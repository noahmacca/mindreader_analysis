# %%
import pandas as pd
import json

# Load the patch activations
MODEL = "CLIP_TINY"
FEATURE_TYPE = "NEURON"
LAYER_TYPE = "FC1"


def make_feature_id(model, feature_type, layer_index, layer_type, feature_index):
    return "-".join(
        [
            str(model),
            str(feature_type),
            str(layer_index),
            str(layer_type),
            str(feature_index),
        ]
    )


dfs = []
for layerIdx in range(0, 10):
    df_i = pd.read_parquet(
        "./data/imagenet_1k_patch_labels/combined_final/final_layer_num_{}_dataset_fold_train.parquet".format(
            layerIdx
        )
    )
    df_i["layerIdx"] = layerIdx
    df_i.drop(
        columns=["batch_index", "image_index", "pred_class", "index"], inplace=True
    )
    df_i.rename(
        columns={
            "neuron_index": "featureIdx",
            "unique_index": "imageIdx",
            "patch_index": "patchIdx",
            "activation_value": "activationValue",
        },
        inplace=True,
    )
    df_i["featureId"] = df_i.apply(
        lambda x: make_feature_id(
            MODEL, FEATURE_TYPE, layerIdx, LAYER_TYPE, x["featureIdx"]
        ),
        axis=1,
    )
    df_i = df_i[df_i["patchIdx"] != 0]
    print("Loaded {} rows for layer {}".format(len(df_i), layerIdx))
    dfs.append(df_i)


df = pd.concat(dfs, axis=0)
del dfs
del df_i

df.head()


# %%
# Make the featureImageActivation table

TOP_N_FEAT_PER_LAYER = 50

# Calculate z score using the "random" values
df_stats_random = (
    df[df["type"] == "random"]
    .groupby(["layerIdx", "featureIdx"])["activationValue"]
    .agg(["mean", "std"])
    .reset_index()
)

df = df.merge(df_stats_random, how="left", on=["layerIdx", "featureIdx"])
df["activationZScore"] = df.apply(
    lambda x: (x["activationValue"] - x["mean"]) / x["std"], axis=1
)
df.drop(columns=["mean", "std"], inplace=True)

# Filter down to just "top" images, and keep top n
print(f"All rows: {len(df)} rows.")
df_top = df[df["type"] == "top"].copy()
print(f"Only type=top: {len(df_top)} rows.")


# # For each top neuron, get the top 10 images
df_top_features = (
    df_top.groupby(["layerIdx", "featureIdx"])["activationValue"].max().reset_index()
)
df_top_features["feature_rank"] = df_top_features.groupby(["layerIdx"])[
    "activationValue"
].rank(method="dense", ascending=False)
df_top_features = df_top_features[
    df_top_features["feature_rank"] <= TOP_N_FEAT_PER_LAYER
]
df_top_features.drop(columns=["feature_rank", "activationValue"], inplace=True)

# Merge df_top with df_top_images to keep only the top TOP_N_FEAT_PER_LAYER images per top neuron
df_top = df_top.merge(df_top_features, on=["layerIdx", "featureIdx"])

print(len(df_top))
print(
    f"Only top {TOP_N_FEAT_PER_LAYER} features. Down to {len(df_top)} rows, using {df_top.memory_usage(deep=True).sum() / 1024**2:.2f} MB in memory"
)

# Write to disk
import os

DB_FILES_PATH = "./data/imagenet_1k_patch_labels/db_files"
# Create the directory for the database files
os.makedirs(DB_FILES_PATH, exist_ok=True)

df_top.rename(columns={"imageIdx": "imageId"})[
    ["featureId", "imageId", "patchIdx", "activationValue", "activationZScore", "label"]
].to_csv(os.path.join(DB_FILES_PATH, "featureImageActivationPatch.csv"), index=False)


# %%
# Make the feature table

df_feat = (
    df_top.groupby(["featureId", "layerIdx", "featureIdx"])["activationValue"]
    .max()
    .reset_index()
)
df_feat.rename(columns={"activationValue": "maxActivation"}, inplace=True)
df_feat["featureType"] = FEATURE_TYPE
df_feat["modelName"] = MODEL
df_feat["layerType"] = LAYER_TYPE

# Get histogram values
import numpy as np
import json


def compute_histogram(data, num_bins=30):
    """Compute histogram for activation values greater than zero and return as JSON string."""
    hist_values, bin_edges = np.histogram(data[data > 0], bins=num_bins)
    histogram_data = [
        {"x": round((bin_edges[i] + bin_edges[i + 1]) / 2, 4), "y": int(hist_values[i])}
        for i in range(len(hist_values))
    ]
    return json.dumps(histogram_data)


# Apply the function outside of the definition using groupby
df_feat_hist = (
    df_top.groupby("featureId")["activationValue"]
    .apply(compute_histogram, num_bins=30)
    .reset_index()
    # .rename(columns={"activationValue", "activationHistVals"})
)
df_feat_hist.rename(columns={"activationValue": "activationHistVals"}, inplace=True)

df_feat = df_feat.merge(df_feat_hist, on="featureId")

# %%
# Calculate autointerp strings and statistics
df_autointerp_labels = df_top[df_top["activationZScore"] > 1.0].copy()

label_string_to_list = lambda x: (
    [] if x == "None" else [i.strip().lower() for i in x.split(",")]
)
df_autointerp_labels["label_list"] = df_autointerp_labels["label"].apply(
    label_string_to_list
)

df_autointerp_labels["label_list_weighted"] = df_autointerp_labels.apply(
    lambda x: [(i, round(x["activationZScore"], 4)) for i in x["label_list"]], axis=1
)

# Calculate score by count * average zScore for each label
df_autointerp_labels = (
    df_autointerp_labels.groupby(["featureId"])["label_list_weighted"]
    .agg("sum")
    .apply(
        lambda x: {
            label: (
                len([label_ for label_, _ in x if label_ == label]),
                int(sum(weight for label_, weight in x if label_ == label)),
            )
            for label in set(label for label, _ in x)
        }
    )
    .apply(
        lambda label_counts: sorted(
            [
                (label, count * avg_activation)
                for label, (count, avg_activation) in label_counts.items()
            ],
            key=lambda item: item[1],
            reverse=True,
        )
    )
    .reset_index(name="concat_labels")
)

# Calculate the gini coefficient for each set of concat_labels
df_autointerp_labels["autointerpScoreGini"] = df_autointerp_labels[
    "concat_labels"
].apply(
    lambda labels: 1
    - sum((count / sum(count for _, count in labels)) ** 2 for _, count in labels)
)

df_autointerp_labels["autoInterp"] = df_autointerp_labels["concat_labels"].apply(
    lambda x: json.dumps(x[:7])
)

df_autointerp_labels["autointerpScoreMax"] = df_autointerp_labels[
    "concat_labels"
].apply(lambda x: x[0][1])

df_autointerp_labels.drop(columns=["concat_labels"], inplace=True)

df_autointerp_labels.head()

# %%
# Add autointerp feature fields and write to disk
df_feat = df_feat.merge(df_autointerp_labels, on="featureId", how="left")

# Write to disk
df_feat.to_csv(os.path.join(DB_FILES_PATH, "feature.csv"), index=False)

df_feat.head()

# %%
