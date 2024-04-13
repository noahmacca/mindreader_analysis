# %%

import pandas as pd

df = pd.read_parquet(
    "./data/imagenet_1k_patch_labels/combined_final/final_layer_num_7_dataset_fold_train.parquet"
)
df.groupby(["unique_index", "neuron_index", "type"])["index"].count()

# %%
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
for layerIdx in range(6, 8):
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
df_top_features = df_top_features[df_top_features["feature_rank"] <= 10]
df_top_features.drop(columns=["feature_rank", "activationValue"], inplace=True)

# Merge df_top with df_top_images to keep only the top 10 images per top neuron
df_top = df_top.merge(df_top_features, on=["layerIdx", "featureIdx"])

print(len(df_top))
print(
    f"Only top 10 features. Down to {len(df_top)} rows, using {df_top.memory_usage(deep=True).sum() / 1024**2:.2f} MB in memory"
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

df_feat.head(20)

# Write to disk
df_feat.to_csv(os.path.join(DB_FILES_PATH, "feature.csv"), index=False)


# %%
from tqdm import tqdm

tqdm.pandas()

df["feature_id"] = df.progress_apply(
    lambda x: make_feature_id(
        MODEL, FEATURE_TYPE, x["layer_index"], LAYER_TYPE, x["feature_index"]
    ),
    axis=1,
)

df.head()

# %%
df.head()
# %%
pd.set_option("display.max_rows", 50)

# %%
# pd.options.display.max_rows = 30

df_counts = df.groupby(["layer_index", "feature_index", "unique_index", "type"])[
    "index"
].count()
df_counts.head(50)

# %%
df[df["neuron_index"] == 161]["label"].value_counts()


# %%
df.info(memory_usage="deep")


# %%


make_feature_id("CLIP_TINY", "NEURON", 1, "FC1", 102)
