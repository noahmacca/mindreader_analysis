# %%
import os
import pandas as pd
import math
import numpy as np
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

from dotenv import load_dotenv

# Write everything to the filesystem to decouple the compute and the actual db uploading
# Something about starting and stopping so many db sessions was causing random db deletions(!)

load_dotenv()
DIRS = {
    "input_data": {
        "images": "./data/images",
        "image_predictions": "./data/image_predictions",
        "activations": "./data/activations",
    },
    "s3_outputs": {
        "image": "./s3_outputs_1/image",
        "neuron-image-activation": "./s3_outputs_1/neuron-image-activation",
    },
    "db_outputs": "./db_outputs",
}


# %%
# IMAGES
image_files = [
    f for f in os.listdir(DIRS["input_data"]["images"]) if f.endswith(".jpeg")
]

image_data = []
for image_file in image_files:
    with open(os.path.join(DIRS["input_data"]["images"], image_file), "rb") as file:
        binary_data = file.read()
        id = int(image_file.split("new_index_")[1].split(".")[0])
        label = image_file.split("_orig")[0]
        image_data.append({"id": id, "label": label, "binary_data": binary_data})

df_image = pd.DataFrame(image_data)

img_idx_to_predicted = pd.read_csv(
    "./data/image_predictions/data.csv", index_col=0
).to_dict()["predicted"]
df_image["predicted"] = df_image["id"].apply(lambda x: img_idx_to_predicted[x])

# Write image files to s3 sync directory
if not os.path.exists(DIRS["s3_outputs"]["image"]):
    os.makedirs(DIRS["s3_outputs"]["image"])

for index, row in df_image.iterrows():
    with open(
        os.path.join(DIRS["s3_outputs"]["image"], "{}.jpg".format(row["id"])), "wb"
    ) as image_file:
        image_file.write(row["binary_data"])

# Write image entries to csv (for later syncing with the db)
if not os.path.exists(DIRS["db_outputs"]):
    os.makedirs(DIRS["db_outputs"])

df_image[["id", "label", "predicted"]].sort_values(by="id").to_csv(
    os.path.join(DIRS["db_outputs"], "image.csv"), index=False
)

del image_data, df_image


# %%
# NEURONS AND ACTIVATION FUNCTIONS
def process_and_save_neuron_data(df_act, is_first=False):
    # transform Neurons table
    dft = (
        df_act.groupby(["neuron_id", "img_idx", "class_name"])["activation_value"]
        .max()
        .reset_index()
        .sort_values(by=["neuron_id", "activation_value"], ascending=[True, False])
    )

    dft1 = (
        dft.groupby("neuron_id")
        .apply(
            lambda x: ", ".join(
                x.sort_values("activation_value", ascending=False).head(5)["class_name"]
            )
        )
        .reset_index()
    )

    dft2 = df_act.groupby(["neuron_id"])["activation_value"].max().reset_index()

    df_neuron = pd.merge(
        dft1, dft2, how="left", left_on="neuron_id", right_on="neuron_id"
    )
    df_neuron.columns = ["neuron_id", "highest_activation_classes", "max_activation"]
    df_neuron.head()

    # Save neurons to file
    # Write image entries to csv (for later syncing with the db)
    if not os.path.exists(DIRS["db_outputs"]):
        os.makedirs(DIRS["db_outputs"])

    df_neuron.rename(
        columns={
            "neuron_id": "id",
            "highest_activation_classes": "topClasses",
            "max_activation": "maxActivation",
        },
        inplace=True,
    )

    if is_first:
        # overwrite neurons file
        df_neuron[["id", "topClasses", "maxActivation"]].to_csv(
            os.path.join(DIRS["db_outputs"], "neuron.csv"), index=False
        )
    else:
        # append
        df_neuron[["id", "topClasses", "maxActivation"]].to_csv(
            os.path.join(DIRS["db_outputs"], "neuron.csv"),
            index=False,
            mode="a",
            header=False,
        )

    print("saved neurons to csv. Cleaning up.")
    del dft, dft1, dft2, df_neuron


def activations_array_to_img_binary(act_in):
    assert math.sqrt(len(act_in)) % 1 == 0, "act_in should be square!"
    side_len = int(math.sqrt(len(act_in)))
    act_in = np.array(act_in).reshape(side_len, side_len)

    # Goes from 0-1
    act_in_norm = (act_in - act_in.min()) / (act_in.max() - act_in.min())

    # Convert the scaled activations to an image
    img = Image.fromarray(
        (plt.cm.inferno(act_in_norm)[:, :, :3] * 255).astype(np.uint8), "RGB"
    )

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return buffered.getvalue()


def process_and_save_neuron_image_activation_data(df_act, is_first=False):
    print("processing neuron_image_activations")

    # Prep neuron_image_activations table
    df_act.sort_values(
        by=["neuron_id", "img_idx", "class_name", "patch_idx"], inplace=True
    )

    tqdm.pandas(desc="Processing neuron activations")
    df_neuron_image_activations = (
        df_act.groupby(["neuron_id", "img_idx", "class_name"])["activation_value"]
        .progress_apply(list)
        .reset_index()
    )

    df_neuron_image_activations["max_activation"] = df_neuron_image_activations[
        "activation_value"
    ].apply(max)

    TOP_N_ACTIVATIONS_PER_NEURON = 50
    before_filter_len = len(df_neuron_image_activations)
    df_neuron_image_activations = (
        df_neuron_image_activations.groupby("neuron_id")
        .apply(
            lambda x: x.nlargest(TOP_N_ACTIVATIONS_PER_NEURON, "max_activation"),
            include_groups=False,
        )
        .reset_index(level="neuron_id")
    )

    after_filter_len = len(df_neuron_image_activations)
    print(
        "Reduced # neuron_image_activations from {} to {} by taking top {} per neuron".format(
            before_filter_len, after_filter_len, TOP_N_ACTIVATIONS_PER_NEURON
        )
    )

    tqdm.pandas(desc="Converting activations to images")

    df_neuron_image_activations["binary_data"] = df_neuron_image_activations[
        "activation_value"
    ].progress_apply(lambda x: activations_array_to_img_binary(x[1:]))

    # Save NeuronImageActivations
    if not os.path.exists(DIRS["s3_outputs"]["neuron-image-activation"]):
        os.makedirs(DIRS["s3_outputs"]["neuron-image-activation"])

    for index, row in df_neuron_image_activations.iterrows():
        image_filename = "neuron-{}-image-{}.jpg".format(
            row["neuron_id"], row["img_idx"]
        )
        with open(
            os.path.join(DIRS["s3_outputs"]["neuron-image-activation"], image_filename),
            "wb",
        ) as image_file:
            image_file.write(row["binary_data"])

    print("saved neuron-image-activation image files to disk.")

    df_neuron_image_activations.rename(
        columns={
            "neuron_id": "neuronId",
            "img_idx": "imageId",
            "max_activation": "maxActivation",
        },
        inplace=True,
    )

    if is_first:
        # overwrite file
        df_neuron_image_activations[["neuronId", "imageId", "maxActivation"]].to_csv(
            os.path.join(DIRS["db_outputs"], "neuron-image-activation.csv"),
            index=False,
        )

    else:
        # append
        df_neuron_image_activations[["neuronId", "imageId", "maxActivation"]].to_csv(
            os.path.join(DIRS["db_outputs"], "neuron-image-activation.csv"),
            index=False,
            mode="a",
            header=False,
        )

    print(
        "saved {} neuronimageactivation entries to csv. Cleaning up.".format(
            len(df_neuron_image_activations)
        )
    )

    del df_neuron_image_activations


# %%
# RUN NEURONS AND ACTIVATIONS CODE
layer_filename_to_id_prefix = {
    "Activations FC1 0.parquet": "0_FC1_",
    "Activations FC1 1.parquet": "1_FC1_",
    "Activations FC1 2.parquet": "2_FC1_",
    "Activations FC1 3.parquet": "3_FC1_",
    "Activations FC1 4.parquet": "4_FC1_",
    "Activations FC1 5.parquet": "5_FC1_",
    "Activations FC1 6.parquet": "6_FC1_",
    "Activations FC1 7.parquet": "7_FC1_",
    "Activations FC1 8.parquet": "8_FC1_",
    "Activations FC1 9.parquet": "9_FC1_",
    "Activations FC2 7.parquet": "7_FC2_",
    "Activations 7.parquet": "7_",
}

include_activation_files = [
    "Activations FC1 0.parquet",
    "Activations FC1 1.parquet",
    "Activations FC1 2.parquet",
    "Activations FC1 3.parquet",
    "Activations FC1 4.parquet",
    "Activations FC1 5.parquet",
    "Activations FC1 6.parquet",
    "Activations FC1 7.parquet",
    "Activations FC1 8.parquet",
    "Activations FC1 9.parquet",
]

# Process all neuron layers
for index, file in enumerate(include_activation_files):
    print("loading {}...".format(file))
    df_act = pd.read_parquet(os.path.join(DIRS["input_data"]["activations"], file))
    df_act["neuron_id"] = df_act["neuron_idx"].apply(
        lambda x: "{}{}".format(layer_filename_to_id_prefix[file], x)
    )
    df_act.rename(columns={"batch_idx": "img_idx"}, inplace=True)
    is_first = index == 0
    print("is_first={}, index={}".format(is_first, index))
    process_and_save_neuron_data(df_act, is_first=is_first)
    process_and_save_neuron_image_activation_data(df_act, is_first=is_first)

    # clean up
    del df_act

# %%
