# %%
import os
import pandas as pd
import base64

from create_db import Image
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

engine = create_engine(os.getenv("DB_CONN_LOCAL"))

# Load images
image_path = "./data/images"
image_files = [f for f in os.listdir(image_path) if f.endswith(".jpeg")]

image_data = []
for image_file in image_files:
    with open(os.path.join(image_path, image_file), "rb") as file:
        binary_data = file.read()
        base64_encoded_data = base64.b64encode(binary_data)
        base64_string = base64_encoded_data.decode("utf-8")
        id = int(image_file.split("new_index_")[1].split(".")[0])
        label = image_file.split("_orig")[0]
        image_data.append({"id": id, "label": label, "data": base64_string})

df_image = pd.DataFrame(image_data)

# Load activations
act_path = "./data/activations"

# List all files in the path directory with .parquet extension
parquet_files = [f for f in os.listdir(act_path) if f.endswith(".parquet")]
print("Parquet files found:", parquet_files)


layer_filename_to_id_prefix = {
    "Activations FC1 7.parquet": "7_FC1_",
    "Activations FC2 7.parquet": "7_FC2_",
    "Activations 7.parquet": "7_",
}

activation_dfs = []

for file in [i for i in parquet_files if "Activations FC1 7.parquet" in i]:
    # Just the FC layers for now
    print("loading {}...".format(file))
    df_i = pd.read_parquet(os.path.join(act_path, file))
    df_i["neuron_id"] = df_i["neuron_idx"].apply(
        lambda x: "{}{}".format(layer_filename_to_id_prefix[file], x)
    )
    activation_dfs.append(df_i)

df_act = pd.concat(activation_dfs, axis=0)
df_act.rename(columns={"batch_idx": "img_idx"}, inplace=True)

# %%
# Append necessary fields onto df_image
# Create a dictionary with img_idx as keys and unique predicted values as values
img_idx_to_predicted = (
    df_act.drop_duplicates(subset="img_idx").set_index("img_idx")["predicted"].to_dict()
)
df_image["predicted"] = df_image["id"].apply(lambda x: img_idx_to_predicted[x])
df_image.head()

# Get image max activation value
df_act.head()

img_idx_to_max_activation = (
    df_act.groupby(["img_idx"])["activation_value"].max().to_dict()
)
df_image["max_activation"] = df_image["id"].apply(
    lambda x: img_idx_to_max_activation[x]
)

df_image.head()

# %%
# Write to the image table
from create_db import Image

Session = sessionmaker(bind=engine)
session = Session()

for index, row in df_image.iterrows():
    image_data = Image(
        id=row["id"],
        label=row["label"],
        predicted=row["predicted"],
        maxActivation=row["max_activation"],
        data=row["data"],
    )
    session.add(image_data)

session.commit()
session.close()

# %%
# now do Neurons table
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

df_neuron = pd.merge(dft1, dft2, how="left", left_on="neuron_id", right_on="neuron_id")
df_neuron.columns = ["neuron_id", "highest_activation_classes", "max_activation"]
df_neuron.head()

# %%
# Write to Neurons table
from create_db import Neuron

Session = sessionmaker(bind=engine)
session = Session()

for index, row in df_neuron.iterrows():
    image_data = Neuron(
        id=row["neuron_id"],
        topClasses=row["highest_activation_classes"],
        maxActivation=row["max_activation"],
    )
    session.add(image_data)

session.commit()
session.close()

# %%
# Prep neuron_image_activations table
from tqdm import tqdm

df_act.sort_values(by=["neuron_id", "img_idx", "class_name", "patch_idx"], inplace=True)

tqdm.pandas(desc="Processing neuron activations")
# Apply the progress bar to the groupby operation
df_neuron_image_activations = (
    df_act.groupby(["neuron_id", "img_idx", "class_name"])["activation_value"]
    .progress_apply(list)
    .reset_index()
)

df_neuron_image_activations["max_activation"] = df_neuron_image_activations[
    "activation_value"
].apply(max)


# %%
# Scale the activation value and base64 encode
# get activations and scale them to match dims of image

import math
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt


def activations_array_to_b64_img(act_in, target_shape):
    assert math.sqrt(len(act_in)) % 1 == 0, "act_in should be square!"
    side_len = int(math.sqrt(len(act_in)))
    assert (
        target_shape[0] % side_len == 0
    ), "target_shape first dim={} should be divisble by side_len={}".format(
        target_shape[0], side_len
    )
    assert (
        target_shape[1] % side_len == 0
    ), "target_shape second dim={} should be divisble by side_len={}".format(
        target_shape[1], side_len
    )
    act_in = np.array(act_in).reshape(side_len, side_len)

    # Goes from 0-1
    act_in_norm = (act_in - act_in.min()) / (act_in.max() - act_in.min())

    # Convert the scaled activations to an image
    img = Image.fromarray(
        (plt.cm.inferno(act_in_norm)[:, :, :3] * 255).astype(np.uint8), "RGB"
    )

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


df_neuron_image_activations["patch_activations_scaled"] = df_neuron_image_activations[
    "activation_value"
].progress_apply(lambda x: activations_array_to_b64_img(x[1:], (224, 224)))


# %%
# Write to NeuronImageActivation table
from create_db import NeuronImageActivation

Session = sessionmaker(bind=engine)
session = Session()

for index, row in df_neuron_image_activations.iterrows():
    if (index % 25000 == 0) | (index == len(df_neuron_image_activations) - 1):
        session.commit()
        print("done {}/{}".format(index, len(df_neuron_image_activations)))
    neuron_image_activation_data = NeuronImageActivation(
        neuronId=row["neuron_id"],
        imageId=row["img_idx"],
        maxActivation=row["max_activation"],
        patchActivations=row["activation_value"],
        patchActivationsScaled=row["patch_activations_scaled"],
    )
    session.add(neuron_image_activation_data)

session.close()
print("done")
