# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from PIL import Image
from datetime import datetime


# Load activations
act_path = "./data/activations"
img_path = "./data/images"

# For saving output figs
fig_dir = "outputs/"
os.makedirs(fig_dir, exist_ok=True)

# List all files in the path directory with .parquet extension
parquet_files = [f for f in os.listdir(act_path) if f.endswith(".parquet")]
print("Parquet files found:", parquet_files)


# Function to load a parquet file
def load_parquet_file(file_path):
    return pd.read_parquet(file_path)


df_act = load_parquet_file(os.path.join(act_path, parquet_files[0]))
print("loaded {} rows from {}".format(len(df_act), parquet_files[0]))
df_act.rename(columns={"batch_idx": "img_idx"}, inplace=True)

# Get image paths
image_files = [f for f in os.listdir(img_path) if f.endswith(".jpeg")]

# create map with batch_idx as the key
img_idx_to_img_file_name = {}
for img in image_files:
    idx = int(img.split("new_index_")[1].split(".")[0])
    img_idx_to_img_file_name[idx] = img


# %%
# display any neuron img pair
def plot_neuron_act_for_img(neuron_idx, img_idx, ax_in=None):
    ref_df_all_act = df_act[
        (df_act["neuron_idx"] == neuron_idx) & (df_act["img_idx"] == img_idx)
    ]

    info = """
    neuron_idx={}
    class_name={}
    predicted={}
    img_idx={}
    """.format(
        neuron_idx,
        ref_df_all_act["class_name"].iloc[0],
        ref_df_all_act["predicted"].iloc[0],
        ref_df_all_act["img_idx"].iloc[0],
    )

    # get activations and scale them to match dims of image
    act_i = np.array(ref_df_all_act["activation_value"])[1:].reshape(14, 14)

    # There's probably a built-in way of doing this
    scaled = np.zeros((224, 224))
    pixels_per_cell = 16
    for i in range(act_i.shape[0]):
        for j in range(act_i.shape[1]):
            scaled[
                i * pixels_per_cell : (i + 1) * pixels_per_cell,
                j * pixels_per_cell : (j + 1) * pixels_per_cell,
            ] = act_i[i][j]

    # for i, ax in enumerate(axs.flatten()):
    img = Image.open(
        os.path.join(
            img_path, img_idx_to_img_file_name[ref_df_all_act["img_idx"].iloc[0]]
        )
    )

    if ax_in == None:
        f, ax = plt.subplots(1, 1)
        ax.imshow(img)
        ax.imshow(scaled, alpha=0.5)
        ax.axis("off")
        ax.set_title(info)
    else:
        ax_in.imshow(img)
        ax_in.imshow(scaled, alpha=0.5)
        ax_in.axis("off")
        ax_in.set_title(info)


# %%
# Get the most highly activating neuron-image pairs, as defined by the max activating
# patch for a given neuron anywhere on a certain image.
df_top_neuron_image_acivations = (
    df_act.groupby(["img_idx", "class_name", "neuron_idx"])["activation_value"]
    .agg(["mean", "max"])
    .reset_index()
    .sort_values(by=["neuron_idx", "max"], ascending=[True, False])
)
df_top_neuron_image_acivations.head()

f, axs = plt.subplots(8, 8, figsize=(32, 32))
for i, ax in enumerate(axs.flatten()):
    ref_row_top_act = df_top_neuron_image_acivations.iloc[i]
    plot_neuron_act_for_img(
        ref_row_top_act["neuron_idx"], ref_row_top_act["img_idx"], ax_in=ax
    )

f.tight_layout()

# save
timestamp = datetime.now().strftime("%b %d %Y %H-%M-%S")
filename = os.path.join(fig_dir, f"top_activations_{timestamp}.png")
plt.savefig(filename)


# %%
# Load the images and display them alongside their activations

df_act.head()
df_top_neuron_image_acivations = (
    df_act[df_act["neuron_idx"] == 0]
    .sort_values(by="activation_value", ascending=False)
    .drop_duplicates(subset="img_idx")
)
fig, axs = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axs.flatten()):
    row = df_top_neuron_image_acivations.iloc[i]
    img = Image.open(os.path.join(img_path, img_idx_to_img_file_name[row["img_idx"]]))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("label={}\npredicted={}".format(row["class_name"], row["predicted"]))

plt.tight_layout()

# %%
