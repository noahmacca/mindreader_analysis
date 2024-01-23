# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from PIL import Image

# Load activations
act_path = "./data/activations"
img_path = "./data/images"

# List all files in the path directory with .parquet extension
parquet_files = [f for f in os.listdir(act_path) if f.endswith(".parquet")]
print("Parquet files found:", parquet_files)


# Function to load a parquet file
def load_parquet_file(file_path):
    return pd.read_parquet(file_path)


df_act = load_parquet_file(os.path.join(act_path, parquet_files[0]))
print("loaded {} rows from {}".format(len(df_act), parquet_files[0]))

# Get image paths
image_files = [f for f in os.listdir(img_path) if f.endswith(".jpeg")]

# create map with batch_idx as the key
batch_idx_to_img_file_name = {}
for img in image_files:
    idx = int(img.split("new_index_")[1].split(".")[0])
    batch_idx_to_img_file_name[idx] = img

# %%
dft = df_act.groupby(["batch_idx", "class_name", "neuron_idx"])["activation_value"].agg(
    ["mean", "max", "count"]
)
dft.reset_index(inplace=True)
dft.sort_values(by=["neuron_idx", "max"], ascending=[True, False])

# %%
dft1 = (
    dft.groupby(["neuron_idx", "batch_idx", "class_name"])["max"]
    .agg(["mean", "sum"])
    .sort_values(by=["neuron_idx", "sum"], ascending=False)
    .reset_index()
)
dft2 = dft1.groupby(["neuron_idx"]).first().reset_index()
dft2.sort_values(by="mean", ascending=False, inplace=True)
dft2.head()


# %%
# For the most highly activating neurons, look at activations on those images
ref_row_top_act = dft2.iloc[15]
ref_df_all_act = df_act[
    (df_act["neuron_idx"] == ref_row_top_act["neuron_idx"])
    & (df_act["batch_idx"] == ref_row_top_act["batch_idx"])
]

info = """
neuron_idx={}
class_name={}
predicted={}
mean max activation for class={}
img batch_idx={}
""".format(
    ref_row_top_act["neuron_idx"],
    ref_row_top_act["class_name"],
    ref_df_all_act["predicted"].iloc[0],
    round(ref_row_top_act["mean"], 5),
    ref_df_all_act["batch_idx"].iloc[0],
)


# get activations
act_i = np.array(ref_df_all_act["activation_value"])[1:].reshape(14, 14)

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
        img_path, batch_idx_to_img_file_name[ref_df_all_act["batch_idx"].iloc[0]]
    )
)

f, ax = plt.subplots(1, 1)
ax.imshow(img)
ax.imshow(scaled, alpha=0.5)
ax.axis("off")
ax.set_title(info)


# %%
# Load the images and display them alongside their activations

df_act.head()
dft = (
    df_act[df_act["neuron_idx"] == 0]
    .sort_values(by="activation_value", ascending=False)
    .drop_duplicates(subset="batch_idx")
)
fig, axs = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axs.flatten()):
    row = dft.iloc[i]
    img = Image.open(
        os.path.join(img_path, batch_idx_to_img_file_name[row["batch_idx"]])
    )
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("label={}\npredicted={}".format(row["class_name"], row["predicted"]))

plt.tight_layout()

# %%
