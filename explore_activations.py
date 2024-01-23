# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image

# %%
# Load some activations
path = "./data/activations"
import os

# List all files in the path directory with .parquet extension
parquet_files = [f for f in os.listdir(path) if f.endswith(".parquet")]
print("Parquet files found:", parquet_files)


# Function to load a parquet file
def load_parquet_file(file_path):
    return pd.read_parquet(file_path)


df = load_parquet_file(os.path.join(path, parquet_files[0]))
df.head()

# %%
# Load the images and display them alongside their activations
image_path = "./data/images"
image_files = [f for f in os.listdir(image_path) if f.endswith(".jpeg")]

# create map with batch_idx as the key
batch_idx_to_img_file_name = {}
for img in image_files:
    idx = int(img.split("new_index_")[1].split(".")[0])
    batch_idx_to_img_file_name[idx] = img

batch_idx_to_img_file_name

df.head()
dft = (
    df[df["neuron_idx"] == 0]
    .sort_values(by="activation_value", ascending=False)
    .drop_duplicates(subset="batch_idx")
)
fig, axs = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axs.flatten()):
    row = dft.iloc[i]
    img = Image.open(
        os.path.join(image_path, batch_idx_to_img_file_name[row["batch_idx"]])
    )
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("label={}\npredicted={}".format(row["class_name"], row["predicted"]))

plt.tight_layout()
