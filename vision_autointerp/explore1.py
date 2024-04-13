# %%
# Read in images and ask GPTV what the core concept(s) present in the image are
print("TODO")

# %%
# Load in images and activaitons for one neuron and ask gpt-v what's in it

# Load activations
import os
import shutil

# Define the source and destination directories
source_dir = "../data/images"
dest_dir = "../data/images_v2"

# Overwrite the destination directory
if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)
os.makedirs(dest_dir)

# List all files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith(".jpeg"):
        # Extract the new index from the filename
        new_index = (
            filename.split("_orig_index_")[1].split("_new_index_")[1].split(".")[0]
        )

        # Construct the new filename
        new_filename = f"{new_index}.jpeg"
        before_dir = os.path.join(source_dir, filename)
        after_dir = os.path.join(dest_dir, new_filename)
        print("copying {} to {}".format(before_dir, after_dir))

        # Copy the file to the new directory with the new filename
        shutil.copy(before_dir, after_dir)

# %%
import pandas as pd

# Get the top activating images for the top neuron
df_neuron = pd.read_csv("../db_outputs/neuron.csv")
df_neuron["layer"] = df_neuron["id"].apply(lambda x: x.split("_")[0])
df_neuron.head()

ref_neuron_id = (
    df_neuron[df_neuron["layer"] == "7"]
    .sort_values(by="maxActivation", ascending=False)
    .iloc[0, 0]
)

# %%
df_act = pd.read_csv("../db_outputs/neuron-image-activation.csv")
ref_neuron_image_ids = (
    df_act[df_act["neuronId"] == ref_neuron_id]
    .sort_values(by="maxActivation", ascending=False)["imageId"][:10]
    .to_list()
)

# Load images into a dictionary with image_id as key
ref_image_map = {}
for image_id in ref_neuron_image_ids:
    image_path = os.path.join(dest_dir, f"{image_id}.jpeg")
    with open(image_path, "rb") as img_file:
        ref_image_map[image_id] = img_file.read()


ref_image_map

# %%
# Get the
