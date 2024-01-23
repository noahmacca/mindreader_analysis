# %%
import os
import pandas as pd

# Load images
image_path = "./data/images"
image_files = [f for f in os.listdir(image_path) if f.endswith(".jpeg")]

image_data = []
for image_file in image_files:
    with open(os.path.join(image_path, image_file), "rb") as file:
        binary_data = file.read()
        id = int(image_file.split("new_index_")[1].split(".")[0])
        label = image_file.split("_orig")[0]
        image_data.append({"id": id, "label": label, "data": binary_data})

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

for file in [i for i in parquet_files if "FC" in i]:
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
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

engine = create_engine(os.getenv("DB_CONN_LOCAL"))

# Assuming df_image is the DataFrame with the image data to be inserted into the database
Session = sessionmaker(bind=engine)
session = Session()

for index, row in df_image.iterrows():
    image_data = Image(
        id=row["id"],
        label=row["label"],
        predicted=row["predicted"],
        max_activation=row["max_activation"],
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
        top_classes=row["highest_activation_classes"],
        max_activation=row["max_activation"],
    )
    session.add(image_data)

session.commit()
session.close()

# %%
# Prep neuron_image_activations table
from tqdm import tqdm

df_act.sort_values(by=["neuron_id", "img_idx", "class_name", "patch_idx"], inplace=True)

# %%
tqdm.pandas(desc="Processing neuron activations")
# Apply the progress bar to the groupby operation
df_neuron_image_activations = (
    df_act.groupby(["neuron_id", "img_idx", "class_name"])["activation_value"]
    .progress_apply(list)
    .reset_index()
)


# %%
from create_db import NeuronImageActivation

for index, row in df_neuron_image_activations.iterrows():
    neuron_image_activation_data = NeuronImageActivation(
        neuron_id=row["neuron_id"],
        image_id=row["img_idx"],
        patch_activations=row["activation_value"],
    )
    session.add(neuron_image_activation_data)

session.commit()
session.close()
