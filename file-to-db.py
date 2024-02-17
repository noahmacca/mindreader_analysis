# %%
import os
import pandas as pd

from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    ForeignKey,
    Enum,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import ARRAY

load_dotenv()

engine = create_engine(os.getenv("DB_CONN_LOCAL"))

Base = declarative_base()


class Image(Base):
    __tablename__ = "Image"
    id = Column(Integer, primary_key=True, nullable=False)
    label = Column(String, nullable=False)
    predicted = Column(String, nullable=False)


class Neuron(Base):
    __tablename__ = "Neuron"
    id = Column(String, primary_key=True, nullable=False)
    topClasses = Column(String, nullable=False)
    maxActivation = Column(Float, nullable=False)


class NeuronImageActivation(Base):
    __tablename__ = "NeuronImageActivation"
    id = Column(Integer, primary_key=True, nullable=False)
    neuronId = Column(String, ForeignKey("Neuron.id"), nullable=False)
    imageId = Column(Integer, ForeignKey("Image.id"), nullable=False)
    maxActivation = Column(Float, nullable=False)


class NeuronCorrelation(Base):
    __tablename__ = "NeuronCorrelation"
    id = Column(Integer, primary_key=True, nullable=False)
    startNeuronId = Column(String, ForeignKey("Neuron.id"), nullable=False)
    endNeuronId = Column(String, ForeignKey("Neuron.id"), nullable=False)
    corr = Column(Float, nullable=False)
    layerLocation = Column(
        Enum("LOWER", "SAME", "HIGHER", name="layer_location_enum"), nullable=False
    )


# Drop existing tables
Base.metadata.drop_all(engine)

# Create the tables in the database
Base.metadata.create_all(engine)
print("Dropped and created new tables")

# %%
# Image Table

# Load dataframe from './db_outputs/Image.csv'
df_image = pd.read_csv("./db_outputs/image.csv")
print("Loaded {} Image rows".format(len(df_image)))

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Add all rows to the Image table
for index, row in df_image.iterrows():
    image = Image(id=row["id"], label=row["label"], predicted=row["predicted"])
    session.add(image)

# Commit and close the session
session.commit()
session.close()
print("Done writing to Image table")


# %%
# Write new Neuron table
df_neuron = pd.read_csv("./db_outputs/neuron.csv")
print("Loaded {} Neuron rows".format(len(df_neuron)))

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Add all rows to the Neuron table
for index, row in df_neuron.iterrows():
    neuron = Neuron(
        id=row["id"], topClasses=row["topClasses"], maxActivation=row["maxActivation"]
    )
    session.add(neuron)

# Commit and close the session
session.commit()
session.close()
print("Done writing to Neuron table")

# %%
# Write new NeuronImageActivation table
df_neuron_image_activation = pd.read_csv("./db_outputs/neuron-image-activation.csv")
print("Loaded {} NeuronImageActivation rows".format(len(df_neuron_image_activation)))

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Add all rows to the NeuronImageActivation table
for index, row in df_neuron_image_activation.iterrows():
    neuron_image_activation = NeuronImageActivation(
        neuronId=row["neuronId"],
        imageId=row["imageId"],
        maxActivation=row["maxActivation"],
    )
    session.add(neuron_image_activation)

# Commit and close the session
session.commit()
session.close()
print("Done writing to NeuronImageActivation table")

# %%
# Write NeuronCorrelation table
df_neuron_correlation = pd.read_csv("./db_outputs/neuron-corrs.csv")
print("Loaded {} NeuronCorrelation rows".format(len(df_neuron_correlation)))

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Add all rows to the NeuronImageActivation table
for index, row in df_neuron_correlation.iterrows():
    neuron_correlation = NeuronCorrelation(
        startNeuronId=row["startNeuronId"],
        endNeuronId=row["endNeuronId"],
        corr=row["corr"],
        layerLocation=row["layerLocation"],
    )
    session.add(neuron_correlation)

# Commit and close the session
session.commit()
session.close()
print("Done writing to NeuronCorrelation table")
