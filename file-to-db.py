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
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import ARRAY

load_dotenv()

engine = create_engine(os.getenv("DB_CONN_PROD"))

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
