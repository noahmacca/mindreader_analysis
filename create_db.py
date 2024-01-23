# %%
from dotenv import load_dotenv
import os
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    ForeignKey,
    Table,
    LargeBinary,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import ARRAY

load_dotenv()

engine = create_engine(os.getenv("DB_CONN_LOCAL"))

Base = declarative_base()


class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True)
    label = Column(String)
    predicted = Column(String)
    max_activation = Column(Float)
    data = Column(LargeBinary)


class Neuron(Base):
    __tablename__ = "neurons"
    id = Column(String, primary_key=True)
    top_classes = Column(String)
    max_activation = Column(Float)


class NeuronImageActivation(Base):
    __tablename__ = "neuron_image_activations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    neuron_id = Column(String, ForeignKey("neurons.id"))
    image_id = Column(Integer, ForeignKey("images.id"))
    patch_activations = Column(ARRAY(Float))


# Drop existing tables
Base.metadata.drop_all(engine)

# Create the tables in the database
Base.metadata.create_all(engine)

# %%
# Add some sample data for testing
with open("data/images/academic_gown_orig_index_400_new_index_412.jpeg", "rb") as file:
    binary_data = file.read()

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Example data to append
image_data = Image(
    id=1,
    label="basset hound",
    predicted="chihuahua",
    max_activation=3.1,
    data=binary_data,
)

neuron_data = Neuron(
    id="7_1_211", top_classes="fires on dogs, faces, bugs", max_activation=3.2
)
neuron_image_activation_data = NeuronImageActivation(
    neuron_id="7_1_211", image_id=1, patch_activations=[1, 0, 0, 122, 255]
)

# Add data to the session
session.add(image_data)
session.add(neuron_data)
session.add(neuron_image_activation_data)

# Commit the transaction
session.commit()

# Close the session
session.close()

# %%
