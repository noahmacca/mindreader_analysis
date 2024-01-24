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
    maxActivation = Column(Float, nullable=False)
    data = Column(String, nullable=False)


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
    patchActivations = Column(ARRAY(Float), nullable=False)
    patchActivationsScaled = Column(String, nullable=False)


# Drop existing tables
Base.metadata.drop_all(engine)

# Create the tables in the database
Base.metadata.create_all(engine)

# %%
# Add some sample data for testing
# with open("data/images/academic_gown_orig_index_400_new_index_412.jpeg", "rb") as file:
#     binary_data = file.read()

# # Create a session
# Session = sessionmaker(bind=engine)
# session = Session()

# # Example data to append
# image_data = Image(
#     id=1,
#     label="basset hound",
#     predicted="chihuahua",
#     max_activation=3.1,
#     data=binary_data,
# )

# neuron_data = Neuron(
#     id="7_1_211", top_classes="fires on dogs, faces, bugs", max_activation=3.2
# )
# neuron_image_activation_data = NeuronImageActivation(
#     neuron_id="7_1_211", image_id=1, patch_activations=[1, 0, 0, 122, 255]
# )

# # Add data to the session
# session.add(image_data)
# session.add(neuron_data)
# session.add(neuron_image_activation_data)

# # Commit the transaction
# session.commit()

# # Close the session
# session.close()
