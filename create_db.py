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
