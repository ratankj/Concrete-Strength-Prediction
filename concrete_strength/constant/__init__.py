import os

ROOT_DIR_KEY = os.getcwd()

# data file path

DATA_DIR = "data"
DATA_DIR_KEY = "concrete_data.csv"

# artfact

ARTIFACT_DIR_KEY = "artfact"

DATASET_URL = "dataset_url"


# data ingestion 

# Data Ingestion constants
DATA_INGESTION_KEY = 'data_ingestion'
TRAIN_DATA_DIR_KEY = 'train'
TEST_DATA_DIR_KEY = 'test' 
RAW_DATA_DIR_KEY = 'raw'

DATA_TRANSFORMATION_ARTIFACT = 'data_transformation'
DATA_TRANSFORMATION_PREPROCESSING_OBJ = 'preprocessor.pkl'

MODEL_ARTIFACT = 'model_trainer'
MODEL_OBJECT = 'model.pkl'