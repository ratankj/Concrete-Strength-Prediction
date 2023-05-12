import os
from datetime import datetime


#to store the log file
def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"

CURRENT_TIME_STAMP = get_current_time_stamp()


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
DATA_INGESTION_RAW_DATA_DIR_KEY= "raw_data_dir"
DATA_INGESTION_INGESTED_DIR_NAME_KEY= "ingested_dir"
RAW_DATA_DIR_KEY = 'raw.csv'
TRAIN_DATA_DIR_KEY = 'train.csv'
TEST_DATA_DIR_KEY = 'test.csv' 


DATA_TRANSFORMATION_ARTIFACT = 'data_transformation'
DATA_TRANSFORMATION_PREPROCESSING_OBJ = 'preprocessor.pkl'

MODEL_ARTIFACT = 'model_trainer'
MODEL_OBJECT = 'model.pkl'