from concrete_strength.constant import *
from concrete_strength.exception import CustomException
from concrete_strength.logger import logging


import os,sys

ROOT_DIR = ROOT_DIR_KEY 

DATASET_PATH = os.path.join(ROOT_DIR,DATA_DIR,DATA_DIR_KEY)


# DATA INGESTION PATH

RAW_FILE_PATH = os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,DATA_INGESTION_KEY,CURRENT_TIME_STAMP,
                             DATA_INGESTION_RAW_DATA_DIR_KEY,RAW_DATA_DIR_KEY)

TEST_FILE_PATH = os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,DATA_INGESTION_KEY,CURRENT_TIME_STAMP,
                              DATA_INGESTION_INGESTED_DIR_NAME_KEY,TEST_DATA_DIR_KEY)

TRAIN_FILE_PATH = os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,DATA_INGESTION_KEY,CURRENT_TIME_STAMP,
                               DATA_INGESTION_INGESTED_DIR_NAME_KEY,TRAIN_DATA_DIR_KEY)


# PREPROCESSING OBJ


PREPROCESSING_OBJ_PATH = os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,
                                      DATA_TRANSFORMATION_ARTIFACT,CURRENT_TIME_STAMP,
                                      DATA_TRANSFORMATION_PREPROCESSING_OBJ)

# MODEL FILE

MODEL_FILE_PATH = os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,
                               MODEL_ARTIFACT,CURRENT_TIME_STAMP,MODEL_OBJECT)
