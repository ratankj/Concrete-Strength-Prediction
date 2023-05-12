import os
import sys
from concrete_strength.logger import logging
from concrete_strength.exception import CustomException
import pandas as pd

from concrete_strength.components.data_ingestion import DataIngestion
from concrete_strength.components.data_transformation import DataTransformation
from concrete_strength.components.model_trainer import ModelTrainer


if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)


# to run this code use
#python concrete_strength/pipeline/training_pipeline.py