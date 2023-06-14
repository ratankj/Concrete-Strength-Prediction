from concrete_strength.exception import CustomException
from concrete_strength.logger import logging
from concrete_strength.constant import *
from concrete_strength.config.configuration import *
from concrete_strength.components.data_transformation import DataTransformation
from concrete_strength.components import data_transformation
import os,sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from concrete_strength.constant.data_base import *
from concrete_strength.data_access.mongo_data import mongodata




@dataclass
class DataIngestionconfig:
    train_data_path:str=TRAIN_FILE_PATH
    test_data_path:str=TEST_FILE_PATH
    raw_data_path:str= RAW_FILE_PATH


class DataIngestion:
    def __init__(self,):
        self.data_ingestion_config = DataIngestionconfig()

    
    def get_data_from_mongo_DB(self) -> str:
        try:
            
            # Raw Data Directory Path
            #raw_data_dir = self.data_ingestion_config.raw_data_dir
            raw_data_dir=self.data_ingestion_config.raw_data_path
            
            # Make Raw data Directory
            #s.makedirs(raw_data_dir, exist_ok=True)
            os.makedirs(os.path.dirname(raw_data_dir),exist_ok=True)

            #file_name = FILE_NAME

            #raw_file_path = os.path.join(raw_data_dir, file_name)

            logging.info(
                f"Downloading file from Mongo DB into :[{raw_data_dir}]")
            
            data=mongodata()
            # Storing mongo data ccs file to raw directoy 
            data.export_collection_as_csv(collection_name=COLLECTION_NAME,database_name=DATABASE_NAME,file_path=raw_data_dir)
            
            logging.info(
                f"File :[{raw_data_dir}] has been downloaded successfully.")
            return raw_data_dir

        except Exception as e:
            raise CustomException(e, sys) from e


    def initiate_data_ingestion(self):
        filepath=self.get_data_from_mongo_DB()

        logging.info("initiate data ingestion config")

        try:
            df=pd.read_csv(filepath)
        

            logging.info("Reading csv file")
            #os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)
            #df.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            
            logging.info(f"dataset path : {DATASET_PATH}")
            logging.info("train test split")

            train_set,test_Set = train_test_split(df,test_size=0.20,random_state=42)

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)

            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path),exist_ok=True)
            test_Set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            
            logging.info(f"train data path, {TRAIN_FILE_PATH}")

            logging.info("data insgested into raw, train and test data")

            return(

                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )


        except Exception as e:
            logging.info("Exception in data ingestion stage")
            raise CustomException(e,sys)



    
# to run data ingestion

if __name__ == "__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr,test_arr,_= data_transformation.initaite_data_transformation(TRAIN_FILE_PATH,
                                                                           TEST_FILE_PATH)

