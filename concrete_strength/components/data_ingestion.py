from concrete_strength.exception import CustomException
from concrete_strength.logger import logging
from concrete_strength.constant import *
from concrete_strength.config.configuration import *
import os,sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass




@dataclass
class DataIngestionconfig:
    train_data_path:str=TRAIN_FILE_PATH
    test_data_path:str=TEST_FILE_PATH
    raw_data_path:str= RAW_FILE_PATH


class DataIngestion:
    def __init__(self,):
        self.data_ingestion_config = DataIngestionconfig()


    def initiate_data_ingestion(self):

        logging.info("initiate data ingestion config")

        try:
            df=pd.read_csv(DATASET_PATH)

            logging.info("Reading csv file")
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)


            logging.info("train test split")

            train_set,test_Set = train_test_split(df,test_size=0.20,random_state=42)
            df.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            df.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
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
