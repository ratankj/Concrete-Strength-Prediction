
# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from concrete_strength.exception import CustomException
from concrete_strength.logger import logging
from concrete_strength.config.configuration import MODEL_FILE_PATH
from concrete_strength.utils.utils import save_object
#from concrete_strength.utils.utils import evaluate_model
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso,Ridge,ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor 
from xgboost import XGBRegressor



from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = MODEL_FILE_PATH


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_model(self,X_train,y_train,X_test,y_test,models):
        try:
            report = {}
            
            for i in range(len(models)):
                model = list(models.values())[i]
                
                model.fit(X_train,y_train)
                
                y_test_pred = model.predict(X_test)
                
                test_model_score = r2_score(y_test,y_test_pred)
                
                report[list(models.keys())[i]] = test_model_score
            
            return report
                
        except Exception as e:
            logging.info("Exception occure while evaluation of model")
            raise CustomException(e,sys)



    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1],
                                                test_array[:,:-1],test_array[:,-1])

            models={
            'LinearRegression':LinearRegression(),
            'DecisionTree':DecisionTreeRegressor(),
            'Gradient Boosting':GradientBoostingRegressor(),
            'Random Forest':RandomForestRegressor(),
            'XGB Regressor':XGBRegressor(),
            'svr' : SVR(kernel='rbf'),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet()

            
        }
            
           

            
            model_report:dict=self.evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)