from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import DataValidationArtifact,DataIngestionArtifact
from src.constants import *
from src.entity.config_entity import DataValidationConfig
from src.utils.main_utils import read_yaml_file,write_yaml
import os
import sys
import json
import pandas as pd
from pandas import DataFrame


class DataValidation:
    
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_config:DataValidationConfig):
        
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config=read_yaml_file(file_path=SCHEMA_FILE_PATH)
        
        except Exception as e:
            raise MyException(e,sys)
        
    def validate_number_of_columns(self,dataframe:DataFrame)->bool:
        
        try:
            status=len(dataframe.columns)==self._schema_config['numerical_columns'] + self._schema_config['categorical_columns']
            logging.info(f"Is required column present: {status}")
            return status
            
        except Exception as e:
            raise MyException(e,sys)
        
        
    def is_column_exist(self, df: DataFrame) -> bool:
       
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")


            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns)>0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")

            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True
        except Exception as e:
            raise MyException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
        

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            validation_error_msg = ""

            expected_columns = self._schema_config['numerical_columns'] + self._schema_config['categorical_columns']

        # Check training dataframe
            missing_train_columns = [col for col in expected_columns if col not in train_df.columns]
            if missing_train_columns:
                validation_error_msg += f"Columns missing in training dataframe: {missing_train_columns}. "
            else:
                logging.info("All required columns present in training dataframe.")

        # Check test dataframe
            missing_test_columns = [col for col in expected_columns if col not in test_df.columns]
            if missing_test_columns:
                validation_error_msg += f"Columns missing in test dataframe: {missing_test_columns}."
            else:
                logging.info("All required columns present in testing dataframe.")

            validation_status = len(validation_error_msg) == 0

        # Create DataValidationArtifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg.strip(),
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

        # Save report
            os.makedirs(os.path.dirname(self.data_validation_config.validation_report_file_path), exist_ok=True)
            with open(self.data_validation_config.validation_report_file_path, "w") as f:
                json.dump({
                    "validation_status": validation_status,
                    "message": validation_error_msg.strip()
                }, f, indent=4)

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise MyException(e, sys) from e