import sys
import os
import json
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact, 
    DataIngestionArtifact, 
    DataValidationArtifact
)
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file,write_yaml


class DataTransformation:
    """
    Handles all data preprocessing, scaling, encoding, and balancing for train/test datasets.
    """
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact

            # Load schema from YAML file
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            logging.info("Schema loaded successfully")
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """Reads CSV file into a pandas DataFrame."""
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Data loaded from {file_path}")
            return df
        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates a preprocessing pipeline:
        - StandardScaler for numeric columns
        - MinMaxScaler for selected columns
        - ColumnTransformer wraps both
        """
        logging.info("Creating data transformer pipeline")

        try:
            
            ss = StandardScaler()
            mm = MinMaxScaler()
            logging.info("Scalers initialized: StandardScaler, MinMaxScaler")

           
            num_features = self._schema_config['num_features']
            mm_columns = self._schema_config['mm_columns']
            logging.info(f"Numeric columns: {num_features}, MinMax columns: {mm_columns}")

     
            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", ss, num_features),
                    ("MinMaxScaler", mm, mm_columns)
                ],
                remainder='passthrough' 
            )

          
            pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Data transformer pipeline created successfully")
            return pipeline
        except Exception as e:
            logging.exception("Error creating data transformer pipeline")
            raise MyException(e, sys) from e

   
    def _map_gender_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maps 'Gender' column to 0/1 integers."""
        logging.info("Mapping 'Gender' column to 0/1")
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        return df

    def _create_dummy_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates one-hot encoded columns for categorical features."""
        logging.info("Creating dummy variables for categorical features")
        df = pd.get_dummies(df, drop_first=True)
        return df

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renames some columns and casts them to integer type."""
        logging.info("Renaming columns and converting types")
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype(int)
        return df

    def _drop_id_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop the 'id' column if it exists."""
        logging.info("Dropping 'id' column if present")
        drop_col = self._schema_config['drop_columns']
        if drop_col in df.columns:
            df = df.drop(drop_col, axis=1)
        return df

    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Applies all preprocessing steps and returns transformed train/test data.
        """
        try:
            logging.info("Starting data transformation")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

        
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]
            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]
            logging.info("Train/Test features and target separated")

            X_train = self._map_gender_column(X_train)
            X_train = self._drop_id_column(X_train)
            X_train = self._create_dummy_columns(X_train)
            X_train = self._rename_columns(X_train)

            X_test = self._map_gender_column(X_test)
            X_test = self._drop_id_column(X_test)
            X_test = self._create_dummy_columns(X_test)
            X_test = self._rename_columns(X_test)
            logging.info("Custom transformations applied to train and test data")

        
            preprocessor = self.get_data_transformer_object()

     
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)
            logging.info("Pipeline transformation applied to train/test data")

            logging.info("Applying SMOTEENN to balance dataset")
            smt = SMOTEENN(sampling_strategy="minority")
            X_train_final, y_train_final = smt.fit_resample(X_train_arr, y_train)
            X_test_final, y_test_final = smt.fit_resample(X_test_arr, y_test)
            logging.info("SMOTEENN applied successfully")

         
            train_arr = np.c_[X_train_final, np.array(y_train_final)]
            test_arr = np.c_[X_test_final, np.array(y_test_final)]
            logging.info("Feature-target concatenation complete")

           
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)
            logging.info("Saved preprocessor and transformed train/test files")

           
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
        except Exception as e:
            logging.exception("Error during data transformation")
            raise MyException(e, sys) from e
