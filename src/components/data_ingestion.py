import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.proj1_data import Proj1Data


class DataIngestion:
    
    def __init__(self,data_ingestion_cofig:DataIngestionConfig=DataIngestionConfig()):
        
        
        try:
            self.data_ingestion_cofig=data_ingestion_cofig
        
        except Exception as e:
            raise MyException(e,sys)
        
        
    def export_data_into_feature_store(self)->DataFrame:
        
        try:
            logging.info("Exporting Data from MongoDB")
            myData=Proj1Data()
            dataframe=myData.export_collection_as_dataframe(collection_name=self.data_ingestion_cofig.collection_name)
            logging.info(f"Dataframe fetched with shape {dataframe.shape}")
            feature_stroe_file_path=self.data_ingestion_cofig.feature_store__file_path
            dir_path=os.path.dirname(feature_stroe_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"Saving exported data into {feature_stroe_file_path}")
            dataframe.to_csv(feature_stroe_file_path,index=False,header=True)
            return dataframe
        
        except Exception as e:
            raise MyException(e,sys)
        
    def split_train_test_data(self,dataframe:DataFrame)->None:
        
        try:
            train_set,test_set=train_test_split(dataframe,test_size=self.data_ingestion_cofig.train_test_split_ratio)
            logging.info("Preformed training test split in dataframe")
            dir_path=os.path.dirname(self.data_ingestion_cofig.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            train_set.to_csv(self.data_ingestion_cofig.training_file_path,index=True,header=True)
            test_set.to_csv(self.data_ingestion_cofig.testing_file_path,index=True,header=True)
            
            logging.info("Exported training and testoing data")
            
        except Exception as e:
            raise MyException(e,sys)
    
    def initiate_data_ingestion(self)->DataIngestionArtifact:
        
        try:
            dataframe=self.export_data_into_feature_store()

            logging.info("Got the data from MongoDB")
            
            self.split_train_test_data(dataframe)
            
            data_ingestion_artifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_cofig.training_file_path,
                                                          test_file_path=self.data_ingestion_cofig.testing_file_path
                                                          )
            
            logging.info(f"Data ingestion artifact:{data_ingestion_artifact}")
            
            return data_ingestion_artifact
        
        except Exception as e:
            raise MyException(e,sys)
        