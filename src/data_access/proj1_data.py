import sys 
import pandas
import numpy as np
import pandas as pd
from typing import Optional


from src.exception import MyException
from src.logger import logging
from src.constants import DATABASE_NAME
from src.configurations.mongo_db_connection import MongoDBClient


class Proj1Data:
    """
    Class to export MongoDb records to pandas dataframe
    """
    
    def __init__(self):
        
        
        try:
            self.mongo_client=MongoDBClient(database_name=DATABASE_NAME)
        
        except Exception as e:
            raise MyException(e,sys)
        
    
    def export_collection_as_dataframe(self,collection_name:str,database_name:Optional[str]=None)->pd.DataFrame:
        
        try:
            if database_name is None:
                collection=self.mongo_client.database[collection_name]
            else:
                collection=self.mongo_client[database_name][collection_name]
        
            print("Fetching Data from mongo DB")
            df=pd.DataFrame(list(collection.find()))
            print(f"Data fetched of length {len(df)}")
        
            if "id" in df.columns.to_list():
                df=df.drop(columns=['id'],axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df

        
        except Exception as e:
            raise MyException(e,sys)
                