import os
import sys
import pandas as pd
import numpy as np
import pymongo

from typing import List
from sklearn.model_selection import train_test_split

from flushot.exception.exception import FluShotException
from flushot.logging.logger import logging

from flushot.entity.config_entity import DataIngestionConfig
from flushot.entity.artifact_entity import DataIngestionArtifact

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_URI")

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise FluShotException(e, sys)
        
    def import_collection_as_dataframe(self):
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)

            collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if '_id' in df.columns.to_list():
                df = df.drop(columns=['_id'], axis=1)

            df.replace({'na': np.nan}, inplace=True)
            return df
        except Exception as e:
            raise FluShotException(e, sys)
    
    def export_data_to_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise FluShotException(e, sys)
        
    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            logging.info('Starting train-test split')

            stratify_col = dataframe['h1n1_vaccine'].astype(str) + "_" + dataframe['seasonal_vaccine'].astype(str)

            # Split based on is_train column
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio, stratify=stratify_col
            )

            logging.info(f"Training set shape: {train_set.shape}")
            logging.info(f"Test set shape: {test_set.shape}")
            logging.info('Exited train-test split logic')

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info('Exporting train and test data to respective files')
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info('Exported train and test data to respective files')
        except Exception as e:
            raise FluShotException(e, sys)
        
    def initiate_data_ingestion(self):
        try:
            dataframe = self.import_collection_as_dataframe()
            dataframe = self.export_data_to_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            dataingestionartifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            return dataingestionartifact
        except Exception as e:
            raise FluShotException(e, sys)