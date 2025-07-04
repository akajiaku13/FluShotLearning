import os
import sys
import json
import certifi
import pandas as pd
import numpy as np
import pymongo

from flushot.exception.exception import FluShotException
from flushot.logging.logger import logging

from dotenv import load_dotenv
load_dotenv()

MONGO_URL = os.getenv('MONGO_URI')
ca = certifi.where()

class FluShotExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise FluShotException(e, sys)
    
    def csv_to_json_converter(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise FluShotException(e, sys)
        
    def insert_data_to_mongodb(self, records, database, collection, batch_size=10000):
        try:
            self.database = database
            self.collection = collection
            self.records = records

            self.mongo_client = pymongo.MongoClient(MONGO_URL)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]

            for i in range(0, len(self.records), batch_size):
                batch = self.records[i:i+batch_size]
                self.collection.insert_many(batch)

            return(len(self.records))
        except Exception as e:
            raise FluShotException(e, sys)

if __name__ == '__main__':
    FILE_PATH = "FluShot_Data/flu_shot_data.csv"
    DATABASE = 'rebeldb'
    Collection = 'FluShot'
    flushotobj = FluShotExtract()
    records = flushotobj.csv_to_json_converter(file_path=FILE_PATH)
    no_records = flushotobj.insert_data_to_mongodb(records=records, database=DATABASE, collection=Collection)

    print(no_records)