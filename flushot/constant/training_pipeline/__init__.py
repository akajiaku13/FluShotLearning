import os
import sys
import numpy as np
import pandas as pd

"""
Defining common constant variables for training pipeline
"""
TARGET_COLUMNS: list[str] = ['h1n1_vaccine', 'seasonal_vaccine']
PIPELINE_NAME: str = 'FluShot'
ARTIFACT_DIR: str = 'artifacts'
FILE_NAME: str = 'flu_shot_data.csv'

TRAIN_FILE_NAME: str = 'train.csv'
TEST_FILE_NAME: str = 'test.csv'

"""
Data Ingestion related constants starting with 'DATA_INGESTION VAR NAME'
"""

DATA_INGESTION_COLLECTION_NAME: str = 'FluShot'
DATA_INGESTION_DATABASE_NAME: str = 'rebeldb'
DATA_INGESTION_DIR_NAME: str = 'data_ingestion'
DATA_INGESTION_FEATURE_STORE_DIR: str = 'feature_store'
DATA_INGESTION_INGESTED_DIR: str = 'ingested'
DATA_INGESTION_USE_IS_TRAIN_COLUMN: bool = True