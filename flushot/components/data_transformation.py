import os
import sys
import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from flushot.exception.exception import FluShotException
from flushot.logging.logger import logging

from flushot.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from flushot.entity.config_entity import DataTransformationConfig

from flushot.utils.main_utils.utils import save_numpy_array_data, save_object

from flushot.constant.training_pipeline import TARGET_COLUMNS, DATA_TRANSFORMATION_IMPUTER_PARAMS

class DataTransformation:
    def __init__(self, data_validation_artifact:DataValidationArtifact, data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise FluShotException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise FluShotException(e, sys)
        
    def get_data_transformer_object(cls) -> Pipeline:
        """
        It initialises a KNNImputer object with the parameters specified in the training_pipeline.py file
        and returns a pipeline object with the KNNImputer object as the first step.

        Args:
            cls: DataTransformation

        Returns:
            A pipeline object
        """
        logging.info('Entered get data transformer object method of transformation classs')
        try:
            imputer:KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f'Initialise KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}')
            processor:Pipeline = Pipeline([('imputer', imputer)])
            return processor
        except Exception as e:
            raise FluShotException(e, sys)
        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info('Starting Data Transformation')
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # train dataframe
            input_feature_train_df = train_df.drop(columns=TARGET_COLUMNS, axis=1)
            target_feature_train_df = train_df[TARGET_COLUMNS]

            # test dataframe
            input_feature_test_df = test_df.drop(columns=TARGET_COLUMNS, axis=1)
            target_feature_test_df = test_df[TARGET_COLUMNS]

            preprocessor = self.get_data_transformer_object()

            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)

            save_object('final_model/preprocessor.pkl', preprocessor_object)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return data_transformation_artifact
        except Exception as e:
            raise FluShotException(e, sys)