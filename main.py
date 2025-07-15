import sys
from flushot.components.data_ingestion import DataIngestion
from flushot.components.data_validation import DataValidation
from flushot.components.data_transformation import DataTransformation
from flushot.components.model_trainer import ModelTrainer

from flushot.entity.config_entity import TrainingPipelineConfig

from flushot.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from flushot.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact

from flushot.exception.exception import FluShotException
from flushot.logging.logger import logging

if __name__=='__main__':
    try:
        trainingpipelineconfig = TrainingPipelineConfig()

        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info('Initiating Data Ingestion')
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        logging.info('Data Initiation Completed')

        
        datavalidationconfig = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact, datavalidationconfig)
        logging.info('Initiating Data validation')
        datavalidationartifact = data_validation.initiate_data_validation()
        logging.info('Data Validation Completed')

        datatransformationconfig = DataTransformationConfig(trainingpipelineconfig)
        data_transformation = DataTransformation(datavalidationartifact, datatransformationconfig)
        logging.info('Initiating data transformation')
        datatransformationartifact = data_transformation.initiate_data_transformation()
        logging.info('Data transformation completed')

        logging.info('Model Training started')
        model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=datatransformationartifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info('Model Training completed')
    except Exception as e:
        raise FluShotException(e, sys)