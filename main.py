import sys
from flushot.components.data_ingestion import DataIngestion
from flushot.components.data_validation import DataValidation

from flushot.entity.config_entity import TrainingPipelineConfig

from flushot.entity.config_entity import DataIngestionConfig, DataValidationConfig
from flushot.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

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
    except Exception as e:
        raise FluShotException(e, sys)