from networksecurity.components.data_ingestion.train import DataIngestionComponent
from networksecurity.components.data_validation.train import DataValidationComponent
from networksecurity.components.data_transformation import DataTransformationComponent
from networksecurity.components.model_trainer import ModelTrainerComponent
from networksecurity.manager.training_pipeline.configuration import ConfigurationManager
from networksecurity.logging.logger import logger
from networksecurity.exceptions.custom_exception import NetworkSecurityException
from networksecurity.entity.training_pipeline.config_entity import TrainingPipelineConfig
from networksecurity.entity.training_pipeline.artifact_entity import (DataIngestionArtifact, DataValidationArtifact, 
                                                    DataTransformationArtifact, ModelTrainerArtifact)
from networksecurity.cloud.s3_syncer import S3Sync
from networksecurity.constants.training_pipeline import AWS_BUCKET_NAME, ARTIFACT_DIR_NAME, FINAL_MODEL_DIR
import sys
import os

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.config_manager = ConfigurationManager(training_pipeline_config=self.training_pipeline_config)
        self.s3_sync = S3Sync()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info(">>>>> Starting data ingestion <<<<<")
            data_ingestion_config = self.config_manager.get_data_ingestion_config()
            data_ingestion_comp = DataIngestionComponent(config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion_comp.initiate_data_ingestion()
            logger.info(" >>>>> Data ingestion completed <<<<<")
            return data_ingestion_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact):
        try:
            logger.info(">>>>> Starting data validation <<<<<")
            data_validation_config = self.config_manager.get_data_validation_config()
            data_validation_comp = DataValidationComponent(data_ingestion_artifact, 
                                                           data_validation_config)
            data_validation_artifact = data_validation_comp.get_data_validation_artifact()
            logger.info(">>>>> Data validation completed <<<<<")
            return data_validation_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_data_transformation(self, data_validation_artifact:DataValidationArtifact) -> DataTransformationArtifact:
        try:
            logger.info(">>>>> Starting data transformation <<<<<")
            data_transformation_config = self.config_manager.get_data_transformation_config()
            data_transformation_comp = DataTransformationComponent(data_validation_artifact, 
                                                                   data_transformation_config)
            data_transformation_artifact = data_transformation_comp.initiate_data_transformation()
            logger.info(">>>>> Data transformation completed <<<<<")
            return data_transformation_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_model_trainer(self, data_transformation_artifact:DataTransformationArtifact):
        try:
            logger.info(">>>>> Starting model trainer <<<<<")
            model_trainer_config = self.config_manager.get_model_trainer_config()
            model_trainer_comp = ModelTrainerComponent(data_transformation_artifact,
                                                    model_trainer_config)
            model_trainer_artifact = model_trainer_comp.initiate_model_trainer()
            logger.info(">>>>> Model training completed <<<<<")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{AWS_BUCKET_NAME}/{ARTIFACT_DIR_NAME}/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(
                local_folder=self.training_pipeline_config.artifact_path,
                aws_bucket_url=aws_bucket_url
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def sync_final_model_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{AWS_BUCKET_NAME}/{FINAL_MODEL_DIR}/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(
                local_folder=self.training_pipeline_config.final_model_path,
                aws_bucket_url=aws_bucket_url
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def initiate_training(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)

            # sync to aws cloud
            self.sync_artifact_dir_to_s3()
            self.sync_final_model_dir_to_s3()
        except Exception as e:
            raise NetworkSecurityException(e, sys)
