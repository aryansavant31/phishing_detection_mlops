from pathlib import Path
import sys
from networksecurity.entity.prediction_pipeline.config_entity import (PredictionPipelineConfig, 
                                                                      DataIngestionConfig,
                                                                      DataValidationConfig,
                                                                      ModelPredictorConfig)
from networksecurity.manager.prediction_pipeline.configuration import ConfigurationManager
from networksecurity.exceptions.custom_exception import NetworkSecurityException
from networksecurity.components.data_ingestion.predict import DataIngestionComponent
from networksecurity.components.data_validation.predict import DataValidationComponent
from networksecurity.components.model_predictor import ModelPredictorComponent
from networksecurity.entity.prediction_pipeline.artifact_entity import (DataIngestionArtifact, DataValidationArtifact)
from networksecurity.cloud.s3_syncer import S3Sync
from networksecurity.constants.prediction_pipeline import AWS_BUCKET_NAME, ARTIFACT_DIR_NAME
import pandas as pd

class PredictionPipeline:
    def __init__(self):
        self.prediction_pipeline_config = PredictionPipelineConfig()
        self.config_manager = ConfigurationManager(self.prediction_pipeline_config)
        self.s3_sync = S3Sync()
    
    def start_data_ingestion(self, input_data_path) -> DataIngestionArtifact:
        try:
            data_ingestion_config = self.config_manager.get_data_ingestion_config()
            if input_data_path:
                data_ingestion_config.raw_input_data_file_path = input_data_path
            data_ingestion_comp = DataIngestionComponent(config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion_comp.initiate_data_ingestion()
            return data_ingestion_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation_config = self.config_manager.get_data_validation_config()
            data_validation_comp = DataValidationComponent(data_ingestion_artifact,
                                                           data_validation_config)
            data_validation_artifact = data_validation_comp.get_data_validation_artifact()
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def start_model_predictor(self, data_validation_artifact:DataValidationArtifact) -> pd.DataFrame:
        try:
            model_predictor_config = self.config_manager.get_model_predictor_config()
            model_predictor_comp = ModelPredictorComponent(data_validation_artifact,
                                                           model_predictor_config)
            output_data =  model_predictor_comp.initiate_model_prediction()
            return output_data
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{AWS_BUCKET_NAME}/{ARTIFACT_DIR_NAME}/{self.prediction_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(
                local_folder=self.prediction_pipeline_config.artifact_path,
                aws_bucket_url=aws_bucket_url
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_batch_prediction(self, input_data_path:Path=None) -> pd.DataFrame:
        try:
            data_ingestion_artifact = self.start_data_ingestion(input_data_path)
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            output_data = self.start_model_predictor(data_validation_artifact)

            # sync prediction artifact to s3 bucket
            self.sync_artifact_dir_to_s3()
            return output_data

        except Exception as e:
            raise NetworkSecurityException(e, sys)