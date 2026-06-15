import os
from networksecurity.constants import (prediction_pipeline)
from networksecurity.constants.prediction_pipeline import data_ingestion, data_validation, model_predictor
from networksecurity.utils.common import read_yaml, create_directories
from networksecurity.entity.prediction_pipeline.config_entity import (DataIngestionConfig, DataValidationConfig, 
                                                  ModelPredictorConfig, PredictionPipelineConfig)

from datetime import datetime

class ConfigurationManager:
    def __init__(self, 
                 prediction_pipeline_config:PredictionPipelineConfig,
                 ):
        self.prediction_pipeline_config = prediction_pipeline_config
        self.artifact_root = self.prediction_pipeline_config.artifact_path
        create_directories(paths_to_dir=[self.artifact_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Assign the DataIngestionConfig attribute
        """
        # make paths
        data_ingestion_root = os.path.join(self.artifact_root, data_ingestion.ROOT_DIR)
        input_data_file_path = os.path.join(data_ingestion_root, 
                                      data_ingestion.INGESTED_DIR, 
                                      data_ingestion.INPUT_FILE_NAME)
        raw_input_data_file_path = os.path.join(data_ingestion.RAW_INPUT_DIR,
                                                data_ingestion.RAW_INPUT_FILE_NAME)

        train_file_path = os.path.join(data_ingestion_root, 
                                       data_ingestion.INGESTED_DIR, 
                                       data_ingestion.TRAIN_FILE_NAME)

        data_ingestion_config = DataIngestionConfig(
            root_dir = data_ingestion_root,
            database_name = data_ingestion.DATABASE_NAME, 
            collection_name = data_ingestion.COLLECTION_NAME,
            input_data_file_path = input_data_file_path,
            train_file_path = train_file_path,
            raw_input_data_file_path = raw_input_data_file_path
        )
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Assign data validation config attributes
        """
        # make paths
        data_validation_root = os.path.join(self.artifact_root, data_validation.ROOT_DIR)
        valid_status_file_path = os.path.join(data_validation_root, 
                                              data_validation.VALID_STATUS_DIR, 
                                              data_validation.VALID_STATUS_FILE_NAME)
        drift_report_file_path = os.path.join(data_validation_root, 
                                              data_validation.DRIFT_REPORT_DIR, 
                                              data_validation.DRIFT_REPORT_FILE_NAME)

        data_validation_config = DataValidationConfig(
            root_dir=data_validation_root,
            valid_status_file_path = valid_status_file_path,
            drift_report_file_path = drift_report_file_path
        )
        return data_validation_config

    
    def get_model_predictor_config(self) -> ModelPredictorConfig:
        """
        Assign the ModelPredictorConfig attribute
        """
        # make paths
        model_predictor_root = os.path.join(self.artifact_root, model_predictor.ROOT_DIR)
        output_file_path = os.path.join(model_predictor_root, model_predictor.PREDICTION_FILE_NAME)
        final_model_file_path = os.path.join(model_predictor.FINAL_MODEL_DIR, model_predictor.FINAL_MODEL_NAME)
        
        model_predictor_config = ModelPredictorConfig(
            root_dir = model_predictor_root,
            output_file_path = output_file_path,
            final_model_file_path = final_model_file_path
        )
        return model_predictor_config