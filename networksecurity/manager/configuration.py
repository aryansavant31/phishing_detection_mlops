import os
from networksecurity.constants import (general, data_ingestion, data_validation, 
                                       data_transformation, model_trainer)
from networksecurity.utils.common import read_yaml, create_directories
from networksecurity.entity.config_entity import (DataIngestionConfig, DataValidationConfig, 
                                                  DataTransformationConfig, ModelTrainerConfig)
from datetime import datetime

class ConfigurationManager:
    def __init__(self, 
                 config_path=general.CONFIG_FILE_PATH,
                 timestamp=datetime.now()
                 ):
        self.config = read_yaml(path_to_yaml=config_path)
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.artifact_root = os.path.join(general.ARTIFACT_DIR_NAME, timestamp)
        create_directories(paths_to_dir=[self.artifact_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Assign the DataIngestionConfig attribute
        """
        config = self.config.data_ingestion

        # make paths
        data_ingestion_root = os.path.join(self.artifact_root, data_ingestion.ROOT_DIR)
        feature_store_file_path = os.path.join(data_ingestion_root, 
                                               data_ingestion.FEATURE_STORE_DIR, 
                                               data_ingestion.RAW_DATA_FILE_NAME)
        train_file_path = os.path.join(data_ingestion_root, 
                                       data_ingestion.INGESTED_DIR, 
                                       data_ingestion.TRAIN_FILE_NAME)
        test_file_path = os.path.join(data_ingestion_root, 
                                      data_ingestion.INGESTED_DIR, 
                                      data_ingestion.TEST_FILE_NAME)

        data_ingestion_config = DataIngestionConfig(
            root_dir = data_ingestion_root,
            database_name = data_ingestion.DATABASE_NAME, 
            collection_name = data_ingestion.COLLECTION_NAME,
            feature_store_file_path = feature_store_file_path,
            train_file_path = train_file_path,
            test_file_path = test_file_path,
            train_test_split_ratio = config.train_test_split_ratio
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
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Assign the DataTransformationConfig attribute
        """
        config = self.config.data_transformation

        # make paths
        data_transformation_root = os.path.join(self.artifact_root, data_transformation.ROOT_DIR)
        transformed_train_file_path = os.path.join(data_transformation_root, 
                                                   data_transformation.TRANSFORMED_DATA_DIR,
                                                   data_transformation.TRANSFORMED_TRAIN_FILE_NAME)
        transformed_test_file_path = os.path.join(data_transformation_root, 
                                                   data_transformation.TRANSFORMED_DATA_DIR,
                                                   data_transformation.TRANSFORMED_TEST_FILE_NAME)
        transformation_object_file_path = os.path.join(data_transformation_root, 
                                                   data_transformation.TRANSFORMATION_OBJECT_DIR,
                                                   data_transformation.PREPROCESSING_OBJECT_FILE_NAME)

        data_transformation_config = DataTransformationConfig(
            root_dir = data_transformation_root,
            transformed_train_file_path = transformed_train_file_path,
            transformed_test_file_path = transformed_test_file_path,
            transformation_object_file_path = transformation_object_file_path,
            imputer_params = config.imputer_params,
            target_col = data_transformation.TARGET_COLUMN
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Assign the ModelTrainerConfig attribute
        """
        config = self.config.model_trainer
        # make paths
        model_trainer_root = os.path.join(self.artifact_root, model_trainer.ROOT_DIR)
        trained_model_file_path = os.path.join(model_trainer_root, 
                                               model_trainer.TRAINED_MODEL_DIR,
                                               model_trainer.TRAINED_MODEL_FILE_NAME)
        model_trainer_config = ModelTrainerConfig(
            root_dir = model_trainer_root,
            trained_model_file_path = trained_model_file_path,
            expected_accuracy = config.expected_accuracy,
            final_model_file_path = model_trainer.FINAL_MODEL_PATH
        )
        return model_trainer_config