from networksecurity.constants.training_pipeline.data_validation import SCHEMA_FILE_PATH
from networksecurity.constants.training_pipeline.data_transformation import TARGET_COLUMN
from networksecurity.entity.training_pipeline.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.training_pipeline.config_entity import DataValidationConfig
from networksecurity.logging.logger import logger
from networksecurity.exceptions.custom_exception import NetworkSecurityException
from networksecurity.utils.common import read_yaml, write_yaml
import sys
import pandas as pd
from scipy.stats import ks_2samp
import os

class DataValidationComponentTools:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        self.data_validation_config = data_validation_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self._schema = read_yaml(SCHEMA_FILE_PATH)

    def check_number_of_columns(self, data:pd.DataFrame) -> bool:
        try:
            req_num_cols = len(self._schema.columns)
            current_num_cols = len(data.columns.to_list())
            return True if req_num_cols == current_num_cols else False
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def check_data_drift(self, 
                         base_data:pd.DataFrame, 
                         current_data:pd.DataFrame,
                         threshold:float=0.01) -> bool:
        try:
            drift_report = {}

            for col in base_data.columns:
                base_feature_values = base_data[col]
                current_feature_values = current_data[col]
                feature_distribution = ks_2samp(base_feature_values, current_feature_values)
                if threshold <= feature_distribution.pvalue:
                    is_drift_found = False
                else:
                    is_drift_found = True

                # update the report
                drift_report.update({col: {
                    "p_value": float(feature_distribution.pvalue),
                    "drift_status": is_drift_found
                }})

            # save drift report
            drift_report_dir_path = os.path.dirname(self.data_validation_config.drift_report_file_path)
            os.makedirs(drift_report_dir_path, exist_ok=True)
            write_yaml(self.data_validation_config.drift_report_file_path, drift_report)
            return is_drift_found
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self, train_data, test_data) -> bool:
        try:
            # PERFORM VALIDATION CHECKS
        # 1. check if number of features match schema
            is_train_valid = self.check_number_of_columns(train_data)
            if not is_train_valid:
                logger.info(f"Train data does NOT contain all columns.\n")
            is_test_valid = self.check_number_of_columns(test_data)
            if not is_test_valid:
                logger.info(f"Test data does NOT contain all columns.\n")

        # 2. check for data drift
            is_drift_found = self.check_data_drift(base_data=train_data, current_data=test_data)

            # analyze the validation results
            if is_train_valid and is_test_valid and not is_drift_found:
                valid_status = True
                logger.info("The incoming data is valid")
            else:
                valid_status = False
                logger.info("Incoming data is not valid")

            # save validation status
            validation_report = {
                "is_data_valid": valid_status
            }
            os.makedirs(os.path.dirname(self.data_validation_config.valid_status_file_path), exist_ok=True)
            write_yaml(self.data_validation_config.valid_status_file_path, validation_report)
            return valid_status
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)