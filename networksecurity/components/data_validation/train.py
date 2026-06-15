from networksecurity.entity.training_pipeline.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.training_pipeline.config_entity import DataValidationConfig
from networksecurity.logging.logger import logger
from networksecurity.exceptions.custom_exception import NetworkSecurityException
import sys
from networksecurity.components.data_validation import DataValidationComponentTools
import pandas as pd

class DataValidationComponent(DataValidationComponentTools):
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        super().__init__(data_ingestion_artifact, data_validation_config)
        self.data_validation_config = data_validation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def get_data_validation_artifact(self) -> DataValidationArtifact:
        try:
            # load train and test data
            train_data = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_data = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            valid_status = self.initiate_data_validation(train_data, test_data)
            data_validation_artifact = DataValidationArtifact(
                    valid_status=valid_status,
                    drift_report_file_path=self.data_validation_config.drift_report_file_path,
                    train_file_path = self.data_ingestion_artifact.train_file_path,
                    test_file_path = self.data_ingestion_artifact.test_file_path
                )
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)