from networksecurity.exceptions.custom_exception import NetworkSecurityException
from networksecurity.logging.logger import logger
from dotenv import load_dotenv
from networksecurity.entity.training_pipeline.config_entity import DataIngestionConfig
from networksecurity.entity.training_pipeline.artifact_entity import DataIngestionArtifact
from networksecurity.components.data_ingestion import DataIngestionComponentTools
import os
import sys

class DataIngestionComponent(DataIngestionComponentTools):
    def __init__(self, config:DataIngestionConfig):
        super().__init__(config)
        self.config = config
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            db = self.import_mongodb_collection()
            self.export_data_to_feature_store(db)
            self.split_into_train_test_data(db)

            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.config.train_file_path,
                test_file_path=self.config.test_file_path
            )
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

