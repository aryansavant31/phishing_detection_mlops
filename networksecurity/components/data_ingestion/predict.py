from networksecurity.exceptions.custom_exception import NetworkSecurityException
from networksecurity.logging.logger import logger
from dotenv import load_dotenv
from networksecurity.entity.prediction_pipeline.config_entity import DataIngestionConfig
from networksecurity.entity.prediction_pipeline.artifact_entity import DataIngestionArtifact
from networksecurity.components.data_ingestion import DataIngestionComponentTools
import os
import sys
import pandas as pd

class DataIngestionComponent(DataIngestionComponentTools):
    def __init__(self, config:DataIngestionConfig):
        super().__init__(config)
        self.config = config
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            # import input data (from local in this test case) and base train data
            base_df = self.import_mongodb_collection()
            input_data = pd.read_csv(self.config.raw_input_data_file_path)

            # save base (train) data and input data
            os.makedirs(os.path.dirname(self.config.train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.input_data_file_path), exist_ok=True)
            base_df.to_csv(self.config.train_file_path, index=False)
            input_data.to_csv(self.config.input_data_file_path, index=False)

            data_ingestion_artifact = DataIngestionArtifact(
                test_file_path = self.config.input_data_file_path,
                train_file_path = self.config.train_file_path
            )
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

