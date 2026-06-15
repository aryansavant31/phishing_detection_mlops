from networksecurity.exceptions.custom_exception import NetworkSecurityException
from networksecurity.logging.logger import logger
import pandas as pd
import numpy as np
import pymongo
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from networksecurity.entity.training_pipeline.config_entity import DataIngestionConfig
from networksecurity.entity.training_pipeline.artifact_entity import DataIngestionArtifact

import os
import sys

load_dotenv()

# get mongodb uri from .env
MONGO_DB_URI = os.getenv("MONGO_DB_URI")

class DataIngestionComponentTools:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        self.config = data_ingestion_config

    def import_mongodb_collection(self) -> pd.DataFrame:
        """
        Read and return mongodb collection as dataframe
        """
        try:
            mongodb_client = pymongo.MongoClient(MONGO_DB_URI)
            collection = mongodb_client[self.config.database_name][self.config.collection_name]
            # convert collection to dataframe
            df = pd.DataFrame(list(collection.find()))

            # remove the id column from collection if exist
            if "_id" in df.columns.to_list():
                df.drop(columns=["_id"], axis=1, inplace=True)

            # replace nan with np.nan
            df.replace(to_replace="nan", value=np.nan, inplace=True)
            logger.info("Imported collection as dataframe from mongodb")
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def export_data_to_feature_store(self, df:pd.DataFrame):
        try:
            feature_store_dir_path = os.path.dirname(self.config.feature_store_file_path)
            os.makedirs(feature_store_dir_path, exist_ok=True)
            df.to_csv(self.config.feature_store_file_path, index=False, header=True)
            logger.info(f"Export raw data into {feature_store_dir_path}")

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def split_into_train_test_data(self, df:pd.DataFrame):
        """
        Split dataset to train and test set and store it as .csv
        """
        try:
            train_data, test_data = train_test_split(df, test_size=self.config.train_test_split_ratio)
            logger.info("Split data into train and test sets")

            # save train, test data
            ingest_data_path = os.path.dirname(self.config.train_file_path)
            os.makedirs(ingest_data_path, exist_ok=True)

            train_data.to_csv(self.config.train_file_path, index=False, header=True)
            test_data.to_csv(self.config.test_file_path, index=False, header=True)
            logger.info(f"Exported train and test data to {ingest_data_path}")

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        db = self.import_mongodb_collection()
        self.export_data_to_feature_store(db)
        self.split_into_train_test_data(db)

        data_ingestion_artifact = DataIngestionArtifact(
            train_file_path=self.config.train_file_path,
            test_file_path=self.config.test_file_path
        )
        return data_ingestion_artifact