from networksecurity.entity.training_pipeline.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from networksecurity.entity.training_pipeline.config_entity import DataTransformationConfig
from networksecurity.exceptions.custom_exception import NetworkSecurityException
from networksecurity.logging.logger import logger
import pandas as pd
import numpy as np
import sys
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from networksecurity.utils.common import save_numpy_array, save_object

class DataTransformationComponent:
    def __init__(self, 
                 data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        self.data_transformation_config = data_transformation_config
        self.data_validation_artifact = data_validation_artifact

    def load_train_test_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            # load train and test data if they are valid
            if self.data_validation_artifact.valid_status is True:
                train_df = pd.read_csv(self.data_validation_artifact.train_file_path)
                test_df = pd.read_csv(self.data_validation_artifact.test_file_path)
                return train_df, test_df
            else:
                raise NetworkSecurityException("Data is NOT valid", sys)
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def split_to_input_features_and_target(self, df:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe data into features (X) and target (y). Then replace -1 in target values with 0
        """
        try:
            X_df = df.drop(columns=[self.data_transformation_config.target_col], axis=-1)
            y_df = df[self.data_transformation_config.target_col]
            y_df = y_df.replace(-1, 0)
            return X_df, y_df
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def get_knn_imputer_object(self) -> Pipeline:
        """
        Initialize KNN imputer and return pipeline object
        """
        try:
            imputer = KNNImputer(**self.data_transformation_config.imputer_params)
            logger.info(f"Initilize KNN imputer with parameters: {self.data_transformation_config.imputer_params}")
            imputer_pipeline = Pipeline([("imputer", imputer)])
            return imputer_pipeline
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            # load train and test data if they are valid
            train_df, test_df = self.load_train_test_data()

        # 1. Seperate features (X) from target (y) in data
            X_train_df, y_train_df = self.split_to_input_features_and_target(train_df)
            X_test_df, y_test_df = self.split_to_input_features_and_target(test_df)

        # 2. Use KNNimputer to handle nan values
            imputer_pipeline = self.get_knn_imputer_object()
            imputer_object = imputer_pipeline.fit(X_train_df)

            transformed_X_train = imputer_object.transform(X_train_df)
            transformed_X_test = imputer_object.transform(X_test_df)

            # combine feature and target
            train_arr = np.c_[transformed_X_train, np.array(y_train_df)]
            test_arr = np.c_[transformed_X_test, np.array(y_test_df)]

            # save numpy data and imputer pipeline object
            save_numpy_array(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array(self.data_transformation_config.transformed_test_file_path, test_arr)
            save_object(self.data_transformation_config.transformation_object_file_path, imputer_object)

            # prepare data transformation artifacts
            data_transformation_artifact = DataTransformationArtifact(
                transformation_object_file_path = self.data_transformation_config.transformation_object_file_path,
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)