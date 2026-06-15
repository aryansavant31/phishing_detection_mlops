from networksecurity.entity.prediction_pipeline.artifact_entity import DataValidationArtifact
from networksecurity.entity.prediction_pipeline.config_entity import ModelPredictorConfig
from networksecurity.exceptions.custom_exception import NetworkSecurityException
import sys
import pandas as pd
from networksecurity.utils.common import load_object
import os

class ModelPredictorComponent:
    def __init__(self, data_validation_artifact:DataValidationArtifact,
                 model_predictor_config:ModelPredictorConfig):
        self.data_validation_artifact = data_validation_artifact
        self.model_predictor_config = model_predictor_config

    def load_input_data(self) -> pd.DataFrame:
        try:
            # load input data if they are valid
            if self.data_validation_artifact.valid_status is True:
                input_df = pd.read_csv(self.data_validation_artifact.input_file_path)
                return input_df
            else:
                raise NetworkSecurityException("Input data is NOT valid", sys)
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_prediction(self) -> pd.DataFrame:
        try:
            # load data
            input_data = self.load_input_data()

            # load model and do prediction
            classifier = load_object(self.model_predictor_config.final_model_file_path)
            y_pred = classifier.predict(input_data)
            input_data["Prediction"] = y_pred

            # save the prediction as csv
            os.makedirs(os.path.dirname(self.model_predictor_config.output_file_path), exist_ok=True)
            input_data.to_csv(self.model_predictor_config.output_file_path)

            return input_data

        except Exception as e:
            raise NetworkSecurityException(e, sys)