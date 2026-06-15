import sys
import os
from networksecurity.logging.logger import logger
from networksecurity.exceptions.custom_exception import NetworkSecurityException
from networksecurity.entity.training_pipeline.config_entity import ModelTrainerConfig, TrainingPipelineConfig
from networksecurity.entity.training_pipeline.artifact_entity import (ModelTrainerArtifact, ClassificationMetricArtifact, 
                                                    DataTransformationArtifact)
from networksecurity.utils.common import load_numpy_array, read_yaml, load_object, save_object
from networksecurity.utils.ml.metrics.classification_metric import get_classification_metrics
from networksecurity.utils.ml.model.model_factory import MODELS
from networksecurity.utils.ml.model.classifier import Classifier
from networksecurity.constants.training_pipeline.model_trainer import MODEL_HYPERPARAMS_FILE_PATH
from networksecurity.utils.ml.training.model_selection import get_best_model
import mlflow
import dagshub
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_URI_DAGSHUB")
REPO_OWNER = os.getenv("REPO_OWNER")

dagshub.init(repo_owner=REPO_OWNER, repo_name='network_security', mlflow=True)

class ModelTrainerComponent:
    def __init__(self, 
                 data_transformation_artifact: DataTransformationArtifact, 
                 model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def create_and_save_classifier(self, model):
        try:
            preprocessor = load_object(self.data_transformation_artifact.transformation_object_file_path)
            classifier = Classifier(preprocessor, model)
            # save classifier as pickle
            save_object(self.model_trainer_config.trained_model_file_path, classifier)
            save_object(TrainingPipelineConfig().final_model_path, classifier)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def get_classification_metrics_of_all_models(self, X_test, y_test, model_report:dict):
        try:
            metric_report = {}
            for model_name in model_report.keys():
                model = model_report[model_name]["model"]
                y_pred_test = model.predict(X_test)
                classification_metric = get_classification_metrics(y_test, y_pred_test)
                metric_report[model_name] = classification_metric
            return metric_report
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def track_experiment_with_mlflow(self, metric_report:dict[str, ClassificationMetricArtifact], best_model):
        try:
            for model_name, metric in metric_report.items():

                with mlflow.start_run(run_name=model_name + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")):
                    mlflow.set_tag("model_name", model_name)
                    mlflow.log_metric("f1_score", metric.f1_score)
                    mlflow.log_metric("precision", metric.precision)
                    mlflow.log_metric("recall", metric.recall)
            
            mlflow.sklearn.log_model(best_model, "model")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def train_and_test_model(self, X_train, y_train, X_test, y_test):
        try:
            hyperparams = read_yaml(MODEL_HYPERPARAMS_FILE_PATH)
            model_report = get_best_model(
                X_train, y_train, X_test, y_test, MODELS, hyperparams
            )
            # get classification metrics for all models
            metric_report = self.get_classification_metrics_of_all_models(
                X_test, y_test, model_report
            )

            # Find BEST MODEL
            best_model_name = max(
                model_report, key=lambda model_name: model_report[model_name]["score"]
            )
            best_model = model_report[best_model_name]["model"]

            # track all model performance with mlflow
            self.track_experiment_with_mlflow(metric_report, best_model)

            # make classifer (preprocessor + model)
            self.create_and_save_classifier(best_model)

            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)

            # create metric artifacts for best model
            classification_train_metric = get_classification_metrics(y_train, y_pred_train)
            classification_test_metric = get_classification_metrics(y_test, y_pred_test)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path = self.model_trainer_config.trained_model_file_path,
                train_metric_artifact = classification_train_metric,
                test_metric_artifact = classification_test_metric
            )
            logger.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            # load transformed data
            train_arr = load_numpy_array(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array(self.data_transformation_artifact.transformed_test_file_path)

            # split data to features and target
            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            # train and test model
            model_trainer_artifact = self.train_and_test_model(
                X_train, y_train, X_test, y_test
            )

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)