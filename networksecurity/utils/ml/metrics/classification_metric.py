from sklearn.metrics import f1_score, precision_score, recall_score
from networksecurity.exceptions.custom_exception import NetworkSecurityException
from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
import sys
import numpy as np

def get_classification_metrics(y_true:np.ndarray, y_pred:np.ndarray) -> ClassificationMetricArtifact:
    try:
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        classification_metric_artifact = ClassificationMetricArtifact(
            f1_score = f1, 
            precision = precision, 
            recall = recall
        )
        return classification_metric_artifact
    except Exception as e:
        raise NetworkSecurityException(e, sys)