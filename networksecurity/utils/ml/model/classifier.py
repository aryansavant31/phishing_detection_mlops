from networksecurity.exceptions.custom_exception import NetworkSecurityException
from sklearn.pipeline import Pipeline
import numpy as np
import sys

class Classifier:
    def __init__(self, preprocessor:Pipeline, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, X:np.ndarray) -> np.ndarray:
        try:
            X_transformed = self.preprocessor.transform(X)
            y_pred = self.model.predict(X_transformed)
            return y_pred
        except Exception as e:
            raise NetworkSecurityException(e, sys)