from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionArtifact:
    train_file_path: Path
    test_file_path: Path

@dataclass
class DataValidationArtifact:
    valid_status: bool
    drift_report_file_path: Path
    train_file_path: Path
    test_file_path: Path

@dataclass
class DataTransformationArtifact:
    transformation_object_file_path: Path
    transformed_train_file_path: Path
    transformed_test_file_path: Path

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision: float
    recall: float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: Path
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact