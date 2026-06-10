from dataclasses import dataclass
from pathlib import Path
from box import ConfigBox

@dataclass
class DataIngestionConfig:
    root_dir: Path
    collection_name: str
    database_name: str
    feature_store_file_path: Path
    train_file_path: Path
    test_file_path: Path
    train_test_split_ratio: float

@dataclass
class DataValidationConfig:
    root_dir: Path
    valid_status_file_path: Path
    drift_report_file_path: Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    transformed_train_file_path: Path
    transformed_test_file_path: Path
    transformation_object_file_path: Path
    imputer_params: ConfigBox
    target_col: str

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    trained_model_file_path: Path
    expected_accuracy: float
    final_model_file_path: Path