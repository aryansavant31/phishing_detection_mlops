from dataclasses import dataclass
from pathlib import Path
from box import ConfigBox
from datetime import datetime
import os
from networksecurity.constants import (training_pipeline)


@dataclass
class TrainingPipelineConfig: # zoomed out perspective
    timestamp:str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    artifact_path:Path = os.path.join(training_pipeline.ARTIFACT_DIR_NAME, timestamp)
    final_model_path:Path = os.path.join(training_pipeline.FINAL_MODEL_DIR, training_pipeline.FINAL_MODEL_NAME)

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