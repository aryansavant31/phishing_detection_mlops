from dataclasses import dataclass
from pathlib import Path
from box import ConfigBox
from datetime import datetime
import os
from networksecurity.constants import (prediction_pipeline)
from networksecurity.constants.training_pipeline import model_trainer


@dataclass
class PredictionPipelineConfig: # zoomed out perspective
    timestamp:str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    artifact_path:Path = os.path.join(prediction_pipeline.ARTIFACT_DIR_NAME, timestamp)

@dataclass
class DataIngestionConfig:
    root_dir: Path
    collection_name: str
    database_name: str
    raw_input_data_file_path: str
    input_data_file_path: Path
    train_file_path: Path

@dataclass
class DataValidationConfig:
    root_dir: Path
    valid_status_file_path: Path
    drift_report_file_path: Path

@dataclass
class ModelPredictorConfig:
    root_dir: Path
    output_file_path: Path
    final_model_file_path: Path