from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionArtifact:
    test_file_path: Path
    train_file_path: Path

@dataclass
class DataValidationArtifact:
    valid_status: bool
    drift_report_file_path: Path
    input_file_path: Path