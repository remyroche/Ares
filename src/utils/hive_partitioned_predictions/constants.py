"""
Constants and configuration for Hive-partitioned prediction storage.

This module defines the base paths and configuration for the Hive partitioning
system. All storage follows a strict hierarchy without metadata.json files.
"""
from pathlib import Path
from typing import Dict


# Base paths for each layer
BASE_ARTIFACT_PATH = Path("artifacts")

LAYER_PATHS: Dict[str, Path] = {
    "specialists": BASE_ARTIFACT_PATH / "specialists",
    "base_models": BASE_ARTIFACT_PATH / "base_models",
    "disagreement_features": BASE_ARTIFACT_PATH / "disagreement_features",
    "meta_layer": BASE_ARTIFACT_PATH / "meta_layer",
}

# Prediction subdirectories
PREDICTIONS_DIR = "predictions"
MODELS_DIR = "models"

# Parquet compression settings
PARQUET_COMPRESSION = "zstd"
PARQUET_ENGINE = "pyarrow"

# Metadata column prefixes (for filtering/debugging only)
METADATA_COLUMNS = [
    "_prediction_date",
    "_model_version",
    "_write_timestamp",
]

# Monthly consolidation settings
CONSOLIDATION_DAY = 1  # Run on 1st of month
CONSOLIDATION_HOUR = 2  # Run at 02:00 AM UTC

# Lock file name for compaction safety
COMPACTION_LOCK_FILE = ".compaction.lock"

# Supported layers
SUPPORTED_LAYERS = list(LAYER_PATHS.keys())
