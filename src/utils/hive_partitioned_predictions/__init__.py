"""
Production-Grade Hive Partitioning for Prediction Storage.

This module provides a complete Hive-partitioned storage system for
ML predictions with no metadata.json files. The filesystem is the
source of truth.

Directory Structure:
    artifacts/{layer}/predictions/
        model_version=v1.2.3/
            year=2025/
                month=11/
                    day=01/
                        data.parquet  # Daily predictions
                    monthly_consolidated.parquet  # Monthly rollup

Features:
- Thread-safe atomic writes
- No metadata.json = no race conditions
- Smart monthly consolidation fallback
- 95%+ reduction in file count
- Efficient partition pruning
- Polars support for blazing-fast reads

Usage:
    # Writing predictions
    >>> from src.utils.hive_partitioned_predictions import HivePartitionedWriter
    >>> writer = HivePartitionedWriter("specialists", "v1.2.3")
    >>> writer.write_predictions(df, datetime.now())

    # Reading predictions
    >>> from src.utils.hive_partitioned_predictions import HivePartitionedReader
    >>> reader = HivePartitionedReader("specialists")
    >>> df = reader.load_recent_predictions(days=56, model_version="v1.2.3")

    # Monthly compaction
    >>> from src.utils.hive_partitioned_predictions import MonthlyCompactor
    >>> compactor = MonthlyCompactor("specialists")
    >>> stats = compactor.compact_previous_month()

    # Scheduled job (run as cron)
    >>> from src.utils.hive_partitioned_predictions import monthly_compaction_job
    >>> results = monthly_compaction_job()
"""

from .writer import HivePartitionedWriter
from .reader import HivePartitionedReader, PolarsHiveReader
from .compactor import MonthlyCompactor
from .jobs import (
    monthly_compaction_job,
    compact_specific_month,
    backfill_compaction
)
from .constants import (
    SUPPORTED_LAYERS,
    LAYER_PATHS,
    PREDICTIONS_DIR,
    MODELS_DIR
)

__version__ = "1.0.0"

__all__ = [
    # Core classes
    "HivePartitionedWriter",
    "HivePartitionedReader",
    "PolarsHiveReader",
    "MonthlyCompactor",
    # Scheduled jobs
    "monthly_compaction_job",
    "compact_specific_month",
    "backfill_compaction",
    # Constants
    "SUPPORTED_LAYERS",
    "LAYER_PATHS",
    "PREDICTIONS_DIR",
    "MODELS_DIR",
]
