"""
Thread-safe Hive-partitioned prediction writer.

NO metadata.json = NO race conditions!
Filesystem is the source of truth.
"""
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
import logging
import tempfile
import os

from .constants import (
    LAYER_PATHS,
    PREDICTIONS_DIR,
    PARQUET_COMPRESSION,
    PARQUET_ENGINE,
)


logger = logging.getLogger(__name__)


class HivePartitionedWriter:
    """
    Write predictions to Hive-partitioned structure.

    Thread-safe: Each day/model_version gets its own directory.
    No shared state = no race conditions!

    Directory structure:
        artifacts/{layer}/predictions/
            model_version=v1.2.3/
                year=2025/
                    month=11/
                        day=01/
                            data.parquet  # 96 rows (15m data)

    Features:
    - Atomic writes using temp files + rename
    - Thread-safe directory creation
    - No metadata.json files
    - Automatic metadata columns for debugging
    - ZSTD compression for efficiency
    """

    def __init__(
        self,
        layer_name: str,
        model_version: str,
        base_path: Optional[Path] = None
    ):
        """
        Initialize the Hive-partitioned writer.

        Args:
            layer_name: Layer name (specialists, base_models, meta_layer, etc.)
            model_version: Model version string (e.g., "v1.2.3")
            base_path: Optional base path override (for testing)
        """
        if layer_name not in LAYER_PATHS and base_path is None:
            raise ValueError(
                f"Unknown layer: {layer_name}. "
                f"Must be one of: {list(LAYER_PATHS.keys())}"
            )

        self.layer_name = layer_name
        self.model_version = model_version

        if base_path is not None:
            self.base_path = base_path / PREDICTIONS_DIR
        else:
            self.base_path = LAYER_PATHS[layer_name] / PREDICTIONS_DIR

        logger.debug(
            f"Initialized HivePartitionedWriter for {layer_name}/"
            f"{model_version} at {self.base_path}"
        )

    def write_predictions(
        self,
        df: pd.DataFrame,
        prediction_date: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Write predictions to Hive-partitioned structure.

        Thread-safe: Each day/model_version gets its own directory.
        No shared state = no race conditions!

        Args:
            df: Predictions DataFrame (must have datetime index)
            prediction_date: Date for this prediction batch
            metadata: Optional metadata dict (added as columns)

        Returns:
            Path to written parquet file

        Raises:
            ValueError: If DataFrame is empty or has no datetime index
        """
        if df.empty:
            raise ValueError("Cannot write empty DataFrame")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "DataFrame must have DatetimeIndex. "
                f"Got: {type(df.index)}"
            )

        # Build Hive partition path
        partition_path = self._build_partition_path(prediction_date)

        # Create directory (thread-safe with exist_ok=True)
        partition_path.mkdir(parents=True, exist_ok=True)

        # Add metadata columns for easier filtering and debugging
        df = self._add_metadata_columns(df, prediction_date, metadata)

        # Write with atomic rename
        filepath = partition_path / "data.parquet"
        self._atomic_write(df, filepath)

        logger.info(
            f"✅ Wrote {len(df)} predictions to "
            f"{filepath.relative_to(self.base_path.parent)}"
        )

        return filepath

    def _build_partition_path(self, prediction_date: datetime) -> Path:
        """Build Hive partition path from date."""
        return (
            self.base_path /
            f"model_version={self.model_version}" /
            f"year={prediction_date.year}" /
            f"month={prediction_date.month:02d}" /
            f"day={prediction_date.day:02d}"
        )

    def _add_metadata_columns(
        self,
        df: pd.DataFrame,
        prediction_date: datetime,
        metadata: Optional[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Add metadata columns to DataFrame.

        These columns start with '_' to indicate they're metadata,
        and can be easily filtered out when loading predictions.
        """
        df = df.copy()

        # Standard metadata columns
        df['_prediction_date'] = prediction_date
        df['_model_version'] = self.model_version
        df['_write_timestamp'] = datetime.now()

        # Custom metadata columns
        if metadata:
            for key, value in metadata.items():
                # Prefix with '_' if not already
                col_name = f"_{key}" if not key.startswith("_") else key
                df[col_name] = value

        return df

    def _atomic_write(self, df: pd.DataFrame, filepath: Path) -> None:
        """
        Atomic write: write to temp, then rename.

        POSIX guarantees that rename() is atomic, so this ensures
        readers never see partial writes.

        Args:
            df: DataFrame to write
            filepath: Target file path
        """
        # Create temp file in same directory (for atomic rename)
        temp_fd, temp_path_str = tempfile.mkstemp(
            dir=filepath.parent,
            prefix=".tmp_",
            suffix=".parquet"
        )
        temp_path = Path(temp_path_str)

        try:
            # Close the file descriptor (pandas will open it)
            os.close(temp_fd)

            # Write to temp file
            df.to_parquet(
                temp_path,
                compression=PARQUET_COMPRESSION,
                index=True,
                engine=PARQUET_ENGINE
            )

            # Atomic rename (POSIX guarantees atomicity)
            temp_path.rename(filepath)

        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(
                f"Failed to write predictions to {filepath}: {e}"
            ) from e

    def get_partition_path(self, prediction_date: datetime) -> Path:
        """
        Get the partition path for a given date.

        Useful for checking if predictions exist before writing.
        """
        return self._build_partition_path(prediction_date)

    def partition_exists(self, prediction_date: datetime) -> bool:
        """Check if a partition already exists for a given date."""
        partition_path = self._build_partition_path(prediction_date)
        filepath = partition_path / "data.parquet"
        return filepath.exists()
