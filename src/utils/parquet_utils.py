# src/utils/parquet_utils.py

import os
import gc
import logging
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

from src.utils.logger import system_logger


class ParquetUtils:
    """Utility class for safe parquet file operations with comprehensive error handling."""

    def __init__(self):
        self.logger = system_logger.getChild("ParquetUtils")

    def validate_parquet_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a parquet file and return detailed information about its structure.

        Args:
            file_path: Path to the parquet file

        Returns:
            Dictionary containing validation results and file information
        """
        result = {
            "valid": False,
            "file_exists": False,
            "file_size": 0,
            "error": None,
            "metadata": None,
            "columns": [],
            "shape": None,
            "dtypes": None,
        }

        try:
            # Check if file exists
            if not os.path.exists(file_path):
                result["error"] = f"File does not exist: {file_path}"
                return result

            result["file_exists"] = True
            result["file_size"] = os.path.getsize(file_path)

            # Try to read a small sample using basic pandas
            try:
                # Read just the first few rows to validate structure
                sample_df = pd.read_parquet(file_path)

                result["columns"] = sample_df.columns.tolist()
                result["shape"] = sample_df.shape
                result["dtypes"] = sample_df.dtypes.to_dict()
                result["valid"] = True

                # Clean up
                del sample_df
                gc.collect()

            except Exception as e:
                result["error"] = f"Failed to read parquet file: {e}"

        except Exception as e:
            result["error"] = f"Validation failed: {e}"

        return result

    def safe_read_parquet(
        self,
        file_path: str,
        columns: Optional[List[str]] = None,
        nrows: Optional[int] = None,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """
        Safely read a parquet file with multiple fallback strategies.

        Args:
            file_path: Path to the parquet file
            columns: List of columns to read
            nrows: Number of rows to read
            **kwargs: Additional arguments for pd.read_parquet

        Returns:
            DataFrame if successful, None otherwise
        """
        self.logger.info(f"🔧 Safe reading parquet file: {file_path}")

        # Strategy 1: Basic pandas read_parquet
        try:
            self.logger.info("   Trying strategy 1/3: Basic pandas read_parquet")
            df = pd.read_parquet(file_path, columns=columns, **kwargs)

            if nrows and len(df) > nrows:
                df = df.head(nrows)

            self.logger.info(f"✅ Successfully read with strategy 1: {df.shape}")
            return df
        except Exception as e:
            self.logger.warning(f"   Strategy 1 failed: {e}")

        # Strategy 2: Try with pyarrow engine
        try:
            self.logger.info("   Trying strategy 2/3: PyArrow engine")
            df = pd.read_parquet(file_path, columns=columns, engine="pyarrow", **kwargs)

            if nrows and len(df) > nrows:
                df = df.head(nrows)

            self.logger.info(f"✅ Successfully read with strategy 2: {df.shape}")
            return df
        except Exception as e:
            self.logger.warning(f"   Strategy 2 failed: {e}")

        # Strategy 3: Try with fastparquet engine
        try:
            self.logger.info("   Trying strategy 3/3: Fastparquet engine")
            df = pd.read_parquet(
                file_path, columns=columns, engine="fastparquet", **kwargs
            )

            if nrows and len(df) > nrows:
                df = df.head(nrows)

            self.logger.info(f"✅ Successfully read with strategy 3: {df.shape}")
            return df
        except Exception as e:
            self.logger.warning(f"   Strategy 3 failed: {e}")

        self.logger.error(f"❌ All strategies failed for file: {file_path}")
        return None

    def repair_parquet_file(
        self, file_path: str, backup_path: Optional[str] = None
    ) -> bool:
        """
        Attempt to repair a corrupted parquet file.

        Args:
            file_path: Path to the parquet file
            backup_path: Path to save backup (optional)

        Returns:
            True if repair was successful, False otherwise
        """
        try:
            # Create backup if requested
            if backup_path:
                import shutil

                shutil.copy2(file_path, backup_path)
                self.logger.info(f"📁 Created backup: {backup_path}")

            # Try to read and rewrite the file
            df = self.safe_read_parquet(file_path)
            if df is not None:
                # Write back to the same file
                df.to_parquet(file_path, index=False)
                self.logger.info(f"✅ Successfully repaired parquet file: {file_path}")
                return True
            else:
                self.logger.error(f"❌ Could not read file for repair: {file_path}")
                return False

        except Exception as e:
            self.logger.error(f"❌ Repair failed: {e}")
            return False


def get_parquet_utils() -> ParquetUtils:
    """Get a fresh instance of ParquetUtils to avoid global state issues."""
    return ParquetUtils()
