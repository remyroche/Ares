"""Safe parquet helpers with error handling and minimal dependencies."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .logger import get_logger
from .error_handler import handle_file_operations, handle_data_processing_errors

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


class ParquetUtils:
    """Utility class for safe parquet file operations."""

    def __init__(self) -> None:
        self.logger = get_logger("ParquetUtils")

    @handle_file_operations(default_return={"valid": False, "error": "validation_error"}, context="ParquetUtils.validate_parquet_file")
    def validate_parquet_file(self, file_path: str, *, sample_rows: Optional[int] = 10) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "valid": False,
            "file_exists": False,
            "file_size": 0,
            "error": None,
            "columns": [],
            "shape": None,
            "dtypes": None,
        }
        if not os.path.exists(file_path):
            result["error"] = f"File does not exist: {file_path}"
            return result
        result["file_exists"] = True
        result["file_size"] = os.path.getsize(file_path)

        if pd is None:
            result["error"] = "pandas not available"
            return result

        # Small sample read to validate structure
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:  # pragma: no cover
            result["error"] = f"Failed to read parquet: {e}"
            return result

        result["columns"] = list(df.columns)
        result["shape"] = tuple(df.shape)
        # dtypes -> str for JSON friendliness
        result["dtypes"] = {k: str(v) for k, v in df.dtypes.to_dict().items()}
        result["valid"] = True
        return result

    @handle_file_operations(default_return=None, context="ParquetUtils.safe_read_parquet")
    @handle_data_processing_errors(default_return=None, context="ParquetUtils.safe_read_parquet")
    def safe_read_parquet(self, file_path: str, *, columns: Optional[List[str]] = None, **read_kwargs: Any):
        if pd is None:
            self.logger.error("pandas not available; cannot read parquet")
            return None
        df = pd.read_parquet(file_path, columns=columns, **read_kwargs)
        return df

    @handle_file_operations(default_return=False, context="ParquetUtils.repair_parquet_file")
    def repair_parquet_file(self, file_path: str, *, backup_path: Optional[str] = None) -> bool:
        """Naive repair: attempt a read; if it succeeds, optionally rewrite to a clean file.
        Returns True if the file is readable after the operation.
        """
        if pd is None:
            self.logger.error("pandas not available; cannot repair parquet")
            return False
        if not os.path.exists(file_path):
            self.logger.error(f"File does not exist: {file_path}")
            return False
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:  # pragma: no cover
            self.logger.error(f"Failed to read parquet: {e}")
            return False

        # Optionally write a backup or re-write to same path
        if backup_path:
            try:
                df.to_parquet(backup_path, index=False)
                self.logger.info(f"Wrote repaired copy to: {backup_path}")
            except Exception as e:  # pragma: no cover
                self.logger.error(f"Failed to write backup parquet: {e}")
                return False
        else:
            try:
                df.to_parquet(file_path, index=False)
                self.logger.info("Rewrote original parquet with a clean copy")
            except Exception as e:  # pragma: no cover
                self.logger.error(f"Failed to rewrite original parquet: {e}")
                return False
        return True


def get_parquet_utils() -> ParquetUtils:
    return ParquetUtils()
