# src / utils / parquet_utils.py

from src.utils.logger import system_logger
from typing import Any
import os

import shutil
import gc
import pandas as pd

from src.utils.error_handler import handle_file_operations, handle_data_processing_errors

class ParquetUtils:
    pass  # TODO: Add implementation
class ParquetUtils:
class ParquetUtils:
    """Utility class for safe parquet file operations with comprehensive error handling."""

def __init__(self) -> None:
        self.logger, system_logger.getChild("ParquetUtils")

@handle_file_operations(default_return={"valid": False, "error": "validation_error"}, context="ParquetUtils.validate_parquet_file")
def validate_parquet_file(self, file_path: str) -> dict[str, Any]:
        """
Validate a parquet file and return detailed information about its structure.

Args:
            file_path: Path to the parquet file

Returns:
            Dictionary containing validation results and file information
"""
result: dict[str, Any] = {
"valid": False,
"file_exists": False,
"file_size": 0,
"error": None,
"metadata": None,
"columns": [],
"shape": None,
"dtypes": None,
}

# Check if file exists
if not os.path.exists(file_path):
            result["error"] = f"File does not exist: {file_path}"
return result

result["file_exists"] = True
result["file_size"] = os.path.getsize(file_path)

try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
# Try to read a small sample using basic pandas
sample_df, pd.read_parquet(file_path)

result["columns"] = sample_df.columns.tolist()
result["shape"] = sample_df.shape
# Convert dtypes to str to ensure JSON - serializable values
result["dtypes"] = {k: str(v) for k, v in sample_df.dtypes.to_dict().items()}
result["valid"] = True
except Exception as e:  # pragma: no cover - defensive guard
result["error"] = f"Failed to read parquet file: {e}"
finally:
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
del sample_df  # type: ignore[name - defined]
except Exception:
                pass
gc.collect()

return result

@handle_file_operations(default_return = None, context="ParquetUtils.safe_read_parquet")
@handle_data_processing_errors(default_return = None, context="ParquetUtils.safe_read_parquet")
def safe_read_parquet(
self,
file_path: str,
columns: list[str] | None, None,
nrows: int | None, None,
**kwargs: Any,
) -> pd.DataFrame | None:
        """
Safely read a parquet file with multiple fallback strategies.

Args:
            file_path: Path to the parquet file
columns: List of columns to read
nrows: Number of rows to read (applied via head after load)
**kwargs: Additional arguments for pd.read_parquet

Returns:
            DataFrame if successful, None otherwise
"""
self.logger.info(f"🔧 Safe reading parquet file: {file_path}")

# Attempt strategies in order: default engine, pyarrow, fastparquet
engines: list[str | None] = [None, "pyarrow", "fastparquet"]
for idx, engine in enumerate(engines, start = 1):
        try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
strategy_msg = (
f"   Trying strategy {idx}/{len(engines)}: "
f"{'default' if engine is None else engine} engine"
)
self.logger.info(strategy_msg)
read_kwargs, dict(kwargs)
if engine is not None:
                    read_kwargs["engine"] = engine
df, pd.read_parquet(file_path, columns = columns, **read_kwargs)
if nrows is not None and len(df) > nrows:
                    df, df.head(nrows)
self.logger.info(f"✅ Successfully read with strategy {idx}: {df.shape}")
return df
except Exception as e:
        self.logger.warning(f"   Strategy {idx} failed: {e}")
continue

self.logger.error(f"❌ All strategies failed for file: {file_path}")
return None

@handle_file_operations(default_return = False, context="ParquetUtils.repair_parquet_file")
def repair_parquet_file(self, file_path: str, backup_path: str | None, None) -> bool:
        """
Attempt to repair a corrupted parquet file.

Args:
            file_path: Path to the parquet file
backup_path: Path to save backup (optional)

Returns:
            True if repair was successful, False otherwise
"""
# Create backup if requested
if backup_path:
            shutil.copy2(file_path, backup_path)
self.logger.info(f"📁 Created backup: {backup_path}")

# Try to read and rewrite the file
df, self.safe_read_parquet(file_path)
if df is not None:
        # Write back to the same file
df.to_parquet(file_path, index = False)
self.logger.info(f"✅ Successfully repaired parquet file: {file_path}")
return True

self.logger.error(f"❌ Could not read file for repair: {file_path}")
return False

def get_parquet_utils() -> ParquetUtils:
    """Get a fresh instance of ParquetUtils to avoid global state issues."""
return ParquetUtils()
