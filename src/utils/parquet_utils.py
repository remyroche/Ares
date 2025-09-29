import gc
import os
import shutil
from typing import Any
import pandas as pd
try:
    import numpy as np
except ImportError:
    np = None

from .logger import system_logger

from src.core.decorators import handles_errors
import logging

# src/utils/parquet_utils.py

class ParquetUtils:
    """Utility class for safe parquet file operations with comprehensive error handling."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild("ParquetUtils")

    @handles_errors(
        default_return={"valid": False, "error": "validation_error"},
        context="ParquetUtils.validate_parquet_file",
    )
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
            # Try to read a small sample using basic pandas
            sample_df = pd.read_parquet(file_path)

            result["columns"] = sample_df.columns.tolist()
            result["shape"] = sample_df.shape
            # Convert dtypes to str to ensure JSON-serializable values
            result["dtypes"] = {
                k: str(v) for k, v in sample_df.dtypes.to_dict().items()
            }
            result["valid"] = True
        except Exception as e:  # pragma: no cover - defensive guard
            result["error"] = f"Failed to read parquet file: {e}"
        finally:
            try:
                del sample_df  # type: ignore[name-defined]
            except Exception as e:
                # Log the exception for debugging but don't fail the operation
                logger.warning(f"Failed to delete sample_df: {e}")
            gc.collect()

        return result

    @handles_errors(default_return = None, context="ParquetUtils.safe_read_parquet")
    def safe_read_parquet(
        self,
        file_path: str,
        columns: list[str] | None = None,
        nrows: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | None:
        """
        Safely read a parquet file with multiple fallback strategies and enhanced schema compatibility.

        Args:
            file_path: Path to the parquet file
            columns: List of columns to read
            nrows: Number of rows to read (applied via head after load)
            **kwargs: Additional arguments for pd.read_parquet

        Returns:
            DataFrame if successful, None otherwise
        """
        self.logger.info(f"🔧 Safe reading parquet file: {file_path}")

        # Enhanced strategies with schema compatibility options
        # Note: use_legacy_dataset is deprecated in newer pandas/pyarrow versions
        strategies = [
            {"engine": "pyarrow", "coerce_int96_timestamp_unit": "ms"},
            {"engine": "pyarrow", "coerce_int96_timestamp_unit": "ns"},  # Alternative timestamp unit
            {"engine": "fastparquet"},
            {"engine": None},  # pandas default
        ]
        
        for idx, strategy in enumerate(strategies, start=1):
            try:
                engine = strategy.get("engine")
                strategy_msg = f"   Trying strategy {idx}/{len(strategies)}: {'default' if engine is None else engine} engine"
                if strategy.get("coerce_int96_timestamp_unit"):
                    strategy_msg += f" ({strategy.get('coerce_int96_timestamp_unit')} timestamps)"
                self.logger.info(strategy_msg)
                
                read_kwargs = dict(kwargs)
                read_kwargs.update({k: v for k, v in strategy.items() if k != "engine"})
                
                if engine is not None:
                    read_kwargs["engine"] = engine
                    
                df = pd.read_parquet(file_path, columns=columns, **read_kwargs)
                
                if nrows is not None and len(df) > nrows:
                    df = df.head(nrows)
                    
                # Apply immediate schema harmonization to prevent downstream issues
                df = self._harmonize_schema_immediately(df)
                
                self.logger.info(f"✅ Successfully read with strategy {idx}: {df.shape}")
                return df
                
            except Exception as e:
                self.logger.warning(f"   Strategy {idx} failed: {e}")
                continue

        self.logger.error(f"❌ All strategies failed for file: {file_path}")
        return None

    @handles_errors(default_return = None, context="ParquetUtils.safe_read_parquet_with_dtype_normalization")
    def safe_read_parquet_with_dtype_normalization(
        self,
        file_path: str,
        columns: list[str] | None = None,
        nrows: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | None:
        """
        Safely read a parquet file with dtype normalization after reading.

        Args:
            file_path: Path to the parquet file
            columns: List of columns to read
            nrows: Number of rows to read (applied via head after load)
            **kwargs: Additional arguments for pd.read_parquet

        Returns:
            DataFrame with normalized dtypes if successful, None otherwise
        """
        df = self.safe_read_parquet(file_path, columns, nrows, **kwargs)
        if df is not None:
            # Normalize dtypes after reading
            df = self._normalize_dtypes(df)
            self.logger.info(f"✅ Dtypes normalized after reading: {df.shape}")
        return df

    def _normalize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize dtypes to ensure consistency across different parquet files.
        Handles pyarrow dictionary encoding, categorical data, and mixed dtypes.

        Args:
            df: DataFrame to normalize

        Returns:
            DataFrame with normalized dtypes
        """
        df_normalized = df.copy()

        # Define standard dtype mappings for vulnerable columns
        dtype_mappings = {
            'year': 'int32',  # Normalize year to int32 regardless of source
            'month': 'int32',  # Normalize month to int32
            'day': 'int32',    # Normalize day to int32
            'symbol': 'string',  # Normalize symbol to string
            'exchange': 'string',  # Normalize exchange to string
            'ticker': 'string',   # Normalize ticker to string
        }

        for col, target_dtype in dtype_mappings.items():
            if col in df_normalized.columns:
                try:
                    # Get original dtype for logging
                    original_dtype = str(df_normalized[col].dtype)

                    if target_dtype == 'string':
                        # Handle dictionary/categorical to string conversion
                        if hasattr(df_normalized[col], 'cat'):
                            # Categorical column
                            df_normalized[col] = df_normalized[col].astype('string')
                        elif str(df_normalized[col].dtype).startswith('dictionary'):
                            # PyArrow dictionary encoding - convert to string first
                            df_normalized[col] = df_normalized[col].astype(str).astype('string')
                        else:
                            # Regular conversion
                            df_normalized[col] = df_normalized[col].astype('string')
                    else:
                        # Handle numeric conversions with dictionary decoding
                        if str(df_normalized[col].dtype).startswith('dictionary'):
                            # PyArrow dictionary encoding - convert to numeric
                            try:
                                # First convert to string to decode dictionary
                                temp_series = df_normalized[col].astype(str)
                                # Then convert to target numeric type
                                df_normalized[col] = pd.to_numeric(temp_series, errors='coerce').astype(target_dtype)
                            except Exception:
                                # Fallback: convert to string first, then numeric
                                df_normalized[col] = pd.to_numeric(df_normalized[col].astype(str), errors='coerce').astype(target_dtype)
                        elif hasattr(df_normalized[col], 'cat'):
                            # Categorical column
                            df_normalized[col] = df_normalized[col].astype(target_dtype)
                        else:
                            # Regular conversion
                            df_normalized[col] = df_normalized[col].astype(target_dtype)

                    self.logger.debug(f"✅ Normalized {col} from {original_dtype} to {target_dtype}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not normalize {col} to {target_dtype}: {e}")
                    # Log more details about the column for debugging
                    self.logger.debug(f"   Column info: dtype={df_normalized[col].dtype}, shape={df_normalized[col].shape}, unique_values={df_normalized[col].nunique() if len(df_normalized[col]) > 0 else 'N/A'}")

        return df_normalized

    def _harmonize_schema_immediately(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Immediately harmonize schema to prevent compatibility issues.
        This is a lightweight version of _normalize_dtypes for immediate use.
        
        Args:
            df: DataFrame to harmonize
            
        Returns:
            DataFrame with harmonized schema
        """
        if df is None or df.empty:
            return df
            
        df_harmonized = df.copy()
        
        # Critical schema harmonization for common compatibility issues
        critical_mappings = {
            'year': 'int32',  # Fix int16 vs dictionary<int32> conflicts
            'month': 'string',  # Fix category vs string conflicts
            'symbol': 'string',  # Fix object vs string conflicts
            'exchange': 'string',  # Fix object vs string conflicts
        }
        
        for col, target_dtype in critical_mappings.items():
            if col in df_harmonized.columns:
                try:
                    original_dtype = str(df_harmonized[col].dtype)
                    
                    # Handle dictionary encoding conflicts (main cause of the error)
                    if str(df_harmonized[col].dtype).startswith('dictionary'):
                        if target_dtype == 'string':
                            df_harmonized[col] = df_harmonized[col].astype(str).astype('string')
                        else:
                            # Convert dictionary to numeric
                            temp_series = df_harmonized[col].astype(str)
                            df_harmonized[col] = pd.to_numeric(temp_series, errors='coerce').astype(target_dtype)
                    elif hasattr(df_harmonized[col], 'cat'):
                        # Handle categorical conflicts
                        if target_dtype == 'string':
                            df_harmonized[col] = df_harmonized[col].astype('string')
                        else:
                            df_harmonized[col] = df_harmonized[col].astype(target_dtype)
                    else:
                        # Regular conversion
                        df_harmonized[col] = df_harmonized[col].astype(target_dtype)
                        
                    self.logger.debug(f"🔧 Harmonized {col}: {original_dtype} -> {target_dtype}")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not harmonize {col}: {e}")
                    
        return df_harmonized

    @handles_errors(default_return = False, context="ParquetUtils.repair_parquet_file")
    def repair_parquet_file(
        self, file_path: str, backup_path: str | None = None
    ) -> bool:
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
        df = self.safe_read_parquet(file_path)
        if df is not None:
            # Write back to the same file
            df.to_parquet(file_path, index=False)
            self.logger.info(f"✅ Successfully repaired parquet file: {file_path}")
            return True

        self.logger.error(f"❌ Could not read file for repair: {file_path}")
        return False

    @handles_errors(default_return=None, context="ParquetUtils.harmonize_schema_after_read")
    def harmonize_schema_after_read(self, df: pd.DataFrame, schema_reference: dict[str, str] | None = None) -> pd.DataFrame | None:
        """
        Harmonize DataFrame schema immediately after reading from parquet to prevent schema incompatibilities.

        This method addresses common parquet schema issues:
        - Inconsistent dtypes for the same logical column (e.g., year as int16 vs dictionary<int32>)
        - Mixed categorical encodings
        - Timestamp format inconsistencies

        Args:
            df: DataFrame read from parquet
            schema_reference: Optional reference schema with column -> dtype mappings

        Returns:
            DataFrame with harmonized schema, or None if harmonization fails
        """
        if df is None or df.empty:
            return df

        try:
            harmonized_df = df.copy()
            harmonization_log = []

            # Handle year column specifically (most common issue)
            if 'year' in harmonized_df.columns:
                original_dtype = harmonized_df['year'].dtype
                try:
                    # Force year to consistent integer type
                    harmonized_df['year'] = harmonized_df['year'].astype('int32')
                    harmonization_log.append(f"year: {original_dtype} -> int32")
                    self.logger.debug(f"✅ Harmonized year column: {original_dtype} -> int32")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not harmonize year column: {e}")
                    # Try fallback to int16
                    try:
                        harmonized_df['year'] = harmonized_df['year'].astype('int16')
                        harmonization_log.append(f"year: {original_dtype} -> int16 (fallback)")
                    except Exception as e2:
                        self.logger.error(f"❌ Could not harmonize year column even with fallback: {e2}")

            # Handle other categorical/dictionary columns
            categorical_cols = ['symbol', 'ticker', 'month', 'exchange', 'pair', 'asset']
            for col in categorical_cols:
                if col in harmonized_df.columns:
                    original_dtype = harmonized_df[col].dtype
                    try:
                        # Convert dictionary/categorical encodings to string for consistency
                        if str(original_dtype).startswith('dictionary') or str(original_dtype).startswith('category'):
                            harmonized_df[col] = harmonized_df[col].astype('string')
                            harmonization_log.append(f"{col}: {original_dtype} -> string")
                            self.logger.debug(f"✅ Harmonized {col}: {original_dtype} -> string")
                        elif harmonized_df[col].dtype == 'object':
                            # Ensure object columns are consistently string
                            harmonized_df[col] = harmonized_df[col].astype('string')
                            harmonization_log.append(f"{col}: {original_dtype} -> string")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not harmonize {col}: {e}")

            # Handle timestamp columns
            timestamp_cols = [col for col in harmonized_df.columns
                            if 'timestamp' in col.lower() or 'time' in col.lower() or 'date' in col.lower()]
            for col in timestamp_cols:
                if col in harmonized_df.columns:
                    original_dtype = harmonized_df[col].dtype
                    try:
                        # Ensure consistent datetime format
                        if not pd.api.types.is_datetime64_any_dtype(harmonized_df[col]):
                            harmonized_df[col] = pd.to_datetime(harmonized_df[col], errors='coerce')
                            harmonization_log.append(f"{col}: {original_dtype} -> datetime64")
                            self.logger.debug(f"✅ Harmonized {col}: {original_dtype} -> datetime64")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not harmonize timestamp column {col}: {e}")

            # Optimize numeric dtypes where safe
            if np is None:
                self.logger.warning("⚠️ NumPy not available, skipping numeric dtype optimization")
                numeric_cols = harmonized_df.select_dtypes(include=['number']).columns
            else:
                numeric_cols = harmonized_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col == 'year':  # Skip year as we already handled it
                    continue

                original_dtype = harmonized_df[col].dtype
                try:
                    if harmonized_df[col].dtype == 'float64':
                        # Check if float32 is sufficient (preserves precision for most financial data)
                        col_min, col_max = harmonized_df[col].min(), harmonized_df[col].max()
                        if col_min >= -3.4e38 and col_max <= 3.4e38:
                            harmonized_df[col] = harmonized_df[col].astype('float32')
                            harmonization_log.append(f"{col}: {original_dtype} -> float32")
                    elif harmonized_df[col].dtype == 'int64':
                        # Check if smaller int type is sufficient
                        col_min, col_max = harmonized_df[col].min(), harmonized_df[col].max()
                        if col_min >= -32768 and col_max <= 32767:
                            harmonized_df[col] = harmonized_df[col].astype('int16')
                            harmonization_log.append(f"{col}: {original_dtype} -> int16")
                        elif col_min >= -2147483648 and col_max <= 2147483647:
                            harmonized_df[col] = harmonized_df[col].astype('int32')
                            harmonization_log.append(f"{col}: {original_dtype} -> int32")
                except Exception as e:
                    self.logger.debug(f"⚠️ Could not optimize dtype for {col}: {e}")

            # Log harmonization summary
            if harmonization_log:
                self.logger.info(f"✅ Schema harmonized: {', '.join(harmonization_log)}")
            else:
                self.logger.debug("ℹ️ No schema harmonization needed")

            return harmonized_df

        except Exception as e:
            self.logger.error(f"❌ Schema harmonization failed: {e}")
            return df

    def safe_read_parquet_with_harmonization(self, file_path: str, harmonize_schema: bool = True,
                                           schema_reference: dict[str, str] | None = None) -> pd.DataFrame | None:
        """
        Read parquet file with automatic schema harmonization.

        Args:
            file_path: Path to parquet file
            harmonize_schema: Whether to harmonize schema after reading
            schema_reference: Optional reference schema

        Returns:
            DataFrame with harmonized schema, or None if reading fails
        """
        # First read the file
        df = self.safe_read_parquet(file_path)

        if df is not None and harmonize_schema:
            df = self.harmonize_schema_after_read(df, schema_reference)

        return df

def get_parquet_utils() -> ParquetUtils:
    """Get a fresh instance of ParquetUtils to avoid global state issues."""
    return ParquetUtils()
