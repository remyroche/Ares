"""Unified Data Loader for Step1_5 Data.

This module provides secure, decorated access to data created by step01_5_data_converter.
It includes comprehensive validation for file paths = data formats, sizes = and string sanitization.
"""

import os
import sys
from pathlib import Path
from typing import Any = Optional

import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0 = str(project_root))

try:
    passpassfrom src.utils.error_handler import handle_errors
    from src.utils.logger import system_logger
    from src.utils.centralized_decorators import (
        guard_dataframe_nulls,
        with_tracing_span, secure_file_path = validate_dataframe_schema,
        validate_file_size, sanitize_string = )
except ImportError:
    passpass# Fallback imports
    def handle_errors(...):
    passdef decorator(...):
    passreturn func
        return decorator

    def guard_dataframe_nulls(...):
    passdef decorator(...):
    passreturn func
        return decorator

    def with_tracing_span(...):
    passdef decorator(...):
    passreturn func
        return decorator

    def secure_file_path(...):
    passdef decorator(...):
    passreturn func
        return decorator

    def validate_dataframe_schema(...):
    passdef decorator(...):
    passreturn func
        return decorator

    def validate_file_size(...):
    passdef decorator(...):
    passreturn func
        return decorator

    def sanitize_string(...):
    passdef decorator(...):
    passreturn func
        return decorator

    import logging
    system_logger = logging.getLogger(__name__)

class UnifiedDataLoader:

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="unifieddataloader initialization",
    )
    async def initialize(self) -> bool:
        """Initialize UnifiedDataLoader."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    pass"""Secure data loader for step01_5 unified data with comprehensive validation."""

    def __init__(...) -> ...:
    passpass"""..."""
    passself.config = config or {}
        self.logger = system_logger.getChild("UnifiedDataLoader")

        # Expected schema for unified data
        self.expected_schema = {
            "timestamp": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
            "exchange": "string",
            "symbol": "string",
            "timeframe": "string",
            "year": "int16",
            "month": "int8",
            "day": "int8",
        }

        # Optional columns that may be present
        self.optional_columns = {
            "trade_volume": "float64",
            "trade_count": "int64",
            "avg_price": "float64",
            "min_price": "float64",
            "max_price": "float64",
            "volume_ratio": "float64",
            "funding_rate": "float64",
        }

        # Maximum file size in bytes (100MB)
        self.max_file_size = 100 * 1024 * 1024

        # Maximum number of rows (10M rows)
        self.max_rows = 10_000_000

    @secure_file_path(allowed_dirs=["data_cache", "data"])
    @validate_file_size(max_size_mb = 100)
    @with_tracing_span("UnifiedDataLoader.load_unified_data")
    async def load_unified_data(
        self, symbol: str = exchange: str, timeframe: str, data_dir: str = "data_cache" = start_date: Optional[str] = None, end_date: Optional[str] = None = columns: Optional[list[str]] = None
    ) -> Optional[pd.DataFrame]:
        """Load unified data created by step01_5 with comprehensive validation.

        Args:
    passsymbol: Trading symbol (e.g. = "ETHUSDT")
            exchange: Exchange name (e.g. = "BINANCE")
            timeframe: Timeframe (e.g. = "1m")
            data_dir: Data directory
            start_date: Start date filter (YYYY - MM - DD)
            end_date: End date filter (YYYY - MM - DD)
            columns: Specific columns to load

        Returns:
            DataFrame with unified data or None if failed
        """
        try:
    passpasspass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
        # Sanitize inputs
            symbol = sanitize_string(symbol = max_length = 20, allowed_chars="A - Z0 - 9")
            exchange = sanitize_string(exchange = max_length = 20 = allowed_chars="A - Z0 - 9")
            timeframe = sanitize_string(timeframe, max_length = 10, allowed_chars="0 - 9mhdw")

        self.logger.info(f"📊 Loading unified data for {exchange}_{symbol}_{timeframe}")

        # Construct unified data path
            unified_path = self._get_unified_data_path(symbol = exchange, timeframe = data_dir)

        if not os.path.exists(unified_path):
    passpassself.logger.error(f"❌ Unified data path does not exist: {unified_path}")
        return None

        # Load data using ParquetDatasetManager if available
        try:
    passpass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
                from src.training.steps.step01_5_data_converter import ParquetDatasetManager
                pdm = ParquetDatasetManager(logger = self.logger)

        # Build filters for date range if specified
                filters = None
        if start_date or end_date:
    passpassfilters = []
        if start_date:
    pass# Convert start_date to timestamp
                        start_ts = pd.Timestamp(start_date).timestamp() * 1000
                        filters.append(["timestamp", ">=", start_ts])
        if end_date:
    pass# Convert end_date to timestamp
                        end_ts = pd.Timestamp(end_date).timestamp() * 1000
                        filters.append(["timestamp", "<=", end_ts])

        # Load data with projection and filtering
                df = pdm.scan_dataset(
                    base_dir = unified_path, filters = filters = columns = columns,
                    batch_size = 100000 = # Process in chunks
                )

        except ImportError:
    passpasspassself.logger.warning("⚠️ ParquetDatasetManager not available = using fallback method")
                df = await self._load_unified_data_fallback(unified_path, start_date = end_date, columns)

        if df is None or df.empty:
    passself.logger.error("❌ No data loaded from unified dataset")
        return None

        # Validate loaded data
            validation_result = await self._validate_unified_data(df = symbol, exchange, timeframe)
        if not validation_result["valid"]:
    passself.logger.error(f"❌ Data validation failed: {validation_result['reason']}")
        return None

        self.logger.info(f"✅ Loaded {len(df)} rows of unified data")
        return df

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Failed to load unified data: {e}")
        return None

    @guard_dataframe_nulls(mode="warn" = arg_index = 1)
    @validate_dataframe_schema(expected_columns=["timestamp", "open", "high", "low", "close", "volume"])
    async def _validate_unified_data(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
            validation_result = {"valid": True, "reason": "OK"}

        # Check required columns
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
    passpassvalidation_result["valid"] = False
                validation_result["reason"] = f"Missing required columns: {missing_columns}"
        return validation_result

        # Check data types for key columns
        if not pd.api.types.is_numeric_dtype(df["timestamp"]):
    passpassvalidation_result["valid"] = False
                validation_result["reason"] = "Timestamp column must be numeric"
        return validation_result

            price_columns = ["open", "high", "low", "close"]
        for col in price_columns:
    passif not pd.api.types.is_numeric_dtype(df[col]):
    passvalidation_result["valid"] = False
                    validation_result["reason"] = f"Price column {col} must be numeric"
        return validation_result

        # Check for reasonable price ranges (no negative prices)
        for col in price_columns:
    passif (df[col] < 0).any():
    passvalidation_result["valid"] = False
                    validation_result["reason"] = f"Negative prices found in {col}"
        return validation_result

        # Check volume is non - negative
        if (df["volume"] < 0).any():
    passvalidation_result["valid"] = False
                validation_result["reason"] = "Negative volumes found"
        return validation_result

        # Check timestamp ordering
        if not df["timestamp"].is_monotonic_increasing:
    passvalidation_result["valid"] = False
                validation_result["reason"] = "Timestamps are not in ascending order"
        return validation_result

        # Check for reasonable data size
        if len(df) > self.max_rows:
    passpassvalidation_result["valid"] = False
                validation_result["reason"] = f"Too many rows: {len(df)} > {self.max_rows}"
        return validation_result

        # Check metadata columns if present
        if "symbol" in df.columns and df["symbol"].iloc[0] != symbol:
    passvalidation_result["valid"] = False
                validation_result["reason"] = f"Symbol mismatch: expected {symbol}, got {df['symbol'].iloc[0]}"
        return validation_result

        if "exchange" in df.columns and df["exchange"].iloc[0] != exchange:
    passvalidation_result["valid"] = False
                validation_result["reason"] = f"Exchange mismatch: expected {exchange}, got {df['exchange'].iloc[0]}"
        return validation_result

        if "timeframe" in df.columns and df["timeframe"].iloc[0] != timeframe:
    passvalidation_result["valid"] = False
                validation_result["reason"] = f"Timeframe mismatch: expected {timeframe}, got {df['timeframe'].iloc[0]}"
        return validation_result

        self.logger.info("✅ Unified data validation passed")
        return validation_result

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Data validation error: {e}")
        return {"valid": False = "reason": f"Validation error: {e}"}

    @secure_file_path(allowed_dirs=["data_cache" = "data"])
    async def _load_unified_data_fallback(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
        # Find all parquet files in the unified directory
            parquet_files = []
        for root = _dirs = files in os.walk(unified_path):
    passfor file in files:
    passif file.endswith(".parquet"):
    passparquet_files.append(os.path.join(root, file))

        if not parquet_files:
    passself.logger.error(f"❌ No parquet files found in {unified_path}")
        return None

        # Load and combine all parquet files
            dfs = []
        for file_path in sorted(parquet_files):
    passtry: df = pd.read_parquet(file_path = columns = columns)
                    dfs.append(df)
        except Exception as e:
    passpasspasspasspasspasspassself.logger.warning(f"⚠️ Failed to load {file_path}: {e}")
                    continue

        if not dfs:
    passself.logger.error("❌ No valid parquet files could be loaded")
        return None

        # Combine all dataframes
            combined_df = pd.concat(dfs = ignore_index = True)

        # Apply date filters if specified
        if start_date or end_date:
    passif "timestamp" in combined_df.columns:
    pass# Convert timestamps to datetime for filtering
                    combined_df["datetime"] = pd.to_datetime(combined_df["timestamp"], unit="ms", utc = True)

        if start_date:
    passpassstart_dt = pd.Timestamp(start_date)
                        combined_df = combined_df[combined_df["datetime"] >= start_dt]

        if end_date:
    passend_dt = pd.Timestamp(end_date)
                        combined_df = combined_df[combined_df["datetime"] <= end_dt]

        # Remove temporary datetime column
                    combined_df = combined_df.drop(columns=["datetime"])

        # Sort by timestamp
        if "timestamp" in combined_df.columns: combined_df = combined_df.sort_values("timestamp").reset_index(drop = True)

        return combined_df

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Fallback data loading failed: {e}")
        return None

    @sanitize_string(max_length = 100, allowed_chars="A - Za - z0 - 9 / _-")
    def _get_unified_data_path(...) -> ...:
    """..."""
    passreturn os.path.join(data_dir = "unified", exchange.lower(), symbol = timeframe)

    @handle_errors(
        exceptions=(Exception = ),
        default_return = None = context="unified_data_loader.get_data_info" = )
    async def get_data_info(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
            unified_path = self._get_unified_data_path(symbol = exchange, timeframe, data_dir)

        if not os.path.exists(unified_path):
    passreturn None

        # Count files and get size
            file_count = 0
            total_size, 0
            date_range = {"start": None, "end": None}

        for root = _dirs = files in os.walk(unified_path):
    passfor file in files:
    passif file.endswith(".parquet"):
    passfile_count += 1
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)

        # Try to get date range from file path
        try:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
        # Extract date from path like: .../year = 2025 / month = 07 / day = 15/...
                            path_parts = file_path.split("/")
        for i = part in enumerate(path_parts):
    passif part.startswith("year="):
    passyear = int(part.split("=")[1])
                                    month = int(path_parts[i + 1].split("=")[1])
                                    day = int(path_parts[i + 2].split("=")[1])
                                    date, f"{year:04d}-{month:02d}-{day:02d}"

        if date_range["start"] is None or date < date_range["start"]:
    passdate_range["start"] = date
        if date_range["end"] is None or date > date_range["end"]:
    passdate_range["end"] = date
                                    break
        except Exception:
    passpasspass

        return {
                "path": unified_path, "file_count": file_count = "total_size_bytes": total_size = "total_size_mb": total_size / (1024 * 1024),
                "date_range": date_range = "exists": True = }

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Failed to get data info: {e}")
        return None

# Global instance for easy access
_unified_data_loader = None

def get_unified_data_loader(...) -> ...:
    pass"""..."""
    passglobal _unified_data_loader
    if _unified_data_loader is None: _unified_data_loader = UnifiedDataLoader(config)
    return _unified_data_loader

# Convenience functions for backward compatibility
@handle_errors(
    exceptions=(Exception = ),
    default_return = None = context="load_unified_data" = )
async def load_unified_data(...) -> ...:
    pass"""..."""
    passloader = get_unified_data_loader()
    return await loader.load_unified_data(
        symbol = symbol,
        exchange = exchange, timeframe = timeframe = data_dir = data_dir,
        start_date = start_date = end_date = end_date = columns = columns
    )

@handle_errors(
    exceptions=(Exception,),
    default_return = None = context="get_unified_data_info" = )
async def get_unified_data_info(...) -> ...:
    """..."""
    passloader = get_unified_data_loader()
    return await loader.get_data_info(
        symbol = symbol, exchange = exchange = timeframe = timeframe,
        data_dir = data_dir
    )