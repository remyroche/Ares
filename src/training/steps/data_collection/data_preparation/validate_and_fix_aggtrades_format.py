#!/usr/bin/env python3
from src.utils.logger import system_logger
from ....core.decorators import handles_errors
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

"""Validate and Fix Aggtrades Format for Step1.

Validates and fixes aggtrades data format to ensure compatibility with all pipeline steps.
Enhanced with comprehensive data stability features including memory management,
error recovery, and advanced validation.
"""

import asyncio
import gc
import psutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List
import threading
import time


# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
import pandas as pd

import numpy as np
import logging

logger = system_logger.getChild("AggtradesFormatValidator")

class AggtradesFormatValidator:
    """Validates and fixes aggtrades data format for pipeline compatibility.
    Enhanced with comprehensive stability features."""

    # Expected columns for aggtrades data
    EXPECTED_COLUMNS = [
        "agg_trade_id",
        "price",
        "quantity",
        "first_trade_id",
        "last_trade_id",
        "timestamp",
        "is_buyer_maker",
    ]

    # Expected data types
    EXPECTED_DTYPES = {
        "agg_trade_id": "int64",
        "price": "float64",
        "quantity": "float64",
        "first_trade_id": "int64",
        "last_trade_id": "int64",
        "timestamp": "datetime64[ns]",
        "is_buyer_maker": "bool",
    }

    # Step1_5 specific requirements
    STEP1_5_REQUIREMENTS = {
        "min_timestamp": "2020-01-01",
        "max_timestamp": "2025-12-31",
        "min_rows": 100,
        "max_rows": 10000000,
    }

    # Step2 feature engineering requirements
    STEP2_REQUIREMENTS = {
        "min_price": 0.000001,
        "max_price": 1000000.0,
        "min_quantity": 0.000001,
        "max_quantity": 1000000.0,
    }

    # Step3 regime discovery requirements
    STEP3_REQUIREMENTS = {
        "min_trades_per_day": 100,
        "max_gap_seconds": 3600,  # 1 hour
        "required_time_span_days": 30,
    }

    # Step4 labeling requirements
    STEP4_REQUIREMENTS = {
        "min_labeling_period_hours": 24,
        "max_labeling_period_hours": 168,  # 1 week
        "required_features": ["price", "quantity", "timestamp"],
    }

    # Memory management settings
    MEMORY_SETTINGS = {
        "max_memory_mb": 2048,  # 2GB limit
        "chunk_size_mb": 256,   # Process in 256MB chunks
        "gc_threshold": 1000,   # GC threshold
        "concurrent_workers": 4,  # Number of concurrent workers
    }

    # Error recovery settings
    ERROR_RECOVERY_SETTINGS = {
        "max_retries": 3,
        "retry_delay": 1.0,
        "circuit_breaker_threshold": 5,
        "circuit_breaker_timeout": 60,
    }

    @log_important_calls
    def __init__(self, data_cache_path: str = "data_cache") -> None:
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok = True)

        # Initialize stability components
        self._memory_monitor = MemoryMonitor()
        self._error_recovery = ErrorRecoveryHandler(self.ERROR_RECOVERY_SETTINGS)
        self._concurrent_processor = ConcurrentProcessor(self.MEMORY_SETTINGS["concurrent_workers"])
        self._stability_metrics = StabilityMetrics()

        # Thread safety
        self._lock = threading.Lock()
        self._active_operations = set()

    @traced(span_name="get_aggtrades_files")
    def get_aggtrades_files(self, symbol: str, exchange: str) -> List[Path]:
        """Get all aggtrades files for a symbol and exchange."""
        pattern = f"aggtrades_{exchange}_{symbol}_*.csv"
        csv_files = list(self.data_cache_path.glob(pattern))

        # Also get parquet files if they exist
        pattern_parquet = f"aggtrades_{exchange}_{symbol}_*.parquet"
        parquet_files = list(self.data_cache_path.glob(pattern_parquet))

        return sorted(csv_files + parquet_files)

    @validates()
    @traced(span_name="validate_file_format")
    @handles_errors(default_return={ "valid": False, "issues": ["Validation failed"], "warnings": [], "file_size": 0, "row_count": 0, "memory_usage_mb": 0.0, "step1_5_compatible": False, "step2_compatible": False, "step3_compatible": False, "step4_compatible": False, }, context="aggtrades_format_validator.validate_file_format" )
    async def validate_file_format(self, file_path: Path) -> Dict[str, Any]:
        """Validate a single aggtrades file format for pipeline compatibility.

        Args:
            file_path: Path to the file to validate

        Returns:
            Dictionary with comprehensive validation results

        """
        logger.info(f"🔍 Validating {file_path.name} for pipeline compatibility")

        result = {
            "valid": False,
            "issues": [],
            "warnings": [],
            "file_size": 0,
            "row_count": 0,
            "memory_usage_mb": 0.0,
            "step1_5_compatible": False,
            "step2_compatible": False,
            "step3_compatible": False,
            "step4_compatible": False,
        }
        
        try:
            # Check file size
            result['file_size'] = file_path.stat().st_size
            
            if result['file_size'] == 0:
                result['issues'].append("Empty file")
                return result
            
            # Read the file
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, parse_dates=['timestamp'])
            elif file_path.suffix.lower() == '.parquet':
                df = standardized_parquet_handler.read_parquet_standardized(file_path)
            else:
                result['issues'].append(f"Unsupported file format: {file_path.suffix}")
                return result
            
            result['row_count'] = len(df)
            result['memory_usage_mb'] = df.memory_usage(deep = True).sum() / 1024 / 1024
            
            if len(df) == 0:
                result['issues'].append("No data rows")
                return result
            
            # Step 1: Basic column validation
            if list(df.columns) != self.EXPECTED_COLUMNS:
                result['issues'].append(
                    f"Invalid columns: expected {self.EXPECTED_COLUMNS}, found {list(df.columns)}"
                )
            
            # Step 2: Data type validation
            for col, expected_dtype in self.EXPECTED_DTYPES.items():
                if col in df.columns:
                    if str(df[col].dtype) != expected_dtype:
                        result['issues'].append(
                            f"Invalid dtype for {col}: expected {expected_dtype}, found {df[col].dtype}"
                        )
                else:
                    result['issues'].append(f"Missing column: {col}")
            
            # Step 3: Step1_5 specific validation
            step1_5_issues = self._validate_step1_5_requirements(df)
            result['issues'].extend(step1_5_issues)
            
            # Step 4: Step2 compatibility (feature engineering requirements)
            step2_issues = self._validate_step2_compatibility(df)
            result['issues'].extend(step2_issues)
            
            # Step 5: Step3 compatibility (regime discovery requirements)
            step3_issues = self._validate_step3_compatibility(df)
            result['issues'].extend(step3_issues)
            
            # Step 6: Step4 compatibility (labeling requirements)
            step4_issues = self._validate_step4_compatibility(df)
            result['issues'].extend(step4_issues)
            
            # Step 7: Data quality checks
            quality_issues = await self._validate_data_quality(df)
            result['issues'].extend(quality_issues)
            
            # Step 8: Memory optimization warnings
            memory_warnings = self._check_memory_optimization(df)
            result['warnings'].extend(memory_warnings)
            
            # Determine compatibility
            result['step1_5_compatible'] = len([i for i in result['issues'] if 'step1_5' in i.lower()]) == 0
            result['step2_compatible'] = len([i for i in result['issues'] if 'step02' in i.lower()]) == 0
            result['step3_compatible'] = len([i for i in result['issues'] if 'step03' in i.lower()]) == 0
            result['step4_compatible'] = len([i for i in result['issues'] if 'step04' in i.lower()]) == 0
            
            # Overall validity
            result['valid'] = len(result['issues']) == 0
                
        except Exception as e:
            result['issues'].append(f"Error reading file: {e}")
        
        return result
    
    @log_all_calls

    def _validate_step1_5_requirements(self, df: pd.DataFrame) -> List[str]:
        """Validate step1_5 specific requirements"""
        issues = []
        
        if 'timestamp' in df.columns:
            # Check timestamp range
            min_timestamp = pd.to_datetime(self.STEP1_5_REQUIREMENTS['min_timestamp'])
            max_timestamp = pd.to_datetime(self.STEP1_5_REQUIREMENTS['max_timestamp'])
            
            if df['timestamp'].min() < min_timestamp:
                issues.append(f"step1_5: Timestamps before {min_timestamp} not supported")
            
            if df['timestamp'].max() > max_timestamp:
                issues.append(f"step1_5: Timestamps after {max_timestamp} not supported")
            
            # Check timestamp ordering
            if not df['timestamp'].is_monotonic_increasing:
                issues.append("step1_5: Timestamps not in ascending order")
        
        # Check row count requirements
        if len(df) < self.STEP1_5_REQUIREMENTS['min_rows']:
            issues.append(f"step1_5: Too few rows ({len(df)} < {self.STEP1_5_REQUIREMENTS['min_rows']})")
        
        if len(df) > self.STEP1_5_REQUIREMENTS['max_rows']:
            issues.append(f"step1_5: Too many rows ({len(df)} > {self.STEP1_5_REQUIREMENTS['max_rows']})")
        
        return issues
    @log_all_calls

    def _validate_step2_compatibility(self, df: pd.DataFrame) -> List[str]:
        """Validate step02 feature engineering compatibility"""
        issues = []
        
        if 'price' in df.columns:
            min_price = df['price'].min()
            max_price = df['price'].max()
            
            if min_price < self.STEP2_REQUIREMENTS['min_price']:
                issues.append(f"step02: Price too low ({min_price} < {self.STEP2_REQUIREMENTS['min_price']})")
            
            if max_price > self.STEP2_REQUIREMENTS['max_price']:
                issues.append(f"step02: Price too high ({max_price} > {self.STEP2_REQUIREMENTS['max_price']})")
        
        if 'quantity' in df.columns:
            min_quantity = df['quantity'].min()
            max_quantity = df['quantity'].max()
            
            if min_quantity < self.STEP2_REQUIREMENTS['min_quantity']:
                issues.append(f"step02: Quantity too low ({min_quantity} < {self.STEP2_REQUIREMENTS['min_quantity']})")
            
            if max_quantity > self.STEP2_REQUIREMENTS['max_quantity']:
                issues.append(f"step02: Quantity too high ({max_quantity} > {self.STEP2_REQUIREMENTS['max_quantity']})")
        
        return issues
    @log_all_calls

    def _validate_step3_compatibility(self, df: pd.DataFrame) -> List[str]:
        """Validate step03 regime discovery compatibility"""
        issues = []
        
        if 'timestamp' in df.columns:
            # Check time span
            time_span = (df['timestamp'].max() - df['timestamp'].min()).days
            if time_span < self.STEP3_REQUIREMENTS['required_time_span_days']:
                issues.append(f"step03: Insufficient time span ({time_span} days < {self.STEP3_REQUIREMENTS['required_time_span_days']} days)")
            
            # Check for large gaps
            time_diffs = df['timestamp'].diff().dropna()
            max_gap = time_diffs.max().total_seconds()
            if max_gap > self.STEP3_REQUIREMENTS['max_gap_seconds']:
                issues.append(f"step03: Large time gap detected ({max_gap:.1f}s > {self.STEP3_REQUIREMENTS['max_gap_seconds']}s)")
        
        return issues
    @log_all_calls

    def _validate_step4_compatibility(self, df: pd.DataFrame) -> List[str]:
        """Validate step04 labeling compatibility"""
        issues = []
        
        # Check required features
        for feature in self.STEP4_REQUIREMENTS['required_features']:
            if feature not in df.columns:
                issues.append(f"step04: Missing required feature: {feature}")
        
        if 'timestamp' in df.columns:
            # Check labeling period requirements
            time_span_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
            
            if time_span_hours < self.STEP4_REQUIREMENTS['min_labeling_period_hours']:
                issues.append(f"step04: Insufficient labeling period ({time_span_hours:.1f}h < {self.STEP4_REQUIREMENTS['min_labeling_period_hours']}h)")
            
            if time_span_hours > self.STEP4_REQUIREMENTS['max_labeling_period_hours']:
                issues.append(f"step04: Excessive labeling period ({time_span_hours:.1f}h > {self.STEP4_REQUIREMENTS['max_labeling_period_hours']}h)")
        
        return issues
    @log_all_calls

    def _validate_data_quality(self, df: pd.DataFrame) -> List[str]:
        """Validate general data quality"""
        issues = []
        
        # Check for null values in critical columns
        critical_columns = ['timestamp', 'price', 'quantity']
        for col in critical_columns:
            if col in df.columns and df[col].isnull().any():
                null_count = df[col].isnull().sum()
                issues.append(f"Data quality: {null_count} null values in {col}")
        
        # Check for duplicate timestamps
        if 'timestamp' in df.columns:
            duplicate_timestamps = df['timestamp'].duplicated().sum()
            if duplicate_timestamps > 0:
                issues.append(f"Data quality: {duplicate_timestamps} duplicate timestamps")
        
        # Check for negative prices or quantities
        if 'price' in df.columns and (df['price'] < 0).any():
            issues.append("Data quality: Negative prices detected")
        
        if 'quantity' in df.columns and (df['quantity'] < 0).any():
            issues.append("Data quality: Negative quantities detected")
        
        return issues
    @log_all_calls

    def _check_memory_optimization(self, df: pd.DataFrame) -> List[str]:
        """Check for memory optimization opportunities"""
        warnings = []
        
        # Check memory usage
        memory_usage_mb = df.memory_usage(deep = True).sum() / 1024 / 1024
        if memory_usage_mb > 100:  # 100 MB threshold
            warnings.append(f"Memory optimization: Large memory usage ({memory_usage_mb:.1f} MB)")
        
        # Check for inefficient data types
        for col, expected_dtype in self.EXPECTED_DTYPES.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if actual_dtype != expected_dtype:
                    warnings.append(f"Memory optimization: {col} has inefficient dtype {actual_dtype} (expected {expected_dtype})")
        
        return warnings

    @traced(span_name="fix_file_format")
    @handles_errors(fallback = False)
    def fix_file_format(self, file_path: Path) -> bool:
        """Fix file format issues to ensure pipeline compatibility.

        Args:
            file_path: Path to the file to fix

        Returns:
            True if successfully fixed, False otherwise

        """
        try:
            logger.info(f"🔧 Fixing format for {file_path.name}")

            # Read the file with robust error handling
            df = self._robust_file_reader(file_path)
            if df is None:
                return False

            # Fix column names if needed
            df = self._fix_column_names(df, file_path)
            if df is None:
                return False

            # Validate and fix data types with enhanced stability
            df = self._fix_data_types_with_validation(df, file_path)
            if df is None:
                return False

            # Enhanced null value handling
            df = self._handle_null_values(df, file_path)
            if df is None:
                return False

            # Ensure timestamp consistency
            df = self._ensure_timestamp_consistency(df, file_path)
            if df is None:
                return False

            # Remove data anomalies
            df = self._remove_data_anomalies(df, file_path)
            if df is None:
                return False

            # Validate final format
            validation_result = self._validate_final_format(df, file_path)
            if not validation_result.get("valid", False):
                logger.error(f"❌ Final validation failed for {file_path.name}: {validation_result.get('issues', [])}")
                return False

            # Save the fixed file with backup
            success = self._save_fixed_file_safely(df, file_path)
            if success:
                logger.info(f"✅ Fixed format for {file_path.name}")
                return True
            else:
                return False

        except Exception as e:
            logger.exception(f"❌ Error fixing {file_path.name}: {e}")
            return False

    def _robust_file_reader(self, file_path: Path) -> pd.DataFrame | None:
        """Robust file reading with multiple fallback strategies."""
        try:
            if file_path.suffix.lower() == '.csv':
                # Try different CSV reading options
                try:
                    df = pd.read_csv(file_path, parse_dates=['timestamp'], low_memory=False)
                except pd.errors.ParserError:
                    # Fallback: try without date parsing
                    df = pd.read_csv(file_path, low_memory=False)
                except UnicodeDecodeError:
                    # Fallback: try with different encoding
                    df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
            elif file_path.suffix.lower() == '.parquet':
                try:
                    df = standardized_parquet_handler.read_parquet_standardized(file_path)
                except Exception as e:
                    logger.warning(f"❌ Failed to read parquet file {file_path.name}: {e}")
                    return None
            else:
                logger.error(f"❌ Unsupported file format: {file_path.suffix}")
                return None

            if df.empty:
                logger.warning(f"⚠️ File {file_path.name} is empty")
                return None

            return df

        except Exception as e:
            logger.exception(f"❌ Error reading file {file_path.name}: {e}")
            return None

    def _fix_column_names(self, df: pd.DataFrame, file_path: Path) -> pd.DataFrame | None:
        """Fix column names with enhanced mapping."""
        try:
            column_mapping = {
                "a": "agg_trade_id",
                "p": "price",
                "q": "quantity",
                "f": "first_trade_id",
                "l": "last_trade_id",
                "T": "timestamp",
                "m": "is_buyer_maker",
                # Additional mappings for common variations
                "id": "agg_trade_id",
                "trade_id": "agg_trade_id",
                "vol": "quantity",
                "volume": "quantity",
                "side": "is_buyer_maker",
                "buyer_maker": "is_buyer_maker",
                "time": "timestamp",
                "datetime": "timestamp",
            }

            if list(df.columns) != self.EXPECTED_COLUMNS:
                # Check if we have the old column names
                if all(col in df.columns for col in column_mapping.keys()):
                    df = df.rename(columns=column_mapping)
                    logger.info(f"✅ Renamed columns using mapping for {file_path.name}")
                else:
                    # Try partial mapping for missing columns
                    missing_cols = [col for col in self.EXPECTED_COLUMNS if col not in df.columns]
                    available_mappings = {k: v for k, v in column_mapping.items() if k in df.columns and v in missing_cols}

                    if available_mappings:
                        df = df.rename(columns=available_mappings)
                        logger.info(f"✅ Applied partial column mapping for {file_path.name}")

                    # For remaining missing columns, try to infer them
                    df = self._infer_missing_columns(df, missing_cols, file_path)

            return df

        except Exception as e:
            logger.exception(f"❌ Error fixing column names for {file_path.name}: {e}")
            return None

    def _infer_missing_columns(self, df: pd.DataFrame, missing_cols: list[str], file_path: Path) -> pd.DataFrame:
        """Infer missing columns based on data patterns."""
        try:
            for col in missing_cols:
                if col == "is_buyer_maker":
                    # Try to infer buyer_maker from available data
                    if "side" in df.columns:
                        df["is_buyer_maker"] = df["side"].astype(bool)
                    elif "buyer" in df.columns:
                        df["is_buyer_maker"] = df["buyer"].astype(bool)
                    else:
                        # Default to True for missing data
                        df["is_buyer_maker"] = True
                        logger.warning(f"⚠️ Inferred is_buyer_maker as True for {file_path.name}")

                elif col == "timestamp":
                    # Try to infer timestamp from available time columns
                    time_cols = [c for c in df.columns if 'time' in c.lower()]
                    if time_cols:
                        df["timestamp"] = pd.to_datetime(df[time_cols[0]], errors='coerce')
                        logger.warning(f"⚠️ Inferred timestamp from {time_cols[0]} for {file_path.name}")

                elif col in ["first_trade_id", "last_trade_id"]:
                    # For trade IDs, use agg_trade_id if available
                    if "agg_trade_id" in df.columns:
                        df[col] = df["agg_trade_id"]
                        logger.warning(f"⚠️ Inferred {col} from agg_trade_id for {file_path.name}")

            return df

        except Exception as e:
            logger.exception(f"❌ Error inferring missing columns for {file_path.name}: {e}")
            return df

    def _fix_data_types_with_validation(self, df: pd.DataFrame, file_path: Path) -> pd.DataFrame | None:
        """Fix data types with enhanced validation and error recovery."""
        try:
            for col, expected_dtype in self.EXPECTED_DTYPES.items():
                if col in df.columns:
                    original_dtype = str(df[col].dtype)

                    try:
                        if expected_dtype == "int64":
                            # Handle nullable integers
                            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                        elif expected_dtype == "float64":
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                            # Check for extreme values
                            if df[col].notna().any():
                                mean_val = df[col].mean()
                                std_val = df[col].std()
                                if std_val > 0:
                                    z_scores = np.abs((df[col] - mean_val) / std_val)
                                    outliers = z_scores > 5  # 5 standard deviations
                                    if outliers.any():
                                        logger.warning(f"⚠️ Found {outliers.sum()} outlier values in {col} for {file_path.name}")
                        elif expected_dtype == "datetime64[ns]":
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                            # Validate timestamp range
                            if df[col].notna().any():
                                min_ts = df[col].min()
                                max_ts = df[col].max()
                                if min_ts < pd.Timestamp('2020-01-01') or max_ts > pd.Timestamp('2030-01-01'):
                                    logger.warning(f"⚠️ Unusual timestamp range in {file_path.name}: {min_ts} to {max_ts}")
                        elif expected_dtype == "bool":
                            df[col] = df[col].astype(bool)

                        logger.debug(f"✅ Fixed dtype for {col}: {original_dtype} -> {expected_dtype}")

                    except Exception as e:
                        logger.warning(f"⚠️ Failed to convert {col} to {expected_dtype}: {e}")
                        # Keep original dtype if conversion fails
                        continue

            return df

        except Exception as e:
            logger.exception(f"❌ Error fixing data types for {file_path.name}: {e}")
            return None

    def _handle_null_values(self, df: pd.DataFrame, file_path: Path) -> pd.DataFrame | None:
        """Enhanced null value handling with interpolation and validation."""
        try:
            critical_columns = ["timestamp", "price", "quantity"]
            non_critical_columns = ["agg_trade_id", "first_trade_id", "last_trade_id", "is_buyer_maker"]

            # Check null values in critical columns
            for col in critical_columns:
                if col in df.columns:
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        if null_count / len(df) > 0.1:  # More than 10% null
                            logger.error(f"❌ Too many null values in critical column {col} for {file_path.name}: {null_count}")
                            return None
                        else:
                            logger.warning(f"⚠️ Found {null_count} null values in {col} for {file_path.name}")

            # Remove rows with null values in critical columns
            df = df.dropna(subset=critical_columns)

            # Handle null values in non-critical columns
            for col in non_critical_columns:
                if col in df.columns:
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        if col in ["agg_trade_id", "first_trade_id", "last_trade_id"]:
                            # Forward fill for ID columns
                            df[col] = df[col].fillna(method='ffill')
                        elif col == "is_buyer_maker":
                            # Default to True for buyer_maker
                            df[col] = df[col].fillna(True)

                        logger.debug(f"✅ Handled {null_count} null values in {col}")

            return df

        except Exception as e:
            logger.exception(f"❌ Error handling null values for {file_path.name}: {e}")
            return None

    def _ensure_timestamp_consistency(self, df: pd.DataFrame, file_path: Path) -> pd.DataFrame | None:
        """Ensure timestamp consistency and handle duplicates."""
        try:
            if "timestamp" not in df.columns:
                logger.error(f"❌ No timestamp column found in {file_path.name}")
                return None

            # Ensure timestamp is datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            # Remove rows with invalid timestamps
            invalid_ts = df["timestamp"].isnull().sum()
            if invalid_ts > 0:
                logger.warning(f"⚠️ Removing {invalid_ts} rows with invalid timestamps from {file_path.name}")
                df = df.dropna(subset=["timestamp"])

            # Sort by timestamp
            df = df.sort_values("timestamp")

            # Check for duplicate timestamps
            duplicate_count = df["timestamp"].duplicated().sum()
            if duplicate_count > 0:
                logger.warning(f"⚠️ Found {duplicate_count} duplicate timestamps in {file_path.name}")
                # Keep first occurrence of duplicates
                df = df.drop_duplicates(subset=["timestamp"], keep='first')

            # Check timestamp monotonicity
            if not df["timestamp"].is_monotonic_increasing:
                logger.warning(f"⚠️ Timestamps not monotonic in {file_path.name}, re-sorting")
                df = df.sort_values("timestamp").reset_index(drop=True)

            return df

        except Exception as e:
            logger.exception(f"❌ Error ensuring timestamp consistency for {file_path.name}: {e}")
            return None

    def _remove_data_anomalies(self, df: pd.DataFrame, file_path: Path) -> pd.DataFrame | None:
        """Remove data anomalies and outliers."""
        try:
            # Remove negative prices or quantities
            if "price" in df.columns:
                negative_prices = (df["price"] < 0).sum()
                if negative_prices > 0:
                    logger.warning(f"⚠️ Removing {negative_prices} rows with negative prices from {file_path.name}")
                    df = df[df["price"] >= 0]

            if "quantity" in df.columns:
                negative_qty = (df["quantity"] < 0).sum()
                if negative_qty > 0:
                    logger.warning(f"⚠️ Removing {negative_qty} rows with negative quantities from {file_path.name}")
                    df = df[df["quantity"] >= 0]

            # Remove extreme outliers in price and quantity
            for col in ["price", "quantity"]:
                if col in df.columns and len(df) > 10:
                    # Use IQR method for outlier detection
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR  # More lenient than usual 1.5
                    upper_bound = Q3 + 3 * IQR

                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
                    outlier_count = outliers.sum()

                    if outlier_count > 0:
                        logger.warning(f"⚠️ Found {outlier_count} outlier values in {col} for {file_path.name}")
                        # Don't remove outliers automatically, just log them

            return df

        except Exception as e:
            logger.exception(f"❌ Error removing data anomalies for {file_path.name}: {e}")
            return None

    def _validate_final_format(self, df: pd.DataFrame, file_path: Path) -> dict[str, Any]:
        """Final validation of the fixed format."""
        validation = {"valid": True, "issues": []}

        try:
            # Check column count
            if len(df.columns) != len(self.EXPECTED_COLUMNS):
                validation["valid"] = False
                validation["issues"].append(f"Column count mismatch: expected {len(self.EXPECTED_COLUMNS)}, got {len(df.columns)}")

            # Check column names
            if list(df.columns) != self.EXPECTED_COLUMNS:
                validation["valid"] = False
                validation["issues"].append(f"Column names mismatch: expected {self.EXPECTED_COLUMNS}, got {list(df.columns)}")

            # Check data types
            for col, expected_dtype in self.EXPECTED_DTYPES.items():
                if col in df.columns:
                    actual_dtype = str(df[col].dtype)
                    if actual_dtype != expected_dtype:
                        validation["issues"].append(f"Data type mismatch for {col}: expected {expected_dtype}, got {actual_dtype}")

            # Check for null values in critical columns
            critical_columns = ["timestamp", "price", "quantity"]
            for col in critical_columns:
                if col in df.columns:
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        validation["valid"] = False
                        validation["issues"].append(f"Null values in critical column {col}: {null_count}")

            # Check data size
            if len(df) == 0:
                validation["valid"] = False
                validation["issues"].append("DataFrame is empty")

            return validation

        except Exception as e:
            validation["valid"] = False
            validation["issues"].append(f"Validation error: {e}")
            return validation

    def _save_fixed_file_safely(self, df: pd.DataFrame, file_path: Path) -> bool:
        """Save the fixed file with backup and validation."""
        try:
            # Create backup
            backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
            if file_path.exists() and not backup_path.exists():
                import shutil
                shutil.copy2(file_path, backup_path)
                logger.info(f"✅ Created backup: {backup_path.name}")

            # Save the fixed file
            if file_path.suffix.lower() == '.csv':
                df.to_csv(file_path, index=False)
            else:
                # Use aggtrades schema for aggtrades data
                standardized_parquet_handler.write_parquet_standardized(df, file_path, schema_name='aggtrades', compression="zstd", index=False)

            # Verify the saved file
            if file_path.exists():
                file_size = file_path.stat().st_size
                logger.info(f"✅ Saved fixed file: {file_path.name} ({file_size} bytes)")
                return True
            else:
                logger.error(f"❌ Failed to save fixed file: {file_path.name}")
                return False

        except Exception as e:
            logger.exception(f"❌ Error saving fixed file {file_path.name}: {e}")
            return False

    @traced(span_name="validate_all_aggtrades_stable")
    @handles_errors(default_return={ "total_files": 0, "valid_files": 0, "invalid_files": 0, "fixed_files": 0, "errors": [], "stability_metrics": {}, }, context="aggtrades_format_validator.validate_all_aggtrades_stable" )
    async def validate_all_aggtrades_stable(
        self, symbol: str, exchange: str, auto_fix: bool = True, use_concurrency: bool = True
    ) -> Dict[str, Any]:
        """Enhanced validation of all aggtrades files with comprehensive stability features."""

        logger.info(f"🔍 Starting stable validation for {exchange}_{symbol}")
        start_time = time.time()

        self._stability_metrics.record_operation_start("batch_validation")

        try:
            aggtrades_files = self.get_aggtrades_files(symbol, exchange)
            logger.info(f"📁 Found {len(aggtrades_files)} aggtrades files")

            validation_result = {
                "total_files": len(aggtrades_files),
                "valid_files": 0,
                "invalid_files": 0,
                "fixed_files": 0,
                "errors": [],
                "stability_metrics": {},
                "processing_stats": {},
            }

            if not aggtrades_files:
                logger.warning(f"⚠️ No aggtrades files found for {exchange}_{symbol}")
                return validation_result

            # Process files with stability features
            if use_concurrency and len(aggtrades_files) > 1:
                # Concurrent processing for multiple files
                results = await self._process_files_concurrently(aggtrades_files, auto_fix)
            else:
                # Sequential processing with stability monitoring
                results = await self._process_files_sequentially(aggtrades_files, auto_fix)

            # Aggregate results
            for result in results:
                if result and result.get("success", False):
                    validation_result["valid_files"] += 1
                    if result.get("fixed", False):
                        validation_result["fixed_files"] += 1
                elif result:
                    validation_result["invalid_files"] += 1
                    if result.get("error"):
                        validation_result["errors"].append(result["error"])

            # Add stability metrics
            validation_result["stability_metrics"] = self._stability_metrics.get_summary()
            validation_result["processing_stats"] = {
                "total_time": time.time() - start_time,
                "avg_time_per_file": (time.time() - start_time) / len(aggtrades_files),
                "memory_usage_mb": self._memory_monitor.get_memory_usage(),
                "concurrent_processing": use_concurrency,
            }

            # Log final results
            logger.info(f"📊 Stable validation complete: {validation_result['valid_files']} valid, {validation_result['invalid_files']} invalid, {validation_result['fixed_files']} fixed")
            logger.info(f"⏱️ Total processing time: {validation_result['processing_stats']['total_time']:.2f}s")
            logger.info(f"🧠 Peak memory usage: {validation_result['stability_metrics']['memory_peak_mb']:.1f} MB")

            self._stability_metrics.record_operation_complete(
                "batch_validation",
                validation_result['processing_stats']['total_time'],
                validation_result['stability_metrics']['memory_peak_mb']
            )

            return validation_result

        except Exception as e:
            processing_time = time.time() - start_time
            self._stability_metrics.record_operation_failure("batch_validation", type(e).__name__)

            logger.exception(f"❌ Stable batch validation failed: {e}")
            return {
                "total_files": 0,
                "valid_files": 0,
                "invalid_files": 0,
                "fixed_files": 0,
                "errors": [str(e)],
                "stability_metrics": self._stability_metrics.get_summary(),
                "processing_stats": {
                    "total_time": processing_time,
                    "error": str(e),
                },
            }

    async def _process_files_concurrently(self, files: List[Path], auto_fix: bool) -> List[Dict[str, Any]]:
        """Process files concurrently with stability monitoring."""

        async def process_single_file(file_path: Path) -> Dict[str, Any]:
            """Process a single file with comprehensive error handling."""
            try:
                # Check memory before processing
                if self._memory_monitor.check_memory_pressure():
                    self._memory_monitor.force_garbage_collection()
                    await asyncio.sleep(0.1)  # Brief pause for GC

                # Validate file with stability features
                validation_result = await validate_file_with_stability(
                    self,
                    file_path,
                    self._memory_monitor,
                    self._error_recovery,
                    self._stability_metrics
                )

                success = validation_result.get("valid", False)
                result = {"success": success, "file": str(file_path), "fixed": False}

                if not success and auto_fix:
                    # Attempt to fix the file
                    fixed = self.fix_file_format(file_path)
                    result["fixed"] = fixed
                    result["success"] = fixed

                    if fixed:
                        logger.info(f"🔧 Successfully fixed {file_path.name}")
                    else:
                        result["error"] = f"Failed to fix {file_path.name}"

                return result

            except Exception as e:
                logger.exception(f"❌ Error processing {file_path.name}: {e}")
                return {"success": False, "file": str(file_path), "error": str(e)}

        # Process files concurrently
        results = await self._concurrent_processor.process_concurrently(
            files, process_single_file
        )

        return results

    async def _process_files_sequentially(self, files: List[Path], auto_fix: bool) -> List[Dict[str, Any]]:
        """Process files sequentially with stability monitoring."""
        results = []

        for file_path in files:
            try:
                # Memory check between files
                if self._memory_monitor.check_memory_pressure():
                    self._memory_monitor.force_garbage_collection()
                    await asyncio.sleep(0.05)

                # Validate file
                validation_result = await validate_file_with_stability(
                    self,
                    file_path,
                    self._memory_monitor,
                    self._error_recovery,
                    self._stability_metrics
                )

                success = validation_result.get("valid", False)
                result = {"success": success, "file": str(file_path), "fixed": False}

                if not success and auto_fix:
                    fixed = self.fix_file_format(file_path)
                    result["fixed"] = fixed
                    result["success"] = fixed

                results.append(result)

            except Exception as e:
                logger.exception(f"❌ Error processing {file_path.name}: {e}")
                results.append({"success": False, "file": str(file_path), "error": str(e)})

        return results

    @traced(span_name="validate_all_aggtrades")
    @handles_errors(default_return={ "total_files": 0, "valid_files": 0, "invalid_files": 0, "fixed_files": 0, "errors": [], }, context="aggtrades_format_validator.validate_all_aggtrades" )
    def validate_all_aggtrades(
        self, symbol: str, exchange: str, auto_fix: bool = True
    ) -> Dict[str, Any]:
        """Validate all aggtrades files for a symbol and exchange.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            auto_fix: Whether to automatically fix issues

        Returns:
            Dictionary with validation results

        """
        logger.info(f"🔍 Validating all aggtrades for {exchange}_{symbol}")

        aggtrades_files = self.get_aggtrades_files(symbol, exchange)
        logger.info(f"📁 Found {len(aggtrades_files)} aggtrades files")

        validation_result = {
            "total_files": len(aggtrades_files),
            "valid_files": 0,
            "invalid_files": 0,
            "fixed_files": 0,
            "errors": [],
        }

        for file_path in aggtrades_files:
            try:
                # Validate file format
                validation = self.validate_file_format(file_path)

                if validation["valid"]:
                    validation_result["valid_files"] += 1
                    logger.debug(f"✅ {file_path.name} is valid")
                else:
                    validation_result["invalid_files"] += 1
                    logger.warning(f"⚠️ {file_path.name} has issues: {validation['issues']}")

                    # Auto-fix if enabled
                    if auto_fix:
                        if self.fix_file_format(file_path):
                            validation_result["fixed_files"] += 1
                            logger.info(f"🔧 Fixed {file_path.name}")

            except Exception as e:
                validation_result["errors"].append(f"Error processing {file_path.name}: {e}")
                logger.exception(f"❌ Error processing {file_path.name}: {e}")

        logger.info(
            f"📊 Validation complete: {validation_result['valid_files']} valid, "
            f"{validation_result['invalid_files']} invalid, "
            f"{validation_result['fixed_files']} fixed"
        )

        return validation_result

    @traced(span_name="generate_compatibility_report")
    def generate_compatibility_report(self, symbol: str, exchange: str) -> str:
        """Generate a comprehensive compatibility report.

        Args:
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Compatibility report string

        """
        aggtrades_files = self.get_aggtrades_files(symbol, exchange)

        report = f"""
🔍 AGGTRADES COMPATIBILITY REPORT FOR {exchange}_{symbol}
{'='*80}

📁 FILES FOUND: {len(aggtrades_files)}

📊 COMPATIBILITY RESULTS:
"""

        total_size = 0
        total_rows = 0
        step1_5_compatible = 0
        step2_compatible = 0
        step3_compatible = 0
        step4_compatible = 0

        for file_path in aggtrades_files:
            try:
                validation = self.validate_file_format(file_path)
                file_size = file_path.stat().st_size
                total_size += file_size

                status = "✅ VALID" if validation["valid"] else "❌ INVALID"
                report += f"• {file_path.name}: {status} ({validation['row_count']} rows, {file_size/1024/1024:.2f} MB)\n"

                if not validation["valid"]:
                    for issue in validation["issues"]:
                        report += f"  - Issue: {issue}\n"

                total_rows += validation["row_count"]
                if validation["step1_5_compatible"]:
                    step1_5_compatible += 1
                if validation["step2_compatible"]:
                    step2_compatible += 1
                if validation["step3_compatible"]:
                    step3_compatible += 1
                if validation["step4_compatible"]:
                    step4_compatible += 1

            except Exception as e:
                report += f"• {file_path.name}: ❌ ERROR ({e})\n"

        report += f"""
📈 SUMMARY:
• Total Files: {len(aggtrades_files)}
• Total Size: {total_size/1024/1024:.2f} MB
• Total Rows: {total_rows:,}

🔧 PIPELINE COMPATIBILITY:
• Step1_5 Compatible: {step1_5_compatible}/{len(aggtrades_files)}
• Step2 Compatible: {step2_compatible}/{len(aggtrades_files)}
• Step3 Compatible: {step3_compatible}/{len(aggtrades_files)}
• Step4 Compatible: {step4_compatible}/{len(aggtrades_files)}

{'='*80}
"""

        return report

class MemoryMonitor:
    """Monitor memory usage and provide memory management utilities."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self._baseline_memory = self.get_memory_usage()

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def check_memory_pressure(self, threshold_mb: int = 512) -> bool:
        """Check if memory usage exceeds threshold."""
        current_memory = self.get_memory_usage()
        return current_memory > threshold_mb

    def force_garbage_collection(self) -> None:
        """Force garbage collection to free memory."""
        gc.collect()
        logger.debug("🧹 Forced garbage collection")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        memory_info = self.process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": self.process.memory_percent(),
            "gc_stats": gc.get_stats(),
        }

class ErrorRecoveryHandler:
    """Handle errors with retry logic and circuit breaker pattern."""

    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings = settings
        self.failure_count = 0
        self.last_failure_time = 0
        self.circuit_open = False

    def should_retry(self, exception: Exception) -> bool:
        """Determine if operation should be retried based on exception type."""
        retryable_exceptions = (
            ConnectionError,
            TimeoutError,
            OSError,
            pd.errors.ParserError,
        )

        if isinstance(exception, retryable_exceptions):
            return self.failure_count < self.settings["max_retries"]
        return False

    def record_failure(self) -> None:
        """Record a failure for circuit breaker logic."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.settings["circuit_breaker_threshold"]:
            self.circuit_open = True
            logger.warning(f"🔌 Circuit breaker opened after {self.failure_count} failures")

    def record_success(self) -> None:
        """Record a success to reset failure count."""
        if not self.circuit_open:
            self.failure_count = max(0, self.failure_count - 1)

    def can_proceed(self) -> bool:
        """Check if operation can proceed based on circuit breaker state."""
        if not self.circuit_open:
            return True

        # Check if circuit breaker timeout has expired
        if time.time() - self.last_failure_time > self.settings["circuit_breaker_timeout"]:
            logger.info("🔌 Circuit breaker timeout expired, attempting to close")
            self.circuit_open = False
            self.failure_count = 0
            return True

        return False

    async def execute_with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic."""
        while True:
            if not self.can_proceed():
                raise Exception("Circuit breaker is open")

            try:
                result = await func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                if self.should_retry(e):
                    self.record_failure()
                    logger.warning(f"⚠️ Retrying operation after error: {e}")
                    await asyncio.sleep(self.settings["retry_delay"])
                    continue
                else:
                    raise e

class ConcurrentProcessor:
    """Handle concurrent processing with proper synchronization."""

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="stability_worker")
        self.semaphore = asyncio.Semaphore(max_workers)

    async def process_concurrently(self, tasks: List[Any], processor_func) -> List[Any]:
        """Process tasks concurrently with proper resource management."""
        async def process_task(task):
            async with self.semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self.executor, processor_func, task)

        # Process tasks concurrently
        results = await asyncio.gather(
            *[process_task(task) for task in tasks],
            return_exceptions=True
        )

        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Error processing task {i}: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)

        return processed_results

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)

class StabilityMetrics:
    """Track stability metrics for monitoring and optimization."""

    def __init__(self) -> None:
        self.metrics = {
            "operations_started": 0,
            "operations_completed": 0,
            "operations_failed": 0,
            "memory_peak_mb": 0,
            "processing_time_total": 0,
            "errors_by_type": {},
            "retries_count": 0,
        }
        self._lock = threading.Lock()

    def record_operation_start(self, operation_type: str) -> None:
        """Record the start of an operation."""
        with self._lock:
            self.metrics["operations_started"] += 1

    def record_operation_complete(self, operation_type: str, duration: float, memory_used: float) -> None:
        """Record a successful operation completion."""
        with self._lock:
            self.metrics["operations_completed"] += 1
            self.metrics["processing_time_total"] += duration
            self.metrics["memory_peak_mb"] = max(self.metrics["memory_peak_mb"], memory_used)

    def record_operation_failure(self, operation_type: str, error_type: str) -> None:
        """Record an operation failure."""
        with self._lock:
            self.metrics["operations_failed"] += 1
            if error_type not in self.metrics["errors_by_type"]:
                self.metrics["errors_by_type"][error_type] = 0
            self.metrics["errors_by_type"][error_type] += 1

    def record_retry(self) -> None:
        """Record a retry attempt."""
        with self._lock:
            self.metrics["retries_count"] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of stability metrics."""
        with self._lock:
            total_operations = self.metrics["operations_started"]
            success_rate = (
                self.metrics["operations_completed"] / total_operations
                if total_operations > 0 else 0
            )
            avg_processing_time = (
                self.metrics["processing_time_total"] / self.metrics["operations_completed"]
                if self.metrics["operations_completed"] > 0 else 0
            )

            return {
                "total_operations": total_operations,
                "success_rate": success_rate,
                "failure_rate": 1 - success_rate,
                "avg_processing_time": avg_processing_time,
                "memory_peak_mb": self.metrics["memory_peak_mb"],
                "retries_count": self.metrics["retries_count"],
                "top_errors": sorted(
                    self.metrics["errors_by_type"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }

# Enhanced validation method with stability features
@traced(span_name="validate_file_with_stability")
@handles_errors(
    default_return={
        "valid": False,
        "issues": ["Stability validation failed"],
        "warnings": [],
        "file_size": 0,
        "row_count": 0,
        "memory_usage_mb": 0.0,
        "stability_score": 0.0,
        "processing_time": 0.0,
    },
    context="enhanced_validation.validate_file_with_stability"
)
async def validate_file_with_stability(
    validator: AggtradesFormatValidator,
    file_path: Path,
    memory_monitor: MemoryMonitor = None,
    error_recovery: ErrorRecoveryHandler = None,
    stability_metrics: StabilityMetrics = None
) -> Dict[str, Any]:
    """Enhanced file validation with comprehensive stability features."""

    start_time = time.time()
    operation_id = f"validate_{file_path.name}_{int(time.time())}"

    if stability_metrics:
        stability_metrics.record_operation_start("file_validation")

    try:
        # Check memory before processing
        if memory_monitor and memory_monitor.check_memory_pressure():
            memory_monitor.force_garbage_collection()
            if memory_monitor.check_memory_pressure():
                raise MemoryError("Insufficient memory for file processing")

        # Validate file format with error recovery
        if error_recovery:
            result = await error_recovery.execute_with_retry(
                validator.validate_file_format, file_path
            )
        else:
            result = validator.validate_file_format(file_path)

        # Calculate stability score
        stability_score = calculate_stability_score(result)

        # Add stability metrics to result
        result["stability_score"] = stability_score
        result["processing_time"] = time.time() - start_time
        result["memory_usage_mb"] = memory_monitor.get_memory_usage() if memory_monitor else 0

        # Record success
        if stability_metrics:
            stability_metrics.record_operation_complete(
                "file_validation",
                result["processing_time"],
                result["memory_usage_mb"]
            )

        return result

    except Exception as e:
        processing_time = time.time() - start_time

        if stability_metrics:
            stability_metrics.record_operation_failure(
                "file_validation",
                type(e).__name__
            )

        logger.exception(f"❌ Stability validation failed for {file_path.name}: {e}")

        return {
            "valid": False,
            "issues": [f"Stability validation failed: {e}"],
            "warnings": [],
            "file_size": 0,
            "row_count": 0,
            "memory_usage_mb": memory_monitor.get_memory_usage() if memory_monitor else 0,
            "stability_score": 0.0,
            "processing_time": processing_time,
            "error_type": type(e).__name__,
        }

def calculate_stability_score(validation_result: Dict[str, Any]) -> float:
    """Calculate a stability score based on validation results."""
    base_score = 1.0

    # Deduct points for issues
    issue_penalty = len(validation_result.get("issues", [])) * 0.2
    base_score -= min(issue_penalty, 0.8)  # Max deduction of 0.8

    # Deduct points for warnings
    warning_penalty = len(validation_result.get("warnings", [])) * 0.05
    base_score -= min(warning_penalty, 0.2)  # Max deduction of 0.2

    # Check data quality factors
    if validation_result.get("row_count", 0) == 0:
        base_score -= 0.5
    elif validation_result.get("row_count", 0) < 100:
        base_score -= 0.2

    # Memory efficiency bonus
    memory_mb = validation_result.get("memory_usage_mb", 0)
    if memory_mb < 100:  # Efficient memory usage
        base_score += 0.1

    return max(0.0, min(1.0, base_score))  # Clamp between 0 and 1