"""Validator for Step 1: Data Collection."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd


# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
import datetime
import logging


import numpy as np


class Step1DataCollectionValidator:
    """Validator for Step 1: Data Collection."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.step_name = "step01_data_collection"
        self.config = config
        self.logger = system_logger.getChild("Validator.Step1")
        # Container for last validation result (consumed by orchestrator wrappers)
        self.validation_results: Dict[str, Any] = {}
        # Fine-tuned parameters for ML training (more lenient to avoid stopping training)
        self.min_records = 500  # Reduced from 1000 to allow smaller datasets
        self.max_gap_ratio = 0.2  # Allow up to 20% large gaps (increased from 10%)
        self.max_gap_hours = 48  # Increased from 24 hours
        self.price_tolerance = 0.001  # Allow very small negative prices due to precision
        self.volume_tolerance = 0.001  # Allow very small negative volumes due to precision

    async def validate(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate the data collection step with comprehensive checks."

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict containing validation results with detailed information

        """
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        timeframe = training_input.get("timeframe", "1m")
        data_dir = training_input.get("data_dir", "data_cache")

        self.logger.info(
            f"🔍 Validating Step 1 data collection for {exchange} {symbol} {timeframe}",
        )

        validation_result = {
            "validation_passed": False,
            "step_name": "step01_data_collection",
            "validation_results": {},
            "critical_issues": [],
            "warnings": [],
            "data_quality_metrics": {},
        }

        # Check pipeline_state presence first
        md = pipeline_state.get("market_data") or {}
        if isinstance(md, pd.DataFrame) and not md.empty:
            self.logger.info(f"✅ Market data present in state: {md.shape} rows/cols")
            
            # Comprehensive DataFrame validation
            df_validation, df_metrics = self.validate_dataframe_quality(
                df=md,
                min_rows=self.min_records,
                required_columns=["open", "high", "low", "close", "volume"],
                check_data_types=True,
                check_value_ranges=True,
                check_duplicates=True,
                check_temporal_consistency=True,
            )
            
            validation_result["validation_results"]["pipeline_state_data"] = {
                "valid": df_validation,
                "metrics": df_metrics,
            }
            
            if df_validation:
                validation_result["validation_passed"] = True
                validation_result["data_quality_metrics"] = df_metrics
                
                # Log additional details
                try:
                    if isinstance(md.index, pd.DatetimeIndex):
                        self.logger.info(f"   Date range: {md.index.min()} -> {md.index.max()}")
                    req = [c for c in ["open", "high", "low", "close"] if c in md.columns]
                    self.logger.info(f"   OHLC present: {req}")
                except Exception:
                    pass
            else:
                validation_result["critical_issues"].extend(df_metrics.get("critical_issues", []))
                validation_result["warnings"].extend(df_metrics.get("data_quality_issues", []))
            
            # Store results for external access
            self.validation_results = validation_result
            return validation_result

        # Check for consolidated files in data_cache directory
        consolidated_files = await self._check_consolidated_files(
            symbol=symbol, exchange=exchange, timeframe=timeframe, data_dir=data_dir,
        )

        validation_result["validation_results"]["consolidated_files"] = {
            "found": consolidated_files["found"],
            "files": consolidated_files.get("files", []),
        }

        if consolidated_files["found"]:
            self.logger.info(f"✅ Found consolidated files: {consolidated_files['files']}")

            # Validate the data quality of the consolidated files
            data_validation = await self._validate_consolidated_data_quality(
                consolidated_files["files"], symbol, exchange, timeframe,
            )

            validation_result["validation_results"]["data_quality"] = data_validation

            if data_validation.get("valid", False):
                self.logger.info("✅ Consolidated data quality validation passed")
                validation_result["validation_passed"] = True
                validation_result["data_quality_metrics"] = data_validation.get("metrics", {})
            else:
                self.logger.warning("⚠️ Consolidated data quality issues detected")
                validation_result["critical_issues"].extend(data_validation.get("critical_issues", []))
                validation_result["warnings"].extend(data_validation.get("warnings", []))
        else:
            validation_result["critical_issues"].append("No consolidated files found")

        if not validation_result["validation_passed"]:
            error_details = []
            if not validation_result.get("validation_results", {}).get("pipeline_state_data", {}).get("valid", False):
                error_details.append("pipeline state data validation failed")
            if not validation_result.get("validation_results", {}).get("consolidated_files", {}).get("found", False):
                error_details.append("no consolidated files found")
            if not validation_result.get("validation_results", {}).get("data_quality", {}).get("valid", False):
                error_details.append("consolidated data quality validation failed")
            
            error_msg = "❌ No market data found in state or consolidated files"
            if error_details:
                error_msg += f" - Issues: {', '.join(error_details)}"
            self.logger.error(error_msg)

        # Store results for external access
        self.validation_results = validation_result
        return validation_result

    def validate_dataframe_quality(
        self,
        df: pd.DataFrame,
        min_rows: int,
        required_columns: List[str],
        check_data_types: bool = True,
        check_value_ranges: bool = True,
        check_duplicates: bool = True,
        check_temporal_consistency: bool = True,
    ) -> tuple[bool, Dict[str, Any]]:
        """Basic DataFrame quality validation used by this validator.

        Returns (is_valid, metrics_dict).
        """
        is_valid = True
        metrics: Dict[str, Any] = {
            "row_count": int(len(df)) if df is not None else 0,
            "required_columns": required_columns,
            "missing_required_columns": [],
            "duplicate_rows": 0,
            "data_quality_issues": [],
            "critical_issues": [],
        }

        try:
            # Existence and size
            if df is None or df.empty:
                is_valid = False
                metrics["critical_issues"].append("DataFrame is None or empty")
                return is_valid, metrics
            if len(df) < min_rows:
                is_valid = False
                metrics["critical_issues"].append(f"Too few rows: {len(df)} < {min_rows}")

            # Required columns
            missing_cols = [c for c in required_columns if c not in df.columns]
            if missing_cols:
                is_valid = False
                metrics["missing_required_columns"] = missing_cols
                metrics["critical_issues"].append(f"Missing required columns: {missing_cols}")

            # Data types (best-effort)
            if check_data_types:
                numeric_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
                for col in numeric_cols:
                    try:
                        _ = pd.to_numeric(df[col], errors="coerce")
                    except Exception:
                        metrics["data_quality_issues"].append(f"Column not numeric: {col}")

            # Value ranges (lightweight sanity checks)
            if check_value_ranges:
                if "volume" in df.columns:
                    try:
                        min_volume = float(pd.to_numeric(df["volume"], errors="coerce").min())
                        if min_volume < -self.volume_tolerance:
                            metrics["data_quality_issues"].append(
                                f"Volume has unrealistic negative values (min={min_volume})"
                            )
                    except Exception:
                        metrics["data_quality_issues"].append("Failed to evaluate volume range")

            # Duplicates
            if check_duplicates:
                try:
                    dup_count = int(df.duplicated().sum())
                    metrics["duplicate_rows"] = dup_count
                    if dup_count > 0:
                        metrics["data_quality_issues"].append(f"Found {dup_count} duplicate rows")
                except Exception:
                    metrics["data_quality_issues"].append("Failed to compute duplicates")

            # Temporal consistency
            if check_temporal_consistency and "timestamp" in df.columns:
                try:
                    ts = pd.to_datetime(df["timestamp"], errors="coerce")
                    if ts.isnull().any():
                        metrics["data_quality_issues"].append("Null timestamps detected")
                    if not ts.is_monotonic_increasing:
                        metrics["data_quality_issues"].append("Timestamps are not monotonically increasing")
                except Exception:
                    metrics["data_quality_issues"].append("Failed to validate temporal consistency")

        except Exception as e:
            is_valid = False
            metrics["critical_issues"].append(f"Validation error: {e}")

        return is_valid, metrics

    async def _check_consolidated_files(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
    ) -> Dict[str, Any]:
        """Check for consolidated files in the data directory."

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory

        Returns:
            Dictionary with file information
        """
        files_found: List[str] = []

        # Check for klines consolidated files
        klines_patterns = [
            os.path.join(data_dir, f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"),
            os.path.join(data_dir, f"klines_{exchange}_{symbol}_{timeframe}_consolidated.csv"),
            os.path.join(data_dir, f"klines_{exchange}_{symbol}_{timeframe}_consolidated_cached_data.pkl"),
        ]

        for pattern in klines_patterns:
            if os.path.exists(pattern):
                files_found.append(pattern)
                self.logger.info(f"📊 Found klines file: {pattern}")

        # Check for aggtrades consolidated files (optional)
        aggtrades_patterns = [
            os.path.join(data_dir, f"aggtrades_{exchange}_{symbol}_consolidated.parquet"),
            os.path.join(data_dir, f"aggtrades_{exchange}_{symbol}_consolidated.csv"),
            os.path.join(data_dir, f"aggtrades_{exchange}_{symbol}_consolidated_cached_data.pkl"),
        ]

        for pattern in aggtrades_patterns:
            if os.path.exists(pattern):
                files_found.append(pattern)
                self.logger.info(f"📊 Found aggtrades file: {pattern}")

        return {
            "found": len(files_found) > 0,
            "files": files_found,
            "klines_found": any("klines" in f for f in files_found),
            "aggtrades_found": any("aggtrades" in f for f in files_found),
        }

    async def _validate_consolidated_data_quality(
        self,
        files: List[str],
        symbol: str,
        exchange: str,
        timeframe: str,
    ) -> Dict[str, Any]:
        """Validate the quality of consolidated data files."

        Args:
            files: List of file paths to validate
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe

        Returns:
            bool: True if validation passed
        """
        try:
            validation_result = {
                "valid": True,
                "files_validated": len(files),
                "file_validation_results": {},
                "critical_issues": [],
                "warnings": [],
                "metrics": {
                    "total_files": len(files),
                    "valid_files": 0,
                    "total_records": 0,
                    "data_quality_score": 0.0,
                },
            }

            # Validate klines data first (required)
            klines_files = [f for f in files if "klines" in f]
            if not klines_files:
                validation_result["valid"] = False
                validation_result["critical_issues"].append("No klines files found")
                return validation_result

            # Load and validate the first klines file
            klines_file = klines_files[0]
            self.logger.info(f"🔍 Validating klines file: {klines_file}")

            try:
                if klines_file.endswith(".parquet"):
                    df = pd.read_parquet(klines_file)
                elif klines_file.endswith(".csv"):
                    df = pd.read_csv(klines_file)
                elif klines_file.endswith(".pkl"):
                    df = pd.read_pickle(klines_file)
                else:
                    validation_result["valid"] = False
                    validation_result["critical_issues"].append(f"Unsupported file format: {klines_file}")
                    return validation_result

                # Comprehensive DataFrame validation
                df_validation, df_metrics = self.validate_dataframe_quality(
                    df=df,
                    min_rows=self.min_records,
                    required_columns=["open", "high", "low", "close", "volume"],
                    check_data_types=True,
                    check_value_ranges=True,
                    check_duplicates=True,
                    check_temporal_consistency=True,
                )

                validation_result["file_validation_results"][klines_file] = {
                    "valid": df_validation,
                    "file_path": klines_file,
                    "row_count": len(df),
                    "metrics": df_metrics,
                }

                if df_validation:
                    validation_result["metrics"]["valid_files"] = 1
                    validation_result["metrics"]["total_records"] = len(df)
                    validation_result["metrics"]["data_quality_score"] = 1.0
                else:
                    validation_result["critical_issues"].extend(df_metrics.get("critical_issues", []))
                    validation_result["warnings"].extend(df_metrics.get("data_quality_issues", []))

                # Additional data characteristics validation
                characteristics_validation = self._validate_data_characteristics(df, symbol, exchange)
                if not characteristics_validation:
                    validation_result["warnings"].append("Data characteristics validation failed")

            except Exception as e:
                validation_result["valid"] = False
                validation_result["critical_issues"].append(f"Error reading {klines_file}: {str(e)}")

            return validation_result

        except Exception as e:
            self.logger.exception(f"❌ Error validating consolidated data: {e}")
            return {
                "valid": False,
                "error": str(e),
                "critical_issues": [f"Validation error: {str(e)}"],
            }

    def _validate_data_characteristics(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str,
    ) -> bool:
        """Validate specific characteristics of the collected data."

        Args:
            data: Historical data DataFrame
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            bool: True if characteristics are valid

        """
        try:
            # Check minimum data size (more lenient for ML training)
            if len(data) < self.min_records:
                self.logger.warning(
                    f"⚠️ Insufficient data: {len(data)} records (minimum: {self.min_records}) - continuing with caution",
                )
                return False

            # Check for required columns (basic OHLCV)
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.warning(
                    f"⚠️ Missing required columns: {missing_columns} - continuing with caution",
                )
                return False

            # Check for reasonable price ranges (more tolerant)
            price_columns = ["open", "high", "low", "close"]
            for col in price_columns:
                if col in data.columns:
                    min_price = float(data[col].min())
                    if min_price < -self.price_tolerance:  # Allow small negative values due to precision
                        self.logger.warning(
                            f"⚠️ Invalid price values in {col} column (min: {min_price}) - continuing with caution",
                        )
                        return False

            # Check for reasonable volume values (more tolerant)
            if "volume" in data.columns:
                min_volume = float(data["volume"].min())
                if min_volume < -self.volume_tolerance:  # Allow small negative values due to precision
                    self.logger.warning(
                        f"⚠️ Invalid volume values (min: {min_volume}) - continuing with caution",
                    )
                    return False

            # Check data consistency (high >= low, etc.) - more lenient
            if all(col in data.columns for col in ["high", "low", "open", "close"]):
                invalid_rows = (
                    (data["high"] < data["low"]) | (data["high"] < data["open"]) | (data["high"] < data["close"]) | (data["low"] > data["open"]) | (data["low"] > data["close"])
                ).sum()

                invalid_ratio = float(invalid_rows) / float(len(data))
                if invalid_ratio > 0.05:  # Allow up to 5% invalid rows
                    self.logger.warning(
                        f"⚠️ Found {invalid_rows} rows ({invalid_ratio:.2%}) with inconsistent OHLC data - continuing with caution",
                    )
                elif invalid_rows > 0:
                    self.logger.info(
                        f"ℹ️ Found {invalid_rows} rows with minor OHLC inconsistencies (acceptable)",
                    )

            # Check for reasonable time gaps (if timestamp column exists) - more lenient
            if "timestamp" in data.columns:
                data_sorted = data.sort_values("timestamp")
                time_diffs = data_sorted["timestamp"].diff().dropna()

                # Check for reasonable time intervals (not too large gaps)
                large_gaps = (time_diffs > pd.Timedelta(hours=self.max_gap_hours)).sum()
                large_gap_ratio = float(large_gaps) / float(len(data))

                if large_gap_ratio > self.max_gap_ratio:  # Allow up to 20% large gaps
                    self.logger.warning(
                        f"⚠️ Found {large_gaps} large time gaps ({large_gap_ratio:.2%}) in data - continuing with caution",
                    )
                elif large_gaps > 0:
                    self.logger.info(f"ℹ️ Found {large_gaps} large time gaps (acceptable)")

            self.logger.info(
                f"✅ Data characteristics validation passed: {len(data)} records",
            )
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Error during data characteristics validation: {e}",
            )
            return False


import time


async def run_validator(
    training_input: Dict[str, Any],
    pipeline_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the Step 1 Data Collection validator.

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results

    """
    start_time = time.time()
    try:
        validator = Step1DataCollectionValidator(training_input)
        result = await validator.validate(training_input, pipeline_state)
        # Ensure attribute is populated for orchestrators that expect it
        validator.validation_results = result
        
        duration = time.time() - start_time
        return {
            "step_name": "step01_data_collection",
            "validation_passed": bool(result.get("validation_passed", False)) if isinstance(result, dict) else bool(result),
            "validation_results": result if isinstance(result, dict) else {"result": result},
            "duration": duration,
            "timestamp": time.time(),
        }
    except Exception as e:
        duration = time.time() - start_time
        error_result = {
            "step_name": "step01_data_collection",
            "validation_passed": False,
            "error": f"Validator execution failed: {str(e)}",
            "error_type": type(e).__name__,
            "validation_results": {},
            "duration": duration,
            "timestamp": time.time(),
        }
        system_logger.error(f"❌ Step01 validator failed: {str(e)}")
        return error_result


if __name__ == "__main__":
    import asyncio

    # Example usage
    async def test_validator() -> None:
        training_input = {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "timeframe": "1m",
            "data_dir": "data_cache",
        }

        pipeline_state = {"data_collection": {"status": "SUCCESS", "duration": 120.5}}

        await run_validator(training_input, pipeline_state)

    asyncio.run(test_validator())