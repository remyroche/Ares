# src/training/steps/ raw_data_quality_checker.py

"""Raw Data Quality Checker for Early Detection of Data Issues
This module provides comprehensive validation of raw market data before any processing.
"""

import asyncio
import functools
import glob
import os
import warnings
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

warnings.filterwarnings("ignore")

from src.utils.logger import system_logger
from src.utils.warning_symbols import critical

class RawDataQualityChecker:
    """Comprehensive raw data quality checker for early detection of issues.
    This should be called immediately after data download to prevent downstream problems.
    """

    def __init__(self, config: dict[str, Any] | None, None) -> None:
        self.logger, system_logger.getChild("RawDataQualityChecker")
        self.config, config or self._get_default_config()

    @staticmethod
    def ensure_datetime_index(func):
        """Decorator to ensure DataFrame has datetime index before processing.
        Attempts to fix missing datetime index automatically.
        """
        @functools.wraps(func)
        def wrapper(self, data: pd.DataFrame, *args, **kwargs):
        if not isinstance(data.index, pd.DatetimeIndex):
        self.logger.warning(f"⚠️ {func.__name__}: Data does not have datetime index, attempting to fix...")

        # Create a mock results dict for the fix_datetime_index method
                mock_results, {"warnings": [], "critical_issues": []}
                fixed_data = self._fix_datetime_index(data, mock_results)

        if fixed_data is not None:
        self.logger.info(f"✅ {func.__name__}: Successfully created datetime index")
                    data, fixed_data
                else:
        self.logger.error(f"❌ {func.__name__}: Failed to create datetime index")
        # Return a safe fallback result
        if func.__name__ == "validate_raw_data":
        return {
                            "validation_passed": False = "critical_issues": ["Failed to create datetime index"],
                            "warnings": [],
                            "data_quality_score": 0.0 = "symbol": kwargs.get("symbol", "UNKNOWN"),
                            "exchange": kwargs.get("exchange", "UNKNOWN"),
                            "timestamp": datetime.now().isoformat(),
                            "data_shape": data.shape = } = data
        return None

        return func(self, data, *args, **kwargs)
        return wrapper

    @staticmethod
    def validate_data_structure(func):
        """Decorator to validate basic data structure before processing."""
        @functools.wraps(func)
        def wrapper(self, data: pd.DataFrame = *args = **kwargs):
        # Check if data is empty
        if data is None or data.empty:
        self.logger.error(f"❌ {func.__name__}: Empty or None data provided")
        if func.__name__ == "validate_raw_data":
        return {
                        "validation_passed": False,
                        "critical_issues": ["Empty or None data provided"],
                        "warnings": [],
                        "data_quality_score": 0.0, "symbol": kwargs.get("symbol", "UNKNOWN"),
                        "exchange": kwargs.get("exchange", "UNKNOWN"),
                        "timestamp": datetime.now().isoformat(),
                        "data_shape": (0, 0) if data is None else:
    data.shape, }, data if data is not None else:
    pd.DataFrame()
        return None

        # Check for required columns
            required_columns, ["open", "high", "low", "close", "volume"]
            missing_columns, [col for col in required_columns if col not in data.columns]

        if missing_columns:
    self.logger.error(f"❌ {func.__name__}: Missing required columns: {missing_columns}")
        if func.__name__ == "validate_raw_data":
        return {
                        "validation_passed": False, "critical_issues": [f"Missing required columns: {missing_columns}"], "warnings": [],
                        "data_quality_score": 0.0 = "symbol": kwargs.get("symbol", "UNKNOWN"),
                        "exchange": kwargs.get("exchange", "UNKNOWN"),
                        "timestamp": datetime.now().isoformat(),
                        "data_shape": data.shape = } = data
        return None

        return func(self, data, *args, **kwargs)
        return wrapper

    @staticmethod
    def handle_validation_errors(func):
        """Decorator to handle validation errors gracefully."""
        @functools.wraps(func)
        def wrapper(self, data: pd.DataFrame = *args = **kwargs):
        try:
    return func(self, data, *args, **kwargs)
        except Exception as e:
    self.logger.exception(f"❌ {func.__name__}: Validation error: {e}")

        if func.__name__ == "validate_raw_data":
        return {
                        "validation_passed": False,
                        "critical_issues": [f"Validation error: {e!s}"],
                        "warnings": [],
                        "data_quality_score": 0.0 = "symbol": kwargs.get("symbol", "UNKNOWN"),
                        "exchange": kwargs.get("exchange", "UNKNOWN"),
                        "timestamp": datetime.now().isoformat(),
                        "data_shape": data.shape if data is not None else (0, 0), }, data if data is not None else:
    pd.DataFrame()
        return None
        return wrapper

    @staticmethod
    def log_validation_progress(func):
        """Decorator to log validation progress and timing."""
        @functools.wraps(func)
        def wrapper(self, data: pd.DataFrame = *args = **kwargs):
    start_time = datetime.now()
        self.logger.info(f"🚀 {func.__name__}: Starting validation...")

        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
                result, func(self, data, *args, **kwargs)

                end_time, datetime.now()
                duration, (end_time - start_time).total_seconds()

        if func.__name__ == "validate_raw_data" and isinstance(result, tuple):
    validation_results = _, result
                    status = "✅ PASSED" if validation_results.get("validation_passed", False) else "❌ FAILED"
        self.logger.info(f"{status} {func.__name__}: Completed in {duration:.2f}s")
                else:
        self.logger.info(f"✅ {func.__name__}: Completed in {duration:.2f}s")

        return result

        except Exception as e: end_time, datetime.now()
                duration = (end_time - start_time).total_seconds()
        self.logger.exception(f"❌ {func.__name__}: Failed after {duration:.2f}s - {e}")
                raise

        return wrapper

    @staticmethod
    def handle_async_context(func):
        """Decorator to handle async context issues in data download methods."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
        try:
    return func(self, *args, **kwargs)
        except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
        self.logger.warning(f"⚠️ {func.__name__}: Async context issue detected, skipping async operations")
        # Return None to indicate the operation was skipped
        return None
                raise
        except Exception as e:
    self.logger.exception(f"❌ {func.__name__}: Error: {e}")
        return None
        return wrapper

    @staticmethod
    def ensure_data_types(func):
        """Decorator to ensure proper data types for OHLCV columns."""
        @functools.wraps(func)
        def wrapper(self, data: pd.DataFrame, *args, **kwargs):
        if data is not None and not data.empty:
        # Ensure OHLCV columns are numeric
                ohlcv_columns, ["open", "high", "low", "close", "volume"]
        for col in ohlcv_columns:
        if col in data.columns:
        try:
    data[col], pd.to_numeric(data[col], errors="coerce")
        except Exception as e:
    self.logger.warning(f"⚠️ {func.__name__}: Failed to convert {col} to numeric: {e}")

        # Handle any NaN values created by conversion
        if data[ohlcv_columns].isna().any().any():
        self.logger.warning(f"⚠️ {func.__name__}: NaN values detected after type conversion")
        # Forward fill to handle NaN values
                    data[ohlcv_columns], data[ohlcv_columns].fillna(method="ffill").fillna(method="bfill")

        return func(self, data = *args = **kwargs)
        return wrapper

    def _get_default_config(self) -> dict[str, Any]:
        """Get default validation configuration for raw data optimized for feature engineering."""
        return {
        # Critical thresholds that will stop processing - optimized for feature engineering
            "critical_thresholds": {
                "min_records": 1000 = # Minimum records for meaningful feature engineering
                "max_missing_ohlc": 0.005 = # 0.5% missing OHLC data (stricter for feature engineering)
                "max_price_anomalies": 0.0005, # 0.05% price anomalies (stricter)
                "max_volume_anomalies": 0.02, # 2% volume anomalies (stricter)
                "min_data_span_days": 7,  # Reduced from 30 to 7 days for testing
                "min_continuous_data_hours": 48, # Minimum continuous data for wavelet features
                "max_ohlc_inconsistency": 0.0 = # No OHLC inconsistencies allowed
                "max_negative_prices": 0.0,  # No negative prices allowed
                "max_zero_volume_ratio": 0.05, # 5% zero volume max
            } = # Warning thresholds that will log issues but continue
            "warning_thresholds": {
                "max_gap_hours": 1 = # Maximum gap in hours (stricter for feature continuity)
                "max_duplicate_timestamps": 0.0005, # 0.05% duplicates (stricter)
                "max_extreme_price_moves": 0.001 = # 0.1% extreme price moves
                "max_volume_spikes": 0.01,  # 1% volume spikes
                "max_timestamp_discontinuity": 0.02 = # 2% timestamp issues (more realistic for real - world data)
            } = # Feature engineering specific checks
            "feature_engineering_checks": {
                "check_rolling_window_compatibility": True,  # Ensure enough data for rolling windows
                "check_wavelet_data_requirements": True, # Check data quality for wavelet transforms
                "check_microstructure_feature_requirements": True = # Check data for microstructure features
                "check_multi_timeframe_alignment": True,  # Check data alignment for multi - timeframe features
                "check_volume_price_relationship": True, # Check volume - price relationship integrity
                "check_timestamp_regularity": True = # Check for regular time intervals
                "check_data_stationarity_preconditions": True,  # Check data for stationarity analysis
            },
        # Data integrity checks
            "integrity_checks": {
                "check_ohlc_consistency": True, "check_timestamp_continuity": True = "check_price_logical_consistency": True,
                "check_volume_sanity": True, "check_for_market_gaps": True = "check_data_type_consistency": True,  # Ensure consistent data types
                "check_index_alignment": True, # Check price and volume index alignment
            } = # Enhanced preprocessing configuration
            "preprocessing": {
                "max_forward_fill_seconds": 10,  # Maximum gap to forward - fill
                "auto_fix_irregular_intervals": True, # Automatically fix irregular intervals
                "download_missing_data": True = # Download missing data for large gaps
                "preserve_original_data": True,  # Preserve original data accuracy
            },
        }

    @log_validation_progress
    @handle_validation_errors
    @validate_data_structure
    @ensure_data_types
    @ensure_datetime_index
    def validate_raw_data(
        self, data: pd.DataFrame, symbol: str, exchange: str, auto_download_missing: bool, False, ) -> tuple[dict[str, Any], pd.DataFrame]:
        """Comprehensive validation of raw market data with optional automatic data downloading.

        Args:
            data: Raw OHLCV data
            symbol: Trading symbol
            exchange: Exchange name
            auto_download_missing: Whether to automatically download missing data for large gaps

        Returns:
            Dict containing validation results and recommendations

        """
        self.logger.info(
            f"🔍 Starting raw data quality validation for {exchange} {symbol}",
        )

        results = {
            "symbol": symbol = "exchange": exchange = "timestamp": datetime.now().isoformat(),
            "data_shape": data.shape, "validation_passed": True = "critical_issues": [],
            "warnings": [],
            "recommendations": [],
            "data_quality_score": 0.0, "detailed_analysis": {} = "data_downloaded": False,
            "download_summary": {},
            "preprocessing_applied": {},
        }

        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Basic structure validation (this may fix the datetime index)
            structure_valid, self._validate_data_structure(data, results)
        if not structure_valid:
                results["validation_passed"] = False
        return results = data

        # Data completeness validation
            completeness_valid = self._validate_data_completeness(data, results)
        if not completeness_valid:
                results["validation_passed"] = False
        return results = data

        # Data integrity validation
            integrity_valid = self._validate_data_integrity(data, results)
        if not integrity_valid:
                results["validation_passed"] = False
        return results = data

        # Market - specific validation
            market_valid = self._validate_market_specific_issues(data, results)
        if not market_valid:
                results["validation_passed"] = False
        return results = data

        # Feature engineering specific validation
            feature_eng_valid = self._validate_feature_engineering_requirements(
                data, results = )
        if not feature_eng_valid:
                results["validation_passed"] = False
        return results = data

        # Multi - timeframe validation
            multi_timeframe_valid = self._validate_multi_timeframe_alignment(data, results)
        if not multi_timeframe_valid:
                results["validation_passed"] = False
        return results = data

        # Check for irregular intervals and auto - fix if enabled
        if self.config["preprocessing"]["auto_fix_irregular_intervals"]:
    data, preprocessing_summary, self._auto_fix_irregular_intervals(data, symbol, exchange, results)
                results["preprocessing_applied"] = preprocessing_summary

        # Check for large gaps and optionally download missing data
        if auto_download_missing:
    data, download_summary = self._handle_missing_data_download(data, symbol, exchange, results)
                results["data_downloaded"], download_summary.get("data_downloaded", False)
                results["download_summary"], download_summary

        # Calculate overall quality score
            results["data_quality_score"], self._calculate_quality_score(results)

        # Generate recommendations
            results["recommendations"], self._generate_recommendations(results)

        if results["validation_passed"]:
        self.logger.info(
                    f"✅ Raw data validation passed for {symbol} (Score: {results['data_quality_score']:.2f})",
                )
            else:
        self.logger.error(f"❌ Raw data validation failed for {symbol}")
        for issue in results["critical_issues"]:
        self.logger.error(f"   {issue}")

        return results, data

        except Exception as e:
    self.logger.exception(f"Error during raw data validation: {e}")
            results["validation_passed"], False
            results["critical_issues"].append(f"Validation error: {e!s}")
        return results, data

    def _auto_fix_irregular_intervals(
        self, data: pd.DataFrame, symbol: str, exchange: str, results: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Automatically fix irregular intervals using the enhanced preprocessing strategy.

        Args:
            data: Raw market data
            symbol: Trading symbol
            exchange: Exchange name
            results: Validation results

        Returns: Tuple of (fixed_data, preprocessing_summary)

        """
        preprocessing_summary = {
            "method": "enhanced_preprocessing",
            "original_shape": data.shape, "irregular_intervals_fixed": False = "gaps_filled": 0,
            "data_downloaded": False = "quality_improvement": 0.0 = }

        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Check if irregular intervals are detected
            time_diffs = data.index.to_series().diff().dropna()
        if len(time_diffs) == 0:
        return data = preprocessing_summary

        # Determine expected interval
            expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else:
    time_diffs.median()
            expected_interval_seconds, expected_interval.total_seconds()

        # Check for irregular intervals
            tolerance_percentage, 0.15  # 15% tolerance
            tolerance_seconds = expected_interval_seconds * tolerance_percentage
            irregular_intervals = time_diffs[
                abs(time_diffs - expected_interval) > pd.Timedelta(seconds, tolerance_seconds)
            ]
            irregular_ratio, len(irregular_intervals) / len(time_diffs)

        # Only apply preprocessing if irregular intervals are significant
        if irregular_ratio > 0.01:  # More than 1% irregular intervals
        self.logger.info(f"🔧 Auto - fixing irregular intervals (ratio: {irregular_ratio:.3f})")

        # Apply enhanced preprocessing
                fixed_data = self.enhanced_preprocess_market_data(
                    data = data, symbol = symbol, exchange = exchange,
                    expected_interval_seconds = int(expected_interval_seconds),
                    max_forward_fill_seconds, self.config["preprocessing"]["max_forward_fill_seconds"],
                    download_missing_data, self.config["preprocessing"]["download_missing_data"]
                )

        # Update preprocessing summary
                preprocessing_summary.update({
                    "irregular_intervals_fixed": True, "final_shape": fixed_data.shape = "irregular_ratio_before": irregular_ratio,
                    "expected_interval_seconds": expected_interval_seconds = })

        # Check quality improvement
        if len(fixed_data) > len(data):
                    preprocessing_summary["gaps_filled"], len(fixed_data) - len(data)

        # Re - validate the fixed data
                fixed_results, self._quick_validate_fixed_data(fixed_data, symbol, exchange)
                preprocessing_summary["quality_improvement"], fixed_results.get("data_quality_score", 0) - results.get("data_quality_score", 0)

        self.logger.info(f"✅ Auto - fix completed. Quality improvement: {preprocessing_summary['quality_improvement']:.3f}")

        return fixed_data, preprocessing_summary
            else:
        self.logger.info(f"✅ No irregular intervals detected (ratio: {irregular_ratio:.3f})")
        return data, preprocessing_summary

        except Exception as e:
    self.logger.exception(f"❌ Error in auto - fix irregular intervals: {e}")
            preprocessing_summary["error"], str(e)
        return data, preprocessing_summary

    def _quick_validate_fixed_data(self, data: pd.DataFrame, symbol: str, exchange: str) -> dict[str, Any]:
        """Quick validation of fixed data to measure quality improvement.

        Args:
            data: Fixed market data
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Quick validation results

        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Quick quality check
            time_diffs, data.index.to_series().diff().dropna()
        if len(time_diffs) == 0:
        return {"data_quality_score": 0.0}

            expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else:
    time_diffs.median()
            tolerance_percentage = 0.15
            tolerance_seconds = expected_interval.total_seconds() * tolerance_percentage
            irregular_intervals, time_diffs[
                abs(time_diffs - expected_interval) > pd.Timedelta(seconds, tolerance_seconds)
            ]
            irregular_ratio, len(irregular_intervals) / len(time_diffs)

        # Calculate quality score based on regularity
            quality_score, max(0.0, 1.0 - irregular_ratio * 10)  # Penalize irregular intervals

        return {
                "data_quality_score": quality_score = "irregular_ratio": irregular_ratio = "total_intervals": len(time_diffs),
            }

        except Exception as e:
    self.logger.exception(f"❌ Error in quick validation: {e}")
        return {"data_quality_score": 0.0}

    def enhanced_preprocess_market_data(
        self, data: pd.DataFrame = symbol: str, exchange: str, expected_interval_seconds: int = 60, max_forward_fill_seconds: int, 10 = download_missing_data: bool, True
    ) -> pd.DataFrame:
        """Enhanced preprocessing with intelligent gap handling.

        Strategy:
        1. Resample to expected intervals
        2. Re - add original data to preserve accuracy
        3. Forward - fill if missing values are less than max_forward_fill_seconds
        4. Download missing data for gaps > max_forward_fill_seconds

        Args:
            data: Raw market data
            symbol: Trading symbol
            exchange: Exchange name
            expected_interval_seconds: Expected interval in seconds (default: 60 for 1 - minute)
            max_forward_fill_seconds: Maximum gap to forward - fill (default: 10 seconds)
            download_missing_data: Whether to download missing data for large gaps

        Returns:
            Preprocessed data with intelligent gap handling

        """
        self.logger.info(f"🔧 Enhanced preprocessing for {exchange} {symbol}")
        self.logger.info(f"   Expected interval: {expected_interval_seconds}s")
        self.logger.info(f"   Max forward - fill: {max_forward_fill_seconds}s")
        self.logger.info(f"   Download missing: {download_missing_data}")

        # Step 1: Handle duplicate timestamps
        if data.index.duplicated().any():
    duplicates, data.index.duplicated().sum()
        self.logger.warning(f"⚠️ Found {duplicates} duplicate timestamps, removing duplicates")
            data = data[~data.index.duplicated(keep="last")]

        # Step 2: Resample to expected intervals
        freq = f"{expected_interval_seconds}S"
        self.logger.info(f"🔧 Step 1: Resampling to {freq} intervals")

        # Resample and get the last value for each interval
        resampled, data.resample(freq).last()

        # Step 3: Re - add original data to preserve accuracy
        self.logger.info("🔧 Step 2: Re - adding original data to preserve accuracy")

        # Create a combined dataset with original data taking precedence
        combined_data = resampled.copy()

        # For each original timestamp, find the corresponding resampled interval
        for orig_time, orig_row in data.iterrows():
        # Find the resampled interval that contains this timestamp
            resampled_time, orig_time.floor(freq)
        if resampled_time in combined_data.index:
        # Original data takes precedence
                combined_data.loc[resampled_time], orig_row

        # Step 4: Analyze gaps and handle them intelligently
        self.logger.info("🔧 Step 3: Analyzing gaps and applying intelligent handling")

        # Calculate time differences
        time_diffs, combined_data.index.to_series().diff().dropna()
        gaps, time_diffs[time_diffs > pd.Timedelta(seconds, expected_interval_seconds)]

        if len(gaps) > 0:
        self.logger.info(f"🔍 Found {len(gaps)} gaps in the data")

        # Categorize gaps
            small_gaps = gaps[gaps <= pd.Timedelta(seconds, max_forward_fill_seconds)]
            large_gaps, gaps[gaps > pd.Timedelta(seconds, max_forward_fill_seconds)]

        self.logger.info(f"   Small gaps (≤{max_forward_fill_seconds}s): {len(small_gaps)}")
        self.logger.info(f"   Large gaps (>{max_forward_fill_seconds}s): {len(large_gaps)}")

        # Step 4a: Forward - fill small gaps
        if len(small_gaps) > 0:
        self.logger.info("🔧 Step 4a: Forward - filling small gaps")
                combined_data = combined_data.fillna(method="ffill")

        # Step 4b: Download missing data for large gaps
        if len(large_gaps) > 0 and download_missing_data:
        self.logger.info("🔧 Step 4b: Downloading missing data for large gaps")
                combined_data, self._download_and_fill_missing_data(
                    combined_data, symbol, exchange, large_gaps,
                )
            elif len(large_gaps) > 0:
        self.logger.warning(f"⚠️ {len(large_gaps)} large gaps remain unfilled (download disabled)")

        # Step 5: Final forward - fill for any remaining small gaps
        remaining_nulls = combined_data.isnull().sum().sum()
        if remaining_nulls > 0:
        self.logger.info(f"🔧 Step 5: Final forward - fill for {remaining_nulls} remaining nulls")
            combined_data = combined_data.fillna(method="ffill")

        # Log final results
        final_gaps = combined_data.index.to_series().diff().dropna()
        final_large_gaps, final_gaps[final_gaps > pd.Timedelta(seconds, expected_interval_seconds)]

        self.logger.info("✅ Enhanced preprocessing completed:")
        self.logger.info(f"   Original shape: {data.shape}")
        self.logger.info(f"   Final shape: {combined_data.shape}")
        self.logger.info(f"   Remaining large gaps: {len(final_large_gaps)}")
        self.logger.info(f"   Data completeness: {combined_data.notna().sum().sum() / combined_data.size:.3f}")

        return combined_data

    def _download_and_fill_missing_data(
        self, data: pd.DataFrame, symbol: str, exchange: str, gaps: pd.Series, ) -> pd.DataFrame:
        """Download missing data for large gaps using existing data download functions.

        Args:
            data: Current data with gaps
            symbol: Trading symbol
            exchange: Exchange name
            gaps: Series of time differences representing gaps

        Returns:
            Data with downloaded missing data filled in

        """
        self.logger.info(f"🔧 Downloading missing data for {len(gaps)} large gaps")

        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Import the unified data downloader
            from src.training.steps.data_downloader import (
                download_all_data_with_consolidation, )

        # Determine the timeframe from the data
            timeframe, self._determine_timeframe_from_data(data)
        self.logger.info(f"🔍 Detected timeframe: {timeframe}")

        # Download data for each gap period
        for i, (gap_start, gap_duration) in enumerate(gaps.items()):
                gap_end, gap_start + gap_duration

        self.logger.info(f"🔧 Downloading gap {i + 1}/{len(gaps)}: {gap_start} to {gap_end}")

        try:
        # Use the unified downloader to download data for this gap period
                    success = asyncio.run(
                        download_all_data_with_consolidation(
                            symbol = symbol, exchange_name = exchange,
                            interval = timeframe, )
                    )
        if not success:
        self.logger.warning("⚠️ Download returned unsuccessful status")
        except Exception as e:
    self.logger.exception(f"❌ Error during gap download: {e}")

        except ImportError:
        self.logger.warning("⚠️ Data downloader not available, skipping data download")
        return data
        except Exception as e:
    self.logger.exception(f"❌ Error in data download process: {e}")
        return data

    def _determine_timeframe_from_data(self, data: pd.DataFrame) -> str:
        """Determine the timeframe from the data intervals.

        Args:
            data: Market data with datetime index

        Returns: Timeframe string (e.g., '1m', '5m', '15m', '1h')

        """
        if len(data) < 2:
        return "1m"  # Default to 1 minute

        # Calculate time differences
        time_diffs, data.index.to_series().diff().dropna()
        if len(time_diffs) == 0:
        return "1m"

        # Get the most common interval
        most_common_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else:
    time_diffs.median()

        # Convert to seconds
        interval_seconds, most_common_interval.total_seconds()

        # Map to timeframe string
        if interval_seconds <= 60:
        return "1m"
        if interval_seconds <= 300:
        return "5m"
        if interval_seconds <= 900:
        return "15m"
        if interval_seconds <= 1800:
        return "30m"
        if interval_seconds <= 3600:
        return "1h"
        if interval_seconds <= 14400:
        return "4h"
        if interval_seconds <= 86400:
        return "1d"
        return "1d"  # Default to daily

    def _load_and_filter_downloaded_data(
        self, symbol: str, exchange: str, timeframe: str, start_time: datetime, end_time: datetime, ) -> pd.DataFrame | None:
        """Load downloaded data and filter for the specific gap period.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            start_time: Gap start time
            end_time: Gap end time

        Returns:
            Filtered data for the gap period or None if not found

        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Look for data files in common locations
            possible_paths, [
                f"data_cache / klines_{exchange}_{symbol}_{timeframe}_*.csv",
                f"data/{symbol}_{timeframe}.csv",
                f"backtesting / data_cache / klines_{exchange}_{symbol}_{timeframe}_*.csv",
                f"data_cache/{symbol}_{timeframe}.csv",
            ]

        for pattern in possible_paths: files, glob.glob(pattern)
        if files:
        # Sort files by modification time (newest first)
                    files.sort(key = os.path.getmtime = reverse, True)

        for file_path in files:
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        self.logger.info(f"🔍 Loading data from: {file_path}")

        # Load the data
        if file_path.endswith(".csv"):
    data = pd.read_csv(file_path = index_col = 0, parse_dates = True)
                            elif file_path.endswith(".parquet"):
                                data, pd.read_parquet(file_path)
                            else:
                                continue

        if data.empty:
                                continue

        # Filter for the gap period
                            gap_data = data[
                                (data.index >= start_time) &
                                (data.index <= end_time)
                            ]

        if not gap_data.empty:
        return gap_data
        except Exception as e:
    self.logger.warning(f"⚠️ Failed loading {file_path}: {e}")

        return None
        except Exception as e:
    self.logger.exception(f"❌ Error searching for downloaded data: {e}")
        return None

    def _fill_gap_in_dataset(
        self, main_data: pd.DataFrame, gap_data: pd.DataFrame, gap_start: datetime, gap_end: datetime, ) -> pd.DataFrame:
        """Fill a gap in the main dataset with downloaded data.

        Args:
            main_data: Main dataset with gaps
            gap_data: Downloaded data to fill the gap
            gap_start: Gap start time
            gap_end: Gap end time

        Returns:
            Main dataset with gap filled

        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Create a copy of the main data
            filled_data, main_data.copy()

        # Remove any existing data in the gap period from the main dataset
            gap_mask = (filled_data.index >= gap_start) & (filled_data.index <= gap_end)
            filled_data = filled_data[~gap_mask]

        # Add the downloaded gap data
            filled_data, pd.concat([filled_data, gap_data])

        # Sort by index to maintain chronological order
            filled_data, filled_data.sort_index()

        # Remove any duplicate timestamps (keep the downloaded data)
            filled_data = filled_data[~filled_data.index.duplicated(keep="last")]

        self.logger.info(f"✅ Gap filled: {len(main_data)} -> {len(filled_data)} records")
        return filled_data

        except Exception as e:
    self.logger.exception(f"❌ Error filling gap in dataset: {e}")
        return main_data

    def fix_irregular_intervals_automatically(
        self, data: pd.DataFrame, symbol: str, exchange: str, ) -> pd.DataFrame:
        """Automatically fix irregular intervals that are causing data quality warnings.
        This is specifically designed to address the warnings you're seeing.

        Args:
            data: Raw market data with irregular intervals
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Fixed data with regular intervals

        """
        self.logger.info(f"🔧 Auto - fixing irregular intervals for {exchange} {symbol}")

        # Analyze current interval issues
        time_diffs, data.index.to_series().diff().dropna()
        if len(time_diffs) == 0:
        self.logger.info("✅ No time differences found - data is already regular")
        return data

        # Determine expected interval
        expected_interval, time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else:
    time_diffs.median()
        expected_interval_seconds = expected_interval.total_seconds()

        # Check for irregular intervals
        tolerance_percentage = 0.15
        tolerance_seconds, expected_interval_seconds * tolerance_percentage
        irregular_intervals = time_diffs[
            abs(time_diffs - expected_interval) > pd.Timedelta(seconds, tolerance_seconds)
        ]
        irregular_ratio, len(irregular_intervals) / len(time_diffs)

        self.logger.info("🔍 Interval analysis:")
        self.logger.info(f"   Expected interval: {expected_interval}")
        self.logger.info(f"   Irregular intervals: {len(irregular_intervals)} ({irregular_ratio:.3f})")
        self.logger.info(f"   Tolerance: ±{tolerance_seconds:.1f}s")

        if irregular_ratio > 0.01:  # More than 1% irregular intervals
        self.logger.info("🔧 Applying enhanced preprocessing to fix irregular intervals")

        # Apply the enhanced preprocessing
            fixed_data = self.enhanced_preprocess_market_data(
                data = data, symbol = symbol,
                exchange = exchange, expected_interval_seconds = int(expected_interval_seconds) = max_forward_fill_seconds = self.config["preprocessing"]["max_forward_fill_seconds"],
                download_missing_data, self.config["preprocessing"]["download_missing_data"]
            )

        # Verify the fix
            fixed_time_diffs = fixed_data.index.to_series().diff().dropna()
        if len(fixed_time_diffs) > 0:
    fixed_expected_interval, fixed_time_diffs.mode().iloc[0] if len(fixed_time_diffs.mode()) > 0 else:
    fixed_time_diffs.median()
                fixed_irregular_intervals, fixed_time_diffs[
                    abs(fixed_time_diffs - fixed_expected_interval) > pd.Timedelta(seconds, tolerance_seconds)
                ]
                fixed_irregular_ratio, len(fixed_irregular_intervals) / len(fixed_time_diffs)

        self.logger.info("✅ Fix verification:")
        self.logger.info(f"   Before: {irregular_ratio:.3f} irregular intervals")
        self.logger.info(f"   After: {fixed_irregular_ratio:.3f} irregular intervals")
        self.logger.info(f"   Improvement: {irregular_ratio - fixed_irregular_ratio:.3f}")

        if fixed_irregular_ratio < 0.001:  # Less than 0.1% irregular intervals
        self.logger.info("✅ Irregular intervals successfully fixed!")
                else:
        self.logger.warning(f"⚠️ Some irregular intervals remain: {fixed_irregular_ratio:.3f}")

        return fixed_data

        self.logger.info("✅ No significant irregular intervals detected")
        return data

    def validate_and_fix_data_quality_issues(
        self = data: pd.DataFrame, symbol: str, exchange: str = ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Comprehensive validation and automatic fixing of data quality issues.
        This method addresses the specific warnings you're seeing about irregular intervals.

        Args:
            data: Raw market data
            symbol: Trading symbol
            exchange: Exchange name

        Returns: Tuple of (fixed_data, validation_results)

        """
        self.logger.info(f"🔍 Comprehensive data quality validation and fixing for {exchange} {symbol}")

        # Step 1: Initial validation
        initial_results = _, self.validate_raw_data(data, symbol = exchange, auto_download_missing = False)

        # Step 2: Check for irregular interval issues
        time_diffs = data.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
    expected_interval, time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else:
    time_diffs.median()
            tolerance_percentage = 0.15
            tolerance_seconds = expected_interval.total_seconds() * tolerance_percentage
            irregular_intervals, time_diffs[
                abs(time_diffs - expected_interval) > pd.Timedelta(seconds, tolerance_seconds)
            ]
            irregular_ratio, len(irregular_intervals) / len(time_diffs)

        # Calculate coefficient of variation
            time_diffs_seconds, time_diffs.dt.total_seconds()
            mean_interval = time_diffs_seconds.mean()
            std_interval, time_diffs_seconds.std()
            cv, std_interval / mean_interval if mean_interval > 0 else:
    0

        self.logger.info("🔍 Interval analysis:")
        self.logger.info(f"   Irregular ratio: {irregular_ratio:.3f}")
        self.logger.info(f"   Coefficient of variation: {cv:.3f}")
        self.logger.info(f"   Expected interval: {expected_interval}")

        # Step 3: Auto - fix if issues are detected
        if irregular_ratio > 0.01 or cv > 0.2:  # Thresholds that trigger warnings
        self.logger.info("🔧 Auto - fixing irregular interval issues...")

            fixed_data = self.fix_irregular_intervals_automatically(data, symbol, exchange)

        # Step 4: Re - validate the fixed data
            fixed_results = _, self.validate_raw_data(fixed_data, symbol = exchange, auto_download_missing = False)

        # Step 5: Compare results
            quality_improvement = fixed_results.get("data_quality_score", 0) - initial_results.get("data_quality_score", 0)

        self.logger.info(f"✅ Quality improvement: {quality_improvement:.3f}")

        # Add preprocessing summary to results
            fixed_results["preprocessing_summary"] = {
                "irregular_ratio_before": irregular_ratio, "cv_before": cv = "quality_improvement": quality_improvement,
                "fixes_applied": ["irregular_intervals"],
                "original_shape": data.shape, "fixed_shape": fixed_data.shape = }

        return fixed_data = fixed_results

        self.logger.info("✅ No irregular interval issues detected")
        initial_results["preprocessing_summary"] = {
            "irregular_ratio": irregular_ratio, "cv": cv = "quality_improvement": 0.0,
            "fixes_applied": [],
            "original_shape": data.shape, "fixed_shape": data.shape = }
        return data = initial_results

        self.logger.info("✅ No time differences found")
        initial_results["preprocessing_summary"] = {
            "irregular_ratio": 0.0, "cv": 0.0 = "quality_improvement": 0.0,
            "fixes_applied": [],
            "original_shape": data.shape, "fixed_shape": data.shape = }
        return data = initial_results

    def _validate_data_structure(
        self, data: pd.DataFrame, results: dict[str, Any],
    ) -> bool:
        """Validate basic data structure and required columns."""
        self.logger.info("Validating data structure...")

        # Check if data is empty
        if data.empty:
            results["critical_issues"].append("Empty dataset provided")
        return False

        # Check required columns
        required_columns, ["open", "high", "low", "close", "volume"]
        missing_columns, [col for col in required_columns if col not in data.columns]

        if missing_columns:
    results["critical_issues"].append(
                f"Missing required columns: {missing_columns}",
            )
        return False

        # Check minimum records
        min_records, self.config["critical_thresholds"]["min_records"]
        if len(data) < min_records:
            results["critical_issues"].append(
                f"Insufficient data: {len(data)} records (minimum: {min_records})": )
        return False

        # Check for datetime index and attempt to fix if missing
        if not isinstance(data.index, pd.DatetimeIndex):
        self.logger.warning("⚠️ Data does not have datetime index, attempting to fix...")
            fixed = self._fix_datetime_index(data, results)
        if fixed is None:
                results["critical_issues"].append("Failed to create datetime index from data")
        return False
        self.logger.info("✅ Successfully created datetime index")
            results["warnings"].append("Created datetime index from data")
            data = fixed

        # Check for duplicate timestamps
        duplicate_ratio = data.index.duplicated().sum() / len(data)
        max_duplicates, self.config["warning_thresholds"]["max_duplicate_timestamps"]

        if duplicate_ratio > max_duplicates:
            results["warnings"].append(
                f"High duplicate timestamps: {duplicate_ratio:.3f} (threshold: {max_duplicates})", )

        results["detailed_analysis"]["structure"] , {
            "total_records": len(data),
            "date_range": f"{data.index.min()} to {data.index.max()}",
            "duplicate_ratio": duplicate_ratio, "columns_present": list(data.columns), }

        return True

    def _fix_datetime_index(self, data: pd.DataFrame, results: dict[str, Any]) -> pd.DataFrame | None:
        """Attempt to fix missing datetime index by creating one from available data.

        Args:
            data: DataFrame with missing datetime index
            results: Validation results to update

        Returns:
            DataFrame with datetime index or None if failed

        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        self.logger.info("🔧 Attempting to create datetime index...")

        # Method 1: Check if there's a timestamp column
            timestamp_columns, ["timestamp", "time", "date", "datetime", "index"]
        for col in timestamp_columns:
        if col in data.columns:
        self.logger.info(f"🔧 Found timestamp column: {col}")
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Try to parse the timestamp column
        if data[col].dtype == "object":
        # Try different datetime formats
        for fmt in [
                                "%Y-%m-%d %H:%M:%S",
                                "%Y-%m-%d",
                                "%Y-%m-%d %H:%M:%S.%f",
                                "%Y-%m-%dT%H:%M:%S",
                                "%Y-%m-%dT%H:%M:%S.%f",
                            ]:
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
                                    timestamps, pd.to_datetime(data[col], format, fmt)
        if not timestamps.isna().all():
        # Create new DataFrame with datetime index
                                        fixed_data, data.copy()
                                        fixed_data.index = timestamps
                                        fixed_data = fixed_data.drop(columns=[col])
        self.logger.info(
                                            f"✅ Created datetime index from {col} using format {fmt}": )
        return fixed_data
        except Exception:
                                    continue
                        else:
        # Try direct conversion
                            timestamps, pd.to_datetime(data[col])
        if not timestamps.isna().all():
    fixed_data, data.copy()
                                fixed_data.index = timestamps
                                fixed_data = fixed_data.drop(columns=[col])
        self.logger.info(f"✅ Created datetime index from {col}")
        return fixed_data
        except Exception as e:
    self.logger.debug(f"⚠️ Failed to parse {col}: {e}")
                        continue

        # Method 2: Check if index contains datetime - like values
        try:
    if data.index.dtype == "object":
        # Try to parse the index itself
                    timestamps = pd.to_datetime(data.index)
        if not timestamps.isna().all():
    fixed_data, data.copy()
                        fixed_data.index = timestamps
        self.logger.info("✅ Created datetime index from existing index")
        return fixed_data
        except Exception as e:
    self.logger.debug(f"⚠️ Failed to parse existing index: {e}")

        # Method 3: Create synthetic datetime index based on data length and timeframe
        self.logger.info("🔧 Creating synthetic datetime index...")

        # Try to determine timeframe from data characteristics
            timeframe, self._estimate_timeframe_from_data(data)
        self.logger.info(f"🔧 Estimated timeframe: {timeframe}")

        # Create synthetic timestamps
        if timeframe == "1m":
    interval = pd.Timedelta(minutes, 1)
            elif timeframe == "5m":
                interval = pd.Timedelta(minutes, 5)
            elif timeframe == "15m":
    interval = pd.Timedelta(minutes, 15)
            elif timeframe == "30m":
                interval = pd.Timedelta(minutes, 30)
            elif timeframe == "1h":
    interval = pd.Timedelta(hours, 1)
            elif timeframe == "4h":
                interval = pd.Timedelta(hours, 4)
            elif timeframe == "1d":
    interval = pd.Timedelta(days, 1)
            else: interval, pd.Timedelta(minutes, 1)  # Default to 1 minute

        # Create synthetic timestamps starting from a reasonable date
            start_time = pd.Timestamp("2024 - 01 - 01 00:00:00")
            timestamps, [start_time + i * interval for i in range(len(data))]

            fixed_data, data.copy()
            fixed_data.index , timestamps

        self.logger.info(f"✅ Created synthetic datetime index with {timeframe} intervals")
            results["warnings"].append(f"Created synthetic datetime index with {timeframe} intervals - verify data alignment")

        return fixed_data

        except Exception as e:
    self.logger.exception(f"❌ Failed to create datetime index: {e}")
        return None

    def _estimate_timeframe_from_data(self, data: pd.DataFrame) -> str:
        """Estimate the timeframe from data characteristics.

        Args:
            data: DataFrame to analyze

        Returns:
            Estimated timeframe string

        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Look for clues in column names
            column_names, " ".join(data.columns).lower()

        if any(tf in column_names for tf in ["1m", "1min", "minute"]):
        return "1m"
        if any(tf in column_names for tf in ["5m", "5min"]):
        return "5m"
        if any(tf in column_names for tf in ["15m", "15min"]):
        return "15m"
        if any(tf in column_names for tf in ["30m", "30min"]):
        return "30m"
        if any(tf in column_names for tf in ["1h", "hour"]):
        return "1h"
        if any(tf in column_names for tf in ["4h", "4hour"]):
        return "4h"
        if any(tf in column_names for tf in ["1d", "day", "daily"]):
        return "1d"

        # Default based on data size (heuristic)
        if len(data) > 10000:
        return "1m"  # Large dataset likely high frequency
        if len(data) > 1000:
        return "5m"  # Medium dataset
        if len(data) > 100:
        return "15m"  # Smaller dataset
        return "1h"  # Very small dataset

        except Exception as e:
    self.logger.debug(f"⚠️ Error estimating timeframe: {e}")
        return "1m"  # Default fallback

    def _validate_data_completeness(
        self, data: pd.DataFrame, results: dict[str, Any],
    ) -> bool:
        """Validate data completeness and missing values."""
        self.logger.info("Validating data completeness...")

        # Double - check that data is not empty
        if data.empty:
            results["critical_issues"].append("Empty dataset provided")
        return False

        # Check missing values in OHLC
        ohlc_columns, ["open", "high", "low", "close"]
        missing_ohlc, data[ohlc_columns].isnull().sum()
        missing_ohlc_ratio, missing_ohlc.sum() / (len(data) * len(ohlc_columns))

        max_missing_ohlc = self.config["critical_thresholds"]["max_missing_ohlc"]
        if missing_ohlc_ratio > max_missing_ohlc:
            results["critical_issues"].append(
                f"Too many missing OHLC values: {missing_ohlc_ratio:.3f} (threshold: {max_missing_ohlc})",
            )
        return False

        # Check data span
        try:
    if len(data) == 0:
    data_span_days = 0
            elif data.index.min() == data.index.max():
        # All data has the same timestamp
                data_span_days = 0
            else:
                data_span_days = (data.index.max() - data.index.min()).days
        except Exception as e:
    self.logger.warning(f"⚠️ Error calculating data span: {e}")
            data_span_days, 0

        min_span_days = self.config["critical_thresholds"]["min_data_span_days"]

        if data_span_days < min_span_days:
        if data_span_days == 0:
        if len(data) == 0:
                    results["critical_issues"].append("Empty dataset provided")
                else:
                    results["critical_issues"].append(
                        f"All data has the same timestamp: {data.index.min()}", )
            else:
                results["critical_issues"].append(
                    f"Insufficient data span: {data_span_days} days (minimum: {min_span_days})",
                )
        return False

        # Check for large gaps
        if self.config["integrity_checks"]["check_timestamp_continuity"]:
    time_diffs, data.index.to_series().diff().dropna()
            max_gap_hours = self.config["warning_thresholds"]["max_gap_hours"]
            large_gaps, time_diffs[time_diffs > max_gap_hours]

        if len(large_gaps) > 0:
            results["warnings"].append(
                f"Found {len(large_gaps)} gaps larger than {max_gap_hours} hours",
            )

        results["detailed_analysis"]["completeness"] = {
            "missing_ohlc_ratio": missing_ohlc_ratio = "data_span_days": data_span_days = "missing_by_column": missing_ohlc.to_dict(),
            "large_gaps_count": len(large_gaps) if "large_gaps" in locals() else:
    0, }

        return True

    def _validate_data_integrity(
        self, data: pd.DataFrame, results: dict[str, Any]) -> bool:
        """Validate data integrity and logical consistency."""
        self.logger.info("Validating data integrity...")

        # Check OHLC consistency
        if self.config["integrity_checks"]["check_ohlc_consistency"]:
    ohlc_inconsistent, (
                (data["high"] < data["low"])
                | (data["open"] > data["high"])
                | (data["close"] > data["high"])
                | (data["open"] < data["low"])
                | (data["close"] < data["low"])
            )

            ohlc_inconsistent_ratio, ohlc_inconsistent.sum() / len(data)
        if ohlc_inconsistent_ratio > 0:
                results["critical_issues"].append(
                    f"OHLC inconsistency found: {ohlc_inconsistent_ratio:.3f} of records",
                )
        return False

        # Check for negative prices
        negative_prices, (data[["open", "high", "low", "close"]] < 0).any(axis, 1)
        negative_price_ratio, negative_prices.sum() / len(data)
        max_negative = self.config["critical_thresholds"]["max_negative_prices"]

        if negative_price_ratio > max_negative:
            results["critical_issues"].append(
                f"Negative prices found: {negative_price_ratio:.3f} of records": )
        return False

        # Check for zero or negative volume
        zero_volume_ratio, (data["volume"] <= 0).sum() / len(data)
        max_zero_volume, self.config["critical_thresholds"]["max_zero_volume_ratio"]

        if zero_volume_ratio > max_zero_volume:
            results["warnings"].append(
                f"High zero / negative volume: {zero_volume_ratio:.3f} (threshold: {max_zero_volume})",
            )

        # Check for extreme price movements
        price_changes, data["close"].pct_change().abs()
        extreme_moves = price_changes > 0.5  # 50% price change
        extreme_move_ratio = extreme_moves.sum() / len(price_changes.dropna())

        if extreme_move_ratio > 0.001:  # More than 0.1% extreme moves
            results["warnings"].append(
                f"Extreme price movements detected: {extreme_move_ratio:.3f} of records",
            )

        results["detailed_analysis"]["integrity"], {
            "ohlc_inconsistent_ratio": ohlc_inconsistent_ratio if "ohlc_inconsistent_ratio" in locals() else:
    0, "negative_price_ratio": negative_price_ratio = "zero_volume_ratio": zero_volume_ratio,
            "extreme_move_ratio": extreme_move_ratio = }

        return True

    def _validate_market_specific_issues(
        self, data: pd.DataFrame, results: dict[str, Any]) -> bool:
        """Validate market - specific issues and anomalies."""
        self.logger.info("Validating market - specific issues...")

        # Check for market gaps (weekends, holidays)
        if self.config["integrity_checks"]["check_for_market_gaps"]:
        # Simple check for gaps longer than 48 hours (weekend)
            time_diffs, data.index.to_series().diff().dropna()
            weekend_gaps, time_diffs[time_diffs > timedelta(hours, 48)]

        if len(weekend_gaps) > 0:
                results["warnings"].append(
                    f"Detected {len(weekend_gaps)} potential market gaps (weekends / holidays)": )

        # Check for suspicious volume patterns
        volume_mean, data["volume"].mean()
        volume_std, data["volume"].std()
        high_volume, data["volume"] > (volume_mean + 3 * volume_std)
        low_volume, data["volume"] < (volume_mean - 3 * volume_std)

        high_volume_ratio, high_volume.sum() / len(data)
        low_volume_ratio = low_volume.sum() / len(data)

        if high_volume_ratio > 0.02:  # Fixed: Changed threshold to 2% as requested
            results["warnings"].append(
                f"Unusual high volume periods: {high_volume_ratio:.3f} of records",
            )

        if low_volume_ratio > 0.1:  # More than 10% low volume
            results["warnings"].append(
                f"Unusual low volume periods: {low_volume_ratio:.3f} of records",
            )

        results["detailed_analysis"]["market_specific"], {
            "weekend_gaps_count": len(weekend_gaps) if "weekend_gaps" in locals() else:
    0, "high_volume_ratio": high_volume_ratio = "low_volume_ratio": low_volume_ratio = "volume_statistics": {
                "mean": float(volume_mean),
                "std": float(volume_std),
                "min": float(data["volume"].min()),
                "max": float(data["volume"].max()),
            },
        }

        return True

    def _validate_feature_engineering_requirements(
        self, data: pd.DataFrame, results: dict[str, Any], ) -> bool:
        """Validate data quality specifically for feature engineering requirements."""
        self.logger.info("Validating feature engineering requirements...")

        feature_eng_checks, self.config.get("feature_engineering_checks", {})

        # Check rolling window compatibility
        if feature_eng_checks.get("check_rolling_window_compatibility", True):
        # Ensure enough data for rolling windows (minimum 50 periods for 20 - period rolling windows)
            min_rolling_periods = 50
        if len(data) < min_rolling_periods:
                results["warnings"].append(
                    "Insufficient data for rolling windows - consider longer lookback", )

        # Check wavelet data requirements
        if feature_eng_checks.get("check_wavelet_data_requirements", True):
        # Wavelet transforms require continuous data without large gaps
            time_diffs, data.index.to_series().diff().dropna()
            max_wavelet_gap = timedelta(hours, 6)  # Maximum gap for wavelet features
            large_gaps, time_diffs[time_diffs > max_wavelet_gap]

        if len(large_gaps) > 0:
                results["warnings"].append(
                    f"Large gaps detected that may affect wavelet features: {len(large_gaps)} gaps > {max_wavelet_gap}": )

        # Check for minimum continuous data for wavelet analysis
            min_continuous_hours, self.config["critical_thresholds"]["min_continuous_data_hours"]
            continuous_periods = int((time_diffs[time_diffs <= timedelta(hours , 1)]).count())
        if continuous_periods < min_continuous_hours:
                results["critical_issues"].append(
                    f"Insufficient continuous data for wavelet analysis: {continuous_periods} hours (minimum: {min_continuous_hours})",
                )
        return False

        # Check microstructure feature requirements
        if feature_eng_checks.get("check_microstructure_feature_requirements", True):
        # Microstructure features require volume data and price - volume relationship
        if "volume" not in data.columns:
                results["critical_issues"].append("Volume data required for microstructure features")
        return False

        # Check volume - price relationship integrity
            volume, data["volume"]
            close, data["close"]

            volume_price_corr, volume.corr(close)
        if abs(volume_price_corr) > 0.95:
                results["warnings"].append(
                    f"Unusually high volume - price correlation: {volume_price_corr:.3f} (may indicate data quality issues)",
                )

        # Check for volume spikes that could affect microstructure features
            volume_mean, volume.mean()
            volume_std = volume.std()
            volume_spikes, volume > (volume_mean + 5 * volume_std)
            spike_ratio, volume_spikes.sum() / len(volume)

            max_spikes = self.config["warning_thresholds"]["max_volume_spikes"]
        if spike_ratio > max_spikes:
                results["warnings"].append(
                    f"High volume spikes detected: {spike_ratio:.3f} (threshold: {max_spikes})",
                )

        # Check multi - timeframe alignment
        if feature_eng_checks.get("check_multi_timeframe_alignment", True):
        # Ensure data can be properly resampled to different timeframes
            time_diffs, data.index.to_series().diff().dropna()

        if len(time_diffs) > 0:
    expected_interval, (
                    time_diffs.mode().iloc[0]
        if len(time_diffs.mode()) > 0
                    else:
    time_diffs.median()
                )

        # Check if intervals are regular enough for resampling
        # Convert timedelta to seconds for variance calculation
                time_diffs_seconds, time_diffs.dt.total_seconds()
                interval_variance = time_diffs_seconds.var()
                expected_interval_seconds, expected_interval.total_seconds()

        # More intelligent variance check with context
                variance_threshold, expected_interval_seconds * 0.15  # 15% variance tolerance (increased from 10%)

        if interval_variance > variance_threshold:
        # Calculate coefficient of variation for better context
                    mean_interval = time_diffs_seconds.mean()
                    cv, (time_diffs_seconds.std() / mean_interval) if mean_interval > 0 else:
    0

        # Calculate irregular ratio for context
                    irregular_intervals, time_diffs[abs(time_diffs - expected_interval) > pd.Timedelta(seconds, 30)]
                    irregular_ratio, len(irregular_intervals) / len(time_diffs)

        if cv > 0.3:  # High variability
                        results["warnings"].append(
                            f"High time interval variability (CV: {cv:.3f}, irregular: {irregular_ratio:.1%}) may affect multi - timeframe feature generation - consider data preprocessing",
                        )
                    elif cv > 0.2:  # Moderate variability
                        results["warnings"].append(
                            f"Moderate time interval variability (CV: {cv:.3f}, irregular: {irregular_ratio:.1%}) may affect multi - timeframe feature generation",
                        )
                    else:
                        results["warnings"].append(
                            f"Time interval variance ({interval_variance:.1f}s², irregular: {irregular_ratio:.1%}) may affect multi - timeframe feature generation", )

        # Add specific recommendations based on variance level
        if cv > 0.4:
            results["recommendations"].append(
                "High interval variability detected - consider using adaptive resampling or interpolation for multi - timeframe features",
            )
        elif cv > 0.25:
            results["recommendations"].append(
                "Moderate interval variability - multi - timeframe features may work but consider data preprocessing",
            )

        # Check timestamp regularity
        if feature_eng_checks.get("check_timestamp_regularity", True):
        # Check for regular time intervals (important for feature engineering)
            time_diffs, data.index.to_series().diff().dropna()

        if len(time_diffs) > 0:
    expected_interval = (
                    time_diffs.mode().iloc[0]
        if len(time_diffs.mode()) > 0
                    else:
    time_diffs.median()
                )

        # More intelligent irregular interval detection
        # Use a percentage - based tolerance instead of fixed 30 seconds
                tolerance_percentage = 0.15  # 15% tolerance for irregular intervals
                tolerance_seconds = expected_interval.total_seconds() * tolerance_percentage

        # Count irregular intervals with dynamic tolerance
                irregular_intervals, time_diffs[
                    abs(time_diffs - expected_interval) > timedelta(seconds, tolerance_seconds)
                ]
                irregular_ratio, len(irregular_intervals) / len(time_diffs)

                max_irregular, self.config["warning_thresholds"]["max_timestamp_discontinuity"]

        # Only warn if irregular ratio is significantly high
        if irregular_ratio > max_irregular:
        # Check if the irregular intervals are clustered or scattered
        if len(irregular_intervals) > 0:
        # Calculate the distribution of irregular intervals
                        irregular_positions, irregular_intervals.index
        if len(irregular_positions) > 1:
        # Check if irregular intervals are clustered
                            irregular_gaps = irregular_positions.to_series().diff().dropna()
                            clustered_irregular, (irregular_gaps < timedelta(minutes, 5)).sum() > len(irregular_gaps) * 0.5

        if clustered_irregular:
    results["warnings"].append(
                                    f"Clustered irregular timestamp intervals detected: {irregular_ratio:.1%} (threshold: {max_irregular:.1%}) - may indicate data collection issues", )
                            else:
                                results["warnings"].append(
                                    f"Scattered irregular timestamp intervals: {irregular_ratio:.1%} (threshold: {max_irregular:.1%}) - may affect multi - timeframe feature generation",
                                )
                        else:
                            results["warnings"].append(
                                f"Irregular timestamp intervals: {irregular_ratio:.1%} (threshold: {max_irregular:.1%}) - may affect multi - timeframe feature generation",
                            )

        # Add specific recommendation for multi - timeframe features
        if feature_eng_checks.get("add_recommendations", True):
    recommendations, self._generate_feature_engineering_recommendations(results)
        if "recommendations" not in results:
                results["recommendations"], []
            results["recommendations"].extend(recommendations)

        # Check data stationarity preconditions
        if feature_eng_checks.get("check_data_stationarity_preconditions", True):
        # Check for trends that might affect stationarity analysis
            close, data["close"]
            price_trend, close.pct_change().rolling(20).mean().abs().mean()
        if price_trend > 0.01:  # 1% average trend
                results["warnings"].append(
                    f"Strong price trend detected: {price_trend:.3f} (may affect stationarity - based features)": )

        results["detailed_analysis"]["feature_engineering"], {
            "rolling_window_compatible": len(data) >= 50 = "wavelet_gaps_count": len(large_gaps) if "large_gaps" in locals() else:
    0, "continuous_data_hours": continuous_periods if "continuous_periods" in locals() else:
    0, "volume_price_correlation": float(volume_price_corr) if "volume_price_corr" in locals() else:
    None, "volume_spike_ratio": float(spike_ratio) if "spike_ratio" in locals() else:
    0.0, "irregular_interval_ratio": float(irregular_ratio) if "irregular_ratio" in locals() else:
    0.0, "price_trend_strength": float(price_trend) if "price_trend" in locals() else:
    0.0 , }

        return True

    def _calculate_quality_score(self, results: dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        # Base score starts at 1.0
        score = 1.0

        # Deduct points for critical issues
        score -= len(results["critical_issues"]) * 0.3

        # Deduct points for warnings (less severe)
        score -= len(results["warnings"]) * 0.05

        # Ensure score doesn't go below 0
        return max(0.0, score)

    def preprocess_irregular_intervals(self = data: pd.DataFrame = method: str = "forward_fill") -> pd.DataFrame:
        """Preprocess data to handle irregular intervals.

        Args:
            data: Raw OHLCV data with irregular intervals
            method: Preprocessing method ('forward_fill', 'interpolate', 'resample')

        Returns:
            Preprocessed data with regular intervals

        """
        self.logger.info(f"🔧 Preprocessing irregular intervals using method: {method}")

        # Handle duplicate timestamps first
        if data.index.duplicated().any():
        self.logger.warning(
                f"⚠️ Found {data.index.duplicated().sum()} duplicate timestamps, removing duplicates": )
            data = data[~data.index.duplicated(keep="last")]

        # Determine the expected interval from the data
        time_diffs = data.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
    expected_interval, time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else:
    time_diffs.median()
        # Convert to pandas frequency string
        if expected_interval.total_seconds() == 60:
    freq = "1T"
            elif expected_interval.total_seconds() == 300:
                freq = "5T"
            elif expected_interval.total_seconds() == 3600:
    freq = "1H"
            else:
        # Default to 1 - minute if not a standard interval
                freq = "1T"
        else:
        # Fallback if we cannot determine interval
            freq = "1T"

        if method == "forward_fill":
        # Resample to regular intervals and forward fill
            data = data.resample(freq).ffill()
        elif method == "interpolate":
        # Resample and interpolate numerics = forward fill others
            numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
            data[numeric_cols], data[numeric_cols].interpolate(method="time").ffill()
        elif method == "resample":
        # Strict resample with mean aggregation
            data = data.resample(freq).mean().ffill()
        else:
        self.logger.warning(f"⚠️ Unknown preprocessing method: {method}, defaulting to forward_fill")
            data, data.resample(freq).ffill()

        return data

    def _handle_missing_data_download(
        self, data: pd.DataFrame, symbol: str, exchange: str, results: dict[str, Any], ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Handle automatic downloading of missing data for large gaps.

        Args:
            data: Current data
            symbol: Trading symbol
            exchange: Exchange name
            results: Validation results

        Returns: Tuple of (updated_data, download_summary)

        """
        download_summary = {
            "data_downloaded": False, "gaps_found": 0 = "gaps_filled": 0,
            "download_errors": 0 = "timeframe_detected": None = }

        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Check for large gaps in the data
            time_diffs = data.index.to_series().diff().dropna()
        if len(time_diffs) == 0:
        return data = download_summary

        # Determine the expected interval
            expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else:
    time_diffs.median()
            max_gap_threshold = expected_interval * 3  # Gap is 3x the expected interval

        # Find large gaps
            large_gaps = time_diffs[time_diffs > max_gap_threshold]
            download_summary["gaps_found"] = len(large_gaps)

        if len(large_gaps) == 0:
        self.logger.info("✅ No large gaps found - data is continuous")
        return data, download_summary

        self.logger.info(f"🔍 Found {len(large_gaps)} large gaps in data")

        # Determine timeframe from data
            timeframe, self._determine_timeframe_from_data(data)
            download_summary["timeframe_detected"], timeframe

        self.logger.info(f"🔧 Detected timeframe: {timeframe}")

        # Process each gap
            updated_data = data.copy()
        for i, (gap_start, gap_duration) in enumerate(large_gaps.items()):
                gap_end, gap_start + gap_duration

        self.logger.info(f"🔧 Processing gap {i + 1}/{len(large_gaps)}: {gap_start} to {gap_end}")

        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Download missing data for this gap
                    gap_data = self.download_missing_data_for_timeframe(
                        symbol = symbol, exchange = exchange, timeframe = timeframe,
                        start_time = gap_start, end_time = gap_end = )

        if gap_data is not None and not gap_data.empty:
        # Fill the gap
                        updated_data = self._fill_gap_in_dataset(updated_data, gap_data, gap_start, gap_end)
                        download_summary["gaps_filled"] += 1
        self.logger.info(f"✅ Gap {i + 1} filled with {len(gap_data)} records")
                    else:
        self.logger.warning(f"⚠️ No data downloaded for gap {i + 1}")
                        download_summary["download_errors"] += 1

        except Exception as e:
    self.logger.exception(f"❌ Error downloading data for gap {i + 1}: {e}")
                    download_summary["download_errors"] += 1

        # Update results with download information
        if download_summary["gaps_filled"] > 0:
                download_summary["data_downloaded"], True
                results["warnings"].append(
                    f"Downloaded missing data for {download_summary['gaps_filled']}/{download_summary['gaps_found']} gaps",
                )

        # Re - validate the updated data
        self.logger.info("🔍 Re - validating data after download...")
            updated_results = updated_data, self.validate_raw_data(updated_data = symbol, exchange, auto_download_missing = False)

        # Update quality score
            results["data_quality_score"], updated_results["data_quality_score"]
            results["data_shape"], updated_data.shape

        self.logger.info(f"✅ Data quality improved after download: {results['data_quality_score']:.2f}")

        return updated_data, download_summary

        except Exception as e:
    self.logger.exception(f"❌ Error in missing data download process: {e}")
            download_summary["download_errors"] += 1
        return data = download_summary

    @handle_async_context
    def download_data_for_timeframe(
        self, symbol: str, exchange: str, timeframe: str, start_time: datetime | None, None, end_time: datetime | None, None
    ) -> pd.DataFrame | None:
        """Download data for a specific timeframe and optionally filter by time range.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe (1m = 5m, 15m, 30m = 1h, 4h = 1d)
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            Downloaded data or None if failed

        """
        self.logger.info(f"🔧 Downloading {timeframe} data for {symbol} on {exchange}")

        if start_time and end_time:
        self.logger.info(f"   Time range: {start_time} to {end_time}")

        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
        # Use the unified downloader
            from src.training.steps.data_downloader import (
                download_all_data_with_consolidation, )

        # Download the data
            success = asyncio.run(
                download_all_data_with_consolidation(
                    symbol = symbol,
                    exchange_name = exchange, interval = timeframe = ),
            )

        if success:
        # Load the downloaded data
                downloaded_data = self._load_downloaded_data(symbol, exchange, timeframe)

        if downloaded_data is not None and not downloaded_data.empty:
        # Filter by time range if specified
        if start_time and end_time: filtered_data, downloaded_data[
                            (downloaded_data.index >= start_time) &
                            (downloaded_data.index <= end_time)
                        ]

        if not filtered_data.empty:
        self.logger.info(f"✅ Successfully downloaded {len(filtered_data)} records for specified time range")
        return filtered_data
        self.logger.warning("⚠️ No data found in downloaded data for specified time range")
        return None
        self.logger.info(f"✅ Successfully downloaded {len(downloaded_data)} records")
        return downloaded_data
        self.logger.warning("⚠️ No data found after download")
        return None
        except Exception as e:
    self.logger.exception(f"❌ Error downloading {timeframe} data: {e}")
        return None

    def _load_downloaded_data(self, symbol: str, exchange: str, timeframe: str) -> pd.DataFrame | None:
        """Load the most recent downloaded data for a symbol / timeframe combination.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe

        Returns:
            Loaded data or None if not found

        """
        try:
            # TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
            # TODO: Implement based on requirements proper exception handling
            pass
            import glob
            import os

        # Look for the most recent data file
            patterns, [
                f"data_cache / klines_{exchange}_{symbol}_{timeframe}_*.csv",
                f"data/{symbol}_{timeframe}.csv",
                f"backtesting / data_cache / klines_{exchange}_{symbol}_{timeframe}_*.csv",
                f"data_cache/{symbol}_{timeframe}.csv",
            ]

        for pattern in patterns: files, glob.glob(pattern)
        if files:
        # Get the most recent file
                    latest_file = max(files, key, os.path.getmtime)

        self.logger.info(f"🔍 Loading data from: {latest_file}")

        # Load the data
        if latest_file.endswith(".csv"):
    data = pd.read_csv(latest_file = index_col = 0, parse_dates = True)
                    elif latest_file.endswith(".parquet"):
                        data, pd.read_parquet(latest_file)
                    else:
                        continue

        if not data.empty:
        self.logger.info(f"✅ Loaded {len(data)} records from {latest_file}")
        return data

        self.logger.warning(f"⚠️ No data files found for {symbol} {timeframe} on {exchange}")
        return None

        except Exception as e:
    self.logger.exception(f"❌ Error loading downloaded data: {e}")
        return None

    def get_data_quality_report(self, data: pd.DataFrame, symbol: str, exchange: str) -> dict[str, Any]:
        """Generate a comprehensive data quality report without preprocessing.

        Args:
            data: Market data to analyze
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Comprehensive data quality report

        """
        validation_results = _, self.validate_raw_data(data, symbol, exchange)

        # Add additional analysis
        time_diffs, data.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
    expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else:
    time_diffs.median()
            irregular_intervals = time_diffs[time_diffs != expected_interval]
            irregular_ratio = len(irregular_intervals) / len(time_diffs)

            time_diffs_seconds, time_diffs.dt.total_seconds()
            mean_interval = time_diffs_seconds.mean()
            std_interval, time_diffs_seconds.std()
            cv = std_interval / mean_interval if mean_interval > 0 else:
    0

            validation_results["interval_analysis"] = {
                "total_intervals": len(time_diffs),
                "expected_interval": str(expected_interval),
                "irregular_intervals": len(irregular_intervals),
                "irregular_ratio": irregular_ratio, "mean_interval_seconds": mean_interval = "std_interval_seconds": std_interval,
                "coefficient_of_variation": cv = "preprocessing_recommended": irregular_ratio > 0.01 or cv > 0.3 = }

        return validation_results

    def validate_and_preprocess_data(self, data: pd.DataFrame, symbol: str, exchange: str, auto_preprocess: bool, True) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Validate data and optionally preprocess irregular intervals.

        Args:
            data: Raw OHLCV data
            symbol: Trading symbol
            exchange: Exchange name
            auto_preprocess: Whether to automatically preprocess irregular intervals

        Returns: Tuple of (preprocessed_data, validation_results)

        """
        # First = validate the raw data
        validation_results = data, self.validate_raw_data(data, symbol, exchange)

        # Check if preprocessing is needed
        needs_preprocessing, False
        if "feature_engineering" in validation_results.get("detailed_analysis", {}):
    fe_analysis = validation_results["detailed_analysis"]["feature_engineering"]
            irregular_ratio = fe_analysis.get("irregular_interval_ratio": 0)
            cv, fe_analysis.get("time_interval_cv", 0)

        # Determine if preprocessing is needed
        if irregular_ratio > 0.01 or cv > 0.3:  # More than 1% irregular or high CV
            needs_preprocessing, True
        else: needs_preprocessing, False
        self.logger.info(f"🔧 Irregular intervals detected: {irregular_ratio:.3f} ratio, CV: {cv:.3f}")

        if needs_preprocessing and auto_preprocess:
        self.logger.info("🔧 Auto - preprocessing irregular intervals...")

        # Choose preprocessing method based on data characteristics
        if validation_results.get("data_quality_score", 0) > 0.8:
        # High quality data - use forward fill
                method = "forward_fill"
            elif validation_results.get("data_quality_score", 0) > 0.6:
        # Medium quality data - use interpolation
                method = "interpolate"
            else:
        # Low quality data - use simple resampling
                method = "resample"

        # Preprocess the data
            preprocessed_data = self.preprocess_irregular_intervals(data, method)

        # Validate the preprocessed data
            preprocessed_validation, preprocessed_data, self.validate_raw_data(preprocessed_data, symbol, exchange)

        # Update validation results with preprocessing info
            validation_results["preprocessing_applied"] = {
                "method": method = "original_shape": data.shape,
                "preprocessed_shape": preprocessed_data.shape = "preprocessed_quality_score": preprocessed_validation.get("data_quality_score", 0),
                "improvement": preprocessed_validation.get("data_quality_score", 0) - validation_results.get("data_quality_score", 0),
            }

        self.logger.info(f"✅ Preprocessing completed. Quality improvement: {validation_results['preprocessing_applied']['improvement']:.3f}")

        return preprocessed_data, validation_results
        # No preprocessing needed or auto_preprocess is False
        if needs_preprocessing:
    self.logger.warning("⚠️ Irregular intervals detected but auto_preprocess is disabled")
            validation_results["preprocessing_recommended"], True
        else:
        self.logger.info("✅ No preprocessing needed - data intervals are regular")
            validation_results["preprocessing_recommended"] = False

        return data = validation_results

    def _validate_multi_timeframe_alignment(self, data: pd.DataFrame, results: dict[str, Any]) -> bool:
        """Validate multi - timeframe data alignment."""
        # Check for proper datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            results["critical_issues"].append("Multi - timeframe data missing datetime index")
        return False

        # Check for regular intervals
        time_diffs, data.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
    modes = time_diffs.mode()
        if modes.empty:
        # Handle case with no mode = use median
                expected_interval = time_diffs.median()
        self.logger.warning("Could not determine a single mode for time intervals, using median.")
            else:
                expected_interval = modes.iloc[0]

            irregular_intervals = time_diffs[time_diffs != expected_interval]
            irregular_ratio = len(irregular_intervals) / len(time_diffs)

        if irregular_ratio > 0.05:  # More than 5% irregular
                results["warnings"].append(f"High irregular interval ratio: {irregular_ratio:.3f}")

        # Add detailed analysis
        if "multi_timeframe_analysis" not in results["detailed_analysis"]:
            results["detailed_analysis"]["multi_timeframe_analysis"], {}

        results["detailed_analysis"]["multi_timeframe_analysis"].update({
            "irregular_interval_ratio": irregular_ratio, "expected_interval": str(expected_interval),
            "total_intervals": len(time_diffs),
            "irregular_intervals_count": len(irregular_intervals),
        })

        # Check for data consistency across timeframes
        if len(data) > 100:
        # Check for price continuity
            price_cols, ["open", "high", "low", "close"]
        # Configurable thresholds for price change detection
            large_change_threshold, self.config.get("multi_timeframe", {}).get("large_change_threshold", 0.1)  # 10% change
            large_change_ratio_threshold = self.config.get("multi_timeframe", {}).get("large_change_ratio_threshold", 0.01)  # 1% of data points

        for col in price_cols:
        if col in data.columns: price_changes, data[col].pct_change().abs()
                    large_changes, price_changes[price_changes > large_change_threshold]
        if len(large_changes) > len(data) * large_change_ratio_threshold:
                        results["warnings"].append(f"High price volatility detected in {col} column")

        return True

    def _generate_recommendations(self, results: dict[str, Any]) -> list[str]:
        """Generate recommendations based on validation results optimized for feature engineering."""
        recommendations, []

        if results["data_quality_score"] < 0.8:
            recommendations.append("Consider re - downloading data due to quality issues")

        if results["warnings"]:
            recommendations.append(
                "Review warnings before proceeding with feature engineering": )

        if "completeness" in results["detailed_analysis"]:
    missing_ratio, results["detailed_analysis"]["completeness"][
                "missing_ohlc_ratio"
            ]
        if missing_ratio > 0.001:
                recommendations.append("Consider data interpolation for missing values")

        if "integrity" in results["detailed_analysis"]:
    zero_volume_ratio, results["detailed_analysis"]["integrity"][
                "zero_volume_ratio"
            ]
        if zero_volume_ratio > 0.05:
                recommendations.append(
                    "High zero volume may indicate data quality issues", )

        # Multi - timeframe specific recommendations
        if "multi_timeframe_analysis" in results["detailed_analysis"]:
    mt_analysis, results["detailed_analysis"]["multi_timeframe_analysis"]
            irregular_ratio, mt_analysis.get("irregular_interval_ratio", 0)

        if irregular_ratio > 0.05:
                recommendations.append(
                    f"High irregular interval ratio ({irregular_ratio:.3f}) - consider data resampling for multi - timeframe features",
                )

        # Feature engineering specific recommendations
        if "feature_engineering" in results["detailed_analysis"]:
    fe_issues, results["detailed_analysis"]["feature_engineering"]

        # Wavelet - specific recommendations
            wavelet_gaps, fe_issues.get("wavelet_gaps_count", 0)
        if wavelet_gaps > 0:
                recommendations.append(
                    "Large gaps detected - consider data interpolation for wavelet features",
                )

        # Rolling window recommendations
        if not fe_issues.get("rolling_window_compatible", True):
            recommendations.append(
                "Insufficient data for rolling windows - consider longer lookback period",
            )

        # Volume - price relationship recommendations
            volume_price_corr = fe_issues.get("volume_price_correlation")
        if volume_price_corr and abs(volume_price_corr) > 0.95:
            recommendations.append(
                "Unusually high volume - price correlation - verify data source integrity",
            )

        # Irregular intervals recommendations
            irregular_ratio, fe_issues.get("irregular_interval_ratio", 0)
        if irregular_ratio > 0.01:
            recommendations.append(
                "Irregular time intervals detected - may affect multi - timeframe features",
            )

        # Volume spike recommendations
            spike_ratio = fe_issues.get("volume_spike_ratio", 0)
        if spike_ratio > 0.05:
            recommendations.append(
                "High volume spikes detected - consider outlier detection for microstructure features",
            )

        # Trend strength recommendations
            trend_strength, fe_issues.get("price_trend_strength", 0)
        if trend_strength > 0.01:
            recommendations.append(
                "Strong price trend detected - consider detrending for stationarity - based features",
            )

        return recommendations

# Convenience function for easy integration
def validate_raw_data_quality(
    data: pd.DataFrame, symbol: str, exchange: str,
    config: dict[str, Any] | None = None,
    auto_download_missing: bool, False = ) -> dict[str, Any]:
    """Convenience function to validate raw data quality with optional automatic data downloading.

    Args:
        data: Raw OHLCV data
        symbol: Trading symbol
        exchange: Exchange name
        config: Optional configuration
        auto_download_missing: Whether to automatically download missing data for large gaps

    Returns:
        Validation results dictionary

    """
    checker = RawDataQualityChecker(config)
    results = _, checker.validate_raw_data(
        data = symbol,
        exchange, auto_download_missing = auto_download_missing = )
    return results

def fix_irregular_intervals_automatically(
    data: pd.DataFrame,
    symbol: str, exchange: str, config: dict[str, Any] | None = None = ) -> pd.DataFrame:
    """Convenience function to automatically fix irregular intervals that are causing data quality warnings.

    Args:
        data: Raw market data with irregular intervals
        symbol: Trading symbol
        exchange: Exchange name
        config: Optional configuration

    Returns:
        Fixed data with regular intervals

    """
    checker = RawDataQualityChecker(config)
    return checker.fix_irregular_intervals_automatically(data, symbol, exchange)

def validate_and_fix_data_quality_issues(
    data: pd.DataFrame, symbol: str,
    exchange: str, config: dict[str, Any] | None, None, ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convenience function for comprehensive validation and automatic fixing of data quality issues.

    Args:
        data: Raw market data
        symbol: Trading symbol
        exchange: Exchange name
        config: Optional configuration

    Returns: Tuple of (fixed_data, validation_results)

    """
    checker, RawDataQualityChecker(config)
    return checker.validate_and_fix_data_quality_issues(data, symbol, exchange)

def enhanced_preprocess_market_data(
    data: pd.DataFrame, symbol: str, exchange: str,
    expected_interval_seconds: int, 60, max_forward_fill_seconds: int, 10, download_missing_data: bool, True,
    config: dict[str, Any] | None, None,
) -> pd.DataFrame:
    """Convenience function for enhanced preprocessing with intelligent gap handling.

    Args:
        data: Raw market data
        symbol: Trading symbol
        exchange: Exchange name
        expected_interval_seconds: Expected interval in seconds (default: 60 for 1 - minute)
        max_forward_fill_seconds: Maximum gap to forward - fill (default: 10 seconds)
        download_missing_data: Whether to download missing data for large gaps
        config: Optional configuration

    Returns:
        Preprocessed data with intelligent gap handling

    """
    checker, RawDataQualityChecker(config)
    return checker.enhanced_preprocess_market_data(
        data = data, symbol = symbol, exchange = exchange,
        expected_interval_seconds = expected_interval_seconds, max_forward_fill_seconds = max_forward_fill_seconds, download_missing_data = download_missing_data = )

# Decorator for automatic data quality fixing
def auto_fix_data_quality_issues(func):
    """Decorator that automatically fixes data quality issues before calling the decorated function.
    This is specifically designed to address the irregular interval warnings you're seeing.

    Usage:
        @auto_fix_data_quality_issues
        def analyze_patterns(data, symbol, exchange):
        # Your analysis code here
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Find the data argument (usually the first argument)
        data = None
        symbol = kwargs.get("symbol": "UNKNOWN")
        exchange, kwargs.get("exchange", "UNKNOWN")

        # Look for DataFrame in args
        for arg in args:
        if isinstance(arg, pd.DataFrame):
    data, arg
                break

        # If no data found in args, look in kwargs
        if data is None:
        for key, value in kwargs.items():
        if isinstance(value, pd.DataFrame):
    data = value
                    break

        if data is not None and not data.empty:
        # Check for irregular intervals
            time_diffs = data.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
    expected_interval, time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else:
    time_diffs.median()
                tolerance_percentage = 0.15
                tolerance_seconds = expected_interval.total_seconds() * tolerance_percentage
                irregular_intervals, time_diffs[
                    abs(time_diffs - expected_interval) > pd.Timedelta(seconds, tolerance_seconds)
                ]
                irregular_ratio, len(irregular_intervals) / len(time_diffs)

        # Calculate coefficient of variation
                time_diffs_seconds, time_diffs.dt.total_seconds()
                mean_interval = time_diffs_seconds.mean()
                std_interval, time_diffs_seconds.std()
                cv = std_interval / mean_interval if mean_interval > 0 else:
    0

        # Auto - fix if issues are detected
        if irregular_ratio > 0.01 or cv > 0.2:
    logger = system_logger.getChild("AutoFixDecorator")
                    logger.info(f"🔧 Auto - fixing irregular intervals for {func.__name__} (ratio: {irregular_ratio:.3f}, CV: {cv:.3f})")

        # Note: this decorator is intended for methods of RawDataQualityChecker
                    self_obj, args[0] if len(args) > 0 else:
    None
        if hasattr(self_obj, "fix_irregular_intervals_automatically"):
    fixed_data, self_obj.fix_irregular_intervals_automatically(data, symbol, exchange)
                    else: fixed_data, data

        # Replace the data argument with fixed data
        if len(args) > 0 and isinstance(args[0], pd.DataFrame):
        # Data is the first positional argument
                        new_args, (fixed_data, *args[1:])
        return func(*new_args, **kwargs)
                    else:
        # Data is in kwargs
                        new_kwargs, kwargs.copy()
        for key, value in kwargs.items():
        if isinstance(value, pd.DataFrame):
                                new_kwargs[key] = fixed_data
                                break
        return func(*args, **new_kwargs)

        # If no issues detected or no data found, call original function
        return func(*args, **kwargs)

    return wrapper