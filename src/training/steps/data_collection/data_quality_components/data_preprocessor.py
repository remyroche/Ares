"""Data Preprocessor Component

Handles all data preprocessing operations for market data.
Extracted from raw_data_quality_checker.py
"""

import asyncio
import glob
import os
from datetime import datetime, timedelta
from typing import Any, Optional, Tuple, List
import pandas as pd
import logging
import numpy as np

from src.utils.logger import system_logger
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

class DataPreprocessor:
    """Handles all data preprocessing operations for market data.

    This class provides functionality for:
    - Fixing irregular intervals
    - Enhanced preprocessing with intelligent gap handling
    - Data resampling and interpolation
    - Gap filling and data continuity
    """
    @log_important_calls
    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.logger = system_logger.getChild("DataPreprocessor")
        self.config = config or self._get_default_config()
    @log_all_calls
    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for preprocessing."""
        return {
            "preprocessing": {
                "max_forward_fill_seconds": 10,
                "auto_fix_irregular_intervals": True,
                "download_missing_data": True,
                "preserve_original_data": True
            },
            "tolerance": {
                "interval_tolerance_percentage": 0.15,
                "irregular_interval_threshold": 0.01
            }
        }

    def fix_irregular_intervals_automatically(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str
    ) -> pd.DataFrame:
        """Automatically fix irregular intervals that are causing data quality warnings.

        Args:
            data: Raw market data with irregular intervals
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Fixed data with regular intervals
        """
        self.logger.info('🔧 Starting automatic irregular interval fixing...')
        self.logger.info(f'📊 Processing {exchange} {symbol} data')
        self.logger.info(f'📈 Input data shape: {data.shape}')

        if data.empty:
            self.logger.warning('⚠️ Empty data provided, returning as-is')
            return data

        time_diffs = data.index.to_series().diff().dropna()
        if len(time_diffs) == 0:
            self.logger.info('✅ No time differences found - data is already regular')
            return data

        self.logger.info(f'📊 Analyzing {len(time_diffs)} time intervals...')

        expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
        expected_interval_seconds = expected_interval.total_seconds()
        tolerance_percentage = self.config["tolerance"]["interval_tolerance_percentage"]
        tolerance_seconds = expected_interval_seconds * tolerance_percentage

        irregular_intervals = time_diffs[abs(time_diffs - expected_interval) > pd.Timedelta(seconds = tolerance_seconds)]
        irregular_ratio = len(irregular_intervals) / len(time_diffs)

        self.logger.info('🔍 Detailed interval analysis:')
        self.logger.info(f'   📅 Expected interval: {expected_interval} ({expected_interval_seconds:.1f}s)')
        self.logger.info(f'   📊 Total intervals analyzed: {len(time_diffs):,}')
        self.logger.info(f'   ⚠️ Irregular intervals found: {len(irregular_intervals):,} ({irregular_ratio:.3f})')
        self.logger.info(f'   🎯 Tolerance threshold: ±{tolerance_seconds:.1f}s ({tolerance_percentage:.1%})')
        self.logger.info(f'   📋 Irregular interval threshold: {self.config["tolerance"]["irregular_interval_threshold"]:.3f}')

        # Show some examples of irregular intervals
        if len(irregular_intervals) > 0:
            self.logger.info('📋 Sample irregular intervals:')
            for i, (timestamp, interval) in enumerate(irregular_intervals.head(5).items()):
                self.logger.info(f'   {i+1}. {timestamp}: {interval} (expected: {expected_interval})')
            if len(irregular_intervals) > 5:
                self.logger.info(f'   ... and {len(irregular_intervals) - 5} more irregular intervals')

        if irregular_ratio > self.config["tolerance"]["irregular_interval_threshold"]:
            self.logger.warning(f'⚠️ Irregular interval ratio {irregular_ratio:.3f} exceeds threshold {self.config["tolerance"]["irregular_interval_threshold"]:.3f}')
            self.logger.info('🔧 Applying enhanced preprocessing to fix irregular intervals...')

            fixed_data = self.enhanced_preprocess_market_data(
                data = data,
                symbol = symbol,
                exchange = exchange,
                expected_interval_seconds = int(expected_interval_seconds),
                max_forward_fill_seconds = self.config['preprocessing']['max_forward_fill_seconds'],
                download_missing_data = self.config['preprocessing']['download_missing_data']
            )

            # Verify the fix
            self.logger.info('🔍 Verifying interval fix...')
            fixed_time_diffs = fixed_data.index.to_series().diff().dropna()
            if len(fixed_time_diffs) > 0:
                fixed_expected_interval = fixed_time_diffs.mode().iloc[0] if len(fixed_time_diffs.mode()) > 0 else fixed_time_diffs.median()
                fixed_irregular_intervals = fixed_time_diffs[abs(fixed_time_diffs - fixed_expected_interval) > pd.Timedelta(seconds = tolerance_seconds)]
                fixed_irregular_ratio = len(fixed_irregular_intervals) / len(fixed_time_diffs)

                self.logger.info('✅ Fix verification results:')
                self.logger.info(f'   📊 Original data shape: {data.shape}')
                self.logger.info(f'   📊 Fixed data shape: {fixed_data.shape}')
                self.logger.info(f'   📈 Before: {irregular_ratio:.3f} irregular intervals ({len(irregular_intervals):,} out of {len(time_diffs):,})')
                self.logger.info(f'   📈 After: {fixed_irregular_ratio:.3f} irregular intervals ({len(fixed_irregular_intervals):,} out of {len(fixed_time_diffs):,})')
                self.logger.info(f'   📈 Improvement: {irregular_ratio - fixed_irregular_ratio:.3f} ({((irregular_ratio - fixed_irregular_ratio) / irregular_ratio * 100):.1f}% reduction)')

                if fixed_irregular_ratio < 0.001:
                    self.logger.info('✅ Irregular intervals successfully fixed! Data is now highly regular')
                elif fixed_irregular_ratio < irregular_ratio * 0.5:
                    self.logger.info('✅ Significant improvement achieved in interval regularity')
                else:
                    self.logger.warning(f'⚠️ Some irregular intervals remain: {fixed_irregular_ratio:.3f}')

                return fixed_data
            else:
                self.logger.warning('⚠️ No time differences found in fixed data - verification failed')
                return fixed_data
        else:
            self.logger.info('✅ No significant irregular intervals detected - data is within acceptable tolerance')
            self.logger.info(f'   📊 Irregular ratio: {irregular_ratio:.3f} ≤ threshold: {self.config["tolerance"]["irregular_interval_threshold"]:.3f}')

        return data

    def enhanced_preprocess_market_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str,
        expected_interval_seconds: int = 60,
        max_forward_fill_seconds: int = 10,
        download_missing_data: bool = True
    ) -> pd.DataFrame:
        """Enhanced preprocessing with intelligent gap handling.

        Strategy:
        1. Resample to expected intervals
        2. Re-add original data to preserve accuracy
        3. Forward-fill if missing values are less than max_forward_fill_seconds
        4. Download missing data for gaps > max_forward_fill_seconds

        Args:
            data: Raw market data
            symbol: Trading symbol
            exchange: Exchange name
            expected_interval_seconds: Expected interval in seconds (default: 60 for 1-minute)
            max_forward_fill_seconds: Maximum gap to forward-fill (default: 10 seconds)
            download_missing_data: Whether to download missing data for large gaps

        Returns:
            Preprocessed data with intelligent gap handling
        """
        self.logger.info('🔧 Starting enhanced preprocessing...')
        self.logger.info(f'📊 Processing {exchange} {symbol} data')
        self.logger.info(f'📈 Input data shape: {data.shape}')
        self.logger.info(f'⚙️ Configuration:')
        self.logger.info(f'   Expected interval: {expected_interval_seconds}s')
        self.logger.info(f'   Max forward-fill: {max_forward_fill_seconds}s')
        self.logger.info(f'   Download missing: {download_missing_data}')

        if data.empty:
            self.logger.warning('⚠️ Empty data provided, returning as-is')
            return data

        # Analyze original data
        original_start = data.index.min()
        original_end = data.index.max()
        original_duration = (original_end - original_start).total_seconds()
        original_points = len(data)

        self.logger.info('📊 Original data analysis:')
        self.logger.info(f'   📅 Time range: {original_start} to {original_end}')
        self.logger.info(f'   ⏱️ Duration: {original_duration:.1f}s ({original_duration/3600:.1f}h)')
        self.logger.info(f'   📈 Data points: {original_points:,}')
        self.logger.info(f'   📊 Columns: {list(data.columns)}')

        # Remove duplicates
        if data.index.duplicated().any():
            duplicates = data.index.duplicated().sum()
            self.logger.warning(f'⚠️ Found {duplicates} duplicate timestamps, removing duplicates')
            data = data[~data.index.duplicated(keep='last')]
            self.logger.info(f'✅ Removed {duplicates} duplicates, new shape: {data.shape}')
        else:
            self.logger.info('✅ No duplicate timestamps found')

        # Step 1: Resample to expected intervals
        freq = f'{expected_interval_seconds}S'
        self.logger.info(f'🔧 Step 1: Resampling to {freq} intervals')
        self.logger.info(f'   📊 Original data points: {len(data):,}')

        resampled = data.resample(freq).last()
        self.logger.info(f'   📈 Resampled data points: {len(resampled):,}')
        self.logger.info(f'   📊 Resampling ratio: {len(resampled)/len(data):.3f}')

        # Step 2: Re-add original data to preserve accuracy
        self.logger.info('🔧 Step 2: Re-adding original data to preserve accuracy')
        combined_data = resampled.copy()
        # Vectorized replacement: floor original timestamps to bucket, deduplicate, update
        orig = data.copy()
        orig.index = orig.index.floor(freq)
        orig = orig[~orig.index.duplicated(keep='last')]
        combined_data.update(orig)

        self.logger.info(f'   📈 Combined data points: {len(combined_data):,}')
        self.logger.info(f'   📊 Data points added from original: {len(orig):,}')

        # Step 3: Analyze gaps and apply intelligent handling
        self.logger.info('🔧 Step 3: Analyzing gaps and applying intelligent handling')
        time_diffs = combined_data.index.to_series().diff().dropna()
        gaps = time_diffs[time_diffs > pd.Timedelta(seconds = expected_interval_seconds)]

        self.logger.info(f'📊 Gap analysis:')
        self.logger.info(f'   📈 Total time differences analyzed: {len(time_diffs):,}')
        self.logger.info(f'   ⚠️ Gaps found: {len(gaps):,}')

        if len(gaps) > 0:
            self.logger.info(f'🔍 Detailed gap analysis:')
            small_gaps = gaps[gaps <= pd.Timedelta(seconds = max_forward_fill_seconds)]
            large_gaps = gaps[gaps > pd.Timedelta(seconds = max_forward_fill_seconds)]

            self.logger.info(f'   📊 Small gaps (≤{max_forward_fill_seconds}s): {len(small_gaps):,}')
            self.logger.info(f'   📊 Large gaps (>{max_forward_fill_seconds}s): {len(large_gaps):,}')

            if len(small_gaps) > 0:
                avg_small_gap = small_gaps.mean().total_seconds()
                max_small_gap = small_gaps.max().total_seconds()
                self.logger.info(f'   📈 Small gap stats: avg={avg_small_gap:.1f}s, max={max_small_gap:.1f}s')

            if len(large_gaps) > 0:
                avg_large_gap = large_gaps.mean().total_seconds()
                max_large_gap = large_gaps.max().total_seconds()
                self.logger.info(f'   📈 Large gap stats: avg={avg_large_gap:.1f}s, max={max_large_gap:.1f}s')

                # Show some examples of large gaps
                self.logger.info('📋 Sample large gaps:')
                for i, (timestamp, gap) in enumerate(large_gaps.head(3).items()):
                    self.logger.info(f'   {i+1}. {timestamp}: {gap.total_seconds():.1f}s')
                if len(large_gaps) > 3:
                    self.logger.info(f'   ... and {len(large_gaps) - 3} more large gaps')

            # Handle small gaps with forward fill
            if len(small_gaps) > 0:
                self.logger.info('🔧 Step 4a: Forward-filling small gaps')
                nulls_before = combined_data.isnull().sum().sum()
                combined_data = combined_data.fillna(method='ffill')
                nulls_after = combined_data.isnull().sum().sum()
                self.logger.info(f'   📊 Nulls before: {nulls_before:,}, after: {nulls_after:,}')
                self.logger.info('✅ Small gaps forward filled')
            else:
                self.logger.info('✅ No small gaps to forward fill')

            # Handle large gaps with data download
            if len(large_gaps) > 0 and download_missing_data:
                self.logger.info('🔧 Step 4b: Downloading missing data for large gaps')
                nulls_before = combined_data.isnull().sum().sum()
                combined_data = self._download_and_fill_missing_data(
                    combined_data, symbol, exchange, large_gaps
                )
                nulls_after = combined_data.isnull().sum().sum()
                self.logger.info(f'   📊 Nulls before: {nulls_before:,}, after: {nulls_after:,}')
                self.logger.info('✅ Large gaps processed with data download')
            elif len(large_gaps) > 0:
                self.logger.warning(f'⚠️ {len(large_gaps)} large gaps remain unfilled (download disabled)')
        else:
            self.logger.info('✅ No gaps found - data is already regular')

        # Final cleanup
        remaining_nulls = combined_data.isnull().sum().sum()
        if remaining_nulls > 0:
            self.logger.info(f'🔧 Step 5: Final forward-fill for {remaining_nulls:,} remaining nulls')
            combined_data = combined_data.fillna(method='ffill')
            final_nulls = combined_data.isnull().sum().sum()
            self.logger.info(f'   📊 Final nulls after cleanup: {final_nulls:,}')
        else:
            self.logger.info('✅ No remaining nulls to clean up')

        # Final verification
        self.logger.info('🔍 Final verification...')
        final_gaps = combined_data.index.to_series().diff().dropna()
        final_large_gaps = final_gaps[final_gaps > pd.Timedelta(seconds = expected_interval_seconds)]
        final_completeness = combined_data.notna().sum().sum() / combined_data.size

        self.logger.info('✅ Enhanced preprocessing completed:')
        self.logger.info(f'   📊 Original shape: {data.shape}')
        self.logger.info(f'   📊 Final shape: {combined_data.shape}')
        self.logger.info(f'   📈 Data points added: {len(combined_data) - original_points:,}')
        self.logger.info(f'   ⚠️ Remaining large gaps: {len(final_large_gaps):,}')
        self.logger.info(f'   📊 Data completeness: {final_completeness:.3f}')
        self.logger.info(f'   📈 Completeness improvement: {((final_completeness - (1 - len(gaps)/len(time_diffs))) * 100):.1f}%')

        return combined_data

    def preprocess_irregular_intervals(
        self,
        data: pd.DataFrame,
        method: str = 'forward_fill'
    ) -> pd.DataFrame:
        """Preprocess data to handle irregular intervals.

        Args:
            data: Raw OHLCV data with irregular intervals
            method: Preprocessing method ('forward_fill', 'interpolate', 'resample')

        Returns:
            Preprocessed data with regular intervals
        """
        self.logger.info(f'🔧 Preprocessing irregular intervals using method: {method}')

        # Remove duplicates
        if data.index.duplicated().any():
            self.logger.warning(f'⚠️ Found {data.index.duplicated().sum()} duplicate timestamps, removing duplicates')
            data = data[~data.index.duplicated(keep='last')]

        # Determine frequency
        time_diffs = data.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
            expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
            seconds = int(expected_interval.total_seconds())
            if seconds <= 60:
                freq = '1T'
            elif seconds <= 300:
                freq = '5T'
            elif seconds <= 900:
                freq = '15T'
            elif seconds <= 1800:
                freq = '30T'
            elif seconds <= 3600:
                freq = '1H'
            elif seconds <= 14400:
                freq = '4H'
            elif seconds <= 86400:
                freq = '1D'
            else:
                freq = '1T'
        else:
            freq = '1T'

        # Apply preprocessing method
        if method == 'forward_fill':
            data = data.resample(freq).ffill()
        elif method == 'interpolate':
            # CRITICAL GAPS: Never interpolate data that contains critical gaps
            # Check for large time gaps that indicate critical data issues
            if hasattr(data.index, 'to_series') and len(data.index) > 1:
                time_gaps = data.index.to_series().diff().dropna()
                max_gap_seconds = time_gaps.max().total_seconds() if len(time_gaps) > 0 else 0

                # If there are gaps > 30 minutes (1800 seconds), this is likely critical data
                if max_gap_seconds > 1800:
                    self.logger.error('🚨 CRITICAL DATA GAPS DETECTED - REFUSING TO INTERPOLATE')
                    self.logger.error(f'🚨 Maximum gap: {max_gap_seconds} seconds - CRITICAL GAPS MUST NOT BE INTERPOLATED')
                    self.logger.warning('⚠️ Switching to forward_fill to preserve data integrity')
                    data = data.resample(freq).ffill()
                else:
                    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
                    data[numeric_cols] = data[numeric_cols].interpolate(method='time').ffill()
            else:
                numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
                data[numeric_cols] = data[numeric_cols].interpolate(method='time').ffill()
        elif method == 'resample':
            data = data.resample(freq).mean().ffill()
        else:
            self.logger.warning(f'⚠️ Unknown preprocessing method: {method}, defaulting to forward_fill')
            data = data.resample(freq).ffill()

        return data
    @log_all_calls

    def _download_and_fill_missing_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        exchange: str,
        gaps: pd.Series
    ) -> pd.DataFrame:
        """Download missing data for large gaps using existing data download functions.

        Args:
            data: Current data with gaps
            symbol: Trading symbol
            exchange: Exchange name
            gaps: Series of time differences representing gaps

        Returns:
            Data with downloaded missing data filled in
        """
        self.logger.info(f'🔧 Downloading missing data for {len(gaps)} large gaps')

        try:
            from ..data_downloader import download_all_data_with_consolidation

            timeframe = self._determine_timeframe_from_data(data)
            self.logger.info(f'🔍 Detected timeframe: {timeframe}')

            for i, (gap_start, gap_duration) in enumerate(gaps.items()):
                gap_end = gap_start + gap_duration
                self.logger.info(f'🔧 Downloading gap {i + 1}/{len(gaps)}: {gap_start} to {gap_end}')

                try:
                    # Run async downloader from sync context
                    success = asyncio.run(download_all_data_with_consolidation(
                        symbol = symbol,
                        exchange_name = exchange,
                        interval = timeframe
                    ))

                    if not success:
                        self.logger.warning('⚠️ Download returned unsuccessful status')

                except Exception as e:
                    self.logger.exception(f'❌ Error during gap download: {e}')

        except ImportError:
            self.logger.warning('⚠️ Data downloader not available, skipping data download')
            return data
        except Exception as e:
            self.logger.exception(f'❌ Error in data download process: {e}')
            return data

        return data
    @log_all_calls

    def _determine_timeframe_from_data(self, data: pd.DataFrame) -> str:
        """Determine the timeframe from the data intervals.

        Args:
            data: Market data with datetime index

        Returns:
            Timeframe string (e.g., '1m', '5m', '15m', '1h')
        """
        if len(data) < 2:
            return '1m'

        time_diffs = data.index.to_series().diff().dropna()
        if len(time_diffs) == 0:
            return '1m'

        most_common_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
        interval_seconds = most_common_interval.total_seconds()

        if interval_seconds <= 60:
            return '1m'
        elif interval_seconds <= 300:
            return '5m'
        elif interval_seconds <= 900:
            return '15m'
        elif interval_seconds <= 1800:
            return '30m'
        elif interval_seconds <= 3600:
            return '1h'
        elif interval_seconds <= 14400:
            return '4h'
        elif interval_seconds <= 86400:
            return '1d'
        else:
            return '1d'
    @log_all_calls

    def _load_and_filter_downloaded_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame | None:
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
            possible_paths = [
                f'data_cache/klines_{exchange}_{symbol}_{timeframe}_*.csv',
                f'data/{symbol}_{timeframe}.csv',
                f'backtesting/data_cache/klines_{exchange}_{symbol}_{timeframe}_*.csv',
                f'data_cache/{symbol}_{timeframe}.csv'
            ]

            for pattern in possible_paths:
                files = glob.glob(pattern)
                if files:
                    files.sort(key = os.path.getmtime, reverse = True)
                    for file_path in files:
                        try:
                            self.logger.info(f'🔍 Loading data from: {file_path}')

                            if file_path.endswith('.csv'):
                                data = pd.read_csv(file_path, index_col = 0, parse_dates = True)
                            elif file_path.endswith('.parquet'):
                                data = standardized_parquet_handler.read_parquet_standardized(file_path)
                            else:
                                continue

                            if data.empty:
                                continue

                            gap_data = data[(data.index >= start_time) & (data.index <= end_time)]
                            if not gap_data.empty:
                                return gap_data

                        except Exception as e:
                            self.logger.warning(f'⚠️ Failed loading {file_path}: {e}')

            return None

        except Exception as e:
            self.logger.exception(f'❌ Error searching for downloaded data: {e}')
            return None
    @log_all_calls

    def _fill_gap_in_dataset(
        self,
        main_data: pd.DataFrame,
        gap_data: pd.DataFrame,
        gap_start: datetime,
        gap_end: datetime
    ) -> pd.DataFrame:
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
            filled_data = main_data.copy()
            gap_mask = (filled_data.index >= gap_start) & (filled_data.index <= gap_end)
            filled_data = filled_data[~gap_mask]
            filled_data = pd.concat([filled_data, gap_data])
            filled_data = filled_data.sort_index()
            filled_data = filled_data[~filled_data.index.duplicated(keep='last')]

            self.logger.info(f'✅ Gap filled: {len(main_data)} -> {len(filled_data)} records')
            return filled_data

        except Exception as e:
            self.logger.exception(f'❌ Error filling gap in dataset: {e}')
            return main_data
            return main_data
