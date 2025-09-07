"""Data Preprocessor Component
from src.utils.logger import system_logger
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

Handles all data preprocessing operations for market data.
Extracted from raw_data_quality_checker.py
"""
import asyncio
import glob
import os
from datetime import datetime, timedelta
from typing import Any, Optional, Tuple
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
import logging
import time


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
        self.logger.info(f'🔧 Auto-fixing irregular intervals for {exchange} {symbol}')
        
        time_diffs = data.index.to_series().diff().dropna()
        if len(time_diffs) == 0:
            self.logger.info('✅ No time differences found - data is already regular')
            return data
            
        expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
        expected_interval_seconds = expected_interval.total_seconds()
        tolerance_percentage = self.config["tolerance"]["interval_tolerance_percentage"]
        tolerance_seconds = expected_interval_seconds * tolerance_percentage
        
        irregular_intervals = time_diffs[abs(time_diffs - expected_interval) > pd.Timedelta(seconds = tolerance_seconds)]
        irregular_ratio = len(irregular_intervals) / len(time_diffs)
        
        self.logger.info('🔍 Interval analysis:')
        self.logger.info(f'   Expected interval: {expected_interval}')
        self.logger.info(f'   Irregular intervals: {len(irregular_intervals)} ({irregular_ratio:.3f})')
        self.logger.info(f'   Tolerance: ±{tolerance_seconds:.1f}s')
        
        if irregular_ratio > self.config["tolerance"]["irregular_interval_threshold"]:
            self.logger.info('🔧 Applying enhanced preprocessing to fix irregular intervals')
            fixed_data = self.enhanced_preprocess_market_data(
                data = data, 
                symbol = symbol, 
                exchange = exchange, 
                expected_interval_seconds = int(expected_interval_seconds),
                max_forward_fill_seconds = self.config['preprocessing']['max_forward_fill_seconds'],
                download_missing_data = self.config['preprocessing']['download_missing_data']
            )
            
            # Verify the fix
            fixed_time_diffs = fixed_data.index.to_series().diff().dropna()
            if len(fixed_time_diffs) > 0:
                fixed_expected_interval = fixed_time_diffs.mode().iloc[0] if len(fixed_time_diffs.mode()) > 0 else fixed_time_diffs.median()
                fixed_irregular_intervals = fixed_time_diffs[abs(fixed_time_diffs - fixed_expected_interval) > pd.Timedelta(seconds = tolerance_seconds)]
                fixed_irregular_ratio = len(fixed_irregular_intervals) / len(fixed_time_diffs)
                
                self.logger.info('✅ Fix verification:')
                self.logger.info(f'   Before: {irregular_ratio:.3f} irregular intervals')
                self.logger.info(f'   After: {fixed_irregular_ratio:.3f} irregular intervals')
                self.logger.info(f'   Improvement: {irregular_ratio - fixed_irregular_ratio:.3f}')
                
                if fixed_irregular_ratio < 0.001:
                    self.logger.info('✅ Irregular intervals successfully fixed!')
                else:
                    self.logger.warning(f'⚠️ Some irregular intervals remain: {fixed_irregular_ratio:.3f}')
                    
                return fixed_data
        else:
            self.logger.info('✅ No significant irregular intervals detected')
            
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
        self.logger.info(f'🔧 Enhanced preprocessing for {exchange} {symbol}')
        self.logger.info(f'   Expected interval: {expected_interval_seconds}s')
        self.logger.info(f'   Max forward-fill: {max_forward_fill_seconds}s')
        self.logger.info(f'   Download missing: {download_missing_data}')
        
        # Remove duplicates
        if data.index.duplicated().any():
            duplicates = data.index.duplicated().sum()
            self.logger.warning(f'⚠️ Found {duplicates} duplicate timestamps, removing duplicates')
            data = data[~data.index.duplicated(keep='last')]
            
        # Step 1: Resample to expected intervals
        freq = f'{expected_interval_seconds}S'
        self.logger.info(f'🔧 Step 1: Resampling to {freq} intervals')
        resampled = data.resample(freq).last()
        
        # Step 2: Re-add original data to preserve accuracy
        self.logger.info('🔧 Step 2: Re-adding original data to preserve accuracy')
        combined_data = resampled.copy()
        # Vectorized replacement: floor original timestamps to bucket, deduplicate, update
        orig = data.copy()
        orig.index = orig.index.floor(freq)
        orig = orig[~orig.index.duplicated(keep='last')]
        combined_data.update(orig)
                
        # Step 3: Analyze gaps and apply intelligent handling
        self.logger.info('🔧 Step 3: Analyzing gaps and applying intelligent handling')
        time_diffs = combined_data.index.to_series().diff().dropna()
        gaps = time_diffs[time_diffs > pd.Timedelta(seconds = expected_interval_seconds)]
        
        if len(gaps) > 0:
            self.logger.info(f'🔍 Found {len(gaps)} gaps in the data')
            small_gaps = gaps[gaps <= pd.Timedelta(seconds = max_forward_fill_seconds)]
            large_gaps = gaps[gaps > pd.Timedelta(seconds = max_forward_fill_seconds)]
            
            self.logger.info(f'   Small gaps (≤{max_forward_fill_seconds}s): {len(small_gaps)}')
            self.logger.info(f'   Large gaps (>{max_forward_fill_seconds}s): {len(large_gaps)}')
            
            # Handle small gaps with forward fill
            if len(small_gaps) > 0:
                self.logger.info('🔧 Step 4a: Forward-filling small gaps')
                combined_data = combined_data.fillna(method='ffill')
                
            # Handle large gaps with data download
            if len(large_gaps) > 0 and download_missing_data:
                self.logger.info('🔧 Step 4b: Downloading missing data for large gaps')
                combined_data = self._download_and_fill_missing_data(
                    combined_data, symbol, exchange, large_gaps
                )
            elif len(large_gaps) > 0:
                self.logger.warning(f'⚠️ {len(large_gaps)} large gaps remain unfilled (download disabled)')
                
        # Final cleanup
        remaining_nulls = combined_data.isnull().sum().sum()
        if remaining_nulls > 0:
            self.logger.info(f'🔧 Step 5: Final forward-fill for {remaining_nulls} remaining nulls')
            combined_data = combined_data.fillna(method='ffill')
            
        # Final verification
        final_gaps = combined_data.index.to_series().diff().dropna()
        final_large_gaps = final_gaps[final_gaps > pd.Timedelta(seconds = expected_interval_seconds)]
        
        self.logger.info('✅ Enhanced preprocessing completed:')
        self.logger.info(f'   Original shape: {data.shape}')
        self.logger.info(f'   Final shape: {combined_data.shape}')
        self.logger.info(f'   Remaining large gaps: {len(final_large_gaps)}')
        self.logger.info(f'   Data completeness: {combined_data.notna().sum().sum() / combined_data.size:.3f}')
        
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
                                data = pd.read_parquet(file_path)
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