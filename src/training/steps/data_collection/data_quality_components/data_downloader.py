"""Data Downloader Component
from src.utils.logger import system_logger
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

Handles all data downloading operations for market data.
Extracted from raw_data_quality_checker.py
"""
import asyncio
import glob
import os
from datetime import datetime
from typing import Any, Optional
import pandas as pd

from src.utils.logger import system_logger
import numpy as np
import json
import logging


class DataDownloader:
    """Handles all data downloading operations for market data.
    
    This class provides functionality for:
    - Downloading missing data for specific timeframes
    - Loading downloaded data from files
    - Managing data download sessions
    - Handling download errors and retries
    """
    @log_important_calls
    
    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.logger = system_logger.getChild("DataDownloader")
        self.config = config or self._get_default_config()
    @log_all_calls
        
    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration for data downloading."""
        return {
            "download": {
                "retry_attempts": 3,
                "retry_delay_seconds": 5,
                "timeout_seconds": 30,
                "max_concurrent_downloads": 5
            },
            "paths": {
                "data_cache": "data_cache",
                "backup_paths": [
                    "data",
                    "backtesting/data_cache"
                ]
            }
        }
        
    def download_missing_data_for_timeframe(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> pd.DataFrame | None:
        """Download missing data for a specific timeframe and time range.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            start_time: Start time for data download
            end_time: End time for data download
            
        Returns:
            Downloaded data or None if failed
        """
        self.logger.info(f'🔧 Downloading {timeframe} data for {symbol} on {exchange}')
        self.logger.info(f'   Time range: {start_time} to {end_time}')
        
        try:
            from ..data_downloader import download_all_data_with_consolidation
            
            success = asyncio.run(download_all_data_with_consolidation(
                symbol = symbol, 
                exchange_name = exchange, 
                interval = timeframe
            ))
            
            if success:
                downloaded_data = self._load_downloaded_data(symbol, exchange, timeframe)
                if downloaded_data is not None and not downloaded_data.empty:
                    # Filter for the specific time range
                    filtered_data = downloaded_data[
                        (downloaded_data.index >= start_time) & 
                        (downloaded_data.index <= end_time)
                    ]
                    
                    if not filtered_data.empty:
                        self.logger.info(f'✅ Successfully downloaded {len(filtered_data)} records for specified time range')
                        return filtered_data
                    else:
                        self.logger.warning('⚠️ No data found in downloaded data for specified time range')
                        return None
                else:
                    self.logger.warning('⚠️ No data found after download')
                    return None
            else:
                self.logger.warning('⚠️ Download returned unsuccessful status')
                return None
                
        except Exception as e:
            self.logger.exception(f'❌ Error downloading {timeframe} data: {e}')
            return None
            
    def download_data_for_timeframe(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        start_time: Optional[datetime] = None, 
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame | None:
        """Download data for a specific timeframe and optionally filter by time range.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Downloaded data or None if failed
        """
        self.logger.info(f'🔧 Downloading {timeframe} data for {symbol} on {exchange}')
        if start_time and end_time:
            self.logger.info(f'   Time range: {start_time} to {end_time}')
            
        try:
            from ..data_downloader import download_all_data_with_consolidation
            
            success = asyncio.run(download_all_data_with_consolidation(
                symbol = symbol, 
                exchange_name = exchange, 
                interval = timeframe
            ))
            
            if success:
                downloaded_data = self._load_downloaded_data(symbol, exchange, timeframe)
                if downloaded_data is not None and not downloaded_data.empty:
                    if start_time and end_time:
                        filtered_data = downloaded_data[
                            (downloaded_data.index >= start_time) & 
                            (downloaded_data.index <= end_time)
                        ]
                        if not filtered_data.empty:
                            self.logger.info(f'✅ Successfully downloaded {len(filtered_data)} records for specified time range')
                            return filtered_data
                        else:
                            self.logger.warning('⚠️ No data found in downloaded data for specified time range')
                            return None
                    else:
                        self.logger.info(f'✅ Successfully downloaded {len(downloaded_data)} records')
                        return downloaded_data
                else:
                    self.logger.warning('⚠️ No data found after download')
                    return None
            else:
                self.logger.warning('⚠️ Download returned unsuccessful status')
                return None
                
        except Exception as e:
            self.logger.exception(f'❌ Error downloading {timeframe} data: {e}')
            return None
    @log_all_calls
            
    def _load_downloaded_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str
    ) -> pd.DataFrame | None:
        """Load the most recent downloaded data for a symbol/timeframe combination.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            
        Returns:
            Loaded data or None if not found
        """
        try:
            patterns = [
                f'data_cache/klines_{exchange}_{symbol}_{timeframe}_*.csv',
                f'data/{symbol}_{timeframe}.csv',
                f'backtesting/data_cache/klines_{exchange}_{symbol}_{timeframe}_*.csv',
                f'data_cache/{symbol}_{timeframe}.csv'
            ]
            
            for pattern in patterns:
                files = glob.glob(pattern)
                if files:
                    latest_file = max(files, key=os.path.getmtime)
                    self.logger.info(f'🔍 Loading data from: {latest_file}')
                    
                    if latest_file.endswith('.csv'):
                        data = pd.read_csv(latest_file, index_col = 0, parse_dates = True)
                    elif latest_file.endswith('.parquet'):
                        data = pd.read_parquet(latest_file)
                    else:
                        continue
                        
                    if not data.empty:
                        self.logger.info(f'✅ Loaded {len(data)} records from {latest_file}')
                        return data
                        
            self.logger.warning(f'⚠️ No data files found for {symbol} {timeframe} on {exchange}')
            return None
            
        except Exception as e:
            self.logger.exception(f'❌ Error loading downloaded data: {e}')
            return None
            
    def handle_missing_data_download(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        exchange: str, 
        results: dict[str, Any]
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Handle automatic downloading of missing data for large gaps.
        
        Args:
            data: Current data
            symbol: Trading symbol
            exchange: Exchange name
            results: Validation results
            
        Returns:
            Tuple of (updated_data, download_summary)
        """
        download_summary = {
            'data_downloaded': False,
            'gaps_found': 0,
            'gaps_filled': 0,
            'download_errors': 0,
            'timeframe_detected': None
        }
        
        try:
            time_diffs = data.index.to_series().diff().dropna()
            if len(time_diffs) == 0:
                return data, download_summary
                
            expected_interval = time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
            max_gap_threshold = expected_interval * 3
            large_gaps = time_diffs[time_diffs > max_gap_threshold]
            download_summary['gaps_found'] = len(large_gaps)
            
            if len(large_gaps) == 0:
                self.logger.info('✅ No large gaps found - data is continuous')
                return data, download_summary
                
            self.logger.info(f'🔍 Found {len(large_gaps)} large gaps in data')
            
            timeframe = self._determine_timeframe_from_data(data)
            download_summary['timeframe_detected'] = timeframe
            self.logger.info(f'🔧 Detected timeframe: {timeframe}')
            
            updated_data = data.copy()
            
            for i, (gap_start, gap_duration) in enumerate(large_gaps.items()):
                gap_end = gap_start + gap_duration
                self.logger.info(f'🔧 Processing gap {i + 1}/{len(large_gaps)}: {gap_start} to {gap_end}')
                
                try:
                    gap_data = self.download_missing_data_for_timeframe(
                        symbol = symbol,
                        exchange = exchange,
                        timeframe = timeframe,
                        start_time = gap_start,
                        end_time = gap_end
                    )
                    
                    if gap_data is not None and not gap_data.empty:
                        updated_data = self._fill_gap_in_dataset(updated_data, gap_data, gap_start, gap_end)
                        download_summary['gaps_filled'] += 1
                        self.logger.info(f'✅ Gap {i + 1} filled with {len(gap_data)} records')
                    else:
                        self.logger.warning(f'⚠️ No data downloaded for gap {i + 1}')
                        download_summary['download_errors'] += 1
                        
                except Exception as e:
                    self.logger.exception(f'❌ Error downloading data for gap {i + 1}: {e}')
                    download_summary['download_errors'] += 1
                    
            if download_summary['gaps_filled'] > 0:
                download_summary['data_downloaded'] = True
                results['warnings'].append(
                    f"Downloaded missing data for {download_summary['gaps_filled']}/{download_summary['gaps_found']} gaps"
                )
                
            self.logger.info('🔍 Re-validating data after download...')
            # Note: This would need to be called from the main validator
            # updated_results, updated_data = self.validate_raw_data(updated_data, symbol, exchange, auto_download_missing = False)
            # results['data_quality_score'] = updated_results['data_quality_score']
            # results['data_shape'] = updated_data.shape
            # self.logger.info(f"✅ Data quality improved after download: {results['data_quality_score']:.2f}")
            
            return updated_data, download_summary
            
        except Exception as e:
            self.logger.exception(f'❌ Error in missing data download process: {e}')
            download_summary['download_errors'] += 1
            return data, download_summary
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