#!/usr/bin/env python3
"""
Enhanced API-Agnostic Data Collector

This module provides API-agnostic data collection using the exchanges/ directory
with comprehensive validation, gap detection, and incremental downloading.

Features:
- API-agnostic data collection using exchange/ directory
- Comprehensive data gap detection
- Incremental downloading (batches start where previous batch ended)
- Batch downloading without erasing previous batches
- Download data for specific periods
- Extensive logging and error handling
- Integration with enhanced validation framework
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.pipeline_standards import pipeline_standards
from src.utils.common_operations import handles_errors, traced, cached, log_execution_time
from src.utils.enhanced_memory_management import memory_efficient, resource_monitor
from src.utils.enhanced_mlflow_integration import with_enhanced_mlflow_logging, log_step_metrics
from .enhanced_validation_framework_with_decorators import (
    DataType, EnhancedDataValidator, get_validator, ValidationSeverity
)
from .exchange_field_mappings import get_exchange_mapper, ExchangeType

logger = system_logger.getChild("EnhancedAPIAgnosticDataCollector")


class DataGapDetector:
    """Comprehensive data gap detection and analysis."""
    
    def __init__(self, exchange: str, symbol: str, timeframe: str):
        self.exchange = exchange.upper()
        self.symbol = symbol
        self.timeframe = timeframe
        self.logger = logger.getChild(f"GapDetector.{exchange}.{symbol}")
        
        # Gap detection configuration
        self.gap_thresholds = {
            'klines': 66.0,      # 1.1 minutes
            'aggtrades': 1.0,    # 1 second
            'futures': 32400.0   # 9 hours
        }
        
        self.logger.info(f"🔍 Initialized gap detector for {self.exchange} {self.symbol} {self.timeframe}")
    
    @handles_errors(fallback=[], context="detect_gaps")
    @traced(span_name="detect_data_gaps", log_args=False, log_result_len_only=True)
    def detect_gaps(self, data: pd.DataFrame, data_type: str) -> List[Dict[str, Any]]:
        """
        Detect gaps in data with extensive logging.
        
        Args:
            data: DataFrame with timestamp column
            data_type: Type of data ('klines', 'aggtrades', 'futures')
            
        Returns:
            List of gap information dictionaries
        """
        self.logger.info(f"🔍 Detecting gaps in {data_type} data with {len(data)} rows")
        
        if data.empty or 'timestamp' not in data.columns:
            self.logger.warning(f"⚠️ No data or timestamp column found for gap detection")
            return []
        
        gaps = []
        threshold = self.gap_thresholds.get(data_type, 60.0)  # Default 1 minute
        
        # Sort data by timestamp
        sorted_data = data.sort_values('timestamp').reset_index(drop=True)
        
        for i in range(1, len(sorted_data)):
            current_ts = sorted_data.iloc[i]['timestamp']
            previous_ts = sorted_data.iloc[i-1]['timestamp']
            
            # Calculate gap in seconds
            gap_seconds = (current_ts - previous_ts) / 1000.0
            
            if gap_seconds > threshold:
                gap_info = {
                    'start_timestamp': previous_ts,
                    'end_timestamp': current_ts,
                    'gap_seconds': gap_seconds,
                    'gap_minutes': gap_seconds / 60.0,
                    'start_time': pd.to_datetime(previous_ts, unit='ms', utc=True),
                    'end_time': pd.to_datetime(current_ts, unit='ms', utc=True),
                    'data_type': data_type,
                    'exchange': self.exchange,
                    'symbol': self.symbol,
                    'timeframe': self.timeframe
                }
                gaps.append(gap_info)
                
                self.logger.warning(f"⚠️ Gap detected: {gap_seconds:.1f}s ({gap_seconds/60:.1f}min) at {gap_info['start_time']}")
        
        self.logger.info(f"📊 Gap detection completed: {len(gaps)} gaps found")
        return gaps
    
    @handles_errors(fallback=None, context="get_gap_summary")
    def get_gap_summary(self, gaps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary of detected gaps."""
        if not gaps:
            return {'total_gaps': 0, 'total_gap_time': 0, 'average_gap': 0}
        
        total_gap_time = sum(gap['gap_seconds'] for gap in gaps)
        average_gap = total_gap_time / len(gaps)
        
        summary = {
            'total_gaps': len(gaps),
            'total_gap_time': total_gap_time,
            'total_gap_minutes': total_gap_time / 60.0,
            'average_gap': average_gap,
            'average_gap_minutes': average_gap / 60.0,
            'largest_gap': max(gaps, key=lambda x: x['gap_seconds']),
            'smallest_gap': min(gaps, key=lambda x: x['gap_seconds'])
        }
        
        self.logger.info(f"📊 Gap Summary:")
        self.logger.info(f"   📈 Total Gaps: {summary['total_gaps']}")
        self.logger.info(f"   ⏱️ Total Gap Time: {summary['total_gap_minutes']:.1f} minutes")
        self.logger.info(f"   📊 Average Gap: {summary['average_gap_minutes']:.1f} minutes")
        
        return summary


class IncrementalDataDownloader:
    """Incremental data downloader with gap detection and batch management."""
    
    def __init__(self, exchange: str, symbol: str, timeframe: str):
        self.exchange = exchange.upper()
        self.symbol = symbol
        self.timeframe = timeframe
        self.logger = logger.getChild(f"IncrementalDownloader.{exchange}.{symbol}")
        
        # Download state
        self.last_timestamp: Optional[int] = None
        self.download_stats = {
            'total_batches': 0,
            'total_rows': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'start_time': None,
            'last_batch_time': None
        }
        
        # Initialize gap detector
        self.gap_detector = DataGapDetector(exchange, symbol, timeframe)
        
        self.logger.info(f"🚀 Initialized incremental downloader for {self.exchange} {self.symbol} {self.timeframe}")
    
    @handles_errors(fallback=None, context="get_last_timestamp")
    @traced(span_name="get_last_timestamp", log_args=False, log_result_len_only=True)
    async def get_last_timestamp(self, data_dir: str, data_type: str) -> Optional[int]:
        """Get the last timestamp from existing data files."""
        try:
            import os
            import glob
            
            self.logger.info(f"🔍 Looking for last timestamp in {data_type} data")
            
            # Look for existing data files
            pattern = f"{data_type}_{self.exchange}_{self.symbol}*_validated.parquet"
            search_path = os.path.join(data_dir, pattern)
            files = glob.glob(search_path)
            
            if not files:
                self.logger.info(f"ℹ️ No existing {data_type} files found")
                return None
            
            # Get the most recent file
            latest_file = max(files, key=os.path.getmtime)
            self.logger.info(f"📖 Reading latest file: {os.path.basename(latest_file)}")
            
            # Read the file and get the last timestamp
            df = pd.read_parquet(latest_file)
            if df.empty or 'timestamp' not in df.columns:
                self.logger.warning(f"⚠️ No data or timestamp column in {latest_file}")
                return None
            
            last_timestamp = df['timestamp'].max()
            last_time = pd.to_datetime(last_timestamp, unit='ms', utc=True)
            
            self.logger.info(f"✅ Last timestamp: {last_timestamp} ({last_time})")
            return int(last_timestamp)
            
        except Exception as e:
            self.logger.exception(f"❌ Error getting last timestamp: {e}")
            return None
    
    @handles_errors(fallback=False, context="download_incremental_batch")
    @traced(span_name="download_incremental_batch", log_args=False, log_result_len_only=True)
    @memory_efficient(batch_size=1000)
    async def download_incremental_batch(
        self, 
        data_type: str, 
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
        batch_size: int = 1000
    ) -> Tuple[bool, List[Dict[str, Any]], Optional[int]]:
        """
        Download incremental batch of data.
        
        Args:
            data_type: Type of data to download
            start_timestamp: Start timestamp (if None, uses last_timestamp)
            end_timestamp: End timestamp (if None, uses current time)
            batch_size: Number of records to download
            
        Returns:
            Tuple of (success, data, next_timestamp)
        """
        batch_start_time = time.time()
        self.download_stats['total_batches'] += 1
        self.download_stats['last_batch_time'] = batch_start_time
        
        if not self.download_stats['start_time']:
            self.download_stats['start_time'] = batch_start_time
        
        self.logger.info(f"📥 Downloading incremental {data_type} batch {self.download_stats['total_batches']}")
        
        try:
            # Determine start timestamp
            if start_timestamp is None:
                start_timestamp = self.last_timestamp
            
            if start_timestamp is None:
                # No previous data, start from 24 hours ago
                start_timestamp = int((datetime.now() - timedelta(hours=24)).timestamp() * 1000)
                self.logger.info(f"ℹ️ No previous timestamp, starting from 24 hours ago")
            
            # Determine end timestamp
            if end_timestamp is None:
                end_timestamp = int(datetime.now().timestamp() * 1000)
            
            self.logger.info(f"🕐 Download period: {pd.to_datetime(start_timestamp, unit='ms', utc=True)} to {pd.to_datetime(end_timestamp, unit='ms', utc=True)}")
            
            # Download data using exchange API
            raw_data = await self._download_from_exchange(
                data_type, start_timestamp, end_timestamp, batch_size
            )
            
            if not raw_data:
                self.logger.warning(f"⚠️ No data downloaded for {data_type}")
                self.download_stats['failed_batches'] += 1
                return False, [], None
            
            # Validate data
            validator = get_validator(DataType(data_type), self.exchange)
            validated_data = validator.validate_batch(raw_data, self.last_timestamp)
            
            if not validated_data:
                self.logger.error(f"❌ Data validation failed for {data_type}")
                self.download_stats['failed_batches'] += 1
                return False, [], None
            
            # Update statistics
            self.download_stats['total_rows'] += len(validated_data)
            self.download_stats['successful_batches'] += 1
            
            # Update last timestamp
            if validated_data:
                self.last_timestamp = validated_data[-1]['timestamp']
            
            # Calculate next timestamp for continuation
            next_timestamp = self.last_timestamp + 1 if self.last_timestamp else None
            
            batch_duration = time.time() - batch_start_time
            self.logger.info(f"✅ Batch {self.download_stats['total_batches']} completed:")
            self.logger.info(f"   📊 Rows: {len(validated_data)}")
            self.logger.info(f"   ⏱️ Duration: {batch_duration:.2f}s")
            self.logger.info(f"   🕐 Last timestamp: {self.last_timestamp}")
            
            return True, validated_data, next_timestamp
            
        except Exception as e:
            self.logger.exception(f"❌ Error downloading incremental batch: {e}")
            self.download_stats['failed_batches'] += 1
            return False, [], None
    
    @handles_errors(fallback=[], context="download_from_exchange")
    async def _download_from_exchange(
        self, 
        data_type: str, 
        start_timestamp: int, 
        end_timestamp: int, 
        batch_size: int
    ) -> List[Dict[str, Any]]:
        """Download data from exchange using API-agnostic interface."""
        try:
            self.logger.info(f"🔄 Downloading {data_type} data from {self.exchange} API")
            
            # Import exchange factory
            from exchange.factory import ExchangeFactory
            
            # Create exchange instance
            exchange_instance = ExchangeFactory.create_exchange(
                exchange_name=self.exchange,
                api_key="",  # Use public endpoints
                api_secret="",
                trade_symbol=self.symbol
            )
            
            # Download data based on type
            if data_type == 'klines':
                raw_data = await self._download_klines(exchange_instance, start_timestamp, end_timestamp, batch_size)
            elif data_type == 'aggtrades':
                raw_data = await self._download_aggtrades(exchange_instance, start_timestamp, end_timestamp, batch_size)
            elif data_type == 'futures':
                raw_data = await self._download_futures(exchange_instance, start_timestamp, end_timestamp, batch_size)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            
            self.logger.info(f"✅ Downloaded {len(raw_data)} {data_type} records from {self.exchange}")
            return raw_data
            
        except Exception as e:
            self.logger.exception(f"❌ Error downloading from exchange: {e}")
            return []
    
    @handles_errors(fallback=[], context="download_klines")
    async def _download_klines(self, exchange_instance, start_timestamp: int, end_timestamp: int, batch_size: int) -> List[Dict[str, Any]]:
        """Download klines data from exchange."""
        try:
            # Convert timestamps to datetime
            start_dt = pd.to_datetime(start_timestamp, unit='ms', utc=True)
            end_dt = pd.to_datetime(end_timestamp, unit='ms', utc=True)
            
            self.logger.info(f"📊 Downloading klines from {start_dt} to {end_dt}")
            
            # Download klines data
            klines_data = await exchange_instance.get_historical_klines(
                symbol=self.symbol,
                interval=self.timeframe,
                start_time=start_dt,
                end_time=end_dt,
                limit=batch_size
            )
            
            # Convert to list of dictionaries
            raw_data = []
            for kline in klines_data:
                raw_data.append({
                    'timestamp': kline.timestamp,
                    'open': kline.open,
                    'high': kline.high,
                    'low': kline.low,
                    'close': kline.close,
                    'volume': kline.volume
                })
            
            return raw_data
            
        except Exception as e:
            self.logger.exception(f"❌ Error downloading klines: {e}")
            return []
    
    @handles_errors(fallback=[], context="download_aggtrades")
    async def _download_aggtrades(self, exchange_instance, start_timestamp: int, end_timestamp: int, batch_size: int) -> List[Dict[str, Any]]:
        """Download aggtrades data from exchange."""
        try:
            # Convert timestamps to datetime
            start_dt = pd.to_datetime(start_timestamp, unit='ms', utc=True)
            end_dt = pd.to_datetime(end_timestamp, unit='ms', utc=True)
            
            self.logger.info(f"📊 Downloading aggtrades from {start_dt} to {end_dt}")
            
            # Download aggtrades data (this would need to be implemented in the exchange interface)
            # For now, return empty list as aggtrades might not be available in all exchanges
            self.logger.warning(f"⚠️ Aggtrades download not implemented for {self.exchange}")
            return []
            
        except Exception as e:
            self.logger.exception(f"❌ Error downloading aggtrades: {e}")
            return []
    
    @handles_errors(fallback=[], context="download_futures")
    async def _download_futures(self, exchange_instance, start_timestamp: int, end_timestamp: int, batch_size: int) -> List[Dict[str, Any]]:
        """Download futures data from exchange."""
        try:
            # Convert timestamps to datetime
            start_dt = pd.to_datetime(start_timestamp, unit='ms', utc=True)
            end_dt = pd.to_datetime(end_timestamp, unit='ms', utc=True)
            
            self.logger.info(f"📊 Downloading futures from {start_dt} to {end_dt}")
            
            # Download futures data (this would need to be implemented in the exchange interface)
            # For now, return empty list as futures might not be available in all exchanges
            self.logger.warning(f"⚠️ Futures download not implemented for {self.exchange}")
            return []
            
        except Exception as e:
            self.logger.exception(f"❌ Error downloading futures: {e}")
            return []
    
    @handles_errors(fallback={}, context="get_download_summary")
    def get_download_summary(self) -> Dict[str, Any]:
        """Get download statistics summary."""
        total_duration = time.time() - self.download_stats['start_time'] if self.download_stats['start_time'] else 0
        
        summary = {
            'exchange': self.exchange,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'total_batches': self.download_stats['total_batches'],
            'successful_batches': self.download_stats['successful_batches'],
            'failed_batches': self.download_stats['failed_batches'],
            'success_rate': self.download_stats['successful_batches'] / self.download_stats['total_batches'] * 100 if self.download_stats['total_batches'] > 0 else 0,
            'total_rows': self.download_stats['total_rows'],
            'total_duration': total_duration,
            'last_timestamp': self.last_timestamp,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"📊 Download Summary for {self.exchange} {self.symbol} {self.timeframe}:")
        self.logger.info(f"   📦 Total Batches: {summary['total_batches']}")
        self.logger.info(f"   ✅ Successful: {summary['successful_batches']}")
        self.logger.info(f"   ❌ Failed: {summary['failed_batches']}")
        self.logger.info(f"   📈 Success Rate: {summary['success_rate']:.1f}%")
        self.logger.info(f"   📊 Total Rows: {summary['total_rows']}")
        self.logger.info(f"   ⏱️ Duration: {summary['total_duration']:.2f}s")
        
        return summary


class EnhancedAPIAgnosticDataCollector:
    """Enhanced API-agnostic data collector with comprehensive features."""
    
    def __init__(self, exchange: str, symbol: str, timeframe: str):
        self.exchange = exchange.upper()
        self.symbol = symbol
        self.timeframe = timeframe
        self.logger = logger.getChild(f"APIAgnosticCollector.{exchange}.{symbol}")
        
        # Initialize components
        self.incremental_downloader = IncrementalDataDownloader(exchange, symbol, timeframe)
        self.gap_detector = DataGapDetector(exchange, symbol, timeframe)
        
        # Collection state
        self.collection_stats = {
            'total_data_collected': 0,
            'klines_rows': 0,
            'aggtrades_rows': 0,
            'futures_rows': 0,
            'gaps_detected': 0,
            'collection_start_time': None,
            'last_collection_time': None
        }
        
        self.logger.info(f"🚀 Initialized Enhanced API-Agnostic Data Collector for {self.exchange} {self.symbol} {self.timeframe}")
    
    @handles_errors(fallback=False, context="collect_data_for_period")
    @traced(span_name="collect_data_for_period", log_args=False, log_result_len_only=True)
    @with_enhanced_mlflow_logging
    async def collect_data_for_period(
        self, 
        start_time: datetime, 
        end_time: datetime, 
        data_types: List[str] = None,
        data_dir: str = "data_cache"
    ) -> Dict[str, Any]:
        """
        Collect data for a specific time period.
        
        Args:
            start_time: Start time for data collection
            end_time: End time for data collection
            data_types: List of data types to collect (default: ['klines'])
            data_dir: Directory to save data
            
        Returns:
            Collection summary
        """
        if data_types is None:
            data_types = ['klines']
        
        collection_start = time.time()
        self.collection_stats['collection_start_time'] = collection_start
        
        self.logger.info(f"📅 Collecting data for period: {start_time} to {end_time}")
        self.logger.info(f"📊 Data types: {data_types}")
        
        # Convert to timestamps
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        collection_results = {}
        
        try:
            for data_type in data_types:
                self.logger.info(f"🔄 Collecting {data_type} data...")
                
                # Download data for the period
                success, data, _ = await self.incremental_downloader.download_incremental_batch(
                    data_type=data_type,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    batch_size=10000  # Large batch for period collection
                )
                
                if success and data:
                    # Save data
                    await self._save_collected_data(data, data_type, data_dir)
                    
                    # Update statistics
                    self.collection_stats[f'{data_type}_rows'] += len(data)
                    self.collection_stats['total_data_collected'] += len(data)
                    
                    collection_results[data_type] = {
                        'success': True,
                        'rows': len(data),
                        'start_time': start_time.isoformat(),
                        'end_time': end_time.isoformat()
                    }
                    
                    self.logger.info(f"✅ Collected {len(data)} {data_type} rows")
                else:
                    collection_results[data_type] = {
                        'success': False,
                        'rows': 0,
                        'error': 'Download or validation failed'
                    }
                    
                    self.logger.error(f"❌ Failed to collect {data_type} data")
            
            # Generate summary
            total_duration = time.time() - collection_start
            summary = {
                'exchange': self.exchange,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_duration': total_duration,
                'collection_results': collection_results,
                'total_rows_collected': self.collection_stats['total_data_collected'],
                'success': all(result['success'] for result in collection_results.values()),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Period collection completed in {total_duration:.2f}s")
            self.logger.info(f"📊 Total rows collected: {summary['total_rows_collected']}")
            
            return summary
            
        except Exception as e:
            self.logger.exception(f"❌ Error collecting data for period: {e}")
            return {
                'exchange': self.exchange,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    @handles_errors(fallback=False, context="collect_incremental_data")
    @traced(span_name="collect_incremental_data", log_args=False, log_result_len_only=True)
    async def collect_incremental_data(
        self, 
        data_types: List[str] = None,
        data_dir: str = "data_cache",
        max_batches: int = 10
    ) -> Dict[str, Any]:
        """
        Collect data incrementally, starting from the last timestamp.
        
        Args:
            data_types: List of data types to collect
            data_dir: Directory to save data
            max_batches: Maximum number of batches to download
            
        Returns:
            Collection summary
        """
        if data_types is None:
            data_types = ['klines']
        
        collection_start = time.time()
        self.collection_stats['collection_start_time'] = collection_start
        
        self.logger.info(f"🔄 Starting incremental data collection")
        self.logger.info(f"📊 Data types: {data_types}")
        self.logger.info(f"📦 Max batches: {max_batches}")
        
        collection_results = {}
        
        try:
            for data_type in data_types:
                self.logger.info(f"🔄 Collecting {data_type} data incrementally...")
                
                # Get last timestamp from existing data
                last_timestamp = await self.incremental_downloader.get_last_timestamp(data_dir, data_type)
                
                if last_timestamp:
                    self.logger.info(f"🕐 Resuming from timestamp: {pd.to_datetime(last_timestamp, unit='ms', utc=True)}")
                else:
                    self.logger.info(f"ℹ️ No existing data found, starting from 24 hours ago")
                
                # Download incremental batches
                batch_results = []
                for batch_num in range(max_batches):
                    self.logger.info(f"📦 Downloading batch {batch_num + 1}/{max_batches}")
                    
                    success, data, next_timestamp = await self.incremental_downloader.download_incremental_batch(
                        data_type=data_type,
                        start_timestamp=last_timestamp,
                        batch_size=1000
                    )
                    
                    if success and data:
                        # Save batch data
                        await self._save_collected_data(data, data_type, data_dir, batch_num)
                        
                        # Update statistics
                        self.collection_stats[f'{data_type}_rows'] += len(data)
                        self.collection_stats['total_data_collected'] += len(data)
                        
                        batch_results.append({
                            'batch': batch_num + 1,
                            'rows': len(data),
                            'success': True
                        })
                        
                        # Update last timestamp for next batch
                        last_timestamp = next_timestamp
                        
                        self.logger.info(f"✅ Batch {batch_num + 1}: {len(data)} rows")
                    else:
                        self.logger.warning(f"⚠️ Batch {batch_num + 1} failed")
                        batch_results.append({
                            'batch': batch_num + 1,
                            'rows': 0,
                            'success': False
                        })
                        break
                
                collection_results[data_type] = {
                    'success': len([r for r in batch_results if r['success']]) > 0,
                    'total_batches': len(batch_results),
                    'successful_batches': len([r for r in batch_results if r['success']]),
                    'total_rows': sum(r['rows'] for r in batch_results),
                    'batch_results': batch_results
                }
                
                self.logger.info(f"✅ {data_type} incremental collection: {collection_results[data_type]['total_rows']} rows in {collection_results[data_type]['successful_batches']} batches")
            
            # Generate summary
            total_duration = time.time() - collection_start
            summary = {
                'exchange': self.exchange,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'total_duration': total_duration,
                'collection_results': collection_results,
                'total_rows_collected': self.collection_stats['total_data_collected'],
                'success': all(result['success'] for result in collection_results.values()),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Incremental collection completed in {total_duration:.2f}s")
            self.logger.info(f"📊 Total rows collected: {summary['total_rows_collected']}")
            
            return summary
            
        except Exception as e:
            self.logger.exception(f"❌ Error in incremental data collection: {e}")
            return {
                'exchange': self.exchange,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    @handles_errors(fallback=False, context="detect_and_fill_gaps")
    @traced(span_name="detect_and_fill_gaps", log_args=False, log_result_len_only=True)
    async def detect_and_fill_gaps(
        self, 
        data_dir: str = "data_cache",
        data_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Detect gaps in existing data and fill them.
        
        Args:
            data_dir: Directory containing data files
            data_types: List of data types to check
            
        Returns:
            Gap detection and filling summary
        """
        if data_types is None:
            data_types = ['klines']
        
        self.logger.info(f"🔍 Detecting and filling gaps in {data_types}")
        
        gap_results = {}
        
        try:
            for data_type in data_types:
                self.logger.info(f"🔍 Checking gaps in {data_type} data...")
                
                # Load existing data
                existing_data = await self._load_existing_data(data_dir, data_type)
                
                if existing_data.empty:
                    self.logger.info(f"ℹ️ No existing {data_type} data found")
                    gap_results[data_type] = {'gaps_found': 0, 'gaps_filled': 0}
                    continue
                
                # Detect gaps
                gaps = self.gap_detector.detect_gaps(existing_data, data_type)
                
                if not gaps:
                    self.logger.info(f"✅ No gaps found in {data_type} data")
                    gap_results[data_type] = {'gaps_found': 0, 'gaps_filled': 0}
                    continue
                
                # Fill gaps
                gaps_filled = 0
                for gap in gaps:
                    self.logger.info(f"🔄 Filling gap: {gap['start_time']} to {gap['end_time']}")
                    
                    success, data, _ = await self.incremental_downloader.download_incremental_batch(
                        data_type=data_type,
                        start_timestamp=gap['start_timestamp'],
                        end_timestamp=gap['end_timestamp'],
                        batch_size=10000
                    )
                    
                    if success and data:
                        # Save gap data
                        await self._save_collected_data(data, data_type, data_dir, gap_id=gap['start_timestamp'])
                        gaps_filled += 1
                        self.logger.info(f"✅ Filled gap with {len(data)} rows")
                    else:
                        self.logger.warning(f"⚠️ Failed to fill gap")
                
                gap_results[data_type] = {
                    'gaps_found': len(gaps),
                    'gaps_filled': gaps_filled,
                    'gap_summary': self.gap_detector.get_gap_summary(gaps)
                }
                
                self.logger.info(f"✅ {data_type} gap analysis: {len(gaps)} gaps found, {gaps_filled} filled")
            
            # Update statistics
            total_gaps = sum(result['gaps_found'] for result in gap_results.values())
            total_filled = sum(result['gaps_filled'] for result in gap_results.values())
            self.collection_stats['gaps_detected'] += total_gaps
            
            summary = {
                'exchange': self.exchange,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'gap_results': gap_results,
                'total_gaps_found': total_gaps,
                'total_gaps_filled': total_filled,
                'success': total_filled > 0,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"✅ Gap detection and filling completed")
            self.logger.info(f"📊 Total gaps: {total_gaps}, Filled: {total_filled}")
            
            return summary
            
        except Exception as e:
            self.logger.exception(f"❌ Error in gap detection and filling: {e}")
            return {
                'exchange': self.exchange,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    @handles_errors(fallback=pd.DataFrame(), context="load_existing_data")
    async def _load_existing_data(self, data_dir: str, data_type: str) -> pd.DataFrame:
        """Load existing data from files."""
        try:
            import os
            import glob
            
            pattern = f"{data_type}_{self.exchange}_{self.symbol}*_validated.parquet"
            search_path = os.path.join(data_dir, pattern)
            files = glob.glob(search_path)
            
            if not files:
                return pd.DataFrame()
            
            # Load and combine all files
            dataframes = []
            for file in files:
                df = pd.read_parquet(file)
                dataframes.append(df)
            
            if dataframes:
                combined_df = pd.concat(dataframes, ignore_index=True)
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                self.logger.info(f"📖 Loaded {len(combined_df)} existing {data_type} rows from {len(files)} files")
                return combined_df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.exception(f"❌ Error loading existing data: {e}")
            return pd.DataFrame()
    
    @handles_errors(fallback=False, context="save_collected_data")
    async def _save_collected_data(self, data: List[Dict[str, Any]], data_type: str, data_dir: str, batch_num: int = None, gap_id: int = None):
        """Save collected data to files."""
        try:
            import os
from src.core.decorators.errors import handles_errors
            
            # Create data directory
            os.makedirs(data_dir, exist_ok=True)
            
            # Generate filename
            if gap_id:
                filename = f"{data_type}_{self.exchange}_{self.symbol}_{self.timeframe}_gap_{gap_id}_validated.parquet"
            elif batch_num is not None:
                filename = f"{data_type}_{self.exchange}_{self.symbol}_{self.timeframe}_batch_{batch_num}_validated.parquet"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{data_type}_{self.exchange}_{self.symbol}_{self.timeframe}_{timestamp}_validated.parquet"
            
            filepath = os.path.join(data_dir, filename)
            
            # Convert to DataFrame and save
            df = pd.DataFrame(data)
            df.to_parquet(filepath, index=False)
            
            self.logger.info(f"💾 Saved {len(data)} {data_type} rows to {filename}")
            
        except Exception as e:
            self.logger.exception(f"❌ Error saving collected data: {e}")
    
    @handles_errors(fallback={}, context="get_collection_summary")
    def get_collection_summary(self) -> Dict[str, Any]:
        """Get comprehensive collection summary."""
        total_duration = time.time() - self.collection_stats['collection_start_time'] if self.collection_stats['collection_start_time'] else 0
        
        summary = {
            'exchange': self.exchange,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'total_data_collected': self.collection_stats['total_data_collected'],
            'klines_rows': self.collection_stats['klines_rows'],
            'aggtrades_rows': self.collection_stats['aggtrades_rows'],
            'futures_rows': self.collection_stats['futures_rows'],
            'gaps_detected': self.collection_stats['gaps_detected'],
            'total_duration': total_duration,
            'download_summary': self.incremental_downloader.get_download_summary(),
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"📊 Collection Summary for {self.exchange} {self.symbol} {self.timeframe}:")
        self.logger.info(f"   📊 Total Data: {summary['total_data_collected']} rows")
        self.logger.info(f"   📈 Klines: {summary['klines_rows']} rows")
        self.logger.info(f"   📈 Aggtrades: {summary['aggtrades_rows']} rows")
        self.logger.info(f"   📈 Futures: {summary['futures_rows']} rows")
        self.logger.info(f"   🕐 Gaps Detected: {summary['gaps_detected']}")
        self.logger.info(f"   ⏱️ Duration: {summary['total_duration']:.2f}s")
        
        return summary


# Convenience functions
@handles_errors(fallback=False, context="collect_data_for_period")
@traced(span_name="collect_data_for_period", log_args=False, log_result_len_only=True)
async def collect_data_for_period(
    exchange: str,
    symbol: str,
    timeframe: str,
    start_time: datetime,
    end_time: datetime,
    data_types: List[str] = None,
    data_dir: str = "data_cache"
) -> Dict[str, Any]:
    """Collect data for a specific time period."""
    collector = EnhancedAPIAgnosticDataCollector(exchange, symbol, timeframe)
    return await collector.collect_data_for_period(start_time, end_time, data_types, data_dir)


@handles_errors(fallback=False, context="collect_incremental_data")
@traced(span_name="collect_incremental_data", log_args=False, log_result_len_only=True)
async def collect_incremental_data(
    exchange: str,
    symbol: str,
    timeframe: str,
    data_types: List[str] = None,
    data_dir: str = "data_cache",
    max_batches: int = 10
) -> Dict[str, Any]:
    """Collect data incrementally."""
    collector = EnhancedAPIAgnosticDataCollector(exchange, symbol, timeframe)
    return await collector.collect_incremental_data(data_types, data_dir, max_batches)


@handles_errors(fallback=False, context="detect_and_fill_gaps")
@traced(span_name="detect_and_fill_gaps", log_args=False, log_result_len_only=True)
async def detect_and_fill_gaps(
    exchange: str,
    symbol: str,
    timeframe: str,
    data_dir: str = "data_cache",
    data_types: List[str] = None
) -> Dict[str, Any]:
    """Detect and fill gaps in existing data."""
    collector = EnhancedAPIAgnosticDataCollector(exchange, symbol, timeframe)
    return await collector.detect_and_fill_gaps(data_dir, data_types)


if __name__ == "__main__":
    # Example usage
    async def test_api_agnostic_collector():
        logger.info("🎯 Testing Enhanced API-Agnostic Data Collector")
        logger.info("=" * 80)
        
        # Test incremental data collection
        logger.info("📊 Testing incremental data collection...")
        result = await collect_incremental_data(
            exchange="BINANCE",
            symbol="ETHUSDT",
            timeframe="1m",
            data_types=["klines"],
            max_batches=3
        )
        
        logger.info(f"✅ Incremental collection result: {result['success']}")
        
        # Test gap detection
        logger.info("🔍 Testing gap detection...")
        gap_result = await detect_and_fill_gaps(
            exchange="BINANCE",
            symbol="ETHUSDT",
            timeframe="1m",
            data_types=["klines"]
        )
        
        logger.info(f"✅ Gap detection result: {gap_result['success']}")
        
        logger.info("=" * 80)
        logger.info("🎉 API-agnostic data collector tests completed!")
        logger.info("=" * 80)
    
    asyncio.run(test_api_agnostic_collector())