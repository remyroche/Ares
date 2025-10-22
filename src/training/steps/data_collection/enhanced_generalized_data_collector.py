#!/usr/bin/env python3
"""
Enhanced Generalized Data Collector

This module provides a comprehensive data collection framework that leverages all
the comprehensive tools available in BaseStep. It serves as a generalized foundation
for all data collection operations with:

- Complete BaseStep integration with all comprehensive utilities
- Hardware optimization and memory management
- Advanced logging with tprint integration
- Data quality validation and cleaning
- Model persistence and caching
- ML common utilities integration
- Comprehensive error handling and validation

Features:
- API-agnostic data collection using exchange/ directory
- Comprehensive data gap detection and filling
- Incremental downloading with batch management
- Advanced data quality assessment and cleaning
- Hardware-optimized data processing
- Comprehensive logging and monitoring
- Integration with all BaseStep comprehensive tools
"""

import asyncio
import sys
import time
import os
import glob
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Awaitable
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import BaseStep for comprehensive tool access
from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger

logger = system_logger.getChild("EnhancedGeneralizedDataCollector")

class EnhancedGeneralizedDataCollector(BaseStep):
    """
    Enhanced generalized data collector that leverages all BaseStep comprehensive tools.
    
    This class provides a comprehensive foundation for all data collection operations
    with full access to BaseStep utilities including:
    - Hardware optimization and memory management
    - Advanced logging with tprint integration
    - Data quality validation and cleaning
    - Model persistence and caching
    - ML common utilities integration
    - Comprehensive error handling and validation
    """
    
    def __init__(self, step_name: str = "enhanced_data_collection", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced generalized data collector.
        
        Args:
            step_name: Name for this autonomous step
            config: Configuration dictionary
        """
        super().__init__(step_name, config)
        
        # Data collection configuration
        self.exchange = config.get('exchange', 'BINANCE').upper() if config else 'BINANCE'
        self.symbol = config.get('symbol', 'ETHUSDT') if config else 'ETHUSDT'
        self.timeframe = config.get('timeframe', '1m') if config else '1m'
        self.data_dir = config.get('data_dir', 'historical_data') if config else 'historical_data'
        
        # Collection state
        self.collection_stats = {
            'total_data_collected': 0,
            'klines_rows': 0,
            'aggtrades_rows': 0,
            'futures_rows': 0,
            'gaps_detected': 0,
            'collection_start_time': None,
            'last_collection_time': None,
            'quality_scores': [],
            'performance_metrics': []
        }
        
        # Initialize comprehensive logging
        self.tprint_info(f"🚀 Initialized Enhanced Generalized Data Collector")
        self.tprint_info(f"   📊 Exchange: {self.exchange}")
        self.tprint_info(f"   📈 Symbol: {self.symbol}")
        self.tprint_info(f"   ⏱️ Timeframe: {self.timeframe}")
        self.tprint_info(f"   📁 Data Directory: {self.data_dir}")
        
        # Log utility availability
        self._log_utility_availability()
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the data collection step using comprehensive BaseStep tools.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Execution result with artifacts and metadata
        """
        try:
            # Update configuration
            self.exchange = config.get('exchange', self.exchange).upper()
            self.symbol = config.get('symbol', self.symbol)
            self.timeframe = config.get('timeframe', self.timeframe)
            self.data_dir = config.get('data_dir', self.data_dir)
            
            # Start comprehensive logging
            self.tprint_step_start("Enhanced Data Collection")
            self.tprint_banner("Data Collection Pipeline")
            
            # Set context for enhanced file naming
            self._set_context(
                symbol=self.symbol,
                exchange=self.exchange,
                information=config.get('information', 'klines'),
                direction=config.get('direction', 'long'),
                model=config.get('model', 'Analyst')
            )
            
            # Determine collection mode
            collection_mode = config.get('collection_mode', 'incremental')
            
            if collection_mode == 'incremental':
                result = await self._collect_incremental_data(config)
            elif collection_mode == 'period':
                result = await self._collect_data_for_period(config)
            elif collection_mode == 'gap_filling':
                result = await self._detect_and_fill_gaps(config)
            else:
                raise ValueError(f"Unknown collection mode: {collection_mode}")
            
            # Generate comprehensive summary
            summary = self._generate_collection_summary(result)
            
            # Log performance metrics
            self.tprint_performance_summary(self.collection_stats['performance_metrics'])
            self.tprint_memory_usage()
            self.tprint_hardware_stats()
            
            # End comprehensive logging
            self.tprint_step_end("Enhanced Data Collection")
            
            return {
                'success': True,
                'artifacts': result.get('artifacts', []),
                'metadata': summary,
                'performance_metrics': self.collection_stats['performance_metrics'],
                'quality_scores': self.collection_stats['quality_scores']
            }
            
        except Exception as e:
            self.tprint_error(f"❌ Data collection failed: {e}")
            self.tprint_exception(e)
            return {
                'success': False,
                'error': str(e),
                'artifacts': [],
                'metadata': {}
            }
    
    async def _collect_incremental_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Collect data incrementally using comprehensive BaseStep tools."""
        self.tprint_operation_start("Incremental Data Collection")
        
        try:
            # Get collection parameters
            data_types = config.get('data_types', ['klines'])
            max_batches = config.get('max_batches', 10)
            batch_size = config.get('batch_size', 1000)
            
            self.tprint_info(f"📊 Collecting incremental data:")
            self.tprint_info(f"   📈 Data types: {data_types}")
            self.tprint_info(f"   📦 Max batches: {max_batches}")
            self.tprint_info(f"   📊 Batch size: {batch_size}")
            
            collection_results = {}
            
            for data_type in data_types:
                self.tprint_operation_start(f"Collecting {data_type} data")
                
                # Get last timestamp using BaseStep utilities
                last_timestamp = await self._get_last_timestamp(data_type)
                
                if last_timestamp:
                    self.tprint_info(f"🕐 Resuming from timestamp: {pd.to_datetime(last_timestamp, unit='ms', utc=True)}")
                else:
                    self.tprint_info(f"ℹ️ No existing data found, starting from 24 hours ago")
                
                # Download incremental batches
                batch_results = []
                for batch_num in range(max_batches):
                    self.tprint_progress(f"Downloading batch {batch_num + 1}/{max_batches}")
                    
                    success, data, next_timestamp = await self._download_incremental_batch(
                        data_type, last_timestamp, batch_size
                    )
                    
                    if success and data:
                        # Process data using comprehensive tools
                        processed_data = await self._process_data_with_comprehensive_tools(data, data_type)
                        
                        # Save data using BaseStep utilities
                        await self._save_data_with_comprehensive_tools(processed_data, data_type, batch_num)
                        
                        # Update statistics
                        self.collection_stats[f'{data_type}_rows'] += len(processed_data)
                        self.collection_stats['total_data_collected'] += len(processed_data)
                        
                        batch_results.append({
                            'batch': batch_num + 1,
                            'rows': len(processed_data),
                            'success': True
                        })
                        
                        # Update last timestamp for next batch
                        last_timestamp = next_timestamp
                        
                        self.tprint_success(f"✅ Batch {batch_num + 1}: {len(processed_data)} rows")
                    else:
                        self.tprint_warning(f"⚠️ Batch {batch_num + 1} failed")
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
                
                self.tprint_operation_end(f"Collecting {data_type} data")
                self.tprint_success(f"✅ {data_type} collection: {collection_results[data_type]['total_rows']} rows")
            
            self.tprint_operation_end("Incremental Data Collection")
            return {
                'success': all(result['success'] for result in collection_results.values()),
                'collection_results': collection_results,
                'artifacts': [f"{dt}_data" for dt in data_types if collection_results[dt]['success']]
            }
            
        except Exception as e:
            self.tprint_error(f"❌ Incremental data collection failed: {e}")
            self.tprint_exception(e)
            return {'success': False, 'error': str(e)}
    
    async def _collect_data_for_period(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Collect data for a specific time period using comprehensive BaseStep tools."""
        self.tprint_operation_start("Period Data Collection")
        
        try:
            # Get period parameters
            start_time = config.get('start_time')
            end_time = config.get('end_time')
            data_types = config.get('data_types', ['klines'])
            
            if not start_time or not end_time:
                raise ValueError("start_time and end_time are required for period collection")
            
            self.tprint_info(f"📅 Collecting data for period: {start_time} to {end_time}")
            self.tprint_info(f"📊 Data types: {data_types}")
            
            # Convert to timestamps
            start_timestamp = int(start_time.timestamp() * 1000)
            end_timestamp = int(end_time.timestamp() * 1000)
            
            collection_results = {}
            
            for data_type in data_types:
                self.tprint_operation_start(f"Collecting {data_type} for period")
                
                # Download data for the period
                success, data, _ = await self._download_incremental_batch(
                    data_type, start_timestamp, end_timestamp, 10000
                )
                
                if success and data:
                    # Process data using comprehensive tools
                    processed_data = await self._process_data_with_comprehensive_tools(data, data_type)
                    
                    # Save data using BaseStep utilities
                    await self._save_data_with_comprehensive_tools(processed_data, data_type)
                    
                    # Update statistics
                    self.collection_stats[f'{data_type}_rows'] += len(processed_data)
                    self.collection_stats['total_data_collected'] += len(processed_data)
                    
                    collection_results[data_type] = {
                        'success': True,
                        'rows': len(processed_data),
                        'start_time': start_time.isoformat(),
                        'end_time': end_time.isoformat()
                    }
                    
                    self.tprint_success(f"✅ Collected {len(processed_data)} {data_type} rows")
                else:
                    collection_results[data_type] = {
                        'success': False,
                        'rows': 0,
                        'error': 'Download or validation failed'
                    }
                    
                    self.tprint_error(f"❌ Failed to collect {data_type} data")
                
                self.tprint_operation_end(f"Collecting {data_type} for period")
            
            self.tprint_operation_end("Period Data Collection")
            return {
                'success': all(result['success'] for result in collection_results.values()),
                'collection_results': collection_results,
                'artifacts': [f"{dt}_data" for dt in data_types if collection_results[dt]['success']]
            }
            
        except Exception as e:
            self.tprint_error(f"❌ Period data collection failed: {e}")
            self.tprint_exception(e)
            return {'success': False, 'error': str(e)}
    
    async def _detect_and_fill_gaps(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect and fill gaps in existing data using comprehensive BaseStep tools."""
        self.tprint_operation_start("Gap Detection and Filling")
        
        try:
            data_types = config.get('data_types', ['klines'])
            
            self.tprint_info(f"🔍 Detecting and filling gaps in {data_types}")
            
            gap_results = {}
            
            for data_type in data_types:
                self.tprint_operation_start(f"Gap analysis for {data_type}")
                
                # Load existing data using BaseStep utilities
                existing_data = await self._load_existing_data(data_type)
                
                if existing_data.empty:
                    self.tprint_info(f"ℹ️ No existing {data_type} data found")
                    gap_results[data_type] = {'gaps_found': 0, 'gaps_filled': 0}
                    continue
                
                # Detect gaps using comprehensive tools
                gaps = await self._detect_gaps_with_comprehensive_tools(existing_data, data_type)
                
                if not gaps:
                    self.tprint_info(f"✅ No gaps found in {data_type} data")
                    gap_results[data_type] = {'gaps_found': 0, 'gaps_filled': 0}
                    continue
                
                # Fill gaps
                gaps_filled = 0
                for gap in gaps:
                    self.tprint_info(f"🔄 Filling gap: {gap['start_time']} to {gap['end_time']}")
                    
                    success, data, _ = await self._download_incremental_batch(
                        data_type, gap['start_timestamp'], gap['end_timestamp'], 10000
                    )
                    
                    if success and data:
                        # Process and save gap data
                        processed_data = await self._process_data_with_comprehensive_tools(data, data_type)
                        await self._save_data_with_comprehensive_tools(processed_data, data_type, gap_id=gap['start_timestamp'])
                        gaps_filled += 1
                        self.tprint_success(f"✅ Filled gap with {len(processed_data)} rows")
                    else:
                        self.tprint_warning(f"⚠️ Failed to fill gap")
                
                gap_results[data_type] = {
                    'gaps_found': len(gaps),
                    'gaps_filled': gaps_filled,
                    'gap_summary': self._get_gap_summary(gaps)
                }
                
                self.tprint_operation_end(f"Gap analysis for {data_type}")
                self.tprint_success(f"✅ {data_type} gap analysis: {len(gaps)} gaps found, {gaps_filled} filled")
            
            # Update statistics
            total_gaps = sum(result['gaps_found'] for result in gap_results.values())
            total_filled = sum(result['gaps_filled'] for result in gap_results.values())
            self.collection_stats['gaps_detected'] += total_gaps
            
            self.tprint_operation_end("Gap Detection and Filling")
            return {
                'success': total_filled > 0,
                'gap_results': gap_results,
                'total_gaps_found': total_gaps,
                'total_gaps_filled': total_filled,
                'artifacts': [f"{dt}_gap_filled_data" for dt in data_types if gap_results[dt]['gaps_filled'] > 0]
            }
            
        except Exception as e:
            self.tprint_error(f"❌ Gap detection and filling failed: {e}")
            self.tprint_exception(e)
            return {'success': False, 'error': str(e)}
    
    async def _get_last_timestamp(self, data_type: str) -> Optional[int]:
        """Get the last timestamp from existing data files using BaseStep utilities."""
        try:
            # Use BaseStep file utilities
            pattern = f"{data_type}_{self.exchange}_{self.symbol}*_validated.parquet"
            search_path = os.path.join(self.data_dir, pattern)
            files = glob.glob(search_path)
            
            if not files:
                return None
            
            # Get the most recent file
            latest_file = max(files, key=os.path.getmtime)
            
            # Read using BaseStep utilities
            df = self._safe_read_parquet(latest_file)
            if df.empty or 'timestamp' not in df.columns:
                return None
            
            last_timestamp = df['timestamp'].max()
            return int(last_timestamp)
            
        except Exception as e:
            self.tprint_warning(f"⚠️ Error getting last timestamp: {e}")
            return None
    
    async def _download_incremental_batch(
        self,
        data_type: str,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
        batch_size: int = 1000
    ) -> Tuple[bool, List[Dict[str, Any]], Optional[int]]:
        """Download incremental batch of data using comprehensive BaseStep tools."""
        try:
            # Determine timestamps
            if start_timestamp is None:
                start_timestamp = int((datetime.now() - timedelta(hours=24)).timestamp() * 1000)
            
            if end_timestamp is None:
                end_timestamp = int(datetime.now().timestamp() * 1000)
            
            # Download data from exchange
            raw_data = await self._download_from_exchange(data_type, start_timestamp, end_timestamp, batch_size)
            
            if not raw_data:
                return False, [], None
            
            # Validate data using comprehensive tools
            validated_data = await self._validate_data_with_comprehensive_tools(raw_data, data_type)
            
            if not validated_data:
                return False, [], None
            
            # Calculate next timestamp
            next_timestamp = validated_data[-1]['timestamp'] + 1 if validated_data else None
            
            return True, validated_data, next_timestamp
            
        except Exception as e:
            self.tprint_error(f"❌ Error downloading incremental batch: {e}")
            return False, [], None
    
    async def _download_from_exchange(
        self,
        data_type: str,
        start_timestamp: int,
        end_timestamp: int,
        batch_size: int
    ) -> List[Dict[str, Any]]:
        """Download data from exchange using API-agnostic interface."""
        try:
            # Import exchange factory
            from exchanges.factory import ExchangeFactory
            
            # Create exchange instance
            exchange_instance = ExchangeFactory.create_exchange(
                exchange_name=self.exchange,
                api_key="",  # Use public endpoints
                api_secret="",
                trade_symbol=self.symbol
            )
            
            # Download data based on type
            if data_type == 'klines':
                return await self._download_klines(exchange_instance, start_timestamp, end_timestamp, batch_size)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            
        except Exception as e:
            self.tprint_error(f"❌ Error downloading from exchange: {e}")
            return []
    
    async def _download_klines(self, exchange_instance, start_timestamp: int, end_timestamp: int, batch_size: int) -> List[Dict[str, Any]]:
        """Download klines data from exchange."""
        try:
            # Convert timestamps to datetime
            start_dt = pd.to_datetime(start_timestamp, unit='ms', utc=True)
            end_dt = pd.to_datetime(end_timestamp, unit='ms', utc=True)
            
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
            self.tprint_error(f"❌ Error downloading klines: {e}")
            return []
    
    async def _process_data_with_comprehensive_tools(self, data: List[Dict[str, Any]], data_type: str) -> List[Dict[str, Any]]:
        """Process data using comprehensive BaseStep tools."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Use hardware optimization
            if self.hardware_utils:
                df = self.hardware_utils['optimize_dataframe'](df)
            
            # Use data quality tools
            if self.data_quality:
                cleaner = self._get_data_cleaner()
                if cleaner:
                    df = cleaner.clean(df)
            
            # Use ML common utilities for validation
            if self.ml_common:
                # Check for data leakage
                leakage_detector = self._get_data_leakage_detector()
                if leakage_detector:
                    leakage_result = leakage_detector.detect_leakage(df)
                    if leakage_result.has_leakage:
                        self.tprint_warning(f"⚠️ Data leakage detected in {data_type}")
            
            # Convert back to list of dictionaries
            return df.to_dict('records')
            
        except Exception as e:
            self.tprint_error(f"❌ Error processing data: {e}")
            return data
    
    async def _validate_data_with_comprehensive_tools(self, data: List[Dict[str, Any]], data_type: str) -> List[Dict[str, Any]]:
        """Validate data using comprehensive BaseStep tools."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Use comprehensive validation
            if self.core_decorators:
                validation_result = self._validate_dataframe_columns(df, ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                if not validation_result:
                    return []
            
            # Use math validation
            if self.math_validation:
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: self._validate_finite(x, default=0))
            
            # Use data quality validation
            if self.data_quality:
                quality_result = self._get_data_quality_assessment(df, data_type)
                if quality_result and quality_result.get('valid', True):
                    self.collection_stats['quality_scores'].append(quality_result.get('quality_score', 0))
            
            return df.to_dict('records')
            
        except Exception as e:
            self.tprint_error(f"❌ Error validating data: {e}")
            return []
    
    async def _save_data_with_comprehensive_tools(
        self,
        data: List[Dict[str, Any]],
        data_type: str,
        batch_num: Optional[int] = None,
        gap_id: Optional[int] = None
    ):
        """Save data using comprehensive BaseStep tools."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Use BaseStep utilities for file operations
            self._ensure_directory(self.data_dir)
            
            # Generate filename
            if gap_id:
                filename = f"{data_type}_{self.exchange}_{self.symbol}_{self.timeframe}_gap_{gap_id}_validated.parquet"
            elif batch_num is not None:
                filename = f"{data_type}_{self.exchange}_{self.symbol}_{self.timeframe}_batch_{batch_num}_validated.parquet"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{data_type}_{self.exchange}_{self.symbol}_{self.timeframe}_{timestamp}_validated.parquet"
            
            filepath = os.path.join(self.data_dir, filename)
            
            # Use BaseStep utilities for saving
            self._safe_to_parquet(df, filepath, index=False)
            
            # Also save as artifact using BaseStep
            self._save_dataframe(df, f"{data_type}_data")
            
            self.tprint_success(f"💾 Saved {len(data)} {data_type} rows to {filename}")
            
        except Exception as e:
            self.tprint_error(f"❌ Error saving data: {e}")
    
    async def _load_existing_data(self, data_type: str) -> pd.DataFrame:
        """Load existing data using BaseStep utilities."""
        try:
            pattern = f"{data_type}_{self.exchange}_{self.symbol}*_validated.parquet"
            search_path = os.path.join(self.data_dir, pattern)
            files = glob.glob(search_path)
            
            if not files:
                return pd.DataFrame()
            
            # Load and combine all files using BaseStep utilities
            dataframes = []
            for file in files:
                df = self._safe_read_parquet(file)
                dataframes.append(df)
            
            if dataframes:
                combined_df = pd.concat(dataframes, ignore_index=True)
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                return combined_df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.tprint_error(f"❌ Error loading existing data: {e}")
            return pd.DataFrame()
    
    async def _detect_gaps_with_comprehensive_tools(self, data: pd.DataFrame, data_type: str) -> List[Dict[str, Any]]:
        """Detect gaps using comprehensive BaseStep tools."""
        try:
            if data.empty or 'timestamp' not in data.columns:
                return []
            
            gaps = []
            threshold = 66.0  # 1.1 minutes for klines
            
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
            
            return gaps
            
        except Exception as e:
            self.tprint_error(f"❌ Error detecting gaps: {e}")
            return []
    
    def _get_gap_summary(self, gaps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary of detected gaps."""
        if not gaps:
            return {'total_gaps': 0, 'total_gap_time': 0, 'average_gap': 0}
        
        total_gap_time = sum(gap['gap_seconds'] for gap in gaps)
        average_gap = total_gap_time / len(gaps)
        
        return {
            'total_gaps': len(gaps),
            'total_gap_time': total_gap_time,
            'total_gap_minutes': total_gap_time / 60.0,
            'average_gap': average_gap,
            'average_gap_minutes': average_gap / 60.0,
            'largest_gap': max(gaps, key=lambda x: x['gap_seconds']),
            'smallest_gap': min(gaps, key=lambda x: x['gap_seconds'])
        }
    
    def _generate_collection_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive collection summary."""
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
            'average_quality_score': np.mean(self.collection_stats['quality_scores']) if self.collection_stats['quality_scores'] else 0,
            'performance_metrics': self.collection_stats['performance_metrics'],
            'timestamp': datetime.now().isoformat()
        }
        
        return summary

# Convenience functions for easy usage
async def collect_data_incremental(
    exchange: str,
    symbol: str,
    timeframe: str,
    data_types: List[str] = None,
    data_dir: str = "historical_data",
    max_batches: int = 10
) -> Dict[str, Any]:
    """Collect data incrementally using the enhanced generalized collector."""
    if data_types is None:
        data_types = ['klines']
    
    config = {
        'exchange': exchange,
        'symbol': symbol,
        'timeframe': timeframe,
        'data_types': data_types,
        'data_dir': data_dir,
        'max_batches': max_batches,
        'collection_mode': 'incremental'
    }
    
    collector = EnhancedGeneralizedDataCollector("incremental_collection", config)
    return await collector.execute(config)

async def collect_data_for_period(
    exchange: str,
    symbol: str,
    timeframe: str,
    start_time: datetime,
    end_time: datetime,
    data_types: List[str] = None,
    data_dir: str = "historical_data"
) -> Dict[str, Any]:
    """Collect data for a specific time period using the enhanced generalized collector."""
    if data_types is None:
        data_types = ['klines']
    
    config = {
        'exchange': exchange,
        'symbol': symbol,
        'timeframe': timeframe,
        'start_time': start_time,
        'end_time': end_time,
        'data_types': data_types,
        'data_dir': data_dir,
        'collection_mode': 'period'
    }
    
    collector = EnhancedGeneralizedDataCollector("period_collection", config)
    return await collector.execute(config)

async def detect_and_fill_gaps(
    exchange: str,
    symbol: str,
    timeframe: str,
    data_dir: str = "historical_data",
    data_types: List[str] = None
) -> Dict[str, Any]:
    """Detect and fill gaps in existing data using the enhanced generalized collector."""
    if data_types is None:
        data_types = ['klines']
    
    config = {
        'exchange': exchange,
        'symbol': symbol,
        'timeframe': timeframe,
        'data_dir': data_dir,
        'data_types': data_types,
        'collection_mode': 'gap_filling'
    }
    
    collector = EnhancedGeneralizedDataCollector("gap_filling", config)
    return await collector.execute(config)

if __name__ == "__main__":
    # Example usage
    async def test_enhanced_collector():
        logger.info("🎯 Testing Enhanced Generalized Data Collector")
        logger.info("=" * 80)
        
        # Test incremental data collection
        logger.info("📊 Testing incremental data collection...")
        result = await collect_data_incremental(
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
        logger.info("🎉 Enhanced generalized data collector tests completed!")
        logger.info("=" * 80)
    
    asyncio.run(test_enhanced_collector())