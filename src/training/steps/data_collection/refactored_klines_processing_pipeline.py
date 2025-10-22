#!/usr/bin/env python3
"""
Refactored Klines Processing Pipeline

This module demonstrates how to refactor existing data collection steps to use
the new generalized data collection tools that leverage BaseStep comprehensive utilities.

The refactored pipeline:
- Inherits from BaseStep for comprehensive tool access
- Uses generalized data collection utilities
- Leverages all BaseStep comprehensive tools
- Maintains backward compatibility
- Provides enhanced functionality

Features:
- Complete BaseStep integration with all comprehensive utilities
- Hardware optimization and memory management
- Advanced logging with tprint integration
- Data quality validation and cleaning
- Model persistence and caching
- ML common utilities integration
- Comprehensive error handling and validation
"""

import asyncio
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import BaseStep for comprehensive tool access
from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger

# Import generalized data collection utilities
from .generalized_data_collection_utils import (
    create_standard_collection_config,
    validate_collection_config,
    validate_klines_data,
    validate_data_quality,
    detect_gaps,
    analyze_gap_patterns,
    generate_filename,
    find_latest_file,
    create_performance_tracker,
    track_operation,
    finalize_performance_tracker
)

logger = system_logger.getChild("RefactoredKlinesProcessingPipeline")

class RefactoredKlinesProcessingPipeline(BaseStep):
    """
    Refactored klines processing pipeline using generalized data collection tools.
    
    This class demonstrates how to refactor existing data collection steps to use
    the new generalized tools while maintaining all functionality and adding
    comprehensive BaseStep integration.
    """
    
    def __init__(self, step_name: str = "refactored_klines_processing", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the refactored klines processing pipeline.
        
        Args:
            step_name: Name for this autonomous step
            config: Configuration dictionary
        """
        super().__init__(step_name, config)
        
        # Initialize configuration using generalized utilities
        self.collection_config = self._initialize_collection_config(config)
        
        # Initialize performance tracking
        self.performance_tracker = create_performance_tracker()
        
        # Initialize comprehensive logging
        self.tprint_info(f"🚀 Initialized Refactored Klines Processing Pipeline")
        self.tprint_info(f"   📊 Exchange: {self.collection_config['exchange']}")
        self.tprint_info(f"   📈 Symbol: {self.collection_config['symbol']}")
        self.tprint_info(f"   ⏱️ Timeframe: {self.collection_config['timeframe']}")
        
        # Log utility availability
        self._log_utility_availability()
    
    def _initialize_collection_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Initialize collection configuration using generalized utilities."""
        if config is None:
            config = {}
        
        # Create standard configuration
        collection_config = create_standard_collection_config(
            exchange=config.get('exchange', 'BINANCE'),
            symbol=config.get('symbol', 'ETHUSDT'),
            timeframe=config.get('timeframe', '1m'),
            data_dir=config.get('data_dir', 'historical_data'),
            collection_mode=config.get('collection_mode', 'incremental'),
            data_types=config.get('data_types', ['klines']),
            max_batches=config.get('max_batches', 10),
            batch_size=config.get('batch_size', 1000),
            start_time=config.get('start_time'),
            end_time=config.get('end_time')
        )
        
        # Validate configuration
        is_valid, error_message = validate_collection_config(collection_config)
        if not is_valid:
            raise ValueError(f"Invalid collection configuration: {error_message}")
        
        return collection_config
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the refactored klines processing pipeline.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Execution result with artifacts and metadata
        """
        try:
            # Update configuration
            self.collection_config.update(config)
            
            # Start comprehensive logging
            self.tprint_step_start("Refactored Klines Processing")
            self.tprint_banner("Klines Processing Pipeline")
            
            # Set context for enhanced file naming
            self._set_context(
                symbol=self.collection_config['symbol'],
                exchange=self.collection_config['exchange'],
                information=self.collection_config['information'],
                direction=self.collection_config['direction'],
                model=self.collection_config['model']
            )
            
            # Execute based on collection mode
            collection_mode = self.collection_config['collection_mode']
            
            if collection_mode == 'incremental':
                result = await self._process_incremental_data()
            elif collection_mode == 'period':
                result = await self._process_period_data()
            elif collection_mode == 'gap_filling':
                result = await self._process_gap_filling()
            else:
                raise ValueError(f"Unknown collection mode: {collection_mode}")
            
            # Finalize performance tracking
            self.performance_tracker = finalize_performance_tracker(self.performance_tracker)
            
            # Generate comprehensive summary
            summary = self._generate_processing_summary(result)
            
            # Log performance metrics
            self.tprint_performance_summary(self.performance_tracker['summary'])
            self.tprint_memory_usage()
            self.tprint_hardware_stats()
            
            # End comprehensive logging
            self.tprint_step_end("Refactored Klines Processing")
            
            return {
                'success': True,
                'artifacts': result.get('artifacts', []),
                'metadata': summary,
                'performance_metrics': self.performance_tracker['summary'],
                'quality_scores': result.get('quality_scores', [])
            }
            
        except Exception as e:
            self.tprint_error(f"❌ Klines processing failed: {e}")
            self.tprint_exception(e)
            return {
                'success': False,
                'error': str(e),
                'artifacts': [],
                'metadata': {}
            }
    
    async def _process_incremental_data(self) -> Dict[str, Any]:
        """Process incremental data using generalized utilities."""
        self.tprint_operation_start("Incremental Data Processing")
        
        try:
            # Track operation start
            operation_start = time.time()
            
            # Get collection parameters
            data_types = self.collection_config['data_types']
            max_batches = self.collection_config['max_batches']
            batch_size = self.collection_config['batch_size']
            
            self.tprint_info(f"📊 Processing incremental data:")
            self.tprint_info(f"   📈 Data types: {data_types}")
            self.tprint_info(f"   📦 Max batches: {max_batches}")
            self.tprint_info(f"   📊 Batch size: {batch_size}")
            
            processing_results = {}
            quality_scores = []
            
            for data_type in data_types:
                self.tprint_operation_start(f"Processing {data_type} data")
                
                # Get last timestamp using BaseStep utilities
                last_timestamp = await self._get_last_timestamp(data_type)
                
                if last_timestamp:
                    self.tprint_info(f"🕐 Resuming from timestamp: {pd.to_datetime(last_timestamp, unit='ms', utc=True)}")
                else:
                    self.tprint_info(f"ℹ️ No existing data found, starting from 24 hours ago")
                
                # Process incremental batches
                batch_results = []
                for batch_num in range(max_batches):
                    self.tprint_progress(f"Processing batch {batch_num + 1}/{max_batches}")
                    
                    # Track batch operation
                    batch_start = time.time()
                    
                    success, data, next_timestamp = await self._process_incremental_batch(
                        data_type, last_timestamp, batch_size
                    )
                    
                    batch_end = time.time()
                    
                    # Track operation
                    track_operation(
                        self.performance_tracker,
                        f"{data_type}_batch_{batch_num + 1}",
                        batch_start,
                        batch_end,
                        success,
                        rows_processed=len(data) if data else 0
                    )
                    
                    if success and data:
                        # Validate data using generalized utilities
                        is_valid, validation_errors = validate_klines_data(data)
                        if not is_valid:
                            self.tprint_warning(f"⚠️ Data validation failed: {validation_errors}")
                            continue
                        
                        # Process data using comprehensive tools
                        processed_data = await self._process_data_with_comprehensive_tools(data, data_type)
                        
                        # Validate data quality
                        df = pd.DataFrame(processed_data)
                        quality_result = validate_data_quality(df, data_type)
                        quality_scores.append(quality_result['quality_score'])
                        
                        # Save data using BaseStep utilities
                        await self._save_data_with_comprehensive_tools(processed_data, data_type, batch_num)
                        
                        batch_results.append({
                            'batch': batch_num + 1,
                            'rows': len(processed_data),
                            'success': True,
                            'quality_score': quality_result['quality_score']
                        })
                        
                        # Update last timestamp for next batch
                        last_timestamp = next_timestamp
                        
                        self.tprint_success(f"✅ Batch {batch_num + 1}: {len(processed_data)} rows (Quality: {quality_result['quality_score']:.1f})")
                    else:
                        self.tprint_warning(f"⚠️ Batch {batch_num + 1} failed")
                        batch_results.append({
                            'batch': batch_num + 1,
                            'rows': 0,
                            'success': False,
                            'quality_score': 0
                        })
                        break
                
                processing_results[data_type] = {
                    'success': len([r for r in batch_results if r['success']]) > 0,
                    'total_batches': len(batch_results),
                    'successful_batches': len([r for r in batch_results if r['success']]),
                    'total_rows': sum(r['rows'] for r in batch_results),
                    'average_quality_score': np.mean([r['quality_score'] for r in batch_results if r['quality_score'] > 0]) if any(r['quality_score'] > 0 for r in batch_results) else 0,
                    'batch_results': batch_results
                }
                
                self.tprint_operation_end(f"Processing {data_type} data")
                self.tprint_success(f"✅ {data_type} processing: {processing_results[data_type]['total_rows']} rows")
            
            # Track overall operation
            operation_end = time.time()
            track_operation(
                self.performance_tracker,
                "incremental_processing",
                operation_start,
                operation_end,
                all(result['success'] for result in processing_results.values()),
                rows_processed=sum(result['total_rows'] for result in processing_results.values())
            )
            
            self.tprint_operation_end("Incremental Data Processing")
            return {
                'success': all(result['success'] for result in processing_results.values()),
                'processing_results': processing_results,
                'quality_scores': quality_scores,
                'artifacts': [f"{dt}_processed_data" for dt in data_types if processing_results[dt]['success']]
            }
            
        except Exception as e:
            self.tprint_error(f"❌ Incremental data processing failed: {e}")
            self.tprint_exception(e)
            return {'success': False, 'error': str(e)}
    
    async def _process_period_data(self) -> Dict[str, Any]:
        """Process data for a specific time period using generalized utilities."""
        self.tprint_operation_start("Period Data Processing")
        
        try:
            # Track operation start
            operation_start = time.time()
            
            # Get period parameters
            start_time = self.collection_config['start_time']
            end_time = self.collection_config['end_time']
            data_types = self.collection_config['data_types']
            
            if not start_time or not end_time:
                raise ValueError("start_time and end_time are required for period processing")
            
            self.tprint_info(f"📅 Processing data for period: {start_time} to {end_time}")
            self.tprint_info(f"📊 Data types: {data_types}")
            
            # Convert to timestamps
            start_timestamp = int(start_time.timestamp() * 1000)
            end_timestamp = int(end_time.timestamp() * 1000)
            
            processing_results = {}
            quality_scores = []
            
            for data_type in data_types:
                self.tprint_operation_start(f"Processing {data_type} for period")
                
                # Process data for the period
                success, data, _ = await self._process_incremental_batch(
                    data_type, start_timestamp, end_timestamp, 10000
                )
                
                if success and data:
                    # Validate data using generalized utilities
                    is_valid, validation_errors = validate_klines_data(data)
                    if not is_valid:
                        self.tprint_warning(f"⚠️ Data validation failed: {validation_errors}")
                        continue
                    
                    # Process data using comprehensive tools
                    processed_data = await self._process_data_with_comprehensive_tools(data, data_type)
                    
                    # Validate data quality
                    df = pd.DataFrame(processed_data)
                    quality_result = validate_data_quality(df, data_type)
                    quality_scores.append(quality_result['quality_score'])
                    
                    # Save data using BaseStep utilities
                    await self._save_data_with_comprehensive_tools(processed_data, data_type)
                    
                    processing_results[data_type] = {
                        'success': True,
                        'rows': len(processed_data),
                        'quality_score': quality_result['quality_score'],
                        'start_time': start_time.isoformat(),
                        'end_time': end_time.isoformat()
                    }
                    
                    self.tprint_success(f"✅ Processed {len(processed_data)} {data_type} rows (Quality: {quality_result['quality_score']:.1f})")
                else:
                    processing_results[data_type] = {
                        'success': False,
                        'rows': 0,
                        'quality_score': 0,
                        'error': 'Processing or validation failed'
                    }
                    
                    self.tprint_error(f"❌ Failed to process {data_type} data")
                
                self.tprint_operation_end(f"Processing {data_type} for period")
            
            # Track overall operation
            operation_end = time.time()
            track_operation(
                self.performance_tracker,
                "period_processing",
                operation_start,
                operation_end,
                all(result['success'] for result in processing_results.values()),
                rows_processed=sum(result['rows'] for result in processing_results.values())
            )
            
            self.tprint_operation_end("Period Data Processing")
            return {
                'success': all(result['success'] for result in processing_results.values()),
                'processing_results': processing_results,
                'quality_scores': quality_scores,
                'artifacts': [f"{dt}_processed_data" for dt in data_types if processing_results[dt]['success']]
            }
            
        except Exception as e:
            self.tprint_error(f"❌ Period data processing failed: {e}")
            self.tprint_exception(e)
            return {'success': False, 'error': str(e)}
    
    async def _process_gap_filling(self) -> Dict[str, Any]:
        """Process gap filling using generalized utilities."""
        self.tprint_operation_start("Gap Filling Processing")
        
        try:
            # Track operation start
            operation_start = time.time()
            
            data_types = self.collection_config['data_types']
            
            self.tprint_info(f"🔍 Processing gap filling for {data_types}")
            
            gap_results = {}
            quality_scores = []
            
            for data_type in data_types:
                self.tprint_operation_start(f"Gap analysis for {data_type}")
                
                # Load existing data using BaseStep utilities
                existing_data = await self._load_existing_data(data_type)
                
                if existing_data.empty:
                    self.tprint_info(f"ℹ️ No existing {data_type} data found")
                    gap_results[data_type] = {'gaps_found': 0, 'gaps_filled': 0}
                    continue
                
                # Detect gaps using generalized utilities
                gaps = detect_gaps(existing_data, data_type)
                
                if not gaps:
                    self.tprint_info(f"✅ No gaps found in {data_type} data")
                    gap_results[data_type] = {'gaps_found': 0, 'gaps_filled': 0}
                    continue
                
                # Analyze gap patterns
                gap_analysis = analyze_gap_patterns(gaps)
                self.tprint_info(f"📊 Gap analysis: {gap_analysis['total_gaps']} gaps found")
                
                # Fill gaps
                gaps_filled = 0
                for gap in gaps:
                    self.tprint_info(f"🔄 Filling gap: {gap['start_time']} to {gap['end_time']}")
                    
                    success, data, _ = await self._process_incremental_batch(
                        data_type, gap['start_timestamp'], gap['end_timestamp'], 10000
                    )
                    
                    if success and data:
                        # Validate data using generalized utilities
                        is_valid, validation_errors = validate_klines_data(data)
                        if not is_valid:
                            self.tprint_warning(f"⚠️ Gap data validation failed: {validation_errors}")
                            continue
                        
                        # Process and save gap data
                        processed_data = await self._process_data_with_comprehensive_tools(data, data_type)
                        
                        # Validate data quality
                        df = pd.DataFrame(processed_data)
                        quality_result = validate_data_quality(df, data_type)
                        quality_scores.append(quality_result['quality_score'])
                        
                        await self._save_data_with_comprehensive_tools(processed_data, data_type, gap_id=gap['start_timestamp'])
                        gaps_filled += 1
                        self.tprint_success(f"✅ Filled gap with {len(processed_data)} rows (Quality: {quality_result['quality_score']:.1f})")
                    else:
                        self.tprint_warning(f"⚠️ Failed to fill gap")
                
                gap_results[data_type] = {
                    'gaps_found': len(gaps),
                    'gaps_filled': gaps_filled,
                    'gap_analysis': gap_analysis
                }
                
                self.tprint_operation_end(f"Gap analysis for {data_type}")
                self.tprint_success(f"✅ {data_type} gap processing: {len(gaps)} gaps found, {gaps_filled} filled")
            
            # Track overall operation
            operation_end = time.time()
            track_operation(
                self.performance_tracker,
                "gap_filling_processing",
                operation_start,
                operation_end,
                any(result['gaps_filled'] > 0 for result in gap_results.values()),
                rows_processed=sum(result['gaps_filled'] for result in gap_results.values())
            )
            
            self.tprint_operation_end("Gap Filling Processing")
            return {
                'success': any(result['gaps_filled'] > 0 for result in gap_results.values()),
                'gap_results': gap_results,
                'quality_scores': quality_scores,
                'artifacts': [f"{dt}_gap_filled_data" for dt in data_types if gap_results[dt]['gaps_filled'] > 0]
            }
            
        except Exception as e:
            self.tprint_error(f"❌ Gap filling processing failed: {e}")
            self.tprint_exception(e)
            return {'success': False, 'error': str(e)}
    
    async def _get_last_timestamp(self, data_type: str) -> Optional[int]:
        """Get the last timestamp from existing data files using BaseStep utilities."""
        try:
            # Use generalized utilities to find latest file
            latest_file = find_latest_file(
                self.collection_config['data_dir'],
                data_type,
                self.collection_config['exchange'],
                self.collection_config['symbol'],
                self.collection_config['timeframe']
            )
            
            if not latest_file:
                return None
            
            # Read using BaseStep utilities
            df = self._safe_read_parquet(latest_file)
            if df.empty or 'timestamp' not in df.columns:
                return None
            
            last_timestamp = df['timestamp'].max()
            return int(last_timestamp)
            
        except Exception as e:
            self.tprint_warning(f"⚠️ Error getting last timestamp: {e}")
            return None
    
    async def _process_incremental_batch(
        self,
        data_type: str,
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
        batch_size: int = 1000
    ) -> Tuple[bool, List[Dict[str, Any]], Optional[int]]:
        """Process incremental batch of data using comprehensive BaseStep tools."""
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
            self.tprint_error(f"❌ Error processing incremental batch: {e}")
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
                exchange_name=self.collection_config['exchange'],
                api_key="",  # Use public endpoints
                api_secret="",
                trade_symbol=self.collection_config['symbol']
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
                symbol=self.collection_config['symbol'],
                interval=self.collection_config['timeframe'],
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
            self._ensure_directory(self.collection_config['data_dir'])
            
            # Generate filename using generalized utilities
            filename = generate_filename(
                data_type,
                self.collection_config['exchange'],
                self.collection_config['symbol'],
                self.collection_config['timeframe'],
                batch_num=batch_num,
                gap_id=gap_id
            )
            
            filepath = os.path.join(self.collection_config['data_dir'], filename)
            
            # Use BaseStep utilities for saving
            self._safe_to_parquet(df, filepath, index=False)
            
            # Also save as artifact using BaseStep
            self._save_dataframe(df, f"{data_type}_processed_data")
            
            self.tprint_success(f"💾 Saved {len(data)} {data_type} rows to {filename}")
            
        except Exception as e:
            self.tprint_error(f"❌ Error saving data: {e}")
    
    async def _load_existing_data(self, data_type: str) -> pd.DataFrame:
        """Load existing data using BaseStep utilities."""
        try:
            # Use generalized utilities to find all files
            files = find_all_files(
                self.collection_config['data_dir'],
                data_type,
                self.collection_config['exchange'],
                self.collection_config['symbol'],
                self.collection_config['timeframe']
            )
            
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
    
    def _generate_processing_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive processing summary."""
        summary = {
            'exchange': self.collection_config['exchange'],
            'symbol': self.collection_config['symbol'],
            'timeframe': self.collection_config['timeframe'],
            'collection_mode': self.collection_config['collection_mode'],
            'processing_results': result.get('processing_results', {}),
            'quality_scores': result.get('quality_scores', []),
            'performance_metrics': self.performance_tracker['summary'],
            'timestamp': datetime.now().isoformat()
        }
        
        return summary

# Convenience functions for easy usage
async def process_klines_incremental(
    exchange: str,
    symbol: str,
    timeframe: str,
    data_dir: str = "historical_data",
    max_batches: int = 10,
    batch_size: int = 1000
) -> Dict[str, Any]:
    """Process klines data incrementally using the refactored pipeline."""
    config = {
        'exchange': exchange,
        'symbol': symbol,
        'timeframe': timeframe,
        'data_dir': data_dir,
        'max_batches': max_batches,
        'batch_size': batch_size,
        'collection_mode': 'incremental'
    }
    
    pipeline = RefactoredKlinesProcessingPipeline("incremental_klines_processing", config)
    return await pipeline.execute(config)

async def process_klines_for_period(
    exchange: str,
    symbol: str,
    timeframe: str,
    start_time: datetime,
    end_time: datetime,
    data_dir: str = "historical_data"
) -> Dict[str, Any]:
    """Process klines data for a specific time period using the refactored pipeline."""
    config = {
        'exchange': exchange,
        'symbol': symbol,
        'timeframe': timeframe,
        'start_time': start_time,
        'end_time': end_time,
        'data_dir': data_dir,
        'collection_mode': 'period'
    }
    
    pipeline = RefactoredKlinesProcessingPipeline("period_klines_processing", config)
    return await pipeline.execute(config)

async def process_klines_gap_filling(
    exchange: str,
    symbol: str,
    timeframe: str,
    data_dir: str = "historical_data"
) -> Dict[str, Any]:
    """Process klines gap filling using the refactored pipeline."""
    config = {
        'exchange': exchange,
        'symbol': symbol,
        'timeframe': timeframe,
        'data_dir': data_dir,
        'collection_mode': 'gap_filling'
    }
    
    pipeline = RefactoredKlinesProcessingPipeline("gap_filling_klines_processing", config)
    return await pipeline.execute(config)

if __name__ == "__main__":
    # Example usage
    async def test_refactored_pipeline():
        logger.info("🎯 Testing Refactored Klines Processing Pipeline")
        logger.info("=" * 80)
        
        # Test incremental processing
        logger.info("📊 Testing incremental processing...")
        result = await process_klines_incremental(
            exchange="BINANCE",
            symbol="ETHUSDT",
            timeframe="1m",
            max_batches=3
        )
        
        logger.info(f"✅ Incremental processing result: {result['success']}")
        
        # Test gap filling
        logger.info("🔍 Testing gap filling...")
        gap_result = await process_klines_gap_filling(
            exchange="BINANCE",
            symbol="ETHUSDT",
            timeframe="1m"
        )
        
        logger.info(f"✅ Gap filling result: {gap_result['success']}")
        
        logger.info("=" * 80)
        logger.info("🎉 Refactored klines processing pipeline tests completed!")
        logger.info("=" * 80)
    
    asyncio.run(test_refactored_pipeline())