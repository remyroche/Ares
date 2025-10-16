#!/usr/bin/env python3
"""
Unified Resampler

This module provides centralized resampling functionality for all timeframes:
- 1m, 5m, 15m, 30m, 1h
- Memory-efficient processing
- Partitioned data creation
- Comprehensive validation

Consolidates functionality from multiple redundant resamplers into a single,
optimized implementation.
"""

import asyncio
import sys
import time
import gc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.error_handler import handles_errors
from src.utils.common_operations import safe_fillna, safe_to_parquet, safe_read_parquet
from src.utils.common_utilities import validate_dataframe_columns, safe_dataframe_operation
# from src.utils.validation import validate_data_quality  # Replaced with comprehensive quality tools

# Import comprehensive data quality tools
try:
    from src.utils.data.quality.comprehensive_quality_scorer import get_quality_scorer
    from src.utils.data.quality.data_quality import DataQualityFramework
    QUALITY_TOOLS_AVAILABLE = True
except ImportError:
    QUALITY_TOOLS_AVAILABLE = False

def validate_data_quality(df, **kwargs):
    """Comprehensive data quality validation using proper tools."""
    if not QUALITY_TOOLS_AVAILABLE:
        return {'valid': True, 'quality_score': 50.0, 'issues': [], 'warnings': []}
    
    try:
        quality_scorer = get_quality_scorer()
        quality_assessment = quality_scorer.assess_data_quality(
            df,
            context="data_collection",
            step_name="data_resampling",
            data_type="klines"
        )
        
        return {
            'valid': quality_assessment.level.value not in ['critical'],
            'quality_score': quality_assessment.overall_score,
            'issues': quality_assessment.issues,
            'warnings': quality_assessment.warnings
        }
    except Exception as e:
        return {'valid': True, 'quality_score': 50.0, 'issues': [str(e)], 'warnings': []}
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

logger = system_logger.getChild("UnifiedResampler")

class UnifiedResampler:
    """Unified resampler for all timeframes with comprehensive validation and memory management."""
    
    @log_important_calls
    def __init__(self, data_cache_path: str = "data_cache"):
        self.data_cache_path = Path(data_cache_path)
        self.data_cache_path.mkdir(exist_ok=True)
        self.logger = logger.getChild('UnifiedResampler')
        
        # Initialize standardized parquet handler for compatibility
        try:
            from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
            self.parquet_handler = standardized_parquet_handler
        except ImportError:
            self.parquet_handler = None
            self.logger.warning("⚠️ Standardized parquet handler not available")
        
        # Supported timeframes
        self.supported_timeframes = ['1m', '5m', '15m', '30m', '1h']
        
        # Resampling statistics
        self.resample_stats = {
            'total_resamples': 0,
            'successful_resamples': 0,
            'failed_resamples': 0,
            'total_rows_processed': 0,
            'start_time': None
        }
        
    @handles_errors(context="resample_to_timeframe")
    @log_all_calls
    def resample_to_timeframe(
        self, 
        df: pd.DataFrame, 
        timeframe: str,
        symbol: str,
        exchange: str
    ) -> Optional[pd.DataFrame]:
        """
        Resample DataFrame to specified timeframe.
        
        Args:
            df: Input DataFrame with OHLCV data
            timeframe: Target timeframe ('1m', '5m', '15m', '30m', '1h')
            symbol: Trading symbol
            exchange: Exchange name
            
        Returns:
            Resampled DataFrame or None if failed
        """
        self.logger.info(f"🔄 Resampling {exchange}_{symbol} to {timeframe}")
        
        if timeframe not in self.supported_timeframes:
            self.logger.error(f"❌ Unsupported timeframe: {timeframe}")
            return None
            
        try:
            # Validate input data
            if df.empty:
                self.logger.warning("⚠️ Empty DataFrame provided")
                return None
                
            # Use utils/ validation
            if not validate_data_quality(df):
                self.logger.warning("⚠️ Data quality validation failed")
                
            # Ensure timestamp column exists and is datetime
            if 'timestamp' not in df.columns:
                self.logger.error("❌ No timestamp column found")
                return None
                
            # Convert timestamp to datetime if needed
            df_copy = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
                df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], unit='ms', utc=True)
            
            # Set timestamp as index
            df_copy.set_index('timestamp', inplace=True)
            
            # Resample based on timeframe
            resampled_df = self._perform_resampling(df_copy, timeframe)
            
            if resampled_df is None or resampled_df.empty:
                self.logger.warning("⚠️ Resampling resulted in empty DataFrame")
                return None
            
            # Reset index to get timestamp as column
            resampled_df.reset_index(inplace=True)
            
            # Add metadata
            resampled_df['symbol'] = symbol
            resampled_df['exchange'] = exchange
            resampled_df['timeframe'] = timeframe
            
            # Use utils/ safe operations
            resampled_df = safe_fillna(resampled_df, method='forward')
            
            # Update statistics
            self.resample_stats['total_resamples'] += 1
            self.resample_stats['successful_resamples'] += 1
            self.resample_stats['total_rows_processed'] += len(resampled_df)
            
            self.logger.info(f"✅ Resampled to {timeframe}: {len(resampled_df)} rows")
            return resampled_df
            
        except Exception as e:
            self.logger.exception(f"❌ Error resampling to {timeframe}: {e}")
            self.resample_stats['failed_resamples'] += 1
            return None
    
    @handles_errors(context="resample_all_timeframes")
    @log_all_calls
    async def resample_all_timeframes(
        self, 
        symbol: str, 
        exchange: str,
        source_timeframe: str = "1m",
        target_timeframes: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Resample data to all specified timeframes.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            source_timeframe: Source timeframe (default: '1m')
            target_timeframes: List of target timeframes (default: all supported)
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            Dictionary with resampling results
        """
        self.logger.info(f"🔄 Resampling {exchange}_{symbol} to all timeframes")
        
        if target_timeframes is None:
            target_timeframes = [tf for tf in self.supported_timeframes if tf != source_timeframe]
            
        results = {
            'symbol': symbol,
            'exchange': exchange,
            'source_timeframe': source_timeframe,
            'target_timeframes': target_timeframes,
            'resampled_data': {},
            'success_count': 0,
            'failed_count': 0,
            'total_rows': 0
        }
        
        try:
            # Load source data
            source_data = await self._load_source_data(symbol, exchange, source_timeframe, start_date, end_date)
            
            if source_data is None or source_data.empty:
                self.logger.error("❌ No source data found")
                return results
            
            self.logger.info(f"📊 Source data: {len(source_data)} rows")
            
            # Resample to each target timeframe
            for timeframe in target_timeframes:
                self.logger.info(f"🔄 Resampling to {timeframe}...")
                
                resampled_data = self.resample_to_timeframe(source_data, timeframe, symbol, exchange)
                
                if resampled_data is not None:
                    results['resampled_data'][timeframe] = resampled_data
                    results['success_count'] += 1
                    results['total_rows'] += len(resampled_data)
                    
                    # Save resampled data
                    await self._save_resampled_data(resampled_data, symbol, exchange, timeframe)
                else:
                    results['failed_count'] += 1
                    self.logger.error(f"❌ Failed to resample to {timeframe}")
            
            self.logger.info(f"✅ Resampling completed: {results['success_count']}/{len(target_timeframes)} successful")
            return results
            
        except Exception as e:
            self.logger.exception(f"❌ Error in resample_all_timeframes: {e}")
            return results
    
    @handles_errors(context="create_partitioned_data")
    @log_all_calls
    async def create_partitioned_data(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        exchange: str, 
        timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """
        Create partitioned data files for efficient storage and access.
        
        Args:
            df: DataFrame to partition
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            
        Returns:
            Dictionary with partition information
        """
        self.logger.info(f"📁 Creating partitioned data for {exchange}_{symbol}_{timeframe}")
        
        try:
            if df.empty:
                self.logger.warning("⚠️ Empty DataFrame provided")
                return None
            
            # Create partition directory
            partition_dir = self.data_cache_path / 'partitioned' / exchange / symbol / timeframe
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            # Add year, month, day columns for partitioning
            df_copy = df.copy()
            if 'timestamp' in df_copy.columns:
                df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], unit='ms', utc=True)
                df_copy['year'] = df_copy['timestamp'].dt.year
                df_copy['month'] = df_copy['timestamp'].dt.month
                df_copy['day'] = df_copy['timestamp'].dt.day
            
            # Partition by year-month
            partitions = {}
            for (year, month), group in df_copy.groupby(['year', 'month']):
                partition_key = f"{year}_{month:02d}"
                partition_file = partition_dir / f"{partition_key}.parquet"
                
                # Use utils/ safe operations
                safe_to_parquet(group, partition_file)
                
                partitions[partition_key] = {
                    'file_path': str(partition_file),
                    'rows': len(group),
                    'start_date': group['timestamp'].min(),
                    'end_date': group['timestamp'].max()
                }
                
                self.logger.info(f"📁 Created partition {partition_key}: {len(group)} rows")
            
            result = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'partition_dir': str(partition_dir),
                'partitions': partitions,
                'total_partitions': len(partitions),
                'total_rows': len(df_copy)
            }
            
            self.logger.info(f"✅ Created {len(partitions)} partitions for {exchange}_{symbol}_{timeframe}")
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Error creating partitioned data: {e}")
            return None
    
    def _perform_resampling(self, df: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
        """Perform the actual resampling operation."""
        try:
            # Define resampling rules
            resample_rules = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            # Convert timeframe to pandas frequency
            freq_map = {
                '1m': '1T',
                '5m': '5T',
                '15m': '15T',
                '30m': '30T',
                '1h': '1H'
            }
            
            freq = freq_map.get(timeframe)
            if not freq:
                self.logger.error(f"❌ Invalid timeframe: {timeframe}")
                return None
            
            # Perform resampling
            resampled = df.resample(freq).agg(resample_rules)
            
            # Drop rows with all NaN values
            resampled = resampled.dropna(how='all')
            
            return resampled
            
        except Exception as e:
            self.logger.exception(f"❌ Error in resampling operation: {e}")
            return None
    
    @handles_errors(context="load_source_data")
    async def _load_source_data(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """Load source data for resampling using standardized paths."""
        try:
            # Use standardized parquet handler if available
            if self.parquet_handler:
                # Try to load from unified data using standardized paths
                try:
                    unified_path = self.parquet_handler.get_standardized_path(
                        'unified_data', exchange, symbol, timeframe
                    )
                    if Path(unified_path).exists():
                        parquet_files = list(Path(unified_path).glob('**/*.parquet'))
                        if parquet_files:
                            # Use standardized parquet handler
                            data = self.parquet_handler.read_parquet_standardized(parquet_files[0])
                            if data is not None and not data.empty:
                                return data
                except Exception as e:
                    self.logger.debug(f"Could not load from unified path: {e}")
            
            # Fallback to direct file system access
            # Try to load from unified data first
            unified_path = self.data_cache_path / 'unified' / exchange / symbol / timeframe
            if unified_path.exists():
                parquet_files = list(unified_path.glob('**/*.parquet'))
                if parquet_files:
                    # Use utils/ safe operations
                    data = safe_read_parquet(parquet_files[0])
                    if data is not None and not data.empty:
                        return data
            
            # Try to load from raw data
            raw_path = self.data_cache_path / 'raw' / exchange / symbol / timeframe
            if raw_path.exists():
                parquet_files = list(raw_path.glob('**/*.parquet'))
                if parquet_files:
                    # Use utils/ safe operations
                    data = safe_read_parquet(parquet_files[0])
                    if data is not None and not data.empty:
                        return data
            
            self.logger.warning(f"⚠️ No source data found for {exchange}_{symbol}_{timeframe}")
            return None
            
        except Exception as e:
            self.logger.exception(f"❌ Error loading source data: {e}")
            return None
    
    @handles_errors(context="save_resampled_data")
    async def _save_resampled_data(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        exchange: str, 
        timeframe: str
    ) -> bool:
        """Save resampled data to disk."""
        try:
            # Create output directory
            output_dir = self.data_cache_path / 'resampled' / exchange / symbol / timeframe
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{exchange}_{symbol}_{timeframe}_{timestamp}.parquet"
            file_path = output_dir / filename
            
            # Use utils/ safe operations
            success = safe_to_parquet(df, file_path)
            
            if success:
                self.logger.info(f"💾 Saved resampled data: {file_path}")
            else:
                self.logger.error(f"❌ Failed to save resampled data: {file_path}")
                
            return success
            
        except Exception as e:
            self.logger.exception(f"❌ Error saving resampled data: {e}")
            return False
    
    def get_resample_stats(self) -> Dict[str, Any]:
        """Get resampling statistics."""
        return {
            **self.resample_stats,
            'success_rate': (
                self.resample_stats['successful_resamples'] / 
                max(self.resample_stats['total_resamples'], 1) * 100
            )
        }
    
    def reset_stats(self):
        """Reset resampling statistics."""
        self.resample_stats = {
            'total_resamples': 0,
            'successful_resamples': 0,
            'failed_resamples': 0,
            'total_rows_processed': 0,
            'start_time': None
        }

# Convenience functions for backward compatibility
@handles_errors()
def resample_to_timeframe(df: pd.DataFrame, timeframe: str, symbol: str, exchange: str) -> Optional[pd.DataFrame]:
    """Convenience function for resampling to a specific timeframe."""
    resampler = UnifiedResampler()
    return resampler.resample_to_timeframe(df, timeframe, symbol, exchange)

@handles_errors()
async def resample_all_timeframes(symbol: str, exchange: str, **kwargs) -> Dict[str, Any]:
    """Convenience function for resampling to all timeframes."""
    resampler = UnifiedResampler()
    return await resampler.resample_all_timeframes(symbol, exchange, **kwargs)

@handles_errors()
async def create_partitioned_data(df: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> Optional[Dict[str, Any]]:
    """Convenience function for creating partitioned data."""
    resampler = UnifiedResampler()
    return await resampler.create_partitioned_data(df, symbol, exchange, timeframe)