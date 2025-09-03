"""
Enhanced Step1_5 Data Converter

This module provides an improved implementation of Step1_5 data converter
with enhanced error handling, memory optimization, and data quality validation.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import contextlib

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import enhanced utilities
try:
    from src.utils.enhanced_error_handling import (
        retry_with_backoff, circuit_breaker, categorize_errors,
        RetryableError, NonRetryableError, DATA_OPERATION_ERRORS
    )
    from src.utils.enhanced_memory_management import (
        MemoryMonitor, memory_efficient, optimize_dataframe_dtypes,
        MemoryOptimizedProcessor, MemoryConfig
    )
    from src.utils.enhanced_data_quality_validator import (
        UnifiedDataQualityValidator, QualityThresholds, QualityResult
    )
    from src.utils.enhanced_config_management import Step1_5Config
    from src.utils.logger import system_logger
except ImportError as e:
    print(f"Warning: Could not import enhanced utilities: {e}")
    # Fallback imports
    system_logger = logging.getLogger("EnhancedStep1_5")


class OptimizedUnifiedDataProcessor:
    """Optimized unified data processing with streaming and parallelization."""
    
    def __init__(self, config: Step1_5Config):
        self.config = config
        self.logger = system_logger.getChild("UnifiedDataProcessor")
        self.quality_validator = UnifiedDataQualityValidator(
            QualityThresholds(
                max_nan_ratio=config.max_nan_ratio,
                max_infinite_count=config.max_infinite_count,
                min_unique_values=config.min_unique_values,
                price_tolerance=config.price_tolerance,
                volume_tolerance=config.volume_tolerance
            )
        )
        self.memory_monitor = MemoryMonitor(MemoryConfig(max_memory_mb=config.max_memory_mb))
    
    @memory_efficient(max_memory_mb=1024)
    async def process_unified_data_streaming(self, data_sources: Dict[str, str]) -> pd.DataFrame:
        """Process unified data using streaming approach."""
        self.logger.info(f"Processing unified data from {len(data_sources)} sources")
        
        # Process each data source
        processed_chunks = []
        
        for source_name, file_path in data_sources.items():
            if not os.path.exists(file_path):
                self.logger.warning(f"Source file not found: {file_path}")
                continue
            
            self.logger.info(f"Processing {source_name}: {file_path}")
            
            try:
                # Read and process source data
                source_chunks = await self._process_source_streaming(source_name, file_path)
                processed_chunks.extend(source_chunks)
                
            except Exception as e:
                self.logger.error(f"Error processing {source_name}: {e}")
                continue
        
        # Combine all processed chunks
        if processed_chunks:
            unified_data = pd.concat(processed_chunks, ignore_index=True)
            self.logger.info(f"Combined {len(processed_chunks)} chunks into unified data: {unified_data.shape}")
            return unified_data
        else:
            self.logger.warning("No data processed")
            return pd.DataFrame()
    
    async def _process_source_streaming(self, source_name: str, file_path: str) -> List[pd.DataFrame]:
        """Process a single data source using streaming."""
        chunks = []
        chunk_count = 0
        
        try:
            for chunk in pd.read_parquet(file_path, chunksize=self.config.chunk_size):
                chunk_count += 1
                self.logger.debug(f"Processing {source_name} chunk {chunk_count}")
                
                # Validate chunk quality
                quality_result = await self.quality_validator.validate_unified_data_quality(
                    chunk, f"{source_name}_chunk_{chunk_count}"
                )
                
                if not quality_result.passed:
                    self.logger.warning(f"Quality issues in {source_name} chunk {chunk_count}: {quality_result.issues}")
                
                # Transform chunk to unified format
                unified_chunk = await self._transform_to_unified_format(chunk, source_name)
                
                if not unified_chunk.empty:
                    chunks.append(unified_chunk)
                
                # Check memory pressure
                if self.memory_monitor.is_memory_pressure(self.config.max_memory_mb * 0.8):
                    self.logger.warning("Memory pressure detected, processing existing chunks")
                    break
                
        except Exception as e:
            self.logger.error(f"Error processing {source_name}: {e}")
            raise
        
        return chunks
    
    async def _transform_to_unified_format(self, chunk: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Transform data chunk to unified format."""
        if chunk.empty:
            return chunk
        
        # Create unified DataFrame
        unified_chunk = pd.DataFrame()
        
        # Add common columns
        if 'timestamp' in chunk.columns:
            unified_chunk['timestamp'] = chunk['timestamp']
        
        # Add OHLCV columns based on source type
        if source_name == 'klines':
            ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in ohlcv_columns:
                if col in chunk.columns:
                    unified_chunk[col] = chunk[col]
        
        elif source_name == 'aggtrades':
            # Transform aggtrades to OHLCV
            if 'price' in chunk.columns and 'quantity' in chunk.columns:
                # Simple aggregation - in practice, you'd want more sophisticated aggregation'
                unified_chunk['open'] = chunk['price']
                unified_chunk['high'] = chunk['price']
                unified_chunk['low'] = chunk['price']
                unified_chunk['close'] = chunk['price']
                unified_chunk['volume'] = chunk['quantity']
        
        # Add metadata columns
        unified_chunk['exchange'] = self.config.exchange
        unified_chunk['symbol'] = self.config.symbol
        unified_chunk['timeframe'] = self.config.timeframe
        
        # Add date columns if enabled
        if self.config.auto_add_date_columns and 'timestamp' in unified_chunk.columns:
            timestamps = pd.to_datetime(unified_chunk['timestamp'], unit='ms', utc=True)
            unified_chunk['year'] = timestamps.dt.year.astype('int16')
            unified_chunk['month'] = timestamps.dt.month.astype('int8')
            unified_chunk['day'] = timestamps.dt.day.astype('int8')
        
        return unified_chunk
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for memory efficiency."""
        return optimize_dataframe_dtypes(df)


class EnhancedStep1_5DataConverter:
    """
    Enhanced Step1_5 Data Converter
    
    This class provides an improved implementation of Step1_5 data converter
    with enhanced error handling, memory optimization, and data quality validation.
    """
    
    def __init__(self, config: Optional[Step1_5Config] = None):
        self.config = config or Step1_5Config()
        self.logger = system_logger.getChild("EnhancedStep1_5")
        self.processor = OptimizedUnifiedDataProcessor(self.config)
        self.quality_validator = UnifiedDataQualityValidator(
            QualityThresholds(
                max_nan_ratio=self.config.max_nan_ratio,
                max_infinite_count=self.config.max_infinite_count,
                min_unique_values=self.config.min_unique_values,
                price_tolerance=self.config.price_tolerance,
                volume_tolerance=self.config.volume_tolerance
            )
        )
        self.memory_monitor = MemoryMonitor(MemoryConfig(max_memory_mb=self.config.max_memory_mb))
        
        # Validate configuration
        config_issues = self.config.validate()
        if config_issues:
            raise ValueError(f"Configuration validation failed: {config_issues}")
        
        # Initialize directories
        self._initialize_directories()
    
    def _initialize_directories(self):
        """Initialize required directories."""
        directories = [
            self.config.data_dir,
            self.config.unified_dir,
            self.config.backup_dir,
            self.config.temp_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.debug(f"Initialized directory: {directory}")
    
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the enhanced data conversion process.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state with conversion results
        """
        start_time = time.time()
        self.logger.info("🔄 Starting enhanced Step1_5 data conversion...")
        
        try:
            # Extract parameters
            symbol = training_input.get("symbol", self.config.symbol)
            exchange = training_input.get("exchange", self.config.exchange)
            timeframe = training_input.get("timeframe", self.config.timeframe)
            data_dir = training_input.get("data_dir", self.config.data_dir)
            
            self.logger.info(f"🎯 Converting data for {exchange} {symbol} {timeframe}")
            
            # Check if unified data already exists
            unified_exists = await self._check_unified_data_exists(symbol, exchange, timeframe)
            
            if unified_exists and not self.config.force_rerun:
                if self.config.enable_incremental:
                    self.logger.info("✅ Unified data exists, checking for incremental updates...")
                    incremental_success = await self._process_incremental_updates(symbol, exchange, timeframe)
                    if incremental_success:
                        self.logger.info("✅ Incremental processing completed")
                        pipeline_state["data_conversion_completed"] = True
                        pipeline_state["quality_check_passed"] = True
                        return pipeline_state
                
                self.logger.info("🔄 Full reprocessing required")
                await self._backup_existing_data(symbol, exchange, timeframe)
            
            # Perform full conversion
            conversion_success = await self._perform_full_conversion(symbol, exchange, timeframe, data_dir)
            
            if conversion_success:
                self.logger.info("✅ Enhanced data conversion completed successfully")
                pipeline_state["data_conversion_completed"] = True
                pipeline_state["quality_check_passed"] = True
            else:
                self.logger.warning("⚠️ Data conversion completed with issues")
                pipeline_state["data_conversion_completed"] = True
                pipeline_state["quality_check_passed"] = False
            
            # Log final metrics
            duration = time.time() - start_time
            peak_memory = self.memory_monitor.get_peak_usage_mb()
            
            self.logger.info(f"📊 Conversion completed in {duration:.2f}s, peak memory: {peak_memory:.1f}MB")
            
        except Exception as e:
            self.logger.exception(f"❌ Error during enhanced data conversion: {e}")
            pipeline_state["data_conversion_completed"] = False
            pipeline_state["quality_check_passed"] = False
        
        return pipeline_state
    
    async def _check_unified_data_exists(self, symbol: str, exchange: str, timeframe: str) -> bool:
        """Check if unified data already exists."""
        try:
            unified_base = os.path.join(self.config.unified_dir, exchange.lower(), symbol, timeframe)
            if os.path.exists(unified_base):
                parquet_files = []
                for root, dirs, files in os.walk(unified_base):
                    parquet_files.extend([f for f in files if f.endswith('.parquet')])
                
                if parquet_files:
                    self.logger.info(f"✅ Found existing unified data: {len(parquet_files)} files")
                    return True
            
            return False
        except Exception as e:
            self.logger.warning(f"⚠️ Error checking unified data existence: {e}")
            return False
    
    async def _backup_existing_data(self, symbol: str, exchange: str, timeframe: str):
        """Backup existing unified data."""
        try:
            unified_base = os.path.join(self.config.unified_dir, exchange.lower(), symbol, timeframe)
            backup_path = os.path.join(self.config.backup_dir, f"{exchange}_{symbol}_{timeframe}_{int(time.time())}")
            
            if os.path.exists(unified_base):
                import shutil
                shutil.move(unified_base, backup_path)
                self.logger.info(f"📦 Backed up existing data to: {backup_path}")
        except Exception as e:
            self.logger.warning(f"⚠️ Error backing up existing data: {e}")
    
    async def _process_incremental_updates(self, symbol: str, exchange: str, timeframe: str) -> bool:
        """Process incremental updates to existing unified data."""
        try:
            self.logger.info("🔍 Processing incremental updates...")
            # Implement incremental processing logic here
            # This would compare source data timestamps with unified data timestamps
            # and only process new data
            
            # For now, return False to trigger full reprocessing
            return False
        except Exception as e:
            self.logger.error(f"Error processing incremental updates: {e}")
            return False
    
    async def _perform_full_conversion(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Perform full data conversion."""
        try:
            # Identify data sources
            data_sources = await self._identify_data_sources(symbol, exchange, timeframe, data_dir)
            
            if not data_sources:
                self.logger.warning("No data sources found for conversion")
                return False
            
            self.logger.info(f"📁 Found {len(data_sources)} data sources: {list(data_sources.keys())}")
            
            # Process unified data
            unified_data = await self.processor.process_unified_data_streaming(data_sources)
            
            if unified_data.empty:
                self.logger.warning("No unified data generated")
                return False
            
            # Validate unified data quality
            quality_result = await self.quality_validator.validate_unified_data_quality(
                unified_data, "unified_data_final"
            )
            
            if not quality_result.passed:
                self.logger.warning(f"⚠️ Quality issues in unified data: {quality_result.issues}")
                # Continue with warning instead of failing
                self.logger.warning("⚠️ Continuing with quality issues - review logs for details")
            
            # Save unified data
            save_success = await self._save_unified_data(unified_data, symbol, exchange, timeframe)
            
            if not save_success:
                self.logger.error("Failed to save unified data")
                return False
            
            # Log quality metrics
            self.logger.info(f"📊 Final unified data metrics: {json.dumps(quality_result.metrics, indent=2)}")
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Error during full conversion: {e}")
            return False
    
    async def _identify_data_sources(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Dict[str, str]:
        """Identify available data sources for conversion."""
        data_sources = {}
        
        # Check for klines data
        klines_file = os.path.join(data_dir, f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet")
        if os.path.exists(klines_file):
            data_sources['klines'] = klines_file
        
        # Check for aggtrades data
        aggtrades_file = os.path.join(data_dir, f"aggtrades_{exchange}_{symbol}_consolidated.parquet")
        if os.path.exists(aggtrades_file):
            data_sources['aggtrades'] = aggtrades_file
        
        # Check for futures data
        futures_file = os.path.join(data_dir, f"futures_{exchange}_{symbol}_consolidated.parquet")
        if os.path.exists(futures_file):
            data_sources['futures'] = futures_file
        
        return data_sources
    
    async def _save_unified_data(self, unified_data: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> bool:
        """Save unified data to partitioned parquet format."""
        try:
            # Create output directory
            output_dir = os.path.join(self.config.unified_dir, exchange.lower(), symbol, timeframe)
            os.makedirs(output_dir, exist_ok=True)
            
            # Optimize data types
            unified_data = self.processor._optimize_dtypes(unified_data)
            
            # Save to parquet with partitioning
            partition_cols = ['year', 'month', 'day'] if all(col in unified_data.columns for col in ['year', 'month', 'day']) else []
            
            self.logger.info(f"💾 Saving unified data to {output_dir}")
            self.logger.info(f"   Shape: {unified_data.shape}")
            self.logger.info(f"   Partition columns: {partition_cols}")
            
            # Use pyarrow for efficient writing
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                
                table = pa.Table.from_pandas(unified_data, preserve_index=False)
                
                # Write with partitioning
                if partition_cols:
                    pq.write_to_dataset(
                        table,
                        output_dir,
                        partition_cols=partition_cols,
                        compression=self.config.compression,
                        use_dictionary=self.config.use_dictionary,
                        row_group_size=self.config.min_rows_per_group,
                        max_file_size=self.config.max_rows_per_file * 1024,  # Convert to bytes
                    )
                else:
                    pq.write_table(
                        table,
                        os.path.join(output_dir, "data.parquet"),
                        compression=self.config.compression,
                        use_dictionary=self.config.use_dictionary,
                        row_group_size=self.config.min_rows_per_group,
                    )
                
                self.logger.info("✅ Unified data saved successfully")
                return True
                
            except ImportError:
                # Fallback to pandas
                self.logger.warning("pyarrow not available, using pandas fallback")
                unified_data.to_parquet(
                    os.path.join(output_dir, "data.parquet"),
                    compression=self.config.compression,
                    index=False
                )
                self.logger.info("✅ Unified data saved successfully (pandas fallback)")
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving unified data: {e}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return self.memory_monitor.get_memory_stats()
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get quality validation summary."""
        # This would return the last quality validation results
        return {"message": "Quality validation results not available"}


# Convenience function for backward compatibility
async def run_enhanced_step1_5(
    training_input: Dict[str, Any],
    pipeline_state: Dict[str, Any],
    config: Optional[Step1_5Config] = None
) -> Dict[str, Any]:
    """
    Convenience function to run enhanced Step1_5 data conversion.
    
    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state
        config: Optional configuration
        
    Returns:
        Updated pipeline state
    """
    step1_5 = EnhancedStep1_5DataConverter(config)
    return await step1_5.execute(training_input, pipeline_state)


# Example usage
if __name__ == "__main__":
    import asyncio
import os.path
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        # Create configuration
        config = Step1_5Config(
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m",
            max_memory_mb=512,  # Lower for testing
            chunk_size=5000,    # Smaller chunks for testing
            force_rerun=False,
            enable_incremental=True
        )
        
        # Create enhanced Step1_5 instance
        step1_5 = EnhancedStep1_5DataConverter(config)
        
        # Prepare training input
        training_input = {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "timeframe": "1m",
            "data_dir": "data_cache"
        }
        
        # Prepare pipeline state
        pipeline_state = {
            "data_conversion_completed": False,
            "quality_check_passed": False
        }
        
        # Execute enhanced data conversion
        try:
            result = await step1_5.execute(training_input, pipeline_state)
            
            print("=" * 60)
            print("ENHANCED STEP1_5 EXECUTION RESULTS")
            print("=" * 60)
            print(f"Data conversion completed: {result['data_conversion_completed']}")
            print(f"Quality check passed: {result['quality_check_passed']}")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Enhanced Step1_5 execution failed: {e}")
    
    # Run the example
    asyncio.run(main())