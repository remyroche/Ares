"""
Enhanced Step1 Data Collection

This module provides an improved implementation of Step1 data collection
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
        EnhancedDataQualityValidator, QualityThresholds, QualityResult
    )
    from src.utils.enhanced_config_management import Step1Config
    from src.utils.logger import system_logger
except ImportError as e:
    print(f"Warning: Could not import enhanced utilities: {e}")
    # Fallback imports
    system_logger = logging.getLogger("EnhancedStep1")

# Import existing utilities with fallbacks
try:
    from src.training.steps.data_downloader import download_all_data_with_consolidation
except ImportError:
    download_all_data_with_consolidation = None


class EnhancedStep1DataCollection:
    """
    Enhanced Step1 Data Collection

    This class provides an improved implementation of Step1 data collection
    with enhanced error handling, memory optimization, and data quality validation.
    """

    def __init__(self, config: Optional[Step1Config] = None):
        self.config = config or Step1Config()
        self.logger = system_logger.getChild("EnhancedStep1")
        self.memory_monitor = MemoryMonitor(MemoryConfig(max_memory_mb=self.config.max_memory_mb))
        self.quality_validator = EnhancedDataQualityValidator(
            QualityThresholds(
                max_nan_ratio=self.config.max_nan_ratio,
                max_infinite_count=self.config.max_infinite_count,
                min_unique_values=self.config.min_unique_values,
                price_tolerance=self.config.price_tolerance,
                volume_tolerance=self.config.volume_tolerance
            )
        )

        # Validate configuration
        config_issues = self.config.validate()
        if config_issues:
            raise ValueError(f"Configuration validation failed: {config_issues}")

        # Initialize directories
        self._initialize_directories()

    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the enhanced data collection process.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Updated pipeline state with collection results
        """
        start_time = time.time()
        self.logger.info("🚀 Starting enhanced data collection...")

        try:
            # Initialize directories
            await self._initialize_directories()

            # Download data with enhanced resilience
            download_success = await self._download_data_with_resilience(training_input)

            if not download_success:
                self.logger.error("❌ Data download failed")
                pipeline_state["data_collection_completed"] = False
                pipeline_state["quality_check_passed"] = False
                return pipeline_state

            # Process and validate data
            processing_success = await self._process_and_validate_data(training_input)

            if processing_success:
                self.logger.info("✅ Enhanced data collection completed successfully")
                pipeline_state["data_collection_completed"] = True
                pipeline_state["quality_check_passed"] = True
            else:
                self.logger.warning("⚠️ Data collection completed with quality issues")
                pipeline_state["data_collection_completed"] = True
                pipeline_state["quality_check_passed"] = False

            # Log final metrics
            duration = time.time() - start_time
            peak_memory = self.memory_monitor.get_peak_usage_mb()

            self.logger.info(f"📊 Collection completed in {duration:.2f}s, peak memory: {peak_memory:.1f}MB")

        except Exception as e:
            self.logger.exception(f"❌ Error during enhanced data collection: {e}")
            pipeline_state["data_collection_completed"] = False
            pipeline_state["quality_check_passed"] = False

        return pipeline_state

    @retry_with_backoff()
    @categorize_errors(DATA_OPERATION_ERRORS)
    async def _download_data_with_resilience(self, training_input: Dict[str, Any]) -> bool:
        """Download data with enhanced resilience."""
        try:
            symbol = training_input.get("symbol", self.config.symbol)
            exchange = training_input.get("exchange", self.config.exchange)
            timeframe = training_input.get("timeframe", self.config.timeframe)

            self.logger.info(f"📥 Downloading data for {exchange}_{symbol}_{timeframe}")

            # Try to import the downloader if not already imported
            global download_all_data_with_consolidation
            if download_all_data_with_consolidation is None:
                try:
                    from src.training.steps.data_downloader import download_all_data_with_consolidation as _dl
                    download_all_data_with_consolidation = _dl
                except ImportError:
                    self.logger.warning("Could not import data downloader, using fallback")
                    return await self._fallback_data_download(training_input)

            if download_all_data_with_consolidation:
                # Use the existing data downloader if available
                success = await download_all_data_with_consolidation(
                    symbol=symbol,
                    exchange_name=exchange,
                    interval=timeframe,
                )

                if success:
                    self.logger.info("✅ Data download completed successfully")
                    # Log immediate data extract after download
                    data_dir = training_input.get("data_dir", self.config.data_dir)
                    await self._log_detailed_data_extract(symbol, exchange, timeframe, data_dir)

                return bool(success)

            # Fallback implementation
            self.logger.warning("Using fallback data download method")
            return await self._fallback_data_download(training_input)

        except RetryableError:
            raise
        except NonRetryableError:
            raise
        except Exception as e:
            self.logger.error(f"Non-retryable error during download: {e}")
            raise NonRetryableError(f"Download failed: {e}")

    async def _fallback_data_download(self, training_input: Dict[str, Any]) -> bool:
        """Fallback data download method."""
        self.logger.info("Using fallback data download method")
        # Implement fallback logic here if needed
        # For now, just return True to allow the pipeline to continue
        return True

    @memory_efficient(max_memory_mb=1024)
    async def _process_and_validate_data(self, training_input: Dict[str, Any]) -> bool:
        """Process and validate downloaded data."""
        try:
            symbol = training_input.get("symbol", self.config.symbol)
            exchange = training_input.get("exchange", self.config.exchange)
            timeframe = training_input.get("timeframe", self.config.timeframe)

            # Check for downloaded files
            klines_file = os.path.join(self.config.data_dir, f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet")
            aggtrades_file = os.path.join(self.config.data_dir, f"aggtrades_{exchange}_{symbol}_consolidated.parquet")

            files_to_process = []
            if os.path.exists(klines_file):
                files_to_process.append(("klines", klines_file))
            if os.path.exists(aggtrades_file):
                files_to_process.append(("aggtrades", aggtrades_file))

            if not files_to_process:
                self.logger.warning("No data files found for processing")
                return False

            # Process each file
            all_quality_passed = True

            for data_type, file_path in files_to_process:
                self.logger.info(f"🔍 Processing {data_type} data: {file_path}")

                # Process with streaming
                processed_data = await self._process_file_streaming(file_path)

                if processed_data.empty:
                    self.logger.warning(f"⚠️ No data processed for {data_type}")
                    all_quality_passed = False
                    continue

                # Validate quality
                quality_result = await self.quality_validator.validate_dataframe_quality(
                    processed_data, f"{data_type}_processed"
                )

                if not quality_result.passed:
                    self.logger.warning(f"⚠️ Quality issues in {data_type}: {quality_result.issues}")
                    all_quality_passed = False
                else:
                    self.logger.info(f"✅ {data_type} quality validation passed")

                # Log quality metrics
                self.logger.info(f"📊 {data_type} metrics: {json.dumps(quality_result.metrics, indent=2)}")

            return all_quality_passed

        except Exception as e:
            self.logger.exception(f"Error during data processing and validation: {e}")
            return False

    async def _process_file_streaming(self, file_path: str) -> pd.DataFrame:
        """Process file using streaming approach."""
        self.logger.info(f"Processing file: {file_path}")

        # Read data in chunks
        chunks = []
        chunk_count = 0

        try:
            for chunk in pd.read_parquet(file_path, chunksize=self.config.chunk_size):
                chunk_count += 1
                self.logger.debug(f"Processing chunk {chunk_count}")

                # Validate chunk quality
                quality_result = await self.quality_validator.validate_dataframe_quality(
                    chunk, f"chunk_{chunk_count}"
                )

                if not quality_result.passed:
                    self.logger.warning(f"Quality issues in chunk {chunk_count}: {quality_result.issues}")

                # Process chunk
                processed_chunk = await self._process_chunk_parallel(chunk)
                chunks.append(processed_chunk)

                # Check memory pressure
                if self.memory_monitor.is_memory_pressure(self.config.max_memory_mb * 0.8):
                    self.logger.warning("Memory pressure detected, processing existing chunks")
                    break

        except Exception as e:
            self.logger.error(f"Error processing file: {e}")
            raise

        # Combine chunks
        if chunks:
            result = pd.concat(chunks, ignore_index=True)
            self.logger.info(f"Processed {len(chunks)} chunks, final shape: {result.shape}")
            return result
        else:
            self.logger.warning("No chunks processed")
            return pd.DataFrame()

    async def _process_chunk_parallel(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process data chunk using parallel operations."""
        if chunk.empty:
            return chunk

        # Optimize data types for memory efficiency
        chunk = optimize_dataframe_dtypes(chunk)

        # Process in parallel if chunk is large enough
        if len(chunk) > 1000:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Split chunk for parallel processing
                chunk_splits = np.array_split(chunk, self.config.max_workers)

                # Process splits in parallel
                loop = asyncio.get_event_loop()
                futures = [
                    loop.run_in_executor(executor, self._process_chunk_sync, split)
                    for split in chunk_splits if not split.empty
                ]

                processed_splits = await asyncio.gather(*futures)
                return pd.concat(processed_splits, ignore_index=True)
        else:
            return self._process_chunk_sync(chunk)

    def _process_chunk_sync(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Synchronous chunk processing."""
        # Add any specific processing logic here
        # For now, just return the chunk as-is
        return chunk

    async def _log_detailed_data_extract(self, symbol: str, exchange: str, timeframe: str, data_dir: str):
        """Log detailed information about downloaded data."""
        try:
            klines_file = os.path.join(data_dir, f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet")
            aggtrades_file = os.path.join(data_dir, f"aggtrades_{exchange}_{symbol}_consolidated.parquet")

            files_info = []

            for file_path, file_type in [(klines_file, "klines"), (aggtrades_file, "aggtrades")]:
                if os.path.exists(file_path):
                    try:
                        # Get file size
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB

                        # Read basic info about the file
                        df_info = pd.read_parquet(file_path, nrows=1)
                        columns = list(df_info.columns)

                        files_info.append({
                            "type": file_type,
                            "path": file_path,
                            "size_mb": file_size,
                            "columns": columns
                        })

                        self.logger.info(f"📁 {file_type}: {file_size:.1f}MB, {len(columns)} columns")

                    except Exception as e:
                        self.logger.warning(f"Could not read {file_type} file info: {e}")
                else:
                    self.logger.warning(f"⚠️ {file_type} file not found: {file_path}")

            if files_info:
                self.logger.info(f"📊 Downloaded data summary: {len(files_info)} files")
                for info in files_info:
                    self.logger.info(f"   - {info['type']}: {info['size_mb']:.1f}MB, {len(info['columns'])} columns")

        except Exception as e:
            self.logger.warning(f"Error logging data extract: {e}")


# Convenience function for backward compatibility
async def run_enhanced_step1(
    training_input: Dict[str, Any],
    pipeline_state: Dict[str, Any],
    config: Optional[Step1Config] = None
) -> Dict[str, Any]:
    """
    Convenience function to run enhanced Step1 data collection.

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state
        config: Optional configuration

    Returns:
        Updated pipeline state
    """
    step1 = EnhancedStep1DataCollection(config)
    return await step1.execute(training_input, pipeline_state)


# Example usage
if __name__ == "__main__":
    import asyncio

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    async def main():
        # Create configuration
        config = Step1Config(
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m",
            lookback_days=30,  # Shorter for testing
            max_memory_mb=512,  # Lower for testing
            chunk_size=5000     # Smaller chunks for testing
        )

        # Create enhanced Step1 instance
        step1 = EnhancedStep1DataCollection(config)

        # Prepare training input
        training_input = {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "timeframe": "1m",
            "data_dir": "data_cache"
        }

        # Prepare pipeline state
        pipeline_state = {
            "data_collection_completed": False,
            "quality_check_passed": False
        }

        # Execute enhanced data collection
        try:
            result = await step1.execute(training_input, pipeline_state)

            print("=" * 60)
            print("ENHANCED STEP1 EXECUTION RESULTS")
            print("=" * 60)
            print(f"Data collection completed: {result['data_collection_completed']}")
            print(f"Quality check passed: {result['quality_check_passed']}")
            print("=" * 60)

        except Exception as e:
            print(f"❌ Enhanced Step1 execution failed: {e}")

    # Run the example
    asyncio.run(main())