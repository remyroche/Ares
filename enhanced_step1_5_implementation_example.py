#!/usr/bin/env python3
"""
Enhanced Step1_5 Data Converter Implementation Example

This file demonstrates the improved implementation of Step1_5 data converter
with enhanced error handling, memory optimization, and data quality validation.
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
import functools

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import existing utilities with fallbacks
try:
    passpassfrom src.utils.logger import system_logger
except ImportError:
    passpasssystem_logger = logging.getLogger("EnhancedStep1_5")

try:
    passself.logger.info("Implementation placeholder - needs specific logic")
except ImportError:
    passpassdef handle_errors(...):
    passdef decorator(...):
    passreturn func
        return decorator


# ============================================================================
# Enhanced Configuration Management
# ============================================================================

@dataclass
class Step1_5Config:

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="step1_5config initialization",
    )
    async def initialize(self) -> bool:
        """Initialize Step1_5Config."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    pass"""Enhanced configuration for Step1_5 data converter."""

    # Basic parameters
    symbol: str = "ETHUSDT"
    exchange: str = "BINANCE"
    timeframe: str = "1m"

    # Performance parameters
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    chunk_size: int = 10000
    max_memory_mb: int = 1024
    max_workers: int = 4
    batch_size: int = 262144

    # Quality thresholds
    max_nan_ratio: float = 0.0  # Zero tolerance for NaN
    max_infinite_count: int = 0  # Zero tolerance for infinite values
    min_unique_values: int = 2
    max_gap_hours: int = 48
    price_tolerance: float = 0.001
    volume_tolerance: float = 0.001

    # Data directories
    data_dir: str = "data_cache"
    unified_dir: str = "data_cache/unified"
    backup_dir: str = "data_cache/backup_pre_unified"
    temp_dir: str = "data_cache/temp"

    # Processing options
    force_rerun: bool = False
    enable_incremental: bool = True
    auto_add_date_columns: bool = True
    compression: str = "snappy"
    use_dictionary: bool = True
    min_rows_per_group: int = 50000
    max_rows_per_file: int = 5_000_000

    def validate(...) -> ...:
    """..."""
    passissues = []

        if self.chunk_size <= 0:
    passissues.append("chunk_size must be positive")
        if self.max_memory_mb <= 0:
    passissues.append("max_memory_mb must be positive")
        if self.max_retries < 0:
    passissu
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="retryableerror initialization",
    )
    async def initialize(self) -> bool:
        """Initialize RetryableError."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
es.append("max_retries must be non-negative")
        if self.max_nan_ratio < 0 or self.max_nan_ratio > 1:
    passissues.append("max_nan_ratio must be between 0 and 1")
        if self.min_rows_per_group >= self.max_rows_per_file:
    passissues.append("min_rows_per_group must be less than max_rows_per_file")

        return issues


# ============================================================================
# Enhanced Error Handling and Resilience
# ============================================================================

class RetryableError(...):
    """..."""
    passpass

class NonRetryableError(...):
    """..."""
    passpass

def retry_with_backoff(...):
    pass"""Decorator for retrying operations with exponential backoff."""
    def decorator(...):
    passpasspass@functools.wraps(func)
        async def wrapper(...):
    passlast_exception = None

            for attempt in range(max_retries + 1):
    passtry:
    passreturn await func(*args, **kwargs)
                except RetryableError as e:
    passpasspasspasspasspasspasslast_exception = e
                    if attempt < max_retries:
    passwait_time = backoff_factor ** attempt
                        logging.warning(f"Retryable error on attempt {attempt + 1}: {e}. Wait
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="memorymonitor initialization",
    )
    async def initialize(self) -> bool:
        """Initialize MemoryMonitor."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
ing {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
    passlogging.error(f"Max retries ({max_retries}) exceeded. Last error: {e}")
                        raise
                except NonRetryableError as e:
    passpasspasspasspasspasspasslogging.error(f"Non-retryable error: {e}")
                    raise
                except Exception as e:
    passpasspasspasspasspasspasslast_exception = e
                    if attempt < max_retries:
    passwait_time = backoff_factor ** attempt
                        logging.warning(f"Unexpected error on attempt {attempt + 1}: {e}. Waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
    passlogging.error(f"Max retries ({max_retries}) exceeded. Last error: {e}")
                        raise

            raise last_exception
        return wrapper
    return decorator


# ============================================================================
# Memory Management
# ============================================================================

class MemoryMonitor:
    pass"""Monitor memory usage during processing."""

    def __init__(...):
    passself.peak_usage = 0
        self.usage_history = []

    def get_usage_mb(...) -> ...:
    """..."""
    passtry:
    passimport psutil
            process = psutil.Process()
            usage_mb = process.memory_info().rss / 1024 / 1024
            self.peak_usage = max(self.peak_usage, usage_mb)
            self.usage_history.append((time.time(), usage_mb))
            return usage_mb
        except ImportError:
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="qualityresult initialization",
    )
    async def initialize(self) -> bool:
        """Initialize QualityResult."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
   
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="enhanceddataqualityvalidator initialization",
    )
    async def initialize(self) -> bool:
        """Initialize EnhancedDataQualityValidator."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
         self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False

    passpassreturn 0.0

    def get_peak_usage_mb(...) -> ...:
    """..."""
    passreturn self.peak_usage

    def is_memory_pressure(...) -> ...:
    """..."""
    passreturn self.get_usage_mb() > threshold_mb


def memory_efficient(...):
    pass"""Decorator for memory-efficient processing."""
    def decorator(...):
    passpass@functools.wraps(func)
        async def wrapper(...):
    passmonitor = MemoryMonitor()

            # Check memory before processing
            initial_memory = monitor.get_usage_mb()
            logging.info(f"Memory before {func.__name__}: {initial_memory:.1f}MB")

            try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
                result = await func(*args, **kwargs)

                # Check memory after processing
                final_memory = monitor.get_usage_mb()
                peak_memory = monitor.get_peak_usage_mb()

                logging.info(f"Memory after {func.__name__}: {final_memory:.1f}MB (peak: {peak_memory:.1f}MB)")

                if peak_memory > max_memory_mb:
    passlogging.warning(f"Peak memory usage ({peak_memory:.1f}MB) exceeded limit ({max_memory_mb}MB)")

                return result
            except Exception as e:
    passpasspasspasspasspasspasslogging.error(f"Error in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator


# ============================================================================
# Enhanced Data Quality Validation
# ============================================================================

@dataclass
class QualityResult:
    pass"""Result of data quality validation."""
    passed: bool = True
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add_issue(...):
    pass"""Add a quality issue."""
        self.issues.append(f"{issue_type}: {description}")
        self.passed = False

    def add_metric(...):
    pass"""Add a quality metric."""
        self.metrics[name] = value


class EnhancedDataQualityValidator:
    pass"""Enhanced data quality validation with real-time monitoring."""

    def __init__(...):
    passpassself.config = config
        self.logger = system_logger.getChild("DataQualityValidator")

    async def validate_unified_data_quality(...) -> ...:
    """..."""
    passresult = QualityResult()

        if df is None or df.empty:
    passresult.add_issue("empty_data", "DataFrame is None or empty")
            return result

        # Basic metrics
        result.add_metric("rows", len(df))
        result.add_metric("columns", len(df.columns))
        result.add_metric("memory_mb", df.memory_usage(deep=True).sum() / 1024 / 1024)

        # Check for NaN values
        nan_counts = df.isnull().sum()
        total_nans = nan_counts.sum()
        nan_ratio = total_nans / (len(df) * len(df.columns))

        result.add_metric("nan_count", total_nans)
        result.add_metric("nan_ratio", nan_ratio)

        if nan_ratio > self.config.max_nan_ratio:
    passpassresult.add_issue("nan_values", f"NaN ratio {nan_ratio:.4f} exceeds threshold {self.config.max_nan_ratio}")

        # Check for infinite values
        infinite_counts = {}
        for col in df.select_dtypes(include=[np.number]).columns:
    passinfinite_count = np.isinf(df[col]).sum()
            if infinite_count > 0:
    passinfinite_counts[col] = infinite_count

        total_infinites = sum(infinite_counts.values())
        result.add_metric("infinite_count", total_infinites)
        result.add_metric("infinite_columns", infinite_counts)

        if total_infinites > self.config.max_infinite_count:
    passresult.add_issue("infinite_values", f"Found {total_infinites} infinite values in columns: {list(infinite_counts.keys())}")

        # Check for constant features
        constant_features = []
        for col in df.columns:
    passunique_count = df[col].nunique()
            if unique_count < self.config.min_unique_values:
    passconstant_features.append(col)

        result.add_metric("constant_features", constant_features)
        if constant_features:
    passresult.add_issue("constant_features", f"Found {len(constant_features)} constant features: {constant_features}")

        # Check for unified data structure
        unified_issues = self._validate_unified_structure(df)
        result.add_metric("unified_structure_issues", unified_issues)
        if unified_issues:
    passpassresult.add_issue("unified_structure", f"Found {len(unified_issues)} unified structure issues")

        # Check for timestamp consistency
        if 'timestamp' in df.columns:
    passpasstimestamp_issues = self._validate_timestamp_consistency(df)
            result.add_metric("timestamp_issues", timestamp_issues)
            if timestamp_issues:
    passresult.add_issue("timestamp_issues", f"Found {len(timestamp_issues)} timestamp issues")

        # Check for data consistency across exchanges/symbols
        if 'exchange' in df.columns and 'symbol' in df.columns:
    passpassconsistency_issues = self._validate_data_consistency(df)
            result.add_metric("consistency_issues", consistency_issues)
            if consistency_issues:
    passresult.add_issue("data_consistency", f"Found {len(consistency_issues)} consistency issues")

        self.logger.info(f"Quality validation for {context}: {'PASSED' if result.passed else 'FAILED'} "
                        f"({len(result.issues)} issues)")

        return result

    def _validate_unified_structure(...) -> ...:
    pass"""..."""
    passissues = []

        # Check required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'exchange', 'symbol', 'timeframe']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
    passpassissues.append({
                "type": "missing_columns",
                "columns": missing_columns
            })

        # Check data types
        if 'timestamp' in df.columns and not pd.api.types.is_integer_dtype(df['timestamp']):
    passissues.append({
                "type": "timestamp_dtype",
                "expected": "int64",
                "actual": str(df['timestamp'].dtype)
            })

        # Check for date columns if auto_add_date_colum
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="optimizedunifieddataprocessor initialization",
    )
    async def initialize(self) -> bool:
        """Initialize OptimizedUnifiedDataProcessor."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
ns is enabled
        if self.config.auto_add_date_columns:
    passpassdate_columns = ['year', 'month', 'day']
            missing_date_columns = [col for col in date_columns if col not in df.columns]
            if missing_date_columns:
    passpassissues.append({
                    "type": "missing_date_columns",
                    "columns": missing_date_columns
                })

        return issues

    def _validate_timestamp_consistency(...) -> ...:
    """..."""
    passissues = []

        # Convert timestamp to datetime if needed
        timestamps = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        # Check for gaps
        expected_interval = pd.Timedelta(minutes=1)  # Assuming 1-minute data
        time_diffs = timestamps.diff().dropna()

        large_gaps = time_diffs[time_diffs > expected_interval * 2]
        if not large_gaps.empty:
    passpassissues.append({
                "type": "large_gaps",
                "count": len(large_gaps),
                "max_gap": large_gaps.max().total_seconds() / 60
            })

        # Check for duplicates
        duplicates = timestamps.duplicated()
        if duplicates.any():
    passpassissues.append({
                "type": "duplicate_timestamps",
                "count": duplicates.sum()
            })

        return issues

    def _validate_data_consistency(...) -> ...:
    """..."""
    passissues = []

        # Check for consistent data across exchanges
        exchange_counts = df['exchange'].value_counts()
        if len(exchange_counts) > 1:
    passpass# Check if all exchanges have similar data volumes
            mean_count = exchange_counts.mean()
            std_count = exchange_counts.std()
            cv = std_count / mean_count if mean_count > 0 else 0

            if cv > 0.5:  # Coefficient of variation > 50%
                issues.append({
                    "type": "uneven_exchange_distribution",
                    "exchange_counts": exchange_counts.to_dict(),
                    "coefficient_of_variation": cv
                })

        # Check for consistent data across symbols
        symbol_counts = df['symbol'].value_counts()
        if len(symbol_counts) > 1:
    passpassmean_count = symbol_counts.mean()
            std_count = symbol_counts.std()
            cv = std_count / mean_count if mean_count > 0 else 0

            if cv > 0.5:
    passissues.append({
                    "type": "uneven_symbol_distribution",
                    "symbol_counts": symbol_counts.to_dict(),
                    "coefficient_of_variation": cv
                })

        return issues


# ============================================================================
# Enhanced Data Processing
# ============================================================================

class OptimizedUnifiedDataProcessor:
    pass"""Optimized unified data processing with streaming and parallelization."""

    def __init__(...):
    passpassself.config = config
        self.logger = system_logger.getChild("UnifiedDataProcessor")
        self.quality_validator = EnhancedDataQualityValidator(config)
        self.memory_monitor = MemoryMonitor()

    @memory_efficient(max_memory_mb=1024)
    async def process_unified_data_streaming(...) -> ...:
    """..."""
    passself.logger.info(f"Processing unified data from {len(data_sources)} sources")

        # Process each data source
        processed_chunks = []

        for source_name, file_path in data_sources.items():
    passif not os.path.exists(file_path):
    passself.logger.warning(f"Source file not found: {file_path}")
                continue

            self.logger.info(f"Processing {source_name}: {file_path}")

            try:
    pass# Read and process source data
                source_chunks = await self._process_source_streaming(source_name, file_path)
                processed_chunks.extend(source_chunks)

            except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error processing {source_name}: {e}")
                continue

        # Combine all processed chunks
        if processed_chunks:
    passunified_data = pd.concat(processed_chunks, ignore_index=True)
            self.logger.info(f"Combined {len(processed_chunks)} chunks into unified data: {unified_data.shape}")
            return unified_data
        else:
    passself.logger.warning("No data processed")
            return pd.DataFrame()

    async def _process_source_streaming(...) -> ...:
    """..."""
    passchunks = []
        chunk_count = 0

        try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            for chunk in pd.read_parquet(file_path, chunksize=self.config.chunk_size):
    passchunk_count += 1
                self.logger.debug(f"Processing {source_name} chunk {chunk_count}")

                # Validate chunk quality
                quality_result 
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="enhancedstep1_5dataconverter initialization",
    )
    async def initialize(self) -> bool:
        """Initialize EnhancedStep1_5DataConverter."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
= await self.quality_validator.validate_unified_data_quality(
                    chunk, f"{source_name}_chunk_{chunk_count}"
                )

                if not quality_result.passed:
    passself.logger.warning(f"Quality issues in {source_name} chunk {chunk_count}: {quality_result.issues}")

                # Transform chunk to unified format
                unified_chunk = await self._transform_to_unified_format(chunk, source_name)

                if not unified_chunk.empty:
    passchunks.append(unified_chunk)

                # Check memory pressure
                if self.memory_monitor.is_memory_pressure(self.config.max_memory_mb * 0.8):
    passself.logger.warning("Memory pressure detected, processing existing chunks")
                    break

        except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error processing {source_name}: {e}")
            raise

        return chunks

    async def _transform_to_unified_format(...) -> ...:
    """..."""
    passif chunk.empty:
    passreturn chunk

        # Create unified DataFrame
        unified_chunk = pd.DataFrame()

        # Add common columns
        if 'timestamp' in chunk.columns:
    passunified_chunk['timestamp'] = chunk['timestamp']

        # Add OHLCV columns based on source type
        if source_name == 'klines':
    passohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in ohlcv_columns:
    passif col in chunk.columns:
    passunified_chunk[col] = chunk[col]

        elif source_name == 'aggtrades':
    passpass# Transform aggtrades to OHLCV
            if 'price' in chunk.columns and 'quantity' in chunk.columns:
    pass# Simple aggregation - in practice, you'd want more sophisticated aggregation
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
    passtimestamps = pd.to_datetime(unified_chunk['timestamp'], unit='ms', utc=True)
            unified_chunk['year'] = timestamps.dt.year.astype('int16')
            unified_chunk['month'] = timestamps.dt.month.astype('int8')
            unified_chunk['day'] = timestamps.dt.day.astype('int8')

        return unified_chunk

    def _optimize_dtypes(...) -> ...:
    """..."""
    passfor col in df.columns:
    passif df[col].dtype == 'float64':
    passdf[col] = pd.to_numeric(df[col], downcast='float')
            elif df[col].dtype == 'int64':
    passpassdf[col] = pd.to_numeric(df[col], downcast='integer')
            elif df[col].dtype == 'object':
    passpass# Try to convert to category if it has few unique values
                if df[col].nunique() / len(df[col]) < 0.5:
    passdf[col] = df[col].astype('category')
        return df


# ============================================================================
# Enhanced Step1_5 Implementation
# ============================================================================

class EnhancedStep1_5DataConverter:
    pass"""
    Enhanced Step1_5 Data Converter

    This class provides an improved implementation of Step1_5 data converter
    with enhanced error handling, memory optimization, and data quality validation.
    """

    def __init__(...):
    passpassself.config = config
        self.logger = system_logger.getChild("EnhancedStep1_5")
        self.processor = OptimizedUnifiedDataProcessor(config)
        self.quality_validator = EnhancedDataQualityValidator(config)
        self.memory_monitor = MemoryMonitor()

        # Validate configuration
        config_issues = config.validate()
        if config_issues:
    passraise ValueError(f"Configuration validation failed: {config_issues}")

        # Initialize directories
        self._initialize_directories()

    def _initialize_directories(...):
    pass"""Initialize required directories."""
        directories = [
            self.config.data_dir,
            self.config.unified_dir,
            self.config.backup_dir,
            self.config.temp_dir
        ]

        for directory in directories:
    passos.makedirs(directory, exist_ok=True)
            self.logger.debug(f"Initialized directory: {directory}")

    async def execute(...) -> ...:
    """..."""
    passstart_time = time.time()
        self.logger.info("🔄 Starting enhanced Step1_5 data conversion...")

        try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            # Extract parameters
            symbol = training_input.get("symbol", self.config.symbol)
            exchange = training_input.get("exchange", self.config.exchange)
            timeframe = training_input.get("timeframe", self.config.timeframe)
            data_dir = training_input.get("data_dir", self.config.data_dir)

            self.logger.info(f"🎯 Converting data for {exchange} {symbol} {timeframe}")

            # Check if unified data already exists
            unified_exists = await self._check_unified_data_exists(symbol, exchange, timeframe)

            if unified_exists and not self.config.force_rerun:
    passpassif self.config.enable_incremental:
    passself.logger.info("✅ Unified data exists, checking for incremental updates...")
                    incremental_success = await self._process_incremental_updates(symbol, exchange, timeframe)
                    if incremental_success:
    passpassself.logger.info("✅ Incremental processing completed")
                        pipeline_state["data_conversion_completed"] = True
                        pipeline_state["quality_check_passed"] = True
                        return pipeline_state

                self.logger.info("🔄 Full reprocessing required")
                await self._backup_existing_data(symbol, exchange, timeframe)

            # Perform full conversion
            conversion_success = await self._perform_full_conversion(symbol, exchange, timeframe, data_dir)

            if conversion_success:
    passself.logger.info("✅ Enhanced data conversion completed successfully")
                pipeline_state["data_conversion_completed"] = True
                pipeline_state["quality_check_passed"] = True
            else:
    passself.logger.warning("⚠️ Data conversion completed with issues")
                pipeline_state["data_conversion_completed"] = True
                pipeline_state["quality_check_passed"] = False

            # Log final metrics
            duration = time.time() - start_time
            peak_memory = self.memory_monitor.get_peak_usage_mb()

            self.logger.info(f"📊 Conversion completed in {duration:.2f}s, peak memory: {peak_memory:.1f}MB")

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"❌ Error during enhanced data conversion: {e}")
            pipeline_state["data_conversion_completed"] = False
            pipeline_state["quality_check_passed"] = False

        return pipeline_state

    async def _check_unified_data_exists(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            unified_base = os.path.join(self.config.unified_dir, exchange.lower(), symbol, timeframe)
            if os.path.exists(unified_base):
    passparquet_files = []
                for root, dirs, files in os.walk(unified_base):
    passparquet_files.extend([f for f in files if f.endswith('.parquet')])

                if parquet_files:
    passpassself.logger.info(f"✅ Found existing unified data: {len(parquet_files)} files")
                    return True

            return False
        except Exception as e:
    passpasspasspasspasspasspassself.logger.warning(f"⚠️ Error checking unified data existence: {e}")
            return False

    async def _backup_existing_data(...):
    pass"""Backup existing unified data."""
        try:
    passunified_base = os.path.join(self.config.unified_dir, exchange.lower(), symbol, timeframe)
            backup_path = os.path.join(self.config.backup_dir, f"{exchange}_{symbol}_{timeframe}_{int(time.time())}")

            if os.path.exists(unified_base):
    passimport shutil
                shutil.move(unified_base, backup_path)
                self.logger.info(f"📦 Backed up existing data to: {backup_path}")
        except Exception as e:
    passpasspasspasspasspasspassself.logger.warning(f"⚠️ Error backing up existing data: {e}")

    async def _process_incremental_updates(...) -> ...:
    """..."""
    passtry:
    passself.logger.info("🔍 Processing incremental updates...")
            # Implement incremental processing logic here
            # This would compare source data timestamps with unified data timestamps
            # and only process new data

            # For now, return False to trigger full reprocessing
            return False
        except Exception as e:
    passpasspasspasspasspasspasspassself.logger.error(f"Error processing incremental updates: {e}")
            return False

    async def _perform_full_conversion(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            # Identify data sources
            data_sources = await self._identify_data_sources(symbol, exchange, timeframe, data_dir)

            if not data_sources:
    passself.logger.warning("No data sources found for conversion")
                return False

            self.logger.info(f"📁 Found {len(data_sources)} data sources: {list(data_sources.keys())}")

            # Process unified data
            unified_data = await self.processor.process_unified_data_streaming(data_sources)

            if unified_data.empty:
    passself.logger.warning("No unified data generated")
                return False

            # Validate unified data quality
            quality_result = await self.quality_validator.validate_unified_data_quality(
                unified_data, "unified_data_final"
            )

            if not quality_result.passed:
    passself.logger.warning(f"⚠️ Quality issues in unified data: {quality_result.issues}")
                # Continue with warning instead of failing
                self.logger.warning("⚠️ Continuing with quality issues - review logs for details")

            # Save unified data
            save_success = await self._save_unified_data(unified_data, symbol, exchange, timeframe)

            if not save_success:
    passpasspassself.logger.error("Failed to save unified data")
                return False

            # Log quality metrics
            self.logger.info(f"📊 Final unified data metrics: {json.dumps(quality_result.metrics, indent=2)}")

            return True

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"Error during full conversion: {e}")
            return False

    async def _identify_data_sources(...) -> ...:
    """..."""
    passdata_sources = {}

        # Check for klines data
        klines_file = os.path.join(data_dir, f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet")
        if os.path.exists(klines_file):
    passpassdata_sources['klines'] = klines_file

        # Check for aggtrades data
        aggtrades_file = os.path.join(data_dir, f"aggtrades_{exchange}_{symbol}_consolidated.parquet")
        if os.path.exists(aggtrades_file):
    passpassdata_sources['aggtrades'] = aggtrades_file

        # Check for futures data
        futures_file = os.path.join(data_dir, f"futures_{exchange}_{symbol}_consolidated.parquet")
        if os.path.exists(futures_file):
    passpassdata_sources['futures'] = futures_file

        return data_sources

    async def _save_unified_data(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
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
    passpassself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
                import pyarrow as pa
                import pyarrow.parquet as pq

                table = pa.Table.from_pandas(unified_data, preserve_index=False)

                # Write with partitioning
                if partition_cols:
    passpasspq.write_to_dataset(
                        table,
                        output_dir,
                        partition_cols=partition_cols,
                        compression=self.config.compression,
                        use_dictionary=self.config.use_dictionary,
                        row_group_size=self.config.min_rows_per_group,
                        max_file_size=self.config.max_rows_per_file * 1024,  # Convert to bytes
                    )
                else:
    passpq.write_table(
                        table,
                        os.path.join(output_dir, "data.parquet"),
                        compression=self.config.compression,
                        use_dictionary=self.config.use_dictionary,
                        row_group_size=self.config.min_rows_per_group,
                    )

                self.logger.info("✅ Unified data saved successfully")
                return True

            except ImportError:
    passpass# Fallback to pandas
                self.logger.warning("pyarrow not available, using pandas fallback")
                unified_data.to_parquet(
                    os.path.join(output_dir, "data.parquet"),
                    compression=self.config.compression,
                    index=False
                )
                self.logger.info("✅ Unified data saved successfully (pandas fallback)")
                return True

        except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error saving unified data: {e}")
            return False


# ============================================================================
# Usage Example
# ============================================================================

async def main(...):
    pass"""Example usage of the enhanced Step1_5 implementation."""

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
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
        result = await step1_5.execute(training_input, pipeline_state)

        print("=" * 60)
        print("ENHANCED STEP1_5 EXECUTION RESULTS")
        print("=" * 60)
        print(f"Data conversion completed: {result['data_conversion_completed']}")
        print(f"Quality check passed: {result['quality_check_passed']}")
        print("=" * 60)

    except Exception as e:
    passpasspasspasspasspasspassprint(f"❌ Enhanced Step1_5 execution failed: {e}")


if __name__ == "__main__":
    pass# Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run the example
    asyncio.run(main())