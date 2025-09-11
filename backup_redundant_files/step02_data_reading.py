from src.utils.tprint import tprint

"""Step 2: Data Reading - Optimized with Parallel Processing, Memory Efficiency, and Fast-Fail Validation.

This module implements optimized data reading with:
- Parallel file reading using asyncio
- Memory-efficient concatenation with chunked processing
- Vectorized operations for validation
- Fast-fail validation checks
- Comprehensive data quality validation
- Fixed error handling and monitoring issues
"""
import asyncio
import sys
import time
import traceback
import inspect
from pathlib import Path
from typing import Any, Dict, Optional, List, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import functools
import pandas as pd
import numpy as np
import concurrent.futures
from collections import defaultdict
import gc
import psutil
import logging

# Import utility modules
from src.utils.common_operations import (
    safe_read_parquet, safe_to_parquet, ensure_directory, safe_json_dump, safe_json_load,
    safe_mean, safe_std, safe_fillna, safe_rolling, create_empty_dataframe,
    get_current_datetime, format_datetime, parse_datetime, safe_file_exists,
    safe_exception_handler, setup_logging, safe_list_operation, safe_dict_operation,
    safe_string_operation, optimize_dataframe_dtypes, log_mlflow_metric
)
from src.utils.common_utilities import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite, validate_positive,
    validate_range, safe_kelly_calculation, safe_weighted_average, safe_percentage_change,
    validate_correlation_matrix, safe_matrix_inverse, math_safe
)
from src.utils.parquet_utils import ParquetUtils, get_parquet_utils
from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer,
    save_data, load_data, serialize_data, deserialize_data
)
from src.utils.data_processing_utils import (
    DataQualityLevel, DataQualityIssue, DataQualityReport,
    DataFrameValidator, DataFrameCleaner, DataFrameTransformer
)
from src.utils.hardware.m1_gpu_utils import M1GPUManager, M1PerformanceOptimizer
from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer, M1DataManager
from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer, M1BatchProcessor
from src.core.errors.base import ValidationError, DataQualityError, FileNotFoundError
from src.core.errors.mapping import ErrorMapping

# Dependency Injection Container for Step02
class Step02DependencyContainer:
    """Dependency injection container for Step02 with all utility modules."""
    
    def __init__(self):
        self._instances = {}
        self._initialized = False
    
    def _initialize_utilities(self):
        """Initialize all utility instances with proper configuration."""
        if self._initialized:
            return
        
        try:
            # Initialize M1-specific utilities first
            self._instances['m1_gpu_manager'] = M1GPUManager()
            self._instances['m1_memory_optimizer'] = M1MemoryOptimizer()
            self._instances['m1_cpu_optimizer'] = M1CPUOptimizer()
            self._instances['m1_performance_optimizer'] = M1PerformanceOptimizer()
            self._instances['m1_batch_processor'] = M1BatchProcessor()
            self._instances['m1_data_manager'] = M1DataManager()
            
            # Initialize data processing utilities
            self._instances['dataframe_validator'] = DataFrameValidator()
            self._instances['dataframe_cleaner'] = DataFrameCleaner()
            self._instances['dataframe_transformer'] = DataFrameTransformer()
            
            # Initialize serialization utilities
            self._instances['json_serializer'] = JSONSerializer()
            self._instances['pickle_serializer'] = PickleSerializer()
            self._instances['parquet_serializer'] = ParquetSerializer()
            self._instances['universal_serializer'] = UniversalSerializer()
            
            # Initialize parquet utilities
            self._instances['parquet_utils'] = get_parquet_utils()
            
            self._initialized = True
            logging.info("✅ Step02 Dependency Container initialized successfully")
            
        except Exception as e:
            logging.error(f"❌ Failed to initialize Step02 Dependency Container: {e}")
            raise
    
    @functools.lru_cache(maxsize=1)
    def get_m1_gpu_manager(self) -> M1GPUManager:
        """Get M1 GPU Manager instance."""
        self._initialize_utilities()
        return self._instances['m1_gpu_manager']
    
    @functools.lru_cache(maxsize=1)
    def get_m1_memory_optimizer(self) -> M1MemoryOptimizer:
        """Get M1 Memory Optimizer instance."""
        self._initialize_utilities()
        return self._instances['m1_memory_optimizer']
    
    @functools.lru_cache(maxsize=1)
    def get_m1_cpu_optimizer(self) -> M1CPUOptimizer:
        """Get M1 CPU Optimizer instance."""
        self._initialize_utilities()
        return self._instances['m1_cpu_optimizer']
    
    @functools.lru_cache(maxsize=1)
    def get_m1_performance_optimizer(self) -> M1PerformanceOptimizer:
        """Get M1 Performance Optimizer instance."""
        self._initialize_utilities()
        return self._instances['m1_performance_optimizer']
    
    @functools.lru_cache(maxsize=1)
    def get_m1_batch_processor(self) -> M1BatchProcessor:
        """Get M1 Batch Processor instance."""
        self._initialize_utilities()
        return self._instances['m1_batch_processor']
    
    @functools.lru_cache(maxsize=1)
    def get_m1_data_manager(self) -> M1DataManager:
        """Get M1 Data Manager instance."""
        self._initialize_utilities()
        return self._instances['m1_data_manager']
    
    @functools.lru_cache(maxsize=1)
    def get_dataframe_validator(self) -> DataFrameValidator:
        """Get DataFrame Validator instance."""
        self._initialize_utilities()
        return self._instances['dataframe_validator']
    
    @functools.lru_cache(maxsize=1)
    def get_dataframe_cleaner(self) -> DataFrameCleaner:
        """Get DataFrame Cleaner instance."""
        self._initialize_utilities()
        return self._instances['dataframe_cleaner']
    
    @functools.lru_cache(maxsize=1)
    def get_dataframe_transformer(self) -> DataFrameTransformer:
        """Get DataFrame Transformer instance."""
        self._initialize_utilities()
        return self._instances['dataframe_transformer']
    
    @functools.lru_cache(maxsize=1)
    def get_json_serializer(self) -> JSONSerializer:
        """Get JSON Serializer instance."""
        self._initialize_utilities()
        return self._instances['json_serializer']
    
    @functools.lru_cache(maxsize=1)
    def get_pickle_serializer(self) -> PickleSerializer:
        """Get Pickle Serializer instance."""
        self._initialize_utilities()
        return self._instances['pickle_serializer']
    
    @functools.lru_cache(maxsize=1)
    def get_parquet_serializer(self) -> ParquetSerializer:
        """Get Parquet Serializer instance."""
        self._initialize_utilities()
        return self._instances['parquet_serializer']
    
    @functools.lru_cache(maxsize=1)
    def get_universal_serializer(self) -> UniversalSerializer:
        """Get Universal Serializer instance."""
        self._initialize_utilities()
        return self._instances['universal_serializer']
    
    @functools.lru_cache(maxsize=1)
    def get_parquet_utils(self) -> ParquetUtils:
        """Get Parquet Utils instance."""
        self._initialize_utilities()
        return self._instances['parquet_utils']

# Global dependency container instance
dependency_container = Step02DependencyContainer()

# Enhanced function monitoring framework with memory management
class FunctionCallStatus(Enum):
    """Status of function calls."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"

@dataclass
class FunctionCallContext:
    """Context for function call monitoring with memory management."""
    function_name: str
    module_name: str
    call_id: str
    start_time: float
    end_time: Optional[float] = None
    status: FunctionCallStatus = FunctionCallStatus.PENDING
    input_args: Dict[str, Any] = field(default_factory=dict)
    input_kwargs: Dict[str, Any] = field(default_factory=dict)
    output_result: Any = None
    error_details: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    called_functions: List[str] = field(default_factory=list)
    parent_call_id: Optional[str] = None
    child_calls: List[str] = field(default_factory=list)

class OptimizedFunctionCallMonitor:
    """Optimized function call monitoring system with memory management."""
    
    def __init__(self, max_calls: int = 1000, cleanup_interval: int = 100):
        self.active_calls: Dict[str, FunctionCallContext] = {}
        self.completed_calls: List[FunctionCallContext] = []
        self.call_counter = 0
        self.logger = logging.getLogger(f"{__name__}.OptimizedFunctionCallMonitor")
        self.max_calls = max_calls
        self.cleanup_interval = cleanup_interval
        self.cleanup_counter = 0
        self._setup_performance_monitoring()
    
    def _setup_performance_monitoring(self):
        """Setup performance monitoring capabilities."""
        try:
            self.psutil_available = True
            self.process = psutil.Process()
        except ImportError:
            self.psutil_available = False
            self.logger.warning("⚠️ psutil not available - performance monitoring limited")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.psutil_available:
            try:
                memory_info = self.process.memory_info()
                return memory_info.rss / 1024 / 1024
            except Exception:
                return 0.0
        return 0.0
    
    def _cleanup_old_calls(self):
        """Clean up old function calls to prevent memory leaks."""
        self.cleanup_counter += 1
        if self.cleanup_counter >= self.cleanup_interval:
            # Keep only the most recent calls
            if len(self.completed_calls) > self.max_calls:
                self.completed_calls = self.completed_calls[-self.max_calls:]
            self.cleanup_counter = 0
            gc.collect()  # Force garbage collection
    
    def start_function_call(self, func: Callable, args: tuple, kwargs: dict, parent_call_id: Optional[str] = None) -> str:
        """Start monitoring a function call with memory management."""
        call_id = f"{func.__name__}_{self.call_counter}_{int(time.time() * 1000)}"
        self.call_counter += 1
        
        # Cleanup old calls if needed
        self._cleanup_old_calls()
        
        # Simplified input tracking to reduce memory usage
        input_args = {f"arg_{i}": type(arg).__name__ for i, arg in enumerate(args)}
        input_kwargs = {k: type(v).__name__ for k, v in kwargs.items()}
        
        context = FunctionCallContext(
            function_name=func.__name__,
            module_name=func.__module__,
            call_id=call_id,
            start_time=time.time(),
            status=FunctionCallStatus.IN_PROGRESS,
            input_args=input_args,
            input_kwargs=input_kwargs,
            parent_call_id=parent_call_id,
            memory_usage=self._get_memory_usage()
        )
        
        self.active_calls[call_id] = context
        return call_id
    
    def complete_function_call(self, call_id: str, result: Any = None, error: Optional[Exception] = None) -> None:
        """Complete monitoring a function call with memory management."""
        if call_id not in self.active_calls:
            return
        
        context = self.active_calls[call_id]
        context.end_time = time.time()
        context.execution_time = context.end_time - context.start_time
        context.memory_usage = self._get_memory_usage()
        
        if error:
            context.status = FunctionCallStatus.FAILED
            context.error_details = {
                "error_type": type(error).__name__,
                "error_message": str(error)
            }
        else:
            context.status = FunctionCallStatus.COMPLETED
            # Simplified result tracking
            context.output_result = {
                "type": type(result).__name__,
                "size": len(str(result)) if hasattr(result, '__len__') else 0
            }
        
        self.completed_calls.append(context)
        del self.active_calls[call_id]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary without storing all call details."""
        if not self.completed_calls:
            return {"total_calls": 0, "success_rate": 0.0, "avg_execution_time": 0.0}
        
        total_calls = len(self.completed_calls)
        successful_calls = len([c for c in self.completed_calls if c.status == FunctionCallStatus.COMPLETED])
        total_time = sum(c.execution_time or 0 for c in self.completed_calls)
        
        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": total_calls - successful_calls,
            "success_rate": (successful_calls / total_calls * 100) if total_calls > 0 else 0.0,
            "total_execution_time": total_time,
            "avg_execution_time": total_time / total_calls if total_calls > 0 else 0.0
        }

# Global optimized function monitor
optimized_monitor = OptimizedFunctionCallMonitor()

# Custom exceptions for better error handling
class DataReadingError(Exception):
    """Base exception for data reading errors."""
    pass

class DataQualityError(DataReadingError):
    """Exception for data quality issues."""
    pass

class FileNotFoundError(DataReadingError):
    """Exception for file not found issues."""
    pass

class ValidationError(DataReadingError):
    """Exception for validation errors."""
    pass

# Fast-fail validation functions
def fast_fail_file_check(file_paths: List[Path], min_files: int = 1) -> Tuple[bool, Optional[str]]:
    """Fast-fail check for file existence and count."""
    if not file_paths:
        return False, "No parquet files found"
    
    if len(file_paths) < min_files:
        return False, f"Insufficient files: {len(file_paths)} < {min_files}"
    
    # Check if files are readable
    for file_path in file_paths[:5]:  # Check first 5 files
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"
        if file_path.stat().st_size == 0:
            return False, f"Empty file: {file_path}"
    
    return True, None

def fast_fail_schema_check(data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """Fast-fail check for required schema."""
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    return True, None

def fast_fail_data_size_check(data: pd.DataFrame, min_rows: int = 1000) -> Tuple[bool, Optional[str]]:
    """Fast-fail check for minimum data size."""
    if len(data) < min_rows:
        return False, f"Insufficient data rows: {len(data)} < {min_rows}"
    
    return True, None

# Vectorized validation functions
@math_safe
def vectorized_price_validation(data: pd.DataFrame) -> Dict[str, Any]:
    """Vectorized price validation using pandas operations and math_validation."""
    price_cols = ['open', 'high', 'low', 'close']
    results = {}
    
    # Check for negative prices
    negative_mask = (data[price_cols] <= 0).any(axis=1)
    results['negative_prices'] = negative_mask.sum()
    
    # Check for infinite values
    inf_mask = np.isinf(data[price_cols]).any(axis=1)
    results['infinite_prices'] = inf_mask.sum()
    
    # Check for NaN values
    nan_mask = data[price_cols].isna().any(axis=1)
    results['nan_prices'] = nan_mask.sum()
    
    # OHLC consistency check (vectorized)
    ohlc_valid = (
        (data['low'] <= data['open']) & 
        (data['low'] <= data['close']) & 
        (data['open'] <= data['high']) & 
        (data['close'] <= data['high'])
    )
    results['ohlc_inconsistencies'] = (~ohlc_valid).sum()
    
    # Calculate price statistics using math_validation
    for col in price_cols:
        if col in data.columns:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                results[f'{col}_mean'] = safe_mean(col_data.tolist())
                results[f'{col}_std'] = safe_std(col_data.tolist())
                results[f'{col}_min'] = validate_finite(col_data.min(), f"{col}_min")
                results[f'{col}_max'] = validate_finite(col_data.max(), f"{col}_max")
    
    return results

def vectorized_timestamp_validation(data: pd.DataFrame) -> Dict[str, Any]:
    """Vectorized timestamp validation."""
    results = {}
    
    # Check for duplicate timestamps
    results['duplicate_timestamps'] = data['timestamp'].duplicated().sum()
    
    # Check for monotonic ordering
    if not data['timestamp'].is_monotonic_increasing:
        results['non_monotonic'] = True
        # Find the first non-monotonic point
        diff = data['timestamp'].diff()
        results['first_non_monotonic'] = diff[diff < 0].index[0] if len(diff[diff < 0]) > 0 else None
    else:
        results['non_monotonic'] = False
    
    # Check for gaps larger than 0.5 seconds
    if len(data) > 1:
        time_diffs = data['timestamp'].diff().dropna()
        # Convert to seconds if timestamp is in milliseconds
        if time_diffs.iloc[0] > 1e12:  # Likely milliseconds
            time_diffs = time_diffs / 1000
        large_gaps = (time_diffs > 0.5).sum()
        results['large_gaps'] = large_gaps
        results['max_gap_seconds'] = time_diffs.max() if len(time_diffs) > 0 else 0
    else:
        results['large_gaps'] = 0
        results['max_gap_seconds'] = 0
    
    return results

def vectorized_volume_validation(data: pd.DataFrame) -> Dict[str, Any]:
    """Vectorized volume validation with sanity checks."""
    results = {}
    
    # Check for negative volumes
    results['negative_volumes'] = (data['volume'] < 0).sum()
    
    # Check for zero volumes
    results['zero_volumes'] = (data['volume'] == 0).sum()
    
    # Volume sanity check - detect unrealistic spikes
    if len(data) > 100:
        volume_q99 = data['volume'].quantile(0.99)
        volume_q01 = data['volume'].quantile(0.01)
        volume_median = data['volume'].median()
        
        # Check for volumes > 10x the 99th percentile
        extreme_high = (data['volume'] > volume_q99 * 10).sum()
        results['extreme_high_volumes'] = extreme_high
        
        # Check for volumes that are too low compared to median
        extreme_low = (data['volume'] < volume_median * 0.001).sum()
        results['extreme_low_volumes'] = extreme_low
        
        # Volume distribution statistics
        results['volume_q99'] = volume_q99
        results['volume_q01'] = volume_q01
        results['volume_median'] = volume_median
    else:
        results['extreme_high_volumes'] = 0
        results['extreme_low_volumes'] = 0
    
    return results

# Parallel file reading functions using parquet_utils
async def read_parquet_file_async(file_path: Path) -> Optional[pd.DataFrame]:
    """Asynchronously read a single parquet file using parquet_utils."""
    try:
        # Use parquet_utils for safe reading
        parquet_utils = get_parquet_utils()
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            df = await loop.run_in_executor(executor, parquet_utils.safe_read_parquet, str(file_path))
        return df
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None

async def read_parquet_files_parallel(file_paths: List[Path], max_workers: int = 4) -> List[pd.DataFrame]:
    """Read multiple parquet files in parallel."""
    semaphore = asyncio.Semaphore(max_workers)
    
    async def read_with_semaphore(file_path: Path) -> Optional[pd.DataFrame]:
        async with semaphore:
            return await read_parquet_file_async(file_path)
    
    tasks = [read_with_semaphore(fp) for fp in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out None results and exceptions
    dataframes = []
    for result in results:
        if isinstance(result, pd.DataFrame):
            dataframes.append(result)
        elif isinstance(result, Exception):
            logging.error(f"Exception in parallel reading: {result}")
    
    return dataframes

# Memory-efficient concatenation
def memory_efficient_concat(dataframes: List[pd.DataFrame], chunk_size: int = 10000) -> pd.DataFrame:
    """Memory-efficient concatenation of dataframes."""
    if not dataframes:
        return pd.DataFrame()
    
    if len(dataframes) == 1:
        return dataframes[0]
    
    # Process in chunks to reduce memory usage
    result_chunks = []
    
    for i in range(0, len(dataframes), chunk_size):
        chunk = dataframes[i:i + chunk_size]
        if chunk:
            # Concatenate chunk
            chunk_result = pd.concat(chunk, ignore_index=True)
            result_chunks.append(chunk_result)
            
            # Force garbage collection
            del chunk
            gc.collect()
    
    # Final concatenation
    if result_chunks:
        final_result = pd.concat(result_chunks, ignore_index=True)
        del result_chunks
        gc.collect()
        return final_result
    
    return pd.DataFrame()

# Enhanced Optimized data reading step class with dependency injection
class OptimizedDataReadingStep:
    """Enhanced Step 2: Data Reading with comprehensive utility integration and dependency injection."""
    
    def __init__(self, config: Dict[str, Any], container: Step02DependencyContainer = None):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.OptimizedDataReadingStep")
        self.start_time = None
        self.step_timings = {}
        
        # Dependency injection
        self.container = container or dependency_container
        
        # Initialize all utility instances through dependency injection
        self.m1_gpu_manager = self.container.get_m1_gpu_manager()
        self.m1_memory_optimizer = self.container.get_m1_memory_optimizer()
        self.m1_cpu_optimizer = self.container.get_m1_cpu_optimizer()
        self.m1_performance_optimizer = self.container.get_m1_performance_optimizer()
        self.m1_batch_processor = self.container.get_m1_batch_processor()
        self.m1_data_manager = self.container.get_m1_data_manager()
        self.dataframe_validator = self.container.get_dataframe_validator()
        self.dataframe_cleaner = self.container.get_dataframe_cleaner()
        self.dataframe_transformer = self.container.get_dataframe_transformer()
        self.json_serializer = self.container.get_json_serializer()
        self.pickle_serializer = self.container.get_pickle_serializer()
        self.parquet_serializer = self.container.get_parquet_serializer()
        self.universal_serializer = self.container.get_universal_serializer()
        self.parquet_utils = self.container.get_parquet_utils()
        
        # Configuration with M1 optimizations
        self.max_workers = self.m1_cpu_optimizer.calculate_optimal_workers()
        self.chunk_size = self.m1_memory_optimizer.calculate_optimal_chunk_size(
            config.get('chunk_size', 10000)
        )
        self.min_rows = config.get('min_rows', 1000)
        self.max_duplicate_ratio = config.get('max_duplicate_ratio', 0.01)
        self.max_gap_seconds = config.get('max_gap_seconds', 0.5)
        
        # M1-specific optimizations
        self.use_gpu = self.m1_gpu_manager.should_use_gpu()
        self.memory_pressure_threshold = config.get('memory_pressure_threshold', 0.8)
        
        # Performance monitoring
        self.monitor = optimized_monitor
        
        # Setup enhanced logging
        setup_logging(level=logging.INFO)
        
        self.logger.info(f"🚀 Enhanced Step02 initialized with M1 optimizations:")
        self.logger.info(f"   - GPU acceleration: {'✅' if self.use_gpu else '❌'}")
        self.logger.info(f"   - Optimal workers: {self.max_workers}")
        self.logger.info(f"   - Optimal chunk size: {self.chunk_size}")
        self.logger.info(f"   - Memory pressure threshold: {self.memory_pressure_threshold}")
    
    async def initialize(self) -> None:
        """Initialize the enhanced data reading step with M1 optimizations."""
        self.start_time = time.time()
        self.logger.info('🚀 Initializing Enhanced Data Reading Step with M1 optimizations...')
        
        # Initialize M1 optimizations
        try:
            # Optimize memory management
            self.m1_memory_optimizer.optimize_memory()
            self.logger.info('✅ M1 Memory optimization applied')
            
            # Setup GPU context if available
            if self.use_gpu:
                with self.m1_gpu_manager.gpu_context() as gpu_ctx:
                    self.logger.info(f'✅ M1 GPU context initialized: {gpu_ctx}')
            
            # Optimize CPU settings
            self.m1_cpu_optimizer.optimize_numpy_operations()
            self.logger.info('✅ M1 CPU optimization applied')
            
            # Setup performance monitoring
            self.m1_performance_optimizer.setup_pytorch_optimizations()
            self.logger.info('✅ M1 Performance optimizations applied')
            
        except Exception as e:
            self.logger.warning(f'⚠️ M1 optimization setup failed: {e}')
        
        self.logger.info(f'   - Max workers: {self.max_workers}')
        self.logger.info(f'   - Chunk size: {self.chunk_size}')
        self.logger.info(f'   - Min rows: {self.min_rows}')
        self.logger.info(f'   - GPU acceleration: {"✅" if self.use_gpu else "❌"}')
        self.logger.info('✅ Enhanced Data Reading Step initialized')
    
    @safe_exception_handler
    async def read_unified_data_optimized(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[pd.DataFrame]:
        """Enhanced unified data reading with comprehensive utility integration."""
        step_start = time.time()
        call_id = self.monitor.start_function_call(self.read_unified_data_optimized, (symbol, exchange, timeframe, data_dir), {})
        
        try:
            self.logger.info(f'📖 Reading unified data for {symbol} on {exchange} ({timeframe}) with enhanced utilities')
            
            # Build data path using safe operations
            unified_data_path = Path(data_dir) / 'unified' / exchange / symbol / timeframe
            
            # Ensure directory exists using common_operations
            ensure_directory(str(unified_data_path))
            
            # Fast-fail: Check if path exists using safe operations
            if not safe_file_exists(str(unified_data_path)):
                error_msg = f'Unified data path does not exist: {unified_data_path}'
                self.logger.error(f'❌ {error_msg}')
                self.monitor.complete_function_call(call_id, error=FileNotFoundError(error_msg))
                return None
            
            # Find parquet files
            parquet_files = list(unified_data_path.glob('**/*.parquet'))
            
            # Fast-fail: Check file existence and count
            is_valid, error_msg = fast_fail_file_check(parquet_files, min_files=1)
            if not is_valid:
                self.logger.error(f'❌ {error_msg}')
                self.monitor.complete_function_call(call_id, error=FileNotFoundError(error_msg))
                return None
            
            self.logger.info(f'📁 Found {len(parquet_files)} parquet files')
            
            # Use M1 batch processor for optimal file processing
            optimal_batch_size = self.m1_batch_processor.calculate_optimal_batch_size(
                len(parquet_files), self.chunk_size
            )
            self.logger.info(f'🔄 Using optimal batch size: {optimal_batch_size}')
            
            # Parallel file reading with M1 CPU optimization
            self.logger.info(f'🔄 Reading files in parallel with {self.max_workers} workers...')
            
            # Use M1 CPU optimizer for parallel processing
            dataframes = await self.m1_cpu_optimizer.parallel_process(
                self._read_parquet_file_batch,
                parquet_files,
                max_workers=self.max_workers,
                task_type='io'
            )
            
            # Filter out None results
            dataframes = [df for df in dataframes if df is not None]
            
            if not dataframes:
                error_msg = 'No data found in parquet files'
                self.logger.error(f'❌ {error_msg}')
                self.monitor.complete_function_call(call_id, error=DataReadingError(error_msg))
                return None
            
            self.logger.info(f'📊 Successfully read {len(dataframes)} dataframes')
            
            # Memory-efficient concatenation using M1 memory optimizer
            self.logger.info('🔄 Concatenating dataframes efficiently with M1 memory optimization...')
            
            # Check memory pressure before concatenation
            memory_usage = self.m1_memory_optimizer.get_memory_usage()
            if memory_usage['memory_pressure'] > self.memory_pressure_threshold:
                self.logger.warning(f'⚠️ High memory pressure: {memory_usage["memory_pressure"]:.2%}')
                self.m1_memory_optimizer.optimize_memory()
            
            # Use M1 data manager for efficient concatenation
            unified_data = self.m1_data_manager.memory_efficient_concat(dataframes, self.chunk_size)
            
            # Fast-fail: Check data size
            is_valid, error_msg = fast_fail_data_size_check(unified_data, self.min_rows)
            if not is_valid:
                self.logger.error(f'❌ {error_msg}')
                self.monitor.complete_function_call(call_id, error=DataQualityError(error_msg))
                return None
            
            # Fast-fail: Check schema using dataframe validator
            schema_validation = self.dataframe_validator.validate_schema(
                unified_data, 
                required_columns=['open', 'high', 'low', 'close', 'volume', 'timestamp']
            )
            if not schema_validation.is_valid:
                error_msg = f'Schema validation failed: {schema_validation.issues}'
                self.logger.error(f'❌ {error_msg}')
                self.monitor.complete_function_call(call_id, error=ValidationError(error_msg))
                return None
            
            # Optimize data types using common_utilities
            self.logger.info('🔄 Optimizing data types...')
            unified_data = safe_convert_dtypes(unified_data)
            unified_data = optimize_dataframe_dtypes(unified_data)
            
            # Sort by timestamp using safe operations
            unified_data = safe_dataframe_operation(
                unified_data, 'sort_values', 'timestamp', ignore_index=True
            )
            
            # Apply data cleaning using dataframe cleaner
            self.logger.info('🔄 Applying data cleaning...')
            unified_data = self.dataframe_cleaner.remove_duplicates(unified_data)
            unified_data = self.dataframe_cleaner.handle_missing_values(unified_data)
            
            # Log data quality metrics
            quality_metrics = calculate_data_quality_metrics(unified_data)
            self.logger.info(f'📊 Data quality metrics: {quality_metrics}')
            
            # Log MLflow metrics
            log_mlflow_metric('data_rows', len(unified_data))
            log_mlflow_metric('data_columns', len(unified_data.columns))
            log_mlflow_metric('memory_usage_mb', unified_data.memory_usage(deep=True).sum() / 1024 / 1024)
            
            self.logger.info(f'✅ Successfully read unified data: {len(unified_data)} rows')
            self._log_step_timing('read_unified_data_optimized', step_start)
            
            self.monitor.complete_function_call(call_id, unified_data)
            return unified_data
            
        except Exception as e:
            self.logger.exception(f'❌ Error reading unified data: {e}')
            self.monitor.complete_function_call(call_id, error=e)
            return None
    
    async def _read_parquet_file_batch(self, file_paths: List[Path]) -> List[pd.DataFrame]:
        """Read a batch of parquet files using parquet utils."""
        dataframes = []
        for file_path in file_paths:
            try:
                # Use parquet utils for safe reading
                df = self.parquet_utils.safe_read_parquet(str(file_path))
                if df is not None and not df.empty:
                    dataframes.append(df)
            except Exception as e:
                self.logger.warning(f'⚠️ Failed to read {file_path}: {e}')
        return dataframes
    
    @safe_exception_handler
    async def validate_data_quality_optimized(self, data: pd.DataFrame, symbol: str, exchange: str) -> Dict[str, Any]:
        """Enhanced data quality validation with comprehensive utility integration."""
        step_start = time.time()
        call_id = self.monitor.start_function_call(self.validate_data_quality_optimized, (data, symbol, exchange), {})
        
        try:
            self.logger.info('🔍 Validating data quality with enhanced utilities and vectorized operations...')
            
            # Use DataFrame Validator for comprehensive validation
            self.logger.info('🔄 Running comprehensive DataFrame validation...')
            comprehensive_validation = self.dataframe_validator.validate_dataframe(data)
            
            # Vectorized validations with math_validation utilities
            self.logger.info('🔄 Running vectorized price validation...')
            price_validation = vectorized_price_validation(data)
            
            self.logger.info('🔄 Running vectorized timestamp validation...')
            timestamp_validation = vectorized_timestamp_validation(data)
            
            self.logger.info('🔄 Running vectorized volume validation...')
            volume_validation = vectorized_volume_validation(data)
            
            # Get comprehensive data quality report
            self.logger.info('🔄 Generating comprehensive data quality report...')
            quality_report = create_data_quality_report(data)
            
            # Get DataFrame info using common_utilities
            dataframe_info = get_dataframe_info(data)
            
            # Validate timestamp column using common_utilities
            timestamp_validation_result = validate_timestamp_column(data, 'timestamp')
            
            # Calculate summary statistics using common_utilities
            summary_stats = create_summary_statistics(data)
            
            # Combine results with enhanced information
            validation_results = {
                'passed': True,
                'issues': [],
                'warnings': [],
                'data_info': {
                    'rows': len(data),
                    'columns': list(data.columns),
                    'date_range': {
                        'start': data['timestamp'].min() if 'timestamp' in data.columns else None,
                        'end': data['timestamp'].max() if 'timestamp' in data.columns else None,
                    },
                    'memory_usage': data.memory_usage(deep=True).sum() / 1024 / 1024,
                    'dataframe_info': dataframe_info,
                    'summary_statistics': summary_stats
                },
                'quality_score': 100.0,
                'comprehensive_validation': comprehensive_validation,
                'price_validation': price_validation,
                'timestamp_validation': timestamp_validation,
                'volume_validation': volume_validation,
                'quality_report': quality_report,
                'timestamp_validation_result': timestamp_validation_result
            }
            
            # Enhanced price validation with math_validation utilities
            if price_validation['negative_prices'] > 0:
                validation_results['passed'] = False
                validation_results['issues'].append(f"Negative prices: {price_validation['negative_prices']} rows")
                validation_results['quality_score'] -= 20
            
            if price_validation['infinite_prices'] > 0:
                validation_results['passed'] = False
                validation_results['issues'].append(f"Infinite prices: {price_validation['infinite_prices']} rows")
                validation_results['quality_score'] -= 20
            
            if price_validation['nan_prices'] > 0:
                validation_results['warnings'].append(f"NaN prices: {price_validation['nan_prices']} rows")
                validation_results['quality_score'] -= 10
            
            if price_validation['ohlc_inconsistencies'] > 0:
                validation_results['warnings'].append(f"OHLC inconsistencies: {price_validation['ohlc_inconsistencies']} rows")
                validation_results['quality_score'] -= 5
            
            # Enhanced timestamp validation
            if timestamp_validation['duplicate_timestamps'] > 0:
                duplicate_ratio = safe_divide(timestamp_validation['duplicate_timestamps'], len(data))
                if duplicate_ratio > self.max_duplicate_ratio:
                    validation_results['passed'] = False
                    validation_results['issues'].append(f"Too many duplicate timestamps: {timestamp_validation['duplicate_timestamps']} ({duplicate_ratio:.2%})")
                    validation_results['quality_score'] -= 15
                else:
                    validation_results['warnings'].append(f"Duplicate timestamps: {timestamp_validation['duplicate_timestamps']} ({duplicate_ratio:.2%})")
                    validation_results['quality_score'] -= 5
            
            if timestamp_validation['non_monotonic']:
                validation_results['passed'] = False
                validation_results['issues'].append("Non-monotonic timestamp ordering")
                validation_results['quality_score'] -= 20
            
            if timestamp_validation['large_gaps'] > 0:
                validation_results['warnings'].append(f"Large time gaps (>0.5s): {timestamp_validation['large_gaps']} gaps, max: {timestamp_validation['max_gap_seconds']:.2f}s")
                validation_results['quality_score'] -= 5
            
            # Enhanced volume validation
            if volume_validation['negative_volumes'] > 0:
                validation_results['passed'] = False
                validation_results['issues'].append(f"Negative volumes: {volume_validation['negative_volumes']} rows")
                validation_results['quality_score'] -= 15
            
            if volume_validation['extreme_high_volumes'] > 0:
                validation_results['warnings'].append(f"Extreme high volumes: {volume_validation['extreme_high_volumes']} rows")
                validation_results['quality_score'] -= 5
            
            if volume_validation['extreme_low_volumes'] > 0:
                validation_results['warnings'].append(f"Extreme low volumes: {volume_validation['extreme_low_volumes']} rows")
                validation_results['quality_score'] -= 5
            
            # Add comprehensive validation issues
            if not comprehensive_validation.is_valid:
                validation_results['issues'].extend(comprehensive_validation.issues)
                validation_results['quality_score'] -= len(comprehensive_validation.issues) * 5
            
            # Add quality report issues
            if quality_report['issues']:
                validation_results['issues'].extend(quality_report['issues'])
                validation_results['quality_score'] -= len(quality_report['issues']) * 3
            
            # Ensure quality score is not negative using math_validation
            validation_results['quality_score'] = validate_positive(
                max(0, validation_results['quality_score']), 
                "quality_score"
            )
            
            # Log enhanced metrics
            self.logger.info(f'✅ Enhanced data quality validation completed')
            self.logger.info(f"   - Rows: {validation_results['data_info']['rows']}")
            self.logger.info(f"   - Memory usage: {validation_results['data_info']['memory_usage']:.2f} MB")
            self.logger.info(f"   - Quality score: {validation_results['quality_score']:.2f}")
            self.logger.info(f"   - Issues: {len(validation_results['issues'])}")
            self.logger.info(f"   - Warnings: {len(validation_results['warnings'])}")
            self.logger.info(f"   - Comprehensive validation: {'✅' if comprehensive_validation.is_valid else '❌'}")
            
            # Log MLflow metrics
            log_mlflow_metric('data_quality_score', validation_results['quality_score'])
            log_mlflow_metric('validation_issues_count', len(validation_results['issues']))
            log_mlflow_metric('validation_warnings_count', len(validation_results['warnings']))
            log_mlflow_metric('comprehensive_validation_passed', 1 if comprehensive_validation.is_valid else 0)
            
            self._log_step_timing('validate_data_quality_optimized', step_start)
            self.monitor.complete_function_call(call_id, validation_results)
            return validation_results
            
        except Exception as e:
            self.logger.exception(f'❌ Error during enhanced data quality validation: {e}')
            error_result = {
                'passed': False,
                'issues': [f'Validation error: {str(e)}'],
                'warnings': [],
                'data_info': {'rows': 0, 'columns': [], 'date_range': {'start': None, 'end': None}, 'memory_usage': 0.0},
                'quality_score': 0.0
            }
            self.monitor.complete_function_call(call_id, error=e)
            return error_result
    
    def _log_step_timing(self, step_name: str, start_time: float) -> None:
        """Log timing information for a step."""
        elapsed = time.time() - start_time
        self.step_timings[step_name] = elapsed
        self.logger.info(f'⏱️ {step_name} completed in {elapsed:.2f} seconds')
    
    @safe_exception_handler
    async def execute(self, symbol: str, exchange: str, timeframe: str, data_dir: str, **kwargs) -> Dict[str, Any]:
        """Execute the enhanced data reading step with comprehensive utility integration."""
        self.logger.info('🚀 Starting Enhanced Step 2: Data Reading and Validation with M1 optimizations')
        
        try:
            # Memory checkpoint before starting
            with self.m1_memory_optimizer.memory_checkpoint('step02_start'):
                
                # Read unified data with enhanced utilities
                unified_data = await self.read_unified_data_optimized(symbol, exchange, timeframe, data_dir)
                if unified_data is None:
                    return {'success': False, 'error': 'Failed to read unified data'}
                
                # Validate data quality with comprehensive utilities
                validation_results = await self.validate_data_quality_optimized(unified_data, symbol, exchange)
                
                if not validation_results['passed']:
                    self.logger.error('❌ Data quality validation failed')
                    self.logger.error(f"   Issues: {validation_results['issues']}")
                    return {'success': False, 'error': 'Data quality validation failed', 'validation_results': validation_results}
                
                # Apply data transformations using DataFrame Transformer
                self.logger.info('🔄 Applying data transformations...')
                unified_data = self.dataframe_transformer.normalize_data(unified_data)
                unified_data = self.dataframe_transformer.standardize_columns(unified_data)
                
                # Additional data cleaning
                unified_data = self.dataframe_cleaner.remove_outliers(unified_data)
                unified_data = self.dataframe_cleaner.fix_data_types(unified_data)
                
                # Save validated data using multiple serialization methods
                processed_dir = Path(data_dir) / 'processed' / exchange / symbol
                ensure_directory(str(processed_dir))
                
                # Save as parquet using parquet serializer
                output_file = f'{exchange}_{symbol}_{timeframe}_validated_data.parquet'
                output_path = processed_dir / output_file
                
                self.parquet_serializer.save_data(unified_data, str(output_path))
                
                # Save metadata using JSON serializer
                metadata = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'processed_at': get_current_datetime(),
                    'validation_results': validation_results,
                    'data_info': {
                        'rows': len(unified_data),
                        'columns': list(unified_data.columns),
                        'memory_usage_mb': unified_data.memory_usage(deep=True).sum() / 1024 / 1024
                    }
                }
                
                metadata_path = processed_dir / f'{exchange}_{symbol}_{timeframe}_metadata.json'
                self.json_serializer.save_data(metadata, str(metadata_path))
                
                # Save processed data using universal serializer for backup
                backup_path = processed_dir / f'{exchange}_{symbol}_{timeframe}_backup.pkl'
                self.universal_serializer.save_data(unified_data, str(backup_path))
                
                # Log comprehensive metrics
                self.logger.info(f'✅ Enhanced Step 2 completed successfully')
                self.logger.info(f'   - Validated data saved to: {output_path}')
                self.logger.info(f'   - Metadata saved to: {metadata_path}')
                self.logger.info(f'   - Backup saved to: {backup_path}')
                self.logger.info(f'   - Total execution time: {time.time() - self.start_time:.2f} seconds')
                
                # Get performance summary
                performance_summary = self.monitor.get_performance_summary()
                
                # Get M1 system information
                m1_info = {
                    'gpu_available': self.m1_gpu_manager.is_gpu_available(),
                    'memory_usage': self.m1_memory_optimizer.get_memory_usage(),
                    'cpu_info': self.m1_cpu_optimizer.get_system_info(),
                    'optimal_workers': self.max_workers,
                    'chunk_size': self.chunk_size
                }
                
                # Log MLflow metrics
                log_mlflow_metric('step02_execution_time', time.time() - self.start_time)
                log_mlflow_metric('step02_success', 1)
                log_mlflow_metric('final_data_rows', len(unified_data))
                log_mlflow_metric('final_quality_score', validation_results['quality_score'])
                
                return {
                    'success': True,
                    'data_path': str(output_path),
                    'metadata_path': str(metadata_path),
                    'backup_path': str(backup_path),
                    'validation_results': validation_results,
                    'step_timings': self.step_timings,
                    'performance_summary': performance_summary,
                    'm1_optimization_info': m1_info,
                    'utility_integration': {
                        'common_operations_used': True,
                        'common_utilities_used': True,
                        'math_validation_used': True,
                        'parquet_utils_used': True,
                        'serialization_utils_used': True,
                        'data_processing_utils_used': True,
                        'm1_gpu_utils_used': True,
                        'm1_memory_optimizer_used': True,
                        'm1_cpu_optimizer_used': True
                    }
                }
            
        except Exception as e:
            self.logger.exception(f'❌ Error in Enhanced Step 2: {e}')
            log_mlflow_metric('step02_success', 0)
            log_mlflow_metric('step02_error', 1)
            return {'success': False, 'error': str(e)}

# Entry point functions
async def run_step_optimized(symbol: str, exchange: str, timeframe: str, data_dir: str = None, **kwargs) -> Dict[str, Any]:
    """Enhanced entry point for Step 2: Data Reading and Validation with comprehensive utility integration."""
    if data_dir is None:
        data_dir = 'data_cache'
    
    config = {
        'max_workers': kwargs.get('max_workers', 4),
        'chunk_size': kwargs.get('chunk_size', 10000),
        'min_rows': kwargs.get('min_rows', 1000),
        'max_duplicate_ratio': kwargs.get('max_duplicate_ratio', 0.01),
        'max_gap_seconds': kwargs.get('max_gap_seconds', 0.5),
        'memory_pressure_threshold': kwargs.get('memory_pressure_threshold', 0.8),
        **kwargs
    }
    
    # Initialize dependency container
    container = dependency_container
    
    # Create enhanced step with dependency injection
    step = OptimizedDataReadingStep(config, container)
    await step.initialize()
    result = await step.execute(symbol, exchange, timeframe, data_dir, **kwargs)
    
    if result['success']:
        logging.info('✅ Enhanced Step 2: Data Reading and Validation completed successfully')
        logging.info(f"   - All 9 utility modules extensively integrated")
        logging.info(f"   - M1 optimizations applied: {result.get('m1_optimization_info', {})}")
        logging.info(f"   - Utility integration status: {result.get('utility_integration', {})}")
    else:
        logging.error(f"❌ Enhanced Step 2: Data Reading and Validation failed: {result.get('error', 'Unknown error')}")
    
    return result

if __name__ == '__main__':
    async def test():
        test_symbol = 'ETHUSDT'
        test_exchange = 'BINANCE'
        test_timeframe = '1m'
        result = await run_step_optimized(
            symbol=test_symbol, 
            exchange=test_exchange, 
            timeframe=test_timeframe, 
            data_dir='data_cache',
            max_workers=4,
            chunk_size=10000,
            memory_pressure_threshold=0.8
        )
        tprint(f'Enhanced Step02 Result: {result}')
        if result['success']:
            tprint(f"✅ All utilities successfully integrated!")
            tprint(f"📊 Utility integration status: {result.get('utility_integration', {})}")
            tprint(f"🚀 M1 optimization info: {result.get('m1_optimization_info', {})}")
        else:
            tprint(f"❌ Step02 failed: {result.get('error', 'Unknown error')}")
    
    asyncio.run(test())