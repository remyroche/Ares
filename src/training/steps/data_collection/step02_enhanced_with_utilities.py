"""
Enhanced Step 2: Data Reading with Comprehensive Utility Integration

This module implements step02 with extensive integration of all utility modules:
- common_operations.py: Safe operations, file handling, data validation
- common_utilities.py: DataFrame operations, data quality metrics
- math_validation.py: Safe mathematical operations, validation
- parquet_utils.py: Optimized parquet file operations
- serialization_utils.py: Data persistence and serialization
- data_processing_utils.py: Comprehensive data validation and cleaning
- m1_gpu_utils.py: M1 GPU optimization and acceleration
- m1_memory_optimizer.py: Memory-efficient processing
- m1_cpu_optimizer.py: Parallel processing optimization

All utilities are integrated through dependency injection for clean architecture.
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

# Import dependency injection container
from .step02_dependency_injection import get_utility_manager, get_container

# Import core error handling
from src.core.errors.base import ValidationError, DataQualityError, FileNotFoundError
from src.core.errors.mapping import ErrorMapping

# Enhanced function monitoring framework with comprehensive utility integration
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
    """Context for function call monitoring with comprehensive utility integration."""
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
    utility_usage: Dict[str, int] = field(default_factory=dict)

class EnhancedFunctionCallMonitor:
    """Enhanced function call monitoring system with comprehensive utility tracking."""
    
    def __init__(self, max_calls: int = 1000, cleanup_interval: int = 100):
        self.active_calls: Dict[str, FunctionCallContext] = {}
        self.completed_calls: List[FunctionCallContext] = []
        self.call_counter = 0
        self.logger = logging.getLogger(f"{__name__}.EnhancedFunctionCallMonitor")
        self.max_calls = max_calls
        self.cleanup_interval = cleanup_interval
        self.cleanup_counter = 0
        self._setup_performance_monitoring()
        
        # Initialize utility manager
        self.utility_manager = get_utility_manager()
        self.utility_manager.initialize()
    
    def _setup_performance_monitoring(self):
        """Setup performance monitoring capabilities."""
        try:
            self.psutil_available = True
            self.process = psutil.Process()
        except ImportError:
            self.psutil_available = False
            self.logger.warning("⚠️ psutil not available - performance monitoring limited")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB using common_operations."""
        if self.psutil_available:
            try:
                memory_info = self.process.memory_info()
                return memory_info.rss / 1024 / 1024
            except Exception:
                return 0.0
        return 0.0
    
    def _cleanup_old_calls(self):
        """Clean up old function calls to prevent memory leaks using M1 memory optimizer."""
        self.cleanup_counter += 1
        if self.cleanup_counter >= self.cleanup_interval:
            # Use M1 memory optimizer for cleanup
            try:
                memory_optimizer = self.utility_manager.m1_memory_optimizer['get_m1_memory_optimizer']()
                memory_optimizer.optimize_memory()
            except Exception as e:
                self.logger.warning(f"Memory optimization failed: {e}")
            
            # Keep only the most recent calls
            if len(self.completed_calls) > self.max_calls:
                self.completed_calls = self.completed_calls[-self.max_calls:]
            self.cleanup_counter = 0
            gc.collect()  # Force garbage collection
    
    def start_function_call(self, func: Callable, args: tuple, kwargs: dict, parent_call_id: Optional[str] = None) -> str:
        """Start monitoring a function call with comprehensive utility tracking."""
        call_id = f"{func.__name__}_{self.call_counter}_{int(time.time() * 1000)}"
        self.call_counter += 1
        
        # Cleanup old calls if needed
        self._cleanup_old_calls()
        
        # Track utility usage
        utility_usage = {}
        
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
            memory_usage=self._get_memory_usage(),
            utility_usage=utility_usage
        )
        
        self.active_calls[call_id] = context
        return call_id
    
    def complete_function_call(self, call_id: str, result: Any = None, error: Optional[Exception] = None) -> None:
        """Complete monitoring a function call with comprehensive utility tracking."""
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
        """Get performance summary with utility usage statistics."""
        if not self.completed_calls:
            return {"total_calls": 0, "success_rate": 0.0, "avg_execution_time": 0.0, "utility_usage": {}}
        
        total_calls = len(self.completed_calls)
        successful_calls = len([c for c in self.completed_calls if c.status == FunctionCallStatus.COMPLETED])
        total_time = sum(c.execution_time or 0 for c in self.completed_calls)
        
        # Aggregate utility usage
        utility_usage = defaultdict(int)
        for call in self.completed_calls:
            for utility, count in call.utility_usage.items():
                utility_usage[utility] += count
        
        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": total_calls - successful_calls,
            "success_rate": (successful_calls / total_calls * 100) if total_calls > 0 else 0.0,
            "total_execution_time": total_time,
            "avg_execution_time": total_time / total_calls if total_calls > 0 else 0.0,
            "utility_usage": dict(utility_usage)
        }

# Global enhanced function monitor
enhanced_monitor = EnhancedFunctionCallMonitor()

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

# Enhanced validation functions using comprehensive utility integration
def enhanced_fast_fail_file_check(file_paths: List[Path], min_files: int = 1, utility_manager=None) -> Tuple[bool, Optional[str]]:
    """Enhanced fast-fail check for file existence and count using common_operations."""
    if utility_manager is None:
        utility_manager = get_utility_manager()
    
    common_ops = utility_manager.common_ops
    
    if not file_paths:
        return False, "No parquet files found"
    
    if len(file_paths) < min_files:
        return False, f"Insufficient files: {len(file_paths)} < {min_files}"
    
    # Check if files are readable using common_operations
    for file_path in file_paths[:5]:  # Check first 5 files
        if not common_ops['safe_file_exists'](file_path):
            return False, f"File does not exist: {file_path}"
        if file_path.stat().st_size == 0:
            return False, f"Empty file: {file_path}"
    
    return True, None

def enhanced_fast_fail_schema_check(data: pd.DataFrame, utility_manager=None) -> Tuple[bool, Optional[str]]:
    """Enhanced fast-fail check for required schema using common_utilities."""
    if utility_manager is None:
        utility_manager = get_utility_manager()
    
    common_utils = utility_manager.common_utils
    
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    is_valid, missing_columns = common_utils['validate_dataframe_columns'](data, required_columns)
    
    if not is_valid:
        return False, f"Missing required columns: {missing_columns}"
    
    return True, None

def enhanced_fast_fail_data_size_check(data: pd.DataFrame, min_rows: int = 1000, utility_manager=None) -> Tuple[bool, Optional[str]]:
    """Enhanced fast-fail check for minimum data size using common_operations."""
    if utility_manager is None:
        utility_manager = get_utility_manager()
    
    common_ops = utility_manager.common_ops
    
    if len(data) < min_rows:
        return False, f"Insufficient data rows: {len(data)} < {min_rows}"
    
    return True, None

# Enhanced vectorized validation functions using comprehensive utility integration
def enhanced_vectorized_price_validation(data: pd.DataFrame, utility_manager=None) -> Dict[str, Any]:
    """Enhanced vectorized price validation using math_validation and common_operations."""
    if utility_manager is None:
        utility_manager = get_utility_manager()
    
    math_validation = utility_manager.math_validation
    common_ops = utility_manager.common_ops
    
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
    
    # Calculate price statistics using math_validation and common_operations
    for col in price_cols:
        if col in data.columns:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                results[f'{col}_mean'] = common_ops['safe_mean'](col_data.tolist())
                results[f'{col}_std'] = common_ops['safe_std'](col_data.tolist())
                results[f'{col}_min'] = math_validation['validate_finite'](col_data.min(), f"{col}_min")
                results[f'{col}_max'] = math_validation['validate_finite'](col_data.max(), f"{col}_max")
    
    return results

def enhanced_vectorized_timestamp_validation(data: pd.DataFrame, utility_manager=None) -> Dict[str, Any]:
    """Enhanced vectorized timestamp validation using common_utilities."""
    if utility_manager is None:
        utility_manager = get_utility_manager()
    
    common_utils = utility_manager.common_utils
    
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

def enhanced_vectorized_volume_validation(data: pd.DataFrame, utility_manager=None) -> Dict[str, Any]:
    """Enhanced vectorized volume validation using math_validation."""
    if utility_manager is None:
        utility_manager = get_utility_manager()
    
    math_validation = utility_manager.math_validation
    
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
        
        # Volume distribution statistics using math_validation
        results['volume_q99'] = math_validation['validate_finite'](volume_q99, "volume_q99")
        results['volume_q01'] = math_validation['validate_finite'](volume_q01, "volume_q01")
        results['volume_median'] = math_validation['validate_finite'](volume_median, "volume_median")
    else:
        results['extreme_high_volumes'] = 0
        results['extreme_low_volumes'] = 0
    
    return results

# Enhanced parallel file reading functions using parquet_utils
async def enhanced_read_parquet_file_async(file_path: Path, utility_manager=None) -> Optional[pd.DataFrame]:
    """Enhanced asynchronous parquet file reading using parquet_utils."""
    if utility_manager is None:
        utility_manager = get_utility_manager()
    
    parquet_utils = utility_manager.parquet_utils
    
    try:
        # Use parquet_utils for safe reading
        parquet_utils_instance = parquet_utils['get_parquet_utils']()
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            df = await loop.run_in_executor(executor, parquet_utils_instance.safe_read_parquet, str(file_path))
        return df
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None

async def enhanced_read_parquet_files_parallel(file_paths: List[Path], max_workers: int = 4, utility_manager=None) -> List[pd.DataFrame]:
    """Enhanced parallel parquet file reading using M1 CPU optimizer."""
    if utility_manager is None:
        utility_manager = get_utility_manager()
    
    m1_cpu_optimizer = utility_manager.m1_cpu_optimizer
    
    # Use M1 CPU optimizer for optimal worker count
    cpu_optimizer = m1_cpu_optimizer['get_m1_cpu_optimizer']()
    optimal_workers = cpu_optimizer.get_optimal_workers_for_task("io_bound")
    max_workers = min(max_workers, optimal_workers)
    
    semaphore = asyncio.Semaphore(max_workers)
    
    async def read_with_semaphore(file_path: Path) -> Optional[pd.DataFrame]:
        async with semaphore:
            return await enhanced_read_parquet_file_async(file_path, utility_manager)
    
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

# Enhanced memory-efficient concatenation using M1 memory optimizer
def enhanced_memory_efficient_concat(dataframes: List[pd.DataFrame], chunk_size: int = 10000, utility_manager=None) -> pd.DataFrame:
    """Enhanced memory-efficient concatenation using M1 memory optimizer."""
    if utility_manager is None:
        utility_manager = get_utility_manager()
    
    m1_memory_optimizer = utility_manager.m1_memory_optimizer
    
    if not dataframes:
        return pd.DataFrame()
    
    if len(dataframes) == 1:
        return dataframes[0]
    
    # Use M1 memory optimizer for efficient concatenation
    memory_optimizer = m1_memory_optimizer['get_m1_memory_optimizer']()
    
    # Check if we should use chunked processing
    total_memory_mb = sum(df.memory_usage(deep=True).sum() for df in dataframes) / (1024**2)
    if memory_optimizer.should_chunk_data(total_memory_mb, "general"):
        return memory_optimizer.memory_efficient_concat(dataframes)
    
    # Process in chunks to reduce memory usage
    result_chunks = []
    
    for i in range(0, len(dataframes), chunk_size):
        chunk = dataframes[i:i + chunk_size]
        if chunk:
            # Concatenate chunk
            chunk_result = pd.concat(chunk, ignore_index=True)
            result_chunks.append(chunk_result)
            
            # Use M1 memory optimizer for cleanup
            memory_optimizer.optimize_memory()
            
            # Force garbage collection
            del chunk
            gc.collect()
    
    # Final concatenation
    if result_chunks:
        final_result = pd.concat(result_chunks, ignore_index=True)
        del result_chunks
        memory_optimizer.optimize_memory()
        gc.collect()
        return final_result
    
    return pd.DataFrame()

# Enhanced data reading step class with comprehensive utility integration
class EnhancedDataReadingStep:
    """Enhanced Step 2: Data Reading with comprehensive utility integration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.EnhancedDataReadingStep")
        self.start_time = None
        self.step_timings = {}
        
        # Initialize utility manager
        self.utility_manager = get_utility_manager()
        self.utility_manager.initialize()
        
        # Configuration
        self.max_workers = config.get('max_workers', 4)
        self.chunk_size = config.get('chunk_size', 10000)
        self.min_rows = config.get('min_rows', 1000)
        self.max_duplicate_ratio = config.get('max_duplicate_ratio', 0.01)
        self.max_gap_seconds = config.get('max_gap_seconds', 0.5)
        
        # Performance monitoring
        self.monitor = enhanced_monitor
        
        # Initialize M1 optimizers
        self._initialize_m1_optimizers()
    
    def _initialize_m1_optimizers(self):
        """Initialize M1 optimizers for performance."""
        try:
            # Initialize M1 GPU manager
            m1_gpu_utils = self.utility_manager.m1_gpu_utils
            self.m1_gpu_manager = m1_gpu_utils['get_m1_gpu_manager']()
            
            # Initialize M1 memory optimizer
            m1_memory_optimizer = self.utility_manager.m1_memory_optimizer
            self.m1_memory_optimizer = m1_memory_optimizer['get_m1_memory_optimizer']()
            
            # Initialize M1 CPU optimizer
            m1_cpu_optimizer = self.utility_manager.m1_cpu_optimizer
            self.m1_cpu_optimizer = m1_cpu_optimizer['get_m1_cpu_optimizer']()
            
            self.logger.info("✅ M1 optimizers initialized successfully")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize M1 optimizers: {e}")
    
    async def initialize(self) -> None:
        """Initialize the enhanced data reading step."""
        self.start_time = time.time()
        self.logger.info('🚀 Initializing Enhanced Data Reading Step...')
        self.logger.info(f'   - Max workers: {self.max_workers}')
        self.logger.info(f'   - Chunk size: {self.chunk_size}')
        self.logger.info(f'   - Min rows: {self.min_rows}')
        
        # Log utility manager health status
        health_status = self.utility_manager.get_health_status()
        self.logger.info(f'   - Utility manager status: {health_status["status"]}')
        
        self.logger.info('✅ Enhanced Data Reading Step initialized')
    
    async def read_unified_data_enhanced(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[pd.DataFrame]:
        """Read unified data with comprehensive utility integration."""
        step_start = time.time()
        call_id = self.monitor.start_function_call(self.read_unified_data_enhanced, (symbol, exchange, timeframe, data_dir), {})
        
        try:
            self.logger.info(f'📖 Reading unified data for {symbol} on {exchange} ({timeframe})')
            
            # Build data path using common_operations
            common_ops = self.utility_manager.common_ops
            unified_data_path = Path(data_dir) / 'unified' / exchange / symbol / timeframe
            
            # Fast-fail: Check if path exists using common_operations
            if not common_ops['safe_file_exists'](unified_data_path):
                error_msg = f'Unified data path does not exist: {unified_data_path}'
                self.logger.error(f'❌ {error_msg}')
                self.monitor.complete_function_call(call_id, error=FileNotFoundError(error_msg))
                return None
            
            # Find parquet files using common_operations
            parquet_files = common_ops['list_parquet_files'](unified_data_path, recursive=True)
            
            # Enhanced fast-fail: Check file existence and count
            is_valid, error_msg = enhanced_fast_fail_file_check(parquet_files, min_files=1, utility_manager=self.utility_manager)
            if not is_valid:
                self.logger.error(f'❌ {error_msg}')
                self.monitor.complete_function_call(call_id, error=FileNotFoundError(error_msg))
                return None
            
            self.logger.info(f'📁 Found {len(parquet_files)} parquet files')
            
            # Enhanced parallel file reading using M1 CPU optimizer
            self.logger.info(f'🔄 Reading files in parallel with {self.max_workers} workers...')
            dataframes = await enhanced_read_parquet_files_parallel(parquet_files, self.max_workers, self.utility_manager)
            
            if not dataframes:
                error_msg = 'No data found in parquet files'
                self.logger.error(f'❌ {error_msg}')
                self.monitor.complete_function_call(call_id, error=DataReadingError(error_msg))
                return None
            
            self.logger.info(f'📊 Successfully read {len(dataframes)} dataframes')
            
            # Enhanced memory-efficient concatenation using M1 memory optimizer
            self.logger.info('🔄 Concatenating dataframes efficiently...')
            unified_data = enhanced_memory_efficient_concat(dataframes, self.chunk_size, self.utility_manager)
            
            # Enhanced fast-fail: Check data size
            is_valid, error_msg = enhanced_fast_fail_data_size_check(unified_data, self.min_rows, self.utility_manager)
            if not is_valid:
                self.logger.error(f'❌ {error_msg}')
                self.monitor.complete_function_call(call_id, error=DataQualityError(error_msg))
                return None
            
            # Enhanced fast-fail: Check schema
            is_valid, error_msg = enhanced_fast_fail_schema_check(unified_data, self.utility_manager)
            if not is_valid:
                self.logger.error(f'❌ {error_msg}')
                self.monitor.complete_function_call(call_id, error=ValidationError(error_msg))
                return None
            
            # Sort by timestamp using common_utilities
            common_utils = self.utility_manager.common_utils
            unified_data = common_utils['safe_dataframe_operation'](unified_data, 'sort_values', by='timestamp')
            unified_data = common_utils['safe_dataframe_operation'](unified_data, 'reset_index', drop=True)
            
            self.logger.info(f'✅ Successfully read unified data: {len(unified_data)} rows')
            self._log_step_timing('read_unified_data_enhanced', step_start)
            
            self.monitor.complete_function_call(call_id, unified_data)
            return unified_data
            
        except Exception as e:
            self.logger.exception(f'❌ Error reading unified data: {e}')
            self.monitor.complete_function_call(call_id, error=e)
            return None
    
    async def validate_data_quality_enhanced(self, data: pd.DataFrame, symbol: str, exchange: str) -> Dict[str, Any]:
        """Validate data quality using comprehensive utility integration."""
        step_start = time.time()
        call_id = self.monitor.start_function_call(self.validate_data_quality_enhanced, (data, symbol, exchange), {})
        
        try:
            self.logger.info('🔍 Validating data quality with comprehensive utility integration...')
            
            # Use data_processing_utils for comprehensive validation
            data_processing_utils = self.utility_manager.data_processing_utils
            
            # Create DataFrame validator
            validator = data_processing_utils['DataFrameValidator']()
            
            # Define schema for validation
            schema = {
                'required_columns': ['open', 'high', 'low', 'close', 'volume', 'timestamp'],
                'data_types': {
                    'open': 'float64',
                    'high': 'float64',
                    'low': 'float64',
                    'close': 'float64',
                    'volume': 'float64',
                    'timestamp': 'datetime64[ns]'
                }
            }
            
            # Comprehensive validation using data_processing_utils
            validation_report = validator.validate_dataframe(data, schema)
            
            # Enhanced vectorized validations
            price_validation = enhanced_vectorized_price_validation(data, self.utility_manager)
            timestamp_validation = enhanced_vectorized_timestamp_validation(data, self.utility_manager)
            volume_validation = enhanced_vectorized_volume_validation(data, self.utility_manager)
            
            # Use common_utilities for data quality metrics
            common_utils = self.utility_manager.common_utils
            data_quality_metrics = common_utils['calculate_data_quality_metrics'](data)
            
            # Combine results
            validation_results = {
                'passed': validation_report.summary['data_quality_score'] >= 70,
                'issues': [issue.description for issue in validation_report.issues],
                'warnings': [issue.description for issue in validation_report.issues if issue.level.value == 'warning'],
                'data_info': {
                    'rows': len(data),
                    'columns': list(data.columns),
                    'date_range': {
                        'start': data['timestamp'].min() if 'timestamp' in data.columns else None,
                        'end': data['timestamp'].max() if 'timestamp' in data.columns else None,
                    },
                    'memory_usage': data.memory_usage(deep=True).sum() / 1024 / 1024,
                },
                'quality_score': validation_report.summary['data_quality_score'],
                'price_validation': price_validation,
                'timestamp_validation': timestamp_validation,
                'volume_validation': volume_validation,
                'data_quality_metrics': data_quality_metrics,
                'validation_report': validation_report
            }
            
            # Enhanced quality score calculation using math_validation
            math_validation = self.utility_manager.math_validation
            
            # Check price validation results
            if price_validation['negative_prices'] > 0:
                validation_results['passed'] = False
                validation_results['issues'].append(f"Negative prices: {price_validation['negative_prices']} rows")
                validation_results['quality_score'] = math_validation['safe_divide'](
                    validation_results['quality_score'] - 20, 1, default=0
                )
            
            if price_validation['infinite_prices'] > 0:
                validation_results['passed'] = False
                validation_results['issues'].append(f"Infinite prices: {price_validation['infinite_prices']} rows")
                validation_results['quality_score'] = math_validation['safe_divide'](
                    validation_results['quality_score'] - 20, 1, default=0
                )
            
            if price_validation['nan_prices'] > 0:
                validation_results['warnings'].append(f"NaN prices: {price_validation['nan_prices']} rows")
                validation_results['quality_score'] = math_validation['safe_divide'](
                    validation_results['quality_score'] - 10, 1, default=0
                )
            
            if price_validation['ohlc_inconsistencies'] > 0:
                validation_results['warnings'].append(f"OHLC inconsistencies: {price_validation['ohlc_inconsistencies']} rows")
                validation_results['quality_score'] = math_validation['safe_divide'](
                    validation_results['quality_score'] - 5, 1, default=0
                )
            
            # Check timestamp validation results
            if timestamp_validation['duplicate_timestamps'] > 0:
                duplicate_ratio = math_validation['safe_divide'](
                    timestamp_validation['duplicate_timestamps'], len(data), default=0
                )
                if duplicate_ratio > self.max_duplicate_ratio:
                    validation_results['passed'] = False
                    validation_results['issues'].append(f"Too many duplicate timestamps: {timestamp_validation['duplicate_timestamps']} ({duplicate_ratio:.2%})")
                    validation_results['quality_score'] = math_validation['safe_divide'](
                        validation_results['quality_score'] - 15, 1, default=0
                    )
                else:
                    validation_results['warnings'].append(f"Duplicate timestamps: {timestamp_validation['duplicate_timestamps']} ({duplicate_ratio:.2%})")
                    validation_results['quality_score'] = math_validation['safe_divide'](
                        validation_results['quality_score'] - 5, 1, default=0
                    )
            
            if timestamp_validation['non_monotonic']:
                validation_results['passed'] = False
                validation_results['issues'].append("Non-monotonic timestamp ordering")
                validation_results['quality_score'] = math_validation['safe_divide'](
                    validation_results['quality_score'] - 20, 1, default=0
                )
            
            if timestamp_validation['large_gaps'] > 0:
                validation_results['warnings'].append(f"Large time gaps (>0.5s): {timestamp_validation['large_gaps']} gaps, max: {timestamp_validation['max_gap_seconds']:.2f}s")
                validation_results['quality_score'] = math_validation['safe_divide'](
                    validation_results['quality_score'] - 5, 1, default=0
                )
            
            # Check volume validation results
            if volume_validation['negative_volumes'] > 0:
                validation_results['passed'] = False
                validation_results['issues'].append(f"Negative volumes: {volume_validation['negative_volumes']} rows")
                validation_results['quality_score'] = math_validation['safe_divide'](
                    validation_results['quality_score'] - 15, 1, default=0
                )
            
            if volume_validation['extreme_high_volumes'] > 0:
                validation_results['warnings'].append(f"Extreme high volumes: {volume_validation['extreme_high_volumes']} rows")
                validation_results['quality_score'] = math_validation['safe_divide'](
                    validation_results['quality_score'] - 5, 1, default=0
                )
            
            if volume_validation['extreme_low_volumes'] > 0:
                validation_results['warnings'].append(f"Extreme low volumes: {volume_validation['extreme_low_volumes']} rows")
                validation_results['quality_score'] = math_validation['safe_divide'](
                    validation_results['quality_score'] - 5, 1, default=0
                )
            
            # Ensure quality score is not negative
            validation_results['quality_score'] = max(0, validation_results['quality_score'])
            
            self.logger.info(f'✅ Data quality validation completed')
            self.logger.info(f"   - Rows: {validation_results['data_info']['rows']}")
            self.logger.info(f"   - Memory usage: {validation_results['data_info']['memory_usage']:.2f} MB")
            self.logger.info(f"   - Quality score: {validation_results['quality_score']:.2f}")
            self.logger.info(f"   - Issues: {len(validation_results['issues'])}")
            self.logger.info(f"   - Warnings: {len(validation_results['warnings'])}")
            
            self._log_step_timing('validate_data_quality_enhanced', step_start)
            self.monitor.complete_function_call(call_id, validation_results)
            return validation_results
            
        except Exception as e:
            self.logger.exception(f'❌ Error during data quality validation: {e}')
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
    
    async def execute(self, symbol: str, exchange: str, timeframe: str, data_dir: str, **kwargs) -> Dict[str, Any]:
        """Execute the enhanced data reading step."""
        self.logger.info('🚀 Starting Enhanced Step 2: Data Reading and Validation')
        
        try:
            # Read unified data with comprehensive utility integration
            unified_data = await self.read_unified_data_enhanced(symbol, exchange, timeframe, data_dir)
            if unified_data is None:
                return {'success': False, 'error': 'Failed to read unified data'}
            
            # Validate data quality with comprehensive utility integration
            validation_results = await self.validate_data_quality_enhanced(unified_data, symbol, exchange)
            
            if not validation_results['passed']:
                self.logger.error('❌ Data quality validation failed')
                self.logger.error(f"   Issues: {validation_results['issues']}")
                return {'success': False, 'error': 'Data quality validation failed', 'validation_results': validation_results}
            
            # Save validated data using serialization_utils
            serialization_utils = self.utility_manager.serialization_utils
            processed_dir = Path(data_dir) / 'processed' / exchange / symbol
            processed_dir.mkdir(parents=True, exist_ok=True)
            output_file = f'{exchange}_{symbol}_{timeframe}_validated_data.parquet'
            output_path = processed_dir / output_file
            
            # Use serialization_utils for safe saving
            serialization_utils['save_parquet'](unified_data, output_path, compression='snappy', index=False)
            
            # Save validation report using serialization_utils
            report_file = f'{exchange}_{symbol}_{timeframe}_validation_report.json'
            report_path = processed_dir / report_file
            serialization_utils['save_json'](validation_results, report_path, indent=2)
            
            self.logger.info(f'✅ Enhanced Step 2 completed successfully')
            self.logger.info(f'   - Validated data saved to: {output_path}')
            self.logger.info(f'   - Validation report saved to: {report_path}')
            self.logger.info(f'   - Total execution time: {time.time() - self.start_time:.2f} seconds')
            
            # Get performance summary with utility usage
            performance_summary = self.monitor.get_performance_summary()
            
            # Get utility manager health status
            utility_health = self.utility_manager.get_health_status()
            
            return {
                'success': True,
                'data_path': str(output_path),
                'report_path': str(report_path),
                'validation_results': validation_results,
                'step_timings': self.step_timings,
                'performance_summary': performance_summary,
                'utility_health': utility_health
            }
            
        except Exception as e:
            self.logger.exception(f'❌ Error in Enhanced Step 2: {e}')
            return {'success': False, 'error': str(e)}

# Entry point functions
async def run_step_enhanced(symbol: str, exchange: str, timeframe: str, data_dir: str = None, **kwargs) -> Dict[str, Any]:
    """Enhanced entry point for Step 2: Data Reading and Validation with comprehensive utility integration."""
    if data_dir is None:
        data_dir = 'data_cache'
    
    config = {
        'max_workers': kwargs.get('max_workers', 4),
        'chunk_size': kwargs.get('chunk_size', 10000),
        'min_rows': kwargs.get('min_rows', 1000),
        'max_duplicate_ratio': kwargs.get('max_duplicate_ratio', 0.01),
        'max_gap_seconds': kwargs.get('max_gap_seconds', 0.5),
        **kwargs
    }
    
    step = EnhancedDataReadingStep(config)
    await step.initialize()
    result = await step.execute(symbol, exchange, timeframe, data_dir, **kwargs)
    
    if result['success']:
        logging.info('✅ Enhanced Step 2: Data Reading and Validation completed successfully')
    else:
        logging.error(f"❌ Enhanced Step 2: Data Reading and Validation failed: {result.get('error', 'Unknown error')}")
    
    return result

if __name__ == '__main__':
    async def test():
        test_symbol = 'ETHUSDT'
        test_exchange = 'BINANCE'
        test_timeframe = '1m'
        result = await run_step_enhanced(symbol=test_symbol, exchange=test_exchange, timeframe=test_timeframe, data_dir='data_cache')
        print(f'Result: {result}')
    
    asyncio.run(test())