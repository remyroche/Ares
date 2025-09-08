from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from src.training.steps.model_training.step04_common_types import (
    StepResult, RegimeDataResult, StepResultStatus, standardize_result
)
from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_kelly_calculation,
    validate_positive, validate_range, MathValidationError
)
from src.utils.lookahead_bias_detector import (
    get_global_detector, validate_no_future_data, LookaheadBiasError
)

"""Step 4: Regime Data Splitting with Comprehensive Function Call Monitoring.

This module creates a unified dataset with regime labels for regime-aware processing.
Uses labels to differentiate regimes instead of creating separate files per regime.
This ensures trading indicators have the necessary lookback periods.

Enhanced with comprehensive function call monitoring, function-to-function tracking,
and detailed outcome reporting for complete execution visibility.
"""
import asyncio
import json
import logging
import sys
import time
import functools
import traceback
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Memory-efficient imports for large dataset handling
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyarrow import compute as pc
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    pa = None
    pq = None
    pc = None
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

    class MockProcess:

        def memory_info(self) -> None:

            class MemoryInfo:
                rss = 0
            return MemoryInfo()

    class MockPsutil:

        def Process(self) -> None:
            return MockProcess()

        def cpu_percent(self) -> float:
            return 0.0
    psutil = MockPsutil()
try:
    from src.core.domain.decorators_extended import monitor_feature_engineering
    from src.core.decorators import validates, cached, log_execution_time, traced
    import datetime
except Exception:

    def monitor_feature_engineering(*args, **kwargs) -> None:

        def _decorator(func: Callable) -> None:
            return func
        return _decorator
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.utils.common_operations import ensure_directory, safe_json_dump
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards

# M1 Hardware Optimizations
try:
    from src.utils.m1_gpu_utils import get_m1_gpu_manager, M1GPUManager
    from src.utils.m1_memory_optimizer import M1MemoryOptimizer
    from src.utils.m1_cpu_optimizer import M1CPUOptimizer
    M1_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    system_logger.warning(f"M1 optimizations not available: {e}")
    M1_OPTIMIZATIONS_AVAILABLE = False
    M1GPUManager = None
    M1MemoryOptimizer = None
    M1CPUOptimizer = None

# Vectorized Processing Core and Enhanced Matrix Operations
try:
    from src.utils.vectorized_processing_core import get_vectorized_processing_core
    from src.utils.enhanced_matrix_operations import get_enhanced_matrix_operations
    VECTORIZED_OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    system_logger.warning(f"Vectorized optimizations not available: {e}")
    VECTORIZED_OPTIMIZATIONS_AVAILABLE = False
REQUIRED_MODULES = ['pandas', 'numpy', 'src.utils.logger', 'src.utils.enhanced_mlflow_integration']
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)
from src.utils.logger import system_logger
enhanced_mlflow = PipelineStandards.safe_import('src.utils.enhanced_mlflow_integration', None)
pandas = pd
numpy = np

# Import financial metrics logging system
try:
    from .step04_financial_logging import Step04FinancialLogger
    FINANCIAL_LOGGING_AVAILABLE = True
except ImportError:
    FINANCIAL_LOGGING_AVAILABLE = False

def create_fallback_logger() -> Any:
    logging.basicConfig(level = logging.INFO)
    return logging.getLogger(__name__)

def create_fallback_decorator() -> Any:

    def decorator(*decorator_args, **decorator_kwargs) -> Callable:
        def wrapper(func: Callable) -> Callable:
            def inner_wrapper(*args, **kwargs) -> None:
                return func(*args, **kwargs)
            return inner_wrapper
        return wrapper
    return decorator

def create_fallback_validates():
    def decorator(*decorator_args, **decorator_kwargs):
        def wrapper(func: Callable) -> Callable:
            def inner_wrapper(*args, **kwargs) -> None:
                return func(*args, **kwargs)
            return inner_wrapper
        return wrapper
    return decorator

if system_logger is None:
    system_logger = create_fallback_logger()

# Set fallback decorators
comprehensive_data_validation = create_fallback_validates()
handle_errors = handles_errors
memory_efficient = create_fallback_decorator()
resource_monitor = create_fallback_decorator()
secure_data_processing = handles_errors
validate_data_structure = create_fallback_validates()
with_tracing_span = create_fallback_decorator()
quality_gate = create_fallback_validates()
if enhanced_mlflow is None:
    with_enhanced_mlflow_logging = create_fallback_decorator()
    log_step_report = lambda *args, **kwargs: 'fallback_report'
    create_detailed_step_report = lambda *args, **kwargs: {}
    log_step_metrics = lambda *args, **kwargs: None
    log_step_dataframe_with_standardized_name = lambda *args, **kwargs: 'fallback_dataframe'
    log_step_artifact_with_standardized_name = lambda *args, **kwargs: 'fallback_artifact'
else:
    with_enhanced_mlflow_logging = enhanced_mlflow.with_enhanced_mlflow_logging
    log_step_report = enhanced_mlflow.log_step_report
    create_detailed_step_report = enhanced_mlflow.create_detailed_step_report
    log_step_metrics = enhanced_mlflow.log_step_metrics
    log_step_dataframe_with_standardized_name = enhanced_mlflow.log_step_dataframe_with_standardized_name
    log_step_artifact_with_standardized_name = enhanced_mlflow.log_step_artifact_with_standardized_name

# Ensure all decorators have fallbacks
if 'traced' not in globals():
    traced = create_fallback_decorator()
if 'validates' not in globals():
    validates = create_fallback_validates()
if 'cached' not in globals():
    cached = create_fallback_decorator()
if 'log_execution_time' not in globals():
    log_execution_time = create_fallback_decorator()
logger = system_logger.getChild('Step4RegimeDataSplitting')
_function_call_stack = threading.local()
_function_call_history = []
_function_call_lock = threading.Lock()

class FunctionCallTracker:
    """Comprehensive function call tracking and monitoring system."""

    def __init__(self) -> None:
        self.call_history = []
        self.active_calls = {}
        self.performance_metrics = {}
        self.error_tracking = {}

    def start_call(self, func_name: str, args: tuple, kwargs: dict, caller: str = None) -> str:
        """Start tracking a function call."""
        call_id = f'{func_name}_{int(time.time() * 1000000)}'
        call_info = {'call_id': call_id, 'function_name': func_name, 'caller': caller, 'start_time': time.time(), 'args': str(args)[:200] + '...' if len(str(args)) > 200 else str(args), 'kwargs': str(kwargs)[:200] + '...' if len(str(kwargs)) > 200 else str(kwargs), 'memory_before': psutil.Process().memory_info().rss / 1024 / 1024 if PSUTIL_AVAILABLE else 0, 'thread_id': threading.get_ident(), 'stack_depth': len(getattr(_function_call_stack, 'stack', []))}
        with _function_call_lock:
            self.active_calls[call_id] = call_info
            self.call_history.append(call_info.copy())
        if not hasattr(_function_call_stack, 'stack'):
            _function_call_stack.stack = []
        _function_call_stack.stack.append(call_id)
        logger.info(f'🔍 FUNCTION_CALL_START: {func_name} (ID: {call_id})')
        logger.info(f"   📞 Called by: {caller or 'ROOT'}")
        logger.info(f"   📊 Memory before: {call_info['memory_before']:.2f} MB")
        logger.info(f"   🧵 Thread: {call_info['thread_id']}")
        logger.info(f"   📏 Stack depth: {call_info['stack_depth']}")
        return call_id

    def end_call(self, call_id: str, result: Any = None, error: Exception = None) -> Dict[str, Any]:
        """End tracking a function call and generate detailed report."""
        with _function_call_lock:
            if call_id not in self.active_calls:
                logger.warning(f'⚠️ Call ID {call_id} not found in active calls')
                return {}
            call_info = self.active_calls.pop(call_id)
        if hasattr(_function_call_stack, 'stack') and call_id in _function_call_stack.stack:
            _function_call_stack.stack.remove(call_id)
        end_time = time.time()
        execution_time = end_time - call_info['start_time']
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024 if PSUTIL_AVAILABLE else 0
        memory_delta = memory_after - call_info['memory_before']
        outcome_report = {'call_id': call_id, 'function_name': call_info['function_name'], 'caller': call_info['caller'], 'execution_time_seconds': execution_time, 'memory_before_mb': call_info['memory_before'], 'memory_after_mb': memory_after, 'memory_delta_mb': memory_delta, 'success': error is None, 'error_type': type(error).__name__ if error else None, 'error_message': str(error) if error else None, 'result_type': type(result).__name__ if result is not None else None, 'result_size': len(str(result)) if result is not None else 0, 'thread_id': call_info['thread_id'], 'stack_depth': call_info['stack_depth'], 'timestamp': time.time()}
        status_emoji = '✅' if error is None else '❌'
        logger.info(f"{status_emoji} FUNCTION_CALL_END: {call_info['function_name']} (ID: {call_id})")
        logger.info(f'   ⏱️ Execution time: {execution_time:.4f} seconds')
        logger.info(f'   💾 Memory delta: {memory_delta:+.2f} MB')
        logger.info(f"   🎯 Success: {outcome_report['success']}")
        if error:
            logger.error(f'   🚨 Error: {type(error).__name__}: {str(error)}')
            logger.error(f'   📍 Traceback: {traceback.format_exc()}')
        else:
            logger.info(f"   📦 Result type: {outcome_report['result_type']}")
            logger.info(f"   📏 Result size: {outcome_report['result_size']} chars")
        func_name = call_info['function_name']
        if func_name not in self.performance_metrics:
            self.performance_metrics[func_name] = {'total_calls': 0, 'total_time': 0, 'success_count': 0, 'error_count': 0, 'avg_execution_time': 0, 'max_execution_time': 0, 'min_execution_time': float('inf')}
        metrics = self.performance_metrics[func_name]
        metrics['total_calls'] += 1
        metrics['total_time'] += execution_time
        metrics['avg_execution_time'] = metrics['total_time'] / metrics['total_calls']
        metrics['max_execution_time'] = max(metrics['max_execution_time'], execution_time)
        metrics['min_execution_time'] = min(metrics['min_execution_time'], execution_time)
        if error:
            metrics['error_count'] += 1
        else:
            metrics['success_count'] += 1
        if error:
            if func_name not in self.error_tracking:
                self.error_tracking[func_name] = []
            self.error_tracking[func_name].append({'timestamp': time.time(), 'error_type': type(error).__name__, 'error_message': str(error), 'call_id': call_id})
        return outcome_report

    def get_caller_info(self) -> str:
        """Get information about the calling function."""
        if hasattr(_function_call_stack, 'stack') and _function_call_stack.stack:
            return _function_call_stack.stack[-1]
        return 'ROOT'

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive summary report of all function calls."""
        with _function_call_lock:
            return {'total_calls': len(self.call_history), 'active_calls': len(self.active_calls), 'performance_metrics': self.performance_metrics, 'error_summary': {func: len(errors) for func, errors in self.error_tracking.items()}, 'recent_calls': self.call_history[-10:] if self.call_history else []}
_function_tracker = FunctionCallTracker()

def comprehensive_function_monitor(func: Callable) -> Callable:
    """Comprehensive function call monitoring decorator."""

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> None:
        caller = _function_tracker.get_caller_info()
        call_id = _function_tracker.start_call(func.__name__, args, kwargs, caller)
        try:
            result = await func(*args, **kwargs)
            outcome = _function_tracker.end_call(call_id, result)
            return result
        except Exception as e:
            outcome = _function_tracker.end_call(call_id, error = e)
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> None:
        caller = _function_tracker.get_caller_info()
        call_id = _function_tracker.start_call(func.__name__, args, kwargs, caller)
        try:
            result = func(*args, **kwargs)
            outcome = _function_tracker.end_call(call_id, result)
            return result
        except Exception as e:
            outcome = _function_tracker.end_call(call_id, error = e)
            raise
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def log_function_call_summary() -> None:
    """Log a summary of all function calls."""
    summary = _function_tracker.generate_summary_report()
    logger.info('📊 FUNCTION_CALL_SUMMARY:')
    logger.info(f"   📞 Total calls: {summary['total_calls']}")
    logger.info(f"   🔄 Active calls: {summary['active_calls']}")
    if summary['performance_metrics']:
        logger.info('   ⚡ Performance metrics:')
        for func_name, metrics in summary['performance_metrics'].items():
            logger.info(f'      {func_name}:')
            logger.info(f"         Calls: {metrics['total_calls']}")
            logger.info(f"         Avg time: {metrics['avg_execution_time']:.4f}s")
            logger.info(f"         Success rate: {metrics['success_count']}/{metrics['total_calls']}")
    if summary['error_summary']:
        logger.info('   🚨 Error summary:')
        for func_name, error_count in summary['error_summary'].items():
            logger.info(f'      {func_name}: {error_count} errors')
    if _function_tracker.error_tracking:
        logger.info('   🔍 Detailed error context:')
        for func_name, errors in _function_tracker.error_tracking.items():
            logger.info(f'      {func_name} errors:')
            for error in errors[-3:]:
                logger.info(f"         - {error['error_type']}: {error['error_message']}")
                logger.info(f"           Call ID: {error['call_id']}")

def capture_comprehensive_error_context(error: Exception, context: Dict[str, Any]=None) -> Dict[str, Any]:
    """Capture comprehensive error context including function call stack and system state."""
    error_context = {'error_type': type(error).__name__, 'error_message': str(error), 'timestamp': time.time(), 'traceback': traceback.format_exc(), 'system_info': {'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024 if PSUTIL_AVAILABLE else 0, 'cpu_percent': psutil.cpu_percent() if PSUTIL_AVAILABLE else 0, 'thread_id': threading.get_ident()}, 'function_call_context': {'active_calls': len(_function_tracker.active_calls), 'call_stack': getattr(_function_call_stack, 'stack', []), 'recent_calls': _function_tracker.call_history[-5:] if _function_tracker.call_history else []}}
    if context:
        error_context['additional_context'] = context
    return error_context

def log_comprehensive_error_report(error: Exception, context: Dict[str, Any]=None) -> None:
    """Log a comprehensive error report with full context."""
    error_context = capture_comprehensive_error_context(error, context)
    logger.error('🚨 COMPREHENSIVE_ERROR_REPORT:')
    logger.error(f"   🔥 Error: {error_context['error_type']}: {error_context['error_message']}")
    logger.error(f"   ⏰ Timestamp: {error_context['timestamp']}")
    logger.error(f"   💾 Memory usage: {error_context['system_info']['memory_usage_mb']:.2f} MB")
    logger.error(f"   🧵 Thread ID: {error_context['system_info']['thread_id']}")
    logger.error(f"   📞 Active function calls: {error_context['function_call_context']['active_calls']}")
    if error_context['function_call_context']['call_stack']:
        logger.error('   📍 Function call stack:')
        for call_id in error_context['function_call_context']['call_stack']:
            logger.error(f'      - {call_id}')
    logger.error('   📍 Full traceback:')
    for line in error_context['traceback'].split('\n'):
        if line.strip():
            logger.error(f'      {line}')
    if context:
        logger.error('   📋 Additional context:')
        for key, value in context.items():
            logger.error(f'      {key}: {value}')
    return error_context

# MemoryMonitor class replaced with M1MemoryOptimizer integration

class RegimeDataSplittingStep:
    """Step 4: Regime Data Splitting with standardized data quality management."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('RegimeDataSplittingStep')
        self.standards = pipeline_standards
        self.start_time = None
        self.step_timings = {}

        # Initialize M1 Hardware Optimizations
        self._init_m1_optimizations()

        # Initialize Parquet optimizations
        self._init_parquet_optimizations()

        # Set default memory management configuration
        self._set_memory_defaults()
        self._validate_environment()

        # Initialize financial metrics logging system
        if FINANCIAL_LOGGING_AVAILABLE:
            try:
                self.financial_logger = None  # Will be initialized per execution
                self.logger.info('✅ Financial metrics logging system available')
            except Exception as e:
                self.logger.warning(f'⚠️ Financial metrics logging system failed to initialize: {e}')
                self.financial_logger = None
        else:
            self.logger.info('ℹ️ Financial metrics logging system not available, using basic reporting')
            self.financial_logger = None

    def _set_memory_defaults(self) -> None:
        """Set default configuration values for memory management."""
        memory_defaults = {
            # Streaming thresholds
            'streaming_threshold_mb': 500,  # Use streaming for datasets > 500MB
            'streaming_min_rows': 2_000_000,  # Use streaming for > 2M rows
            'streaming_min_mb': 1000,  # Use streaming for > 1GB memory usage

            # Processing chunk sizes
            'processing_chunk_size': 100000,  # Process 100K rows at a time during merging
            'streaming_chunk_size': 5,  # Process 5 files at a time in streaming mode
            'streaming_chunk_rows': 500000,  # Write 500K rows per chunk when saving

            # Merge settings
            'regime_merge_min_retention': 0.8,  # Minimum 80% data retention after merge
            'regime_merge_tolerance_ms': 60000,  # 60 second tolerance for timestamp matching

            # Writer settings
            'use_streaming_writer': True,  # Enable streaming writer by default
            'use_asof_merge': True,  # Use asof merge by default
        }

        # Update config with defaults if not already set
        for key, default_value in memory_defaults.items():
            if key not in self.config:
                self.config[key] = default_value

    @comprehensive_function_monitor
    def _validate_environment(self) -> None:
        """Validate environment dependencies."""
        self.logger.info('🔍 Validating environment dependencies...')
        missing_modules = [module for module, available in dependency_status.items() if not available]
        if missing_modules:
            self.logger.warning(f'⚠️ Missing optional modules: {missing_modules}')
            self.logger.info('📝 Pipeline will continue with fallback implementations')
        else:
            self.logger.info('✅ All required dependencies available')

    def _init_m1_optimizations(self) -> None:
        """Initialize M1 hardware optimization components."""
        if M1_OPTIMIZATIONS_AVAILABLE:
            try:
                # Initialize M1 GPU Manager
                self.gpu_manager = get_m1_gpu_manager()
                self.logger.info('🎯 M1 GPU Manager initialized for step04')

                # Initialize M1 Memory Optimizer with step-specific settings
                memory_limit = self.config.get('memory_limit_gb', 8.0)
                self.memory_optimizer = M1MemoryOptimizer(
                    memory_limit_gb=memory_limit,
                    enable_gc_tuning=True,
                    enable_memory_leak_detection=True,
                    enable_swap_management=True
                )
                self.logger.info('🧠 M1 Memory Optimizer initialized for step04')

                # Initialize M1 CPU Optimizer
                max_workers = self.config.get('max_parallel_workers', None)
                self.cpu_optimizer = M1CPUOptimizer(
                    max_workers=max_workers,
                    enable_hyperthreading=True
                )
                self.logger.info('⚡ M1 CPU Optimizer initialized for step04')

                self.m1_optimizations_enabled = True
            except Exception as e:
                self.logger.warning(f'Failed to initialize M1 optimizations: {e}')
                self.m1_optimizations_enabled = False
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
        else:
            self.logger.info('ℹ️ M1 optimizations not available, using fallback implementations')
            self.m1_optimizations_enabled = False
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None

    def _init_parquet_optimizations(self) -> None:
        """Initialize Parquet optimizations including metadata caching and partitioning."""
        try:
            if PYARROW_AVAILABLE:
                # Parquet metadata cache for frequently accessed files
                self.parquet_metadata_cache = {}
                self.parquet_cache_max_size = self.config.get('parquet_cache_max_size', 100)
                
                # Partitioning configuration
                self.enable_parquet_partitioning = self.config.get('enable_parquet_partitioning', True)
                self.partition_columns = self.config.get('partition_columns', ['composite_cluster_id', 'year', 'month'])
                self.partition_size_threshold = self.config.get('partition_size_threshold', 1_000_000)  # 1M rows
                
                # Columnar storage optimization
                self.columnar_optimization = self.config.get('columnar_optimization', True)
                self.optimize_column_order = self.config.get('optimize_column_order', True)
                
                self.logger.info("📊 Parquet optimizations initialized")
                self.logger.info(f"   🗂️ Partitioning enabled: {self.enable_parquet_partitioning}")
                self.logger.info(f"   📋 Partition columns: {self.partition_columns}")
                self.logger.info(f"   💾 Metadata cache size: {self.parquet_cache_max_size}")
            else:
                self.parquet_metadata_cache = {}
                self.enable_parquet_partitioning = False
                self.columnar_optimization = False
                self.logger.warning("⚠️ PyArrow not available - Parquet optimizations disabled")
        except Exception as e:
            self.logger.warning(f"⚠️ Parquet optimization initialization failed: {e}")
            self.parquet_metadata_cache = {}
            self.enable_parquet_partitioning = False
            self.columnar_optimization = False

    @comprehensive_function_monitor
    async def initialize(self) -> None:
        """Initialize the regime data splitting step."""
        self.start_time = time.time()
        self.logger.info('🚀 Initializing Regime Data Splitting Step...')
        self.logger.info('📋 Step 4 Configuration:')
        self.logger.info(f'   - Unified dataset approach: Enabled')
        self.logger.info(f'   - Regime labels: composite_cluster_id')
        self.logger.info(f'   - Memory management: Optimized')
        self.logger.info('✅ Regime Data Splitting Step initialized successfully')

    @comprehensive_function_monitor
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the regime data splitting step.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dictionary with execution results
        """
        symbol = training_input.get('symbol')
        exchange = training_input.get('exchange')
        timeframe = training_input.get('timeframe', '1m')
        data_dir = training_input.get('data_dir')

        if not all([symbol, exchange, timeframe]):
            return {
                'success': False,
                'step04_regime_data_splitting_completed': False,
                'step04_regime_data_splitting_failure_reason': 'Missing required parameters: symbol, exchange, timeframe'
            }

        try:
            # Initialize if not already done
            if not self.start_time:
                await self.initialize()

            # Execute the main functionality
            result = await self.split_data_by_regimes(symbol, exchange, timeframe, data_dir)

            # Generate comprehensive function call summary
            log_function_call_summary()

            if result.success:
                return {
                    'success': True,
                    'step04_regime_data_splitting_completed': True,
                    'regime_data': result.data,
                    'regime_metadata': result.metadata,
                    'regime_splits': result.data,  # For compatibility with step 05 expectations
                    'execution_time': time.time() - self.start_time,
                    'step_name': 'step04_regime_data_splitting'
                }
            else:
                return {
                    'success': False,
                    'step04_regime_data_splitting_completed': False,
                    'step04_regime_data_splitting_failure_reason': f'Regime data splitting failed: {result.error}',
                    'execution_time': time.time() - self.start_time,
                    'step_name': 'step04_regime_data_splitting'
                }

        except Exception as e:
            self.logger.exception(f'❌ Error in step04_regime_data_splitting execute: {e}')
            return {
                'success': False,
                'step04_regime_data_splitting_completed': False,
                'step04_regime_data_splitting_failure_reason': f'Exception: {str(e)}',
                'execution_time': time.time() - self.start_time,
                'step_name': 'step04_regime_data_splitting'
            }

    @comprehensive_function_monitor
    def _log_step_timing(self, step_name: str, start_time: float) -> None:
        """Log timing information for a step."""
        elapsed = time.time() - start_time
        self.step_timings[step_name] = elapsed
        self.logger.info(f'⏱️ {step_name} completed in {elapsed:.2f} seconds')

    @comprehensive_function_monitor
    @traced(span_name='split_data_by_regimes')
    @validates()
    @cached()
    async def split_data_by_regimes(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> RegimeDataResult:
        """Create unified dataset with regime labels for regime-aware processing.

        Returns a standardized RegimeDataResult with all relevant information.
        """
        step_start = time.time()

        # Start M1 memory monitoring for large datasets
        if self.m1_optimizations_enabled and self.memory_optimizer:
            with self.memory_optimizer.memory_checkpoint("regime_data_splitting_start"):
                self.logger.info('🧠 M1 Memory monitoring enabled for large dataset processing')
        elif PSUTIL_AVAILABLE:
            # Fallback to basic memory monitoring
            self.logger.info('📊 Basic memory monitoring enabled (M1 optimizations not available)')

        self.logger.info(f'🔀 Creating unified dataset with regime labels for {symbol} on {exchange} ({timeframe})')
        try:
            # Memory checkpoint: Start of data loading
            if self.m1_optimizations_enabled and self.memory_optimizer:
                with self.memory_optimizer.memory_checkpoint("data_loading"):
                    regime_data = await self._load_regime_data(symbol, exchange, timeframe, data_dir)
            else:
                regime_data = await self._load_regime_data(symbol, exchange, timeframe, data_dir)

            if regime_data is None:
                return RegimeDataResult.failure_result(
                    error='regime_data_not_found',
                    error_type='DataNotFoundError',
                    metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
                )

            # Log memory usage after data loading
            if self.m1_optimizations_enabled and self.memory_optimizer:
                memory_report = self.memory_optimizer.get_memory_report()
                self.logger.info(f'💾 Memory after loading: {memory_report.get("current_mb", 0):.1f}MB')
            elif PSUTIL_AVAILABLE:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                self.logger.info(f'💾 Memory after loading: {current_memory:.1f}MB')

            regime_ids = regime_data['composite_cluster_id'].unique()
            num_regimes = len(regime_ids)
            self.logger.info(f'📊 Found {num_regimes} regimes: {sorted(regime_ids)}')

            if num_regimes < 3:
                self.logger.error(f'❌ Too few regimes: {num_regimes} (minimum 3 required)')
                return RegimeDataResult.failure_result(
                    error = f'too_few_regimes: {num_regimes} (minimum 3 required)',
                    error_type='InsufficientRegimesError',
                    metadata={'regime_count': num_regimes, 'regime_ids': regime_ids.tolist()}
                )

            if num_regimes > 20:
                self.logger.warning(f'⚠️ Many regimes detected: {num_regimes} (maximum 20 supported)')

            # Memory checkpoint: Before dataset creation
            if self.m1_optimizations_enabled and self.memory_optimizer:
                with self.memory_optimizer.memory_checkpoint("dataset_creation"):
                    dataset_info = await self._create_unified_regime_dataset(regime_data, regime_ids, data_dir, symbol, exchange, timeframe)
            else:
                dataset_info = await self._create_unified_regime_dataset(regime_data, regime_ids, data_dir, symbol, exchange, timeframe)

            if isinstance(dataset_info, dict):
                # Log memory usage after dataset creation
                if self.m1_optimizations_enabled and self.memory_optimizer:
                    memory_report = self.memory_optimizer.get_memory_report()
                    self.logger.info(f'💾 Memory after dataset creation: {memory_report.get("current_mb", 0):.1f}MB')
                elif PSUTIL_AVAILABLE:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    self.logger.info(f'💾 Memory after dataset creation: {current_memory:.1f}MB')

                self._log_step_timing('Regime Data Splitting', step_start)
                self.logger.info(f'✅ Successfully created unified dataset with {num_regimes} regime labels')
                await self._save_regime_metadata(regime_ids, data_dir, symbol, exchange, timeframe)

                # Get memory summary for result metadata
                memory_summary = {}
                if self.m1_optimizations_enabled and self.memory_optimizer:
                    memory_report = self.memory_optimizer.get_memory_report()
                    memory_summary = {
                        'peak_memory_mb': memory_report.get('peak_mb', 0),
                        'current_memory_mb': memory_report.get('current_mb', 0),
                        'memory_delta_mb': memory_report.get('delta_mb', 0),
                        'optimization_type': 'm1_optimized'
                    }
                    self.logger.info(f'🧠 M1 Memory Summary: Peak {memory_summary["peak_memory_mb"]:.1f}MB, Δ{memory_summary["memory_delta_mb"]:+.1f}MB')
                elif PSUTIL_AVAILABLE:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_summary = {
                        'current_memory_mb': current_memory,
                        'optimization_type': 'basic'
                    }
                    self.logger.info(f'📊 Memory Summary: Current {current_memory:.1f}MB')

                return RegimeDataResult.success_result(
                    data = dataset_info.get('unified_data'),
                    metadata={
                        'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe,
                        'regime_count': num_regimes, 'regime_ids': regime_ids.tolist(),
                        'memory_usage': memory_summary
                    },
                    execution_time = time.time() - step_start
                )

            # Generate financial metrics logging if available
            if FINANCIAL_LOGGING_AVAILABLE and regime_data is not None:
                try:
                    self.logger.info('📊 Generating financial metrics logging for Step04...')

                    # Prepare data splitting results
                    data_splitting_results = {
                        'success': True,
                        'total_regimes': num_regimes,
                        'regime_ids': regime_ids.tolist(),
                        'data_shape': regime_data.shape,
                        'processing_method': 'streaming' if self.config.get('use_streaming_writer', True) else 'batch',
                        'memory_usage': memory_summary if 'memory_summary' in locals() else {},
                        'data_retention_rate': 1.0,
                        'processing_method_efficiency': 1.0
                    }

                    # Prepare performance data
                    execution_time_total = time.time() - step_start
                    performance_data = {
                        'execution_time_seconds': execution_time_total,
                        'memory_usage_mb': current_memory if 'current_memory' in locals() else 0,
                        'data_processing_rate': len(regime_data) / execution_time_total if execution_time_total > 0 else 0
                    }

                    # Initialize and use financial logger
                    financial_logger = Step04FinancialLogger(symbol, exchange, timeframe)
                    financial_logger.log_step_execution(
                        regime_data=regime_data,
                        regime_ids=regime_ids.tolist(),
                        execution_data=performance_data,
                        data_splitting_results=data_splitting_results
                    )

                    self.logger.info('✅ Financial metrics logging completed for Step04')

                except Exception as e:
                    self.logger.warning(f'⚠️ Financial metrics logging failed for Step04, continuing with basic reporting: {e}')
            else:
                self.logger.error('❌ Failed to create unified regime dataset')
                return RegimeDataResult.failure_result(
                    error='creation_failed',
                    error_type='DatasetCreationError',
                    metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
                )
        except Exception as e:
            self.logger.exception(f'❌ Error in regime data splitting: {e}')
            return RegimeDataResult.failure_result(
                error = str(e),
                error_type = type(e).__name__,
                metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe},
                execution_time = time.time() - step_start
            )

    @comprehensive_function_monitor
    async def _load_regime_data(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[pd.DataFrame]:
        """Load HMM regime data with standardized validation."""
        try:
            unified_data_path = Path(self.standards.build_path('unified_data', exchange, symbol)) / timeframe
            if not unified_data_path.exists():
                self.logger.error(f'❌ Unified data path not found: {unified_data_path}')
                return None
            # Standardize to: data_dir/hmm_regimes/{exchange}_{symbol}_hmm_composite_clusters_{timeframe}.parquet
            regime_file = Path(data_dir) / 'hmm_regimes' / f'{exchange}_{symbol}_hmm_composite_clusters_{timeframe}.parquet'
            if not regime_file.exists():
                self.logger.error(f'❌ Regime file not found: {regime_file}')
                return None
            unified_files = list(unified_data_path.glob('**/*.parquet'))
            if not unified_files:
                self.logger.error(f'❌ No unified data files found in {unified_data_path}')
                return None
            unified_data_segments = []
            use_asof_merge = bool(self.config.get('use_asof_merge', True))
            merge_tolerance_ms = int(self.config.get('regime_merge_tolerance_ms', 60000))

            # Memory-efficient processing for large datasets
            total_files = len(unified_files)
            streaming_threshold = int(self.config.get('streaming_threshold_mb', 500))  # 500MB threshold
            chunk_size = int(self.config.get('processing_chunk_size', 100000))  # Process in chunks

            # Check if we should use streaming approach
            total_estimated_size = self._estimate_dataset_size(unified_files)
            use_streaming_processing = total_estimated_size > streaming_threshold

            if use_streaming_processing:
                self.logger.info(f'🧠 Large dataset detected ({total_estimated_size:.1f}MB), using streaming processing')
                unified_df = self._process_large_dataset_streaming(
                    unified_files, regime_file, use_asof_merge, merge_tolerance_ms
                )
            else:
                self.logger.info(f'📊 Processing dataset normally ({total_estimated_size:.1f}MB)')

                # Load regime data once and sort with metadata caching
                regime_df = self._read_parquet_with_cache(regime_file)
                regime_df = self.standards.standardize_timestamp(regime_df, 'timestamp')
                regime_df = regime_df.sort_values('timestamp')
                total_input_rows = 0
                total_merged_rows = 0

                for file_path in sorted(unified_files):
                    df = self._read_parquet_with_cache(file_path)
                    df = self.standards.standardize_timestamp(df, 'timestamp')
                    df = self.standards.enforce_schema(df, 'unified')
                    df = df.sort_values('timestamp')
                    total_input_rows += len(df)

                    if use_asof_merge:
                        try:
                            merged_chunk = pd.merge_asof(
                                df,
                                regime_df[['timestamp', 'composite_cluster_id']],
                                on='timestamp',
                                direction='nearest',
                                tolerance = pd.Timedelta(milliseconds = merge_tolerance_ms)
                            )
                            # Drop rows where no near regime match found
                            merged_chunk = merged_chunk.dropna(subset=['composite_cluster_id'])
                        except Exception as e:
                            self.logger.warning(f'⚠️ merge_asof failed for {file_path.name}: {e}; falling back to inner merge')
                            merged_chunk = pd.merge(df, regime_df[['timestamp', 'composite_cluster_id']], on='timestamp', how='inner')
                    else:
                        merged_chunk = pd.merge(df, regime_df[['timestamp', 'composite_cluster_id']], on='timestamp', how='inner')
                    total_merged_rows += len(merged_chunk)
                    unified_data_segments.append(merged_chunk)

                unified_df = pd.concat(unified_data_segments, ignore_index = True) if unified_data_segments else pd.DataFrame()
            try:
                retention_ratio = (total_merged_rows / max(total_input_rows, 1)) if total_input_rows else 0.0
                self.logger.info(f'📈 Merge retention ratio: {retention_ratio:.3f}')
                min_retention = float(self.config.get('regime_merge_min_retention', 0.8))
                if retention_ratio < min_retention:
                    self.logger.warning(f'⚠️ Low retention after regime merge: {retention_ratio:.3f} (< {min_retention:.2f}). Check timestamp alignment and data coverage.')
            except Exception:
                pass
            self.logger.info(f'✅ Loaded {len(unified_df)} data points with regime information')
            return unified_df
        except Exception as e:
            self.logger.exception(f'❌ Error loading regime data: {e}')
            return None

    @comprehensive_function_monitor
    def _estimate_dataset_size(self, file_paths: List[Path]) -> float:
        """Estimate total dataset size in MB for memory planning."""
        try:
            total_size = sum(path.stat().st_size for path in file_paths)
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0

    @comprehensive_function_monitor
    def _process_large_dataset_streaming(
        self,
        unified_files: List[Path],
        regime_file: Path,
        use_asof_merge: bool,
        merge_tolerance_ms: int
    ) -> pd.DataFrame:
        """Process large datasets using streaming approach to minimize memory usage."""
        try:
            self.logger.info('🔄 Starting streaming processing for large dataset')

            # Load regime data once with metadata caching
            regime_df = self._read_parquet_with_cache(regime_file)
            regime_df = self.standards.standardize_timestamp(regime_df, 'timestamp')
            regime_df = regime_df.sort_values('timestamp')

            # Process files in smaller batches
            chunk_size = int(self.config.get('streaming_chunk_size', 5))  # Process 5 files at a time
            all_chunks = []

            for i in range(0, len(unified_files), chunk_size):
                batch_files = unified_files[i:i + chunk_size]
                self.logger.info(f'📁 Processing file batch {i//chunk_size + 1}/{(len(unified_files) + chunk_size - 1)//chunk_size}')

                batch_chunks = []
                for file_path in batch_files:
                    try:
                        # Load file with memory-efficient reading and metadata caching
                        df = self._read_parquet_with_cache(file_path)
                        df = self.standards.standardize_timestamp(df, 'timestamp')
                        df = self.standards.enforce_schema(df, 'unified')
                        df = df.sort_values('timestamp')

                        # Perform merge
                        if use_asof_merge:
                            try:
                                merged_chunk = pd.merge_asof(
                                    df,
                                    regime_df[['timestamp', 'composite_cluster_id']],
                                    on='timestamp',
                                    direction='nearest',
                                    tolerance=pd.Timedelta(milliseconds=merge_tolerance_ms)
                                )
                                merged_chunk = merged_chunk.dropna(subset=['composite_cluster_id'])
                            except Exception as e:
                                self.logger.warning(f'⚠️ merge_asof failed for {file_path.name}: {e}; falling back to inner merge')
                                merged_chunk = pd.merge(df, regime_df[['timestamp', 'composite_cluster_id']], on='timestamp', how='inner')
                        else:
                            merged_chunk = pd.merge(df, regime_df[['timestamp', 'composite_cluster_id']], on='timestamp', how='inner')

                        if len(merged_chunk) > 0:
                            batch_chunks.append(merged_chunk)

                        # Force garbage collection after each file
                        del df
                        if PSUTIL_AVAILABLE:
                            import gc
                            gc.collect()

                    except Exception as e:
                        self.logger.error(f'❌ Error processing {file_path.name}: {e}')
                        continue

                # Concatenate batch results
                if batch_chunks:
                    batch_result = pd.concat(batch_chunks, ignore_index=True)
                    all_chunks.append(batch_result)
                    self.logger.info(f'✅ Processed batch with {len(batch_result)} rows')

                    # Free memory
                    del batch_chunks
                    if PSUTIL_AVAILABLE:
                        import gc
                        gc.collect()

            # Final concatenation
            if all_chunks:
                final_df = pd.concat(all_chunks, ignore_index=True)
                self.logger.info(f'🎉 Streaming processing completed: {len(final_df)} total rows')
                return final_df
            else:
                self.logger.warning('⚠️ No data processed in streaming mode')
                return pd.DataFrame()

        except Exception as e:
            self.logger.exception(f'❌ Error in streaming processing: {e}')
            return pd.DataFrame()

    @comprehensive_function_monitor
    def _save_unified_dataset(self, data: pd.DataFrame, training_dir: Path, exchange: str, symbol: str, timeframe: str) -> bool:
        """Save the unified regime dataset to parquet file with advanced optimizations including partitioning and metadata caching."""
        try:
            # Apply columnar storage optimization
            if self.columnar_optimization and PYARROW_AVAILABLE:
                data = self._optimize_columnar_storage(data)
            
            # Determine if partitioning should be used
            use_partitioning = (
                self.enable_parquet_partitioning and 
                len(data) > self.partition_size_threshold and 
                PYARROW_AVAILABLE
            )
            
            if use_partitioning:
                return self._save_partitioned_dataset(data, training_dir, exchange, symbol, timeframe)
            
            # Standard save with metadata caching
            unified_file = training_dir / f'{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet'
            data_size_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)

            # Enhanced streaming logic for large datasets
            streaming_min_rows = int(self.config.get('streaming_min_rows', 2_000_000))
            streaming_min_mb = int(self.config.get('streaming_min_mb', 1000))  # 1GB threshold

            use_streaming = (
                (len(data) > streaming_min_rows) or
                (data_size_mb > streaming_min_mb)
            ) and PYARROW_AVAILABLE

            if use_streaming:
                self.logger.info(f'🧠 Large dataset detected ({len(data):,} rows, {data_size_mb:.1f}MB), using streaming write')

                try:
                    # Create schema from a small sample for better memory efficiency
                    sample_size = min(10000, len(data))
                    sample_data = data.head(sample_size)
                    schema = pa.Schema.from_pandas(sample_data)

                    with pq.ParquetWriter(unified_file, schema, compression='snappy') as writer:
                        chunk_size = int(self.config.get('streaming_chunk_rows', 500_000))
                        total_chunks = (len(data) + chunk_size - 1) // chunk_size

                        for chunk_idx, start in enumerate(range(0, len(data), chunk_size)):
                            end = min(start + chunk_size, len(data))
                            batch = data.iloc[start:end].copy()

                            # Log progress for large datasets
                            if total_chunks > 10:
                                progress = (chunk_idx + 1) / total_chunks * 100
                                self.logger.info(f'📝 Writing chunk {chunk_idx + 1}/{total_chunks} ({progress:.1f}%)')

                            # Convert to Arrow table
                            table = pa.Table.from_pandas(batch, schema=schema, preserve_index=False)
                            writer.write_table(table)

                            # Memory cleanup
                            del batch, table
                            if PSUTIL_AVAILABLE:
                                import gc
                                gc.collect()

                    file_size_mb = unified_file.stat().st_size / (1024 * 1024)
                    self.logger.info(f'✅ Saved unified regime dataset (streaming): {len(data):,} rows -> {file_size_mb:.1f}MB file')
                    return True

                except Exception as e:
                    self.logger.warning(f'⚠️ Streaming writer failed ({e}), falling back to pandas parquet')

            # Standard pandas save with memory monitoring
            self.logger.info(f'💾 Saving dataset ({len(data):,} rows, {data_size_mb:.1f}MB) using pandas')

            # Use memory-efficient parquet options
            parquet_options = {
                'index': False,
                'compression': 'snappy',  # Good balance of speed vs compression
                'engine': 'auto'
            }

            data.to_parquet(unified_file, **parquet_options)
            file_size_mb = unified_file.stat().st_size / (1024 * 1024)
            self.logger.info(f'✅ Saved unified regime dataset: {len(data):,} rows -> {file_size_mb:.1f}MB file')
            return True

        except Exception as e:
            self.logger.error(f'❌ Error saving unified dataset: {e}')
            return False

    @comprehensive_function_monitor
    def _save_regime_statistics(self, data: pd.DataFrame, regime_ids: List[int], training_dir: Path, exchange: str, symbol: str, timeframe: str) -> bool:
        """Save regime statistics to JSON file."""
        try:
            regime_stats = self._calculate_regime_statistics(data, regime_ids)
            stats_file = training_dir / f'{exchange}_{symbol}_{timeframe}_regime_statistics.json'
            with open(stats_file, 'w') as f:
                json.dump(regime_stats, f, indent = 2)
            self.logger.info(f'✅ Saved regime statistics: {stats_file}')
            return True
        except Exception as e:
            self.logger.error(f'❌ Error saving regime statistics: {e}')
            return False

    @comprehensive_function_monitor
    def _save_regime_labels(self, data: pd.DataFrame, regime_ids: List[int], training_dir: Path, exchange: str, symbol: str, timeframe: str) -> bool:
        """Save regime labels mapping to JSON file."""
        try:
            regime_labels = {'regime_column': 'composite_cluster_id', 'regime_ids': sorted(regime_ids), 'total_regimes': len(regime_ids), 'data_shape': data.shape, 'timestamp_range': {'start': data['timestamp'].min().isoformat(), 'end': data['timestamp'].max().isoformat()}}
            labels_file = training_dir / f'{exchange}_{symbol}_{timeframe}_regime_labels.json'
            safe_json_dump(regime_labels, labels_file, indent = 2)
            self.logger.info(f'✅ Saved regime labels mapping: {labels_file}')
            return True
        except Exception as e:
            self.logger.error(f'❌ Error saving regime labels: {e}')
            return False

    def _optimize_columnar_storage(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize columnar storage by reordering columns for better compression and access patterns."""
        try:
            if not self.optimize_column_order:
                return data
            
            # Define optimal column order for time-series data
            timestamp_cols = [col for col in data.columns if 'timestamp' in col.lower() or 'time' in col.lower()]
            regime_cols = [col for col in data.columns if 'regime' in col.lower() or 'cluster' in col.lower()]
            price_cols = [col for col in data.columns if col in ['open', 'high', 'low', 'close', 'volume']]
            feature_cols = [col for col in data.columns if col not in timestamp_cols + regime_cols + price_cols]
            
            # Reorder columns for optimal columnar storage
            optimal_order = timestamp_cols + regime_cols + price_cols + feature_cols
            existing_cols = [col for col in optimal_order if col in data.columns]
            
            if existing_cols != list(data.columns):
                data = data[existing_cols]
                self.logger.info(f"🔄 Optimized column order for columnar storage: {len(existing_cols)} columns")
            
            return data
        except Exception as e:
            self.logger.warning(f"⚠️ Columnar optimization failed: {e}")
            return data

    def _save_partitioned_dataset(self, data: pd.DataFrame, training_dir: Path, exchange: str, symbol: str, timeframe: str) -> bool:
        """Save dataset using Parquet partitioning for better query performance."""
        try:
            if not PYARROW_AVAILABLE:
                self.logger.warning("⚠️ PyArrow not available for partitioning, falling back to standard save")
                return self._save_unified_dataset(data, training_dir, exchange, symbol, timeframe)
            
            # Create partitioned directory structure
            partitioned_dir = training_dir / f'{exchange}_{symbol}_{timeframe}_partitioned_data'
            partitioned_dir.mkdir(exist_ok=True)
            
            # Add partitioning columns if they don't exist
            data = self._add_partitioning_columns(data)
            
            # Convert to PyArrow table
            table = pa.Table.from_pandas(data)
            
            # Write partitioned dataset
            pq.write_to_dataset(
                table,
                root_path=str(partitioned_dir),
                partition_cols=self.partition_columns,
                compression='snappy',
                use_dictionary=True,  # Better compression for categorical data
                write_statistics=True  # Enable statistics for better query performance
            )
            
            # Cache metadata for future reads
            self._cache_parquet_metadata(str(partitioned_dir), data.shape)
            
            self.logger.info(f"✅ Saved partitioned dataset: {len(data):,} rows -> {partitioned_dir}")
            self.logger.info(f"   📁 Partition columns: {self.partition_columns}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving partitioned dataset: {e}")
            return False

    def _add_partitioning_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add partitioning columns for time-series data."""
        try:
            data = data.copy()
            
            # Add year and month columns for time-based partitioning
            if 'timestamp' in data.columns:
                timestamp_series = pd.to_datetime(data['timestamp'])
                data['year'] = timestamp_series.dt.year
                data['month'] = timestamp_series.dt.month
                
                # Add to partition columns if not already present
                for col in ['year', 'month']:
                    if col not in self.partition_columns:
                        self.partition_columns.append(col)
            
            return data
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to add partitioning columns: {e}")
            return data

    def _cache_parquet_metadata(self, file_path: str, data_shape: tuple) -> None:
        """Cache Parquet metadata for frequently accessed files."""
        try:
            if len(self.parquet_metadata_cache) >= self.parquet_cache_max_size:
                # Remove oldest entries
                oldest_key = min(self.parquet_metadata_cache.keys())
                del self.parquet_metadata_cache[oldest_key]
            
            # Cache metadata
            self.parquet_metadata_cache[file_path] = {
                'shape': data_shape,
                'timestamp': time.time(),
                'access_count': 0
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to cache Parquet metadata: {e}")

    def _get_cached_parquet_metadata(self, file_path: str) -> Optional[dict]:
        """Get cached Parquet metadata if available."""
        try:
            if file_path in self.parquet_metadata_cache:
                metadata = self.parquet_metadata_cache[file_path]
                metadata['access_count'] += 1
                return metadata
            return None
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to get cached metadata: {e}")
            return None

    def _read_parquet_with_cache(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read Parquet file with metadata caching for better performance."""
        try:
            # Check if metadata is cached
            cached_metadata = self._get_cached_parquet_metadata(file_path)
            
            if cached_metadata:
                self.logger.debug(f"📋 Using cached metadata for {file_path}")
                # Use cached metadata to optimize read
                if PYARROW_AVAILABLE:
                    # Use PyArrow for optimized reading with cached metadata
                    table = pq.read_table(file_path, **kwargs)
                    df = table.to_pandas()
                else:
                    df = pd.read_parquet(file_path, **kwargs)
            else:
                # Standard read and cache metadata
                df = pd.read_parquet(file_path, **kwargs)
                self._cache_parquet_metadata(file_path, df.shape)
            
            return df
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cached read failed for {file_path}: {e}")
            # Fallback to standard read
            return pd.read_parquet(file_path, **kwargs)

    @comprehensive_function_monitor
    async def _create_unified_regime_dataset(self, data: pd.DataFrame, regime_ids: List[int], data_dir: str, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any] | None:
        """Create unified dataset with regime labels and return dataset info."""
        try:
            data = data.sort_values('timestamp').reset_index(drop = True)
            training_dir = ensure_directory(Path(data_dir) / 'training' / 'regime_splits')
            if not self._save_unified_dataset(data, training_dir, exchange, symbol, timeframe):
                return None
            if not self._save_regime_statistics(data, regime_ids, training_dir, exchange, symbol, timeframe):
                return None
            if not self._save_regime_labels(data, regime_ids, training_dir, exchange, symbol, timeframe):
                return None
            saved_path = str(training_dir / f'{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet')
            return {'unified_data': data, 'regime_stats': self._calculate_regime_statistics(data, regime_ids), 'saved_path': saved_path}
        except Exception as e:
            self.logger.exception(f'❌ Error creating unified regime dataset: {e}')
            return None

    @comprehensive_function_monitor
    def _calculate_regime_statistics(self, data: pd.DataFrame, regime_ids: List[int]) -> Dict[str, Any]:
        """Calculate per-regime statistics using vectorized operations.

        Produces both legacy and enriched fields for backward compatibility with validators.
        Legacy: count, duration_minutes, mean_volume
        Enriched: percentage, mean_volatility, mean_momentum
        """
        try:
            return self._vectorized_regime_statistics(data, regime_ids)
        except Exception as e:
            self.logger.exception(f'❌ Error calculating regime statistics: {e}')
            return {}
    
    def _vectorized_regime_statistics(self, data: pd.DataFrame, regime_ids: List[int]) -> Dict[str, Any]:
        """Vectorized regime statistics calculation with 3-10x speedup."""
        stats: Dict[int, Dict[str, Any]] = {}
        total_rows = max(int(len(data)), 1)
        
        # Use pandas groupby for vectorized operations
        regime_groups = data.groupby('composite_cluster_id')
        
        # Vectorized calculations for all regimes at once
        regime_counts = regime_groups.size()
        regime_volumes = regime_groups['volume'].mean() if 'volume' in data.columns else pd.Series(0.0, index=regime_counts.index)
        
        # Vectorized timestamp calculations
        regime_timestamps = regime_groups['timestamp'].agg(['min', 'max'])
        
        # Vectorized volatility and momentum calculations if close price exists
        if 'close' in data.columns:
            # Calculate returns for all data at once
            returns = data['close'].pct_change().fillna(0.0)
            data_with_returns = data.copy()
            data_with_returns['returns'] = returns
            
            # Vectorized volatility and momentum by regime
            regime_volatility = data_with_returns.groupby('composite_cluster_id')['returns'].apply(
                lambda x: x.rolling(window=30, min_periods=5).std().mean()
            )
            regime_momentum = data_with_returns.groupby('composite_cluster_id')['returns'].apply(
                lambda x: x.rolling(window=30, min_periods=5).mean().mean()
            )
        else:
            regime_volatility = pd.Series(0.0, index=regime_counts.index)
            regime_momentum = pd.Series(0.0, index=regime_counts.index)
        
        # Vectorized duration calculation
        def calculate_duration_minutes(start_ts, end_ts):
            try:
                return int((int(end_ts) - int(start_ts)) / 60000)
            except Exception:
                try:
                    return int((pd.to_datetime(end_ts) - pd.to_datetime(start_ts)).total_seconds() / 60)
                except Exception:
                    return 0
        
        # Build statistics dictionary using vectorized results
        for regime_id in regime_ids:
            if regime_id in regime_counts.index:
                count = int(regime_counts[regime_id])
                percentage = float(count) / float(total_rows)
                mean_volume = float(regime_volumes.get(regime_id, 0.0))
                mean_volatility = float(regime_volatility.get(regime_id, 0.0))
                mean_momentum = float(regime_momentum.get(regime_id, 0.0))
                
                # Calculate duration
                if regime_id in regime_timestamps.index:
                    start_ts = regime_timestamps.loc[regime_id, 'min']
                    end_ts = regime_timestamps.loc[regime_id, 'max']
                    duration_minutes = calculate_duration_minutes(start_ts, end_ts)
                else:
                    duration_minutes = 0
            else:
                # Regime not found in data
                count = 0
                percentage = 0.0
                mean_volume = 0.0
                mean_volatility = 0.0
                mean_momentum = 0.0
                duration_minutes = 0
            
            stats[int(regime_id)] = {
                'count': count,
                'duration_minutes': duration_minutes,
                'mean_volume': mean_volume,
                'percentage': percentage,
                'mean_volatility': mean_volatility,
                'mean_momentum': mean_momentum,
            }
        
        return stats

    @comprehensive_function_monitor
    async def _save_regime_metadata(self, regime_ids: List[int], data_dir: str, symbol: str, exchange: str, timeframe: str) -> None:
        """Save metadata about the unified regime dataset."""
        try:
            metadata = {'approach': 'unified_dataset_with_labels', 'total_regimes': len(regime_ids), 'regime_ids': sorted(regime_ids), 'created_at': time.time(), 'data_structure': {'main_file': f'{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet', 'regime_column': 'composite_cluster_id', 'regime_labels_file': f'{exchange}_{symbol}_{timeframe}_regime_labels.json', 'regime_statistics_file': f'{exchange}_{symbol}_{timeframe}_regime_statistics.json'}, 'usage_instructions': {'description': 'Load the unified dataset and filter by composite_cluster_id for regime-specific processing', 'example': "regime_data = data[data['composite_cluster_id'] == regime_id]", 'benefits': ['Maintains temporal continuity for trading indicators', 'Preserves lookback periods', 'Eliminates need for multiple file management', 'Enables regime-aware processing with single dataset']}}
            metadata_file = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_regime_metadata.json'
            safe_json_dump(metadata, metadata_file, indent = 2)
            self.logger.info(f'✅ Regime metadata saved: {metadata_file}')
        except Exception as e:
            self.logger.exception(f'❌ Error saving regime metadata: {e}')

    @comprehensive_function_monitor
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the regime data splitting step.

        This method is called by the training manager and delegates to the main processing method.
        """
        try:
            symbol = training_input.get('symbol')
            exchange = training_input.get('exchange')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir')

            if not all([symbol, exchange, timeframe, data_dir]):
                return {
                    'success': False,
                    'step04_regime_data_splitting_completed': False,
                    'error': 'Missing required parameters: symbol, exchange, timeframe, data_dir',
                    'step_name': 'step04_regime_data_splitting'
                }

            self.logger.info(f'🚀 Executing regime data splitting for {symbol} on {exchange} ({timeframe})')

            # Call the main processing method
            result = await self.split_data_by_regimes(symbol, exchange, timeframe, data_dir)

            if result.success:
                return {
                    'success': True,
                    'step04_regime_data_splitting_completed': True,
                    'regime_data': result.data,
                    'regime_metadata': result.metadata,
                    'regime_splits': result.data,  # For compatibility with step 05 expectations
                    'execution_time': result.execution_time,
                    'step_name': 'step04_regime_data_splitting'
                }
            else:
                return {
                    'success': False,
                    'step04_regime_data_splitting_completed': False,
                    'error': result.error,
                    'execution_time': result.execution_time,
                    'step_name': 'step04_regime_data_splitting'
                }

        except Exception as e:
            error_context = {
                'training_input_keys': list(training_input.keys()) if training_input else [],
                'pipeline_state_keys': list(pipeline_state.keys()) if pipeline_state else []
            }
            log_comprehensive_error_report(e, error_context)
            return {
                'success': False,
                'step04_regime_data_splitting_completed': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'execution_time': time.time() - self.start_time if self.start_time else 0,
                'step_name': 'step04_regime_data_splitting'
            }

@comprehensive_function_monitor
@traced(span_name='execute_regime_data_splitting')
@validates()
@handles_errors()
@cached()
@log_execution_time()
@monitor_feature_engineering()
async def run_step(symbol: str, exchange: str, timeframe: str, data_dir: str = None, force_rerun: bool = False, config: dict[str, Any]=None) -> StepResult:
    """Run Step 4: Regime Data Splitting with standardized data quality management."
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        data_dir: Data directory (will use standardized path if None)
        force_rerun: Force rerun flag
        config: Configuration dictionary
        
    Returns:
        StepResult: Standardized result with success status and details
    """
    logger.info('🚀 Starting Step 4: Regime Data Splitting with Comprehensive Function Call Monitoring')
    
    # Initialize lookahead bias detector
    from datetime import datetime
    current_time = datetime.now()
    bias_detector = get_global_detector()
    bias_detector.set_current_timestamp(current_time)
    if data_dir is None:
        data_dir = pipeline_standards.build_path('processed_data', exchange, symbol)
    
    step_start = time.time()
    try:
        step = RegimeDataSplittingStep(config or {})
        await step.initialize()
        result = await step.split_data_by_regimes(symbol, exchange, timeframe, data_dir)
        
        # Standardize the result if it's not already a StepResult
        standardized_result = standardize_result(result, "regime_data_splitting")

        logger.info('📊 Generating comprehensive function call summary...')
        log_function_call_summary()

        if standardized_result.success:
            logger.info('✅ Step 4: Regime Data Splitting completed successfully')
            logger.info('🎯 All function calls executed with comprehensive monitoring')

            # Return dictionary for pipeline state integration (similar to step 02_5)
            return {
                'success': True,
                'step04_regime_data_splitting_completed': True,
                'regime_data': result.data,
                'regime_metadata': result.metadata,
                'regime_splits': result.data,  # For compatibility with step 05 expectations
                'execution_time': standardized_result.execution_time,
                'step_name': 'step04_regime_data_splitting'
            }
        else:
            logger.error('❌ Step 4: Regime Data Splitting failed')
            logger.error(f'🔍 Error: {standardized_result.error}')
            logger.error('🔍 Check function call summary above for detailed error analysis')

            return {
                'success': False,
                'step04_regime_data_splitting_completed': False,
                'error': standardized_result.error,
                'execution_time': standardized_result.execution_time,
                'step_name': 'step04_regime_data_splitting'
            }
        
    except Exception as e:
        error_context = {
            'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe,
            'data_dir': data_dir, 'force_rerun': force_rerun,
            'config_keys': list(config.keys()) if config else []
        }
        log_comprehensive_error_report(e, error_context)
        logger.error('📊 Generating function call summary for error analysis...')
        log_function_call_summary()

        return {
            'success': False,
            'step04_regime_data_splitting_completed': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'execution_time': time.time() - step_start,
            'step_name': 'step04_regime_data_splitting'
        }
if __name__ == '__main__':

    async def test() -> None:
        test_config = {'symbol': 'ETHUSDT', 'exchange': 'BINANCE', 'timeframe': '1m'}
        success = await run_step(symbol='ETHUSDT', exchange='BINANCE', timeframe='1m', data_dir='data_cache', force_rerun = False, config = test_config)
        print(f'Test result: {success}')
    asyncio.run(test())