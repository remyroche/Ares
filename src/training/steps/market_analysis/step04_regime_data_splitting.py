from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from src.training.steps.model_training.step04_common_types import (
    StepResult, RegimeDataResult, StepResultStatus, standardize_result
)

"""Step 4: Regime Data Splitting with Comprehensive Function Call Monitoring.

This module creates a unified dataset with regime labels for regime-aware processing.
Uses labels to differentiate regimes instead of creating separate files per regime.
This ensures trading indicators have the necessary lookback periods.

Enhanced with comprehensive function call monitoring, function-to-function tracking,
and detailed outcome reporting for complete execution visibility.
"""
import asyncio
import sys
import time
import functools
import traceback
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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
from src.core.decorators import handles_errors, traced, validates, cached, log_execution_time
try:
    from src.core.domain.decorators_extended import monitor_feature_engineering
except Exception:

    def monitor_feature_engineering(*args, **kwargs) -> None:

        def _decorator(func: Callable) -> None:
            return func
        return _decorator
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.utils.common_operations import ensure_directory, safe_json_dump
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
REQUIRED_MODULES = ['pandas', 'numpy', 'src.utils.logger', 'src.utils.enhanced_mlflow_integration']
dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)
from src.utils.logger import system_logger
enhanced_mlflow = PipelineStandards.safe_import('src.utils.enhanced_mlflow_integration', None)
pandas = pd
numpy = np

def create_fallback_logger() -> Any:
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def create_fallback_decorator() -> Any:

    def decorator(func: Callable) -> Callable:

        def wrapper(*args, **kwargs) -> None:
            return func(*args, **kwargs)
        return wrapper
    return decorator
if system_logger is None:
    system_logger = create_fallback_logger()
comprehensive_data_validation = validates
handle_errors = handles_errors
memory_efficient = cached
resource_monitor = log_execution_time
secure_data_processing = handles_errors
validate_data_structure = validates
with_tracing_span = traced
quality_gate = validates
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

    def start_call(self, func_name: str, args: tuple, kwargs: dict, caller: str=None) -> str:
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

    def end_call(self, call_id: str, result: Any=None, error: Exception=None) -> Dict[str, Any]:
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
            outcome = _function_tracker.end_call(call_id, error=e)
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
            outcome = _function_tracker.end_call(call_id, error=e)
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

class RegimeDataSplittingStep:
    """Step 4: Regime Data Splitting with standardized data quality management."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('RegimeDataSplittingStep')
        self.standards = pipeline_standards
        self.start_time = None
        self.step_timings = {}
        self._validate_environment()

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
        self.logger.info(f'🔀 Creating unified dataset with regime labels for {symbol} on {exchange} ({timeframe})')
        try:
            regime_data = await self._load_regime_data(symbol, exchange, timeframe, data_dir)
            if regime_data is None:
                return RegimeDataResult.failure_result(
                    error='regime_data_not_found',
                    error_type='DataNotFoundError',
                    metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
                )
            
            regime_ids = regime_data['composite_cluster_id'].unique()
            num_regimes = len(regime_ids)
            self.logger.info(f'📊 Found {num_regimes} regimes: {sorted(regime_ids)}')
            
            if num_regimes < 3:
                self.logger.error(f'❌ Too few regimes: {num_regimes} (minimum 3 required)')
                return RegimeDataResult.failure_result(
                    error=f'too_few_regimes: {num_regimes} (minimum 3 required)',
                    error_type='InsufficientRegimesError',
                    metadata={'regime_count': num_regimes, 'regime_ids': regime_ids.tolist()}
                )
            
            if num_regimes > 20:
                self.logger.warning(f'⚠️ Many regimes detected: {num_regimes} (maximum 20 supported)')
            
            dataset_info = await self._create_unified_regime_dataset(regime_data, regime_ids, data_dir, symbol, exchange, timeframe)
            if isinstance(dataset_info, dict):
                self._log_step_timing('Regime Data Splitting', step_start)
                self.logger.info(f'✅ Successfully created unified dataset with {num_regimes} regime labels')
                await self._save_regime_metadata(regime_ids, data_dir, symbol, exchange, timeframe)
                
                return RegimeDataResult.success_result(
                    data=dataset_info.get('unified_data'),
                    metadata={
                        'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe,
                        'regime_count': num_regimes, 'regime_ids': regime_ids.tolist()
                    },
                    execution_time=time.time() - step_start
                )
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
                error=str(e),
                error_type=type(e).__name__,
                metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe},
                execution_time=time.time() - step_start
            )

    @comprehensive_function_monitor
    async def _load_regime_data(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[pd.DataFrame]:
        """Load HMM regime data with standardized validation."""
        try:
            unified_data_path = Path(self.standards.build_path('unified_data', exchange, symbol)) / timeframe
            if not unified_data_path.exists():
                self.logger.error(f'❌ Unified data path not found: {unified_data_path}')
                return None
            regime_primary = Path('data') / 'hmm_regimes' / f'{exchange}_{symbol}_{timeframe}_composite_clusters.parquet'
            regime_alternative = Path(data_dir) / 'hmm_regimes' / f'{exchange}_{symbol}_{timeframe}_composite_clusters.parquet'
            regime_file = regime_primary if regime_primary.exists() else regime_alternative
            if not regime_file.exists():
                self.logger.error(f'❌ Regime file not found: {regime_primary} or {regime_alternative}')
                return None
            unified_files = list(unified_data_path.glob('**/*.parquet'))
            if not unified_files:
                self.logger.error(f'❌ No unified data files found in {unified_data_path}')
                return None
            unified_data = []
            for file_path in sorted(unified_files):
                df = pandas.read_parquet(file_path)
                df = self.standards.standardize_timestamp(df, 'timestamp')
                df = self.standards.enforce_schema(df, 'unified')
                unified_data.append(df)
            unified_df = pd.concat(unified_data, ignore_index=True)
            regime_df = pandas.read_parquet(regime_file)
            regime_df = self.standards.standardize_timestamp(regime_df, 'timestamp')
            merged_data = pd.merge(unified_df, regime_df[['timestamp', 'composite_cluster_id']], on='timestamp', how='inner')
            try:
                retention_ratio = len(merged_data) / max(len(unified_df), 1) if len(unified_df) else 0.0
                self.logger.info(f'📈 Merge retention ratio: {retention_ratio:.3f}')
                min_retention = float(self.config.get('regime_merge_min_retention', 0.8))
                if retention_ratio < min_retention:
                    self.logger.warning(f'⚠️ Low retention after regime merge: {retention_ratio:.3f} (< {min_retention:.2f}). Check timestamp alignment and data coverage.')
            except Exception:
                pass
            self.logger.info(f'✅ Loaded {len(merged_data)} data points with regime information')
            return merged_data
        except Exception as e:
            self.logger.exception(f'❌ Error loading regime data: {e}')
            return None

    @comprehensive_function_monitor
    def _save_unified_dataset(self, data: pd.DataFrame, training_dir: Path, exchange: str, symbol: str, timeframe: str) -> bool:
        """Save the unified regime dataset to parquet file."""
        try:
            unified_file = training_dir / f'{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet'
            data.to_parquet(unified_file, index=False)
            self.logger.info(f'✅ Saved unified regime dataset: {len(data)} rows -> {unified_file}')
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
            import json
            with open(stats_file, 'w') as f:
                json.dump(regime_stats, f, indent=2)
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
            safe_json_dump(regime_labels, labels_file, indent=2)
            self.logger.info(f'✅ Saved regime labels mapping: {labels_file}')
            return True
        except Exception as e:
            self.logger.error(f'❌ Error saving regime labels: {e}')
            return False

    @comprehensive_function_monitor
    async def _create_unified_regime_dataset(self, data: pd.DataFrame, regime_ids: List[int], data_dir: str, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any] | None:
        """Create unified dataset with regime labels and return dataset info."""
        try:
            data = data.sort_values('timestamp').reset_index(drop=True)
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
        """Calculate simple per-regime statistics compatible with tests."""
        try:
            stats: Dict[int, Dict[str, Any]] = {}
            for regime_id in regime_ids:
                regime_data = data[data['composite_cluster_id'] == regime_id]
                if len(regime_data) == 0:
                    stats[int(regime_id)] = {'count': 0, 'duration_minutes': 0, 'mean_volume': 0.0}
                    continue
                start_ts = regime_data['timestamp'].min()
                end_ts = regime_data['timestamp'].max()
                try:
                    duration_minutes = int((int(end_ts) - int(start_ts)) / 60000)
                except Exception:
                    duration_minutes = int((pd.to_datetime(end_ts) - pd.to_datetime(start_ts)).total_seconds() / 60)
                mean_volume = float(regime_data['volume'].mean()) if 'volume' in regime_data.columns else 0.0
                stats[int(regime_id)] = {'count': int(len(regime_data)), 'duration_minutes': duration_minutes, 'mean_volume': mean_volume}
            return stats
        except Exception as e:
            self.logger.exception(f'❌ Error calculating regime statistics: {e}')
            return {}

    @comprehensive_function_monitor
    async def _save_regime_metadata(self, regime_ids: List[int], data_dir: str, symbol: str, exchange: str, timeframe: str) -> None:
        """Save metadata about the unified regime dataset."""
        try:
            metadata = {'approach': 'unified_dataset_with_labels', 'total_regimes': len(regime_ids), 'regime_ids': sorted(regime_ids), 'created_at': time.time(), 'data_structure': {'main_file': f'{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet', 'regime_column': 'composite_cluster_id', 'regime_labels_file': f'{exchange}_{symbol}_{timeframe}_regime_labels.json', 'regime_statistics_file': f'{exchange}_{symbol}_{timeframe}_regime_statistics.json'}, 'usage_instructions': {'description': 'Load the unified dataset and filter by composite_cluster_id for regime-specific processing', 'example': "regime_data = data[data['composite_cluster_id'] == regime_id]", 'benefits': ['Maintains temporal continuity for trading indicators', 'Preserves lookback periods', 'Eliminates need for multiple file management', 'Enables regime-aware processing with single dataset']}}
            metadata_file = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_regime_metadata.json'
            safe_json_dump(metadata, metadata_file, indent=2)
            self.logger.info(f'✅ Regime metadata saved: {metadata_file}')
        except Exception as e:
            self.logger.exception(f'❌ Error saving regime metadata: {e}')

@comprehensive_function_monitor
@traced(span_name='execute_regime_data_splitting')
@validates()
@handles_errors()
@cached()
@log_execution_time()
@monitor_feature_engineering()
async def run_step(symbol: str, exchange: str, timeframe: str, data_dir: str=None, force_rerun: bool=False, config: dict[str, Any]=None) -> StepResult:
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
        else:
            logger.error('❌ Step 4: Regime Data Splitting failed')
            logger.error(f'🔍 Error: {standardized_result.error}')
            logger.error('🔍 Check function call summary above for detailed error analysis')
        
        return standardized_result
        
    except Exception as e:
        error_context = {
            'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 
            'data_dir': data_dir, 'force_rerun': force_rerun, 
            'config_keys': list(config.keys()) if config else []
        }
        log_comprehensive_error_report(e, error_context)
        logger.error('📊 Generating function call summary for error analysis...')
        log_function_call_summary()
        
        return StepResult.failure_result(
            error=str(e),
            error_type=type(e).__name__,
            metadata=error_context,
            execution_time=time.time() - step_start
        )
if __name__ == '__main__':

    async def test() -> None:
        test_config = {'symbol': 'ETHUSDT', 'exchange': 'BINANCE', 'timeframe': '1m'}
        success = await run_step(symbol='ETHUSDT', exchange='BINANCE', timeframe='1m', data_dir='data_cache', force_rerun=False, config=test_config)
        print(f'Test result: {success}')
    asyncio.run(test())