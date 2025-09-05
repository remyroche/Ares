from typing import List, Dict, Any, Tuple, Optional
import time
import traceback
import functools
import inspect
import gc
from pathlib import Path
import json
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd

try:
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
try:
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
'Step 7: Enhanced Matrix Operations - Refactored to use BaseStep.\n\nThis module performs advanced matrix operations for comprehensive data analysis\nafter feature engineering, with GPU/MPS acceleration support.\nIncludes comprehensive function call validation, tracking, and detailed outcome reporting.\n'
from src.training.base_step import BaseStep
from src.core.decorators import handles_errors
from src.training.steps.model_training.matrix_components import MatrixProcessor, DiverseLookbackIntegrator, MatrixOptimizer
from src.utils.logger import system_logger

class FunctionCallTracker:
    """Comprehensive function call tracking and validation system."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.call_stack = []
        self.function_calls = {}
        self.function_to_function_calls = {}
        self.completion_reports = {}
        self.start_time = time.time()

    def track_function_call(self, func_name: str, args: tuple, kwargs: dict, caller: str=None) -> None:
        """Track function call initiation."""
        call_id = f'{func_name}_{len(self.call_stack)}_{int(time.time() * 1000)}'
        call_info = {'call_id': call_id, 'function_name': func_name, 'caller': caller, 'args_count': len(args), 'kwargs_count': len(kwargs), 'start_time': time.time(), 'args_types': [type(arg).__name__ for arg in args], 'kwargs_keys': list(kwargs.keys())}
        self.call_stack.append(call_id)
        self.function_calls[call_id] = call_info
        if caller:
            if caller not in self.function_to_function_calls:
                self.function_to_function_calls[caller] = []
            self.function_to_function_calls[caller].append({'called_function': func_name, 'call_id': call_id, 'timestamp': time.time()})
        self.logger.debug(f'🔍 Function call initiated: {func_name} (ID: {call_id})')
        return call_id

    def track_function_completion(self, call_id: str, result: Any=None, error: Exception=None) -> None:
        """Track function call completion with detailed outcome."""
        if call_id not in self.function_calls:
            self.logger.warning(f'⚠️ Unknown call ID: {call_id}')
            return
        call_info = self.function_calls[call_id]
        end_time = time.time()
        duration = end_time - call_info['start_time']
        completion_report = {'call_id': call_id, 'function_name': call_info['function_name'], 'caller': call_info['caller'], 'duration_seconds': duration, 'success': error is None, 'error': str(error) if error else None, 'error_type': type(error).__name__ if error else None, 'result_type': type(result).__name__ if result is not None else None, 'result_size': self._get_result_size(result), 'end_time': end_time, 'stack_depth': len(self.call_stack)}
        self.completion_reports[call_id] = completion_report
        if call_id in self.call_stack:
            self.call_stack.remove(call_id)
        status = '✅' if error is None else '❌'
        self.logger.info(f"{status} Function completed: {call_info['function_name']} (ID: {call_id}, Duration: {duration:.3f}s)")
        if error:
            self.logger.error(f"❌ Function error: {call_info['function_name']} - {error}")
            self.logger.debug(f'Error traceback: {traceback.format_exc()}')
        return completion_report

    def _get_result_size(self, result: Any) -> str:
        """Get human-readable size of result."""
        if result is None:
            return 'None'
        elif isinstance(result, (list, tuple)):
            return f'len={len(result)}'
        elif isinstance(result, dict):
            return f'keys={len(result)}'
        elif isinstance(result, np.ndarray):
            return f'shape={result.shape}'
        elif isinstance(result, pd.DataFrame):
            return f'shape={result.shape}'
        else:
            return f'type={type(result).__name__}'

    def get_call_summary(self) -> Dict[str, Any]:
        """Get comprehensive call summary."""
        total_calls = len(self.function_calls)
        successful_calls = len([r for r in self.completion_reports.values() if r['success']])
        failed_calls = total_calls - successful_calls
        total_duration = sum((r['duration_seconds'] for r in self.completion_reports.values()))
        return {'total_function_calls': total_calls, 'successful_calls': successful_calls, 'failed_calls': failed_calls, 'success_rate': successful_calls / total_calls if total_calls > 0 else 0, 'total_duration_seconds': total_duration, 'average_duration_seconds': total_duration / total_calls if total_calls > 0 else 0, 'function_to_function_calls': len(self.function_to_function_calls), 'max_stack_depth': max((r['stack_depth'] for r in self.completion_reports.values()), default=0), 'session_duration_seconds': time.time() - self.start_time}

def comprehensive_function_tracker(logger: logging.Logger) -> None:
    """Decorator for comprehensive function call tracking."""

    def decorator(func: Callable) -> None:

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> None:
            frame = inspect.currentframe().f_back
            caller_name = frame.f_code.co_name if frame else 'unknown'
            tracker = None
            if args and hasattr(args[0], 'call_tracker'):
                tracker = args[0].call_tracker
            if tracker is None:
                tracker = FunctionCallTracker(logger)
            call_id = tracker.track_function_call(func.__name__, args, kwargs, caller_name)
            try:
                if inspect.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                tracker.track_function_completion(call_id, result)
                return result
            except Exception as e:
                tracker.track_function_completion(call_id, error=e)
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> None:
            frame = inspect.currentframe().f_back
            caller_name = frame.f_code.co_name if frame else 'unknown'
            tracker = None
            if args and hasattr(args[0], 'call_tracker'):
                tracker = args[0].call_tracker
            if tracker is None:
                tracker = FunctionCallTracker(logger)
            call_id = tracker.track_function_call(func.__name__, args, kwargs, caller_name)
            try:
                result = func(*args, **kwargs)
                tracker.track_function_completion(call_id, result)
                return result
            except Exception as e:
                tracker.track_function_completion(call_id, error=e)
                raise
        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    return decorator

class EnhancedErrorHandler:
    """Enhanced error handling with detailed context and recovery mechanisms."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.error_history = []
        self.recovery_attempts = {}
        self.error_patterns = {}

    def handle_error(self, error: Exception, context: Dict[str, Any], recovery_strategies: List[str]=None) -> None:
        """Handle error with detailed context and recovery strategies."""
        error_info = {'timestamp': time.time(), 'error_type': type(error).__name__, 'error_message': str(error), 'context': context, 'traceback': traceback.format_exc(), 'recovery_strategies': recovery_strategies or []}
        self.error_history.append(error_info)
        error_key = f"{type(error).__name__}_{context.get('function_name', 'unknown')}"
        if error_key not in self.error_patterns:
            self.error_patterns[error_key] = 0
        self.error_patterns[error_key] += 1
        self.logger.error(f"❌ Error in {context.get('function_name', 'unknown')}: {error}")
        self.logger.debug(f'Error context: {context}')
        self.logger.debug(f'Recovery strategies: {recovery_strategies}')
        return error_info

    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary."""
        return {'total_errors': len(self.error_history), 'error_patterns': self.error_patterns, 'recovery_attempts': self.recovery_attempts, 'recent_errors': self.error_history[-5:] if self.error_history else []}

class ComprehensiveValidator:
    """Comprehensive validation framework for step07 operations."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.validation_results = {}
        self.validation_rules = {}

    def validate_input_data(self, data: Any, data_type: str) -> Tuple[bool, List[str]]:
        """Validate input data based on type."""
        errors = []
        if data_type == 'dataframe' and PANDAS_AVAILABLE:
            if not isinstance(data, pd.DataFrame):
                errors.append('Data is not a pandas DataFrame')
            elif data.empty:
                errors.append('DataFrame is empty')
            elif data.isnull().all().any():
                errors.append('DataFrame has columns with all null values')
        elif data_type == 'numpy_array' and NUMPY_AVAILABLE:
            if not isinstance(data, np.ndarray):
                errors.append('Data is not a numpy array')
            elif data.size == 0:
                errors.append('Array is empty')
            elif np.isnan(data).all():
                errors.append('Array contains only NaN values')
        elif data_type == 'dict':
            if not isinstance(data, dict):
                errors.append('Data is not a dictionary')
            elif not data:
                errors.append('Dictionary is empty')
        is_valid = len(errors) == 0
        if not is_valid:
            self.logger.warning(f'⚠️ Input validation failed: {errors}')
        else:
            self.logger.debug(f'✅ Input validation passed for {data_type}')
        return (is_valid, errors)

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary."""
        return {'validation_results': self.validation_results, 'validation_rules': self.validation_rules, 'total_validations': len(self.validation_results)}

class PerformanceMonitor:
    """Performance monitoring and resource usage tracking for all functions."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.performance_metrics = {}
        self.resource_usage = {}
        self.start_time = time.time()
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
            self.psutil_available = True
        else:
            self.process = None
            self.psutil_available = False
            self.logger.warning('⚠️ psutil not available - limited performance monitoring')

    def start_monitoring(self, function_name: str) -> Dict[str, Any]:
        """Start monitoring performance for a function."""
        if self.psutil_available:
            initial_memory = self.process.memory_info().rss / 1024 / 1024
            initial_cpu = self.process.cpu_percent()
        else:
            initial_memory = 0.0
            initial_cpu = 0.0
        metrics = {'function_name': function_name, 'start_time': time.time(), 'initial_memory_mb': initial_memory, 'initial_cpu_percent': initial_cpu, 'initial_gc_count': gc.get_count(), 'psutil_available': self.psutil_available}
        self.performance_metrics[function_name] = metrics
        return metrics

    def stop_monitoring(self, function_name: str) -> Dict[str, Any]:
        """Stop monitoring and calculate performance metrics."""
        if function_name not in self.performance_metrics:
            self.logger.warning(f'⚠️ No monitoring data found for {function_name}')
            return {}
        metrics = self.performance_metrics[function_name]
        end_time = time.time()
        duration = end_time - metrics['start_time']
        if self.psutil_available:
            final_memory = self.process.memory_info().rss / 1024 / 1024
            final_cpu = self.process.cpu_percent()
        else:
            final_memory = 0.0
            final_cpu = 0.0
        final_gc_count = gc.get_count()
        metrics.update({'end_time': end_time, 'duration_seconds': duration, 'final_memory_mb': final_memory, 'final_cpu_percent': final_cpu, 'final_gc_count': final_gc_count, 'memory_delta_mb': final_memory - metrics['initial_memory_mb'], 'cpu_delta_percent': final_cpu - metrics['initial_cpu_percent'], 'gc_delta': tuple((f - i for f, i in zip(final_gc_count, metrics['initial_gc_count'])))})
        self.logger.info(f'📊 Performance metrics for {function_name}:')
        self.logger.info(f'   Duration: {duration:.3f}s')
        if self.psutil_available:
            self.logger.info(f"   Memory delta: {metrics['memory_delta_mb']:.1f} MB")
            self.logger.info(f"   CPU delta: {metrics['cpu_delta_percent']:.1f}%")
        else:
            self.logger.info('   Memory/CPU monitoring: Not available (psutil missing)')
        self.logger.info(f"   GC delta: {metrics['gc_delta']}")
        return metrics

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        total_duration = sum((m.get('duration_seconds', 0) for m in self.performance_metrics.values()))
        total_memory_delta = sum((m.get('memory_delta_mb', 0) for m in self.performance_metrics.values()))
        return {'total_functions_monitored': len(self.performance_metrics), 'total_duration_seconds': total_duration, 'total_memory_delta_mb': total_memory_delta, 'average_duration_seconds': total_duration / len(self.performance_metrics) if self.performance_metrics else 0, 'average_memory_delta_mb': total_memory_delta / len(self.performance_metrics) if self.performance_metrics else 0, 'session_duration_seconds': time.time() - self.start_time, 'psutil_available': self.psutil_available, 'function_metrics': self.performance_metrics}

class EnhancedMatrixOperationsStep(BaseStep):
    """Step 7: Enhanced Matrix Operations using standardized base class."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize enhanced matrix operations step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, '07', 'enhanced_matrix_operations')
        self.logger = system_logger.getChild('EnhancedMatrixOperationsStep')
        self.call_tracker = FunctionCallTracker(self.logger)
        self.logger.info('🔍 Initialized comprehensive function call tracking system')
        self.error_handler = EnhancedErrorHandler(self.logger)
        self.validator = ComprehensiveValidator(self.logger)
        self.logger.info('🛡️ Initialized enhanced error handling and validation system')
        self.performance_monitor = PerformanceMonitor(self.logger)
        self.logger.info('📊 Initialized performance monitoring system')
        self.matrix_config = config.get('matrix_operations_config', {'use_gpu': True, 'use_diverse_lookback': True, 'optimization_level': 'high', 'batch_size': 1000, 'feature_selection': {'method': 'mutual_info', 'top_k': 50, 'min_importance': 0.01}, 'matrix_computations': {'correlation_matrix': True, 'covariance_matrix': True, 'feature_interaction_matrix': True, 'regime_transition_matrix': True}})
        self.matrix_processor = None
        self.lookback_integrator = None
        self.matrix_optimizer = None

    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        try:
            self.matrix_processor = MatrixProcessor(use_gpu=self.matrix_config.get('use_gpu', True), batch_size=self.matrix_config.get('batch_size', 1000))
            if self.matrix_config.get('use_diverse_lookback', True):
                self.lookback_integrator = DiverseLookbackIntegrator(self.config)
            self.matrix_optimizer = MatrixOptimizer(optimization_level=self.matrix_config.get('optimization_level', 'high'))
            self.logger.info('✅ Enhanced matrix operations components initialized')
        except ImportError as e:
            self.logger.warning(f'⚠️ Some matrix components not available: {e}')

    async def initialize(self) -> None:
        """Initialize the step (BaseStep contract)."""
        self._initialize_step()

    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the step (BaseStep contract)."""
        try:
            is_valid, errors = self.validate_inputs(training_input, pipeline_state)
            if not is_valid and errors:
                self.logger.warning(f'Input validation issues: {errors}')
        except Exception:
            pass
        updated_state = await self.execute_logic(training_input, pipeline_state)
        if isinstance(updated_state, dict):
            updated_state['step07_enhanced_matrix_operations_completed'] = True
            return updated_state
        else:
            pipeline_state['step07_enhanced_matrix_operations_completed'] = True
            return pipeline_state

    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step inputs.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        if 'engineered_data' not in pipeline_state:
            if not all((f'{split}_data' in pipeline_state for split in ['train', 'val', 'test'])):
                errors.append('No engineered data from step 6')
        if 'selected_features' not in pipeline_state:
            self.logger.warning('No selected features, will use all features')
        else:
            try:
                data_any = pipeline_state.get('engineered_data', {}).get('train')
                if isinstance(data_any, pd.DataFrame):
                    missing = [f for f in pipeline_state['selected_features'] if f not in data_any.columns]
                    if missing:
                        self.logger.warning(f"Selected features missing in train data: {missing[:10]}{('...' if len(missing) > 10 else '')}")
            except Exception:
                pass
        if self.matrix_config.get('matrix_computations', {}).get('regime_transition_matrix', False):
            if 'regime_labels' not in pipeline_state:
                self.logger.warning('Regime labels not available for transition matrix')
        return (len(errors) == 0, errors)

    @comprehensive_function_tracker(None)
    @handles_errors(exceptions=(Exception,), default_return={'success': False}, context='enhanced matrix operations execution')
    async def execute_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced matrix operations logic.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state
        """
        self.logger.info('🔢 Starting enhanced matrix operations...')
        data_dict = self._get_data_to_process(pipeline_state)
        selected_features = pipeline_state.get('selected_features', [])
        if self.lookback_integrator and selected_features:
            self.logger.info('🔄 Optimizing lookback periods...')
            lookback_results = await self._optimize_lookback_periods(data_dict, selected_features)
            pipeline_state['lookback_optimization'] = lookback_results
        matrix_results = {}
        for split_name, data in data_dict.items():
            self.logger.info(f'🧮 Computing matrices for {split_name} split...')
            split_matrices = await self._compute_matrices(data, selected_features, pipeline_state)
            matrix_results[split_name] = split_matrices
            try:
                n_feats = len([c for c in data.columns if c.startswith('feature_')])
                self.logger.info(f'✅ {split_name}: matrices computed; features={n_feats}, keys={list(split_matrices.keys())}')
            except Exception:
                pass
        self.logger.info('📊 Analyzing feature importance...')
        importance_results = await self._analyze_feature_importance(data_dict, selected_features, matrix_results)
        optimization_insights = self._generate_optimization_insights(matrix_results, importance_results)
        reports = self._generate_matrix_reports(matrix_results, importance_results, optimization_insights)
        pipeline_state.update({'matrix_results': matrix_results, 'feature_importance': importance_results, 'optimization_insights': optimization_insights, 'matrix_reports': reports, 'matrix_config': self.matrix_config})
        await self._save_outputs(training_input, pipeline_state)
        call_summary = self.call_tracker.get_call_summary()
        self.logger.info('📊 COMPREHENSIVE FUNCTION CALL SUMMARY:')
        self.logger.info(f"   Total function calls: {call_summary['total_function_calls']}")
        self.logger.info(f"   Successful calls: {call_summary['successful_calls']}")
        self.logger.info(f"   Failed calls: {call_summary['failed_calls']}")
        self.logger.info(f"   Success rate: {call_summary['success_rate']:.2%}")
        self.logger.info(f"   Total duration: {call_summary['total_duration_seconds']:.3f}s")
        self.logger.info(f"   Average duration: {call_summary['average_duration_seconds']:.3f}s")
        self.logger.info(f"   Function-to-function calls: {call_summary['function_to_function_calls']}")
        self.logger.info(f"   Max stack depth: {call_summary['max_stack_depth']}")
        self.logger.info(f"   Session duration: {call_summary['session_duration_seconds']:.3f}s")
        pipeline_state['function_call_summary'] = call_summary
        pipeline_state['function_completion_reports'] = self.call_tracker.completion_reports
        pipeline_state['function_to_function_calls'] = self.call_tracker.function_to_function_calls
        performance_summary = self.performance_monitor.get_performance_summary()
        pipeline_state['performance_summary'] = performance_summary
        self.logger.info('📊 PERFORMANCE MONITORING SUMMARY:')
        self.logger.info(f"   Functions monitored: {performance_summary['total_functions_monitored']}")
        self.logger.info(f"   Total duration: {performance_summary['total_duration_seconds']:.3f}s")
        self.logger.info(f"   Total memory delta: {performance_summary['total_memory_delta_mb']:.1f} MB")
        self.logger.info(f"   Average duration: {performance_summary['average_duration_seconds']:.3f}s")
        self.logger.info(f"   psutil available: {performance_summary['psutil_available']}")
        error_summary = self.error_handler.get_error_summary()
        pipeline_state['error_summary'] = error_summary
        if error_summary['total_errors'] > 0:
            self.logger.warning(f'⚠️ ERROR HANDLING SUMMARY:')
            self.logger.warning(f"   Total errors: {error_summary['total_errors']}")
            self.logger.warning(f"   Error patterns: {error_summary['error_patterns']}")
            self.logger.warning(f"   Recovery attempts: {error_summary['recovery_attempts']}")
        else:
            self.logger.info('✅ No errors encountered during execution')
        validation_summary = self.validator.get_validation_summary()
        pipeline_state['validation_summary'] = validation_summary
        self.logger.info(f'🔍 VALIDATION SUMMARY:')
        self.logger.info(f"   Total validations: {validation_summary['total_validations']}")
        return pipeline_state

    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step outputs.
        
        Args:
            pipeline_state: Updated pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        if 'matrix_results' not in pipeline_state:
            errors.append('No matrix results in pipeline state')
            return (False, errors)
        matrix_results = pipeline_state['matrix_results']
        for split_name, matrices in matrix_results.items():
            if not isinstance(matrices, dict):
                errors.append(f'Invalid matrix results for {split_name}')
                continue
            expected_matrices = []
            matrix_computations = self.matrix_config.get('matrix_computations', {})
            if matrix_computations.get('correlation_matrix', True):
                expected_matrices.append('correlation_matrix')
            if matrix_computations.get('covariance_matrix', True):
                expected_matrices.append('covariance_matrix')
            missing_matrices = set(expected_matrices) - set(matrices.keys())
            if missing_matrices:
                errors.append(f'Missing matrices for {split_name}: {missing_matrices}')
        if 'feature_importance' not in pipeline_state:
            errors.append('No feature importance analysis results')
        return (len(errors) == 0, errors)

    def _get_data_to_process(self, pipeline_state: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Get data splits to process.
        
        Args:
            pipeline_state: Current pipeline state
            
        Returns:
            Dictionary of data splits
        """
        data_dict = {}
        if 'engineered_data' in pipeline_state:
            return pipeline_state['engineered_data']
        for split in ['train', 'val', 'test']:
            if f'{split}_data' in pipeline_state:
                data_dict[split] = pipeline_state[f'{split}_data']
        return data_dict

    async def _optimize_lookback_periods(self, data_dict: Dict[str, pd.DataFrame], selected_features: List[str]) -> Dict[str, Any]:
        """Optimize lookback periods using diverse lookback optimizer.
        
        Args:
            data_dict: Dictionary of data splits
            selected_features: List of selected features
            
        Returns:
            Lookback optimization results
        """
        if self.lookback_integrator:
            train_data = data_dict.get('train', next(iter(data_dict.values())))
            return await self.lookback_integrator.optimize_lookback_periods(train_data, selected_features)
        else:
            return {'optimized_periods': {'short': [5, 10, 20], 'medium': [50, 100], 'long': [200]}, 'method': 'default'}

    @comprehensive_function_tracker(None)
    async def _compute_matrices(self, data: pd.DataFrame, selected_features: List[str], pipeline_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Compute various matrices for the data.
        
        Args:
            data: Data to process
            selected_features: List of selected features
            pipeline_state: Pipeline state for additional context
            
        Returns:
            Dictionary of computed matrices
        """
        matrices = {}
        if selected_features:
            feature_data = data[selected_features]
        else:
            feature_cols = [col for col in data.columns if col.startswith('feature_')]
            feature_data = data[feature_cols]
        matrix_computations = self.matrix_config.get('matrix_computations', {})
        if matrix_computations.get('correlation_matrix', True):
            if self.matrix_processor:
                matrices['correlation_matrix'] = await self.matrix_processor.compute_correlation_matrix(feature_data)
            else:
                matrices['correlation_matrix'] = feature_data.corr().values
        if matrix_computations.get('covariance_matrix', True):
            if self.matrix_processor:
                matrices['covariance_matrix'] = await self.matrix_processor.compute_covariance_matrix(feature_data)
            else:
                matrices['covariance_matrix'] = feature_data.cov().values
        if matrix_computations.get('feature_interaction_matrix', True):
            matrices['feature_interaction_matrix'] = self._compute_interaction_matrix(feature_data)
        if matrix_computations.get('regime_transition_matrix', True) and 'regime_label' in data.columns:
            matrices['regime_transition_matrix'] = self._compute_regime_transition_matrix(data['regime_label'])
        return matrices

    def _compute_interaction_matrix(self, feature_data: pd.DataFrame) -> np.ndarray:
        """Compute feature interaction matrix.
        
        Args:
            feature_data: Feature data
            
        Returns:
            Interaction matrix
        """
        n_features = len(feature_data.columns)
        interaction_matrix = np.zeros((n_features, n_features))
        standardized = (feature_data - feature_data.mean()) / (feature_data.std() + 1e-08)
        for i in range(n_features):
            for j in range(i, n_features):
                interaction = (standardized.iloc[:, i] * standardized.iloc[:, j]).mean()
                interaction_matrix[i, j] = interaction
                interaction_matrix[j, i] = interaction
        return interaction_matrix

    def _compute_regime_transition_matrix(self, regime_labels: pd.Series) -> np.ndarray:
        """Compute regime transition matrix.
        
        Args:
            regime_labels: Series of regime labels
            
        Returns:
            Transition matrix
        """
        unique_regimes = sorted(regime_labels.unique())
        n_regimes = len(unique_regimes)
        transition_matrix = np.zeros((n_regimes, n_regimes))
        regime_to_idx = {regime: idx for idx, regime in enumerate(unique_regimes)}
        for i in range(len(regime_labels) - 1):
            from_regime = regime_to_idx[regime_labels.iloc[i]]
            to_regime = regime_to_idx[regime_labels.iloc[i + 1]]
            transition_matrix[from_regime, to_regime] += 1
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)
        return transition_matrix

    @comprehensive_function_tracker(None)
    async def _analyze_feature_importance(self, data_dict: Dict[str, pd.DataFrame], selected_features: List[str], matrix_results: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Analyze feature importance using various methods.
        
        Args:
            data_dict: Dictionary of data splits
            selected_features: List of selected features
            matrix_results: Computed matrices
            
        Returns:
            Feature importance results
        """
        importance_results = {}
        train_data = data_dict.get('train', next(iter(data_dict.values())))
        train_matrices = matrix_results.get('train', {})
        if selected_features:
            feature_cols = selected_features
        else:
            feature_cols = [col for col in train_data.columns if col.startswith('feature_')]
        if 'correlation_matrix' in train_matrices:
            corr_matrix = train_matrices['correlation_matrix']
            if 'label' in train_data.columns:
                feature_data = train_data[feature_cols]
                target_corr = feature_data.corrwith(train_data['label']).abs()
                importance_results['correlation_importance'] = target_corr.to_dict()
        feature_data = train_data[feature_cols]
        variance_importance = feature_data.var()
        importance_results['variance_importance'] = variance_importance.to_dict()
        if 'covariance_matrix' in train_matrices:
            cov_matrix = train_matrices['covariance_matrix']
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            eigenvalue_importance = np.abs(eigenvectors).dot(np.abs(eigenvalues))
            importance_results['eigenvalue_importance'] = {feature_cols[i]: float(eigenvalue_importance[i]) for i in range(len(feature_cols))}
        aggregated_importance = self._aggregate_importance_scores(importance_results, feature_cols)
        importance_results['aggregated_importance'] = aggregated_importance
        return importance_results

    def _aggregate_importance_scores(self, importance_results: Dict[str, Dict[str, float]], feature_names: List[str]) -> Dict[str, float]:
        """Aggregate multiple importance scores.
        
        Args:
            importance_results: Dictionary of importance scores by method
            feature_names: List of feature names
            
        Returns:
            Aggregated importance scores
        """
        aggregated = {}
        for feature in feature_names:
            scores = []
            for method, importance_dict in importance_results.items():
                if isinstance(importance_dict, dict) and feature in importance_dict:
                    score = importance_dict[feature]
                    if not np.isnan(score):
                        scores.append(score)
            if scores:
                normalized_scores = []
                for method, importance_dict in importance_results.items():
                    if isinstance(importance_dict, dict) and feature in importance_dict:
                        values = list(importance_dict.values())
                        min_val = min(values)
                        max_val = max(values)
                        if max_val > min_val:
                            normalized = (importance_dict[feature] - min_val) / (max_val - min_val)
                            normalized_scores.append(normalized)
                if normalized_scores:
                    aggregated[feature] = np.mean(normalized_scores)
        return aggregated

    def _generate_optimization_insights(self, matrix_results: Dict[str, Dict[str, np.ndarray]], importance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimization insights from matrix analysis.
        
        Args:
            matrix_results: Computed matrices
            importance_results: Feature importance results
            
        Returns:
            Optimization insights
        """
        insights = {'feature_recommendations': [], 'matrix_insights': [], 'optimization_suggestions': []}
        if 'aggregated_importance' in importance_results:
            aggregated = importance_results['aggregated_importance']
            sorted_features = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
            top_k = self.matrix_config.get('feature_selection', {}).get('top_k', 50)
            top_features = [f[0] for f in sorted_features[:top_k]]
            insights['feature_recommendations'] = top_features
            min_importance = self.matrix_config.get('feature_selection', {}).get('min_importance', 0.01)
            low_importance = [f[0] for f in sorted_features if f[1] < min_importance]
            if low_importance:
                insights['optimization_suggestions'].append(f'Consider removing {len(low_importance)} low-importance features')
        for split_name, matrices in matrix_results.items():
            if 'correlation_matrix' in matrices:
                corr_matrix = matrices['correlation_matrix']
                high_corr_pairs = []
                n_features = corr_matrix.shape[0]
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        if abs(corr_matrix[i, j]) > 0.95:
                            high_corr_pairs.append((i, j, corr_matrix[i, j]))
                if high_corr_pairs:
                    insights['matrix_insights'].append(f'{split_name}: Found {len(high_corr_pairs)} highly correlated feature pairs')
                    insights['optimization_suggestions'].append('Consider removing redundant features from highly correlated pairs')
        return insights

    def _generate_matrix_reports(self, matrix_results: Dict[str, Dict[str, np.ndarray]], importance_results: Dict[str, Any], optimization_insights: Dict[str, Any]) -> Dict[str, str]:
        """Generate reports for matrix analysis.
        
        Args:
            matrix_results: Computed matrices
            importance_results: Feature importance results
            optimization_insights: Optimization insights
            
        Returns:
            Dictionary of reports
        """
        reports = {}
        summary_lines = ['Enhanced Matrix Operations Summary', '=' * 40, '', 'Matrix Computations:']
        for split_name, matrices in matrix_results.items():
            summary_lines.append(f'\n{split_name.upper()} split:')
            for matrix_name, matrix in matrices.items():
                if isinstance(matrix, np.ndarray):
                    summary_lines.append(f'  {matrix_name}: {matrix.shape} (min={matrix.min():.3f}, max={matrix.max():.3f})')
        if 'aggregated_importance' in importance_results:
            aggregated = importance_results['aggregated_importance']
            top_5 = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)[:5]
            summary_lines.extend(['', 'Top 5 Important Features:'])
            for feature, score in top_5:
                summary_lines.append(f'  {feature}: {score:.3f}')
        reports['summary'] = '\n'.join(summary_lines)
        opt_lines = ['Optimization Insights', '=' * 40, '']
        if optimization_insights.get('feature_recommendations'):
            opt_lines.extend([f"Recommended features: {len(optimization_insights['feature_recommendations'])}", ''])
        for insight in optimization_insights.get('matrix_insights', []):
            opt_lines.append(f'- {insight}')
        opt_lines.append('\nOptimization Suggestions:')
        for suggestion in optimization_insights.get('optimization_suggestions', []):
            opt_lines.append(f'- {suggestion}')
        reports['optimization'] = '\n'.join(opt_lines)
        return reports

    async def _save_outputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> None:
        """Save step outputs to disk.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Pipeline state with results
        """
        output_dir = Path(training_input.get('output_dir', 'output')) / 'step07_matrix_operations'
        output_dir.mkdir(parents=True, exist_ok=True)
        if 'matrix_results' in pipeline_state:
            for split_name, matrices in pipeline_state['matrix_results'].items():
                split_dir = output_dir / split_name
                split_dir.mkdir(exist_ok=True)
                for matrix_name, matrix in matrices.items():
                    if isinstance(matrix, np.ndarray):
                        np.save(split_dir / f'{matrix_name}.npy', matrix)
                self.logger.info(f'💾 Saved matrices for {split_name} split')
        if 'feature_importance' in pipeline_state:
            importance_path = output_dir / 'feature_importance.json'
            with open(importance_path, 'w') as f:
                json.dump(pipeline_state['feature_importance'], f, indent=2)
            self.logger.info(f'💾 Saved feature importance to {importance_path}')
        if 'optimization_insights' in pipeline_state:
            insights_path = output_dir / 'optimization_insights.json'
            with open(insights_path, 'w') as f:
                json.dump(pipeline_state['optimization_insights'], f, indent=2)
            self.logger.info(f'💾 Saved optimization insights')
        if 'matrix_reports' in pipeline_state:
            for report_name, content in pipeline_state['matrix_reports'].items():
                report_path = output_dir / f'{report_name}_report.txt'
                with open(report_path, 'w') as f:
                    f.write(content)
                self.logger.info(f'💾 Saved {report_name} report')
        try:
            symbol = training_input.get('symbol', 'UNKNOWN')
            exchange = training_input.get('exchange', 'UNKNOWN')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir', 'data/training')
            features_dir = Path(data_dir)
            features_dir.mkdir(parents=True, exist_ok=True)
            selected_features = pipeline_state.get('selected_features', [])
            engineered_data = pipeline_state.get('engineered_data', {})

            def _save_split(df: pd.DataFrame, split_name: str) -> None:
                if df is None:
                    return
                if selected_features:
                    available = [c for c in selected_features if c in df.columns]
                    if available:
                        df_to_save = df[available]
                    else:
                        df_to_save = df
                else:
                    df_to_save = df
                out_path = features_dir / f'{exchange}_{symbol}_{timeframe}_features_filtered_{split_name}.parquet'
                try:
                    df_to_save.to_parquet(out_path)
                    self.logger.info(f'💾 Saved filtered features: {out_path}')
                except Exception as e:
                    self.logger.warning(f'⚠️ Failed to save filtered {split_name} features: {e}')
            train_df = engineered_data.get('train') if isinstance(engineered_data, dict) else None
            val_df = engineered_data.get('val') if isinstance(engineered_data, dict) else None
            _save_split(train_df, 'train')
            _save_split(val_df, 'val')
        except Exception as e:
            self.logger.warning(f'⚠️ Skipped filtered feature persistence due to error: {e}')

    def get_required_inputs(self) -> list:
        """Get list of required inputs for this step."""
        return ['engineered_data or split data with features', 'selected_features (optional)']

    def get_produced_outputs(self) -> list:
        """Get list of outputs produced by this step."""
        return ['matrix_results', 'feature_importance', 'optimization_insights', 'matrix_reports']

    def get_dependencies(self) -> list:
        """Get list of step dependencies."""
        return ['06_feature_engineering']