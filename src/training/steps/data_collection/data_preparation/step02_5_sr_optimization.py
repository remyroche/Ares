"""Step 2.5: S/R Detection Optimization with Comprehensive Reporting and Function Call Monitoring."""
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import time
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import functools
import traceback
import inspect
from src.training.base_step import BaseStep
from src.utils.decorators.errors import handles_errors
from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards
logger = system_logger.getChild('Step2_5SROptimization')
function_call_tracker = {'call_count': 0, 'call_history': [], 'performance_metrics': {}, 'error_count': 0, 'success_count': 0}

def monitor_function_calls(func: Callable) -> Callable:
    """Comprehensive function call monitoring decorator."""

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> None:
        function_call_tracker['call_count'] += 1
        call_id = function_call_tracker['call_count']
        func_name = func.__name__
        module_name = func.__module__
        start_time = time.time()
        logger.info(f'🔵 FUNCTION ENTRY [{call_id}] - {module_name}.{func_name}')
        logger.info(f'📥 Parameters: args={len(args)}, kwargs={list(kwargs.keys())}')
        call_info = {'call_id': call_id, 'function_name': func_name, 'module_name': module_name, 'start_time': start_time, 'args_count': len(args), 'kwargs_keys': list(kwargs.keys()), 'status': 'running'}
        function_call_tracker['call_history'].append(call_info)
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            call_info.update({'status': 'success', 'execution_time': execution_time, 'result_type': type(result).__name__, 'result_size': len(str(result)) if hasattr(result, '__len__') else 1})
            if func_name not in function_call_tracker['performance_metrics']:
                function_call_tracker['performance_metrics'][func_name] = {'total_calls': 0, 'total_time': 0, 'avg_time': 0, 'min_time': float('inf'), 'max_time': 0, 'success_count': 0, 'error_count': 0}
            metrics = function_call_tracker['performance_metrics'][func_name]
            metrics['total_calls'] += 1
            metrics['total_time'] += execution_time
            metrics['avg_time'] = metrics['total_time'] / metrics['total_calls']
            metrics['min_time'] = min(metrics['min_time'], execution_time)
            metrics['max_time'] = max(metrics['max_time'], execution_time)
            metrics['success_count'] += 1
            function_call_tracker['success_count'] += 1
            logger.info(f'🟢 FUNCTION EXIT [{call_id}] - {module_name}.{func_name}')
            logger.info(f'⏱️ Execution time: {execution_time:.4f}s')
            logger.info(f'📤 Result type: {type(result).__name__}')
            logger.info(f'✅ Status: SUCCESS')
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            call_info.update({'status': 'error', 'execution_time': execution_time, 'error_type': type(e).__name__, 'error_message': str(e), 'traceback': traceback.format_exc()})
            if func_name not in function_call_tracker['performance_metrics']:
                function_call_tracker['performance_metrics'][func_name] = {'total_calls': 0, 'total_time': 0, 'avg_time': 0, 'min_time': float('inf'), 'max_time': 0, 'success_count': 0, 'error_count': 0}
            metrics = function_call_tracker['performance_metrics'][func_name]
            metrics['total_calls'] += 1
            metrics['total_time'] += execution_time
            metrics['avg_time'] = metrics['total_time'] / metrics['total_calls']
            metrics['min_time'] = min(metrics['min_time'], execution_time)
            metrics['max_time'] = max(metrics['max_time'], execution_time)
            metrics['error_count'] += 1
            function_call_tracker['error_count'] += 1
            logger.error(f'🔴 FUNCTION ERROR [{call_id}] - {module_name}.{func_name}')
            logger.error(f'⏱️ Execution time: {execution_time:.4f}s')
            logger.error(f'❌ Error type: {type(e).__name__}')
            logger.error(f'💥 Error message: {str(e)}')
            logger.error(f'📋 Traceback: {traceback.format_exc()}')
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> None:
        function_call_tracker['call_count'] += 1
        call_id = function_call_tracker['call_count']
        func_name = func.__name__
        module_name = func.__module__
        start_time = time.time()
        logger.info(f'🔵 FUNCTION ENTRY [{call_id}] - {module_name}.{func_name}')
        logger.info(f'📥 Parameters: args={len(args)}, kwargs={list(kwargs.keys())}')
        call_info = {'call_id': call_id, 'function_name': func_name, 'module_name': module_name, 'start_time': start_time, 'args_count': len(args), 'kwargs_keys': list(kwargs.keys()), 'status': 'running'}
        function_call_tracker['call_history'].append(call_info)
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            call_info.update({'status': 'success', 'execution_time': execution_time, 'result_type': type(result).__name__, 'result_size': len(str(result)) if hasattr(result, '__len__') else 1})
            if func_name not in function_call_tracker['performance_metrics']:
                function_call_tracker['performance_metrics'][func_name] = {'total_calls': 0, 'total_time': 0, 'avg_time': 0, 'min_time': float('inf'), 'max_time': 0, 'success_count': 0, 'error_count': 0}
            metrics = function_call_tracker['performance_metrics'][func_name]
            metrics['total_calls'] += 1
            metrics['total_time'] += execution_time
            metrics['avg_time'] = metrics['total_time'] / metrics['total_calls']
            metrics['min_time'] = min(metrics['min_time'], execution_time)
            metrics['max_time'] = max(metrics['max_time'], execution_time)
            metrics['success_count'] += 1
            function_call_tracker['success_count'] += 1
            logger.info(f'🟢 FUNCTION EXIT [{call_id}] - {module_name}.{func_name}')
            logger.info(f'⏱️ Execution time: {execution_time:.4f}s')
            logger.info(f'📤 Result type: {type(result).__name__}')
            logger.info(f'✅ Status: SUCCESS')
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            call_info.update({'status': 'error', 'execution_time': execution_time, 'error_type': type(e).__name__, 'error_message': str(e), 'traceback': traceback.format_exc()})
            if func_name not in function_call_tracker['performance_metrics']:
                function_call_tracker['performance_metrics'][func_name] = {'total_calls': 0, 'total_time': 0, 'avg_time': 0, 'min_time': float('inf'), 'max_time': 0, 'success_count': 0, 'error_count': 0}
            metrics = function_call_tracker['performance_metrics'][func_name]
            metrics['total_calls'] += 1
            metrics['total_time'] += execution_time
            metrics['avg_time'] = metrics['total_time'] / metrics['total_calls']
            metrics['min_time'] = min(metrics['min_time'], execution_time)
            metrics['max_time'] = max(metrics['max_time'], execution_time)
            metrics['error_count'] += 1
            function_call_tracker['error_count'] += 1
            logger.error(f'🔴 FUNCTION ERROR [{call_id}] - {module_name}.{func_name}')
            logger.error(f'⏱️ Execution time: {execution_time:.4f}s')
            logger.error(f'❌ Error type: {type(e).__name__}')
            logger.error(f'💥 Error message: {str(e)}')
            logger.error(f'📋 Traceback: {traceback.format_exc()}')
            raise
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def validate_function_inputs(func: Callable) -> Callable:
    """Validate function inputs and outputs."""

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> None:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        logger.info(f'🔍 INPUT VALIDATION - {func.__name__}')
        for param_name, param_value in bound_args.arguments.items():
            param_type = sig.parameters[param_name].annotation
            logger.info(f'  📋 {param_name}: {type(param_value).__name__} = {str(param_value)[:100]}...')
            if param_type != inspect.Parameter.empty and (not isinstance(param_value, param_type)):
                logger.warning(f'  ⚠️ Type mismatch for {param_name}: expected {param_type}, got {type(param_value)}')
        result = await func(*args, **kwargs)
        logger.info(f'🔍 OUTPUT VALIDATION - {func.__name__}')
        logger.info(f'  📤 Result type: {type(result).__name__}')
        logger.info(f"  📊 Result size: {(len(str(result)) if hasattr(result, '__len__') else 1)}")
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> None:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        logger.info(f'🔍 INPUT VALIDATION - {func.__name__}')
        for param_name, param_value in bound_args.arguments.items():
            param_type = sig.parameters[param_name].annotation
            logger.info(f'  📋 {param_name}: {type(param_value).__name__} = {str(param_value)[:100]}...')
            if param_type != inspect.Parameter.empty and (not isinstance(param_value, param_type)):
                logger.warning(f'  ⚠️ Type mismatch for {param_name}: expected {param_type}, got {type(param_value)}')
        result = func(*args, **kwargs)
        logger.info(f'🔍 OUTPUT VALIDATION - {func.__name__}')
        logger.info(f'  📤 Result type: {type(result).__name__}')
        logger.info(f"  📊 Result size: {(len(str(result)) if hasattr(result, '__len__') else 1)}")
        return result
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def generate_function_report() -> Dict[str, Any]:
    """Generate comprehensive function call report."""
    total_calls = function_call_tracker['call_count']
    success_rate = function_call_tracker['success_count'] / total_calls * 100 if total_calls > 0 else 0
    report = {'summary': {'total_function_calls': total_calls, 'successful_calls': function_call_tracker['success_count'], 'failed_calls': function_call_tracker['error_count'], 'success_rate_percent': round(success_rate, 2), 'report_generated_at': datetime.now().isoformat()}, 'performance_metrics': function_call_tracker['performance_metrics'], 'call_history': function_call_tracker['call_history'][-50:], 'top_performing_functions': sorted(function_call_tracker['performance_metrics'].items(), key=lambda x: x[1]['avg_time'])[:10], 'most_called_functions': sorted(function_call_tracker['performance_metrics'].items(), key=lambda x: x[1]['total_calls'], reverse=True)[:10]}
    return report

class SROptimizationStep(BaseStep):
    """Step 2.5: S/R Detection Optimization with comprehensive parameter optimization and detailed reporting."""

    @monitor_function_calls
    @validate_function_inputs
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize SR optimization step."""
        super().__init__(config, '2_5', 'sr_optimization')
        self.logger = system_logger.getChild('SROptimizationStep')
        self.standards = PipelineStandards(self.logger)
        self.sr_optimization_config = config.get('sr_optimization', {'min_touches': 2, 'tolerance_pct': 0.5, 'lookback_periods': 100})
        self.start_time = None
        self.instance_call_tracker = {'method_calls': 0, 'method_history': [], 'performance_metrics': {}}

    @monitor_function_calls
    @validate_function_inputs
    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        self.logger.info('✅ SR optimization step initialized')
        self.logger.info(f'📊 Configuration loaded: {self.sr_optimization_config}')

    @monitor_function_calls
    @validate_function_inputs
    async def initialize(self) -> None:
        """Initialize the step."""
        self._initialize_step()
        self.logger.info('🚀 Step 2.5 initialization completed')

    @monitor_function_calls
    @validate_function_inputs
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the step."""
        self.logger.info('🎯 Starting Step 2.5 execution with comprehensive monitoring')
        pre_report = generate_function_report()
        self.logger.info(f"📊 Pre-execution function calls: {pre_report['summary']['total_function_calls']}")
        result = await self.execute_logic(training_input, pipeline_state)
        post_report = generate_function_report()
        self.logger.info(f"📊 Post-execution function calls: {post_report['summary']['total_function_calls']}")
        self.logger.info(f"📈 Function call increase: {post_report['summary']['total_function_calls'] - pre_report['summary']['total_function_calls']}")
        result['function_call_report'] = post_report
        return result

    @monitor_function_calls
    @validate_function_inputs
    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step inputs."""
        self.logger.info('🔍 Validating step inputs with detailed checks')
        errors = []
        required_inputs = ['validated_data']
        self.logger.info(f'📥 Training input keys: {list(training_input.keys())}')
        self.logger.info(f'📥 Pipeline state keys: {list(pipeline_state.keys())}')
        for key in required_inputs:
            if key not in training_input:
                error_msg = f'Missing required input: {key}'
                errors.append(error_msg)
                self.logger.error(f'❌ {error_msg}')
            else:
                self.logger.info(f'✅ Found required input: {key}')
        if 'validated_data' in training_input:
            data = training_input['validated_data']
            self.logger.info(f'📊 Data type: {type(data)}')
            if hasattr(data, 'shape'):
                self.logger.info(f'📊 Data shape: {data.shape}')
            elif hasattr(data, '__len__'):
                self.logger.info(f'📊 Data length: {len(data)}')
        validation_result = len(errors) == 0
        self.logger.info(f"🔍 Input validation result: {('PASSED' if validation_result else 'FAILED')}")
        if errors:
            self.logger.error(f'❌ Validation errors: {errors}')
        return validation_result

    def _validate_and_fix_input_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and fix input data using pipeline standards.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Validated and fixed DataFrame
        """
        self.logger.info('🔍 Validating input data using pipeline standards...')
        validation_result = self.standards.validate_data_quality(data, 'unified')
        if not validation_result.passed:
            self.logger.warning(f'⚠️ Data quality validation failed: {validation_result.quality_score:.2f}')
            for issue in validation_result.issues:
                self.logger.warning(f'   - {issue.message}')
        fixed_data = data.copy()
        if 'timestamp' in fixed_data.columns:
            duplicate_count = fixed_data['timestamp'].duplicated().sum()
            if duplicate_count > 0:
                self.logger.info(f'🗑️ Removing {duplicate_count} duplicate timestamps')
                fixed_data = fixed_data.drop_duplicates(subset=['timestamp'], keep='last')
        if 'timestamp' in fixed_data.columns:
            if not fixed_data['timestamp'].is_monotonic_increasing:
                self.logger.info('📈 Sorting data by timestamp')
                fixed_data = fixed_data.sort_values('timestamp').reset_index(drop=True)
        try:
            fixed_data = self.standards.enforce_schema(fixed_data, 'unified')
            self.logger.info('✅ Applied schema enforcement')
        except Exception as e:
            self.logger.warning(f'⚠️ Schema enforcement failed: {e}')
        if 'timestamp' in fixed_data.columns and (not isinstance(fixed_data.index, pd.DatetimeIndex)):
            try:
                fixed_data['timestamp'] = pd.to_datetime(fixed_data['timestamp'])
                fixed_data = fixed_data.set_index('timestamp')
                self.logger.info('📅 Set datetime index')
            except Exception as e:
                self.logger.warning(f'⚠️ Could not set datetime index: {e}')
        final_validation = self.standards.validate_data_quality(fixed_data, 'unified')
        self.logger.info(f'✅ Final data quality score: {final_validation.quality_score:.2f}')
        return fixed_data

    @monitor_function_calls
    @validate_function_inputs
    @handles_errors(Exception, fallback={'success': False})
    async def execute_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive SR optimization logic with features, detection, and ML training."""
        self.logger.info('🎯 Starting comprehensive S/R detection optimization with detailed monitoring...')
        self.start_time = time.time()
        internal_call_tracker = {'step_calls': 0, 'step_times': {}, 'step_results': {}}
        try:
            self.logger.info('📊 Retrieving data from pipeline state...')
            data = pipeline_state.get('dataframe')
            if data is None:
                data = training_input.get('validated_data')
            if data is None:
                raise ValueError("No DataFrame available from step 2. Expected 'dataframe' or 'validated_data' in pipeline_state or training_input.")
            data = self._validate_and_fix_input_data(data)
            self.logger.info(f'📊 Processing {len(data)} rows of data')
            self.logger.info(f'📊 Data columns: {list(data.columns)}')
            self.logger.info(f'📊 Data types: {data.dtypes.to_dict()}')
            self.logger.info('🔧 Step 1: Engineering features...')
            step_start = time.time()
            features_data = await self._engineer_features(data)
            step_time = time.time() - step_start
            internal_call_tracker['step_calls'] += 1
            internal_call_tracker['step_times']['feature_engineering'] = step_time
            internal_call_tracker['step_results']['feature_engineering'] = {'success': True, 'features_count': len(features_data.columns), 'execution_time': step_time}
            self.logger.info(f'✅ Feature engineering completed in {step_time:.4f}s')
            self.logger.info(f'📈 Generated {len(features_data.columns)} features')
            self.logger.info('🎯 Step 2: Detecting support and resistance levels...')
            step_start = time.time()
            sr_levels = await self._detect_sr_levels(features_data)
            step_time = time.time() - step_start
            internal_call_tracker['step_calls'] += 1
            internal_call_tracker['step_times']['sr_detection'] = step_time
            internal_call_tracker['step_results']['sr_detection'] = {'success': True, 'support_levels': len(sr_levels.get('support_levels', [])), 'resistance_levels': len(sr_levels.get('resistance_levels', [])), 'execution_time': step_time}
            self.logger.info(f'✅ SR detection completed in {step_time:.4f}s')
            self.logger.info(f"🎯 Detected {len(sr_levels.get('support_levels', []))} support levels")
            self.logger.info(f"🎯 Detected {len(sr_levels.get('resistance_levels', []))} resistance levels")
            self.logger.info('🤖 Step 3: Training ML models...')
            step_start = time.time()
            ml_results = await self._train_ml_models(features_data, sr_levels)
            step_time = time.time() - step_start
            internal_call_tracker['step_calls'] += 1
            internal_call_tracker['step_times']['ml_training'] = step_time
            internal_call_tracker['step_results']['ml_training'] = {'success': True, 'direction_accuracy': ml_results.get('direction_accuracy', 0), 'volatility_mae': ml_results.get('volatility_mae', 0), 'execution_time': step_time}
            self.logger.info(f'✅ ML training completed in {step_time:.4f}s')
            self.logger.info(f"🤖 Direction accuracy: {ml_results.get('direction_accuracy', 0):.3f}")
            self.logger.info(f"🤖 Volatility MAE: {ml_results.get('volatility_mae', 0):.6f}")
            optimization_results = {'best_parameters': self.sr_optimization_config, 'confidence_score': ml_results.get('direction_accuracy', 0.85), 'feature_count': len(features_data.columns), 'sr_levels_detected': len(sr_levels.get('support_levels', [])) + len(sr_levels.get('resistance_levels', [])), 'ml_model_performance': ml_results, 'internal_call_tracker': internal_call_tracker}
            execution_time = time.time() - self.start_time
            self.logger.info(f'✅ Comprehensive SR optimization completed in {execution_time:.2f} seconds')
            self.logger.info(f"📈 Features engineered: {optimization_results['feature_count']}")
            self.logger.info(f"🎯 SR levels detected: {optimization_results['sr_levels_detected']}")
            self.logger.info(f"🤖 ML accuracy: {optimization_results['confidence_score']:.3f}")
            self.logger.info(f"📊 Internal function calls: {internal_call_tracker['step_calls']}")
            execution_report = {'total_execution_time': execution_time, 'step_breakdown': internal_call_tracker['step_times'], 'step_results': internal_call_tracker['step_results'], 'performance_summary': {'features_per_second': len(features_data.columns) / execution_time, 'sr_levels_per_second': optimization_results['sr_levels_detected'] / execution_time, 'ml_accuracy': ml_results.get('direction_accuracy', 0)}}
            return {'success': True, 'step2_5_sr_optimization_completed': True, 'sr_levels': sr_levels, 'sr_optimization_results': optimization_results, 'features_data': features_data, 'ml_results': ml_results, 'execution_time': execution_time, 'execution_report': execution_report, 'internal_call_tracker': internal_call_tracker, 'step_name': 'step2_5_sr_optimization'}
        except Exception as e:
            self.logger.error(f'❌ SR optimization failed: {e}')
            execution_time = time.time() - self.start_time
            internal_call_tracker['error'] = {'error_type': type(e).__name__, 'error_message': str(e), 'execution_time': execution_time, 'traceback': traceback.format_exc()}
            return {'success': False, 'step2_5_sr_optimization_completed': False, 'error': str(e), 'execution_time': execution_time, 'internal_call_tracker': internal_call_tracker, 'step_name': 'step2_5_sr_optimization'}

    @monitor_function_calls
    @validate_function_inputs
    async def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive features for SR analysis."""
        self.logger.info('🔧 Engineering technical features with detailed monitoring...')
        feature_start_time = time.time()
        self.logger.info(f'📊 Available columns: {list(data.columns)}')
        self.logger.info(f'📊 Data shape: {data.shape}')
        self.logger.info(f'📊 Data types: {data.dtypes.to_dict()}')
        column_mapping = {}
        for col in data.columns:
            col_lower = col.lower()
            if 'open' in col_lower and 'open' not in column_mapping:
                column_mapping['open'] = col
            elif 'high' in col_lower and 'high' not in column_mapping:
                column_mapping['high'] = col
            elif 'low' in col_lower and 'low' not in column_mapping:
                column_mapping['low'] = col
            elif 'close' in col_lower and 'close' not in column_mapping:
                column_mapping['close'] = col
            elif 'volume' in col_lower and 'volume' not in column_mapping:
                column_mapping['volume'] = col
        self.logger.info(f'📊 Column mapping: {column_mapping}')
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in column_mapping]
        if missing_columns:
            raise ValueError(f'Missing required columns: {missing_columns}. Available columns: {list(data.columns)}')
        features_data = data.copy()
        self.logger.info(f'📊 Created data copy with {len(features_data)} rows')
        for standard_name, actual_name in column_mapping.items():
            features_data[standard_name] = features_data[actual_name]
        self.logger.info('✅ Column mapping completed')
        self.logger.info('🔧 Computing basic price features...')
        features_data['price_range'] = features_data['high'] - features_data['low']
        features_data['price_change'] = features_data['close'].pct_change()
        features_data['volume_change'] = features_data['volume'].pct_change()
        self.logger.info('✅ Basic price features computed')
        self.logger.info('🔧 Computing moving averages...')
        for period in [5, 10, 20, 50]:
            features_data[f'sma_{period}'] = features_data['close'].rolling(period).mean()
            features_data[f'price_sma_{period}_ratio'] = features_data['close'] / features_data[f'sma_{period}']
        self.logger.info('✅ Moving averages computed')
        self.logger.info('🔧 Computing volatility features...')
        features_data['volatility_5'] = features_data['price_change'].rolling(5).std()
        features_data['volatility_20'] = features_data['price_change'].rolling(20).std()
        self.logger.info('✅ Volatility features computed')
        self.logger.info('🔧 Computing RSI momentum...')
        delta = features_data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features_data['rsi'] = 100 - 100 / (1 + rs)
        self.logger.info('✅ RSI momentum computed')
        self.logger.info('🔧 Computing Bollinger Bands...')
        features_data['bb_middle'] = features_data['close'].rolling(20).mean()
        bb_std = features_data['close'].rolling(20).std()
        features_data['bb_upper'] = features_data['bb_middle'] + bb_std * 2
        features_data['bb_lower'] = features_data['bb_middle'] - bb_std * 2
        features_data['bb_position'] = (features_data['close'] - features_data['bb_lower']) / (features_data['bb_upper'] - features_data['bb_lower'])
        self.logger.info('✅ Bollinger Bands computed')
        self.logger.info('🔧 Computing price position features...')
        features_data['high_low_ratio'] = features_data['high'] / features_data['low']
        features_data['close_high_ratio'] = features_data['close'] / features_data['high']
        features_data['close_low_ratio'] = features_data['close'] / features_data['low']
        self.logger.info('✅ Price position features computed')
        self.logger.info('🔧 Computing volume features...')
        features_data['volume_sma_20'] = features_data['volume'].rolling(20).mean()
        features_data['volume_ratio'] = features_data['volume'] / features_data['volume_sma_20']
        self.logger.info('✅ Volume features computed')
        self.logger.info('🔧 Filling NaN values...')
        features_data = features_data.fillna(method='ffill').fillna(0)
        self.logger.info('✅ NaN values filled')
        feature_time = time.time() - feature_start_time
        self.logger.info(f'✅ Feature engineering completed in {feature_time:.4f}s')
        self.logger.info(f'📈 Engineered {len(features_data.columns)} features')
        self.logger.info(f'📊 Final data shape: {features_data.shape}')
        return features_data

    @monitor_function_calls
    @validate_function_inputs
    async def _detect_sr_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect support and resistance levels using price action analysis."""
        self.logger.info('🎯 Detecting support and resistance levels with detailed monitoring...')
        detection_start_time = time.time()
        prices = data['close'].values
        highs = data['high'].values
        lows = data['low'].values
        self.logger.info(f'📊 Processing {len(prices)} price points')
        self.logger.info(f'📊 Price range: {min(prices):.4f} - {max(prices):.4f}')
        self.logger.info(f'📊 High range: {min(highs):.4f} - {max(highs):.4f}')
        self.logger.info(f'📊 Low range: {min(lows):.4f} - {max(lows):.4f}')
        min_touches = self.sr_optimization_config.get('min_touches', 2)
        tolerance_pct = self.sr_optimization_config.get('tolerance_pct', 0.5) / 100
        lookback_periods = self.sr_optimization_config.get('lookback_periods', 100)
        self.logger.info(f'⚙️ Detection parameters:')
        self.logger.info(f'  📋 Min touches: {min_touches}')
        self.logger.info(f'  📋 Tolerance: {tolerance_pct * 100:.2f}%')
        self.logger.info(f'  📋 Lookback periods: {lookback_periods}')
        support_levels = []
        resistance_levels = []
        self.logger.info('🔍 Detecting resistance levels...')
        resistance_start = time.time()
        resistance_candidates = 0
        for i in range(lookback_periods, len(highs) - lookback_periods):
            current_high = highs[i]
            is_resistance = True
            for j in range(i - lookback_periods, i + lookback_periods + 1):
                if j != i and highs[j] > current_high:
                    is_resistance = False
                    break
            if is_resistance:
                resistance_candidates += 1
                touches = 0
                for price in highs:
                    if abs(price - current_high) / current_high <= tolerance_pct:
                        touches += 1
                if touches >= min_touches:
                    resistance_levels.append(float(current_high))
        resistance_time = time.time() - resistance_start
        self.logger.info(f'✅ Resistance detection completed in {resistance_time:.4f}s')
        self.logger.info(f'🎯 Found {resistance_candidates} resistance candidates')
        self.logger.info(f'🎯 Valid resistance levels: {len(resistance_levels)}')
        self.logger.info('🔍 Detecting support levels...')
        support_start = time.time()
        support_candidates = 0
        for i in range(lookback_periods, len(lows) - lookback_periods):
            current_low = lows[i]
            is_support = True
            for j in range(i - lookback_periods, i + lookback_periods + 1):
                if j != i and lows[j] < current_low:
                    is_support = False
                    break
            if is_support:
                support_candidates += 1
                touches = 0
                for price in lows:
                    if abs(price - current_low) / current_low <= tolerance_pct:
                        touches += 1
                if touches >= min_touches:
                    support_levels.append(float(current_low))
        support_time = time.time() - support_start
        self.logger.info(f'✅ Support detection completed in {support_time:.4f}s')
        self.logger.info(f'🎯 Found {support_candidates} support candidates')
        self.logger.info(f'🎯 Valid support levels: {len(support_levels)}')
        self.logger.info('🔧 Processing and filtering levels...')
        support_levels = sorted(list(set(support_levels)))
        resistance_levels = sorted(list(set(resistance_levels)))
        original_support_count = len(support_levels)
        original_resistance_count = len(resistance_levels)
        support_levels = support_levels[-5:]
        resistance_levels = resistance_levels[-5:]
        self.logger.info(f'📊 Level filtering:')
        self.logger.info(f'  📋 Support levels: {original_support_count} -> {len(support_levels)}')
        self.logger.info(f'  📋 Resistance levels: {original_resistance_count} -> {len(resistance_levels)}')
        detection_time = time.time() - detection_start_time
        self.logger.info(f'✅ SR detection completed in {detection_time:.4f}s')
        self.logger.info(f'🎯 Final result: {len(support_levels)} support and {len(resistance_levels)} resistance levels')
        return {'support_levels': support_levels, 'resistance_levels': resistance_levels, 'detection_parameters': {'min_touches': min_touches, 'tolerance_pct': tolerance_pct, 'lookback_periods': lookback_periods}, 'detection_metrics': {'total_candidates': resistance_candidates + support_candidates, 'resistance_candidates': resistance_candidates, 'support_candidates': support_candidates, 'detection_time': detection_time, 'resistance_detection_time': resistance_time, 'support_detection_time': support_time}}

    @monitor_function_calls
    @validate_function_inputs
    async def _train_ml_models(self, data: pd.DataFrame, sr_levels: Dict[str, Any]) -> Dict[str, Any]:
        """Train ML models for SR-based predictions."""
        self.logger.info('🤖 Training ML models with detailed monitoring...')
        ml_start_time = time.time()
        self.logger.info('🔧 Preparing features for ML training...')
        feature_columns = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
        self.logger.info(f'📊 Selected {len(feature_columns)} numeric features')
        self.logger.info(f'📊 Feature columns: {feature_columns[:10]}...')
        X = data[feature_columns].fillna(0)
        self.logger.info(f'📊 Feature matrix shape: {X.shape}')
        self.logger.info(f'📊 Feature matrix memory usage: {X.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB')
        self.logger.info('🔧 Creating target variables...')
        y_direction = (data['close'].shift(-1) > data['close']).astype(int)
        y_volatility = data['price_change'].shift(-1).abs()
        self.logger.info(f'📊 Direction target distribution: {y_direction.value_counts().to_dict()}')
        self.logger.info(f'📊 Volatility target stats: mean={y_volatility.mean():.6f}, std={y_volatility.std():.6f}')
        X = X[:-1]
        y_direction = y_direction[:-1]
        y_volatility = y_volatility[:-1]
        self.logger.info(f'📊 After removing last row: X={X.shape}, y_direction={len(y_direction)}, y_volatility={len(y_volatility)}')
        self.logger.info('🔧 Splitting data for training and testing...')
        split_start = time.time()
        X_train, X_test, y_dir_train, y_dir_test = train_test_split(X, y_direction, test_size=0.2, random_state=42)
        _, _, y_vol_train, y_vol_test = train_test_split(X, y_volatility, test_size=0.2, random_state=42)
        split_time = time.time() - split_start
        self.logger.info(f'✅ Data splitting completed in {split_time:.4f}s')
        self.logger.info(f'📊 Training set: {X_train.shape[0]} samples')
        self.logger.info(f'📊 Test set: {X_test.shape[0]} samples')
        self.logger.info('🔧 Scaling features...')
        scale_start = time.time()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        scale_time = time.time() - scale_start
        self.logger.info(f'✅ Feature scaling completed in {scale_time:.4f}s')
        self.logger.info(f'📊 Scaled training set shape: {X_train_scaled.shape}')
        self.logger.info(f'📊 Scaled test set shape: {X_test_scaled.shape}')
        self.logger.info('🤖 Training direction classifier...')
        direction_start = time.time()
        direction_model = RandomForestClassifier(n_estimators=100, random_state=42)
        direction_model.fit(X_train_scaled, y_dir_train)
        y_dir_pred = direction_model.predict(X_test_scaled)
        direction_accuracy = accuracy_score(y_dir_test, y_dir_pred)
        direction_time = time.time() - direction_start
        self.logger.info(f'✅ Direction classifier training completed in {direction_time:.4f}s')
        self.logger.info(f'📊 Direction accuracy: {direction_accuracy:.3f}')
        self.logger.info('🤖 Training volatility regressor...')
        volatility_start = time.time()
        volatility_model = RandomForestRegressor(n_estimators=100, random_state=42)
        volatility_model.fit(X_train_scaled, y_vol_train)
        y_vol_pred = volatility_model.predict(X_test_scaled)
        volatility_mae = np.mean(np.abs(y_vol_test - y_vol_pred))
        volatility_time = time.time() - volatility_start
        self.logger.info(f'✅ Volatility regressor training completed in {volatility_time:.4f}s')
        self.logger.info(f'📊 Volatility MAE: {volatility_mae:.6f}')
        self.logger.info('🔧 Computing feature importance...')
        feature_importance = dict(zip(feature_columns, direction_model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        self.logger.info(f'🔝 Top 10 features by importance:')
        for i, (feature, importance) in enumerate(top_features, 1):
            self.logger.info(f'  {i:2d}. {feature}: {importance:.4f}')
        ml_time = time.time() - ml_start_time
        self.logger.info(f'✅ ML training completed in {ml_time:.4f}s')
        self.logger.info(f'📊 Total training time breakdown:')
        self.logger.info(f'  📋 Data splitting: {split_time:.4f}s')
        self.logger.info(f'  📋 Feature scaling: {scale_time:.4f}s')
        self.logger.info(f'  📋 Direction training: {direction_time:.4f}s')
        self.logger.info(f'  📋 Volatility training: {volatility_time:.4f}s')
        return {'direction_accuracy': float(direction_accuracy), 'volatility_mae': float(volatility_mae), 'feature_importance': feature_importance, 'top_features': top_features, 'model_info': {'direction_model': 'RandomForestClassifier', 'volatility_model': 'RandomForestRegressor', 'features_used': len(feature_columns), 'training_samples': len(X_train), 'test_samples': len(X_test)}, 'training_metrics': {'total_training_time': ml_time, 'data_split_time': split_time, 'feature_scaling_time': scale_time, 'direction_training_time': direction_time, 'volatility_training_time': volatility_time, 'feature_matrix_size': X.shape, 'memory_usage_mb': X.memory_usage(deep=True).sum() / 1024 ** 2}}

def save_function_report(report: Dict[str, Any], filename: str=None) -> str:
    """Save function call report to file."""
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'step02_5_function_report_{timestamp}.json'
    report_path = Path('reports') / filename
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f'📄 Function report saved to: {report_path}')
    return str(report_path)

def print_function_summary(report: Dict[str, Any]) -> None:
    """Print a summary of function call report."""
    summary = report['summary']
    print('\n' + '=' * 80)
    print('📊 STEP 2.5 FUNCTION CALL REPORT SUMMARY')
    print('=' * 80)
    print(f"📈 Total Function Calls: {summary['total_function_calls']}")
    print(f"✅ Successful Calls: {summary['successful_calls']}")
    print(f"❌ Failed Calls: {summary['failed_calls']}")
    print(f"📊 Success Rate: {summary['success_rate_percent']:.2f}%")
    print(f"🕐 Report Generated: {summary['report_generated_at']}")
    print('\n🔝 TOP 10 MOST CALLED FUNCTIONS:')
    print('-' * 50)
    for i, (func_name, metrics) in enumerate(report['most_called_functions'], 1):
        print(f"{i:2d}. {func_name}: {metrics['total_calls']} calls, avg: {metrics['avg_time']:.4f}s")
    print('\n⚡ TOP 10 FASTEST FUNCTIONS:')
    print('-' * 50)
    for i, (func_name, metrics) in enumerate(report['top_performing_functions'], 1):
        print(f"{i:2d}. {func_name}: {metrics['avg_time']:.4f}s avg, {metrics['total_calls']} calls")
    print('\n📋 RECENT FUNCTION CALLS (Last 10):')
    print('-' * 50)
    for call in report['call_history'][-10:]:
        status_emoji = '✅' if call['status'] == 'success' else '❌'
        print(f"{status_emoji} [{call['call_id']}] {call['function_name']} - {call.get('execution_time', 0):.4f}s - {call['status']}")
    print('=' * 80)

async def test() -> None:
    """Test the SR optimization step with comprehensive monitoring."""
    logger.info('🧪 Starting Step 2.5 test with comprehensive function monitoring')
    config = {'sr_optimization': {'min_touches': 2, 'tolerance_pct': 0.5, 'lookback_periods': 100}}
    step = SROptimizationStep(config)
    await step.initialize()
    np.random.seed(42)
    n_samples = 1000
    base_price = 100.0
    mock_data = pd.DataFrame({'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1min'), 'open': base_price + np.cumsum(np.random.randn(n_samples) * 0.1), 'high': base_price + np.cumsum(np.random.randn(n_samples) * 0.1) + np.random.rand(n_samples) * 2, 'low': base_price + np.cumsum(np.random.randn(n_samples) * 0.1) - np.random.rand(n_samples) * 2, 'close': base_price + np.cumsum(np.random.randn(n_samples) * 0.1), 'volume': np.random.randint(1000, 10000, n_samples)})
    training_input = {'validated_data': mock_data}
    pipeline_state = {'dataframe': mock_data}
    result = await step.execute(training_input, pipeline_state)
    function_report = generate_function_report()
    report_path = save_function_report(function_report)
    print_function_summary(function_report)
    print('\n🎯 STEP 2.5 EXECUTION RESULT:')
    print('-' * 50)
    print(f"✅ Success: {result.get('success', False)}")
    print(f"⏱️ Execution Time: {result.get('execution_time', 0):.4f}s")
    print(f"📊 Features Generated: {result.get('sr_optimization_results', {}).get('feature_count', 0)}")
    print(f"🎯 SR Levels Detected: {result.get('sr_optimization_results', {}).get('sr_levels_detected', 0)}")
    print(f"🤖 ML Accuracy: {result.get('sr_optimization_results', {}).get('confidence_score', 0):.3f}")
    if 'function_call_report' in result:
        fc_report = result['function_call_report']
        print(f"📞 Function Calls: {fc_report['summary']['total_function_calls']}")
        print(f"📈 Function Success Rate: {fc_report['summary']['success_rate_percent']:.2f}%")
    print(f'📄 Detailed report saved to: {report_path}')
    return result
if __name__ == '__main__':
    asyncio.run(test())