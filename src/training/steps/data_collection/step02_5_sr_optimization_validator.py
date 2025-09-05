from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np

"""Step 2.5: S/R Detection Optimization Validator with Comprehensive Function Call Monitoring.

This module validates the S/R detection optimization step to ensure:
1. Optimization results are properly saved
2. Optimized parameters are correctly formatted
3. Configuration is updated with optimized parameters
4. All required artifacts are present
5. Comprehensive function call monitoring and reporting
"""
import asyncio
import sys
import json
import time
import functools
import traceback
import inspect
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Callable
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
try:
    from src.utils.centralized_decorators import handle_errors, monitor_step_execution, secure_step_execution, validate_pipeline_step, quality_gate
except ImportError:

    def handle_errors(*args, **kwargs) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator

    def monitor_step_execution(*args, **kwargs) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator

    def secure_step_execution(*args, **kwargs) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator

    def validate_pipeline_step(*args, **kwargs) -> bool:

        def decorator(func: Callable) -> None:
            return func
        return decorator

    def quality_gate(*args, **kwargs) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator
try:
    from src.utils.decorators.errors import handles_errors
except ImportError:

    def handles_errors(*args, **kwargs) -> None:

        def decorator(func: Callable) -> None:
            return func
        return decorator
try:
    from src.utils.logger import system_logger
except ImportError:
    import logging

    system_logger = logging.getLogger(__name__)

def safe_json_load(file_path: Union[str, Path]) -> None:
    """Safe JSON loading with fallback."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}
logger = system_logger.getChild('Step2_5SROptimizationValidator')
validator_function_tracker = {'call_count': 0, 'call_history': [], 'performance_metrics': {}, 'error_count': 0, 'success_count': 0}

def monitor_validator_function_calls(func: Callable) -> Callable:
    """Comprehensive function call monitoring decorator for validator."""

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> None:
        validator_function_tracker['call_count'] += 1
        call_id = validator_function_tracker['call_count']
        func_name = func.__name__
        module_name = func.__module__
        start_time = time.time()
        logger.info(f'🔵 VALIDATOR FUNCTION ENTRY [{call_id}] - {module_name}.{func_name}')
        logger.info(f'📥 Parameters: args={len(args)}, kwargs={list(kwargs.keys())}')
        call_info = {'call_id': call_id, 'function_name': func_name, 'module_name': module_name, 'start_time': start_time, 'args_count': len(args), 'kwargs_keys': list(kwargs.keys()), 'status': 'running'}
        validator_function_tracker['call_history'].append(call_info)
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            call_info.update({'status': 'success', 'execution_time': execution_time, 'result_type': type(result).__name__, 'result_size': len(str(result)) if hasattr(result, '__len__') else 1})
            if func_name not in validator_function_tracker['performance_metrics']:
                validator_function_tracker['performance_metrics'][func_name] = {'total_calls': 0, 'total_time': 0, 'avg_time': 0, 'min_time': float('inf'), 'max_time': 0, 'success_count': 0, 'error_count': 0}
            metrics = validator_function_tracker['performance_metrics'][func_name]
            metrics['total_calls'] += 1
            metrics['total_time'] += execution_time
            metrics['avg_time'] = metrics['total_time'] / metrics['total_calls']
            metrics['min_time'] = min(metrics['min_time'], execution_time)
            metrics['max_time'] = max(metrics['max_time'], execution_time)
            metrics['success_count'] += 1
            validator_function_tracker['success_count'] += 1
            logger.info(f'🟢 VALIDATOR FUNCTION EXIT [{call_id}] - {module_name}.{func_name}')
            logger.info(f'⏱️ Execution time: {execution_time:.4f}s')
            logger.info(f'📤 Result type: {type(result).__name__}')
            logger.info(f'✅ Status: SUCCESS')
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            call_info.update({'status': 'error', 'execution_time': execution_time, 'error_type': type(e).__name__, 'error_message': str(e), 'traceback': traceback.format_exc()})
            if func_name not in validator_function_tracker['performance_metrics']:
                validator_function_tracker['performance_metrics'][func_name] = {'total_calls': 0, 'total_time': 0, 'avg_time': 0, 'min_time': float('inf'), 'max_time': 0, 'success_count': 0, 'error_count': 0}
            metrics = validator_function_tracker['performance_metrics'][func_name]
            metrics['total_calls'] += 1
            metrics['total_time'] += execution_time
            metrics['avg_time'] = metrics['total_time'] / metrics['total_calls']
            metrics['min_time'] = min(metrics['min_time'], execution_time)
            metrics['max_time'] = max(metrics['max_time'], execution_time)
            metrics['error_count'] += 1
            validator_function_tracker['error_count'] += 1
            logger.error(f'🔴 VALIDATOR FUNCTION ERROR [{call_id}] - {module_name}.{func_name}')
            logger.error(f'⏱️ Execution time: {execution_time:.4f}s')
            logger.error(f'❌ Error type: {type(e).__name__}')
            logger.error(f'💥 Error message: {str(e)}')
            logger.error(f'📋 Traceback: {traceback.format_exc()}')
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> None:
        validator_function_tracker['call_count'] += 1
        call_id = validator_function_tracker['call_count']
        func_name = func.__name__
        module_name = func.__module__
        start_time = time.time()
        logger.info(f'🔵 VALIDATOR FUNCTION ENTRY [{call_id}] - {module_name}.{func_name}')
        logger.info(f'📥 Parameters: args={len(args)}, kwargs={list(kwargs.keys())}')
        call_info = {'call_id': call_id, 'function_name': func_name, 'module_name': module_name, 'start_time': start_time, 'args_count': len(args), 'kwargs_keys': list(kwargs.keys()), 'status': 'running'}
        validator_function_tracker['call_history'].append(call_info)
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            call_info.update({'status': 'success', 'execution_time': execution_time, 'result_type': type(result).__name__, 'result_size': len(str(result)) if hasattr(result, '__len__') else 1})
            if func_name not in validator_function_tracker['performance_metrics']:
                validator_function_tracker['performance_metrics'][func_name] = {'total_calls': 0, 'total_time': 0, 'avg_time': 0, 'min_time': float('inf'), 'max_time': 0, 'success_count': 0, 'error_count': 0}
            metrics = validator_function_tracker['performance_metrics'][func_name]
            metrics['total_calls'] += 1
            metrics['total_time'] += execution_time
            metrics['avg_time'] = metrics['total_time'] / metrics['total_calls']
            metrics['min_time'] = min(metrics['min_time'], execution_time)
            metrics['max_time'] = max(metrics['max_time'], execution_time)
            metrics['success_count'] += 1
            validator_function_tracker['success_count'] += 1
            logger.info(f'🟢 VALIDATOR FUNCTION EXIT [{call_id}] - {module_name}.{func_name}')
            logger.info(f'⏱️ Execution time: {execution_time:.4f}s')
            logger.info(f'📤 Result type: {type(result).__name__}')
            logger.info(f'✅ Status: SUCCESS')
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            call_info.update({'status': 'error', 'execution_time': execution_time, 'error_type': type(e).__name__, 'error_message': str(e), 'traceback': traceback.format_exc()})
            if func_name not in validator_function_tracker['performance_metrics']:
                validator_function_tracker['performance_metrics'][func_name] = {'total_calls': 0, 'total_time': 0, 'avg_time': 0, 'min_time': float('inf'), 'max_time': 0, 'success_count': 0, 'error_count': 0}
            metrics = validator_function_tracker['performance_metrics'][func_name]
            metrics['total_calls'] += 1
            metrics['total_time'] += execution_time
            metrics['avg_time'] = metrics['total_time'] / metrics['total_calls']
            metrics['min_time'] = min(metrics['min_time'], execution_time)
            metrics['max_time'] = max(metrics['max_time'], execution_time)
            metrics['error_count'] += 1
            validator_function_tracker['error_count'] += 1
            logger.error(f'🔴 VALIDATOR FUNCTION ERROR [{call_id}] - {module_name}.{func_name}')
            logger.error(f'⏱️ Execution time: {execution_time:.4f}s')
            logger.error(f'❌ Error type: {type(e).__name__}')
            logger.error(f'💥 Error message: {str(e)}')
            logger.error(f'📋 Traceback: {traceback.format_exc()}')
            raise
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def validate_validator_inputs(func: Callable) -> Callable:
    """Validate validator function inputs and outputs."""

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> None:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        logger.info(f'🔍 VALIDATOR INPUT VALIDATION - {func.__name__}')
        for param_name, param_value in bound_args.arguments.items():
            param_type = sig.parameters[param_name].annotation
            logger.info(f'  📋 {param_name}: {type(param_value).__name__} = {str(param_value)[:100]}...')
            if param_type != inspect.Parameter.empty:
                try:
                    if hasattr(param_type, '__origin__') or str(param_type).startswith('typing.'):
                        pass
                    elif not isinstance(param_value, param_type):
                        logger.warning(f'  ⚠️ Type mismatch for {param_name}: expected {param_type}, got {type(param_value)}')
                except TypeError:
                    pass
        result = await func(*args, **kwargs)
        logger.info(f'🔍 VALIDATOR OUTPUT VALIDATION - {func.__name__}')
        logger.info(f'  📤 Result type: {type(result).__name__}')
        logger.info(f"  📊 Result size: {(len(str(result)) if hasattr(result, '__len__') else 1)}")
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> None:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        logger.info(f'🔍 VALIDATOR INPUT VALIDATION - {func.__name__}')
        for param_name, param_value in bound_args.arguments.items():
            param_type = sig.parameters[param_name].annotation
            logger.info(f'  📋 {param_name}: {type(param_value).__name__} = {str(param_value)[:100]}...')
            if param_type != inspect.Parameter.empty:
                try:
                    if hasattr(param_type, '__origin__') or str(param_type).startswith('typing.'):
                        pass
                    elif not isinstance(param_value, param_type):
                        logger.warning(f'  ⚠️ Type mismatch for {param_name}: expected {param_type}, got {type(param_value)}')
                except TypeError:
                    pass
        result = func(*args, **kwargs)
        logger.info(f'🔍 VALIDATOR OUTPUT VALIDATION - {func.__name__}')
        logger.info(f'  📤 Result type: {type(result).__name__}')
        logger.info(f"  📊 Result size: {(len(str(result)) if hasattr(result, '__len__') else 1)}")
        return result
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def generate_validator_report() -> Dict[str, Any]:
    """Generate comprehensive validator function call report."""
    total_calls = validator_function_tracker['call_count']
    success_rate = validator_function_tracker['success_count'] / total_calls * 100 if total_calls > 0 else 0
    report = {'summary': {'total_function_calls': total_calls, 'successful_calls': validator_function_tracker['success_count'], 'failed_calls': validator_function_tracker['error_count'], 'success_rate_percent': round(success_rate, 2), 'report_generated_at': datetime.now().isoformat(), 'validator_type': 'step02_5_sr_optimization_validator'}, 'performance_metrics': validator_function_tracker['performance_metrics'], 'call_history': validator_function_tracker['call_history'][-50:], 'top_performing_functions': sorted(validator_function_tracker['performance_metrics'].items(), key=lambda x: x[1]['avg_time'])[:10], 'most_called_functions': sorted(validator_function_tracker['performance_metrics'].items(), key=lambda x: x[1]['total_calls'], reverse=True)[:10]}
    return report

class SROptimizationValidator:
    """Validator for S/R detection optimization step with comprehensive monitoring."""

    @monitor_validator_function_calls
    @validate_validator_inputs
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('SROptimizationValidator')
        self.validation_results = {}
        self.validation_start_time = None
        self.validation_tracker = {'validation_steps': 0, 'step_times': {}, 'step_results': {}, 'total_validation_time': 0}
        self.logger.info('✅ SROptimizationValidator initialized with comprehensive monitoring')

    @monitor_validator_function_calls
    @validate_validator_inputs
    @handles_errors(fallback=False)
    async def validate_step(self, symbol: str, exchange: str, timeframe: str) -> bool:
        """Validate the S/R optimization step with detailed monitoring."""
        try:
            self.logger.info('🔍 Starting S/R optimization validation with comprehensive monitoring...')
            self.validation_start_time = time.time()
            validation_passed = True
            validation_details = []
            pre_report = generate_validator_report()
            self.logger.info(f"📊 Pre-validation function calls: {pre_report['summary']['total_function_calls']}")
            self.logger.info('🔍 Step 1: Validating optimization results file...')
            step_start = time.time()
            results_validation = await self._validate_optimization_results()
            step_time = time.time() - step_start
            self.validation_tracker['validation_steps'] += 1
            self.validation_tracker['step_times']['optimization_results'] = step_time
            self.validation_tracker['step_results']['optimization_results'] = {'valid': results_validation['valid'], 'errors': results_validation.get('errors', []), 'execution_time': step_time}
            if not results_validation['valid']:
                validation_passed = False
                validation_details.extend(results_validation['errors'])
                self.logger.error(f"❌ Optimization results validation failed: {results_validation['errors']}")
            else:
                self.logger.info(f'✅ Optimization results validation passed in {step_time:.4f}s')
            self.logger.info('🔍 Step 2: Validating optimized parameters...')
            step_start = time.time()
            params_validation = await self._validate_optimized_parameters()
            step_time = time.time() - step_start
            self.validation_tracker['validation_steps'] += 1
            self.validation_tracker['step_times']['optimized_parameters'] = step_time
            self.validation_tracker['step_results']['optimized_parameters'] = {'valid': params_validation['valid'], 'errors': params_validation.get('errors', []), 'execution_time': step_time}
            if not params_validation['valid']:
                validation_passed = False
                validation_details.extend(params_validation['errors'])
                self.logger.error(f"❌ Optimized parameters validation failed: {params_validation['errors']}")
            else:
                self.logger.info(f'✅ Optimized parameters validation passed in {step_time:.4f}s')
            self.logger.info('🔍 Step 3: Validating configuration updates...')
            step_start = time.time()
            config_validation = await self._validate_configuration_updates()
            step_time = time.time() - step_start
            self.validation_tracker['validation_steps'] += 1
            self.validation_tracker['step_times']['configuration_updates'] = step_time
            self.validation_tracker['step_results']['configuration_updates'] = {'valid': config_validation['valid'], 'errors': config_validation.get('errors', []), 'execution_time': step_time}
            if not config_validation['valid']:
                validation_passed = False
                validation_details.extend(config_validation['errors'])
                self.logger.error(f"❌ Configuration updates validation failed: {config_validation['errors']}")
            else:
                self.logger.info(f'✅ Configuration updates validation passed in {step_time:.4f}s')
            self.logger.info('🔍 Step 4: Validating artifact quality...')
            step_start = time.time()
            quality_validation = await self._validate_artifact_quality()
            step_time = time.time() - step_start
            self.validation_tracker['validation_steps'] += 1
            self.validation_tracker['step_times']['artifact_quality'] = step_time
            self.validation_tracker['step_results']['artifact_quality'] = {'valid': quality_validation['valid'], 'errors': quality_validation.get('errors', []), 'execution_time': step_time}
            if not quality_validation['valid']:
                validation_passed = False
                validation_details.extend(quality_validation['errors'])
                self.logger.error(f"❌ Artifact quality validation failed: {quality_validation['errors']}")
            else:
                self.logger.info(f'✅ Artifact quality validation passed in {step_time:.4f}s')
            total_validation_time = time.time() - self.validation_start_time
            self.validation_tracker['total_validation_time'] = total_validation_time
            post_report = generate_validator_report()
            self.logger.info(f"📊 Post-validation function calls: {post_report['summary']['total_function_calls']}")
            self.logger.info(f"📈 Function call increase: {post_report['summary']['total_function_calls'] - pre_report['summary']['total_function_calls']}")
            self.validation_results = {'valid': validation_passed, 'details': validation_details, 'timestamp': time.time(), 'step': 'step02_5_sr_optimization', 'validation_tracker': self.validation_tracker, 'function_call_report': post_report}
            if validation_passed:
                self.logger.info('✅ S/R optimization validation passed')
                self.logger.info(f'📊 Total validation time: {total_validation_time:.4f}s')
                self.logger.info(f"📊 Validation steps completed: {self.validation_tracker['validation_steps']}")
            else:
                self.logger.error(f'❌ S/R optimization validation failed: {validation_details}')
                self.logger.error(f'📊 Total validation time: {total_validation_time:.4f}s')
                self.logger.error(f"📊 Validation steps completed: {self.validation_tracker['validation_steps']}")
            return validation_passed
        except Exception as e:
            self.logger.error(f'Failed to validate S/R optimization: {e}')
            self.logger.error(f'📋 Traceback: {traceback.format_exc()}')
            if self.validation_start_time:
                total_validation_time = time.time() - self.validation_start_time
                self.validation_tracker['total_validation_time'] = total_validation_time
                self.validation_tracker['error'] = {'error_type': type(e).__name__, 'error_message': str(e), 'traceback': traceback.format_exc()}
            return False

    @monitor_validator_function_calls
    @validate_validator_inputs
    @handles_errors(default_return={'valid': False, 'errors': ['Validation failed']}, context='optimization_results_validation')
    async def _validate_optimization_results(self) -> Dict[str, Any]:
        """Validate optimization results file with detailed monitoring."""
        try:
            self.logger.info('📊 Validating optimization results file with detailed checks...')
            validation_start = time.time()
            errors = []
            validation_details = {'files_checked': [], 'fields_validated': [], 'validation_time': 0}
            self.logger.info('🔍 Checking optimization results file existence...')
            results_file = Path('data/optimization/sr_optimization_results.json')
            validation_details['files_checked'].append(str(results_file))
            if not results_file.exists():
                error_msg = 'Optimization results file not found'
                errors.append(error_msg)
                self.logger.error(f'❌ {error_msg}: {results_file}')
                return {'valid': False, 'errors': errors, 'details': validation_details}
            else:
                self.logger.info(f'✅ Found optimization results file: {results_file}')
            self.logger.info('🔍 Checking SR predictor results file existence...')
            sr_results_file = Path('optimization_results.json')
            validation_details['files_checked'].append(str(sr_results_file))
            if not sr_results_file.exists():
                error_msg = 'SR predictor optimization results file not found'
                errors.append(error_msg)
                self.logger.error(f'❌ {error_msg}: {sr_results_file}')
                return {'valid': False, 'errors': errors, 'details': validation_details}
            else:
                self.logger.info(f'✅ Found SR predictor results file: {sr_results_file}')
            self.logger.info('🔍 Validating JSON format...')
            try:
                results_data = safe_json_load(results_file)
                self.logger.info(f'✅ JSON format valid, loaded {len(results_data)} top-level keys')
            except json.JSONDecodeError as e:
                error_msg = f'Invalid JSON format in optimization results: {e}'
                errors.append(error_msg)
                self.logger.error(f'❌ {error_msg}')
                return {'valid': False, 'errors': errors, 'details': validation_details}
            self.logger.info('🔍 Validating required fields...')
            required_fields = ['method_weights', 'strength_weights', 'dbscan_params', 'timeframe_weights', 'advanced_params', 'performance_metrics', 'validation_metrics', 'metadata']
            for field in required_fields:
                validation_details['fields_validated'].append(field)
                if field not in results_data:
                    error_msg = f'Missing required field: {field}'
                    errors.append(error_msg)
                    self.logger.error(f'❌ {error_msg}')
                else:
                    self.logger.info(f'✅ Found required field: {field}')
            self.logger.info('🔍 Validating metadata...')
            if 'metadata' in results_data:
                metadata = results_data['metadata']
                self.logger.info(f'📊 Metadata keys: {list(metadata.keys())}')
                if 'step' not in metadata or metadata['step'] != 'step02_5_sr_optimization':
                    error_msg = 'Invalid step metadata'
                    errors.append(error_msg)
                    self.logger.error(f"❌ {error_msg}: {metadata.get('step', 'None')}")
                else:
                    self.logger.info(f"✅ Valid step metadata: {metadata['step']}")
                if 'timestamp' not in metadata:
                    error_msg = 'Missing timestamp in metadata'
                    errors.append(error_msg)
                    self.logger.error(f'❌ {error_msg}')
                else:
                    self.logger.info(f"✅ Found timestamp: {metadata['timestamp']}")
            else:
                error_msg = 'Missing metadata section'
                errors.append(error_msg)
                self.logger.error(f'❌ {error_msg}')
            validation_time = time.time() - validation_start
            validation_details['validation_time'] = validation_time
            validation_result = len(errors) == 0
            self.logger.info(f'📊 Optimization results validation completed in {validation_time:.4f}s')
            self.logger.info(f"📊 Validation result: {('PASSED' if validation_result else 'FAILED')}")
            self.logger.info(f'📊 Errors found: {len(errors)}')
            return {'valid': validation_result, 'errors': errors, 'details': validation_details}
        except Exception as e:
            self.logger.error(f'❌ Optimization results validation error: {e}')
            self.logger.error(f'📋 Traceback: {traceback.format_exc()}')
            return {'valid': False, 'errors': [f'Validation error: {e}'], 'details': {'error': str(e)}}

    @monitor_validator_function_calls
    @validate_validator_inputs
    @handles_errors(default_return={'valid': False, 'errors': ['Validation failed']}, context='optimized_parameters_validation')
    async def _validate_optimized_parameters(self) -> Dict[str, Any]:
        """Validate optimized parameters structure and values with detailed monitoring."""
        try:
            self.logger.info('⚙️ Validating optimized parameters with detailed checks...')
            validation_start = time.time()
            errors = []
            validation_details = {'parameters_checked': [], 'validation_time': 0}
            self.logger.info('🔍 Loading optimization results...')
            results_file = Path('data/optimization/sr_optimization_results.json')
            if not results_file.exists():
                error_msg = 'Optimization results file not found'
                self.logger.error(f'❌ {error_msg}: {results_file}')
                return {'valid': False, 'errors': [error_msg], 'details': validation_details}
            results_data = safe_json_load(results_file)
            self.logger.info(f'✅ Loaded optimization results with {len(results_data)} keys')
            self.logger.info('🔍 Validating method weights...')
            validation_details['parameters_checked'].append('method_weights')
            method_weights = results_data.get('method_weights', {})
            if not isinstance(method_weights, dict):
                error_msg = 'Method weights must be a dictionary'
                errors.append(error_msg)
                self.logger.error(f'❌ {error_msg}: {type(method_weights)}')
            else:
                self.logger.info(f'✅ Method weights is a dictionary with {len(method_weights)} entries')
                for method, weight in method_weights.items():
                    if not isinstance(weight, (int, float)) or weight < 0:
                        error_msg = f'Invalid method weight for {method}: {weight}'
                        errors.append(error_msg)
                        self.logger.error(f'❌ {error_msg}')
                    else:
                        self.logger.info(f'✅ Valid method weight for {method}: {weight}')
            self.logger.info('🔍 Validating strength weights...')
            validation_details['parameters_checked'].append('strength_weights')
            strength_weights = results_data.get('strength_weights', {})
            if not isinstance(strength_weights, dict):
                error_msg = 'Strength weights must be a dictionary'
                errors.append(error_msg)
                self.logger.error(f'❌ {error_msg}: {type(strength_weights)}')
            else:
                self.logger.info(f'✅ Strength weights is a dictionary with {len(strength_weights)} entries')
                for strength, weight in strength_weights.items():
                    if not isinstance(weight, (int, float)) or weight < 0:
                        error_msg = f'Invalid strength weight for {strength}: {weight}'
                        errors.append(error_msg)
                        self.logger.error(f'❌ {error_msg}')
                    else:
                        self.logger.info(f'✅ Valid strength weight for {strength}: {weight}')
            self.logger.info('🔍 Validating DBSCAN parameters...')
            validation_details['parameters_checked'].append('dbscan_params')
            dbscan_params = results_data.get('dbscan_params', {})
            if not isinstance(dbscan_params, dict):
                error_msg = 'DBSCAN parameters must be a dictionary'
                errors.append(error_msg)
                self.logger.error(f'❌ {error_msg}: {type(dbscan_params)}')
            else:
                self.logger.info(f'✅ DBSCAN parameters is a dictionary with {len(dbscan_params)} entries')
                if 'eps' in dbscan_params:
                    if not isinstance(dbscan_params['eps'], (int, float)):
                        error_msg = 'DBSCAN eps must be a number'
                        errors.append(error_msg)
                        self.logger.error(f"❌ {error_msg}: {type(dbscan_params['eps'])}")
                    else:
                        self.logger.info(f"✅ Valid DBSCAN eps: {dbscan_params['eps']}")
                if 'min_samples' in dbscan_params:
                    if not isinstance(dbscan_params['min_samples'], int):
                        error_msg = 'DBSCAN min_samples must be an integer'
                        errors.append(error_msg)
                        self.logger.error(f"❌ {error_msg}: {type(dbscan_params['min_samples'])}")
                    else:
                        self.logger.info(f"✅ Valid DBSCAN min_samples: {dbscan_params['min_samples']}")
            self.logger.info('🔍 Validating performance metrics...')
            validation_details['parameters_checked'].append('performance_metrics')
            performance_metrics = results_data.get('performance_metrics', {})
            if not isinstance(performance_metrics, dict):
                error_msg = 'Performance metrics must be a dictionary'
                errors.append(error_msg)
                self.logger.error(f'❌ {error_msg}: {type(performance_metrics)}')
            else:
                self.logger.info(f'✅ Performance metrics is a dictionary with {len(performance_metrics)} entries')
                required_metrics = ['optimization_score', 'sharpe_ratio', 'win_rate']
                for metric in required_metrics:
                    if metric not in performance_metrics:
                        error_msg = f'Missing performance metric: {metric}'
                        errors.append(error_msg)
                        self.logger.error(f'❌ {error_msg}')
                    elif not isinstance(performance_metrics[metric], (int, float)):
                        error_msg = f'Invalid performance metric {metric}: {performance_metrics[metric]}'
                        errors.append(error_msg)
                        self.logger.error(f'❌ {error_msg}')
                    else:
                        self.logger.info(f'✅ Valid performance metric {metric}: {performance_metrics[metric]}')
            validation_time = time.time() - validation_start
            validation_details['validation_time'] = validation_time
            validation_result = len(errors) == 0
            self.logger.info(f'📊 Optimized parameters validation completed in {validation_time:.4f}s')
            self.logger.info(f"📊 Validation result: {('PASSED' if validation_result else 'FAILED')}")
            self.logger.info(f'📊 Errors found: {len(errors)}')
            return {'valid': validation_result, 'errors': errors, 'details': validation_details}
        except Exception as e:
            self.logger.error(f'❌ Optimized parameters validation error: {e}')
            self.logger.error(f'📋 Traceback: {traceback.format_exc()}')
            return {'valid': False, 'errors': [f'Parameter validation error: {e}'], 'details': {'error': str(e)}}

    @handles_errors(default_return={'valid': False, 'errors': ['Validation failed']}, context='configuration_validation')
    async def _validate_configuration_updates(self) -> Dict[str, Any]:
        """Validate that configuration has been updated with optimized parameters."""
        try:
            self.logger.info('🔧 Validating configuration updates...')
            errors = []
            sr_config = self.config.get('sr_breakout_predictor', {})
            if not sr_config:
                errors.append('SR breakout predictor configuration not found')
                return {'valid': False, 'errors': errors}
            if not sr_config.get('use_optimized_params', False):
                errors.append('use_optimized_params not enabled in SR configuration')
            if 'optimization_results_file' not in sr_config:
                errors.append('optimization_results_file path not set in SR configuration')
            sr_opt_config = self.config.get('sr_detection_optimization', {})
            if not sr_opt_config:
                errors.append('SR detection optimization configuration not found')
            optimized_params = ['optimized_method_weights', 'optimized_strength_weights', 'optimized_dbscan_params', 'optimized_timeframe_weights', 'optimized_advanced_params']
            for param in optimized_params:
                if param not in sr_opt_config:
                    errors.append(f'Optimized parameter {param} not found in configuration')
            return {'valid': len(errors) == 0, 'errors': errors}
        except Exception as e:
            return {'valid': False, 'errors': [f'Configuration validation error: {e}']}

    @handles_errors(default_return={'valid': False, 'errors': ['Validation failed']}, context='artifact_quality_validation')
    async def _validate_artifact_quality(self) -> Dict[str, Any]:
        """Validate the quality of optimization artifacts."""
        try:
            self.logger.info('🎯 Validating artifact quality...')
            errors = []
            results_file = Path('data/optimization/sr_optimization_results.json')
            if not results_file.exists():
                return {'valid': False, 'errors': ['Optimization results file not found']}
            results_data = safe_json_load(results_file)
            performance_metrics = results_data.get('performance_metrics', {})
            optimization_score = performance_metrics.get('optimization_score', 0)
            if optimization_score <= 0:
                errors.append(f'Low optimization score: {optimization_score}')
            sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
            if sharpe_ratio < 0.3:
                errors.append(f'Low Sharpe ratio: {sharpe_ratio}')
            win_rate = performance_metrics.get('win_rate', 0)
            if win_rate < 0.5:
                errors.append(f'Low win rate: {win_rate}')
            validation_metrics = results_data.get('validation_metrics', {})
            cross_validation_score = validation_metrics.get('cross_validation_score', 0)
            if cross_validation_score < 0.6:
                errors.append(f'Low cross-validation score: {cross_validation_score}')
            metadata = results_data.get('metadata', {})
            optimization_time = metadata.get('optimization_time', 0)
            if optimization_time > 3600:
                errors.append(f'Optimization took too long: {optimization_time}s')
            n_trials = metadata.get('n_trials', 0)
            if n_trials < 10:
                errors.append(f'Too few optimization trials: {n_trials}')
            return {'valid': len(errors) == 0, 'errors': errors}
        except Exception as e:
            return {'valid': False, 'errors': [f'Quality validation error: {e}']}

    @monitor_validator_function_calls
    @validate_validator_inputs
    def get_validation_results(self) -> Dict[str, Any]:
        """Get validation results with detailed information."""
        self.logger.info('📊 Retrieving validation results...')
        if 'function_call_report' not in self.validation_results:
            self.validation_results['function_call_report'] = generate_validator_report()
        self.logger.info(f'📊 Validation results contain {len(self.validation_results)} keys')
        return self.validation_results

@monitor_validator_function_calls
@validate_validator_inputs
@handles_errors(fallback=False)
async def run_validation(config: dict[str, Any], symbol: str, exchange: str, timeframe: str) -> bool:
    """Run validation for the S/R optimization step with comprehensive monitoring."""
    try:
        logger.info('🚀 Starting Step 2.5: S/R Detection Optimization Validation with comprehensive monitoring')
        validation_start_time = time.time()
        pre_report = generate_validator_report()
        logger.info(f"📊 Pre-validation function calls: {pre_report['summary']['total_function_calls']}")
        logger.info('🔧 Creating SROptimizationValidator instance...')
        validator = SROptimizationValidator(config)
        logger.info('🔍 Running validation step...')
        success = await validator.validate_step(symbol, exchange, timeframe)
        results = validator.get_validation_results()
        validation_time = time.time() - validation_start_time
        post_report = generate_validator_report()
        logger.info(f"📊 Post-validation function calls: {post_report['summary']['total_function_calls']}")
        logger.info(f"📈 Function call increase: {post_report['summary']['total_function_calls'] - pre_report['summary']['total_function_calls']}")
        if success:
            logger.info('✅ Step 2.5: S/R Detection Optimization Validation completed successfully')
            logger.info(f'📊 Total validation time: {validation_time:.4f}s')
            logger.info(f"📊 Function calls made: {post_report['summary']['total_function_calls']}")
            logger.info(f"📊 Success rate: {post_report['summary']['success_rate_percent']:.2f}%")
        else:
            logger.error(f"❌ Step 2.5: S/R Detection Optimization Validation failed: {results.get('details', [])}")
            logger.error(f'📊 Total validation time: {validation_time:.4f}s')
            logger.error(f"📊 Function calls made: {post_report['summary']['total_function_calls']}")
            logger.error(f"📊 Success rate: {post_report['summary']['success_rate_percent']:.2f}%")
        results['validation_summary'] = {'total_validation_time': validation_time, 'function_call_report': post_report, 'validation_success': success}
        return success
    except Exception as e:
        logger.error(f'Failed to run S/R optimization validation: {e}')
        logger.error(f'📋 Traceback: {traceback.format_exc()}')
        return False

@monitor_validator_function_calls
@validate_validator_inputs
@handles_errors(default_return={'step_name': 'step02_5_sr_optimization', 'validation_passed': False, 'error': 'Validation wrapper failed'})
async def run_validator(training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper to integrate with the validator orchestrator.

    Expects the standard signature (training_input, pipeline_state), extracts
    symbol/exchange/timeframe from training_input, loads configuration from
    pipeline_state if available, and invokes run_validation.
    """
    # Extract parameters with safe defaults
    symbol = training_input.get('symbol', 'ETHUSDT') if isinstance(training_input, dict) else 'ETHUSDT'
    exchange = training_input.get('exchange', 'BINANCE') if isinstance(training_input, dict) else 'BINANCE'
    timeframe = training_input.get('timeframe', '1m') if isinstance(training_input, dict) else '1m'
    # Load config if available
    config = {}
    if isinstance(pipeline_state, dict):
        config = pipeline_state.get('config', {}) or pipeline_state.get('training_config', {}) or {}
    success = await run_validation(config, symbol, exchange, timeframe)
    return {'step_name': 'step02_5_sr_optimization', 'validation_passed': bool(success)}

def save_validator_report(report: Dict[str, Any], filename: str=None) -> str:
    """Save validator function call report to file."""
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'step02_5_validator_report_{timestamp}.json'
    report_path = Path('reports') / filename
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f'📄 Validator report saved to: {report_path}')
    return str(report_path)

def print_validator_summary(report: Dict[str, Any]) -> None:
    """Print a summary of validator function call report."""
    summary = report['summary']
    print('\n' + '=' * 80)
    print('📊 STEP 2.5 VALIDATOR FUNCTION CALL REPORT SUMMARY')
    print('=' * 80)
    print(f"📈 Total Function Calls: {summary['total_function_calls']}")
    print(f"✅ Successful Calls: {summary['successful_calls']}")
    print(f"❌ Failed Calls: {summary['failed_calls']}")
    print(f"📊 Success Rate: {summary['success_rate_percent']:.2f}%")
    print(f"🕐 Report Generated: {summary['report_generated_at']}")
    print(f"🔧 Validator Type: {summary['validator_type']}")
    print('\n🔝 TOP 10 MOST CALLED VALIDATOR FUNCTIONS:')
    print('-' * 50)
    for i, (func_name, metrics) in enumerate(report['most_called_functions'], 1):
        print(f"{i:2d}. {func_name}: {metrics['total_calls']} calls, avg: {metrics['avg_time']:.4f}s")
    print('\n⚡ TOP 10 FASTEST VALIDATOR FUNCTIONS:')
    print('-' * 50)
    for i, (func_name, metrics) in enumerate(report['top_performing_functions'], 1):
        print(f"{i:2d}. {func_name}: {metrics['avg_time']:.4f}s avg, {metrics['total_calls']} calls")
    print('\n📋 RECENT VALIDATOR FUNCTION CALLS (Last 10):')
    print('-' * 50)
    for call in report['call_history'][-10:]:
        status_emoji = '✅' if call['status'] == 'success' else '❌'
        print(f"{status_emoji} [{call['call_id']}] {call['function_name']} - {call.get('execution_time', 0):.4f}s - {call['status']}")
    print('=' * 80)

def _create_test_config() -> Dict[str, Any]:
    """Create test configuration for validation."""
    return {'sr_breakout_predictor': {'use_optimized_params': True, 'optimization_results_file': 'optimization_results.json'}, 'sr_detection_optimization': {'optimized_method_weights': {'fractal': 0.8, 'volume': 0.6}, 'optimized_strength_weights': {'volume': 0.7, 'price': 0.3}, 'optimized_dbscan_params': {'eps': 0.1, 'min_samples': 5}, 'optimized_timeframe_weights': {'15m': 0.3, '1h': 0.3, '4h': 0.25, '1d': 0.15}, 'optimized_advanced_params': {'fibonacci_sensitivity': 0.8}}}

async def _run_test_validation() -> bool:
    """Run test validation with test configuration and comprehensive monitoring."""
    logger.info('🧪 Starting Step 2.5 validator test with comprehensive monitoring')
    test_config = _create_test_config()
    logger.info(f'📊 Test configuration created with {len(test_config)} sections')
    success = await run_validation(test_config, 'ETHUSDT', 'BINANCE', '1m')
    validator_report = generate_validator_report()
    report_path = save_validator_report(validator_report)
    print_validator_summary(validator_report)
    print('\n🎯 STEP 2.5 VALIDATOR TEST RESULT:')
    print('-' * 50)
    print(f'✅ Validation Success: {success}')
    print(f"📞 Function Calls: {validator_report['summary']['total_function_calls']}")
    print(f"📈 Success Rate: {validator_report['summary']['success_rate_percent']:.2f}%")
    print(f'📄 Detailed report saved to: {report_path}')
    return success

async def test_validator() -> None:
    """Test the validator with comprehensive monitoring."""
    logger.info('🧪 Starting comprehensive validator test')
    test_config = _create_test_config()
    validator = SROptimizationValidator(test_config)
    success = await validator.validate_step('ETHUSDT', 'BINANCE', '1m')
    results = validator.get_validation_results()
    validator_report = generate_validator_report()
    report_path = save_validator_report(validator_report)
    print_validator_summary(validator_report)
    print('\n🎯 VALIDATOR TEST DETAILED RESULTS:')
    print('-' * 50)
    print(f'✅ Validation Success: {success}')
    print(f"📊 Validation Steps: {results.get('validation_tracker', {}).get('validation_steps', 0)}")
    print(f"⏱️ Total Validation Time: {results.get('validation_tracker', {}).get('total_validation_time', 0):.4f}s")
    if 'validation_tracker' in results:
        tracker = results['validation_tracker']
        print(f"📊 Step Times: {tracker.get('step_times', {})}")
        print(f"📊 Step Results: {tracker.get('step_results', {})}")
    print(f'📄 Detailed report saved to: {report_path}')
    return success
if __name__ == '__main__':
    success = asyncio.run(test_validator())
    print(f"\n🎯 Final Result: Validation {('SUCCESSFUL' if success else 'FAILED')}")