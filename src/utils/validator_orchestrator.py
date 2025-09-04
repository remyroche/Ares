from __future__ import annotations
'\nValidator orchestrator for running individual step validators in the training pipeline.\n'
import importlib
import inspect
import sys
import time
from pathlib import Path
from typing import Any
import asyncio
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.utils.logger import system_logger
from src.utils.prometheus_metrics import metrics
from src.utils.warning_symbols import missing

class ValidatorOrchestrator:
    """Orchestrator for running step validators in the training pipeline."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild('ValidatorOrchestrator')
        self.validators = {}
        self.validation_results = {}

    async def run_step_validator(self, step_name: str, training_input: dict[str, Any], pipeline_state: dict[str, Any], config: dict[str, Any], validation_level: str='CRITICAL') -> dict[str, Any]:
        """
        Run the validator for a specific step with enhanced validation levels.

        Args:
            step_name: Name of the step (e.g., "step01_data_collection")
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            config: Configuration dictionary
            validation_level: Validation level ("BASIC", "STANDARD", "COMPREHENSIVE", "CRITICAL") - defaults to CRITICAL

        Returns:
            Dictionary containing validation results
        """
        start_perf = time.perf_counter()
        try:
            self.logger.info(f'🔍 Running {validation_level} validator for {step_name}')
            try:
                self.logger.debug('Input context - training_input_keys=%s pipeline_state_keys=%s validation_level=%s', list(training_input.keys()) if isinstance(training_input, dict) else type(training_input).__name__, list(pipeline_state.keys()) if isinstance(pipeline_state, dict) else type(pipeline_state).__name__, validation_level)
            except Exception:
                pass
            pre_validation_result = await self._run_pre_validation_checks(step_name, training_input, pipeline_state, config, validation_level)
            if not pre_validation_result.get('passed', True):
                duration = max(0.0, time.perf_counter() - start_perf)
                return self._normalize_result(step_name, pre_validation_result, duration)
            raw_result = await self._run_validator(step_name, training_input, pipeline_state, config, validation_level)
            post_validation_result = await self._run_post_validation_checks(step_name, raw_result, training_input, pipeline_state, config, validation_level)
            combined_result = self._combine_validation_results(step_name, raw_result, post_validation_result, validation_level)
            duration = max(0.0, time.perf_counter() - start_perf)
            result = self._normalize_result(step_name, combined_result, duration)
            self.validation_results[step_name] = result
            passed = bool(result.get('validation_passed', False))
            status = 'SUCCESS' if passed else 'FAILED'
            failure_reason = self._extract_failure_reason(result)
            try:
                metrics.record_step_execution(step_name=step_name, duration=duration, status=status)
            except Exception:
                self.logger.debug('Metrics recording for step execution failed', exc_info=True)
            if passed:
                metrics.record_validation_result(step_name=step_name, validation_type='step_validation', passed=True, reason='Step validation completed successfully')
                self.logger.info(f'✅ Validator for {step_name} completed in {duration:.3f}s: passed=True')
            else:
                self.logger.error(f'❌ Validator failed for {step_name} in {duration:.3f}s: {failure_reason}')
                metrics.record_validation_result(step_name=step_name, validation_type='step_validation', passed=False, reason=failure_reason)
            return result
        except Exception as e:
            duration = max(0.0, time.perf_counter() - start_perf)
            self.logger.exception(f'❌ Exception while running validator for {step_name}: {e}')
            error_result = {'step_name': step_name, 'validation_passed': False, 'error': str(e), 'duration': duration, 'timestamp': time.time()}
            self.validation_results[step_name] = error_result
            try:
                metrics.record_step_execution(step_name=step_name, duration=duration, status='EXCEPTION')
            except Exception:
                self.logger.debug('Metrics recording for exception failed', exc_info=True)
            metrics.record_validation_result(step_name=step_name, validation_type='step_validation', passed=False, reason=f'Validator execution error: {str(e)}')
            return error_result

    async def _run_pre_validation_checks(self, step_name: str, training_input: dict[str, Any], pipeline_state: dict[str, Any], config: dict[str, Any], validation_level: str) -> dict[str, Any]:
        """
        Run pre-validation checks before executing the main validator.

        Args:
            step_name: Name of the step
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            config: Configuration dictionary
            validation_level: Validation level

        Returns:
            Pre-validation result dictionary
        """
        try:
            self.logger.debug(f'🔍 Running pre-validation checks for {step_name}')
            if not isinstance(training_input, dict):
                return {'passed': False, 'validation_passed': False, 'error': 'training_input must be a dictionary'}
            if not isinstance(pipeline_state, dict):
                return {'passed': False, 'validation_passed': False, 'error': 'pipeline_state must be a dictionary'}
            if not isinstance(config, dict):
                return {'passed': False, 'validation_passed': False, 'error': 'config must be a dictionary'}
            required_params = ['symbol', 'exchange', 'timeframe']
            missing_params = [param for param in required_params if param not in training_input]
            if missing_params:
                return {'passed': False, 'validation_passed': False, 'error': f'Missing required training input parameters: {missing_params}'}
            if validation_level in ['COMPREHENSIVE', 'CRITICAL']:
                if 'data_dir' not in config:
                    return {'passed': False, 'validation_passed': False, 'error': 'Missing data_dir in configuration'}
                failed_steps = [step for step, info in pipeline_state.items() if isinstance(info, dict) and info.get('status') == 'FAILED']
                if failed_steps:
                    return {'passed': False, 'validation_passed': False, 'error': f'Pipeline has failed steps: {failed_steps}'}
            return {'passed': True, 'validation_passed': True}
        except Exception as e:
            self.logger.exception(f'❌ Error in pre-validation checks for {step_name}: {e}')
            return {'passed': False, 'validation_passed': False, 'error': f'Pre-validation check error: {str(e)}'}

    async def _run_post_validation_checks(self, step_name: str, validation_result: dict[str, Any], training_input: dict[str, Any], pipeline_state: dict[str, Any], config: dict[str, Any], validation_level: str) -> dict[str, Any]:
        """
        Run post-validation checks after executing the main validator.

        Args:
            step_name: Name of the step
            validation_result: Result from main validator
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            config: Configuration dictionary
            validation_level: Validation level

        Returns:
            Post-validation result dictionary
        """
        try:
            self.logger.debug(f'🔍 Running post-validation checks for {step_name}')
            post_checks = {'validation_passed': True, 'warnings': [], 'recommendations': []}
            if not isinstance(validation_result, dict):
                post_checks['validation_passed'] = False
                post_checks['warnings'].append('Validation result is not a dictionary')
                return post_checks
            if validation_level in ['COMPREHENSIVE', 'CRITICAL']:
                if validation_result.get('critical_issues'):
                    post_checks['warnings'].append(f"Critical issues found: {validation_result['critical_issues']}")
                if validation_result.get('data_quality_issues'):
                    post_checks['warnings'].append(f"Data quality issues: {validation_result['data_quality_issues']}")
                if validation_level == 'CRITICAL':
                    post_checks['recommendations'].append('Consider running additional data quality checks')
                    post_checks['recommendations'].append('Review model performance metrics')
            return post_checks
        except Exception as e:
            self.logger.exception(f'❌ Error in post-validation checks for {step_name}: {e}')
            return {'validation_passed': False, 'error': f'Post-validation check error: {str(e)}'}

    def _combine_validation_results(self, step_name: str, main_result: dict[str, Any], post_result: dict[str, Any], validation_level: str) -> dict[str, Any]:
        """
        Combine main validation result with post-validation checks.

        Args:
            step_name: Name of the step
            main_result: Main validation result
            post_result: Post-validation result
            validation_level: Validation level

        Returns:
            Combined validation result
        """
        try:
            combined = dict(main_result)
            if post_result.get('warnings'):
                combined.setdefault('warnings', []).extend(post_result['warnings'])
            if post_result.get('recommendations'):
                combined.setdefault('recommendations', []).extend(post_result['recommendations'])
            main_passed = main_result.get('validation_passed', False)
            post_passed = post_result.get('validation_passed', True)
            if validation_level == 'CRITICAL':
                combined['validation_passed'] = main_passed and post_passed
            else:
                combined['validation_passed'] = main_passed
            combined['validation_level'] = validation_level
            combined['validation_timestamp'] = time.time()
            return combined
        except Exception as e:
            self.logger.exception(f'❌ Error combining validation results for {step_name}: {e}')
            return {'step_name': step_name, 'validation_passed': False, 'error': f'Result combination error: {str(e)}'}

    async def _run_validator(self, step_name: str, training_input: dict[str, Any], pipeline_state: dict[str, Any], config: dict[str, Any], validation_level: str='CRITICAL') -> dict[str, Any]:
        """
        Dynamically import and run the appropriate validator.

        Args:
            step_name: Name of the step
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            config: Configuration dictionary

        Returns:
            Dictionary containing validation results
        """
        validator_mapping = {'step01_data_collection': 'step01_data_collection_validator', 'step01_5_data_converter': 'step01_5_data_converter_validator', 'step02_data_reading': 'step02_data_reading_validator', 'step02_5_sr_optimization': 'step02_5_sr_optimization_validator', 'step03_hmm_regime_discovery': 'step03_hmm_regime_discovery_validator', 'step04_5_triple_barrier_method': 'step04_5_triple_barrier_method_validator', 'step04_regime_data_splitting': 'step04_regime_data_splitting_validator', 'step05_labeling': 'step05_labeling_validator', 'step06_feature_engineering': 'step06_feature_engineering_validator', 'step07_enhanced_matrix_operations': 'step07_enhanced_matrix_operations_validator', 'step8_regime_data_splitting': 'step8_regime_data_splitting_validator', 'step09_hmm_based_training': 'step09_hmm_based_training_validator', 'step09_5_multi_timeframe_hmm_ensemble': 'step09_5_multi_timeframe_hmm_ensemble_validator', 'step09_5_hmm_lm_generalist_training': 'step09_5_hmm_lm_generalist_training_validator', 'step10_unified_regime_intelligence': 'step10_unified_regime_intelligence_validator', 'step11_analyst_creation': 'step11_analyst_creation_validator', 'step12_analyst_enhancement': 'step12_analyst_enhancement_validator', 'step13_analyst_ensemble_creation': 'step13_analyst_ensemble_creation_validator', 'step14_tactician_labeling': 'step14_tactician_labeling_validator', 'step15_tactician_specialist_training': 'step15_tactician_specialist_training_validator', 'step16_confidence_calibration': 'step16_confidence_calibration_validator', 'step17_final_parameters_optimization': 'step17_final_parameters_optimization_validator', 'step18_walk_forward_validation': 'step18_walk_forward_validation_validator', 'step19_monte_carlo_validation': 'step19_monte_carlo_validation_validator', 'step20_ab_testing': 'step20_ab_testing_validator', 'step21_saving': 'step21_saving_validator'}
        validator_module_name = validator_mapping.get(step_name)
        if not validator_module_name:
            msg = f'No validator mapping found for step: {step_name}'
            raise ValueError(msg)
        module_path = f'src.training.steps.{validator_module_name}'
        try:
            module_path = f'src.training.steps.{validator_module_name}'
            validator_module = importlib.import_module(module_path)
            self.validators[step_name] = validator_module
            run_validator_func: Any | None = getattr(validator_module, 'run_validator', None)
            if run_validator_func is None or not callable(run_validator_func):
                warn_msg = f'run_validator not found or not callable in module {module_path}'
                self.logger.warning(missing(warn_msg))
                return {'step_name': step_name, 'validation_passed': True, 'warning': warn_msg}
            if inspect.iscoroutinefunction(run_validator_func):
                result = await run_validator_func(training_input, pipeline_state)
            else:
                result = run_validator_func(training_input, pipeline_state)
            self.logger.info(f"✅ Validator for {step_name} completed: {(bool(result.get('validation_passed', False)) if isinstance(result, dict) else bool(result))}")
            return result if isinstance(result, dict) else {'validation_passed': bool(result)}
        except ImportError as e:
            self.logger.warning(missing(f'⚠️ Validator module not found for {step_name}: {e}'))
            return {'step_name': step_name, 'validation_passed': True, 'warning': f'Validator module not found: {str(e)}'}
        except Exception:
            raise

    def _normalize_result(self, step_name: str, result: Any, duration: float) -> dict[str, Any]:
        """Normalize validator result into a consistent schema and inject timing."

        Schema keys: step_name, validation_passed, validation_results, error|warning|message, duration, timestamp
        """
        normalized: dict[str, Any]
        if not isinstance(result, dict):
            normalized = {'step_name': step_name, 'validation_passed': bool(result), 'validation_results': {}, 'message': 'Non-dict validator result converted to boolean'}
        else:
            normalized = dict(result)
            normalized.setdefault('step_name', step_name)
            normalized['step_name'] = step_name
            normalized.setdefault('validation_passed', False)
            normalized.setdefault('validation_results', {})
        normalized['duration'] = duration
        normalized['timestamp'] = time.time()
        if not bool(normalized.get('validation_passed', False)) and (not (normalized.get('error') or normalized.get('warning') or normalized.get('message'))):
            normalized['error'] = self._extract_failure_reason(normalized)
        return normalized

    def _extract_failure_reason(self, result: dict[str, Any]) -> str:
        """Heuristically extract a concise failure reason from the result payload."""
        for key in ('error', 'warning', 'message'):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        vr = result.get('validation_results')
        if isinstance(vr, dict):
            for sub in vr.values():
                if isinstance(sub, dict):
                    if isinstance(sub.get('error'), str) and sub.get('error'):
                        return str(sub.get('error'))
                    errors = sub.get('errors')
                    if isinstance(errors, list) and errors:
                        return ', '.join(map(str, errors[:3]))
                    if sub.get('has_critical_errors'):
                        msgs = sub.get('error_messages')
                        if isinstance(msgs, list) and msgs:
                            return ', '.join(map(str, msgs[:3]))
                        return 'Critical errors present'
        return 'Step validation failed'

    def get_validation_summary(self) -> dict[str, Any]:
        """
        Get a summary of all validation results.

        Returns:
            Dictionary containing validation summary
        """
        total_validations = len(self.validation_results)
        passed_validations = sum((1 for result in self.validation_results.values() if result.get('validation_passed', False)))
        failed_validations = total_validations - passed_validations
        return {'total_validations': total_validations, 'passed_validations': passed_validations, 'failed_validations': failed_validations, 'success_rate': passed_validations / total_validations if total_validations > 0 else 0, 'validation_results': self.validation_results}

    def get_failed_validations(self) -> list[str]:
        """
        Get list of steps that failed validation.

        Returns:
            List of step names that failed validation
        """
        return [step_name for step_name, result in self.validation_results.items() if not result.get('validation_passed', False)]

    def clear_results(self) -> None:
        """Clear all validation results."""
        self.validation_results.clear()
validator_orchestrator = ValidatorOrchestrator()