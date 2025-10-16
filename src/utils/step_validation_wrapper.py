"""Step validation wrapper using pipeline standards for error prevention."""
import pandas as pd

from typing import Any, Dict, List, Optional, Tuple, Union

import traceback

from .logger import system_logger
from src.utils.pipeline_standards import PipelineStandards, ValidationResult

import logging
import numpy as np

class StepValidationWrapper:
    """Wraps pipeline steps with validation using pipeline standards."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild('StepValidationWrapper')
        self.standards = PipelineStandards(self.logger)
        self.validation_history = []

    def wrap_step_with_validation(self, step_class: Any, step_name: str, schema_name: str='unified') -> None:
        """
        Wrap a step class with validation using pipeline standards.

        Args:
            step_class: The step class to wrap
            step_name: Name of the step
            schema_name: Schema name for validation

        Returns:
            Wrapped step class with validation
        """

        class ValidatedStepWrapper(step_class):

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.validation_wrapper = StepValidationWrapper()
                self.validated_step_name = step_name
                self.schema_name = schema_name

            async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
                """Execute step with validation using pipeline standards."""
                self.logger.info(f'🛡️ Executing {self.validated_step_name} with pipeline standards validation')
                input_validation = self._validate_step_input(pipeline_state)
                if not input_validation['passed']:
                    self.logger.error(f'❌ Input validation failed for {self.validated_step_name}')
                    return {'success': False, 'error': f"Input validation failed: {input_validation['issues']}", 'step_name': self.validated_step_name}
                try:
                    result = await super().execute(training_input, pipeline_state)
                    output_validation = self._validate_step_output(result, pipeline_state)
                    if not output_validation['passed']:
                        self.logger.warning(f'⚠️ Output validation issues in {self.validated_step_name}')
                        if 'dataframe' in result or 'validated_data' in result:
                            data = result.get('dataframe') or result.get('validated_data')
                            if isinstance(data, pd.DataFrame):
                                fixed_data = self._apply_pipeline_standards_fixes(data)
                                result['dataframe'] = fixed_data
                                result['validation_fixes_applied'] = True
                                self.logger.info('🔧 Applied pipeline standards fixes to output')
                    return result
                except Exception as e:
                    self.logger.exception(f'❌ Step {self.validated_step_name} failed: {e}')
                    return {'success': False, 'error': str(e), 'step_name': self.validated_step_name, 'traceback': traceback.format_exc()}

            def _validate_step_input(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
                """Validate step input using pipeline standards."""
                validation_result = {'passed': True, 'issues': [], 'warnings': []}
                data = pipeline_state.get('dataframe') or pipeline_state.get('validated_data')
                if data is None:
                    validation_result['passed'] = False
                    validation_result['issues'].append('No DataFrame found in pipeline state')
                    return validation_result
                if not isinstance(data, pd.DataFrame):
                    validation_result['passed'] = False
                    validation_result['issues'].append(f'Data is not DataFrame: {type(data)}')
                    return validation_result
                standards_validation = self.validation_wrapper.standards.validate_data_quality(data, self.schema_name)
                if not standards_validation.passed:
                    validation_result['passed'] = False
                    for issue in standards_validation.issues:
                        validation_result['issues'].append(issue.message)
                for warning in standards_validation.warnings:
                    validation_result['warnings'].append(warning.message)
                return validation_result

            def _validate_step_output(self, output: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
                """Validate step output using pipeline standards."""
                validation_result = {'passed': True, 'issues': [], 'warnings': [], 'quality_score': 100.0}
                data = output.get('dataframe') or output.get('validated_data')
                if isinstance(data, pd.DataFrame):
                    standards_validation = self.validation_wrapper.standards.validate_data_quality(data, self.schema_name)
                    validation_result['quality_score'] = standards_validation.quality_score
                    if not standards_validation.passed:
                        validation_result['passed'] = False
                        for issue in standards_validation.issues:
                            validation_result['issues'].append(issue.message)
                    for warning in standards_validation.warnings:
                        validation_result['warnings'].append(warning.message)
                return validation_result

            def _apply_pipeline_standards_fixes(self, data: pd.DataFrame) -> pd.DataFrame:
                """Apply fixes using pipeline standards."""
                self.logger.info('🔧 Applying pipeline standards fixes...')
                fixed_data = data.copy()
                if 'timestamp' in fixed_data.columns:
                    duplicate_count = fixed_data['timestamp'].duplicated().sum()
                    if duplicate_count > 0:
                        self.logger.info(f'🗑️ Removing {duplicate_count} duplicate timestamps')
                        fixed_data = fixed_data.drop_duplicates(subset=['timestamp'], keep='last')
                if 'timestamp' in fixed_data.columns:
                    if not fixed_data['timestamp'].is_monotonic_increasing:
                        self.logger.info('📈 Sorting data by timestamp')
                        fixed_data = fixed_data.sort_values('timestamp').reset_index(drop = True)
                try:
                    fixed_data = self.validation_wrapper.standards.enforce_schema(fixed_data, self.schema_name)
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
                return fixed_data
        return ValidatedStepWrapper

    def create_validated_step(self, step_class: Any, step_name: str, schema_name: str='unified') -> Any:
        """Create a validated step instance."""
        ValidatedStep = self.wrap_step_with_validation(step_class, step_name, schema_name)
        return ValidatedStep

    def validate_cross_step_consistency(self, step_outputs: Dict[str, Dict[str, Any]]) -> ValidationResult:
        """Validate consistency across multiple steps using pipeline standards."""
        return self.standards.validate_cross_step_consistency(step_outputs, list(step_outputs.keys()))

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {'total_validations': len(self.validation_history), 'pipeline_standards_used': True, 'validation_method': 'pipeline_standards'}
step_validator_wrapper = StepValidationWrapper()
