"""Error prevention system to prevent error propagation between steps."""
import pandas as pd

from typing import Any, Dict, List, Optional, Tuple, Union

import traceback
from datetime import datetime
from .logger import system_logger
from typing import Dict, List, Optional, Union, Any, Tuple

from .logger import system_logger
import logging
import numpy as np
import time

class ErrorPreventionSystem:
    """Prevents error propagation between pipeline steps."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild('ErrorPreventionSystem')
        self.error_patterns = {'data_quality': ['duplicate timestamps', 'non-monotonic datetime index', 'index is not DatetimeIndex', 'DataFrame is ambiguous'], 'data_structure': ['dict object has no attribute columns', 'Could not find DataFrame', 'No data available', 'object has no attribute'], 'module_imports': ['No module named', 'cannot import name', 'name.*is not defined'], 'validation': ['validation_results', 'metrics.*is not defined', 'CONFIG.*is not defined']}
        self.prevention_actions = {}

    def analyze_step_output(self, step_name: str, output: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze step output for potential error patterns.
        
        Args:
            step_name: Name of the step
            output: Step output data
            pipeline_state: Current pipeline state
            
        Returns:
            Analysis results with recommendations
        """
        analysis = {'step_name': step_name, 'timestamp': datetime.now().isoformat(), 'issues_found': [], 'recommendations': [], 'data_quality_score': 100.0, 'risk_level': 'LOW'}
        if 'dataframe' in output or 'validated_data' in output:
            data = output.get('dataframe') or output.get('validated_data')
            if isinstance(data, pd.DataFrame):
                data_analysis = self._analyze_dataframe_quality(data)
                analysis['issues_found'].extend(data_analysis['issues'])
                analysis['recommendations'].extend(data_analysis['recommendations'])
                analysis['data_quality_score'] = data_analysis['quality_score']
        output_str = str(output).lower()
        for category, patterns in self.error_patterns.items():
            for pattern in patterns:
                if pattern.lower() in output_str:
                    analysis['issues_found'].append(f'{category}: {pattern}')
                    analysis['recommendations'].append(self._get_prevention_recommendation(category, pattern))
        if len(analysis['issues_found']) > 3 or analysis['data_quality_score'] < 70:
            analysis['risk_level'] = 'HIGH'
        elif len(analysis['issues_found']) > 1 or analysis['data_quality_score'] < 85:
            analysis['risk_level'] = 'MEDIUM'
        return analysis

    def _analyze_dataframe_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze DataFrame quality for potential issues."""
        analysis = {'issues': [], 'recommendations': [], 'quality_score': 100.0}
        if 'timestamp' in data.columns:
            duplicate_count = data['timestamp'].duplicated().sum()
            if duplicate_count > 0:
                analysis['issues'].append(f'duplicate_timestamps: {duplicate_count}')
                analysis['recommendations'].append('remove_duplicate_timestamps')
                analysis['quality_score'] -= min(20, duplicate_count / len(data) * 100)
        if 'timestamp' in data.columns:
            if not data['timestamp'].is_monotonic_increasing:
                analysis['issues'].append('non_monotonic_index')
                analysis['recommendations'].append('sort_by_timestamp')
                analysis['quality_score'] -= 10
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                analysis['issues'].append('index_not_datetime')
                analysis['recommendations'].append('set_datetime_index')
                analysis['quality_score'] -= 5
        critical_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in critical_columns if col not in data.columns]
        if missing_columns:
            analysis['issues'].append(f'missing_columns: {missing_columns}')
            analysis['recommendations'].append('validate_required_columns')
            analysis['quality_score'] -= len(missing_columns) * 10
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in data.columns and (not pd.api.types.is_numeric_dtype(data[col])):
                analysis['issues'].append(f'non_numeric_column: {col}')
                analysis['recommendations'].append(f'convert_to_numeric: {col}')
                analysis['quality_score'] -= 5
        analysis['quality_score'] = max(0, analysis['quality_score'])
        return analysis

    def _get_prevention_recommendation(self, category: str, pattern: str) -> str:
        """Get prevention recommendation for error pattern."""
        recommendations = {'data_quality': {'duplicate timestamps': 'apply_data_quality_fixer', 'non-monotonic datetime index': 'sort_and_fix_index', 'index is not DatetimeIndex': 'convert_to_datetime_index', 'DataFrame is ambiguous': 'fix_dataframe_boolean_operations'}, 'data_structure': {'dict object has no attribute columns': 'validate_dataframe_structure', 'Could not find DataFrame': 'ensure_dataframe_in_pipeline_state', 'No data available': 'check_data_availability', 'object has no attribute': 'validate_object_attributes'}, 'module_imports': {'No module named': 'use_graceful_module_handler', 'cannot import name': 'create_fallback_implementation', 'name.*is not defined': 'fix_variable_references'}, 'validation': {'validation_results': 'ensure_validation_results_attribute', 'metrics.*is not defined': 'fix_metrics_reference', 'CONFIG.*is not defined': 'import_config_module'}}
        return recommendations.get(category, {}).get(pattern, 'investigate_and_fix')

    def apply_preventive_fixes(self, step_name: str, data: Any, recommendations: List[str]) -> Tuple[Any, Dict[str, Any]]:
        """
        Apply preventive fixes based on recommendations.
        
        Args:
            step_name: Name of the step
            data: Data to fix
            recommendations: List of recommendations
            
        Returns:
            Tuple of (fixed_data, fix_report)
        """
        fix_report = {'step_name': step_name, 'applied_fixes': [], 'fixes_failed': [], 'original_issues': len(recommendations)}
        if not isinstance(data, pd.DataFrame):
            self.logger.warning(f'⚠️ Data is not DataFrame for step {step_name}: {type(data)}')
            return (data, fix_report)
        fixed_data = data.copy()
        for recommendation in recommendations:
            try:
                if recommendation == 'remove_duplicate_timestamps':
                    original_count = len(fixed_data)
                    fixed_data = fixed_data.drop_duplicates(subset=['timestamp'], keep='last')
                    removed = original_count - len(fixed_data)
                    if removed > 0:
                        fix_report['applied_fixes'].append(f'removed_{removed}_duplicates')
                        self.logger.info(f'🗑️ Removed {removed} duplicate timestamps')
                elif recommendation == 'sort_by_timestamp':
                    if 'timestamp' in fixed_data.columns:
                        fixed_data = fixed_data.sort_values('timestamp').reset_index(drop = True)
                        fix_report['applied_fixes'].append('sorted_by_timestamp')
                        self.logger.info('📈 Sorted data by timestamp')
                elif recommendation == 'set_datetime_index':
                    if 'timestamp' in fixed_data.columns:
                        fixed_data = fixed_data.set_index('timestamp')
                        fix_report['applied_fixes'].append('set_datetime_index')
                        self.logger.info('📅 Set datetime index')
                elif recommendation.startswith('convert_to_numeric:'):
                    column = recommendation.split(':')[1].strip()
                    if column in fixed_data.columns:
                        fixed_data[column] = pd.to_numeric(fixed_data[column], errors='coerce')
                        fix_report['applied_fixes'].append(f'converted_{column}_to_numeric')
                        self.logger.info(f'🔢 Converted {column} to numeric')
                elif recommendation == 'validate_dataframe_structure':
                    if not isinstance(fixed_data, pd.DataFrame):
                        self.logger.error(f'❌ Data is not DataFrame: {type(fixed_data)}')
                        fix_report['fixes_failed'].append('dataframe_structure_validation')
                    else:
                        fix_report['applied_fixes'].append('validated_dataframe_structure')
            except Exception as e:
                self.logger.error(f"❌ Failed to apply fix '{recommendation}': {e}")
                fix_report['fixes_failed'].append(recommendation)
        return (fixed_data, fix_report)

    def validate_step_input(self, step_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate step input to prevent errors.
        
        Args:
            step_name: Name of the step
            input_data: Input data for the step
            
        Returns:
            Validation results
        """
        validation = {'step_name': step_name, 'valid': True, 'issues': [], 'warnings': [], 'recommendations': []}
        if 'dataframe' not in input_data and 'validated_data' not in input_data:
            validation['valid'] = False
            validation['issues'].append('No DataFrame found in input')
            validation['recommendations'].append('Ensure previous step provides DataFrame')
        data = input_data.get('dataframe') or input_data.get('validated_data')
        if isinstance(data, pd.DataFrame):
            if len(data) == 0:
                validation['valid'] = False
                validation['issues'].append('Empty DataFrame')
                validation['recommendations'].append('Check data source')
            critical_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in critical_columns if col not in data.columns]
            if missing_columns:
                validation['warnings'].append(f'Missing columns: {missing_columns}')
                validation['recommendations'].append('Add missing columns or handle gracefully')
        return validation

    def create_step_safety_wrapper(self, step_class: Any, step_name: str) -> Any:
        """
        Create a safety wrapper for a step class.
        
        Args:
            step_class: The step class to wrap
            step_name: Name of the step
            
        Returns:
            Wrapped step class with error prevention
        """

        class SafeStepWrapper(step_class):

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.error_prevention = ErrorPreventionSystem()
                self.safe_step_name = step_name

            async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
                """Execute step with error prevention."""
                self.logger.info(f'🛡️ Executing {self.safe_step_name} with error prevention')
                input_validation = self.error_prevention.validate_step_input(self.safe_step_name, pipeline_state)
                if not input_validation['valid']:
                    self.logger.error(f'❌ Input validation failed for {self.safe_step_name}')
                    return {'success': False, 'error': f"Input validation failed: {input_validation['issues']}", 'step_name': self.safe_step_name}
                try:
                    result = await super().execute(training_input, pipeline_state)
                    output_analysis = self.error_prevention.analyze_step_output(self.safe_step_name, result, pipeline_state)
                    if output_analysis['risk_level'] == 'HIGH':
                        self.logger.warning(f'⚠️ High risk issues detected in {self.safe_step_name}')
                        if 'dataframe' in result or 'validated_data' in result:
                            data = result.get('dataframe') or result.get('validated_data')
                            if isinstance(data, pd.DataFrame):
                                fixed_data, fix_report = self.error_prevention.apply_preventive_fixes(self.safe_step_name, data, output_analysis['recommendations'])
                                result['dataframe'] = fixed_data
                                result['fix_report'] = fix_report
                                self.logger.info(f"🔧 Applied {len(fix_report['applied_fixes'])} preventive fixes")
                    return result
                except Exception as e:
                    self.logger.exception(f'❌ Step {self.safe_step_name} failed: {e}')
                    return {'success': False, 'error': str(e), 'step_name': self.safe_step_name, 'traceback': traceback.format_exc()}
        return SafeStepWrapper
error_prevention = ErrorPreventionSystem()