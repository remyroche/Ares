"""Step validation system to ensure data quality between pipeline steps."""
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import json
from datetime import datetime
from .logger import system_logger

import logging
import time

class StepValidationSystem:
    """Validates data quality and structure between pipeline steps."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild('StepValidationSystem')
        self.validation_rules = {'step02_data_reading': {'required_outputs': ['dataframe', 'validation_results'], 'data_quality_threshold': 85.0, 'min_rows': 1000, 'required_columns': ['open', 'high', 'low', 'close', 'volume', 'timestamp'], 'max_duplicate_ratio': 0.1, 'require_datetime_index': True}, 'step02_5_sr_optimization': {'required_inputs': ['dataframe'], 'data_quality_threshold': 80.0, 'min_rows': 500, 'required_columns': ['open', 'high', 'low', 'close', 'volume'], 'max_duplicate_ratio': 0.05, 'require_datetime_index': True}, 'step2_5_sr_optimization': {'required_inputs': ['dataframe'], 'data_quality_threshold': 80.0, 'min_rows': 500, 'required_columns': ['open', 'high', 'low', 'close', 'volume'], 'max_duplicate_ratio': 0.05, 'require_datetime_index': True}}
        self.validation_history = []

    def validate_step_output(self, step_name: str, output: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate step output against defined rules.

        Args:
            step_name: Name of the step
            output: Step output data
            pipeline_state: Current pipeline state

        Returns:
            Validation results
        """
        validation_result = {'step_name': step_name, 'timestamp': datetime.now().isoformat(), 'passed': True, 'issues': [], 'warnings': [], 'data_quality_score': 100.0, 'recommendations': []}
        rules = self.validation_rules.get(step_name, {})
        if not rules:
            self.logger.warning(f'⚠️ No validation rules defined for step {step_name}')
            return validation_result
        required_outputs = rules.get('required_outputs', [])
        for required_output in required_outputs:
            if required_output not in output:
                validation_result['passed'] = False
                validation_result['issues'].append(f'Missing required output: {required_output}')
                validation_result['recommendations'].append(f'Ensure {required_output} is included in step output')
        dataframe = output.get('dataframe') or output.get('validated_data')
        if isinstance(dataframe, pd.DataFrame):
            df_validation = self._validate_dataframe(dataframe, rules, step_name)
            validation_result['issues'].extend(df_validation['issues'])
            validation_result['warnings'].extend(df_validation['warnings'])
            validation_result['data_quality_score'] = df_validation['quality_score']
            validation_result['recommendations'].extend(df_validation['recommendations'])
            if df_validation['quality_score'] < rules.get('data_quality_threshold', 80.0):
                validation_result['passed'] = False
        self.validation_history.append(validation_result)
        if validation_result['passed']:
            self.logger.info(f"✅ Step {step_name} validation passed (quality: {validation_result['data_quality_score']:.1f})")
        else:
            self.logger.error(f"❌ Step {step_name} validation failed: {validation_result['issues']}")
        return validation_result

    def _validate_dataframe(self, df: pd.DataFrame, rules: Dict[str, Any], step_name: str) -> Dict[str, Any]:
        """Validate DataFrame against step rules."""
        validation = {'issues': [], 'warnings': [], 'recommendations': [], 'quality_score': 100.0}
        min_rows = rules.get('min_rows', 0)
        if len(df) < min_rows:
            validation['issues'].append(f'Insufficient rows: {len(df)} < {min_rows}')
            validation['recommendations'].append('Check data source and ensure sufficient data')
            validation['quality_score'] -= 20
        required_columns = rules.get('required_columns', [])
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation['issues'].append(f'Missing required columns: {missing_columns}')
            validation['recommendations'].append('Add missing columns or handle gracefully')
            validation['quality_score'] -= len(missing_columns) * 10
        max_duplicate_ratio = rules.get('max_duplicate_ratio', 0.1)
        if 'timestamp' in df.columns:
            duplicate_count = df['timestamp'].duplicated().sum()
            duplicate_ratio = duplicate_count / len(df) if len(df) > 0 else 0
            if duplicate_ratio > max_duplicate_ratio:
                validation['issues'].append(f'Too many duplicates: {duplicate_ratio:.2%} > {max_duplicate_ratio:.2%}')
                validation['recommendations'].append('Remove duplicate timestamps')
                validation['quality_score'] -= min(30, duplicate_ratio * 100)
        require_datetime_index = rules.get('require_datetime_index', False)
        if require_datetime_index:
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    validation['warnings'].append('Index is not DatetimeIndex')
                    validation['recommendations'].append('Set timestamp column as index')
                    validation['quality_score'] -= 5
                else:
                    validation['issues'].append('No timestamp column for datetime index')
                    validation['recommendations'].append('Add timestamp column')
                    validation['quality_score'] -= 15
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns and (not pd.api.types.is_numeric_dtype(df[col])):
                validation['warnings'].append(f'Non-numeric column: {col}')
                validation['recommendations'].append(f'Convert {col} to numeric')
                validation['quality_score'] -= 5
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns)) if len(df) > 0 else 0
        if missing_ratio > 0.05:
            validation['warnings'].append(f'High missing value ratio: {missing_ratio:.2%}')
            validation['recommendations'].append('Handle missing values')
            validation['quality_score'] -= min(10, missing_ratio * 100)
        validation['quality_score'] = max(0, validation['quality_score'])
        return validation

    def validate_step_input(self, step_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate step input to ensure it meets requirements.

        Args:
            step_name: Name of the step
            input_data: Input data for the step

        Returns:
            Validation results
        """
        validation_result = {'step_name': step_name, 'timestamp': datetime.now().isoformat(), 'passed': True, 'issues': [], 'warnings': [], 'recommendations': []}
        rules = self.validation_rules.get(step_name, {})
        if not rules:
            return validation_result
        required_inputs = rules.get('required_inputs', [])
        for required_input in required_inputs:
            if required_input not in input_data:
                validation_result['passed'] = False
                validation_result['issues'].append(f'Missing required input: {required_input}')
                validation_result['recommendations'].append(f'Ensure {required_input} is available from previous step')
        dataframe = input_data.get('dataframe') or input_data.get('validated_data')
        if isinstance(dataframe, pd.DataFrame):
            df_validation = self._validate_dataframe(dataframe, rules, step_name)
            validation_result['issues'].extend(df_validation['issues'])
            validation_result['warnings'].extend(df_validation['warnings'])
            validation_result['recommendations'].extend(df_validation['recommendations'])
            if df_validation['quality_score'] < rules.get('data_quality_threshold', 80.0):
                validation_result['passed'] = False
        return validation_result

    def apply_data_fixes(self, step_name: str, data: pd.DataFrame, validation_result: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply data fixes based on validation results.

        Args:
            step_name: Name of the step
            data: DataFrame to fix
            validation_result: Validation results

        Returns:
            Tuple of (fixed_data, fix_report)
        """
        fix_report = {'step_name': step_name, 'applied_fixes': [], 'fixes_failed': [], 'original_issues': len(validation_result['issues'])}
        fixed_data = data.copy()
        for recommendation in validation_result['recommendations']:
            try:
                if recommendation == 'Remove duplicate timestamps':
                    if 'timestamp' in fixed_data.columns:
                        original_count = len(fixed_data)
                        fixed_data = fixed_data.drop_duplicates(subset=['timestamp'], keep='last')
                        removed = original_count - len(fixed_data)
                        if removed > 0:
                            fix_report['applied_fixes'].append(f'removed_{removed}_duplicates')
                            self.logger.info(f'🗑️ Removed {removed} duplicate timestamps')
                elif recommendation == 'Set timestamp column as index':
                    if 'timestamp' in fixed_data.columns:
                        fixed_data = fixed_data.set_index('timestamp')
                        fix_report['applied_fixes'].append('set_datetime_index')
                        self.logger.info('📅 Set datetime index')
                elif recommendation.startswith('Convert') and 'to numeric' in recommendation:
                    column = recommendation.split()[1]
                    if column in fixed_data.columns:
                        fixed_data[column] = pd.to_numeric(fixed_data[column], errors='coerce')
                        fix_report['applied_fixes'].append(f'converted_{column}_to_numeric')
                        self.logger.info(f'🔢 Converted {column} to numeric')
                elif recommendation == 'Handle missing values':
                    fixed_data = fixed_data.fillna(method='ffill').fillna(method='bfill')
                    fix_report['applied_fixes'].append('filled_missing_values')
                    self.logger.info('🔧 Filled missing values')
            except Exception as e:
                self.logger.error(f"❌ Failed to apply fix '{recommendation}': {e}")
                fix_report['fixes_failed'].append(recommendation)
        return (fixed_data, fix_report)

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations performed."""
        if not self.validation_history:
            return {'message': 'No validations performed yet'}
        total_validations = len(self.validation_history)
        passed_validations = sum((1 for v in self.validation_history if v['passed']))
        failed_validations = total_validations - passed_validations
        avg_quality_score = np.mean([v['data_quality_score'] for v in self.validation_history])
        return {'total_validations': total_validations, 'passed_validations': passed_validations, 'failed_validations': failed_validations, 'success_rate': passed_validations / total_validations * 100, 'average_quality_score': avg_quality_score, 'recent_validations': self.validation_history[-5:]}

    def save_validation_report(self, output_path: Path) -> None:
        """Save validation history to file."""
        report_data = {'summary': self.get_validation_summary(), 'validation_history': self.validation_history, 'validation_rules': self.validation_rules, 'generated_at': datetime.now().isoformat()}
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent = 2, default = str)
        self.logger.info(f'💾 Saved validation report to {output_path}')
step_validator = StepValidationSystem()
