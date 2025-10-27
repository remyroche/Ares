"""Step validation initializer to apply pipeline standards validation to all steps."""
import pandas as pd

from typing import Any, Dict, List, Optional, Tuple, Union, Type

import traceback

from .logger import system_logger
from src.utils.pipeline_standards import PipelineStandards

import collections
import logging
import numpy as np

class StepValidationInitializer:
    """Initializes all pipeline steps with comprehensive validation."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild('StepValidationInitializer')
        self.standards = PipelineStandards(self.logger)
        self.initialized_steps = {}
        self.validation_summary = {'steps_processed': 0, 'steps_successful': 0, 'steps_failed': 0, 'validation_methods_added': 0, 'errors_encountered': []}

    def initialize_all_steps(self) -> Dict[str, Any]:
        """Initialize all pipeline steps with validation."""
        self.logger.info('🚀 Initializing all pipeline steps with validation...')
        steps_config = {
            'step01_data_collection': {'module': 'src.training.steps.data_collection.sub_pipeline', 'class': 'DataCollectionSubPipeline', 'priority': 1},
            'step02_data_reading': {'module': 'src.training.steps.data_collection.step02_data_reading', 'class': 'DataReadingStep', 'priority': 2},
            'step2_5_sr_optimization': {'module': 'src.training.steps.market_analysis.sub_pipeline', 'class': 'MarketAnalysisSubPipeline', 'priority': 3},
            'step05_labeling': {'module': 'src.training.steps.step5_labeling', 'class': 'LabelingStep', 'priority': 6},
            'step06_advanced_feature_engineering': {'module': 'src.training.steps.data_collection.feature_generation.utils.step06_feature_engineering', 'class': 'FeatureEngineeringStep', 'priority': 7},
            'step08_advanced_feature_selection': {'module': 'src.training.steps.data_collection.feature_generation.utils.step08_advanced_feature_selection', 'class': 'FeatureSelectionStep', 'priority': 8},
            # Simplified model training steps
            'analyst_model_training': {'module': 'src.training.steps.model_training.simplified.analyst_model_training', 'class': 'AnalystModelTrainer', 'priority': 10},
            'tactician_model_training': {'module': 'src.training.steps.model_training.simplified.tactician_model_training', 'class': 'TacticianModelTrainer', 'priority': 11},
            # Legacy step names for backward compatibility
            'step11_analyst_creation': {'module': 'src.training.steps.model_training.simplified.analyst_model_training', 'class': 'AnalystModelTrainer', 'priority': 11},
            'step12_analyst_enhancement': {'module': 'src.training.steps.model_training.simplified.analyst_model_training', 'class': 'AnalystModelTrainer', 'priority': 12},
            'step13_analyst_ensemble_creation': {'module': 'src.training.steps.model_training.simplified.analyst_model_training', 'class': 'AnalystModelTrainer', 'priority': 13},
            'step14_tactician_labeling': {'module': 'src.training.steps.model_training.simplified.tactician_model_training', 'class': 'TacticianModelTrainer', 'priority': 14},
            'step15_tactician_specialist_training': {'module': 'src.training.steps.model_training.simplified.tactician_model_training', 'class': 'TacticianModelTrainer', 'priority': 15},
            # Consolidated backtesting step
            'consolidated_backtesting': {'module': 'src.training.steps.backtesting.consolidated_backtesting_step', 'class': 'ConsolidatedBacktestingStep', 'priority': 18},
            'step18_walk_forward_validation': {'module': 'src.training.steps.backtesting.consolidated_backtesting_step', 'class': 'ConsolidatedBacktestingStep', 'priority': 18},
            'step19_monte_carlo_validation': {'module': 'src.training.steps.backtesting.consolidated_backtesting_step', 'class': 'ConsolidatedBacktestingStep', 'priority': 19},
            'step20_ab_testing': {'module': 'src.training.steps.backtesting.consolidated_backtesting_step', 'class': 'ConsolidatedBacktestingStep', 'priority': 20},
            'step21_model_persistence': {'module': 'src.training.steps.backtesting.consolidated_backtesting_step', 'class': 'ConsolidatedBacktestingStep', 'priority': 21}
        }
        # Remove deprecated Step07 from configuration
        steps_config.pop('step07_enhanced_matrix_operations', None)
        sorted_steps = sorted(steps_config.items(), key = lambda x: x[1]['priority'])
        for step_name, config in sorted_steps:
            try:
                self.logger.info(f'🔧 Initializing {step_name}...')
                step_class = self._import_step_class(config['module'], config['class'])
                if step_class is None:
                    self.logger.warning(f'⚠️ Could not import {step_name}')
                    self.validation_summary['steps_failed'] += 1
                    self.validation_summary['errors_encountered'].append(f'Import failed: {step_name}')
                    continue
                validated_step_class = self._add_validation_to_step(step_class, step_name)
                self.initialized_steps[step_name] = validated_step_class
                self.validation_summary['steps_successful'] += 1
                self.validation_summary['validation_methods_added'] += 2
                self.logger.info(f'✅ Initialized {step_name}')
            except Exception as e:
                self.logger.error(f'❌ Failed to initialize {step_name}: {e}')
                self.validation_summary['steps_failed'] += 1
                self.validation_summary['errors_encountered'].append(f'Initialization failed: {step_name} - {str(e)}')
            self.validation_summary['steps_processed'] += 1
        self.logger.info(f"🎯 Initialization complete: {self.validation_summary['steps_successful']}/{self.validation_summary['steps_processed']} steps successful")
        return self.validation_summary

    def _import_step_class(self, module_path: str, class_name: str) -> Optional[Type]:
        """Import step class dynamically."""
        try:
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            self.logger.warning(f'⚠️ Could not import {class_name} from {module_path}: {e}')
            return None

    def _add_validation_to_step(self, step_class: Type, step_name: str) -> Type:
        """Add validation methods to a step class."""
        original_init = step_class.__init__

        def new_init(self, *args, **kwargs) -> None:
            original_init(self, *args, **kwargs)
            if not hasattr(self, 'standards'):
                self.standards = PipelineStandards(self.logger)
            if not hasattr(self, 'validated_step_name'):
                self.validated_step_name = step_name
        step_class.__init__ = new_init
        step_class._validate_and_fix_input_data = self._create_validation_method(step_name)
        if hasattr(step_class, 'execute'):
            original_execute = step_class.execute

            async def validated_execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
                """Execute step with comprehensive validation."""
                self.logger.info(f'🛡️ Executing {self.validated_step_name} with validation...')
                pre_validation = self._validate_pre_execution(training_input, pipeline_state)
                if not pre_validation['passed']:
                    self.logger.error(f'❌ Pre-execution validation failed for {self.validated_step_name}')
                    return {'success': False, 'error': f"Pre-execution validation failed: {pre_validation['issues']}", 'step_name': self.validated_step_name}
                try:
                    result = await original_execute(self, training_input, pipeline_state)
                    result = self._validate_and_fix_output(result, pipeline_state)
                    return result
                except Exception as e:
                    self.logger.exception(f'❌ Step {self.validated_step_name} failed: {e}')
                    return {'success': False, 'error': str(e), 'step_name': self.validated_step_name, 'traceback': traceback.format_exc()}
            step_class.execute = validated_execute
        step_class._validate_pre_execution = self._create_pre_execution_validation_method(step_name)
        step_class._validate_and_fix_output = self._create_post_execution_validation_method(step_name)
        return step_class

    def _create_validation_method(self, step_name: str) -> None:
        """Create a validation method for a step."""

        def _validate_and_fix_input_data(self, data: pd.DataFrame) -> pd.DataFrame:
            """
            Validate and fix input data using pipeline standards.

            Args:
                data: Input DataFrame

            Returns:
                Validated and fixed DataFrame
            """
            self.logger.info(f'🔍 Validating input data for {self.validated_step_name}...')
            validation_result = self.standards.validate_data_quality(data, 'unified')
            if not validation_result.passed:
                self.logger.warning(f'⚠️ Data quality issues detected: {validation_result.quality_score:.2f}')
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
                    fixed_data = fixed_data.sort_values('timestamp').reset_index(drop = True)
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
            critical_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in critical_columns:
                if col in fixed_data.columns:
                    if fixed_data[col].isnull().any():
                        self.logger.info(f'🔧 Handling missing values in {col}')
                        fixed_data[col] = fixed_data[col].fillna(method='ffill').fillna(method='bfill')
            final_validation = self.standards.validate_data_quality(fixed_data, 'unified')
            self.logger.info(f'✅ Final data quality score: {final_validation.quality_score:.2f}')
            return fixed_data
        return _validate_and_fix_input_data

    def _create_pre_execution_validation_method(self, step_name: str) -> None:
        """Create pre-execution validation method."""

        def _validate_pre_execution(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
            """Validate inputs before step execution."""
            validation_result = {'passed': True, 'issues': [], 'warnings': [], 'quality_score': 100.0}
            data = pipeline_state.get('dataframe') or pipeline_state.get('validated_data')
            if data is None:
                validation_result['passed'] = False
                validation_result['issues'].append('No DataFrame found in pipeline state')
                return validation_result
            if not isinstance(data, pd.DataFrame):
                validation_result['passed'] = False
                validation_result['issues'].append(f'Data is not DataFrame: {type(data)}')
                return validation_result
            standards_validation = self.standards.validate_data_quality(data, 'unified')
            validation_result['quality_score'] = standards_validation.quality_score
            if not standards_validation.passed:
                validation_result['passed'] = False
                for issue in standards_validation.issues:
                    validation_result['issues'].append(issue.message)
            for warning in standards_validation.warnings:
                validation_result['warnings'].append(warning.message)
            return validation_result
        return _validate_pre_execution

    def _create_post_execution_validation_method(self, step_name: str) -> None:
        """Create post-execution validation method."""

        def _validate_and_fix_output(self, result: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
            """Validate and fix step output."""
            if not result.get('success', True):
                return result
            data = result.get('dataframe') or result.get('validated_data')
            if isinstance(data, pd.DataFrame):
                output_validation = self.standards.validate_data_quality(data, 'unified')
                if not output_validation.passed or output_validation.quality_score < 80.0:
                    self.logger.warning(f'⚠️ Output validation issues in {self.validated_step_name}: {output_validation.quality_score:.2f}')
                    fixed_data = self._validate_and_fix_input_data(data)
                    result['dataframe'] = fixed_data
                    result['validation_fixes_applied'] = True
                    result['original_quality_score'] = output_validation.quality_score
                    final_validation = self.standards.validate_data_quality(fixed_data, 'unified')
                    result['final_quality_score'] = final_validation.quality_score
                    self.logger.info(f'✅ Applied fixes, quality improved: {output_validation.quality_score:.2f} → {final_validation.quality_score:.2f}')
            return result
        return _validate_and_fix_output

    def get_initialized_steps(self) -> Dict[str, Type]:
        """Get all initialized steps."""
        return self.initialized_steps

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary."""
        return {**self.validation_summary, 'initialized_steps': list(self.initialized_steps.keys()), 'validation_features': ['Pre-execution validation', 'Post-execution validation', 'Data quality fixes', 'Schema enforcement', 'Duplicate removal', 'Index sorting', 'Datetime conversion', 'Missing value handling'], 'pipeline_standards_integration': True}
step_validation_initializer = StepValidationInitializer()
