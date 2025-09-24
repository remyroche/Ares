"""Step validation updater to apply pipeline standards validation to all steps."""
import pandas as pd

from typing import Any, Dict, List, Optional, Tuple, Union, Type

from .logger import system_logger
from src.utils.pipeline_standards import PipelineStandards

import logging
import numpy as np

class StepValidationUpdater:
    """Updates existing steps with pipeline standards validation."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild('StepValidationUpdater')
        self.standards = PipelineStandards(self.logger)
        self.updated_steps = []

    def add_validation_to_step_class(self, step_class: Type, step_name: str) -> Type:
        """
        Add validation methods to an existing step class.
        
        Args:
            step_class: The step class to update
            step_name: Name of the step
            
        Returns:
            Updated step class with validation
        """
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
                """Execute step with validation."""
                self.logger.info(f'🛡️ Executing {self.validated_step_name} with validation...')
                if pipeline_state and ('dataframe' in pipeline_state or 'validated_data' in pipeline_state):
                    data = pipeline_state.get('dataframe') or pipeline_state.get('validated_data')
                    if isinstance(data, pd.DataFrame):
                        data = self._validate_and_fix_input_data(data)
                        pipeline_state['dataframe'] = data
                try:
                    result = await original_execute(self, training_input, pipeline_state)
                    if isinstance(result, dict) and ('dataframe' in result or 'validated_data' in result):
                        output_data = result.get('dataframe') or result.get('validated_data')
                        if isinstance(output_data, pd.DataFrame):
                            validated_output = self._validate_and_fix_input_data(output_data)
                            result['dataframe'] = validated_output
                            result['validation_applied'] = True
                    return result
                except Exception as e:
                    self.logger.exception(f'❌ Step {self.validated_step_name} failed: {e}')
                    return {'success': False, 'error': str(e), 'step_name': self.validated_step_name}
            step_class.execute = validated_execute
        self.updated_steps.append(step_name)
        self.logger.info(f'✅ Added validation to {step_name}')
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

    def update_all_steps(self) -> Dict[str, bool]:
        """Update all known steps with validation."""
        update_results = {}
        steps_to_update = ['step06_advanced_feature_engineering', 'step08_advanced_feature_selection', 'step10_unified_regime_intelligence', 'step11_analyst_creation', 'step12_analyst_enhancement', 'step13_analyst_ensemble_creation', 'step14_tactician_labeling', 'step15_tactician_specialist_training', 'step16_confidence_calibration', 'step17_parameter_optimization', 'step18_walk_forward_validation', 'step19_monte_carlo_validation', 'step20_ab_testing', 'step21_model_persistence']
        for step_name in steps_to_update:
            try:
                step_class = self._import_step_class(step_name)
                if step_class:
                    self.add_validation_to_step_class(step_class, step_name)
                    update_results[step_name] = True
                    self.logger.info(f'✅ Updated {step_name}')
                else:
                    update_results[step_name] = False
                    self.logger.warning(f'⚠️ Could not import {step_name}')
            except Exception as e:
                update_results[step_name] = False
                self.logger.error(f'❌ Failed to update {step_name}: {e}')
        return update_results

    def _import_step_class(self, step_name: str) -> Optional[Type]:
        """Import step class dynamically."""
        step_imports = {
            'step06_advanced_feature_engineering': ('src.training.steps.data_collection.feature_generation.utils.step06_feature_engineering', 'FeatureEngineeringStep'),
            'step08_advanced_feature_selection': ('src.training.steps.data_collection.feature_generation.utils.step08_advanced_feature_selection', 'FeatureSelectionStep'),
            # Simplified model training steps
            'analyst_model_training': ('src.training.steps.model_training.simplified.analyst_model_training', 'AnalystModelTrainer'),
            'tactician_model_training': ('src.training.steps.model_training.simplified.tactician_model_training', 'TacticianModelTrainer'),
            # Legacy step names for backward compatibility
            'step11_analyst_creation': ('src.training.steps.model_training.simplified.analyst_model_training', 'AnalystModelTrainer'),
            'step12_analyst_enhancement': ('src.training.steps.model_training.simplified.analyst_model_training', 'AnalystModelTrainer'),
            'step13_analyst_ensemble_creation': ('src.training.steps.model_training.simplified.analyst_model_training', 'AnalystModelTrainer'),
            'step14_tactician_labeling': ('src.training.steps.model_training.simplified.tactician_model_training', 'TacticianModelTrainer'),
            'step15_tactician_specialist_training': ('src.training.steps.model_training.simplified.tactician_model_training', 'TacticianModelTrainer'),
            # Consolidated backtesting step
            'consolidated_backtesting': ('src.training.steps.backtesting.consolidated_backtesting_step', 'ConsolidatedBacktestingStep'),
            'step18_walk_forward_validation': ('src.training.steps.backtesting.consolidated_backtesting_step', 'ConsolidatedBacktestingStep'),
            'step19_monte_carlo_validation': ('src.training.steps.backtesting.consolidated_backtesting_step', 'ConsolidatedBacktestingStep'),
            'step20_ab_testing': ('src.training.steps.backtesting.consolidated_backtesting_step', 'ConsolidatedBacktestingStep'),
            'step21_model_persistence': ('src.training.steps.backtesting.consolidated_backtesting_step', 'ConsolidatedBacktestingStep')
        }
        if step_name not in step_imports:
            return None
        module_path, class_name = step_imports[step_name]
        try:
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            return None

    def get_update_summary(self) -> Dict[str, Any]:
        """Get summary of updates performed."""
        return {'total_steps_updated': len(self.updated_steps), 'updated_steps': self.updated_steps, 'validation_methods_added': ['_validate_and_fix_input_data', 'execute (wrapped with validation)'], 'pipeline_standards_integration': True}
step_validation_updater = StepValidationUpdater()