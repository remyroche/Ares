"""Validated step factory for consistent pipeline standards validation across all steps."""
import pandas as pd

from typing import Any, Dict, List, Optional, Tuple, Union, Type

import traceback

from .logger import system_logger
from src.utils.pipeline_standards import PipelineStandards, ValidationResult

import logging
import numpy as np

class ValidatedStepFactory:
    """Factory for creating validated steps with consistent pipeline standards validation."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild('ValidatedStepFactory')
        self.standards = PipelineStandards(self.logger)
        self.validation_history = []
        self.step_schemas = {'data_reading': 'unified', 'sr_optimization': 'unified', 'hmm_regime_discovery': 'unified', 'regime_data_splitting': 'unified', 'labeling': 'unified', 'feature_engineering': 'unified', 'matrix_operations': 'unified', 'feature_selection': 'unified', 'hmm_training': 'unified', 'regime_intelligence': 'unified', 'analyst_creation': 'unified', 'analyst_enhancement': 'unified', 'analyst_ensemble': 'unified', 'tactician_labeling': 'unified', 'tactician_training': 'unified', 'confidence_calibration': 'unified', 'parameter_optimization': 'unified', 'walk_forward_validation': 'unified', 'monte_carlo_validation': 'unified', 'ab_testing': 'unified', 'model_persistence': 'unified'}

    def create_validated_step(self, step_class: Type, step_name: str, step_type: str = None) -> Type:
        """
        Create a validated step class with pipeline standards validation.
        
        Args:
            step_class: The original step class
            step_name: Name of the step
            step_type: Type of step for schema selection
            
        Returns:
            Validated step class
        """
        schema_name = self.step_schemas.get(step_type, 'unified')

        class ValidatedStep(step_class):

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.validation_factory = ValidatedStepFactory()
                self.validated_step_name = step_name
                self.schema_name = schema_name
                self.step_type = step_type
                if not hasattr(self, 'standards'):
                    self.standards = PipelineStandards(self.logger)

            async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
                """Execute step with comprehensive validation using pipeline standards."""
                self.logger.info(f'🛡️ Executing {self.validated_step_name} with pipeline standards validation')
                pre_validation = self._validate_pre_execution(training_input, pipeline_state)
                if not pre_validation['passed']:
                    self.logger.error(f'❌ Pre-execution validation failed for {self.validated_step_name}')
                    return {'success': False, 'error': f"Pre-execution validation failed: {pre_validation['issues']}", 'step_name': self.validated_step_name}
                try:
                    result = await super().execute(training_input, pipeline_state)
                    result = self._validate_and_fix_output(result, pipeline_state)
                    return result
                except Exception as e:
                    self.logger.exception(f'❌ Step {self.validated_step_name} failed: {e}')
                    return {'success': False, 'error': str(e), 'step_name': self.validated_step_name, 'traceback': traceback.format_exc()}

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
                standards_validation = self.standards.validate_data_quality(data, self.schema_name)
                validation_result['quality_score'] = standards_validation.quality_score
                if not standards_validation.passed:
                    validation_result['passed'] = False
                    for issue in standards_validation.issues:
                        validation_result['issues'].append(issue.message)
                for warning in standards_validation.warnings:
                    validation_result['warnings'].append(warning.message)
                step_validation = self._step_specific_pre_validation(data, training_input, pipeline_state)
                if not step_validation['passed']:
                    validation_result['passed'] = False
                    validation_result['issues'].extend(step_validation['issues'])
                return validation_result

            def _validate_and_fix_output(self, result: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
                """Validate and fix step output."""
                if not result.get('success', True):
                    return result
                data = result.get('dataframe') or result.get('validated_data')
                if isinstance(data, pd.DataFrame):
                    output_validation = self.standards.validate_data_quality(data, self.schema_name)
                    if not output_validation.passed or output_validation.quality_score < 80.0:
                        self.logger.warning(f'⚠️ Output validation issues in {self.validated_step_name}: {output_validation.quality_score:.2f}')
                        fixed_data = self._apply_pipeline_standards_fixes(data)
                        result['dataframe'] = fixed_data
                        result['validation_fixes_applied'] = True
                        result['original_quality_score'] = output_validation.quality_score
                        final_validation = self.standards.validate_data_quality(fixed_data, self.schema_name)
                        result['final_quality_score'] = final_validation.quality_score
                        self.logger.info(f'✅ Applied fixes, quality improved: {output_validation.quality_score:.2f} → {final_validation.quality_score:.2f}')
                return result

            def _step_specific_pre_validation(self, data: pd.DataFrame, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
                """Step-specific pre-execution validation."""
                validation_result = {'passed': True, 'issues': [], 'warnings': []}
                if len(data) == 0:
                    validation_result['passed'] = False
                    validation_result['issues'].append('Empty DataFrame')
                if self.step_type == 'labeling':
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing_cols = [col for col in required_cols if col not in data.columns]
                    if missing_cols:
                        validation_result['passed'] = False
                        validation_result['issues'].append(f'Missing columns for labeling: {missing_cols}')
                elif self.step_type == 'feature_engineering':
                    if len(data) < 100:
                        validation_result['passed'] = False
                        validation_result['issues'].append('Insufficient data for feature engineering')
                elif self.step_type == 'hmm_training':
                    if 'regime_labels' not in pipeline_state:
                        validation_result['warnings'].append('No regime labels found in pipeline state')
                elif self.step_type == 'regime_data_splitting':
                    if 'regime_discovery' not in pipeline_state:
                        validation_result['passed'] = False
                        validation_result['issues'].append('No regime discovery results found')
                return validation_result

            def _apply_pipeline_standards_fixes(self, data: pd.DataFrame) -> pd.DataFrame:
                """Apply comprehensive fixes using pipeline standards."""
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
                    fixed_data = self.standards.enforce_schema(fixed_data, self.schema_name)
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
                return fixed_data
        return ValidatedStep

    def create_all_validated_steps(self) -> Dict[str, Type]:
        """Create validated versions of all pipeline steps."""
        validated_steps = {}
        step_imports = {
            'step02_data_reading': ('src.training.steps.data_collection.step02_data_reading', 'DataReadingStep'),
            'step02_5_sr_optimization': ('src.training.steps.market_analysis.sub_pipeline', 'MarketAnalysisSubPipeline'),
            'step2_5_sr_optimization': ('src.training.steps.market_analysis.sub_pipeline', 'MarketAnalysisSubPipeline'),
            'step04_regime_data_splitting': ('src.training.steps.market_analysis.regime_data_splitting.main', 'RegimeDataSplittingStep'),
            'step05_labeling': ('src.training.steps.step5_labeling', 'LabelingStep'),
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
        # Deprecated Step07 removed from imports
        for step_name, (module_path, class_name) in step_imports.items():
            try:
                module = __import__(module_path, fromlist=[class_name])
                step_class = getattr(module, class_name)
                step_type = self._determine_step_type(step_name)
                validated_step = self.create_validated_step(step_class, step_name, step_type)
                validated_steps[step_name] = validated_step
                self.logger.info(f'✅ Created validated step: {step_name}')
            except ImportError as e:
                self.logger.warning(f'⚠️ Could not import {step_name}: {e}')
            except AttributeError as e:
                self.logger.warning(f'⚠️ Could not find class {class_name} in {module_path}: {e}')
            except Exception as e:
                self.logger.error(f'❌ Error creating validated step {step_name}: {e}')
        return validated_steps

    def _determine_step_type(self, step_name: str) -> str:
        """Determine step type for schema selection."""
        if 'data_reading' in step_name:
            return 'data_reading'
        elif 'sr_optimization' in step_name:
            return 'sr_optimization'
        elif 'hmm_regime_discovery' in step_name:
            return 'hmm_regime_discovery'
        elif 'regime_data_splitting' in step_name:
            return 'regime_data_splitting'
        elif 'labeling' in step_name:
            return 'labeling'
        elif 'feature_engineering' in step_name:
            return 'feature_engineering'
        elif 'matrix_operations' in step_name:
            return 'matrix_operations'
        elif 'feature_selection' in step_name:
            return 'feature_selection'
        elif 'hmm_based_training' in step_name or 'hmm_training' in step_name:
            return 'hmm_training'
        elif 'regime_intelligence' in step_name:
            return 'regime_intelligence'
        elif 'analyst_creation' in step_name:
            return 'analyst_creation'
        elif 'analyst_enhancement' in step_name:
            return 'analyst_enhancement'
        elif 'analyst_ensemble' in step_name:
            return 'analyst_ensemble'
        elif 'tactician_labeling' in step_name:
            return 'tactician_labeling'
        elif 'tactician_specialist_training' in step_name or 'tactician_training' in step_name:
            return 'tactician_training'
        elif 'confidence_calibration' in step_name:
            return 'confidence_calibration'
        elif 'parameter_optimization' in step_name:
            return 'parameter_optimization'
        elif 'walk_forward_validation' in step_name:
            return 'walk_forward_validation'
        elif 'monte_carlo_validation' in step_name:
            return 'monte_carlo_validation'
        elif 'ab_testing' in step_name:
            return 'ab_testing'
        elif 'model_persistence' in step_name or 'saving' in step_name:
            return 'model_persistence'
        else:
            return 'unified'

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation activities."""
        return {'total_validated_steps': len(self.step_schemas), 'available_schemas': list(self.step_schemas.keys()), 'validation_history_count': len(self.validation_history), 'factory_initialized': True}
validated_step_factory = ValidatedStepFactory()