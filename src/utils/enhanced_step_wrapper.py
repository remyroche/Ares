"""Enhanced step wrapper integrating data streaming, cross-step validation, and advanced quality metrics."""
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Type, Callable

import traceback
from datetime import datetime
from .logger import system_logger
from src.utils.pipeline_standards import PipelineStandards
from src.utils.data_streaming_manager import DataStreamingManager
from src.utils.cross_step_validator import CrossStepValidator
from src.utils.data_quality.advanced_quality_metrics import AdvancedQualityMetrics

import logging
import time

class EnhancedStepWrapper:
    """Enhanced step wrapper with comprehensive validation and streaming capabilities."""

    def __init__(self, step_class: Type, step_name: str, enable_streaming: bool = True, enable_cross_step_validation: bool = True, enable_advanced_quality: bool = True) -> None:
        """
        Initialize enhanced step wrapper.

        Args:
            step_class: The step class to wrap
            step_name: Name of the step
            enable_streaming: Enable data streaming for large datasets
            enable_cross_step_validation: Enable cross-step validation
            enable_advanced_quality: Enable advanced quality metrics
        """
        self.logger = system_logger.getChild('EnhancedStepWrapper')
        self.step_class = step_class
        self.step_name = step_name
        self.enable_streaming = enable_streaming
        self.enable_cross_step_validation = enable_cross_step_validation
        self.enable_advanced_quality = enable_advanced_quality
        self.standards = PipelineStandards(self.logger)
        self.streaming_manager = DataStreamingManager() if enable_streaming else None
        self.cross_step_validator = CrossStepValidator() if enable_cross_step_validation else None
        self.advanced_quality = AdvancedQualityMetrics() if enable_advanced_quality else None
        self.performance_metrics = {'executions': 0, 'successful_executions': 0, 'failed_executions': 0, 'total_processing_time': 0.0, 'average_quality_score': 0.0, 'streaming_used': 0, 'validation_issues_found': 0}
        self.logger.info(f'🚀 EnhancedStepWrapper initialized for {step_name}')

    def create_enhanced_step(self) -> Type:
        """Create enhanced step class with all improvements."""

        class EnhancedStep(self.step_class):

            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.enhanced_wrapper = EnhancedStepWrapper(self.__class__.__bases__[0], self.step_name if hasattr(self, 'step_name') else 'unknown_step')
                self.enhanced_step_name = self.enhanced_wrapper.step_name
                if not hasattr(self, 'standards'):
                    self.standards = self.enhanced_wrapper.standards

            async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
                """Execute step with comprehensive enhancements."""
                start_time = datetime.now()
                self.enhanced_wrapper.performance_metrics['executions'] += 1
                self.logger.info(f'🛡️ Executing enhanced {self.enhanced_step_name}...')
                try:
                    pre_result = await self._pre_execution_enhancement(training_input, pipeline_state)
                    if not pre_result['success']:
                        return pre_result
                    execution_result = await self._execute_with_enhancements(training_input, pipeline_state)
                    final_result = await self._post_execution_enhancement(execution_result, pipeline_state)
                    execution_time = (datetime.now() - start_time).total_seconds()
                    self.enhanced_wrapper.performance_metrics['total_processing_time'] += execution_time
                    self.enhanced_wrapper.performance_metrics['successful_executions'] += 1
                    return final_result
                except Exception as e:
                    self.logger.exception(f'❌ Enhanced step execution failed: {e}')
                    self.enhanced_wrapper.performance_metrics['failed_executions'] += 1
                    return {'success': False, 'error': str(e), 'step_name': self.enhanced_step_name, 'traceback': traceback.format_exc(), 'enhancement_metadata': {'streaming_used': False, 'validation_performed': False, 'quality_assessment_performed': False}}

            async def _pre_execution_enhancement(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
                """Pre-execution validation and preparation."""
                self.logger.info('🔍 Pre-execution enhancement...')
                input_data = pipeline_state.get('dataframe') or pipeline_state.get('validated_data')
                if input_data is None or not isinstance(input_data, pd.DataFrame):
                    return {'success': False, 'error': 'No valid DataFrame found in pipeline state', 'step_name': self.enhanced_step_name}
                if self.enhanced_wrapper.enable_advanced_quality:
                    quality_assessment = self.enhanced_wrapper.advanced_quality.comprehensive_quality_assessment(input_data, context='pre_execution', step_name = self.enhanced_step_name)
                    if quality_assessment.issues_found > 0:
                        self.logger.warning(f'⚠️ Quality issues detected: {quality_assessment.issues_found} issues, {quality_assessment.warnings_found} warnings')
                        for metric in quality_assessment.metrics:
                            if metric.severity in ['error', 'critical']:
                                self.logger.warning(f'   - {metric.message}')
                    pipeline_state['pre_execution_quality'] = quality_assessment
                if self.enhanced_wrapper.enable_cross_step_validation:
                    previous_step = pipeline_state.get('previous_step_name', 'unknown')
                    validation_result = self.enhanced_wrapper.cross_step_validator.validate_step_transition(previous_step, self.enhanced_step_name, input_data, input_data)
                    if not validation_result['passed']:
                        self.logger.warning(f"⚠️ Cross-step validation issues: {len(validation_result['issues'])} issues")
                        self.enhanced_wrapper.performance_metrics['validation_issues_found'] += len(validation_result['issues'])
                    pipeline_state['cross_step_validation'] = validation_result
                return {'success': True}

            async def _execute_with_enhancements(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
                """Execute step with streaming enhancements if needed."""
                input_data = pipeline_state.get('dataframe') or pipeline_state.get('validated_data')
                if self.enhanced_wrapper.enable_streaming and self.enhanced_wrapper.streaming_manager.should_chunk_data(input_data):
                    self.logger.info('🌊 Using data streaming for large dataset...')
                    self.enhanced_wrapper.performance_metrics['streaming_used'] += 1

                    async def process_chunk(chunk_data: pd.DataFrame) -> pd.DataFrame:
                        chunk_pipeline_state = pipeline_state.copy()
                        chunk_pipeline_state['dataframe'] = chunk_data
                        chunk_result = await super(EnhancedStep, self).execute(training_input, chunk_pipeline_state)
                        if chunk_result.get('success', True) and 'dataframe' in chunk_result:
                            return chunk_result['dataframe']
                        else:
                            return chunk_data
                    try:
                        processed_data = self.enhanced_wrapper.streaming_manager.process_large_dataset(input_data, process_chunk, combine_results = True)
                        return {'success': True, 'dataframe': processed_data, 'enhancement_metadata': {'streaming_used': True, 'original_rows': len(input_data), 'processed_rows': len(processed_data)}}
                    except Exception as e:
                        self.logger.error(f'❌ Streaming execution failed: {e}')
                        self.logger.info('🔄 Falling back to regular execution...')
                return await super(EnhancedStep, self).execute(training_input, pipeline_state)

            async def _post_execution_enhancement(self, execution_result: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
                """Post-execution validation and quality assessment."""
                self.logger.info('🔍 Post-execution enhancement...')
                if not execution_result.get('success', True):
                    return execution_result
                output_data = execution_result.get('dataframe') or execution_result.get('validated_data')
                if output_data is None or not isinstance(output_data, pd.DataFrame):
                    return execution_result
                enhancement_metadata = execution_result.get('enhancement_metadata', {})
                if self.enhanced_wrapper.enable_advanced_quality:
                    output_quality = self.enhanced_wrapper.advanced_quality.comprehensive_quality_assessment(output_data, context='post_execution', step_name = self.enhanced_step_name)
                    current_avg = self.enhanced_wrapper.performance_metrics['average_quality_score']
                    total_executions = self.enhanced_wrapper.performance_metrics['successful_executions']
                    new_avg = (current_avg * (total_executions - 1) + output_quality.overall_score) / total_executions
                    self.enhanced_wrapper.performance_metrics['average_quality_score'] = new_avg
                    self.logger.info(f'📊 Output quality score: {output_quality.overall_score:.1f}/100')
                    enhancement_metadata['output_quality_score'] = output_quality.overall_score
                    enhancement_metadata['quality_issues'] = output_quality.issues_found
                    enhancement_metadata['quality_warnings'] = output_quality.warnings_found
                if self.enhanced_wrapper.enable_cross_step_validation:
                    input_data = pipeline_state.get('dataframe') or pipeline_state.get('validated_data')
                    if input_data is not None:
                        output_validation = self.enhanced_wrapper.cross_step_validator.validate_step_transition(self.enhanced_step_name, f'{self.enhanced_step_name}_output', input_data, output_data)
                        enhancement_metadata['consistency_score'] = output_validation['consistency_score']
                        enhancement_metadata['consistency_issues'] = len(output_validation['issues'])
                        if not output_validation['passed']:
                            self.logger.warning(f"⚠️ Output consistency issues: {len(output_validation['issues'])} issues")
                pipeline_state['previous_step_name'] = self.enhanced_step_name
                pipeline_state['dataframe'] = output_data
                execution_result['enhancement_metadata'] = enhancement_metadata
                execution_result['step_name'] = self.enhanced_step_name
                return execution_result
        return EnhancedStep

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = self.performance_metrics.copy()
        if metrics['executions'] > 0:
            metrics['success_rate'] = metrics['successful_executions'] / metrics['executions']
            metrics['average_execution_time'] = metrics['total_processing_time'] / metrics['executions']
        else:
            metrics['success_rate'] = 0.0
            metrics['average_execution_time'] = 0.0
        return metrics

    def get_enhancement_summary(self) -> Dict[str, Any]:
        """Get enhancement summary."""
        return {'step_name': self.step_name, 'streaming_enabled': self.enable_streaming, 'cross_step_validation_enabled': self.enable_cross_step_validation, 'advanced_quality_enabled': self.enable_advanced_quality, 'performance_metrics': self.get_performance_metrics(), 'streaming_metrics': self.streaming_manager.get_performance_metrics() if self.streaming_manager else None, 'validation_summary': self.cross_step_validator.get_consistency_summary() if self.cross_step_validator else None, 'quality_summary': self.advanced_quality.get_quality_summary() if self.advanced_quality else None}

class EnhancedPipelineManager:
    """Manager for creating enhanced pipeline steps."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild('EnhancedPipelineManager')
        self.enhanced_steps = {}
        self.performance_summary = {'total_steps_enhanced': 0, 'total_executions': 0, 'total_successful_executions': 0, 'average_quality_score': 0.0, 'streaming_usage_count': 0}

    def create_enhanced_step(self, step_class: Type, step_name: str, enable_streaming: bool = True, enable_cross_step_validation: bool = True, enable_advanced_quality: bool = True) -> Type:
        """Create enhanced step with all improvements."""
        wrapper = EnhancedStepWrapper(step_class, step_name, enable_streaming, enable_cross_step_validation, enable_advanced_quality)
        enhanced_step = wrapper.create_enhanced_step()
        self.enhanced_steps[step_name] = {'wrapper': wrapper, 'step_class': enhanced_step}
        self.performance_summary['total_steps_enhanced'] += 1
        self.logger.info(f'✅ Enhanced step created: {step_name}')
        return enhanced_step

    def enhance_all_pipeline_steps(self) -> Dict[str, Type]:
        """Enhance all known pipeline steps."""
        enhanced_steps = {}
        step_configs = {'step02_data_reading': ('src.training.steps.data_collection.step02_data_reading', 'DataReadingStep'), 'step2_5_sr_optimization': ('src.training.steps.market_analysis.sub_pipeline', 'MarketAnalysisSubPipeline'), 'step04_regime_data_splitting': ('src.training.steps.market_analysis.regime_data_splitting.main', 'RegimeDataSplittingStep'), 'step05_labeling': ('src.training.steps.step5_labeling', 'LabelingStep'), 'step06_advanced_feature_engineering': ('src.training.steps.data_collection.feature_generation.utils.step06_feature_engineering', 'FeatureEngineeringStep'), 'step08_advanced_feature_selection': ('src.training.steps.data_collection.feature_generation.utils.step08_advanced_feature_selection', 'FeatureSelectionStep'), 'step10_unified_regime_intelligence': ('src.training.steps.model_training.step10_unified_regime_intelligence', 'RegimeIntelligenceStep'), 'step11_analyst_creation': ('src.training.steps.model_training.step11_analyst_creation', 'AnalystCreationStep'), 'step12_analyst_enhancement': ('src.training.steps.model_training.step12_analyst_enhancement', 'AnalystEnhancementStep'), 'step13_analyst_ensemble_creation': ('src.training.steps.model_training.step13_analyst_ensemble_creation', 'AnalystEnsembleStep'), 'step14_tactician_labeling': ('src.training.steps.model_training.step14_tactician_labeling', 'TacticianLabelingStep'), 'step15_tactician_specialist_training': ('src.training.steps.model_training.step15_tactician_specialist_training', 'TacticianTrainingStep'), 'step16_confidence_calibration': ('src.training.steps.model_training.validation.step16_confidence_calibration', 'ConfidenceCalibrationStep'), 'step17_parameter_optimization': ('src.training.steps.optimisation.step17_final_parameters_optimization', 'ParameterOptimizationStep'), 'step18_walk_forward_validation': ('src.training.steps.model_training.validation.step18_walk_forward_validation', 'WalkForwardValidationStep'), 'step19_monte_carlo_validation': ('src.training.steps.model_training.validation.step19_monte_carlo_validation', 'MonteCarloValidationStep'), 'step20_ab_testing': ('src.training.steps.model_training.validation.step20_ab_testing', 'ABTestingStep'), 'step21_model_persistence': ('src.training.steps.backtesting.step21_saving', 'ModelPersistenceStep')}
        for step_name, (module_path, class_name) in step_configs.items():
            try:
                module = __import__(module_path, fromlist=[class_name])
                step_class = getattr(module, class_name)
                enhanced_step = self.create_enhanced_step(step_class, step_name)
                enhanced_steps[step_name] = enhanced_step
            except (ImportError, AttributeError) as e:
                self.logger.warning(f'⚠️ Could not enhance {step_name}: {e}')
            except Exception as e:
                self.logger.error(f'❌ Error enhancing {step_name}: {e}')
        self.logger.info(f'🎯 Enhanced {len(enhanced_steps)} pipeline steps')
        return enhanced_steps

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline enhancement summary."""
        total_executions = sum((wrapper.get_performance_metrics()['executions'] for wrapper in self.enhanced_steps.values()))
        total_successful = sum((wrapper.get_performance_metrics()['successful_executions'] for wrapper in self.enhanced_steps.values()))
        total_streaming = sum((wrapper.get_performance_metrics()['streaming_used'] for wrapper in self.enhanced_steps.values()))
        avg_quality = np.mean([wrapper.get_performance_metrics()['average_quality_score'] for wrapper in self.enhanced_steps.values()]) if self.enhanced_steps else 0.0
        return {'total_steps_enhanced': len(self.enhanced_steps), 'total_executions': total_executions, 'total_successful_executions': total_successful, 'success_rate': total_successful / total_executions if total_executions > 0 else 0.0, 'average_quality_score': avg_quality, 'streaming_usage_count': total_streaming, 'enhanced_steps': list(self.enhanced_steps.keys())}
enhanced_step_wrapper = EnhancedStepWrapper
enhanced_pipeline_manager = EnhancedPipelineManager()
