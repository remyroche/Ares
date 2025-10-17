"""
Unified Training Pipeline - ModularComponent Implementation

This module provides a unified training pipeline that orchestrates all models training
components using the ModularComponent architecture. It coordinates:
- Analyst models training
- Analyst ensemble training
- Tactician models training
- ML-based entry timing labeling
- Cross-component validation and monitoring

ENHANCED FEATURES:
- ModularComponent architecture with comprehensive orchestration
- Cross-component state management and monitoring
- Enhanced error handling and recovery
- Configuration management and validation
- Training progress tracking and health monitoring
- Automated component coordination
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
import traceback
from dataclasses import dataclass
from enum import Enum

from .unified_data_driven_pipeline.core.modular_architecture import (
    ModularComponent, ErrorInfo, ErrorSeverity, ErrorCategory, ValidationResult
)

# Import migrated components
from .components.analyst_models_training_modular import (
    AnalystModelsTrainingModular, create_analyst_models_training
)
from .components.analyst_ensemble_training_modular import (
    AnalystEnsembleTrainingModular, create_analyst_ensemble_training
)
from .components.ml_entry_timing_labeler_modular import (
    MLEntryTimingLabelerModular, create_ml_entry_timing_labeler
)


class TrainingPhase(Enum):
    """Training phases."""
    INITIALIZATION = "initialization"
    ANALYST_MODELS = "analyst_models"
    ANALYST_ENSEMBLE = "analyst_ensemble"
    TACTICIAN_MODELS = "tactician_models"
    ML_LABELING = "ml_labeling"
    VALIDATION = "validation"
    COMPLETION = "completion"


@dataclass
class UnifiedTrainingConfig:
    """Configuration for unified training pipeline."""
    analyst_models_config: Dict[str, Any]
    analyst_ensemble_config: Dict[str, Any]
    tactician_models_config: Dict[str, Any]
    ml_labeling_config: Dict[str, Any]
    pipeline_config: Dict[str, Any]
    validation_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]


@dataclass
class UnifiedTrainingResult:
    """Result of unified training pipeline."""
    success: bool
    phase_results: Dict[str, Any]
    overall_metrics: Dict[str, float]
    training_time: float
    errors: List[str]
    warnings: List[str]
    component_health: Dict[str, Any]
    recommendations: List[str]


class UnifiedTrainingPipelineModular(ModularComponent):
    """
    ModularComponent implementation of Unified Training Pipeline.
    
    This component orchestrates all models training components with comprehensive
    state management, performance monitoring, and error handling.
    """
    
    def __init__(
        self,
        name: str = "unified_training_pipeline",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Unified Training Pipeline.
        
        Args:
            name: Component name
            config: Configuration dictionary
            logger: Logger instance
        """
        # Set default configuration
        default_config = {
            'pipeline': {
                'phases': ['analyst_models', 'analyst_ensemble', 'tactician_models', 'ml_labeling'],
                'parallel_execution': False,
                'error_handling': 'continue',
                'monitoring': True
            },
            'analyst_models': {
                'model_types': ['lightgbm', 'lightgbm_patchtst', 'catboost', 'stacker_lgbm_calibrated'],
                'regime_aware': True,
                'timeframe': '15m'
            },
            'analyst_ensemble': {
                'base_models': ['lightgbm', 'lightgbm_patchtst', 'catboost', 'stacker_lgbm_calibrated'],
                'ensemble_method': 'voting',
                'regime_aware': True,
                'timeframe': '15m'
            },
            'tactician_models': {
                'model_types': ['lgbm_gru', 'catboost', 'causal_tcn', 'stacker_lgbm_calibrated'],
                'regime_aware': True,
                'timeframe': '5m'
            },
            'ml_labeling': {
                'labeling_method': 'iterative',
                'ml_model_type': 'random_forest',
                'quality_threshold': 0.7,
                'max_iterations': 5
            },
            'validation': {
                'cross_validation': True,
                'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
                'thresholds': {'accuracy': 0.8, 'f1_score': 0.75}
            },
            'monitoring': {
                'health_checks': True,
                'performance_tracking': True,
                'error_reporting': True,
                'progress_updates': True
            }
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(name, default_config, logger)
        
        # Unified training configuration
        self.unified_config = UnifiedTrainingConfig(
            analyst_models_config=self.get_config('analyst_models', {}),
            analyst_ensemble_config=self.get_config('analyst_ensemble', {}),
            tactician_models_config=self.get_config('tactician_models', {}),
            ml_labeling_config=self.get_config('ml_labeling', {}),
            pipeline_config=self.get_config('pipeline', {}),
            validation_config=self.get_config('validation', {}),
            monitoring_config=self.get_config('monitoring', {})
        )
        
        # Component instances
        self._components = {}
        self._phase_results = {}
        self._overall_metrics = {}
        self._component_health = {}
        self._current_phase = TrainingPhase.INITIALIZATION
        
        self.logger.info(f"Initialized UnifiedTrainingPipelineModular: {name}")
    
    def _initialize_resources(self) -> bool:
        """Initialize component-specific resources."""
        try:
            # Initialize component instances
            self._initialize_components()
            
            # Initialize pipeline state
            self.set_ml_state('pipeline_initialized', True)
            self.set_ml_state('current_phase', self._current_phase.value)
            self.set_ml_state('phases_completed', [])
            self.set_ml_state('overall_success', False)
            
            self.logger.info("Unified training pipeline resources initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Resource initialization failed: {e}")
            return False
    
    def _cleanup_resources(self) -> None:
        """Cleanup component-specific resources."""
        try:
            # Cleanup all components
            for comp_name, component in self._components.items():
                try:
                    component.cleanup()
                    self.logger.info(f"Cleaned up component: {comp_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup component {comp_name}: {e}")
            
            # Clear state
            self._components.clear()
            self._phase_results.clear()
            self._overall_metrics.clear()
            self._component_health.clear()
            
            # Clear pipeline state
            self.set_ml_state('pipeline_initialized', False)
            self.set_ml_state('phases_completed', [])
            
            self.logger.info("Unified training pipeline resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"Resource cleanup failed: {e}")
    
    def _initialize_components(self) -> None:
        """Initialize all training components."""
        try:
            # Initialize Analyst Models Training
            if 'analyst_models' in self.unified_config.pipeline_config.get('phases', []):
                self._components['analyst_models'] = create_analyst_models_training(
                    config=self.unified_config.analyst_models_config,
                    logger=self.logger
                )
                self.logger.info("Initialized Analyst Models Training component")
            
            # Initialize Analyst Ensemble Training
            if 'analyst_ensemble' in self.unified_config.pipeline_config.get('phases', []):
                self._components['analyst_ensemble'] = create_analyst_ensemble_training(
                    config=self.unified_config.analyst_ensemble_config,
                    logger=self.logger
                )
                self.logger.info("Initialized Analyst Ensemble Training component")
            
            # Initialize ML Entry Timing Labeler
            if 'ml_labeling' in self.unified_config.pipeline_config.get('phases', []):
                self._components['ml_labeling'] = create_ml_entry_timing_labeler(
                    config=self.unified_config.ml_labeling_config,
                    logger=self.logger
                )
                self.logger.info("Initialized ML Entry Timing Labeler component")
            
            # Note: Tactician models would be initialized here when migrated
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise
    
    def _process_data(self, data: Any, **kwargs) -> Any:
        """Process data with unified training pipeline logic."""
        try:
            self.logger.info("Starting unified training pipeline")
            
            # Validate input data
            if not self._validate_pipeline_data(data):
                raise ValueError("Invalid pipeline data")
            
            # Start pipeline execution
            if not self._start_pipeline():
                raise RuntimeError("Failed to start pipeline")
            
            # Execute training phases
            pipeline_result = self._execute_training_phases(data)
            
            # Stop pipeline execution
            self._stop_pipeline()
            
            # Prepare result
            result = UnifiedTrainingResult(
                success=pipeline_result['success'],
                phase_results=self._phase_results,
                overall_metrics=self._overall_metrics,
                training_time=self.get_ml_state('total_training_time', 0),
                errors=pipeline_result['errors'],
                warnings=pipeline_result['warnings'],
                component_health=self._component_health,
                recommendations=pipeline_result['recommendations']
            )
            
            self.logger.info(f"Unified training pipeline completed in {result.training_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Unified training pipeline failed: {e}")
            self._stop_pipeline()
            raise
    
    def _validate_pipeline_data(self, data: Any) -> bool:
        """Validate pipeline data."""
        try:
            if not isinstance(data, dict):
                self.logger.error("Pipeline data must be a dictionary")
                return False
            
            # Check for required data sections
            required_sections = ['analyst_data', 'tactician_data', 'labeling_data']
            for section in required_sections:
                if section not in data:
                    self.logger.warning(f"Missing data section: {section}")
            
            # Validate each data section
            for section, section_data in data.items():
                if not self._validate_section_data(section, section_data):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline data validation failed: {e}")
            return False
    
    def _validate_section_data(self, section: str, data: Any) -> bool:
        """Validate specific data section."""
        try:
            if section == 'analyst_data':
                required_keys = ['X_train', 'y_train']
            elif section == 'tactician_data':
                required_keys = ['X_train', 'y_train']
            elif section == 'labeling_data':
                required_keys = ['features', 'market_data']
            else:
                return True  # Unknown section, skip validation
            
            for key in required_keys:
                if key not in data:
                    self.logger.error(f"Missing required key {key} in {section}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Section data validation failed for {section}: {e}")
            return False
    
    def _start_pipeline(self) -> bool:
        """Start pipeline execution."""
        try:
            self.set_ml_state('pipeline_started', True)
            self.set_ml_state('pipeline_start_time', time.time())
            self._current_phase = TrainingPhase.INITIALIZATION
            
            self.logger.info("Pipeline execution started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start pipeline: {e}")
            return False
    
    def _stop_pipeline(self) -> None:
        """Stop pipeline execution."""
        try:
            self.set_ml_state('pipeline_started', False)
            self.set_ml_state('pipeline_end_time', time.time())
            
            # Calculate total pipeline time
            start_time = self.get_ml_state('pipeline_start_time', time.time())
            total_time = time.time() - start_time
            self.set_ml_state('total_training_time', total_time)
            
            self.logger.info(f"Pipeline execution stopped after {total_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to stop pipeline: {e}")
    
    def _execute_training_phases(self, data: Any) -> Dict[str, Any]:
        """Execute all training phases."""
        try:
            phases = self.unified_config.pipeline_config.get('phases', [])
            errors = []
            warnings = []
            recommendations = []
            
            for phase in phases:
                try:
                    self.logger.info(f"Executing phase: {phase}")
                    self._current_phase = TrainingPhase(phase)
                    self.set_ml_state('current_phase', phase)
                    
                    # Execute phase
                    phase_result = self._execute_phase(phase, data)
                    
                    if phase_result['success']:
                        self._phase_results[phase] = phase_result
                        self.logger.info(f"✅ Phase {phase} completed successfully")
                    else:
                        self._phase_results[phase] = phase_result
                        errors.extend(phase_result['errors'])
                        warnings.extend(phase_result['warnings'])
                        self.logger.error(f"❌ Phase {phase} failed: {phase_result['errors']}")
                        
                        # Handle error based on configuration
                        error_handling = self.unified_config.pipeline_config.get('error_handling', 'continue')
                        if error_handling == 'stop':
                            break
                    
                    # Update completed phases
                    completed_phases = self.get_ml_state('phases_completed', [])
                    completed_phases.append(phase)
                    self.set_ml_state('phases_completed', completed_phases)
                    
                except Exception as e:
                    error_msg = f"Phase {phase} execution failed: {str(e)}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
                    
                    # Handle error based on configuration
                    error_handling = self.unified_config.pipeline_config.get('error_handling', 'continue')
                    if error_handling == 'stop':
                        break
            
            # Calculate overall metrics
            self._calculate_overall_metrics()
            
            # Generate recommendations
            recommendations = self._generate_recommendations()
            
            # Determine overall success
            success = len(errors) == 0 and len(self._phase_results) > 0
            
            return {
                'success': success,
                'errors': errors,
                'warnings': warnings,
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Phase execution failed: {e}")
            return {
                'success': False,
                'errors': [str(e)],
                'warnings': [],
                'recommendations': []
            }
    
    def _execute_phase(self, phase: str, data: Any) -> Dict[str, Any]:
        """Execute specific training phase."""
        try:
            if phase not in self._components:
                return {
                    'success': False,
                    'errors': [f"Component for phase {phase} not available"],
                    'warnings': [],
                    'metrics': {}
                }
            
            component = self._components[phase]
            
            # Get phase-specific data
            phase_data = self._get_phase_data(phase, data)
            
            # Execute component
            result = component.process(phase_data)
            
            # Update component health
            self._component_health[phase] = component.get_health_report()
            
            return {
                'success': True,
                'errors': [],
                'warnings': [],
                'metrics': result.metrics if hasattr(result, 'metrics') else {},
                'result': result
            }
            
        except Exception as e:
            self.logger.error(f"Phase {phase} execution failed: {e}")
            return {
                'success': False,
                'errors': [str(e)],
                'warnings': [],
                'metrics': {}
            }
    
    def _get_phase_data(self, phase: str, data: Any) -> Any:
        """Get phase-specific data from pipeline data."""
        if phase == 'analyst_models':
            return data.get('analyst_data', data)
        elif phase == 'analyst_ensemble':
            return data.get('analyst_data', data)
        elif phase == 'tactician_models':
            return data.get('tactician_data', data)
        elif phase == 'ml_labeling':
            return data.get('labeling_data', data)
        else:
            return data
    
    def _calculate_overall_metrics(self) -> None:
        """Calculate overall pipeline metrics."""
        try:
            overall_metrics = {
                'total_phases': len(self._phase_results),
                'successful_phases': sum(1 for r in self._phase_results.values() if r['success']),
                'failed_phases': sum(1 for r in self._phase_results.values() if not r['success']),
                'overall_accuracy': 0.0,
                'overall_precision': 0.0,
                'overall_recall': 0.0,
                'overall_f1_score': 0.0
            }
            
            # Aggregate metrics from successful phases
            successful_metrics = []
            for phase_result in self._phase_results.values():
                if phase_result['success'] and 'metrics' in phase_result:
                    metrics = phase_result['metrics']
                    if isinstance(metrics, dict):
                        successful_metrics.append(metrics)
            
            if successful_metrics:
                # Calculate average metrics
                overall_metrics['overall_accuracy'] = np.mean([
                    m.get('accuracy', 0) for m in successful_metrics
                ])
                overall_metrics['overall_precision'] = np.mean([
                    m.get('precision', 0) for m in successful_metrics
                ])
                overall_metrics['overall_recall'] = np.mean([
                    m.get('recall', 0) for m in successful_metrics
                ])
                overall_metrics['overall_f1_score'] = np.mean([
                    m.get('f1_score', 0) for m in successful_metrics
                ])
            
            self._overall_metrics = overall_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate overall metrics: {e}")
            self._overall_metrics = {}
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on pipeline execution."""
        recommendations = []
        
        # Check overall success rate
        if self._overall_metrics.get('successful_phases', 0) < self._overall_metrics.get('total_phases', 0):
            recommendations.append("Some phases failed - review error logs and consider retrying")
        
        # Check accuracy thresholds
        accuracy = self._overall_metrics.get('overall_accuracy', 0)
        if accuracy < self.unified_config.validation_config.get('thresholds', {}).get('accuracy', 0.8):
            recommendations.append("Overall accuracy below threshold - consider adjusting model parameters")
        
        f1_score = self._overall_metrics.get('overall_f1_score', 0)
        if f1_score < self.unified_config.validation_config.get('thresholds', {}).get('f1_score', 0.75):
            recommendations.append("Overall F1 score below threshold - consider data augmentation or model tuning")
        
        # Check component health
        for comp_name, health in self._component_health.items():
            if health.get('overall_health') == 'poor':
                recommendations.append(f"Component {comp_name} health is poor - review configuration and logs")
        
        return recommendations
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for this component."""
        return {
            'min_size': 100,
            'max_size': 1000000,
            'required_keys': ['analyst_data'],
            'data_types': ['dict'],
            'required_columns': ['analyst_data']
        }
    
    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        """Validate data with component-specific rules."""
        errors = []
        warnings = []
        metadata = {}
        
        if isinstance(data, dict):
            # Check for required data sections
            required_sections = ['analyst_data']
            for section in required_sections:
                if section not in data:
                    errors.append(f"Missing required data section: {section}")
            
            # Check data consistency
            for section, section_data in data.items():
                if isinstance(section_data, dict) and 'X_train' in section_data:
                    X_train = section_data['X_train']
                    if hasattr(X_train, 'shape'):
                        metadata[f'{section}_shape'] = X_train.shape
                        
                        if len(X_train) < 100:
                            warnings.append(f"{section} data is small, consider more data")
        
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        summary = super().get_training_summary()
        
        # Add pipeline-specific information
        summary.update({
            'pipeline_config': {
                'phases': self.unified_config.pipeline_config.get('phases', []),
                'parallel_execution': self.unified_config.pipeline_config.get('parallel_execution', False),
                'error_handling': self.unified_config.pipeline_config.get('error_handling', 'continue')
            },
            'components': list(self._components.keys()),
            'phase_results': self._phase_results,
            'overall_metrics': self._overall_metrics,
            'component_health': self._component_health,
            'current_phase': self._current_phase.value
        })
        
        return summary


def create_unified_training_pipeline(
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> UnifiedTrainingPipelineModular:
    """
    Factory function to create Unified Training Pipeline.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Initialized UnifiedTrainingPipelineModular instance
    """
    return UnifiedTrainingPipelineModular(
        name="unified_training_pipeline",
        config=config,
        logger=logger
    )