#!/usr/bin/env python3
"""
Step Validators for Model Training Pipeline

This module provides individual validators for each step of the model training pipeline,
ensuring proper validation at each stage with comprehensive error handling.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.pipeline_validation_framework import (
    BaseValidator,
    ValidationLevel,
    ValidationResult,
    ValidationReport,
)
from src.utils.enhanced_common_operations import (
    validate_dataframe_integrity,
    validate_pipeline_step_output,
    DataValidationError,
    DataProcessingError,
)


class ModelTrainingStepValidator(BaseValidator):
    """Base validator for model training steps."""
    
    def __init__(self, step_name: str, validation_level: ValidationLevel = ValidationLevel.CRITICAL):
        super().__init__(step_name, validation_level)
        self.logger = system_logger.getChild(f"ModelTrainingValidator.{step_name}")
    
    async def validate(self, data: Any, context: Dict[str, Any]) -> ValidationReport:
        """Validate model training step data."""
        start_time = time.time()
        details = {}
        warnings = []
        errors = []
        recommendations = []
        
        try:
            # Basic validation
            if data is None:
                errors.append("Step output is None")
                result = ValidationResult.FAILED
            else:
                # Validate based on data type
                if isinstance(data, dict):
                    result = await self._validate_dict_output(data, context, details, warnings, errors, recommendations)
                elif isinstance(data, pd.DataFrame):
                    result = await self._validate_dataframe_output(data, context, details, warnings, errors, recommendations)
                else:
                    result = await self._validate_other_output(data, context, details, warnings, errors, recommendations)
            
            # Add step-specific validation
            step_result = await self._validate_step_specific(data, context, details, warnings, errors, recommendations)
            if step_result == ValidationResult.FAILED:
                result = ValidationResult.FAILED
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            result = ValidationResult.FAILED
            self.logger.exception(f"Step validation failed: {e}")
        
        duration = time.time() - start_time
        return self._create_report(
            result=result,
            duration=duration,
            details=details,
            warnings=warnings,
            errors=errors,
            recommendations=recommendations
        )
    
    async def _validate_dict_output(
        self, 
        data: Dict[str, Any], 
        context: Dict[str, Any], 
        details: Dict[str, Any],
        warnings: List[str],
        errors: List[str],
        recommendations: List[str]
    ) -> ValidationResult:
        """Validate dictionary output."""
        details["output_type"] = "dict"
        details["keys"] = list(data.keys())
        
        # Check for required keys
        required_keys = self._get_required_keys()
        if required_keys:
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                errors.append(f"Missing required keys: {missing_keys}")
                return ValidationResult.FAILED
        
        return ValidationResult.PASSED
    
    async def _validate_dataframe_output(
        self, 
        data: pd.DataFrame, 
        context: Dict[str, Any], 
        details: Dict[str, Any],
        warnings: List[str],
        errors: List[str],
        recommendations: List[str]
    ) -> ValidationResult:
        """Validate DataFrame output."""
        details["output_type"] = "dataframe"
        details["shape"] = data.shape
        details["columns"] = list(data.columns)
        
        # Validate DataFrame integrity
        integrity_results = validate_dataframe_integrity(data, self._get_required_columns())
        
        if not integrity_results['is_valid']:
            errors.extend(integrity_results['errors'])
            return ValidationResult.FAILED
        
        if integrity_results['warnings']:
            warnings.extend(integrity_results['warnings'])
        
        details.update(integrity_results['statistics'])
        return ValidationResult.PASSED
    
    async def _validate_other_output(
        self, 
        data: Any, 
        context: Dict[str, Any], 
        details: Dict[str, Any],
        warnings: List[str],
        errors: List[str],
        recommendations: List[str]
    ) -> ValidationResult:
        """Validate other types of output."""
        details["output_type"] = type(data).__name__
        return ValidationResult.PASSED
    
    @abstractmethod
    async def _validate_step_specific(
        self, 
        data: Any, 
        context: Dict[str, Any], 
        details: Dict[str, Any],
        warnings: List[str],
        errors: List[str],
        recommendations: List[str]
    ) -> ValidationResult:
        """Validate step-specific requirements."""
        pass
    
    def _get_required_keys(self) -> List[str]:
        """Get required keys for dictionary output."""
        return []
    
    def _get_required_columns(self) -> List[str]:
        """Get required columns for DataFrame output."""
        return []


class HMMTrainingValidator(ModelTrainingStepValidator):
    """Validator for HMM-based training step."""
    
    def __init__(self):
        super().__init__("HMMTraining", ValidationLevel.CRITICAL)
    
    async def _validate_step_specific(
        self, 
        data: Any, 
        context: Dict[str, Any], 
        details: Dict[str, Any],
        warnings: List[str],
        errors: List[str],
        recommendations: List[str]
    ) -> ValidationResult:
        """Validate HMM training specific requirements."""
        if isinstance(data, dict):
            # Check for HMM model
            if 'hmm_model' not in data:
                errors.append("Missing HMM model in output")
                return ValidationResult.FAILED
            
            # Check for training metrics
            if 'training_metrics' not in data:
                errors.append("Missing training metrics in output")
                return ValidationResult.FAILED
            
            # Validate training metrics
            metrics = data['training_metrics']
            if not isinstance(metrics, dict):
                errors.append("Training metrics must be a dictionary")
                return ValidationResult.FAILED
            
            # Check for required metrics
            required_metrics = ['accuracy', 'loss', 'convergence_iterations']
            missing_metrics = [metric for metric in required_metrics if metric not in metrics]
            if missing_metrics:
                warnings.append(f"Missing metrics: {missing_metrics}")
            
            # Validate metric values
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    if np.isnan(metric_value) or np.isinf(metric_value):
                        errors.append(f"Invalid metric value for {metric_name}: {metric_value}")
                    elif metric_name == 'accuracy' and (metric_value < 0 or metric_value > 1):
                        warnings.append(f"Accuracy out of range [0,1]: {metric_value}")
            
            details["hmm_validation"] = {
                "model_present": 'hmm_model' in data,
                "metrics_present": 'training_metrics' in data,
                "metrics_count": len(metrics) if isinstance(metrics, dict) else 0
            }
        
        return ValidationResult.PASSED if not errors else ValidationResult.FAILED


class RegimeIntelligenceValidator(ModelTrainingStepValidator):
    """Validator for unified regime intelligence step."""
    
    def __init__(self):
        super().__init__("RegimeIntelligence", ValidationLevel.CRITICAL)
    
    async def _validate_step_specific(
        self, 
        data: Any, 
        context: Dict[str, Any], 
        details: Dict[str, Any],
        warnings: List[str],
        errors: List[str],
        recommendations: List[str]
    ) -> ValidationResult:
        """Validate regime intelligence specific requirements."""
        if isinstance(data, dict):
            # Check for regime intelligence components
            required_components = ['regime_classifier', 'intelligence_metrics', 'regime_insights']
            missing_components = [comp for comp in required_components if comp not in data]
            if missing_components:
                errors.append(f"Missing regime intelligence components: {missing_components}")
                return ValidationResult.FAILED
            
            # Validate regime classifier
            classifier = data['regime_classifier']
            if classifier is None:
                errors.append("Regime classifier is None")
                return ValidationResult.FAILED
            
            # Validate intelligence metrics
            metrics = data['intelligence_metrics']
            if not isinstance(metrics, dict):
                errors.append("Intelligence metrics must be a dictionary")
                return ValidationResult.FAILED
            
            # Check for required metrics
            required_metrics = ['regime_accuracy', 'transition_accuracy', 'confidence_score']
            missing_metrics = [metric for metric in required_metrics if metric not in metrics]
            if missing_metrics:
                warnings.append(f"Missing intelligence metrics: {missing_metrics}")
            
            # Validate regime insights
            insights = data['regime_insights']
            if not isinstance(insights, list):
                errors.append("Regime insights must be a list")
                return ValidationResult.FAILED
            
            if len(insights) == 0:
                warnings.append("No regime insights generated")
            
            details["regime_intelligence_validation"] = {
                "classifier_present": classifier is not None,
                "metrics_present": isinstance(metrics, dict),
                "insights_count": len(insights) if isinstance(insights, list) else 0
            }
        
        return ValidationResult.PASSED if not errors else ValidationResult.FAILED


class AnalystCreationValidator(ModelTrainingStepValidator):
    """Validator for analyst creation step."""
    
    def __init__(self):
        super().__init__("AnalystCreation", ValidationLevel.CRITICAL)
    
    async def _validate_step_specific(
        self, 
        data: Any, 
        context: Dict[str, Any], 
        details: Dict[str, Any],
        warnings: List[str],
        errors: List[str],
        recommendations: List[str]
    ) -> ValidationResult:
        """Validate analyst creation specific requirements."""
        if isinstance(data, dict):
            # Check for analyst components
            required_components = ['analysts', 'creation_metrics', 'analyst_configs']
            missing_components = [comp for comp in required_components if comp not in data]
            if missing_components:
                errors.append(f"Missing analyst components: {missing_components}")
                return ValidationResult.FAILED
            
            # Validate analysts
            analysts = data['analysts']
            if not isinstance(analysts, dict):
                errors.append("Analysts must be a dictionary")
                return ValidationResult.FAILED
            
            if len(analysts) == 0:
                errors.append("No analysts created")
                return ValidationResult.FAILED
            
            # Validate each analyst
            for analyst_name, analyst_data in analysts.items():
                if not isinstance(analyst_data, dict):
                    errors.append(f"Analyst {analyst_name} data must be a dictionary")
                    continue
                
                # Check for required analyst attributes
                required_attrs = ['model', 'performance_metrics', 'configuration']
                missing_attrs = [attr for attr in required_attrs if attr not in analyst_data]
                if missing_attrs:
                    warnings.append(f"Analyst {analyst_name} missing attributes: {missing_attrs}")
            
            # Validate creation metrics
            metrics = data['creation_metrics']
            if not isinstance(metrics, dict):
                errors.append("Creation metrics must be a dictionary")
                return ValidationResult.FAILED
            
            details["analyst_creation_validation"] = {
                "analysts_count": len(analysts),
                "analysts_present": len(analysts) > 0,
                "metrics_present": isinstance(metrics, dict)
            }
        
        return ValidationResult.PASSED if not errors else ValidationResult.FAILED


class AnalystEnhancementValidator(ModelTrainingStepValidator):
    """Validator for analyst enhancement step."""
    
    def __init__(self):
        super().__init__("AnalystEnhancement", ValidationLevel.CRITICAL)
    
    async def _validate_step_specific(
        self, 
        data: Any, 
        context: Dict[str, Any], 
        details: Dict[str, Any],
        warnings: List[str],
        errors: List[str],
        recommendations: List[str]
    ) -> ValidationResult:
        """Validate analyst enhancement specific requirements."""
        if isinstance(data, dict):
            # Check for enhanced analysts
            if 'enhanced_analysts' not in data:
                errors.append("Missing enhanced analysts in output")
                return ValidationResult.FAILED
            
            enhanced_analysts = data['enhanced_analysts']
            if not isinstance(enhanced_analysts, dict):
                errors.append("Enhanced analysts must be a dictionary")
                return ValidationResult.FAILED
            
            if len(enhanced_analysts) == 0:
                errors.append("No enhanced analysts created")
                return ValidationResult.FAILED
            
            # Check for enhancement metrics
            if 'enhancement_metrics' not in data:
                errors.append("Missing enhancement metrics in output")
                return ValidationResult.FAILED
            
            metrics = data['enhancement_metrics']
            if not isinstance(metrics, dict):
                errors.append("Enhancement metrics must be a dictionary")
                return ValidationResult.FAILED
            
            # Validate improvement metrics
            if 'improvement_scores' in metrics:
                improvement_scores = metrics['improvement_scores']
                if not isinstance(improvement_scores, dict):
                    errors.append("Improvement scores must be a dictionary")
                else:
                    # Check for positive improvements
                    negative_improvements = [name for name, score in improvement_scores.items() if score < 0]
                    if negative_improvements:
                        warnings.append(f"Negative improvements found: {negative_improvements}")
            
            details["analyst_enhancement_validation"] = {
                "enhanced_analysts_count": len(enhanced_analysts),
                "enhancement_metrics_present": isinstance(metrics, dict),
                "improvement_scores_present": 'improvement_scores' in metrics
            }
        
        return ValidationResult.PASSED if not errors else ValidationResult.FAILED


class EnsembleCreationValidator(ModelTrainingStepValidator):
    """Validator for ensemble creation step."""
    
    def __init__(self):
        super().__init__("EnsembleCreation", ValidationLevel.CRITICAL)
    
    async def _validate_step_specific(
        self, 
        data: Any, 
        context: Dict[str, Any], 
        details: Dict[str, Any],
        warnings: List[str],
        errors: List[str],
        recommendations: List[str]
    ) -> ValidationResult:
        """Validate ensemble creation specific requirements."""
        if isinstance(data, dict):
            # Check for ensemble components
            required_components = ['ensembles', 'ensemble_metrics', 'ensemble_configs']
            missing_components = [comp for comp in required_components if comp not in data]
            if missing_components:
                errors.append(f"Missing ensemble components: {missing_components}")
                return ValidationResult.FAILED
            
            # Validate ensembles
            ensembles = data['ensembles']
            if not isinstance(ensembles, dict):
                errors.append("Ensembles must be a dictionary")
                return ValidationResult.FAILED
            
            if len(ensembles) == 0:
                errors.append("No ensembles created")
                return ValidationResult.FAILED
            
            # Validate each ensemble
            for ensemble_name, ensemble_data in ensembles.items():
                if not isinstance(ensemble_data, dict):
                    errors.append(f"Ensemble {ensemble_name} data must be a dictionary")
                    continue
                
                # Check for required ensemble attributes
                required_attrs = ['models', 'weights', 'performance_metrics']
                missing_attrs = [attr for attr in required_attrs if attr not in ensemble_data]
                if missing_attrs:
                    warnings.append(f"Ensemble {ensemble_name} missing attributes: {missing_attrs}")
                
                # Validate weights sum to 1
                if 'weights' in ensemble_data:
                    weights = ensemble_data['weights']
                    if isinstance(weights, (list, np.ndarray)):
                        weight_sum = sum(weights)
                        if abs(weight_sum - 1.0) > 0.01:  # Allow small floating point errors
                            warnings.append(f"Ensemble {ensemble_name} weights don't sum to 1: {weight_sum}")
            
            # Validate ensemble metrics
            metrics = data['ensemble_metrics']
            if not isinstance(metrics, dict):
                errors.append("Ensemble metrics must be a dictionary")
                return ValidationResult.FAILED
            
            details["ensemble_creation_validation"] = {
                "ensembles_count": len(ensembles),
                "ensembles_present": len(ensembles) > 0,
                "metrics_present": isinstance(metrics, dict)
            }
        
        return ValidationResult.PASSED if not errors else ValidationResult.FAILED


class TacticianTrainingValidator(ModelTrainingStepValidator):
    """Validator for tactician training step."""
    
    def __init__(self):
        super().__init__("TacticianTraining", ValidationLevel.CRITICAL)
    
    async def _validate_step_specific(
        self, 
        data: Any, 
        context: Dict[str, Any], 
        details: Dict[str, Any],
        warnings: List[str],
        errors: List[str],
        recommendations: List[str]
    ) -> ValidationResult:
        """Validate tactician training specific requirements."""
        if isinstance(data, dict):
            # Check for tactician components
            required_components = ['tacticians', 'training_metrics', 'tactician_configs']
            missing_components = [comp for comp in required_components if comp not in data]
            if missing_components:
                errors.append(f"Missing tactician components: {missing_components}")
                return ValidationResult.FAILED
            
            # Validate tacticians
            tacticians = data['tacticians']
            if not isinstance(tacticians, dict):
                errors.append("Tacticians must be a dictionary")
                return ValidationResult.FAILED
            
            if len(tacticians) == 0:
                errors.append("No tacticians created")
                return ValidationResult.FAILED
            
            # Validate each tactician
            for tactician_name, tactician_data in tacticians.items():
                if not isinstance(tactician_data, dict):
                    errors.append(f"Tactician {tactician_name} data must be a dictionary")
                    continue
                
                # Check for required tactician attributes
                required_attrs = ['model', 'specialization', 'performance_metrics']
                missing_attrs = [attr for attr in required_attrs if attr not in tactician_data]
                if missing_attrs:
                    warnings.append(f"Tactician {tactician_name} missing attributes: {missing_attrs}")
            
            # Validate training metrics
            metrics = data['training_metrics']
            if not isinstance(metrics, dict):
                errors.append("Training metrics must be a dictionary")
                return ValidationResult.FAILED
            
            # Check for required metrics
            required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            missing_metrics = [metric for metric in required_metrics if metric not in metrics]
            if missing_metrics:
                warnings.append(f"Missing tactician metrics: {missing_metrics}")
            
            details["tactician_training_validation"] = {
                "tacticians_count": len(tacticians),
                "tacticians_present": len(tacticians) > 0,
                "metrics_present": isinstance(metrics, dict)
            }
        
        return ValidationResult.PASSED if not errors else ValidationResult.FAILED


# Validator registry
VALIDATOR_REGISTRY = {
    'hmm_training': HMMTrainingValidator,
    'regime_intelligence': RegimeIntelligenceValidator,
    'analyst_creation': AnalystCreationValidator,
    'analyst_enhancement': AnalystEnhancementValidator,
    'ensemble_creation': EnsembleCreationValidator,
    'tactician_training': TacticianTrainingValidator,
}


def get_validator(step_name: str) -> ModelTrainingStepValidator:
    """Get validator for a specific step."""
    if step_name not in VALIDATOR_REGISTRY:
        raise ValueError(f"No validator found for step: {step_name}")
    
    return VALIDATOR_REGISTRY[step_name]()


async def validate_model_training_step(
    step_name: str, 
    data: Any, 
    context: Dict[str, Any]
) -> ValidationReport:
    """Validate a model training step."""
    validator = get_validator(step_name)
    return await validator.validate(data, context)