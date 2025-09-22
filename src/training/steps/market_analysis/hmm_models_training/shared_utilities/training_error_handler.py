"""
Centralized Training Error Handler

Provides standardized error handling for training operations across all
HMM training components.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import time
from src.utils.ml_common.monitoring.enhanced_error_detector import (
    get_global_error_detector,
)


@dataclass
class TrainingMetrics:
    """Enhanced training metrics container with additional monitoring."""
    accuracy: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    training_time: float = 0.0
    convergence_epochs: int = 0
    memory_usage_mb: float = 0.0
    validation_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class ModelResult:
    """Enhanced model result container with additional metadata."""
    model: Any
    metrics: TrainingMetrics
    feature_importance: Optional[Dict[str, float]] = None
    predictions: Optional[Any] = None
    probabilities: Optional[Any] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    training_history: Optional[Dict[str, List[float]]] = None
    overfitting_analysis: Optional[Dict[str, Any]] = None


class TrainingErrorHandler:
    """Centralized error handling for training operations."""
    
    @staticmethod
    def handle_model_creation_error(model_type: str, error: Exception) -> ModelResult:
        """
        Standardized model creation error handling with preserved stack trace.
        
        Args:
            model_type: Type of model that failed to create
            error: Exception that occurred during creation
            
        Returns:
            ModelResult with error information
        """
        import traceback
        
        # Preserve the full stack trace
        error_traceback = traceback.format_exc()
        detailed_error_message = f"Failed to create {model_type}: {str(error)}"
        
        # Record with global enhanced detector for unified monitoring/classification
        detector = get_global_error_detector()
        classification = detector.detect_and_classify_error(
            error,
            {
                'component': 'hmm_training_model_creation',
                'model_type': model_type,
            },
        )
        suggestions_msg = (
            f"Suggested actions: {', '.join(classification.suggested_actions)}"
            if getattr(classification, 'suggested_actions', None)
            else 'Suggested actions: none'
        )
        classification_msg = (
            f"Classification => severity={classification.severity.value}, "
            f"category={classification.category.value}, "
            f"confidence={classification.classification_confidence:.2f}"
        )
        
        return ModelResult(
            model=None,
            metrics=TrainingMetrics(
                error_message=detailed_error_message,
                training_time=0.0,
                warnings=[
                    f"Full traceback: {error_traceback}",
                    classification_msg,
                    suggestions_msg,
                ],
            ),
        )
    
    @staticmethod
    def handle_training_error(model_type: str, error: Exception, training_time: float) -> ModelResult:
        """
        Standardized training error handling with preserved stack trace.
        
        Args:
            model_type: Type of model that failed to train
            error: Exception that occurred during training
            training_time: Time spent before failure
            
        Returns:
            ModelResult with error information
        """
        import traceback
        
        # Preserve the full stack trace
        error_traceback = traceback.format_exc()
        detailed_error_message = f"Failed to train {model_type}: {str(error)}"
        
        # Record with global enhanced detector for unified monitoring/classification
        detector = get_global_error_detector()
        classification = detector.detect_and_classify_error(
            error,
            {
                'component': 'hmm_training',
                'model_type': model_type,
                'execution_time': training_time,
            },
        )
        suggestions_msg = (
            f"Suggested actions: {', '.join(classification.suggested_actions)}"
            if getattr(classification, 'suggested_actions', None)
            else 'Suggested actions: none'
        )
        classification_msg = (
            f"Classification => severity={classification.severity.value}, "
            f"category={classification.category.value}, "
            f"confidence={classification.classification_confidence:.2f}"
        )
        
        return ModelResult(
            model=None,
            metrics=TrainingMetrics(
                error_message=detailed_error_message,
                training_time=training_time,
                warnings=[
                    f"Full traceback: {error_traceback}",
                    classification_msg,
                    suggestions_msg,
                ],
            ),
        )
    
    @staticmethod
    def handle_validation_error(validation_type: str, error: Exception) -> ModelResult:
        """
        Standardized validation error handling.
        
        Args:
            validation_type: Type of validation that failed
            error: Exception that occurred during validation
            
        Returns:
            ModelResult with error information
        """
        # Record with global enhanced detector for unified monitoring/classification
        detector = get_global_error_detector()
        classification = detector.detect_and_classify_error(
            error,
            {
                'component': 'hmm_training_validation',
                'model_type': validation_type,
            },
        )
        classification_msg = (
            f"Classification => severity={classification.severity.value}, "
            f"category={classification.category.value}, "
            f"confidence={classification.classification_confidence:.2f}"
        )
        suggestions_msg = (
            f"Suggested actions: {', '.join(classification.suggested_actions)}"
            if getattr(classification, 'suggested_actions', None)
            else 'Suggested actions: none'
        )
        
        return ModelResult(
            model=None,
            metrics=TrainingMetrics(
                error_message=f"Validation failed for {validation_type}: {str(error)}",
                training_time=0.0,
                warnings=[classification_msg, suggestions_msg],
            ),
        )
    
    @staticmethod
    def handle_feature_selection_error(error: Exception, fallback_features: List[str]) -> ModelResult:
        """
        Standardized feature selection error handling.
        
        Args:
            error: Exception that occurred during feature selection
            fallback_features: Features to use as fallback
            
        Returns:
            ModelResult with error information
        """
        # Record with global enhanced detector for unified monitoring/classification
        detector = get_global_error_detector()
        classification = detector.detect_and_classify_error(
            error,
            {
                'component': 'hmm_feature_selection',
                'model_type': 'feature_selection',
            },
        )
        classification_msg = (
            f"Classification => severity={classification.severity.value}, "
            f"category={classification.category.value}, "
            f"confidence={classification.classification_confidence:.2f}"
        )
        suggestions_msg = (
            f"Suggested actions: {', '.join(classification.suggested_actions)}"
            if getattr(classification, 'suggested_actions', None)
            else 'Suggested actions: none'
        )
        
        return ModelResult(
            model=None,
            metrics=TrainingMetrics(
                error_message=f"Feature selection failed: {str(error)}. Using fallback features: {len(fallback_features)}",
                training_time=0.0,
                warnings=[classification_msg, suggestions_msg],
            ),
        )