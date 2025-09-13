"""
ML Common - Models Module

This module contains all model-related functionality including:
- Model factories and creation
- Multi-output models
- Model training and evaluation
- Model registry
"""

from .model_factory import (
    EnhancedModelFactory, ModelType, ModelConfig,
    create_analyst_models, create_tactician_models, create_model_factory
)
from .multi_output_models import (
    MultiOutputConfig, MultiOutputModel, MultiOutputStackingModel, MultiOutputResult,
    prepare_multi_output_targets, create_analyst_outputs, create_tactician_outputs,
    create_multi_output_stacking_model
)
from .model_training import EnhancedModelTrainer, train_model_with_confidence_metrics
from .model_evaluation import ModelEvaluationUtilities as ModelEvaluator
from .model_registry import ModelRegistry

__all__ = [
    # Model Factory
    'EnhancedModelFactory', 'ModelType', 'ModelConfig',
    'create_analyst_models', 'create_tactician_models', 'create_model_factory',
    
    # Multi-Output Models
    'MultiOutputConfig', 'MultiOutputModel', 'MultiOutputStackingModel', 'MultiOutputResult',
    'prepare_multi_output_targets', 'create_analyst_outputs', 'create_tactician_outputs',
    'create_multi_output_stacking_model',
    
    # Model Training
    'EnhancedModelTrainer', 'train_model_with_confidence_metrics',
    
    # Model Evaluation
    'ModelEvaluator',
    
    # Model Registry
    'ModelRegistry'
]