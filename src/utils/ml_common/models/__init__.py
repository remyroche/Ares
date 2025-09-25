"""Aggregated exports for ``src.utils.ml_common.models``."""

from .model_factory import (
    EnhancedModelFactory,
    ModelType,
    ModelConfig,
    create_model_factory,
)
from .enhanced_model_trainer import (
    EnhancedModelTrainer,
    train_model_with_confidence_metrics,
)
from .model_evaluator import ModelEvaluator, EvaluationResult
from .model_registry import ModelRegistry
from .multi_output_models import (
    MultiOutputConfig,
    MultiOutputModel,
    MultiOutputStackingModel,
    MultiOutputResult,
    prepare_multi_output_targets,
    create_analyst_outputs,
    create_tactician_outputs,
    create_multi_output_stacking_model,
)

__all__ = [
    "EnhancedModelFactory",
    "ModelType",
    "ModelConfig",
    "create_model_factory",
    "EnhancedModelTrainer",
    "train_model_with_confidence_metrics",
    "ModelEvaluator",
    "EvaluationResult",
    "ModelRegistry",
    "MultiOutputConfig",
    "MultiOutputModel",
    "MultiOutputStackingModel",
    "MultiOutputResult",
    "prepare_multi_output_targets",
    "create_analyst_outputs",
    "create_tactician_outputs",
    "create_multi_output_stacking_model",
]