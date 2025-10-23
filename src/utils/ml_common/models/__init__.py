"""
ML Common Models Module
"""

from .model_factory import (
    EnhancedModelFactory, ModelType, ModelConfig,
    create_model_factory
)

# Lazy import for EnhancedModelTrainer to avoid circular dependencies
def get_enhanced_model_trainer():
    """Lazy import for EnhancedModelTrainer to avoid circular dependencies."""
    from .enhanced_model_trainer import EnhancedModelTrainer
    return EnhancedModelTrainer

from .model_training import (
    train_model_with_confidence_metrics
)

from ..post_training.model_evaluation import (
    ModelEvaluator
)

from .model_registry import (
    ModelRegistry
)

from .multi_output_models import (
    MultiOutputConfig, MultiOutputModel, MultiOutputStackingModel, MultiOutputResult,
    prepare_multi_output_targets, create_analyst_outputs, create_tactician_outputs,
    create_multi_output_stacking_model
)
