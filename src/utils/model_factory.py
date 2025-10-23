"""
Compatibility re-exports for model factory utilities.

Provides a stable import path `src.utils.model_factory` across the codebase.
"""

from src.utils.ml_common.models.model_factory import (
    EnhancedModelFactory,
    ModelType,
    ModelConfig,
    create_model_factory,
)

# Unified alias for callers that expect UnifiedModelFactory symbol
UnifiedModelFactory = EnhancedModelFactory

__all__ = [
    "EnhancedModelFactory",
    "ModelType",
    "ModelConfig",
    "create_model_factory",
    "UnifiedModelFactory",
]
