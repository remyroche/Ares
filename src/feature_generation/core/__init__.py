"""
Core Feature Generation Framework

This module provides the core framework for the unified feature generation system,
including the main classes and interfaces for feature generation, registration,
and management.
"""

from .feature_bank import FeatureBank
from .feature_generator import FeatureGenerator, FeatureCategory, VectorizedFeatureGenerator
from .feature_registry import FeatureRegistry
from .factory import (
    get_feature_generator,
    get_feature_bank,
    register_feature_generator,
    list_available_features,
    list_available_categories
)
# New utility mixins
from .optimization_mixin import OptimizationMixin
from .rolling_operations_mixin import RollingOperationsMixin
from .vectorbt_optimization_mixin import VectorBTOptimizationMixin
# Factory pattern
from .generator_factory import GeneratorFactory, get_generator_factory, create_generator

def _initialize_default_bank():
    """Initialize the default feature bank with standard generators."""
    try:
        bank = get_feature_bank()
        # Basic initialization - the bank will auto-register available generators
        return bank
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to initialize default bank: {e}")
        return None

__all__ = [
    "FeatureBank",
    "FeatureGenerator", 
    "FeatureCategory",
    "FeatureRegistry",
    "VectorizedFeatureGenerator",
    "get_feature_generator",
    "get_feature_bank",
    "register_feature_generator",
    "list_available_features",
    "list_available_categories",
    # New utility mixins
    "OptimizationMixin",
    "RollingOperationsMixin",
    "VectorBTOptimizationMixin",
    # Factory pattern
    "GeneratorFactory",
    "get_generator_factory",
    "create_generator",
    "_initialize_default_bank"
]