"""
Factory system for component creation.

This module provides intelligent factory classes for creating
optimized components with automatic configuration and optimization.
"""

from .scaler_factory import ScalerFactory, create_optimized_scaler, create_batch_scaler
from .optimizer_factory import OptimizerFactory, create_optimizer, create_vectorbt_optimizer
from .registry_factory import RegistryFactory, create_registry, create_feature_registry
from .unified_factory import UnifiedFactory, create_optimized_component

__all__ = [
    'ScalerFactory',
    'create_optimized_scaler',
    'create_batch_scaler',
    'OptimizerFactory',
    'create_optimizer',
    'create_vectorbt_optimizer',
    'RegistryFactory',
    'create_registry',
    'create_feature_registry',
    'UnifiedFactory',
    'create_optimized_component'
]