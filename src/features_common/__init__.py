"""
Common utilities shared between feature_generation and feature_engineering_roadmap.

This module provides base classes and shared functionality to reduce duplication
between the two feature systems.
"""

__version__ = "1.0.0"

# Import common utilities
from .utils import (
    TPRINT_AVAILABLE, tprint,
    VECTORBT_OPTIMIZER_AVAILABLE, VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer,
    UnifiedVectorizationManager, get_unified_vectorization_manager
)

# Core imports
from .transforms.base_scaler import BaseScaler, create_optimized_scaler, create_optimized_batch_scaler
from .transforms.vectorbt_scaler import VectorBTScaler, VectorBTBatchScaler
from .optimization.cv_base import BaseCVSplitter, PurgedCVSplitter
from .registry.base_registry import BaseFeatureRegistry

__all__ = [
    'BaseScaler',
    'VectorBTScaler',
    'VectorBTBatchScaler',
    'BaseCVSplitter',
    'PurgedCVSplitter',
    'BaseFeatureRegistry',
    'create_optimized_scaler',
    'create_optimized_batch_scaler',
]

# Add VectorBT optimization components to __all__ if available
if VECTORBT_OPTIMIZER_AVAILABLE:
    __all__.extend([
        'VectorBTRollingOptimizer',
        'get_vectorbt_rolling_optimizer',
        'UnifiedVectorizationManager',
        'get_unified_vectorization_manager',
    ])

if TPRINT_AVAILABLE:
    tprint(f"🔧 [features_common] Module initialized with {len(__all__)} exports", color="cyan")
