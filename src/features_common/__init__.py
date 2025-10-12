"""
Common utilities shared between feature_generation and feature_engineering_roadmap.

This module provides base classes and shared functionality to reduce duplication
between the two feature systems.
"""

__version__ = "1.0.0"

from .transforms.base_scaler import BaseScaler, create_optimized_scaler, create_optimized_batch_scaler
from .transforms.vectorbt_scaler import VectorBTScaler, VectorBTBatchScaler
from .optimization.cv_base import BaseCVSplitter, PurgedCVSplitter
from .registry.base_registry import BaseFeatureRegistry

# Import VectorBT optimization components
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    from src.feature_generation.utils.unified_vectorization_manager import UnifiedVectorizationManager, get_unified_vectorization_manager
    VECTORBT_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_OPTIMIZER_AVAILABLE = False
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None
    UnifiedVectorizationManager = None
    get_unified_vectorization_manager = None

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
