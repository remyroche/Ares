"""
Preprocessing module for regime discovery system.

This module now imports optimized preprocessing components from the optimization/
directory instead of maintaining legacy implementations.
"""

# Import optimized preprocessing components
from ..optimization.optimized_preprocessor import (
    OptimizedPreprocessor,
    PreprocessingConfig,
    create_optimized_preprocessor
)

from ..optimization.optimized_dimensionality_reducer import (
    OptimizedDimensionalityReducer,
    DimensionalityReductionConfig,
    create_optimized_dimensionality_reducer
)

from .temporal_window_handler import TemporalWindowHandler

# Legacy aliases for backward compatibility
FeatureProcessor = OptimizedPreprocessor
FeatureProcessingResult = dict  # Simple alias for backward compatibility
DimensionalityReducer = OptimizedDimensionalityReducer

__all__ = [
    # Optimized components
    'OptimizedPreprocessor',
    'PreprocessingConfig',
    'create_optimized_preprocessor',
    'OptimizedDimensionalityReducer',
    'DimensionalityReductionConfig',
    'create_optimized_dimensionality_reducer',
    'TemporalWindowHandler',
    
    # Legacy aliases for backward compatibility
    'FeatureProcessor',
    'FeatureProcessingResult',
    'DimensionalityReducer'
]