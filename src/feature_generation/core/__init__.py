"""
Feature Generation Core Module

This module provides the core functionality for feature generation,
including both legacy and optimized implementations.
"""

from .feature_generator import (
    FeatureGenerator,
    FeatureResult,
    VectorizedFeatureGenerator,
    FeatureConfig,
    FeatureCategory
)

from .vectorbt_optimization_mixin import VectorBTOptimizationMixin

# Import optimized feature generator
from .optimized_feature_generator import OptimizedFeatureGenerator

# Import all optimized category generators
from ..categories.optimized_trend import (
    OptimizedTrendFeatureGenerator,
    create_optimized_trend_generator,
    generate_trend_features_optimized
)

from ..categories.optimized_volatility import (
    OptimizedVolatilityFeatureGenerator,
    create_optimized_volatility_generator,
    generate_volatility_features_optimized
)

from ..categories.optimized_returns import (
    OptimizedReturnsFeatureGenerator,
    create_optimized_returns_generator,
    generate_returns_features_optimized
)

from ..categories.optimized_cross_category import (
    OptimizedCrossCategoryFeatureGenerator,
    create_optimized_cross_category_generator,
    generate_cross_category_features_optimized
)

# Import factory for easy switching between implementations
from .feature_generator_factory import (
    FeatureGeneratorFactory,
    GeneratorType,
    create_feature_generator,
    get_available_generators
)

__all__ = [
    # Core classes
    'FeatureGenerator',
    'FeatureResult',
    'VectorizedFeatureGenerator',
    'FeatureConfig',
    'FeatureCategory',
    'VectorBTOptimizationMixin',
    
    # Optimized base class
    'OptimizedFeatureGenerator',
    
    # Optimized category generators
    'OptimizedTrendFeatureGenerator',
    'OptimizedVolatilityFeatureGenerator',
    'OptimizedReturnsFeatureGenerator',
    'OptimizedCrossCategoryFeatureGenerator',
    
    # Convenience functions
    'create_optimized_trend_generator',
    'generate_trend_features_optimized',
    'create_optimized_volatility_generator',
    'generate_volatility_features_optimized',
    'create_optimized_returns_generator',
    'generate_returns_features_optimized',
    'create_optimized_cross_category_generator',
    'generate_cross_category_features_optimized',
    
    # Factory pattern
    'FeatureGeneratorFactory',
    'GeneratorType',
    'create_feature_generator',
    'get_available_generators'
]