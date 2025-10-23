"""
Acceleration Feature Generators

This module provides acceleration-based feature generators by importing from
the VectorBT-optimized acceleration module.
"""

# Import all acceleration generators from the VectorBT module
from .vectorbt_acceleration import (
    AccelerationFeatureGenerator,
    VectorBTMomentumGenerator,
    VectorBTPriceAccelerationGenerator,
    VectorBTPriceJerkGenerator,
    VectorBTTrendStrengthGenerator,
    VectorBTTrendConsistencyGenerator,
    VectorBTVolumeAccelerationGenerator,
    VectorBTVolatilityAccelerationGenerator,
    VectorBTMomentumAccelerationGenerator,
    VectorBTAccelerationMomentumGenerator,
    VectorBTAccelerationVolatilityGenerator,
    VectorBTAccelerationTrendStrengthGenerator,
    VectorBTAccelerationConsistencyGenerator,
    VectorBTAccelerationRegimeGenerator,
    VectorBTMultiTimeframeAccelerationGenerator,
    VectorBTAccelerationCorrelationGenerator,
    VectorBTAccelerationDivergenceGenerator,
    OptimizedAccelerationBatchGenerator,
    create_vectorbt_acceleration_generators,
    create_default_vectorbt_acceleration_generators,
    create_optimized_acceleration_batch_generator,
    create_acceleration_generators,
    create_default_acceleration_generators
)

# Create aliases for backward compatibility
MomentumGenerator = VectorBTMomentumGenerator
PriceAccelerationGenerator = VectorBTPriceAccelerationGenerator
PriceJerkGenerator = VectorBTPriceJerkGenerator
TrendStrengthGenerator = VectorBTTrendStrengthGenerator
TrendConsistencyGenerator = VectorBTTrendConsistencyGenerator
VolumeAccelerationGenerator = VectorBTVolumeAccelerationGenerator
VolatilityAccelerationGenerator = VectorBTVolatilityAccelerationGenerator
MomentumAccelerationGenerator = VectorBTMomentumAccelerationGenerator
AccelerationMomentumGenerator = VectorBTAccelerationMomentumGenerator
AccelerationVolatilityGenerator = VectorBTAccelerationVolatilityGenerator
AccelerationTrendStrengthGenerator = VectorBTAccelerationTrendStrengthGenerator
AccelerationConsistencyGenerator = VectorBTAccelerationConsistencyGenerator
AccelerationRegimeGenerator = VectorBTAccelerationRegimeGenerator
MultiTimeframeAccelerationGenerator = VectorBTMultiTimeframeAccelerationGenerator
AccelerationCorrelationGenerator = VectorBTAccelerationCorrelationGenerator
AccelerationDivergenceGenerator = VectorBTAccelerationDivergenceGenerator

# Re-export the main functions that the feature bank expects
def create_default_acceleration_generators():
    """Create default acceleration generators."""
    return create_default_acceleration_generators()

# Export all the classes and functions
__all__ = [
    'AccelerationFeatureGenerator',
    'VectorBTMomentumGenerator',
    'VectorBTPriceAccelerationGenerator',
    'VectorBTPriceJerkGenerator',
    'VectorBTTrendStrengthGenerator',
    'VectorBTTrendConsistencyGenerator',
    'VectorBTVolumeAccelerationGenerator',
    'VectorBTVolatilityAccelerationGenerator',
    'VectorBTMomentumAccelerationGenerator',
    'VectorBTAccelerationMomentumGenerator',
    'VectorBTAccelerationVolatilityGenerator',
    'VectorBTAccelerationTrendStrengthGenerator',
    'VectorBTAccelerationConsistencyGenerator',
    'VectorBTAccelerationRegimeGenerator',
    'VectorBTMultiTimeframeAccelerationGenerator',
    'VectorBTAccelerationCorrelationGenerator',
    'VectorBTAccelerationDivergenceGenerator',
    'OptimizedAccelerationBatchGenerator',
    'MomentumGenerator',
    'PriceAccelerationGenerator',
    'PriceJerkGenerator',
    'TrendStrengthGenerator',
    'TrendConsistencyGenerator',
    'VolumeAccelerationGenerator',
    'VolatilityAccelerationGenerator',
    'MomentumAccelerationGenerator',
    'AccelerationMomentumGenerator',
    'AccelerationVolatilityGenerator',
    'AccelerationTrendStrengthGenerator',
    'AccelerationConsistencyGenerator',
    'AccelerationRegimeGenerator',
    'MultiTimeframeAccelerationGenerator',
    'AccelerationCorrelationGenerator',
    'AccelerationDivergenceGenerator',
    'create_vectorbt_acceleration_generators',
    'create_default_vectorbt_acceleration_generators',
    'create_optimized_acceleration_batch_generator',
    'create_acceleration_generators',
    'create_default_acceleration_generators'
]
