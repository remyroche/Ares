"""
Order Flow Feature Generators

This module provides order flow-based feature generators by importing from
the microstructure features module.
"""

# Import order flow generators from microstructure features
from .microstructure_features import (
    OrderFlowImbalanceGenerator,
    VectorBTOrderFlowImbalanceGenerator,
    VectorBTMarketOrderFlowGenerator,
    VectorBTVolumeWeightedOrderFlowGenerator,
    VectorBTOrderFlowMomentumGenerator,
    VectorBTOrderFlowVolatilityGenerator,
    VectorBTOrderFlowTrendStrengthGenerator,
    VectorBTOrderFlowConsistencyGenerator,
    VectorBTOrderFlowAccelerationGenerator,
    VectorBTOrderFlowJerkGenerator,
    VectorBTOrderFlowRegimeGenerator
)

def create_default_order_flow_generators():
    """Create default order flow generators."""
    generators = []
    
    # Add basic order flow generators
    generators.append(OrderFlowImbalanceGenerator())
    
    # Add VectorBT-optimized order flow generators
    for window in [5, 10, 20, 50]:
        generators.append(VectorBTOrderFlowImbalanceGenerator(window))
        generators.append(VectorBTMarketOrderFlowGenerator(window))
        generators.append(VectorBTVolumeWeightedOrderFlowGenerator(window))
        generators.append(VectorBTOrderFlowMomentumGenerator(window))
        generators.append(VectorBTOrderFlowVolatilityGenerator(window))
        generators.append(VectorBTOrderFlowTrendStrengthGenerator(window))
        generators.append(VectorBTOrderFlowConsistencyGenerator(window))
        generators.append(VectorBTOrderFlowAccelerationGenerator(window))
        generators.append(VectorBTOrderFlowJerkGenerator(window))
        generators.append(VectorBTOrderFlowRegimeGenerator(window))
    
    return generators

# Export all the classes and functions
__all__ = [
    'OrderFlowImbalanceGenerator',
    'VectorBTOrderFlowImbalanceGenerator',
    'VectorBTMarketOrderFlowGenerator',
    'VectorBTVolumeWeightedOrderFlowGenerator',
    'VectorBTOrderFlowMomentumGenerator',
    'VectorBTOrderFlowVolatilityGenerator',
    'VectorBTOrderFlowTrendStrengthGenerator',
    'VectorBTOrderFlowConsistencyGenerator',
    'VectorBTOrderFlowAccelerationGenerator',
    'VectorBTOrderFlowJerkGenerator',
    'VectorBTOrderFlowRegimeGenerator',
    'create_default_order_flow_generators'
]
