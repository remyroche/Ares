"""
Test Suite for Enhanced NAS Clustering

This module provides comprehensive testing and benchmarking for the enhanced NAS clustering
system with true Neural Architecture Search capabilities.
"""

from .test_enhanced_nas_clusterer import (
    TestEnhancedNASClusterer,
    TestTrueNASIntegration,
    TestMultiObjectiveOptimization
)

from .test_evolutionary_search import (
    TestEvolutionaryArchitectureSearch,
    TestArchitecturePopulation,
    TestGeneticAlgorithm
)

from .test_regime_networks import (
    TestVolatilityRegimeNetwork,
    TestTrendRegimeNetwork,
    TestVolumeRegimeNetwork,
    TestHybridRegimeNetwork
)

from .test_multi_objective import (
    TestParetoFrontier,
    TestNSGAIIOptimizer,
    TestWeightedSumOptimizer
)

from .test_integration import (
    TestNASIntegration,
    TestPipelineCompatibility,
    TestPerformanceBenchmarks
)

__all__ = [
    # Enhanced NAS clusterer tests
    'TestEnhancedNASClusterer',
    'TestTrueNASIntegration',
    'TestMultiObjectiveOptimization',
    
    # Evolutionary search tests
    'TestEvolutionaryArchitectureSearch',
    'TestArchitecturePopulation',
    'TestGeneticAlgorithm',
    
    # Regime network tests
    'TestVolatilityRegimeNetwork',
    'TestTrendRegimeNetwork',
    'TestVolumeRegimeNetwork',
    'TestHybridRegimeNetwork',
    
    # Multi-objective optimization tests
    'TestParetoFrontier',
    'TestNSGAIIOptimizer',
    'TestWeightedSumOptimizer',
    
    # Integration tests
    'TestNASIntegration',
    'TestPipelineCompatibility',
    'TestPerformanceBenchmarks'
]