"""
Essential Neural Architecture Search (NAS) Module

This module provides essential Neural Architecture Search capabilities,
focusing on core NAS components for dynamic architecture discovery.

Key Features:
- Evolutionary architecture search with genetic algorithms
- Essential search space definition
- Multi-objective optimization
- Architecture evaluation and validation
"""

from .evolutionary_search import (
    EvolutionaryArchitectureSearch,
    ArchitecturePopulation,
    GeneticAlgorithm,
    ArchitectureIndividual,
    FitnessEvaluator,
    RegimeDetectionFitnessEvaluator
)

from .search_space import (
    SearchSpace,
    LayerType,
    ActivationFunction,
    ConnectionType,
    ArchitectureConstraints,
    LayerConfig,
    ConnectionConfig,
    get_default_search_space
)

__all__ = [
    # Essential evolutionary search
    'EvolutionaryArchitectureSearch',
    'ArchitecturePopulation',
    'GeneticAlgorithm',
    'ArchitectureIndividual',
    'FitnessEvaluator',
    'RegimeDetectionFitnessEvaluator',
    
    # Essential search space
    'SearchSpace',
    'LayerType',
    'ActivationFunction',
    'ConnectionType',
    'ArchitectureConstraints',
    'LayerConfig',
    'ConnectionConfig',
    'get_default_search_space'
]