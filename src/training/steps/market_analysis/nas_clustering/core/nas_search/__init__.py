"""
Neural Architecture Search (NAS) Core Module

This module provides true Neural Architecture Search capabilities for regime detection,
including evolutionary algorithms, reinforcement learning-based search, and architecture
generation for financial time series analysis.

Key Features:
- Evolutionary architecture search with genetic algorithms
- Reinforcement learning-based architecture optimization
- Multi-objective optimization for regime detection
- Hardware-optimized architecture evaluation
- Integration with existing matrix operations and hardware optimization
"""

from .evolutionary_search import (
    EvolutionaryArchitectureSearch,
    ArchitecturePopulation,
    GeneticAlgorithm,
    ArchitectureIndividual,
    FitnessEvaluator
)

from .reinforcement_search import (
    ReinforcementLearningSearch,
    ArchitectureController,
    PolicyNetwork,
    RewardCalculator,
    ExperienceReplay
)

from .architecture_generator import (
    ArchitectureGenerator,
    LayerGenerator,
    ConnectionGenerator,
    ArchitectureValidator
)

from .search_space import (
    SearchSpace,
    LayerType,
    ActivationFunction,
    ConnectionType,
    ArchitectureConstraints
)

__all__ = [
    # Evolutionary search
    'EvolutionaryArchitectureSearch',
    'ArchitecturePopulation',
    'GeneticAlgorithm',
    'ArchitectureIndividual',
    'FitnessEvaluator',
    
    # Reinforcement learning search
    'ReinforcementLearningSearch',
    'ArchitectureController',
    'PolicyNetwork',
    'RewardCalculator',
    'ExperienceReplay',
    
    # Architecture generation
    'ArchitectureGenerator',
    'LayerGenerator',
    'ConnectionGenerator',
    'ArchitectureValidator',
    
    # Search space
    'SearchSpace',
    'LayerType',
    'ActivationFunction',
    'ConnectionType',
    'ArchitectureConstraints'
]