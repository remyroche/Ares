"""
Base Configuration for NAS-TAS Architecture Search

This module defines the base configuration classes for unified NAS-TAS optimization.
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


class ArchitectureType(Enum):
    """Types of neural architectures for NAS-TAS."""
    HYBRID = "hybrid"
    TREE_BASED = "tree_based"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    HYBRID_NEURAL_TREE = "hybrid_neural_tree"
    NEURAL_ONLY = "neural_only"
    TREE_ONLY = "tree_only"


class SearchStrategy(Enum):
    """Search strategies for architecture optimization."""
    EVOLUTIONARY = "evolutionary"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GRID_SEARCH = "grid_search"


class OptimizationMode(Enum):
    """Optimization modes for architecture search."""
    STANDARD = "standard"
    REGIME_AWARE = "regime_aware"
    ECONOMIC_FOCUSED = "economic_focused"
    TRADING_OPTIMIZED = "trading_optimized"
    HYBRID = "hybrid"


@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithms."""

    # Search space parameters
    search_strategy: SearchStrategy = SearchStrategy.EVOLUTIONARY
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7

    # Evaluation parameters
    evaluation_metric: str = "accuracy"
    cv_folds: int = 5
    test_size: float = 0.2

    # Hardware parameters
    use_gpu: bool = True
    parallel_processing: bool = True
    max_workers: int = 4

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.001

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'search_strategy': self.search_strategy.value,
            'population_size': self.population_size,
            'generations': self.generations,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'evaluation_metric': self.evaluation_metric,
            'cv_folds': self.cv_folds,
            'test_size': self.test_size,
            'use_gpu': self.use_gpu,
            'parallel_processing': self.parallel_processing,
            'max_workers': self.max_workers,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_threshold': self.early_stopping_threshold
        }


@dataclass
class UnifiedArchitectureConfig:
    """Unified configuration for NAS-TAS architecture search."""

    # Architecture parameters
    architecture_type: ArchitectureType = ArchitectureType.HYBRID
    primary_architecture: ArchitectureType = ArchitectureType.TREE_BASED
    secondary_architecture: ArchitectureType = ArchitectureType.NEURAL_NETWORK

    # Search parameters
    search_strategy: SearchStrategy = SearchStrategy.EVOLUTIONARY
    population_size: int = 50
    generations: int = 100

    # Neural network parameters
    enable_neural_odes: bool = True
    enable_vision_transformers: bool = True
    enable_meta_learning: bool = True

    # Tree parameters
    enable_tree_ensembles: bool = True
    enable_gradient_boosting: bool = True
    enable_random_forest: bool = True

    # Integration parameters
    enable_hybrid_integration: bool = True
    enable_cross_validation: bool = True
    enable_early_stopping: bool = True

    # Performance parameters
    target_accuracy: float = 0.8
    target_efficiency: float = 0.7
    target_robustness: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'architecture_type': self.architecture_type.value,
            'primary_architecture': self.primary_architecture.value,
            'secondary_architecture': self.secondary_architecture.value,
            'search_strategy': self.search_strategy.value,
            'population_size': self.population_size,
            'generations': self.generations,
            'enable_neural_odes': self.enable_neural_odes,
            'enable_vision_transformers': self.enable_vision_transformers,
            'enable_meta_learning': self.enable_meta_learning,
            'enable_tree_ensembles': self.enable_tree_ensembles,
            'enable_gradient_boosting': self.enable_gradient_boosting,
            'enable_random_forest': self.enable_random_forest,
            'enable_hybrid_integration': self.enable_hybrid_integration,
            'enable_cross_validation': self.enable_cross_validation,
            'enable_early_stopping': self.enable_early_stopping,
            'target_accuracy': self.target_accuracy,
            'target_efficiency': self.target_efficiency,
            'target_robustness': self.target_robustness
        }
