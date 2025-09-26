"""
Unified Search Engine for NAS and TAS

This module provides a unified interface for all search strategies used by both
Neural Architecture Search (NAS) and Tree Architecture Search (TAS) systems.
It consolidates Bayesian optimization, evolutionary algorithms, reinforcement learning,
and hybrid search strategies into a single, extensible framework.

Key Features:
- Unified interface for all search algorithms
- Architecture-agnostic search strategies
- Advanced optimization techniques (NSGA-II, SPEA2, Bayesian TPE)
- Hardware optimization and parallel processing
- Comprehensive logging and monitoring
- Extensible plugin architecture for new search strategies
"""

import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Protocol
from typing import runtime_checkable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import asyncio
from abc import ABC, abstractmethod
import json
import pickle
from pathlib import Path
import warnings
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from collections import OrderedDict, deque
import threading
from math import isfinite

# Import unified utilities lazily when needed to avoid circular dependencies
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking helper only
    from src.utils.nas_tas.shared_utils.common_operations_bridge import (
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    )
else:
    get_m1_gpu_manager = None  # type: ignore
    get_m1_memory_optimizer = None  # type: ignore
    get_m1_cpu_optimizer = None  # type: ignore


def _load_m1_gpu_manager():
    global get_m1_gpu_manager
    if get_m1_gpu_manager is None:  # pragma: no branch - lazy load
        from src.utils.nas_tas.shared_utils.common_operations_bridge import get_m1_gpu_manager as _impl

        get_m1_gpu_manager = _impl
    return get_m1_gpu_manager  # type: ignore[return-value]


def _load_m1_memory_optimizer():
    global get_m1_memory_optimizer
    if get_m1_memory_optimizer is None:  # pragma: no branch - lazy load
        from src.utils.nas_tas.shared_utils.common_operations_bridge import get_m1_memory_optimizer as _impl

        get_m1_memory_optimizer = _impl
    return get_m1_memory_optimizer  # type: ignore[return-value]


def _load_m1_cpu_optimizer():
    global get_m1_cpu_optimizer
    if get_m1_cpu_optimizer is None:  # pragma: no branch - lazy load
        from src.utils.nas_tas.shared_utils.common_operations_bridge import get_m1_cpu_optimizer as _impl

        get_m1_cpu_optimizer = _impl
    return get_m1_cpu_optimizer  # type: ignore[return-value]


_BAYESIAN_DEPENDENCIES: Dict[str, Any] = {}


def _get_bayesian_dependencies() -> Dict[str, Any]:
    """Lazily import Bayesian optimizer dependencies to avoid circular imports."""
    if not _BAYESIAN_DEPENDENCIES:
        try:
            from .bayesian_tpe_optimizer import BayesianTPEOptimizer, BayesianTPEConfig

            _BAYESIAN_DEPENDENCIES.update(
                {
                    'optimizer_cls': BayesianTPEOptimizer,
                    'config_cls': BayesianTPEConfig,
                }
            )
        except ImportError as exc:
            _BAYESIAN_DEPENDENCIES.update(
                {
                    'optimizer_cls': None,
                    'config_cls': None,
                    'error': exc,
                }
            )
    return _BAYESIAN_DEPENDENCIES


def _bayesian_available() -> bool:
    deps = _get_bayesian_dependencies()
    return bool(deps.get('optimizer_cls'))

from src.utils.nas_tas.shared_utils.math_validation_bridge import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, safe_correlation,
    safe_covariance, safe_mean, safe_std, safe_percentile,
    validate_correlation_matrix, safe_matrix_inverse, math_safe,
    MathValidation, MathValidationError
)

from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

from .search_strategies import (
    Candidate as StrategyCandidate,
    Evaluation as StrategyEvaluation,
    SearchState as StrategyState,
    StrategyRegistry,
    RandomSearchStrategy as PluginRandomStrategy,
    GridSearchStrategy as PluginGridStrategy,
    OptunaSearchStrategy as PluginOptunaStrategy,
    HyperbandSearchStrategy as PluginHyperbandStrategy,
)
from .shared_utils.meta_warmstart import MetaWarmStartConfig, MetaWarmStarter
from .shared_utils.uncertainty_surrogate import (
    BayesianEnsembleConfig,
    BayesianEnsembleSurrogate,
)
from .shared_utils.time_series_cv import BlockedPurgedCV, BlockedPurgedCVConfig
from .shared_utils.hardware_costs import (
    HardwareConstraintConfig,
    HardwareCostEvaluator,
)
from .shared_utils.overfitting_tests import hansen_spa_test

logger = logging.getLogger(__name__)


def _warning_to_log(message, category, filename, lineno, file=None, line=None):
    """Redirect warnings to the module logger instead of silencing them."""
    logger.warning(
        "%s:%s: %s: %s",
        Path(filename).name,
        lineno,
        category.__name__,
        message,
    )


warnings.showwarning = _warning_to_log
warnings.filterwarnings("default")


class ArchitectureType(Enum):
    """Types of architectures supported by the unified search engine."""
    NEURAL = "neural"
    TREE = "tree"
    HYBRID = "hybrid"


class SearchStrategy(Enum):
    """Available search strategies."""
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    RANDOM = "random"
    HYBRID = "hybrid"
    ENHANCED_BAYESIAN = "enhanced_bayesian"
    ADAPTIVE_EVOLUTIONARY = "adaptive_evolutionary"
    NSGA2 = "nsga2"
    SPEA2 = "spea2"


class OptimizationObjective(Enum):
    """Types of optimization objectives."""
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    STABILITY = "stability"
    PROFITABILITY = "profitability"
    ECONOMIC_SIGNIFICANCE = "economic_significance"
    TRADING_VIABILITY = "trading_viability"
    COMPLEXITY = "complexity"
    MEMORY_USAGE = "memory_usage"
    SHARPE_RATIO = "sharpe_ratio"
    DOWNSIDE_DEVIATION = "downside_deviation"
    EXECUTION_LATENCY = "execution_latency"
    TAIL_LATENCY = "tail_latency"
    COLD_START_LATENCY = "cold_start_latency"


@dataclass
class SearchConfig:
    """Unified configuration for search strategies."""

    # Core search parameters
    architecture_type: ArchitectureType = ArchitectureType.HYBRID
    search_strategy: SearchStrategy = SearchStrategy.ENHANCED_BAYESIAN
    max_iterations: int = 100
    population_size: int = 50
    elite_size: int = 5
    max_candidates_per_batch: int = 1
    random_seed: Optional[int] = None
    cache_results: bool = True

    # Optimization objectives
    objectives: List[OptimizationObjective] = field(default_factory=lambda: [
        OptimizationObjective.SHARPE_RATIO,
        OptimizationObjective.DOWNSIDE_DEVIATION,
        OptimizationObjective.EXECUTION_LATENCY,
    ])
    objective_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])
    objective_names: List[str] = field(default_factory=list)
    objective_directions: List[str] = field(default_factory=list)

    # Search strategy specific parameters
    bayesian_config: Dict[str, Any] = field(default_factory=lambda: {
        'n_initial_points': 10,
        'acquisition_function': 'ei',
        'random_state': 42
    })
    
    evolutionary_config: Dict[str, Any] = field(default_factory=lambda: {
        'mutation_rate': 0.1,
        'crossover_rate': 0.8,
        'selection_method': 'tournament',
        'tournament_size': 3
    })
    
    rl_config: Dict[str, Any] = field(default_factory=lambda: {
        'learning_rate': 0.01,
        'exploration_rate': 0.1,
        'reward_decay': 0.9,
        'episode_length': 100
    })
    
    # Hardware optimization
    enable_parallel_processing: bool = True
    n_jobs: int = -1
    enable_gpu_acceleration: bool = True
    memory_limit_gb: Optional[float] = None
    
    # Monitoring and logging
    enable_logging: bool = True
    log_level: str = 'INFO'
    save_intermediate_results: bool = True
    checkpoint_frequency: int = 10
    
    # Advanced features
    enable_early_stopping: bool = True
    early_stopping_patience: int = 20
    convergence_threshold: float = 0.01
    enable_adaptive_parameters: bool = True
    enable_meta_warm_start: bool = True
    meta_warm_start_config: Dict[str, Any] = field(default_factory=dict)
    meta_warm_start_blueprints: List[Dict[str, Any]] = field(default_factory=list)
    enable_progressive_search: bool = True
    progressive_search_stages: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"max_depth": 2, "max_width": 32},
        {"max_depth": 4, "max_width": 64},
        {"max_depth": 8, "max_width": 128},
    ])
    progressive_stability_window: int = 5
    progressive_stability_tolerance: float = 0.01
    enable_uncertainty_guided_promotion: bool = True
    uncertainty_bonus: float = 1.0
    enable_time_series_cv: bool = True
    time_series_cv_config: Dict[str, Any] = field(default_factory=dict)
    enable_overfitting_defense: bool = True
    spa_bootstrap_iterations: int = 500
    spa_transaction_cost_bp: float = 1.0
    enable_hardware_constraints: bool = True
    hardware_latency_budget_ms: float = 5.0
    hardware_tail_latency_ms: float = 10.0
    hardware_cold_start_ms: float = 60.0
    hardware_memory_budget_mb: float = 4096.0

    # Constraint handling
    enable_constraint_validation: bool = True
    max_parameter_count: Optional[int] = None
    max_flops: Optional[float] = None

    def __post_init__(self) -> None:
        self._validate_core_parameters()
        self._validate_objectives()
        self._validate_strategy_configs()

        if not self.objective_names:
            self.objective_names = [
                obj.value if hasattr(obj, 'value') else str(obj)
                for obj in self.objectives
            ]
        if not self.objective_directions:
            self.objective_directions = [
                'maximize'
                if objective in {
                    OptimizationObjective.ACCURACY,
                    OptimizationObjective.EFFICIENCY,
                    OptimizationObjective.PROFITABILITY,
                    OptimizationObjective.ECONOMIC_SIGNIFICANCE,
                    OptimizationObjective.TRADING_VIABILITY,
                    OptimizationObjective.SHARPE_RATIO,
                }
                else 'minimize'
                for objective in self.objectives
            ]

        if len(self.objective_directions) != len(self.objective_names):
            raise ValueError("Objective directions must match the number of objective names")

        if len(set(self.objective_names)) != len(self.objective_names):
            raise ValueError("Objective names must be unique")

        allowed_directions = {'maximize', 'minimize'}
        invalid = [direction for direction in self.objective_directions if direction not in allowed_directions]
        if invalid:
            raise ValueError(f"Objective directions must be one of {allowed_directions}; invalid values: {invalid}")

        # Normalize weights if they sum to approximately 1 and ensure positivity
        if self.objective_weights:
            if len(self.objective_weights) != len(self.objectives):
                raise ValueError("Objective weights must match the number of objectives")
            if any(weight <= 0 for weight in self.objective_weights):
                raise ValueError("Objective weights must be strictly positive")
            weight_sum = sum(self.objective_weights)
            if not 0.99 <= weight_sum <= 1.01:
                raise ValueError("Objective weights must sum to 1.0 within tolerance")

    def _validate_core_parameters(self) -> None:
        numeric_checks = {
            'max_iterations': self.max_iterations,
            'population_size': self.population_size,
            'elite_size': self.elite_size,
            'max_candidates_per_batch': self.max_candidates_per_batch,
        }

        for name, value in numeric_checks.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if self.random_seed is not None and (not isinstance(self.random_seed, int) or self.random_seed < 0):
            raise ValueError("random_seed must be a non-negative integer when provided")

        if self.memory_limit_gb is not None:
            if not isinstance(self.memory_limit_gb, (int, float)) or self.memory_limit_gb <= 0:
                raise ValueError("memory_limit_gb must be positive when provided")

        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero; use -1 for all cores or a positive integer")

        if self.max_candidates_per_batch > self.population_size:
            raise ValueError("max_candidates_per_batch cannot exceed population_size")

    def _validate_objectives(self) -> None:
        if not self.objectives:
            raise ValueError("At least one optimization objective must be provided")

        for objective in self.objectives:
            if not isinstance(objective, OptimizationObjective):
                raise TypeError("Objectives must be instances of OptimizationObjective")

    def _validate_strategy_configs(self) -> None:
        def _validate_positive(config: Dict[str, Any], keys: List[str]) -> None:
            for key in keys:
                value = config.get(key)
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    if value <= 0 or not isfinite(float(value)):
                        raise ValueError(f"{key} must be positive and finite")
                    if key.endswith('rate') and value > 1:
                        raise ValueError(f"{key} must be between 0 and 1")
                    if key == 'reward_decay' and not 0 < value <= 1:
                        raise ValueError("reward_decay must be between 0 and 1")
                    if key.startswith('n_') and not float(value).is_integer():
                        raise ValueError(f"{key} must be an integer value")
                else:
                    raise TypeError(f"{key} must be numeric")

        _validate_positive(self.bayesian_config, ['n_initial_points'])
        _validate_positive(self.evolutionary_config, ['mutation_rate', 'crossover_rate'])
        _validate_positive(self.rl_config, ['learning_rate', 'exploration_rate', 'reward_decay'])


@dataclass
class SearchResult:
    """Result from unified search."""
    
    # Core results
    best_architecture: Dict[str, Any]
    best_scores: Dict[OptimizationObjective, float]
    pareto_frontier: List[Dict[str, Any]]
    
    # Search metadata
    search_strategy: SearchStrategy
    architecture_type: ArchitectureType
    total_iterations: int
    execution_time: float
    
    # Performance metrics
    convergence_achieved: bool
    optimization_history: List[Dict[str, float]]
    search_statistics: Dict[str, Any]
    
    # Advanced metrics
    diversity_scores: List[float]
    exploration_exploitation_ratio: float
    hardware_utilization: Dict[str, float]
    
    # Metadata
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    success: bool = True
    error_message: Optional[str] = None


@runtime_checkable
class SearchStrategyInterface(Protocol):
    """Protocol for search strategy implementations."""
    
    def search(self, 
               search_space: Dict[str, Any], 
               objective_function: Callable,
               config: SearchConfig) -> SearchResult:
        """Perform search using this strategy."""
        ...


class BayesianSearchStrategy:
    """Bayesian optimization search strategy."""

    def __init__(
        self,
        dependency_resolver: Callable[[], Dict[str, Any]] = _get_bayesian_dependencies,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.optimizer = None
        self._dependency_resolver = dependency_resolver
        self._dependency_warning_emitted = False

    def search(self,
               search_space: Dict[str, Any],
               objective_function: Callable,
               config: SearchConfig) -> SearchResult:
        """Perform Bayesian optimization search."""
        try:
            tprint_info("🔍 Starting Bayesian optimization search...")
            start_time = time.time()
            
            # Initialize Bayesian optimizer lazily to avoid circular imports
            dependencies = self._dependency_resolver()
            optimizer_cls = dependencies.get('optimizer_cls')
            config_cls = dependencies.get('config_cls')

            if optimizer_cls and config_cls:
                bayesian_config = config_cls(**config.bayesian_config)
                self.optimizer = optimizer_cls(bayesian_config)
            else:
                if dependencies.get('error') and not self._dependency_warning_emitted:
                    warnings.warn(
                        "Bayesian optimizer dependencies unavailable; falling back to random search",
                        RuntimeWarning,
                    )
                    self.logger.warning(
                        "Bayesian optimizer unavailable, using fallback: %s",
                        dependencies.get('error'),
                    )
                    self._dependency_warning_emitted = True
                # Fallback implementation
                self.optimizer = self._create_fallback_optimizer(config)
            
            # Perform optimization
            optimization_result = self.optimizer.optimize(
                objective_function=objective_function,
                search_space=search_space,
                max_iterations=config.max_iterations
            )
            
            # Create result
            result = SearchResult(
                best_architecture=optimization_result.best_parameters,
                best_scores={obj: optimization_result.best_score for obj in config.objectives},
                pareto_frontier=[optimization_result.best_parameters],
                search_strategy=SearchStrategy.BAYESIAN,
                architecture_type=config.architecture_type,
                total_iterations=config.max_iterations,
                execution_time=time.time() - start_time,
                convergence_achieved=optimization_result.convergence_achieved,
                optimization_history=optimization_result.optimization_history,
                search_statistics={'method': 'bayesian_optimization'},
                diversity_scores=[1.0],
                exploration_exploitation_ratio=0.5,
                hardware_utilization={'cpu': 1.0, 'memory': 0.8}
            )
            
            tprint_success(f"✅ Bayesian search completed in {result.execution_time:.2f}s")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Bayesian search failed: {e}")
            return SearchResult(
                best_architecture={},
                best_scores={},
                pareto_frontier=[],
                search_strategy=SearchStrategy.BAYESIAN,
                architecture_type=config.architecture_type,
                total_iterations=0,
                execution_time=0.0,
                convergence_achieved=False,
                optimization_history=[],
                search_statistics={},
                diversity_scores=[],
                exploration_exploitation_ratio=0.0,
                hardware_utilization={},
                success=False,
                error_message=str(e)
            )
    
    def _create_fallback_optimizer(self, config: SearchConfig):
        """Create fallback optimizer when ML common is not available."""
        # Simple random search fallback
        return RandomSearchStrategy()


class EvolutionarySearchStrategy:
    """Evolutionary algorithm search strategy."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.population_initializer: Optional[
            Callable[[Dict[str, Any], int], List[Dict[str, Any]]]
        ] = None

    def set_population_initializer(
        self, initializer: Callable[[Dict[str, Any], int], List[Dict[str, Any]]]
    ) -> None:
        self.population_initializer = initializer
        
    def search(self, 
               search_space: Dict[str, Any], 
               objective_function: Callable,
               config: SearchConfig) -> SearchResult:
        """Perform evolutionary search."""
        try:
            tprint_info("🧬 Starting evolutionary search...")
            start_time = time.time()
            
            # Initialize population
            population = self._initialize_population(search_space, config.population_size)
            fitness_scores = []
            optimization_history = []
            pareto_frontier = []
            
            # Evolution loop
            for generation in range(config.max_iterations):
                # Evaluate fitness
                current_fitness = []
                for individual in population:
                    score = objective_function(individual)
                    current_fitness.append(score)
                
                fitness_scores.extend(current_fitness)
                optimization_history.append({
                    'generation': generation,
                    'best_fitness': max(current_fitness),
                    'avg_fitness': np.mean(current_fitness),
                    'population_diversity': self._calculate_diversity(population)
                })
                
                # Update Pareto frontier
                pareto_frontier = self._update_pareto_frontier(
                    population, current_fitness, pareto_frontier
                )
                
                # Early stopping check
                if config.enable_early_stopping and self._check_convergence(
                    optimization_history, config.early_stopping_patience
                ):
                    tprint_info(f"🛑 Early stopping at generation {generation}")
                    break
                
                # Create next generation
                if generation < config.max_iterations - 1:
                    population = self._create_next_generation(
                        population, current_fitness, config
                    )
            
            # Find best individual
            best_idx = np.argmax(fitness_scores)
            best_architecture = population[best_idx]
            best_score = fitness_scores[best_idx]
            
            # Create result
            result = SearchResult(
                best_architecture=best_architecture,
                best_scores={config.objectives[0]: best_score},
                pareto_frontier=pareto_frontier,
                search_strategy=SearchStrategy.EVOLUTIONARY,
                architecture_type=config.architecture_type,
                total_iterations=generation + 1,
                execution_time=time.time() - start_time,
                convergence_achieved=self._check_convergence(
                    optimization_history, config.early_stopping_patience
                ),
                optimization_history=optimization_history,
                search_statistics={
                    'method': 'evolutionary',
                    'final_population_size': len(population),
                    'total_evaluations': len(fitness_scores)
                },
                diversity_scores=[h['population_diversity'] for h in optimization_history],
                exploration_exploitation_ratio=self._calculate_exploration_ratio(optimization_history),
                hardware_utilization={'cpu': 1.0, 'memory': 0.9}
            )
            
            tprint_success(f"✅ Evolutionary search completed in {result.execution_time:.2f}s")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Evolutionary search failed: {e}")
            return SearchResult(
                best_architecture={},
                best_scores={},
                pareto_frontier=[],
                search_strategy=SearchStrategy.EVOLUTIONARY,
                architecture_type=config.architecture_type,
                total_iterations=0,
                execution_time=0.0,
                convergence_achieved=False,
                optimization_history=[],
                search_statistics={},
                diversity_scores=[],
                exploration_exploitation_ratio=0.0,
                hardware_utilization={},
                success=False,
                error_message=str(e)
            )
    
    def _initialize_population(self, search_space: Dict[str, Any], population_size: int) -> List[Dict[str, Any]]:
        """Initialize population using optional warm start callback."""
        if self.population_initializer is not None:
            try:
                return self.population_initializer(search_space, population_size)
            except Exception as exc:  # pragma: no cover - fallback
                self.logger.warning("Warm start initializer failed: %s", exc)

        population = []
        for _ in range(population_size):
            individual = {}
            for param, values in search_space.items():
                if isinstance(values, list):
                    individual[param] = np.random.choice(values)
                elif isinstance(values, tuple) and len(values) == 2:
                    individual[param] = np.random.uniform(values[0], values[1])
                else:
                    individual[param] = values
            population.append(individual)
        return population
    
    def _calculate_diversity(self, population: List[Dict[str, Any]]) -> float:
        """Calculate population diversity."""
        if len(population) <= 1:
            return 0.0
        
        # Simple diversity metric based on parameter variance
        diversity_scores = []
        param_names = list(population[0].keys())
        
        for param in param_names:
            values = [ind[param] for ind in population if isinstance(ind[param], (int, float))]
            if len(values) > 1:
                diversity_scores.append(np.std(values))
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def _update_pareto_frontier(self, 
                               population: List[Dict[str, Any]], 
                               fitness: List[float],
                               current_frontier: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update Pareto frontier."""
        # Simple implementation - keep best individuals
        combined = list(zip(population, fitness))
        combined.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top 10% as Pareto frontier
        frontier_size = max(1, len(population) // 10)
        return [ind for ind, _ in combined[:frontier_size]]
    
    def _check_convergence(self, 
                          optimization_history: List[Dict[str, Any]], 
                          patience: int) -> bool:
        """Check for convergence."""
        if len(optimization_history) < patience:
            return False
        
        recent_scores = [h['best_fitness'] for h in optimization_history[-patience:]]
        return max(recent_scores) - min(recent_scores) < 0.01
    
    def _create_next_generation(self, 
                               population: List[Dict[str, Any]], 
                               fitness: List[float],
                               config: SearchConfig) -> List[Dict[str, Any]]:
        """Create next generation through selection, crossover, and mutation."""
        new_population = []
        
        # Elitism - keep best individuals
        elite_indices = np.argsort(fitness)[-config.elite_size:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate offspring
        while len(new_population) < config.population_size:
            # Selection
            parent1 = self._tournament_selection(population, fitness, config.evolutionary_config.get('tournament_size', 3))
            parent2 = self._tournament_selection(population, fitness, config.evolutionary_config.get('tournament_size', 3))
            
            # Crossover
            if np.random.random() < config.evolutionary_config.get('crossover_rate', 0.8):
                offspring = self._crossover(parent1, parent2)
            else:
                offspring = parent1.copy()
            
            # Mutation
            if np.random.random() < config.evolutionary_config.get('mutation_rate', 0.1):
                offspring = self._mutate(offspring)
            
            new_population.append(offspring)
        
        return new_population[:config.population_size]
    
    def _tournament_selection(self, 
                             population: List[Dict[str, Any]], 
                             fitness: List[float], 
                             tournament_size: int) -> Dict[str, Any]:
        """Tournament selection."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Uniform crossover."""
        offspring = {}
        for param in parent1.keys():
            if np.random.random() < 0.5:
                offspring[param] = parent1[param]
            else:
                offspring[param] = parent2[param]
        return offspring
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Simple mutation."""
        mutated = individual.copy()
        param_to_mutate = np.random.choice(list(individual.keys()))
        
        if isinstance(individual[param_to_mutate], (int, float)):
            # Add Gaussian noise
            noise = np.random.normal(0, 0.1)
            mutated[param_to_mutate] = individual[param_to_mutate] + noise
        
        return mutated
    
    def _calculate_exploration_ratio(self, optimization_history: List[Dict[str, Any]]) -> float:
        """Calculate exploration vs exploitation ratio."""
        if len(optimization_history) < 2:
            return 0.5
        
        diversity_scores = [h['population_diversity'] for h in optimization_history]
        return np.mean(diversity_scores) / (np.mean(diversity_scores) + 1.0)


class RandomSearchStrategy:
    """Random search strategy."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def search(self, 
               search_space: Dict[str, Any], 
               objective_function: Callable,
               config: SearchConfig) -> SearchResult:
        """Perform random search."""
        try:
            tprint_info("🎲 Starting random search...")
            start_time = time.time()
            
            best_architecture = {}
            best_score = float('-inf')
            optimization_history = []
            
            # Random search loop
            for iteration in range(config.max_iterations):
                # Generate random architecture
                architecture = {}
                for param, values in search_space.items():
                    if isinstance(values, list):
                        architecture[param] = np.random.choice(values)
                    elif isinstance(values, tuple) and len(values) == 2:
                        architecture[param] = np.random.uniform(values[0], values[1])
                    else:
                        architecture[param] = values
                
                # Evaluate
                score = objective_function(architecture)
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_architecture = architecture.copy()
                
                optimization_history.append({
                    'iteration': iteration,
                    'score': score,
                    'best_score': best_score
                })
            
            # Create result
            result = SearchResult(
                best_architecture=best_architecture,
                best_scores={config.objectives[0]: best_score},
                pareto_frontier=[best_architecture],
                search_strategy=SearchStrategy.RANDOM,
                architecture_type=config.architecture_type,
                total_iterations=config.max_iterations,
                execution_time=time.time() - start_time,
                convergence_achieved=False,
                optimization_history=optimization_history,
                search_statistics={'method': 'random_search'},
                diversity_scores=[1.0] * len(optimization_history),
                exploration_exploitation_ratio=1.0,
                hardware_utilization={'cpu': 1.0, 'memory': 0.5}
            )
            
            tprint_success(f"✅ Random search completed in {result.execution_time:.2f}s")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Random search failed: {e}")
            return SearchResult(
                best_architecture={},
                best_scores={},
                pareto_frontier=[],
                search_strategy=SearchStrategy.RANDOM,
                architecture_type=config.architecture_type,
                total_iterations=0,
                execution_time=0.0,
                convergence_achieved=False,
                optimization_history=[],
                search_statistics={},
                diversity_scores=[],
                exploration_exploitation_ratio=0.0,
                hardware_utilization={},
                success=False,
                error_message=str(e)
            )


class UnifiedSearchEngine:
    """
    Unified search engine that provides a single interface for all search strategies
    used by both NAS and TAS systems.
    """
    
    def __init__(self, config: Optional[SearchConfig] = None):
        """Initialize unified search engine."""
        self.config = config or SearchConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            random.seed(self.config.random_seed)
            self.logger.info(
                "Search engine random seed configured",
                extra={"seed": self.config.random_seed},
            )

        # Initialize pluggable strategies
        self.strategy_registry = StrategyRegistry()
        self._register_default_strategies()
        self.strategy_aliases = {
            SearchStrategy.BAYESIAN: "optuna",
            SearchStrategy.ENHANCED_BAYESIAN: "optuna",
            SearchStrategy.NSGA2: "optuna",
            SearchStrategy.SPEA2: "optuna",
            SearchStrategy.RANDOM: "random",
            SearchStrategy.HYBRID: "hyperband",
        }
        self.legacy_strategies = {
            SearchStrategy.EVOLUTIONARY: EvolutionarySearchStrategy(),
            SearchStrategy.ADAPTIVE_EVOLUTIONARY: EvolutionarySearchStrategy(),
            SearchStrategy.REINFORCEMENT_LEARNING: EvolutionarySearchStrategy(),
        }
        for legacy_strategy in self.legacy_strategies.values():
            if hasattr(legacy_strategy, "set_population_initializer"):
                legacy_strategy.set_population_initializer(self._initialize_population)

        # Initialize hardware optimizers
        self.hardware_optimizers: Dict[str, Any] = {}
        self.hardware_warnings: List[str] = []
        if self.config.enable_parallel_processing:
            self._initialize_hardware_optimizers()

        # Performance monitoring
        self.search_history = []
        self.performance_metrics = {}
        self._cache_lock = threading.RLock()
        self.evaluation_cache: "OrderedDict[Tuple, StrategyEvaluation]" = OrderedDict()
        self._cache_hits = 0
        self._max_cache_size = 1000  # Limit cache size to prevent memory issues
        self._cache_evictions = 0
        self._cache_requests = 0

        # Meta-learning warm start utilities
        self.meta_warm_starter: Optional[MetaWarmStarter] = None
        if self.config.enable_meta_warm_start:
            warm_config = MetaWarmStartConfig(
                historical_blueprints=self.config.meta_warm_start_blueprints,
            ).copy_with_overrides(self.config.meta_warm_start_config)
            self.meta_warm_starter = MetaWarmStarter(warm_config)

        # Progressive search tracking
        self._progressive_stage_index = 0
        self._validation_window: deque = deque(maxlen=self.config.progressive_stability_window)

        # Uncertainty-aware surrogate
        self.uncertainty_surrogate = BayesianEnsembleSurrogate(BayesianEnsembleConfig())

        # Time-series cross validation helper
        self.time_series_cv = None
        if self.config.enable_time_series_cv:
            cv_cfg = BlockedPurgedCVConfig(**self.config.time_series_cv_config)
            self.time_series_cv = BlockedPurgedCV(cv_cfg)

        # Hardware-aware evaluator
        hw_config = HardwareConstraintConfig(
            latency_budget_ms=self.config.hardware_latency_budget_ms,
            tail_latency_budget_ms=self.config.hardware_tail_latency_ms,
            cold_start_budget_ms=self.config.hardware_cold_start_ms,
            memory_budget_mb=self.config.hardware_memory_budget_mb,
        )
        self.hardware_evaluator = HardwareCostEvaluator(hw_config)

        self.logger.info(
            "Unified Search Engine initialised",
            extra={
                "architecture_type": self.config.architecture_type.value,
                "default_strategy": self.config.search_strategy.value,
                "available_plugins": self.strategy_registry.available(),
            },
        )

        tprint_info("🚀 Unified Search Engine initialized")
        tprint_info(f"   Architecture type: {self.config.architecture_type.value}")
        tprint_info(f"   Search strategy: {self.config.search_strategy.value}")
        tprint_info(f"   Available strategies: {self.strategy_registry.available()}")
    
    def _manage_cache_size(self):
        """Manage cache size to prevent memory issues."""
        with self._cache_lock:
            removed = 0
            while len(self.evaluation_cache) > self._max_cache_size:
                self.evaluation_cache.popitem(last=False)
                self._cache_evictions += 1
                removed += 1
            if removed:
                self.logger.debug("Cleaned cache: removed %s entries", removed)

    def clear_cache(self):
        """Clear the evaluation cache."""
        with self._cache_lock:
            self.evaluation_cache.clear()
            self._cache_hits = 0
            self._cache_evictions = 0
            self._cache_requests = 0
        self.logger.info("Evaluation cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            cache_size = len(self.evaluation_cache)
        total_requests = max(1, self._cache_requests)
        return {
            'cache_size': cache_size,
            'max_cache_size': self._max_cache_size,
            'cache_hits': self._cache_hits,
            'cache_evictions': self._cache_evictions,
            'cache_hit_rate': self._cache_hits / total_requests,
        }

    def _wrap_objective_function(
        self, objective_function: Callable[[Dict[str, Any]], Any]
    ) -> Callable[[Dict[str, Any]], Dict[str, float]]:
        """Augment the user objective with CV, uncertainty & hardware checks."""

        def wrapped(params: Dict[str, Any]) -> Dict[str, float]:
            candidate_params = self._apply_candidate_stage_constraints(dict(params))

            if self.time_series_cv is not None:
                raw_result = self.time_series_cv.evaluate(objective_function, candidate_params)
            else:
                raw_result = objective_function(candidate_params)

            metrics = self._normalize_metrics(raw_result)

            # Ensure required objectives exist
            sharpe = float(metrics.get('sharpe_ratio', metrics.get('sharpe', 0.0)))
            downside = float(metrics.get('downside_deviation', metrics.get('downside_risk', metrics.get('volatility', 0.0))))
            latency_ms = float(metrics.get('latency_ms', metrics.get('execution_latency', 0.0)))

            metrics['sharpe_ratio_raw'] = sharpe
            metrics['downside_deviation'] = downside
            metrics['execution_latency'] = latency_ms

            # Hardware-aware evaluation
            hardware_costs = self.hardware_evaluator.estimate(candidate_params, metrics)
            metrics.update({
                'latency_ms': hardware_costs['latency_ms'],
                'tail_latency': hardware_costs['tail_latency_ms'],
                'cold_start_latency': hardware_costs['cold_start_ms'],
                'memory_mb': hardware_costs['memory_mb'],
            })

            penalty = 0.0
            if self.config.enable_hardware_constraints and not self.hardware_evaluator.validate(hardware_costs):
                penalty = self.hardware_evaluator.constraint_penalty(hardware_costs)
                metrics['hardware_penalty'] = penalty
            else:
                metrics['hardware_penalty'] = 0.0

            # Update surrogate and compute UCB
            self.uncertainty_surrogate.update(candidate_params, sharpe)
            if self.config.enable_uncertainty_guided_promotion:
                ucb_score = self.uncertainty_surrogate.compute_ucb(candidate_params)
                if ucb_score is not None:
                    metrics['sharpe_ratio_ucb'] = float(ucb_score)
                    sharpe = float(ucb_score)

            # Apply penalty and ensure objective sign conventions
            sharpe -= penalty
            metrics['sharpe_ratio'] = sharpe
            metrics['execution_latency'] = latency_ms + penalty

            self._maybe_update_progressive_stage(metrics)

            return metrics

        return wrapped

    def _initialize_hardware_optimizers(self):
        """Initialize hardware optimization components."""
        try:
            if self.config.enable_gpu_acceleration:
                try:
                    self.hardware_optimizers['gpu'] = _load_m1_gpu_manager()()
                except Exception as exc:  # pragma: no cover - GPU optional
                    warning = f"GPU optimizer unavailable: {exc}"
                    self.hardware_warnings.append(warning)
                    tprint_warning(f"⚠️ {warning}")
                    self.logger.debug("GPU optimizer unavailable", exc_info=exc)

            if self.config.memory_limit_gb:
                try:
                    memory_optimizer_factory = _load_m1_memory_optimizer()
                    self.hardware_optimizers['memory'] = memory_optimizer_factory(self.config.memory_limit_gb)
                except Exception as exc:  # pragma: no cover - memory optimizer optional
                    warning = f"Memory optimizer unavailable: {exc}"
                    self.hardware_warnings.append(warning)
                    tprint_warning(f"⚠️ {warning}")

            try:
                cpu_optimizer_factory = _load_m1_cpu_optimizer()
                self.hardware_optimizers['cpu'] = cpu_optimizer_factory()
            except Exception as exc:  # pragma: no cover - CPU optimizer optional
                warning = f"CPU optimizer unavailable: {exc}"
                self.hardware_warnings.append(warning)
                tprint_warning(f"⚠️ {warning}")
                self.logger.debug("CPU optimizer unavailable", exc_info=exc)

            tprint_success("✅ Hardware optimizers initialized")

        except Exception as e:
            warning = f"Hardware optimization setup failed: {e}"
            self.hardware_warnings.append(warning)
            tprint_warning(f"⚠️ {warning}")
            self.hardware_optimizers = {}

    def _initialize_population(
        self, search_space: Dict[str, Any], population_size: int
    ) -> List[Dict[str, Any]]:
        """Warm-start aware population initialization used by legacy strategies."""

        effective_space = self._apply_progressive_search_space(search_space)

        def _random_sample(space: Dict[str, Any]) -> Dict[str, Any]:
            sample: Dict[str, Any] = {}
            for param, values in space.items():
                if isinstance(values, dict):
                    if 'choices' in values:
                        sample[param] = np.random.choice(values['choices'])
                    elif {'low', 'high'}.issubset(values):
                        sample[param] = np.random.uniform(values['low'], values['high'])
                    else:
                        sample[param] = values
                elif isinstance(values, list):
                    sample[param] = np.random.choice(values)
                elif isinstance(values, tuple) and len(values) == 2:
                    sample[param] = np.random.uniform(values[0], values[1])
                else:
                    sample[param] = values
            return sample

        if self.meta_warm_starter is not None:
            try:
                return self.meta_warm_starter.warm_start(
                    effective_space, population_size, _random_sample
                )
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning("Meta warm start failed, falling back to random: %s", exc)

        return [_random_sample(effective_space) for _ in range(population_size)]

    def _apply_progressive_search_space(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        if not self.config.enable_progressive_search or not self.config.progressive_search_stages:
            return search_space

        stage_idx = min(self._progressive_stage_index, len(self.config.progressive_search_stages) - 1)
        stage = self.config.progressive_search_stages[stage_idx]
        adjusted = copy.deepcopy(search_space)
        max_depth = stage.get('max_depth')
        max_width = stage.get('max_width')
        unlocked = set(stage.get('unlocked_params', []))

        for key, definition in adjusted.items():
            if isinstance(definition, dict) and {'low', 'high'}.issubset(definition):
                high = definition['high']
                if max_depth is not None and ('depth' in key or 'layer' in key):
                    definition['high'] = min(high, max_depth)
                if max_width is not None and any(token in key for token in ('width', 'units', 'channels')):
                    definition['high'] = min(definition['high'], max_width)
            elif isinstance(definition, (list, tuple)) and len(definition) == 2:
                low, high = definition
                if max_depth is not None and ('depth' in key or 'layer' in key):
                    adjusted[key] = (low, min(high, max_depth))
                elif max_width is not None and any(token in key for token in ('width', 'units', 'channels')):
                    adjusted[key] = (low, min(high, max_width))

            if unlocked and key not in unlocked and stage_idx > 0:
                # Freeze parameters not yet unlocked by stage
                if isinstance(definition, dict) and 'choices' in definition and definition['choices']:
                    adjusted[key] = {'choices': [definition['choices'][0]]}
                elif isinstance(definition, (list, tuple)) and definition:
                    adjusted[key] = definition[0]

        return adjusted

    def _apply_candidate_stage_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.config.enable_progressive_search or not self.config.progressive_search_stages:
            return params

        stage_idx = min(self._progressive_stage_index, len(self.config.progressive_search_stages) - 1)
        stage = self.config.progressive_search_stages[stage_idx]
        max_depth = stage.get('max_depth')
        max_width = stage.get('max_width')
        adjusted = dict(params)

        for key, value in list(adjusted.items()):
            if not isinstance(value, (int, float)):
                continue
            if max_depth is not None and ('depth' in key or 'layer' in key):
                adjusted[key] = min(value, max_depth)
            if max_width is not None and any(token in key for token in ('width', 'units', 'channels')):
                adjusted[key] = min(value, max_width)

        return adjusted

    def _maybe_update_progressive_stage(self, metrics: Dict[str, float]) -> None:
        if not self.config.enable_progressive_search:
            return

        score = metrics.get('sharpe_ratio')
        if score is None:
            return

        self._validation_window.append(float(score))
        if len(self._validation_window) < self.config.progressive_stability_window:
            return

        window = np.array(self._validation_window)
        tolerance = self.config.progressive_stability_tolerance
        if np.std(window) <= tolerance * max(abs(np.mean(window)), 1e-6):
            if self._progressive_stage_index < len(self.config.progressive_search_stages) - 1:
                self._progressive_stage_index += 1
                self._validation_window.clear()
                self.logger.info(
                    "Progressive search advanced to stage %s", self._progressive_stage_index + 1
                )

    def _register_default_strategies(self) -> None:
        """Register built-in strategy plugins."""

        def _wrap(factory):
            return lambda random_seed=None, **kwargs: factory(random_seed=random_seed, **kwargs)

        self.strategy_registry.register("random", _wrap(PluginRandomStrategy))
        self.strategy_registry.register("grid", _wrap(PluginGridStrategy))
        self.strategy_registry.register("optuna", _wrap(PluginOptunaStrategy))
        self.strategy_registry.register("hyperband", _wrap(PluginHyperbandStrategy))

    def _resolve_strategy(self, strategy: SearchStrategy) -> Any:
        """Resolve an enum to either a plugin or legacy implementation."""

        plugin_name = self.strategy_aliases.get(strategy)
        if plugin_name is None and strategy.value in self.strategy_registry.available():
            plugin_name = strategy.value

        if plugin_name is not None:
            return self.strategy_registry.create(plugin_name, random_seed=self.config.random_seed)

        if strategy in self.legacy_strategies:
            return self.legacy_strategies[strategy]

        raise ValueError(f"Unsupported search strategy: {strategy}")

    def _run_legacy_strategy(
        self,
        strategy_impl: SearchStrategyInterface,
        search_space: Dict[str, Any],
        objective_function: Callable[[Dict[str, Any]], Any],
    ) -> SearchResult:
        with memory_checkpoint("unified_search_legacy"):
            return strategy_impl.search(
                objective_function=objective_function,
                search_space=search_space,
                config=self.config,
            )

    def _run_plugin_strategy(
        self,
        strategy_impl: Any,
        search_space: Dict[str, Any],
        objective_function: Callable[[Dict[str, Any]], Any],
        strategy: SearchStrategy,
        start_time: float,
    ) -> SearchResult:
        state = StrategyState()
        strategy_config = self._build_strategy_config()
        strategy_impl.initialize(search_space, lambda params: {}, state, strategy_config)

        with memory_checkpoint("unified_search_plugin"):
            while strategy_impl.should_continue(state):
                candidates = strategy_impl.sample_candidates(
                    state, strategy_config.get("max_candidates_per_batch", 1)
                )
                if not candidates:
                    state.terminated = True
                    break

                evaluations: List[StrategyEvaluation] = []
                for candidate in candidates:
                    evaluation = self._evaluate_candidate(candidate, objective_function)
                    if evaluation is not None:
                        evaluations.append(evaluation)

                if not evaluations:
                    # Prevent endless loops if everything violates constraints
                    state.terminated = True
                    break

                strategy_impl.update_state(state, evaluations)

        summary = strategy_impl.finalize(state)
        execution_time = time.time() - start_time
        return self._build_result_from_summary(
            summary=summary,
            search_strategy=strategy,
            state=state,
            execution_time=execution_time,
        )

    def _build_strategy_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "max_iterations": self.config.max_iterations,
            "max_candidates_per_batch": self.config.max_candidates_per_batch,
            "objective_names": self.config.objective_names,
            "objective_directions": self.config.objective_directions,
            "enable_pruning": self.config.enable_early_stopping,
            "hyperband_min_resource": self.config.bayesian_config.get('min_resource', 1),
            "hyperband_max_resource": self.config.max_iterations,
            "max_trials": self.config.bayesian_config.get('n_trials'),
        }
        config.update(self.config.bayesian_config)
        return config

    def _evaluate_candidate(
        self,
        candidate: StrategyCandidate,
        objective_function: Callable[[Dict[str, Any]], Any],
    ) -> Optional[StrategyEvaluation]:
        key = candidate.cache_key()
        if self.config.cache_results:
            with self._cache_lock:
                self._cache_requests += 1
                cached = self.evaluation_cache.get(key)
                if cached is not None:
                    # Maintain LRU ordering by re-inserting
                    self.evaluation_cache.pop(key)
                    self.evaluation_cache[key] = cached
                    self._cache_hits += 1
                    return cached

        if self.config.enable_constraint_validation and not self._validate_candidate_constraints(candidate.params):
            self.logger.debug(
                "Candidate rejected by constraints", extra={"params": candidate.params}
            )
            return None

        metrics = self._normalize_metrics(objective_function(candidate.params))
        evaluation = StrategyEvaluation(candidate=candidate, metrics=metrics)
        if self.config.cache_results:
            with self._cache_lock:
                self.evaluation_cache[key] = evaluation
                self._manage_cache_size()  # Manage cache size after adding new entry
        return evaluation

    def _normalize_metrics(self, raw_result: Any) -> Dict[str, float]:
        if isinstance(raw_result, dict):
            return {k: float(v) for k, v in raw_result.items()}
        if isinstance(raw_result, (list, tuple)):
            names = self.config.objective_names or ["score"]
            return {
                names[idx] if idx < len(names) else f"metric_{idx}": float(value)
                for idx, value in enumerate(raw_result)
            }
        if isinstance(raw_result, (np.ndarray,)):
            array = raw_result.tolist()
            return self._normalize_metrics(array)
        return {"score": float(raw_result)}

    def _validate_candidate_constraints(self, params: Dict[str, Any]) -> bool:
        if self.config.max_parameter_count is not None:
            param_count = params.get("parameter_count") or params.get("n_parameters")
            if param_count is not None and param_count > self.config.max_parameter_count:
                return False
        if self.config.max_flops is not None:
            flops = params.get("flops") or params.get("estimated_flops")
            if flops is not None and flops > self.config.max_flops:
                return False
        return True

    def _run_overfitting_checks(self, result: SearchResult) -> None:
        if not self.config.enable_overfitting_defense:
            return
        if not result.optimization_history:
            return

        scores = [entry.get('score') for entry in result.optimization_history if 'score' in entry]
        scores = [float(score) for score in scores if score is not None]
        if not scores:
            return

        spa = hansen_spa_test(
            scores,
            n_bootstrap=self.config.spa_bootstrap_iterations,
            transaction_cost_bp=self.config.spa_transaction_cost_bp,
        )
        result.search_statistics.setdefault('overfitting_check', {})
        result.search_statistics['overfitting_check'].update(
            {
                'p_value': spa.p_value,
                'threshold': spa.threshold,
                'passes': spa.passes,
            }
        )

    def _build_result_from_summary(
        self,
        summary: Dict[str, Any],
        search_strategy: SearchStrategy,
        state: StrategyState,
        execution_time: float,
    ) -> SearchResult:
        best_scores: Dict[OptimizationObjective, float] = {}
        for objective, name in zip(self.config.objectives, self.config.objective_names):
            if name in summary.get("best_metrics", {}):
                best_scores[objective] = float(summary["best_metrics"][name])

        optimization_history = []
        primary_metric = self.config.objective_names[0] if self.config.objective_names else "score"
        for idx, entry in enumerate(summary.get("history", [])):
            metrics = entry.get("metrics", {})
            optimization_history.append(
                {
                    "iteration": idx,
                    "score": float(metrics.get(primary_metric, metrics.get("score", 0.0))),
                }
            )

        return SearchResult(
            best_architecture=summary.get("best_params", {}),
            best_scores=best_scores,
            pareto_frontier=summary.get("pareto_front", []),
            search_strategy=search_strategy,
            architecture_type=self.config.architecture_type,
            total_iterations=state.iteration,
            execution_time=execution_time,
            convergence_achieved=state.terminated,
            optimization_history=optimization_history,
            search_statistics={
                "evaluations": len(state.history),
                "cache_hits": self._cache_hits,
                "plugin_state": summary,
            },
            diversity_scores=[],
            exploration_exploitation_ratio=0.0,
            hardware_utilization={},
        )
    
    def search(self,
               search_space: Dict[str, Any],
               objective_function: Callable[[Dict[str, Any]], Any],
               strategy: Optional[SearchStrategy] = None) -> SearchResult:
        """Perform architecture search using the specified strategy."""

        search_strategy = strategy or self.config.search_strategy
        start_time = time.time()
        self._cache_hits = 0
        with self._cache_lock:
            self.evaluation_cache.clear()

        try:
            self._validate_search_inputs(search_space, objective_function)
            strategy_impl = self._resolve_strategy(search_strategy)

            self._progressive_stage_index = 0
            self._validation_window.clear()

            effective_space = self._apply_progressive_search_space(search_space)
            wrapped_objective = self._wrap_objective_function(objective_function)

            tprint_info(f"🔍 Starting {search_strategy.value} search...")
            tprint_info(f"   Search space size: {len(search_space)} parameters")
            tprint_info(f"   Max iterations: {self.config.max_iterations}")

            if isinstance(strategy_impl, SearchStrategyInterface):
                result = self._run_legacy_strategy(
                    strategy_impl, effective_space, wrapped_objective
                )
            else:
                result = self._run_plugin_strategy(
                    strategy_impl, effective_space, wrapped_objective, search_strategy, start_time
                )

            self._update_performance_metrics(result)
            self.search_history.append(result)
            if self.config.save_intermediate_results:
                self._save_intermediate_result(result)

            self._run_overfitting_checks(result)

            tprint_success("✅ Search completed successfully")
            tprint_info(
                f"   Best score: {max(result.best_scores.values()) if result.best_scores else 0:.4f}"
            )
            tprint_info(f"   Execution time: {result.execution_time:.2f}s")
            tprint_info(f"   Convergence: {'Yes' if result.convergence_achieved else 'No'}")
            return result

        except Exception as e:
            self.logger.exception("Search failed", exc_info=e)
            tprint_error(f"❌ Search failed: {e}")
            return SearchResult(
                best_architecture={},
                best_scores={},
                pareto_frontier=[],
                search_strategy=search_strategy,
                architecture_type=self.config.architecture_type,
                total_iterations=0,
                execution_time=time.time() - start_time,
                convergence_achieved=False,
                optimization_history=[],
                search_statistics={},
                diversity_scores=[],
                exploration_exploitation_ratio=0.0,
                hardware_utilization={},
                success=False,
                error_message=str(e)
            )
    
    def _validate_search_inputs(
        self,
        search_space: Dict[str, Any],
        objective_function: Callable[[Dict[str, Any]], Any],
    ) -> None:
        if not isinstance(search_space, dict):
            raise TypeError("Search space must be a dictionary")
        if not callable(objective_function):
            raise TypeError("Objective function must be callable")

        self._validate_search_space(search_space)

        if self.config.objective_weights and len(self.config.objective_weights) != len(self.config.objectives):
            raise ValueError("Objective weights must match the number of objectives")

    def _validate_search_space(self, search_space: Dict[str, Any]) -> None:
        """Validate search space definition."""
        if not search_space:
            raise ValueError("Search space cannot be empty")

        for param, values in search_space.items():
            if not isinstance(param, str) or not param.strip():
                raise TypeError("Search space keys must be non-empty strings")

            if isinstance(values, dict):
                has_bounds = {'low', 'high'}.issubset(values.keys())
                has_choices = 'choices' in values
                if not (has_bounds or has_choices):
                    raise ValueError(
                        f"Parameter {param} must define bounds ['low', 'high'] or 'choices'"
                    )
                if has_bounds:
                    low = values['low']
                    high = values['high']
                    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                        raise TypeError(f"Bounds for parameter {param} must be numeric")
                    if not np.isfinite(low) or not np.isfinite(high):
                        raise ValueError(f"Bounds for parameter {param} must be finite")
                    if low >= high:
                        raise ValueError(
                            f"Lower bound must be less than upper bound for parameter {param}"
                        )
                    if high - low < 1e-9:
                        raise ValueError(
                            f"Bounds for parameter {param} are unrealistically narrow"
                        )
                    if abs(high) > 1e9 or abs(low) > 1e9:
                        raise ValueError(
                            f"Bounds for parameter {param} exceed realistic limits (|value| > 1e9)"
                        )
                    step = values.get('step')
                    if step is not None and step <= 0:
                        raise ValueError(
                            f"Step for parameter {param} must be positive when provided"
                        )
                if has_choices:
                    choices = values['choices']
                    if not isinstance(choices, (list, tuple)) or not choices:
                        raise ValueError(f"Parameter {param} choices must be a non-empty sequence")
                    if any(choice is None for choice in choices):
                        raise ValueError(f"Parameter {param} choices cannot contain None values")
                    if len(set(choices)) != len(choices):
                        raise ValueError(f"Parameter {param} choices contain duplicates")
                continue

            if isinstance(values, tuple):
                if len(values) != 2:
                    raise ValueError(
                        f"Parameter {param} tuple must have exactly 2 values (min, max)"
                    )
                low, high = values
                if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                    raise TypeError(f"Bounds for parameter {param} must be numeric")
                if not np.isfinite(low) or not np.isfinite(high):
                    raise ValueError(f"Bounds for parameter {param} must be finite")
                if low >= high:
                    raise ValueError(
                        f"Lower bound must be less than upper bound for parameter {param}"
                    )
                if abs(high) > 1e9 or abs(low) > 1e9:
                    raise ValueError(
                        f"Bounds for parameter {param} exceed realistic limits (|value| > 1e9)"
                    )
                continue

            if isinstance(values, list):
                if not values:
                    raise ValueError(f"Parameter {param} list cannot be empty")
                if any(value is None for value in values):
                    raise ValueError(f"Parameter {param} list cannot contain None values")
                continue

            raise ValueError(
                f"Parameter {param} must be defined using a dict, tuple or list"
            )
    
    def _update_performance_metrics(self, result: SearchResult):
        """Update performance metrics."""
        strategy_name = result.search_strategy.value
        
        if strategy_name not in self.performance_metrics:
            self.performance_metrics[strategy_name] = {
                'total_searches': 0,
                'successful_searches': 0,
                'avg_execution_time': 0.0,
                'avg_best_score': 0.0,
                'convergence_rate': 0.0
            }
        
        metrics = self.performance_metrics[strategy_name]
        metrics['total_searches'] += 1
        
        if result.success:
            metrics['successful_searches'] += 1
        
        # Update averages
        total = metrics['total_searches']
        successful = metrics['successful_searches']
        
        if successful > 0:
            metrics['avg_execution_time'] = (
                (metrics['avg_execution_time'] * (successful - 1) + result.execution_time) / successful
            )
            
            best_score = max(result.best_scores.values()) if result.best_scores else 0.0
            metrics['avg_best_score'] = (
                (metrics['avg_best_score'] * (successful - 1) + best_score) / successful
            )
            
            metrics['convergence_rate'] = successful / total if total > 0 else 0.0
    
    def _save_intermediate_result(self, result: SearchResult):
        """Save intermediate search result."""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"search_result_{result.search_strategy.value}_{timestamp}.json"
            
            # Convert result to serializable format
            result_dict = {
                'best_architecture': result.best_architecture,
                'best_scores': {k.value: v for k, v in result.best_scores.items()},
                'search_strategy': result.search_strategy.value,
                'architecture_type': result.architecture_type.value,
                'execution_time': result.execution_time,
                'success': result.success,
                'timestamp': result.timestamp
            }
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            tprint_debug(f"💾 Intermediate result saved: {filename}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to save intermediate result: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all search strategies."""
        summary = {
            'total_searches': sum(m['total_searches'] for m in self.performance_metrics.values()),
            'strategies_performance': self.performance_metrics,
            'search_history_length': len(self.search_history),
            'recommended_strategy': self._get_recommended_strategy()
        }
        
        return summary
    
    def _get_recommended_strategy(self) -> Optional[SearchStrategy]:
        """Get recommended search strategy based on performance."""
        if not self.performance_metrics:
            return None
        
        # Find strategy with best average score and highest success rate
        best_strategy = None
        best_score = -1
        
        for strategy_name, metrics in self.performance_metrics.items():
            if metrics['successful_searches'] > 0:
                score = metrics['avg_best_score'] * metrics['convergence_rate']
                if score > best_score:
                    best_score = score
                    best_strategy = SearchStrategy(strategy_name)
        
        return best_strategy
    
    def compare_strategies(self, 
                          search_space: Dict[str, Any], 
                          objective_function: Callable,
                          strategies: List[SearchStrategy],
                          iterations_per_strategy: int = 50) -> Dict[SearchStrategy, SearchResult]:
        """Compare multiple search strategies."""
        tprint_info(f"⚖️ Comparing {len(strategies)} search strategies...")
        
        results = {}
        original_max_iterations = self.config.max_iterations
        
        try:
            for strategy in strategies:
                tprint_info(f"🔍 Testing {strategy.value}...")
                
                # Temporarily modify config for comparison
                self.config.max_iterations = iterations_per_strategy
                
                # Perform search
                result = self.search(search_space, objective_function, strategy)
                results[strategy] = result
                
                tprint_info(f"   Best score: {max(result.best_scores.values()) if result.best_scores else 0:.4f}")
                tprint_info(f"   Time: {result.execution_time:.2f}s")
                tprint_info(f"   Convergence: {'Yes' if result.convergence_achieved else 'No'}")
            
            # Restore original config
            self.config.max_iterations = original_max_iterations
            
            # Print comparison summary
            tprint_info("📊 Strategy Comparison Summary:")
            for strategy, result in results.items():
                best_score = max(result.best_scores.values()) if result.best_scores else 0.0
                tprint_info(f"   {strategy.value}: Score={best_score:.4f}, Time={result.execution_time:.2f}s")
            
            return results
            
        except Exception as e:
            tprint_error(f"❌ Strategy comparison failed: {e}")
            # Restore original config
            self.config.max_iterations = original_max_iterations
            return {}


# Convenience functions
def create_unified_search_engine(config: Optional[SearchConfig] = None) -> UnifiedSearchEngine:
    """Create a unified search engine with specified configuration."""
    return UnifiedSearchEngine(config)


def quick_search(search_space: Dict[str, Any], 
                objective_function: Callable,
                strategy: SearchStrategy = SearchStrategy.ENHANCED_BAYESIAN,
                max_iterations: int = 100) -> SearchResult:
    """Quick search using default configuration."""
    config = SearchConfig(
        search_strategy=strategy,
        max_iterations=max_iterations
    )
    engine = UnifiedSearchEngine(config)
    return engine.search(search_space, objective_function)


# Export main classes and functions
__all__ = [
    'UnifiedSearchEngine',
    'SearchConfig',
    'SearchResult',
    'SearchStrategy',
    'ArchitectureType',
    'OptimizationObjective',
    'BayesianSearchStrategy',
    'EvolutionarySearchStrategy',
    'RandomSearchStrategy',
    'create_unified_search_engine',
    'quick_search'
]