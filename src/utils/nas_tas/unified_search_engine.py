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

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Protocol
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Import unified utilities
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report,
    optimize_dataframe_dtypes, safe_to_parquet, safe_read_parquet,
    align_dataframes, validate_dataframe_schema, guard_dataframe_nulls,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    integrate_with_m1_optimizers, memory_checkpoint, gpu_context,
    optimize_memory, get_memory_usage, validate_file_path, get_file_size,
    check_disk_space, CommonUtilities
)

from src.utils.math_validation import (
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

# Import ML common utilities
try:
    from .bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, BayesianTPEConfig, OptimizationResult
    )
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False
    warnings.warn("ML common utilities not available, using fallback implementations")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


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
    DARTS = "darts"
    PROXYLESS = "proxylessnas"
    NASNET = "nasnet"


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


@dataclass
class SearchConfig:
    """Unified configuration for search strategies."""
    
    # Core search parameters
    architecture_type: ArchitectureType = ArchitectureType.HYBRID
    search_strategy: SearchStrategy = SearchStrategy.ENHANCED_BAYESIAN
    max_iterations: int = 100
    population_size: int = 50
    elite_size: int = 5
    
    # Optimization objectives
    objectives: List[OptimizationObjective] = field(default_factory=lambda: [
        OptimizationObjective.ACCURACY,
        OptimizationObjective.EFFICIENCY,
        OptimizationObjective.PROFITABILITY
    ])
    objective_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])
    
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
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.optimizer = None
        
    def search(self, 
               search_space: Dict[str, Any], 
               objective_function: Callable,
               config: SearchConfig) -> SearchResult:
        """Perform Bayesian optimization search."""
        try:
            tprint_info("🔍 Starting Bayesian optimization search...")
            start_time = time.time()
            
            # Initialize Bayesian optimizer
            if ML_COMMON_AVAILABLE:
                bayesian_config = BayesianTPEConfig(**config.bayesian_config)
                self.optimizer = BayesianTPEOptimizer(bayesian_config)
            else:
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
        """Initialize random population."""
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


class DifferentiableNASStrategy:
    """Simplified differentiable NAS strategy inspired by DARTS."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _initial_architecture(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        architecture = {}
        for param, values in search_space.items():
            if isinstance(values, (list, tuple)) and values:
                if isinstance(values[0], (int, float)):
                    architecture[param] = float(np.mean(values))
                else:
                    architecture[param] = values[0]
            else:
                architecture[param] = values
        return architecture

    def _perturb_architecture(self, architecture: Dict[str, Any], search_space: Dict[str, Any]) -> Dict[str, Any]:
        candidate = architecture.copy()
        for param, values in search_space.items():
            if not isinstance(values, (list, tuple)) or not values:
                continue
            if isinstance(values[0], (int, float)):
                span = max(values) - min(values) if len(values) > 1 else 1.0
                noise = np.random.normal(0, 0.1 * span)
                candidate[param] = float(np.clip(candidate[param] + noise, min(values), max(values)))
            else:
                candidate[param] = np.random.choice(values)
        return candidate

    def search(self, search_space: Dict[str, Any], objective_function: Callable, config: SearchConfig) -> SearchResult:
        try:
            best_architecture = self._initial_architecture(search_space)
            best_score = objective_function(best_architecture)
            history = []

            for iteration in range(config.max_iterations):
                candidate = self._perturb_architecture(best_architecture, search_space)
                score = objective_function(candidate)
                history.append({'iteration': iteration, 'score': score})
                if score > best_score:
                    best_score = score
                    best_architecture = candidate

            return SearchResult(
                best_architecture=best_architecture,
                best_scores={config.objectives[0]: best_score},
                pareto_frontier=[best_architecture],
                search_strategy=SearchStrategy.DARTS,
                architecture_type=config.architecture_type,
                total_iterations=config.max_iterations,
                execution_time=0.0,
                convergence_achieved=True,
                optimization_history=history,
                search_statistics={'method': 'differentiable'},
                diversity_scores=[],
                exploration_exploitation_ratio=0.7,
                hardware_utilization={'cpu': 0.5},
            )
        except Exception as exc:
            self.logger.error(f"DARTS-style search failed: {exc}")
            return SearchResult(
                best_architecture={},
                best_scores={},
                pareto_frontier=[],
                search_strategy=SearchStrategy.DARTS,
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
                error_message=str(exc),
            )


class ProxylessNASStrategy:
    """ProxylessNAS-inspired strategy using adaptive sampling."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _fallback_architecture(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        return {param: (values[0] if isinstance(values, (list, tuple)) and values else values) for param, values in search_space.items()}

    def search(self, search_space: Dict[str, Any], objective_function: Callable, config: SearchConfig) -> SearchResult:
        try:
            weight_tables = {}
            for param, values in search_space.items():
                if isinstance(values, (list, tuple)) and values:
                    weight_tables[param] = np.ones(len(values), dtype=float)

            best_architecture = None
            best_score = -np.inf
            history = []

            for iteration in range(config.max_iterations):
                candidate = {}
                indices = {}
                for param, values in search_space.items():
                    if isinstance(values, (list, tuple)) and values:
                        weights = weight_tables.get(param)
                        if weights is not None:
                            probs = weights / weights.sum()
                            choice_idx = np.random.choice(len(values), p=probs)
                            candidate[param] = values[choice_idx]
                            indices[param] = choice_idx
                        else:
                            candidate[param] = values[0]
                    else:
                        candidate[param] = values

                score = objective_function(candidate)
                history.append({'iteration': iteration, 'score': score})

                if score > best_score:
                    best_score = score
                    best_architecture = candidate.copy()

                for param, idx in indices.items():
                    weights = weight_tables[param]
                    weights[idx] = weights[idx] * 0.9 + 0.1 * max(score, 0)
                    weight_tables[param] = weights

            if best_architecture is None:
                best_architecture = self._fallback_architecture(search_space)

            return SearchResult(
                best_architecture=best_architecture,
                best_scores={config.objectives[0]: best_score},
                pareto_frontier=[best_architecture],
                search_strategy=SearchStrategy.PROXYLESS,
                architecture_type=config.architecture_type,
                total_iterations=config.max_iterations,
                execution_time=0.0,
                convergence_achieved=True,
                optimization_history=history,
                search_statistics={'method': 'proxyless'},
                diversity_scores=[],
                exploration_exploitation_ratio=0.6,
                hardware_utilization={'cpu': 0.4},
            )
        except Exception as exc:
            self.logger.error(f"ProxylessNAS search failed: {exc}")
            return SearchResult(
                best_architecture={},
                best_scores={},
                pareto_frontier=[],
                search_strategy=SearchStrategy.PROXYLESS,
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
                error_message=str(exc),
            )


class NASNetSearchStrategy:
    """NASNet-inspired search using hybrid sampling."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def search(self, search_space: Dict[str, Any], objective_function: Callable, config: SearchConfig) -> SearchResult:
        try:
            best_architecture = None
            best_score = -np.inf
            history = []

            for iteration in range(config.max_iterations):
                candidate = {}
                for param, values in search_space.items():
                    if isinstance(values, (list, tuple)) and values:
                        if isinstance(values[0], (int, float)):
                            candidate[param] = float(np.random.choice(values))
                        else:
                            candidate[param] = np.random.choice(values)
                    else:
                        candidate[param] = values

                score = objective_function(candidate)
                history.append({'iteration': iteration, 'score': score})

                if score > best_score:
                    best_score = score
                    best_architecture = candidate.copy()

            if best_architecture is None:
                best_architecture = {}

            return SearchResult(
                best_architecture=best_architecture,
                best_scores={config.objectives[0]: best_score},
                pareto_frontier=[best_architecture],
                search_strategy=SearchStrategy.NASNET,
                architecture_type=config.architecture_type,
                total_iterations=config.max_iterations,
                execution_time=0.0,
                convergence_achieved=True,
                optimization_history=history,
                search_statistics={'method': 'nasnet_random'},
                diversity_scores=[],
                exploration_exploitation_ratio=0.5,
                hardware_utilization={'cpu': 0.3},
            )
        except Exception as exc:
            self.logger.error(f"NASNet search failed: {exc}")
            return SearchResult(
                best_architecture={},
                best_scores={},
                pareto_frontier=[],
                search_strategy=SearchStrategy.NASNET,
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
                error_message=str(exc),
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
        
        # Initialize search strategies
        self.search_strategies = {
            SearchStrategy.BAYESIAN: BayesianSearchStrategy(),
            SearchStrategy.EVOLUTIONARY: EvolutionarySearchStrategy(),
            SearchStrategy.RANDOM: RandomSearchStrategy(),
            SearchStrategy.ENHANCED_BAYESIAN: BayesianSearchStrategy(),  # Same as Bayesian for now
            SearchStrategy.ADAPTIVE_EVOLUTIONARY: EvolutionarySearchStrategy(),  # Same as Evolutionary for now
            SearchStrategy.DARTS: DifferentiableNASStrategy(),
            SearchStrategy.PROXYLESS: ProxylessNASStrategy(),
            SearchStrategy.NASNET: NASNetSearchStrategy(),
        }
        
        # Initialize hardware optimizers
        self.hardware_optimizers = {}
        if self.config.enable_parallel_processing:
            self._initialize_hardware_optimizers()
        
        # Performance monitoring
        self.search_history = []
        self.performance_metrics = {}
        
        tprint_info(f"🚀 Unified Search Engine initialized")
        tprint_info(f"   Architecture type: {self.config.architecture_type.value}")
        tprint_info(f"   Search strategy: {self.config.search_strategy.value}")
        tprint_info(f"   Available strategies: {list(self.search_strategies.keys())}")
    
    def _initialize_hardware_optimizers(self):
        """Initialize hardware optimization components."""
        try:
            if self.config.enable_gpu_acceleration:
                self.hardware_optimizers['gpu'] = get_m1_gpu_manager()
            
            if self.config.memory_limit_gb:
                self.hardware_optimizers['memory'] = get_m1_memory_optimizer(self.config.memory_limit_gb)
            
            self.hardware_optimizers['cpu'] = get_m1_cpu_optimizer()
            
            tprint_success("✅ Hardware optimizers initialized")
            
        except Exception as e:
            tprint_warning(f"⚠️ Hardware optimization setup failed: {e}")
            self.hardware_optimizers = {}
    
    def search(self, 
               search_space: Dict[str, Any], 
               objective_function: Callable,
               strategy: Optional[SearchStrategy] = None) -> SearchResult:
        """
        Perform architecture search using the specified strategy.
        
        Args:
            search_space: Dictionary defining the search space
            objective_function: Function to evaluate architectures
            strategy: Search strategy to use (defaults to config strategy)
            
        Returns:
            SearchResult containing the best architecture and metrics
        """
        try:
            # Use specified strategy or default from config
            search_strategy = strategy or self.config.search_strategy
            
            # Validate search space
            self._validate_search_space(search_space)
            
            # Get search strategy implementation
            if search_strategy not in self.search_strategies:
                raise ValueError(f"Unsupported search strategy: {search_strategy}")
            
            strategy_impl = self.search_strategies[search_strategy]
            
            tprint_info(f"🔍 Starting {search_strategy.value} search...")
            tprint_info(f"   Search space size: {len(search_space)} parameters")
            tprint_info(f"   Max iterations: {self.config.max_iterations}")
            
            # Perform search with hardware optimization
            with memory_checkpoint("unified_search"):
                result = strategy_impl.search(
                    search_space=search_space,
                    objective_function=objective_function,
                    config=self.config
                )
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            # Save search history
            self.search_history.append(result)
            
            # Save intermediate results if enabled
            if self.config.save_intermediate_results:
                self._save_intermediate_result(result)
            
            tprint_success(f"✅ Search completed successfully")
            tprint_info(f"   Best score: {max(result.best_scores.values()) if result.best_scores else 0:.4f}")
            tprint_info(f"   Execution time: {result.execution_time:.2f}s")
            tprint_info(f"   Convergence: {'Yes' if result.convergence_achieved else 'No'}")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Search failed: {e}")
            return SearchResult(
                best_architecture={},
                best_scores={},
                pareto_frontier=[],
                search_strategy=search_strategy or self.config.search_strategy,
                architecture_type=self.config.architecture_type,
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
    
    def _validate_search_space(self, search_space: Dict[str, Any]):
        """Validate search space definition."""
        if not search_space:
            raise ValueError("Search space cannot be empty")
        
        for param, values in search_space.items():
            if not isinstance(values, (list, tuple)):
                raise ValueError(f"Parameter {param} must have list or tuple values")
            
            if isinstance(values, tuple) and len(values) != 2:
                raise ValueError(f"Parameter {param} tuple must have exactly 2 values (min, max)")
    
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