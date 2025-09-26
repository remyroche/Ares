"""
Evolutionary Search for Neural Architecture Search (NAS)

This module provides comprehensive evolutionary algorithms for neural architecture search,
leveraging advanced optimization techniques and hardware-specific optimizations.

Key Features:
- Population-based evolutionary search with genetic operators
- M1 hardware optimization integration
- Advanced fitness evaluation with cross-validation
- Parallel processing and memory optimization
- Integration with ML common utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
import random
import concurrent.futures
from pathlib import Path

# Import utility modules
try:
    from src.utils.nas_tas.shared_utils.common_operations_bridge import (
        safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
        calculate_data_quality_metrics, safe_merge_dataframes, create_summary_statistics,
        safe_drop_columns, safe_rename_columns, validate_timestamp_column,
        safe_timestamp_conversion, get_dataframe_info, safe_filter_dataframe,
        create_data_quality_report, optimize_dataframe_dtypes, safe_to_parquet,
        safe_read_parquet, list_parquet_files, get_memory_usage, optimize_memory,
        memory_checkpoint, gpu_context, integrate_with_m1_optimizers,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        cleanup_m1_optimizers, CommonUtilities
    )
except ImportError:
    # Fallback implementations
    def safe_dataframe_operation(df, operation, *args, **kwargs):
        return operation(df, *args, **kwargs)
    def validate_dataframe_columns(df, required_columns):
        return all(col in df.columns for col in required_columns)
    def get_memory_usage():
        return 0.0
    def optimize_memory():
        return {'success': True}
    def memory_checkpoint(name):
        from contextlib import contextmanager
        return contextmanager(lambda: (yield))
    def gpu_context(name):
        from contextlib import contextmanager
        return contextmanager(lambda: (yield))

try:
    from src.utils.nas_tas.shared_utils.math_validation_bridge import (
        safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
        validate_positive, validate_range, safe_correlation, safe_covariance,
        safe_mean, safe_std, safe_percentile, MathValidation
    )
except ImportError:
    # Fallback math functions
    def safe_divide(a, b, default=0.0):
        return a / b if b != 0 else default
    def safe_log(x, default=0.0):
        return np.log(x) if x > 0 else default
    def validate_finite(value, name="value"):
        return float(value)
    def safe_mean(x, default=0.0):
        return np.mean(x) if len(x) > 0 else default
    def safe_std(x, default=0.0):
        return np.std(x) if len(x) > 1 else default

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_debug, tprint_performance, tprint_progress, tprint_structured,
        tprint_timer, LogLevel, TPrintConfig, configure_tprint
    )
except ImportError:
    # Fallback logging
    def tprint(*args, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}]", *args)
    def tprint_info(*args, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}] INFO:", *args)
    def tprint_warning(*args, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}] WARNING:", *args)
    def tprint_error(*args, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}] ERROR:", *args)
    def tprint_success(*args, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}] SUCCESS:", *args)

# Import ML common utilities
try:
    from src.utils.nas_tas.shared_utils.ml_common_bridge import (
        BayesianTPEOptimizer, BayesianTPEConfig, optimize_with_bayesian_tpe
    )
except ImportError:
    class BayesianTPEOptimizer:
        def __init__(self, config=None):
            self.config = config
        def optimize(self, objective_function, search_space, **kwargs):
            return {'success': False, 'best_params': {}, 'best_score': 0.0}
    
    class BayesianTPEConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def optimize_with_bayesian_tpe(objective_function, search_space, config=None, **kwargs):
        return {'success': False, 'best_params': {}, 'best_score': 0.0}

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class EvolutionaryConfig:
    """Configuration for evolutionary search."""
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    tournament_size: int = 3
    selection_pressure: float = 2.0
    diversity_weight: float = 0.1
    early_stopping_patience: int = 20
    convergence_threshold: float = 1e-6
    min_fitness_threshold: Optional[float] = None
    n_workers: int = 4
    use_parallel_evaluation: bool = True


class EvolutionaryTreeSearch:
    """Evolutionary search for tree architectures."""
    
    def __init__(self, config: EvolutionaryConfig):
        self.config = config
        self.population = []
        self.fitness_scores = []
        self.generation = 0
        self.param_metadata: Dict[str, Dict[str, Any]] = {}
        self.best_individual = None
        self.best_score = -np.inf
        self.fitness_history = []
        self.diversity_history = []
        
        # Hardware optimization
        try:
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
        except:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
        
        tprint_info("🧬 EvolutionaryTreeSearch initialized")
    
    def search(self, search_space: Dict[str, Any], objective_function: Callable) -> Dict[str, Any]:
        """Perform evolutionary search for optimal tree architecture."""
        tprint_info("🚀 Starting evolutionary tree search")
        
        # Initialize population
        self.param_metadata = self._build_param_metadata(search_space)
        self._initialize_population(search_space)
        
        # Evolution loop
        for generation in range(self.config.generations):
            tprint_info(f"🔄 Generation {generation + 1}/{self.config.generations}")
            
            # Evaluate fitness
            self._evaluate_population(objective_function)
            
            # Update best individual
            current_best_idx = np.argmax(self.fitness_scores)
            current_best_score = self.fitness_scores[current_best_idx]
            
            if current_best_score > self.best_score:
                self.best_score = current_best_score
                self.best_individual = self.population[current_best_idx].copy()
                tprint_success(f"🏆 New best score: {self.best_score:.4f}")
            
            # Track metrics
            avg_fitness = np.mean(self.fitness_scores)
            diversity = self._calculate_diversity()
            
            self.fitness_history.append(avg_fitness)
            self.diversity_history.append(diversity)
            
            tprint_info(f"📊 Avg fitness: {avg_fitness:.4f}, Diversity: {diversity:.4f}")
            
            # Check convergence
            if self._check_convergence():
                tprint_info(f"🎯 Convergence reached at generation {generation + 1}")
                break
            
            # Create next generation
            if generation < self.config.generations - 1:
                self._create_next_generation()
            
            # Memory optimization
            if generation % 5 == 0:
                try:
                    optimize_memory()
                except (RuntimeError, OSError, AttributeError) as e:
                    # Handle memory optimization failures
                    tprint_warning(f"Memory optimization failed during evolution: {e}")
                    pass
        
        tprint_success(f"✅ Evolution completed - Best score: {self.best_score:.4f}")
        return self.best_individual
    
    def _initialize_population(self, search_space: Dict[str, Any]):
        """Initialize random population."""
        self.population = []
        for _ in range(self.config.population_size):
            individual = {}
            for param in search_space:
                individual[param] = self._sample_parameter(param)
            self.population.append(individual)
    
    def _evaluate_population(self, objective_function: Callable):
        """Evaluate fitness of population."""
        if self.config.use_parallel_evaluation and len(self.population) > 1:
            self._evaluate_population_parallel(objective_function)
        else:
            self._evaluate_population_sequential(objective_function)
    
    def _evaluate_population_sequential(self, objective_function: Callable):
        """Sequential evaluation of population."""
        self.fitness_scores = []
        for i, individual in enumerate(self.population):
            try:
                fitness = objective_function(individual)
                self.fitness_scores.append(fitness)
                tprint_progress(i + 1, len(self.population), f"Fitness: {fitness:.4f}")
            except Exception as e:
                tprint_warning(f"⚠️ Evaluation failed for individual {i}: {e}")
                self.fitness_scores.append(0.0)
    
    def _evaluate_population_parallel(self, objective_function: Callable):
        """Parallel evaluation of population."""
        self.fitness_scores = []
        
        try:
            max_workers = self.config.n_workers if self.cpu_optimizer is None else self.cpu_optimizer.get_optimal_worker_count()
        except:
            max_workers = self.config.n_workers
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(objective_function, individual): i 
                for i, individual in enumerate(self.population)
            }
            
            # Initialize scores
            self.fitness_scores = [0.0] * len(self.population)
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    fitness = future.result()
                    self.fitness_scores[idx] = fitness
                    completed += 1
                    tprint_progress(completed, len(self.population), f"Fitness: {fitness:.4f}")
                except Exception as e:
                    tprint_warning(f"⚠️ Evaluation failed for individual {idx}: {e}")
                    self.fitness_scores[idx] = 0.0
    
    def _create_next_generation(self):
        """Create next generation through selection, crossover, and mutation."""
        # Sort population by fitness
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        sorted_population = [self.population[i] for i in sorted_indices]
        sorted_fitness = [self.fitness_scores[i] for i in sorted_indices]
        
        # Keep elite individuals
        elite_size = min(self.config.elite_size, len(sorted_population))
        new_population = sorted_population[:elite_size]
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Select parents
            parent1 = self._tournament_selection(sorted_population, sorted_fitness)
            parent2 = self._tournament_selection(sorted_population, sorted_fitness)
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.config.mutation_rate:
                child2 = self._mutate(child2)

            new_population.extend([
                self._enforce_constraints(child1),
                self._enforce_constraints(child2)
            ])

        # Trim to population size
        self.population = new_population[:self.config.population_size]

    def _tournament_selection(self, population: List[Dict], fitness_scores: List[float]) -> Dict[str, Any]:
        """Tournament selection."""
        tournament_size = min(self.config.tournament_size, len(population))
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform crossover between two parents."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Single-point crossover
        keys = list(parent1.keys())
        if len(keys) > 1:
            crossover_point = random.randint(1, len(keys))
            for i in range(crossover_point):
                child1[keys[i]], child2[keys[i]] = child2[keys[i]], child1[keys[i]]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mutation to individual."""
        mutated = individual.copy()
        key = random.choice(list(mutated.keys()))

        meta = self.param_metadata.get(key, {})
        value = mutated[key]

        if meta.get("type") == "categorical":
            choices = [choice for choice in meta.get("values", []) if choice != value]
            if choices:
                mutated[key] = random.choice(choices)
        elif meta.get("type") == "integer":
            min_val = meta.get("min", value)
            max_val = meta.get("max", value)
            span = max(1, max_val - min_val)
            window = max(1, int(round(0.1 * span)))
            lower = max(min_val, int(round(value)) - window)
            upper = min(max_val, int(round(value)) + window)
            mutated[key] = random.randint(lower, upper)
        elif meta.get("type") == "float":
            min_val = float(meta.get("min", value))
            max_val = float(meta.get("max", value))
            span = max(1e-9, max_val - min_val)
            delta = span * 0.1
            mutated_value = float(value) + random.uniform(-delta, delta)
            mutated[key] = float(np.clip(mutated_value, min_val, max_val))

        return mutated

    def _build_param_metadata(self, search_space: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Build metadata describing parameter domains."""
        metadata: Dict[str, Dict[str, Any]] = {}
        for param, values in search_space.items():
            if isinstance(values, list):
                metadata[param] = {"type": "categorical", "values": list(values)}
            elif isinstance(values, tuple) and len(values) == 2:
                low, high = values
                if all(isinstance(v, int) for v in values):
                    metadata[param] = {"type": "integer", "min": int(low), "max": int(high)}
                else:
                    metadata[param] = {"type": "float", "min": float(low), "max": float(high)}
            else:
                metadata[param] = {"type": "fixed", "value": values}
        return metadata

    def _sample_parameter(self, name: str) -> Any:
        """Sample a parameter value based on metadata."""
        meta = self.param_metadata.get(name, {})
        if meta.get("type") == "categorical":
            return random.choice(meta.get("values", []))
        if meta.get("type") == "integer":
            return random.randint(meta.get("min", 0), meta.get("max", 1))
        if meta.get("type") == "float":
            return random.uniform(meta.get("min", 0.0), meta.get("max", 1.0))
        return meta.get("value")

    def _enforce_constraints(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure individuals remain inside the feasible domain."""
        constrained = individual.copy()
        for param, meta in self.param_metadata.items():
            if param not in constrained:
                continue

            if meta.get("type") == "integer":
                value = int(round(constrained[param]))
                constrained[param] = int(np.clip(value, meta.get("min", value), meta.get("max", value)))
            elif meta.get("type") == "float":
                value = float(constrained[param])
                constrained[param] = float(np.clip(value, meta.get("min", value), meta.get("max", value)))
            elif meta.get("type") == "categorical":
                choices = meta.get("values", [])
                if choices and constrained[param] not in choices:
                    constrained[param] = random.choice(choices)
            elif meta.get("type") == "fixed":
                constrained[param] = meta.get("value")

        return constrained
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0
        
        # Simple diversity metric based on parameter differences
        total_differences = 0
        comparisons = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                individual1, individual2 = self.population[i], self.population[j]
                differences = 0
                
                for key in individual1.keys():
                    if key in individual2:
                        if isinstance(individual1[key], (int, float)) and isinstance(individual2[key], (int, float)):
                            differences += abs(individual1[key] - individual2[key])
                        elif individual1[key] != individual2[key]:
                            differences += 1
                
                total_differences += differences
                comparisons += 1
        
        return total_differences / comparisons if comparisons > 0 else 0.0
    
    def _check_convergence(self) -> bool:
        """Check if the algorithm has converged."""
        if len(self.fitness_history) < self.config.early_stopping_patience:
            return False

        # Check if fitness has improved significantly in recent generations
        recent_fitness = self.fitness_history[-self.config.early_stopping_patience:]
        improvement = max(recent_fitness) - min(recent_fitness)
        has_stalled = improvement < self.config.convergence_threshold

        threshold = self.config.min_fitness_threshold
        meets_absolute = True
        if threshold is not None:
            meets_absolute = max(self.best_score, max(recent_fitness)) >= threshold

        return has_stalled and meets_absolute


class TreeGeneticAlgorithm:
    """Tree genetic algorithm for architecture search."""
    
    def __init__(self, config: EvolutionaryConfig):
        self.config = config
        self.population = []
        self.fitness_scores = []
        self.generation = 0
        self.best_individual = None
        self.best_score = -np.inf
        self.param_metadata: Dict[str, Dict[str, Any]] = {}
    
    def search(self, search_space: Dict[str, Any], objective_function: Callable) -> Dict[str, Any]:
        """Perform genetic algorithm search for optimal tree architecture."""
        tprint_info("🧬 Starting tree genetic algorithm search")
        
        # Initialize population
        self.param_metadata = self._build_param_metadata(search_space)
        self._initialize_population(search_space)
        
        # Evolution loop
        for generation in range(self.config.generations):
            # Evaluate fitness
            self._evaluate_population(objective_function)
            
            # Update best
            current_best_idx = np.argmax(self.fitness_scores)
            current_best_score = self.fitness_scores[current_best_idx]
            
            if current_best_score > self.best_score:
                self.best_score = current_best_score
                self.best_individual = self.population[current_best_idx].copy()
            
            tprint_info(f"Generation {generation + 1}: Best fitness = {self.best_score:.4f}")
            
            # Create next generation
            if generation < self.config.generations - 1:
                self._create_next_generation()
        
        return self.best_individual
    
    def _initialize_population(self, search_space: Dict[str, Any]):
        """Initialize population with random individuals."""
        self.population = []
        for _ in range(self.config.population_size):
            individual = self._create_random_individual(search_space)
            self.population.append(individual)

    def _create_random_individual(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Create a random individual from search space."""
        return {param: self._sample_parameter(param) for param in search_space}
    
    def _evaluate_population(self, objective_function: Callable):
        """Evaluate fitness of all individuals in population."""
        self.fitness_scores = []
        for individual in self.population:
            try:
                fitness = objective_function(individual)
                self.fitness_scores.append(fitness)
            except Exception as e:
                tprint_warning(f"⚠️ Evaluation failed: {e}")
                self.fitness_scores.append(0.0)
    
    def _create_next_generation(self):
        """Create next generation through selection, crossover, and mutation."""
        # Sort by fitness
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        sorted_population = [self.population[i] for i in sorted_indices]

        # Keep elite
        elite_size = min(self.config.elite_size, len(sorted_population))
        new_population = [self._enforce_constraints(ind.copy()) for ind in sorted_population[:elite_size]]

        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Select parents
            parent1 = self._tournament_selection(sorted_population)
            parent2 = self._tournament_selection(sorted_population)

            # Crossover
            if random.random() < self.config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            new_population.extend([
                self._enforce_constraints(child1),
                self._enforce_constraints(child2)
            ])

        # Mutation
        for i in range(elite_size, len(new_population)):
            if random.random() < self.config.mutation_rate:
                new_population[i] = self._enforce_constraints(self._mutate(new_population[i]))
            else:
                new_population[i] = self._enforce_constraints(new_population[i])

        self.population = new_population[:self.config.population_size]
    
    def _tournament_selection(self, population: List[Dict]) -> Dict[str, Any]:
        """Tournament selection."""
        tournament_size = min(self.config.tournament_size, len(population))
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform crossover between two parents."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Single-point crossover
        crossover_point = random.randint(1, len(parent1))
        keys = list(parent1.keys())
        
        for i, key in enumerate(keys):
            if i < crossover_point:
                child1[key], child2[key] = child2[key], child1[key]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate an individual."""
        mutated = individual.copy()
        for key in list(mutated.keys()):
            if random.random() < 0.1:
                mutated[key] = self._mutate_value(key, mutated[key])
        return mutated

    def _mutate_value(self, key: str, value: Any) -> Any:
        meta = self.param_metadata.get(key, {})
        if meta.get("type") == "categorical":
            choices = [choice for choice in meta.get("values", []) if choice != value]
            return random.choice(choices) if choices else value
        if meta.get("type") == "integer":
            min_val = meta.get("min", value)
            max_val = meta.get("max", value)
            span = max(1, max_val - min_val)
            window = max(1, int(round(0.1 * span)))
            lower = max(min_val, int(round(value)) - window)
            upper = min(max_val, int(round(value)) + window)
            return random.randint(lower, upper)
        if meta.get("type") == "float":
            min_val = float(meta.get("min", value))
            max_val = float(meta.get("max", value))
            span = max(1e-9, max_val - min_val)
            delta = span * 0.1
            mutated_value = float(value) + random.uniform(-delta, delta)
            return float(np.clip(mutated_value, min_val, max_val))
        if meta.get("type") == "fixed":
            return meta.get("value", value)
        return value

    def _build_param_metadata(self, search_space: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        metadata: Dict[str, Dict[str, Any]] = {}
        for param, values in search_space.items():
            if isinstance(values, list):
                metadata[param] = {"type": "categorical", "values": list(values)}
            elif isinstance(values, tuple) and len(values) == 2:
                low, high = values
                if all(isinstance(v, int) for v in values):
                    metadata[param] = {"type": "integer", "min": int(low), "max": int(high)}
                else:
                    metadata[param] = {"type": "float", "min": float(low), "max": float(high)}
            else:
                metadata[param] = {"type": "fixed", "value": values}
        return metadata

    def _sample_parameter(self, name: str) -> Any:
        meta = self.param_metadata.get(name, {})
        if meta.get("type") == "categorical":
            return random.choice(meta.get("values", []))
        if meta.get("type") == "integer":
            return random.randint(meta.get("min", 0), meta.get("max", 1))
        if meta.get("type") == "float":
            return random.uniform(meta.get("min", 0.0), meta.get("max", 1.0))
        return meta.get("value")

    def _enforce_constraints(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        constrained = individual.copy()
        for param, meta in self.param_metadata.items():
            if param not in constrained:
                continue

            if meta.get("type") == "integer":
                value = int(round(constrained[param]))
                constrained[param] = int(np.clip(value, meta.get("min", value), meta.get("max", value)))
            elif meta.get("type") == "float":
                value = float(constrained[param])
                constrained[param] = float(np.clip(value, meta.get("min", value), meta.get("max", value)))
            elif meta.get("type") == "categorical":
                choices = meta.get("values", [])
                if choices and constrained[param] not in choices:
                    constrained[param] = random.choice(choices)
            elif meta.get("type") == "fixed":
                constrained[param] = meta.get("value")

        return constrained


class TreeNSGA2:
    """Tree NSGA-II algorithm for multi-objective optimization."""

    def __init__(self, config: EvolutionaryConfig):
        self.config = config
        self.population: List[Dict[str, Any]] = []
        self.objective_values: List[List[float]] = []
        self.objective_names: List[str] = []
        self.rank_map: Dict[int, int] = {}
        self.crowding_map: Dict[int, float] = {}

    def search(self, search_space: Dict[str, Any],
               objective_function: Callable[[Dict[str, Any]], Union[Dict[str, float], List[float], Tuple[float, ...]]]
               ) -> List[Dict[str, Any]]:
        """Perform NSGA-II search for optimal tree architectures."""

        if not callable(objective_function):
            raise ValueError("objective_function must be callable")

        population_size = max(2, self.config.population_size)
        tprint_info("🧬 Starting tree NSGA-II search")
        

        # Initialize population and evaluate
        self.population = [self._create_random_individual(search_space) for _ in range(population_size)]
        self.objective_values = [self._evaluate_individual(individual, objective_function) for individual in self.population]
        self._update_rankings()

        for generation in range(self.config.generations):
            offspring = self._generate_offspring(search_space)
            offspring_values = [self._evaluate_individual(child, objective_function) for child in offspring]

            combined_population = self.population + offspring
            combined_values = self.objective_values + offspring_values

            fronts, _ = self._fast_non_dominated_sort(combined_values)
            crowding_map = self._compute_crowding_distances(fronts, combined_values)

            self.population, self.objective_values = self._select_next_population(
                combined_population,
                combined_values,
                fronts,
                crowding_map,
                population_size
            )

            self._update_rankings()

            tprint_info(f"Generation {generation + 1}/{self.config.generations} completed")

        # Final Pareto front from last population
        fronts, _ = self._fast_non_dominated_sort(self.objective_values)
        pareto_front = fronts[0] if fronts else []

        results = []
        for idx in pareto_front:
            results.append({
                'params': self.population[idx],
                'objectives': self._format_objectives(self.objective_values[idx])
            })

        return results

    def _format_objectives(self, values: List[float]) -> Dict[str, float]:
        if not self.objective_names:
            self.objective_names = [f"objective_{i}" for i in range(len(values))]
        return {name: float(val) for name, val in zip(self.objective_names, values)}

    def _create_random_individual(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Create a random individual from search space."""
        return {param: self._sample_parameter(param) for param in search_space}

    def _build_param_metadata(self, search_space: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        metadata: Dict[str, Dict[str, Any]] = {}
        for param, values in search_space.items():
            if isinstance(values, list):
                metadata[param] = {"type": "categorical", "values": list(values)}
            elif isinstance(values, tuple) and len(values) == 2:
                low, high = values
                if all(isinstance(v, int) for v in values):
                    metadata[param] = {"type": "integer", "min": int(low), "max": int(high)}
                else:
                    metadata[param] = {"type": "float", "min": float(low), "max": float(high)}
            else:
                metadata[param] = {"type": "fixed", "value": values}
        return metadata

    def _sample_parameter(self, name: str) -> Any:
        meta = self.param_metadata.get(name, {})
        if meta.get("type") == "categorical":
            return random.choice(meta.get("values", []))
        if meta.get("type") == "integer":
            return random.randint(meta.get("min", 0), meta.get("max", 1))
        if meta.get("type") == "float":
            return random.uniform(meta.get("min", 0.0), meta.get("max", 1.0))
        return meta.get("value")
    
    def _evaluate_population(self, objectives: List[str]):
        """Evaluate fitness of all individuals in population."""
        self.fitness_scores = []
        for individual in self.population:
            fitness = self._evaluate_individual(individual, objectives)
            self.fitness_scores.append(fitness)
    
    def _evaluate_individual(self, individual: Dict[str, Any], objectives: List[str]) -> List[float]:
        """Evaluate fitness of a single individual for multiple objectives."""
        # Placeholder implementation - should be replaced with actual evaluation
        return [random.random() for _ in objectives]
    
    def _nsga2_operations(self):
        """Perform NSGA-II operations."""
        # Simplified implementation
        pass
    
    def _get_pareto_front(self) -> List[Dict[str, Any]]:
        """Get Pareto front of non-dominated solutions."""
        # Simplified implementation
        return self.population[:10]  # Return top 10 individuals
        individual = {}
        for param, config in search_space.items():
            individual[param] = self._sample_from_space(config)
        return individual

    def _sample_from_space(self, config: Any) -> Any:
        if isinstance(config, dict):
            param_type = config.get('type', 'float')
            if param_type == 'int':
                return int(np.random.randint(config['low'], config['high'] + 1))
            if param_type == 'float':
                if config.get('log'):
                    low = np.log(config['low'])
                    high = np.log(config['high'])
                    return float(np.exp(np.random.uniform(low, high)))
                return float(np.random.uniform(config['low'], config['high']))
            if param_type == 'categorical':
                return np.random.choice(config['choices'])
        elif isinstance(config, list):
            return np.random.choice(config)
        elif isinstance(config, tuple) and len(config) == 2:
            low, high = config
            if all(isinstance(v, int) for v in config):
                return int(np.random.randint(low, high + 1))
            return float(np.random.uniform(low, high))
        return config

    def _evaluate_individual(self, individual: Dict[str, Any], objective_function: Callable) -> List[float]:
        try:
            values = objective_function(individual)
            if isinstance(values, dict):
                if not self.objective_names:
                    self.objective_names = list(values.keys())
                return [float(values[name]) for name in self.objective_names]
            values_list = list(values)
            if not self.objective_names:
                self.objective_names = [f"objective_{i}" for i in range(len(values_list))]
            return [float(v) for v in values_list]
        except Exception as e:
            tprint_warning(f"⚠️ Objective evaluation failed: {e}")
            return [float('-inf')] * (len(self.objective_names) if self.objective_names else 2)  # Assume at least 2 objectives for NSGA-II

    def _generate_offspring(self, search_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        offspring = []
        while len(offspring) < self.config.population_size:
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            child1, child2 = self._crossover(parent1, parent2)
            child1 = self._mutate(child1, search_space)
            child2 = self._mutate(child2, search_space)

            offspring.extend([child1, child2])

        return offspring[:self.config.population_size]

    def _tournament_selection(self) -> Dict[str, Any]:
        candidates = np.random.choice(len(self.population), size=2, replace=True)
        idx1, idx2 = int(candidates[0]), int(candidates[1])
        return self._better_individual(idx1, idx2)

    def _better_individual(self, idx1: int, idx2: int) -> Dict[str, Any]:
        rank1 = self.rank_map.get(idx1, 0)
        rank2 = self.rank_map.get(idx2, 0)
        if rank1 < rank2:
            return self.population[idx1]
        if rank2 < rank1:
            return self.population[idx2]
        crowd1 = self.crowding_map.get(idx1, 0.0)
        crowd2 = self.crowding_map.get(idx2, 0.0)
        if crowd1 > crowd2:
            return self.population[idx1]
        return self.population[idx2]

    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if np.random.rand() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()
        keys = list(parent1.keys())
        for key in keys:
            if np.random.rand() < 0.5:
                child1[key], child2[key] = child2[key], child1[key]
        return child1, child2

    def _mutate(self, individual: Dict[str, Any], search_space: Dict[str, Any]) -> Dict[str, Any]:
        mutated = individual.copy()
        for key in mutated.keys():
            if np.random.rand() < self.config.mutation_rate:
                mutated[key] = self._mutate_value(mutated[key], search_space.get(key))
        return mutated

    def _mutate_value(self, value: Any, config: Any) -> Any:
        if isinstance(config, dict):
            if config['type'] == 'int':
                step = max(1, (config['high'] - config['low']) // 10)
                return int(np.clip(value + np.random.randint(-step, step + 1), config['low'], config['high']))
            if config['type'] == 'float':
                range_width = config['high'] - config['low']
                return float(np.clip(value + np.random.uniform(-0.1 * range_width, 0.1 * range_width),
                                      config['low'], config['high']))
            if config['type'] == 'categorical':
                choices = [choice for choice in config['choices'] if choice != value]
                return np.random.choice(choices) if choices else value
        elif isinstance(config, list):
            choices = [choice for choice in config if choice != value]
            return np.random.choice(choices) if choices else value
        elif isinstance(config, tuple) and len(config) == 2:
            low, high = config
            if all(isinstance(v, int) for v in config):
                step = max(1, (high - low) // 10 or 1)
                return int(np.clip(value + np.random.randint(-step, step + 1), low, high))
            range_width = high - low
            return float(np.clip(value + np.random.uniform(-0.1 * range_width, 0.1 * range_width), low, high))
        return value

    def _update_rankings(self):
        fronts, rank_map = self._fast_non_dominated_sort(self.objective_values)
        crowding_map = self._compute_crowding_distances(fronts, self.objective_values)
        self.rank_map = {idx: rank_map[idx] for idx in range(len(self.population))}
        self.crowding_map = crowding_map

    def _fast_non_dominated_sort(self, values: List[List[float]]) -> Tuple[List[List[int]], Dict[int, int]]:
        S = {i: [] for i in range(len(values))}
        domination_count = {i: 0 for i in range(len(values))}
        fronts: List[List[int]] = [[]]
        rank_map: Dict[int, int] = {}

        for p in range(len(values)):
            S[p] = []
            domination_count[p] = 0
            for q in range(len(values)):
                if p == q:
                    continue
                if self._dominates(values[p], values[q]):
                    S[p].append(q)
                elif self._dominates(values[q], values[p]):
                    domination_count[p] += 1
            if domination_count[p] == 0:
                rank_map[p] = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        rank_map[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        if not fronts[-1]:
            fronts.pop()

        return fronts, rank_map

    def _compute_crowding_distances(self, fronts: List[List[int]], values: List[List[float]]) -> Dict[int, float]:
        distances: Dict[int, float] = {idx: 0.0 for idx in range(len(values))}
        for front in fronts:
            if len(front) <= 2:
                for idx in front:
                    distances[idx] = float('inf')
                continue

            front_values = [values[idx] for idx in front]
            n_objectives = len(front_values[0])
            for m in range(n_objectives):
                sorted_indices = sorted(front, key=lambda idx: values[idx][m])
                distances[sorted_indices[0]] = float('inf')
                distances[sorted_indices[-1]] = float('inf')

                objective_min = values[sorted_indices[0]][m]
                objective_max = values[sorted_indices[-1]][m]
                if objective_max == objective_min:
                    continue

                for idx in range(1, len(sorted_indices) - 1):
                    prev_val = values[sorted_indices[idx - 1]][m]
                    next_val = values[sorted_indices[idx + 1]][m]
                    distances[sorted_indices[idx]] += (next_val - prev_val) / (objective_max - objective_min)

        return distances

    def _select_next_population(self,
                                combined_population: List[Dict[str, Any]],
                                combined_values: List[List[float]],
                                fronts: List[List[int]],
                                crowding_map: Dict[int, float],
                                population_size: int) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
        new_population: List[Dict[str, Any]] = []
        new_values: List[List[float]] = []

        for front in fronts:
            if len(new_population) + len(front) <= population_size:
                for idx in front:
                    new_population.append(combined_population[idx])
                    new_values.append(combined_values[idx])
            else:
                remaining = population_size - len(new_population)
                sorted_front = sorted(front, key=lambda idx: crowding_map.get(idx, 0.0), reverse=True)
                for idx in sorted_front[:remaining]:
                    new_population.append(combined_population[idx])
                    new_values.append(combined_values[idx])
                break

        return new_population, new_values

    def _dominates(self, values_p: List[float], values_q: List[float]) -> bool:
        better_or_equal = all(p >= q for p, q in zip(values_p, values_q))
        strictly_better = any(p > q for p, q in zip(values_p, values_q))
        return better_or_equal and strictly_better


# Export main classes
__all__ = [
    'EvolutionaryTreeSearch',
    'TreeGeneticAlgorithm', 
    'TreeNSGA2',
    'EvolutionaryConfig'
]