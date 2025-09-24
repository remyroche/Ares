"""
Multi-Objective Optimization for Neural Architecture Search

This module provides NSGA-II optimization and NAS objectives for multi-objective
neural architecture search, integrating with existing utility modules.

Key Features:
- NSGA-II multi-objective optimization
- Neural Architecture Search (NAS) objectives
- Integration with M1 hardware optimizations
- Pareto front analysis and knee point selection
- Comprehensive evaluation metrics
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import time
from datetime import datetime
import json
from pathlib import Path

# Import existing utilities
try:
    from src.utils.common_operations import (
        safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
        safe_mean, safe_std, safe_percentage_change, safe_weighted_average,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, memory_checkpoint, gpu_context
    )
    COMMON_OPS_AVAILABLE = True
except ImportError:
    COMMON_OPS_AVAILABLE = False

try:
    from src.utils.math_validation import (
        validate_finite, validate_positive, validate_range,
        safe_correlation, safe_covariance, safe_mean, safe_std
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError:
    MATH_VALIDATION_AVAILABLE = False

try:
    from src.utils.tprint import tprint, tprint_info, tprint_error, tprint_success
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)

try:
    from src.utils.ml_common.optimization.pareto import (
        Solution, ParetoFront, compute_pareto_front, select_knee_point,
        compute_hypervolume, scalarize_financial_goals, filter_by_constraints
    )
    PARETO_AVAILABLE = True
except ImportError:
    PARETO_AVAILABLE = False

try:
    from src.utils.ml_common.evaluation.unified_evaluator import (
        compute_classification_metrics, compute_regression_metrics,
        evaluate_model, compute_sharpe_ratio
    )
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class NSGAIIConfig:
    """Configuration for NSGA-II optimization."""
    
    # Population parameters
    population_size: int = 100
    max_generations: int = 50
    crossover_probability: float = 0.9
    mutation_probability: float = 0.1
    
    # Selection parameters
    tournament_size: int = 2
    crowding_distance_alpha: float = 0.1
    
    # Convergence parameters
    convergence_threshold: float = 1e-6
    max_stagnation_generations: int = 10
    
    # Hardware optimization
    use_m1_optimization: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    
    # Memory management
    memory_limit_gb: float = 8.0
    enable_memory_checkpointing: bool = True

@dataclass
class NASObjective:
    """Neural Architecture Search objective definition."""
    
    name: str
    weight: float = 1.0
    direction: str = 'maximize'  # 'maximize' or 'minimize'
    target_value: Optional[float] = None
    threshold: Optional[float] = None
    
    def evaluate(self, metrics: Dict[str, float]) -> float:
        """Evaluate objective from metrics dictionary."""
        if self.name not in metrics:
            return 0.0
        
        value = metrics[self.name]
        
        # Apply direction
        if self.direction == 'minimize':
            value = -value
        
        # Apply weight
        return value * self.weight

@dataclass
class Individual:
    """Individual in the NSGA-II population."""
    
    # Architecture parameters
    architecture: Dict[str, Any]
    
    # Objectives
    objectives: List[float] = field(default_factory=list)
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # NSGA-II specific
    rank: int = 0
    crowding_distance: float = 0.0
    dominated_count: int = 0
    dominated_solutions: List[int] = field(default_factory=list)
    
    # Metadata
    generation: int = 0
    evaluation_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

class NSGAIIOptimizer:
    """NSGA-II Multi-Objective Optimizer for Neural Architecture Search."""
    
    def __init__(self, config: Optional[NSGAIIConfig] = None):
        """Initialize NSGA-II optimizer."""
        self.config = config or NSGAIIConfig()
        self.logger = logger.getChild('NSGAIIOptimizer')
        
        # Initialize hardware optimizations
        self.m1_integration = None
        if self.config.use_m1_optimization and COMMON_OPS_AVAILABLE:
            try:
                self.m1_integration = integrate_with_m1_optimizers()
                tprint_success("✅ M1 optimization integration successful")
            except Exception as e:
                tprint_error(f"❌ M1 optimization integration failed: {e}")
        
        # Population tracking
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individuals: List[Individual] = []
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Pareto front
        self.pareto_front: List[Individual] = []
        self.pareto_front_history: List[List[Individual]] = []
        
        tprint_info(f"🚀 NSGA-II Optimizer initialized with population size {self.config.population_size}")
    
    def optimize(self, 
                 architecture_space: Dict[str, Any],
                 objectives: List[NASObjective],
                 evaluation_function: Callable[[Dict[str, Any]], Dict[str, float]],
                 initial_population: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Perform NSGA-II multi-objective optimization.
        
        Args:
            architecture_space: Search space for neural architectures
            objectives: List of objectives to optimize
            evaluation_function: Function to evaluate architecture performance
            initial_population: Optional initial population
            
        Returns:
            Optimization results dictionary
        """
        tprint_info("🎯 Starting NSGA-II multi-objective optimization")
        start_time = time.time()
        
        try:
            # Initialize population
            self._initialize_population(architecture_space, initial_population)
            
            # Main optimization loop
            for generation in range(self.config.max_generations):
                self.generation = generation
                tprint_info(f"🔄 Generation {generation + 1}/{self.config.max_generations}")
                
                # Evaluate population
                self._evaluate_population(objectives, evaluation_function)
                
                # Apply NSGA-II operations
                self._apply_nsga_ii_operations()
                
                # Update Pareto front
                self._update_pareto_front()
                
                # Check convergence
                if self._check_convergence():
                    tprint_success(f"✅ Optimization converged at generation {generation + 1}")
                    break
                
                # Record generation statistics
                self._record_generation_stats()
            
            # Final results
            optimization_time = time.time() - start_time
            results = self._compile_results(optimization_time)
            
            tprint_success(f"🏆 NSGA-II optimization completed in {optimization_time:.2f}s")
            return results
            
        except Exception as e:
            tprint_error(f"❌ NSGA-II optimization failed: {e}")
            self.logger.error(f"Optimization failed: {e}")
            return {'error': str(e), 'success': False}
    
    def _initialize_population(self, 
                              architecture_space: Dict[str, Any],
                              initial_population: Optional[List[Dict[str, Any]]] = None):
        """Initialize the population with random architectures."""
        tprint_info(f"🧬 Initializing population of {self.config.population_size} individuals")
        
        self.population = []
        
        if initial_population:
            # Use provided initial population
            for i, arch in enumerate(initial_population[:self.config.population_size]):
                individual = Individual(
                    architecture=arch,
                    generation=0
                )
                self.population.append(individual)
        else:
            # Generate random population
            for i in range(self.config.population_size):
                architecture = self._sample_architecture(architecture_space)
                individual = Individual(
                    architecture=architecture,
                    generation=0
                )
                self.population.append(individual)
        
        tprint_success(f"✅ Population initialized with {len(self.population)} individuals")
    
    def _sample_architecture(self, architecture_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample a random architecture from the search space."""
        architecture = {}
        
        for param_name, param_config in architecture_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get('type', 'float')
                
                if param_type == 'int':
                    low = param_config.get('low', 1)
                    high = param_config.get('high', 10)
                    architecture[param_name] = np.random.randint(low, high + 1)
                
                elif param_type == 'float':
                    low = param_config.get('low', 0.0)
                    high = param_config.get('high', 1.0)
                    architecture[param_name] = np.random.uniform(low, high)
                
                elif param_type == 'categorical':
                    choices = param_config.get('choices', [])
                    architecture[param_name] = np.random.choice(choices)
                
                elif param_type == 'list':
                    # For layer configurations
                    min_layers = param_config.get('min_layers', 1)
                    max_layers = param_config.get('max_layers', 5)
                    n_layers = np.random.randint(min_layers, max_layers + 1)
                    
                    layers = []
                    for _ in range(n_layers):
                        layer_config = self._sample_layer_config(param_config.get('layer_config', {}))
                        layers.append(layer_config)
                    
                    architecture[param_name] = layers
            
            elif isinstance(param_config, (list, tuple)):
                # Simple range specification
                if len(param_config) == 2:
                    low, high = param_config
                    if isinstance(low, int) and isinstance(high, int):
                        architecture[param_name] = np.random.randint(low, high + 1)
                    else:
                        architecture[param_name] = np.random.uniform(low, high)
        
        return architecture
    
    def _sample_layer_config(self, layer_config: Dict[str, Any]) -> Dict[str, Any]:
        """Sample a random layer configuration."""
        config = {}
        
        for param_name, param_spec in layer_config.items():
            if isinstance(param_spec, dict):
                param_type = param_spec.get('type', 'int')
                
                if param_type == 'int':
                    low = param_spec.get('low', 32)
                    high = param_spec.get('high', 512)
                    config[param_name] = np.random.randint(low, high + 1)
                
                elif param_type == 'float':
                    low = param_spec.get('low', 0.0)
                    high = param_spec.get('high', 1.0)
                    config[param_name] = np.random.uniform(low, high)
                
                elif param_type == 'categorical':
                    choices = param_spec.get('choices', ['relu'])
                    config[param_name] = np.random.choice(choices)
        
        return config
    
    def _evaluate_population(self, 
                           objectives: List[NASObjective],
                           evaluation_function: Callable[[Dict[str, Any]], Dict[str, float]]):
        """Evaluate the population using the provided evaluation function."""
        tprint_info(f"📊 Evaluating population of {len(self.population)} individuals")
        
        for i, individual in enumerate(self.population):
            if not individual.objectives:  # Skip if already evaluated
                try:
                    # Evaluate architecture
                    start_time = time.time()
                    metrics = evaluation_function(individual.architecture)
                    evaluation_time = time.time() - start_time
                    
                    individual.metrics = metrics
                    individual.evaluation_time = evaluation_time
                    
                    # Calculate objectives
                    individual.objectives = []
                    for objective in objectives:
                        obj_value = objective.evaluate(metrics)
                        individual.objectives.append(obj_value)
                    
                    if (i + 1) % 10 == 0:
                        tprint_info(f"   Evaluated {i + 1}/{len(self.population)} individuals")
                
                except Exception as e:
                    tprint_error(f"❌ Evaluation failed for individual {i}: {e}")
                    # Set default objectives for failed evaluations
                    individual.objectives = [0.0] * len(objectives)
                    individual.metrics = {}
        
        tprint_success(f"✅ Population evaluation completed")
    
    def _apply_nsga_ii_operations(self):
        """Apply NSGA-II genetic operations."""
        # Create offspring through crossover and mutation
        offspring = []
        
        while len(offspring) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if np.random.random() < self.config.crossover_probability:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            # Mutation
            if np.random.random() < self.config.mutation_probability:
                child1 = self._mutate(child1)
            if np.random.random() < self.config.mutation_probability:
                child2 = self._mutate(child2)
            
            offspring.extend([child1, child2])
        
        # Combine parent and offspring populations
        combined_population = self.population + offspring[:self.config.population_size]
        
        # Apply NSGA-II selection
        self.population = self._nsga_ii_selection(combined_population)
    
    def _tournament_selection(self) -> Individual:
        """Tournament selection for parent selection."""
        tournament_indices = np.random.choice(
            len(self.population), 
            size=self.config.tournament_size, 
            replace=False
        )
        
        tournament_individuals = [self.population[i] for i in tournament_indices]
        
        # Select best individual based on rank and crowding distance
        best_individual = tournament_individuals[0]
        for individual in tournament_individuals[1:]:
            if (individual.rank < best_individual.rank or 
                (individual.rank == best_individual.rank and 
                 individual.crowding_distance > best_individual.crowding_distance)):
                best_individual = individual
        
        return best_individual
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        # Simple uniform crossover for architecture parameters
        child1_arch = parent1.architecture.copy()
        child2_arch = parent2.architecture.copy()
        
        for param_name in parent1.architecture.keys():
            if np.random.random() < 0.5:
                # Swap parameters
                child1_arch[param_name] = parent2.architecture[param_name]
                child2_arch[param_name] = parent1.architecture[param_name]
        
        child1 = Individual(
            architecture=child1_arch,
            generation=self.generation + 1
        )
        child2 = Individual(
            architecture=child2_arch,
            generation=self.generation + 1
        )
        
        return child1, child2
    
    def _mutate(self, individual: Individual) -> Individual:
        """Apply mutation to an individual."""
        mutated_arch = individual.architecture.copy()
        
        # Random parameter mutation
        for param_name, param_value in mutated_arch.items():
            if np.random.random() < 0.1:  # 10% chance to mutate each parameter
                if isinstance(param_value, int):
                    # Integer mutation
                    mutated_arch[param_name] = max(1, param_value + np.random.randint(-2, 3))
                elif isinstance(param_value, float):
                    # Float mutation
                    noise = np.random.normal(0, 0.1)
                    mutated_arch[param_name] = max(0.0, param_value + noise)
                elif isinstance(param_value, list):
                    # List mutation (for layer configurations)
                    if len(param_value) > 0:
                        # Add or remove a layer
                        if np.random.random() < 0.5 and len(param_value) > 1:
                            # Remove a layer
                            mutated_arch[param_name] = param_value[:-1]
                        else:
                            # Add a layer
                            new_layer = self._sample_layer_config({'units': {'type': 'int', 'low': 32, 'high': 512}})
                            mutated_arch[param_name].append(new_layer)
        
        mutated_individual = Individual(
            architecture=mutated_arch,
            generation=self.generation + 1
        )
        
        return mutated_individual
    
    def _nsga_ii_selection(self, combined_population: List[Individual]) -> List[Individual]:
        """Apply NSGA-II selection to combined population."""
        # Calculate dominance relationships
        self._calculate_dominance(combined_population)
        
        # Fast non-dominated sorting
        fronts = self._fast_non_dominated_sort(combined_population)
        
        # Select individuals from fronts
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= self.config.population_size:
                selected.extend(front)
            else:
                # Calculate crowding distance for the last front
                self._calculate_crowding_distance(front)
                # Sort by crowding distance (descending)
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                # Add remaining individuals
                remaining = self.config.population_size - len(selected)
                selected.extend(front[:remaining])
                break
        
        return selected
    
    def _calculate_dominance(self, population: List[Individual]):
        """Calculate dominance relationships between individuals."""
        for i, individual in enumerate(population):
            individual.dominated_count = 0
            individual.dominated_solutions = []
            
            for j, other in enumerate(population):
                if i != j:
                    if self._dominates(individual, other):
                        individual.dominated_solutions.append(j)
                    elif self._dominates(other, individual):
                        individual.dominated_count += 1
    
    def _dominates(self, individual1: Individual, individual2: Individual) -> bool:
        """Check if individual1 dominates individual2."""
        if not individual1.objectives or not individual2.objectives:
            return False
        
        # At least one objective is better
        at_least_one_better = False
        
        for obj1, obj2 in zip(individual1.objectives, individual2.objectives):
            if obj1 < obj2:  # Assuming minimization
                return False
            elif obj1 > obj2:
                at_least_one_better = True
        
        return at_least_one_better
    
    def _fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """Fast non-dominated sorting algorithm."""
        fronts = []
        current_front = []
        
        # Find first front (non-dominated solutions)
        for individual in population:
            if individual.dominated_count == 0:
                individual.rank = 0
                current_front.append(individual)
        
        fronts.append(current_front)
        
        # Find subsequent fronts
        while current_front:
            next_front = []
            
            for individual in current_front:
                for dominated_idx in individual.dominated_solutions:
                    dominated_individual = population[dominated_idx]
                    dominated_individual.dominated_count -= 1
                    
                    if dominated_individual.dominated_count == 0:
                        dominated_individual.rank = len(fronts)
                        next_front.append(dominated_individual)
            
            if next_front:
                fronts.append(next_front)
                current_front = next_front
            else:
                break
        
        return fronts
    
    def _calculate_crowding_distance(self, front: List[Individual]):
        """Calculate crowding distance for individuals in a front."""
        if len(front) <= 2:
            for individual in front:
                individual.crowding_distance = float('inf')
            return
        
        # Initialize crowding distance
        for individual in front:
            individual.crowding_distance = 0.0
        
        # Calculate for each objective
        n_objectives = len(front[0].objectives)
        for obj_idx in range(n_objectives):
            # Sort by objective value
            front.sort(key=lambda x: x.objectives[obj_idx])
            
            # Set boundary points to infinity
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate range
            obj_values = [individual.objectives[obj_idx] for individual in front]
            obj_range = max(obj_values) - min(obj_values)
            
            if obj_range > 0:
                # Calculate crowding distance for intermediate points
                for i in range(1, len(front) - 1):
                    distance = (obj_values[i + 1] - obj_values[i - 1]) / obj_range
                    front[i].crowding_distance += distance
    
    def _update_pareto_front(self):
        """Update the Pareto front with current non-dominated solutions."""
        # Find non-dominated solutions in current population
        non_dominated = []
        for individual in self.population:
            is_dominated = False
            for other in self.population:
                if individual != other and self._dominates(other, individual):
                    is_dominated = True
                    break
            
            if not is_dominated:
                non_dominated.append(individual)
        
        self.pareto_front = non_dominated
        self.pareto_front_history.append(non_dominated.copy())
        
        # Update best individuals
        if not self.best_individuals or len(non_dominated) > len(self.best_individuals):
            self.best_individuals = non_dominated.copy()
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self.pareto_front_history) < 2:
            return False
        
        # Check if Pareto front has stabilized
        current_front = self.pareto_front
        previous_front = self.pareto_front_history[-2]
        
        # Simple convergence check based on front similarity
        if len(current_front) == len(previous_front):
            # Check if objectives have changed significantly
            max_change = 0.0
            for i, (current, previous) in enumerate(zip(current_front, previous_front)):
                for obj_current, obj_previous in zip(current.objectives, previous.objectives):
                    change = abs(obj_current - obj_previous)
                    max_change = max(max_change, change)
            
            if max_change < self.config.convergence_threshold:
                return True
        
        return False
    
    def _record_generation_stats(self):
        """Record statistics for the current generation."""
        stats = {
            'generation': self.generation,
            'population_size': len(self.population),
            'pareto_front_size': len(self.pareto_front),
            'best_objectives': [min(obj_values) for obj_values in zip(*[ind.objectives for ind in self.pareto_front])] if self.pareto_front else [],
            'average_evaluation_time': np.mean([ind.evaluation_time for ind in self.population]),
            'timestamp': datetime.now().isoformat()
        }
        
        self.optimization_history.append(stats)
    
    def _compile_results(self, optimization_time: float) -> Dict[str, Any]:
        """Compile final optimization results."""
        results = {
            'success': True,
            'optimization_time': optimization_time,
            'generations': self.generation + 1,
            'population_size': len(self.population),
            'pareto_front_size': len(self.pareto_front),
            'best_individuals': [
                {
                    'architecture': ind.architecture,
                    'objectives': ind.objectives,
                    'metrics': ind.metrics,
                    'rank': ind.rank,
                    'crowding_distance': ind.crowding_distance
                }
                for ind in self.pareto_front
            ],
            'optimization_history': self.optimization_history,
            'convergence_info': {
                'converged': self._check_convergence(),
                'final_generation': self.generation + 1
            }
        }
        
        return results

def create_nas_objectives(objective_configs: List[Dict[str, Any]]) -> List[NASObjective]:
    """
    Create NAS objectives from configuration.
    
    Args:
        objective_configs: List of objective configurations
        
    Returns:
        List of NASObjective instances
    """
    objectives = []
    
    for config in objective_configs:
        objective = NASObjective(
            name=config.get('name', 'unknown'),
            weight=config.get('weight', 1.0),
            direction=config.get('direction', 'maximize'),
            target_value=config.get('target_value'),
            threshold=config.get('threshold')
        )
        objectives.append(objective)
    
    return objectives

def create_default_nas_objectives() -> List[NASObjective]:
    """Create default NAS objectives for neural architecture search."""
    default_configs = [
        {
            'name': 'accuracy',
            'weight': 0.4,
            'direction': 'maximize',
            'target_value': 0.95
        },
        {
            'name': 'efficiency',
            'weight': 0.3,
            'direction': 'maximize',
            'target_value': 1.0
        },
        {
            'name': 'robustness',
            'weight': 0.2,
            'direction': 'maximize',
            'target_value': 0.9
        },
        {
            'name': 'complexity',
            'weight': 0.1,
            'direction': 'minimize',
            'target_value': 0.0
        }
    ]
    
    return create_nas_objectives(default_configs)

def create_financial_nas_objectives() -> List[NASObjective]:
    """Create financial trading NAS objectives."""
    financial_configs = [
        {
            'name': 'sharpe_ratio',
            'weight': 0.4,
            'direction': 'maximize',
            'target_value': 2.0
        },
        {
            'name': 'max_drawdown',
            'weight': 0.3,
            'direction': 'minimize',
            'target_value': 0.1
        },
        {
            'name': 'win_rate',
            'weight': 0.2,
            'direction': 'maximize',
            'target_value': 0.6
        },
        {
            'name': 'profit_factor',
            'weight': 0.1,
            'direction': 'maximize',
            'target_value': 1.5
        }
    ]
    
    return create_nas_objectives(financial_configs)

# Example usage and testing functions
def example_architecture_evaluation(architecture: Dict[str, Any]) -> Dict[str, float]:
    """Example architecture evaluation function."""
    # Simulate evaluation metrics
    metrics = {
        'accuracy': np.random.uniform(0.7, 0.95),
        'efficiency': np.random.uniform(0.5, 1.0),
        'robustness': np.random.uniform(0.6, 0.9),
        'complexity': np.random.uniform(0.1, 0.8),
        'training_time': np.random.uniform(10, 100),
        'inference_time': np.random.uniform(0.1, 1.0)
    }
    
    return metrics

def run_example_optimization():
    """Run an example NSGA-II optimization."""
    tprint_info("🚀 Running example NSGA-II optimization")
    
    # Define architecture search space
    architecture_space = {
        'n_layers': {'type': 'int', 'low': 2, 'high': 8},
        'layer_sizes': {
            'type': 'list',
            'min_layers': 2,
            'max_layers': 8,
            'layer_config': {
                'units': {'type': 'int', 'low': 32, 'high': 512},
                'activation': {'type': 'categorical', 'choices': ['relu', 'tanh', 'swish', 'gelu']},
                'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5}
            }
        },
        'learning_rate': {'type': 'float', 'low': 0.0001, 'high': 0.01},
        'batch_size': {'type': 'int', 'low': 16, 'high': 128}
    }
    
    # Create objectives
    objectives = create_default_nas_objectives()
    
    # Create optimizer
    config = NSGAIIConfig(
        population_size=50,
        max_generations=20,
        use_m1_optimization=True
    )
    optimizer = NSGAIIOptimizer(config)
    
    # Run optimization
    results = optimizer.optimize(
        architecture_space=architecture_space,
        objectives=objectives,
        evaluation_function=example_architecture_evaluation
    )
    
    tprint_success(f"✅ Optimization completed: {results.get('pareto_front_size', 0)} Pareto solutions found")
    return results

if __name__ == "__main__":
    # Run example optimization
    results = run_example_optimization()
    print(f"Results: {results}")