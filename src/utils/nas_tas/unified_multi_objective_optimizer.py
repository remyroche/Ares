"""
Unified Multi-Objective Optimizer for NAS and TAS Systems

This module consolidates all multi-objective optimization algorithms for both
Neural Architecture Search (NAS) and Tree Architecture Search (TAS) systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from importlib import import_module
import logging
import time

_LEGACY_NAS_NSGA: Optional[type] = None


def _resolve_legacy_nsga() -> type:
    """Resolve the legacy NSGA-II optimizer, failing fast when unavailable."""
    global _LEGACY_NAS_NSGA
    if _LEGACY_NAS_NSGA is not None:
        return _LEGACY_NAS_NSGA

    module = import_module('src.training.steps.market_analysis.nas_regime.core.nas_search')
    legacy_cls = getattr(module, 'NSGAIIOptimizer', None)
    if legacy_cls is None:
        raise ImportError("NSGAIIOptimizer not found in legacy NAS module")

    _LEGACY_NAS_NSGA = legacy_cls
    return _LEGACY_NAS_NSGA

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)

class ObjectiveType(Enum):
    """Types of optimization objectives."""
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    STABILITY = "stability"
    PROFITABILITY = "profitability"
    ECONOMIC_SIGNIFICANCE = "economic_significance"
    TRADING_VIABILITY = "trading_viability"
    COMPLEXITY = "complexity"
    MEMORY_USAGE = "memory_usage"
    SPEED = "speed"
    ROBUSTNESS = "robustness"
    SHARPE_RATIO = "sharpe_ratio"
    DOWNSIDE_DEVIATION = "downside_deviation"
    EXECUTION_LATENCY = "execution_latency"
    TAIL_LATENCY = "tail_latency"
    COLD_START_LATENCY = "cold_start_latency"

class OptimizationAlgorithm(Enum):
    """Available optimization algorithms."""
    NSGA2 = "nsga2"
    SPEA2 = "spea2"
    BAYESIAN = "bayesian"
    RANDOM = "random"
    HYBRID = "hybrid"
    WEIGHTED_SUM = "weighted_sum"

@dataclass
class UnifiedMultiObjectiveConfig:
    """Configuration for unified multi-objective optimization."""
    
    # Core optimization parameters
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.NSGA2
    objectives: List[ObjectiveType] = field(default_factory=lambda: [
        ObjectiveType.SHARPE_RATIO,
        ObjectiveType.DOWNSIDE_DEVIATION,
        ObjectiveType.EXECUTION_LATENCY,
    ])
    objective_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])
    objective_directions: List[str] = field(default_factory=lambda: ['maximize', 'minimize', 'minimize'])
    
    # Optimization parameters
    max_iterations: int = 100
    population_size: int = 50
    elite_size: int = 5
    convergence_threshold: float = 0.01
    convergence_patience: int = 20
    
    # Hardware optimization
    enable_parallel_processing: bool = True
    n_jobs: int = -1
    enable_gpu_acceleration: bool = True
    
    # Monitoring and logging
    enable_logging: bool = True
    log_level: str = 'INFO'
    save_intermediate_results: bool = True

    def __post_init__(self) -> None:
        if len(self.objective_weights) != len(self.objectives):
            raise ValueError("Objective weights must match number of objectives")
        if len(self.objective_directions) != len(self.objectives):
            self.objective_directions = ['maximize'] * len(self.objectives)
        allowed = {'maximize', 'minimize'}
        for direction in self.objective_directions:
            if direction not in allowed:
                raise ValueError("Objective directions must be 'maximize' or 'minimize'")

@dataclass
class ParetoSolution:
    """Represents a solution on the Pareto frontier."""
    parameters: Dict[str, Any]
    objectives: Dict[ObjectiveType, float]
    rank: int = 0
    crowding_distance: float = 0.0
    dominated_count: int = 0

@dataclass
class UnifiedOptimizationResult:
    """Comprehensive multi-objective optimization result."""
    
    # Core results
    best_parameters: Dict[str, Any]
    best_scores: Dict[ObjectiveType, float]
    pareto_frontier: List[ParetoSolution]
    optimization_history: List[Dict[str, Any]]
    
    # Optimization metadata
    algorithm: OptimizationAlgorithm
    total_iterations: int
    execution_time: float
    convergence_achieved: bool
    
    # Performance metrics
    hypervolume: float = 0.0
    total_evaluations: int = 0
    
    # Metadata
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    success: bool = True
    error_message: Optional[str] = None

class UnifiedMultiObjectiveOptimizer:
    """
    Unified multi-objective optimizer that consolidates all optimization algorithms
    for both NAS and TAS systems.
    """
    
    def __init__(self, config: Optional[UnifiedMultiObjectiveConfig] = None):
        """Initialize unified multi-objective optimizer."""
        self.config = config or UnifiedMultiObjectiveConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Performance monitoring
        self.optimization_history = []
        self.performance_metrics = {}
        
        tprint_info(f"🚀 Unified Multi-Objective Optimizer initialized")
        tprint_info(f"   Algorithm: {self.config.algorithm.value}")
        tprint_info(f"   Objectives: {[obj.value for obj in self.config.objectives]}")
        tprint_info(f"   Max iterations: {self.config.max_iterations}")
    
    def optimize(self, 
                objective_functions: Dict[ObjectiveType, Callable],
                parameter_bounds: Dict[str, Tuple[float, float]],
                algorithm: Optional[OptimizationAlgorithm] = None) -> UnifiedOptimizationResult:
        """
        Optimize multiple objectives simultaneously using the specified algorithm.
        
        Args:
            objective_functions: Dictionary mapping objective types to evaluation functions
            parameter_bounds: Dictionary defining parameter bounds
            algorithm: Optimization algorithm to use (defaults to config algorithm)
            
        Returns:
            UnifiedOptimizationResult containing optimization results
        """
        try:
            # Use specified algorithm or default from config
            opt_algorithm = algorithm or self.config.algorithm
            
            # Validate inputs
            self._validate_objectives(objective_functions)
            self._validate_parameter_bounds(parameter_bounds)
            
            tprint_info(f"🎯 Starting {opt_algorithm.value} optimization...")
            tprint_info(f"   Objectives: {list(objective_functions.keys())}")
            tprint_info(f"   Parameter bounds: {list(parameter_bounds.keys())}")
            
            start_time = time.time()
            
            # Route to appropriate optimizer
            if opt_algorithm == OptimizationAlgorithm.NSGA2:
                result = self._optimize_nsga2(objective_functions, parameter_bounds)
            elif opt_algorithm == OptimizationAlgorithm.BAYESIAN:
                result = self._optimize_bayesian(objective_functions, parameter_bounds)
            elif opt_algorithm == OptimizationAlgorithm.RANDOM:
                result = self._optimize_random(objective_functions, parameter_bounds)
            else:
                raise ValueError(f"Unsupported optimization algorithm: {opt_algorithm}")
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            # Save optimization history
            self.optimization_history.append(result)
            
            tprint_success(f"✅ Optimization completed successfully")
            tprint_info(f"   Best weighted score: {sum(result.best_scores.values()):.4f}")
            tprint_info(f"   Pareto frontier size: {len(result.pareto_frontier)}")
            tprint_info(f"   Execution time: {result.execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Optimization failed: {e}")
            return UnifiedOptimizationResult(
                best_parameters={},
                best_scores={},
                pareto_frontier=[],
                optimization_history=[],
                algorithm=algorithm or self.config.algorithm,
                total_iterations=0,
                execution_time=0.0,
                convergence_achieved=False,
                success=False,
                error_message=str(e)
            )

    def _validate_objectives(self, objective_functions: Dict[ObjectiveType, Callable]):
        """Validate objective functions."""
        for obj in self.config.objectives:
            if obj not in objective_functions:
                raise ValueError(f"Objective function for {obj.value} not provided")

    def _objective_direction(self, obj_type: ObjectiveType) -> str:
        try:
            idx = self.config.objectives.index(obj_type)
        except ValueError:
            return 'maximize'
        return self.config.objective_directions[idx]
    
    def _validate_parameter_bounds(self, parameter_bounds: Dict[str, Tuple[float, float]]):
        """Validate parameter bounds."""
        if not parameter_bounds:
            raise ValueError("Parameter bounds cannot be empty")
        
        for param, bounds in parameter_bounds.items():
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError(f"Parameter {param} must have tuple bounds (min, max)")
            
            if bounds[0] >= bounds[1]:
                raise ValueError(f"Parameter {param} min value must be less than max value")
    
    def _optimize_nsga2(self, 
                       objective_functions: Dict[ObjectiveType, Callable],
                       parameter_bounds: Dict[str, Tuple[float, float]]) -> UnifiedOptimizationResult:
        """Optimize using NSGA-II algorithm."""
        try:
            tprint_info("🧬 Starting NSGA-II optimization...")
            start_time = time.time()
            
            # Initialize population
            population = self._initialize_population(parameter_bounds)
            
            optimization_history = []
            
            # Evolution loop
            for generation in range(self.config.max_iterations):
                # Evaluate population
                self._evaluate_population(population, objective_functions)
                
                # Create offspring
                offspring = self._create_offspring(population, parameter_bounds)
                
                # Evaluate offspring
                self._evaluate_population(offspring, objective_functions)
                
                # Combine population and offspring
                combined_population = population + offspring
                
                # Non-dominated sorting
                fronts = self._non_dominated_sorting(combined_population)
                
                # Select next generation
                population = self._environmental_selection(fronts)
                
                # Calculate metrics
                best_scores = self._get_best_scores(population)
                pareto_frontier = fronts[0] if fronts else []
                
                # Record history
                optimization_history.append({
                    'generation': generation,
                    'best_scores': best_scores,
                    'population_size': len(population),
                    'pareto_size': len(pareto_frontier)
                })
            
            # Calculate final metrics
            final_pareto = fronts[0] if fronts else []
            hypervolume = self._calculate_hypervolume(final_pareto)
            
            # Find best solution (highest weighted sum)
            best_solution = self._find_best_weighted_solution(population)
            
            # Create result
            result = UnifiedOptimizationResult(
                best_parameters=best_solution.parameters,
                best_scores=best_solution.objectives,
                pareto_frontier=final_pareto,
                optimization_history=optimization_history,
                algorithm=OptimizationAlgorithm.NSGA2,
                total_iterations=generation + 1,
                execution_time=time.time() - start_time,
                convergence_achieved=True,  # Simplified for now
                hypervolume=hypervolume,
                total_evaluations=len(population) * (generation + 1)
            )
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ NSGA-II optimization failed: {e}")
            raise
    
    def _optimize_bayesian(self, 
                          objective_functions: Dict[ObjectiveType, Callable],
                          parameter_bounds: Dict[str, Tuple[float, float]]) -> UnifiedOptimizationResult:
        """Optimize using Bayesian optimization."""
        try:
            tprint_info("🎯 Starting Bayesian optimization...")
            start_time = time.time()
            
            # Simple Bayesian optimization implementation
            best_solution = None
            best_weighted_score = float('-inf')
            optimization_history = []
            
            # Random search as fallback (simplified implementation)
            for iteration in range(self.config.max_iterations):
                # Generate random parameters
                parameters = {}
                for param, (min_val, max_val) in parameter_bounds.items():
                    parameters[param] = np.random.uniform(min_val, max_val)
                
                # Evaluate all objectives
                objectives = {}
                weighted_score = 0.0
                
                for obj_type, obj_func in objective_functions.items():
                    try:
                        score = obj_func(parameters)
                        objectives[obj_type] = score
                        weight = self.config.objective_weights[self.config.objectives.index(obj_type)]
                        weighted_score += weight * score
                    except Exception as e:
                        self.logger.warning(f"Objective {obj_type} failed: {e}")
                        objectives[obj_type] = 0.0
                
                # Update best solution
                if weighted_score > best_weighted_score:
                    best_weighted_score = weighted_score
                    best_solution = ParetoSolution(
                        parameters=parameters,
                        objectives=objectives
                    )
                
                optimization_history.append({
                    'iteration': iteration,
                    'weighted_score': weighted_score,
                    'best_weighted_score': best_weighted_score
                })
            
            # Create result
            result = UnifiedOptimizationResult(
                best_parameters=best_solution.parameters if best_solution else {},
                best_scores=best_solution.objectives if best_solution else {},
                pareto_frontier=[best_solution] if best_solution else [],
                optimization_history=optimization_history,
                algorithm=OptimizationAlgorithm.BAYESIAN,
                total_iterations=self.config.max_iterations,
                execution_time=time.time() - start_time,
                convergence_achieved=True,
                hypervolume=0.0,
                total_evaluations=self.config.max_iterations
            )
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Bayesian optimization failed: {e}")
            raise
    
    def _optimize_random(self, 
                        objective_functions: Dict[ObjectiveType, Callable],
                        parameter_bounds: Dict[str, Tuple[float, float]]) -> UnifiedOptimizationResult:
        """Optimize using random search."""
        try:
            tprint_info("🎲 Starting random optimization...")
            start_time = time.time()
            
            best_solution = None
            best_weighted_score = float('-inf')
            optimization_history = []
            
            # Random search loop
            for iteration in range(self.config.max_iterations):
                # Generate random parameters
                parameters = {}
                for param, (min_val, max_val) in parameter_bounds.items():
                    parameters[param] = np.random.uniform(min_val, max_val)
                
                # Evaluate all objectives
                objectives = {}
                weighted_score = 0.0
                
                for obj_type, obj_func in objective_functions.items():
                    try:
                        score = obj_func(parameters)
                        objectives[obj_type] = score
                        weight = self.config.objective_weights[self.config.objectives.index(obj_type)]
                        weighted_score += weight * score
                    except Exception as e:
                        self.logger.warning(f"Objective {obj_type} failed: {e}")
                        objectives[obj_type] = 0.0
                
                # Update best solution
                if weighted_score > best_weighted_score:
                    best_weighted_score = weighted_score
                    best_solution = ParetoSolution(
                        parameters=parameters,
                        objectives=objectives
                    )
                
                optimization_history.append({
                    'iteration': iteration,
                    'weighted_score': weighted_score,
                    'best_weighted_score': best_weighted_score
                })
            
            # Create result
            result = UnifiedOptimizationResult(
                best_parameters=best_solution.parameters if best_solution else {},
                best_scores=best_solution.objectives if best_solution else {},
                pareto_frontier=[best_solution] if best_solution else [],
                optimization_history=optimization_history,
                algorithm=OptimizationAlgorithm.RANDOM,
                total_iterations=self.config.max_iterations,
                execution_time=time.time() - start_time,
                convergence_achieved=False,
                hypervolume=0.0,
                total_evaluations=self.config.max_iterations
            )
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Random optimization failed: {e}")
            raise
    
    def _initialize_population(self, parameter_bounds: Dict[str, Tuple[float, float]]) -> List[ParetoSolution]:
        """Initialize random population."""
        population = []
        for _ in range(self.config.population_size):
            parameters = {}
            for param, (min_val, max_val) in parameter_bounds.items():
                parameters[param] = np.random.uniform(min_val, max_val)
            
            solution = ParetoSolution(
                parameters=parameters,
                objectives={obj: 0.0 for obj in self.config.objectives}
            )
            population.append(solution)
        
        return population
    
    def _evaluate_population(self, population: List[ParetoSolution], 
                           objective_functions: Dict[ObjectiveType, Callable]):
        """Evaluate objective functions for population."""
        for solution in population:
            for obj_type, obj_func in objective_functions.items():
                try:
                    score = obj_func(solution.parameters)
                    solution.objectives[obj_type] = score
                except Exception as e:
                    self.logger.warning(f"Objective {obj_type} failed: {e}")
                    solution.objectives[obj_type] = 0.0
    
    def _create_offspring(self, population: List[ParetoSolution], 
                         parameter_bounds: Dict[str, Tuple[float, float]]) -> List[ParetoSolution]:
        """Create offspring through crossover and mutation."""
        offspring = []
        
        while len(offspring) < self.config.population_size:
            # Selection (simple random selection for now)
            parent1 = np.random.choice(population)
            parent2 = np.random.choice(population)
            
            # Crossover (simple uniform crossover)
            child_params = {}
            for param in parent1.parameters.keys():
                if np.random.random() < 0.5:
                    child_params[param] = parent1.parameters[param]
                else:
                    child_params[param] = parent2.parameters[param]
            
            # Mutation (simple Gaussian noise)
            for param, (min_val, max_val) in parameter_bounds.items():
                if np.random.random() < 0.1:  # 10% mutation rate
                    noise = np.random.normal(0, 0.1 * (max_val - min_val))
                    child_params[param] = np.clip(
                        child_params[param] + noise, min_val, max_val
                    )
            
            child = ParetoSolution(
                parameters=child_params,
                objectives={obj: 0.0 for obj in self.config.objectives}
            )
            offspring.append(child)
        
        return offspring
    
    def _non_dominated_sorting(self, population: List[ParetoSolution]) -> List[List[ParetoSolution]]:
        """Perform non-dominated sorting."""
        fronts = []
        remaining_population = population.copy()
        
        # Calculate domination relationships
        for i, sol1 in enumerate(population):
            sol1.dominated_count = 0
            
            for j, sol2 in enumerate(population):
                if i != j:
                    if self._dominates(sol2, sol1):
                        sol1.dominated_count += 1
        
        # Build fronts
        while remaining_population:
            current_front = []
            next_remaining = []
            
            for sol in remaining_population:
                if sol.dominated_count == 0:
                    current_front.append(sol)
                else:
                    next_remaining.append(sol)
            
            fronts.append(current_front)
            
            # Update domination counts for next front
            for sol in current_front:
                for other_sol in next_remaining:
                    if self._dominates(sol, other_sol):
                        other_sol.dominated_count -= 1
            
            remaining_population = next_remaining
        
        # Assign ranks
        for rank, front in enumerate(fronts):
            for sol in front:
                sol.rank = rank
        
        return fronts
    
    def _dominates(self, sol1: ParetoSolution, sol2: ParetoSolution) -> bool:
        """Check if sol1 dominates sol2."""
        at_least_one_better = False
        
        for obj_type in self.config.objectives:
            val1 = sol1.objectives.get(obj_type, 0.0)
            val2 = sol2.objectives.get(obj_type, 0.0)
            
            direction = self._objective_direction(obj_type)
            if direction == 'maximize':
                if val1 < val2:
                    return False
                elif val1 > val2:
                    at_least_one_better = True
            else:  # minimize
                if val1 > val2:
                    return False
                elif val1 < val2:
                    at_least_one_better = True
        
        return at_least_one_better
    
    def _environmental_selection(self, fronts: List[List[ParetoSolution]]) -> List[ParetoSolution]:
        """Select next generation using environmental selection."""
        flattened_population: List[ParetoSolution] = [sol for front in fronts for sol in front]

        if not flattened_population:
            return []

        legacy_cls = _resolve_legacy_nsga()
        legacy_optimizer = legacy_cls(
            objectives=[obj.value for obj in self.config.objectives],
            population_size=self.config.population_size,
        )

        class _Wrapper:
            def __init__(self, solution: ParetoSolution, score: float):
                self.solution = solution
                self.fitness_score = score

        wrapped_population = [
            _Wrapper(solution=sol, score=self._weighted_score(sol))
            for sol in flattened_population
        ]
        selected_wrapped = legacy_optimizer.optimize(wrapped_population)
        selected_solutions = [wrapper.solution for wrapper in selected_wrapped if hasattr(wrapper, "solution")]
        if not selected_solutions:
            raise RuntimeError("Legacy NSGA-II selection returned no candidates")
        return selected_solutions[: self.config.population_size]
    
    def _get_best_scores(self, population: List[ParetoSolution]) -> Dict[ObjectiveType, float]:
        """Get best scores across all objectives."""
        best_scores = {}

        for obj_type in self.config.objectives:
            direction = self._objective_direction(obj_type)
            best_score = float('-inf') if direction == 'maximize' else float('inf')
            for sol in population:
                score = sol.objectives.get(obj_type, 0.0)
                if direction == 'maximize':
                    if score > best_score:
                        best_score = score
                else:
                    if score < best_score:
                        best_score = score
            best_scores[obj_type] = best_score

        return best_scores

    def _weighted_score(self, sol: ParetoSolution) -> float:
        """Calculate weighted score respecting objective directions."""
        weighted_score = 0.0
        for weight, obj_type in zip(self.config.objective_weights, self.config.objectives):
            score = sol.objectives.get(obj_type, 0.0)
            if self._objective_direction(obj_type) == 'minimize':
                score = -score
            weighted_score += weight * score
        return weighted_score

    def _find_best_weighted_solution(self, population: List[ParetoSolution]) -> ParetoSolution:
        """Find best solution using weighted sum."""
        best_solution = None
        best_weighted_score = float('-inf')

        for sol in population:
            weighted_score = self._weighted_score(sol)

            if weighted_score > best_weighted_score:
                best_weighted_score = weighted_score
                best_solution = sol

        return best_solution or ParetoSolution({}, {})

    def _calculate_hypervolume(self, pareto_frontier: List[ParetoSolution]) -> float:
        """Calculate hypervolume of Pareto frontier."""
        if len(pareto_frontier) < 2:
            return 0.0
        
        # Simple hypervolume calculation (2D case)
        if len(self.config.objectives) == 2:
            obj1_values = [sol.objectives[list(sol.objectives.keys())[0]] for sol in pareto_frontier]
            obj2_values = [sol.objectives[list(sol.objectives.keys())[1]] for sol in pareto_frontier]
            
            # Sort by first objective
            sorted_indices = np.argsort(obj1_values)
            hypervolume = 0.0
            
            for i in range(len(sorted_indices)):
                if i == 0:
                    hypervolume += obj1_values[sorted_indices[i]] * obj2_values[sorted_indices[i]]
                else:
                    prev_idx = sorted_indices[i-1]
                    curr_idx = sorted_indices[i]
                    hypervolume += (obj1_values[curr_idx] - obj1_values[prev_idx]) * obj2_values[curr_idx]
            
            return hypervolume
        
        return 0.0  # Placeholder for higher dimensions
    
    def _update_performance_metrics(self, result: UnifiedOptimizationResult):
        """Update performance metrics."""
        algorithm_name = result.algorithm.value
        
        if algorithm_name not in self.performance_metrics:
            self.performance_metrics[algorithm_name] = {
                'total_optimizations': 0,
                'successful_optimizations': 0,
                'avg_execution_time': 0.0,
                'avg_best_score': 0.0
            }
        
        metrics = self.performance_metrics[algorithm_name]
        metrics['total_optimizations'] += 1
        
        if result.success:
            metrics['successful_optimizations'] += 1
            
            # Update averages
            successful = metrics['successful_optimizations']
            metrics['avg_execution_time'] = (
                (metrics['avg_execution_time'] * (successful - 1) + result.execution_time) / successful
            )
            
            best_score = sum(result.best_scores.values())
            metrics['avg_best_score'] = (
                (metrics['avg_best_score'] * (successful - 1) + best_score) / successful
            )


# Convenience functions
def create_unified_multi_objective_optimizer(config: Optional[UnifiedMultiObjectiveConfig] = None) -> UnifiedMultiObjectiveOptimizer:
    """Create a unified multi-objective optimizer with specified configuration."""
    return UnifiedMultiObjectiveOptimizer(config)


def quick_multi_objective_optimization(objective_functions: Dict[ObjectiveType, Callable],
                                      parameter_bounds: Dict[str, Tuple[float, float]],
                                      algorithm: OptimizationAlgorithm = OptimizationAlgorithm.NSGA2,
                                      max_iterations: int = 100) -> UnifiedOptimizationResult:
    """Quick multi-objective optimization using default configuration."""
    config = UnifiedMultiObjectiveConfig(
        algorithm=algorithm,
        max_iterations=max_iterations
    )
    optimizer = UnifiedMultiObjectiveOptimizer(config)
    return optimizer.optimize(objective_functions, parameter_bounds)


# Export main classes and functions
__all__ = [
    'UnifiedMultiObjectiveOptimizer',
    'UnifiedMultiObjectiveConfig',
    'UnifiedOptimizationResult',
    'ParetoSolution',
    'ObjectiveType',
    'OptimizationAlgorithm',
    'create_unified_multi_objective_optimizer',
    'quick_multi_objective_optimization'
]