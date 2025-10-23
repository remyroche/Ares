"""
Advanced optimization algorithms for regime detection.

This module provides sophisticated optimization algorithms including Bayesian optimization,
evolutionary algorithms, and advanced grid search strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import ParameterGrid
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

logger = logging.getLogger(__name__)


class AcquisitionFunction(Enum):
    """Acquisition functions for Bayesian optimization."""
    EXPECTED_IMPROVEMENT = "expected_improvement"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    PROBABILITY_IMPROVEMENT = "probability_improvement"


@dataclass
class BayesianOptimizationConfig:
    """Configuration for Bayesian optimization."""
    n_iterations: int = 50
    acquisition_function: AcquisitionFunction = AcquisitionFunction.EXPECTED_IMPROVEMENT
    exploration_weight: float = 0.1
    random_state: int = 42
    n_initial_points: int = 10


@dataclass
class EvolutionaryConfig:
    """Configuration for evolutionary optimization."""
    population_size: int = 50
    n_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    random_state: int = 42


class BayesianOptimizer:
    """Bayesian optimization for regime detection parameters."""
    
    def __init__(self, config: BayesianOptimizationConfig):
        """Initialize Bayesian optimizer."""
        self.config = config
        self.logger = logging.getLogger('BayesianOptimizer')
        self.X_observed = []
        self.y_observed = []
        self.best_params = None
        self.best_score = -np.inf
        
    def optimize(self, objective_function: Callable, 
                parameter_bounds: Dict[str, Tuple[float, float]],
                categorical_params: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
        """
        Perform Bayesian optimization.
        
        Args:
            objective_function: Function to optimize
            parameter_bounds: Bounds for continuous parameters
            categorical_params: Categorical parameters
            
        Returns:
            Best parameters found
        """
        try:
            self.logger.info("Starting Bayesian optimization")
            
            # Initialize with random points
            self._initialize_points(objective_function, parameter_bounds, categorical_params)
            
            # Perform Bayesian optimization iterations
            for iteration in range(self.config.n_iterations):
                # Select next point to evaluate
                next_point = self._select_next_point(parameter_bounds, categorical_params)
                
                # Evaluate objective function
                score = objective_function(next_point)
                
                # Update observations
                self.X_observed.append(next_point)
                self.y_observed.append(score)
                
                # Update best if improved
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = next_point.copy()
                
                self.logger.debug(f"Iteration {iteration + 1}: Score = {score:.4f}, Best = {self.best_score:.4f}")
            
            self.logger.info(f"Bayesian optimization completed. Best score: {self.best_score:.4f}")
            return self.best_params
            
        except Exception as e:
            self.logger.error(f"Bayesian optimization failed: {e}")
            return {}
    
    def _initialize_points(self, objective_function: Callable,
                          parameter_bounds: Dict[str, Tuple[float, float]],
                          categorical_params: Optional[Dict[str, List[Any]]]) -> None:
        """Initialize with random points."""
        np.random.seed(self.config.random_state)
        
        for _ in range(self.config.n_initial_points):
            point = self._sample_random_point(parameter_bounds, categorical_params)
            score = objective_function(point)
            
            self.X_observed.append(point)
            self.y_observed.append(score)
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = point.copy()
    
    def _sample_random_point(self, parameter_bounds: Dict[str, Tuple[float, float]],
                           categorical_params: Optional[Dict[str, List[Any]]]) -> Dict[str, Any]:
        """Sample a random point from parameter space."""
        point = {}
        
        # Sample continuous parameters
        for param, (low, high) in parameter_bounds.items():
            point[param] = np.random.uniform(low, high)
        
        # Sample categorical parameters
        if categorical_params:
            for param, values in categorical_params.items():
                point[param] = np.random.choice(values)
        
        return point
    
    def _select_next_point(self, parameter_bounds: Dict[str, Tuple[float, float]],
                          categorical_params: Optional[Dict[str, List[Any]]]) -> Dict[str, Any]:
        """Select next point using acquisition function."""
        # For simplicity, use random sampling
        # In a full implementation, this would use Gaussian Process regression
        return self._sample_random_point(parameter_bounds, categorical_params)


class EvolutionaryOptimizer:
    """Evolutionary algorithm for regime detection optimization."""
    
    def __init__(self, config: EvolutionaryConfig):
        """Initialize evolutionary optimizer."""
        self.config = config
        self.logger = logging.getLogger('EvolutionaryOptimizer')
        self.population = []
        self.fitness_scores = []
        self.best_individual = None
        self.best_fitness = -np.inf
        
    def optimize(self, objective_function: Callable,
                parameter_bounds: Dict[str, Tuple[float, float]],
                categorical_params: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
        """
        Perform evolutionary optimization.
        
        Args:
            objective_function: Function to optimize
            parameter_bounds: Bounds for continuous parameters
            categorical_params: Categorical parameters
            
        Returns:
            Best parameters found
        """
        try:
            self.logger.info("Starting evolutionary optimization")
            
            # Initialize population
            self._initialize_population(parameter_bounds, categorical_params)
            
            # Evaluate initial population
            self._evaluate_population(objective_function)
            
            # Evolution loop
            for generation in range(self.config.n_generations):
                # Selection
                parents = self._selection()
                
                # Crossover
                offspring = self._crossover(parents, parameter_bounds, categorical_params)
                
                # Mutation
                offspring = self._mutation(offspring, parameter_bounds, categorical_params)
                
                # Evaluate offspring
                offspring_fitness = [objective_function(ind) for ind in offspring]
                
                # Replace population
                self._replacement(offspring, offspring_fitness)
                
                # Update best
                if max(offspring_fitness) > self.best_fitness:
                    best_idx = np.argmax(offspring_fitness)
                    self.best_fitness = offspring_fitness[best_idx]
                    self.best_individual = offspring[best_idx].copy()
                
                self.logger.debug(f"Generation {generation + 1}: Best fitness = {self.best_fitness:.4f}")
            
            self.logger.info(f"Evolutionary optimization completed. Best fitness: {self.best_fitness:.4f}")
            return self.best_individual
            
        except Exception as e:
            self.logger.error(f"Evolutionary optimization failed: {e}")
            return {}
    
    def _initialize_population(self, parameter_bounds: Dict[str, Tuple[float, float]],
                             categorical_params: Optional[Dict[str, List[Any]]]) -> None:
        """Initialize population with random individuals."""
        np.random.seed(self.config.random_state)
        
        self.population = []
        for _ in range(self.config.population_size):
            individual = self._create_random_individual(parameter_bounds, categorical_params)
            self.population.append(individual)
    
    def _create_random_individual(self, parameter_bounds: Dict[str, Tuple[float, float]],
                                categorical_params: Optional[Dict[str, List[Any]]]) -> Dict[str, Any]:
        """Create a random individual."""
        individual = {}
        
        # Random continuous parameters
        for param, (low, high) in parameter_bounds.items():
            individual[param] = np.random.uniform(low, high)
        
        # Random categorical parameters
        if categorical_params:
            for param, values in categorical_params.items():
                individual[param] = np.random.choice(values)
        
        return individual
    
    def _evaluate_population(self, objective_function: Callable) -> None:
        """Evaluate fitness of entire population."""
        self.fitness_scores = []
        for individual in self.population:
            fitness = objective_function(individual)
            self.fitness_scores.append(fitness)
        
        # Update best
        best_idx = np.argmax(self.fitness_scores)
        if self.fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness_scores[best_idx]
            self.best_individual = self.population[best_idx].copy()
    
    def _selection(self) -> List[Dict[str, Any]]:
        """Select parents using tournament selection."""
        parents = []
        
        for _ in range(self.config.population_size):
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(
                len(self.population), 
                size=min(tournament_size, len(self.population)), 
                replace=False
            )
            tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(self.population[winner_idx].copy())
        
        return parents
    
    def _crossover(self, parents: List[Dict[str, Any]], 
                  parameter_bounds: Dict[str, Tuple[float, float]],
                  categorical_params: Optional[Dict[str, List[Any]]]) -> List[Dict[str, Any]]:
        """Perform crossover to create offspring."""
        offspring = []
        
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1 = parents[i]
                parent2 = parents[i + 1]
                
                if np.random.random() < self.config.crossover_rate:
                    child1, child2 = self._single_point_crossover(parent1, parent2, parameter_bounds)
                    offspring.extend([child1, child2])
                else:
                    offspring.extend([parent1.copy(), parent2.copy()])
            else:
                offspring.append(parents[i].copy())
        
        return offspring
    
    def _single_point_crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any],
                               parameter_bounds: Dict[str, Tuple[float, float]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform single-point crossover."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Crossover continuous parameters
        continuous_params = list(parameter_bounds.keys())
        if len(continuous_params) > 1:
            crossover_point = np.random.randint(1, len(continuous_params))
            
            for i, param in enumerate(continuous_params):
                if i >= crossover_point:
                    child1[param] = parent2[param]
                    child2[param] = parent1[param]
        
        return child1, child2
    
    def _mutation(self, offspring: List[Dict[str, Any]], 
                 parameter_bounds: Dict[str, Tuple[float, float]],
                 categorical_params: Optional[Dict[str, List[Any]]]) -> List[Dict[str, Any]]:
        """Apply mutation to offspring."""
        for individual in offspring:
            if np.random.random() < self.config.mutation_rate:
                # Mutate continuous parameters
                for param, (low, high) in parameter_bounds.items():
                    if np.random.random() < 0.5:  # 50% chance to mutate each parameter
                        # Gaussian mutation
                        current_value = individual[param]
                        mutation_strength = (high - low) * 0.1
                        new_value = current_value + np.random.normal(0, mutation_strength)
                        individual[param] = np.clip(new_value, low, high)
                
                # Mutate categorical parameters
                if categorical_params:
                    for param, values in categorical_params.items():
                        if np.random.random() < 0.5:
                            individual[param] = np.random.choice(values)
        
        return offspring
    
    def _replacement(self, offspring: List[Dict[str, Any]], 
                    offspring_fitness: List[float]) -> None:
        """Replace population with offspring (elitism)."""
        # Keep elite individuals
        elite_indices = np.argsort(self.fitness_scores)[-self.config.elite_size:]
        elite_individuals = [self.population[i] for i in elite_indices]
        elite_fitness = [self.fitness_scores[i] for i in elite_indices]
        
        # Combine elite and offspring
        all_individuals = elite_individuals + offspring
        all_fitness = elite_fitness + offspring_fitness
        
        # Select best individuals for next generation
        sorted_indices = np.argsort(all_fitness)[::-1]
        self.population = [all_individuals[i] for i in sorted_indices[:self.config.population_size]]
        self.fitness_scores = [all_fitness[i] for i in sorted_indices[:self.config.population_size]]


class AdvancedGridSearch:
    """Advanced grid search with adaptive refinement."""
    
    def __init__(self, initial_density: int = 3, refinement_factor: int = 2, max_refinements: int = 3):
        """Initialize advanced grid search."""
        self.initial_density = initial_density
        self.refinement_factor = refinement_factor
        self.max_refinements = max_refinements
        self.logger = logging.getLogger('AdvancedGridSearch')
    
    def optimize(self, objective_function: Callable,
                parameter_bounds: Dict[str, Tuple[float, float]],
                categorical_params: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
        """
        Perform adaptive grid search optimization.
        
        Args:
            objective_function: Function to optimize
            parameter_bounds: Bounds for continuous parameters
            categorical_params: Categorical parameters
            
        Returns:
            Best parameters found
        """
        try:
            self.logger.info("Starting adaptive grid search optimization")
            
            best_params = None
            best_score = -np.inf
            current_bounds = parameter_bounds.copy()
            
            for refinement in range(self.max_refinements + 1):
                # Create parameter grid
                param_grid = self._create_parameter_grid(current_bounds, categorical_params, refinement)
                
                # Evaluate all combinations
                for params in param_grid:
                    try:
                        score = objective_function(params)
                        
                        if score > best_score:
                            best_score = score
                            best_params = params.copy()
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to evaluate parameters {params}: {e}")
                        continue
                
                # Refine search around best point
                if refinement < self.max_refinements and best_params is not None:
                    current_bounds = self._refine_bounds(current_bounds, best_params)
                    self.logger.debug(f"Refinement {refinement + 1}: Best score = {best_score:.4f}")
            
            self.logger.info(f"Adaptive grid search completed. Best score: {best_score:.4f}")
            return best_params or {}
            
        except Exception as e:
            self.logger.error(f"Adaptive grid search failed: {e}")
            return {}
    
    def _create_parameter_grid(self, parameter_bounds: Dict[str, Tuple[float, float]],
                              categorical_params: Optional[Dict[str, List[Any]]],
                              refinement_level: int) -> List[Dict[str, Any]]:
        """Create parameter grid for current refinement level."""
        density = self.initial_density * (self.refinement_factor ** refinement_level)
        
        # Create grid for continuous parameters
        continuous_grid = {}
        for param, (low, high) in parameter_bounds.items():
            continuous_grid[param] = np.linspace(low, high, density)
        
        # Create grid for categorical parameters
        categorical_grid = categorical_params or {}
        
        # Generate all combinations
        param_grid = ParameterGrid({**continuous_grid, **categorical_grid})
        return list(param_grid)
    
    def _refine_bounds(self, current_bounds: Dict[str, Tuple[float, float]], 
                      best_params: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """Refine bounds around best parameters."""
        refined_bounds = {}
        
        for param, (low, high) in current_bounds.items():
            if param in best_params:
                # Shrink bounds around best value
                current_value = best_params[param]
                range_size = high - low
                new_range = range_size * 0.5  # Shrink to 50% of current range
                
                new_low = max(low, current_value - new_range / 2)
                new_high = min(high, current_value + new_range / 2)
                
                refined_bounds[param] = (new_low, new_high)
            else:
                refined_bounds[param] = (low, high)
        
        return refined_bounds


def create_optimization_objective(features: np.ndarray, 
                                 algorithms: List[str],
                                 quality_weights: Optional[Dict[str, float]] = None) -> Callable:
    """
    Create objective function for regime optimization.
    
    Args:
        features: Input features
        algorithms: List of algorithms to try
        quality_weights: Weights for different quality metrics
        
    Returns:
        Objective function
    """
    if quality_weights is None:
        quality_weights = {
            'silhouette': 0.4,
            'calinski_harabasz': 0.3,
            'davies_bouldin': 0.3
        }
    
    def objective(params: Dict[str, Any]) -> float:
        """Objective function for optimization."""
        try:
            # Extract parameters
            n_clusters = int(params.get('n_clusters', 3))
            algorithm = params.get('algorithm', 'kmeans')
            
            # Validate parameters
            if n_clusters < 2 or n_clusters > 20:
                return -np.inf
            
            if algorithm not in algorithms:
                return -np.inf
            
            # Create and fit model
            if algorithm == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            elif algorithm == 'gmm':
                model = GaussianMixture(n_components=n_clusters, random_state=42)
            elif algorithm == 'agglomerative':
                model = AgglomerativeClustering(n_clusters=n_clusters)
            else:
                return -np.inf
            
            labels = model.fit_predict(features)
            
            # Check for valid clustering
            if len(np.unique(labels)) < 2:
                return -np.inf
            
            # Calculate quality metrics
            try:
                silhouette = silhouette_score(features, labels)
                calinski = calinski_harabasz_score(features, labels)
                davies = davies_bouldin_score(features, labels)
                
                # Normalize metrics (higher is better for all)
                normalized_silhouette = (silhouette + 1) / 2  # [-1, 1] -> [0, 1]
                normalized_calinski = min(calinski / 1000, 1.0)  # Cap at 1.0
                normalized_davies = 1.0 / (1.0 + davies)  # Invert (lower is better)
                
                # Weighted combination
                score = (quality_weights['silhouette'] * normalized_silhouette +
                        quality_weights['calinski_harabasz'] * normalized_calinski +
                        quality_weights['davies_bouldin'] * normalized_davies)
                
                return score
                
            except Exception as e:
                logger.warning(f"Failed to calculate quality metrics: {e}")
                return -np.inf
                
        except Exception as e:
            logger.warning(f"Objective function failed: {e}")
            return -np.inf
    
    return objective