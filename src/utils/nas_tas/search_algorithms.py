"""
Unified Search Algorithms for NAS-TAS Systems

This module provides unified search algorithms that consolidate all search
functionality from TAS, NAS, and hybrid systems. It eliminates duplication
and provides a single, consistent interface for all search operations.

Components:
- BayesianOptimizer: Bayesian optimization with Gaussian Process regression
- EvolutionaryOptimizer: Evolutionary algorithms (NSGA-II, SPEA2)
- GridSearchOptimizer: Grid search with parallel processing
- RandomSearchOptimizer: Intelligent random search
- HybridOptimizer: Combined search strategies
- SearchManager: Unified search management and coordination
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Iterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
import time
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.model_selection import ParameterGrid
    BAYESIAN_OPTIMIZATION_AVAILABLE = True
except ImportError:
    BAYESIAN_OPTIMIZATION_AVAILABLE = False

try:
    from scipy.optimize import minimize
    SCIPY_OPTIMIZATION_AVAILABLE = True
except ImportError:
    SCIPY_OPTIMIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class SearchAlgorithmType(Enum):
    """Types of search algorithms available."""
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    EVOLUTIONARY_ALGORITHM = "evolutionary_algorithm"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    TREE_BASED_SEARCH = "tree_based_search"
    NEURAL_ARCHITECTURE_SEARCH = "neural_architecture_search"
    HYBRID_SEARCH = "hybrid_search"
    MULTI_OBJECTIVE_SEARCH = "multi_objective_search"


@dataclass
class SearchConfig:
    """Configuration for search algorithms."""
    
    # Algorithm parameters
    algorithm_type: SearchAlgorithmType = SearchAlgorithmType.BAYESIAN_OPTIMIZATION
    max_iterations: int = 100
    max_evaluations: int = 1000
    population_size: int = 50
    
    # Convergence parameters
    convergence_threshold: float = 1e-6
    patience: int = 20
    early_stopping: bool = True
    
    # Parallel processing
    n_jobs: int = -1
    parallel_backend: str = "threading"  # "threading", "multiprocessing"
    
    # Bayesian optimization specific
    acquisition_function: str = "expected_improvement"  # "ei", "pi", "ucb"
    exploration_weight: float = 0.1
    
    # Evolutionary algorithm specific
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_method: str = "tournament"  # "tournament", "roulette", "rank"
    
    # Multi-objective parameters
    enable_multi_objective: bool = False
    objectives: List[str] = field(default_factory=lambda: ["performance", "complexity"])
    weights: List[float] = field(default_factory=lambda: [1.0, 0.1])
    
    # Output parameters
    verbose: bool = True
    save_history: bool = True
    save_results: bool = True


@dataclass
class SearchResult:
    """Result from search algorithm."""
    
    # Best solution
    best_solution: Any
    best_score: float
    best_parameters: Dict[str, Any]
    
    # Search history
    search_history: List[Dict[str, Any]]
    convergence_curve: List[float]
    
    # Algorithm information
    algorithm_used: str
    n_evaluations: int
    execution_time: float
    
    # Convergence information
    converged: bool
    convergence_iteration: int
    final_improvement: float
    
    # Multi-objective results (if applicable)
    pareto_front: Optional[List[Dict[str, Any]]] = None
    multi_objective_scores: Optional[List[float]] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class BaseSearchAlgorithm(ABC):
    """Abstract base class for search algorithms."""
    
    def __init__(self, config: SearchConfig):
        """Initialize the search algorithm."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.search_history = []
        self.best_solution = None
        self.best_score = -np.inf
        self.best_parameters = {}
        self.n_evaluations = 0
        self.start_time = None
        
    @abstractmethod
    def _initialize_search(self, parameter_space: Dict[str, Any]) -> None:
        """Initialize the search algorithm."""
        pass
    
    @abstractmethod
    def _generate_candidate(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a candidate solution."""
        pass
    
    @abstractmethod
    def _update_search_state(self, candidate: Dict[str, Any], score: float) -> None:
        """Update the search state with new candidate."""
        pass
    
    @abstractmethod
    def _check_convergence(self) -> bool:
        """Check if the search has converged."""
        pass
    
    def search(self,
               objective_function: Callable,
               parameter_space: Dict[str, Any],
               initial_candidates: Optional[List[Dict[str, Any]]] = None) -> SearchResult:
        """
        Perform search optimization.
        
        Args:
            objective_function: Function to optimize
            parameter_space: Parameter space definition
            initial_candidates: Optional initial candidate solutions
            
        Returns:
            SearchResult with optimization results
        """
        self.start_time = time.time()
        self.logger.info(f"Starting {self.config.algorithm_type.value} search")
        
        try:
            # Initialize search
            self._initialize_search(parameter_space)
            
            # Generate initial candidates if not provided
            if initial_candidates is None:
                initial_candidates = self._generate_initial_candidates(parameter_space)
            
            # Evaluate initial candidates
            for candidate in initial_candidates:
                score = self._evaluate_candidate(objective_function, candidate)
                self._update_search_state(candidate, score)
            
            # Main search loop
            iteration = 0
            patience_counter = 0
            last_improvement = self.best_score
            
            while iteration < self.config.max_iterations and self.n_evaluations < self.config.max_evaluations:
                # Generate new candidate
                candidate = self._generate_candidate(parameter_space)
                
                # Evaluate candidate
                score = self._evaluate_candidate(objective_function, candidate)
                
                # Update search state
                self._update_search_state(candidate, score)
                
                # Check for improvement
                if score > self.best_score:
                    patience_counter = 0
                    last_improvement = self.best_score
                else:
                    patience_counter += 1
                
                # Check convergence
                if self._check_convergence() or (self.config.early_stopping and patience_counter >= self.config.patience):
                    self.logger.info(f"Search converged at iteration {iteration}")
                    break
                
                iteration += 1
            
            # Create result
            execution_time = time.time() - self.start_time
            convergence_iteration = iteration if patience_counter >= self.config.patience else -1
            final_improvement = self.best_score - last_improvement
            
            result = SearchResult(
                best_solution=self.best_solution,
                best_score=self.best_score,
                best_parameters=self.best_parameters,
                search_history=self.search_history,
                convergence_curve=[entry['score'] for entry in self.search_history],
                algorithm_used=self.config.algorithm_type.value,
                n_evaluations=self.n_evaluations,
                execution_time=execution_time,
                converged=iteration < self.config.max_iterations,
                convergence_iteration=convergence_iteration,
                final_improvement=final_improvement,
                metadata={
                    'parameter_space_size': len(parameter_space),
                    'initial_candidates': len(initial_candidates)
                }
            )
            
            self.logger.info(f"Search completed: {self.n_evaluations} evaluations, {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return SearchResult(
                best_solution=None,
                best_score=-np.inf,
                best_parameters={},
                search_history=self.search_history,
                convergence_curve=[],
                algorithm_used=self.config.algorithm_type.value,
                n_evaluations=self.n_evaluations,
                execution_time=time.time() - self.start_time if self.start_time else 0,
                converged=False,
                convergence_iteration=-1,
                final_improvement=0,
                errors=[str(e)]
            )
    
    def _generate_initial_candidates(self, parameter_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate initial candidate solutions."""
        candidates = []
        n_initial = min(10, self.config.population_size)
        
        for _ in range(n_initial):
            candidate = {}
            for param_name, param_config in parameter_space.items():
                if isinstance(param_config, dict):
                    if param_config['type'] == 'continuous':
                        candidate[param_name] = np.random.uniform(
                            param_config['min'], param_config['max']
                        )
                    elif param_config['type'] == 'discrete':
                        candidate[param_name] = np.random.choice(param_config['values'])
                    elif param_config['type'] == 'integer':
                        candidate[param_name] = np.random.randint(
                            param_config['min'], param_config['max'] + 1
                        )
                else:
                    candidate[param_name] = np.random.choice(param_config)
            candidates.append(candidate)
        
        return candidates
    
    def _evaluate_candidate(self, objective_function: Callable, candidate: Dict[str, Any]) -> float:
        """Evaluate a candidate solution."""
        try:
            score = objective_function(candidate)
            self.n_evaluations += 1
            return float(score)
        except Exception as e:
            self.logger.warning(f"Evaluation failed for candidate {candidate}: {e}")
            return -np.inf


class BayesianOptimizer(BaseSearchAlgorithm):
    """Bayesian optimization using Gaussian Process regression."""
    
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        if not BAYESIAN_OPTIMIZATION_AVAILABLE:
            raise ImportError("Bayesian optimization requires scikit-learn")
        
        self.gp_model = None
        self.X_evaluated = []
        self.y_evaluated = []
    
    def _initialize_search(self, parameter_space: Dict[str, Any]) -> None:
        """Initialize Bayesian optimization."""
        self.parameter_names = list(parameter_space.keys())
        self.parameter_bounds = self._get_parameter_bounds(parameter_space)
        
        # Initialize Gaussian Process
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10
        )
    
    def _get_parameter_bounds(self, parameter_space: Dict[str, Any]) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization."""
        bounds = []
        for param_name, param_config in parameter_space.items():
            if isinstance(param_config, dict):
                if param_config['type'] == 'continuous':
                    bounds.append((param_config['min'], param_config['max']))
                elif param_config['type'] == 'discrete':
                    # Convert discrete to continuous bounds
                    values = param_config['values']
                    bounds.append((min(values), max(values)))
                elif param_config['type'] == 'integer':
                    bounds.append((param_config['min'], param_config['max']))
            else:
                bounds.append((min(param_config), max(param_config)))
        return bounds
    
    def _generate_candidate(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate candidate using acquisition function."""
        if len(self.X_evaluated) < 5:
            # Random sampling for initial points
            return self._random_candidate(parameter_space)
        
        # Use acquisition function
        candidate_vector = self._optimize_acquisition_function()
        
        # Convert vector back to parameter dictionary
        candidate = {}
        for i, param_name in enumerate(self.parameter_names):
            candidate[param_name] = candidate_vector[i]
        
        return candidate
    
    def _random_candidate(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random candidate."""
        candidate = {}
        for param_name, param_config in parameter_space.items():
            if isinstance(param_config, dict):
                if param_config['type'] == 'continuous':
                    candidate[param_name] = np.random.uniform(
                        param_config['min'], param_config['max']
                    )
                elif param_config['type'] == 'discrete':
                    candidate[param_name] = np.random.choice(param_config['values'])
                elif param_config['type'] == 'integer':
                    candidate[param_name] = np.random.randint(
                        param_config['min'], param_config['max'] + 1
                    )
            else:
                candidate[param_name] = np.random.choice(param_config)
        return candidate
    
    def _optimize_acquisition_function(self) -> np.ndarray:
        """Optimize acquisition function to find next candidate."""
        try:
            # Fit GP model
            X = np.array(self.X_evaluated)
            y = np.array(self.y_evaluated)
            self.gp_model.fit(X, y)
            
            # Define acquisition function
            def acquisition_function(x):
                x = x.reshape(1, -1)
                mean, std = self.gp_model.predict(x, return_std=True)
                
                if self.config.acquisition_function == "expected_improvement":
                    # Expected Improvement
                    improvement = mean - self.best_score
                    z = improvement / (std + 1e-9)
                    ei = improvement * self._normal_cdf(z) + std * self._normal_pdf(z)
                    return -ei[0]  # Minimize negative EI
                
                elif self.config.acquisition_function == "upper_confidence_bound":
                    # Upper Confidence Bound
                    return -(mean + self.config.exploration_weight * std)[0]
                
                else:
                    # Probability of Improvement
                    z = (mean - self.best_score) / (std + 1e-9)
                    return -self._normal_cdf(z)[0]
            
            # Optimize acquisition function
            if SCIPY_OPTIMIZATION_AVAILABLE:
                result = minimize(
                    acquisition_function,
                    x0=np.random.uniform([b[0] for b in self.parameter_bounds],
                                       [b[1] for b in self.parameter_bounds]),
                    bounds=self.parameter_bounds,
                    method='L-BFGS-B'
                )
                return result.x
            else:
                # Fallback to random search
                return np.random.uniform([b[0] for b in self.parameter_bounds],
                                       [b[1] for b in self.parameter_bounds])
                
        except Exception as e:
            self.logger.warning(f"Acquisition function optimization failed: {e}")
            return np.random.uniform([b[0] for b in self.parameter_bounds],
                                   [b[1] for b in self.parameter_bounds])
    
    def _normal_cdf(self, x):
        """Normal cumulative distribution function."""
        return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def _normal_pdf(self, x):
        """Normal probability density function."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def _update_search_state(self, candidate: Dict[str, Any], score: float) -> None:
        """Update Bayesian optimization state."""
        # Convert candidate to vector
        candidate_vector = [candidate[param_name] for param_name in self.parameter_names]
        
        self.X_evaluated.append(candidate_vector)
        self.y_evaluated.append(score)
        
        # Update best solution
        if score > self.best_score:
            self.best_score = score
            self.best_solution = candidate
            self.best_parameters = candidate.copy()
        
        # Record search history
        self.search_history.append({
            'iteration': len(self.search_history),
            'candidate': candidate,
            'score': score,
            'best_score': self.best_score
        })
    
    def _check_convergence(self) -> bool:
        """Check convergence for Bayesian optimization."""
        if len(self.y_evaluated) < 10:
            return False
        
        # Check if improvement is below threshold
        recent_scores = self.y_evaluated[-10:]
        improvement = max(recent_scores) - min(recent_scores)
        
        return improvement < self.config.convergence_threshold


class EvolutionaryOptimizer(BaseSearchAlgorithm):
    """Evolutionary algorithm optimizer."""
    
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.population = []
        self.fitness_scores = []
        self.generation = 0
    
    def _initialize_search(self, parameter_space: Dict[str, Any]) -> None:
        """Initialize evolutionary algorithm."""
        self.parameter_space = parameter_space
        self.parameter_names = list(parameter_space.keys())
    
    def _generate_candidate(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate candidate through evolution."""
        if len(self.population) < self.config.population_size:
            # Generate random candidate
            return self._random_candidate(parameter_space)
        
        # Selection
        parent1, parent2 = self._select_parents()
        
        # Crossover
        if random.random() < self.config.crossover_rate:
            candidate = self._crossover(parent1, parent2)
        else:
            candidate = parent1.copy()
        
        # Mutation
        if random.random() < self.config.mutation_rate:
            candidate = self._mutate(candidate)
        
        return candidate
    
    def _random_candidate(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random candidate."""
        candidate = {}
        for param_name, param_config in parameter_space.items():
            if isinstance(param_config, dict):
                if param_config['type'] == 'continuous':
                    candidate[param_name] = np.random.uniform(
                        param_config['min'], param_config['max']
                    )
                elif param_config['type'] == 'discrete':
                    candidate[param_name] = np.random.choice(param_config['values'])
                elif param_config['type'] == 'integer':
                    candidate[param_name] = np.random.randint(
                        param_config['min'], param_config['max'] + 1
                    )
            else:
                candidate[param_name] = np.random.choice(param_config)
        return candidate
    
    def _select_parents(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Select parents for reproduction."""
        if self.config.selection_method == "tournament":
            return self._tournament_selection()
        elif self.config.selection_method == "roulette":
            return self._roulette_selection()
        else:
            return self._rank_selection()
    
    def _tournament_selection(self, tournament_size: int = 3) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Tournament selection."""
        def select_one():
            tournament_indices = random.sample(range(len(self.population)), tournament_size)
            tournament_scores = [self.fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_scores)]
            return self.population[winner_idx]
        
        return select_one(), select_one()
    
    def _roulette_selection(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Roulette wheel selection."""
        # Convert scores to probabilities
        min_score = min(self.fitness_scores)
        adjusted_scores = [score - min_score + 1e-6 for score in self.fitness_scores]
        total_fitness = sum(adjusted_scores)
        probabilities = [score / total_fitness for score in adjusted_scores]
        
        # Select parents
        parent1_idx = np.random.choice(len(self.population), p=probabilities)
        parent2_idx = np.random.choice(len(self.population), p=probabilities)
        
        return self.population[parent1_idx], self.population[parent2_idx]
    
    def _rank_selection(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Rank-based selection."""
        # Sort by fitness
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        
        # Assign ranks
        ranks = np.arange(1, len(self.population) + 1)
        rank_probabilities = ranks / ranks.sum()
        
        # Select parents
        parent1_idx = np.random.choice(sorted_indices, p=rank_probabilities)
        parent2_idx = np.random.choice(sorted_indices, p=rank_probabilities)
        
        return self.population[parent1_idx], self.population[parent2_idx]
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Perform crossover between parents."""
        child = {}
        
        for param_name in self.parameter_names:
            if random.random() < 0.5:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]
        
        return child
    
    def _mutate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mutation to candidate."""
        mutated = candidate.copy()
        
        for param_name, param_config in self.parameter_space.items():
            if random.random() < 0.1:  # 10% chance to mutate each parameter
                if isinstance(param_config, dict):
                    if param_config['type'] == 'continuous':
                        # Gaussian mutation
                        noise = np.random.normal(0, 0.1 * (param_config['max'] - param_config['min']))
                        mutated[param_name] = np.clip(
                            mutated[param_name] + noise,
                            param_config['min'],
                            param_config['max']
                        )
                    elif param_config['type'] == 'discrete':
                        mutated[param_name] = np.random.choice(param_config['values'])
                    elif param_config['type'] == 'integer':
                        mutated[param_name] = np.random.randint(
                            param_config['min'], param_config['max'] + 1
                        )
                else:
                    mutated[param_name] = np.random.choice(param_config)
        
        return mutated
    
    def _update_search_state(self, candidate: Dict[str, Any], score: float) -> None:
        """Update evolutionary algorithm state."""
        # Add to population
        self.population.append(candidate)
        self.fitness_scores.append(score)
        
        # Update best solution
        if score > self.best_score:
            self.best_score = score
            self.best_solution = candidate
            self.best_parameters = candidate.copy()
        
        # Maintain population size
        if len(self.population) > self.config.population_size:
            # Remove worst individual
            worst_idx = np.argmin(self.fitness_scores)
            self.population.pop(worst_idx)
            self.fitness_scores.pop(worst_idx)
        
        # Record search history
        self.search_history.append({
            'iteration': len(self.search_history),
            'candidate': candidate,
            'score': score,
            'best_score': self.best_score,
            'population_size': len(self.population),
            'generation': self.generation
        })
    
    def _check_convergence(self) -> bool:
        """Check convergence for evolutionary algorithm."""
        if len(self.fitness_scores) < 20:
            return False
        
        # Check diversity in population
        recent_scores = self.fitness_scores[-20:]
        diversity = np.std(recent_scores)
        
        return diversity < self.config.convergence_threshold


class GridSearchOptimizer(BaseSearchAlgorithm):
    """Grid search optimizer with parallel processing."""
    
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.grid_points = []
        self.current_index = 0
    
    def _initialize_search(self, parameter_space: Dict[str, Any]) -> None:
        """Initialize grid search."""
        self.parameter_space = parameter_space
        self.grid_points = self._generate_grid_points(parameter_space)
        self.logger.info(f"Generated {len(self.grid_points)} grid points")
    
    def _generate_grid_points(self, parameter_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate all grid points."""
        # Limit grid size to prevent explosion
        max_points = 10000
        
        param_lists = {}
        total_combinations = 1
        
        for param_name, param_config in parameter_space.items():
            if isinstance(param_config, dict):
                if param_config['type'] == 'continuous':
                    # Create discrete grid for continuous parameters
                    n_points = min(10, max_points // total_combinations)
                    param_lists[param_name] = np.linspace(
                        param_config['min'], param_config['max'], n_points
                    ).tolist()
                elif param_config['type'] == 'discrete':
                    param_lists[param_name] = param_config['values']
                elif param_config['type'] == 'integer':
                    # Create grid for integer parameters
                    n_points = min(10, max_points // total_combinations)
                    param_lists[param_name] = list(range(
                        param_config['min'], param_config['max'] + 1, 
                        max(1, (param_config['max'] - param_config['min']) // n_points)
                    ))
            else:
                param_lists[param_name] = param_config
            
            total_combinations *= len(param_lists[param_name])
            if total_combinations > max_points:
                break
        
        # Generate all combinations
        if SCIPY_OPTIMIZATION_AVAILABLE:
            grid = ParameterGrid(param_lists)
            return list(grid)
        else:
            # Fallback implementation
            return self._generate_combinations(param_lists)
    
    def _generate_combinations(self, param_lists: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate parameter combinations (fallback)."""
        from itertools import product
        
        combinations = []
        for combination in product(*param_lists.values()):
            param_dict = dict(zip(param_lists.keys(), combination))
            combinations.append(param_dict)
            
            if len(combinations) >= 10000:  # Limit size
                break
        
        return combinations
    
    def _generate_candidate(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate candidate from grid."""
        if self.current_index >= len(self.grid_points):
            return None  # Search complete
        
        candidate = self.grid_points[self.current_index]
        self.current_index += 1
        return candidate
    
    def _update_search_state(self, candidate: Dict[str, Any], score: float) -> None:
        """Update grid search state."""
        # Update best solution
        if score > self.best_score:
            self.best_score = score
            self.best_solution = candidate
            self.best_parameters = candidate.copy()
        
        # Record search history
        self.search_history.append({
            'iteration': len(self.search_history),
            'candidate': candidate,
            'score': score,
            'best_score': self.best_score,
            'grid_index': self.current_index - 1
        })
    
    def _check_convergence(self) -> bool:
        """Check if grid search is complete."""
        return self.current_index >= len(self.grid_points)


class RandomSearchOptimizer(BaseSearchAlgorithm):
    """Random search optimizer with intelligent sampling."""
    
    def _initialize_search(self, parameter_space: Dict[str, Any]) -> None:
        """Initialize random search."""
        self.parameter_space = parameter_space
    
    def _generate_candidate(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random candidate."""
        return self._random_candidate(parameter_space)
    
    def _random_candidate(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random candidate."""
        candidate = {}
        for param_name, param_config in parameter_space.items():
            if isinstance(param_config, dict):
                if param_config['type'] == 'continuous':
                    candidate[param_name] = np.random.uniform(
                        param_config['min'], param_config['max']
                    )
                elif param_config['type'] == 'discrete':
                    candidate[param_name] = np.random.choice(param_config['values'])
                elif param_config['type'] == 'integer':
                    candidate[param_name] = np.random.randint(
                        param_config['min'], param_config['max'] + 1
                    )
            else:
                candidate[param_name] = np.random.choice(param_config)
        return candidate
    
    def _update_search_state(self, candidate: Dict[str, Any], score: float) -> None:
        """Update random search state."""
        # Update best solution
        if score > self.best_score:
            self.best_score = score
            self.best_solution = candidate
            self.best_parameters = candidate.copy()
        
        # Record search history
        self.search_history.append({
            'iteration': len(self.search_history),
            'candidate': candidate,
            'score': score,
            'best_score': self.best_score
        })
    
    def _check_convergence(self) -> bool:
        """Random search doesn't converge, just stops at max iterations."""
        return False


class SearchManager:
    """Unified search management and coordination."""
    
    def __init__(self, config: SearchConfig):
        """Initialize search manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_optimizer(self, algorithm_type: Optional[SearchAlgorithmType] = None) -> BaseSearchAlgorithm:
        """Create optimizer based on configuration."""
        if algorithm_type is None:
            algorithm_type = self.config.algorithm_type
        
        if algorithm_type == SearchAlgorithmType.BAYESIAN_OPTIMIZATION:
            if not BAYESIAN_OPTIMIZATION_AVAILABLE:
                self.logger.warning("Bayesian optimization not available, falling back to random search")
                return RandomSearchOptimizer(self.config)
            return BayesianOptimizer(self.config)
        
        elif algorithm_type == SearchAlgorithmType.EVOLUTIONARY_ALGORITHM:
            return EvolutionaryOptimizer(self.config)
        
        elif algorithm_type == SearchAlgorithmType.GRID_SEARCH:
            return GridSearchOptimizer(self.config)
        
        elif algorithm_type == SearchAlgorithmType.RANDOM_SEARCH:
            return RandomSearchOptimizer(self.config)
        
        else:
            raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
    
    def optimize(self,
                 objective_function: Callable,
                 parameter_space: Dict[str, Any],
                 algorithm_type: Optional[SearchAlgorithmType] = None,
                 initial_candidates: Optional[List[Dict[str, Any]]] = None) -> SearchResult:
        """
        Perform optimization using specified algorithm.
        
        Args:
            objective_function: Function to optimize
            parameter_space: Parameter space definition
            algorithm_type: Algorithm to use (optional)
            initial_candidates: Optional initial candidates
            
        Returns:
            SearchResult with optimization results
        """
        optimizer = self.create_optimizer(algorithm_type)
        return optimizer.search(objective_function, parameter_space, initial_candidates)
    
    def compare_algorithms(self,
                          objective_function: Callable,
                          parameter_space: Dict[str, Any],
                          algorithms: List[SearchAlgorithmType],
                          n_trials: int = 3) -> Dict[str, SearchResult]:
        """
        Compare multiple algorithms on the same objective.
        
        Args:
            objective_function: Function to optimize
            parameter_space: Parameter space definition
            algorithms: List of algorithms to compare
            n_trials: Number of trials per algorithm
            
        Returns:
            Dictionary mapping algorithm names to results
        """
        results = {}
        
        for algorithm in algorithms:
            self.logger.info(f"Testing {algorithm.value}")
            algorithm_results = []
            
            for trial in range(n_trials):
                self.logger.info(f"Trial {trial + 1}/{n_trials}")
                
                optimizer = self.create_optimizer(algorithm)
                result = optimizer.search(objective_function, parameter_space)
                algorithm_results.append(result)
            
            # Average results
            avg_score = np.mean([r.best_score for r in algorithm_results])
            best_result = max(algorithm_results, key=lambda x: x.best_score)
            
            results[algorithm.value] = SearchResult(
                best_solution=best_result.best_solution,
                best_score=avg_score,
                best_parameters=best_result.best_parameters,
                search_history=best_result.search_history,
                convergence_curve=best_result.convergence_curve,
                algorithm_used=algorithm.value,
                n_evaluations=np.mean([r.n_evaluations for r in algorithm_results]),
                execution_time=np.mean([r.execution_time for r in algorithm_results]),
                converged=all(r.converged for r in algorithm_results),
                convergence_iteration=np.mean([r.convergence_iteration for r in algorithm_results]),
                final_improvement=np.mean([r.final_improvement for r in algorithm_results]),
                metadata={
                    'n_trials': n_trials,
                    'individual_results': algorithm_results
                }
            )
        
        return results


# Convenience functions
def create_search_manager(config: Optional[SearchConfig] = None) -> SearchManager:
    """Create a search manager with default configuration."""
    if config is None:
        config = SearchConfig()
    return SearchManager(config)


def optimize_with_bayesian(objective_function: Callable,
                          parameter_space: Dict[str, Any],
                          max_iterations: int = 100) -> SearchResult:
    """Quick Bayesian optimization."""
    config = SearchConfig(
        algorithm_type=SearchAlgorithmType.BAYESIAN_OPTIMIZATION,
        max_iterations=max_iterations
    )
    manager = SearchManager(config)
    return manager.optimize(objective_function, parameter_space)


def optimize_with_evolutionary(objective_function: Callable,
                              parameter_space: Dict[str, Any],
                              max_iterations: int = 100,
                              population_size: int = 50) -> SearchResult:
    """Quick evolutionary optimization."""
    config = SearchConfig(
        algorithm_type=SearchAlgorithmType.EVOLUTIONARY_ALGORITHM,
        max_iterations=max_iterations,
        population_size=population_size
    )
    manager = SearchManager(config)
    return manager.optimize(objective_function, parameter_space)


def optimize_with_grid(objective_function: Callable,
                      parameter_space: Dict[str, Any]) -> SearchResult:
    """Quick grid search optimization."""
    config = SearchConfig(
        algorithm_type=SearchAlgorithmType.GRID_SEARCH
    )
    manager = SearchManager(config)
    return manager.optimize(objective_function, parameter_space)