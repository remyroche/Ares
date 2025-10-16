"""
Advanced Search Strategies for Hybrid NAS-TAS Regime Detection.

Provides common search strategy utilities including Bayesian optimization and grid optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass
import time
from datetime import datetime
from abc import ABC, abstractmethod
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

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


@dataclass
class SearchStrategyConfig:
    """Configuration for search strategy operations."""
    max_iterations: int = 100
    n_initial_points: int = 10
    acquisition_function: str = "expected_improvement"  # "expected_improvement", "upper_confidence_bound"
    exploration_weight: float = 0.1
    convergence_threshold: float = 1e-6
    parallel_evaluations: int = 1
    random_state: int = 42
    use_bayesian_optimization: bool = True
    use_grid_optimization: bool = True


@dataclass
class OptimizationResult:
    """Result from optimization operations."""
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


class AdvancedSearchStrategy(ABC):
    """Abstract base class for advanced search strategies."""
    
    def __init__(self, config: SearchStrategyConfig):
        """Initialize the search strategy.
        
        Args:
            config: Search strategy configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def optimize(self, objective_function: Callable, parameter_space: Dict[str, Any]) -> OptimizationResult:
        """Optimize the objective function over the parameter space.
        
        Args:
            objective_function: Function to optimize
            parameter_space: Parameter space definition
            
        Returns:
            OptimizationResult with optimization results
        """
        pass


class BayesianOptimizer(AdvancedSearchStrategy):
    """Bayesian optimization using Gaussian Process regression."""
    
    def __init__(self, config: SearchStrategyConfig):
        """Initialize the Bayesian optimizer.
        
        Args:
            config: Search strategy configuration
        """
        super().__init__(config)
        
        if not BAYESIAN_OPTIMIZATION_AVAILABLE:
            self.logger.warning("⚠️ Bayesian optimization not available - scikit-learn required")
        
        self.logger.info("✅ Bayesian Optimizer initialized")
    
    def optimize(self, objective_function: Callable, parameter_space: Dict[str, Any]) -> OptimizationResult:
        """Optimize using Bayesian optimization.
        
        Args:
            objective_function: Function to optimize
            parameter_space: Parameter space definition
            
        Returns:
            OptimizationResult with optimization results
        """
        try:
            self.logger.info("🔍 Starting Bayesian optimization...")
            start_time = time.time()
            
            if not BAYESIAN_OPTIMIZATION_AVAILABLE:
                raise ImportError("Bayesian optimization requires scikit-learn")
            
            # Initialize optimization
            optimization_history = []
            best_score = -np.inf
            best_parameters = {}
            
            # Generate initial points
            initial_points = self._generate_initial_points(parameter_space)
            
            # Evaluate initial points
            for point in initial_points:
                score = objective_function(point)
                optimization_history.append({
                    'parameters': point.copy(),
                    'score': score,
                    'iteration': len(optimization_history)
                })
                
                if score > best_score:
                    best_score = score
                    best_parameters = point.copy()
            
            # Bayesian optimization loop
            for iteration in range(self.config.n_initial_points, self.config.max_iterations):
                # Fit Gaussian Process
                gp = self._fit_gaussian_process(optimization_history)
                
                # Select next point using acquisition function
                next_point = self._select_next_point(gp, parameter_space, optimization_history)
                
                # Evaluate objective function
                score = objective_function(next_point)
                
                # Update history
                optimization_history.append({
                    'parameters': next_point.copy(),
                    'score': score,
                    'iteration': iteration
                })
                
                # Update best if improved
                if score > best_score:
                    best_score = score
                    best_parameters = next_point.copy()
                
                # Check convergence
                if self._check_convergence(optimization_history):
                    self.logger.info(f"✅ Convergence reached at iteration {iteration}")
                    break
            
            execution_time = time.time() - start_time
            
            # Create convergence info
            convergence_info = {
                'total_iterations': len(optimization_history),
                'convergence_reached': len(optimization_history) < self.config.max_iterations,
                'score_improvement': best_score - optimization_history[0]['score'] if optimization_history else 0.0
            }
            
            self.logger.info(f"✅ Bayesian optimization completed: {best_score:.4f} in {execution_time:.2f}s")
            
            return OptimizationResult(
                best_parameters=best_parameters,
                best_score=best_score,
                optimization_history=optimization_history,
                convergence_info=convergence_info,
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Bayesian optimization failed: {e}")
            return OptimizationResult(
                best_parameters={},
                best_score=-np.inf,
                optimization_history=[],
                convergence_info={'error': str(e)},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _generate_initial_points(self, parameter_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate initial points for Bayesian optimization."""
        try:
            initial_points = []
            np.random.seed(self.config.random_state)
            
            for _ in range(self.config.n_initial_points):
                point = {}
                for param_name, param_config in parameter_space.items():
                    if isinstance(param_config, dict):
                        if param_config['type'] == 'continuous':
                            min_val = param_config['min']
                            max_val = param_config['max']
                            point[param_name] = np.random.uniform(min_val, max_val)
                        elif param_config['type'] == 'discrete':
                            choices = param_config['choices']
                            point[param_name] = np.random.choice(choices)
                        elif param_config['type'] == 'integer':
                            min_val = param_config['min']
                            max_val = param_config['max']
                            point[param_name] = np.random.randint(min_val, max_val + 1)
                    else:
                        # Simple range
                        if isinstance(param_config, (list, tuple)) and len(param_config) == 2:
                            min_val, max_val = param_config
                            point[param_name] = np.random.uniform(min_val, max_val)
                        else:
                            point[param_name] = param_config
                
                initial_points.append(point)
            
            return initial_points
            
        except Exception as e:
            self.logger.warning(f"⚠️ Initial point generation failed: {e}")
            return []
    
    def _fit_gaussian_process(self, optimization_history: List[Dict[str, Any]]) -> GaussianProcessRegressor:
        """Fit Gaussian Process to optimization history."""
        try:
            # Extract parameters and scores
            X = []
            y = []
            
            for record in optimization_history:
                param_vector = []
                for param_name in sorted(record['parameters'].keys()):
                    param_vector.append(record['parameters'][param_name])
                X.append(param_vector)
                y.append(record['score'])
            
            X = np.array(X)
            y = np.array(y)
            
            # Create kernel
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            
            # Fit Gaussian Process
            gp = GaussianProcessRegressor(kernel=kernel, random_state=self.config.random_state)
            gp.fit(X, y)
            
            return gp
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gaussian Process fitting failed: {e}")
            # Return a simple fallback
            return None
    
    def _select_next_point(self, gp: GaussianProcessRegressor, parameter_space: Dict[str, Any], 
                          optimization_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select next point using acquisition function."""
        try:
            if gp is None:
                # Fallback to random selection
                return self._generate_initial_points(parameter_space)[0]
            
            # Generate candidate points
            n_candidates = 1000
            candidates = self._generate_initial_points(parameter_space)
            candidates = candidates * (n_candidates // len(candidates) + 1)
            candidates = candidates[:n_candidates]
            
            # Evaluate acquisition function
            best_acquisition = -np.inf
            best_point = candidates[0]
            
            for candidate in candidates:
                # Convert to vector
                param_vector = []
                for param_name in sorted(candidate.keys()):
                    param_vector.append(candidate[param_name])
                param_vector = np.array(param_vector).reshape(1, -1)
                
                # Get GP predictions
                mean, std = gp.predict(param_vector, return_std=True)
                
                # Calculate acquisition function
                if self.config.acquisition_function == "expected_improvement":
                    acquisition = self._expected_improvement(mean, std, optimization_history)
                elif self.config.acquisition_function == "upper_confidence_bound":
                    acquisition = self._upper_confidence_bound(mean, std)
                else:
                    acquisition = mean[0]  # Fallback to mean
                
                if acquisition > best_acquisition:
                    best_acquisition = acquisition
                    best_point = candidate
            
            return best_point
            
        except Exception as e:
            self.logger.warning(f"⚠️ Next point selection failed: {e}")
            return self._generate_initial_points(parameter_space)[0]
    
    def _expected_improvement(self, mean: np.ndarray, std: np.ndarray, 
                            optimization_history: List[Dict[str, Any]]) -> float:
        """Calculate expected improvement acquisition function."""
        try:
            if not optimization_history:
                return mean[0]
            
            # Get best score so far
            best_score = max(record['score'] for record in optimization_history)
            
            # Calculate expected improvement
            improvement = mean[0] - best_score
            z = improvement / (std[0] + 1e-8)
            
            # Expected improvement formula
            from scipy.stats import norm
            ei = improvement * norm.cdf(z) + std[0] * norm.pdf(z)
            
            return ei
            
        except Exception:
            return mean[0]
    
    def _upper_confidence_bound(self, mean: np.ndarray, std: np.ndarray) -> float:
        """Calculate upper confidence bound acquisition function."""
        try:
            return mean[0] + self.config.exploration_weight * std[0]
        except Exception:
            return mean[0]
    
    def _check_convergence(self, optimization_history: List[Dict[str, Any]]) -> bool:
        """Check if optimization has converged."""
        try:
            if len(optimization_history) < 10:
                return False
            
            # Check if improvement is below threshold
            recent_scores = [record['score'] for record in optimization_history[-10:]]
            score_std = np.std(recent_scores)
            
            return score_std < self.config.convergence_threshold
            
        except Exception:
            return False


class GridOptimizer(AdvancedSearchStrategy):
    """Grid search optimization for parameter space exploration."""
    
    def __init__(self, config: SearchStrategyConfig):
        """Initialize the grid optimizer.
        
        Args:
            config: Search strategy configuration
        """
        super().__init__(config)
        self.logger.info("✅ Grid Optimizer initialized")
    
    def optimize(self, objective_function: Callable, parameter_space: Dict[str, Any]) -> OptimizationResult:
        """Optimize using grid search.
        
        Args:
            objective_function: Function to optimize
            parameter_space: Parameter space definition
            
        Returns:
            OptimizationResult with optimization results
        """
        try:
            self.logger.info("🔍 Starting grid optimization...")
            start_time = time.time()
            
            # Generate parameter grid
            param_grid = self._generate_parameter_grid(parameter_space)
            
            # Initialize optimization
            optimization_history = []
            best_score = -np.inf
            best_parameters = {}
            
            # Evaluate all parameter combinations
            for i, params in enumerate(param_grid):
                try:
                    score = objective_function(params)
                    optimization_history.append({
                        'parameters': params.copy(),
                        'score': score,
                        'iteration': i
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_parameters = params.copy()
                    
                    # Log progress
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"   Evaluated {i + 1}/{len(param_grid)} combinations")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Evaluation failed for parameters {params}: {e}")
                    optimization_history.append({
                        'parameters': params.copy(),
                        'score': -np.inf,
                        'iteration': i,
                        'error': str(e)
                    })
            
            execution_time = time.time() - start_time
            
            # Create convergence info
            convergence_info = {
                'total_combinations': len(param_grid),
                'successful_evaluations': len([h for h in optimization_history if h['score'] != -np.inf]),
                'score_improvement': best_score - optimization_history[0]['score'] if optimization_history else 0.0
            }
            
            self.logger.info(f"✅ Grid optimization completed: {best_score:.4f} in {execution_time:.2f}s")
            
            return OptimizationResult(
                best_parameters=best_parameters,
                best_score=best_score,
                optimization_history=optimization_history,
                convergence_info=convergence_info,
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Grid optimization failed: {e}")
            return OptimizationResult(
                best_parameters={},
                best_score=-np.inf,
                optimization_history=[],
                convergence_info={'error': str(e)},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _generate_parameter_grid(self, parameter_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate parameter grid for grid search."""
        try:
            # Convert parameter space to sklearn ParameterGrid format
            param_grid_dict = {}
            
            for param_name, param_config in parameter_space.items():
                if isinstance(param_config, dict):
                    if param_config['type'] == 'continuous':
                        # Create discrete range for continuous parameters
                        min_val = param_config['min']
                        max_val = param_config['max']
                        n_points = param_config.get('n_points', 10)
                        param_grid_dict[param_name] = np.linspace(min_val, max_val, n_points).tolist()
                    elif param_config['type'] == 'discrete':
                        param_grid_dict[param_name] = param_config['choices']
                    elif param_config['type'] == 'integer':
                        min_val = param_config['min']
                        max_val = param_config['max']
                        param_grid_dict[param_name] = list(range(min_val, max_val + 1))
                else:
                    # Simple range or single value
                    if isinstance(param_config, (list, tuple)) and len(param_config) == 2:
                        min_val, max_val = param_config
                        param_grid_dict[param_name] = np.linspace(min_val, max_val, 10).tolist()
                    else:
                        param_grid_dict[param_name] = [param_config]
            
            # Generate all combinations
            param_grid = list(ParameterGrid(param_grid_dict))
            
            # Limit grid size if too large
            max_combinations = 10000
            if len(param_grid) > max_combinations:
                self.logger.warning(f"⚠️ Grid too large ({len(param_grid)} combinations), sampling {max_combinations}")
                np.random.seed(self.config.random_state)
                param_grid = np.random.choice(param_grid, max_combinations, replace=False).tolist()
            
            return param_grid
            
        except Exception as e:
            self.logger.warning(f"⚠️ Parameter grid generation failed: {e}")
            return []


class SearchStrategyManager:
    """Manager for coordinating different search strategies."""
    
    def __init__(self, config: SearchStrategyConfig):
        """Initialize the search strategy manager.
        
        Args:
            config: Search strategy configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize optimizers
        self.bayesian_optimizer = None
        self.grid_optimizer = None
        self.evolutionary_optimizer = None

        if config.use_bayesian_optimization and BAYESIAN_OPTIMIZATION_AVAILABLE:
            self.bayesian_optimizer = BayesianOptimizer(config)

        if config.use_grid_optimization:
            self.grid_optimizer = GridOptimizer(config)

        # Always initialize evolutionary optimizer
        self.evolutionary_optimizer = EvolutionarySearch(config)

        self.logger.info("✅ Search Strategy Manager initialized")
    
    def optimize_with_strategy(self, objective_function: Callable, parameter_space: Dict[str, Any],
                              strategy: str = "auto") -> OptimizationResult:
        """Optimize using specified strategy.

        Args:
            objective_function: Function to optimize
            parameter_space: Parameter space definition
            strategy: Optimization strategy ("bayesian", "grid", "evolutionary", "auto")

        Returns:
            OptimizationResult with optimization results
        """
        try:
            self.logger.info(f"🔍 Starting optimization with strategy: {strategy}")

            if strategy == "bayesian" or (strategy == "auto" and self.bayesian_optimizer is not None):
                if self.bayesian_optimizer is None:
                    raise ValueError("Bayesian optimizer not available")
                return self.bayesian_optimizer.optimize(objective_function, parameter_space)

            elif strategy == "grid" or (strategy == "auto" and self.grid_optimizer is not None):
                if self.grid_optimizer is None:
                    raise ValueError("Grid optimizer not available")
                return self.grid_optimizer.optimize(objective_function, parameter_space)

            elif strategy == "evolutionary" or strategy == "auto":
                # Use evolutionary optimizer as fallback or default
                return self.evolutionary_optimizer.optimize(objective_function, parameter_space)

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        except Exception as e:
            self.logger.error(f"❌ Strategy optimization failed: {e}")
            return OptimizationResult(
                best_parameters={},
                best_score=-np.inf,
                optimization_history=[],
                convergence_info={'error': str(e)},
                execution_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def compare_strategies(self, objective_function: Callable, parameter_space: Dict[str, Any]) -> Dict[str, OptimizationResult]:
        """Compare different optimization strategies.
        
        Args:
            objective_function: Function to optimize
            parameter_space: Parameter space definition
            
        Returns:
            Dictionary mapping strategy names to results
        """
        try:
            results = {}
            
            # Run Bayesian optimization if available
            if self.bayesian_optimizer is not None:
                self.logger.info("🔍 Running Bayesian optimization...")
                results['bayesian'] = self.bayesian_optimizer.optimize(objective_function, parameter_space)
            
            # Run grid optimization if available
            if self.grid_optimizer is not None:
                self.logger.info("🔍 Running grid optimization...")
                results['grid'] = self.grid_optimizer.optimize(objective_function, parameter_space)

            # Always run evolutionary optimization
            self.logger.info("🔍 Running evolutionary optimization...")
            results['evolutionary'] = self.evolutionary_optimizer.optimize(objective_function, parameter_space)
            
            # Compare results
            if results:
                best_strategy = max(results.keys(), key=lambda k: results[k].best_score)
                self.logger.info(f"✅ Best strategy: {best_strategy} (score: {results[best_strategy].best_score:.4f})")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Strategy comparison failed: {e}")
            return {}


def create_search_strategy_manager(config: SearchStrategyConfig) -> SearchStrategyManager:
    """Create a search strategy manager instance.

    Args:
        config: Search strategy configuration

    Returns:
        SearchStrategyManager instance
    """
    return SearchStrategyManager(config)


class EvolutionarySearch(AdvancedSearchStrategy):
    """Evolutionary search algorithm for optimization."""

    def __init__(self, config: SearchStrategyConfig):
        """Initialize the evolutionary search.

        Args:
            config: Search strategy configuration
        """
        super().__init__(config)
        self.logger.info("✅ Evolutionary Search initialized")

    def optimize(self, objective_function: Callable, parameter_space: Dict[str, Any]) -> OptimizationResult:
        """Optimize using evolutionary search.

        Args:
            objective_function: Function to optimize
            parameter_space: Parameter space definition

        Returns:
            OptimizationResult with optimization results
        """
        try:
            self.logger.info("🔍 Starting evolutionary optimization...")
            start_time = time.time()

            # Initialize population
            population = self._initialize_population(parameter_space)
            best_score = -np.inf
            best_parameters = {}

            optimization_history = []

            # Evolutionary loop
            for generation in range(self.config.max_iterations):
                self.logger.info(f"🔄 Generation {generation + 1}/{self.config.max_iterations}")

                # Evaluate population
                for individual in population:
                    score = objective_function(individual)
                    individual['fitness'] = score
                    optimization_history.append({
                        'parameters': individual.copy(),
                        'score': score,
                        'generation': generation
                    })

                    if score > best_score:
                        best_score = score
                        best_parameters = individual.copy()

                # Select best individuals
                population = self._select_best_individuals(population, self.config.max_iterations)

                # Apply crossover
                population = self._apply_crossover(population, parameter_space)

                # Apply mutation
                population = self._apply_mutation(population, parameter_space)

                # Check convergence
                if self._check_convergence(optimization_history):
                    self.logger.info(f"✅ Convergence reached at generation {generation}")
                    break

            execution_time = time.time() - start_time

            # Create convergence info
            convergence_info = {
                'total_generations': len(set(h['generation'] for h in optimization_history)),
                'convergence_reached': len(set(h['generation'] for h in optimization_history)) < self.config.max_iterations,
                'score_improvement': best_score - optimization_history[0]['score'] if optimization_history else 0.0
            }

            self.logger.info(f"✅ Evolutionary optimization completed: {best_score:.4f} in {execution_time:.2f}s")

            return OptimizationResult(
                best_parameters=best_parameters,
                best_score=best_score,
                optimization_history=optimization_history,
                convergence_info=convergence_info,
                execution_time=execution_time,
                success=True
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Evolutionary optimization failed: {e}")
            return OptimizationResult(
                best_parameters={},
                best_score=-np.inf,
                optimization_history=[],
                convergence_info={'error': str(e)},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )

    def _initialize_population(self, parameter_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Initialize population for evolutionary search."""
        try:
            population = []
            np.random.seed(self.config.random_state)

            for _ in range(self.config.n_initial_points):
                individual = {}
                for param_name, param_config in parameter_space.items():
                    if isinstance(param_config, dict):
                        if param_config['type'] == 'continuous':
                            min_val = param_config['min']
                            max_val = param_config['max']
                            individual[param_name] = np.random.uniform(min_val, max_val)
                        elif param_config['type'] == 'discrete':
                            choices = param_config['choices']
                            individual[param_name] = np.random.choice(choices)
                        elif param_config['type'] == 'integer':
                            min_val = param_config['min']
                            max_val = param_config['max']
                            individual[param_name] = np.random.randint(min_val, max_val + 1)
                    else:
                        # Simple range
                        if isinstance(param_config, (list, tuple)) and len(param_config) == 2:
                            min_val, max_val = param_config
                            individual[param_name] = np.random.uniform(min_val, max_val)
                        else:
                            individual[param_name] = param_config

                population.append(individual)

            return population

        except Exception as e:
            self.logger.warning(f"⚠️ Population initialization failed: {e}")
            return []

    def _select_best_individuals(self, population: List[Dict[str, Any]], elite_size: int) -> List[Dict[str, Any]]:
        """Select best individuals for next generation."""
        try:
            # Sort by fitness
            sorted_population = sorted(population, key=lambda x: x.get('fitness', 0), reverse=True)

            # Select elite individuals
            elite = sorted_population[:elite_size]

            return elite

        except Exception as e:
            self.logger.warning(f"⚠️ Individual selection failed: {e}")
            return population

    def _apply_crossover(self, population: List[Dict[str, Any]], parameter_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply crossover to create new individuals."""
        try:
            if len(population) < 2:
                return population

            new_population = population.copy()

            for i in range(len(population)):
                # Select two parents randomly
                parent1 = np.random.choice(population)
                parent2 = np.random.choice(population)

                # Create offspring through crossover
                offspring = {}

                for param_name in parameter_space.keys():
                    if np.random.random() < 0.5:
                        offspring[param_name] = parent1[param_name]
                    else:
                        offspring[param_name] = parent2[param_name]

                new_population.append(offspring)

            return new_population

        except Exception as e:
            self.logger.warning(f"⚠️ Crossover failed: {e}")
            return population

    def _apply_mutation(self, population: List[Dict[str, Any]], parameter_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply mutation to individuals."""
        try:
            mutated_population = []

            for individual in population:
                mutated_individual = individual.copy()

                for param_name, param_config in parameter_space.items():
                    if np.random.random() < 0.1:  # 10% mutation rate
                        if isinstance(param_config, dict):
                            if param_config['type'] == 'continuous':
                                min_val = param_config['min']
                                max_val = param_config['max']
                                # Small mutation
                                current_val = mutated_individual[param_name]
                                mutation = np.random.normal(0, (max_val - min_val) * 0.1)
                                mutated_val = np.clip(current_val + mutation, min_val, max_val)
                                mutated_individual[param_name] = mutated_val
                            elif param_config['type'] == 'discrete':
                                choices = param_config['choices']
                                mutated_individual[param_name] = np.random.choice(choices)
                            elif param_config['type'] == 'integer':
                                min_val = param_config['min']
                                max_val = param_config['max']
                                current_val = mutated_individual[param_name]
                                mutation = np.random.randint(-2, 3)  # Small integer mutation
                                mutated_val = np.clip(current_val + mutation, min_val, max_val)
                                mutated_individual[param_name] = mutated_val

                mutated_population.append(mutated_individual)

            return mutated_population

        except Exception as e:
            self.logger.warning(f"⚠️ Mutation failed: {e}")
            return population

    def _check_convergence(self, optimization_history: List[Dict[str, Any]]) -> bool:
        """Check if optimization has converged."""
        try:
            if len(optimization_history) < 10:
                return False

            # Check if improvement is below threshold
            recent_scores = [record['score'] for record in optimization_history[-10:]]
            score_std = np.std(recent_scores)

            return score_std < self.config.convergence_threshold

        except Exception:
            return False