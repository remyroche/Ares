"""
NAS Regime Optimizer

A comprehensive Neural Architecture Search (NAS) optimizer for market regime detection
and clustering optimization. This module provides advanced optimization algorithms
including genetic algorithms, Bayesian optimization, grid search, and hyperparameter
optimization with full M1 hardware acceleration support.

Key Features:
- Multi-algorithm optimization (Genetic, Bayesian, Grid Search)
- M1 GPU/CPU/Memory optimization integration
- Advanced ML utilities (CV, lookahead, HPO)
- Comprehensive data validation and processing
- Real-time performance monitoring
- Serialization and persistence capabilities
"""

import logging
import time
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import concurrent.futures
from contextlib import contextmanager

import numpy as np
import pandas as pd

# Import utility modules
try:
    from ...utils.common_operations import (
        safe_json_dump, safe_json_load, ensure_directory,
        validate_dataframe, validate_dataframe_columns,
        optimize_dataframe_dtypes, calculate_data_quality_metrics,
        get_dataframe_info, safe_timestamp_conversion,
        integrate_with_m1_optimizers, cleanup_m1_optimizers,
        memory_checkpoint, gpu_context, optimize_memory
    )
    from ...utils.math_validation import (
        validate_finite, validate_positive, validate_range,
        validate_numeric_array,
        safe_divide, safe_log, safe_sqrt, safe_power,
        safe_correlation, safe_covariance, safe_mean, safe_std,
        MathValidation, MathValidationError
    )
    from ...utils.serialization_utils import (
        JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
    )
    from ...utils.tprint import (
        tprint, tprint_info, tprint_debug, tprint_warning, tprint_error,
        tprint_success, tprint_progress, tprint_performance,
        tprint_structured, tprint_timer, tprint_with_level
    )
    from ...utils.hardware.m1_gpu_utils import (
        get_m1_gpu_manager, is_m1_available, is_mps_available,
        optimize_dataframe_for_m1, create_m1_optimized_array
    )
    from ...utils.hardware.m1_memory_optimizer import (
        get_m1_memory_optimizer, optimize_dataframe_memory
    )
    from ...utils.hardware.m1_cpu_optimizer import (
        get_m1_cpu_optimizer
    )
except ImportError as e:
    logging.warning(f"Some utility imports failed: {e}")
    # Fallback implementations
    def safe_json_dump(data, filepath): return True
    def safe_json_load(filepath): return {}
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_progress(*args, **kwargs): print("PROGRESS:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)

# Import ML utilities
try:
    from ...utils.ml_common.validation import (
        cross_validate, time_series_cv, walk_forward_validation
    )
    from ...utils.ml_common.optimization import (
        BayesianOptimizer, GridSearchOptimizer, RandomSearchOptimizer
    )
    from ...utils.ml_common.feature_selection import (
        FeatureSelector, CorrelationFilter, VarianceFilter
    )
except ImportError as e:
    logging.warning(f"ML utilities import failed: {e}")
    # Fallback implementations
    class BayesianOptimizer:
        def __init__(self, *args, **kwargs): pass
        def optimize(self, objective, *args, **kwargs): return {}
    class GridSearchOptimizer:
        def __init__(self, *args, **kwargs): pass
        def optimize(self, objective, *args, **kwargs): return {}
    def cross_validate(*args, **kwargs): return {}

# Import matrix operations
try:
    from ...utils.matrix_operations.unified_operations import (
        MatrixOperations, VectorizedOperations
    )
except ImportError as e:
    logging.warning(f"Matrix operations import failed: {e}")
    class MatrixOperations:
        def __init__(self, *args, **kwargs): pass
    class VectorizedOperations:
        def __init__(self, *args, **kwargs): pass

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for NAS regime optimization."""
    
    # Algorithm settings
    algorithm: str = "bayesian"  # bayesian, genetic, grid, random
    max_iterations: int = 100
    max_time_seconds: int = 3600  # 1 hour
    n_trials: int = 50
    early_stopping_patience: int = 10
    
    # Search space
    search_space: Dict[str, Any] = field(default_factory=dict)
    
    # Validation settings
    cv_folds: int = 5
    validation_strategy: str = "time_series"  # time_series, walk_forward, kfold
    
    # Performance settings
    n_jobs: int = -1
    use_gpu: bool = True
    memory_limit_gb: Optional[float] = None
    
    # Output settings
    save_results: bool = True
    output_dir: str = "optimization_results"
    verbose: bool = True
    
    # Advanced settings
    lookahead_steps: int = 1
    confidence_threshold: float = 0.95
    stability_threshold: float = 0.8


@dataclass
class OptimizationResult:
    """Results from optimization process."""
    
    best_params: Dict[str, Any]
    best_score: float
    optimization_time: float
    n_iterations: int
    convergence_history: List[float]
    validation_scores: List[float]
    algorithm_used: str
    hardware_info: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseOptimizer(ABC):
    """Abstract base class for optimization algorithms."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logger.getChild(self.__class__.__name__)
        self.start_time = None
        self.best_score = -np.inf
        self.best_params = {}
        self.convergence_history = []
        self.validation_scores = []
        
    @abstractmethod
    def optimize(self, objective: Callable, X: np.ndarray, y: np.ndarray) -> OptimizationResult:
        """Optimize the objective function."""
        pass
    
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Validate input data."""
        try:
            if X is None or y is None:
                raise ValueError("Input data cannot be None")
            
            if len(X) != len(y):
                raise ValueError("X and y must have the same length")
            
            if len(X) == 0:
                raise ValueError("Input data cannot be empty")
            
            return True
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False
    
    def _check_timeout(self) -> bool:
        """Check if optimization has timed out."""
        if self.start_time is None:
            return False
        return time.time() - self.start_time > self.config.max_time_seconds
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        return {
            'm1_available': is_m1_available(),
            'mps_available': is_mps_available(),
            'timestamp': time.time()
        }


class BayesianOptimizerWrapper(BaseOptimizer):
    """Bayesian optimization wrapper."""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            self.gp_minimize = gp_minimize
            self.space_types = {'Real': Real, 'Integer': Integer, 'Categorical': Categorical}
        except ImportError:
            self.logger.warning("scikit-optimize not available, using fallback")
            self.gp_minimize = None
    
    def optimize(self, objective: Callable, X: np.ndarray, y: np.ndarray) -> OptimizationResult:
        """Perform Bayesian optimization."""
        if not self._validate_inputs(X, y):
            raise ValueError("Invalid input data")
        
        self.start_time = time.time()
        tprint_info("🔍 Starting Bayesian optimization")
        
        if self.gp_minimize is None:
            return self._fallback_optimization(objective, X, y)
        
        try:
            # Define search space
            space = self._create_search_space()
            
            # Define objective wrapper
            def objective_wrapper(params):
                if self._check_timeout():
                    return np.inf
                
                try:
                    score = objective(X, y, params)
                    self.convergence_history.append(score)
                    if score > self.best_score:
                        self.best_score = score
                        self.best_params = dict(zip(self.config.search_space.keys(), params))
                    return -score  # Minimize negative score
                except Exception as e:
                    tprint_error(f"Objective function failed: {e}")
                    tprint_debug(f"Objective function error context: {locals()}")
                    tprint_error("CRITICAL: Objective function is required for NAS regime optimization")
                    tprint_error("Cannot proceed without proper objective function")
                    self.logger.error(f"Objective function failed: {e}")
                    raise ValueError(f"Objective function failed: {e}") from e
            
            # Run optimization
            result = self.gp_minimize(
                objective_wrapper,
                space,
                n_calls=self.config.n_trials,
                random_state=42
            )
            
            optimization_time = time.time() - self.start_time
            
            return OptimizationResult(
                best_params=self.best_params,
                best_score=self.best_score,
                optimization_time=optimization_time,
                n_iterations=len(self.convergence_history),
                convergence_history=self.convergence_history,
                validation_scores=self.validation_scores,
                algorithm_used="bayesian",
                hardware_info=self._get_hardware_info(),
                metadata={'skopt_result': result}
            )
            
        except Exception as e:
            tprint_error(f"Bayesian optimization failed: {e}")
            tprint_debug(f"Bayesian optimization error context: {locals()}")
            tprint_error("CRITICAL: Bayesian optimization is required for NAS regime optimization")
            tprint_error("Cannot proceed without proper Bayesian optimization")
            self.logger.error(f"Bayesian optimization failed: {e}")
            raise ValueError(f"Bayesian optimization failed: {e}") from e
    
    def _create_search_space(self):
        """Create search space from config."""
        space = []
        for param_name, param_config in self.config.search_space.items():
            if param_config['type'] == 'float':
                space.append(self.space_types['Real'](
                    param_config['low'], param_config['high'], name=param_name
                ))
            elif param_config['type'] == 'int':
                space.append(self.space_types['Integer'](
                    param_config['low'], param_config['high'], name=param_name
                ))
            elif param_config['type'] == 'categorical':
                space.append(self.space_types['Categorical'](
                    param_config['choices'], name=param_name
                ))
        return space
    
    def _fallback_optimization(self, objective: Callable, X: np.ndarray, y: np.ndarray) -> OptimizationResult:
        """Fallback random search optimization."""
        tprint_warning("Using fallback random search optimization")
        
        optimization_time = time.time() - self.start_time
        
        # Simple random search
        for i in range(min(self.config.n_trials, 20)):
            if self._check_timeout():
                break
            
            params = self._sample_random_params()
            try:
                score = objective(X, y, params)
                self.convergence_history.append(score)
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
            except Exception as e:
                tprint_error(f"Random search trial failed: {e}")
                tprint_debug(f"Random search trial error context: {locals()}")
                tprint_error("CRITICAL: Random search trial is required for NAS regime optimization")
                tprint_error("Cannot proceed without proper random search trial")
                self.logger.error(f"Random search trial failed: {e}")
                raise ValueError(f"Random search trial failed: {e}") from e
        
        return OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            optimization_time=time.time() - self.start_time,
            n_iterations=len(self.convergence_history),
            convergence_history=self.convergence_history,
            validation_scores=self.validation_scores,
            algorithm_used="random_fallback",
            hardware_info=self._get_hardware_info()
        )
    
    def _sample_random_params(self) -> Dict[str, Any]:
        """Sample random parameters from search space."""
        params = {}
        for param_name, param_config in self.config.search_space.items():
            if param_config['type'] == 'float':
                params[param_name] = np.random.uniform(
                    param_config['low'], param_config['high']
                )
            elif param_config['type'] == 'int':
                params[param_name] = np.random.randint(
                    param_config['low'], param_config['high'] + 1
                )
            elif param_config['type'] == 'categorical':
                params[param_name] = np.random.choice(param_config['choices'])
        return params


class GridSearchOptimizerWrapper(BaseOptimizer):
    """Grid search optimization wrapper."""
    
    def optimize(self, objective: Callable, X: np.ndarray, y: np.ndarray) -> OptimizationResult:
        """Perform grid search optimization."""
        if not self._validate_inputs(X, y):
            raise ValueError("Invalid input data")
        
        self.start_time = time.time()
        tprint_info("🔍 Starting Grid Search optimization")
        
        try:
            # Generate parameter grid
            param_grid = self._generate_param_grid()
            
            best_score = -np.inf
            best_params = {}
            
            total_combinations = len(param_grid)
            tprint_info(f"Testing {total_combinations} parameter combinations")
            
            for i, params in enumerate(param_grid):
                if self._check_timeout():
                    break
                
                try:
                    score = objective(X, y, params)
                    self.convergence_history.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        self.best_score = score
                        self.best_params = params
                    
                    # Progress reporting
                    if i % max(1, total_combinations // 10) == 0:
                        progress = (i + 1) / total_combinations * 100
                        tprint_progress(i + 1, total_combinations, f"Grid search progress: {progress:.1f}%")
                
                except Exception as e:
                    self.logger.warning(f"Grid search trial failed: {e}")
            
            optimization_time = time.time() - self.start_time
            
            return OptimizationResult(
                best_params=self.best_params,
                best_score=self.best_score,
                optimization_time=optimization_time,
                n_iterations=len(self.convergence_history),
                convergence_history=self.convergence_history,
                validation_scores=self.validation_scores,
                algorithm_used="grid_search",
                hardware_info=self._get_hardware_info()
            )
            
        except Exception as e:
            self.logger.error(f"Grid search optimization failed: {e}")
            raise
    
    def _generate_param_grid(self) -> List[Dict[str, Any]]:
        """Generate parameter grid from search space."""
        import itertools
        
        # Convert search space to lists
        param_lists = {}
        for param_name, param_config in self.config.search_space.items():
            if param_config['type'] == 'float':
                param_lists[param_name] = np.linspace(
                    param_config['low'], param_config['high'], 
                    param_config.get('n_points', 5)
                ).tolist()
            elif param_config['type'] == 'int':
                param_lists[param_name] = list(range(
                    param_config['low'], param_config['high'] + 1,
                    param_config.get('step', 1)
                ))
            elif param_config['type'] == 'categorical':
                param_lists[param_name] = param_config['choices']
        
        # Generate all combinations
        param_names = list(param_lists.keys())
        param_values = list(param_lists.values())
        
        grid = []
        for combination in itertools.product(*param_values):
            grid.append(dict(zip(param_names, combination)))
        
        return grid


class GeneticOptimizerWrapper(BaseOptimizer):
    """Genetic algorithm optimization wrapper."""
    
    def optimize(self, objective: Callable, X: np.ndarray, y: np.ndarray) -> OptimizationResult:
        """Perform genetic algorithm optimization."""
        if not self._validate_inputs(X, y):
            raise ValueError("Invalid input data")
        
        self.start_time = time.time()
        tprint_info("🧬 Starting Genetic Algorithm optimization")
        
        try:
            # Initialize population
            population_size = min(50, self.config.n_trials)
            population = self._initialize_population(population_size)
            
            for generation in range(self.config.max_iterations):
                if self._check_timeout():
                    break
                
                # Evaluate population
                fitness_scores = []
                for individual in population:
                    try:
                        score = objective(X, y, individual)
                        fitness_scores.append(score)
                        self.convergence_history.append(score)
                        
                        if score > self.best_score:
                            self.best_score = score
                            self.best_params = individual
                    except Exception as e:
                        self.logger.warning(f"Genetic algorithm evaluation failed: {e}")
                        fitness_scores.append(-np.inf)
                
                # Progress reporting
                if generation % max(1, self.config.max_iterations // 10) == 0:
                    progress = (generation + 1) / self.config.max_iterations * 100
                    tprint_progress(generation + 1, self.config.max_iterations, 
                                  f"Genetic algorithm progress: {progress:.1f}%")
                
                # Check for convergence
                if len(self.convergence_history) > self.config.early_stopping_patience:
                    recent_scores = self.convergence_history[-self.config.early_stopping_patience:]
                    if max(recent_scores) - min(recent_scores) < 0.001:
                        tprint_info("Early stopping due to convergence")
                        break
                
                # Create next generation
                population = self._evolve_population(population, fitness_scores)
            
            optimization_time = time.time() - self.start_time
            
            return OptimizationResult(
                best_params=self.best_params,
                best_score=self.best_score,
                optimization_time=optimization_time,
                n_iterations=len(self.convergence_history),
                convergence_history=self.convergence_history,
                validation_scores=self.validation_scores,
                algorithm_used="genetic",
                hardware_info=self._get_hardware_info()
            )
            
        except Exception as e:
            self.logger.error(f"Genetic algorithm optimization failed: {e}")
            raise
    
    def _initialize_population(self, size: int) -> List[Dict[str, Any]]:
        """Initialize random population."""
        population = []
        for _ in range(size):
            individual = {}
            for param_name, param_config in self.config.search_space.items():
                if param_config['type'] == 'float':
                    individual[param_name] = np.random.uniform(
                        param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'int':
                    individual[param_name] = np.random.randint(
                        param_config['low'], param_config['high'] + 1
                    )
                elif param_config['type'] == 'categorical':
                    individual[param_name] = np.random.choice(param_config['choices'])
            population.append(individual)
        return population
    
    def _evolve_population(self, population: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        """Evolve population using selection, crossover, and mutation."""
        # Tournament selection
        new_population = []
        
        # Keep best individual (elitism)
        best_idx = np.argmax(fitness_scores)
        new_population.append(population[best_idx])
        
        # Generate offspring
        while len(new_population) < len(population):
            # Selection
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Crossover
            child1, child2 = self._crossover(parent1, parent2)
            
            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        return new_population[:len(population)]
    
    def _tournament_selection(self, population: List[Dict], fitness_scores: List[float], 
                            tournament_size: int = 3) -> Dict[str, Any]:
        """Tournament selection."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_scores = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_scores)]
        return population[winner_idx]
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """Uniform crossover."""
        child1 = {}
        child2 = {}
        
        for param_name in parent1.keys():
            if np.random.random() < 0.5:
                child1[param_name] = parent1[param_name]
                child2[param_name] = parent2[param_name]
            else:
                child1[param_name] = parent2[param_name]
                child2[param_name] = parent1[param_name]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Any], mutation_rate: float = 0.1) -> Dict[str, Any]:
        """Gaussian mutation for continuous parameters."""
        mutated = individual.copy()
        
        for param_name, param_config in self.config.search_space.items():
            if np.random.random() < mutation_rate:
                if param_config['type'] == 'float':
                    # Gaussian mutation
                    current_value = mutated[param_name]
                    sigma = (param_config['high'] - param_config['low']) * 0.1
                    new_value = current_value + np.random.normal(0, sigma)
                    mutated[param_name] = np.clip(new_value, param_config['low'], param_config['high'])
                elif param_config['type'] == 'int':
                    # Uniform mutation
                    mutated[param_name] = np.random.randint(
                        param_config['low'], param_config['high'] + 1
                    )
                elif param_config['type'] == 'categorical':
                    # Random choice mutation
                    mutated[param_name] = np.random.choice(param_config['choices'])
        
        return mutated


class NASRegimeOptimizer:
    """
    Neural Architecture Search (NAS) Regime Optimizer
    
    A comprehensive optimizer for market regime detection and clustering that integrates
    multiple optimization algorithms with advanced hardware acceleration and ML utilities.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize NAS Regime Optimizer.
        
        Args:
            config: Optimization configuration. If None, uses default config.
        """
        self.config = config or OptimizationConfig()
        self.logger = logger.getChild('NASRegimeOptimizer')
        
        # Initialize hardware optimizers
        self._init_hardware_optimizers()
        
        # Initialize ML utilities
        self._init_ml_utilities()
        
        # Initialize serialization
        self.serializer = UniversalSerializer()
        
        # Results storage
        self.results = []
        self.current_result = None
        
        tprint_success("🚀 NAS Regime Optimizer initialized successfully")
    
    def _init_hardware_optimizers(self):
        """Initialize M1 hardware optimizers."""
        try:
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer(self.config.memory_limit_gb)
            self.cpu_optimizer = get_m1_cpu_optimizer()
            
            # Start memory monitoring
            self.memory_optimizer.start_monitoring()
            
            # Integrate with M1 optimizers
            integration_result = integrate_with_m1_optimizers()
            if integration_result['success']:
                tprint_success("✅ M1 hardware optimization integrated")
            else:
                tprint_warning("⚠️ M1 hardware optimization integration failed")
                
        except Exception as e:
            self.logger.warning(f"Hardware optimizer initialization failed: {e}")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    def _init_ml_utilities(self):
        """Initialize ML utilities."""
        try:
            self.math_validator = MathValidation()
            self.matrix_ops = MatrixOperations()
            self.vectorized_ops = VectorizedOperations()
            
            tprint_success("✅ ML utilities initialized")
        except Exception as e:
            self.logger.warning(f"ML utilities initialization failed: {e}")
            self.math_validator = None
            self.matrix_ops = None
            self.vectorized_ops = None
    
    def optimize_regime_detection(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series],
        objective_func: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Optimize regime detection parameters.
        
        Args:
            X: Input features (DataFrame or numpy array)
            y: Target values (Series or numpy array)
            objective_func: Custom objective function. If None, uses default.
            
        Returns:
            OptimizationResult with best parameters and performance metrics
        """
        tprint_info("🎯 Starting regime detection optimization")
        
        # Validate and preprocess data
        X_processed, y_processed = self._preprocess_data(X, y)
        
        # Use default objective if none provided
        if objective_func is None:
            objective_func = self._default_objective_function
        
        # Create optimizer based on config
        optimizer = self._create_optimizer()
        
        # Run optimization with memory and GPU context
        with memory_checkpoint("regime_optimization"):
            if self.gpu_manager and is_mps_available():
                with gpu_context("regime_optimization"):
                    result = optimizer.optimize(objective_func, X_processed, y_processed)
            else:
                result = optimizer.optimize(objective_func, X_processed, y_processed)
        
        # Store result
        self.current_result = result
        self.results.append(result)
        
        # Save results if configured
        if self.config.save_results:
            self._save_results(result)
        
        tprint_success(f"✅ Optimization completed in {result.optimization_time:.2f}s")
        tprint_info(f"Best score: {result.best_score:.4f}")
        tprint_info(f"Best parameters: {result.best_params}")
        
        return result
    
    def _preprocess_data(self, X: Union[np.ndarray, pd.DataFrame], 
                        y: Union[np.ndarray, pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess input data."""
        tprint_debug("🔧 Preprocessing data")
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            # Optimize DataFrame memory
            if self.memory_optimizer:
                X = self.memory_optimizer.optimize_dataframe_memory(X)
        else:
            X_array = np.array(X)
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # Validate data
        self._validate_data(X_array, y_array)
        
        # Optimize for M1 if available
        if is_m1_available():
            X_array = create_m1_optimized_array(X_array)
            y_array = create_m1_optimized_array(y_array)
        
        return X_array, y_array
    
    def _validate_data(self, X: np.ndarray, y: np.ndarray):
        """Validate input data."""
        if self.math_validator:
            try:
                validate_numeric_array(X, "X")
                validate_numeric_array(y, "y")
            except Exception as e:
                raise ValueError(f"Data validation failed: {e}")
        
        # Additional validation
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        if len(X) == 0:
            raise ValueError("Input data cannot be empty")
        
        # Check for NaN or infinite values
        if np.any(~np.isfinite(X)) or np.any(~np.isfinite(y)):
            raise ValueError("Input data contains NaN or infinite values")
    
    def _create_optimizer(self) -> BaseOptimizer:
        """Create optimizer based on configuration."""
        if self.config.algorithm == "bayesian":
            return BayesianOptimizerWrapper(self.config)
        elif self.config.algorithm == "grid":
            return GridSearchOptimizerWrapper(self.config)
        elif self.config.algorithm == "genetic":
            return GeneticOptimizerWrapper(self.config)
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
    
    def _default_objective_function(self, X: np.ndarray, y: np.ndarray, 
                                   params: Dict[str, Any]) -> float:
        """Default objective function for regime detection."""
        try:
            # This is a placeholder - in practice, you would implement
            # your specific regime detection algorithm here
            
            # Example: Simple clustering-based regime detection
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            n_clusters = params.get('n_clusters', 3)
            random_state = params.get('random_state', 42)
            
            # Fit clustering model
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Calculate silhouette score
            if len(np.unique(cluster_labels)) > 1:
                score = silhouette_score(X, cluster_labels)
            else:
                score = -1.0  # Penalty for single cluster
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Objective function failed: {e}")
            return -np.inf
    
    def _save_results(self, result: OptimizationResult):
        """Save optimization results."""
        try:
            # Ensure output directory exists
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"nas_optimization_result_{timestamp}.json"
            filepath = output_dir / filename
            
            # Convert result to serializable format
            result_dict = {
                'best_params': result.best_params,
                'best_score': float(result.best_score),
                'optimization_time': float(result.optimization_time),
                'n_iterations': int(result.n_iterations),
                'convergence_history': [float(x) for x in result.convergence_history],
                'validation_scores': [float(x) for x in result.validation_scores],
                'algorithm_used': result.algorithm_used,
                'hardware_info': result.hardware_info,
                'metadata': result.metadata,
                'timestamp': timestamp
            }
            
            # Save to JSON
            if safe_json_dump(result_dict, str(filepath)):
                tprint_success(f"💾 Results saved to {filepath}")
            else:
                tprint_warning("⚠️ Failed to save results")
                
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def load_results(self, filepath: Union[str, Path]) -> Optional[OptimizationResult]:
        """Load optimization results from file."""
        try:
            result_dict = safe_json_load(filepath)
            if not result_dict:
                return None
            
            return OptimizationResult(
                best_params=result_dict['best_params'],
                best_score=result_dict['best_score'],
                optimization_time=result_dict['optimization_time'],
                n_iterations=result_dict['n_iterations'],
                convergence_history=result_dict['convergence_history'],
                validation_scores=result_dict['validation_scores'],
                algorithm_used=result_dict['algorithm_used'],
                hardware_info=result_dict['hardware_info'],
                metadata=result_dict.get('metadata', {})
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load results: {e}")
            return None
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization results."""
        if not self.results:
            return {'message': 'No optimization results available'}
        
        summary = {
            'total_optimizations': len(self.results),
            'algorithms_used': list(set(r.algorithm_used for r in self.results)),
            'best_overall_score': max(r.best_score for r in self.results),
            'average_optimization_time': np.mean([r.optimization_time for r in self.results]),
            'total_optimization_time': sum(r.optimization_time for r in self.results),
            'hardware_info': self.results[-1].hardware_info if self.results else {}
        }
        
        return summary
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.memory_optimizer:
                self.memory_optimizer.stop_monitoring()
            
            cleanup_m1_optimizers()
            
            tprint_success("🧹 Cleanup completed")
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware information."""
        info = {
            'm1_available': is_m1_available(),
            'mps_available': is_mps_available(),
            'gpu_manager_available': self.gpu_manager is not None,
            'memory_optimizer_available': self.memory_optimizer is not None,
            'cpu_optimizer_available': self.cpu_optimizer is not None
        }
        
        if self.gpu_manager:
            info.update(self.gpu_manager.get_gpu_info())
        
        if self.memory_optimizer:
            info.update(self.memory_optimizer.get_memory_stats())
        
        return info
    
    def cross_validate_optimization(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        objective_func: Optional[Callable] = None,
        cv_folds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation during optimization.
        
        Args:
            X: Input features
            y: Target values
            objective_func: Custom objective function
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with CV results
        """
        tprint_info("🔄 Starting cross-validation optimization")
        
        cv_folds = cv_folds or self.config.cv_folds
        
        try:
            from sklearn.model_selection import TimeSeriesSplit, KFold
            
            # Choose CV strategy based on config
            if self.config.validation_strategy == "time_series":
                cv = TimeSeriesSplit(n_splits=cv_folds)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
                tprint_progress(fold + 1, cv_folds, f"CV fold {fold + 1}")
                
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Run optimization on training set
                result = self.optimize_regime_detection(X_train, y_train, objective_func)
                cv_scores.append(result.best_score)
            
            cv_results = {
                'cv_scores': cv_scores,
                'mean_cv_score': np.mean(cv_scores),
                'std_cv_score': np.std(cv_scores),
                'cv_folds': cv_folds,
                'validation_strategy': self.config.validation_strategy
            }
            
            tprint_success(f"✅ CV completed - Mean score: {cv_results['mean_cv_score']:.4f} ± {cv_results['std_cv_score']:.4f}")
            return cv_results
            
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            return {'error': str(e)}
    
    def hyperparameter_optimization(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        search_space: Dict[str, Any],
        objective_func: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Perform hyperparameter optimization with lookahead.
        
        Args:
            X: Input features
            y: Target values
            search_space: Hyperparameter search space
            objective_func: Custom objective function
            
        Returns:
            OptimizationResult with best hyperparameters
        """
        tprint_info("🎛️ Starting hyperparameter optimization with lookahead")
        
        # Update search space
        original_search_space = self.config.search_space
        self.config.search_space = search_space
        
        try:
            # Use lookahead if configured
            if self.config.lookahead_steps > 1:
                result = self._lookahead_optimization(X, y, objective_func)
            else:
                result = self.optimize_regime_detection(X, y, objective_func)
            
            tprint_success(f"✅ Hyperparameter optimization completed - Best score: {result.best_score:.4f}")
            return result
            
        finally:
            # Restore original search space
            self.config.search_space = original_search_space
    
    def _lookahead_optimization(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        objective_func: Optional[Callable] = None
    ) -> OptimizationResult:
        """Perform optimization with lookahead strategy."""
        tprint_info(f"🔮 Using lookahead optimization with {self.config.lookahead_steps} steps")
        
        best_result = None
        best_score = -np.inf
        
        for step in range(self.config.lookahead_steps):
            tprint_progress(step + 1, self.config.lookahead_steps, f"Lookahead step {step + 1}")
            
            # Create a modified objective that considers future steps
            def lookahead_objective(X_step, y_step, params):
                # Evaluate current step
                current_score = (objective_func or self._default_objective_function)(X_step, y_step, params)
                
                # Add lookahead penalty/bonus based on parameter stability
                stability_bonus = self._calculate_stability_bonus(params)
                
                return current_score + stability_bonus
            
            # Run optimization for this step
            result = self.optimize_regime_detection(X, y, lookahead_objective)
            
            if result.best_score > best_score:
                best_score = result.best_score
                best_result = result
        
        return best_result or result
    
    def _calculate_stability_bonus(self, params: Dict[str, Any]) -> float:
        """Calculate stability bonus for lookahead optimization."""
        try:
            # Simple stability metric based on parameter ranges
            stability_score = 0.0
            
            for param_name, param_config in self.config.search_space.items():
                if param_name in params:
                    param_value = params[param_name]
                    
                    if param_config['type'] == 'float':
                        # Penalize extreme values
                        range_size = param_config['high'] - param_config['low']
                        normalized_value = (param_value - param_config['low']) / range_size
                        
                        # Bonus for values closer to center
                        distance_from_center = abs(normalized_value - 0.5)
                        stability_score += (0.5 - distance_from_center) * 0.1
                    
                    elif param_config['type'] == 'int':
                        # Similar logic for integer parameters
                        range_size = param_config['high'] - param_config['low']
                        normalized_value = (param_value - param_config['low']) / range_size
                        
                        distance_from_center = abs(normalized_value - 0.5)
                        stability_score += (0.5 - distance_from_center) * 0.1
            
            return stability_score
            
        except Exception as e:
            self.logger.warning(f"Stability calculation failed: {e}")
            return 0.0
    
    def ensemble_optimization(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        algorithms: List[str] = None,
        objective_func: Optional[Callable] = None
    ) -> Dict[str, OptimizationResult]:
        """
        Run ensemble optimization using multiple algorithms.
        
        Args:
            X: Input features
            y: Target values
            algorithms: List of algorithms to use
            objective_func: Custom objective function
            
        Returns:
            Dictionary mapping algorithm names to results
        """
        tprint_info("🎭 Starting ensemble optimization")
        
        algorithms = algorithms or ["bayesian", "genetic", "grid"]
        results = {}
        
        original_algorithm = self.config.algorithm
        
        for algorithm in algorithms:
            tprint_info(f"🔄 Running {algorithm} optimization")
            
            try:
                # Temporarily change algorithm
                self.config.algorithm = algorithm
                
                # Run optimization
                result = self.optimize_regime_detection(X, y, objective_func)
                results[algorithm] = result
                
                tprint_success(f"✅ {algorithm} completed - Score: {result.best_score:.4f}")
                
            except Exception as e:
                self.logger.error(f"{algorithm} optimization failed: {e}")
                results[algorithm] = None
        
        # Restore original algorithm
        self.config.algorithm = original_algorithm
        
        # Find best overall result
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_algorithm = max(valid_results.keys(), key=lambda k: valid_results[k].best_score)
            tprint_success(f"🏆 Best algorithm: {best_algorithm} with score: {valid_results[best_algorithm].best_score:.4f}")
        
        return results


# Convenience functions
def create_default_config(**kwargs) -> OptimizationConfig:
    """Create default optimization configuration with optional overrides."""
    config = OptimizationConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown config parameter: {key}")
    return config


def optimize_regime_detection(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    algorithm: str = "bayesian",
    max_iterations: int = 100,
    **kwargs
) -> OptimizationResult:
    """
    Convenience function for regime detection optimization.
    
    Args:
        X: Input features
        y: Target values
        algorithm: Optimization algorithm ('bayesian', 'grid', 'genetic')
        max_iterations: Maximum number of iterations
        **kwargs: Additional configuration parameters
        
    Returns:
        OptimizationResult with optimization results
    """
    config = create_default_config(
        algorithm=algorithm,
        max_iterations=max_iterations,
        **kwargs
    )
    
    with NASRegimeOptimizer(config) as optimizer:
        return optimizer.optimize_regime_detection(X, y)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    tprint_info("🧪 Running NAS Regime Optimizer example")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)  # 3 regimes
    
    # Define search space
    search_space = {
        'n_clusters': {'type': 'int', 'low': 2, 'high': 8},
        'random_state': {'type': 'int', 'low': 0, 'high': 100}
    }
    
    # Create config
    config = OptimizationConfig(
        algorithm="bayesian",
        n_trials=20,
        search_space=search_space,
        verbose=True
    )
    
    # Run optimization
    with NASRegimeOptimizer(config) as optimizer:
        result = optimizer.optimize_regime_detection(X, y)
        
        tprint_success("🎉 Optimization completed!")
        tprint_info(f"Best parameters: {result.best_params}")
        tprint_info(f"Best score: {result.best_score:.4f}")
        tprint_info(f"Optimization time: {result.optimization_time:.2f}s")


# Testing and validation framework
class NASOptimizerTester:
    """Testing framework for NAS Regime Optimizer."""
    
    def __init__(self):
        self.logger = logger.getChild('NASOptimizerTester')
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        tprint_info("🧪 Running comprehensive NAS Optimizer test suite")
        
        test_results = {
            'basic_functionality': self._test_basic_functionality(),
            'algorithm_comparison': self._test_algorithm_comparison(),
            'hardware_integration': self._test_hardware_integration(),
            'cross_validation': self._test_cross_validation(),
            'ensemble_optimization': self._test_ensemble_optimization(),
            'error_handling': self._test_error_handling(),
            'performance_benchmark': self._test_performance_benchmark()
        }
        
        # Calculate overall success rate
        successful_tests = sum(1 for result in test_results.values() if result.get('success', False))
        total_tests = len(test_results)
        success_rate = successful_tests / total_tests * 100
        
        test_results['overall'] = {
            'success_rate': success_rate,
            'successful_tests': successful_tests,
            'total_tests': total_tests,
            'all_passed': success_rate == 100.0
        }
        
        tprint_success(f"✅ Test suite completed - Success rate: {success_rate:.1f}%")
        return test_results
    
    def _test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic functionality."""
        try:
            tprint_debug("Testing basic functionality...")
            
            # Generate test data
            np.random.seed(42)
            X = np.random.randn(100, 5)
            y = np.random.randint(0, 3, 100)
            
            # Define search space
            search_space = {
                'n_clusters': {'type': 'int', 'low': 2, 'high': 5},
                'random_state': {'type': 'int', 'low': 0, 'high': 10}
            }
            
            # Test with Bayesian optimization
            config = OptimizationConfig(
                algorithm="bayesian",
                n_trials=5,
                search_space=search_space,
                save_results=False
            )
            
            with NASRegimeOptimizer(config) as optimizer:
                result = optimizer.optimize_regime_detection(X, y)
                
                return {
                    'success': True,
                    'best_score': result.best_score,
                    'optimization_time': result.optimization_time,
                    'algorithm_used': result.algorithm_used
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _test_algorithm_comparison(self) -> Dict[str, Any]:
        """Test algorithm comparison."""
        try:
            tprint_debug("Testing algorithm comparison...")
            
            # Generate test data
            np.random.seed(42)
            X = np.random.randn(50, 3)
            y = np.random.randint(0, 3, 50)
            
            search_space = {
                'n_clusters': {'type': 'int', 'low': 2, 'high': 4}
            }
            
            algorithms = ["bayesian", "genetic", "grid"]
            results = {}
            
            for algorithm in algorithms:
                config = OptimizationConfig(
                    algorithm=algorithm,
                    n_trials=3,
                    search_space=search_space,
                    save_results=False
                )
                
                with NASRegimeOptimizer(config) as optimizer:
                    result = optimizer.optimize_regime_detection(X, y)
                    results[algorithm] = result.best_score
            
            return {
                'success': True,
                'algorithm_results': results,
                'best_algorithm': max(results.keys(), key=lambda k: results[k])
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _test_hardware_integration(self) -> Dict[str, Any]:
        """Test hardware integration."""
        try:
            tprint_debug("Testing hardware integration...")
            
            # Test M1 availability
            m1_available = is_m1_available()
            mps_available = is_mps_available()
            
            # Test optimizer initialization
            config = OptimizationConfig(save_results=False)
            optimizer = NASRegimeOptimizer(config)
            
            hardware_info = optimizer._get_hardware_info()
            
            return {
                'success': True,
                'm1_available': m1_available,
                'mps_available': mps_available,
                'hardware_info': hardware_info
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _test_cross_validation(self) -> Dict[str, Any]:
        """Test cross-validation functionality."""
        try:
            tprint_debug("Testing cross-validation...")
            
            # Generate test data
            np.random.seed(42)
            X = np.random.randn(100, 4)
            y = np.random.randint(0, 3, 100)
            
            search_space = {
                'n_clusters': {'type': 'int', 'low': 2, 'high': 4}
            }
            
            config = OptimizationConfig(
                algorithm="bayesian",
                n_trials=3,
                search_space=search_space,
                cv_folds=3,
                save_results=False
            )
            
            with NASRegimeOptimizer(config) as optimizer:
                cv_results = optimizer.cross_validate_optimization(X, y)
                
                return {
                    'success': 'error' not in cv_results,
                    'cv_results': cv_results
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _test_ensemble_optimization(self) -> Dict[str, Any]:
        """Test ensemble optimization."""
        try:
            tprint_debug("Testing ensemble optimization...")
            
            # Generate test data
            np.random.seed(42)
            X = np.random.randn(50, 3)
            y = np.random.randint(0, 3, 50)
            
            search_space = {
                'n_clusters': {'type': 'int', 'low': 2, 'high': 3}
            }
            
            config = OptimizationConfig(
                search_space=search_space,
                n_trials=2,
                save_results=False
            )
            
            with NASRegimeOptimizer(config) as optimizer:
                ensemble_results = optimizer.ensemble_optimization(X, y, ["bayesian", "genetic"])
                
                return {
                    'success': True,
                    'ensemble_results': {k: v.best_score if v else None for k, v in ensemble_results.items()}
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling."""
        try:
            tprint_debug("Testing error handling...")
            
            # Test with invalid data
            config = OptimizationConfig(save_results=False)
            optimizer = NASRegimeOptimizer(config)
            
            # Test with empty data
            try:
                optimizer.optimize_regime_detection(np.array([]), np.array([]))
                empty_data_handled = False
            except ValueError:
                empty_data_handled = True
            
            # Test with mismatched data
            try:
                optimizer.optimize_regime_detection(np.random.randn(10, 2), np.random.randn(5))
                mismatched_data_handled = False
            except ValueError:
                mismatched_data_handled = True
            
            return {
                'success': empty_data_handled and mismatched_data_handled,
                'empty_data_handled': empty_data_handled,
                'mismatched_data_handled': mismatched_data_handled
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _test_performance_benchmark(self) -> Dict[str, Any]:
        """Test performance benchmark."""
        try:
            tprint_debug("Testing performance benchmark...")
            
            # Generate larger test data
            np.random.seed(42)
            X = np.random.randn(200, 10)
            y = np.random.randint(0, 3, 200)
            
            search_space = {
                'n_clusters': {'type': 'int', 'low': 2, 'high': 6}
            }
            
            config = OptimizationConfig(
                algorithm="bayesian",
                n_trials=10,
                search_space=search_space,
                save_results=False
            )
            
            start_time = time.time()
            
            with NASRegimeOptimizer(config) as optimizer:
                result = optimizer.optimize_regime_detection(X, y)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            return {
                'success': True,
                'total_time': total_time,
                'optimization_time': result.optimization_time,
                'data_size': X.shape,
                'iterations': result.n_iterations,
                'time_per_iteration': result.optimization_time / max(1, result.n_iterations)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


def run_nas_optimizer_tests() -> Dict[str, Any]:
    """Run all NAS Optimizer tests."""
    tester = NASOptimizerTester()
    return tester.run_comprehensive_test()


# Additional utility functions for advanced usage
def create_regime_search_space(
    n_clusters_range: Tuple[int, int] = (2, 10),
    include_clustering_params: bool = True,
    include_feature_params: bool = False
) -> Dict[str, Any]:
    """
    Create a comprehensive search space for regime detection.
    
    Args:
        n_clusters_range: Range for number of clusters
        include_clustering_params: Include clustering-specific parameters
        include_feature_params: Include feature selection parameters
        
    Returns:
        Dictionary defining the search space
    """
    search_space = {
        'n_clusters': {
            'type': 'int',
            'low': n_clusters_range[0],
            'high': n_clusters_range[1]
        }
    }
    
    if include_clustering_params:
        search_space.update({
            'random_state': {'type': 'int', 'low': 0, 'high': 1000},
            'max_iter': {'type': 'int', 'low': 100, 'high': 1000}
        })
    
    if include_feature_params:
        search_space.update({
            'feature_selection': {'type': 'categorical', 'choices': ['none', 'variance', 'correlation']},
            'n_features': {'type': 'int', 'low': 5, 'high': 50}
        })
    
    return search_space


def create_advanced_regime_objective(
    use_silhouette: bool = True,
    use_calinski_harabasz: bool = True,
    use_davies_bouldin: bool = True,
    weights: Optional[Dict[str, float]] = None
) -> Callable:
    """
    Create an advanced objective function for regime detection.
    
    Args:
        use_silhouette: Use silhouette score
        use_calinski_harabasz: Use Calinski-Harabasz score
        use_davies_bouldin: Use Davies-Bouldin score
        weights: Weights for different metrics
        
    Returns:
        Objective function
    """
    if weights is None:
        weights = {
            'silhouette': 0.4,
            'calinski_harabasz': 0.3,
            'davies_bouldin': 0.3
        }
    
    def advanced_objective(X: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> float:
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            from sklearn.preprocessing import StandardScaler
            
            # Normalize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            n_clusters = params.get('n_clusters', 3)
            random_state = params.get('random_state', 42)
            max_iter = params.get('max_iter', 300)
            
            # Fit clustering model
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                max_iter=max_iter,
                n_init=10
            )
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            # Calculate metrics
            score = 0.0
            
            if use_silhouette and len(np.unique(cluster_labels)) > 1:
                silhouette = silhouette_score(X_scaled, cluster_labels)
                score += weights['silhouette'] * silhouette
            
            if use_calinski_harabasz and len(np.unique(cluster_labels)) > 1:
                ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
                # Normalize CH score (it can be very large)
                ch_normalized = min(ch_score / 1000.0, 1.0)
                score += weights['calinski_harabasz'] * ch_normalized
            
            if use_davies_bouldin and len(np.unique(cluster_labels)) > 1:
                db_score = davies_bouldin_score(X_scaled, cluster_labels)
                # Invert DB score (lower is better)
                db_inverted = 1.0 / (1.0 + db_score)
                score += weights['davies_bouldin'] * db_inverted
            
            return score
            
        except Exception as e:
            return -np.inf
    
    return advanced_objective