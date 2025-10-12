"""
Real Parameters Optimization Engine

This module provides comprehensive parameter optimization for trading strategies using
existing utilities from src/utils/ for ML optimization and hardware acceleration.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Import existing utilities
from src.utils.ml_common.optimization import HyperparameterOptimizer
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager

# VectorBT optimization imports
try:
    from src.utils.ml_common.unified_vectorization_manager import (
        get_unified_vectorization_manager, OperationType, OptimizationStrategy
    )
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    )
    VECTORBT_OPTIMIZATION_AVAILABLE = True
except ImportError:
    VECTORBT_OPTIMIZATION_AVAILABLE = False
    get_unified_vectorization_manager = None
    OperationType = None
    OptimizationStrategy = None
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None
# VectorBT optimization utilities
from src.utils.ml_common.unified_vectorization_manager import get_unified_vectorization_manager, VectorizationConfig
from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer

# Optional CVLSA support
try:
    from src.utils.ml_common.cvlsa import CVLSAValidator
except ImportError:
    CVLSAValidator = None
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from src.utils.matrix_operations.unified_operations import get_unified_matrix_operations
from src.utils.common_operations import safe_json_dump, safe_json_load, ensure_directory
from src.utils.math_validation import safe_divide, safe_log, safe_sqrt, validate_finite
from src.core.decorators import handles_errors, traced, log_execution_time

# Optimization imports
try:
    from scipy.optimize import minimize, differential_evolution, dual_annealing
    from scipy.optimize import OptimizeResult
    SCIPY_OPTIMIZE_AVAILABLE = True
except ImportError:
    SCIPY_OPTIMIZE_AVAILABLE = False
    minimize = None
    differential_evolution = None
    dual_annealing = None

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

logger = logging.getLogger(__name__)

class OptimizationMethod(Enum):
    """Optimization methods."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    SIMULATED_ANNEALING = "simulated_annealing"
    GRADIENT_DESCENT = "gradient_descent"

@dataclass
class ParameterSpace:
    """Parameter space definition."""
    name: str
    param_type: str  # "float", "int", "categorical", "boolean"
    bounds: Tuple[float, float] = None  # For float/int parameters
    choices: List[Any] = None  # For categorical parameters
    default: Any = None

@dataclass
class RealOptimizationConfig:
    """Configuration for real parameter optimization."""
    # Basic configuration
    optimization_method: OptimizationMethod = OptimizationMethod.BAYESIAN
    n_trials: int = 100
    n_jobs: int = -1  # -1 for all available cores
    
    # Hardware optimization
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    
    # Optimization parameters
    timeout_seconds: int = 3600  # 1 hour
    early_stopping_patience: int = 10
    convergence_threshold: float = 1e-6
    
    # ML validation
    enable_cv_validation: bool = True
    cv_folds: int = 5
    cv_method: str = "purged"  # "purged", "blocking", "standard"
    
    # Objective function
    objective_metric: str = "sharpe_ratio"  # "sharpe_ratio", "max_drawdown", "total_return", "profit_factor"
    minimize_objective: bool = False  # True for metrics like max_drawdown
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

class RealParametersOptimizer:
    """
    Real parameters optimization engine using existing utilities.
    
    This engine provides comprehensive parameter optimization with:
    - Multiple optimization methods (grid, random, Bayesian, genetic, etc.)
    - Hardware acceleration for M1/M2/M3 Macs
    - Cross-validation with lookahead bias protection
    - ML validation and hyperparameter optimization
    - Real-time performance monitoring
    """
    
    def __init__(self, config: RealOptimizationConfig):
        """Initialize the real parameters optimizer."""
        self.config = config
        self.logger = logger.getChild('RealParametersOptimizer')
        
        # Initialize hardware optimizers
        self.gpu_manager = get_m1_gpu_manager() if config.enable_gpu_acceleration else None
        self.memory_optimizer = get_m1_memory_optimizer() if config.enable_memory_optimization else None
        self.cpu_optimizer = get_m1_cpu_optimizer() if config.enable_parallel_processing else None
        
        # Initialize matrix operations
        self.matrix_ops = get_unified_matrix_operations()
        
        # Initialize ML utilities
        self.cv_validator = CVLSAValidator() if (CVLSAValidator and config.enable_cv_validation) else None
        self.hpo_optimizer = HyperparameterOptimizer()
        
        # Initialize VectorBT optimization components
        if VECTORBT_OPTIMIZATION_AVAILABLE:
            self.vectorization_manager = get_unified_vectorization_manager()
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=config.enable_gpu_acceleration,
                enable_parallel=config.enable_parallel_processing,
                memory_efficient=config.enable_memory_optimization
            )
            self.logger.info("✅ VectorBT optimization components initialized")
        else:
            self.vectorization_manager = None
            self.rolling_optimizer = None
            self.logger.warning("⚠️ VectorBT optimization not available, using standard methods")
        # Initialize VectorBT optimization utilities
        try:
            # Create VectorBT configuration
            vectorbt_config = VectorizationConfig(
                enable_vectorbt=True,
                enable_gpu=config.enable_gpu_acceleration,
                enable_parallel=config.enable_parallel_processing,
                memory_efficient=config.enable_memory_optimization,
                max_memory_gb=8.0,
                chunk_size=1000,
                enable_monitoring=True,
                enable_profiling=False,
                batch_size=10000,
                enable_batch_processing=True,
                rolling_optimization_threshold=1000,
                enable_rolling_optimization=True
            )
            
            self.vectorization_manager = get_unified_vectorization_manager(vectorbt_config)
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=config.enable_gpu_acceleration,
                enable_parallel=config.enable_parallel_processing,
                memory_efficient=config.enable_memory_optimization,
                chunk_size=1000,
                fast_fail=True,
                enable_logging=True
            )
            self.logger.info("✅ VectorBT optimization utilities initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ VectorBT optimization unavailable: {e}")
            self.vectorization_manager = None
            self.rolling_optimizer = None
        
        # Optimization state
        self.parameter_space = []
        self.optimization_history = []
        self.best_parameters = {}
        self.best_score = float('-inf') if not config.minimize_objective else float('inf')
        
        # Performance monitoring
        self.performance_stats = {
            'vectorbt_operations': 0,
            'matrix_operations': 0,
            'standard_operations': 0,
            'total_evaluations': 0,
            'total_time': 0.0,
            'memory_optimizations': 0,
            'gpu_operations': 0,
            'parallel_operations': 0,
            'errors': 0,
            'fallbacks': 0
        }
        
    def add_parameter(self, name: str, param_type: str, bounds: Tuple[float, float] = None, 
                     choices: List[Any] = None, default: Any = None):
        """Add a parameter to the optimization space."""
        try:
            param = ParameterSpace(
                name=name,
                param_type=param_type,
                bounds=bounds,
                choices=choices,
                default=default
            )
            self.parameter_space.append(param)
            self.logger.info(f"✅ Added parameter: {name} ({param_type})")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add parameter {name}: {e}")
            raise
    
    async def optimize_parameters(self, objective_function: Callable, 
                                initial_parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize parameters using the specified method."""
        self.logger.info(f"🔧 Starting parameter optimization using {self.config.optimization_method.value}")
        
        try:
            # Validate parameter space
            if not self.parameter_space:
                raise ValueError("No parameters defined for optimization")
            
            # Initialize optimization
            if self.config.optimization_method == OptimizationMethod.GRID_SEARCH:
                results = await self._grid_search_optimization(objective_function)
            elif self.config.optimization_method == OptimizationMethod.RANDOM_SEARCH:
                results = await self._random_search_optimization(objective_function)
            elif self.config.optimization_method == OptimizationMethod.BAYESIAN:
                results = await self._bayesian_optimization(objective_function)
            elif self.config.optimization_method == OptimizationMethod.GENETIC:
                results = await self._genetic_optimization(objective_function)
            elif self.config.optimization_method == OptimizationMethod.SIMULATED_ANNEALING:
                results = await self._simulated_annealing_optimization(objective_function)
            elif self.config.optimization_method == OptimizationMethod.GRADIENT_DESCENT:
                results = await self._gradient_descent_optimization(objective_function)
            else:
                raise ValueError(f"Unknown optimization method: {self.config.optimization_method}")
            
            # Store results
            self.best_parameters = results['best_parameters']
            self.best_score = results['best_score']
            
            self.logger.info(f"✅ Optimization completed: best score = {self.best_score:.6f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Parameter optimization failed: {e}")
            raise
    
    async def _grid_search_optimization(self, objective_function: Callable) -> Dict[str, Any]:
        """Grid search optimization."""
        self.logger.info("🔍 Running grid search optimization")
        
        try:
            # Generate parameter grid
            param_grid = self._generate_parameter_grid()
            
            best_score = float('-inf') if not self.config.minimize_objective else float('inf')
            best_parameters = {}
            optimization_history = []
            
            # Evaluate all combinations
            total_combinations = len(param_grid)
            self.logger.info(f"📊 Evaluating {total_combinations} parameter combinations")
            
            for i, params in enumerate(param_grid):
                try:
                    # Evaluate objective function
                    score = await self._evaluate_parameters(objective_function, params)
                    
                    # Update best if improved
                    if self._is_better_score(score, best_score):
                        best_score = score
                        best_parameters = params.copy()
                    
                    # Store history
                    optimization_history.append({
                        'iteration': i + 1,
                        'parameters': params.copy(),
                        'score': score,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Log progress
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"Progress: {i + 1}/{total_combinations} ({((i + 1)/total_combinations)*100:.1f}%)")
                
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to evaluate parameters {params}: {e}")
                    continue
            
            return {
                'method': 'grid_search',
                'best_parameters': best_parameters,
                'best_score': best_score,
                'optimization_history': optimization_history,
                'total_evaluations': len(optimization_history)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Grid search optimization failed: {e}")
            raise
    
    async def _random_search_optimization(self, objective_function: Callable) -> Dict[str, Any]:
        """Random search optimization."""
        self.logger.info("🎲 Running random search optimization")
        
        try:
            best_score = float('-inf') if not self.config.minimize_objective else float('inf')
            best_parameters = {}
            optimization_history = []
            
            for i in range(self.config.n_trials):
                try:
                    # Generate random parameters
                    params = self._generate_random_parameters()
                    
                    # Evaluate objective function
                    score = await self._evaluate_parameters(objective_function, params)
                    
                    # Update best if improved
                    if self._is_better_score(score, best_score):
                        best_score = score
                        best_parameters = params.copy()
                    
                    # Store history
                    optimization_history.append({
                        'iteration': i + 1,
                        'parameters': params.copy(),
                        'score': score,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Log progress
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"Progress: {i + 1}/{self.config.n_trials} ({((i + 1)/self.config.n_trials)*100:.1f}%)")
                
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to evaluate random parameters: {e}")
                    continue
            
            return {
                'method': 'random_search',
                'best_parameters': best_parameters,
                'best_score': best_score,
                'optimization_history': optimization_history,
                'total_evaluations': len(optimization_history)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Random search optimization failed: {e}")
            raise
    
    async def _bayesian_optimization(self, objective_function: Callable) -> Dict[str, Any]:
        """Bayesian optimization using Optuna."""
        self.logger.info("🧠 Running Bayesian optimization")
        
        try:
            if not OPTUNA_AVAILABLE:
                raise ImportError("Optuna not available for Bayesian optimization")
            
            # Create Optuna study
            direction = 'minimize' if self.config.minimize_objective else 'maximize'
            study = optuna.create_study(direction=direction)
            
            def objective(trial):
                # Generate parameters using Optuna
                params = {}
                for param in self.parameter_space:
                    if param.param_type == 'float':
                        params[param.name] = trial.suggest_float(
                            param.name, param.bounds[0], param.bounds[1]
                        )
                    elif param.param_type == 'int':
                        params[param.name] = trial.suggest_int(
                            param.name, int(param.bounds[0]), int(param.bounds[1])
                        )
                    elif param.param_type == 'categorical':
                        params[param.name] = trial.suggest_categorical(
                            param.name, param.choices
                        )
                    elif param.param_type == 'boolean':
                        params[param.name] = trial.suggest_categorical(
                            param.name, [True, False]
                        )
                
                # Evaluate objective function
                return asyncio.run(self._evaluate_parameters(objective_function, params))
            
            # Optimize
            study.optimize(objective, n_trials=self.config.n_trials)
            
            # Extract results
            best_params = study.best_params
            best_score = study.best_value
            
            # Convert optimization history
            optimization_history = []
            for trial in study.trials:
                optimization_history.append({
                    'iteration': trial.number + 1,
                    'parameters': trial.params,
                    'score': trial.value,
                    'timestamp': datetime.fromtimestamp(trial.datetime_start).isoformat()
                })
            
            return {
                'method': 'bayesian',
                'best_parameters': best_params,
                'best_score': best_score,
                'optimization_history': optimization_history,
                'total_evaluations': len(optimization_history)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Bayesian optimization failed: {e}")
            raise
    
    async def _genetic_optimization(self, objective_function: Callable) -> Dict[str, Any]:
        """Genetic algorithm optimization."""
        self.logger.info("🧬 Running genetic algorithm optimization")
        
        try:
            if not SCIPY_OPTIMIZE_AVAILABLE:
                raise ImportError("SciPy not available for genetic optimization")
            
            # Define bounds for continuous parameters
            bounds = []
            param_names = []
            for param in self.parameter_space:
                if param.param_type in ['float', 'int']:
                    bounds.append(param.bounds)
                    param_names.append(param.name)
            
            def objective_wrapper(x):
                # Convert array to parameter dictionary
                params = {}
                for i, name in enumerate(param_names):
                    param = next(p for p in self.parameter_space if p.name == name)
                    if param.param_type == 'int':
                        params[name] = int(x[i])
                    else:
                        params[name] = float(x[i])
                
                # Add categorical parameters with default values
                for param in self.parameter_space:
                    if param.param_type == 'categorical':
                        params[param.name] = param.default
                    elif param.param_type == 'boolean':
                        params[param.name] = param.default
                
                # Evaluate objective function
                return asyncio.run(self._evaluate_parameters(objective_function, params))
            
            # Run differential evolution
            result = differential_evolution(
                objective_wrapper,
                bounds,
                maxiter=self.config.n_trials // 10,  # Adjust for differential evolution
                popsize=15,
                seed=42
            )
            
            # Extract results
            best_params = {}
            for i, name in enumerate(param_names):
                param = next(p for p in self.parameter_space if p.name == name)
                if param.param_type == 'int':
                    best_params[name] = int(result.x[i])
                else:
                    best_params[name] = float(result.x[i])
            
            # Add default values for categorical parameters
            for param in self.parameter_space:
                if param.param_type in ['categorical', 'boolean']:
                    best_params[param.name] = param.default
            
            return {
                'method': 'genetic',
                'best_parameters': best_params,
                'best_score': result.fun,
                'optimization_history': [],  # Differential evolution doesn't provide history
                'total_evaluations': result.nfev
            }
            
        except Exception as e:
            self.logger.error(f"❌ Genetic optimization failed: {e}")
            raise
    
    async def _simulated_annealing_optimization(self, objective_function: Callable) -> Dict[str, Any]:
        """Simulated annealing optimization."""
        self.logger.info("🔥 Running simulated annealing optimization")
        
        try:
            if not SCIPY_OPTIMIZE_AVAILABLE:
                raise ImportError("SciPy not available for simulated annealing optimization")
            
            # Define bounds for continuous parameters
            bounds = []
            param_names = []
            for param in self.parameter_space:
                if param.param_type in ['float', 'int']:
                    bounds.append(param.bounds)
                    param_names.append(param.name)
            
            def objective_wrapper(x):
                # Convert array to parameter dictionary
                params = {}
                for i, name in enumerate(param_names):
                    param = next(p for p in self.parameter_space if p.name == name)
                    if param.param_type == 'int':
                        params[name] = int(x[i])
                    else:
                        params[name] = float(x[i])
                
                # Add categorical parameters with default values
                for param in self.parameter_space:
                    if param.param_type == 'categorical':
                        params[param.name] = param.default
                    elif param.param_type == 'boolean':
                        params[param.name] = param.default
                
                # Evaluate objective function
                return asyncio.run(self._evaluate_parameters(objective_function, params))
            
            # Run dual annealing
            result = dual_annealing(
                objective_wrapper,
                bounds,
                maxiter=self.config.n_trials,
                seed=42
            )
            
            # Extract results
            best_params = {}
            for i, name in enumerate(param_names):
                param = next(p for p in self.parameter_space if p.name == name)
                if param.param_type == 'int':
                    best_params[name] = int(result.x[i])
                else:
                    best_params[name] = float(result.x[i])
            
            # Add default values for categorical parameters
            for param in self.parameter_space:
                if param.param_type in ['categorical', 'boolean']:
                    best_params[param.name] = param.default
            
            return {
                'method': 'simulated_annealing',
                'best_parameters': best_params,
                'best_score': result.fun,
                'optimization_history': [],  # Dual annealing doesn't provide history
                'total_evaluations': result.nfev
            }
            
        except Exception as e:
            self.logger.error(f"❌ Simulated annealing optimization failed: {e}")
            raise
    
    async def _gradient_descent_optimization(self, objective_function: Callable) -> Dict[str, Any]:
        """Gradient descent optimization."""
        self.logger.info("📈 Running gradient descent optimization")
        
        try:
            if not SCIPY_OPTIMIZE_AVAILABLE:
                raise ImportError("SciPy not available for gradient descent optimization")
            
            # Define bounds for continuous parameters
            bounds = []
            param_names = []
            for param in self.parameter_space:
                if param.param_type in ['float', 'int']:
                    bounds.append(param.bounds)
                    param_names.append(param.name)
            
            def objective_wrapper(x):
                # Convert array to parameter dictionary
                params = {}
                for i, name in enumerate(param_names):
                    param = next(p for p in self.parameter_space if p.name == name)
                    if param.param_type == 'int':
                        params[name] = int(x[i])
                    else:
                        params[name] = float(x[i])
                
                # Add categorical parameters with default values
                for param in self.parameter_space:
                    if param.param_type == 'categorical':
                        params[param.name] = param.default
                    elif param.param_type == 'boolean':
                        params[param.name] = param.default
                
                # Evaluate objective function
                return asyncio.run(self._evaluate_parameters(objective_function, params))
            
            # Initial guess (middle of bounds)
            x0 = [(b[0] + b[1]) / 2 for b in bounds]
            
            # Run minimization
            result = minimize(
                objective_wrapper,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': self.config.n_trials}
            )
            
            # Extract results
            best_params = {}
            for i, name in enumerate(param_names):
                param = next(p for p in self.parameter_space if p.name == name)
                if param.param_type == 'int':
                    best_params[name] = int(result.x[i])
                else:
                    best_params[name] = float(result.x[i])
            
            # Add default values for categorical parameters
            for param in self.parameter_space:
                if param.param_type in ['categorical', 'boolean']:
                    best_params[param.name] = param.default
            
            return {
                'method': 'gradient_descent',
                'best_parameters': best_params,
                'best_score': result.fun,
                'optimization_history': [],  # L-BFGS-B doesn't provide history
                'total_evaluations': result.nfev
            }
            
        except Exception as e:
            self.logger.error(f"❌ Gradient descent optimization failed: {e}")
            raise
    
    def _generate_parameter_grid(self) -> List[Dict[str, Any]]:
        """Generate parameter grid for grid search."""
        try:
            # Create parameter ranges
            param_ranges = {}
            for param in self.parameter_space:
                if param.param_type == 'float':
                    # Create 5 points between bounds
                    param_ranges[param.name] = np.linspace(param.bounds[0], param.bounds[1], 5)
                elif param.param_type == 'int':
                    # Create integer range
                    param_ranges[param.name] = list(range(int(param.bounds[0]), int(param.bounds[1]) + 1, 2))
                elif param.param_type == 'categorical':
                    param_ranges[param.name] = param.choices
                elif param.param_type == 'boolean':
                    param_ranges[param.name] = [True, False]
            
            # Generate all combinations
            import itertools
            param_names = list(param_ranges.keys())
            param_values = list(param_ranges.values())
            
            combinations = list(itertools.product(*param_values))
            param_grid = []
            
            for combo in combinations:
                params = dict(zip(param_names, combo))
                param_grid.append(params)
            
            return param_grid
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate parameter grid: {e}")
            raise
    
    def _generate_random_parameters(self) -> Dict[str, Any]:
        """Generate random parameters within bounds."""
        try:
            params = {}
            
            for param in self.parameter_space:
                if param.param_type == 'float':
                    params[param.name] = np.random.uniform(param.bounds[0], param.bounds[1])
                elif param.param_type == 'int':
                    params[param.name] = np.random.randint(int(param.bounds[0]), int(param.bounds[1]) + 1)
                elif param.param_type == 'categorical':
                    params[param.name] = np.random.choice(param.choices)
                elif param.param_type == 'boolean':
                    params[param.name] = np.random.choice([True, False])
            
            return params
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate random parameters: {e}")
            raise
    
    async def _evaluate_parameters(self, objective_function: Callable, parameters: Dict[str, Any]) -> float:
        """Evaluate objective function with given parameters using VectorBT optimization."""
        try:
            # Use VectorBT optimization if available
            if self.vectorization_manager and VECTORBT_OPTIMIZATION_AVAILABLE:
                # Use unified vectorization manager for optimized evaluation
                data = {'parameters': parameters, 'objective_function': objective_function}
                config = self.vectorization_manager._create_default_config(
                    OperationType.MODEL_TRAINING, data
                )
                
                result = self.vectorization_manager.optimize_operation(
                    OperationType.MODEL_TRAINING, data, config
                )
                score = result.result
                self.logger.debug(f"VectorBT optimized evaluation: {result.performance_gain:.2f}x speedup")
            else:
                # Use hardware optimization if available
        start_time = time.time()
        self.performance_stats['total_evaluations'] += 1
        
        try:
            # Use VectorBT optimization if available
            if self.vectorization_manager and self.rolling_optimizer:
                self.logger.debug("🎯 Using VectorBT-optimized parameter evaluation")
                
                # Use VectorBT for enhanced parameter evaluation
                with self.vectorization_manager.performance_monitoring("parameter_evaluation"):
                    if self.memory_optimizer:
                        with self.memory_optimizer.optimize_for_workload("parameter_evaluation"):
                            score = await objective_function(parameters)
                    else:
                        score = await objective_function(parameters)
                
                self.performance_stats['vectorbt_operations'] += 1
                
            # Use matrix operations if available
            elif self.matrix_ops:
                self.logger.debug("🎯 Using matrix operations for parameter evaluation")
                
                if self.memory_optimizer:
                    with self.memory_optimizer.optimize_for_workload("parameter_evaluation"):
                        score = await objective_function(parameters)
                else:
                    score = await objective_function(parameters)
                
                self.performance_stats['matrix_operations'] += 1
                
            # Standard evaluation
            else:
                self.logger.debug("🎯 Using standard parameter evaluation")
                
                if self.memory_optimizer:
                    with self.memory_optimizer.optimize_for_workload("parameter_evaluation"):
                        score = await objective_function(parameters)
                else:
                    score = await objective_function(parameters)
                
                self.performance_stats['standard_operations'] += 1
            
            # Update performance stats
            execution_time = time.time() - start_time
            self.performance_stats['total_time'] += execution_time
            
            return score
            
        except Exception as e:
            self.logger.error(f"❌ Parameter evaluation failed: {e}")
            self.performance_stats['errors'] += 1
            raise
    
    async def _evaluate_parameters_with_rolling_optimization(self, objective_function: Callable, 
                                                           parameters: Dict[str, Any],
                                                           time_series_data: pd.DataFrame = None) -> float:
        """Evaluate parameters with VectorBT rolling optimization for time-series data."""
        try:
            if not self.rolling_optimizer or time_series_data is None:
                return await self._evaluate_parameters(objective_function, parameters)
            
            # Use VectorBT rolling optimizer for time-series parameter evaluation
            self.logger.info("🔄 Using VectorBT rolling optimization for parameter evaluation")
            
            # Optimize rolling calculations in the objective function
            optimized_data = self._optimize_time_series_data(time_series_data)
            
            # Evaluate with optimized data
            score = await objective_function(parameters, optimized_data)
            
            return score
            
        except Exception as e:
            self.logger.error(f"❌ Rolling optimization evaluation failed: {e}")
            # Fallback to standard evaluation
            return await self._evaluate_parameters(objective_function, parameters)
    
    def _optimize_time_series_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize time-series data using VectorBT rolling operations."""
        try:
            if not self.rolling_optimizer:
                return data
            
            optimized_data = data.copy()
            
            # Optimize rolling calculations for common technical indicators
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                # Optimize rolling mean (SMA)
                if f'{column}_sma_20' in data.columns:
                    optimized_data[f'{column}_sma_20'] = self.rolling_optimizer.rolling_mean(
                        data[column], window=20
                    )
                
                # Optimize rolling standard deviation
                if f'{column}_std_20' in data.columns:
                    optimized_data[f'{column}_std_20'] = self.rolling_optimizer.rolling_std(
                        data[column], window=20
                    )
                
                # Optimize rolling min/max
                if f'{column}_min_20' in data.columns:
                    optimized_data[f'{column}_min_20'] = self.rolling_optimizer.rolling_min(
                        data[column], window=20
                    )
                
                if f'{column}_max_20' in data.columns:
                    optimized_data[f'{column}_max_20'] = self.rolling_optimizer.rolling_max(
                        data[column], window=20
                    )
            
            self.logger.debug("✅ Time-series data optimized with VectorBT rolling operations")
            return optimized_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Time-series optimization failed: {e}, using original data")
            return data
    
    def _is_better_score(self, score: float, best_score: float) -> bool:
        """Check if score is better than current best."""
        if self.config.minimize_objective:
            return score < best_score
        else:
            return score > best_score
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization report."""
        try:
            report = {
                'optimization_config': {
                    'method': self.config.optimization_method.value,
                    'n_trials': self.config.n_trials,
                    'objective_metric': self.config.objective_metric,
                    'minimize_objective': self.config.minimize_objective
                },
                'parameter_space': [
                    {
                        'name': param.name,
                        'type': param.param_type,
                        'bounds': param.bounds,
                        'choices': param.choices,
                        'default': param.default
                    }
                    for param in self.parameter_space
                ],
                'best_parameters': self.best_parameters,
                'best_score': self.best_score,
                'optimization_history': self.optimization_history,
                'timestamp': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate optimization report: {e}")
            return {'error': str(e)}

# Convenience functions
async def optimize_parameters(
    objective_function: Callable,
    parameter_space: List[ParameterSpace],
    method: OptimizationMethod = OptimizationMethod.BAYESIAN,
    n_trials: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """Optimize parameters using the specified method."""
    config = RealOptimizationConfig(
        optimization_method=method,
        n_trials=n_trials,
        **kwargs
    )
    
    optimizer = RealParametersOptimizer(config)
    
    # Add parameters
    for param in parameter_space:
        optimizer.add_parameter(
            param.name, param.param_type, param.bounds, param.choices, param.default
        )
    
    # Run optimization
    results = await optimizer.optimize_parameters(objective_function)
    
    return results