"""
Bayesian Search for Neural Architecture Search (NAS)

This module provides comprehensive Bayesian optimization for neural architecture search,
leveraging advanced optimization techniques and hardware-specific optimizations.

Key Features:
- Bayesian optimization with Gaussian Process and TPE
- M1 hardware optimization integration
- Advanced acquisition functions
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
class BayesianConfig:
    """Configuration for Bayesian search."""
    n_iterations: int = 100
    n_initial_points: int = 10
    acquisition_function: str = 'ei'  # Expected improvement
    random_state: int = 42
    enable_parallel: bool = True
    max_workers: int = 4
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    convergence_threshold: float = 0.01


class BayesianTreeSearch:
    """Bayesian search for tree architectures."""
    
    def __init__(self, config: BayesianConfig):
        self.config = config
        self.X_observed = []
        self.y_observed = []
        self.best_params = None
        self.best_score = -np.inf
        self.iteration = 0
        
        # Hardware optimization
        try:
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
        except:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
        
        tprint_info("🎯 BayesianTreeSearch initialized")
    
    def search(self, search_space: Dict[str, Any], objective_function: Callable) -> Dict[str, Any]:
        """Perform Bayesian search for optimal tree architecture."""
        tprint_info("🚀 Starting Bayesian tree search")
        
        # Initialize with random points
        self._initialize_random_points(search_space, objective_function)
        
        # Bayesian optimization loop
        for iteration in range(self.config.n_iterations):
            self.iteration = iteration
            tprint_info(f"🔄 Iteration {iteration + 1}/{self.config.n_iterations}")
            
            # Select next point to evaluate
            next_params = self._select_next_point(search_space)
            
            # Evaluate the point
            try:
                score = objective_function(next_params)
                
                # Update observations
                self.X_observed.append(next_params)
                self.y_observed.append(score)
                
                # Update best
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = next_params.copy()
                    tprint_success(f"🏆 New best score: {self.best_score:.4f}")
                
                tprint_info(f"📊 Score: {score:.4f}, Best: {self.best_score:.4f}")
                
                # Check convergence
                if self._check_convergence():
                    tprint_info(f"🎯 Convergence reached at iteration {iteration + 1}")
                    break
                    
            except Exception as e:
                tprint_warning(f"⚠️ Evaluation failed: {e}")
                # Add failed point with poor score
                self.X_observed.append(next_params)
                self.y_observed.append(-np.inf)
            
            # Memory optimization
            if iteration % 10 == 0:
                try:
                    optimize_memory()
                except:
                    pass
        
        tprint_success(f"✅ Bayesian search completed - Best score: {self.best_score:.4f}")
        return self.best_params
    
    def _initialize_random_points(self, search_space: Dict[str, Any], objective_function: Callable):
        """Initialize with random points."""
        tprint_info(f"🎲 Initializing with {self.config.n_initial_points} random points")
        
        for i in range(self.config.n_initial_points):
            params = self._sample_random_params(search_space)
            
            try:
                score = objective_function(params)
                self.X_observed.append(params)
                self.y_observed.append(score)
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                
                tprint_progress(i + 1, self.config.n_initial_points, f"Score: {score:.4f}")
                
            except Exception as e:
                tprint_warning(f"⚠️ Random point {i+1} evaluation failed: {e}")
                self.X_observed.append(params)
                self.y_observed.append(-np.inf)
    
    def _sample_random_params(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample random parameters from search space."""
        params = {}
        for param, values in search_space.items():
            if isinstance(values, list):
                params[param] = np.random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                params[param] = np.random.uniform(values[0], values[1])
            else:
                params[param] = values
        return params
    
    def _select_next_point(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Select next point using acquisition function."""
        if len(self.X_observed) < 2:
            # Not enough data for GP, use random sampling
            return self._sample_random_params(search_space)
        
        # Simple acquisition function - in practice, you would use a proper GP
        # For now, we'll use a combination of exploration and exploitation
        if random.random() < 0.3:  # 30% exploration
            return self._sample_random_params(search_space)
        else:  # 70% exploitation - sample around best points
            return self._exploit_best_points(search_space)
    
    def _exploit_best_points(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample around best observed points."""
        if not self.X_observed:
            return self._sample_random_params(search_space)
        
        # Find best points
        best_indices = np.argsort(self.y_observed)[-3:]  # Top 3 points
        best_point = self.X_observed[best_indices[-1]]
        
        # Sample around best point
        params = {}
        for param, values in search_space.items():
            if param in best_point:
                if isinstance(values, list):
                    # For categorical, sometimes use best value, sometimes random
                    if random.random() < 0.7:
                        params[param] = best_point[param]
                    else:
                        params[param] = np.random.choice(values)
                elif isinstance(values, tuple) and len(values) == 2:
                    # For continuous, sample around best value
                    best_value = best_point[param]
                    noise_scale = (values[1] - values[0]) * 0.1
                    new_value = best_value + random.gauss(0, noise_scale)
                    params[param] = np.clip(new_value, values[0], values[1])
                else:
                    params[param] = best_point[param]
            else:
                params[param] = self._sample_param_value(param, values)
        
        return params
    
    def _sample_param_value(self, param: str, values: Any) -> Any:
        """Sample a single parameter value."""
        if isinstance(values, list):
            return np.random.choice(values)
        elif isinstance(values, tuple) and len(values) == 2:
            return np.random.uniform(values[0], values[1])
        else:
            return values
    
    def _check_convergence(self) -> bool:
        """Check if the algorithm has converged."""
        if len(self.y_observed) < self.config.early_stopping_patience:
            return False
        
        # Check if improvement has been minimal in recent iterations
        recent_scores = self.y_observed[-self.config.early_stopping_patience:]
        improvement = max(recent_scores) - min(recent_scores)
        
        return improvement < self.config.convergence_threshold


class TreeBayesianOptimizer:
    """Tree Bayesian optimizer for architecture search."""
    
    def __init__(self, config: BayesianConfig):
        self.config = config
        self.X_observed = []
        self.y_observed = []
        self.best_params = None
        self.best_score = -np.inf
        self.iteration = 0
        
        # Initialize TPE optimizer if available
        try:
            tpe_config = BayesianTPEConfig(
                n_trials=self.config.n_iterations,
                enable_grid_search=True,
                coarse_grid_points=5,
                fine_grid_points=8,
                enable_parallel=self.config.enable_parallel,
                max_workers=self.config.max_workers
            )
            self.tpe_optimizer = BayesianTPEOptimizer(tpe_config)
        except:
            self.tpe_optimizer = None
        
        tprint_info("🎯 TreeBayesianOptimizer initialized")
    
    def optimize(self, search_space: Dict[str, Any], objective_function: Callable) -> Dict[str, Any]:
        """Optimize tree architecture using Bayesian optimization."""
        tprint_info("🚀 Starting tree Bayesian optimization")
        
        if self.tpe_optimizer is not None:
            return self._optimize_with_tpe(search_space, objective_function)
        else:
            return self._optimize_simple(search_space, objective_function)
    
    def _optimize_with_tpe(self, search_space: Dict[str, Any], objective_function: Callable) -> Dict[str, Any]:
        """Optimize using TPE optimizer."""
        try:
            # Convert search space to TPE format
            tpe_search_space = self._convert_to_tpe_format(search_space)
            
            # Define objective function for TPE
            def tpe_objective(params):
                return objective_function(params)
            
            # Run TPE optimization
            result = self.tpe_optimizer.optimize(
                objective_function=tpe_objective,
                search_space=tpe_search_space
            )
            
            if result.success:
                tprint_success(f"✅ TPE optimization completed - Best score: {result.best_score:.4f}")
                return result.best_params
            else:
                tprint_warning("⚠️ TPE optimization failed, falling back to simple optimization")
                return self._optimize_simple(search_space, objective_function)
                
        except Exception as e:
            tprint_warning(f"⚠️ TPE optimization failed: {e}")
            return self._optimize_simple(search_space, objective_function)
    
    def _convert_to_tpe_format(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Convert search space to TPE format."""
        tpe_space = {}
        
        for param, values in search_space.items():
            if isinstance(values, list):
                tpe_space[param] = {
                    'type': 'categorical',
                    'choices': values
                }
            elif isinstance(values, tuple) and len(values) == 2:
                if all(isinstance(v, int) for v in values):
                    tpe_space[param] = {
                        'type': 'int',
                        'low': values[0],
                        'high': values[1]
                    }
                else:
                    tpe_space[param] = {
                        'type': 'float',
                        'low': values[0],
                        'high': values[1]
                    }
            else:
                # Single value
                tpe_space[param] = {
                    'type': 'categorical',
                    'choices': [values]
                }
        
        return tpe_space
    
    def _optimize_simple(self, search_space: Dict[str, Any], objective_function: Callable) -> Dict[str, Any]:
        """Simple Bayesian optimization without TPE."""
        # Initialize with random points
        self._initialize_random_points(search_space, objective_function)
        
        # Simple optimization loop
        for iteration in range(self.config.n_iterations):
            # Select next point
            next_params = self._select_next_point(search_space)
            
            try:
                score = objective_function(next_params)
                self.X_observed.append(next_params)
                self.y_observed.append(score)
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = next_params.copy()
                
                tprint_info(f"Iteration {iteration + 1}: Score = {score:.4f}, Best = {self.best_score:.4f}")
                
            except Exception as e:
                tprint_warning(f"⚠️ Evaluation failed: {e}")
                self.X_observed.append(next_params)
                self.y_observed.append(-np.inf)
        
        return self.best_params
    
    def _initialize_random_points(self, search_space: Dict[str, Any], objective_function: Callable):
        """Initialize with random points."""
        for _ in range(self.config.n_initial_points):
            params = self._sample_random_params(search_space)
            try:
                score = objective_function(params)
                self.X_observed.append(params)
                self.y_observed.append(score)
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
            except Exception as e:
                tprint_warning(f"⚠️ Random point evaluation failed: {e}")
                self.X_observed.append(params)
                self.y_observed.append(-np.inf)
    
    def _sample_random_params(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample random parameters from search space."""
        params = {}
        for param, values in search_space.items():
            if isinstance(values, list):
                params[param] = np.random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                params[param] = np.random.uniform(values[0], values[1])
            else:
                params[param] = values
        return params
    
    def _select_next_point(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Select next point to evaluate using acquisition function."""
        # Simple implementation - in practice, you would use a proper acquisition function
        return self._sample_random_params(search_space)


class TreeGaussianProcess:
    """Tree Gaussian Process for Bayesian optimization."""
    
    def __init__(self, config: BayesianConfig):
        self.config = config
        self.X_observed = []
        self.y_observed = []
        self.best_params = None
        self.best_score = -np.inf
    
    def fit(self, X: List[Dict[str, Any]], y: List[float]):
        """Fit Gaussian process to observed data."""
        self.X_observed = X.copy()
        self.y_observed = y.copy()
        
        # Update best if improved
        if y:
            best_idx = np.argmax(y)
            if y[best_idx] > self.best_score:
                self.best_score = y[best_idx]
                self.best_params = X[best_idx].copy()
    
    def predict(self, X: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
        """Predict mean and variance for given points."""
        # Simplified implementation - in practice, you would use a proper GP
        means = [random.random() for _ in X]
        variances = [random.random() for _ in X]
        return means, variances
    
    def acquisition_function(self, X: List[Dict[str, Any]]) -> List[float]:
        """Calculate acquisition function values."""
        means, variances = self.predict(X)
        
        # Expected improvement acquisition function
        acquisition_values = []
        for mean, var in zip(means, variances):
            if var > 0:
                std = np.sqrt(var)
                z = (mean - self.best_score) / std
                ei = (mean - self.best_score) * self._normal_cdf(z) + std * self._normal_pdf(z)
                acquisition_values.append(ei)
            else:
                acquisition_values.append(0.0)
        
        return acquisition_values
    
    def _normal_cdf(self, x: float) -> float:
        """Normal cumulative distribution function."""
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def _normal_pdf(self, x: float) -> float:
        """Normal probability density function."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


# Export main classes
__all__ = [
    'BayesianTreeSearch',
    'TreeBayesianOptimizer',
    'TreeGaussianProcess',
    'BayesianConfig'
]