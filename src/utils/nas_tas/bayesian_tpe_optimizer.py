"""
Bayesian TPE Optimizer with Automatic Grid Search Integration

This module provides a comprehensive Bayesian TPE optimization system that automatically
calls the dedicated grid utils (coarse then fine grid) as a first step, followed by
Bayesian TPE optimization for final refinement.

Key Features:
- Automatic grid search integration (coarse → fine → TPE)
- Comprehensive logging and error handling
- Configurable optimization strategies
- Support for multiple optimization backends (Optuna, scikit-optimize)
- Memory-efficient optimization tracking
- Parallel processing support
- Early stopping and convergence detection

Usage:
    from src.utils.nas_tas.bayesian_tpe_optimizer import BayesianTPEOptimizer
    
    optimizer = BayesianTPEOptimizer(config)
    results = optimizer.optimize(objective_function, search_space)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os

# Import existing grid utilities
from src.utils.ml_common.optimization.grid_utils import build_coarse_grid_from_search_space, build_fine_grid_around_best
from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization

# Optional dependencies with graceful fallback
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

try:
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class BayesianTPEConfig:
    """Configuration for Bayesian TPE optimization."""
    
    # Grid search configuration
    enable_grid_search: bool = True
    coarse_grid_points: int = 5
    fine_grid_points: int = 8
    
    # TPE configuration
    n_trials: int = 50
    timeout_seconds: Optional[int] = None
    random_state: int = 42
    
    # Optimization backend
    backend: str = 'optuna'  # 'optuna', 'skopt', 'skopt_forest'
    
    # Parallel processing
    enable_parallel: bool = True
    max_workers: int = 4
    
    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # Convergence detection
    enable_convergence_detection: bool = True
    convergence_threshold: float = 0.01
    convergence_patience: int = 15
    
    # Memory management
    max_history_size: int = 1000
    enable_memory_cleanup: bool = True
    
    # Logging
    log_level: str = 'INFO'
    log_file: Optional[str] = None
    enable_progress_logging: bool = True
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    monitor_memory: bool = True
    monitor_time: bool = True


@dataclass
class OptimizationResult:
    """Result of Bayesian TPE optimization."""
    
    best_params: Dict[str, Any]
    best_score: float
    optimization_time: float
    n_trials: int
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    grid_search_results: Optional[Dict[str, Any]] = None
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


class BayesianTPEOptimizer:
    """
    Bayesian TPE Optimizer with automatic grid search integration.
    
    This optimizer automatically performs:
    1. Coarse grid search to identify promising regions
    2. Fine grid search around best coarse results
    3. Bayesian TPE optimization for final refinement
    """
    
    def __init__(self, config: Optional[BayesianTPEConfig] = None):
        """Initialize Bayesian TPE optimizer."""
        self.config = config or BayesianTPEConfig()
        self.logger = self._setup_logging()
        
        # Validate configuration
        self._validate_config()
        
        # Initialize optimization state
        self.optimization_history = []
        self.best_score = -np.inf
        self.best_params = None
        self.start_time = None
        
        # Initialize performance monitoring
        if self.config.enable_performance_monitoring:
            self.performance_metrics = {
                'memory_usage': [],
                'execution_times': [],
                'convergence_rates': []
            }
        
        self.logger.info(f"🚀 Bayesian TPE Optimizer initialized")
        self.logger.info(f"   → Backend: {self.config.backend}")
        self.logger.info(f"   → Grid search: {'enabled' if self.config.enable_grid_search else 'disabled'}")
        self.logger.info(f"   → Parallel processing: {'enabled' if self.config.enable_parallel else 'disabled'}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(f"{__name__}.BayesianTPEOptimizer")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _validate_config(self):
        """Validate configuration parameters."""
        valid_backends = ['optuna', 'skopt', 'skopt_forest']
        if self.config.backend not in valid_backends:
            raise ValueError(
                f"Invalid backend: {self.config.backend}. Must be one of {valid_backends}"
            )

        if self.config.backend == 'optuna' and not OPTUNA_AVAILABLE:
            raise ImportError("Optuna backend requested but not available. Install optuna or use 'skopt' backend.")

        if self.config.backend in {'skopt', 'skopt_forest'} and not SKOPT_AVAILABLE:
            raise ImportError(
                "scikit-optimize backend requested but not available. Install scikit-optimize or use 'optuna' backend."
            )
        
        if self.config.n_trials <= 0:
            raise ValueError("n_trials must be positive")
        
        if self.config.coarse_grid_points <= 0:
            raise ValueError("coarse_grid_points must be positive")
        
        if self.config.fine_grid_points <= 0:
            raise ValueError("fine_grid_points must be positive")
    
    def optimize(self, 
                 objective_function: Callable,
                 search_space: Dict[str, Any],
                 X: Optional[np.ndarray] = None,
                 y: Optional[np.ndarray] = None,
                 **kwargs) -> OptimizationResult:
        """
        Perform Bayesian TPE optimization with automatic grid search.
        
        Args:
            objective_function: Function to optimize (should return a score to maximize)
            search_space: Dictionary defining the search space
            X: Optional feature matrix (for data-driven optimizations)
            y: Optional target values (for supervised optimizations)
            **kwargs: Additional arguments passed to objective function
            
        Returns:
            OptimizationResult with best parameters and optimization details
        """
        self.start_time = time.time()
        self.logger.info("🎯 Starting Bayesian TPE optimization")
        
        try:
            # Step 1: Grid search (if enabled)
            grid_results = None
            if self.config.enable_grid_search:
                self.logger.info("🔍 Step 1: Performing grid search")
                grid_results = self._perform_grid_search(objective_function, search_space, X, y, **kwargs)
                
                if grid_results and grid_results.get('success', False):
                    self.logger.info(f"✅ Grid search completed - Best score: {grid_results['best_score']:.4f}")
                    # Update search space around best grid result
                    search_space = self._refine_search_space(search_space, grid_results['best_params'])
                else:
                    self.logger.warning("⚠️ Grid search failed, proceeding with original search space")
            
            # Step 2: Bayesian TPE optimization
            self.logger.info("🎲 Step 2: Performing Bayesian TPE optimization")
            tpe_results = self._perform_tpe_optimization(objective_function, search_space, X, y, **kwargs)
            
            # Combine results
            final_result = self._combine_results(grid_results, tpe_results)
            
            # Log final results
            self.logger.info(f"✅ Optimization completed in {final_result.optimization_time:.2f}s")
            self.logger.info(f"📊 Best score: {final_result.best_score:.4f}")
            self.logger.info(f"🔧 Best parameters: {final_result.best_params}")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"❌ Optimization failed: {e}")
            return OptimizationResult(
                best_params={},
                best_score=-np.inf,
                optimization_time=time.time() - self.start_time,
                n_trials=0,
                success=False,
                error_message=str(e)
            )
    
    def _perform_grid_search(self, 
                           objective_function: Callable,
                           search_space: Dict[str, Any],
                           X: Optional[np.ndarray],
                           y: Optional[np.ndarray],
                           **kwargs) -> Dict[str, Any]:
        """Perform coarse and fine grid search."""
        try:
            # Coarse grid search
            self.logger.info(f"   → Coarse grid search with {self.config.coarse_grid_points} points")
            coarse_grid = build_coarse_grid_from_search_space(
                search_space, 
                self.config.coarse_grid_points
            )
            
            if not coarse_grid:
                self.logger.warning("⚠️ No coarse grid points generated")
                return {'success': False, 'error': 'No coarse grid points'}
            
            # Evaluate coarse grid
            coarse_results = self._evaluate_grid_points(
                objective_function, coarse_grid, X, y, **kwargs
            )
            
            if not coarse_results['success']:
                return coarse_results
            
            best_coarse_params = coarse_results['best_params']
            best_coarse_score = coarse_results['best_score']
            
            self.logger.info(f"   → Coarse grid best score: {best_coarse_score:.4f}")
            
            # Fine grid search around best coarse result
            self.logger.info(f"   → Fine grid search with {self.config.fine_grid_points} points")
            fine_grid = build_fine_grid_around_best(
                search_space,
                best_coarse_params,
                self.config.fine_grid_points
            )
            
            if not fine_grid:
                self.logger.warning("⚠️ No fine grid points generated, using coarse results")
                return coarse_results
            
            # Evaluate fine grid
            fine_results = self._evaluate_grid_points(
                objective_function, fine_grid, X, y, **kwargs
            )
            
            if not fine_results['success']:
                self.logger.warning("⚠️ Fine grid search failed, using coarse results")
                return coarse_results
            
            # Return best result
            if fine_results['best_score'] > coarse_results['best_score']:
                self.logger.info(f"   → Fine grid improved score: {fine_results['best_score']:.4f}")
                return fine_results
            else:
                self.logger.info("   → Fine grid did not improve, using coarse results")
                return coarse_results
                
        except Exception as e:
            self.logger.error(f"❌ Grid search failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _evaluate_grid_points(self, 
                             objective_function: Callable,
                             grid_points: List[Dict[str, Any]],
                             X: Optional[np.ndarray],
                             y: Optional[np.ndarray],
                             **kwargs) -> Dict[str, Any]:
        """Evaluate grid points with parallel processing support."""
        try:
            best_score = -np.inf
            best_params = None
            successful_evaluations = 0
            failed_evaluations = 0
            
            if self.config.enable_parallel and len(grid_points) > 1:
                # Parallel evaluation
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    future_to_params = {
                        executor.submit(self._evaluate_single_point, objective_function, params, X, y, **kwargs): params
                        for params in grid_points
                    }
                    
                    for future in as_completed(future_to_params):
                        params = future_to_params[future]
                        try:
                            score = future.result()
                            if score is not None and not np.isnan(score):
                                successful_evaluations += 1
                                if score > best_score:
                                    best_score = score
                                    best_params = params.copy()
                            else:
                                failed_evaluations += 1
                        except Exception as e:
                            self.logger.warning(f"⚠️ Evaluation failed for params {params}: {e}")
                            failed_evaluations += 1
            else:
                # Sequential evaluation
                for i, params in enumerate(grid_points):
                    try:
                        score = self._evaluate_single_point(objective_function, params, X, y, **kwargs)
                        if score is not None and not np.isnan(score):
                            successful_evaluations += 1
                            if score > best_score:
                                best_score = score
                                best_params = params.copy()
                        else:
                            failed_evaluations += 1
                        
                        if self.config.enable_progress_logging and (i + 1) % 10 == 0:
                            self.logger.info(f"   → Evaluated {i + 1}/{len(grid_points)} points")
                            
                    except Exception as e:
                        self.logger.warning(f"⚠️ Evaluation failed for params {params}: {e}")
                        failed_evaluations += 1
            
            if successful_evaluations == 0:
                return {'success': False, 'error': 'All evaluations failed'}
            
            self.logger.info(f"   → Grid evaluation: {successful_evaluations} successful, {failed_evaluations} failed")
            
            return {
                'success': True,
                'best_score': best_score,
                'best_params': best_params,
                'successful_evaluations': successful_evaluations,
                'failed_evaluations': failed_evaluations
            }
            
        except Exception as e:
            self.logger.error(f"❌ Grid evaluation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _evaluate_single_point(self, 
                              objective_function: Callable,
                              params: Dict[str, Any],
                              X: Optional[np.ndarray],
                              y: Optional[np.ndarray],
                              **kwargs) -> Optional[float]:
        """Evaluate a single parameter combination."""
        try:
            # Prepare arguments for objective function
            objective_args = {'params': params}
            if X is not None:
                objective_args['X'] = X
            if y is not None:
                objective_args['y'] = y
            objective_args.update(kwargs)
            
            # Call objective function
            score = objective_function(**objective_args)
            
            # Validate score
            if not isinstance(score, (int, float, np.number)):
                self.logger.warning(f"⚠️ Invalid score type: {type(score)}")
                return None
            
            if np.isnan(score) or np.isinf(score):
                self.logger.warning(f"⚠️ Invalid score value: {score}")
                return None
            
            return float(score)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Single point evaluation failed: {e}")
            return None
    
    def _perform_tpe_optimization(self,
                                 objective_function: Callable,
                                 search_space: Dict[str, Any],
                                 X: Optional[np.ndarray],
                                 y: Optional[np.ndarray],
                                 **kwargs) -> Dict[str, Any]:
        """Perform Bayesian TPE optimization."""
        try:
            if self.config.backend == 'optuna':
                return self._optimize_with_optuna(objective_function, search_space, X, y, **kwargs)
            elif self.config.backend in {'skopt', 'skopt_forest'}:
                use_forest = self.config.backend == 'skopt_forest'
                return self._optimize_with_skopt(
                    objective_function,
                    search_space,
                    X,
                    y,
                    use_forest=use_forest,
                    **kwargs
                )
            else:
                raise ValueError(f"Unknown backend: {self.config.backend}")
                
        except Exception as e:
            self.logger.error(f"❌ TPE optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _optimize_with_optuna(self, 
                             objective_function: Callable,
                             search_space: Dict[str, Any],
                             X: Optional[np.ndarray],
                             y: Optional[np.ndarray],
                             **kwargs) -> Dict[str, Any]:
        """Optimize using Optuna TPE sampler."""
        try:
            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.config.random_state),
                pruner=MedianPruner() if self.config.enable_early_stopping else None
            )
            
            def objective(trial):
                # Sample parameters from search space
                params = {}
                for param_name, param_config in search_space.items():
                    if param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name, 
                            param_config['low'], 
                            param_config['high']
                        )
                    elif param_config['type'] == 'float':
                        if param_config.get('log', False):
                            params[param_name] = trial.suggest_float(
                                param_name,
                                param_config['low'],
                                param_config['high'],
                                log=True
                            )
                        else:
                            params[param_name] = trial.suggest_float(
                                param_name,
                                param_config['low'],
                                param_config['high']
                            )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )
                
                # Evaluate objective
                score = self._evaluate_single_point(objective_function, params, X, y, **kwargs)
                
                if score is None:
                    raise optuna.TrialPruned()
                
                return score
            
            # Optimize
            study.optimize(
                objective, 
                n_trials=self.config.n_trials,
                timeout=self.config.timeout_seconds
            )
            
            return {
                'success': True,
                'best_score': study.best_value,
                'best_params': study.best_params,
                'study': study,
                'n_trials': len(study.trials)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Optuna optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _optimize_with_skopt(self,
                           objective_function: Callable,
                           search_space: Dict[str, Any],
                           X: Optional[np.ndarray],
                           y: Optional[np.ndarray],
                           use_forest: bool = False,
                           **kwargs) -> Dict[str, Any]:
        """Optimize using scikit-optimize."""
        try:
            # Convert search space to scikit-optimize dimensions
            dimensions = []
            for param_name, param_config in search_space.items():
                if param_config['type'] == 'int':
                    dimensions.append(Integer(
                        low=param_config['low'],
                        high=param_config['high'],
                        name=param_name
                    ))
                elif param_config['type'] == 'float':
                    dimensions.append(Real(
                        low=param_config['low'],
                        high=param_config['high'],
                        name=param_name
                    ))
                elif param_config['type'] == 'categorical':
                    dimensions.append(Categorical(
                        categories=param_config['choices'],
                        name=param_name
                    ))
            
            @use_named_args(dimensions)
            def objective_wrapper(**params):
                score = self._evaluate_single_point(objective_function, params, X, y, **kwargs)
                return -score if score is not None else 1e6  # Minimize negative score
            
            optimizer_func = forest_minimize if use_forest else gp_minimize

            optimizer_kwargs = {
                'func': objective_wrapper,
                'dimensions': dimensions,
                'n_calls': self.config.n_trials,
                'random_state': self.config.random_state
            }

            if self.config.enable_parallel:
                optimizer_kwargs['n_jobs'] = self.config.max_workers

            if use_forest:
                optimizer_kwargs['base_estimator'] = 'rf'

            result = optimizer_func(**optimizer_kwargs)
            
            # Extract best parameters
            best_params = dict(zip([dim.name for dim in dimensions], result.x))
            best_score = -result.fun  # Convert back to maximization
            
            return {
                'success': True,
                'best_score': best_score,
                'best_params': best_params,
                'result': result,
                'n_trials': len(result.func_vals)
            }
            
        except Exception as e:
            self.logger.error(f"❌ scikit-optimize optimization failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _refine_search_space(self, 
                           original_search_space: Dict[str, Any],
                           best_params: Dict[str, Any]) -> Dict[str, Any]:
        """Refine search space around best parameters."""
        try:
            refined_space = {}
            
            for param_name, param_config in original_search_space.items():
                if param_name not in best_params:
                    refined_space[param_name] = param_config
                    continue
                
                best_value = best_params[param_name]
                
                if param_config['type'] == 'int':
                    # Narrow range around best value
                    range_size = param_config['high'] - param_config['low']
                    narrow_range = max(1, int(range_size * 0.2))
                    
                    refined_space[param_name] = {
                        'type': 'int',
                        'low': max(param_config['low'], best_value - narrow_range),
                        'high': min(param_config['high'], best_value + narrow_range)
                    }
                    
                elif param_config['type'] == 'float':
                    # Narrow range around best value
                    range_size = param_config['high'] - param_config['low']
                    narrow_range = range_size * 0.2
                    
                    refined_space[param_name] = {
                        'type': 'float',
                        'low': max(param_config['low'], best_value - narrow_range),
                        'high': min(param_config['high'], best_value + narrow_range),
                        'log': param_config.get('log', False)
                    }
                    
                elif param_config['type'] == 'categorical':
                    # Keep original choices
                    refined_space[param_name] = param_config
            
            self.logger.info("🔧 Search space refined around best grid parameters")
            return refined_space
            
        except Exception as e:
            self.logger.warning(f"⚠️ Search space refinement failed: {e}")
            return original_search_space
    
    def _combine_results(self, 
                        grid_results: Optional[Dict[str, Any]],
                        tpe_results: Dict[str, Any]) -> OptimizationResult:
        """Combine grid search and TPE results."""
        try:
            # Determine best result
            if grid_results and grid_results.get('success', False) and tpe_results.get('success', False):
                if tpe_results['best_score'] > grid_results['best_score']:
                    best_score = tpe_results['best_score']
                    best_params = tpe_results['best_params']
                    best_method = 'tpe'
                else:
                    best_score = grid_results['best_score']
                    best_params = grid_results['best_params']
                    best_method = 'grid'
            elif tpe_results.get('success', False):
                best_score = tpe_results['best_score']
                best_params = tpe_results['best_params']
                best_method = 'tpe'
            elif grid_results and grid_results.get('success', False):
                best_score = grid_results['best_score']
                best_params = grid_results['best_params']
                best_method = 'grid'
            else:
                best_score = -np.inf
                best_params = {}
                best_method = 'none'
            
            # Create optimization history
            optimization_history = []
            if grid_results and grid_results.get('success', False):
                optimization_history.append({
                    'stage': 'grid_search',
                    'best_score': grid_results['best_score'],
                    'best_params': grid_results['best_params']
                })
            
            if tpe_results.get('success', False):
                optimization_history.append({
                    'stage': 'tpe_optimization',
                    'best_score': tpe_results['best_score'],
                    'best_params': tpe_results['best_params']
                })
            
            # Create convergence info
            convergence_info = {
                'best_method': best_method,
                'grid_search_used': grid_results is not None and grid_results.get('success', False),
                'tpe_optimization_used': tpe_results.get('success', False),
                'total_trials': tpe_results.get('n_trials', 0)
            }
            
            return OptimizationResult(
                best_params=best_params,
                best_score=best_score,
                optimization_time=time.time() - self.start_time,
                n_trials=tpe_results.get('n_trials', 0),
                convergence_info=convergence_info,
                grid_search_results=grid_results,
                optimization_history=optimization_history,
                success=best_method != 'none'
            )
            
        except Exception as e:
            self.logger.error(f"❌ Result combination failed: {e}")
            return OptimizationResult(
                best_params={},
                best_score=-np.inf,
                optimization_time=time.time() - self.start_time,
                n_trials=0,
                success=False,
                error_message=str(e)
            )


# Convenience functions
def optimize_with_bayesian_tpe(objective_function: Callable,
                              search_space: Dict[str, Any],
                              config: Optional[BayesianTPEConfig] = None,
                              **kwargs) -> OptimizationResult:
    """
    Convenience function for Bayesian TPE optimization.
    
    Args:
        objective_function: Function to optimize
        search_space: Search space definition
        config: Optimization configuration
        **kwargs: Additional arguments for objective function
        
    Returns:
        OptimizationResult
    """
    optimizer = BayesianTPEOptimizer(config)
    return optimizer.optimize(objective_function, search_space, **kwargs)


def create_search_space_from_bounds(bounds: Dict[str, Tuple[float, float]],
                                  param_types: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Create search space from parameter bounds.
    
    Args:
        bounds: Dictionary mapping parameter names to (low, high) bounds
        param_types: Optional dictionary mapping parameter names to types ('int', 'float', 'categorical')
        
    Returns:
        Search space dictionary
    """
    search_space = {}
    param_types = param_types or {}
    
    for param_name, (low, high) in bounds.items():
        param_type = param_types.get(param_name, 'float')
        
        if param_type == 'int':
            search_space[param_name] = {
                'type': 'int',
                'low': int(low),
                'high': int(high)
            }
        elif param_type == 'categorical':
            search_space[param_name] = {
                'type': 'categorical',
                'choices': list(range(int(low), int(high) + 1))
            }
        else:  # float
            search_space[param_name] = {
                'type': 'float',
                'low': float(low),
                'high': float(high)
            }
    
    return search_space


# Export main classes and functions
__all__ = [
    'BayesianTPEOptimizer',
    'BayesianTPEConfig', 
    'OptimizationResult',
    'optimize_with_bayesian_tpe',
    'create_search_space_from_bounds'
]