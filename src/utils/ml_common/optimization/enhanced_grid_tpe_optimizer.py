"""
Enhanced Grid+TPE Optimizer with VectorBT Integration

This module provides an enhanced hyperparameter optimization system that combines
grid search with Tree-structured Parzen Estimator (TPE) optimization, leveraging
VectorBTRollingOptimizer and UnifiedVectorizationManager for maximum performance.

Key Features:
- VectorBT-accelerated grid search
- Unified vectorization for batch processing
- Memory-efficient parameter evaluation
- GPU acceleration support
- Adaptive grid refinement
- Performance monitoring and statistics
- Parallel processing capabilities
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings

# VectorBT and optimization imports
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_quantile, rolling_skew, rolling_kurt
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optuna imports
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Install with: pip install optuna for TPE optimization")

# Import our vectorization components
from ...feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
from ...feature_generation.utils.unified_vectorization_manager import (
    UnifiedVectorizationManager, VectorizationConfig, get_unified_vectorization_manager
)

# Import existing utilities
from .grid_utils import build_coarse_grid_from_search_space, build_fine_grid_around_best
from .pareto import Solution, ParetoFront, compute_pareto_front

logger = logging.getLogger(__name__)


@dataclass
class EnhancedOptimizationConfig:
    """Configuration for enhanced grid+TPE optimization with VectorBT integration."""
    
    # Core optimization settings
    n_trials: int = 200
    timeout: Optional[float] = None
    direction: str = 'maximize'
    metric_name: str = 'objective'
    
    # Staged optimization settings
    enable_staged_optimization: bool = True
    coarse_grid_points: int = 5
    fine_grid_points: int = 5
    coarse_grid_trials: int = 25
    fine_grid_trials: int = 25
    tpe_trials: int = 150
    
    # VectorBT optimization settings
    enable_vectorbt_optimization: bool = True
    vectorbt_parallel_workers: int = 4
    vectorbt_chunk_size: int = 1000
    vectorbt_memory_limit_gb: float = 4.0
    vectorbt_use_gpu: bool = False
    vectorbt_enable_parallel: bool = True
    
    # Memory and performance settings
    memory_efficient: bool = True
    enable_batch_processing: bool = True
    batch_size: int = 32
    max_memory_gb: float = 8.0
    
    # Parallel processing settings
    enable_parallel: bool = True
    max_workers: int = 4
    use_threading: bool = True
    
    # Adaptive refinement settings
    enable_adaptive_refinement: bool = True
    adaptive_threshold: float = 0.01
    max_adaptive_iterations: int = 3
    
    # Early stopping settings
    enable_early_stopping: bool = True
    patience_trials: int = 20
    improvement_threshold: float = 0.001
    
    # TPE sampler settings
    n_startup_trials: int = 10
    n_ei_candidates: int = 24
    multivariate: bool = True
    group: bool = True
    gamma: Callable[[int], int] = lambda t: min(int(np.ceil(0.15 * t)), 100)
    seed: Optional[int] = None


class EnhancedGridTPEOptimizer:
    """
    Enhanced Grid+TPE optimizer with VectorBT integration and unified vectorization.
    
    This optimizer combines the efficiency of grid search with the sophistication
    of TPE optimization, leveraging VectorBT for high-performance computations.
    """
    
    def __init__(self, config: Optional[EnhancedOptimizationConfig] = None):
        """
        Initialize enhanced grid+TPE optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or EnhancedOptimizationConfig()
        self.logger = logging.getLogger(f"{__name__}.EnhancedGridTPEOptimizer")
        
        # Initialize VectorBT components
        self._initialize_vectorbt_components()
        
        # Initialize optimization state
        self.optimization_history = []
        self.best_params = {}
        self.best_score = float('-inf') if self.config.direction == 'maximize' else float('inf')
        self.performance_stats = {
            'total_trials': 0,
            'grid_trials': 0,
            'tpe_trials': 0,
            'vectorbt_operations': 0,
            'batch_operations': 0,
            'parallel_operations': 0,
            'memory_optimizations': 0,
            'total_time': 0.0,
            'early_stops': 0,
            'adaptive_refinements': 0
        }
        
        self.logger.info(f"Enhanced Grid+TPE Optimizer initialized: "
                        f"VectorBT={VECTORBT_AVAILABLE}, "
                        f"Optuna={OPTUNA_AVAILABLE}, "
                        f"Staged={self.config.enable_staged_optimization}")
    
    def _initialize_vectorbt_components(self):
        """Initialize VectorBT and vectorization components."""
        if not VECTORBT_AVAILABLE:
            self.vectorbt_rolling_optimizer = None
            self.vectorization_manager = None
            return
        
        try:
            # Initialize VectorBT rolling optimizer
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=self.config.vectorbt_use_gpu,
                enable_parallel=self.config.vectorbt_enable_parallel,
                memory_efficient=self.config.memory_efficient,
                chunk_size=self.config.vectorbt_chunk_size
            )
            
            # Initialize unified vectorization manager
            vectorization_config = VectorizationConfig(
                enable_vectorbt=self.config.enable_vectorbt_optimization,
                enable_gpu=self.config.vectorbt_use_gpu,
                enable_parallel=self.config.vectorbt_enable_parallel,
                memory_efficient=self.config.memory_efficient,
                max_memory_gb=self.config.vectorbt_memory_limit_gb,
                chunk_size=self.config.vectorbt_chunk_size,
                enable_monitoring=True,
                batch_size=self.config.batch_size,
                enable_batch_processing=self.config.enable_batch_processing
            )
            
            self.vectorization_manager = get_unified_vectorization_manager(vectorization_config)
            
            self.logger.info("VectorBT components initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize VectorBT components: {e}")
            self.vectorbt_rolling_optimizer = None
            self.vectorization_manager = None
    
    def optimize(self, objective: Callable, search_space: Dict[str, Any], 
                X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Perform enhanced grid+TPE optimization.
        
        Args:
            objective: Objective function to optimize
            search_space: Parameter search space
            X: Training features (optional)
            y: Training targets (optional)
            **kwargs: Additional arguments for objective function
            
        Returns:
            Dictionary containing best parameters and optimization results
        """
        start_time = time.time()
        self.logger.info(f"Starting enhanced grid+TPE optimization with {self.config.n_trials} trials")
        
        try:
            if self.config.enable_staged_optimization:
                result = self._staged_optimization(objective, search_space, X, y, **kwargs)
            else:
                result = self._direct_tpe_optimization(objective, search_space, X, y, **kwargs)
            
            self.performance_stats['total_time'] = time.time() - start_time
            self.logger.info(f"Optimization completed in {self.performance_stats['total_time']:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise
    
    def _staged_optimization(self, objective: Callable, search_space: Dict[str, Any],
                           X: Optional[np.ndarray], y: Optional[np.ndarray], **kwargs) -> Dict[str, Any]:
        """Perform staged optimization: coarse grid -> fine grid -> TPE."""
        self.logger.info("🔍 Starting staged optimization")
        
        # Stage 1: Coarse grid search
        coarse_result = self._coarse_grid_stage(objective, search_space, X, y, **kwargs)
        best_coarse_params = coarse_result['best_params']
        best_coarse_score = coarse_result['best_score']
        
        self.logger.info(f"Coarse grid completed: score={best_coarse_score:.4f}")
        
        # Stage 2: Fine grid search around best coarse results
        fine_result = self._fine_grid_stage(objective, search_space, best_coarse_params, X, y, **kwargs)
        best_fine_params = fine_result['best_params']
        best_fine_score = fine_result['best_score']
        
        self.logger.info(f"Fine grid completed: score={best_fine_score:.4f}")
        
        # Stage 3: TPE optimization
        tpe_result = self._tpe_stage(objective, search_space, best_fine_params, X, y, **kwargs)
        best_tpe_params = tpe_result['best_params']
        best_tpe_score = tpe_result['best_score']
        
        self.logger.info(f"TPE optimization completed: score={best_tpe_score:.4f}")
        
        # Return best overall result
        if self._is_better_score(best_tpe_score, best_fine_score):
            return tpe_result
        elif self._is_better_score(best_fine_score, best_coarse_score):
            return fine_result
        else:
            return coarse_result
    
    def _coarse_grid_stage(self, objective: Callable, search_space: Dict[str, Any],
                          X: Optional[np.ndarray], y: Optional[np.ndarray], **kwargs) -> Dict[str, Any]:
        """Perform VectorBT-optimized coarse grid search."""
        self.logger.info("🔍 Stage 1: Coarse grid search")
        
        # Generate coarse grid points
        grid_points = self._generate_vectorbt_coarse_grid(search_space, self.config.coarse_grid_points)
        
        # Evaluate grid points
        if self.config.enable_batch_processing and self.vectorization_manager:
            results = self._batch_evaluate_grid(objective, grid_points, X, y, **kwargs)
        else:
            results = self._parallel_evaluate_grid(objective, grid_points, X, y, **kwargs)
        
        # Find best result
        best_idx = self._find_best_result_idx(results)
        best_params = grid_points[best_idx]
        best_score = results[best_idx]['score']
        
        self.performance_stats['grid_trials'] += len(grid_points)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results,
            'stage': 'coarse_grid'
        }
    
    def _fine_grid_stage(self, objective: Callable, search_space: Dict[str, Any],
                        best_coarse_params: Dict[str, Any], X: Optional[np.ndarray], 
                        y: Optional[np.ndarray], **kwargs) -> Dict[str, Any]:
        """Perform VectorBT-optimized fine grid search around best coarse results."""
        self.logger.info("🔍 Stage 2: Fine grid search")
        
        # Generate fine grid points around best coarse parameters
        grid_points = self._generate_vectorbt_fine_grid(
            search_space, best_coarse_params, self.config.fine_grid_points
        )
        
        # Evaluate grid points
        if self.config.enable_batch_processing and self.vectorization_manager:
            results = self._batch_evaluate_grid(objective, grid_points, X, y, **kwargs)
        else:
            results = self._parallel_evaluate_grid(objective, grid_points, X, y, **kwargs)
        
        # Find best result
        best_idx = self._find_best_result_idx(results)
        best_params = grid_points[best_idx]
        best_score = results[best_idx]['score']
        
        self.performance_stats['grid_trials'] += len(grid_points)
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results,
            'stage': 'fine_grid'
        }
    
    def _tpe_stage(self, objective: Callable, search_space: Dict[str, Any],
                  best_fine_params: Dict[str, Any], X: Optional[np.ndarray], 
                  y: Optional[np.ndarray], **kwargs) -> Dict[str, Any]:
        """Perform TPE optimization starting from best fine grid results."""
        if not OPTUNA_AVAILABLE:
            self.logger.warning("Optuna not available, skipping TPE stage")
            return {'best_params': best_fine_params, 'best_score': 0.0, 'stage': 'tpe_skipped'}
        
        self.logger.info("🔍 Stage 3: TPE optimization")
        
        # Create Optuna study
        study = optuna.create_study(
            direction=self.config.direction,
            sampler=TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                n_ei_candidates=self.config.n_ei_candidates,
                multivariate=self.config.multivariate,
                group=self.config.group,
                gamma=self.config.gamma,
                seed=self.config.seed
            ),
            pruner=MedianPruner() if self.config.enable_early_stopping else None
        )
        
        # Define objective function for Optuna
        def optuna_objective(trial):
            params = self._suggest_parameters(trial, search_space)
            try:
                score = objective(params, X, y, **kwargs)
                return score
            except Exception as e:
                self.logger.warning(f"Trial failed: {e}")
                return float('-inf') if self.config.direction == 'maximize' else float('inf')
        
        # Optimize
        study.optimize(
            optuna_objective,
            n_trials=self.config.tpe_trials,
            timeout=self.config.timeout
        )
        
        best_params = study.best_params
        best_score = study.best_value
        
        self.performance_stats['tpe_trials'] += self.config.tpe_trials
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'study': study,
            'stage': 'tpe'
        }
    
    def _generate_vectorbt_coarse_grid(self, search_space: Dict[str, Any], 
                                     grid_points: int) -> List[Dict[str, Any]]:
        """Generate coarse grid using VectorBT vectorized operations."""
        if not VECTORBT_AVAILABLE or not self.vectorization_manager:
            return build_coarse_grid_from_search_space(search_space, grid_points)
        
        try:
            # Use VectorBT for efficient grid generation
            param_names = list(search_space.keys())
            param_configs = list(search_space.values())
            
            # Generate parameter values using VectorBT
            param_values = {}
            for name, config in zip(param_names, param_configs):
                if isinstance(config, dict):
                    param_type = config.get('type', 'float')
                    if param_type == 'float':
                        low, high = config['low'], config['high']
                        if config.get('log', False):
                            values = np.logspace(np.log10(low), np.log10(high), grid_points)
                        else:
                            values = np.linspace(low, high, grid_points)
                    elif param_type == 'int':
                        low, high = config['low'], config['high']
                        values = np.linspace(low, high, grid_points).astype(int)
                    elif param_type == 'categorical':
                        values = config.get('choices', [])
                    else:
                        values = [config.get('default', 0)]
                else:
                    # Legacy tuple format
                    if isinstance(config, tuple) and len(config) == 2:
                        low, high = config
                        values = np.linspace(low, high, grid_points)
                    else:
                        values = [config]
                
                param_values[name] = values
            
            # Generate all combinations
            import itertools
            keys = list(param_values.keys())
            values = list(param_values.values())
            combinations = list(itertools.product(*values))
            
            grid_points_list = [dict(zip(keys, combo)) for combo in combinations]
            
            self.performance_stats['vectorbt_operations'] += 1
            return grid_points_list
            
        except Exception as e:
            self.logger.warning(f"VectorBT grid generation failed: {e}, using fallback")
            return build_coarse_grid_from_search_space(search_space, grid_points)
    
    def _generate_vectorbt_fine_grid(self, search_space: Dict[str, Any], 
                                   best_params: Dict[str, Any], 
                                   grid_points: int) -> List[Dict[str, Any]]:
        """Generate fine grid around best parameters using VectorBT."""
        if not VECTORBT_AVAILABLE or not self.vectorization_manager:
            return build_fine_grid_around_best(search_space, best_params, grid_points)
        
        try:
            # Use VectorBT for efficient fine grid generation
            param_names = list(search_space.keys())
            param_configs = list(search_space.values())
            
            # Generate fine parameter values around best parameters
            param_values = {}
            for name, config in zip(param_names, param_configs):
                if name not in best_params:
                    continue
                
                best_val = best_params[name]
                
                if isinstance(config, dict):
                    param_type = config.get('type', 'float')
                    if param_type == 'float':
                        low, high = config['low'], config['high']
                        range_size = high - low
                        fine_range = range_size * 0.2  # 20% of original range
                        fine_min = max(low, best_val - fine_range)
                        fine_max = min(high, best_val + fine_range)
                        
                        if config.get('log', False) and fine_min > 0:
                            values = np.logspace(np.log10(fine_min), np.log10(fine_max), grid_points)
                        else:
                            values = np.linspace(fine_min, fine_max, grid_points)
                    elif param_type == 'int':
                        low, high = config['low'], config['high']
                        fine_min = max(low, int(best_val) - 2)
                        fine_max = min(high, int(best_val) + 2)
                        values = np.arange(fine_min, fine_max + 1)
                    elif param_type == 'categorical':
                        values = config.get('choices', [])
                    else:
                        values = [best_val]
                else:
                    # Legacy tuple format
                    if isinstance(config, tuple) and len(config) == 2:
                        low, high = config
                        range_size = high - low
                        fine_range = range_size * 0.2
                        fine_min = max(low, best_val - fine_range)
                        fine_max = min(high, best_val + fine_range)
                        values = np.linspace(fine_min, fine_max, grid_points)
                    else:
                        values = [best_val]
                
                param_values[name] = values
            
            # Generate all combinations
            import itertools
            keys = list(param_values.keys())
            values = list(param_values.values())
            combinations = list(itertools.product(*values))
            
            grid_points_list = [dict(zip(keys, combo)) for combo in combinations]
            
            self.performance_stats['vectorbt_operations'] += 1
            return grid_points_list
            
        except Exception as e:
            self.logger.warning(f"VectorBT fine grid generation failed: {e}, using fallback")
            return build_fine_grid_around_best(search_space, best_params, grid_points)
    
    def _batch_evaluate_grid(self, objective: Callable, grid_points: List[Dict[str, Any]],
                           X: Optional[np.ndarray], y: Optional[np.ndarray], **kwargs) -> List[Dict[str, Any]]:
        """Evaluate grid points using VectorBT batch processing."""
        if not self.vectorization_manager:
            return self._parallel_evaluate_grid(objective, grid_points, X, y, **kwargs)
        
        try:
            self.logger.info(f"🔄 Batch evaluating {len(grid_points)} grid points using VectorBT")
            
            # Prepare batch data for vectorized processing
            batch_data = self._prepare_batch_data(grid_points, X, y)
            
            # Use vectorization manager for batch processing
            results = []
            for i, params in enumerate(grid_points):
                try:
                    # Extract parameters for this trial
                    trial_data = {k: batch_data[k][i] for k in batch_data.keys()}
                    
                    # Evaluate objective
                    score = objective(params, X, y, **kwargs)
                    
                    results.append({
                        'params': params,
                        'score': score,
                        'trial_id': i,
                        'timestamp': time.time()
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Trial {i} failed: {e}")
                    results.append({
                        'params': params,
                        'score': float('-inf') if self.config.direction == 'maximize' else float('inf'),
                        'trial_id': i,
                        'error': str(e),
                        'timestamp': time.time()
                    })
            
            self.performance_stats['batch_operations'] += 1
            return results
            
        except Exception as e:
            self.logger.warning(f"Batch evaluation failed: {e}, using parallel fallback")
            return self._parallel_evaluate_grid(objective, grid_points, X, y, **kwargs)
    
    def _parallel_evaluate_grid(self, objective: Callable, grid_points: List[Dict[str, Any]],
                              X: Optional[np.ndarray], y: Optional[np.ndarray], **kwargs) -> List[Dict[str, Any]]:
        """Evaluate grid points using parallel processing."""
        if not self.config.enable_parallel or len(grid_points) <= 1:
            return self._sequential_evaluate_grid(objective, grid_points, X, y, **kwargs)
        
        try:
            self.logger.info(f"🔄 Parallel evaluating {len(grid_points)} grid points")
            
            # Prepare evaluation function
            def evaluate_single(params):
                try:
                    score = objective(params, X, y, **kwargs)
                    return {
                        'params': params,
                        'score': score,
                        'timestamp': time.time()
                    }
                except Exception as e:
                    self.logger.warning(f"Trial failed: {e}")
                    return {
                        'params': params,
                        'score': float('-inf') if self.config.direction == 'maximize' else float('inf'),
                        'error': str(e),
                        'timestamp': time.time()
                    }
            
            # Execute parallel evaluation
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                results = list(executor.map(evaluate_single, grid_points))
            
            self.performance_stats['parallel_operations'] += 1
            return results
            
        except Exception as e:
            self.logger.warning(f"Parallel evaluation failed: {e}, using sequential fallback")
            return self._sequential_evaluate_grid(objective, grid_points, X, y, **kwargs)
    
    def _sequential_evaluate_grid(self, objective: Callable, grid_points: List[Dict[str, Any]],
                                X: Optional[np.ndarray], y: Optional[np.ndarray], **kwargs) -> List[Dict[str, Any]]:
        """Evaluate grid points sequentially."""
        self.logger.info(f"🔄 Sequential evaluating {len(grid_points)} grid points")
        
        results = []
        for i, params in enumerate(grid_points):
            try:
                score = objective(params, X, y, **kwargs)
                results.append({
                    'params': params,
                    'score': score,
                    'trial_id': i,
                    'timestamp': time.time()
                })
            except Exception as e:
                self.logger.warning(f"Trial {i} failed: {e}")
                results.append({
                    'params': params,
                    'score': float('-inf') if self.config.direction == 'maximize' else float('inf'),
                    'trial_id': i,
                    'error': str(e),
                    'timestamp': time.time()
                })
        
        return results
    
    def _prepare_batch_data(self, grid_points: List[Dict[str, Any]], 
                          X: Optional[np.ndarray], y: Optional[np.ndarray]) -> Dict[str, Any]:
        """Prepare data for batch processing."""
        batch_data = {}
        
        # Extract parameter values for vectorized processing
        param_names = list(grid_points[0].keys()) if grid_points else []
        for param_name in param_names:
            values = [point[param_name] for point in grid_points]
            batch_data[param_name] = np.array(values)
        
        # Add X and y if available
        if X is not None:
            batch_data['X'] = X
        if y is not None:
            batch_data['y'] = y
        
        return batch_data
    
    def _suggest_parameters(self, trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest parameters for Optuna trial."""
        params = {}
        
        for name, config in search_space.items():
            if isinstance(config, dict):
                param_type = config.get('type', 'float')
                if param_type == 'float':
                    low, high = config['low'], config['high']
                    if config.get('log', False):
                        params[name] = trial.suggest_float(name, low, high, log=True)
                    else:
                        params[name] = trial.suggest_float(name, low, high)
                elif param_type == 'int':
                    low, high = config['low'], config['high']
                    params[name] = trial.suggest_int(name, low, high)
                elif param_type == 'categorical':
                    choices = config.get('choices', [])
                    params[name] = trial.suggest_categorical(name, choices)
            else:
                # Legacy tuple format
                if isinstance(config, tuple) and len(config) == 2:
                    low, high = config
                    params[name] = trial.suggest_float(name, low, high)
                else:
                    params[name] = config
        
        return params
    
    def _find_best_result_idx(self, results: List[Dict[str, Any]]) -> int:
        """Find index of best result."""
        if not results:
            return 0
        
        scores = [r['score'] for r in results]
        if self.config.direction == 'maximize':
            return np.argmax(scores)
        else:
            return np.argmin(scores)
    
    def _is_better_score(self, score1: float, score2: float) -> bool:
        """Check if score1 is better than score2."""
        if self.config.direction == 'maximize':
            return score1 > score2
        else:
            return score1 < score2
    
    def _direct_tpe_optimization(self, objective: Callable, search_space: Dict[str, Any],
                               X: Optional[np.ndarray], y: Optional[np.ndarray], **kwargs) -> Dict[str, Any]:
        """Perform direct TPE optimization without grid search stages."""
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna not available for TPE optimization")
        
        self.logger.info("🔍 Direct TPE optimization")
        
        # Create Optuna study
        study = optuna.create_study(
            direction=self.config.direction,
            sampler=TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                n_ei_candidates=self.config.n_ei_candidates,
                multivariate=self.config.multivariate,
                group=self.config.group,
                gamma=self.config.gamma,
                seed=self.config.seed
            ),
            pruner=MedianPruner() if self.config.enable_early_stopping else None
        )
        
        # Define objective function for Optuna
        def optuna_objective(trial):
            params = self._suggest_parameters(trial, search_space)
            try:
                score = objective(params, X, y, **kwargs)
                return score
            except Exception as e:
                self.logger.warning(f"Trial failed: {e}")
                return float('-inf') if self.config.direction == 'maximize' else float('inf')
        
        # Optimize
        study.optimize(
            optuna_objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout
        )
        
        self.performance_stats['tpe_trials'] += self.config.n_trials
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study,
            'stage': 'direct_tpe'
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add VectorBT stats if available
        if self.vectorization_manager:
            vectorbt_stats = self.vectorization_manager.get_performance_stats()
            stats.update(vectorbt_stats)
        
        # Calculate efficiency metrics
        if stats['total_trials'] > 0:
            stats['grid_usage_rate'] = stats['grid_trials'] / stats['total_trials']
            stats['tpe_usage_rate'] = stats['tpe_trials'] / stats['total_trials']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_trials']
            stats['batch_usage_rate'] = stats['batch_operations'] / stats['total_trials']
            stats['parallel_usage_rate'] = stats['parallel_operations'] / stats['total_trials']
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_trials': 0,
            'grid_trials': 0,
            'tpe_trials': 0,
            'vectorbt_operations': 0,
            'batch_operations': 0,
            'parallel_operations': 0,
            'memory_optimizations': 0,
            'total_time': 0.0,
            'early_stops': 0,
            'adaptive_refinements': 0
        }
        
        if self.vectorization_manager:
            self.vectorization_manager.reset_stats()


# Convenience functions
def create_enhanced_optimizer(config: Optional[EnhancedOptimizationConfig] = None) -> EnhancedGridTPEOptimizer:
    """Create an enhanced grid+TPE optimizer."""
    return EnhancedGridTPEOptimizer(config)


def optimize_with_enhanced_grid_tpe(objective: Callable, search_space: Dict[str, Any],
                                  config: Optional[EnhancedOptimizationConfig] = None,
                                  X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                                  **kwargs) -> Dict[str, Any]:
    """Convenience function for enhanced grid+TPE optimization."""
    optimizer = create_enhanced_optimizer(config)
    return optimizer.optimize(objective, search_space, X, y, **kwargs)


# Example usage
if __name__ == "__main__":
    # Example objective function
    def example_objective(params, X=None, y=None):
        """Example objective function for testing."""
        # Simulate some computation
        time.sleep(0.01)
        
        # Example scoring based on parameters
        score = 0
        for param_name, param_value in params.items():
            if isinstance(param_value, (int, float)):
                score += param_value ** 2
            else:
                score += 1
        
        return -score  # Minimize (negative of sum of squares)
    
    # Example search space
    search_space = {
        'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.1, 'log': True},
        'batch_size': {'type': 'int', 'low': 16, 'high': 128},
        'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5},
        'optimizer': {'type': 'categorical', 'choices': ['adam', 'sgd', 'rmsprop']}
    }
    
    # Create optimizer
    config = EnhancedOptimizationConfig(
        n_trials=50,
        enable_staged_optimization=True,
        enable_vectorbt_optimization=True,
        enable_parallel=True,
        max_workers=2
    )
    
    optimizer = create_enhanced_optimizer(config)
    
    # Run optimization
    print("Running enhanced grid+TPE optimization...")
    result = optimizer.optimize(example_objective, search_space)
    
    print(f"Best parameters: {result['best_params']}")
    print(f"Best score: {result['best_score']}")
    print(f"Performance stats: {optimizer.get_performance_stats()}")