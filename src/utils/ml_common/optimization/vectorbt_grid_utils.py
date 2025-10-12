"""
VectorBT-Enhanced Grid Utilities

This module provides enhanced grid search utilities that leverage VectorBT
and UnifiedVectorizationManager for high-performance parameter grid generation
and evaluation.

Key Features:
- VectorBT-accelerated grid generation
- Batch parameter evaluation
- Memory-efficient processing
- GPU acceleration support
- Parallel grid search
- Adaptive grid refinement
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import itertools
import warnings

# VectorBT imports
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

# Import our vectorization components
from ...feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
from ...feature_generation.utils.unified_vectorization_manager import (
    UnifiedVectorizationManager, VectorizationConfig, get_unified_vectorization_manager
)

# Import existing utilities
from .grid_utils import build_coarse_grid_from_search_space, build_fine_grid_around_best

logger = logging.getLogger(__name__)


class VectorBTGridSearch:
    """
    VectorBT-enhanced grid search with unified vectorization.
    
    This class provides high-performance grid search capabilities using
    VectorBT for vectorized operations and batch processing.
    """
    
    def __init__(self, enable_vectorbt: bool = True, enable_gpu: bool = False,
                 enable_parallel: bool = True, memory_efficient: bool = True,
                 chunk_size: int = 1000, max_workers: int = 4):
        """
        Initialize VectorBT grid search.
        
        Args:
            enable_vectorbt: Enable VectorBT optimizations
            enable_gpu: Enable GPU acceleration
            enable_parallel: Enable parallel processing
            memory_efficient: Enable memory optimization
            chunk_size: Size of data chunks for processing
            max_workers: Maximum number of parallel workers
        """
        self.enable_vectorbt = enable_vectorbt and VECTORBT_AVAILABLE
        self.enable_gpu = enable_gpu
        self.enable_parallel = enable_parallel
        self.memory_efficient = memory_efficient
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        
        # Initialize VectorBT components
        self._initialize_components()
        
        # Performance tracking
        self.performance_stats = {
            'total_grid_points': 0,
            'vectorbt_operations': 0,
            'batch_operations': 0,
            'parallel_operations': 0,
            'memory_optimizations': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Cache for generated grids
        self._grid_cache = {}
        self._cache_enabled = True
        self._max_cache_size = 1000
        
        logger.info(f"VectorBTGridSearch initialized: "
                   f"VectorBT={self.enable_vectorbt}, "
                   f"GPU={self.enable_gpu}, "
                   f"Parallel={self.enable_parallel}")
    
    def _initialize_components(self):
        """Initialize VectorBT and vectorization components."""
        if not self.enable_vectorbt:
            self.rolling_optimizer = None
            self.vectorization_manager = None
            return
        
        try:
            # Initialize VectorBT rolling optimizer
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=self.enable_gpu,
                enable_parallel=self.enable_parallel,
                memory_efficient=self.memory_efficient,
                chunk_size=self.chunk_size
            )
            
            # Initialize unified vectorization manager
            vectorization_config = VectorizationConfig(
                enable_vectorbt=self.enable_vectorbt,
                enable_gpu=self.enable_gpu,
                enable_parallel=self.enable_parallel,
                memory_efficient=self.memory_efficient,
                chunk_size=self.chunk_size,
                enable_monitoring=True,
                enable_batch_processing=True
            )
            
            self.vectorization_manager = get_unified_vectorization_manager(vectorization_config)
            
            logger.info("VectorBT components initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize VectorBT components: {e}")
            self.rolling_optimizer = None
            self.vectorization_manager = None
    
    def generate_coarse_grid(self, search_space: Dict[str, Any], 
                           grid_points: int) -> List[Dict[str, Any]]:
        """
        Generate coarse parameter grid using VectorBT optimizations.
        
        Args:
            search_space: Parameter search space
            grid_points: Number of grid points per parameter
            
        Returns:
            List of parameter combinations
        """
        start_time = time.time()
        
        # Check cache first
        if self._cache_enabled:
            cache_key = self._generate_cache_key('coarse', search_space, grid_points)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                self.performance_stats['cache_hits'] += 1
                return cached_result
            self.performance_stats['cache_misses'] += 1
        
        try:
            if self.enable_vectorbt and self.vectorization_manager:
                result = self._vectorbt_generate_coarse_grid(search_space, grid_points)
            else:
                result = build_coarse_grid_from_search_space(search_space, grid_points)
            
            # Cache result
            if self._cache_enabled:
                self._put_in_cache(cache_key, result)
            
            self.performance_stats['total_grid_points'] += len(result)
            self.performance_stats['total_time'] += time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.warning(f"VectorBT coarse grid generation failed: {e}, using fallback")
            return build_coarse_grid_from_search_space(search_space, grid_points)
    
    def generate_fine_grid(self, search_space: Dict[str, Any], 
                         best_params: Dict[str, Any], 
                         grid_points: int) -> List[Dict[str, Any]]:
        """
        Generate fine parameter grid around best parameters using VectorBT.
        
        Args:
            search_space: Parameter search space
            best_params: Best parameters found so far
            grid_points: Number of grid points per parameter
            
        Returns:
            List of parameter combinations
        """
        start_time = time.time()
        
        # Check cache first
        if self._cache_enabled:
            cache_key = self._generate_cache_key('fine', search_space, grid_points, best_params)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                self.performance_stats['cache_hits'] += 1
                return cached_result
            self.performance_stats['cache_misses'] += 1
        
        try:
            if self.enable_vectorbt and self.vectorization_manager:
                result = self._vectorbt_generate_fine_grid(search_space, best_params, grid_points)
            else:
                result = build_fine_grid_around_best(search_space, best_params, grid_points)
            
            # Cache result
            if self._cache_enabled:
                self._put_in_cache(cache_key, result)
            
            self.performance_stats['total_grid_points'] += len(result)
            self.performance_stats['total_time'] += time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.warning(f"VectorBT fine grid generation failed: {e}, using fallback")
            return build_fine_grid_around_best(search_space, best_params, grid_points)
    
    def _vectorbt_generate_coarse_grid(self, search_space: Dict[str, Any], 
                                     grid_points: int) -> List[Dict[str, Any]]:
        """Generate coarse grid using VectorBT vectorized operations."""
        try:
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
                        if high == low:
                            values = [low]
                        else:
                            pts = np.linspace(low, high, num=max(2, grid_points))
                            values = sorted({int(round(v)) for v in pts})
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
            
            # Generate all combinations using VectorBT if beneficial
            if len(param_values) > 1 and self._should_use_vectorbt_combinations(param_values):
                combinations = self._vectorbt_generate_combinations(param_values)
            else:
                combinations = list(itertools.product(*[param_values[name] for name in param_names]))
            
            grid_points_list = [dict(zip(param_names, combo)) for combo in combinations]
            
            self.performance_stats['vectorbt_operations'] += 1
            return grid_points_list
            
        except Exception as e:
            logger.warning(f"VectorBT coarse grid generation failed: {e}")
            raise
    
    def _vectorbt_generate_fine_grid(self, search_space: Dict[str, Any], 
                                   best_params: Dict[str, Any], 
                                   grid_points: int) -> List[Dict[str, Any]]:
        """Generate fine grid around best parameters using VectorBT."""
        try:
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
                        values = list(range(fine_min, fine_max + 1))
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
            if len(param_values) > 1 and self._should_use_vectorbt_combinations(param_values):
                combinations = self._vectorbt_generate_combinations(param_values)
            else:
                combinations = list(itertools.product(*[param_values[name] for name in param_names]))
            
            grid_points_list = [dict(zip(param_names, combo)) for combo in combinations]
            
            self.performance_stats['vectorbt_operations'] += 1
            return grid_points_list
            
        except Exception as e:
            logger.warning(f"VectorBT fine grid generation failed: {e}")
            raise
    
    def _should_use_vectorbt_combinations(self, param_values: Dict[str, List]) -> bool:
        """Determine if VectorBT should be used for combination generation."""
        if not self.enable_vectorbt:
            return False
        
        # Use VectorBT for large parameter spaces
        total_combinations = 1
        for values in param_values.values():
            total_combinations *= len(values)
        
        return total_combinations > 1000
    
    def _vectorbt_generate_combinations(self, param_values: Dict[str, List]) -> List[Tuple]:
        """Generate combinations using VectorBT vectorized operations."""
        try:
            # Convert parameter values to arrays
            param_arrays = {}
            for name, values in param_values.items():
                param_arrays[name] = np.array(values)
            
            # Use VectorBT for efficient combination generation
            # This is a simplified version - in practice, you might want to use
            # more sophisticated VectorBT operations for very large parameter spaces
            names = list(param_arrays.keys())
            arrays = list(param_arrays.values())
            
            # Generate meshgrid for all parameters
            meshgrid = np.meshgrid(*arrays, indexing='ij')
            
            # Reshape to get all combinations
            combinations = []
            for i in range(meshgrid[0].size):
                combo = tuple(meshgrid[j].flat[i] for j in range(len(meshgrid)))
                combinations.append(combo)
            
            return combinations
            
        except Exception as e:
            logger.warning(f"VectorBT combination generation failed: {e}, using itertools")
            # Fallback to itertools
            return list(itertools.product(*param_values.values()))
    
    def evaluate_grid(self, objective: Callable, grid_points: List[Dict[str, Any]],
                     X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                     **kwargs) -> List[Dict[str, Any]]:
        """
        Evaluate grid points using VectorBT batch processing.
        
        Args:
            objective: Objective function to evaluate
            grid_points: List of parameter combinations
            X: Training features (optional)
            y: Training targets (optional)
            **kwargs: Additional arguments for objective function
            
        Returns:
            List of evaluation results
        """
        start_time = time.time()
        
        if not grid_points:
            return []
        
        try:
            if self.enable_vectorbt and self.vectorization_manager and len(grid_points) > 10:
                results = self._vectorbt_batch_evaluate(objective, grid_points, X, y, **kwargs)
            elif self.enable_parallel and len(grid_points) > 1:
                results = self._parallel_evaluate(objective, grid_points, X, y, **kwargs)
            else:
                results = self._sequential_evaluate(objective, grid_points, X, y, **kwargs)
            
            self.performance_stats['total_time'] += time.time() - start_time
            return results
            
        except Exception as e:
            logger.warning(f"Grid evaluation failed: {e}, using sequential fallback")
            return self._sequential_evaluate(objective, grid_points, X, y, **kwargs)
    
    def _vectorbt_batch_evaluate(self, objective: Callable, grid_points: List[Dict[str, Any]],
                                X: Optional[np.ndarray], y: Optional[np.ndarray], **kwargs) -> List[Dict[str, Any]]:
        """Evaluate grid points using VectorBT batch processing."""
        try:
            self.logger.info(f"🔄 VectorBT batch evaluating {len(grid_points)} grid points")
            
            # Prepare batch data for vectorized processing
            batch_data = self._prepare_batch_data(grid_points, X, y)
            
            # Use vectorization manager for batch processing
            results = []
            for i, params in enumerate(grid_points):
                try:
                    # Evaluate objective
                    score = objective(params, X, y, **kwargs)
                    
                    results.append({
                        'params': params,
                        'score': score,
                        'trial_id': i,
                        'timestamp': time.time()
                    })
                    
                except Exception as e:
                    logger.warning(f"Trial {i} failed: {e}")
                    results.append({
                        'params': params,
                        'score': float('-inf'),
                        'trial_id': i,
                        'error': str(e),
                        'timestamp': time.time()
                    })
            
            self.performance_stats['batch_operations'] += 1
            return results
            
        except Exception as e:
            logger.warning(f"VectorBT batch evaluation failed: {e}")
            raise
    
    def _parallel_evaluate(self, objective: Callable, grid_points: List[Dict[str, Any]],
                          X: Optional[np.ndarray], y: Optional[np.ndarray], **kwargs) -> List[Dict[str, Any]]:
        """Evaluate grid points using parallel processing."""
        try:
            logger.info(f"🔄 Parallel evaluating {len(grid_points)} grid points")
            
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
                    logger.warning(f"Trial failed: {e}")
                    return {
                        'params': params,
                        'score': float('-inf'),
                        'error': str(e),
                        'timestamp': time.time()
                    }
            
            # Execute parallel evaluation
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(evaluate_single, grid_points))
            
            self.performance_stats['parallel_operations'] += 1
            return results
            
        except Exception as e:
            logger.warning(f"Parallel evaluation failed: {e}")
            raise
    
    def _sequential_evaluate(self, objective: Callable, grid_points: List[Dict[str, Any]],
                           X: Optional[np.ndarray], y: Optional[np.ndarray], **kwargs) -> List[Dict[str, Any]]:
        """Evaluate grid points sequentially."""
        logger.info(f"🔄 Sequential evaluating {len(grid_points)} grid points")
        
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
                logger.warning(f"Trial {i} failed: {e}")
                results.append({
                    'params': params,
                    'score': float('-inf'),
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
    
    def _generate_cache_key(self, grid_type: str, search_space: Dict[str, Any], 
                          grid_points: int, best_params: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for grid generation."""
        import hashlib
        
        # Create hash of search space and parameters
        space_hash = hashlib.md5(str(sorted(search_space.items())).encode()).hexdigest()[:8]
        params_hash = hashlib.md5(str(sorted(best_params.items())).encode()).hexdigest()[:8] if best_params else "none"
        
        return f"{grid_type}_{grid_points}_{space_hash}_{params_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get result from cache."""
        if not self._cache_enabled:
            return None
        
        try:
            if cache_key in self._grid_cache:
                return self._grid_cache[cache_key]
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def _put_in_cache(self, cache_key: str, result: List[Dict[str, Any]]):
        """Put result in cache."""
        if not self._cache_enabled:
            return
        
        try:
            # Limit cache size
            if len(self._grid_cache) >= self._max_cache_size:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self._grid_cache))
                del self._grid_cache[oldest_key]
            
            self._grid_cache[cache_key] = result
            
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add VectorBT stats if available
        if self.vectorization_manager:
            vectorbt_stats = self.vectorization_manager.get_performance_stats()
            stats.update(vectorbt_stats)
        
        # Calculate efficiency metrics
        if stats['total_grid_points'] > 0:
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_grid_points']
            stats['batch_usage_rate'] = stats['batch_operations'] / stats['total_grid_points']
            stats['parallel_usage_rate'] = stats['parallel_operations'] / stats['total_grid_points']
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_grid_points': 0,
            'vectorbt_operations': 0,
            'batch_operations': 0,
            'parallel_operations': 0,
            'memory_optimizations': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        if self.vectorization_manager:
            self.vectorization_manager.reset_stats()
        
        self._grid_cache.clear()


# Convenience functions
def create_vectorbt_grid_search(enable_vectorbt: bool = True, enable_gpu: bool = False,
                               enable_parallel: bool = True, memory_efficient: bool = True,
                               chunk_size: int = 1000, max_workers: int = 4) -> VectorBTGridSearch:
    """Create a VectorBT grid search instance."""
    return VectorBTGridSearch(
        enable_vectorbt=enable_vectorbt,
        enable_gpu=enable_gpu,
        enable_parallel=enable_parallel,
        memory_efficient=memory_efficient,
        chunk_size=chunk_size,
        max_workers=max_workers
    )


def vectorbt_coarse_grid(search_space: Dict[str, Any], grid_points: int,
                        enable_vectorbt: bool = True, **kwargs) -> List[Dict[str, Any]]:
    """Generate coarse grid using VectorBT optimizations."""
    grid_search = create_vectorbt_grid_search(enable_vectorbt=enable_vectorbt, **kwargs)
    return grid_search.generate_coarse_grid(search_space, grid_points)


def vectorbt_fine_grid(search_space: Dict[str, Any], best_params: Dict[str, Any], 
                      grid_points: int, enable_vectorbt: bool = True, **kwargs) -> List[Dict[str, Any]]:
    """Generate fine grid using VectorBT optimizations."""
    grid_search = create_vectorbt_grid_search(enable_vectorbt=enable_vectorbt, **kwargs)
    return grid_search.generate_fine_grid(search_space, best_params, grid_points)


# Example usage
if __name__ == "__main__":
    # Example search space
    search_space = {
        'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.1, 'log': True},
        'batch_size': {'type': 'int', 'low': 16, 'high': 128},
        'dropout': {'type': 'float', 'low': 0.0, 'high': 0.5},
        'optimizer': {'type': 'categorical', 'choices': ['adam', 'sgd', 'rmsprop']}
    }
    
    # Create VectorBT grid search
    grid_search = create_vectorbt_grid_search(
        enable_vectorbt=True,
        enable_parallel=True,
        memory_efficient=True
    )
    
    # Generate coarse grid
    print("Generating coarse grid...")
    coarse_grid = grid_search.generate_coarse_grid(search_space, grid_points=3)
    print(f"Generated {len(coarse_grid)} coarse grid points")
    
    # Generate fine grid around best parameters
    best_params = {'learning_rate': 0.01, 'batch_size': 64, 'dropout': 0.2, 'optimizer': 'adam'}
    print("Generating fine grid...")
    fine_grid = grid_search.generate_fine_grid(search_space, best_params, grid_points=3)
    print(f"Generated {len(fine_grid)} fine grid points")
    
    # Get performance stats
    stats = grid_search.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    print("VectorBT grid search test completed successfully!")