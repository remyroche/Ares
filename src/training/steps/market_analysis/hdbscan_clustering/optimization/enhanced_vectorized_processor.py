"""
Enhanced Vectorized Computations for HDBSCAN Clustering

This module provides comprehensive vectorized computations using
VectorBTRollingOptimizer and UnifiedVectorizationManager.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import time
import gc
import psutil

# Import VectorBT optimization components
from src.utils.ml_common.unified_vectorization_manager import (
    UnifiedVectorizationManager,
    VectorizationConfig,
    get_unified_vectorization_manager,
    OperationType,
    OptimizationStrategy
)

# Import VectorBTRollingOptimizer
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer,
        get_vectorbt_rolling_optimizer
    )
    VECTORBT_ROLLING_AVAILABLE = True
except ImportError:
    VECTORBT_ROLLING_AVAILABLE = False
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None

# Import HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    hdbscan = None

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance, tprint_progress, tprint_timer,
    tprint_logged, LogLevel
)
from src.utils.common_operations import optimize_dataframe_memory, get_memory_usage

logger = logging.getLogger(__name__)

@dataclass
class VectorizedProcessingConfig:
    """Configuration for vectorized processing."""
    # VectorBT settings
    enable_vectorbt: bool = True
    enable_gpu: bool = False
    enable_parallel: bool = True
    
    # Memory optimization
    memory_efficient: bool = True
    max_memory_gb: float = 8.0
    chunk_size: int = 1000
    
    # Performance optimization
    enable_rolling_optimization: bool = True
    enable_distance_optimization: bool = True
    enable_clustering_optimization: bool = True
    
    # VectorBT rolling settings
    rolling_optimization_threshold: int = 1000
    enable_rolling_optimization: bool = True

class EnhancedVectorizedProcessor:
    """
    Enhanced vectorized processor with comprehensive VectorBT optimizations.
    
    Provides:
    - VectorBT rolling operations
    - Optimized distance calculations
    - Vectorized mathematical operations
    - HDBSCAN clustering optimization
    """
    
    @tprint_logged(LogLevel.INFO, include_args=True)
    def __init__(self, config: Optional[VectorizedProcessingConfig] = None):
        """Initialize the enhanced vectorized processor."""
        start_time = time.perf_counter()
        initial_memory = get_memory_usage()
        
        self.config = config or VectorizedProcessingConfig()
        
        tprint_info("Initializing EnhancedVectorizedProcessor")
        tprint_debug(f"Config: enable_vectorbt={self.config.enable_vectorbt}, enable_gpu={self.config.enable_gpu}")
        
        # Initialize UnifiedVectorizationManager
        with tprint_timer("Vectorization manager initialization"):
            vectorization_config = VectorizationConfig(
                enable_vectorbt=self.config.enable_vectorbt,
                enable_gpu=self.config.enable_gpu,
                memory_efficient=self.config.memory_efficient,
                max_memory_gb=self.config.max_memory_gb,
                chunk_size=self.config.chunk_size,
                enable_parallel=self.config.enable_parallel
            )
            self.vectorization_manager = get_unified_vectorization_manager(vectorization_config)
            tprint_debug(f"Vectorization manager initialized: vectorbt={self.config.enable_vectorbt}, gpu={self.config.enable_gpu}")
        
        # Initialize VectorBTRollingOptimizer
        with tprint_timer("VectorBT rolling optimizer initialization"):
            if VECTORBT_ROLLING_AVAILABLE and self.config.enable_rolling_optimization:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=self.config.enable_gpu,
                    enable_parallel=self.config.enable_parallel,
                    memory_efficient=self.config.memory_efficient,
                    chunk_size=self.config.chunk_size
                )
                tprint_debug("VectorBT rolling optimizer initialized successfully")
            else:
                self.rolling_optimizer = None
                tprint_debug("VectorBT rolling optimizer not available or disabled")
        
        # Performance tracking
        self.performance_stats = {
            'vectorized_operations': 0,
            'rolling_operations': 0,
            'distance_calculations': 0,
            'clustering_operations': 0,
            'vectorbt_usage_rate': 0.0,
            'gpu_usage_rate': 0.0,
            'memory_optimizations': 0,
            'processing_time': 0.0,
            'initialization_time': 0.0,
            'initial_memory_mb': initial_memory
        }
        
        # Track initialization performance
        init_time = time.perf_counter() - start_time
        final_memory = get_memory_usage()
        self.performance_stats['initialization_time'] = init_time
        self.performance_stats['memory_usage_mb'] = final_memory
        
        tprint_success("✅ EnhancedVectorizedProcessor initialized")
        tprint_performance("Vectorized processor initialization", init_time)
        tprint_debug(f"Memory usage: {initial_memory:.2f}MB -> {final_memory:.2f}MB (delta: {final_memory - initial_memory:+.2f}MB)")
        
        logger.info("✅ EnhancedVectorizedProcessor initialized")
    
    def vectorized_rolling_operation(self, data: pd.DataFrame, 
                                   operation: str, 
                                   window: int, 
                                   **kwargs) -> pd.DataFrame:
        """Perform vectorized rolling operations using VectorBT optimization."""
        start_time = time.time()
        
        if self.rolling_optimizer and len(data) > self.config.rolling_optimization_threshold:
            # Use VectorBT rolling optimization
            result = self._vectorbt_rolling_operation(data, operation, window, **kwargs)
            self.performance_stats['vectorbt_usage_rate'] += 1
        else:
            # Use standard pandas rolling
            result = self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        # Update stats
        self.performance_stats['vectorized_operations'] += 1
        self.performance_stats['rolling_operations'] += 1
        self.performance_stats['processing_time'] += time.time() - start_time
        
        return result
    
    def _vectorbt_rolling_operation(self, data: pd.DataFrame, 
                                  operation: str, 
                                  window: int, 
                                  **kwargs) -> pd.DataFrame:
        """Perform rolling operation using VectorBT optimization."""
        try:
            if operation == 'mean':
                return self.rolling_optimizer.rolling_mean(data, window, **kwargs)
            elif operation == 'std':
                return self.rolling_optimizer.rolling_std(data, window, **kwargs)
            elif operation == 'var':
                return self.rolling_optimizer.rolling_var(data, window, **kwargs)
            elif operation == 'min':
                return self.rolling_optimizer.rolling_min(data, window, **kwargs)
            elif operation == 'max':
                return self.rolling_optimizer.rolling_max(data, window, **kwargs)
            elif operation == 'sum':
                return self.rolling_optimizer.rolling_sum(data, window, **kwargs)
            elif operation == 'quantile':
                q = kwargs.get('q', 0.5)
                return self.rolling_optimizer.rolling_quantile(data, window, q=q, **kwargs)
            elif operation == 'skew':
                return self.rolling_optimizer.rolling_skew(data, window, **kwargs)
            elif operation == 'kurt':
                return self.rolling_optimizer.rolling_kurt(data, window, **kwargs)
            elif operation == 'corr':
                other = kwargs.get('other')
                return self.rolling_optimizer.rolling_corr(data, other, window, **kwargs)
            elif operation == 'cov':
                other = kwargs.get('other')
                return self.rolling_optimizer.rolling_cov(data, other, window, **kwargs)
            else:
                # Fallback to pandas
                return self._pandas_rolling_operation(data, operation, window, **kwargs)
        except Exception as e:
            logger.warning(f"⚠️ VectorBT rolling operation failed: {e}, falling back to pandas")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.DataFrame, 
                                 operation: str, 
                                 window: int, 
                                 **kwargs) -> pd.DataFrame:
        """Perform rolling operation using pandas."""
        rolling_obj = data.rolling(window=window, **kwargs)
        
        if operation == 'mean':
            return rolling_obj.mean()
        elif operation == 'std':
            return rolling_obj.std()
        elif operation == 'var':
            return rolling_obj.var()
        elif operation == 'min':
            return rolling_obj.min()
        elif operation == 'max':
            return rolling_obj.max()
        elif operation == 'sum':
            return rolling_obj.sum()
        elif operation == 'quantile':
            q = kwargs.get('q', 0.5)
            return rolling_obj.quantile(q)
        elif operation == 'skew':
            return rolling_obj.skew()
        elif operation == 'kurt':
            return rolling_obj.kurt()
        elif operation == 'corr':
            other = kwargs.get('other')
            return rolling_obj.corr(other)
        elif operation == 'cov':
            other = kwargs.get('other')
            return rolling_obj.cov(other)
        else:
            raise ValueError(f"Unsupported rolling operation: {operation}")
    
    def vectorized_distance_calculation(self, X: np.ndarray, 
                                       metric: str = 'euclidean') -> np.ndarray:
        """Calculate vectorized distances using optimized methods."""
        start_time = time.time()
        
        if self.config.enable_distance_optimization:
            # Use VectorBT optimization for distance calculations
            result = self._vectorbt_distance_calculation(X, metric)
        else:
            # Use standard scipy distance calculation
            result = self._scipy_distance_calculation(X, metric)
        
        # Update stats
        self.performance_stats['vectorized_operations'] += 1
        self.performance_stats['distance_calculations'] += 1
        self.performance_stats['processing_time'] += time.time() - start_time
        
        return result
    
    def _vectorbt_distance_calculation(self, X: np.ndarray, 
                                     metric: str) -> np.ndarray:
        """Calculate distances using VectorBT optimization."""
        try:
            # Use VectorBT for distance calculations
            if hasattr(self.vectorization_manager, 'calculate_distances'):
                return self.vectorization_manager.calculate_distances(X, metric)
            else:
                # Fallback to scipy
                return self._scipy_distance_calculation(X, metric)
        except Exception as e:
            logger.warning(f"⚠️ VectorBT distance calculation failed: {e}, falling back to scipy")
            return self._scipy_distance_calculation(X, metric)
    
    def _scipy_distance_calculation(self, X: np.ndarray, 
                                   metric: str) -> np.ndarray:
        """Calculate distances using scipy."""
        from scipy.spatial.distance import pdist, squareform
        
        # Calculate pairwise distances
        distances = pdist(X, metric=metric)
        
        # Convert to square matrix
        distance_matrix = squareform(distances)
        
        return distance_matrix
    
    def vectorized_mathematical_operations(self, data: pd.DataFrame, 
                                          operations: List[str]) -> pd.DataFrame:
        """Perform vectorized mathematical operations."""
        start_time = time.time()
        
        result_df = data.copy()
        
        for operation in operations:
            if operation == 'log':
                result_df = self._vectorized_log(result_df)
            elif operation == 'sqrt':
                result_df = self._vectorized_sqrt(result_df)
            elif operation == 'square':
                result_df = self._vectorized_square(result_df)
            elif operation == 'abs':
                result_df = self._vectorized_abs(result_df)
            elif operation == 'exp':
                result_df = self._vectorized_exp(result_df)
            elif operation == 'sin':
                result_df = self._vectorized_sin(result_df)
            elif operation == 'cos':
                result_df = self._vectorized_cos(result_df)
            elif operation == 'tan':
                result_df = self._vectorized_tan(result_df)
            else:
                logger.warning(f"⚠️ Unsupported mathematical operation: {operation}")
        
        # Update stats
        self.performance_stats['vectorized_operations'] += 1
        self.performance_stats['processing_time'] += time.time() - start_time
        
        return result_df
    
    def _vectorized_log(self, data: pd.DataFrame) -> pd.DataFrame:
        """Vectorized logarithm operation."""
        return np.log(np.maximum(data, 1e-10))
    
    def _vectorized_sqrt(self, data: pd.DataFrame) -> pd.DataFrame:
        """Vectorized square root operation."""
        return np.sqrt(np.maximum(data, 0))
    
    def _vectorized_square(self, data: pd.DataFrame) -> pd.DataFrame:
        """Vectorized square operation."""
        return data ** 2
    
    def _vectorized_abs(self, data: pd.DataFrame) -> pd.DataFrame:
        """Vectorized absolute value operation."""
        return np.abs(data)
    
    def _vectorized_exp(self, data: pd.DataFrame) -> pd.DataFrame:
        """Vectorized exponential operation."""
        return np.exp(data)
    
    def _vectorized_sin(self, data: pd.DataFrame) -> pd.DataFrame:
        """Vectorized sine operation."""
        return np.sin(data)
    
    def _vectorized_cos(self, data: pd.DataFrame) -> pd.DataFrame:
        """Vectorized cosine operation."""
        return np.cos(data)
    
    def _vectorized_tan(self, data: pd.DataFrame) -> pd.DataFrame:
        """Vectorized tangent operation."""
        return np.tan(data)
    
    def optimized_hdbscan_clustering(self, features_df: pd.DataFrame, 
                                   **hdbscan_params) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform optimized HDBSCAN clustering."""
        start_time = time.time()
        
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not available")
        
        # Use VectorBT optimization for distance calculations
        if self.config.enable_clustering_optimization:
            # Precompute distances using VectorBT
            distance_matrix = self.vectorized_distance_calculation(
                features_df.values, 
                metric=hdbscan_params.get('metric', 'euclidean')
            )
            
            # Use precomputed distances
            hdbscan_params['metric'] = 'precomputed'
            clusterer = hdbscan.HDBSCAN(**hdbscan_params)
            cluster_labels = clusterer.fit_predict(distance_matrix)
        else:
            # Use standard HDBSCAN
            clusterer = hdbscan.HDBSCAN(**hdbscan_params)
            cluster_labels = clusterer.fit_predict(features_df)
        
        # Create clustering info
        clustering_info = {
            'clusterer': clusterer,
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'n_noise_points': list(cluster_labels).count(-1),
            'cluster_persistence': getattr(clusterer, 'cluster_persistence_', None),
            'condensed_tree': getattr(clusterer, 'condensed_tree_', None)
        }
        
        # Update stats
        self.performance_stats['vectorized_operations'] += 1
        self.performance_stats['clustering_operations'] += 1
        self.performance_stats['processing_time'] += time.time() - start_time
        
        return cluster_labels, clustering_info
    
    def vectorized_feature_engineering(self, data: pd.DataFrame, 
                                      feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Perform vectorized feature engineering."""
        start_time = time.time()
        
        features_df = data.copy()
        
        for config in feature_configs:
            feature_type = config.get('type', 'rolling')
            feature_name = config.get('name', 'feature')
            params = config.get('params', {})
            
            if feature_type == 'rolling':
                operation = params.get('operation', 'mean')
                window = params.get('window', 20)
                column = params.get('column', 'close')
                
                if column in features_df.columns:
                    rolling_result = self.vectorized_rolling_operation(
                        features_df[[column]], operation, window
                    )
                    features_df[f'{feature_name}_{operation}_{window}'] = rolling_result[column]
            
            elif feature_type == 'mathematical':
                operations = params.get('operations', ['log', 'sqrt'])
                column = params.get('column', 'close')
                
                if column in features_df.columns:
                    math_result = self.vectorized_mathematical_operations(
                        features_df[[column]], operations
                    )
                    for op in operations:
                        features_df[f'{feature_name}_{op}'] = math_result[column]
            
            elif feature_type == 'distance':
                metric = params.get('metric', 'euclidean')
                window = params.get('window', 20)
                
                # Calculate rolling distances
                rolling_distances = self.vectorized_rolling_operation(
                    features_df, 'corr', window
                )
                features_df[f'{feature_name}_distance_{window}'] = rolling_distances.mean(axis=1)
        
        # Update stats
        self.performance_stats['vectorized_operations'] += 1
        self.performance_stats['processing_time'] += time.time() - start_time
        
        return features_df
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add vectorization manager stats
        vectorization_stats = self.vectorization_manager.get_performance_stats()
        stats['vectorization_stats'] = vectorization_stats
        
        # Add rolling optimizer stats
        if self.rolling_optimizer:
            rolling_stats = self.rolling_optimizer.get_performance_stats()
            stats['rolling_optimizer_stats'] = rolling_stats
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'vectorized_operations': 0,
            'rolling_operations': 0,
            'distance_calculations': 0,
            'clustering_operations': 0,
            'vectorbt_usage_rate': 0.0,
            'gpu_usage_rate': 0.0,
            'memory_optimizations': 0,
            'processing_time': 0.0
        }
        
        # Reset vectorization manager stats
        self.vectorization_manager.reset_stats()
        
        # Reset rolling optimizer stats
        if self.rolling_optimizer:
            self.rolling_optimizer.reset_stats()

# Convenience function
def create_enhanced_vectorized_processor(
    enable_vectorbt: bool = True,
    enable_gpu: bool = False,
    enable_parallel: bool = True,
    memory_efficient: bool = True,
    max_memory_gb: float = 8.0,
    chunk_size: int = 1000
) -> EnhancedVectorizedProcessor:
    """
    Create an enhanced vectorized processor with specified configuration.
    
    Args:
        enable_vectorbt: Enable VectorBT optimization
        enable_gpu: Enable GPU acceleration
        enable_parallel: Enable parallel processing
        memory_efficient: Enable memory optimization
        max_memory_gb: Maximum memory usage in GB
        chunk_size: Chunk size for processing
        
    Returns:
        EnhancedVectorizedProcessor instance
    """
    config = VectorizedProcessingConfig(
        enable_vectorbt=enable_vectorbt,
        enable_gpu=enable_gpu,
        enable_parallel=enable_parallel,
        memory_efficient=memory_efficient,
        max_memory_gb=max_memory_gb,
        chunk_size=chunk_size
    )
    
    return EnhancedVectorizedProcessor(config)
