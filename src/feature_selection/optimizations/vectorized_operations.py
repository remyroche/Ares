"""
Vectorized Operations for Feature Selection

This module provides highly optimized vectorized operations for feature selection
using NumPy and hardware acceleration where available.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from scipy.stats import spearmanr, pearsonr

# Import hardware optimization tools
from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, HardwareConfig
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug

logger = logging.getLogger(__name__)

@dataclass
class VectorizationConfig:
    """Configuration for vectorized operations."""
    # Vectorization settings
    enable_vectorization: bool = True
    enable_hardware_acceleration: bool = True
    use_optimized_algorithms: bool = True

    # Memory optimization
    chunk_size: int = 10000
    memory_limit_mb: int = 2048

    # Algorithm-specific settings
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01
    mutual_info_k: int = 5

    # Performance monitoring
    enable_timing: bool = True
    log_performance: bool = True

    # CPU core thresholds (for compatibility)
    parallel_cpu_cores_threshold: int = 4
    vectorbt_parallel_cpu_cores_threshold: int = 2

    # GPU configuration (for compatibility)
    gpu_data_size_threshold: int = 1000000
    enable_gpu_fallback: bool = True

class VectorizedFeatureSelector:
    """Feature selector with vectorized operations and hardware optimization."""

    def __init__(self, config: Optional[VectorizationConfig] = None):
        """Initialize vectorized feature selector."""
        self.config = config or VectorizationConfig()
        self.logger = logger.getChild('VectorizedFeatureSelector')

        # Initialize hardware tools
        if self.config.enable_hardware_acceleration:
            self.cpu_optimizer = M1CPUOptimizer()
            hw_config = HardwareConfig(
                cpu_optimization_level='aggressive',
                enable_adaptive_optimization=True
            )
            self.hardware_manager = UnifiedHardwareManager(hw_config)
        else:
            self.cpu_optimizer = None
            self.hardware_manager = None

        # Performance tracking
        self.performance_stats = {
            'vectorized_operations': 0,
            'total_time': 0.0,
            'speedup_vs_naive': 0.0
        }

        tprint_success("🚀 VectorizedFeatureSelector initialized")

    def _time_operation(self, operation_name: str, func: callable, *args, **kwargs) -> Any:
        """Time an operation and log performance."""
        if not self.config.enable_timing:
            return func(*args, **kwargs)

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        execution_time = end_time - start_time
        self.performance_stats['total_time'] += execution_time

        if self.config.log_performance:
            tprint_performance(f"⏱️ {operation_name}: {execution_time:.3f}s")

        return result

    def vectorized_correlation_filter(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """Vectorized correlation-based feature filtering."""
        threshold = threshold or self.config.correlation_threshold

        def _correlation_filter():
            # Compute correlation matrix efficiently
            corr_matrix = np.corrcoef(X.T)

            # Vectorized high correlation detection
            high_corr_mask = np.abs(corr_matrix) > threshold
            np.fill_diagonal(high_corr_mask, False)  # Exclude diagonal

            # Find features to remove (vectorized)
            to_remove = np.any(high_corr_mask, axis=1)

            return ~to_remove  # Return features to keep

        result = self._time_operation("Vectorized Correlation Filter", _correlation_filter)
        self.performance_stats['vectorized_operations'] += 1

        return result

    def vectorized_variance_filter(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """Vectorized variance-based feature filtering."""
        threshold = threshold or self.config.variance_threshold

        def _variance_filter():
            # Vectorized variance computation
            variances = np.var(X, axis=0)

            # Vectorized threshold comparison
            return variances > threshold

        result = self._time_operation("Vectorized Variance Filter", _variance_filter)
        self.performance_stats['vectorized_operations'] += 1

        return result

    def vectorized_mutual_information(self, X: np.ndarray, y: np.ndarray, k: int = None) -> np.ndarray:
        """Vectorized mutual information computation."""
        k = k or self.config.mutual_info_k

        def _mutual_info():
            from sklearn.feature_selection import mutual_info_regression

            # Compute mutual information for all features
            mi_scores = mutual_info_regression(X, y, random_state=42)

            # Vectorized top-k selection
            top_k_indices = np.argsort(mi_scores)[-k:]

            # Create boolean mask
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[top_k_indices] = True

            return mask

        result = self._time_operation("Vectorized Mutual Information", _mutual_info)
        self.performance_stats['vectorized_operations'] += 1

        return result

    def vectorized_stability_selection(self, X: np.ndarray, y: np.ndarray,
                                     n_bootstrap: int = 75) -> np.ndarray:
        """Vectorized stability selection with optimized operations."""
        def _stability_selection():
            n_samples, n_features = X.shape
            stability_scores = np.zeros(n_features)

            # Pre-allocate arrays for efficiency
            bootstrap_indices = np.zeros(n_samples, dtype=int)

            for bootstrap_iter in range(n_bootstrap):
                # Vectorized bootstrap sampling
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)

                # Get bootstrap sample
                X_bootstrap = X[bootstrap_indices]
                y_bootstrap = y[bootstrap_indices]

                # Compute feature importance (vectorized)
                importance = self._compute_feature_importance_vectorized(X_bootstrap, y_bootstrap)

                # Vectorized feature selection
                n_selected = max(1, int(0.7 * n_features))  # Select 70% of features
                selected_indices = np.argsort(importance)[-n_selected:]

                # Update stability scores (vectorized)
                stability_scores[selected_indices] += 1

            # Normalize stability scores
            stability_scores = stability_scores / n_bootstrap

            return stability_scores

        result = self._time_operation("Vectorized Stability Selection", _stability_selection)
        self.performance_stats['vectorized_operations'] += 1

        return result

    def _compute_feature_importance_vectorized(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorized feature importance computation."""
        try:
            from sklearn.ensemble import RandomForestRegressor

            # Use Random Forest for feature importance
            rf = RandomForestRegressor(
                n_estimators=50,  # Reduced for speed
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X, y)

            return rf.feature_importances_

        except Exception as e:
            self.logger.warning(f"RF importance failed: {e}")
            # Fallback to mutual information
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(X, y, random_state=42)
            return mi_scores / np.sum(mi_scores)  # Normalize

    def vectorized_feature_selection(self, X: np.ndarray, y: np.ndarray,
                                   method: str = 'comprehensive', **kwargs) -> Dict[str, Any]:
        """Perform vectorized feature selection."""
        tprint_info(f"🚀 Starting vectorized {method} selection")

        start_time = time.time()

        try:
            # Apply vectorized filters
            filters_applied = []
            selected_mask = np.ones(X.shape[1], dtype=bool)

            # Variance filter
            if method in ['comprehensive', 'filter']:
                variance_mask = self.vectorized_variance_filter(X)
                selected_mask &= variance_mask
                filters_applied.append('variance')
                tprint_debug(f"📊 Variance filter: {np.sum(variance_mask)}/{X.shape[1]} features")

            # Correlation filter
            if method in ['comprehensive', 'filter']:
                correlation_mask = self.vectorized_correlation_filter(X)
                selected_mask &= correlation_mask
                filters_applied.append('correlation')
                tprint_debug(f"📊 Correlation filter: {np.sum(correlation_mask)}/{X.shape[1]} features")

            # Mutual information filter
            if method in ['comprehensive', 'filter']:
                mi_mask = self.vectorized_mutual_information(X, y)
                selected_mask &= mi_mask
                filters_applied.append('mutual_info')
                tprint_debug(f"📊 MI filter: {np.sum(mi_mask)}/{X.shape[1]} features")

            # Stability selection
            if method in ['comprehensive', 'stability']:
                stability_scores = self.vectorized_stability_selection(X, y)
                stability_threshold = kwargs.get('stability_threshold', 0.6)
                stability_mask = stability_scores >= stability_threshold
                selected_mask &= stability_mask
                filters_applied.append('stability')
                tprint_debug(f"📊 Stability filter: {np.sum(stability_mask)}/{X.shape[1]} features")

            # Get selected features
            selected_indices = np.where(selected_mask)[0]
            selected_features = [f"feature_{i}" for i in selected_indices]

            # Calculate feature scores
            feature_scores = {}
            if len(selected_indices) > 0:
                # Use mutual information as base scores
                mi_scores = self._compute_feature_importance_vectorized(X, y)
                for i, idx in enumerate(selected_indices):
                    feature_scores[f"feature_{idx}"] = float(mi_scores[idx])

            end_time = time.time()
            execution_time = end_time - start_time

            result = {
                'success': True,
                'selected_features': selected_features,
                'selected_indices': selected_indices.tolist(),
                'feature_scores': feature_scores,
                'n_selected': len(selected_features),
                'n_total': X.shape[1],
                'filters_applied': filters_applied,
                'execution_time': execution_time,
                'method': f'vectorized_{method}'
            }

            tprint_success(f"✅ Vectorized selection completed: {len(selected_features)}/{X.shape[1]} features "
                         f"in {execution_time:.3f}s")

            return result

        except Exception as e:
            self.logger.error(f"Vectorized selection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()

        if stats['vectorized_operations'] > 0:
            stats['avg_time_per_operation'] = stats['total_time'] / stats['vectorized_operations']
        else:
            stats['avg_time_per_operation'] = 0.0

        tprint_performance(f"📊 Vectorization Stats: {stats['vectorized_operations']} operations, "
                         f"{stats['avg_time_per_operation']:.3f}s avg")

        return stats

class OptimizedCorrelationFilter:
    """Optimized correlation-based feature filtering."""

    def __init__(self, threshold: float = 0.95):
        """Initialize correlation filter."""
        self.threshold = threshold
        self.logger = logger.getChild('OptimizedCorrelationFilter')

    def filter_features(self, X: np.ndarray) -> np.ndarray:
        """Filter features based on correlation."""
        return VectorizedFeatureSelector().vectorized_correlation_filter(X, self.threshold)

class OptimizedVarianceFilter:
    """Optimized variance-based feature filtering."""

    def __init__(self, threshold: float = 0.01):
        """Initialize variance filter."""
        self.threshold = threshold
        self.logger = logger.getChild('OptimizedVarianceFilter')

    def filter_features(self, X: np.ndarray) -> np.ndarray:
        """Filter features based on variance."""
        return VectorizedFeatureSelector().vectorized_variance_filter(X, self.threshold)

def create_vectorized_selector(config: Optional[VectorizationConfig] = None) -> VectorizedFeatureSelector:
    """Create a vectorized feature selector."""
    return VectorizedFeatureSelector(config)
