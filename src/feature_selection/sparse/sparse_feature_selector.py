"""
Sparse Matrix Feature Selection

This module provides optimized feature selection operations for sparse matrices
with memory-efficient algorithms and hardware optimization.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix, issparse, spmatrix
from scipy.sparse.linalg import svds
from sklearn.feature_selection import mutual_info_regression, f_regression

# Import hardware optimization tools
# Lazy imports to avoid circular imports
def get_optimization_level():
    """Lazy import of OptimizationLevel to avoid circular imports."""
    try:
        from src.utils.hardware.constants import OptimizationLevel
        return OptimizationLevel
    except ImportError:
        from enum import Enum
        class OptimizationLevel(Enum):
            MINIMAL = "minimal"
            BALANCED = "balanced"
            AGGRESSIVE = "aggressive"
            MAXIMUM = "maximum"
        return OptimizationLevel

def get_workload_type():
    """Lazy import of WorkloadType to avoid circular imports."""
    try:
        from src.utils.hardware.constants import WorkloadType
        return WorkloadType
    except ImportError:
        from enum import Enum
        class WorkloadType(Enum):
            MATRIX_OPERATIONS = "matrix_operations"
            BACKTESTING = "backtesting"
            MONTE_CARLO = "monte_carlo"
            ML_TRAINING = "ml_training"
            DATA_PROCESSING = "data_processing"
            FEATURE_ENGINEERING = "feature_engineering"
            GENERAL = "general"
        return WorkloadType

# Use lazy imports
OptimizationLevel = get_optimization_level()
WorkloadType = get_workload_type()

# Lazy import of other hardware utilities
def get_hardware_manager():
    """Lazy import of hardware manager to avoid circular imports."""
    try:
        from src.utils.hardware import get_integrated_hardware_manager
        return get_integrated_hardware_manager()
    except ImportError:
        return None

def memory_efficient(*args, **kwargs):
    """Lazy import of memory_efficient decorator."""
    try:
        from src.utils.hardware import memory_efficient as _memory_efficient
        return _memory_efficient(*args, **kwargs)
    except ImportError:
        def decorator(func):
            return func
        return decorator

def performance_tracked(*args, **kwargs):
    """Lazy import of performance_tracked decorator."""
    try:
        from src.utils.hardware import performance_tracked as _performance_tracked
        return _performance_tracked(*args, **kwargs)
    except ImportError:
        def decorator(func):
            return func
        return decorator

def smart_cache(*args, **kwargs):
    """Lazy import of smart_cache decorator."""
    try:
        from src.utils.hardware import smart_cache as _smart_cache
        return _smart_cache(*args, **kwargs)
    except ImportError:
        def decorator(func):
            return func
        return decorator
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug

logger = logging.getLogger(__name__)

@dataclass
class SparseConfig:
    """Configuration for sparse matrix operations."""
    # Sparse matrix settings
    sparse_threshold: float = 0.1  # Use sparse if >10% zeros
    max_density: float = 0.5  # Convert to dense if density > 50%

    # Memory optimization
    memory_limit_gb: float = 8.0
    enable_memory_monitoring: bool = True
    enable_compression: bool = True

    # Algorithm settings
    enable_sparse_algorithms: bool = True
    use_approximate_methods: bool = True
    approximation_rank: int = 100

    # Performance monitoring
    enable_timing: bool = True
    log_performance: bool = True

class SparseFeatureSelector:
    """Feature selector optimized for sparse matrices."""

    def __init__(self, config: Optional[SparseConfig] = None):
        """Initialize sparse feature selector."""
        self.config = config or SparseConfig()
        self.logger = logger.getChild('SparseFeatureSelector')

        # Initialize hardware tools
        if self.config.enable_memory_monitoring:
            self.hardware_manager = get_integrated_hardware_manager()
        else:
            self.hardware_manager = None

        # Performance tracking
        self.performance_stats = {
            'sparse_operations': 0,
            'dense_conversions': 0,
            'total_time': 0.0,
            'memory_saved_mb': 0.0
        }

        tprint_success("📊 SparseFeatureSelector initialized")

    def _is_sparse_beneficial(self, X: Union[np.ndarray, spmatrix]) -> bool:
        """Determine if sparse representation is beneficial."""
        if issparse(X):
            return True

        # Check sparsity
        zero_ratio = np.count_nonzero(X == 0) / X.size
        return zero_ratio > self.config.sparse_threshold

    def _should_convert_to_dense(self, X: spmatrix) -> bool:
        """Determine if sparse matrix should be converted to dense."""
        density = X.nnz / X.size
        return density > self.config.max_density

    def _ensure_sparse(self, X: Union[np.ndarray, spmatrix]) -> spmatrix:
        """Ensure matrix is in sparse format if beneficial."""
        if issparse(X):
            return X

        if self._is_sparse_beneficial(X):
            return csr_matrix(X)
        else:
            return X

    def _sparse_variance_filter(self, X: spmatrix, threshold: float = 0.01) -> np.ndarray:
        """Variance filter for sparse matrices."""
        def _compute_variance():
            # Efficient sparse variance computation
            mean = np.array(X.mean(axis=0)).flatten()
            X_centered = X - mean
            variance = np.array((X_centered.multiply(X_centered)).mean(axis=0)).flatten()
            return variance > threshold

        return self._time_operation("Sparse Variance Filter", _compute_variance)

    def _sparse_correlation_filter(self, X: spmatrix, threshold: float = 0.95) -> np.ndarray:
        """Correlation filter for sparse matrices."""
        def _compute_correlation():
            # For sparse matrices, we need to be more careful about correlation
            # Convert to dense for correlation computation (unavoidable)
            if X.shape[1] > 1000:  # Large number of features
                # Use sampling for large sparse matrices
                sample_size = min(1000, X.shape[1])
                sample_indices = np.random.choice(X.shape[1], size=sample_size, replace=False)
                X_sample = X[:, sample_indices].toarray()
                corr_matrix = np.corrcoef(X_sample.T)
            else:
                # Convert to dense for correlation
                X_dense = X.toarray()
                corr_matrix = np.corrcoef(X_dense.T)

            # Find highly correlated features
            high_corr_mask = np.abs(corr_matrix) > threshold
            np.fill_diagonal(high_corr_mask, False)
            to_remove = np.any(high_corr_mask, axis=1)

            return ~to_remove

        return self._time_operation("Sparse Correlation Filter", _compute_correlation)

    def _sparse_mutual_information(self, X: spmatrix, y: np.ndarray, k: int = 5) -> np.ndarray:
        """Mutual information for sparse matrices."""
        def _compute_mi():
            # Convert to dense for MI computation (sklearn requirement)
            X_dense = X.toarray()
            mi_scores = mutual_info_regression(X_dense, y, random_state=42)

            # Select top k features
            top_k_indices = np.argsort(mi_scores)[-k:]
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[top_k_indices] = True

            return mask

        return self._time_operation("Sparse Mutual Information", _compute_mi)

    def _sparse_f_score_filter(self, X: spmatrix, y: np.ndarray, k: int = 5) -> np.ndarray:
        """F-score filter for sparse matrices."""
        def _compute_f_score():
            # Convert to dense for F-score computation
            X_dense = X.toarray()
            f_scores, _ = f_regression(X_dense, y)

            # Select top k features
            top_k_indices = np.argsort(f_scores)[-k:]
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[top_k_indices] = True

            return mask

        return self._time_operation("Sparse F-Score Filter", _compute_f_score)

    def _sparse_svd_selection(self, X: spmatrix, n_components: int = 50) -> np.ndarray:
        """SVD-based feature selection for sparse matrices."""
        def _compute_svd():
            # Use sparse SVD for large matrices
            if X.shape[1] > 1000:
                # Use approximate SVD
                U, s, Vt = svds(X, k=min(n_components, X.shape[1]-1))
                # Select features with highest singular values
                feature_importance = np.sum(np.abs(Vt), axis=0)
            else:
                # Use full SVD
                U, s, Vt = np.linalg.svd(X.toarray(), full_matrices=False)
                feature_importance = np.sum(np.abs(Vt), axis=0)

            # Select top features
            top_indices = np.argsort(feature_importance)[-n_components:]
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[top_indices] = True

            return mask

        return self._time_operation("Sparse SVD Selection", _compute_svd)

    def _time_operation(self, operation_name: str, func: callable) -> Any:
        """Time an operation and log performance."""
        if not self.config.enable_timing:
            return func()

        start_time = time.time()
        result = func()
        end_time = time.time()

        execution_time = end_time - start_time
        self.performance_stats['total_time'] += execution_time

        if self.config.log_performance:
            tprint_performance(f"⏱️ {operation_name}: {execution_time:.3f}s")

        return result

    @memory_efficient(memory_threshold_mb=400.0, auto_cleanup=True)
    @performance_tracked(log_performance=True, track_memory=True)
    def select_features_sparse(self, X: Union[np.ndarray, spmatrix], y: np.ndarray,
                              method: str = 'comprehensive', **kwargs) -> Dict[str, Any]:
        """Select features using sparse matrix operations."""
        tprint_info(f"📊 Starting sparse {method} selection: {X.shape}")

        start_time = time.time()

        try:
            # Ensure sparse format if beneficial
            X_sparse = self._ensure_sparse(X)

            # Track conversions
            if not issparse(X) and issparse(X_sparse):
                self.performance_stats['sparse_operations'] += 1
                tprint_debug("📊 Converted to sparse matrix")
            elif issparse(X) and not issparse(X_sparse):
                self.performance_stats['dense_conversions'] += 1
                tprint_debug("📊 Converted to dense matrix")

            # Check if we should convert to dense
            if issparse(X_sparse) and self._should_convert_to_dense(X_sparse):
                X_sparse = X_sparse.toarray()
                self.performance_stats['dense_conversions'] += 1
                tprint_debug("📊 Converted to dense (high density)")

            # Apply sparse-aware filters
            filters_applied = []
            selected_mask = np.ones(X_sparse.shape[1], dtype=bool)

            # Variance filter
            if method in ['comprehensive', 'filter']:
                variance_mask = self._sparse_variance_filter(X_sparse, kwargs.get('variance_threshold', 0.01))
                selected_mask &= variance_mask
                filters_applied.append('variance')
                tprint_debug(f"📊 Variance filter: {np.sum(variance_mask)}/{X_sparse.shape[1]} features")

            # Correlation filter (only for dense matrices)
            if method in ['comprehensive', 'filter'] and not issparse(X_sparse):
                correlation_mask = self._sparse_correlation_filter(X_sparse, kwargs.get('correlation_threshold', 0.95))
                selected_mask &= correlation_mask
                filters_applied.append('correlation')
                tprint_debug(f"📊 Correlation filter: {np.sum(correlation_mask)}/{X_sparse.shape[1]} features")

            # Mutual information filter
            if method in ['comprehensive', 'filter']:
                mi_mask = self._sparse_mutual_information(X_sparse, y, kwargs.get('k', 5))
                selected_mask &= mi_mask
                filters_applied.append('mutual_info')
                tprint_debug(f"📊 MI filter: {np.sum(mi_mask)}/{X_sparse.shape[1]} features")

            # F-score filter
            if method in ['comprehensive', 'filter']:
                f_score_mask = self._sparse_f_score_filter(X_sparse, y, kwargs.get('k', 5))
                selected_mask &= f_score_mask
                filters_applied.append('f_score')
                tprint_debug(f"📊 F-score filter: {np.sum(f_score_mask)}/{X_sparse.shape[1]} features")

            # SVD-based selection
            if method in ['comprehensive', 'dimensionality']:
                n_components = kwargs.get('n_components', 50)
                svd_mask = self._sparse_svd_selection(X_sparse, n_components)
                selected_mask &= svd_mask
                filters_applied.append('svd')
                tprint_debug(f"📊 SVD filter: {np.sum(svd_mask)}/{X_sparse.shape[1]} features")

            # Get selected features
            selected_indices = np.where(selected_mask)[0]
            selected_features = [f"feature_{i}" for i in selected_indices]

            # Calculate feature scores
            feature_scores = {}
            if len(selected_indices) > 0:
                # Use mutual information as base scores
                if issparse(X_sparse):
                    X_dense = X_sparse.toarray()
                else:
                    X_dense = X_sparse

                mi_scores = mutual_info_regression(X_dense, y, random_state=42)
                for i, idx in enumerate(selected_indices):
                    feature_scores[f"feature_{idx}"] = float(mi_scores[idx])

            end_time = time.time()
            execution_time = end_time - start_time

            # Calculate memory savings
            if issparse(X_sparse):
                memory_saved = (X_sparse.data.nbytes + X_sparse.indices.nbytes + X_sparse.indptr.nbytes) - (X_sparse.toarray().nbytes)
                self.performance_stats['memory_saved_mb'] += memory_saved / (1024 * 1024)

            result = {
                'success': True,
                'selected_features': selected_features,
                'selected_indices': selected_indices.tolist(),
                'feature_scores': feature_scores,
                'n_selected': len(selected_features),
                'n_total': X_sparse.shape[1],
                'filters_applied': filters_applied,
                'execution_time': execution_time,
                'method': f'sparse_{method}',
                'sparse_used': issparse(X_sparse),
                'memory_saved_mb': self.performance_stats['memory_saved_mb']
            }

            tprint_success(f"✅ Sparse selection completed: {len(selected_features)}/{X_sparse.shape[1]} features "
                         f"in {execution_time:.3f}s")

            return result

        except Exception as e:
            self.logger.error(f"Sparse selection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()

        if stats['sparse_operations'] > 0:
            stats['avg_time_per_operation'] = stats['total_time'] / stats['sparse_operations']
        else:
            stats['avg_time_per_operation'] = 0.0

        tprint_performance(f"📊 Sparse Stats: {stats['sparse_operations']} operations, "
                         f"{stats['memory_saved_mb']:.1f}MB saved")

        return stats

class SparseMatrixProcessor:
    """Processor for sparse matrix operations."""

    def __init__(self, config: Optional[SparseConfig] = None):
        """Initialize sparse matrix processor."""
        self.config = config or SparseConfig()
        self.selector = SparseFeatureSelector(self.config)

    def process_sparse_data(self, X: Union[np.ndarray, spmatrix], y: np.ndarray,
                          method: str = 'comprehensive', **kwargs) -> Dict[str, Any]:
        """Process sparse data with feature selection."""
        return self.selector.select_features_sparse(X, y, method, **kwargs)

def create_sparse_selector(config: Optional[SparseConfig] = None) -> SparseFeatureSelector:
    """Create a sparse feature selector."""
    return SparseFeatureSelector(config)
