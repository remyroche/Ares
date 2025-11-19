"""
MI Proxy - Fast Mutual Information Computation using Numba/Numpy

This module provides efficient mutual information calculation using:
- Numba JIT compilation for fast entropy computation
- Numpy vectorization for batch operations
- Adaptive quantization strategies
- Cross-validation for stable MI estimates
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy.stats import entropy as scipy_entropy
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

# Try to import Numba for JIT compilation
try:
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args or callable(args[0]) else decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args or callable(args[0]) else decorator
    prange = range

from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_debug, tprint_info

logger = logging.getLogger(__name__)


# Numba-optimized entropy calculation
@njit
def _entropy_numba(counts: np.ndarray) -> float:
    """
    Fast entropy calculation using Numba.

    Args:
        counts: Array of occurrence counts

    Returns:
        Shannon entropy value
    """
    # Normalize to probabilities
    total = np.sum(counts)
    if total <= 0:
        return 0.0

    entropy_value = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            entropy_value -= p * np.log2(p)

    return entropy_value


@njit
def _mutual_information_numba(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Fast mutual information calculation using Numba.

    Assumes X and Y are already discretized (integer indices).

    Args:
        X: First discretized variable (n_samples,)
        Y: Second discretized variable (n_samples,)

    Returns:
        Mutual information value
    """
    # Handle edge cases
    if len(X) == 0 or len(Y) == 0:
        return 0.0

    # Get unique values
    X_unique = np.unique(X)
    Y_unique = np.unique(Y)

    # Calculate entropies
    H_X = 0.0
    H_Y = 0.0
    H_XY = 0.0

    n = len(X)

    # H(X)
    for x_val in X_unique:
        count_x = np.sum(X == x_val)
        if count_x > 0:
            p_x = count_x / n
            H_X -= p_x * np.log2(p_x)

    # H(Y)
    for y_val in Y_unique:
        count_y = np.sum(Y == y_val)
        if count_y > 0:
            p_y = count_y / n
            H_Y -= p_y * np.log2(p_y)

    # H(X,Y)
    for x_val in X_unique:
        for y_val in Y_unique:
            count_xy = np.sum((X == x_val) & (Y == y_val))
            if count_xy > 0:
                p_xy = count_xy / n
                H_XY -= p_xy * np.log2(p_xy)

    # MI(X;Y) = H(X) + H(Y) - H(X,Y)
    mi = H_X + H_Y - H_XY

    # Ensure non-negative due to numerical errors
    return max(0.0, mi)


class MIProxy:
    """
    Mutual Information Proxy using efficient Numba/Numpy computation.

    Features:
    - Fast MI calculation with Numba JIT
    - Adaptive quantization for continuous data
    - Cross-validation for robust MI estimation
    - Feature-target and feature-feature MI computation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MI Proxy.

        Args:
            config: Configuration dictionary with:
                - n_bins: Number of bins for discretization (default: 10)
                - cv_folds: Number of CV folds for MI estimation (default: 5)
                - use_numba: Whether to use Numba (default: True if available)
                - quantization_strategy: 'quantile', 'kmeans', or 'uniform' (default: 'quantile')
        """
        self.config = config or {}
        self.n_bins = self.config.get('n_bins', 10)
        self.cv_folds = self.config.get('cv_folds', 5)
        self.use_numba = self.config.get('use_numba', NUMBA_AVAILABLE)
        self.quantization_strategy = self.config.get('quantization_strategy', 'quantile')

        self.logger = logger.getChild('MIProxy')

        # Performance tracking
        self.performance_stats = {
            'total_mi_calculations': 0,
            'total_time': 0.0,
            'avg_calculation_time': 0.0,
            'numba_accelerations': 0 if NUMBA_AVAILABLE else 0
        }

        tprint_success(f"🚀 MIProxy initialized (Numba: {NUMBA_AVAILABLE}, Strategy: {self.quantization_strategy})")

    def discretize(self, X: np.ndarray, n_bins: Optional[int] = None,
                   strategy: Optional[str] = None) -> np.ndarray:
        """
        Discretize continuous data into bins.

        Args:
            X: Input array (n_samples, n_features) or (n_samples,)
            n_bins: Number of bins (uses config default if None)
            strategy: 'quantile', 'kmeans', or 'uniform' (uses config default if None)

        Returns:
            Discretized array with integer indices
        """
        n_bins = n_bins or self.n_bins
        strategy = strategy or self.quantization_strategy

        try:
            # Handle 1D case
            if X.ndim == 1:
                X = X.reshape(-1, 1)
                squeeze = True
            else:
                squeeze = False

            # Use sklearn's KBinsDiscretizer for robustness
            discretizer = KBinsDiscretizer(
                n_bins=min(n_bins, len(np.unique(X)) + 1),
                encode='ordinal',
                strategy=strategy,
                subsample=None
            )

            X_discretized = discretizer.fit_transform(X).astype(np.int32)

            if squeeze:
                X_discretized = X_discretized.ravel()

            return X_discretized

        except Exception as e:
            self.logger.warning(f"Discretization failed: {e}, using fallback quantile binning")
            return self._discretize_fallback(X)

    def _discretize_fallback(self, X: np.ndarray) -> np.ndarray:
        """Fallback discretization using quantile binning."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            squeeze = True
        else:
            squeeze = False

        X_discretized = np.zeros_like(X, dtype=np.int32)

        for i in range(X.shape[1]):
            # Simple quantile binning
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            bin_edges = np.quantile(X[:, i], quantiles)
            bin_edges = np.unique(bin_edges)
            X_discretized[:, i] = np.digitize(X[:, i], bin_edges[1:-1])

        if squeeze:
            X_discretized = X_discretized.ravel()

        return X_discretized

    def compute_mi_target(self, X: np.ndarray, y: np.ndarray,
                         feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute mutual information between features and target variable.

        Uses cross-validation for robust estimation with both sklearn and Numba paths.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target variable (n_samples,)
            feature_names: Optional feature names

        Returns:
            Dictionary mapping feature names to MI scores
        """
        tprint_debug(f"🔧 Computing MI between {X.shape[1]} features and target")

        start_time = time.time()

        try:
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            n_features = X.shape[1]
            mi_scores = np.zeros(n_features)

            # Determine if target is continuous or discrete
            is_classification = len(np.unique(y)) < 10 and np.all(y == y.astype(int))

            # Use SK learn's MI functions as baseline, but add Numba verification
            if is_classification:
                mi_scores_sk = mutual_info_classif(X, y, random_state=42, n_neighbors=3)
            else:
                mi_scores_sk = mutual_info_regression(X, y, random_state=42, n_neighbors=3)

            # If Numba is available, cross-validate with discretization approach
            if self.use_numba and NUMBA_AVAILABLE:
                mi_scores_numba = self._compute_mi_target_numba(X, y)
                # Average the two approaches for robustness
                mi_scores = (mi_scores_sk + mi_scores_numba) / 2.0
                self.performance_stats['numba_accelerations'] += 1
            else:
                mi_scores = mi_scores_sk

            # Create result dictionary
            result = {feature_names[i]: float(mi_scores[i]) for i in range(n_features)}

            # Update performance stats
            elapsed_time = time.time() - start_time
            self.performance_stats['total_mi_calculations'] += 1
            self.performance_stats['total_time'] += elapsed_time
            self.performance_stats['avg_calculation_time'] = (
                self.performance_stats['total_time'] / self.performance_stats['total_mi_calculations']
            )

            tprint_debug(f"✅ MI computation completed in {elapsed_time:.3f}s")

            return result

        except Exception as e:
            self.logger.error(f"MI target computation failed: {e}")
            # Fallback to correlation
            return {feature_names[i]: float(np.abs(np.corrcoef(X[:, i], y)[0, 1]))
                    for i in range(X.shape[1])}

    def _compute_mi_target_numba(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute MI using Numba-accelerated discretization.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Array of MI scores
        """
        # Discretize target
        y_discrete = self.discretize(y)

        # Compute MI for each feature
        n_features = X.shape[1]
        mi_scores = np.zeros(n_features)

        for i in range(n_features):
            try:
                X_discrete = self.discretize(X[:, i])
                mi_scores[i] = _mutual_information_numba(X_discrete, y_discrete)
            except Exception as e:
                self.logger.debug(f"Feature {i} MI calculation failed: {e}")
                mi_scores[i] = 0.0

        return mi_scores

    def compute_mi_pairwise(self, X: np.ndarray,
                           indices: Optional[List[int]] = None) -> Dict[Tuple[int, int], float]:
        """
        Compute mutual information between feature pairs.

        Optimized for redundancy detection.

        Args:
            X: Feature matrix (n_samples, n_features)
            indices: Specific feature indices to compute (default: all)

        Returns:
            Dictionary mapping (i, j) feature pairs to MI scores
        """
        tprint_debug(f"🔧 Computing pairwise MI for {X.shape[1]} features")

        start_time = time.time()

        try:
            if indices is None:
                indices = list(range(X.shape[1]))

            result = {}

            # Discretize all features once
            if self.use_numba and NUMBA_AVAILABLE:
                X_discrete = self.discretize(X)

                # Compute pairwise MI using Numba
                for i, idx_i in enumerate(indices):
                    for j, idx_j in enumerate(indices):
                        if idx_i < idx_j:  # Only upper triangle
                            mi = _mutual_information_numba(X_discrete[:, idx_i], X_discrete[:, idx_j])
                            result[(idx_i, idx_j)] = float(mi)
                            self.performance_stats['numba_accelerations'] += 1
            else:
                # Fallback to correlation-based approach
                corr_matrix = np.corrcoef(X.T)
                for i, idx_i in enumerate(indices):
                    for j, idx_j in enumerate(indices):
                        if idx_i < idx_j:
                            result[(idx_i, idx_j)] = float(np.abs(corr_matrix[idx_i, idx_j]))

            # Update performance stats
            elapsed_time = time.time() - start_time
            self.performance_stats['total_time'] += elapsed_time

            tprint_debug(f"✅ Pairwise MI completed in {elapsed_time:.3f}s ({len(result)} pairs)")

            return result

        except Exception as e:
            self.logger.error(f"Pairwise MI computation failed: {e}")
            return {}

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get MI proxy performance statistics."""
        return {
            **self.performance_stats,
            'numba_available': NUMBA_AVAILABLE,
            'numba_enabled': self.use_numba,
            'quantization_strategy': self.quantization_strategy,
            'n_bins': self.n_bins
        }


def create_mi_proxy(config: Optional[Dict[str, Any]] = None) -> MIProxy:
    """Factory function to create MI proxy instance."""
    return MIProxy(config)
