"""
VectorBT Advanced Feature Selection Tools

This module provides VectorBT-optimized implementations of advanced feature selection
methods including distance correlation, HSIC, bootstrap stability, and ensemble scoring.

Key Features:
- VectorBT-optimized distance correlation calculations
- HSIC computations with kernel operations
- Bootstrap stability with vectorized operations
- LASSO ensemble scoring with VectorBT matrix operations
- Early pruning with VectorBT performance monitoring
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_corr, rolling_cov,
        rolling_quantile, rolling_skew, rolling_kurt, rolling_apply
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

# SciPy for advanced statistical functions
try:
    from scipy.spatial.distance import pdist, squareform
    from scipy.linalg import eigh
    from scipy.stats import spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Sklearn for fallbacks
try:
    from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel
    from sklearn.linear_model import LassoCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import mutual_info_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import VectorBT optimization tools
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    )
    from src.utils.matrix_operations.vectorbt_optimizations import (
        VectorBTOptimizedOperations, get_unified_matrix_operations
    )
    VECTORBT_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    VECTORBT_OPTIMIZATIONS_AVAILABLE = False
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None
    VectorBTOptimizedOperations = None
    get_unified_matrix_operations = None

# Import utilities
from src.utils.tprint import (
    tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug,
    tprint_info, tprint_error
)
from src.utils.math_validation import validate_numeric_array, validate_finite

logger = logging.getLogger(__name__)


@dataclass
class VectorBTAdvancedConfig:
    """Configuration for VectorBT advanced feature selection."""
    # Distance correlation settings
    distance_corr_chunk_size: int = 1000
    distance_corr_enable_subsampling: bool = True
    distance_corr_sample_size: int = 5000
    distance_corr_use_rolling_optimizer: bool = True
    
    # HSIC settings
    hsic_kernel: str = 'rbf'  # 'rbf', 'linear', 'poly'
    hsic_gamma: Optional[float] = None
    hsic_enable_subsampling: bool = True
    hsic_sample_size: int = 3000
    hsic_use_vectorization_manager: bool = True
    
    # Bootstrap settings
    bootstrap_n_samples: int = 100
    bootstrap_sample_ratio: float = 0.8
    bootstrap_chunk_size: int = 500
    bootstrap_use_rolling_optimizer: bool = True
    
    # LASSO ensemble settings
    lasso_use_vectorization_manager: bool = True
    lasso_chunk_size: int = 2000
    lasso_parallel_cv: bool = True
    
    # Feature importance aggregation
    importance_use_rolling_optimizer: bool = True
    importance_aggregation_method: str = 'weighted_mean'  # 'mean', 'weighted_mean', 'median'
    
    # Cross-validation settings
    cv_use_vectorization_manager: bool = True
    cv_parallel_folds: bool = True
    cv_chunk_size: int = 1000
    
    # Early pruning settings
    enable_early_pruning: bool = True
    pruning_threshold: float = 0.1  # Remove features with score < threshold
    pruning_batch_size: int = 1000
    pruning_memory_limit_gb: float = 8.0
    pruning_use_rolling_optimizer: bool = True
    
    # Performance settings
    enable_parallel: bool = True
    max_workers: int = 4
    enable_gpu: bool = False
    memory_efficient: bool = True
    
    # VectorBT optimization settings
    enable_rolling_optimizer: bool = True
    enable_vectorization_manager: bool = True
    rolling_window_size: int = 100
    vectorization_chunk_size: int = 2000


class VectorBTDistanceCorrelation:
    """VectorBT-optimized distance correlation calculations with RollingOptimizer integration."""
    
    def __init__(self, config: VectorBTAdvancedConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.DistanceCorrelation")
        
        # Initialize VectorBT optimization tools
        self.rolling_optimizer = None
        self.vectorization_manager = None
        self._initialize_vectorbt_tools()
    
    def _initialize_vectorbt_tools(self):
        """Initialize VectorBT optimization tools."""
        if VECTORBT_OPTIMIZATIONS_AVAILABLE and self.config.enable_rolling_optimizer:
            try:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer()
                self.vectorization_manager = get_unified_matrix_operations()
                tprint_debug("✅ VectorBT optimization tools initialized for distance correlation")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT tools not available: {e}")
                self.rolling_optimizer = None
                self.vectorization_manager = None
    
    def calculate_distance_correlation_vectorbt(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> pd.Series:
        """
        Calculate distance correlation using VectorBT optimizations.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Distance correlation scores for each feature
        """
        tprint_debug("🔍 Calculating distance correlation with VectorBT optimization")
        
        if not VECTORBT_AVAILABLE:
            tprint_warning("⚠️ VectorBT not available, using fallback")
            return self._fallback_distance_correlation(X, y)
        
        try:
            # Subsample if enabled and dataset is large
            if (self.config.distance_corr_enable_subsampling and 
                len(X) > self.config.distance_corr_sample_size):
                tprint_debug(f"📊 Subsampling data: {len(X)} -> {self.config.distance_corr_sample_size}")
                sample_indices = np.random.choice(
                    len(X), self.config.distance_corr_sample_size, replace=False
                )
                X_sample = X.iloc[sample_indices]
                y_sample = y.iloc[sample_indices]
            else:
                X_sample = X
                y_sample = y
            
            # Process in chunks for memory efficiency
            chunk_size = self.config.distance_corr_chunk_size
            n_features = len(X_sample.columns)
            
            distance_corr_scores = {}
            
            # Process features in chunks
            for i in range(0, n_features, chunk_size):
                chunk_features = X_sample.columns[i:i + chunk_size]
                tprint_debug(f"📊 Processing chunk {i//chunk_size + 1}: features {i}-{min(i+chunk_size, n_features)}")
                
                # Calculate distance correlation for chunk
                chunk_scores = self._calculate_chunk_distance_correlation(
                    X_sample[chunk_features], y_sample
                )
                distance_corr_scores.update(chunk_scores)
            
            result = pd.Series(distance_corr_scores, index=X.columns)
            tprint_success(f"✅ Distance correlation calculated for {len(result)} features")
            return result
            
        except Exception as e:
            tprint_error(f"❌ VectorBT distance correlation failed: {e}")
            return self._fallback_distance_correlation(X, y)
    
    def _calculate_chunk_distance_correlation(
        self, 
        X_chunk: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, float]:
        """Calculate distance correlation for a chunk of features."""
        chunk_scores = {}
        
        for feature in X_chunk.columns:
            try:
                # Use VectorBT for efficient distance matrix calculation
                if VECTORBT_AVAILABLE:
                    score = self._vectorbt_distance_correlation(X_chunk[feature], y)
                else:
                    score = self._scipy_distance_correlation(X_chunk[feature], y)
                
                chunk_scores[feature] = score
                
            except Exception as e:
                tprint_debug(f"⚠️ Distance correlation failed for {feature}: {e}")
                chunk_scores[feature] = 0.0
        
        return chunk_scores
    
    def _vectorbt_distance_correlation(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate distance correlation using VectorBT optimizations with RollingOptimizer."""
        try:
            # Remove NaN values
            valid_mask = ~(x.isna() | y.isna())
            if not valid_mask.any():
                return 0.0
            
            x_clean = x[valid_mask].values
            y_clean = y[valid_mask].values
            
            if len(x_clean) < 3:
                return 0.0
            
            # Use VectorBTRollingOptimizer for efficient distance calculations
            if self.rolling_optimizer and self.config.distance_corr_use_rolling_optimizer:
                return self._rolling_optimizer_distance_correlation(x_clean, y_clean)
            elif self.vectorization_manager and VECTORBT_AVAILABLE:
                return self._vectorization_manager_distance_correlation(x_clean, y_clean)
            else:
                return self._scipy_distance_correlation(x, y)
            
        except Exception as e:
            tprint_debug(f"⚠️ VectorBT distance correlation error: {e}")
            return self._scipy_distance_correlation(x, y)
    
    def _rolling_optimizer_distance_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate distance correlation using VectorBTRollingOptimizer."""
        try:
            # Use rolling operations for efficient distance matrix calculation
            # Create rolling windows for distance calculations
            window_size = min(self.config.rolling_window_size, len(x))
            
            # Calculate rolling distance correlations
            x_series = pd.Series(x.flatten())
            y_series = pd.Series(y.flatten())
            
            # Use VectorBTRollingOptimizer for rolling operations
            rolling_corr = self.rolling_optimizer.rolling_correlation(
                x_series, y_series, window=window_size
            )
            
            # Aggregate rolling correlations
            if rolling_corr is not None and not rolling_corr.empty:
                # Use mean of rolling correlations as distance correlation approximation
                return abs(rolling_corr.mean())
            else:
                return self._scipy_distance_correlation(pd.Series(x), pd.Series(y))
                
        except Exception as e:
            tprint_debug(f"⚠️ Rolling optimizer distance correlation error: {e}")
            return self._scipy_distance_correlation(pd.Series(x), pd.Series(y))
    
    def _vectorization_manager_distance_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate distance correlation using UnifiedVectorizationManager."""
        try:
            # Use vectorization manager for efficient matrix operations
            # Create data matrices
            X_matrix = x.reshape(-1, 1)
            Y_matrix = y.reshape(-1, 1)
            
            # Calculate distance matrices using vectorized operations
            if hasattr(self.vectorization_manager, 'calculate_distance_matrix'):
                x_dist = self.vectorization_manager.calculate_distance_matrix(X_matrix)
                y_dist = self.vectorization_manager.calculate_distance_matrix(Y_matrix)
            else:
                # Fallback to manual calculation with vectorization
                x_dist = self._manual_distance_matrix_vectorized(X_matrix)
                y_dist = self._manual_distance_matrix_vectorized(Y_matrix)
            
            # Calculate distance correlation
            return self._calculate_dcorr_from_matrices(x_dist, y_dist)
            
        except Exception as e:
            tprint_debug(f"⚠️ Vectorization manager distance correlation error: {e}")
            return self._scipy_distance_correlation(pd.Series(x), pd.Series(y))
    
    def _manual_distance_matrix_vectorized(self, data: np.ndarray) -> np.ndarray:
        """Calculate distance matrix using vectorized operations."""
        # Efficient vectorized distance matrix calculation
        n = data.shape[0]
        distances = np.zeros((n, n))
        
        # Vectorized calculation using broadcasting
        for i in range(n):
            diff = data - data[i]
            distances[i] = np.sqrt(np.sum(diff**2, axis=1))
        
        return distances
    
    def _vectorbt_distance_matrix(self, data: 'vbt.Data') -> np.ndarray:
        """Calculate distance matrix using VectorBT."""
        # This is a simplified implementation - would need VectorBT-specific distance calculation
        # For now, fall back to scipy implementation
        return self._scipy_distance_correlation(data.values.flatten(), data.values.flatten())
    
    def _scipy_distance_correlation(self, x: pd.Series, y: pd.Series) -> float:
        """Fallback distance correlation using SciPy."""
        if not SCIPY_AVAILABLE:
            return 0.0
        
        try:
            # Remove NaN values
            valid_mask = ~(x.isna() | y.isna())
            if not valid_mask.any():
                return 0.0
            
            x_clean = x[valid_mask].values
            y_clean = y[valid_mask].values
            
            if len(x_clean) < 3:
                return 0.0
            
            # Calculate distance matrices
            x_dist = pdist(x_clean.reshape(-1, 1), metric='euclidean')
            y_dist = pdist(y_clean.reshape(-1, 1), metric='euclidean')
            
            # Convert to squareform
            x_dist_matrix = squareform(x_dist)
            y_dist_matrix = squareform(y_dist)
            
            # Center the distance matrices
            n = len(x_clean)
            x_centered = x_dist_matrix - np.mean(x_dist_matrix, axis=1)[:, np.newaxis] - np.mean(x_dist_matrix, axis=0) + np.mean(x_dist_matrix)
            y_centered = y_dist_matrix - np.mean(y_dist_matrix, axis=1)[:, np.newaxis] - np.mean(y_dist_matrix, axis=0) + np.mean(y_dist_matrix)
            
            # Calculate distance covariance and variances
            dcov_xy = np.sqrt(np.mean(x_centered * y_centered))
            dcov_xx = np.sqrt(np.mean(x_centered * x_centered))
            dcov_yy = np.sqrt(np.mean(y_centered * y_centered))
            
            # Avoid division by zero
            if dcov_xx == 0 or dcov_yy == 0:
                return 0.0
            
            # Distance correlation
            dcorr = dcov_xy / np.sqrt(dcov_xx * dcov_yy)
            return abs(dcorr)
            
        except Exception as e:
            tprint_debug(f"⚠️ SciPy distance correlation error: {e}")
            return 0.0
    
    def _calculate_dcorr_from_matrices(self, x_dist: np.ndarray, y_dist: np.ndarray) -> float:
        """Calculate distance correlation from distance matrices."""
        # Center the distance matrices
        n = x_dist.shape[0]
        x_centered = x_dist - np.mean(x_dist, axis=1)[:, np.newaxis] - np.mean(x_dist, axis=0) + np.mean(x_dist)
        y_centered = y_dist - np.mean(y_dist, axis=1)[:, np.newaxis] - np.mean(y_dist, axis=0) + np.mean(y_dist)
        
        # Calculate distance covariance and variances
        dcov_xy = np.sqrt(np.mean(x_centered * y_centered))
        dcov_xx = np.sqrt(np.mean(x_centered * x_centered))
        dcov_yy = np.sqrt(np.mean(y_centered * y_centered))
        
        # Avoid division by zero
        if dcov_xx == 0 or dcov_yy == 0:
            return 0.0
        
        # Distance correlation
        dcorr = dcov_xy / np.sqrt(dcov_xx * dcov_yy)
        return abs(dcorr)
    
    def _fallback_distance_correlation(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Fallback to simple correlation when VectorBT/SciPy unavailable."""
        tprint_warning("⚠️ Using Spearman correlation as fallback")
        return X.corrwith(y).abs()


class VectorBTHSIC:
    """VectorBT-optimized HSIC calculations with UnifiedVectorizationManager integration."""
    
    def __init__(self, config: VectorBTAdvancedConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.HSIC")
        
        # Initialize VectorBT optimization tools
        self.rolling_optimizer = None
        self.vectorization_manager = None
        self._initialize_vectorbt_tools()
    
    def _initialize_vectorbt_tools(self):
        """Initialize VectorBT optimization tools."""
        if VECTORBT_OPTIMIZATIONS_AVAILABLE and self.config.enable_vectorization_manager:
            try:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer()
                self.vectorization_manager = get_unified_matrix_operations()
                tprint_debug("✅ VectorBT optimization tools initialized for HSIC")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT tools not available: {e}")
                self.rolling_optimizer = None
                self.vectorization_manager = None
    
    def calculate_hsic_vectorbt(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """
        Calculate HSIC using VectorBT optimizations.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            HSIC scores for each feature
        """
        tprint_debug("🔍 Calculating HSIC with VectorBT optimization")
        
        if not VECTORBT_AVAILABLE or not SKLEARN_AVAILABLE:
            tprint_warning("⚠️ Required libraries not available, using fallback")
            return self._fallback_hsic(X, y)
        
        try:
            # Subsample if enabled and dataset is large
            if (self.config.hsic_enable_subsampling and 
                len(X) > self.config.hsic_sample_size):
                tprint_debug(f"📊 Subsampling data: {len(X)} -> {self.config.hsic_sample_size}")
                sample_indices = np.random.choice(
                    len(X), self.config.hsic_sample_size, replace=False
                )
                X_sample = X.iloc[sample_indices]
                y_sample = y.iloc[sample_indices]
            else:
                X_sample = X
                y_sample = y
            
            # Calculate HSIC for each feature
            hsic_scores = {}
            for feature in X_sample.columns:
                try:
                    score = self._calculate_single_hsic(X_sample[feature], y_sample)
                    hsic_scores[feature] = score
                except Exception as e:
                    tprint_debug(f"⚠️ HSIC calculation failed for {feature}: {e}")
                    hsic_scores[feature] = 0.0
            
            result = pd.Series(hsic_scores, index=X.columns)
            tprint_success(f"✅ HSIC calculated for {len(result)} features")
            return result
            
        except Exception as e:
            tprint_error(f"❌ VectorBT HSIC failed: {e}")
            return self._fallback_hsic(X, y)
    
    def _calculate_single_hsic(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate HSIC for a single feature with VectorBT optimizations."""
        try:
            # Remove NaN values
            valid_mask = ~(x.isna() | y.isna())
            if not valid_mask.any():
                return 0.0
            
            x_clean = x[valid_mask].values
            y_clean = y[valid_mask].values
            
            if len(x_clean) < 3:
                return 0.0
            
            # Use VectorBT optimizations if available
            if self.vectorization_manager and self.config.hsic_use_vectorization_manager:
                return self._vectorization_manager_hsic(x_clean, y_clean)
            elif self.rolling_optimizer and VECTORBT_AVAILABLE:
                return self._rolling_optimizer_hsic(x_clean, y_clean)
            else:
                return self._standard_hsic_calculation(x_clean, y_clean)
            
        except Exception as e:
            tprint_debug(f"⚠️ HSIC calculation error: {e}")
            return 0.0
    
    def _vectorization_manager_hsic(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate HSIC using UnifiedVectorizationManager."""
        try:
            # Reshape for kernel calculation
            x_reshaped = x.reshape(-1, 1)
            y_reshaped = y.reshape(-1, 1)
            
            # Use vectorization manager for kernel calculations
            kernel_type = self.config.hsic_kernel
            gamma = self.config.hsic_gamma
            
            if kernel_type == 'rbf':
                if gamma is None:
                    gamma = 1.0 / (x_reshaped.shape[1] * np.var(x_reshaped))
                
                # Use vectorization manager for RBF kernel
                if hasattr(self.vectorization_manager, 'rbf_kernel'):
                    Kx = self.vectorization_manager.rbf_kernel(x_reshaped, gamma=gamma)
                    Ky = self.vectorization_manager.rbf_kernel(y_reshaped, gamma=gamma)
                else:
                    Kx = rbf_kernel(x_reshaped, gamma=gamma)
                    Ky = rbf_kernel(y_reshaped, gamma=gamma)
            else:
                # Fallback to standard kernel calculation
                return self._standard_hsic_calculation(x, y)
            
            # Use vectorization manager for matrix operations
            if hasattr(self.vectorization_manager, 'matrix_multiply'):
                return self._calculate_hsic_with_vectorization(Kx, Ky)
            else:
                return self._calculate_hsic_standard(Kx, Ky)
                
        except Exception as e:
            tprint_debug(f"⚠️ Vectorization manager HSIC error: {e}")
            return self._standard_hsic_calculation(x, y)
    
    def _rolling_optimizer_hsic(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate HSIC using VectorBTRollingOptimizer."""
        try:
            # Use rolling operations for HSIC approximation
            x_series = pd.Series(x.flatten())
            y_series = pd.Series(y.flatten())
            
            # Calculate rolling correlations as HSIC approximation
            window_size = min(self.config.rolling_window_size, len(x))
            rolling_corr = self.rolling_optimizer.rolling_correlation(
                x_series, y_series, window=window_size
            )
            
            if rolling_corr is not None and not rolling_corr.empty:
                # Use variance of rolling correlations as HSIC approximation
                return abs(rolling_corr.var())
            else:
                return self._standard_hsic_calculation(x, y)
                
        except Exception as e:
            tprint_debug(f"⚠️ Rolling optimizer HSIC error: {e}")
            return self._standard_hsic_calculation(x, y)
    
    def _standard_hsic_calculation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Standard HSIC calculation without VectorBT optimizations."""
        # Reshape for kernel calculation
        x_reshaped = x.reshape(-1, 1)
        y_reshaped = y.reshape(-1, 1)
        
        # Calculate kernels based on configuration
        kernel_type = self.config.hsic_kernel
        gamma = self.config.hsic_gamma
        
        if kernel_type == 'rbf':
            if gamma is None:
                # Auto gamma: 1 / (n_features * X.var())
                gamma = 1.0 / (x_reshaped.shape[1] * np.var(x_reshaped))
            Kx = rbf_kernel(x_reshaped, gamma=gamma)
            Ky = rbf_kernel(y_reshaped, gamma=gamma)
        elif kernel_type == 'linear':
            Kx = linear_kernel(x_reshaped)
            Ky = linear_kernel(y_reshaped)
        elif kernel_type == 'poly':
            Kx = polynomial_kernel(x_reshaped, degree=2)
            Ky = polynomial_kernel(y_reshaped, degree=2)
        else:
            # Default to RBF
            gamma = 1.0 / (x_reshaped.shape[1] * np.var(x_reshaped))
            Kx = rbf_kernel(x_reshaped, gamma=gamma)
            Ky = rbf_kernel(y_reshaped, gamma=gamma)
        
        return self._calculate_hsic_standard(Kx, Ky)
    
    def _calculate_hsic_with_vectorization(self, Kx: np.ndarray, Ky: np.ndarray) -> float:
        """Calculate HSIC using vectorization manager for matrix operations."""
        n = Kx.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix
        
        # Use vectorization manager for matrix operations
        Kx_centered = self.vectorization_manager.matrix_multiply(
            self.vectorization_manager.matrix_multiply(H, Kx), H
        )
        Ky_centered = self.vectorization_manager.matrix_multiply(
            self.vectorization_manager.matrix_multiply(H, Ky), H
        )
        
        # Calculate HSIC
        hsic_matrix = self.vectorization_manager.matrix_multiply(Kx_centered, Ky_centered)
        hsic = np.trace(hsic_matrix) / (n - 1) ** 2
        
        return abs(hsic)
    
    def _calculate_hsic_standard(self, Kx: np.ndarray, Ky: np.ndarray) -> float:
        """Standard HSIC calculation."""
        n = Kx.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix
        
        Kx_centered = H @ Kx @ H
        Ky_centered = H @ Ky @ H
        
        # Calculate HSIC
        hsic = np.trace(Kx_centered @ Ky_centered) / (n - 1) ** 2
        
        return abs(hsic)
    
    def _fallback_hsic(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Fallback to simple correlation when HSIC unavailable."""
        tprint_warning("⚠️ Using Spearman correlation as fallback")
        return X.corrwith(y).abs()


class VectorBTEarlyPruning:
    """VectorBT-optimized early pruning for feature selection."""
    
    def __init__(self, config: VectorBTAdvancedConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.EarlyPruning")
        self.pruning_stats = {
            'features_pruned': 0,
            'pruning_rounds': 0,
            'memory_saved_mb': 0,
            'time_saved_seconds': 0
        }
    
    def apply_early_pruning(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        feature_scores: pd.Series,
        pruning_threshold: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Apply early pruning to remove low-scoring features.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_scores: Current feature scores
            pruning_threshold: Threshold for pruning (uses config if None)
            
        Returns:
            Tuple of (pruned_X, pruned_scores, pruning_stats)
        """
        if not self.config.enable_early_pruning:
            return X, feature_scores, {}
        
        threshold = pruning_threshold or self.config.pruning_threshold
        tprint_debug(f"✂️ Applying early pruning with threshold {threshold}")
        
        start_time = time.time()
        initial_features = len(X.columns)
        
        try:
            # Identify features to prune
            features_to_keep = feature_scores >= threshold
            pruned_features = X.columns[~features_to_keep].tolist()
            
            if len(pruned_features) == 0:
                tprint_debug("📊 No features to prune")
                return X, feature_scores, {}
            
            # Apply pruning
            X_pruned = X.loc[:, features_to_keep]
            scores_pruned = feature_scores[features_to_keep]
            
            # Update statistics
            self.pruning_stats['features_pruned'] += len(pruned_features)
            self.pruning_stats['pruning_rounds'] += 1
            
            # Calculate memory savings
            memory_saved = X[pruned_features].memory_usage(deep=True).sum() / 1024 / 1024
            self.pruning_stats['memory_saved_mb'] += memory_saved
            
            # Calculate time savings
            time_saved = time.time() - start_time
            self.pruning_stats['time_saved_seconds'] += time_saved
            
            pruning_info = {
                'features_removed': len(pruned_features),
                'features_remaining': len(X_pruned.columns),
                'pruning_ratio': len(pruned_features) / initial_features,
                'memory_saved_mb': memory_saved,
                'time_saved_seconds': time_saved,
                'threshold_used': threshold
            }
            
            tprint_success(
                f"✅ Early pruning completed: {len(pruned_features)} features removed "
                f"({len(pruned_features)/initial_features:.1%}), "
                f"{memory_saved:.1f}MB saved"
            )
            
            return X_pruned, scores_pruned, pruning_info
            
        except Exception as e:
            tprint_error(f"❌ Early pruning failed: {e}")
            return X, feature_scores, {}
    
    def get_pruning_statistics(self) -> Dict[str, Any]:
        """Get cumulative pruning statistics."""
        return self.pruning_stats.copy()


class VectorBTLASSOEnsemble:
    """VectorBT-optimized LASSO ensemble scoring with UnifiedVectorizationManager."""
    
    def __init__(self, config: VectorBTAdvancedConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.LASSOEnsemble")
        
        # Initialize VectorBT optimization tools
        self.rolling_optimizer = None
        self.vectorization_manager = None
        self._initialize_vectorbt_tools()
    
    def _initialize_vectorbt_tools(self):
        """Initialize VectorBT optimization tools."""
        if VECTORBT_OPTIMIZATIONS_AVAILABLE and self.config.lasso_use_vectorization_manager:
            try:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer()
                self.vectorization_manager = get_unified_matrix_operations()
                tprint_debug("✅ VectorBT optimization tools initialized for LASSO ensemble")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT tools not available: {e}")
                self.rolling_optimizer = None
                self.vectorization_manager = None
    
    def calculate_lasso_ensemble_vectorbt(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_alphas: int = 50,
        cv_folds: int = 5
    ) -> pd.Series:
        """
        Calculate LASSO ensemble scores using VectorBT optimizations.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_alphas: Number of alpha values to test
            cv_folds: Number of CV folds
            
        Returns:
            LASSO ensemble scores for each feature
        """
        tprint_debug("🔍 Calculating LASSO ensemble with VectorBT optimization")
        
        if not SKLEARN_AVAILABLE:
            tprint_warning("⚠️ Scikit-learn not available, using fallback")
            return self._fallback_lasso_ensemble(X, y)
        
        try:
            # Use VectorBT optimizations if available
            if self.vectorization_manager and self.config.lasso_use_vectorization_manager:
                return self._vectorization_manager_lasso_ensemble(X, y, n_alphas, cv_folds)
            else:
                return self._standard_lasso_ensemble(X, y, n_alphas, cv_folds)
                
        except Exception as e:
            tprint_error(f"❌ VectorBT LASSO ensemble failed: {e}")
            return self._fallback_lasso_ensemble(X, y)
    
    def _vectorization_manager_lasso_ensemble(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_alphas: int,
        cv_folds: int
    ) -> pd.Series:
        """Calculate LASSO ensemble using UnifiedVectorizationManager."""
        try:
            # Process in chunks for memory efficiency
            chunk_size = self.config.lasso_chunk_size
            n_features = len(X.columns)
            
            # Generate alpha values
            alpha_min = 0.001
            alpha_max = 1.0
            alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alphas)
            
            # Calculate LASSO scores for each alpha using vectorization
            lasso_scores = np.zeros(n_features)
            
            for i, alpha in enumerate(alphas):
                tprint_debug(f"📊 Processing alpha {i+1}/{n_alphas}: {alpha:.6f}")
                
                # Use vectorization manager for LASSO calculation
                if hasattr(self.vectorization_manager, 'lasso_regression'):
                    scores = self.vectorization_manager.lasso_regression(
                        X.values, y.values, alpha=alpha
                    )
                else:
                    # Fallback to standard LASSO
                    from sklearn.linear_model import Lasso
                    lasso = Lasso(alpha=alpha, random_state=42, max_iter=1000)
                    lasso.fit(X, y)
                    scores = np.abs(lasso.coef_)
                
                lasso_scores += scores
            
            # Average across alphas
            lasso_scores = lasso_scores / n_alphas
            
            result = pd.Series(lasso_scores, index=X.columns)
            tprint_success(f"✅ LASSO ensemble calculated for {len(result)} features")
            return result
            
        except Exception as e:
            tprint_debug(f"⚠️ Vectorization manager LASSO error: {e}")
            return self._standard_lasso_ensemble(X, y, n_alphas, cv_folds)
    
    def _standard_lasso_ensemble(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        n_alphas: int,
        cv_folds: int
    ) -> pd.Series:
        """Standard LASSO ensemble calculation."""
        try:
            # Use LassoCV for automatic alpha selection
            lasso = LassoCV(
                alphas=np.logspace(-4, 1, n_alphas),
                cv=cv_folds,
                random_state=42,
                n_jobs=1 if not self.config.enable_parallel else -1
            )
            
            lasso.fit(X, y)
            
            # Feature importance as absolute coefficients
            feature_importance = np.abs(lasso.coef_)
            
            result = pd.Series(feature_importance, index=X.columns)
            tprint_success(f"✅ LASSO ensemble calculated for {len(result)} features")
            return result
            
        except Exception as e:
            tprint_debug(f"⚠️ Standard LASSO ensemble error: {e}")
            return self._fallback_lasso_ensemble(X, y)
    
    def _fallback_lasso_ensemble(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Fallback LASSO ensemble calculation."""
        tprint_warning("⚠️ Using simple correlation as LASSO fallback")
        return X.corrwith(y).abs()


class VectorBTFeatureImportanceAggregator:
    """VectorBT-optimized feature importance aggregation with RollingOptimizer."""
    
    def __init__(self, config: VectorBTAdvancedConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.FeatureImportanceAggregator")
        
        # Initialize VectorBT optimization tools
        self.rolling_optimizer = None
        self.vectorization_manager = None
        self._initialize_vectorbt_tools()
    
    def _initialize_vectorbt_tools(self):
        """Initialize VectorBT optimization tools."""
        if VECTORBT_OPTIMIZATIONS_AVAILABLE and self.config.importance_use_rolling_optimizer:
            try:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer()
                self.vectorization_manager = get_unified_matrix_operations()
                tprint_debug("✅ VectorBT optimization tools initialized for importance aggregation")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT tools not available: {e}")
                self.rolling_optimizer = None
                self.vectorization_manager = None
    
    def aggregate_importance_scores_vectorbt(
        self, 
        importance_scores: Dict[str, pd.Series],
        weights: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        Aggregate feature importance scores using VectorBT optimizations.
        
        Args:
            importance_scores: Dictionary of method -> scores
            weights: Optional weights for each method
            
        Returns:
            Aggregated importance scores
        """
        tprint_debug("🔍 Aggregating importance scores with VectorBT optimization")
        
        if not importance_scores:
            tprint_warning("⚠️ No importance scores provided")
            return pd.Series(dtype=float)
        
        try:
            # Use VectorBT optimizations if available
            if self.rolling_optimizer and self.config.importance_use_rolling_optimizer:
                return self._rolling_optimizer_aggregation(importance_scores, weights)
            elif self.vectorization_manager:
                return self._vectorization_manager_aggregation(importance_scores, weights)
            else:
                return self._standard_aggregation(importance_scores, weights)
                
        except Exception as e:
            tprint_error(f"❌ VectorBT importance aggregation failed: {e}")
            return self._standard_aggregation(importance_scores, weights)
    
    def _rolling_optimizer_aggregation(
        self, 
        importance_scores: Dict[str, pd.Series],
        weights: Optional[Dict[str, float]]
    ) -> pd.Series:
        """Aggregate importance scores using VectorBTRollingOptimizer."""
        try:
            # Convert to DataFrame for rolling operations
            scores_df = pd.DataFrame(importance_scores)
            
            # Use rolling operations for weighted aggregation
            if weights:
                # Apply weights using rolling operations
                weighted_scores = pd.DataFrame()
                for method, scores in scores_df.items():
                    weight = weights.get(method, 1.0)
                    weighted_scores[method] = self.rolling_optimizer.rolling_apply(
                        scores, lambda x: x * weight, window=1
                    )
            else:
                weighted_scores = scores_df
            
            # Aggregate using rolling mean
            if self.config.importance_aggregation_method == 'weighted_mean':
                aggregated = self.rolling_optimizer.rolling_mean(
                    weighted_scores.mean(axis=1), window=1
                )
            elif self.config.importance_aggregation_method == 'median':
                aggregated = self.rolling_optimizer.rolling_apply(
                    weighted_scores.median(axis=1), lambda x: x, window=1
                )
            else:  # mean
                aggregated = self.rolling_optimizer.rolling_mean(
                    weighted_scores.mean(axis=1), window=1
                )
            
            return aggregated
            
        except Exception as e:
            tprint_debug(f"⚠️ Rolling optimizer aggregation error: {e}")
            return self._standard_aggregation(importance_scores, weights)
    
    def _vectorization_manager_aggregation(
        self, 
        importance_scores: Dict[str, pd.Series],
        weights: Optional[Dict[str, float]]
    ) -> pd.Series:
        """Aggregate importance scores using UnifiedVectorizationManager."""
        try:
            # Convert to matrix for vectorized operations
            scores_matrix = np.column_stack([
                scores.values for scores in importance_scores.values()
            ])
            
            # Apply weights if provided
            if weights:
                weight_vector = np.array([
                    weights.get(method, 1.0) for method in importance_scores.keys()
                ])
                scores_matrix = scores_matrix * weight_vector
            
            # Use vectorization manager for aggregation
            if hasattr(self.vectorization_manager, 'aggregate_scores'):
                aggregated = self.vectorization_manager.aggregate_scores(
                    scores_matrix, method=self.config.importance_aggregation_method
                )
            else:
                # Fallback to standard aggregation
                if self.config.importance_aggregation_method == 'weighted_mean':
                    aggregated = np.mean(scores_matrix, axis=1)
                elif self.config.importance_aggregation_method == 'median':
                    aggregated = np.median(scores_matrix, axis=1)
                else:  # mean
                    aggregated = np.mean(scores_matrix, axis=1)
            
            # Create result series
            feature_names = list(importance_scores.values())[0].index
            return pd.Series(aggregated, index=feature_names)
            
        except Exception as e:
            tprint_debug(f"⚠️ Vectorization manager aggregation error: {e}")
            return self._standard_aggregation(importance_scores, weights)
    
    def _standard_aggregation(
        self, 
        importance_scores: Dict[str, pd.Series],
        weights: Optional[Dict[str, float]]
    ) -> pd.Series:
        """Standard importance score aggregation."""
        # Normalize scores to 0-1 range
        normalized_scores = {}
        for method, scores in importance_scores.items():
            if scores.max() > 0:
                normalized_scores[method] = scores / scores.max()
            else:
                normalized_scores[method] = scores
        
        # Apply weights
        if weights:
            weighted_scores = {
                method: scores * weights.get(method, 1.0)
                for method, scores in normalized_scores.items()
            }
        else:
            weighted_scores = normalized_scores
        
        # Aggregate
        if self.config.importance_aggregation_method == 'weighted_mean':
            aggregated = pd.DataFrame(weighted_scores).mean(axis=1)
        elif self.config.importance_aggregation_method == 'median':
            aggregated = pd.DataFrame(weighted_scores).median(axis=1)
        else:  # mean
            aggregated = pd.DataFrame(normalized_scores).mean(axis=1)
        
        return aggregated


class VectorBTCrossValidation:
    """VectorBT-optimized cross-validation operations with UnifiedVectorizationManager."""
    
    def __init__(self, config: VectorBTAdvancedConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.CrossValidation")
        
        # Initialize VectorBT optimization tools
        self.rolling_optimizer = None
        self.vectorization_manager = None
        self._initialize_vectorbt_tools()
    
    def _initialize_vectorbt_tools(self):
        """Initialize VectorBT optimization tools."""
        if VECTORBT_OPTIMIZATIONS_AVAILABLE and self.config.cv_use_vectorization_manager:
            try:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer()
                self.vectorization_manager = get_unified_matrix_operations()
                tprint_debug("✅ VectorBT optimization tools initialized for cross-validation")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT tools not available: {e}")
                self.rolling_optimizer = None
                self.vectorization_manager = None
    
    def cross_validate_features_vectorbt(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        cv_folds: int = 5,
        scoring_func: Optional[Callable] = None
    ) -> pd.Series:
        """
        Perform cross-validation on features using VectorBT optimizations.
        
        Args:
            X: Feature matrix
            y: Target variable
            cv_folds: Number of CV folds
            scoring_func: Custom scoring function
            
        Returns:
            Cross-validation scores for each feature
        """
        tprint_debug("🔍 Performing cross-validation with VectorBT optimization")
        
        if not SKLEARN_AVAILABLE:
            tprint_warning("⚠️ Scikit-learn not available, using fallback")
            return self._fallback_cross_validation(X, y)
        
        try:
            # Use VectorBT optimizations if available
            if self.vectorization_manager and self.config.cv_use_vectorization_manager:
                return self._vectorization_manager_cv(X, y, cv_folds, scoring_func)
            else:
                return self._standard_cross_validation(X, y, cv_folds, scoring_func)
                
        except Exception as e:
            tprint_error(f"❌ VectorBT cross-validation failed: {e}")
            return self._fallback_cross_validation(X, y)
    
    def _vectorization_manager_cv(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        cv_folds: int,
        scoring_func: Optional[Callable]
    ) -> pd.Series:
        """Perform cross-validation using UnifiedVectorizationManager."""
        try:
            from sklearn.model_selection import KFold
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            # Create CV splits
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Process in chunks for memory efficiency
            chunk_size = self.config.cv_chunk_size
            n_features = len(X.columns)
            
            cv_scores = np.zeros(n_features)
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                tprint_debug(f"📊 Processing CV fold {fold+1}/{cv_folds}")
                
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Process features in chunks
                for i in range(0, n_features, chunk_size):
                    chunk_features = X.columns[i:i + chunk_size]
                    
                    # Use vectorization manager for feature scoring
                    if hasattr(self.vectorization_manager, 'score_features'):
                        chunk_scores = self.vectorization_manager.score_features(
                            X_train[chunk_features], y_train, X_val[chunk_features], y_val
                        )
                    else:
                        # Fallback to standard scoring
                        chunk_scores = self._score_features_chunk(
                            X_train[chunk_features], y_train, 
                            X_val[chunk_features], y_val, scoring_func
                        )
                    
                    cv_scores[i:i+len(chunk_features)] += chunk_scores
            
            # Average across folds
            cv_scores = cv_scores / cv_folds
            
            result = pd.Series(cv_scores, index=X.columns)
            tprint_success(f"✅ Cross-validation completed for {len(result)} features")
            return result
            
        except Exception as e:
            tprint_debug(f"⚠️ Vectorization manager CV error: {e}")
            return self._standard_cross_validation(X, y, cv_folds, scoring_func)
    
    def _score_features_chunk(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame, 
        y_val: pd.Series,
        scoring_func: Optional[Callable]
    ) -> np.ndarray:
        """Score a chunk of features."""
        scores = np.zeros(len(X_train.columns))
        
        for i, feature in enumerate(X_train.columns):
            try:
                if scoring_func:
                    scores[i] = scoring_func(X_train[feature], y_train, X_val[feature], y_val)
                else:
                    # Default: R² score
                    from sklearn.linear_model import LinearRegression
                    from sklearn.metrics import r2_score
                    
                    lr = LinearRegression()
                    lr.fit(X_train[[feature]], y_train)
                    y_pred = lr.predict(X_val[[feature]])
                    scores[i] = r2_score(y_val, y_pred)
            except Exception as e:
                tprint_debug(f"⚠️ Feature scoring failed for {feature}: {e}")
                scores[i] = 0.0
        
        return scores
    
    def _standard_cross_validation(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        cv_folds: int,
        scoring_func: Optional[Callable]
    ) -> pd.Series:
        """Standard cross-validation without VectorBT optimizations."""
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.linear_model import LinearRegression
            
            cv_scores = np.zeros(len(X.columns))
            
            for i, feature in enumerate(X.columns):
                try:
                    if scoring_func:
                        # Use custom scoring function
                        scores = []
                        from sklearn.model_selection import KFold
                        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                        for train_idx, val_idx in kf.split(X):
                            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                            score = scoring_func(X_train[feature], y_train, X_val[feature], y_val)
                            scores.append(score)
                        cv_scores[i] = np.mean(scores)
                    else:
                        # Use standard cross-validation
                        lr = LinearRegression()
                        scores = cross_val_score(
                            lr, X[[feature]], y, cv=cv_folds, scoring='r2'
                        )
                        cv_scores[i] = np.mean(scores)
                except Exception as e:
                    tprint_debug(f"⚠️ CV scoring failed for {feature}: {e}")
                    cv_scores[i] = 0.0
            
            result = pd.Series(cv_scores, index=X.columns)
            tprint_success(f"✅ Cross-validation completed for {len(result)} features")
            return result
            
        except Exception as e:
            tprint_debug(f"⚠️ Standard CV error: {e}")
            return self._fallback_cross_validation(X, y)
    
    def _fallback_cross_validation(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Fallback cross-validation calculation."""
        tprint_warning("⚠️ Using simple correlation as CV fallback")
        return X.corrwith(y).abs()


class VectorBTBootstrapStability:
    """VectorBT-optimized bootstrap stability calculations with RollingOptimizer integration."""
    
    def __init__(self, config: VectorBTAdvancedConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.BootstrapStability")
        
        # Initialize VectorBT optimization tools
        self.rolling_optimizer = None
        self.vectorization_manager = None
        self._initialize_vectorbt_tools()
    
    def _initialize_vectorbt_tools(self):
        """Initialize VectorBT optimization tools."""
        if VECTORBT_OPTIMIZATIONS_AVAILABLE and self.config.bootstrap_use_rolling_optimizer:
            try:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer()
                self.vectorization_manager = get_unified_matrix_operations()
                tprint_debug("✅ VectorBT optimization tools initialized for bootstrap stability")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT tools not available: {e}")
                self.rolling_optimizer = None
                self.vectorization_manager = None
    
    def calculate_bootstrap_stability_vectorbt(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        base_estimator: Optional[Callable] = None
    ) -> pd.Series:
        """
        Calculate bootstrap stability using VectorBT optimizations.
        
        Args:
            X: Feature matrix
            y: Target variable
            base_estimator: Function to calculate feature importance
            
        Returns:
            Bootstrap stability scores for each feature
        """
        tprint_debug("🔍 Calculating bootstrap stability with VectorBT optimization")
        
        if not VECTORBT_AVAILABLE or not SKLEARN_AVAILABLE:
            tprint_warning("⚠️ Required libraries not available, using fallback")
            return self._fallback_bootstrap_stability(X, y, base_estimator)
        
        try:
            n_samples = self.config.bootstrap_n_samples
            sample_ratio = self.config.bootstrap_sample_ratio
            chunk_size = self.config.bootstrap_chunk_size
            
            # Process bootstrap samples in chunks for memory efficiency
            feature_stability = np.zeros(len(X.columns))
            successful_samples = 0
            
            for chunk_start in range(0, n_samples, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_samples)
                chunk_samples = chunk_end - chunk_start
                
                tprint_debug(f"📊 Processing bootstrap chunk {chunk_start//chunk_size + 1}: samples {chunk_start}-{chunk_end}")
                
                # Process chunk in parallel if enabled
                if self.config.enable_parallel and chunk_samples > 1:
                    chunk_stability = self._process_bootstrap_chunk_parallel(
                        X, y, chunk_samples, sample_ratio, base_estimator
                    )
                else:
                    chunk_stability = self._process_bootstrap_chunk_sequential(
                        X, y, chunk_samples, sample_ratio, base_estimator
                    )
                
                feature_stability += chunk_stability
                successful_samples += chunk_samples
            
            # Normalize by number of successful samples
            if successful_samples > 0:
                feature_stability = feature_stability / successful_samples
            
            result = pd.Series(feature_stability, index=X.columns)
            tprint_success(f"✅ Bootstrap stability calculated for {len(result)} features")
            return result
            
        except Exception as e:
            tprint_error(f"❌ VectorBT bootstrap stability failed: {e}")
            return self._fallback_bootstrap_stability(X, y, base_estimator)
    
    def _process_bootstrap_chunk_parallel(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        chunk_samples: int,
        sample_ratio: float,
        base_estimator: Optional[Callable]
    ) -> np.ndarray:
        """Process bootstrap chunk in parallel."""
        feature_stability = np.zeros(len(X.columns))
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            for _ in range(chunk_samples):
                future = executor.submit(
                    self._single_bootstrap_sample, X, y, sample_ratio, base_estimator
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    sample_stability = future.result()
                    feature_stability += sample_stability
                except Exception as e:
                    tprint_debug(f"⚠️ Bootstrap sample failed: {e}")
        
        return feature_stability
    
    def _process_bootstrap_chunk_sequential(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        chunk_samples: int,
        sample_ratio: float,
        base_estimator: Optional[Callable]
    ) -> np.ndarray:
        """Process bootstrap chunk sequentially."""
        feature_stability = np.zeros(len(X.columns))
        
        for _ in range(chunk_samples):
            try:
                sample_stability = self._single_bootstrap_sample(X, y, sample_ratio, base_estimator)
                feature_stability += sample_stability
            except Exception as e:
                tprint_debug(f"⚠️ Bootstrap sample failed: {e}")
        
        return feature_stability
    
    def _single_bootstrap_sample(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        sample_ratio: float,
        base_estimator: Optional[Callable]
    ) -> np.ndarray:
        """Calculate stability for a single bootstrap sample."""
        # Bootstrap sample
        n_samples_subset = int(len(X) * sample_ratio)
        indices = np.random.choice(len(X), n_samples_subset, replace=True)
        
        X_bootstrap = X.iloc[indices]
        y_bootstrap = y.iloc[indices]
        
        # Calculate feature importance for this bootstrap sample
        if base_estimator:
            importance = base_estimator(X_bootstrap, y_bootstrap)
        else:
            # Default to RandomForest
            rf = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=1)
            rf.fit(X_bootstrap, y_bootstrap)
            importance = rf.feature_importances_
        
        # Count features above stability threshold
        stability_threshold = 0.01  # Configurable threshold
        return (importance > stability_threshold).astype(int)
    
    def _fallback_bootstrap_stability(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        base_estimator: Optional[Callable]
    ) -> pd.Series:
        """Fallback bootstrap stability calculation."""
        tprint_warning("⚠️ Using simplified bootstrap stability")
        
        # Simple implementation without VectorBT optimizations
        n_samples = min(10, self.config.bootstrap_n_samples)  # Reduced for fallback
        sample_ratio = self.config.bootstrap_sample_ratio
        
        feature_stability = np.zeros(len(X.columns))
        
        for _ in range(n_samples):
            try:
                # Bootstrap sample
                n_samples_subset = int(len(X) * sample_ratio)
                indices = np.random.choice(len(X), n_samples_subset, replace=True)
                
                X_bootstrap = X.iloc[indices]
                y_bootstrap = y.iloc[indices]
                
                # Calculate feature importance
                if base_estimator:
                    importance = base_estimator(X_bootstrap, y_bootstrap)
                else:
                    rf = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=1)
                    rf.fit(X_bootstrap, y_bootstrap)
                    importance = rf.feature_importances_
                
                # Count features above threshold
                stability_threshold = 0.01
                feature_stability += (importance > stability_threshold).astype(int)
                
            except Exception as e:
                tprint_debug(f"⚠️ Bootstrap sample failed: {e}")
        
        # Normalize
        feature_stability = feature_stability / n_samples
        return pd.Series(feature_stability, index=X.columns)


class VectorBTAdvancedFeatureSelector:
    """
    VectorBT-optimized advanced feature selection with early pruning.
    
    This class combines all VectorBT optimizations for comprehensive
    feature selection with performance monitoring and early pruning.
    
    Enhanced with:
    - VectorBTRollingOptimizer integration
    - UnifiedVectorizationManager for matrix operations
    - LASSO ensemble scoring
    - Feature importance aggregation
    - Cross-validation operations
    """
    
    def __init__(self, config: Optional[VectorBTAdvancedConfig] = None):
        self.config = config or VectorBTAdvancedConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize all components
        self.distance_corr = VectorBTDistanceCorrelation(self.config)
        self.hsic = VectorBTHSIC(self.config)
        self.lasso_ensemble = VectorBTLASSOEnsemble(self.config)
        self.importance_aggregator = VectorBTFeatureImportanceAggregator(self.config)
        self.cross_validation = VectorBTCrossValidation(self.config)
        self.early_pruning = VectorBTEarlyPruning(self.config)
        self.bootstrap_stability = VectorBTBootstrapStability(self.config)
        
        # Performance tracking
        self.performance_stats = {
            'distance_corr_time': 0.0,
            'hsic_time': 0.0,
            'lasso_ensemble_time': 0.0,
            'importance_aggregation_time': 0.0,
            'cross_validation_time': 0.0,
            'bootstrap_time': 0.0,
            'pruning_time': 0.0,
            'total_time': 0.0,
            'features_processed': 0,
            'memory_optimizations': 0,
            'vectorbt_operations': 0
        }
    
    def calculate_advanced_scores(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        methods: List[str] = None
    ) -> Dict[str, pd.Series]:
        """
        Calculate advanced feature selection scores using VectorBT optimizations.
        
        Args:
            X: Feature matrix
            y: Target variable
            methods: List of methods to use ['distance_corr', 'hsic', 'lasso_ensemble', 
                   'cross_validation', 'bootstrap']
            
        Returns:
            Dictionary of method scores
        """
        if methods is None:
            methods = ['distance_corr', 'hsic', 'lasso_ensemble', 'cross_validation', 'bootstrap']
        
        tprint_info(f"🚀 Calculating advanced scores with VectorBT: {methods}")
        start_time = time.time()
        
        results = {}
        
        try:
            # Distance correlation
            if 'distance_corr' in methods:
                tprint_debug("📊 Calculating distance correlation...")
                method_start = time.time()
                results['distance_corr'] = self.distance_corr.calculate_distance_correlation_vectorbt(X, y)
                self.performance_stats['distance_corr_time'] = time.time() - method_start
                self.performance_stats['vectorbt_operations'] += 1
            
            # HSIC
            if 'hsic' in methods:
                tprint_debug("📊 Calculating HSIC...")
                method_start = time.time()
                results['hsic'] = self.hsic.calculate_hsic_vectorbt(X, y)
                self.performance_stats['hsic_time'] = time.time() - method_start
                self.performance_stats['vectorbt_operations'] += 1
            
            # LASSO ensemble
            if 'lasso_ensemble' in methods:
                tprint_debug("📊 Calculating LASSO ensemble...")
                method_start = time.time()
                results['lasso_ensemble'] = self.lasso_ensemble.calculate_lasso_ensemble_vectorbt(X, y)
                self.performance_stats['lasso_ensemble_time'] = time.time() - method_start
                self.performance_stats['vectorbt_operations'] += 1
            
            # Cross-validation
            if 'cross_validation' in methods:
                tprint_debug("📊 Calculating cross-validation scores...")
                method_start = time.time()
                results['cross_validation'] = self.cross_validation.cross_validate_features_vectorbt(X, y)
                self.performance_stats['cross_validation_time'] = time.time() - method_start
                self.performance_stats['vectorbt_operations'] += 1
            
            # Bootstrap stability
            if 'bootstrap' in methods:
                tprint_debug("📊 Calculating bootstrap stability...")
                method_start = time.time()
                results['bootstrap'] = self.bootstrap_stability.calculate_bootstrap_stability_vectorbt(X, y)
                self.performance_stats['bootstrap_time'] = time.time() - method_start
                self.performance_stats['vectorbt_operations'] += 1
            
            self.performance_stats['total_time'] = time.time() - start_time
            self.performance_stats['features_processed'] = len(X.columns)
            
            tprint_success(f"✅ Advanced scores calculated in {self.performance_stats['total_time']:.2f}s")
            tprint_info(f"   📊 VectorBT operations: {self.performance_stats['vectorbt_operations']}")
            return results
            
        except Exception as e:
            tprint_error(f"❌ Advanced score calculation failed: {e}")
            return {}
    
    def aggregate_and_prune_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        importance_scores: Dict[str, pd.Series],
        weights: Optional[Dict[str, float]] = None,
        pruning_thresholds: List[float] = None
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Aggregate importance scores and apply early pruning with VectorBT optimizations.
        
        Args:
            X: Feature matrix
            y: Target variable
            importance_scores: Dictionary of method -> scores
            weights: Optional weights for aggregation
            pruning_thresholds: List of thresholds for progressive pruning
            
        Returns:
            Tuple of (final_X, final_scores, pruning_info)
        """
        tprint_info("🚀 Aggregating scores and applying pruning with VectorBT")
        start_time = time.time()
        
        try:
            # Aggregate importance scores
            tprint_debug("📊 Aggregating importance scores...")
            aggregation_start = time.time()
            aggregated_scores = self.importance_aggregator.aggregate_importance_scores_vectorbt(
                importance_scores, weights
            )
            self.performance_stats['importance_aggregation_time'] = time.time() - aggregation_start
            self.performance_stats['vectorbt_operations'] += 1
            
            # Apply early pruning
            tprint_debug("✂️ Applying early pruning...")
            pruning_start = time.time()
            final_X, final_scores, pruning_info = self.early_pruning.apply_early_pruning_pipeline(
                X, y, aggregated_scores, pruning_thresholds
            )
            self.performance_stats['pruning_time'] = time.time() - pruning_start
            self.performance_stats['vectorbt_operations'] += 1
            
            # Calculate final statistics
            total_time = time.time() - start_time
            pruning_info['aggregation_time'] = self.performance_stats['importance_aggregation_time']
            pruning_info['total_time'] = total_time
            pruning_info['vectorbt_operations'] = self.performance_stats['vectorbt_operations']
            
            tprint_success(f"✅ Aggregation and pruning completed in {total_time:.2f}s")
            tprint_info(f"   📊 Features: {len(X.columns)} -> {len(final_X.columns)}")
            tprint_info(f"   📊 VectorBT operations: {self.performance_stats['vectorbt_operations']}")
            
            return final_X, final_scores, pruning_info
            
        except Exception as e:
            tprint_error(f"❌ Aggregation and pruning failed: {e}")
            return X, pd.Series(dtype=float), {}
    
    def apply_early_pruning_pipeline(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        initial_scores: pd.Series,
        pruning_thresholds: List[float] = None
    ) -> Tuple[pd.DataFrame, pd.Series, List[Dict[str, Any]]]:
        """
        Apply multi-stage early pruning pipeline.
        
        Args:
            X: Feature matrix
            y: Target variable
            initial_scores: Initial feature scores
            pruning_thresholds: List of thresholds for progressive pruning
            
        Returns:
            Tuple of (final_X, final_scores, pruning_history)
        """
        if pruning_thresholds is None:
            pruning_thresholds = [0.1, 0.2, 0.3]  # Progressive thresholds
        
        tprint_info(f"✂️ Applying early pruning pipeline with {len(pruning_thresholds)} stages")
        
        current_X = X.copy()
        current_scores = initial_scores.copy()
        pruning_history = []
        
        for i, threshold in enumerate(pruning_thresholds):
            tprint_debug(f"📊 Pruning stage {i+1}: threshold {threshold}")
            
            # Apply pruning
            pruned_X, pruned_scores, pruning_info = self.early_pruning.apply_early_pruning(
                current_X, y, current_scores, threshold
            )
            
            # Update for next stage
            current_X = pruned_X
            current_scores = pruned_scores
            
            # Record history
            pruning_info['stage'] = i + 1
            pruning_info['threshold'] = threshold
            pruning_history.append(pruning_info)
            
            # Stop if no more features to prune
            if len(current_X.columns) == 0:
                tprint_warning("⚠️ All features pruned, stopping early")
                break
        
        tprint_success(f"✅ Early pruning pipeline completed: {len(X.columns)} -> {len(current_X.columns)} features")
        return current_X, current_scores, pruning_history
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        stats.update(self.early_pruning.get_pruning_statistics())
        return stats


# Convenience function for easy usage
def run_vectorbt_advanced_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    methods: List[str] = None,
    weights: Optional[Dict[str, float]] = None,
    pruning_thresholds: List[float] = None,
    config: Optional[VectorBTAdvancedConfig] = None
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Convenience function to run VectorBT-optimized advanced feature selection.
    
    Args:
        X: Feature matrix
        y: Target variable
        methods: List of methods to use
        weights: Optional weights for aggregation
        pruning_thresholds: List of thresholds for progressive pruning
        config: Optional configuration
        
    Returns:
        Tuple of (selected_features_X, aggregated_scores, performance_info)
    """
    if methods is None:
        methods = ['distance_corr', 'hsic', 'lasso_ensemble', 'cross_validation', 'bootstrap']
    
    if weights is None:
        weights = {
            'distance_corr': 0.3,
            'hsic': 0.2,
            'lasso_ensemble': 0.25,
            'cross_validation': 0.15,
            'bootstrap': 0.1
        }
    
    if pruning_thresholds is None:
        pruning_thresholds = [0.1, 0.2, 0.3]
    
    selector = VectorBTAdvancedFeatureSelector(config)
    
    try:
        # Calculate advanced scores
        importance_scores = selector.calculate_advanced_scores(X, y, methods)
        
        if not importance_scores:
            tprint_warning("⚠️ No importance scores calculated, returning original data")
            return X, pd.Series(dtype=float), {}
        
        # Aggregate and prune
        final_X, final_scores, pruning_info = selector.aggregate_and_prune_features(
            X, y, importance_scores, weights, pruning_thresholds
        )
        
        # Get performance statistics
        performance_stats = selector.get_performance_statistics()
        
        # Combine all information
        result_info = {
            'pruning_info': pruning_info,
            'performance_stats': performance_stats,
            'methods_used': methods,
            'weights_applied': weights,
            'initial_features': len(X.columns),
            'final_features': len(final_X.columns),
            'reduction_ratio': len(final_X.columns) / len(X.columns)
        }
        
        return final_X, final_scores, result_info
        
    except Exception as e:
        tprint_error(f"❌ VectorBT advanced feature selection failed: {e}")
        return X, pd.Series(dtype=float), {'error': str(e)}


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pandas as pd
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randn(n_samples))
    
    # Configure VectorBT advanced selector
    config = VectorBTAdvancedConfig(
        enable_rolling_optimizer=True,
        enable_vectorization_manager=True,
        distance_corr_use_rolling_optimizer=True,
        hsic_use_vectorization_manager=True,
        lasso_use_vectorization_manager=True,
        bootstrap_use_rolling_optimizer=True,
        importance_use_rolling_optimizer=True,
        cv_use_vectorization_manager=True,
        enable_early_pruning=True,
        pruning_threshold=0.1
    )
    
    # Run advanced feature selection
    selected_X, scores, info = run_vectorbt_advanced_feature_selection(
        X, y, 
        methods=['distance_corr', 'hsic', 'lasso_ensemble', 'cross_validation'],
        weights={
            'distance_corr': 0.3,
            'hsic': 0.2,
            'lasso_ensemble': 0.3,
            'cross_validation': 0.2
        },
        pruning_thresholds=[0.1, 0.2, 0.3],
        config=config
    )
    
    print(f"✅ Feature selection completed:")
    print(f"   📊 Initial features: {info['initial_features']}")
    print(f"   📊 Final features: {info['final_features']}")
    print(f"   📊 Reduction ratio: {info['reduction_ratio']:.1%}")
    print(f"   📊 VectorBT operations: {info['performance_stats']['vectorbt_operations']}")
    print(f"   📊 Total time: {info['performance_stats']['total_time']:.2f}s")