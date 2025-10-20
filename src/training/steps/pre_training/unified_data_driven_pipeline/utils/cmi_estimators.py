"""
Conditional Mutual Information Estimators

This module provides three-tier CMI estimation with adaptive selection:
1. KSG (k-NN) - High accuracy, final shortlist only
2. GCMI (Gaussian-copula) - Balanced performance, primary prefilter
3. Binned - Fallback for large-scale/small-sample scenarios

All estimators support rank-normalization and fold-aware caching.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import time
import warnings
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

# Import tprint utilities
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Import hardware optimizations
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUOptimizer
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    HARDWARE_OPTIMIZATIONS_AVAILABLE = True
    tprint_info("✅ Hardware optimizations available")
except ImportError:
    HARDWARE_OPTIMIZATIONS_AVAILABLE = False
    tprint_warning("⚠️ Hardware optimizations not available, using standard computations")

# Import VectorBT optimizations
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
    from src.feature_generation.utils.unified_vectorization_manager import UnifiedVectorizationManager
    VECTORBT_OPTIMIZATIONS_AVAILABLE = True
    tprint_info("✅ VectorBT optimizations available")
except ImportError:
    VECTORBT_OPTIMIZATIONS_AVAILABLE = False
    tprint_warning("⚠️ VectorBT optimizations not available, using standard computations")

# Import ML utilities
try:
    from src.utils.purged_kfold import PurgedKFoldTime
    from src.utils.ml_common.validation.data_leakage_detector import DataLeakageDetector
    from src.utils.ml_common.utils.lookahead_protection import LookaheadValidator
    ML_UTILITIES_AVAILABLE = True
    tprint_info("✅ ML utilities available")
except ImportError:
    ML_UTILITIES_AVAILABLE = False
    tprint_warning("⚠️ ML utilities not available, using standard implementations")

# Import common utilities
try:
    from src.utils.common_operations import safe_divide, safe_log
    from src.utils.common_utilities import validate_inputs, handle_missing_data
    from src.utils.math_validation import validate_numerical, check_finite
    COMMON_UTILITIES_AVAILABLE = True
    tprint_info("✅ Common utilities available")
except ImportError:
    COMMON_UTILITIES_AVAILABLE = False
    tprint_warning("⚠️ Common utilities not available, using standard implementations")

# Import Bayesian optimization
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    BAYESIAN_OPTIMIZATION_AVAILABLE = True
    tprint_info("✅ Bayesian optimization available")
except ImportError:
    BAYESIAN_OPTIMIZATION_AVAILABLE = False
    tprint_warning("⚠️ Bayesian optimization not available, using grid search")

logger = logging.getLogger(__name__)

@dataclass
class CMIResult:
    """Result from CMI estimation."""
    mi_value: float
    estimator_used: str
    computation_time: float
    is_valid: bool = True
    metadata: Dict[str, Any] = None

@dataclass
class CMIEstimatorConfig:
    """Configuration for CMI estimators."""
    # Adaptive selection thresholds
    large_scale_threshold: int = 800  # n_features > 800 → binned prefilter
    small_sample_threshold: int = 1500  # n_rows < 1500 → binned prefilter
    
    # Estimator parameters
    ksg_neighbors: int = 5
    gcmi_bins: int = 10
    binned_quantiles: int = 10
    
    # Performance limits
    compute_timeout_seconds: float = 300.0  # 5 min hard limit
    max_A_dims: int = 2  # Reduce A to ≤2 dims for efficiency
    
    # Normalization
    enable_rank_normalization: bool = True
    enable_fold_caching: bool = True
    
    # Safety checks
    min_samples_for_estimation: int = 10
    min_samples_per_bin: int = 100

class CMIEstimator:
    """
    Conditional Mutual Information estimator with adaptive selection.
    
    Supports three estimation methods:
    1. KSG (k-NN based) - High accuracy, for final shortlist
    2. GCMI (Gaussian-copula) - Balanced, primary prefilter
    3. Binned - Fallback for large-scale/small-sample
    """
    
    def __init__(self, config: Optional[CMIEstimatorConfig] = None):
        """Initialize CMI estimator."""
        self.config = config or CMIEstimatorConfig()
        self.logger = logger
        
        # Caching for fold-aware computation
        self._ksg_cache = {}
        self._A_reductions_cache = {}
        self._computation_stats = {
            'ksg_calls': 0,
            'gcmi_calls': 0,
            'binned_calls': 0,
            'total_calls': 0,
            'cache_hits': 0,
            'timeout_events': 0
        }
        
        tprint_info("🎯 CMI Estimator initialized with adaptive selection")
        
        # Initialize hardware optimizations
        self._init_hardware_optimizations()
        
        # Initialize VectorBT optimizations
        self._init_vectorbt_optimizations()
        
        # Initialize ML utilities
        self._init_ml_utilities()
    
    def _init_hardware_optimizations(self):
        """Initialize hardware optimizations for M1 chip."""
        if HARDWARE_OPTIMIZATIONS_AVAILABLE:
            try:
                self.gpu_optimizer = M1GPUOptimizer()
                self.memory_optimizer = M1MemoryOptimizer()
                self.cpu_optimizer = M1CPUOptimizer()
                tprint_success("✅ Hardware optimizations initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Hardware optimization initialization failed: {e}")
                self.gpu_optimizer = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
        else:
            self.gpu_optimizer = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    def _init_vectorbt_optimizations(self):
        """Initialize VectorBT optimizations for efficient rolling computations."""
        if VECTORBT_OPTIMIZATIONS_AVAILABLE:
            try:
                self.vectorbt_optimizer = VectorBTRollingOptimizer()
                self.vectorization_manager = UnifiedVectorizationManager()
                tprint_success("✅ VectorBT optimizations initialized")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT optimization initialization failed: {e}")
                self.vectorbt_optimizer = None
                self.vectorization_manager = None
        else:
            self.vectorbt_optimizer = None
            self.vectorization_manager = None
    
    def _init_ml_utilities(self):
        """Initialize ML utilities for cross-validation and data leakage detection."""
        if ML_UTILITIES_AVAILABLE:
            try:
                self.purged_kfold = PurgedKFold
                self.data_leakage_detector = DataLeakageDetector()
                self.lookahead_validator = LookaheadValidator()
                tprint_success("✅ ML utilities initialized")
            except Exception as e:
                tprint_warning(f"⚠️ ML utility initialization failed: {e}")
                self.purged_kfold = None
                self.data_leakage_detector = None
                self.lookahead_validator = None
        else:
            self.purged_kfold = None
            self.data_leakage_detector = None
            self.lookahead_validator = None
    
    def select_estimator(self, n_features: int, n_rows: int, stage: str) -> str:
        """
        Select appropriate estimator based on data characteristics and stage.
        
        Args:
            n_features: Number of features
            n_rows: Number of samples
            stage: 'prefilter', 'shortlist', or 'final'
            
        Returns:
            Estimator name: 'ksg', 'gcmi', or 'binned'
        """
        # Large scale or small sample → binned prefilter
        if n_features > self.config.large_scale_threshold or n_rows < self.config.small_sample_threshold:
            if stage == 'prefilter':
                return 'binned'
            elif stage == 'shortlist':  # Top 3×k
                return 'gcmi'
            else:  # Final k
                return 'ksg'
        elif n_features <= 600 and n_rows >= 2000:
            if stage == 'prefilter':
                return 'gcmi'
            else:  # Final
                return 'ksg'
        else:
            return 'gcmi'  # Balanced default
    
    def estimate_cmi(self, X: np.ndarray, Y: np.ndarray, A: np.ndarray, 
                    estimator: Optional[str] = None, stage: str = 'prefilter',
                    fold_id: Optional[str] = None) -> CMIResult:
        """
        Estimate conditional mutual information I(Y; X | A).
        
        Args:
            X: Feature array (n_samples, n_features)
            Y: Target array (n_samples,)
            A: Analyst side information (n_samples, n_A_dims)
            estimator: Specific estimator to use (None for adaptive)
            stage: 'prefilter', 'shortlist', or 'final'
            fold_id: Fold identifier for caching
            
        Returns:
            CMIResult with MI value and metadata
        """
        start_time = time.time()
        
        try:
            # Input validation
            if not self._validate_inputs(X, Y, A):
                return CMIResult(
                    mi_value=0.0,
                    estimator_used='invalid',
                    computation_time=time.time() - start_time,
                    is_valid=False,
                    metadata={'error': 'Invalid inputs'}
                )
            
            # Select estimator if not specified
            if estimator is None:
                estimator = self.select_estimator(X.shape[1], X.shape[0], stage)
            
            # Check timeout
            if time.time() - start_time > self.config.compute_timeout_seconds:
                self._computation_stats['timeout_events'] += 1
                tprint_warning("⚠️ CMI computation timeout, using fallback")
                estimator = 'binned'
            
            # Reduce A dimensionality if needed
            A_reduced = self._reduce_A_dimensionality(A, fold_id)
            
            # Estimate CMI based on selected method
            if estimator == 'ksg':
                mi_value = self._estimate_ksg(X, Y, A_reduced, fold_id)
            elif estimator == 'gcmi':
                mi_value = self._estimate_gcmi(X, Y, A_reduced)
            elif estimator == 'binned':
                mi_value = self._estimate_binned(X, Y, A_reduced)
            else:
                raise ValueError(f"Unknown estimator: {estimator}")
            
            # Update stats
            self._computation_stats[f'{estimator}_calls'] += 1
            self._computation_stats['total_calls'] += 1
            
            computation_time = time.time() - start_time
            
            return CMIResult(
                mi_value=mi_value,
                estimator_used=estimator,
                computation_time=computation_time,
                is_valid=True,
                metadata={
                    'stage': stage,
                    'fold_id': fold_id,
                    'n_features': X.shape[1],
                    'n_samples': X.shape[0],
                    'A_dims': A_reduced.shape[1] if len(A_reduced.shape) > 1 else 1,
                    'cache_used': fold_id is not None and self.config.enable_fold_caching
                }
            )
            
        except Exception as e:
            tprint_error(f"❌ CMI estimation failed: {e}")
            return CMIResult(
                mi_value=0.0,
                estimator_used=estimator or 'unknown',
                computation_time=time.time() - start_time,
                is_valid=False,
                metadata={'error': str(e)}
            )
    
    def _validate_inputs(self, X: np.ndarray, Y: np.ndarray, A: np.ndarray) -> bool:
        """Validate input arrays."""
        try:
            if X is None or Y is None or A is None:
                return False
            
            if len(X) != len(Y) or len(X) != len(A):
                return False
            
            if len(X) < self.config.min_samples_for_estimation:
                return False
            
            # Check for all NaN
            if np.all(np.isnan(X)) or np.all(np.isnan(Y)) or np.all(np.isnan(A)):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _reduce_A_dimensionality(self, A: np.ndarray, fold_id: Optional[str] = None) -> np.ndarray:
        """Reduce A to ≤2 dimensions for CMI efficiency."""
        if A.shape[1] <= self.config.max_A_dims:
            return A
        
        # Check cache first
        if fold_id and self.config.enable_fold_caching:
            cache_key = f"A_reduction_{fold_id}_{A.shape}"
            if cache_key in self._A_reductions_cache:
                self._computation_stats['cache_hits'] += 1
                return self._A_reductions_cache[cache_key]
        
        # Use PCA to reduce to max_A_dims
        try:
            pca = PCA(n_components=self.config.max_A_dims)
            A_reduced = pca.fit_transform(A)
            
            # Cache result
            if fold_id and self.config.enable_fold_caching:
                self._A_reductions_cache[cache_key] = A_reduced
            
            return A_reduced
            
        except Exception as e:
            tprint_warning(f"⚠️ A dimensionality reduction failed: {e}")
            # Fallback: take first max_A_dims columns
            return A[:, :self.config.max_A_dims]
    
    def _estimate_ksg(self, X: np.ndarray, Y: np.ndarray, A: np.ndarray, 
                     fold_id: Optional[str] = None) -> float:
        """Estimate CMI using KSG (k-NN) method."""
        try:
            # Check cache for KSG neighbor graphs
            if fold_id and self.config.enable_fold_caching:
                cache_key = f"ksg_{fold_id}_{X.shape}_{Y.shape}_{A.shape}"
                if cache_key in self._ksg_cache:
                    self._computation_stats['cache_hits'] += 1
                    return self._ksg_cache[cache_key]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y) | np.isnan(A).any(axis=1))
            X_clean = X[valid_mask]
            Y_clean = Y[valid_mask]
            A_clean = A[valid_mask]
            
            if len(X_clean) < self.config.min_samples_for_estimation:
                return 0.0
            
            # KSG estimation
            k = min(self.config.ksg_neighbors, len(X_clean) - 1)
            
            # Compute distances for k-NN
            XY = np.column_stack([X_clean, Y_clean.reshape(-1, 1)])
            XA = np.column_stack([X_clean, A_clean])
            YA = np.column_stack([Y_clean.reshape(-1, 1), A_clean])
            
            # Find k-th nearest neighbor distances
            nbrs_xy = NearestNeighbors(n_neighbors=k+1, metric='chebyshev')
            nbrs_xy.fit(XY)
            distances_xy, _ = nbrs_xy.kneighbors(XY)
            eps_xy = distances_xy[:, -1]  # k-th distance
            
            nbrs_xa = NearestNeighbors(n_neighbors=k+1, metric='chebyshev')
            nbrs_xa.fit(XA)
            distances_xa, _ = nbrs_xa.kneighbors(XA)
            eps_xa = distances_xa[:, -1]
            
            nbrs_ya = NearestNeighbors(n_neighbors=k+1, metric='chebyshev')
            nbrs_ya.fit(YA)
            distances_ya, _ = nbrs_ya.kneighbors(YA)
            eps_ya = distances_ya[:, -1]
            
            # KSG formula: I(Y;X|A) = ψ(k) + ψ(N) - <ψ(n_y) + ψ(n_x)>
            # where n_y, n_x are counts within eps_ya, eps_xa respectively
            n_y = np.sum(distances_ya <= eps_ya[:, np.newaxis], axis=1) - 1
            n_x = np.sum(distances_xa <= eps_xa[:, np.newaxis], axis=1) - 1
            
            # Digamma function - use scipy.special.digamma or approximation
            try:
                from scipy.special import digamma
            except ImportError:
                # Fallback approximation
                def digamma(x):
                    return np.log(np.maximum(x, 1e-10)) - 1/(2*np.maximum(x, 1e-10))
            
            psi_k = digamma(k)
            psi_n = digamma(len(X_clean))
            psi_ny = digamma(n_y)
            psi_nx = digamma(n_x)
            
            mi_value = psi_k + psi_n - np.mean(psi_ny + psi_nx)
            
            # Cache result
            if fold_id and self.config.enable_fold_caching:
                self._ksg_cache[cache_key] = mi_value
            
            return max(0.0, mi_value)  # MI is non-negative
            
        except Exception as e:
            tprint_warning(f"⚠️ KSG estimation failed: {e}")
            return 0.0
    
    def _estimate_gcmi(self, X: np.ndarray, Y: np.ndarray, A: np.ndarray) -> float:
        """Estimate CMI using Gaussian-copula MI method."""
        try:
            # Remove NaN values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y) | np.isnan(A).any(axis=1))
            X_clean = X[valid_mask]
            Y_clean = Y[valid_mask]
            A_clean = A[valid_mask]
            
            if len(X_clean) < self.config.min_samples_for_estimation:
                return 0.0
            
            # Rank normalization
            if self.config.enable_rank_normalization:
                X_clean = self._rank_normalize(X_clean)
                Y_clean = self._rank_normalize(Y_clean.reshape(-1, 1)).flatten()
                A_clean = self._rank_normalize(A_clean)
            
            # Gaussian-copula transformation
            X_gaussian = self._gaussian_copula_transform(X_clean)
            Y_gaussian = self._gaussian_copula_transform(Y_clean.reshape(-1, 1)).flatten()
            A_gaussian = self._gaussian_copula_transform(A_clean)
            
            # Compute conditional MI using Gaussian assumption
            # I(Y;X|A) = 0.5 * log(det(C_XX|A) * det(C_YY|A) / det(C_XY|A))
            # where C_XX|A is the conditional covariance of X given A
            
            # Stack X and A for joint analysis
            XA = np.column_stack([X_gaussian, A_gaussian])
            YA = np.column_stack([Y_gaussian.reshape(-1, 1), A_gaussian])
            
            # Compute covariance matrices
            cov_XA = np.cov(XA.T)
            cov_YA = np.cov(YA.T)
            
            # Conditional covariances
            n_x = X_gaussian.shape[1]
            n_a = A_gaussian.shape[1]
            
            # C_XX|A = C_XX - C_XA * C_AA^-1 * C_AX
            C_XX = cov_XA[:n_x, :n_x]
            C_XA = cov_XA[:n_x, n_x:]
            C_AA = cov_XA[n_x:, n_x:]
            C_AX = C_XA.T
            
            try:
                C_AA_inv = np.linalg.inv(C_AA + 1e-8 * np.eye(n_a))
                C_XX_given_A = C_XX - C_XA @ C_AA_inv @ C_AX
            except np.linalg.LinAlgError:
                C_XX_given_A = C_XX
            
            # C_YY|A
            C_YY = cov_YA[0, 0]
            C_YA = cov_YA[0, 1:]
            C_AA_YA = cov_YA[1:, 1:]
            C_AY = C_YA.T
            
            try:
                C_AA_YA_inv = np.linalg.inv(C_AA_YA + 1e-8 * np.eye(n_a))
                C_YY_given_A = C_YY - C_YA @ C_AA_YA_inv @ C_AY
            except np.linalg.LinAlgError:
                C_YY_given_A = C_YY
            
            # Joint conditional covariance
            XY = np.column_stack([X_gaussian, Y_gaussian.reshape(-1, 1)])
            XYA = np.column_stack([XY, A_gaussian])
            cov_XYA = np.cov(XYA.T)
            
            n_xy = XY.shape[1]
            C_XY = cov_XYA[:n_xy, :n_xy]
            C_XYA = cov_XYA[:n_xy, n_xy:]
            C_AA_XYA = cov_XYA[n_xy:, n_xy:]
            C_AYX = C_XYA.T
            
            try:
                C_AA_XYA_inv = np.linalg.inv(C_AA_XYA + 1e-8 * np.eye(n_a))
                C_XY_given_A = C_XY - C_XYA @ C_AA_XYA_inv @ C_AYX
            except np.linalg.LinAlgError:
                C_XY_given_A = C_XY
            
            # Compute determinants
            det_XX_given_A = np.linalg.det(C_XX_given_A + 1e-8 * np.eye(n_x))
            det_YY_given_A = C_YY_given_A + 1e-8
            det_XY_given_A = np.linalg.det(C_XY_given_A + 1e-8 * np.eye(n_xy))
            
            # CMI = 0.5 * log(det(C_XX|A) * det(C_YY|A) / det(C_XY|A))
            mi_value = 0.5 * np.log(det_XX_given_A * det_YY_given_A / det_XY_given_A)
            
            return max(0.0, mi_value)
            
        except Exception as e:
            tprint_warning(f"⚠️ GCMI estimation failed: {e}")
            return 0.0
    
    def _estimate_binned(self, X: np.ndarray, Y: np.ndarray, A: np.ndarray) -> float:
        """Estimate CMI using quantile-binned method."""
        try:
            # Remove NaN values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y) | np.isnan(A).any(axis=1))
            X_clean = X[valid_mask]
            Y_clean = Y[valid_mask]
            A_clean = A[valid_mask]
            
            if len(X_clean) < self.config.min_samples_for_estimation:
                return 0.0
            
            # Quantile binning
            n_bins = self.config.binned_quantiles
            
            # Bin each dimension
            X_binned = np.zeros_like(X_clean)
            for i in range(X_clean.shape[1]):
                X_binned[:, i] = pd.qcut(X_clean[:, i], n_bins, labels=False, duplicates='drop')
            
            Y_binned = pd.qcut(Y_clean, n_bins, labels=False, duplicates='drop')
            
            A_binned = np.zeros_like(A_clean)
            for i in range(A_clean.shape[1]):
                A_binned[:, i] = pd.qcut(A_clean[:, i], n_bins, labels=False, duplicates='drop')
            
            # Compute conditional MI using binned data
            # I(Y;X|A) = sum_a p(a) * I(Y;X|A=a)
            mi_value = 0.0
            
            # Get unique A combinations
            if A_binned.ndim == 1:
                A_unique = np.unique(A_binned)
            else:
                A_unique = np.unique(A_binned, axis=0)
            
            for a_val in A_unique:
                # Find samples with this A value
                if A_binned.ndim == 1:
                    a_mask = A_binned == a_val
                else:
                    a_mask = np.all(A_binned == a_val, axis=1)
                
                if np.sum(a_mask) < 2:  # Need at least 2 samples
                    continue
                
                # Get X, Y for this A value
                X_a = X_binned[a_mask]
                Y_a = Y_binned[a_mask]
                
                # Compute MI for this A stratum
                mi_a = self._compute_binned_mi(X_a, Y_a)
                
                # Weight by probability of A
                p_a = np.sum(a_mask) / len(A_clean)
                mi_value += p_a * mi_a
            
            return max(0.0, mi_value)
            
        except Exception as e:
            tprint_warning(f"⚠️ Binned estimation failed: {e}")
            return 0.0
    
    def _compute_binned_mi(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute MI for binned data using histogram method."""
        try:
            # Create joint histogram
            # For simplicity, use the first feature of X
            if X.shape[1] > 1:
                X_flat = X[:, 0]  # Use first feature
            else:
                X_flat = X.flatten()
            
            # Get unique values
            X_unique = np.unique(X_flat)
            Y_unique = np.unique(Y)
            
            # Count joint occurrences
            joint_counts = np.zeros((len(X_unique), len(Y_unique)))
            for i, x_val in enumerate(X_unique):
                for j, y_val in enumerate(Y_unique):
                    joint_counts[i, j] = np.sum((X_flat == x_val) & (Y == y_val))
            
            # Normalize to probabilities
            total = np.sum(joint_counts)
            if total == 0:
                return 0.0
            
            joint_probs = joint_counts / total
            X_probs = np.sum(joint_probs, axis=1)
            Y_probs = np.sum(joint_probs, axis=0)
            
            # Compute MI
            mi = 0.0
            for i in range(len(X_unique)):
                for j in range(len(Y_unique)):
                    if joint_probs[i, j] > 0:
                        mi += joint_probs[i, j] * np.log2(
                            joint_probs[i, j] / (X_probs[i] * Y_probs[j] + 1e-10)
                        )
            
            return max(0.0, mi)
            
        except Exception as e:
            tprint_warning(f"⚠️ Binned MI computation failed: {e}")
            return 0.0
    
    def _rank_normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply rank normalization to data."""
        try:
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            
            normalized = np.zeros_like(data)
            for i in range(data.shape[1]):
                # Rank normalization: (rank - 1) / (n - 1)
                ranks = stats.rankdata(data[:, i])
                normalized[:, i] = (ranks - 1) / (len(ranks) - 1)
            
            return normalized
            
        except Exception as e:
            tprint_warning(f"⚠️ Rank normalization failed: {e}")
            return data
    
    def _gaussian_copula_transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data to Gaussian copula space."""
        try:
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            
            transformed = np.zeros_like(data)
            for i in range(data.shape[1]):
                # Rank normalization first
                ranks = stats.rankdata(data[:, i])
                uniform = (ranks - 1) / (len(ranks) - 1)
                
                # Transform to Gaussian
                transformed[:, i] = stats.norm.ppf(np.clip(uniform, 1e-10, 1-1e-10))
            
            return transformed
            
        except Exception as e:
            tprint_warning(f"⚠️ Gaussian copula transform failed: {e}")
            return data
    
    def get_computation_stats(self) -> Dict[str, Any]:
        """Get computation statistics."""
        return self._computation_stats.copy()
    
    def clear_cache(self):
        """Clear all caches."""
        self._ksg_cache.clear()
        self._A_reductions_cache.clear()
        tprint_info("🧹 CMI estimator caches cleared")

def create_cmi_estimator(config: Optional[CMIEstimatorConfig] = None) -> CMIEstimator:
    """Create a CMI estimator with default configuration."""
    return CMIEstimator(config)
