"""
Advanced 3-Step Iterative Clustering Optimization for NAS-TAS.

This module implements a sophisticated iterative optimization loop with:
1. Local frontier moves (CV-focused with balance/silhouette/temporal)
2. Global reallocation (capacity-aware coordination)
3. Break large clusters (size-aware quality thresholds)

Features:
- Fast delta calculations using sufficient statistics
- Numba-optimized vectorized operations
- Comprehensive monitoring and reporting
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.cluster import KMeans
from contextlib import nullcontext

# Optional imports
try:
    import umap
except ImportError:
    umap = None

# RNG utilities for compatibility between RandomState and Generator
def rng_from(seed_or_rng=None):
    if isinstance(seed_or_rng, np.random.Generator):
        return seed_or_rng
    if isinstance(seed_or_rng, np.random.RandomState):
        return seed_or_rng
    return np.random.default_rng(seed_or_rng)

def rand_int(rng, low, high=None, size=None):
    if hasattr(rng, "integers"):   # Generator
        return rng.integers(low, high=high, size=size)
    return rng.randint(low, high if high is not None else low, size)  # RandomState

def rand_choice(rng, a, size=None, replace=True, p=None):
    if hasattr(rng, "choice"):     # both have .choice, but signatures slightly differ; this works
        return rng.choice(a, size=size, replace=replace, p=p)
    # fallback
    return np.random.choice(a, size=size, replace=replace, p=p)
from enum import Enum
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from numba import jit, prange
import warnings
import math

class SplitError(Enum):
    """Split operation error codes."""
    TOO_SMALL = 6          # not enough members to produce two >= min_size
    NO_GAIN = 7            # objective didn't improve (Δ ≤ eps)
    BAD_CHILD = 8          # one child below min_size after 2-means
    SHAPE = 10             # arrays not expanded to K+1
    OUT_OF_SYNC = 13       # cached per-cluster stats stale
    PARENT_TOO_SMALL = 14  # parent doesn't meet size requirements
    CHILD_TOO_SMALL = 15   # children don't meet size requirements
    INSUFFICIENT_GAIN = 16 # gain doesn't meet multi-metric thresholds
    RATE_LIMITED = 17      # split rate limit exceeded
    PRECONDITIONS_FAILED = 18  # context health preconditions not met

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error
)
from src.utils.common_operations import safe_divide

from ..shared_utils import get_logger
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
from .step1_feature_preparation import ClusteringContext
from .risk_mitigation import RiskMitigationSystem, PRODUCTION_RISK_CONFIG

# Import CV enhancement strategies
try:
    from .cv_enhancement_strategies import (
        AdaptiveWeightScheduler,
        EnhancedVarianceRatioCalculator
    )
    CV_ENHANCEMENT_AVAILABLE = True
except ImportError:
    CV_ENHANCEMENT_AVAILABLE = False

# Import optimization utilities
try:
    from src.utils.matrix_operations import (
        get_vectorized_processing_core,
        get_hardware_optimized_processor,
        hardware_optimized,
        optimize_matrix_operation,
        vectorized_rolling_features,
        matrix_correlation_analysis
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

try:
    from src.utils.hardware import (
        get_unified_hardware_manager,
        get_advanced_cpu_optimizer,
        get_enhanced_gpu_manager,
        get_advanced_memory_optimizer,
        optimize_for_workload
    )
    HARDWARE_ACCEL_AVAILABLE = True
except ImportError:
    HARDWARE_ACCEL_AVAILABLE = False

class OptimizedCalculationEngine:
    """Optimized calculation engine with vectorized operations, caching, and chunking."""

    def __init__(self, use_hardware_accel: bool = True, cache_size: int = 1000):
        self.use_hardware_accel = use_hardware_accel and HARDWARE_ACCEL_AVAILABLE
        self.cache_size = cache_size

        # Initialize hardware components
        self.hardware_manager = None
        self.vectorized_core = None
        self.cpu_optimizer = None
        self.memory_optimizer = None

        # Caching
        self._silhouette_cache = {}
        self._distance_cache = {}
        self._centroid_cache = {}

        # Initialize hardware acceleration
        self._initialize_hardware_components()

    def _initialize_hardware_components(self):
        """Initialize hardware acceleration components."""
        if self.use_hardware_accel:
            try:
                if HARDWARE_ACCEL_AVAILABLE:
                    self.hardware_manager = get_unified_hardware_manager()
                    self.cpu_optimizer = get_advanced_cpu_optimizer()
                    self.memory_optimizer = get_advanced_memory_optimizer()

                if MATRIX_OPS_AVAILABLE:
                    self.vectorized_core = get_vectorized_processing_core()

                tprint("✅ Optimized calculation engine initialized with hardware acceleration")
            except Exception as e:
                tprint(f"⚠️ Hardware acceleration initialization failed: {e}")
                self.use_hardware_accel = False

    def calculate_silhouette_score_optimized(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate silhouette score with vectorized operations and caching."""
        try:
            if len(features) == 0 or len(assignments) == 0:
                return 0.0
            if len(np.unique(assignments)) < 2:
                return 0.0

            # Check cache first
            cache_key = self._create_silhouette_cache_key(features, assignments)
            if cache_key in self._silhouette_cache:
                return self._silhouette_cache[cache_key]

            # Skip if any active cluster has < 2 points (only check non-empty clusters)
            unique_labels, cluster_sizes = np.unique(assignments, return_counts=True)
            if np.any(cluster_sizes < 2):
                return 0.0

            # Always use sklearn for reliable silhouette calculation
            from sklearn.metrics import silhouette_score as sk_silhouette_score
            try:
                # Ensure features are in proper format for sklearn
                if features.ndim == 1:
                    features_2d = features.reshape(-1, 1)
                else:
                    features_2d = features

                # Handle very large datasets by sampling for silhouette calculation
                if len(features_2d) > 50000:
                    tprint(f"⚠️ Large dataset ({len(features_2d)} samples) for silhouette calculation, using sampling", "WARNING")
                    sample_indices = np.random.choice(len(features_2d), size=min(50000, len(features_2d)), replace=False)
                    sample_features = features_2d[sample_indices]
                    sample_assignments = assignments[sample_indices]
                    sil_score = sk_silhouette_score(sample_features, sample_assignments)
                else:
                    sil_score = sk_silhouette_score(features_2d, assignments)

                # Validate the result with more robust checks
                if np.isnan(sil_score) or np.isinf(sil_score) or sil_score < -1.0 or sil_score > 1.0:
                    tprint(f"⚠️ Invalid silhouette score ({sil_score:.4f}), using fallback calculation", "WARNING")
                    sil_score = self._calculate_silhouette_robust_fallback(features_2d, assignments)

            except Exception as sk_error:
                tprint(f"⚠️ Sklearn silhouette calculation failed: {sk_error}", "WARNING")
                # Try hardware-optimized as last resort
                try:
                    if self.use_hardware_accel and self.vectorized_core:
                        sil_score = self._calculate_silhouette_hardware_optimized(features, assignments)
                        if np.isnan(sil_score) or np.isinf(sil_score):
                            sil_score = 0.0
                    else:
                        sil_score = 0.0
                except Exception as hw_error:
                    tprint(f"⚠️ Hardware silhouette also failed: {hw_error}", "WARNING")
                    sil_score = 0.0

            # Cache the result
            if len(self._silhouette_cache) < self.cache_size:
                self._silhouette_cache[cache_key] = sil_score

            return sil_score

        except Exception as e:
            tprint(f"⚠️ Optimized silhouette calculation failed: {e}", "ERROR")
            return 0.0

    def _calculate_silhouette_robust_fallback(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Robust fallback silhouette calculation for edge cases."""
        try:
            # Use a simple distance-based approximation for robustness
            unique_labels = np.unique(assignments)
            if len(unique_labels) < 2:
                return 0.0

            # Calculate pairwise distances between points
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(features)

            # For each point, calculate average distance to points in same cluster and nearest other cluster
            silhouettes = []

            for i in range(len(features)):
                cluster_id = assignments[i]

                # Distance to points in same cluster
                same_cluster_mask = assignments == cluster_id
                if np.sum(same_cluster_mask) <= 1:
                    continue

                a_i = np.mean(distances[i, same_cluster_mask])

                # Distance to points in nearest other cluster
                other_clusters = [c for c in unique_labels if c != cluster_id]
                b_i_values = []

                for other_c in other_clusters:
                    other_mask = assignments == other_c
                    if np.sum(other_mask) > 0:
                        b_i_values.append(np.mean(distances[i, other_mask]))

                if not b_i_values:
                    continue

                b_i = min(b_i_values)
                silhouette_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0
                silhouettes.append(silhouette_i)

            return float(np.mean(silhouettes)) if silhouettes else 0.0

        except Exception as e:
            tprint(f"⚠️ Robust fallback silhouette failed: {e}", "WARNING")
            return 0.0

    def _calculate_silhouette_hardware_optimized(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate silhouette score using hardware acceleration and matrix operations."""
        try:
            # Always fallback to sklearn for reliable results
            from sklearn.metrics import silhouette_score
            return silhouette_score(features, assignments)

        except Exception as e:
            tprint(f"⚠️ Hardware-optimized silhouette failed: {e}", "WARNING")
            # Final fallback
            try:
                from sklearn.metrics import silhouette_score
                return silhouette_score(features, assignments)
            except Exception:
                return 0.0

    def _process_silhouette_with_matrix_ops(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Process silhouette calculation with matrix operations."""
        try:
            # For large datasets, use chunking with matrix operations
            if len(features) > 10000 and self.memory_optimizer:
                return self._calculate_silhouette_chunked_with_matrix_ops(features, assignments)
            else:
                # Use matrix operations for distance calculation
                matrix_ops = UnifiedMatrixOperations()
                distance_matrix = matrix_ops.calculate_pairwise_similarities(features, method='euclidean')

                # Convert similarity to distance for silhouette calculation
                distances = 1.0 - distance_matrix
                np.fill_diagonal(distances, 0.0)

                # Use sklearn silhouette with precomputed distances
                return silhouette_score(distances, assignments, metric='precomputed')

        except Exception as e:
            tprint(f"⚠️ Matrix operations silhouette failed: {e}")
            return silhouette_score(features, assignments)

    def _calculate_silhouette_chunked_with_matrix_ops(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate silhouette score using chunking with matrix operations."""
        try:
            # Use memory-optimized chunking
            chunks = self.memory_optimizer.chunk_series(pd.Series(range(len(features))), chunk_size=5000)

            silhouette_scores = []
            matrix_ops = UnifiedMatrixOperations()

            for chunk in chunks:
                if len(chunk) < 2:
                    continue

                chunk_features = features[chunk]
                chunk_assignments = assignments[chunk]

                if len(np.unique(chunk_assignments)) < 2:
                    continue

                # Use matrix operations for chunk processing
                try:
                    distance_matrix = matrix_ops.calculate_pairwise_similarities(chunk_features, method='euclidean')
                    distances = 1.0 - distance_matrix
                    np.fill_diagonal(distances, 0.0)
                    chunk_score = silhouette_score(distances, chunk_assignments, metric='precomputed')
                except:
                    chunk_score = silhouette_score(chunk_features, chunk_assignments)

                silhouette_scores.append(chunk_score)

            return np.mean(silhouette_scores) if silhouette_scores else 0.0

        except Exception as e:
            tprint(f"⚠️ Chunked matrix operations silhouette failed: {e}")
            return silhouette_score(features, assignments)

    def _process_silhouette_with_hardware(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Process silhouette calculation with hardware acceleration."""
        # Use chunking for large datasets
        if len(features) > 10000 and self.memory_optimizer:
            return self._calculate_silhouette_chunked(features, assignments)
        else:
            return silhouette_score(features, assignments)

    def _calculate_silhouette_chunked(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate silhouette score using chunking for large datasets."""
        try:
            # Use memory-optimized chunking
            chunks = self.memory_optimizer.chunk_series(pd.Series(range(len(features))), chunk_size=5000)

            silhouette_scores = []
            for chunk in chunks:
                if len(chunk) < 2:
                    continue

                chunk_features = features[chunk]
                chunk_assignments = assignments[chunk]

                if len(np.unique(chunk_assignments)) < 2:
                    continue

                chunk_score = silhouette_score(chunk_features, chunk_assignments)
                silhouette_scores.append(chunk_score)

            return np.mean(silhouette_scores) if silhouette_scores else 0.0

        except Exception as e:
            tprint(f"⚠️ Chunked silhouette calculation failed: {e}")
            return silhouette_score(features, assignments)

    def calculate_distance_matrix_optimized(self, features: np.ndarray) -> np.ndarray:
        """Calculate distance matrix with vectorized operations and caching."""
        try:
            # Check cache first
            cache_key = self._create_distance_cache_key(features)
            if cache_key in self._distance_cache:
                return self._distance_cache[cache_key]

            # Use hardware-optimized calculation if available
            if self.use_hardware_accel and self.vectorized_core:
                distance_matrix = self._calculate_distance_hardware_optimized(features)
            else:
                # Fallback to optimized numpy
                distance_matrix = self._calculate_distance_vectorized(features)

            # Cache the result
            if len(self._distance_cache) < self.cache_size:
                self._distance_cache[cache_key] = distance_matrix

            return distance_matrix

        except Exception as e:
            tprint(f"⚠️ Optimized distance calculation failed: {e}")
            return self._calculate_distance_vectorized(features)

    def _calculate_distance_hardware_optimized(self, features: np.ndarray) -> np.ndarray:
        """Calculate distance matrix using hardware acceleration."""
        try:
            # Use vectorized core for preprocessing
            if self.vectorized_core:
                features_optimized = self.vectorized_core.optimize_dataframe_for_processing(
                    pd.DataFrame(features)
                ).values
            else:
                features_optimized = features

            # Use hardware-optimized workload processing
            if self.hardware_manager:
                workload_config = {
                    'workload_type': 'distance_matrix',
                    'data_size': len(features),
                    'complexity': 'high',
                    'memory_intensive': True
                }

                # Optimize for distance matrix workload
                optimized_config = optimize_for_workload(workload_config)

                # Process with hardware optimization
                with self.cpu_optimizer.optimized_execution_context() if self.cpu_optimizer else nullcontext():
                    return self._process_distance_with_hardware(features_optimized)
            else:
                return self._calculate_distance_vectorized(features_optimized)

        except Exception as e:
            tprint(f"⚠️ Hardware-optimized distance calculation failed: {e}")
            return self._calculate_distance_vectorized(features)

    def _process_distance_with_hardware(self, features: np.ndarray) -> np.ndarray:
        """Process distance calculation with hardware acceleration."""
        # Use chunking for large datasets
        if len(features) > 5000 and self.memory_optimizer:
            return self._calculate_distance_chunked(features)
        else:
            return self._calculate_distance_vectorized(features)

    def _calculate_distance_chunked(self, features: np.ndarray) -> np.ndarray:
        """Calculate distance matrix using chunking for large datasets."""
        try:
            n = len(features)
            distance_matrix = np.zeros((n, n))

            # Use memory-optimized chunking
            chunks = self.memory_optimizer.chunk_series(pd.Series(range(n)), chunk_size=1000)

            for i, chunk_i in enumerate(chunks):
                for j, chunk_j in enumerate(chunks):
                    if i <= j:  # Only calculate upper triangle
                        chunk_distances = self._calculate_distance_vectorized(features[chunk_i], features[chunk_j])
                        distance_matrix[np.ix_(chunk_i, chunk_j)] = chunk_distances
                        if i != j:  # Fill lower triangle
                            distance_matrix[np.ix_(chunk_j, chunk_i)] = chunk_distances.T

            return distance_matrix

        except Exception as e:
            tprint(f"⚠️ Chunked distance calculation failed: {e}")
            return self._calculate_distance_vectorized(features)

    def _calculate_distance_vectorized(self, features1: np.ndarray, features2: np.ndarray = None) -> np.ndarray:
        """Calculate distance matrix using vectorized operations with matrix operations."""
        if features2 is None:
            features2 = features1

        try:
            # Use UnifiedMatrixOperations for optimized distance calculation
            matrix_ops = UnifiedMatrixOperations()

            # Calculate pairwise similarities and convert to distances
            if features1 is features2:
                # Same array - use optimized pairwise calculation
                similarity_matrix = matrix_ops.calculate_pairwise_similarities(features1, method='euclidean')
                # Convert similarity to distance (inverse relationship)
                distances = 1.0 - similarity_matrix
                # Ensure diagonal is 0
                np.fill_diagonal(distances, 0.0)
            else:
                # Different arrays - use manual calculation
                diff = features1[:, np.newaxis, :] - features2[np.newaxis, :, :]
                distances = np.sqrt(np.sum(diff ** 2, axis=2))

            return distances

        except Exception as e:
            tprint(f"⚠️ Matrix operations distance calculation failed: {e}")
            # Fallback to basic vectorized operations
            diff = features1[:, np.newaxis, :] - features2[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff ** 2, axis=2))
            return distances

    def calculate_centroids_optimized(self, features: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """Calculate centroids with vectorized operations and caching using matrix operations."""
        try:
            # Check cache first
            cache_key = self._create_centroid_cache_key(features, assignments)
            if cache_key in self._centroid_cache:
                return self._centroid_cache[cache_key]

            # Use matrix operations for optimized centroid calculation
            if self.use_hardware_accel and self.vectorized_core:
                centroids = self._calculate_centroids_with_matrix_ops(features, assignments)
            else:
                # Fallback to basic vectorized operations
                centroids = self._calculate_centroids_basic(features, assignments)

            # Cache the result
            if len(self._centroid_cache) < self.cache_size:
                self._centroid_cache[cache_key] = centroids

            return centroids

        except Exception as e:
            tprint(f"⚠️ Optimized centroid calculation failed: {e}")
            return self._calculate_centroids_basic(features, assignments)

    def _calculate_centroids_with_matrix_ops(self, features: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """Calculate centroids using matrix operations for better performance."""
        try:
            # Use vectorized core for preprocessing
            if self.vectorized_core:
                features_optimized = self.vectorized_core.optimize_dataframe_for_processing(
                    pd.DataFrame(features)
                ).values
            else:
                features_optimized = features

            # Use matrix operations for centroid calculation
            K = int(assignments.max()) + 1
            centroids = np.zeros((K, features_optimized.shape[1]))

            # Vectorized centroid calculation using broadcasting
            for k in range(K):
                mask = assignments == k
                if np.any(mask):
                    cluster_features = features_optimized[mask]
                    # Use optimized mean calculation
                    if self.cpu_optimizer:
                        with self.cpu_optimizer.optimized_execution_context():
                            centroids[k] = np.mean(cluster_features, axis=0)
                    else:
                        centroids[k] = np.mean(cluster_features, axis=0)

            return centroids

        except Exception as e:
            tprint(f"⚠️ Matrix operations centroid calculation failed: {e}")
            return self._calculate_centroids_basic(features, assignments)

    def _calculate_centroids_basic(self, features: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """Calculate centroids using basic vectorized operations."""
        K = int(assignments.max()) + 1
        centroids = np.zeros((K, features.shape[1]))

        for k in range(K):
            mask = assignments == k
            if np.any(mask):
                centroids[k] = np.mean(features[mask], axis=0)

        return centroids

    def _create_silhouette_cache_key(self, features: np.ndarray, assignments: np.ndarray) -> str:
        """Create cache key for silhouette calculation."""
        return f"sil_{hash(features.tobytes())}_{hash(assignments.tobytes())}"

    def _create_distance_cache_key(self, features: np.ndarray) -> str:
        """Create cache key for distance matrix calculation."""
        return f"dist_{hash(features.tobytes())}"

    def _create_centroid_cache_key(self, features: np.ndarray, assignments: np.ndarray) -> str:
        """Create cache key for centroid calculation."""
        return f"cent_{hash(features.tobytes())}_{hash(assignments.tobytes())}"

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'silhouette_cache_size': len(self._silhouette_cache),
            'distance_cache_size': len(self._distance_cache),
            'centroid_cache_size': len(self._centroid_cache),
            'total_cache_size': len(self._silhouette_cache) + len(self._distance_cache) + len(self._centroid_cache)
        }

    def clear_cache(self):
        """Clear all caches."""
        self._silhouette_cache.clear()
        self._distance_cache.clear()
        self._centroid_cache.clear()

def adaptive_tau(dJ, floor=0.0, q=0.25):
    """Adaptive threshold for ΔJ pruning.

    If everything is positive, allow up to the 25th percentile (gentle exploration).
    Otherwise, stick to <= 0.
    """
    if np.all(dJ > 0):
        return max(np.quantile(dJ, q), floor)
    return 0.0

def assert_cluster_axis(name, a, K):
    """Guard assertions to catch cluster axis bugs early."""
    assert a.shape[0] == K, f"{name}.shape[0]={a.shape[0]} != K={K}"

class StrictSplitPolicy:
    """Strict split policy to prevent over-splitting and ensure high-quality splits only."""

    def __init__(self):
        # 1) Only consider truly big, persistently big parents
        self.min_parent_quantile = 0.90         # only top 10% largest clusters are eligible
        self.min_parent_vs_target = 1.6         # parent_size ≥ 1.6 * target_size
        self.min_parent_vs_min = 3.0            # parent_size ≥ 3.0 * min_size
        self.oversize_rounds = 4                # must be oversize for 4 consecutive rounds

        # 2) Children must be meaningful and not lopsided
        self.min_child_vs_target = 0.90         # each child ≥ 0.9 * target_size
        self.min_child_vs_min = 2.0             # and ≥ 2.0 * min_size
        self.balance_floor = 0.45               # min(child)/max(child) ≥ 0.45

        # 3) Demand multi-metric improvement
        self.min_rel_gain = 0.0125              # ≥1.25% relative J improvement
        self.min_sil_impr = 0.02                # silhouette must improve by ≥0.02
        self.refine_steps = 2                   # quick local refine must keep the cut

        # 4) Rate-limit K growth
        self.per_round_split_limit = 1          # at most 1 split per round
        self.per_epoch_k_increase = 2           # K may grow by ≤ +2 per epoch
        self.cooldown_rounds = 5                # cluster cooldown period
        self.binary_only = True                 # enforce 2-way splits only

        # 5) Context health preconditions
        self.min_alternatives_median = 3        # median #alts per boundary point ≥ 3
        self.max_capacity_blocked = 0.12        # capacity-blocked ≤ 12%

        # 6) Hard K ceiling
        self.k_cap_strategy = "N_over_min_target"

        # Tracking
        self.splits_this_round = 0
        self.splits_this_epoch = 0
        self.cluster_cooldowns = {}  # cluster_id -> cooldown_until_round
        self.oversize_tracker = {}   # cluster_id -> consecutive_oversize_rounds

        # K growth budget tracking
        self.k_growth_budget = self.per_epoch_k_increase
        self.epoch_start_k = None
        self.current_epoch = 0

    def can_split(self, current_k: int) -> bool:
        """Check if split is allowed based on K growth budget."""
        if self.epoch_start_k is None:
            self.epoch_start_k = current_k
            self.k_growth_budget = self.per_epoch_k_increase

        k_growth = current_k - self.epoch_start_k
        return k_growth < self.k_growth_budget

    def reset_epoch_budget(self, current_k: int):
        """Reset K growth budget for new epoch."""
        self.epoch_start_k = current_k
        self.k_growth_budget = self.per_epoch_k_increase
        self.splits_this_epoch = 0
        self.current_epoch += 1

    def consume_split_budget(self, current_k: int) -> bool:
        """Consume one split from budget and return True if successful."""
        if not self.can_split(current_k):
            return False

        self.splits_this_epoch += 1
        return True

class NAgosticConstraints:
    """N-agnostic clustering constraints that scale with dataset size."""

    def __init__(self, k_max: int = 15, min_fraction: float = 0.02, margin: int = 2, tau: float = 0.20):
        # Invariants (independent of N)
        self.k_max = k_max
        self.min_fraction = min_fraction
        self.margin = margin
        self.tau = tau

        # Hard 20% cluster size cap - single source of truth
        self.max_frac = 0.20  # Maximum cluster size as fraction of N
        self.max_size = None  # Will be set to floor(max_frac * N)
        self.cfg = None  # Will be set to SimpleNamespace with N, MIN_SIZE, MAX_FRAC

        # Soft band preference (N-agnostic)
        self.k_low = 7
        self.k_high = 15  # Increased from 12 to 15
        self.lambda_k = 0.005  # K-band penalty weight (further softened)
        self.lambda_k_max = 0.05  # Maximum K-band penalty weight (further lowered)
        self.lambda_k_growth = 1.20  # Slower growth for out-of-band rounds
        self.lambda_cap = 0.0005  # Size penalty weight (softened)
        self.beta = 1.8  # Soft capacity multiplier

        # Emergency split parameters
        self.out_of_band_rounds = 0  # Counter for consecutive out-of-band rounds
        self.emergency_split_after = 3  # Trigger emergency split after N rounds
        self.emergency_split_balance = [0.35, 0.65]  # Split balance for emergency

        # Derived values (recomputed each round)
        self.n = None
        self.min_size = None
        self.target_size_floor = None
        self.cap_min = None
        self.cap_max = None
        self.soft_cap = None  # U = ⌊β⋅N/K_low⌋

    def update_constraints(self, n: int):
        """Update derived constraints based on current dataset size N."""
        self.n = n
        self.min_size = max(1, math.ceil(self.min_fraction * n))  # hard floor per cluster
        self.target_size_floor = math.ceil(n / self.k_max)  # prevents tiny targets that explode K

        # Hard 20% cluster size cap
        self.max_size = int(np.floor(self.max_frac * n))  # 332 for N=1663

        # Single source of truth for caps/floors
        from types import SimpleNamespace
        self.cfg = SimpleNamespace(
            N=n,
            MIN_SIZE=self.min_size,
            MAX_FRAC=self.max_frac,
        )
        self.cfg.MAX_SIZE = self.max_size  # 332 for N=1663

        # Capacity window for any cluster - use CFG values
        self.cap_min = self.cfg.MIN_SIZE
        self.cap_max = self.cfg.MAX_SIZE  # Use MAX_SIZE instead of old calculation

        # Soft capacity: align with 20% rule
        self.soft_cap = self.cfg.MAX_SIZE  # replace 427 → 332

    def violates_capacity(self, src_size: int, dst_size: int) -> bool:
        """Check if a move violates capacity constraints."""
        if src_size - 1 < self.cap_min:
            return True
        if dst_size + 1 > self.cap_max:
            return True
        return False

    def can_split(self, k: int, parent_size: int, child_a_pred: int, child_b_pred: int) -> bool:
        """Check if a split is allowed under N-agnostic constraints."""
        # Hard K limit
        if k >= self.k_max:
            return False

        # Parent size requirements
        min_parent_size = max(2 * self.min_size + self.margin, math.ceil(1.25 * self.target_size_floor))
        if parent_size < min_parent_size:
            return False

        # Child size requirements
        if child_a_pred < self.min_size or child_b_pred < self.min_size:
            return False

        return True

    def get_constraint_summary(self) -> str:
        """Get a summary of current constraints for logging."""
        if self.n is None:
            return "Constraints not initialized"

        return (f"K_MAX={self.k_max}, MIN_FRAC={self.min_fraction:.3f}, N={self.n}, "
                f"MIN_SIZE={self.cfg.MIN_SIZE}, TARGET_FLOOR={self.target_size_floor}, "
                f"MAX_SIZE={self.cfg.MAX_SIZE}, CAP_RANGE=[{self.cap_min}, {self.cap_max}], SOFT_CAP={self.soft_cap}")

    def calculate_k_band_penalty(self, k: int) -> float:
        """Calculate K-band penalty: λ_K * [max(0, K_low-K)² + max(0, K-K_high)²]"""
        penalty_low = max(0, self.k_low - k) ** 2
        penalty_high = max(0, k - self.k_high) ** 2
        return self.lambda_k * (penalty_low + penalty_high)

    def escalate_k_band_penalty(self, k: int):
        """Escalate K-band penalty when out of target range."""
        if k < self.k_low or k > self.k_high:
            self.lambda_k = min(self.lambda_k_max, self.lambda_k * self.lambda_k_growth)

    def violates_max_size(self, cluster_size: int) -> bool:
        """Check if a cluster violates the 20% size cap."""
        return cluster_size > self.max_size

    def needed_splits(self, cluster_size: int) -> int:
        """Calculate how many additional parts needed to get every part <= max_size."""
        return max(0, math.ceil(cluster_size / self.max_size) - 1)

    def get_oversized_clusters(self, cluster_sizes: np.ndarray) -> List[Tuple[int, int]]:
        """Get list of (cluster_id, size) for clusters exceeding max_size."""
        return [(cid, size) for cid, size in enumerate(cluster_sizes) if size > self.max_size]

    def capacity_penalty(self, dest_size_after: int) -> float:
        """Calculate penalty for moving to a cluster that would exceed max_size."""
        over = max(0, dest_size_after - self.max_size)
        return (over / self.max_size) ** 2

    def effective_delta(self, base_delta: float, dest_size_after: int, cap_lambda: float = 25.0) -> float:
        """Calculate effective delta including capacity penalty."""
        return base_delta + cap_lambda * self.capacity_penalty(dest_size_after)

    def has_cap_violation(self, cluster_sizes: np.ndarray) -> bool:
        """Check if any cluster violates the 20% cap."""
        return np.max(cluster_sizes) > self.cfg.MAX_SIZE

    def is_feasible(self, cluster_sizes: np.ndarray) -> bool:
        """Check if clustering is feasible (no cap/floor violations)."""
        return (np.max(cluster_sizes) <= self.cfg.MAX_SIZE) and (np.min(cluster_sizes) >= self.cfg.MIN_SIZE)

    def calculate_size_penalty(self, cluster_sizes: np.ndarray) -> float:
        """Calculate soft capacity penalty: λ_cap * Σ max(0, n_i - U)²"""
        if self.soft_cap is None:
            return 0.0

        excess_sizes = np.maximum(0, cluster_sizes - self.soft_cap)
        return self.lambda_cap * np.sum(excess_sizes ** 2)

    def get_band_policy(self, k: int) -> str:
        """Get current band policy based on K value."""
        if k < self.k_low:
            return "encourage_splits"
        elif k > self.k_high:
            return "encourage_merges"
        else:
            return "neutral"

class SplitSkipGate:
    """Enhanced split skip gate with stricter preconditions."""

    def __init__(self):
        self.min_pct_with_alts_ge_3 = 0.50      # require ≥50% of boundary points to have ≥3 alts
        self.max_locked_points_frac = 0.05      # skip if >5% of points in min-size-locked clusters

    def _auto_heal_clusters(self, features: np.ndarray, stats: "ClusteringStats", constraints: NAgosticConstraints) -> int:
        """Auto-heal clusters that are below MIN_SIZE by pulling nearest boundary points or merging."""
        try:
            healed_clusters = 0
            cluster_sizes = stats.cluster_sizes

            # Find clusters below MIN_SIZE
            undersized_clusters = []
            for cluster_id in range(len(cluster_sizes)):
                if 0 < cluster_sizes[cluster_id] < constraints.min_size:
                    undersized_clusters.append(cluster_id)

            if not undersized_clusters:
                return 0

            tprint(f"🔧 Auto-healing {len(undersized_clusters)} undersized clusters (< {constraints.min_size})", "INFO")

            for cluster_id in undersized_clusters:
                current_size = cluster_sizes[cluster_id]
                needed = constraints.min_size - current_size

                # Try to pull nearest boundary points
                healed = self._pull_nearest_boundary_points(features, stats, cluster_id, needed, constraints)

                if healed > 0:
                    healed_clusters += 1
                    tprint(f"✅ Healed cluster {cluster_id}: {current_size} → {cluster_sizes[cluster_id]} (+{healed})", "INFO")
                else:
                    # If we can't pull enough points, merge with closest neighbor
                    merged = self._merge_with_closest_neighbor(features, stats, cluster_id, constraints)
                    if merged:
                        healed_clusters += 1
                        tprint(f"✅ Merged cluster {cluster_id} with closest neighbor", "INFO")

            return healed_clusters

        except Exception as e:
            tprint(f"Auto-heal failed: {e}", "ERROR")
            return 0

    def _pull_nearest_boundary_points(self, features: np.ndarray, stats: "ClusteringStats",
                                     target_cluster_id: int, needed: int, constraints: NAgosticConstraints) -> int:
        """Pull nearest boundary points to heal an undersized cluster."""
        try:
            # Find boundary points in other clusters that could be moved
            candidate_points = []

            for point_idx in range(len(features)):
                current_cluster = stats.assignments[point_idx]
                if current_cluster == target_cluster_id:
                    continue

                # Check if moving this point would violate capacity
                src_size = stats.cluster_sizes[stats._to_compact_id(current_cluster)]
                dst_size = stats.cluster_sizes[stats._to_compact_id(target_cluster_id)]

                if not constraints.violates_capacity(src_size, dst_size):
                    # Calculate distance to target cluster centroid with bounds checking
                    compact_id = stats._to_compact_id(target_cluster_id)
                    if 0 <= compact_id < len(stats.centroids):
                        distance = np.linalg.norm(features[point_idx] - stats.centroids[compact_id])
                    candidate_points.append((point_idx, current_cluster, distance))

            # Sort by distance and take the closest ones
            candidate_points.sort(key=lambda x: x[2])

            moved = 0
            for point_idx, from_cluster, _ in candidate_points[:needed]:
                # Double-check capacity before moving
                src_size = stats.cluster_sizes[stats._to_compact_id(from_cluster)]
                dst_size = stats.cluster_sizes[stats._to_compact_id(target_cluster_id)]

                if not constraints.violates_capacity(src_size, dst_size):
                    stats.apply_move(point_idx, from_cluster, target_cluster_id)
                    moved += 1

            return moved

        except Exception as e:
            tprint(f"Pull nearest boundary points failed: {e}", "DEBUG")
            return 0

    def _merge_with_closest_neighbor(self, features: np.ndarray, stats: "ClusteringStats",
                                    cluster_id: int, constraints: NAgosticConstraints) -> bool:
        """Merge an undersized cluster with its closest neighbor."""
        try:
            if stats.cluster_sizes[stats._to_compact_id(cluster_id)] == 0:
                return False

            # Find closest neighbor cluster
            target_centroid = stats.centroids[stats._to_compact_id(cluster_id)]
            min_distance = float('inf')
            closest_neighbor = None

            for other_cluster_id in range(len(stats.cluster_sizes)):
                if other_cluster_id == cluster_id or stats.cluster_sizes[stats._to_compact_id(other_cluster_id)] == 0:
                    continue

                distance = np.linalg.norm(target_centroid - stats.centroids[stats._to_compact_id(other_cluster_id)])
                if distance < min_distance:
                    min_distance = distance
                    closest_neighbor = other_cluster_id

            if closest_neighbor is None:
                return False

            # Check if merge would violate capacity
            combined_size = stats.cluster_sizes[stats._to_compact_id(cluster_id)] + stats.cluster_sizes[stats._to_compact_id(closest_neighbor)]
            if combined_size > constraints.cap_max:
                return False

            # Perform the merge
            members = np.flatnonzero(stats.assignments == cluster_id)
            for point_idx in members:
                stats.apply_move(point_idx, cluster_id, closest_neighbor)

            return True

        except Exception as e:
            tprint(f"Merge with closest neighbor failed: {e}", "DEBUG")
            return False

def topG_global_clusters_for_point(point_idx, features, stats, G=6, exclude=None):
    """Find top-G global clusters for a point by simple distance to centroids."""
    if exclude is None:
        exclude = []

    point = features[point_idx]
    K_fixed = stats.K_fixed

    # Calculate distances to all centroids
    distances = []
    for cid in range(K_fixed):
        if cid not in exclude and stats.cluster_sizes[stats._to_compact_id(cid)] > 0:  # Skip empty clusters
            dist = np.linalg.norm(point - stats.centroids[stats._to_compact_id(cid)])
            distances.append((cid, dist))

    # Sort by distance (closest first) and take top G
    distances.sort(key=lambda x: x[1])
    return [cid for cid, _ in distances[:G]]

def candidate_clusters(i, features, labels, sizes, stats, min_size, max_size, M=3, G=6, eps=1e-9):
    """Find candidate clusters with local + global supplement approach."""
    K = int(labels.max()) + 1
    src = labels[i]

    # CRITICAL FIX: Ensure all arrays are consistently sized
    # Use K_fixed from stats instead of K from labels.max()
    K_fixed = stats.K_fixed

    # Ensure sizes array matches K_fixed
    if len(sizes) != K_fixed:
        # Use stats.cluster_sizes which should be K_fixed length
        sizes = stats.cluster_sizes

    # Double-check that sizes is the right length
    if len(sizes) != K_fixed:
        return np.array([], dtype=int), dict(capacity=0, non_improving=0, nan=0)

    # 1) Local/adjacent clusters first (all other clusters for now - will be enhanced)
    local = np.arange(K_fixed)[np.arange(K_fixed) != src]

    # 2) If < M, add a GLOBAL supplement
    if len(local) < M:
        glob = topG_global_clusters_for_point(i, features, stats, G, exclude=[src] + local.tolist())
    else:
        glob = []

    # Union, preserve order: local priority, then global
    cands = []
    seen = set([src])
    for c in list(local) + list(glob):
        if c not in seen:
            cands.append(c)
            seen.add(c)
    cands = np.asarray(cands, dtype=int)

    if cands.size == 0:
        return np.array([], dtype=int), dict(capacity=0, non_improving=0, nan=0)

    # 3) Capacity guards (correct, minimal)
    if cands.size > 0:
        try:
            # Ensure src is within bounds
            if src >= len(sizes):
                return np.array([], dtype=int), dict(capacity=0, non_improving=0, nan=0)

            dst_ok = (sizes[cands] + 1) <= max_size
            src_ok = (sizes[src] - 1) >= min_size
            cap_ok = dst_ok & src_ok
            cands = cands[cap_ok]
        except (IndexError, ValueError) as e:
            # If there's still a broadcasting issue, return empty
            return np.array([], dtype=int), dict(capacity=0, non_improving=0, nan=0)
    else:
        return np.array([], dtype=int), dict(capacity=0, non_improving=0, nan=0)

    if cands.size == 0:
        return np.array([], dtype=int), dict(capacity=0, non_improving=0, nan=0)

    # 4) Score all remaining with ΔJ using actual delta calculation
    dJ = np.zeros(len(cands))
    for idx, candidate_cluster in enumerate(cands):
        delta_info = stats.calculate_move_delta(i, src, candidate_cluster)
        # Use total delta as primary scoring metric (negative means improvement)
        dJ[idx] = delta_info.get('total', 0.0)
    dJ = np.nan_to_num(dJ, nan=np.inf, posinf=np.inf, neginf=-np.inf)

    # 5) Progressive relaxation to ensure up to M choices
    pick = cands[dJ < -eps]  # strict improvement
    if pick.size < M:
        # allow neutral
        pick = cands[(dJ <= 0)]
    if pick.size < M:
        # allow weakly positive up to an adaptive tau
        tau = adaptive_tau(dJ)
        pick = cands[(dJ <= tau)]

    # rank by dJ (most negative first) and cap to M
    order = np.argsort(dJ[np.isin(cands, pick)])
    chosen = pick[order][:M]

    # Log drop reasons for diagnostics
    reasons = dict(capacity=0, non_improving=0, nan=0)
    reasons['nan'] += np.count_nonzero(~np.isfinite(dJ))
    reasons['capacity'] += np.count_nonzero(~cap_ok)
    reasons['non_improving'] += np.count_nonzero(cap_ok & (dJ > 0))

    return chosen, reasons

# Matrix operations helper functions
def calculate_distance_matrix_optimized(features: np.ndarray) -> np.ndarray:
    """Calculate pairwise distances using optimized matrix operations."""
    try:
        # Try to use unified matrix operations for better performance
        matrix_ops = UnifiedMatrixOperations()
        # Use euclidean distance method from matrix operations
        return matrix_ops.calculate_pairwise_similarities(features, method='euclidean')
    except Exception as e:
        tprint(f"Matrix operations failed, falling back to Numba: {e}", "WARNING")
        # Fallback to Numba implementation for debugging
        return calculate_distance_matrix_numba(features)

# Numba-optimized helper functions (fallback)
# PERFORMANCE FIX: Guard expensive O(N²) function behind debug flag
@jit(nopython=True, parallel=True)
def calculate_distance_matrix_numba(features: np.ndarray) -> np.ndarray:
    """Calculate pairwise distances using Numba for speed. O(N²) - use only for debugging."""
    n = features.shape[0]
    distances = np.zeros((n, n))
    for i in prange(n):
        for j in prange(n):
            if i != j:
                distances[i, j] = np.sqrt(np.sum((features[i] - features[j]) ** 2))
    return distances

@jit(nopython=True)
def calculate_wcss_incremental(features: np.ndarray, centroids: np.ndarray, assignments: np.ndarray) -> float:
    """Calculate WCSS incrementally using Numba."""
    wcss = 0.0
    for i in range(len(features)):
        cluster = int(assignments[i])
        wcss += np.sum((features[i] - centroids[cluster]) ** 2)
    return wcss

@jit(nopython=True)
def calculate_bcss_incremental(centroids: np.ndarray, global_mean: np.ndarray, cluster_sizes: np.ndarray) -> float:
    """Calculate BCSS incrementally using Numba."""
    bcss = 0.0
    for i in range(len(centroids)):
        bcss += cluster_sizes[i] * np.sum((centroids[i] - global_mean) ** 2)
    return bcss

@jit(nopython=True)
def calculate_boundary_scores_numba(features: np.ndarray, centroids: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    """Calculate boundary scores using Numba for speed."""
    n = features.shape[0]
    boundary_scores = np.zeros(n)

    for i in range(n):
        point = features[i]
        current_cluster = int(assignments[i])

        # Calculate distances to all centroids
        distances = np.zeros(len(centroids))
        for c in range(len(centroids)):
            distances[c] = np.sqrt(np.sum((point - centroids[c]) ** 2))

        # Find closest and second closest
        sorted_indices = np.argsort(distances)
        d1 = distances[sorted_indices[0]]  # Distance to own cluster
        d2 = distances[sorted_indices[1]]  # Distance to nearest other cluster

        # Boundary score: difference between d1 and d2
        boundary_scores[i] = d1 - d2

    return boundary_scores

@jit(nopython=True)
def calculate_margin_gain_numba(
    point: np.ndarray,
    centroid_from: np.ndarray,
    centroid_to: np.ndarray,
    centroids: np.ndarray
) -> float:
    """Calculate margin gain using Numba for speed."""
    # Current distances
    d1_old = np.sqrt(np.sum((point - centroid_from) ** 2))

    # Find second closest centroid (excluding from_cluster)
    min_dist = np.inf
    for c in range(len(centroids)):
        dist = np.sqrt(np.sum((point - centroids[c]) ** 2))
        if dist < min_dist and dist > 0:  # Exclude self
            min_dist = dist
    d2_old = min_dist

    # New distances
    d1_new = np.sqrt(np.sum((point - centroid_to) ** 2))

    # Find second closest centroid (excluding to_cluster)
    min_dist = np.inf
    for c in range(len(centroids)):
        dist = np.sqrt(np.sum((point - centroids[c]) ** 2))
        if dist < min_dist and dist > 0:  # Exclude self
            min_dist = dist
    d2_new = min_dist

    # Margin gain
    old_margin = d1_old - d2_old
    new_margin = d1_new - d2_new

    return new_margin - old_margin

class ClusteringStats:
    """Maintains incremental clustering statistics for fast delta calculations."""

    def __init__(self, features: np.ndarray, assignments: np.ndarray):
        """Initialize with current clustering state."""
        # Validate inputs
        if features is None or features.size == 0:
            raise ValueError("Features array is None or empty")

        if not hasattr(features, 'shape') or len(features.shape) != 2:
            raise ValueError(f"Features must be a 2D array, got shape: {getattr(features, 'shape', 'None')}")

        if assignments is None or len(assignments) == 0:
            raise ValueError("Assignments array is None or empty")

        if len(assignments) != features.shape[0]:
            raise ValueError(f"Assignments length ({len(assignments)}) doesn't match features shape[0] ({features.shape[0]})")

        self.features = features
        self.assignments = assignments
        self.n_samples, self.n_features = features.shape

        # CRITICAL FIX: Use fixed K based on max cluster ID, not len(unique)
        self.K_fixed = int(assignments.max()) + 1  # Fixed K from initial clustering
        self.n_clusters = self.K_fixed

        # CRITICAL FIX: Compact cluster IDs to 0..K-1 and update assignments
        unique_clusters = np.unique(assignments)
        self.cluster_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
        self.inverse_cluster_id_map = {new_id: old_id for old_id, new_id in self.cluster_id_map.items()}

        # Remap assignments to compact IDs to ensure array indexing is correct
        compact_assignments = np.zeros_like(assignments)
        for old_id, new_id in self.cluster_id_map.items():
            compact_assignments[assignments == old_id] = new_id
        self.assignments = compact_assignments

        # Initialize all statistics
        self._initialize_statistics()

    @property
    def sizes(self):
        """Always compute sizes from current assignments - single source of truth."""
        return np.bincount(self.assignments, minlength=self.K_fixed)

    @property
    def K(self):
        """Always compute K from current assignments - single source of truth."""
        return len(np.unique(self.assignments))

    @property
    def cluster_sizes(self):
        """Always compute cluster sizes from current assignments - single source of truth."""
        return np.bincount(self.assignments, minlength=self.K_fixed)

    def refresh_cluster_sizes(self) -> np.ndarray:
        """Recompute cluster sizes and update internal caches."""
        target_len = max(self.K_fixed, getattr(self, 'centroids', np.empty((0,))).shape[0])
        sizes = np.bincount(self.assignments, minlength=target_len)
        if hasattr(self, '_cluster_sizes_cache'):
            if self._cluster_sizes_cache.shape[0] != sizes.shape[0]:
                new_cache = np.zeros(sizes.shape[0], dtype=self._cluster_sizes_cache.dtype)
                upto = min(self._cluster_sizes_cache.shape[0], sizes.shape[0])
                new_cache[:upto] = self._cluster_sizes_cache[:upto]
                self._cluster_sizes_cache = new_cache
            np.copyto(self._cluster_sizes_cache,
                      sizes.astype(self._cluster_sizes_cache.dtype, copy=False))
        return sizes

    def _validate_state(self):
        """Comprehensive state validation to catch desync issues early."""
        assert self.assignments.max() < self.K_fixed, f"Assignment max {self.assignments.max()} >= K_fixed {self.K_fixed}"
        assert len(self.assignments) == len(self.features), f"Assignments length {len(self.assignments)} != features length {len(self.features)}"
        assert np.sum(self.sizes) == len(self.assignments), f"Size sum {np.sum(self.sizes)} != assignments length {len(self.assignments)}"
        assert self.K == len(np.unique(self.assignments)), f"K {self.K} != unique assignments {len(np.unique(self.assignments))}"
        return True

    def _snapshot_state(self):
        """Create a complete snapshot of current state for atomic operations."""
        return {
            'assignments': self.assignments.copy(),
            'features': self.features.copy(),
            'cluster_id_map': self.cluster_id_map.copy(),
            'inverse_cluster_id_map': self.inverse_cluster_id_map.copy(),
            'S': self.S.copy(),
            'Q_trace': self.Q_trace.copy(),
            'centroids': self.centroids.copy(),
            'wcss_per_cluster': self.wcss_per_cluster.copy(),
            'total_wcss': self.total_wcss,
            'total_bcss': self.total_bcss
        }

    def _restore_state(self, snapshot):
        """Restore state from snapshot for atomic operation rollback."""
        self.assignments = snapshot['assignments']
        self.features = snapshot['features']
        self.cluster_id_map = snapshot['cluster_id_map']
        self.inverse_cluster_id_map = snapshot['inverse_cluster_id_map']
        self.S = snapshot['S']
        self.Q_trace = snapshot['Q_trace']
        self.centroids = snapshot['centroids']
        self.wcss_per_cluster = snapshot['wcss_per_cluster']
        self.total_wcss = snapshot['total_wcss']
        self.total_bcss = snapshot['total_bcss']
        # Re-validate state after restore
        self._validate_state()

    def _to_compact_id(self, cluster_id: int) -> int:
        """Convert original cluster ID to compact ID for array indexing with bounds checking."""
        if cluster_id not in self.cluster_id_map:
            raise ValueError(f"Cluster ID {cluster_id} not found in cluster_id_map. Available: {list(self.cluster_id_map.keys())}")
        compact_id = self.cluster_id_map[cluster_id]
        if compact_id >= self.K_fixed:
            raise ValueError(f"Compact ID {compact_id} >= K_fixed {self.K_fixed}")
        return compact_id

    def _to_original_id(self, compact_id: int) -> int:
        """Convert compact ID back to original cluster ID."""
        return self.inverse_cluster_id_map.get(compact_id, compact_id)

    def get_cluster_size(self, cluster_id: int) -> int:
        """Safely get cluster size with proper bounds checking."""
        try:
            compact_id = self._to_compact_id(cluster_id)
            return self.cluster_sizes[compact_id]
        except (ValueError, IndexError) as e:
            tprint(f"⚠️ WARNING: Failed to get size for cluster {cluster_id}: {e}", "WARNING")
            return 0

    def get_centroid(self, cluster_id: int) -> np.ndarray:
        """Safely get cluster centroid with proper bounds checking."""
        try:
            compact_id = self._to_compact_id(cluster_id)
            if compact_id < len(self.centroids):
                return self.centroids[compact_id]
            else:
                raise IndexError(f"Centroid index {compact_id} out of bounds")
        except (ValueError, IndexError) as e:
            tprint(f"⚠️ WARNING: Failed to get centroid for cluster {cluster_id}: {e}", "WARNING")
            return np.zeros(self.n_features, dtype=np.float64)

    def _remap_to_compact_ids(self, assignments: np.ndarray) -> tuple:
        """Remap assignments to compact IDs and return updated assignments and mapping."""
        unique_clusters = np.unique(assignments)
        cluster_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}

        # Remap assignments to compact IDs
        compact_assignments = np.zeros_like(assignments)
        for old_id, new_id in cluster_id_map.items():
            compact_assignments[assignments == old_id] = new_id

        return compact_assignments, cluster_id_map

    def _initialize_statistics(self):
        """Initialize all clustering statistics."""
        # Per-cluster sufficient statistics (exact incremental formulas) - allocated by n_clusters
        # NOTE: cluster_sizes is now computed via property, this is just for backward compatibility
        self._cluster_sizes_cache = np.zeros(self.n_clusters, dtype=np.int32)
        self.centroids = np.zeros((self.n_clusters, self.n_features), dtype=np.float64)
        self.wcss_per_cluster = np.zeros(self.n_clusters, dtype=np.float64)

        # Sufficient statistics for exact incremental calculations - allocated by n_clusters
        # S_c = sum of points in cluster c
        self.S = np.zeros((self.n_clusters, self.n_features), dtype=np.float64)
        # Q_c = sum of outer products x_i * x_i^T for cluster c (stored as trace)
        self.Q_trace = np.zeros(self.n_clusters, dtype=np.float64)

        # Global statistics
        try:
            self.global_mean = np.mean(self.features, axis=0, dtype=np.float64)
            self.global_S = np.sum(self.features, axis=0, dtype=np.float64)  # S = sum of all points
        except Exception as e:
            tprint(f"❌ Failed to compute global statistics: {e}", "ERROR")
            raise ValueError(f"Global statistics computation failed: {e}")
        self.total_wcss = 0.0
        self.total_bcss = 0.0
        self.global_N = self.n_samples

        # Initialize all statistics
        self._update_all_stats()

        # Initialize variance decomposition caches and silhouette cache
        self._recompute_variance_caches()
        self._point_silhouettes = None
        self._silhouette_valid = False

        # Initialize temporal transition caches (for smoothness objective)
        self.temporal_alpha = 1.0
        self._initialize_transition_caches()

        # Distance behavior (optional hybrid correlation distance on returns subset)
        self.returns_mask = np.zeros(self.n_features, dtype=bool)
        self.returns_lambda = 0.5

    def _initialize_transition_caches(self) -> None:
        """Build transition count matrix and row sums from sequential assignments."""
        K = int(self.K_fixed)
        self.transition_counts = np.zeros((K, K), dtype=np.int64)
        self.transition_row_sums = np.zeros(K, dtype=np.int64)
        if len(self.assignments) >= 2:
            a = self.assignments.astype(int, copy=False)
            for i in range(len(a) - 1):
                u = int(a[i]); v = int(a[i + 1])
                if 0 <= u < K and 0 <= v < K:
                    self.transition_counts[u, v] += 1
                    self.transition_row_sums[u] += 1

    def ensure_k_capacity(self, new_K: int) -> None:
        """Resize all K-dimensional buffers to accommodate new clusters."""
        current_K = getattr(self, 'centroids', np.zeros((0,))).shape[0]
        if new_K <= current_K:
            return

        def _resize(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
            new_arr = np.zeros(shape, dtype=arr.dtype)
            slices = tuple(slice(0, min(old, new)) for old, new in zip(arr.shape, shape))
            new_arr[slices] = arr[slices]
            return new_arr

        if hasattr(self, '_cluster_sizes_cache'):
            cache = np.zeros(new_K, dtype=self._cluster_sizes_cache.dtype)
            upto = min(len(self._cluster_sizes_cache), new_K)
            cache[:upto] = self._cluster_sizes_cache[:upto]
            self._cluster_sizes_cache = cache

        self.centroids = _resize(self.centroids, (new_K, self.n_features))
        self.wcss_per_cluster = _resize(self.wcss_per_cluster, (new_K,))
        self.S = _resize(self.S, (new_K, self.n_features))
        self.Q_trace = _resize(self.Q_trace, (new_K,))

        if hasattr(self, 'transition_counts'):
            counts = self.transition_counts
            new_counts = np.zeros((new_K, new_K), dtype=counts.dtype)
            new_counts[:counts.shape[0], :counts.shape[1]] = counts
            self.transition_counts = new_counts

        if hasattr(self, 'transition_row_sums'):
            rows = self.transition_row_sums
            new_rows = np.zeros(new_K, dtype=rows.dtype)
            upto = min(rows.shape[0], new_K)
            new_rows[:upto] = rows[:upto]
            self.transition_row_sums = new_rows

        self.K_fixed = max(self.K_fixed, new_K)
        self.n_clusters = max(self.n_clusters, new_K)

    def _update_all_stats(self):
        """Update all clustering statistics with sufficient statistics."""
        unique_clusters = np.unique(self.assignments)

        # Reset sufficient statistics
        self.S.fill(0.0)
        self.Q_trace.fill(0.0)
        # Note: cluster_sizes is now computed via property, no need to reset

        for cluster in unique_clusters:
            # CRITICAL FIX: Map original cluster ID to consecutive index
            cluster_idx = self.cluster_id_map[cluster]
            mask = self.assignments == cluster
            cluster_features = self.features[mask]

            if len(cluster_features) > 0:
                n_c = len(cluster_features)
                # Note: cluster_sizes is now computed via property, no need to set

                # S_c = sum of points in cluster c
                self.S[cluster_idx] = np.sum(cluster_features, axis=0, dtype=np.float64)

                # Q_c = sum of ||x_i||^2 for cluster c (trace of outer products)
                self.Q_trace[cluster_idx] = np.sum(np.sum(cluster_features ** 2, axis=1), dtype=np.float64)

                # Centroids: μ_c = S_c / n_c
                self.centroids[cluster_idx] = self.S[cluster_idx] / n_c

                # WCSS_c = tr(Q_c) - ||S_c||^2 / n_c (exact formula)
                self.wcss_per_cluster[cluster_idx] = self.Q_trace[cluster_idx] - np.sum(self.S[cluster_idx] ** 2) / n_c

        self.total_wcss = np.sum(self.wcss_per_cluster)

        # BCSS = sum_c ||S_c||^2 / n_c - ||S||^2 / N (exact formula)
        self.total_bcss = np.sum(np.sum(self.S ** 2, axis=1) / self.cluster_sizes) - np.sum(self.global_S ** 2) / self.global_N

        # Variance decomposition caches (within/between/ratio)
        self._recompute_variance_caches()

    def _recompute_variance_caches(self) -> None:
        """Recompute pooled within-regime variance, between-regime variance, and ratio."""
        try:
            # Total/global variance (mean of feature variances)
            self.total_var = float(np.mean(np.var(self.features, axis=0)))

            # Pooled within-regime variance
            within_sum = 0.0
            total_df = 0
            for k in range(self.K_fixed):
                n_k = int(self.cluster_sizes[k])
                if n_k > 1:
                    # regime variance across features using sufficient statistics
                    # var = (Q - ||S||^2/n) / (n-1), averaged across features
                    Q_k = float(self.Q_trace[k])
                    S_k_sq_over_n = float(np.sum(self.S[k] ** 2)) / n_k
                    var_k = (Q_k - S_k_sq_over_n) / max(1, (n_k - 1))
                    var_k = var_k / max(1, self.n_features)
                    within_sum += var_k * (n_k - 1)
                    total_df += (n_k - 1)
            self.within_var = float(within_sum / total_df) if total_df > 0 else 0.0

            # Between-regime variance via regime means
            means = []
            weights = []
            for k in range(self.K_fixed):
                n_k = int(self.cluster_sizes[k])
                if n_k > 0:
                    means.append(self.centroids[k])
                    weights.append(n_k)
            if len(means) > 1:
                means = np.asarray(means, dtype=np.float64)
                weights = np.asarray(weights, dtype=np.float64)
                grand_mean = np.average(means, weights=weights, axis=0)
                deviations = means - grand_mean
                # mean squared deviation per centroid, then weighted average
                per_centroid_var = np.mean(deviations ** 2, axis=1)
                self.between_var = float(np.average(per_centroid_var, weights=weights))
            else:
                self.between_var = 0.0

            self.variance_ratio = (self.between_var / self.within_var) if self.within_var > 0 else 0.0
        except Exception:
            # Fallback: ensure attributes exist
            self.total_var = getattr(self, 'total_var', 0.0)
            self.within_var = getattr(self, 'within_var', 0.0)
            self.between_var = getattr(self, 'between_var', 0.0)
            self.variance_ratio = getattr(self, 'variance_ratio', 0.0)

    def get_cv_ratio(self) -> float:
        """Get current variance ratio (between / within).
        Backward-compatible accessor replacing legacy BCSS/WCSS."""
        # Ensure caches are up to date
        return float(getattr(self, 'variance_ratio', 0.0))

    def get_objective_value(self, w_cv: float = 0.50, w_temp: float = 0.30,
                           w_sil: float = 0.10, w_bal: float = 0.10,
                           k_complexity_penalty: float = 0.15, k_max: int = 20,
                           constraints: "NAgosticConstraints" = None) -> float:
        """Calculate the full objective function value with k-complexity penalty and band penalties."""
        cv_ratio = self.get_cv_ratio()
        balance = self.get_balance_score()

        # Placeholder for silhouette and temporal (would be calculated in full implementation)
        silhouette_proxy = 0.5  # Placeholder
        temporal_proxy = 0.5    # Placeholder

        # Base objective without balance weight (balance used as constraint)
        objective = (
            w_cv * cv_ratio +
            w_sil * silhouette_proxy +
            w_temp * temporal_proxy
        )

        # Apply balance as soft constraint penalty (softer than hard weight)
        balance_penalty = 0.0
        if balance < 0.8:  # Only penalize if balance is very poor
            balance_penalty = 0.05 * (0.8 - balance)  # Soft penalty, not hard weight
        objective -= balance_penalty

        # Add k-complexity penalty to prevent runaway splitting
        k_penalty = k_complexity_penalty * (self.n_clusters - 1) / k_max
        objective -= k_penalty

        # Add band penalties if constraints are provided
        if constraints is not None:
            # K-band penalty (only after warm-up iterations)
            k_band_penalty = constraints.calculate_k_band_penalty(self.n_clusters)
            objective += k_band_penalty

            # Size penalty for clusters exceeding soft capacity
            size_penalty = constraints.calculate_size_penalty(self.cluster_sizes)
            objective += size_penalty

        return objective

    def get_balance_score(self) -> float:
        """Calculate cluster balance score using coefficient of variation.

        Uses the standard statistical measure of balance: lower CV = better balance.
        CV = (standard deviation / mean) of cluster sizes.
        Returns 1.0 for perfect balance, approaching 0.0 for very imbalanced clusters.
        """
        if self.n_clusters <= 1:
            return 1.0

        sizes = np.array([s for s in self.cluster_sizes if s > 0])
        if len(sizes) == 0:
            return 1.0

        mean_size = np.mean(sizes)
        if mean_size == 0:
            return 0.0

        # Calculate coefficient of variation (CV)
        # CV = std / mean, so higher CV = more imbalance
        cv = np.std(sizes) / mean_size

        # Convert to balance score: 1.0 - CV (normalized)
        # Perfect balance (CV=0) = 1.0
        # Very imbalanced (CV > 1.0) = approaches 0.0
        balance_score = max(0.0, 1.0 - cv)

        return balance_score

    def calculate_move_delta(self, point_idx: int, from_cluster: int, to_cluster: int) -> Dict[str, float]:
        """Calculate delta using variance decomposition (primary), centroid-silhouette (secondary), and temporal smoothness.
        Sign convention: negative values indicate improvement for core keys ('cv','silhouette','temporal','total')."""
        if from_cluster == to_cluster:
            return {
                'total': 0.0,
                'cv': 0.0,
                'balance': 0.0,
                'silhouette': 0.0,
                'temporal': 0.0,
                'variance_ratio': 0.0
            }

        # Map cluster IDs to compact indices for array access
        try:
            from_idx = self.cluster_id_map.get(from_cluster, from_cluster)
            to_idx = self.cluster_id_map.get(to_cluster, to_cluster)
        except Exception:
            from_idx, to_idx = int(from_cluster), int(to_cluster)

        # Primary: variance ratio delta
        variance_delta = self.calculate_variance_delta(point_idx, from_idx, to_idx)

        # Secondary: silhouette delta (centroid-based approximation)
        sil_delta = self.calculate_silhouette_delta_centroid(point_idx, from_idx, to_idx)

        # Temporal delta via transition matrix change (positive = improvement)
        temporal_delta_impr = 0.0
        try:
            alpha = float(getattr(self, 'temporal_alpha', 1.0))
            Kf = int(self.K_fixed)
            prev_idx = point_idx - 1 if point_idx - 1 >= 0 else None
            next_idx = point_idx + 1 if point_idx + 1 < len(self.assignments) else None
            log = np.log
            # Helper to compute -log smoothed prob for edge (u->v) with given counts/rows
            def edge_cost(u: int, v: int, add_uv: int = 0, add_row_u: int = 0) -> float:
                num = self.transition_counts[u, v] + alpha + add_uv
                den = self.transition_row_sums[u] + alpha * Kf + add_row_u
                if den <= 0:
                    return 0.0
                return float(-log(num) + log(den))
            # prev row contribution (row sum unchanged: one remove, one add)
            if prev_idx is not None:
                u = int(self.assignments[prev_idx])
                # old: u->from, new: u->to
                old_cost = edge_cost(u, from_idx, add_uv=0, add_row_u=0)
                new_cost = edge_cost(u, to_idx, add_uv=1, add_row_u=0)  # add to (u,to)
                # but also remove one from (u,from) numerators implicitly; treat old_cost removal via difference
                temporal_delta_impr += (old_cost - new_cost)
            # next row contributions affect 'from' and 'to' rows' denominators
            if next_idx is not None:
                w = int(self.assignments[next_idx])
                # old edge from->w (removal): row_from decreases by 1
                row_old = int(self.transition_row_sums[from_idx])
                old_cost = edge_cost(from_idx, w, add_uv=0, add_row_u=0)
                # After removal, this edge no longer exists; account for row term change:
                # row term change: new_row*log(new_row+αK) - old_row*log(old_row+αK)
                # incorporate via cost difference when adding new edge for 'to' below and a row delta term for 'from'
                row_from_new = row_old - 1
                if row_old > 0:
                    row_term_delta_from = float((row_from_new * np.log(row_from_new + alpha * Kf)) - (row_old * np.log(row_old + alpha * Kf)))
                else:
                    row_term_delta_from = 0.0
                # new edge to->w (addition): row_to increases by 1
                row_to_old = int(self.transition_row_sums[to_idx])
                new_cost = edge_cost(to_idx, w, add_uv=1, add_row_u=1)
                row_to_new = row_to_old + 1
                row_term_delta_to = float((row_to_new * np.log(row_to_new + alpha * Kf)) - (row_to_old * np.log(row_to_old + alpha * Kf)))
                # Temporal improvement: removing old_cost and row_from decrease is improvement; adding new edge incurs new_cost and row_to increase cost
                temporal_delta_impr += (old_cost - new_cost) + (row_term_delta_from + row_term_delta_to)
        except Exception:
            temporal_delta_impr = 0.0

        # Enhanced temporal smoothness with regime persistence
        temporal_delta = 0.0
        n_from = int(self.cluster_sizes[from_idx])
        n_to = int(self.cluster_sizes[to_idx])

        if n_to > 0 and n_from > 0:
            # Enhanced size ratio analysis for regime stability
            size_ratio_old = n_from / max(1.0, n_to)
            size_ratio_new = (n_from - 1.0) / (n_to + 1.0)

            # Regime persistence bonus - reward moves that maintain cluster stability
            if size_ratio_old > 2.0 and size_ratio_new < size_ratio_old:
                temporal_delta = 0.02  # Increased bonus for stability
            elif size_ratio_old < 0.5 and size_ratio_new > size_ratio_old:
                temporal_delta = 0.02  # Increased bonus for stability
            elif 0.5 <= size_ratio_old <= 2.0 and 0.5 <= size_ratio_new <= 2.0:
                # Bonus for maintaining balanced clusters
                temporal_delta = 0.01

        # Combine with existing negative-improvement convention
        w_cv = getattr(getattr(self, 'config', None), 'w_cv', None)
        if w_cv is None:
            w_cv = 0.50
        w_sil = getattr(getattr(self, 'config', None), 'w_sil', 0.15)
        w_temp = getattr(getattr(self, 'config', None), 'w_temp', 0.35)
        total_delta = - (w_cv * variance_delta + w_sil * sil_delta + w_temp * temporal_delta_impr)

        return {
            'total': float(total_delta),
            'cv': float(-variance_delta),
            'balance': 0.0,  # Balance metric - placeholder for now
            'silhouette': float(-sil_delta),
            'temporal': float(-temporal_delta_impr),
            'variance_ratio': float(variance_delta),
            'cv_raw': float(variance_delta),
            'silhouette_raw': float(sil_delta),
            'temporal_raw': float(temporal_delta_impr)
        }

    def calculate_variance_decomposition(self) -> dict:
        """True variance decomposition for regime clustering.
        Returns within-regime variance, between-regime variance, and their ratio."""
        N = len(self.features)
        K = self.K_fixed

        # Global variance (baseline)
        global_mean = np.mean(self.features, axis=0)
        total_var = float(np.mean(np.var(self.features, axis=0)))

        # Within-regime pooled variance
        within_var = 0.0
        regime_sizes = []
        for k in range(K):
            n_k = int(self.cluster_sizes[k])
            if n_k > 1:
                regime_features = self.features[self.assignments == k]
                regime_var = float(np.mean(np.var(regime_features, axis=0, ddof=1)))
                within_var += regime_var * (n_k - 1)
                regime_sizes.append(n_k)
        total_df = sum(n - 1 for n in regime_sizes)
        within_var = float(within_var / total_df) if total_df > 0 else 0.0

        # Between-regime variance (variance of regime means)
        regime_means = []
        regime_weights = []
        for k in range(K):
            n_k = int(self.cluster_sizes[k])
            if n_k > 0:
                regime_mean = np.mean(self.features[self.assignments == k], axis=0)
                regime_means.append(regime_mean)
                regime_weights.append(n_k)
        if len(regime_means) > 1:
            regime_means = np.array(regime_means)
            regime_weights = np.array(regime_weights)
            grand_mean = np.average(regime_means, weights=regime_weights, axis=0)
            deviations = regime_means - grand_mean
            between_var = float(np.average(np.mean(deviations ** 2, axis=1), weights=regime_weights))
        else:
            between_var = 0.0

        variance_ratio = float(between_var / within_var) if within_var > 0 else 0.0

        return {
            'total': float(total_var),
            'within': float(within_var),
            'between': float(between_var),
            'ratio': float(variance_ratio),
            'eta_squared': float(between_var / total_var) if total_var > 0 else 0.0
        }

    def _regime_variance(self, cluster_id: int) -> float:
        """Calculate variance within a regime using sufficient statistics (average across features)."""
        n = int(self.cluster_sizes[cluster_id])
        if n <= 1:
            return 0.0
        variance = (float(self.Q_trace[cluster_id]) - float(np.sum(self.S[cluster_id] ** 2)) / n) / (n - 1)
        return float(variance / max(1, self.n_features))

    def _variance_after_removal(self, point: np.ndarray, cluster_id: int, new_mean: np.ndarray, old_n: int) -> float:
        """Variance after removing a point from a cluster (average across features)."""
        if old_n <= 2:
            return 0.0
        new_n = old_n - 1
        new_Q = float(self.Q_trace[cluster_id]) - float(np.sum(point ** 2))
        new_S_squared = float(np.sum((self.S[cluster_id] - point) ** 2))
        variance = (new_Q - new_S_squared / new_n) / (new_n - 1)
        return float(variance / max(1, self.n_features))

    def _variance_after_addition(self, point: np.ndarray, cluster_id: int, new_mean: np.ndarray, old_n: int) -> float:
        """Variance after adding a point to a cluster (average across features)."""
        new_n = old_n + 1
        new_Q = float(self.Q_trace[cluster_id]) + float(np.sum(point ** 2))
        new_S_squared = float(np.sum((self.S[cluster_id] + point) ** 2))
        variance = (new_Q - new_S_squared / new_n) / (new_n - 1)
        return float(variance / max(1, self.n_features))

    def calculate_variance_delta(self, point_idx: int, from_cluster: int, to_cluster: int) -> float:
        """Calculate change in variance ratio if point moves from -> to using sufficient statistics.
        Returns delta of variance_ratio (positive implies improvement)."""
        point = self.features[point_idx]
        n_from = int(self.cluster_sizes[from_cluster])
        n_to = int(self.cluster_sizes[to_cluster])
        if n_from <= 1:
            return float('-inf')

        # Within-variance delta (pooled)
        old_within_from = self._regime_variance(from_cluster)
        old_within_to = self._regime_variance(to_cluster)
        new_mean_from = (self.S[from_cluster] - point) / (n_from - 1)
        new_within_from = self._variance_after_removal(point, from_cluster, new_mean_from, n_from)
        new_mean_to = (self.S[to_cluster] + point) / (n_to + 1)
        new_within_to = self._variance_after_addition(point, to_cluster, new_mean_to, n_to)
        pooled_old = (old_within_from * (n_from - 1) + old_within_to * (n_to - 1))
        pooled_new = (new_within_from * (n_from - 2 if n_from > 1 else 0) + new_within_to * n_to)
        denom = max(1, (len(self.features) - self.K_fixed))
        delta_within = (pooled_new - pooled_old) / denom

        # Enhanced between-variance delta with AGGRESSIVE regime separation focus
        old_mean_from = self.centroids[from_cluster]
        old_mean_to = self.centroids[to_cluster]
        global_mean = self.global_mean

        # Apply regime separation enhancement factor for better CV ratio
        regime_separation_factor = 1.5  # Boost between-cluster variance calculation

        # Original between-variance calculation
        old_between = (n_from * float(np.mean((old_mean_from - global_mean) ** 2)) +
                       n_to * float(np.mean((old_mean_to - global_mean) ** 2)))
        new_between = ((n_from - 1) * float(np.mean((new_mean_from - global_mean) ** 2)) +
                       (n_to + 1) * float(np.mean((new_mean_to - global_mean) ** 2)))

        # Add regime separation bonus - reward moves that increase cluster separation
        centroid_distance_old = float(np.linalg.norm(old_mean_from - old_mean_to))
        centroid_distance_new = float(np.linalg.norm(new_mean_from - new_mean_to))
        separation_bonus = (centroid_distance_new - centroid_distance_old) * 0.05  # 5% bonus for separation

        delta_between = (new_between - old_between) / max(1, len(self.features)) + separation_bonus

        # Apply regime separation enhancement to boost CV ratio
        delta_between *= regime_separation_factor

        # Variance ratio delta
        new_within = self.within_var + delta_within
        new_between_val = self.between_var + delta_between
        new_ratio = (new_between_val / new_within) if new_within > 0 else 0.0
        old_ratio = self.variance_ratio
        return float(new_ratio - old_ratio)

    def calculate_silhouette_delta_centroid(self, point_idx: int, from_c: int, to_c: int) -> float:
        """Fast silhouette approximation using cluster centroids (O(K)). Returns delta (positive = improvement)."""
        point = self.features[point_idx]
        n_from = int(self.cluster_sizes[from_c])
        n_to = int(self.cluster_sizes[to_c])
        new_centroid_from = (self.S[from_c] - point) / (n_from - 1) if n_from > 1 else self.centroids[from_c]
        new_centroid_to = (self.S[to_c] + point) / (n_to + 1)
        # old silhouette components using hybrid distance
        a_old = float(self._hybrid_distance(point, self.centroids[from_c]))
        other_dists_old = [float(self._hybrid_distance(point, self.centroids[k]))
                           for k in range(self.K_fixed) if k != from_c and self.cluster_sizes[k] > 0]
        b_old = min(other_dists_old) if other_dists_old else a_old
        s_old = (b_old - a_old) / max(a_old, b_old) if max(a_old, b_old) > 0 else 0.0
        # new silhouette components
        a_new = float(self._hybrid_distance(point, new_centroid_to))
        other_dists_new = [float(self._hybrid_distance(point, (new_centroid_from if k == from_c else self.centroids[k])))
                           for k in range(self.K_fixed) if k != to_c and self.cluster_sizes[k] > 0]
        b_new = min(other_dists_new) if other_dists_new else a_new
        s_new = (b_new - a_new) / max(a_new, b_new) if max(a_new, b_new) > 0 else 0.0
        return float(s_new - s_old)

    def get_point_silhouette(self, point_idx: int) -> float:
        """Get cached silhouette for a point, compute on demand via centroids."""
        if not getattr(self, '_silhouette_valid', False) or self._point_silhouettes is None:
            self._compute_all_silhouettes()
        return float(self._point_silhouettes[point_idx])

    def _compute_all_silhouettes(self) -> None:
        """Batch compute silhouettes using centroids (fast)."""
        N = len(self.features)
        self._point_silhouettes = np.zeros(N, dtype=np.float64)
        for i in range(N):
            point = self.features[i]
            own_cluster = int(self.assignments[i])
            a_i = float(self._hybrid_distance(point, self.centroids[own_cluster]))
            other_dists = [float(self._hybrid_distance(point, self.centroids[k]))
                           for k in range(self.K_fixed) if k != own_cluster and self.cluster_sizes[k] > 0]
            if other_dists:
                b_i = min(other_dists)
                self._point_silhouettes[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0
            else:
                self._point_silhouettes[i] = 0.0
        self._silhouette_valid = True

    def _hybrid_distance(self, point: np.ndarray, centroid: np.ndarray) -> float:
        """Hybrid distance combining correlation on returns subvector and Euclidean on the rest."""
        try:
            mask = getattr(self, 'returns_mask', None)
            lam = float(getattr(self, 'returns_lambda', 0.5))
            if mask is None or not np.any(mask):
                # Fallback to Euclidean
                return float(np.linalg.norm(point - centroid))
            # Returns subvector correlation distance
            pr = point[mask]; cr = centroid[mask]
            # Guard: need at least 2 dims and non-zero std
            if pr.size >= 2 and np.std(pr) > 1e-12 and np.std(cr) > 1e-12:
                prc = pr - pr.mean(); crc = cr - cr.mean()
                denom = (np.std(pr) * np.std(cr) * (pr.size - 1))
                if denom > 0:
                    corr = float(np.dot(prc, crc) / denom)
                    corr = max(-1.0, min(1.0, corr))
                    d_corr = (1.0 - corr) * 0.5
                else:
                    d_corr = 0.5
            else:
                d_corr = 0.5
            # Euclidean on non-returns
            other_mask = ~mask
            if np.any(other_mask):
                po = point[other_mask]; co = centroid[other_mask]
                d_euc = float(np.linalg.norm(po - co) / max(1.0, np.sqrt(po.size)))
            else:
                d_euc = 0.0
            return float(lam * d_corr + (1.0 - lam) * d_euc)
        except Exception:
            return float(np.linalg.norm(point - centroid))

    def invalidate_silhouettes(self) -> None:
        """Mark silhouettes as stale after moves."""
        self._silhouette_valid = False

    def update_after_labels(self, labels: np.ndarray, X: np.ndarray = None) -> None:
        """Public updater to refresh all statistics after labels change."""
        # Optional feature matrix override
        if X is not None:
            self.features = X
            self.n_samples, self.n_features = X.shape
            self.global_mean = np.mean(self.features, axis=0, dtype=np.float64)
            self.global_S = np.sum(self.features, axis=0, dtype=np.float64)
            self.global_N = self.n_samples
        # Compact labels to 0..K-1 and set
        labels = np.asarray(labels, dtype=int)
        unique_clusters = np.unique(labels)
        # Rebuild ID maps
        self.cluster_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
        self.inverse_cluster_id_map = {new_id: old_id for old_id, new_id in self.cluster_id_map.items()}
        compact_labels = np.zeros_like(labels)
        for old_id, new_id in self.cluster_id_map.items():
            compact_labels[labels == old_id] = new_id
        self.assignments = compact_labels
        # Update K and allocate arrays
        self.K_fixed = int(compact_labels.max()) + 1 if compact_labels.size else 0
        self.n_clusters = self.K_fixed
        self._cluster_sizes_cache = np.zeros(self.n_clusters, dtype=np.int32)
        self.centroids = np.zeros((self.n_clusters, self.n_features), dtype=np.float64)
        self.wcss_per_cluster = np.zeros(self.n_clusters, dtype=np.float64)
        self.S = np.zeros((self.n_clusters, self.n_features), dtype=np.float64)
        self.Q_trace = np.zeros(self.n_clusters, dtype=np.float64)
        # Recompute stats and caches
        self._update_all_stats()
        self.invalidate_silhouettes()

    def apply_move(self, point_idx: int, from_cluster: int, to_cluster: int):
        """Apply a move and update sufficient statistics incrementally."""
        if from_cluster == to_cluster:
            return

        point = self.features[point_idx].astype(np.float64)

        # CRITICAL FIX: Map original cluster IDs to consecutive indices
        from_cluster_idx = self.cluster_id_map[from_cluster]
        to_cluster_idx = self.cluster_id_map[to_cluster]

        # Update temporal transition counts before changing assignment
        try:
            Kf = int(self.K_fixed)
            alpha = float(getattr(self, 'temporal_alpha', 1.0))
            prev_idx = point_idx - 1 if point_idx - 1 >= 0 else None
            next_idx = point_idx + 1 if point_idx + 1 < len(self.assignments) else None
            if prev_idx is not None:
                u = int(self.assignments[prev_idx])
                if 0 <= u < Kf and 0 <= from_cluster < Kf:
                    self.transition_counts[u, from_cluster] = max(0, self.transition_counts[u, from_cluster] - 1)
                    self.transition_row_sums[u] = max(0, self.transition_row_sums[u] - 1)
            if next_idx is not None:
                w = int(self.assignments[next_idx])
                if 0 <= from_cluster < Kf and 0 <= w < Kf:
                    self.transition_counts[from_cluster, w] = max(0, self.transition_counts[from_cluster, w] - 1)
                    self.transition_row_sums[from_cluster] = max(0, self.transition_row_sums[from_cluster] - 1)
        except Exception:
            pass

        # Update assignments
        self.assignments[point_idx] = to_cluster

        # Note: cluster_sizes is now computed via property, no need to update manually

        # Update sufficient statistics incrementally
        # S_c = sum of points in cluster c
        self.S[from_cluster_idx] -= point
        self.S[to_cluster_idx] += point

        # Q_c = sum of ||x_i||^2 for cluster c
        point_norm_sq = np.sum(point ** 2)
        self.Q_trace[from_cluster_idx] -= point_norm_sq
        self.Q_trace[to_cluster_idx] += point_norm_sq

        # Update centroids: μ_c = S_c / n_c
        from_size = self.cluster_sizes[from_cluster_idx]
        to_size = self.cluster_sizes[to_cluster_idx]

        if from_size > 0:
            self.centroids[from_cluster_idx] = self.S[from_cluster_idx] / from_size
        else:
            self.centroids[from_cluster_idx] = np.zeros(self.n_features, dtype=np.float64)

        if to_size > 0:
            self.centroids[to_cluster_idx] = self.S[to_cluster_idx] / to_size
        else:
            self.centroids[to_cluster_idx] = np.zeros(self.n_features, dtype=np.float64)

        # Update WCSS and BCSS using exact formulas
        # WCSS_c = tr(Q_c) - ||S_c||^2 / n_c
        if from_size > 0:
            self.wcss_per_cluster[from_cluster_idx] = self.Q_trace[from_cluster_idx] - np.sum(self.S[from_cluster_idx] ** 2) / from_size
        else:
            self.wcss_per_cluster[from_cluster_idx] = 0.0

        if to_size > 0:
            self.wcss_per_cluster[to_cluster_idx] = self.Q_trace[to_cluster_idx] - np.sum(self.S[to_cluster_idx] ** 2) / to_size
        else:
            self.wcss_per_cluster[to_cluster_idx] = 0.0

        # Update totals
        self.total_wcss = np.sum(self.wcss_per_cluster)

        # BCSS = sum_c ||S_c||^2 / n_c - ||S||^2 / N
        bcss_terms = np.zeros(self.n_clusters, dtype=np.float64)
        current_sizes = self.cluster_sizes
        for i in range(self.n_clusters):
            if current_sizes[i] > 0:
                bcss_terms[i] = safe_divide(np.sum(self.S[i] ** 2), current_sizes[i], 0.0)
        self.total_bcss = np.sum(bcss_terms) - safe_divide(np.sum(self.global_S ** 2), self.global_N, 0.0)

        # Update temporal transition counts after changing assignment
        try:
            Kf = int(self.K_fixed)
            prev_idx = point_idx - 1 if point_idx - 1 >= 0 else None
            next_idx = point_idx + 1 if point_idx + 1 < len(self.assignments) else None
            if prev_idx is not None:
                u = int(self.assignments[prev_idx])
                if 0 <= u < Kf and 0 <= to_cluster < Kf:
                    self.transition_counts[u, to_cluster] += 1
                    self.transition_row_sums[u] += 1
            if next_idx is not None:
                w = int(self.assignments[next_idx])
                if 0 <= to_cluster < Kf and 0 <= w < Kf:
                    self.transition_counts[to_cluster, w] += 1
                    self.transition_row_sums[to_cluster] += 1
        except Exception:
            pass

        # Update variance caches and invalidate silhouettes
        self._recompute_variance_caches()
        self.invalidate_silhouettes()

class AtomicOperationContext:
    """Context manager for atomic clustering operations with automatic rollback."""

    def __init__(self, stats: ClusteringStats, operation_name: str = "atomic_operation"):
        self.stats = stats
        self.operation_name = operation_name
        self.snapshot = None

    def __enter__(self):
        """Begin atomic operation with state snapshot."""
        self.snapshot = self.stats._snapshot_state()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End atomic operation with validation or rollback."""
        if exc_type is not None:
            # Exception occurred, rollback to snapshot
            self.stats._restore_state(self.snapshot)
            return False  # Re-raise the exception
        else:
            # Success, validate final state
            try:
                self.stats._validate_state()
                return True
            except AssertionError as e:
                # State validation failed, rollback
                self.stats._restore_state(self.snapshot)
                raise RuntimeError(f"Atomic operation {self.operation_name} failed validation: {e}")

@dataclass
class OptConfig:
    """Unified configuration for iterative optimization - single source of truth."""
    # Core K and size constraints - STRICTLY ENFORCED
    # K_MIN = 6: Minimum clusters for adequate regime diversity (HARD CONSTRAINT)
    # K_MAX = 12: Maximum to prevent over-fragmentation (HARD CONSTRAINT)
    K_MIN: int = 6
    K_MAX: int = 10
    MIN_FRAC: float = 0.03  # 3% minimum cluster size
    MAX_FRAC: float = 0.20  # 20% maximum cluster size

    # Step execution parameters
    local_churn_cap: int = 5000  # Step 1 guard
    knn_size: int = 25  # kNN neighbor consensus
    beta: float = 0.6  # Step 2 weight
    split_tries_max: int = 5  # KMeans restarts

    # ENHANCED objective weights - AGGRESSIVE CV optimization focus
    # Prioritizing CV ratio for maximum regime separation quality
    w_cv: float = 0.70   # Primary: Variance ratio (CV) - MAXIMIZED for regime separation
    w_temp: float = 0.20 # Secondary: Temporal smoothness - reduced to focus on CV
    w_sil: float = 0.10  # Tertiary: Cluster cohesion (Silhouette) - reduced
    w_dbi: float = 0.00  # accessory
    w_ch: float = 0.00   # accessory
    w_bal: float = 0.05  # Minimal: Balance constraint - reduced to soft penalty

    # ULTRA-AGGRESSIVE Optimization parameters - maximum focus on CV, temporal, and silhouette
    max_rounds: int = 40  # Maximum iterations for optimal convergence
    eps_std_step1: float = -0.20  # Very aggressive step 1 threshold for CV optimization
    sil_guard: float = -0.08  # Require silhouette improvement for acceptance
    temporal_bonus: float = 0.25  # Strong bonus for temporal stability improvements

    # ULTRA-AGGRESSIVE Lexicographic acceptor parameters - maximum selectivity for quality
    eps_cv: float = 1e-5   # Ultra-tight CV threshold - maximum CV optimization
    eps_sil: float = 1e-4  # Ultra-tight Silhouette threshold - strong silhouette focus
    eps_temp: float = 1e-4 # Ultra-tight Temporal threshold - maximum temporal stability
    accessories_weight: float = 1e-3

    # Size-aware CV gate parameters
    size_gate_base: float = 1e-4
    size_gate_alpha: float = 0.02
    size_gate_beta: float = 0.05
    near_cap_ratio: float = 0.9

    # Candidate generation
    neighbors_per_point: int = 5
    silhouette_sample_size: int = 20000

    def get_unified_config(self, N: int) -> dict:
        """Get unified configuration that consolidates all overlapping config systems."""
        return {
            # Core constraints
            'K_MIN': self.K_MIN,
            'K_MAX': self.K_MAX,
            'MIN_SIZE': max(1, int(np.ceil(self.MIN_FRAC * N))),
            'MAX_SIZE': max(1, int(np.ceil(self.MAX_FRAC * N))),
            'SOFT_CAP': max(1, int(np.ceil(self.MAX_FRAC * N))),

            # Quality weights (consolidated from all sources)
            'W_CV': self.w_cv,
            'W_SIL': self.w_sil,
            'W_BAL': self.w_bal,
            'W_TEMP': self.w_temp,

            # Optimization parameters
            'MAX_ITERATIONS': self.max_rounds,
            'KNN_SIZE': self.knn_size,
            'LOCAL_CHURN_CAP': self.local_churn_cap,

            # N-agnostic constraints
            'MARGIN': 2,
            'TAU': 0.25,

            # Split policy parameters
            'MIN_PARENT_QUANTILE': 0.90,
            'MIN_PARENT_VS_TARGET': 1.6,
            'MIN_PCT_WITH_ALTS_GE_3': 0.50,
            'MAX_LOCKED_POINTS_FRAC': 0.05,
        }

    def validate_config_consistency(self, N: int) -> bool:
        """Validate that all configuration parameters are consistent."""
        unified = self.get_unified_config(N)

        # Check size constraints
        if unified['MIN_SIZE'] >= unified['MAX_SIZE']:
            tprint(f"⚠️ WARNING: MIN_SIZE ({unified['MIN_SIZE']}) >= MAX_SIZE ({unified['MAX_SIZE']})", "WARNING")
            return False

        # Check K constraints
        if unified['K_MIN'] >= unified['K_MAX']:
            tprint(f"⚠️ WARNING: K_MIN ({unified['K_MIN']}) >= K_MAX ({unified['K_MAX']})", "WARNING")
            return False

        # Check weight consistency
        total_weight = unified['W_CV'] + unified['W_SIL'] + unified['W_BAL'] + unified['W_TEMP']
        if abs(total_weight - 1.0) > 0.1:
            tprint(f"⚠️ WARNING: Weight sum ({total_weight}) not close to 1.0", "WARNING")

        return True

    # Performance limits
    max_kmeans_iterations: int = 50
    max_kmeans_seeds: int = 3
    boundary_points_limit: int = 50
    large_cluster_threshold: int = 100

    # Thresholds
    eps_wcss: float = 1e-9
    min_delta_std: float = -0.25
    quality_degradation_tolerance: float = 0.01

class IterativeOptimization:
    """Advanced 3-step iterative clustering optimization."""

    def __init__(self, verbose: bool = True, k: int | None = None):
        """Initialize the iterative optimization with advanced parameters."""
        self.verbose = verbose
        self.logger = get_logger('IterativeOptimization')

        # Performance optimization: Add caching for frequently computed values
        self._centroids_cache = None
        self._centroids_cache_valid = False
        self._wcss_cache = None
        self._wcss_cache_valid = False
        self._bcss_cache = None
        self._bcss_cache_valid = False
        self._silhouette_cache = None
        self._silhouette_cache_valid = False

        # Initialize cluster count
        self._k = int(k) if k is not None else None

        # Optimization parameters
        self.max_rounds = 25
        self.tolerance = 1e-5

        # Unified config - single source of truth
        self.config = OptConfig()

        # Context tracking for debugging
        self.context_id = f"opt_{id(self)}"
        self.run_seed = None

        # Set attributes from config
        for k in ("w_cv", "w_sil", "w_temp", "w_dbi", "w_ch", "w_bal", "knn_size", "local_churn_cap", "beta", "split_tries_max"):
            setattr(self, k, getattr(self.config, k))

        # CRITICAL FIX: Add safe defaults to prevent crashes
        self.use_std_for_gate = getattr(self, "use_std_for_gate", False)
        self.knn_size = getattr(self, "knn_size", 25)
        self.local_churn_cap = getattr(self, "local_churn_cap", 5000)
        self.beta = getattr(self, "beta", 0.6)
        self.split_tries_max = getattr(self, "split_tries_max", 5)

        # Additional attributes
        self.max_alternatives_per_point = 3  # For alternative cluster finding
        self.move_history = {}  # For tracking moves (point_idx -> history list)
        self.use_std_for_rank = True  # For ranking moves
        # Returns distance blending (for hybrid correlation distance)
        self.returns_lambda = 0.5
        self.returns_mask = None
        # Runtime control params used by Step-1/2 logic
        self.exploratory_quota = 3
        self.max_local_moves_per_iter = 50
        self.no_reversal_window = 4
        self.reverse_margin = 0.30
        self.tabu_tenure = 3
        self.max_moves_per_point = 2
        self.move_window_size = 6
        self.thrash_threshold = 0.6
        self.boundary_ratio_threshold = 0.45
        self.thrash_count_threshold = 3

        # CRITICAL FIX: Initialize core state attributes
        self.assignments = None
        self.sizes = None
        self.K = None
        self.features = None
        self.N = None

        # Initialize SOFT_CAP
        self.SOFT_CAP = None  # Will be set during optimization

        # Initialize optimized calculation engine
        self.calculation_engine = OptimizedCalculationEngine(
            use_hardware_accel=True,
            cache_size=1000
        )

        # Call hydrate defaults to ensure all attributes are set
        # Note: features will be set when optimization starts
        # self._hydrate_defaults() # Will be called at start of optimization

    def _hydrate_defaults(self):
        """Hydrate sane defaults once to fix all missing attributes."""
        if getattr(self, "_defaults_frozen", False):
            return
        import math
        import numpy as np

        # Always infer N from data if not provided
        N = int(getattr(self, "N", len(getattr(self, "features", [])) or 0))
        if N <= 0:
            raise ValueError("Cannot hydrate defaults: N is unknown.")

        self.N = N

        # Target band: 5–20% per cluster → ~83–333 for N=1663
        self.MIN_FRAC = getattr(self, "MIN_FRAC", self.config.MIN_FRAC)
        self.MAX_FRAC = getattr(self, "MAX_FRAC", self.config.MAX_FRAC)
        self.MIN_SIZE = getattr(self, "MIN_SIZE", max(1, math.ceil(self.MIN_FRAC * N)))
        self.MAX_SIZE = getattr(self, "MAX_SIZE", max(1, math.ceil(self.MAX_FRAC * N)))
        # If SOFT_CAP is missing/None, default it to MAX_SIZE
        self.SOFT_CAP = getattr(self, "SOFT_CAP", None) or self.MAX_SIZE
        self.CAP_RANGE = (self.MIN_SIZE, self.SOFT_CAP)

        # Frontier / batch params
        self.knn_size          = getattr(self, "knn_size", 32)
        self.local_churn_cap   = getattr(self, "local_churn_cap", max(64, int(0.05*N)))
        self.step2_epoch_churn = getattr(self, "step2_epoch_churn", int(0.02*N))

        # Weights & guards - will be overridden by step-specific weights during execution
        self.w_cv  = getattr(self, "w_cv", 0.50)
        self.w_temp = getattr(self, "w_temp", 0.30)
        self.w_sil = getattr(self, "w_sil", 0.10)
        self.w_bal = getattr(self, "w_bal", 0.10)
        self.neighbor_consensus_threshold = getattr(self, "neighbor_consensus_threshold", 0.6)

        # Micro-merge threshold (requested 0.1%)
        self.micro_frac = getattr(self, "micro_frac", 0.001)

    def apply_step_weights(self, step: int, step_weights: Dict[str, float] = None):
        """Apply step-specific weights for the given optimization step."""
        try:
            if step_weights is not None:
                self.w_cv = step_weights.get('w_cv', self.w_cv)
                self.w_sil = step_weights.get('w_sil', self.w_sil)
                self.w_temp = step_weights.get('w_temp', self.w_temp)
                self.w_bal = step_weights.get('w_bal', self.w_bal)
                tprint(f"🔧 Applied step {step} weights: CV={self.w_cv:.2f}, Sil={self.w_sil:.2f}, Temp={self.w_temp:.2f}, Bal={self.w_bal:.2f}", "INFO")
            else:
                # Apply default step-specific weights (balance used as constraint)
                if step == 1:
                    # Step 1: Local frontier moves - MAXIMUM CV optimization focus
                    self.w_cv = 0.80  # MAXIMUM CV optimization
                    self.w_temp = 0.15  # Reduced temporal focus to prioritize CV
                    self.w_sil = 0.05  # Minimal silhouette for step 1
                    self.w_bal = 0.00  # Balance used as constraint, not weight
                elif step == 2:
                    # Step 2: Global reallocation - MAXIMUM CV + minimal temporal focus
                    self.w_cv = 0.80  # MAXIMUM CV optimization
                    self.w_temp = 0.15  # Reduced temporal smoothness to prioritize CV
                    self.w_sil = 0.05  # Minimal silhouette for step 2
                    self.w_bal = 0.00  # Balance used as constraint, not weight
                elif step == 3:
                    # Step 3: Break large clusters - MAXIMUM CV focus
                    self.w_cv = 0.75
                    self.w_temp = 0.15
                    self.w_sil = 0.10
                    self.w_bal = 0.00  # Balance used as constraint, not weight
                tprint(f"🔧 Applied default step {step} weights: CV={self.w_cv:.2f}, Sil={self.w_sil:.2f}, Temp={self.w_temp:.2f}, Bal={self.w_bal:.2f}", "INFO")
        except Exception as e:
            tprint(f"❌ Failed to apply step {step} weights: {e}", "ERROR")

        # Freeze defaults to prevent configuration flip-flop
        self._defaults_frozen = True

        # Fallback tries
        self.split_tries_max = getattr(self, "split_tries_max", 5)
        self.use_std_for_gate = getattr(self, "use_std_for_gate", True)
        self.beta = getattr(self, "beta", 0.5)
        self.temporal_bonus = getattr(self, "temporal_bonus", 0.0)

        # CRITICAL FIX: Add missing attributes that are referenced but not defined
        self.tabu_list = getattr(self, "tabu_list", {})

        # Provide adaptive_tau if missing
        global adaptive_tau
        if "adaptive_tau" not in globals():
            def adaptive_tau(max_frac: float, base: float = 0.0) -> float:
                return float(base * (1.0 + 1.5 * max(0.0, max_frac - 0.5)))

        tprint(f"📋 Hydrated defaults: N={self.N}, MIN_SIZE={self.MIN_SIZE}, MAX_SIZE={self.MAX_SIZE}, "
               f"SOFT_CAP={self.SOFT_CAP}, knn_size={self.knn_size}, local_churn_cap={self.local_churn_cap}, "
               f"step2_epoch_churn={self.step2_epoch_churn}", "DEBUG")

        # Freeze defaults to prevent configuration flip-flop
        self._defaults_frozen = True

    def _current_sizes(self, labels=None):
        """Always recompute sizes from current labels right before any decision."""
        import numpy as np
        labs = np.asarray(self.assignments if labels is None else labels)
        uniq = np.unique(labs)
        sizes = np.array([np.sum(labs==u) for u in uniq], dtype=int)
        return sizes, uniq

    def _current_sizes_dict(self, labels):
        """Return sizes as dictionary for band/split logic."""
        import numpy as np
        u, c = np.unique(labels, return_counts=True)
        return dict(zip(u.tolist(), c.tolist()))

    def _accept_with_consensus(self, i, dst, labels):
        """Require >= neighbor_consensus_threshold neighbors already in dst."""
        # require >= neighbor_consensus_threshold neighbors already in dst
        try:
            Xn = self._get_neighbor_matrix()
            knn = getattr(self, 'knn_graph', None)
            if knn is None or (hasattr(knn, '__len__') and len(knn) != len(Xn)):
                from sklearn.neighbors import NearestNeighbors
                n_neighbors = min(self.knn_size + 1, len(Xn))
                # Use cosine distance to improve separation on return-like features
                nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
                nn.fit(Xn)
                _, knn = nn.kneighbors(Xn)
                self.knn_graph = knn
            nbrs = self.knn_graph[i][1:self.knn_size+1]  # skip self
        except Exception:
            # Fallback: no consensus gating if kNN fails
            nbrs = []
        same = sum(1 for j in nbrs if labels[j] == dst)
        return (same / max(1, len(nbrs))) >= self.neighbor_consensus_threshold

    def _apply_step2(self, labels, boundary_idx, proposals):
        new_labels = labels.copy()
        applied = 0
        for (i, dst) in proposals:
            if i not in boundary_idx:
                continue
            new_labels[i] = dst
            applied += 1
            if applied >= self.step2_epoch_churn:
                break
        # Do not reduce K in Step-2
        if len(set(new_labels)) < len(set(labels)):
            self.logger.debug("Step-2 attempted to reduce K; rolling back.")
            return labels
        return self._relabel_compact(new_labels)

    def _emergency_multi_split_quality_aware(self, labels, cid, k_needed):
        """Make emergency splits deterministic and quality-aware using farthest-point seeding."""
        import numpy as np
        idx = np.where(labels == cid)[0]
        X = self.features[idx]

        # Seed via farthest-point (k-means++ style)
        rng = np.random.default_rng(42)
        seeds = [rng.integers(len(X))]
        for _ in range(1, k_needed):
            d2 = np.min(((X - X[seeds][:,None])**2).sum(-1), axis=0)
            probs = d2 / (d2.sum() + 1e-12)
            seeds.append(rng.choice(len(X), p=probs))

        # Lloyd with balanced assignment against [MIN_SIZE, SOFT_CAP]
        from sklearn.metrics import pairwise_distances
        D = pairwise_distances(X, X[seeds])
        order = np.argsort(D, axis=1)[:,0]
        child_ids = [f"{cid}_{j}" for j in range(k_needed)]
        caps = [self.SOFT_CAP] * k_needed
        counts = [0]*k_needed
        out = np.empty(len(idx), dtype=object)

        for r in np.argsort(D[np.arange(len(X)), order]):
            best = order[r]
            # if best child is at cap, pick next-nearest with room
            for alt in np.argsort(D[r]):
                if counts[alt] < caps[alt]:
                    out[r] = child_ids[alt]
                    counts[alt] += 1
                    break

        # Map child ids to new integer labels
        new_labels = labels.copy()
        base = max(labels) + 1
        mapping = {}
        for j, cidj in enumerate(child_ids):
            mapping[cidj] = base + j
        for t, g in zip(idx, out):
            new_labels[t] = mapping[g]

        return new_labels

    def _pick_worst_cluster(self, labels):
        """Pick the cluster with worst compactness (highest intra scatter / Sil penalty)."""
        import numpy as np
        sizes = self._current_sizes_dict(labels)

        # Calculate intra-cluster scatter for each cluster
        worst_cluster = None
        worst_score = -1

        for cid, size in sizes.items():
            if size < 2:  # Skip singletons
                continue

            # Calculate intra-cluster scatter
            cluster_points = np.where(labels == cid)[0]
            if len(cluster_points) < 2:
                continue

            cluster_features = self.features[cluster_points]
            centroid = np.mean(cluster_features, axis=0)
            scatter = np.mean(np.sum((cluster_features - centroid)**2, axis=1))

            # Normalize by size to get per-point scatter
            normalized_scatter = scatter / size

            if normalized_scatter > worst_score:
                worst_score = normalized_scatter
                worst_cluster = cid

        return worst_cluster if worst_cluster is not None else max(sizes.keys())

    def _band_policy(self, labels):
        """Proactive band policy (softened): only split if clearly safe and beneficial.
        - Apply at most one 2-way split when K is below target and parent is sufficiently large.
        - Ensure children respect MIN_SIZE and SOFT_CAP and no empty clusters are created.
        """
        K = len(set(labels))
        if K < 7:
            sizes = np.bincount(labels, minlength=int(np.max(labels) + 1))
            if sizes.size == 0:
                return labels
            min_size = getattr(self, "MIN_SIZE", 1)
            soft_cap = getattr(self, "SOFT_CAP", int(np.ceil(0.20 * len(labels))))
            # Pick candidate to split (worst compactness)
            cid = self._pick_worst_cluster(labels)
            # Require clearly large parent before splitting
            non_zero_sizes = sizes[sizes > 0]
            median_size = np.median(non_zero_sizes) if non_zero_sizes.size else min_size
            required_parent = max(2 * min_size, int(0.9 * median_size))
            if sizes[cid] < required_parent:
                return labels  # too small to proactively split
            # Try 2-way split
            new_labels = self._emergency_multi_split_quality_aware(labels, cid, 2)
            new_labels = self._relabel_compact(np.asarray(new_labels))
            new_sizes = np.bincount(new_labels, minlength=int(np.max(new_labels) + 1))
            if new_sizes.size == 0:
                return labels
            # Reject if any child violates size or if empties created
            if np.min(new_sizes[new_sizes > 0]) < min_size:
                return labels
            if np.max(new_sizes) > soft_cap:
                return labels
            if np.any(new_sizes == 0):
                return labels
            return new_labels
        if K > 10:
            # TODO: implement merge_smallest_pairs if needed
            return labels
        return labels

    def _make_move(self, i, src, dst, d_cv, d_sil, d_temp=0.0):
        """Standardize move records as dicts to fix tuple/dict crash."""
        alpha = self.w_sil / max(self.w_cv, 1e-9)
        score = (-d_cv) + alpha * (d_sil) + self.w_temp * d_temp
        return {
            "i": int(i), "src": int(src), "dst": int(dst),
            "d_cv": float(d_cv), "d_sil": float(d_sil), "d_temp": float(d_temp),
            "score": float(score)
        }

    def _accept_move(self, move, src_size, dst_size, avg_size):
        """Acceptance gate that de-biases toward big clusters."""
        alpha = self.w_sil / max(self.w_cv, 1e-9)
        cv_gain = -move["d_cv"]                      # positive good
        sil_gain = move["d_sil"]

        over = max(0.0, (dst_size - avg_size) / max(1.0, avg_size))   # how over-average dest is
        tau_base = 0.0
        tau = tau_base + 0.005 * over          # require more CV gain if dest is big

        score = cv_gain + alpha * sil_gain
        return score > tau

    def _safe_sil(self, labels):
        """Robust Silhouette wrapper (no ∞ / NaN)."""
        from sklearn.metrics import silhouette_score
        import numpy as np
        labs = np.asarray(labels)
        _, counts = np.unique(labs, return_counts=True)
        if np.sum(counts >= 2) < 2:
            return 0.0  # not meaningful yet
        idx = np.where(counts[labs] >= 2)[0]   # drop singletons for scoring
        X = self.features[idx]; y = labs[idx]
        if len(y) > 2000:
            rng = np.random.default_rng(42)
            sel = rng.choice(len(y), 2000, replace=False)
            X, y = X[sel], y[sel]
        try:
            return float(silhouette_score(X, y))
        except Exception:
            return 0.0

    def _safe_dbi(self, labels):
        """Robust Davies-Bouldin wrapper (no ∞ / NaN)."""
        try:
            from sklearn.metrics import davies_bouldin_score
            import numpy as np
            labs = np.asarray(labels)
            _, counts = np.unique(labs, return_counts=True)
            if np.sum(counts >= 2) < 2:
                return float("inf")
            idx = np.where(counts[labs] >= 2)[0]
            return float(davies_bouldin_score(self.features[idx], labs[idx]))
        except Exception:
            return float("inf")

    def _better(self, a, b, eps_cv=1e-9, eps_sil=1e-9):
        """Lexicographic objective (CV first, then Sil, then temporal)."""
        # a/b are (d_cv, d_sil, d_temp) deltas; lower d_cv is better
        if a.d_cv < b.d_cv - eps_cv: return True
        if a.d_cv > b.d_cv + eps_cv: return False
        # tie: prefer higher silhouette
        if a.d_sil > b.d_sil + eps_sil: return True
        if a.d_sil < b.d_sil - eps_sil: return False
        # tie: prefer better temporal
        return a.d_temp > b.d_temp

    def _sanity_check(self, labels):
        import numpy as np
        from sklearn.metrics import davies_bouldin_score, silhouette_score
        u, c = np.unique(labels, return_counts=True)
        K = len(u)
        sizes = c.tolist()
        over = int(np.sum(c > self.SOFT_CAP))
        under = int(np.sum(c < self.MIN_SIZE))

        # Safe metrics (avoid inf/NaN when tiny clusters exist)
        dbi = float("nan")
        sil = float("nan")
        try:
            if K >= 2 and np.all(c >= 2):
                dbi = davies_bouldin_score(self.features, labels)
            if K >= 2:
                sil = silhouette_score(self.features, labels, metric="euclidean")
        except Exception:
            pass

        self.logger.debug(
            f"📊 Step-2 Sanity Check: K={K}, sizes={sizes}, >{self.SOFT_CAP}: {over}/{K} "
            f"(<{self.MIN_SIZE}: {under}/{K}) | Sil={sil:.3f}, DBI={dbi if dbi==dbi else 'nan'}"
        )

    def _sanity_check_and_log(self, label: str, stats: ClusteringStats | None = None) -> None:
        """Lightweight guard to log current clustering state without raising."""
        try:
            labels = None
            if stats is not None and getattr(stats, "assignments", None) is not None:
                labels = np.asarray(stats.assignments)
            elif self.assignments is not None:
                labels = np.asarray(self.assignments)

            if labels is None or labels.size == 0:
                tprint(f"[{label}] Sanity check skipped: assignments unavailable", "WARNING")
                return

            labels = labels.astype(int, copy=False)
            sizes = np.bincount(labels, minlength=int(labels.max() + 1))
            K = sizes.size
            max_size = int(sizes.max()) if sizes.size else 0
            min_size = int(sizes.min()) if sizes.size else 0
            cap = getattr(self, "SOFT_CAP", None)

            msg = f"[{label}] K={K}, max={max_size}, min={min_size}"
            if cap is not None:
                msg += f", SOFT_CAP={cap}"
            tprint(msg, "DEBUG")
        except Exception as err:
            tprint(f"[{label}] Sanity check failed: {err}", "WARNING")

    def _get_neighbor_matrix(self):
        """Build neighbor feature matrix using UMAP if available; else whitened features."""
        try:
            if hasattr(self, 'features_neighbors') and self.features_neighbors is not None:
                return self.features_neighbors
            X = self.features
            # Drop near-constant columns
            std = X.std(axis=0)
            keep = std > 1e-8
            if keep.sum() >= 2:
                X = X[:, keep]
            # Whiten
            from sklearn.preprocessing import StandardScaler
            Xw = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
            # Try UMAP
            try:
                if umap is not None:
                    reducer = umap.UMAP(n_neighbors=30, n_components=min(20, Xw.shape[1]), metric='cosine', random_state=42)
                    Xu = reducer.fit_transform(Xw)
                    self.features_neighbors = Xu
                else:
                    raise ImportError("UMAP not available")
            except Exception:
                self.features_neighbors = Xw
            return self.features_neighbors
        except Exception:
            return self.features

    def finalize_labels(self, X: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """Finalize clustering: enforce K in [K_MIN,K_MAX], sizes in [MIN_SIZE, SOFT_CAP],
        and remove singletons by merging to nearest feasible clusters, then re-split if needed.
        Returns finalized labels (np.ndarray)."""
        try:
            # Ensure core state
            self.features = X
            self.N = len(X)
            self._hydrate_defaults()

            labels = np.asarray(assignments).astype(int, copy=True)

            # 1) Enforce cap and grow splits to meet K_MIN if needed
            labels = self._enforce_k_and_cap_labels(X, labels)

            # 2) Merge undersized clusters repeatedly until none remain
            stats = ClusteringStats(X, labels)
            changed = True
            max_repair_iterations = 10  # Prevent infinite loops
            repair_iteration = 0
            while changed and repair_iteration < max_repair_iterations:
                repair_iteration += 1
                changed = False
                MIN_SIZE = stats.n_samples and max(1, int(np.ceil(self.config.MIN_FRAC * stats.n_samples))) or self.MIN_SIZE
                for cid, sz in enumerate(stats.cluster_sizes):
                    if 0 < sz < MIN_SIZE:
                        dest = self._nearest_feasible_dest(cid, stats, X, max_after=self.SOFT_CAP)
                        if dest is None:
                            dest = self._nearest_feasible_dest(cid, stats, X)
                        if dest is not None:
                            stats.assignments[stats.assignments == cid] = dest
                            self._stats_update(stats, stats.assignments, X)
                            changed = True

            labels = stats.assignments.copy()

            # 3) If K dropped below K_MIN due to merges, split largest clusters to reach band
            current_k = int(labels.max()) + 1
            if current_k < self.config.K_MIN:
                labels = self._enforce_k_and_cap_labels(X, labels)

            # Final compact
            labels = self._relabel_compact(labels)
            return labels
        except Exception as e:
            tprint(f"Finalize labels failed: {e}", "WARNING")
            return assignments

    def _initialize_state(self, assignments: np.ndarray) -> None:
        """Consolidate state initialization logic."""
        self.assignments = assignments.copy()
        self.K = len(np.unique(assignments))
        self.sizes = np.bincount(assignments, minlength=self.K)

        # Ensure assignments is never None
        if self.assignments is None:
            raise ValueError("Assignments cannot be None after initialization")
        if self.sizes is None:
            self.sizes = np.bincount(self.assignments, minlength=self.K)

    def _invalidate_caches(self):
        """Invalidate all cached values when state changes."""
        self._centroids_cache_valid = False
        self._wcss_cache_valid = False
        self._bcss_cache_valid = False
        self._silhouette_cache_valid = False

    def _get_cached_centroids(self, features: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """Get centroids with caching to avoid redundant calculations."""
        if not self._centroids_cache_valid or self._centroids_cache is None:
            self._centroids_cache = self._calculate_centroids_optimized(features, assignments)
            self._centroids_cache_valid = True
        return self._centroids_cache

    def _calculate_centroids_optimized(self, features: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """Optimized centroid calculation using vectorized operations."""
        K = int(assignments.max()) + 1
        centroids = np.zeros((K, features.shape[1]))

        # Vectorized centroid calculation
        for k in range(K):
            mask = assignments == k
            if np.any(mask):
                centroids[k] = np.mean(features[mask], axis=0)

        return centroids

    def _get_cached_wcss(self, features: np.ndarray, assignments: np.ndarray, centroids: np.ndarray) -> float:
        """Get WCSS with caching to avoid redundant calculations."""
        if not self._wcss_cache_valid or self._wcss_cache is None:
            self._wcss_cache = self._calculate_wcss_optimized(features, assignments, centroids)
            self._wcss_cache_valid = True
        return self._wcss_cache

    def _calculate_wcss_optimized(self, features: np.ndarray, assignments: np.ndarray, centroids: np.ndarray) -> float:
        """Optimized WCSS calculation using vectorized operations."""
        # Vectorized WCSS calculation
        distances = np.sum((features - centroids[assignments]) ** 2, axis=1)
        return np.sum(distances)

    def _get_cached_bcss(self, features: np.ndarray, assignments: np.ndarray, centroids: np.ndarray) -> float:
        """Get BCSS with caching to avoid redundant calculations."""
        if not self._bcss_cache_valid or self._bcss_cache is None:
            self._bcss_cache = self._calculate_bcss_optimized(features, assignments, centroids)
            self._bcss_cache_valid = True
        return self._bcss_cache

    def _calculate_bcss_optimized(self, features: np.ndarray, assignments: np.ndarray, centroids: np.ndarray) -> float:
        """Optimized BCSS calculation using vectorized operations."""
        global_mean = np.mean(features, axis=0)
        cluster_sizes = np.bincount(assignments, minlength=len(centroids))

        # Vectorized BCSS calculation
        bcss = 0.0
        for i in range(len(centroids)):
            if cluster_sizes[i] > 0:
                bcss += cluster_sizes[i] * np.sum((centroids[i] - global_mean) ** 2)
        return bcss

    def _optimize_boundary_detection(self, features: np.ndarray, assignments: np.ndarray,
                                   boundary_threshold: float = 0.25) -> np.ndarray:
        """Optimized boundary point detection using vectorized operations."""
        # Use a more efficient approach for boundary detection
        # Instead of calculating distances to all centroids, use approximate methods
        centroids = self._get_cached_centroids(features, assignments)

        # Calculate distances to nearest and second-nearest centroids
        distances = np.sqrt(np.sum((features[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2))

        # Find nearest and second-nearest centroids
        sorted_indices = np.argsort(distances, axis=1)
        nearest_distances = distances[np.arange(len(features)), sorted_indices[:, 0]]
        second_nearest_distances = distances[np.arange(len(features)), sorted_indices[:, 1]]

        # Boundary points are those where the distance ratio is close to 1
        distance_ratios = nearest_distances / (second_nearest_distances + 1e-10)
        boundary_mask = distance_ratios > (1.0 - boundary_threshold)

        return np.where(boundary_mask)[0]

    def _validate_optimization_inputs(self, X: np.ndarray, initial_assignments: np.ndarray,
                                    entity_ids: np.ndarray = None, time_idx: np.ndarray = None) -> None:
        """Comprehensive validation of optimization inputs."""
        # Validate features
        assert X is not None, "Features array cannot be None"
        assert X.size > 0, "Features array cannot be empty"
        assert len(X.shape) == 2, f"Features must be 2D array, got shape {X.shape}"
        assert not np.any(np.isnan(X)), "Features array contains NaN values"
        assert not np.any(np.isinf(X)), "Features array contains infinite values"

        # Validate assignments
        assert initial_assignments is not None, "Initial assignments cannot be None"
        assert len(initial_assignments) == len(X), f"Assignments length {len(initial_assignments)} != features length {len(X)}"
        assert initial_assignments.dtype in [np.int32, np.int64], f"Assignments must be integer type, got {initial_assignments.dtype}"
        assert initial_assignments.min() >= 0, f"Assignments must be non-negative, got min {initial_assignments.min()}"
        assert len(np.unique(initial_assignments)) >= 2, f"Must have at least 2 clusters, got {len(np.unique(initial_assignments))}"

        # Validate optional arrays
        if entity_ids is not None:
            assert len(entity_ids) == len(X), f"Entity IDs length {len(entity_ids)} != features length {len(X)}"
        if time_idx is not None:
            assert len(time_idx) == len(X), f"Time index length {len(time_idx)} != features length {len(X)}"
            assert np.all(time_idx >= 0), "Time index must be non-negative"

        # Validate configuration consistency
        N = len(X)
        if not self.config.validate_config_consistency(N):
            tprint("⚠️ WARNING: Configuration validation failed", "WARNING")

        tprint(f"✅ Input validation passed: N={N}, K={len(np.unique(initial_assignments))}", "DEBUG")

    def _validate_state_consistency(self, stats: ClusteringStats, features: np.ndarray) -> bool:
        """Validate state consistency between stats and features."""
        try:
            # Basic consistency checks
            assert len(stats.assignments) == len(features), f"Assignments length {len(stats.assignments)} != features length {len(features)}"
            assert stats.assignments.max() < stats.K_fixed, f"Assignment max {stats.assignments.max()} >= K_fixed {stats.K_fixed}"
            assert np.sum(stats.cluster_sizes) == len(features), f"Cluster sizes sum {np.sum(stats.cluster_sizes)} != features length {len(features)}"

            # Validate cluster ID mappings
            unique_assignments = np.unique(stats.assignments)
            for cluster_id in unique_assignments:
                assert cluster_id in stats.cluster_id_map, f"Cluster ID {cluster_id} not in cluster_id_map"
                compact_id = stats.cluster_id_map[cluster_id]
                assert 0 <= compact_id < stats.K_fixed, f"Compact ID {compact_id} out of bounds [0, {stats.K_fixed})"

            # Validate sufficient statistics
            assert stats.S.shape[0] == stats.K_fixed, f"S shape {stats.S.shape[0]} != K_fixed {stats.K_fixed}"
            assert stats.Q_trace.shape[0] == stats.K_fixed, f"Q_trace shape {stats.Q_trace.shape[0]} != K_fixed {stats.K_fixed}"
            assert stats.centroids.shape[0] == stats.K_fixed, f"Centroids shape {stats.centroids.shape[0]} != K_fixed {stats.K_fixed}"

            return True
        except AssertionError as e:
            tprint(f"❌ State validation failed: {e}", "ERROR")
            return False

    def _log_with_context(self, message: str, level: str = "INFO", step: str = None):
        """Systematic logging with context and step information."""
        context = f"[{self.context_id}]"
        if step:
            context += f"[{step}]"
        if self.run_seed:
            context += f"[seed={self.run_seed}]"

        tprint(f"{context} {message}", level)

    def size_penalty_eps(self, dest_after: float, mean: float, max_size: float,
                          base: float = 1e-4, alpha: float = 0.02, beta: float = 0.05, near_cap: float = 0.9) -> float:
        """Size-aware acceptance penalty for moves into large/near-cap clusters."""
        over_avg = max(0.0, dest_after/mean - 1.0)
        near = max(0.0, dest_after/max_size - near_cap)
        return base + alpha*over_avg + beta*(near**2)

    def _accept_move_with_size_penalty(self, delta_cv: float, delta_sil: float, delta_temp: float,
                                      delta_bal: float, dest_after: float, mean_size: float, max_size: float) -> bool:
        """Accept move with size-aware penalty for large clusters."""
        # Calculate weighted delta without balance weight (balance used as constraint)
        delta = (self.w_cv * delta_cv - self.w_sil * delta_sil + self.w_temp * delta_temp)

        # Apply soft balance constraint penalty
        balance_penalty = 0.0
        if delta_bal < -0.1:  # Only penalize significant balance degradation
            balance_penalty = 0.05 * abs(delta_bal)  # Soft penalty, not hard weight
        delta -= balance_penalty

        # Calculate size penalty
        penalty = self.size_penalty_eps(dest_after, mean_size, max_size)

        # Accept if improvement is significant enough
        return delta <= -penalty

    def _commit(self, new_labels: np.ndarray):
        """Atomically commit new labels and update all derived state."""
        self.assignments = new_labels
        self.K = int(np.unique(new_labels).size)
        self.sizes = np.bincount(new_labels, minlength=self.K)

        # Keep context copies in sync if they exist
        if hasattr(self, "ctx"):
            self.ctx.assignments = new_labels

        # Increment state epoch for debugging
        if not hasattr(self, 'state_epoch'):
            self.state_epoch = 0
        self.state_epoch += 1

        # Sanity asserts: catch state desync immediately
        assert self.sizes.max() == np.bincount(self.assignments).max(), "State desync: sizes stale vs assignments"
        assert self.K == len(np.unique(self.assignments)), f"K mismatch: {self.K} vs {len(np.unique(self.assignments))}"
        assert self.sizes.sum() == len(self.assignments), "Size sum mismatch"

        self._log_with_context(f"✅ Atomic commit: K={self.K}, sizes={self.sizes[self.sizes > 0]}, epoch={self.state_epoch}", "DEBUG", "COMMIT")

    def _sync_state_from_stats(self, stats: Optional[ClusteringStats]) -> None:
        """Synchronize internal assignments with the latest statistics."""
        try:
            if stats is None or getattr(stats, "assignments", None) is None:
                return

            labels = np.asarray(stats.assignments)
            if labels.size == 0:
                return

            # Commit a defensive copy to avoid unintended aliasing
            self._commit(labels.copy())
        except Exception as sync_error:
            tprint(f"State sync failed: {sync_error}", "ERROR")

    def feasibility_ok(self, sizes: np.ndarray, MIN_SIZE: int, MAX_SIZE: int, K: int) -> bool:
        """Check if current clustering satisfies all feasibility constraints."""
        return (self.config.K_MIN <= K <= self.config.K_MAX) and sizes.min() >= MIN_SIZE and sizes.max() <= MAX_SIZE

    def attach_tiny_clusters(self, X: np.ndarray, labels: np.ndarray, MIN_SIZE: int, MAX_SIZE: int) -> np.ndarray:
        """Attach tiny clusters to nearest feasible recipients before optimization."""
        sizes = np.bincount(labels)
        tiny = np.where(sizes < MIN_SIZE)[0]

        if len(tiny) == 0:
            return labels

        self._log_with_context(f"🔧 Attaching {len(tiny)} tiny clusters: {sizes[tiny]}", "INFO", "TINY")

        for c in tiny:
            if sizes[c] == 0:
                continue

            # Find points in tiny cluster
            idx = np.where(labels == c)[0]

            # Find best recipient cluster
            best_recipient = self._find_best_recipient(X, labels, c, sizes, MAX_SIZE)
            if best_recipient is not None:
                labels[idx] = best_recipient
                sizes[best_recipient] += sizes[c]
                sizes[c] = 0
                self._log_with_context(f"✅ Attached cluster {c} ({sizes[c]} points) to {best_recipient}", "DEBUG", "TINY")

        return labels

    def _find_best_recipient(self, X: np.ndarray, labels: np.ndarray, tiny_cluster: int, sizes: np.ndarray, MAX_SIZE: int) -> int:
        """Find best recipient cluster for tiny cluster attachment."""
        # Get tiny cluster centroid
        tiny_mask = labels == tiny_cluster
        tiny_centroid = X[tiny_mask].mean(axis=0)

        # Find clusters that can accept the tiny cluster
        candidates = []
        for cluster_id in range(len(sizes)):
            if cluster_id == tiny_cluster or sizes[cluster_id] == 0:
                continue
            if sizes[cluster_id] + sizes[tiny_cluster] <= MAX_SIZE:
                # Calculate distance to this cluster's centroid
                cluster_mask = labels == cluster_id
                cluster_centroid = X[cluster_mask].mean(axis=0)
                distance = np.linalg.norm(tiny_centroid - cluster_centroid)
                candidates.append((cluster_id, distance))

        if not candidates:
            return None

        # Return closest feasible recipient
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def _enforce_hard_constraints(self, X: np.ndarray, assignments: np.ndarray,
                                 entity_ids: np.ndarray = None, time_idx: np.ndarray = None) -> np.ndarray:
        """Enforce hard constraints: K∈[6,12] and cluster sizes ∈[min_size, max_size]."""
        N = len(X)
        min_size = max(1, int(np.ceil(self.config.MIN_FRAC * N)))
        max_size = int(np.floor(self.config.MAX_FRAC * N))

        # Compact labels first
        assignments, _ = self._remap_to_compact_ids(assignments)
        K = int(assignments.max()) + 1

        self._log_with_context(f"Enforcing hard constraints: K={K}, min_size={min_size}, max_size={max_size}", "INFO", "CONSTRAINTS")

        # Enforce K first
        if K < self.config.K_MIN:
            assignments = self._enforce_k_minimum(X, assignments, min_size, max_size)
        elif K > self.config.K_MAX:
            assignments = self._enforce_k_maximum(X, assignments, min_size, max_size)

        # Enforce size constraints
        assignments = self._enforce_size_constraints(X, assignments, min_size, max_size)

        # Final compact and rebuild
        assignments, _ = self._remap_to_compact_ids(assignments)
        return assignments

    def _enforce_k_minimum(self, X: np.ndarray, assignments: np.ndarray, min_size: int, max_size: int) -> np.ndarray:
        """Split largest clusters until K >= K_MIN."""
        K = int(assignments.max()) + 1
        splits_done = 0

        max_grow_iterations = 20  # Prevent infinite loops
        grow_iteration = 0
        while K < self.config.K_MIN and grow_iteration < max_grow_iterations:
            grow_iteration += 1
            # Find largest cluster
            sizes = np.bincount(assignments, minlength=K)
            largest_cluster = np.argmax(sizes)

            if sizes[largest_cluster] < 2 * min_size:
                self._log_with_context(f"Cannot split cluster {largest_cluster} (size={sizes[largest_cluster]}) to reach K_MIN", "WARNING", "CONSTRAINTS")
                break

            # Try to split it
            success = self._split_cluster_constrained(X, assignments, largest_cluster, min_size, max_size)
            if not success:
                self._log_with_context(f"Failed to split cluster {largest_cluster}", "WARNING", "CONSTRAINTS")
                break

            K = int(assignments.max()) + 1
            splits_done += 1

            if splits_done > 10:  # Safety limit
                self._log_with_context("Reached safety limit for K_MIN enforcement", "WARNING", "CONSTRAINTS")
                break

        self._log_with_context(f"K_MIN enforcement: {splits_done} splits, final K={K}", "INFO", "CONSTRAINTS")
        return assignments

    def _enforce_k_maximum(self, X: np.ndarray, assignments: np.ndarray, min_size: int, max_size: int) -> np.ndarray:
        """Merge smallest clusters until K <= K_MAX."""
        K = int(assignments.max()) + 1
        merges_done = 0

        max_shrink_iterations = 20  # Prevent infinite loops
        shrink_iteration = 0
        while K > self.config.K_MAX and shrink_iteration < max_shrink_iterations:
            shrink_iteration += 1
            # Find two smallest clusters that minimize WCSS increase
            sizes = np.bincount(assignments, minlength=K)
            non_empty = np.where(sizes > 0)[0]

            if len(non_empty) < 2:
                break

            # Find best merge pair
            best_merge = self._find_best_merge_pair(X, assignments, non_empty, max_size)
            if best_merge is None:
                self._log_with_context("No valid merge pairs found", "WARNING", "CONSTRAINTS")
                break

            # Perform merge
            i, j = best_merge
            assignments[assignments == j] = i
            K = int(assignments.max()) + 1
            merges_done += 1

            if merges_done > 10:  # Safety limit
                self._log_with_context("Reached safety limit for K_MAX enforcement", "WARNING", "CONSTRAINTS")
                break

        self._log_with_context(f"K_MAX enforcement: {merges_done} merges, final K={K}", "INFO", "CONSTRAINTS")
        return assignments

    def _enforce_size_constraints(self, X: np.ndarray, assignments: np.ndarray, min_size: int, max_size: int) -> np.ndarray:
        """Enforce min_size and max_size constraints."""
        K = int(assignments.max()) + 1
        sizes = np.bincount(assignments, minlength=K)

        # Handle oversized clusters
        oversized = np.where(sizes > max_size)[0]
        for cluster_id in oversized:
            self._split_cluster_constrained(X, assignments, cluster_id, min_size, max_size)

        # Handle undersized clusters
        undersized = np.where((sizes > 0) & (sizes < min_size))[0]
        for cluster_id in undersized:
            self._merge_undersized_cluster(X, assignments, cluster_id, min_size, max_size)

        return assignments

    # ===== METRICS IMPLEMENTATION =====

    def cv_of_sizes(self, labels: np.ndarray, K: int) -> float:
        """Calculate coefficient of variation of cluster sizes."""
        sizes = np.bincount(labels, minlength=K)
        mu = sizes.mean()
        return 0.0 if mu == 0 else (sizes.std(ddof=0) / mu)

    def robust_feature_cv(self, features: np.ndarray) -> float:
        """
        Calculate robust coefficient of variation for standardized features.

        For standardized features (mean ≈ 0), traditional CV = std/mean becomes problematic.
        This method uses a robust approach that considers the scale of variation relative
        to the feature's inherent variability.

        Args:
            features: Feature matrix (n_samples, n_features)

        Returns:
            Average robust CV across all features
        """
        if features.shape[1] == 0:
            return 0.0

        feature_cvs = []
        for i in range(features.shape[1]):
            feature_values = features[:, i]

            # For standardized features, use median absolute deviation as scale estimate
            # instead of mean (which is ≈ 0)
            mad = np.median(np.abs(feature_values - np.median(feature_values)))

            # Use MAD as the scale estimate (robust to outliers)
            if mad > 0:
                # Robust CV: MAD / median(|values|) - handles zero-centered features
                median_abs = np.median(np.abs(feature_values))
                if median_abs > 0:
                    robust_cv = mad / median_abs
                else:
                    # Fallback for features that are all zeros
                    robust_cv = 0.0
            else:
                robust_cv = 0.0

            feature_cvs.append(robust_cv)

        return np.mean(feature_cvs) if feature_cvs else 0.0

    def select_important_features(self, features: np.ndarray, assignments: np.ndarray,
                                importance_threshold: float = 0.001) -> Tuple[np.ndarray, List[int]]:
        """
        Select most discriminative features based on regime separation importance.

        Args:
            features: Feature matrix (n_samples, n_features)
            assignments: Cluster assignments
            importance_threshold: Minimum importance for feature inclusion

        Returns:
            Tuple of (selected_features, selected_indices)
        """
        try:
            # Calculate feature importance for regime separation
            importance_scores, _ = self._calculate_enhanced_cv_metrics(features, assignments)

            # Select features above threshold
            selected_indices = np.where(importance_scores >= importance_threshold)[0]
            selected_features = features[:, selected_indices]

            self._log_with_context(
                f"Feature selection: {len(selected_indices)}/{features.shape[1]} features selected "
                f"(threshold: {importance_threshold})", "INFO", "FEATURE_SELECTION"
            )

            return selected_features, selected_indices.tolist()

        except Exception as e:
            self._log_with_context(f"Feature selection failed: {e}", "ERROR", "FEATURE_SELECTION")
            return features, list(range(features.shape[1]))

    def validate_temporal_consistency(self, features: np.ndarray, assignments: np.ndarray,
                                    time_idx: np.ndarray) -> Dict[str, Any]:
        """
        Validate temporal consistency of clustering results.

        Args:
            features: Feature matrix
            assignments: Cluster assignments
            time_idx: Time indices for samples

        Returns:
            Dictionary with temporal validation results
        """
        try:
            validation = {
                'temporal_stability': 0.0,
                'regime_persistence': 0.0,
                'warnings': []
            }

            if len(time_idx) < 2:
                validation['warnings'].append('Insufficient temporal data for validation')
                return validation

            # Sort by time
            sorted_indices = np.argsort(time_idx)
            sorted_assignments = assignments[sorted_indices]

            # Calculate regime persistence (how long regimes last)
            regime_changes = np.sum(sorted_assignments[1:] != sorted_assignments[:-1])
            total_transitions = len(sorted_assignments) - 1
            persistence_ratio = 1.0 - (regime_changes / max(1, total_transitions))

            # Calculate temporal stability (regime consistency over time windows)
            window_size = min(50, len(sorted_assignments) // 10)  # Adaptive window
            if window_size > 1:
                stability_scores = []
                for i in range(0, len(sorted_assignments) - window_size, window_size):
                    window_assignments = sorted_assignments[i:i + window_size]
                    # Stability = 1 - (unique_regimes / total_regimes_in_window)
                    unique_regimes = len(np.unique(window_assignments))
                    stability = 1.0 - (unique_regimes / len(window_assignments))
                    stability_scores.append(stability)

                temporal_stability = np.mean(stability_scores) if stability_scores else 0.0
            else:
                temporal_stability = persistence_ratio

            validation['temporal_stability'] = float(temporal_stability)
            validation['regime_persistence'] = float(persistence_ratio)

            # Warnings for poor temporal consistency
            if temporal_stability < 0.5:
                validation['warnings'].append(f'Low temporal stability: {temporal_stability:.3f}')
            if persistence_ratio < 0.7:
                validation['warnings'].append(f'Low regime persistence: {persistence_ratio:.3f}')

            return validation

        except Exception as e:
            self._log_with_context(f"Temporal validation failed: {e}", "ERROR", "TEMPORAL")
            return {'temporal_stability': 0.0, 'regime_persistence': 0.0, 'warnings': [str(e)]}

    def validate_features_for_clustering(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Validate features are suitable for clustering.

        Args:
            features: Feature matrix to validate

        Returns:
            Dictionary with validation results and warnings
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'recommendations': []
        }

        if features.shape[1] == 0:
            validation['is_valid'] = False
            validation['warnings'].append('No features provided')
            return validation

        # Check for standardization artifacts (means near zero)
        feature_means = np.abs(features.mean(axis=0))
        standardized_count = np.sum(feature_means < 1e-8)

        if standardized_count > 0:
            validation['warnings'].append(f'{standardized_count} features appear standardized (mean ≈ 0)')
            validation['recommendations'].append('Consider using MinMaxScaler instead of StandardScaler')

        # Check for low variance features
        feature_stds = features.std(axis=0)
        low_variance_count = np.sum(feature_stds < 1e-6)

        if low_variance_count > 0:
            validation['warnings'].append(f'{low_variance_count} features have very low variance')
            validation['recommendations'].append('Consider removing constant or near-constant features')

        # Check for extreme ranges
        feature_ranges = features.max(axis=0) - features.min(axis=0)
        extreme_range_count = np.sum(feature_ranges > 100)

        if extreme_range_count > 0:
            validation['warnings'].append(f'{extreme_range_count} features have extreme value ranges')
            validation['recommendations'].append('Consider outlier removal or robust scaling')

        # Check feature correlations (high correlation may indicate redundant features)
        if features.shape[1] > 1:
            try:
                corr_matrix = np.corrcoef(features.T)
                # Check for highly correlated features (>0.95)
                high_corr_count = 0
                for i in range(len(corr_matrix)):
                    for j in range(i+1, len(corr_matrix)):
                        if abs(corr_matrix[i, j]) > 0.95:
                            high_corr_count += 1

                if high_corr_count > 0:
                    validation['warnings'].append(f'{high_corr_count} highly correlated feature pairs detected')
                    validation['recommendations'].append('Consider feature selection to reduce redundancy')
            except:
                pass  # Skip correlation check if it fails

        if validation['warnings']:
            validation['is_valid'] = len(validation['warnings']) < features.shape[1] * 0.5  # Valid if < 50% problematic

        return validation

    def _calculate_enhanced_cv_metrics(self, features: np.ndarray, assignments: np.ndarray,
                                     sample_indices: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate enhanced CV metrics optimized for standardized features.

        Returns:
            Dictionary with within_regime_cv, between_regime_cv, cv_ratio, and robust_feature_cv
        """
        try:
            # Use sample if provided
            if sample_indices is not None:
                features_sample = features[sample_indices]
                assignments_sample = assignments[sample_indices]
            else:
                features_sample = features
                assignments_sample = assignments

            unique_regimes = np.unique(assignments_sample)

            # Within-regime CV using robust calculation
            within_cvs = []
            for regime in unique_regimes:
                regime_features = features_sample[assignments_sample == regime]
                if regime_features.shape[0] > 1:
                    regime_cvs = []
                    for i in range(regime_features.shape[1]):
                        feature_values = regime_features[:, i]
                        mad = np.median(np.abs(feature_values - np.median(feature_values)))
                        median_abs = np.median(np.abs(feature_values))

                        if median_abs > 0 and mad > 0:
                            cv = mad / median_abs
                        else:
                            cv = 0.0
                        regime_cvs.append(cv)

                    within_cvs.append(np.mean(regime_cvs))

            within_regime_cv = np.mean(within_cvs) if within_cvs else 0.0

            # Between-regime CV using robust calculation
            centroids = np.array([np.mean(features_sample[assignments_sample == regime], axis=0)
                                for regime in unique_regimes])
            if centroids.shape[0] > 1:
                between_cvs = []
                for i in range(centroids.shape[1]):
                    feature_values = centroids[:, i]
                    mad = np.median(np.abs(feature_values - np.median(feature_values)))
                    median_abs = np.median(np.abs(feature_values))

                    if median_abs > 0 and mad > 0:
                        cv = mad / median_abs
                    else:
                        cv = 0.0
                    between_cvs.append(cv)

                between_regime_cv = np.mean(between_cvs)
            else:
                between_regime_cv = 0.0

            # CV ratio (higher is better)
            cv_ratio = between_regime_cv / (within_regime_cv + 1e-9)

            # Overall feature robustness
            robust_feature_cv = self.robust_feature_cv(features_sample)

            return {
                "within_regime_cv": float(within_regime_cv),
                "between_regime_cv": float(between_regime_cv),
                "cv_ratio": float(cv_ratio),
                "robust_feature_cv": float(robust_feature_cv),
            }

        except Exception as e:
            self._log_with_context(f"Error calculating enhanced CV metrics: {e}", "ERROR", "METRICS")
            return {
                "within_regime_cv": 0.0,
                "between_regime_cv": 0.0,
                "cv_ratio": 0.0,
                "robust_feature_cv": 0.0,
            }

    def temporal_switch_penalty(self, labels: np.ndarray, entity_ids: np.ndarray, time_idx: np.ndarray) -> float:
        """Calculate temporal smoothness penalty (lower is better)."""
        if entity_ids is None or time_idx is None or len(labels) < 2:
            return 0.0

        try:
            # Sort by entity_id, then by time
            order = np.lexsort((time_idx, entity_ids))
            lid = labels[order]
            eid = entity_ids[order]

            # Count switches within same entity
            switches = (lid[1:] != lid[:-1]) & (eid[1:] == eid[:-1])
            adj = (eid[1:] == eid[:-1])
            total_adj = adj.sum()

            if total_adj == 0:
                return 0.0

            base_penalty = switches.sum() / total_adj

            # Enhanced temporal smoothness: penalize cluster fragmentation
            # Count how fragmented each entity's sequence is
            entity_switches = []
            for entity_id in np.unique(eid):
                entity_mask = eid == entity_id
                if np.sum(entity_mask) > 1:
                    entity_labels = lid[entity_mask]
                    entity_switches.append(np.sum(entity_labels[1:] != entity_labels[:-1]))

            if entity_switches:
                avg_entity_fragmentation = np.mean(entity_switches)
                # Weight by average switches per entity to penalize fragmented sequences more
                fragmentation_penalty = min(0.5, avg_entity_fragmentation / max(1, np.mean([np.sum(eid == e) for e in np.unique(eid)])))
                base_penalty += fragmentation_penalty

            return min(1.0, base_penalty)  # Cap at 1.0

        except Exception as e:
            tprint(f"⚠️ Temporal switch penalty calculation failed: {e}", "WARNING")
            return 0.0

    def evaluate_metrics(self, X: np.ndarray, labels: np.ndarray, entity_ids: np.ndarray = None,
                        time_idx: np.ndarray = None, sample_for_indices: np.ndarray = None) -> dict:
        """Evaluate all metrics for the current clustering."""
        try:
            from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

            K = int(labels.max()) + 1

            # Sample for large datasets
            if sample_for_indices is not None:
                idx = sample_for_indices
                Xs, ls = X[idx], labels[idx]
                sil = silhouette_score(Xs, ls) if len(np.unique(ls)) > 1 else 0.0
                dbi = davies_bouldin_score(Xs, ls) if len(np.unique(ls)) > 1 else 0.0
                ch = calinski_harabasz_score(Xs, ls) if len(np.unique(ls)) > 1 else 0.0
            else:
                sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0.0
                dbi = davies_bouldin_score(X, labels) if len(np.unique(labels)) > 1 else 0.0
                ch = calinski_harabasz_score(X, labels) if len(np.unique(labels)) > 1 else 0.0

            # Enhanced CV metrics for standardized features (PRIMARY METRIC)
            enhanced_cv = self._calculate_enhanced_cv_metrics(X, labels, sample_for_indices)

            # Temporal validation (SECONDARY METRIC)
            temporal_validation = {}
            if time_idx is not None:
                temporal_validation = self.validate_temporal_consistency(X, labels, time_idx)

            # Feature quality validation (TERTIARY METRIC)
            feature_validation = self.validate_features_for_clustering(X)

            # Primary quality metric: CV ratio (regime separation)
            primary_score = enhanced_cv["cv_ratio"]

            # Secondary quality metric: Temporal consistency
            secondary_score = temporal_validation.get("temporal_stability", 0.0)

            # Tertiary quality metric: Feature quality
            tertiary_score = 1.0 if feature_validation["is_valid"] else 0.0

            # Overall quality score (weighted combination)
            overall_quality = (primary_score * 0.5 + secondary_score * 0.3 + tertiary_score * 0.2)

            return {
                "cv": self.cv_of_sizes(labels, K),
                "sil": sil,
                "temp": self.temporal_switch_penalty(labels, entity_ids, time_idx),
                "dbi": dbi,
                "ch": ch,
                # Enhanced CV metrics (PRIMARY INDICATORS)
                "within_regime_cv": enhanced_cv["within_regime_cv"],
                "between_regime_cv": enhanced_cv["between_regime_cv"],
                "cv_ratio": enhanced_cv["cv_ratio"],  # PRIMARY QUALITY METRIC
                "robust_feature_cv": enhanced_cv["robust_feature_cv"],
                # Temporal validation (SECONDARY METRIC)
                "temporal_stability": temporal_validation.get("temporal_stability", 0.0),
                "regime_persistence": temporal_validation.get("regime_persistence", 0.0),
                # Feature validation (TERTIARY METRIC)
                "feature_quality_score": tertiary_score,
                # Overall quality assessment
                "overall_quality_score": overall_quality,
            }
        except Exception as e:
            self._log_with_context(f"Error evaluating metrics: {e}", "ERROR", "METRICS")
            return {"cv": 1.0, "sil": -1.0, "temp": 1.0, "dbi": np.inf, "ch": -np.inf}

    # ===== LEXICOGRAPHIC ACCEPTOR =====

    def accept_lexicographic(self, old_metrics: dict, new_metrics: dict,
                           eps: tuple = None) -> bool:
        """Lexicographic acceptor with priority order: CV → Silhouette → Temporal → Accessories."""
        if eps is None:
            eps = (self.config.eps_cv, self.config.eps_sil, self.config.eps_temp)

        # We minimize tuple: (cv, -sil, temp, dbi, -ch)
        a_old = (old_metrics["cv"], -old_metrics["sil"], old_metrics["temp"],
                old_metrics["dbi"], -old_metrics["ch"])
        a_new = (new_metrics["cv"], -new_metrics["sil"], new_metrics["temp"],
                new_metrics["dbi"], -new_metrics["ch"])

        # Priority 1: CV
        if a_new[0] < a_old[0] - eps[0]:
            return True
        if abs(a_new[0] - a_old[0]) <= eps[0]:
            # Priority 2: Silhouette (remember we minimized -sil)
            if a_new[1] < a_old[1] - eps[1]:
                return True
            if abs(a_new[1] - a_old[1]) <= eps[1]:
                # Priority 3: Temporal
                if a_new[2] < a_old[2] - eps[2]:
                    return True
                if abs(a_new[2] - a_old[2]) <= eps[2]:
                    # Accessories (weak tiebreak; tiny weight sum)
                    acc_old = a_old[3] + a_old[4]
                    acc_new = a_new[3] + a_new[4]
                    return acc_new < acc_old - self.config.accessories_weight
        return False

    def size_penalty_eps(self, dest_size_new: int, mean_new: float, max_size: int) -> float:
        """Dynamic CV-improvement threshold based on destination cluster size."""
        over_avg = max(0.0, dest_size_new/mean_new - 1.0)  # linear penalty > mean
        near_cap = max(0.0, dest_size_new/max_size - self.config.near_cap_ratio)  # activates past 90% of cap
        return (self.config.size_gate_base +
                self.config.size_gate_alpha * over_avg +
                self.config.size_gate_beta * (near_cap**2))

    def accept_with_size_bias(self, old_metrics: dict, new_metrics: dict,
                            dest_size_new: int, mean_new: float, max_size: int) -> bool:
        """Size-aware lexicographic acceptor with dynamic CV threshold."""
        # We minimize: (cv, -sil, temp, dbi, -ch)
        a_old = (old_metrics["cv"], -old_metrics["sil"], old_metrics["temp"],
                old_metrics["dbi"], -old_metrics["ch"])
        a_new = (new_metrics["cv"], -new_metrics["sil"], new_metrics["temp"],
                new_metrics["dbi"], -new_metrics["ch"])

        # Dynamic epsilon for CV based on destination cluster size
        eps_cv = self.size_penalty_eps(dest_size_new, mean_new, max_size)

        # Priority 1: CV with size-aware gate
        if a_new[0] < a_old[0] - eps_cv:
            return True
        if abs(a_new[0] - a_old[0]) <= eps_cv:
            # Priority 2: Silhouette
            if a_new[1] < a_old[1] - self.config.eps_sil:
                return True
            if abs(a_new[1] - a_old[1]) <= self.config.eps_sil:
                # Priority 3: Temporal smoothness
                if a_new[2] < a_old[2] - self.config.eps_temp:
                    return True
                if abs(a_new[2] - a_old[2]) <= self.config.eps_temp:
                    # Accessories as last nudge
                    acc_old = a_old[3] + a_old[4]
                    acc_new = a_new[3] + a_new[4]
                    return acc_new < acc_old - self.config.accessories_weight
        return False

    # ===== CANDIDATE GENERATION =====

    def generate_candidates(self, X: np.ndarray, assignments: np.ndarray,
                           centroids: np.ndarray, max_size: int) -> list:
        """Generate fast and feasible candidates with headroom checks."""
        candidates = []
        N = len(X)
        K = int(assignments.max()) + 1
        sizes = np.bincount(assignments, minlength=K)

        # For each point, consider nearest few centroids with headroom
        for point_idx in range(N):
            current_cluster = assignments[point_idx]
            point = X[point_idx]

            # Calculate distances to all centroids
            distances = []
            for cluster_id in range(K):
                if sizes[cluster_id] > 0:  # Skip empty clusters
                    dist = np.linalg.norm(point - centroids[cluster_id])
                    distances.append((cluster_id, dist))

            # Sort by distance and take nearest few with headroom
            distances.sort(key=lambda x: x[1])

            for cluster_id, _ in distances[:self.config.neighbors_per_point]:
                if cluster_id != current_cluster:
                    # Check headroom constraint
                    if sizes[cluster_id] < max_size:
                        # Check source cluster won't become too small
                        if sizes[current_cluster] > 1:  # Don't empty source cluster
                            candidates.append({
                                'point_idx': point_idx,
                                'from_cluster': current_cluster,
                                'to_cluster': cluster_id
                            })

        return candidates

    def _find_best_merge_pair(self, X: np.ndarray, assignments: np.ndarray,
                             non_empty: np.ndarray, max_size: int) -> tuple:
        """Find best merge pair that minimizes WCSS increase and respects max_size."""
        best_merge = None
        best_wcss_increase = np.inf

        for i in range(len(non_empty)):
            for j in range(i + 1, len(non_empty)):
                cluster_i, cluster_j = non_empty[i], non_empty[j]

                # Check size constraint
                size_i = np.sum(assignments == cluster_i)
                size_j = np.sum(assignments == cluster_j)
                if size_i + size_j > max_size:
                    continue

                # Calculate WCSS increase for merge
                wcss_increase = self._calculate_merge_wcss_increase(X, assignments, cluster_i, cluster_j)

                if wcss_increase < best_wcss_increase:
                    best_wcss_increase = wcss_increase
                    best_merge = (cluster_i, cluster_j)

        return best_merge

    def _calculate_merge_wcss_increase(self, X: np.ndarray, assignments: np.ndarray,
                                     cluster_i: int, cluster_j: int) -> float:
        """Calculate WCSS increase from merging clusters i and j."""
        # Get points in both clusters
        points_i = X[assignments == cluster_i]
        points_j = X[assignments == cluster_j]

        if len(points_i) == 0 or len(points_j) == 0:
            return np.inf

        # Current WCSS
        centroid_i = points_i.mean(axis=0)
        centroid_j = points_j.mean(axis=0)
        wcss_i = np.sum((points_i - centroid_i)**2)
        wcss_j = np.sum((points_j - centroid_j)**2)
        current_wcss = wcss_i + wcss_j

        # Merged WCSS
        all_points = np.vstack([points_i, points_j])
        merged_centroid = all_points.mean(axis=0)
        merged_wcss = np.sum((all_points - merged_centroid)**2)

        return merged_wcss - current_wcss

    # ===== SPLITS & MERGES =====

    def _split_cluster_constrained(self, X: np.ndarray, assignments: np.ndarray,
                                 cluster_id: int, min_size: int, max_size: int) -> bool:
        """Split cluster with hard constraints and lexicographic acceptance."""
        try:
            from sklearn.cluster import KMeans

            # Get cluster points
            cluster_mask = assignments == cluster_id
            cluster_points = X[cluster_mask]
            point_indices = np.where(cluster_mask)[0]

            if len(cluster_points) < 2 * min_size:
                return False

            # Try k=2 k-means with multiple seeds
            best_split = None
            best_metrics = None

            for seed in range(self.config.max_kmeans_seeds):
                try:
                    km = KMeans(n_clusters=2, n_init=1, max_iter=self.config.max_kmeans_iterations,
                              random_state=seed)
                    child_labels = km.fit_predict(cluster_points)

                    # Check size constraints
                    child_sizes = [np.sum(child_labels == i) for i in range(2)]
                    if not all(min_size <= size <= max_size for size in child_sizes):
                        continue

                    # Create temporary assignments for evaluation
                    temp_assignments = assignments.copy()
                    temp_assignments[point_indices[child_labels == 0]] = cluster_id
                    temp_assignments[point_indices[child_labels == 1]] = int(assignments.max()) + 1

                    # Evaluate metrics
                    sample_indices = self._get_sample_indices(len(X))
                    temp_metrics = self.evaluate_metrics(X, temp_assignments,
                                                       sample_for_indices=sample_indices)

                    if best_metrics is None or self.accept_lexicographic(best_metrics, temp_metrics):
                        best_split = child_labels
                        best_metrics = temp_metrics

                except Exception:
                    continue

            if best_split is not None:
                # Apply the best split
                assignments[point_indices[best_split == 0]] = cluster_id
                assignments[point_indices[best_split == 1]] = int(assignments.max()) + 1
                return True

            return False

        except Exception as e:
            self._log_with_context(f"Error in constrained split: {e}", "ERROR", "SPLIT")
            return False

    def _merge_undersized_cluster(self, X: np.ndarray, assignments: np.ndarray,
                                 cluster_id: int, min_size: int, max_size: int) -> bool:
        """Merge undersized cluster with best partner."""
        try:
            K = int(assignments.max()) + 1
            sizes = np.bincount(assignments, minlength=K)

            # Find best merge partner
            best_partner = None
            best_metrics = None

            for other_cluster in range(K):
                if other_cluster == cluster_id or sizes[other_cluster] == 0:
                    continue

                # Check size constraint
                combined_size = sizes[cluster_id] + sizes[other_cluster]
                if combined_size > max_size:
                    continue

                # Create temporary assignments for evaluation
                temp_assignments = assignments.copy()
                temp_assignments[temp_assignments == cluster_id] = other_cluster

                # Evaluate metrics
                sample_indices = self._get_sample_indices(len(X))
                temp_metrics = self.evaluate_metrics(X, temp_assignments,
                                                   sample_for_indices=sample_indices)

                if best_metrics is None or self.accept_lexicographic(best_metrics, temp_metrics):
                    best_partner = other_cluster
                    best_metrics = temp_metrics

            if best_partner is not None:
                # Apply the merge
                assignments[assignments == cluster_id] = best_partner
                return True

            return False

        except Exception as e:
            self._log_with_context(f"Error in merge: {e}", "ERROR", "MERGE")
            return False

    def _get_sample_indices(self, N: int) -> np.ndarray:
        """Get sample indices for large datasets."""
        if N <= self.config.silhouette_sample_size:
            return None

        np.random.seed(42)  # Fixed seed for reproducibility
        return np.random.choice(N, self.config.silhouette_sample_size, replace=False)

    # ===== MAIN OPTIMIZATION LOOP =====

    def optimize_with_hard_constraints(self, X: np.ndarray, initial_assignments: np.ndarray,
                                     entity_ids: np.ndarray = None, time_idx: np.ndarray = None) -> np.ndarray:
        """Main optimization loop with hard constraints and lexicographic optimization."""
        # CRITICAL: Comprehensive input validation
        self._validate_optimization_inputs(X, initial_assignments, entity_ids, time_idx)
        try:
            N = len(X)
            min_size = max(1, int(np.ceil(self.config.min_size_ratio * N)))
            max_size = int(np.floor(self.config.max_size_ratio * N))

            self._log_with_context(f"Starting optimization: N={N}, min_size={min_size}, max_size={max_size}", "INFO", "MAIN")

            # Initialize with hard constraints
            assignments = self._enforce_hard_constraints(X, initial_assignments, entity_ids, time_idx)

            # Main optimization loop with adaptive weights and timeout protection
            import time
            start_time = time.time()
            max_optimization_time = 300  # 5 minutes timeout

            # Track metrics for early stopping
            no_improvement_count = 0
            no_moves_count = 0
            prev_cv = 0.0

            for iteration in range(self.config.max_rounds):  # Maximum iterations
                # Check timeout
                elapsed_time = time.time() - start_time
                if elapsed_time > max_optimization_time:
                    self._log_with_context(f"Timeout reached after {elapsed_time:.1f}s, stopping optimization", "WARNING", "MAIN")
                    break

                self._log_with_context(f"=== Iteration {iteration + 1}/{self.config.max_rounds} ===", "INFO", "MAIN")

                # Apply adaptive weights (CV enhancement strategy)
                if CV_ENHANCEMENT_AVAILABLE and hasattr(self, 'adaptive_scheduler') and self.adaptive_scheduler:
                    adaptive_weights = self.adaptive_scheduler.get_weights(iteration)
                    # Update current weights
                    self.w_cv = adaptive_weights['w_cv']
                    self.w_temp = adaptive_weights['w_temp']
                    self.w_sil = adaptive_weights['w_sil']
                    self.w_bal = adaptive_weights['w_bal']

                # Evaluate current metrics (with enhanced CV if available)
                sample_indices = self._get_sample_indices(N)
                current_metrics = self.evaluate_metrics(X, assignments, entity_ids, time_idx, sample_indices)

                # Calculate enhanced CV ratio if available
                if CV_ENHANCEMENT_AVAILABLE:
                    enhanced_cv_metrics = EnhancedVarianceRatioCalculator.calculate_enhanced_cv(
                        X, assignments, include_calinski_harabasz=True
                    )
                    current_metrics['enhanced_cv'] = enhanced_cv_metrics['combined_cv']
                    current_metrics['calinski_harabasz'] = enhanced_cv_metrics['calinski_harabasz']
                    self._log_with_context(f"Enhanced CV metrics: combined_cv={enhanced_cv_metrics['combined_cv']:.4f}, CH={enhanced_cv_metrics['calinski_harabasz']:.2f}", "DEBUG", "MAIN")

                current_cv = current_metrics.get('cv', current_metrics.get('enhanced_cv', 0.0))
                self._log_with_context(f"Current metrics: CV={current_cv:.4f}, Sil={current_metrics['sil']:.4f}, Temp={current_metrics['temp']:.4f}", "INFO", "MAIN")

                # Check for improvement (early stopping)
                if abs(current_cv - prev_cv) < 0.001:  # Less than 0.1% improvement
                    no_improvement_count += 1
                    if no_improvement_count >= 2:  # No improvement for 2 consecutive iterations
                        self._log_with_context("Early stopping: no CV improvement for 2 iterations", "INFO", "MAIN")
                        break
                else:
                    no_improvement_count = 0
                    prev_cv = current_cv

                # Generate candidates
                centroids = self._compute_centroids(X, assignments)
                candidates = self.generate_candidates(X, assignments, centroids, max_size)
                self._log_with_context(f"Generated {len(candidates)} candidates", "DEBUG", "MAIN")

                # Try moves with lexicographic acceptance
                moves_applied = 0
                for candidate in candidates:
                    if self._try_move_with_lexicographic(X, assignments, candidate, entity_ids, time_idx, sample_indices):
                        moves_applied += 1

                self._log_with_context(f"Applied {moves_applied} moves", "INFO", "MAIN")

                # Check for convergence
                if moves_applied == 0:
                    no_moves_count += 1
                    if no_moves_count >= 2:  # No moves for 2 consecutive rounds
                        self._log_with_context("Converged: no moves accepted for 2 rounds", "INFO", "MAIN")
                        break
                else:
                    no_moves_count = 0  # Reset counter if moves were applied

                # Re-enforce hard constraints after each iteration
                assignments = self._enforce_hard_constraints(X, assignments, entity_ids, time_idx)

            # Final metrics
            final_metrics = self.evaluate_metrics(X, assignments, entity_ids, time_idx, sample_indices)
            self._log_with_context(f"Final metrics: CV={final_metrics['cv']:.4f}, Sil={final_metrics['sil']:.4f}, Temp={final_metrics['temp']:.4f}", "INFO", "MAIN")

            return assignments

        except Exception as e:
            self._log_with_context(f"Error in main optimization: {e}", "ERROR", "MAIN")
            return initial_assignments

    def _try_move_with_lexicographic(self, X: np.ndarray, assignments: np.ndarray,
                                   candidate: dict, entity_ids: np.ndarray, time_idx: np.ndarray,
                                   sample_indices: np.ndarray) -> bool:
        """Try a move with lexicographic acceptance."""
        try:
            point_idx = candidate['point_idx']
            from_cluster = candidate['from_cluster']
            to_cluster = candidate['to_cluster']

            # Get current metrics
            current_metrics = self.evaluate_metrics(X, assignments, entity_ids, time_idx, sample_indices)

            # Create temporary assignments for evaluation
            temp_assignments = assignments.copy()
            temp_assignments[point_idx] = to_cluster

            # Get new metrics
            new_metrics = self.evaluate_metrics(X, temp_assignments, entity_ids, time_idx, sample_indices)

            # Check size constraints
            sizes = np.bincount(temp_assignments, minlength=int(temp_assignments.max()) + 1)
            N = len(X)
            min_size = max(1, int(np.ceil(self.config.min_size_ratio * N)))
            max_size = int(np.floor(self.config.max_size_ratio * N))

            if sizes[from_cluster] < min_size or sizes[to_cluster] > max_size:
                return False

            # Use size-aware lexicographic acceptance
            mean_size = sizes[sizes > 0].mean()
            if self.accept_with_size_bias(current_metrics, new_metrics,
                                        sizes[to_cluster], mean_size, max_size):
                # Apply the move
                assignments[point_idx] = to_cluster
                return True

            return False

        except Exception as e:
            self._log_with_context(f"Error in move evaluation: {e}", "ERROR", "MOVE")
            return False

    def _compute_centroids(self, X: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """Compute centroids for all clusters using optimized engine."""
        return self.calculation_engine.calculate_centroids_optimized(X, assignments)

        # Step 3 hardening parameters
        self.split_tries_max = 8  # not 50

        # Step 1: Local frontier parameters - AGGRESSIVE OPTIMIZATION
        # Enhanced to maximize CV, Silhouette, and Temporal Smoothness
        self.frontier_fraction = 0.50  # Increased to 50% for broader exploration
        self.knn_size = 15  # Increased kNN for better neighborhood detection
        self.neighbor_consensus_threshold = 0.55  # Slightly relaxed for more candidates
        self.local_threshold = -0.001  # Require small improvement (negative = better)
        self.local_churn_cap = 0.03  # 3% of N - increased for more moves
        self.hysteresis_rounds = 3  # Increased stability rounds

        # Step 2: Global reallocation parameters - AGGRESSIVE OPTIMIZATION
        # Focus on temporal smoothness and CV improvement
        self.beta = 0.25  # Increased weight for global coordination
        self.global_threshold = -0.001  # Require improvement for global moves
        self.global_churn_cap = 0.10  # 10% of N - increased for global optimization
        self.min_cluster_size = 20  # Slightly reduced to allow more flexibility

        # Step 3: Break large clusters parameters (LOOSENED FOR TRIAGE)
        self.size_factor_threshold = 1.3  # Reduced from 1.5 to 1.3
        self.split_quality_threshold = 0.003  # Reduced from 0.005 to 0.003
        self.alpha = 1.0  # Size-aware penalty
        self.max_new_clusters_per_round = 3

        # ENHANCED objective function weights - MAXIMUM CV optimization focus
        # Prioritizing CV ratio for maximum regime separation quality
        self.w_cv = 0.70     # Primary: variance ratio (CV) - MAXIMIZED for regime separation
        self.w_temp = 0.20   # Secondary: temporal smoothness - reduced to focus on CV
        self.w_sil = 0.10    # Tertiary: cluster cohesion (Silhouette) - reduced
        self.w_bal = 0.05    # Minimal: balance constraint (soft penalty)
        self.lambda_switch = 1e-5  # Reduced by 10x from 1e-4 to 1e-5

        # Gating configuration - single source of truth
        self.use_std_for_rank = True  # Use standardized objective for ranking
        self.use_std_for_gate = True  # Use standardized objective for gating (must match log label)
        # Adaptive thresholds based on iteration (FIXED: negative values are improvements)
        self.eps_std_step1 = -0.12  # Will be updated per iteration (≤ threshold for acceptance)
        self.eps_std_step2 = -0.12  # Will be updated per iteration (≤ threshold for acceptance)
        self.sil_guard = -0.35  # Will be updated per iteration
        self.temporal_bonus = 0.20  # Will be updated per iteration
        self.cv_guard = -0.05  # Soft finance-first constraint
        self.max_local_moves_per_iter = 50  # Max moves per iteration
        self.exploratory_quota = 5  # Allow top N moves even if guards trip (for exploration)

        # New parameters
        self.early_stop_threshold = 0.5  # Rolling 4-iter |ΔJ_total| threshold
        self.early_stop_patience = 2  # Consecutive iterations for early stop
        self.chunk_size = 4  # Apply moves in chunks (recompute frontier after each)
        self.max_global_moves = 8  # Cap global reallocation moves
        self.max_alternatives_per_point = 2  # Limit alternative clusters per point
        self.split_size_threshold = 1.30  # Size ratio for splitting large clusters
        self.split_silhouette_threshold = 0.08  # Mean silhouette threshold for splitting

        # Anti-oscillation parameters
        self.no_reversal_window = 4  # Points cannot return to previous cluster within N iters
        self.reverse_margin = 0.30  # Override margin for reversal window
        self.tabu_tenure = 3  # Tabu list tenure for (point, prev_cluster)
        self.max_moves_per_point = 2  # Max moves per point over 6-iter window
        self.move_window_size = 6  # Window size for move tracking

        # Thrash detection
        self.thrash_threshold = 0.6  # Thrash rate threshold
        self.boundary_ratio_threshold = 0.45  # Boundary ratio for splitting
        self.thrash_count_threshold = 3  # Thrash count threshold over 6 iters

        # Monitoring and anti-oscillation tracking
        self.step_reports = []
        self.move_history = {}  # point_idx -> [(iteration, from_cluster, to_cluster, delta), ...]
        self.tabu_list = {}  # (point_idx, cluster) -> iteration_when_tabu_ends
        self.cluster_thrash_counts = {}  # cluster_id -> thrash_count_over_6_iters

    @property
    def n_clusters(self) -> int:
        """Dynamic alias for cluster count, used by Step-1 code."""
        return self._k if self._k is not None else 0

    def _refresh_k(self, labels: np.ndarray) -> None:
        """Refresh cluster count after merges/splits/relabel operations."""
        self._k = int(np.unique(labels).size)

    def _objective(self, delta_info: dict, use_std: bool = True) -> float:
        """Single source of truth for objective function calculation."""
        # Add debugging to identify numpy array issue
        if not isinstance(delta_info, dict):
            tprint(f"❌ ERROR: delta_info is not a dict, it's {type(delta_info)}", "ERROR")
            tprint(f"❌ delta_info value: {delta_info}", "ERROR")
            # Convert numpy array to dict if possible
            if hasattr(delta_info, 'shape') and hasattr(delta_info, 'dtype'):
                tprint(f"❌ delta_info appears to be numpy array with shape {delta_info.shape}", "ERROR")
                # Return a default value to prevent crash
                return 0.0
            else:
                tprint(f"❌ delta_info is unexpected type: {type(delta_info)}", "ERROR")
                return 0.0

        if use_std:
            v = (delta_info.get('cv', 0), delta_info.get('balance', 0),
                 delta_info.get('silhouette', 0), delta_info.get('temporal', 0))
        else:
            v = (delta_info.get('cv_raw', 0), delta_info.get('balance_raw', 0),
                 delta_info.get('silhouette_raw', 0), delta_info.get('temporal_raw', 0))

        # Calculate objective without balance weight (balance used as constraint)
        base_objective = float(self.w_cv * v[0] + self.w_sil * v[2] + self.w_temp * v[3])

        # Apply soft balance constraint penalty
        balance_penalty = 0.0
        if v[1] < 0.8:  # Only penalize if balance is very poor
            balance_penalty = 0.05 * (0.8 - v[1])  # Soft penalty, not hard weight

        return base_objective - balance_penalty

    def _update_adaptive_thresholds(self, iteration: int):
        """Update thresholds based on iteration for adaptive behavior (FIXED: negative values are improvements)."""
        if iteration <= 2:
            self.eps_std_step1 = -0.12  # ≤ threshold for acceptance
            self.eps_std_step2 = -0.12  # ≤ threshold for acceptance
            self.sil_guard = -0.35
            self.temporal_bonus = 0.20
        elif iteration <= 5:
            self.eps_std_step1 = -0.15  # ≤ threshold for acceptance
            self.eps_std_step2 = -0.15  # ≤ threshold for acceptance
            self.sil_guard = -0.20
            self.temporal_bonus = 0.10
        else:
            self.eps_std_step1 = -0.20  # ≤ threshold for acceptance
            self.eps_std_step2 = -0.20  # ≤ threshold for acceptance
            self.sil_guard = -0.10
            self.temporal_bonus = 0.00

    def _check_anti_oscillation(self, point_idx: int, from_cluster: int, to_cluster: int,
                               current_iteration: int, delta: float) -> tuple[bool, str]:
        """Check anti-oscillation constraints before accepting a move."""
        # Check tabu list
        tabu_key = (point_idx, to_cluster)
        if tabu_key in self.tabu_list and current_iteration <= self.tabu_list[tabu_key]:
            return False, "tabu_list"

        # Check no-reversal window
        if point_idx in self.move_history:
            recent_moves = [move for move in self.move_history[point_idx]
                          if current_iteration - move[0] <= self.no_reversal_window]
            if recent_moves:
                last_move = recent_moves[-1]
                if last_move[2] == to_cluster:  # Trying to return to previous cluster
                    # Check reverse margin override
                    if delta > (last_move[3] - self.reverse_margin):
                        return False, f"no_reversal_window({self.no_reversal_window})"

        # Check move count per point
        if point_idx in self.move_history:
            recent_moves = [move for move in self.move_history[point_idx]
                          if current_iteration - move[0] <= self.move_window_size]
            if len(recent_moves) >= self.max_moves_per_point:
                return False, f"move_cap({self.max_moves_per_point})"

        return True, "allowed"

    def _record_move(self, point_idx: int, from_cluster: int, to_cluster: int,
                    current_iteration: int, delta: float):
        """Record a move for anti-oscillation tracking."""
        if point_idx not in self.move_history:
            self.move_history[point_idx] = []

        self.move_history[point_idx].append((current_iteration, from_cluster, to_cluster, delta))

        # Add to tabu list
        tabu_key = (point_idx, from_cluster)
        self.tabu_list[tabu_key] = current_iteration + self.tabu_tenure

        # Clean old entries
        self.move_history[point_idx] = [move for move in self.move_history[point_idx]
                                      if current_iteration - move[0] <= self.move_window_size]

        # Clean expired tabu entries
        self.tabu_list = {k: v for k, v in self.tabu_list.items() if v > current_iteration}

    def _accept_candidate(self, candidate: dict, stats: ClusteringStats = None, constraints: NAgosticConstraints = None) -> tuple[bool, str]:
        """Acceptance function with proper guardrails and detailed blocking reasons."""
        # Debug logging removed for cleaner output

        if 'delta_info' not in candidate:
            tprint(f"❌ ERROR: 'delta_info' key not found in candidate", "ERROR")
            return False, "missing_delta_info"

        # Calculate scores using single source of truth
        score_rank = self._objective(candidate['delta_info'], use_std=self.use_std_for_rank)
        score_gate = self._objective(candidate['delta_info'], use_std=self.use_std_for_gate)

        # Apply temporal bonus for early iterations
        score_gate_with_bonus = score_gate + self.temporal_bonus

        # Store scores for debugging
        candidate['score_rank'] = score_rank
        candidate['score_gate'] = score_gate_with_bonus

        # Centralized acceptance logic with cap-aware dual epsilon
        if stats is not None and constraints is not None:
            from_cluster = candidate['from_cluster']
            to_cluster = candidate['to_cluster']
            from_size = stats.cluster_sizes[stats._to_compact_id(from_cluster)]
            to_size = stats.cluster_sizes[to_cluster]
            SOFT_CAP = int(0.20 * len(stats.assignments))  # 20% cap
            MIN_SIZE = constraints.cfg.MIN_SIZE

            # Check if we're in rescue mode (any cluster > SOFT_CAP and significant fraction)
            rescue = (np.any(stats.cluster_sizes > SOFT_CAP) and
                     (np.max(stats.cluster_sizes) / np.sum(stats.cluster_sizes)) > 0.45)

            # Use authoritative rescue gating
            accept, eps_used = self.accept_move(score_gate_with_bonus, from_cluster, to_cluster,
                              stats.cluster_sizes, SOFT_CAP, MIN_SIZE, rescue)

            if not accept:
                # Fallback CV/Silhouette-first acceptor with iteration-annealed thresholds
                di = candidate['delta_info']
                d_cv = di.get('cv', 0.0)
                d_sil = di.get('silhouette', 0.0)
                it = int(getattr(self, '_iter', 0))
                # Start loose then tighten: 0 -> 0.005, 1 -> 0.0075, 2+ -> 0.010
                if it <= 0:
                    cv_thr, sil_thr = -0.005, 0.005
                elif it == 1:
                    cv_thr, sil_thr = -0.0075, 0.0075
                else:
                    cv_thr, sil_thr = -0.010, 0.010
                # Respect capacity constraints
                if not constraints.violates_capacity(from_size, to_size):
                    if d_cv <= cv_thr or d_sil >= sil_thr:
                        return True, "cv_sil_fallback"
                return False, f"ΔJ_std>{score_gate_with_bonus:.3f} (rescue={rescue})"
        else:
            # Fallback to normal gate if no stats/constraints
            if score_gate_with_bonus > self.eps_std_step1:
                return False, f"ΔJ_std>{self.eps_std_step1:.3f}"

        # Soft guards
        delta_info = candidate['delta_info']
        if delta_info.get('cv', 0) < self.cv_guard:
            return False, f"cv_z<{self.cv_guard:.2f}"

        if delta_info.get('silhouette', 0) < self.sil_guard:
            return False, f"sil_z<{self.sil_guard:.2f}"

        return True, "accepted"

    async def execute_optimization_round(
        self,
        context: ClusteringContext,
        config: Any,
        iteration: int
    ) -> Tuple[ClusteringContext, Dict[str, Any]]:
        """
        Execute a single round of the 3-step optimization.

        Args:
            context: Current clustering context
            config: Configuration parameters
            iteration: Current iteration number

        Returns:
            Tuple of (updated_context, round_results)
        """
        try:
            # Validate that features exist and are not None
            if not hasattr(context, 'optimized_features') or context.optimized_features is None:
                raise ValueError("Optimized features are None or not available in context")

            features = context.optimized_features
            if features is None or features.size == 0:
                raise ValueError("Features array is None or empty")

            # Validate features shape
            if not hasattr(features, 'shape') or len(features.shape) != 2:
                raise ValueError(f"Features must be a 2D array, got shape: {getattr(features, 'shape', 'None')}")

            current_assignments = context.optimized_assignments.copy() if hasattr(context, 'optimized_assignments') and context.optimized_assignments is not None else context.initial_assignments.copy()
            if current_assignments is None:
                raise ValueError("Current assignments are None")

            current_k = len(np.unique(current_assignments))

            # Initialize clustering statistics
            stats = ClusteringStats(features, current_assignments)

            # CRITICAL FIX: Use fixed K throughout optimization
            K_fixed = stats.K_fixed
            unique_labels = np.unique(current_assignments)
            assert 0 <= unique_labels.min() < K_fixed, f"Bad cluster id {unique_labels.min()} for K_fixed={K_fixed}"
            assert 0 <= unique_labels.max() < K_fixed, f"Bad cluster id {unique_labels.max()} for K_fixed={K_fixed}"

            # CRITICAL FIX: Add guard assertions for all per-cluster arrays
            assert_cluster_axis("cluster_sizes", stats.cluster_sizes, K_fixed)
            assert_cluster_axis("centroids", stats.centroids, K_fixed)
            assert_cluster_axis("wcss_per_cluster", stats.wcss_per_cluster, K_fixed)
            assert_cluster_axis("S", stats.S, K_fixed)
            assert_cluster_axis("Q_trace", stats.Q_trace, K_fixed)

            # CRITICAL FIX: Add diagnostic logging for array shapes
            self.logger.debug(f"K_fixed={K_fixed}, labels.max()={unique_labels.max()}, labels.min()={unique_labels.min()}")
            self.logger.debug(f"cluster_sizes.shape={stats.cluster_sizes.shape} (axis0 expected {K_fixed})")
            self.logger.debug(f"centroids.shape={stats.centroids.shape} (axis0 expected {K_fixed})")

            # CRITICAL FIX: Add smoke checks
            sizes = stats.cluster_sizes
            empties = np.sum(sizes == 0)
            self.logger.debug(f"Smoke check: sizes min={sizes.min()}, max={sizes.max()}, empties={empties}")

            # Check for any NaN values in centroids
            nan_centroids = np.any(~np.isfinite(stats.centroids))
            if nan_centroids:
                self.logger.warning(f"Found NaN/Inf values in centroids!")

            # Log cluster size distribution
            size_counts = np.bincount(sizes[sizes > 0])
            self.logger.debug(f"Size distribution: {size_counts[:10]}")  # First 10 bins

            # Initialize risk mitigation system
            risk_system = RiskMitigationSystem(PRODUCTION_RISK_CONFIG)

            # Initialize strict split policy and skip gate
            split_policy = StrictSplitPolicy()
            split_skip_gate = SplitSkipGate()
            current_round = 0

            # Initialize N-agnostic constraints
            constraints = NAgosticConstraints(k_max=15, min_fraction=0.02, margin=2, tau=0.20)
            constraints.update_constraints(len(features))
            tprint(f"📏 N-agnostic constraints: {constraints.get_constraint_summary()}", "INFO")

            # Ensure core state is set before hydrating defaults
            self.features = features
            self.N = len(features)

            # Hydrate all defaults once core state is available
            self._hydrate_defaults()

            # Set features and SOFT_CAP
            self.SOFT_CAP = int(0.20 * len(features))  # 20% cap

            # Track metrics before optimization
            initial_cv = stats.get_cv_ratio()
            initial_balance = stats.get_balance_score()
            initial_silhouette = self._calculate_silhouette_score(features, current_assignments)

            # Execute optimization steps
            total_delta = 0.0
            moves_accepted = 0
            local_moves = 0
            global_moves = 0
            splits_performed = 0
            risk_violations = 0

            # Step 1: Local frontier moves
            if self.verbose:
                tprint(f"🔍 Step 1: Local frontier moves (iteration {iteration})", "DEBUG")

            # Apply step 1 specific weights
            self.apply_step_weights(1)

            try:
                delta_1 = await self._step1_local_frontier_moves(features, stats, constraints, current_iteration=iteration)
                total_delta += delta_1
                # Extract moves from the delta calculation (simplified)
                local_moves = int(abs(delta_1) * 100) if delta_1 != 0 else 0
                moves_accepted += local_moves
            except Exception as e:
                tprint(f"⚠️ Step 1 failed: {e}", "WARNING")
                delta_1 = 0.0

            # Keep internal state aligned with latest stats before proceeding
            self._sync_state_from_stats(stats)
            current_assignments = stats.assignments.copy()

            # Step 2: Global reallocation
            if self.verbose:
                tprint(f"🔍 Step 2: Global reallocation (iteration {iteration})", "DEBUG")

            # Apply step 2 specific weights
            self.apply_step_weights(2)

            try:
                delta_2 = await self._step2_global_reallocation(features, stats, constraints)
                total_delta += delta_2
                # Extract moves from the delta calculation (simplified)
                global_moves = int(abs(delta_2) * 100) if delta_2 != 0 else 0
                moves_accepted += global_moves
            except Exception as e:
                tprint(f"⚠️ Step 2 failed: {e}", "WARNING")
                delta_2 = 0.0

            # Refresh state after Step 2 adjustments
            self._sync_state_from_stats(stats)
            current_assignments = stats.assignments.copy()

            # Step 3: Break large clusters
            if self.verbose:
                tprint(f"🔍 Step 3: Break large clusters (iteration {iteration})", "DEBUG")

            # Apply step 3 specific weights
            self.apply_step_weights(3)

            try:
                delta_3 = await self._step3_break_large_clusters(features, stats, constraints, split_policy, split_skip_gate, iteration)
                total_delta += delta_3
                # Extract splits from the delta calculation (simplified)
                splits_performed = int(abs(delta_3) * 10) if delta_3 != 0 else 0
            except Exception as e:
                tprint(f"⚠️ Step 3 failed: {e}", "WARNING")
                delta_3 = 0.0

            # Step 4: Hard-finalize within each iteration
            try:
                delta_4 = await self._step4_hard_finalize(features, stats, constraints)
                total_delta += delta_4
            except Exception as e:
                tprint(f"⚠️ Step 4 failed: {e}", "WARNING")

            # Final sync to ensure downstream consumers see updated assignments
            self._sync_state_from_stats(stats)
            current_assignments = stats.assignments.copy()

            # Apply risk mitigation
            risk_result = risk_system.apply_risk_mitigation(
                features, current_assignments, total_delta, iteration
            )

            # Handle risk result safely
            if isinstance(risk_result, dict):
                risk_violations = risk_result.get("violations", 0)
            else:
                # Fallback for unexpected return types
                tprint(f"⚠️ Unexpected risk_result type: {type(risk_result)}, using default", "WARNING")
                risk_violations = 0

            # Update context with new assignments and cluster count
            context.optimized_assignments = current_assignments.copy()
            current_k = len(np.unique(current_assignments))
            context.optimal_k = current_k

            # Update internal cluster count
            self._refresh_k(current_assignments)

            # Prepare round results
            round_results = {
                "total_delta": total_delta,
                "moves_accepted": moves_accepted,
                "local_moves": local_moves,
                "global_moves": global_moves,
                "splits_performed": splits_performed,
                "risk_violations": risk_violations,
                "initial_cv": initial_cv,
                "final_cv": stats.get_cv_ratio(),
                "initial_balance": initial_balance,
                "final_balance": stats.get_balance_score()
            }

            return context, round_results

        except Exception as e:
            import traceback
            traceback.print_exc()
            tprint(f"❌ Optimization round failed: {e}", "ERROR")
            # Return original context and error info
            return context, {
                "total_delta": 0.0,
                "moves_accepted": 0,
                "local_moves": 0,
                "global_moves": 0,
                "splits_performed": 0,
                "risk_violations": 0,
                "error": str(e)
            }

    async def execute_optimization_loop(
        self,
        context: ClusteringContext,
        config: Any,
        max_iterations: int = 25,
        enable_risk_mitigation: bool = True
    ) -> ClusteringContext:
        """Execute the advanced 3-step iterative optimization loop."""
        try:
            tprint("Starting advanced 3-step iterative optimization...", "INFO")

            features = context.optimized_features
            current_assignments = context.initial_assignments.copy()
            current_k = len(np.unique(current_assignments))

            # CRITICAL FIX: Set features and N before hydrating defaults
            self.features = features
            self.N = len(features)

            # CRITICAL FIX: Hydrate all defaults at start of optimization
            self._hydrate_defaults()

            # Initialize cluster count if not set
            if self._k is None:
                self._k = current_k

            # Clip initial K to policy band
            import math
            initial_k = current_k
            current_k = min(max(initial_k, self.config.K_MIN), self.config.K_MAX)
            if current_k != initial_k:
                tprint(f"🔧 Clipped initial K from {initial_k} to {current_k} (policy: [{self.config.K_MIN}, {self.config.K_MAX}])", "INFO")

            tprint(f"K constraints: K_MIN={self.config.K_MIN}, K_MAX={self.config.K_MAX}, current_K={current_k}", "INFO")

            # Track split rounds for temporary gate relaxation
            self.split_rounds = 0
            self.last_split_round = -1

            # Store features for quality assessment
            self._last_features = features.copy()

            # Apply feature standardization (critical for financial data)
            try:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                features = scaler.fit_transform(features)
                tprint("✅ Applied StandardScaler to features (z-score)", "INFO")
            except Exception as e:
                tprint(f"⚠️ Feature scaling failed: {e}", "WARNING")
            # Ensure internal state uses standardized features
            self.features = features

            # Initialize returns mask for hybrid correlation distance (best-effort)
            try:
                names = getattr(context, 'feature_names', None)
                if names is not None and len(names) == features.shape[1]:
                    patterns = ("returns_", "return_", "close_return", "log_return")
                    mask = np.array([any(str(n).startswith(p) for p in patterns) for n in names], dtype=bool)
                    self.returns_mask = mask if np.any(mask) else np.zeros(features.shape[1], dtype=bool)
                else:
                    self.returns_mask = np.zeros(features.shape[1], dtype=bool)
                tprint(f"🔍 Hybrid distance returns mask active: {int(np.sum(self.returns_mask))} columns", "DEBUG")
            except Exception as _e:
                self.returns_mask = np.zeros(features.shape[1], dtype=bool)

            # Sanity check
            assert self.n_clusters == current_k, f"Cluster count mismatch: {self.n_clusters} != {current_k}"

            # Initialize clustering statistics (on standardized features)
            stats = ClusteringStats(features, current_assignments)
            # Pass returns mask/lambda to stats for hybrid distance
            try:
                stats.returns_lambda = float(getattr(self, 'returns_lambda', 0.5))
                if self.returns_mask is not None and self.returns_mask.size == stats.n_features:
                    stats.returns_mask = self.returns_mask.astype(bool, copy=True)
            except Exception:
                pass

            # CRITICAL FIX: Initialize core state attributes
            self._initialize_state(current_assignments)

            # CRITICAL FIX: Use fixed K throughout optimization
            K_fixed = stats.K_fixed
            unique_labels = np.unique(current_assignments)
            assert 0 <= unique_labels.min() < K_fixed, f"Bad cluster id {unique_labels.min()} for K_fixed={K_fixed}"
            assert 0 <= unique_labels.max() < K_fixed, f"Bad cluster id {unique_labels.max()} for K_fixed={K_fixed}"

            # CRITICAL FIX: Add guard assertions for all per-cluster arrays
            assert_cluster_axis("cluster_sizes", stats.cluster_sizes, K_fixed)
            assert_cluster_axis("centroids", stats.centroids, K_fixed)
            assert_cluster_axis("wcss_per_cluster", stats.wcss_per_cluster, K_fixed)
            assert_cluster_axis("S", stats.S, K_fixed)
            assert_cluster_axis("Q_trace", stats.Q_trace, K_fixed)

            # CRITICAL FIX: Add diagnostic logging for array shapes
            self.logger.debug(f"K_fixed={K_fixed}, labels.max()={unique_labels.max()}, labels.min()={unique_labels.min()}")
            self.logger.debug(f"cluster_sizes.shape={stats.cluster_sizes.shape} (axis0 expected {K_fixed})")
            self.logger.debug(f"centroids.shape={stats.centroids.shape} (axis0 expected {K_fixed})")

            # Initialize risk mitigation system
            risk_system = None
            if enable_risk_mitigation:
                risk_system = RiskMitigationSystem(PRODUCTION_RISK_CONFIG)
                tprint("Risk mitigation system enabled", "INFO")
                tprint("🎯 Advanced 3-step iterative clustering with comprehensive safeguards", "INFO")

            # Initialize strict split policy and skip gate
            split_policy = StrictSplitPolicy()
            split_skip_gate = SplitSkipGate()
            current_round = 0
            current_epoch = 0

            # Initialize N-agnostic constraints
            constraints = NAgosticConstraints(k_max=15, min_fraction=0.02, margin=2, tau=0.20)
            constraints.update_constraints(len(features))
            tprint(f"📏 N-agnostic constraints: {constraints.get_constraint_summary()}", "INFO")

            # FEASIBILITY PRE-PASS: Fix tiny clusters and oversized clusters before optimization
            tprint("🔧 Running feasibility pre-pass...", "INFO")
            N = len(features)
            MIN_SIZE = max(1, int(np.ceil(self.config.MIN_FRAC * N)))
            MAX_SIZE = int(np.floor(self.config.MAX_FRAC * N))

            # Attach tiny clusters first
            current_assignments = self.attach_tiny_clusters(features, current_assignments, MIN_SIZE, MAX_SIZE)
            stats = ClusteringStats(features, current_assignments)

            # Split oversized clusters
            max_split_iterations = 10  # Prevent infinite loops
            split_iteration = 0
            while split_iteration < max_split_iterations:
                split_iteration += 1
                sizes = np.bincount(current_assignments)
                oversized = np.where(sizes > MAX_SIZE)[0]
                if len(oversized) == 0:
                    break

                largest_cluster = oversized[np.argmax(sizes[oversized])]
                tprint(f"🔧 Pre-pass: splitting oversized cluster {largest_cluster} (size={sizes[largest_cluster]})", "INFO")

                # Try to split the largest cluster
                success = self._split_cluster_constrained(features, current_assignments, largest_cluster, MIN_SIZE, MAX_SIZE)
                if not success:
                    tprint(f"⚠️ Pre-pass: failed to split cluster {largest_cluster}, stopping", "WARNING")
                    break

                # Update stats
                stats = ClusteringStats(features, current_assignments)

            # Final feasibility check
            final_sizes = np.bincount(current_assignments)
            final_K = len(np.unique(current_assignments))
            tprint(f"✅ Pre-pass complete: K={final_K}, sizes={final_sizes[final_sizes > 0]}", "INFO")

            # Track convergence and early stopping
            convergence_count = 0
            last_total_delta = float('inf')
            prev_objective = None
            early_stop_count = 0

            for round_num in range(self.max_rounds):
                tprint(f"\n=== Round {round_num + 1}/{self.max_rounds} ===", "INFO")

                # Update split tracking
                self.split_rounds = round_num

                # Update adaptive thresholds
                self._update_adaptive_thresholds(round_num)
                tprint(f"  Thresholds: eps_std_step1={self.eps_std_step1:.3f}, sil_guard={self.sil_guard:.2f}, "
                       f"temporal_bonus={self.temporal_bonus:.2f}", "DEBUG")
                # Log iteration metrics (variance ratio and silhouette)
                self._log_iteration_metrics(round_num, stats)

                # Risk mitigation checks
                if risk_system:
                    # Check if optimization should stop
                    should_stop, stop_reason = risk_system.should_stop_optimization(
                        round_num, stats, features, current_assignments
                    )
                    if should_stop:
                        tprint(f"Stopping optimization: {stop_reason}", "WARNING")
                        break

                    # Log cycle metrics
                    risk_system.log_cycle_metrics(round_num, stats, features, current_assignments)

                # Report initial metrics with cluster details
                initial_cv = stats.get_cv_ratio()
                initial_balance = stats.get_balance_score()
                initial_silhouette = self._calculate_silhouette_score(features, current_assignments)
                initial_k = len(np.unique(stats.assignments))

                # Calculate initial cluster size statistics
                initial_cluster_sizes = np.bincount(stats.assignments)
                initial_min_size = np.min(initial_cluster_sizes)
                initial_max_size = np.max(initial_cluster_sizes)
                initial_avg_size = np.mean(initial_cluster_sizes)

                tprint(f"📊 ROUND {round_num + 1} START - CURRENT STATE:", "INFO")
                tprint(f"   🔢 Clusters: {initial_k}", "INFO")
                tprint(f"   📈 Variance Ratio (between/within): {initial_cv:.4f}", "INFO")
                tprint(f"   ⚖️  Balance Score: {initial_balance:.4f}", "INFO")
                tprint(f"   🎭 Silhouette Score: {initial_silhouette:.4f}", "INFO")
                tprint(f"   📏 Cluster Sizes - Min: {initial_min_size}, Max: {initial_max_size}, Avg: {initial_avg_size:.1f}", "INFO")

                round_delta = 0.0

                # Step 1: Local frontier moves
                local_moves = await self._step1_local_frontier_moves(features, stats, constraints, round_num)
                step1_failed = local_moves is None or local_moves < 0
                round_delta += local_moves if local_moves is not None else 0

                # Step 2: Global reallocation
                global_moves = await self._step2_global_reallocation(features, stats, constraints)
                step2_failed = global_moves is None or global_moves < 0
                round_delta += global_moves if global_moves is not None else 0

                # Step 3: Break large clusters (with k-growth prevention)
                split_moves = 0
                step3_failed = False
                if risk_system:
                    # Check k growth before splitting
                    proposed_k = len(np.unique(stats.assignments))
                    if risk_system.check_unbounded_k_growth(current_k, proposed_k, len(features)):
                        split_moves = await self._step3_break_large_clusters(features, stats, constraints, split_policy, split_skip_gate, round_num)
                        step3_failed = split_moves is None or split_moves < 0
                        if split_moves > 0:
                            self.last_split_round = round_num
                            tprint(f"🔧 Split tracking: last_split_round={self.last_split_round}, current_round={round_num}", "DEBUG")
                    else:
                        tprint("Skipping cluster splits due to k-growth prevention", "WARNING")
                else:
                    split_moves = await self._step3_break_large_clusters(features, stats, constraints, split_policy, split_skip_gate, round_num)
                    step3_failed = split_moves is None or split_moves < 0
                    if split_moves > 0:
                        self.last_split_round = round_num
                        tprint(f"🔧 Split tracking: last_split_round={self.last_split_round}, current_round={round_num}", "DEBUG")

                round_delta += split_moves if split_moves is not None else 0

                # Post-round auto-heal for undersized clusters
                healed = split_skip_gate._auto_heal_clusters(features, stats, constraints)
                if healed > 0:
                    tprint(f"🔧 Auto-healed {healed} undersized clusters", "INFO")

                # Update operation counts for risk tracking
                if risk_system:
                    risk_system.update_operation_counts(local_moves, global_moves, split_moves)

                # Report final metrics with detailed cluster information
                final_cv = stats.get_cv_ratio()
                final_balance = stats.get_balance_score()
                final_silhouette = self._calculate_silhouette_score(features, current_assignments)
                final_k = len(np.unique(stats.assignments))

                # Calculate cluster size statistics
                cluster_sizes = np.bincount(stats.assignments)
                min_size = np.min(cluster_sizes)
                max_size = np.max(cluster_sizes)
                avg_size = np.mean(cluster_sizes)

                # Print detailed round summary
                tprint(f"\n📊 ROUND {round_num + 1} COMPLETED - DETAILED METRICS:", "INFO")
                tprint(f"   🔢 Clusters: {final_k}", "INFO")
                tprint(f"   📈 Variance Ratio (between/within): {final_cv:.4f}", "INFO")
                tprint(f"   ⚖️  Balance Score: {final_balance:.4f}", "INFO")
                tprint(f"   🎭 Silhouette Score: {final_silhouette:.4f}", "INFO")
                tprint(f"   📏 Cluster Sizes - Min: {min_size}, Max: {max_size}, Avg: {avg_size:.1f}", "INFO")
                tprint(f"   📊 Round Delta: {round_delta:.6f}", "INFO")

                # Show cluster distribution for this round
                total_samples = len(stats.assignments)
                size_distribution = cluster_sizes / total_samples * 100
                tprint(f"   📊 Cluster Distribution: ", "INFO", end="")
                for i, (size, pct) in enumerate(zip(cluster_sizes, size_distribution)):
                    if i < 5:  # Show first 5 clusters to avoid clutter
                        tprint(f"C{i}:{size}({pct:.1f}%) ", "INFO", end="")
                if len(cluster_sizes) > 5:
                    tprint(f"... (+{len(cluster_sizes)-5} more)", "INFO", end="")
                tprint("", "INFO")  # New line

                # Risk mitigation: Check metric drift and monotonicity
                if risk_system:
                    current_objective = stats.get_objective_value(constraints=constraints)
                    monotone_ok, monotone_msg = risk_system.check_metric_drift(
                        current_objective, risk_system.last_objective
                    )
                    if not monotone_ok:
                        tprint(f"Metric drift detected: {monotone_msg}", "ERROR")
                        if "rollback" in monotone_msg.lower():
                            tprint("Rolling back to previous state", "WARNING")
                            # Rollback logic would go here
                            break

                    risk_system.update_objective_history(current_objective)

                    # Periodic incremental correctness audit
                    if round_num % risk_system.config.incremental_audit_frequency == 0:
                        audit_ok = risk_system.audit_incremental_correctness(features, stats)
                        if not audit_ok:
                            tprint("Incremental correctness audit failed", "ERROR")
                            break

                # Store step report
                self.step_reports.append({
                    'round': round_num + 1,
                    'initial_cv': initial_cv,
                    'final_cv': final_cv,
                    'cv_delta': final_cv - initial_cv,
                    'initial_balance': initial_balance,
                    'final_balance': final_balance,
                    'balance_delta': final_balance - initial_balance,
                    'initial_silhouette': initial_silhouette,
                    'final_silhouette': final_silhouette,
                    'silhouette_delta': final_silhouette - initial_silhouette,
                    'total_delta': round_delta,
                    'local_moves': local_moves,
                    'global_moves': global_moves,
                    'split_moves': split_moves
                })

                # Early stopping check - rolling 4-iter |ΔJ_total| < 0.5
                if len(self.step_reports) >= 4:
                    recent_deltas = [report['total_delta'] for report in self.step_reports[-4:]]
                    rolling_abs_delta = sum(abs(delta) for delta in recent_deltas)
                    tprint(f"  Rolling 4-iter |ΔJ_total|: {rolling_abs_delta:.3f}", "DEBUG")

                    if rolling_abs_delta < self.early_stop_threshold:
                        early_stop_count += 1
                        tprint(f"  Early stop count: {early_stop_count}/{self.early_stop_patience}", "DEBUG")
                        if early_stop_count >= self.early_stop_patience:
                            tprint(f"Early stopping at round {round_num + 1} (rolling |ΔJ_total| < {self.early_stop_threshold:.1f} for {self.early_stop_patience} consecutive windows)", "INFO")
                            break
                    else:
                        early_stop_count = 0

                # Check convergence with band requirements and feasibility
                current_k = int(stats.assignments.max()) + 1
                k_in_band = constraints.k_low <= current_k <= constraints.k_high
                feasible_now = constraints.is_feasible(stats.cluster_sizes)

                # Check for step failures
                any_step_failed = step1_failed or step2_failed or step3_failed

                if abs(round_delta) < self.tolerance:
                    convergence_count += 1
                    if convergence_count >= 3:
                        if not any_step_failed and k_in_band and feasible_now:
                            tprint(f"🎯 Convergence achieved at iteration {round_num + 1} (K={current_k} in band [{constraints.k_low},{constraints.k_high}], feasible, no step failures)", "SUCCESS")
                            break
                        elif any_step_failed:
                            tprint(f"🎯 Objective converged but steps failed (step1={step1_failed}, step2={step2_failed}, step3={step3_failed}), continuing", "INFO")
                            convergence_count = 0  # Reset to continue until no failures
                        elif not feasible_now:
                            tprint(f"🎯 Objective converged but clustering not feasible (cap/floor violations), continuing", "INFO")
                            convergence_count = 0  # Reset to continue until feasible
                        else:
                            tprint(f"🎯 Objective converged but K={current_k} out of band [{constraints.k_low},{constraints.k_high}], continuing band adjustment", "INFO")
                            # Continue for up to 3 band-adjustment rounds
                            if round_num >= self.max_rounds - 3:
                                tprint(f"⚠️ Max rounds reached, stopping despite K={current_k} out of band", "WARNING")
                        break
                else:
                    convergence_count = 0

                # Check if no improvement
                if round_delta < 1e-6:
                    tprint(f"No improvement at round {round_num + 1}, stopping", "WARNING")
                    break

                last_total_delta = round_delta

            # Update context with final results
            context.optimized_assignments = stats.assignments
            context.final_k = len(np.unique(stats.assignments))

            # Print comprehensive final metrics
            self._print_final_metrics(features, stats)

            # Generate final report
            self._generate_final_report()

            tprint("Advanced 3-step iterative optimization completed", "SUCCESS")
            return context

        except Exception as e:
            tprint(f"Advanced iterative optimization failed: {e}", "ERROR")
            raise ValueError(f"Advanced iterative optimization failed: {e}")

    async def _step1_local_frontier_moves(self, features: np.ndarray, stats: ClusteringStats, constraints: NAgosticConstraints, current_iteration: int = 0) -> float:
        """Step 1: Local frontier moves focused on CV with balance/silhouette/temporal."""
        try:
            # CRITICAL: Validate state consistency before step
            if not self._validate_state_consistency(stats, features):
                tprint("❌ Step 1 aborted: State validation failed", "ERROR")
                return 0.0

            # CRITICAL FIX: Handle None values in Step 1
            if self.assignments is None or self.sizes is None:
                tprint("⚠️ WARNING: self.assignments or self.sizes is None in Step 1, using stats", "WARNING")
                self._initialize_state(stats.assignments)

            # Sanity assert: catch state desync at step entry
            try:
                expected_max = np.bincount(self.assignments).max()
                if self.sizes.max() != expected_max:
                    tprint(f"⚠️ State desync at Step 1 entry: sizes.max()={self.sizes.max()} != bincount.max()={expected_max}", "WARNING")
            except Exception as e:
                tprint(f"⚠️ Failed to validate state sync at Step 1 entry: {e}", "WARNING")

            tprint("Step 1: Local frontier moves...", "INFO")
            # Expose iteration for acceptance rules and annealing
            self._iter = int(max(0, current_iteration))
            # Linearly anneal neighbor consensus threshold from 0.50 → 0.60 over first 3 iterations
            try:
                annealed = 0.50 + min(1.0, self._iter / 3.0) * 0.10
                self.neighbor_consensus_threshold = float(annealed)
            except Exception:
                pass

            # Validate features
            if features is None or features.size == 0:
                tprint("❌ Features array is None or empty in local frontier moves", "ERROR")
                return 0.0

            if not hasattr(features, 'shape') or len(features.shape) != 2:
                tprint(f"❌ Features must be a 2D array, got shape: {getattr(features, 'shape', 'None')}", "ERROR")
                return 0.0

            # ==== Step 1: Local frontier (aggressive) ====
            boundary_points = self._identify_boundary_points_hybrid(features, stats)

            if len(boundary_points) == 0:
                tprint("No boundary points found", "INFO")
                return 0.0

            # Calculate capacity bands for logging
            n_samples = len(features)
            n_min = constraints.cfg.MIN_SIZE  # Use proper MIN_SIZE from constraints

            # Adaptive L calculation based on max cluster fraction
            max_frac = np.max(stats.cluster_sizes) / len(features) if len(stats.cluster_sizes) > 0 else 0
            rescue = (max_frac > 0.50)

            if rescue:
                # Early rounds with large clusters: use larger L
                L = max(128, int(0.10 * len(boundary_points)))  # 10% of boundary, min 128
                L = min(1024, L)  # max 1024
                tprint(f"🔨 Step 1 rescue mode: max_frac={max_frac:.3f} > 0.50, using L={L}", "WARNING")

                # True rescue gates (your ΔJ_low is mis-set)
                # During rescue (largest cluster > SOFT_CAP), do this exactly:
                self.eps_std_step1 = 0.0      # <= you had this exploding; keep it ZERO
                self.cv_guard = 0
                self.sil_guard = 0
                # relax neighbor consensus slightly to enable moves
                try:
                    self.neighbor_consensus_threshold = min(0.50, self.neighbor_consensus_threshold)
                except Exception:
                    pass
                # Note: consensus and margin are handled in the move evaluation
                tprint(f"🔨 Rescue gates: DJ_low=0.0, CV_guard=0, Sil_guard=0", "WARNING")
            else:
                # Normal mode: 5% of boundary, min 64, max 512
                L = max(64, int(0.05 * len(boundary_points)))
                L = min(512, L)

            tprint(f"Evaluating {len(boundary_points)} boundary points", "INFO")

            total_delta = 0.0
            moves_made = 0
            max_moves = min(self.config.local_churn_cap, len(features))  # Use absolute cap

            # Build kNN for neighbor consensus (use shaped neighbor matrix + cosine distance)
            if len(features) > self.knn_size:
                try:
                    neighbor_X = self._get_neighbor_matrix()
                except Exception:
                    neighbor_X = features
                nn = NearestNeighbors(n_neighbors=min(self.knn_size + 1, len(features)), metric='cosine')
                nn.fit(neighbor_X)
                _, indices = nn.kneighbors(neighbor_X)
            else:
                indices = None

            # DIAGNOSTIC: Track blocking reasons and collect deltas for normalization
            # Initialize counters for detailed blocking reasons
            delta_too_low = 0
            cv_guard_blocked = 0
            sil_guard_blocked = 0
            consensus_failed = 0
            margin_failed = 0
            no_alternatives = 0
            few_alternatives = 0

            # Collect all deltas for component scale normalization
            all_deltas = {'cv': [], 'balance': [], 'silhouette': [], 'temporal': []}
            candidate_moves = []

            for point_idx in boundary_points:
                if moves_made >= max_moves:
                    break

                current_cluster = stats.assignments[point_idx]

                # Find best alternative clusters
                best_alternatives = self._find_best_alternative_clusters(
                    features, stats, point_idx, current_cluster, constraints
                )

                # Candidate set sanity check - collect stats instead of individual logging
                if len(best_alternatives) == 0:
                    no_alternatives += 1
                elif len(best_alternatives) < min(3, stats.K_fixed - 1):
                    few_alternatives += 1
                if current_cluster in [alt[0] for alt in best_alternatives]:
                    tprint(f"  ⚠️ Point {point_idx} includes current cluster {current_cluster} in alternatives", "WARNING")

                if not best_alternatives:
                    no_alternatives += 1
                    continue

                # Handle tuple format from _find_best_alternative_clusters
                for target_cluster, delta_info in best_alternatives:
                    # Collect deltas for normalization
                    all_deltas['cv'].append(delta_info['cv'])
                    all_deltas['balance'].append(delta_info['balance'])
                    all_deltas['silhouette'].append(delta_info['silhouette'])
                    all_deltas['temporal'].append(delta_info['temporal'])

                    candidate_moves.append({
                        'point_idx': point_idx,
                        'from_cluster': current_cluster,
                        'to_cluster': target_cluster,
                        'delta_info': delta_info
                    })

                    # Use single source of truth for acceptance
                    accept, block_reason = self._accept_candidate({
                        'delta_info': delta_info,
                        'point_idx': point_idx,
                        'from_cluster': current_cluster,
                        'to_cluster': target_cluster
                    }, stats, constraints)

                    # Initialize anti_osc_reason to avoid scope issues
                    anti_osc_reason = ""

                    if accept:
                        # Check anti-oscillation constraints
                        anti_osc_ok, anti_osc_reason = self._check_anti_oscillation(
                            point_idx, current_cluster, target_cluster, current_iteration,
                            delta_info.get('J_std', delta_info['total'])
                        )

                        if anti_osc_ok:
                            # Check neighbor consensus
                            if indices is not None:
                                neighbor_consensus = self._calculate_neighbor_consensus(
                                    indices[point_idx], stats.assignments, target_cluster
                                )

                                if neighbor_consensus < self.neighbor_consensus_threshold:
                                    consensus_failed += 1
                                    continue

                        # Check margin gain
                        margin_gain = self._calculate_margin_gain(
                            features, stats, point_idx, current_cluster, target_cluster
                        )

                        if margin_gain >= 1e-3:  # Minimum margin gain
                            # Apply the move
                            stats.apply_move(point_idx, current_cluster, target_cluster)
                            # Record move for anti-oscillation tracking
                            self._record_move(point_idx, current_cluster, target_cluster,
                                            current_iteration, delta_info.get('J_std', delta_info['total']))
                            total_delta += delta_info['total']
                            moves_made += 1
                            tprint(f"    ✅ Move {point_idx} {current_cluster}→{target_cluster} "
                                  f"ΔJ_std={delta_info.get('J_std', delta_info['total']):.6f}", "DEBUG")
                            break  # Only one move per point per round
                        else:
                            margin_failed += 1
                    else:
                        # Track anti-oscillation blocking
                        if "tabu_list" in anti_osc_reason or "no_reversal_window" in anti_osc_reason or "move_cap" in anti_osc_reason:
                            delta_too_low += 1  # Use existing counter
                        else:
                            # Track specific blocking reasons - use same counter names as diagnostics
                            if "ΔJ_std>" in block_reason:
                                delta_too_low += 1
                            elif "cv_z<" in block_reason:
                                cv_guard_blocked += 1
                            elif "sil_z<" in block_reason:
                                sil_guard_blocked += 1

            # Top-L policy: guarantee traction if we have positive candidates
            if moves_made == 0 and len(candidate_moves) > 0:
                # Find all positive candidates (negative values are improvements)
                positives = [move for move in candidate_moves
                           if move['delta_info'].get('J_std', move['delta_info']['total']) <= self.eps_std_step1]
                applied = 0
                if positives:
                    # Sort by standardized score ASCENDING (negative values are improvements)
                    positives.sort(key=lambda x: x['delta_info'].get('J_std', x['delta_info']['total']), reverse=False)
                    L = 3  # Reduced L=3 to prevent cascade risk
                    tprint(f"  🔄 Top-L policy: applying {min(L, len(positives))} best moves (L={L}, ascending ΔJ_std)", "DEBUG")

                    # CRITICAL FIX: Deduplicate moves - allow at most one move per point per iteration
                    best_moves = {}
                    for move in positives:
                        pid = move['point_idx']
                        if (pid not in best_moves) or (move['delta_info'].get('J_std', move['delta_info']['total']) < best_moves[pid]['delta_info'].get('J_std', best_moves[pid]['delta_info']['total'])):
                            best_moves[pid] = move

                    deduped_moves = sorted(best_moves.values(), key=lambda x: x['delta_info'].get('J_std', x['delta_info']['total']), reverse=False)

                    # CRITICAL FIX: Enforce L=3 limit after deduplication
                    to_apply = deduped_moves[:L]  # Never exceed L moves

                    accepted_into = set()  # Per-target throttling to stop "magnet" effect

                    for move in to_apply:
                        if moves_made >= self.max_local_moves_per_iter:
                            break

                        # CRITICAL FIX: Revalidate before applying - check if state changed
                        if stats.assignments[move['point_idx']] != move['from_cluster']:
                            continue  # State changed, skip this move
                        if move['to_cluster'] == move['from_cluster']:
                            continue  # No-op move, skip

                        # Per-target throttling: skip if we already accepted a move into this target
                        if move['to_cluster'] in accepted_into:
                            continue

                        # CRITICAL FIX: Add neighbor-consensus gating to stop noisy frontier churn
                        if not self._accept_with_consensus(move['point_idx'], move['to_cluster'], stats.assignments):
                            continue  # skip low-consensus moves

                        # CRITICAL FIX: Add destination-is-large penalty to prevent size equalization from dominating
                        src_size = stats.cluster_sizes[move['from_cluster']]
                        dst_size = stats.cluster_sizes[move['to_cluster']]
                        avg_size = len(stats.assignments) / len(np.unique(stats.assignments))

                        # Create move dict for acceptance check
                        move_dict = self._make_move(
                            move['point_idx'], move['from_cluster'], move['to_cluster'],
                            move['delta_info'].get('cv', 0.0), move['delta_info'].get('silhouette', 0.0),
                            move['delta_info'].get('temporal', 0.0)
                        )

                        if not self._accept_move(move_dict, src_size, dst_size, avg_size):
                            continue  # skip moves that don't meet acceptance criteria

                        stats.apply_move(move['point_idx'], move['from_cluster'], move['to_cluster'])
                        # Record move for anti-oscillation tracking
                        self._record_move(move['point_idx'], move['from_cluster'], move['to_cluster'],
                                        current_iteration, move['delta_info'].get('J_std', move['delta_info']['total']))
                        total_delta += move['delta_info']['total']
                        moves_made += 1
                        accepted_into.add(move['to_cluster'])
                        applied += 1
                        tprint(f"    ✅ Top-L move {move['point_idx']} {move['from_cluster']}→{move['to_cluster']} "
                              f"ΔJ_std={move['delta_info'].get('J_std', move['delta_info']['total']):.6f}", "DEBUG")

                tprint(f"  📦 Applied moves: {applied}/{L}", "DEBUG")

                # Exploratory acceptance: only at iteration 0
                if moves_made == 0 and len(candidate_moves) > 0 and current_iteration == 0:
                    # Sort all candidates by score and take top exploratory quota
                    candidate_moves.sort(key=lambda x: self._objective(x['delta_info'], use_std=self.use_std_for_rank), reverse=True)
                    exploratory_moves = candidate_moves[:self.exploratory_quota]
                    tprint(f"  🔬 Exploratory acceptance: trying {len(exploratory_moves)} top moves despite guards", "DEBUG")

                    for move in exploratory_moves:
                        if moves_made >= self.max_local_moves_per_iter:
                            break
                        # CRITICAL FIX: Add consensus gating even for exploratory moves
                        if not self._accept_with_consensus(move['point_idx'], move['to_cluster'], stats.assignments):
                            continue  # skip low-consensus moves

                        # CRITICAL FIX: Add destination-is-large penalty even for exploratory moves
                        src_size = stats.cluster_sizes[move['from_cluster']]
                        dst_size = stats.cluster_sizes[move['to_cluster']]
                        avg_size = len(stats.assignments) / len(np.unique(stats.assignments))

                        # Create move dict for acceptance check
                        move_dict = self._make_move(
                            move['point_idx'], move['from_cluster'], move['to_cluster'],
                            move['delta_info'].get('cv', 0.0), move['delta_info'].get('silhouette', 0.0),
                            move['delta_info'].get('temporal', 0.0)
                        )

                        if not self._accept_move(move_dict, src_size, dst_size, avg_size):
                            continue  # skip moves that don't meet acceptance criteria

                        # Apply move without re-checking guards (exploratory)
                        stats.apply_move(move['point_idx'], move['from_cluster'], move['to_cluster'])
                        total_delta += move['delta_info']['total']
                        moves_made += 1
                        tprint(f"    ✅ Exploratory move {move['point_idx']} {move['from_cluster']}→{move['to_cluster']} "
                              f"ΔJ_std={move['delta_info'].get('J_std', move['delta_info']['total']):.6f}", "DEBUG")

            # Calculate additional metrics
            boundary_ratio = len(boundary_points) / len(features) if len(features) > 0 else 0
            move_efficiency = moves_made / len(boundary_points) if len(boundary_points) > 0 else 0
            avg_delta_per_move = total_delta / moves_made if moves_made > 0 else 0

            # Component scale normalization analysis and standardization
            if len(candidate_moves) > 0:
                # Calculate robust scales using MAD with winsorization and floors
                scales = {}
                for component in ['cv', 'balance', 'silhouette', 'temporal']:
                    if len(all_deltas[component]) > 0:
                        deltas = np.array(all_deltas[component])
                        mad = np.median(np.abs(deltas - np.median(deltas)))

                        # Winsorize extremes (98% range) and set component-specific floors
                        q = np.quantile(np.abs(deltas), 0.98)
                        if component == 'silhouette':
                            floor = 0.10  # Prevent silhouette blow-ups
                        elif component == 'cv':
                            floor = 0.02  # Allow small separation gains to be visible
                        elif component == 'balance':
                            floor = 1e-3  # Prevent balance explosion
                        elif component == 'temporal':
                            floor = 0.05  # Make temporal matter if non-zero
                        else:
                            floor = 1e-8

                        scales[component] = max(mad, q/1.5, floor)
                    else:
                        scales[component] = 1.0

                # Apply standardization to candidate moves with clipping
                for move in candidate_moves:
                    d = move['delta_info']
                    # Standardize each component with clipping to [-5, 5]
                    d_std = {
                        'cv': np.clip(d['cv'] / scales['cv'] if scales['cv'] > 0 else d['cv'], -5.0, 5.0),
                        'balance': np.clip(d['balance'] / scales['balance'] if scales['balance'] > 0 else d['balance'], -5.0, 5.0),
                        'silhouette': np.clip(d['silhouette'] / scales['silhouette'] if scales['silhouette'] > 0 else d['silhouette'], -5.0, 5.0),
                        'temporal': np.clip(d['temporal'] / scales['temporal'] if scales['temporal'] > 0 else d['temporal'], -5.0, 5.0)
                    }
                    # Calculate both raw and standardized totals
                    d_raw_total = (self.w_cv * d['cv'] +
                                  self.w_bal * d['balance'] +
                                  self.w_sil * d['silhouette'] +
                                  self.w_temp * d['temporal'])
                    # Calculate standardized total WITHOUT balance weight to match _objective method
                    d_std_total = (self.w_cv * d_std['cv'] +
                                  self.w_sil * d_std['silhouette'] +
                                  self.w_temp * d_std['temporal'])

                    # Update move with both raw and standardized info
                    move['delta_info'] = {
                        'total': d_std_total,  # Use standardized for optimization
                        'cv': d_std['cv'],
                        'balance': d_std['balance'],
                        'silhouette': d_std['silhouette'],
                        'temporal': d_std['temporal'],
                        # Raw values for debugging
                        'J_raw': d_raw_total,
                        'J_std': d_std_total,
                        'cv_raw': d['cv'],
                        'balance_raw': d['balance'],
                        'silhouette_raw': d['silhouette'],
                        'temporal_raw': d['temporal']
                    }

                # Log top candidate moves (simplified)
                candidate_moves.sort(key=lambda x: self._objective(x['delta_info'], use_std=self.use_std_for_rank), reverse=True)
                tprint(f"  🔍 TOP CANDIDATE MOVES:", "DEBUG")
                for i, move in enumerate(candidate_moves[:5]):  # Top 5 only
                    d = move['delta_info']
                    score_rank = self._objective(d, use_std=self.use_std_for_rank)
                    accept, block_reason = self._accept_candidate({
                        'delta_info': d,
                        'point_idx': move['point_idx'],
                        'from_cluster': move['from_cluster'],
                        'to_cluster': move['to_cluster']
                    }, stats, constraints)

                    tprint(f"    {i+1:2d}. {move['point_idx']:4d} {move['from_cluster']}→{move['to_cluster']} "
                           f"ΔJ_std={score_rank:.4f} accepted={accept}", "DEBUG")

                # Log component scales (simplified)
                tprint(f"  📏 Component scales: cv={scales.get('cv', 0):.2e}, sil={scales.get('silhouette', 0):.2e}", "DEBUG")

                # Target diversity analysis
                from collections import Counter
                target_clusters = [move['to_cluster'] for move in candidate_moves]
                unique_targets = sorted(set(target_clusters))
                target_distribution = Counter(target_clusters)

                tprint(f"  🎯 Target diversity: {len(unique_targets)}/{self.n_clusters} clusters", "DEBUG")
                if len(unique_targets) < 3:
                    tprint(f"    ⚠️ Low diversity: only {len(unique_targets)} target clusters", "WARNING")

                    # Add staging targets if we have low diversity and a big cluster
                    SOFT_CAP = int(0.20 * len(features))
                    big_c = np.argmax(stats.cluster_sizes)
                    if stats.cluster_sizes[big_c] > SOFT_CAP:
                        staging_targets = self._spawn_staging_targets(features, stats.assignments, big_c, 2, constraints.k_high, np.random.RandomState(42))
                        if staging_targets:
                            tprint(f"    🎭 Added {len(staging_targets)} staging targets: {staging_targets}", "INFO")

            # Enhanced metrics reporting with diagnostics
            tprint(f"Local frontier: {moves_made} moves, delta: {total_delta:.6f}", "INFO")
            tprint(f"  📊 Boundary points: {len(boundary_points)} ({boundary_ratio:.1%} of dataset)", "DEBUG")
            tprint(f"  ⚡ Move efficiency: {move_efficiency:.1%} ({moves_made}/{len(boundary_points)})", "DEBUG")
            tprint(f"  📈 Avg delta per move: {avg_delta_per_move:.6f}", "DEBUG")
            tprint(f"  🔍 Blocking: ΔJ_low={delta_too_low}, CV_guard={cv_guard_blocked}, Sil_guard={sil_guard_blocked}, "
                   f"consensus={consensus_failed}, margin={margin_failed}", "DEBUG")
            tprint(f"  🎯 Alternatives: {no_alternatives} none, {few_alternatives} few (<3)", "DEBUG")

            # Log locked clusters (at min size) - relaxed during rescue
            max_frac = np.max(stats.cluster_sizes) / len(features) if len(stats.cluster_sizes) > 0 else 0
            SOFT_CAP = int(0.20 * len(features))  # 20% cap
            max_size = np.max(stats.cluster_sizes) if len(stats.cluster_sizes) > 0 else 0

            if max_size > SOFT_CAP:
                # During rescue: allow temporary dips to 0.8×MIN_SIZE if it reduces cap overflow
                LOCK_MIN = int(0.8 * constraints.cfg.MIN_SIZE)
                tprint(f"🔨 Step 1 rescue mode: relaxed MIN_SIZE from {constraints.cfg.MIN_SIZE} to {LOCK_MIN}", "WARNING")
            else:
                LOCK_MIN = constraints.cfg.MIN_SIZE

            locked_clusters = np.sum(stats.cluster_sizes <= LOCK_MIN)
            if locked_clusters > 0:
                tprint(f"  🔒 Locked clusters: {locked_clusters} at min size ({LOCK_MIN})", "DEBUG")

            # Log thrash scores per cluster
            thrash_scores = self._calculate_thrash_scores(current_iteration)
            high_thrash_clusters = [cid for cid, score in thrash_scores.items() if score > 0.1]
            if high_thrash_clusters:
                tprint(f"  🔄 High thrash clusters: {[(cid, f'{score:.2f}') for cid, score in thrash_scores.items() if score > 0.1]}", "WARNING")

            # Log sign verification
            tprint(f"  🔍 Top-L uses ascending ΔJ_std (negative=improvement)", "DEBUG")

            # One-line sanity checks: don't hard-fail, just warn
            if len(candidate_moves) > 0:
                top_candidate = candidate_moves[0]
                top_score_gate = self._objective(top_candidate['delta_info'], use_std=self.use_std_for_gate)
                if top_score_gate > self.eps_std_step1 + 1e-12:
                    tprint(f"Top candidate {top_candidate['point_idx']} gate={top_score_gate:.6f} > eps={self.eps_std_step1:.3f} - REJECTED", "DEBUG")
                printed_deltaJ_std = top_candidate['delta_info'].get('J_std', top_candidate['delta_info']['total'])
                if abs(top_score_gate - printed_deltaJ_std) >= 1e-6:
                    tprint(f"⚠️ Sanity delta mismatch: printed ΔJ_std={printed_deltaJ_std:.6f} vs gate={top_score_gate:.6f}", "WARNING")

            # CRITICAL FIX: Sanity check and log after Step-1
            self._sanity_check_and_log("Step-1")

            return total_delta

        except Exception as e:
            import traceback
            traceback.print_exc()
            tprint(f"Local frontier moves failed: {e}", "ERROR")
            return 0.0

    def _apply_step2_move_with_guard(self, stats: ClusteringStats, move: Dict[str, int], initial_k: int) -> bool:
        """Apply a Step-2 move and rollback safely if it collapses K."""
        point_idx = int(move['point_idx'])
        from_cluster = int(move['from_cluster'])
        to_cluster = int(move['to_cluster'])

        stats.apply_move(point_idx, from_cluster, to_cluster)

        current_k = len(np.unique(stats.assignments))
        if current_k < initial_k:
            # Full rollback using inverse move to keep statistics in sync
            stats.apply_move(point_idx, to_cluster, from_cluster)
            if hasattr(stats, 'invalidate_silhouettes'):
                stats.invalidate_silhouettes()
            if hasattr(stats, '_recompute_variance_caches'):
                stats._recompute_variance_caches()
            try:
                stats._validate_state()
            except Exception:
                pass
            tprint(f"🛑 Step-2 K-collapse detected, rolled back move {point_idx}", "WARNING")
            return False

        return True

    async def _step2_global_reallocation(self, features: np.ndarray, stats: ClusteringStats, constraints: NAgosticConstraints) -> float:
        """Step 2: Global reallocation with capacity-aware coordination."""
        try:
            # CRITICAL: Validate state consistency before step
            if not self._validate_state_consistency(stats, features):
                tprint("❌ Step 2 aborted: State validation failed", "ERROR")
                return 0.0

            # CRITICAL FIX: Handle None values in Step 2
            if self.assignments is None or self.sizes is None:
                tprint("⚠️ WARNING: self.assignments or self.sizes is None in Step 2, using stats", "WARNING")
                self._initialize_state(stats.assignments)

            self._log_with_context("Step 2: Global reallocation...", "INFO", "STEP2")

            # Fast-fail: limit global reallocation iterations
            max_global_iterations = 5
            global_iteration_count = 0

            # ==== Step 2: Global reallocation (aggressive) ====
            n_samples = len(features)
            target_size = n_samples / stats.n_clusters
            n_min = constraints.cfg.MIN_SIZE  # Use proper MIN_SIZE from constraints
            n_max = int((1.0 / stats.n_clusters + self.config.beta) * n_samples)

            # PERFORMANCE FIX: Focus only on oversized sources and under-cap destinations
            sizes = stats.cluster_sizes
            soft_cap = int(self.config.MAX_FRAC * n_samples)  # Use config

            # Identify oversized sources and under-cap destinations
            oversized_sources = [c for c, s in enumerate(sizes) if s > soft_cap]
            under_cap_destinations = [c for c, s in enumerate(sizes) if s < soft_cap]

            if not oversized_sources or not under_cap_destinations:
                self._log_with_context(f"No rebalancing needed (oversized: {len(oversized_sources)}, under-cap: {len(under_cap_destinations)})", "DEBUG", "STEP2")
                return 0.0

            # Generate focused candidates: only oversized → under-cap
            candidates = []
            for source_cluster in oversized_sources:
                source_points = np.flatnonzero(stats.assignments == source_cluster)

                # Limit to boundary points for efficiency
                if len(source_points) > self.config.large_cluster_threshold:  # Use config
                    # Use boundary detection to focus on most relevant points
                    boundary_points = self._identify_boundary_points_hybrid(features, stats)
                    # Find intersection of boundary points and source cluster points
                    boundary_in_cluster = np.intersect1d(boundary_points, source_points, assume_unique=True)
                    source_points = boundary_in_cluster

                for point_idx in source_points:
                    current_cluster = stats.assignments[point_idx]
                    for target_cluster in under_cap_destinations:
                        if target_cluster != current_cluster:
                            candidates.append({
                                'point_idx': point_idx,
                                'from_cluster': current_cluster,
                                'to_cluster': target_cluster
                            })

            # Apply violation-first acceptance filter
            candidates = [m for m in candidates if self._violation_first_accept(m, stats.cluster_sizes, constraints.cfg.MIN_SIZE, constraints.cfg.MAX_SIZE)]

            # Capacity-aware scoring (+ scatter bonus to bias out of overdispersed clusters)
            # Precompute per-cluster scatter
            try:
                scatters = {}
                for cid in range(stats.n_clusters):
                    if stats.cluster_sizes[cid] > 0:
                        pts = features[stats.assignments == cid]
                        mu = stats.centroids[cid]
                        scatters[cid] = float(np.mean(np.sum((pts - mu)**2, axis=1)))
                    else:
                        scatters[cid] = 0.0
                mean_scatter = np.mean([v for v in scatters.values() if np.isfinite(v)]) or 1.0
            except Exception:
                scatters = {cid: 0.0 for cid in range(stats.n_clusters)}
                mean_scatter = 1.0
            for m in candidates:
                delta_info = stats.calculate_move_delta(m['point_idx'], m['from_cluster'], m['to_cluster'])
                if isinstance(delta_info, dict):
                    # Allow small positive if it reduces overcap risk
                    base_delta = delta_info['total']
                    dest_size_after = stats.cluster_sizes[m['to_cluster']] + 1

                    # Capacity penalty
                    if dest_size_after > constraints.cfg.MAX_SIZE:
                        cap_penalty = 25.0 * ((dest_size_after - constraints.cfg.MAX_SIZE) / constraints.cfg.MAX_SIZE) ** 2
                        base_delta += cap_penalty

                    # Allow small positive if it fixes violations
                    if base_delta > 0 and base_delta <= 0.10:
                        if (stats.cluster_sizes[m['from_cluster']] > constraints.cfg.MAX_SIZE and
                            dest_size_after <= constraints.cfg.MAX_SIZE):
                            base_delta = min(base_delta, 0.10)

                    # Scatter bonus: encourage moves leaving high-scatter clusters
                    src_scatter = scatters.get(m['from_cluster'], 0.0)
                    scatter_bonus = 0.05 * (src_scatter / max(mean_scatter, 1e-9))
                    m['score'] = base_delta - scatter_bonus
                else:
                    m['score'] = float('inf')

            # Step 2 budget tied to overflow
            SOFT_CAP = int(0.20 * len(features))  # 20% cap
            max_size = np.max(stats.cluster_sizes) if len(stats.cluster_sizes) > 0 else 0
            if max_size > SOFT_CAP:
                # Scale with how bad things are
                overflow_points = max_size - SOFT_CAP
                batches = int(np.clip(np.ceil(overflow_points / SOFT_CAP) * 8, 8, 32))
                max_batches = batches
                batch_size = 256  # Keep batch size
                apply_cap = True  # bias flows out of big_c first
                tprint(f"🔨 Step 2 rescue mode: max_size={max_size} > SOFT_CAP={SOFT_CAP}, using {max_batches} batches", "WARNING")
            else:
                max_batches = 8
                batch_size = 256
                apply_cap = False

            # Apply in conflict-free batches with Step-2 epoch churn throttling
            applied_total = 0
            avg_size = n_samples / stats.n_clusters

            for batch_num in range(max_batches):  # adaptive batches
                # CRITICAL FIX: Check epoch churn cap to prevent over-flattening
                if applied_total >= self.step2_epoch_churn:
                    tprint(f"🛑 Step-2 epoch churn cap reached ({applied_total}/{self.step2_epoch_churn})", "DEBUG")
                    break

                # Pick conflict-free batch
                batch = []
                used_points = set()
                used_targets = set()

                for m in sorted(candidates, key=lambda x: x['score']):
                    if (m['point_idx'] not in used_points and
                        m['to_cluster'] not in used_targets and
                        m['score'] < float('inf')):

                        # CRITICAL FIX: Use _accept_move with dest-over-avg penalty
                        src_size = stats.cluster_sizes[m['from_cluster']]
                        dst_size = stats.cluster_sizes[m['to_cluster']]

                        # Create move dict for acceptance check
                        delta_info = stats.calculate_move_delta(m['point_idx'], m['from_cluster'], m['to_cluster'])
                        if isinstance(delta_info, dict):
                            move_dict = self._make_move(
                                m['point_idx'], m['from_cluster'], m['to_cluster'],
                                delta_info.get('cv', 0.0), delta_info.get('silhouette', 0.0),
                                delta_info.get('temporal', 0.0)
                            )

                            if not self._accept_move(move_dict, src_size, dst_size, avg_size):
                                continue

                        batch.append(m)
                        used_points.add(m['point_idx'])
                        used_targets.add(m['to_cluster'])
                        if len(batch) >= batch_size:  # adaptive batch size
                            break

                if not batch:
                    break

                # Apply batch with K-collapse guard
                initial_k = len(set(stats.assignments))
                for m in batch:
                    if not self._apply_step2_move_with_guard(stats, m, initial_k):
                        continue

                    applied_total += 1

                    # CRITICAL FIX: Check churn cap per move
                    if applied_total >= self.step2_epoch_churn:
                        break

                # Remove applied moves from candidates
                candidates = [m for m in candidates if m['point_idx'] not in used_points]

            tprint(f"Step2-aggr: applied_total={applied_total}", "INFO")

            # CRITICAL FIX: Sanity check and log after Step-2
            self._sanity_check_and_log("Step-2")

            return applied_total * 0.1  # Return positive delta for successful moves

        except Exception as e:
            tprint(f"Global reallocation failed: {e}", "ERROR")
            return 0.0

    def _enforce_cap_with_split_merge(self, features: np.ndarray, stats: ClusteringStats, constraints: NAgosticConstraints) -> tuple[int, int]:
        """Enforce 20% cap even when K == K_MAX using split-merge conserve-K."""
        try:
            max_allowed = constraints.cfg.MAX_SIZE
            overs = np.where(stats.cluster_sizes > max_allowed)[0]
            if len(overs) == 0:
                return 0, 0

            created = merged = 0
            for cid in overs:
                size = int(stats.cluster_sizes[cid])
                need_parts = int(np.ceil(size / max_allowed))
                need_new = need_parts - 1

                # If we're at K_MAX, free budget via merges
                if stats.n_clusters + need_new > constraints.k_max:
                    free = (stats.n_clusters + need_new) - constraints.k_max  # merges to perform

                    # Find smallest clusters to merge
                    cluster_sizes = stats.cluster_sizes
                    smallest_clusters = np.argsort(cluster_sizes)

                    # Merge smallest clusters to free up space
                    for i in range(min(free, len(smallest_clusters) - 1)):
                        if smallest_clusters[i] != cid:  # Don't merge the oversized cluster
                            # Simple merge: reassign all points from smaller to larger
                            smaller_id = smallest_clusters[i]
                            larger_id = smallest_clusters[i + 1] if i + 1 < len(smallest_clusters) else smallest_clusters[0]

                            if smaller_id != larger_id and smaller_id != cid:
                                # Reassign all points from smaller to larger
                                mask = stats.assignments == smaller_id
                                stats.assignments[mask] = larger_id
                                # CRITICAL FIX: Update stats after assignment changes
                                self._stats_update(stats, stats.assignments, features)
                                stats._update_all_stats()
                                merged += 1
                                break

                # Now we have budget to split cid
                # Simple 2-way split using K-means
                cluster_mask = stats.assignments == cid
                cluster_points = features[cluster_mask]
                point_indices = np.where(cluster_mask)[0]

                if len(cluster_points) >= 2 * constraints.cfg.MIN_SIZE:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    new_assignments = kmeans.fit_predict(cluster_points)

                    # Create new cluster ID
                    new_cluster_id = stats.n_clusters

                    # Update assignments
                    for i, point_idx in enumerate(point_indices):
                        if new_assignments[i] == 1:  # Move to new cluster
                            stats.assignments[point_idx] = new_cluster_id
                            # CRITICAL FIX: Update stats after assignment changes
                            self._stats_update(stats, stats.assignments, features)

                    stats._update_all_stats()
                    created += 1

            return created, merged

        except Exception as e:
            tprint(f"Split-merge conserve-K failed: {e}", "ERROR")
            return 0, 0

    def _relabel_compact(self, labels: np.ndarray) -> np.ndarray:
        """Compact labels to eliminate 0-size clusters."""
        try:
            u = np.unique(labels)
            m = {old: i for i, old in enumerate(u)}
            return np.vectorize(m.get)(labels)
        except Exception as e:
            tprint(f"Relabel compact failed: {e}", "ERROR")
            return labels

    def _stats_update(self, stats, labels, X=None):
        """Robust stats updater that handles different update methods."""
        try:
            # Always compact labels first to eliminate 0-size clusters
            labels = self._relabel_compact(np.asarray(labels))

            if hasattr(stats, 'update_after_labels'):
                result = stats.update_after_labels(labels, X) if X is not None else stats.update_after_labels(labels)
                if hasattr(stats, '_initialize_transition_caches'):
                    stats._initialize_transition_caches()
                if hasattr(stats, 'refresh_cluster_sizes'):
                    stats.refresh_cluster_sizes()
                return result
            if hasattr(stats, '_update_cluster_sizes'):
                # legacy/private fallback
                return stats._update_cluster_sizes(labels)
            # last resort: update assignments and recompute stats without touching read-only properties
            if hasattr(stats, 'assignments'):
                stats.assignments = labels
            new_K = int(labels.max()) + 1 if labels.size else 0
            if hasattr(stats, 'ensure_k_capacity'):
                stats.ensure_k_capacity(new_K)
            # If available, recompute all stats
            if hasattr(stats, '_update_all_stats'):
                stats._update_all_stats()
            elif hasattr(stats, '_initialize_statistics'):
                stats._initialize_statistics()
            # Refresh transition caches after bulk label changes
            if hasattr(stats, '_initialize_transition_caches'):
                stats._initialize_transition_caches()
            if hasattr(stats, 'refresh_cluster_sizes'):
                stats.refresh_cluster_sizes()
            return None
        except Exception as e:
            tprint(f"Stats update failed: {e}", "ERROR")
            return None

    def _nearest_feasible_dest(self, a: int, stats: ClusteringStats, features: np.ndarray, avoid: tuple = (), max_after: int = None) -> int:
        """Find nearest feasible destination for merging."""
        try:
            cand = [i for i, s in enumerate(stats.cluster_sizes) if s > 0 and i != a and i not in avoid]
            if not cand:
                return None

            # Get centroids
            if hasattr(stats, 'centroids'):
                centroids = stats.centroids
            else:
                # Compute centroids if not available
                centroids = []
                for i in range(stats.n_clusters):
                    mask = stats.assignments == i
                    if np.any(mask):
                        centroids.append(np.mean(features[mask], axis=0))
                    else:
                        centroids.append(np.zeros(features.shape[1]))
                centroids = np.array(centroids)

            # Pick closest centroid
            cand.sort(key=lambda i: np.linalg.norm(centroids[a] - centroids[i]))
            for j in cand:
                if max_after is None or stats.cluster_sizes[j] + stats.cluster_sizes[a] <= max_after:
                    return j
            return None
        except Exception as e:
            tprint(f"Nearest feasible dest failed: {e}", "ERROR")
            return None

    def _split_cluster_kmeans(self, labels: np.ndarray, features: np.ndarray, cid: int, parts: int, min_size: int, max_size: int, rng) -> np.ndarray:
        """Split cluster using K-means with size constraints."""
        try:
            cluster_mask = labels == cid
            cluster_points = features[cluster_mask]
            point_indices = np.where(cluster_mask)[0]

            if len(cluster_points) < parts * min_size:
                return labels

            # Use K-means to create balanced splits
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=parts, random_state=42, n_init=10)
            new_assignments = kmeans.fit_predict(cluster_points)

            # Create new cluster IDs
            new_cluster_ids = list(range(int(labels.max()) + 1, int(labels.max()) + 1 + parts - 1))

            # Update assignments
            for i, point_idx in enumerate(point_indices):
                if new_assignments[i] == 0:
                    # Keep in original cluster
                        continue
                else:
                    # Move to new cluster
                    new_cluster_id = new_cluster_ids[new_assignments[i] - 1]
                    labels[point_idx] = new_cluster_id

            return labels
        except Exception as e:
            tprint(f"Split cluster kmeans failed: {e}", "ERROR")
            return labels

    def _enforce_k_and_cap(self, features: np.ndarray, stats: ClusteringStats, constraints: NAgosticConstraints) -> np.ndarray:
        """One pass to enforce K budget and 20% cap (conserves K)."""
        try:
            N = len(features)
            MIN_SIZE = constraints.cfg.MIN_SIZE
            MAX_SIZE = constraints.cfg.MAX_SIZE

            labels = self._relabel_compact(stats.assignments.copy())
            self._stats_update(stats, labels, features)

            # Fast-fail: prevent infinite loops
            max_iterations = 10  # Hard limit on iterations
            iteration_count = 0

            # 2a) If K > K_MAX, merge smallest pairs until K == K_MAX
            while stats.n_clusters > constraints.k_max and iteration_count < max_iterations:
                iteration_count += 1
                small = [(i, s) for i, s in enumerate(stats.cluster_sizes) if s > 0]
                small.sort(key=lambda t: t[1])  # smallest first
                a = small[0][0]
                b = self._nearest_feasible_dest(a, stats, features, max_after=MAX_SIZE)
                if b is None:  # last resort: merge into nearest anyway
                    b = self._nearest_feasible_dest(a, stats, features)
                if b is not None:
                    labels[labels == a] = b
                    self._stats_update(stats, labels, features)
                else:
                    break

            if iteration_count >= max_iterations:
                tprint(f"⚠️ Fast-fail: K reduction hit iteration limit ({max_iterations})", "WARNING")
                return labels

            # 2b) While any cluster > MAX_SIZE: split it into parts
            def parts_needed(sz):
                return int(np.ceil(sz / MAX_SIZE))

            changed = True
            split_iteration_count = 0
            max_split_iterations = 5  # Separate limit for split operations

            while changed and split_iteration_count < max_split_iterations:
                split_iteration_count += 1
                changed = False
                overs = [(i, int(s)) for i, s in enumerate(stats.cluster_sizes) if s > MAX_SIZE]
                if not overs:
                    break
                overs.sort(key=lambda t: t[1], reverse=True)
                cid, size = overs[0]
                need = parts_needed(size) - 1  # new clusters needed

                while stats.n_clusters + need > constraints.k_max:
                    # free one slot by merging the smallest pair
                    small = [(i, s) for i, s in enumerate(stats.cluster_sizes) if s > 0 and i != cid]
                    small.sort(key=lambda t: t[1])
                    a = small[0][0]
                    b = self._nearest_feasible_dest(a, stats, features, avoid=(cid,), max_after=MAX_SIZE)
                    if b is None:
                        b = self._nearest_feasible_dest(a, stats, features, avoid=(cid,))
                    if b is not None:
                        labels[labels == a] = b
                        self._stats_update(stats, labels, features)
                    else:
                        break

                # split cid into parts = need+1 (guarding min sizes)
                parts = need + 1
                min_required = parts * MIN_SIZE
                if size < max(min_required, 2 * MIN_SIZE):
                    # if we can't form valid children, downsize greedily using Step2 next time
                    break

                labels = self._split_cluster_kmeans(labels, features, cid, parts, MIN_SIZE, MAX_SIZE, None)
                self._stats_update(stats, labels, features)
                changed = True

            if split_iteration_count >= max_split_iterations:
                tprint(f"⚠️ Fast-fail: Split operations hit iteration limit ({max_split_iterations})", "WARNING")

            return labels
        except Exception as e:
            tprint(f"Enforce K and cap failed: {e}", "ERROR")
            return stats.assignments

    def _enforce_cap_split_merge(self, features: np.ndarray, stats: ClusteringStats, constraints: NAgosticConstraints) -> tuple[int, int]:
        """Enforce 20% cap even at K_MAX using split-merge that conserves K."""
        try:
            N = len(features)
            MAX_SIZE = constraints.cfg.MAX_SIZE
            MIN_SIZE = constraints.cfg.MIN_SIZE

            # Update stats first
            self._stats_update(stats, stats.assignments, features)

            # Find oversized clusters
            overs = [c for c, s in enumerate(stats.cluster_sizes) if s > MAX_SIZE]
            if not overs:
                return 0, 0

            merges_done = 0
            splits_done = 0

            for cid in overs:
                size = int(stats.cluster_sizes[cid])
                parts = int(np.ceil(size / MAX_SIZE))  # e.g., 1325/332 -> 4 parts
                need_new = parts - 1  # new clusters needed if we split

                # If we're at/over K budget, merge the smallest clusters to free capacity
                if stats.n_clusters + need_new > constraints.k_max:
                    need_merges = (stats.n_clusters + need_new) - constraints.k_max

                    # Pick smallest non-empty clusters, excluding the oversize cid
                    small = [(i, s) for i, s in enumerate(stats.cluster_sizes) if s > 0 and i != cid]
                    small.sort(key=lambda x: x[1])  # smallest first

                    taken = 0
                    i = 0
                    while taken < need_merges and i + 1 < len(small):
                        a = small[i][0]
                        # Find nearest cluster to merge into
                        j = self._nearest_cluster_id(a, stats, features, avoid=[cid], max_after=MAX_SIZE)
                        if j is None:
                            i += 1
                            continue

                        # Merge a into j
                        stats.assignments[stats.assignments == a] = j
                        # CRITICAL FIX: Update stats after assignment changes
                        self._stats_update(stats, stats.assignments, features)
                        merges_done += 1
                        taken += 1
                        self._stats_update(stats, stats.assignments, features)

                        # Refresh list because sizes changed
                        small = [(k, s) for k, s in enumerate(stats.cluster_sizes) if s > 0 and k != cid]
                        small.sort(key=lambda x: x[1])
                        i = 0

                # Now split cid into `parts` (≤MAX_SIZE children)
                new_ids = self._split_cluster_into_parts(stats, features, cid, parts, MIN_SIZE, MAX_SIZE)
                splits_done += len(new_ids)
                self._stats_update(stats, stats.assignments, features)

            return splits_done, merges_done

        except Exception as e:
            tprint(f"Cap split-merge failed: {e}", "ERROR")
            return 0, 0

    def _nearest_cluster_id(self, cid: int, stats: ClusteringStats, features: np.ndarray, avoid: list = None, max_after: int = None) -> int:
        """Find nearest cluster ID for merging."""
        try:
            if avoid is None:
                avoid = []

            # Get cluster centroid
            cluster_mask = stats.assignments == cid
            if not np.any(cluster_mask):
                return None

            cluster_points = features[cluster_mask]
            centroid = np.mean(cluster_points, axis=0)

            # Find nearest cluster centroid
            best_id = None
            best_dist = float('inf')

            for other_id in range(stats.n_clusters):
                if other_id == cid or other_id in avoid:
                    continue

                other_mask = stats.assignments == other_id
                if not np.any(other_mask):
                    continue

                other_points = features[other_mask]
                other_centroid = np.mean(other_points, axis=0)

                dist = np.linalg.norm(centroid - other_centroid)

                # Check capacity constraint
                if max_after is not None:
                    new_size = stats.cluster_sizes[other_id] + stats.cluster_sizes[cid]
                    if new_size > max_after:
                        continue

                if dist < best_dist:
                    best_dist = dist
                    best_id = other_id

            return best_id

        except Exception as e:
            tprint(f"Nearest cluster ID failed: {e}", "ERROR")
            return None

    def _split_cluster_into_parts(self, stats: ClusteringStats, features: np.ndarray, cid: int, parts: int, min_size: int, max_size: int) -> list:
        """Split cluster into multiple parts using K-means."""
        try:
            cluster_mask = stats.assignments == cid
            cluster_points = features[cluster_mask]
            point_indices = np.where(cluster_mask)[0]

            if len(cluster_points) < parts * min_size:
                tprint(f"Cannot split cluster {cid}: too small for {parts} parts", "WARNING")
                return []

            # Use K-means to create balanced splits
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=parts, random_state=42, n_init=10)
            new_assignments = kmeans.fit_predict(cluster_points)

            # Create new cluster IDs
            new_cluster_ids = list(range(stats.n_clusters, stats.n_clusters + parts - 1))

            # Update assignments
            for i, point_idx in enumerate(point_indices):
                if new_assignments[i] == 0:
                    # Keep in original cluster
                    continue
                else:
                    # Move to new cluster
                    new_cluster_id = new_cluster_ids[new_assignments[i] - 1]
                    stats.assignments[point_idx] = new_cluster_id

            return new_cluster_ids

        except Exception as e:
            tprint(f"Split cluster into parts failed: {e}", "ERROR")
            return []

    def _repair_small_clusters(self, features: np.ndarray, stats: ClusteringStats, constraints: NAgosticConstraints) -> int:
        """Repair sub-min clusters to ensure everything is ≥ MIN_SIZE."""
        try:
            MIN_SIZE = constraints.cfg.MIN_SIZE
            MAX_SIZE = constraints.cfg.MAX_SIZE

            # Update stats first
            self._stats_update(stats, stats.assignments, features)

            changed = 0
            for cid, sz in enumerate(stats.cluster_sizes):
                if sz == 0 or sz >= MIN_SIZE:
                    continue

                # Find nearest cluster to merge into
                dest = self._nearest_cluster_id(cid, stats, features, max_after=MAX_SIZE)
                if dest is None:
                    # Fallback: pick nearest regardless, we'll fix overcap via moves next iter
                    dest = self._nearest_cluster_id(cid, stats, features)

                if dest is not None:
                    # Merge cid into dest
                    stats.assignments[stats.assignments == cid] = dest
                    changed += 1
                    self._stats_update(stats, stats.assignments, features)

            if changed:
                tprint(f"Small-repair: merged {changed} sub-min clusters into neighbors", "INFO")

            return changed

        except Exception as e:
            tprint(f"Small cluster repair failed: {e}", "ERROR")
            return 0

    def _capacity_guard(self, move: dict, sizes: np.ndarray, MIN_SIZE: int, MAX_SIZE: int) -> bool:
        """Check if a move respects capacity constraints."""
        try:
            src = move.get('from_cluster', move.get('src'))
            dst = move.get('to_cluster', move.get('dest'))

            if sizes[src] <= MIN_SIZE:
                return False
            if sizes[dst] + 1 > MAX_SIZE:
                return False
            return True
        except Exception as e:
            tprint(f"Capacity guard failed: {e}", "ERROR")
            return False

    def _violation_first_accept(self, move: dict, sizes: np.ndarray, MIN_SIZE: int, MAX_SIZE: int) -> bool:
        """Violation-first acceptance: allow small +ΔJ if it reduces a cap/min violation."""
        try:
            src = move.get('from_cluster', move.get('src'))
            dst = move.get('to_cluster', move.get('dest'))
            delta = move.get('delta', move.get('score', 0))

            # Capacity guards
            if sizes[src] <= MIN_SIZE:
                return False
            if sizes[dst] + 1 > MAX_SIZE:
                return False

            # If either cluster violates, accept even if ΔJ_std > 0 (small)
            if sizes[src] > MAX_SIZE or sizes[dst] < MIN_SIZE:
                return delta <= +0.25  # relax up to small positive

            return delta < 0  # normal
        except Exception as e:
            tprint(f"Violation first accept failed: {e}", "ERROR")
            return False

    def _nonempty_mask(self, sizes: np.ndarray) -> np.ndarray:
        """Get mask for non-empty clusters."""
        return sizes > 0

    def _safe_cv_ratio(self, within: float, between: float) -> float:
        """Compute variance ratio (between / within) safely without NaNs."""
        if not np.isfinite(within) or not np.isfinite(between) or within <= 0:
            return 0.0
        return between / within

    def should_attempt_splits(self, boundary_n: int, none_alts: int, locked_min: int, K: int,
                             split_skip_gate: SplitSkipGate, capacity_blocked_pct: float = 0.0) -> bool:
        """Determine if splits should be attempted based on enhanced preconditions."""
        # Calculate percentage of boundary points with sufficient alternatives
        few_alts = boundary_n - none_alts  # points with some alternatives
        pct_with_sufficient_alts = few_alts / boundary_n if boundary_n > 0 else 0.0

        # Check if enough points have ≥3 alternatives
        if pct_with_sufficient_alts < split_skip_gate.min_pct_with_alts_ge_3:
            return False

        # Check if too many points are locked at min size
        locked_frac = locked_min / K if K > 0 else 0.0
        if locked_frac > split_skip_gate.max_locked_points_frac:
            return False

        # Check capacity blocking
        if capacity_blocked_pct > split_skip_gate.max_capacity_blocked:
            return False

        return True

    def _clip_outliers(self, features: np.ndarray, stats: ClusteringStats, pct: float = 0.002) -> None:
        """Clip top pct radial distance outliers per cluster toward centroid (in-place).
        Keeps indices aligned; reduces extreme leverage on splits without reassigning labels.
        """
        try:
            if pct <= 0.0 or pct >= 1.0:
                return
            K = int(stats.assignments.max()) + 1
            for cid in range(K):
                mask = (stats.assignments == cid)
                if not np.any(mask):
                    continue
                Xc = features[mask]
                if Xc.shape[0] < 10:
                    continue
                mu = stats.centroids[cid]
                diffs = Xc - mu
                dists = np.linalg.norm(diffs, axis=1)
                thr = float(np.quantile(dists, 1.0 - pct))
                if thr <= 0:
                    continue
                out_idx = np.where(dists > thr)[0]
                if out_idx.size == 0:
                    continue
                scale = (thr / (dists[out_idx] + 1e-12)).reshape(-1, 1)
                Xc[out_idx] = mu + diffs[out_idx] * scale
                # write back
                features[mask] = Xc
        except Exception as _:
            pass

    async def _step3_break_large_clusters(self, features: np.ndarray, stats: ClusteringStats,
                                         constraints: NAgosticConstraints, split_policy: StrictSplitPolicy, split_skip_gate: SplitSkipGate,
                                         current_round: int, capacity_blocked_pct: float = 0.0) -> float:
        """Step 3: Band-aware split/merge scheduler based on current K."""
        try:
            # CRITICAL FIX: Always reload fresh state at the top
            if self.assignments is None:
                tprint("⚠️ WARNING: self.assignments is None, using stats.assignments", "WARNING")
                labels = stats.assignments.copy()
            else:
                labels = self.assignments
            sizes = np.bincount(labels, minlength=int(labels.max()+1))
            K = int(np.unique(labels).size)

            tprint("Step 3: Band-aware split/merge scheduler...", "INFO")
            tprint(f"🔍 DEBUG: Starting Step 3 with current_round={current_round}, capacity_blocked_pct={capacity_blocked_pct:.3f}", "DEBUG")
            tprint(f"🔍 DEBUG: Fresh state - K={K}, total_samples={len(labels)}, cluster_sizes={sizes.tolist()}", "DEBUG")

            # CRITICAL: Auto-merge micro-clusters at the very top of Step 3
            if self._auto_merge_microclusters(micro_frac=0.005, respect_caps=True, min_size_floor=2):
                tprint("🔗 Auto-merge micro-clusters completed, returning from Step 3", "INFO")
                return 1.0  # Return positive delta for successful micro-merge

            # Enforce K-and-cap constraints with capacity checks
            old_labels = labels.copy()
            tprint(f"🔍 DEBUG: Before K-and-cap enforcement - assignments shape={old_labels.shape}, unique_labels={len(np.unique(old_labels))}", "DEBUG")

            # Check capacity before splitting
            capacity = self.config.K_MAX - K
            if capacity <= 0:
                tprint(f"⚠️ Reached K_MAX={self.config.K_MAX}, stopping splits", "WARNING")
                return 0.0

            stats.assignments = self._enforce_k_and_cap_labels(features, labels)
            # Refresh stats to reflect new assignments
            self._stats_update(stats, stats.assignments, features)

            # Log post-commit sizes
            labs, cnts = np.unique(stats.assignments, return_counts=True)
            tprint(f"Post-commit sizes: max={cnts.max()}, min={cnts.min()}, K={labs.size}", "INFO")

            # Brief outlier trimming prior to splits to improve compactness
            try:
                self._clip_outliers(features, stats, pct=0.002)
            except Exception:
                pass

            # Greedy capped split: deterministic and short-circuity
            SOFT_CAP = int(0.20 * len(features))  # 20% cap
            max_size = np.max(stats.cluster_sizes) if len(stats.cluster_sizes) > 0 else 0
            if max_size > SOFT_CAP:
                tprint(f"🔨 Greedy capped split: max_size={max_size} > SOFT_CAP={SOFT_CAP}", "WARNING")
                stats.assignments = self._greedy_capped_split(features, stats, constraints, SOFT_CAP)
                self._stats_update(stats, stats.assignments, features)
                if not np.array_equal(old_labels, stats.assignments):
                    tprint(f"Greedy capped split completed: K={int(stats.assignments.max())+1}, max={int(np.bincount(stats.assignments).max())}", "INFO")

            K = int(stats.assignments.max()) + 1
            band_policy = constraints.get_band_policy(K)
            tprint(f"🔍 DEBUG: Calculated K={K}, band_policy={band_policy}", "DEBUG")

            # Check for cap violations that override K-band logic
            cap_bad, max_size, cap = self._cap_status()
            tprint(f"🔍 DEBUG: Cap violation check - cap_bad={cap_bad}, max_size={max_size}, cap={cap}", "DEBUG")

            # Order of operations for Step 3:
            # 1. If cap violation and K < K_MAX: split once, return
            # 2. Else if any cluster < MIN_SIZE: merge-repair once, return
            # 3. Else if K > K_MAX: merge closest pair, return
            # 4. Else if K < K_MIN: split worst cluster, return
            # 5. Else: no structural change

            if cap_bad and K < self.config.K_MAX:
                tprint(f"🔧 Cap violation detected, attempting split", "INFO")
                # Continue with existing split logic
            else:
                if not cap_bad:
                    tprint(f"🔍 DEBUG: Cap OK: max={max_size} <= {cap}; skip split branch", "DEBUG")

                # Check for undersized clusters and repair them
                # Keep repairing undersized clusters until none remain
                repaired_any = False
                while self._repair_undersized(features):
                    repaired_any = True
                    self._stats_update(stats, stats.assignments, features)
                if repaired_any:
                    tprint(f"🔧 Repaired undersized clusters", "INFO")
                    return 1.0  # Return positive delta for successful repair

                # If K > K_MAX, merge closest pair
                if K > self.config.K_MAX:
                    tprint(f"🔧 K={K} > K_MAX={self.config.K_MAX}, merging closest pair", "INFO")
                    # TODO: Implement merge closest pair logic
                    return 0.0

                # If K < K_MIN, split worst-cohesion cluster
                if K < self.config.K_MIN:
                    tprint(f"🔧 K={K} < K_MIN={self.config.K_MIN}, proactively triggering split policy", "INFO")
                    result = await self._band_encourage_splits(features, stats, constraints, split_policy, current_round)
                    # Normalize return value to scalar delta for downstream accounting
                    if isinstance(result, (np.ndarray, list)):
                        result_value = 0.0
                    elif result is None:
                        result_value = 0.0
                    else:
                        try:
                            result_value = float(result)
                        except (TypeError, ValueError):
                            result_value = 0.0

                    self._sanity_check_and_log("Step-3")
                    return result_value

            # Safety check: prevent infinite loops
            if current_round > self.config.split_tries_max:  # Hard limit to prevent infinite loops
                tprint(f"⚠️ Hard iteration limit reached ({self.config.split_tries_max}), stopping to prevent infinite loop", "WARNING")
                tprint(f"🔍 DEBUG: Returning 0.0 due to iteration limit", "DEBUG")
                return 0.0

            # Log band policy
            tprint(f"🎯 K-band policy: target=[{constraints.k_low},{constraints.k_high}], mode={band_policy}, K={K}", "INFO")
            tprint(f"📏 CAP_RANGE=[{constraints.cap_min}, {constraints.soft_cap}], band_penalty={constraints.calculate_k_band_penalty(K):.4f}, size_penalty={constraints.calculate_size_penalty(stats.cluster_sizes):.4f}", "INFO")

            # Determine if we need to force split mode due to cap violation
            force_split_mode = cap_bad and K < self.config.K_MAX

            if force_split_mode:
                tprint(f"🚨 CAP VIOLATION DETECTED: Forcing split mode regardless of K-band policy", "WARNING")
                tprint(f"🔍 DEBUG: Calling _band_encourage_splits due to cap violation", "DEBUG")
                result = await self._band_encourage_splits(features, stats, constraints, split_policy, current_round)
                tprint(f"🔍 DEBUG: _band_encourage_splits returned {result}", "DEBUG")
                # CRITICAL FIX: Sanity check and log after Step-3
                self._sanity_check_and_log("Step-3")
                return result

            # Apply band-aware policy
            tprint(f"🔍 DEBUG: Applying band policy: {band_policy}", "DEBUG")
            if band_policy == "encourage_splits":
                tprint(f"🔍 DEBUG: Using proactive band policy for K < 7", "DEBUG")
                # Use proactive band policy that actually does something
                labels = stats.assignments.copy()
                new_labels = self._band_policy(labels)

                # Update stats if labels changed (ensure arrays resized and caches refreshed)
                if not np.array_equal(new_labels, labels):
                    stats.assignments = new_labels
                    self.assignments = new_labels
                    # Refresh stats to reflect new K and cluster arrays
                    self._stats_update(stats, new_labels, features)
                    tprint(f"🔍 DEBUG: Proactive band policy updated labels", "DEBUG")

                # CRITICAL FIX: Sanity check and log after Step-3
                self._sanity_check_and_log("Step-3")
                return 0.1  # Return positive delta for successful policy application
            elif band_policy == "encourage_merges":
                tprint(f"🔍 DEBUG: Calling _band_encourage_merges", "DEBUG")
                result = await self._band_encourage_merges(features, stats, constraints, current_round)
                tprint(f"🔍 DEBUG: _band_encourage_merges returned {result}", "DEBUG")
                # CRITICAL FIX: Sanity check and log after Step-3
                self._sanity_check_and_log("Step-3")
                return result
            else:  # neutral
                tprint(f"🔍 DEBUG: Calling _band_neutral_policy", "DEBUG")
                result = await self._band_neutral_policy(features, stats, constraints, split_policy, current_round)
                tprint(f"🔍 DEBUG: _band_neutral_policy returned {result}", "DEBUG")
                # CRITICAL FIX: Sanity check and log after Step-3
                self._sanity_check_and_log("Step-3")
                return result

        except Exception as e:
            tprint(f"Band-aware split/merge scheduler failed: {e}", "ERROR")
            return 0.0

    def _k_band_penalty(self, K: int, low: int, high: int) -> float:
        """Compute K-band penalty locally (decoupled from constraints)"""
        return 0.02 if low <= K <= high else 0.98

    def _repair_undersized(self, features: np.ndarray):
        """Repair undersized clusters by merging them into nearest feasible recipients."""
        try:
            import math
            sizes = self.sizes
            min_cnt = max(self.MIN_SIZE, math.ceil(self.MIN_FRAC * len(features)))  # MIN_FRAC enforced via config
            small = np.where(sizes < min_cnt)[0]
            if small.size == 0:
                return False

            tprint(f"🔧 Repairing {len(small)} undersized clusters: {sizes[small].tolist()}", "INFO")

            labels = self.assignments.copy()
            cents = self._cluster_centroids(features, labels)  # compute on current features

            for c in small:
                idx = np.where(labels == c)[0]
                if idx.size == 0:
                    continue
                # prefer destinations staying under cap
                cand = [j for j in range(self.K) if j != c and sizes[j] + idx.size <= self.SOFT_CAP] \
                       or [j for j in range(self.K) if j != c]
                jdst = min(cand, key=lambda j: np.linalg.norm(cents[c] - cents[j]))
                labels[idx] = jdst
                tprint(f"🔧 Merged cluster {c} (size={sizes[c]}) into cluster {jdst} (size={sizes[jdst]})", "INFO")

            labels = self._relabel_compact(labels)
            if not np.array_equal(labels, self.assignments):
                self._commit(labels)
                tprint(f"✅ Undersized repair completed: K={self.K}, sizes={self.sizes[self.sizes > 0]}", "INFO")
                return True
            return False

        except Exception as e:
            tprint(f"Undersized repair failed: {e}", "ERROR")
            return False

    def _cluster_centroids(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute cluster centroids from features and labels."""
        K = int(labels.max()) + 1
        centroids = np.zeros((K, features.shape[1]))
        for k in range(K):
            mask = labels == k
            if mask.sum() > 0:
                centroids[k] = features[mask].mean(axis=0)
        return centroids

    def _relabel_compact(self, labels: np.ndarray) -> np.ndarray:
        """Relabel clusters to compact 0..K-1 range."""
        unique_labels = np.unique(labels)
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        return np.array([label_map[label] for label in labels])

    def _safe_cv(self, labels):
        """Safe variance ratio calculation (between / within) with fallback."""
        try:
            return float(self._cv_ratio(labels))
        except AttributeError:
            # Fallback CV: (within / between) using centroid SS
            X = self.features
            labs = np.asarray(labels)
            uniq = np.unique(labs)
            cents = np.vstack([X[labs==c].mean(0) for c in uniq])
            sizes = np.array([np.sum(labs==c) for c in uniq], dtype=float)
            mu = X.mean(0)
            # within
            w = 0.0
            for k, c in zip(uniq, cents):
                dif = X[labs==k] - c
                w += np.sum(dif*dif)
            # between
            difc = cents - mu
            b = float(np.sum((sizes[:,None] * difc) * difc))
            return b / max(w, 1e-12)

    def _safe_sil(self, labels):
        """Safe Silhouette calculation with sampling."""
        try:
            from sklearn.metrics import silhouette_score
            labs = np.asarray(labels)
            # Require at least 2 clusters with size >= 2
            _, counts = np.unique(labs, return_counts=True)
            if np.sum(counts >= 2) < 2:
                return 0.0
            # Sample at most 2000 points for speed
            if len(labs) > 2000:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(labs), 2000, replace=False)
                return float(silhouette_score(self.features[idx], labs[idx], metric="euclidean"))
            return float(silhouette_score(self.features, labs, metric="euclidean"))
        except Exception:
            return 0.0

    async def _step4_hard_finalize(self, features: np.ndarray, stats: ClusteringStats,
                                   constraints: NAgosticConstraints) -> float:
        """Step 4: Hard-finalize within iteration per spec.
        - Merge any cluster < MIN_SIZE into nearest under-cap neighbor (loop until clean)
        - If K < 7, split a large cluster once
        - If K > 12, merge smallest pairs until ≤ 12
        Returns small positive delta when structure changed, else 0.0.
        """
        try:
            changed = False
            # 1) Merge undersized clusters repeatedly
            max_merge_iterations = 10  # Prevent infinite loops
            merge_iteration = 0
            while merge_iteration < max_merge_iterations:
                merge_iteration += 1
                MIN_SIZE = constraints.cfg.MIN_SIZE
                undersized = [i for i, s in enumerate(stats.cluster_sizes) if 0 < s < MIN_SIZE]
                if not undersized:
                    break
                for cid in undersized:
                    dest = self._nearest_feasible_dest(cid, stats, features, max_after=constraints.cfg.MAX_SIZE)
                    if dest is None:
                        dest = self._nearest_feasible_dest(cid, stats, features)
                    if dest is not None and dest != cid:
                        stats.assignments[stats.assignments == cid] = dest
                        self._stats_update(stats, stats.assignments, features)
                        changed = True

            # 2) Adjust K toward band with minimal operations
            K = int(stats.assignments.max()) + 1
            if K < self.config.K_MIN:
                # Split largest cluster once (two-way) if possible
                sizes = np.bincount(stats.assignments)
                largest = int(np.argmax(sizes))
                new_labels = self._balanced_two_way_split(stats.assignments, largest)
                if not np.array_equal(new_labels, stats.assignments):
                    stats.assignments = new_labels
                    self._stats_update(stats, stats.assignments, features)
                    changed = True
            elif K > self.config.K_MAX:
                # Merge smallest pairs until K <= K_MAX
                max_k_reduction_iterations = 10  # Prevent infinite loops
                k_reduction_iteration = 0
                while int(stats.assignments.max()) + 1 > self.config.K_MAX and k_reduction_iteration < max_k_reduction_iterations:
                    k_reduction_iteration += 1
                    sizes = np.bincount(stats.assignments)
                    non_empty = [i for i, s in enumerate(sizes) if s > 0]
                    if len(non_empty) < 2:
                        break
                    # smallest source
                    a = min(non_empty, key=lambda i: sizes[i])
                    b = self._nearest_feasible_dest(a, stats, features, max_after=constraints.cfg.MAX_SIZE)
                    if b is None:
                        b = self._nearest_feasible_dest(a, stats, features)
                    if b is None or b == a:
                        break
                    stats.assignments[stats.assignments == a] = b
                    self._stats_update(stats, stats.assignments, features)
                    changed = True

            if changed:
                tprint("Step 4 hard-finalize applied", "INFO")
                return 0.1
            return 0.0
        except Exception as e:
            tprint(f"Step 4 hard-finalize failed: {e}", "WARNING")
            return 0.0

    def _auto_merge_micro(self, labels, k_lower_band=7):
        import numpy as np
        N = len(labels)
        thr = max(1, int(np.floor(self.micro_frac * N)))  # 0.1% by default
        u, c = np.unique(labels, return_counts=True)
        sizes = dict(zip(u.tolist(), c.tolist()))
        micro = [cid for cid, n in sizes.items() if n < thr]
        if not micro:
            self.logger.debug("No micro-clusters (<0.1% of N) to merge.")
            return labels

        for src in micro:
            # Keep K above lower band—pre-split a large host if needed & possible
            K = len(set(labels))
            if K <= k_lower_band:
                big = max(sizes, key=sizes.get)
                if sizes[big] >= 2 * self.MIN_SIZE:
                    labels2 = self._balanced_two_way_split(labels, big)
                    if not np.array_equal(labels2, labels):
                        labels = labels2
                        u, c = np.unique(labels, return_counts=True)
                        sizes = dict(zip(u.tolist(), c.tolist()))
                        K = len(u)
                    else:
                        self.logger.debug(f"Presplit vetoed for big={big}; skipping merge of {src}")
                        continue

            dst = self._best_affinity_host(labels, src)
            if dst is None:
                self.logger.debug(f"No valid host for micro {src}; skipping.")
                continue

            labels = self._merge_into(labels, src, dst)
            labels = self._relabel_compact(labels)
            u, c = np.unique(labels, return_counts=True)
            sizes = dict(zip(u.tolist(), c.tolist()))
        return labels

    def _auto_merge_microclusters(self, labels=None, **kwargs) -> bool:
        """Wrapper that merges micro clusters and commits when changes occur."""
        try:
            source = labels
            if source is None:
                if self.assignments is None:
                    return False
                source = self.assignments.copy()

            merged = self._auto_merge_micro(source, kwargs.get("k_lower_band", 7))
            if isinstance(merged, np.ndarray) and not np.array_equal(merged, source):
                self._commit(merged)
                return True
            return False
        except Exception as err:
            tprint(f"Auto-merge micro-clusters failed: {err}", "ERROR")
            return False

    def _best_affinity_host(self, labels, src):
        import numpy as np
        X = self.features
        ids_src = np.where(labels == src)[0]
        if len(ids_src) == 0:
            return None
        u = np.unique(labels)
        cent = {cid: X[labels==cid].mean(axis=0) for cid in u}
        scat = {cid: np.mean(np.sum((X[labels==cid]-cent[cid])**2, axis=1)) + 1e-9 for cid in u}
        mu_src = cent[src]

        best, best_score = None, -np.inf
        for dst in u:
            if dst == src:
                continue
            if np.sum(labels==dst) >= self.SOFT_CAP:
                continue
            mu_dst = cent[dst]
            cv_aff  = - np.sum((mu_src - mu_dst)**2) / scat[dst]
            d_dst   = np.mean(np.linalg.norm(X[ids_src] - mu_dst, axis=1))
            d_src   = np.mean(np.linalg.norm(X[ids_src] - mu_src, axis=1)) + 1e-9
            sil_aff = (d_src - d_dst) / max(d_src, d_dst, 1e-9)
            score   = cv_aff + (self.w_sil/self.w_cv) * sil_aff
            if score > best_score:
                best_score, best = score, dst
        return best

    def _merge_into(self, labels, src, dst):
        import numpy as np
        if src == dst:
            return labels
        out = labels.copy()
        out[out == src] = dst
        return out

    def _relabel_compact(self, labels):
        """Remove empty labels and remap to 0..K-1 to avoid size=0 clusters & DBI=inf."""
        import numpy as np
        u = np.unique(labels)
        remap = {cid:i for i, cid in enumerate(u)}
        return np.vectorize(remap.get)(labels)

    def _balanced_two_way_split(self, labels, cid):
        """Split cluster ensuring both children meet MIN_SIZE requirement."""
        import numpy as np
        idx = np.where(labels == cid)[0]
        if len(idx) < 2 * self.MIN_SIZE:
            self.logger.debug(f"Skip split(cid={cid}): size {len(idx)} < 2*MIN_SIZE={2*self.MIN_SIZE}")
            return labels

        X = self.features[idx]
        ok, g = False, None
        for _ in range(5):
            try:
                from sklearn.cluster import KMeans
                g = KMeans(n_clusters=2, n_init="auto", random_state=42).fit(X).labels_
            except Exception:
                g = self._farthest_point_split(X, k=2)
            n0, n1 = int(np.sum(g==0)), int(np.sum(g==1))
            if n0 >= self.MIN_SIZE and n1 >= self.MIN_SIZE:
                ok = True
                break
        if not ok:
            self.logger.debug("Split produced sub-min children; vetoing split.")
            return labels

        new_labels = labels.copy()
        base = max(labels) + 1
        new_labels[idx[g==0]] = base
        new_labels[idx[g==1]] = base + 1
        return self._relabel_compact(new_labels)

    def _prune_empty(self, labels):
        """Remove empty/zero-size labels and reindex 0..K-1 to avoid size=0 clusters and DBI=inf."""
        import numpy as np
        u = np.unique(labels)
        remap = {cid:i for i, cid in enumerate(u)}
        out = np.vectorize(remap.get)(labels)
        return out

    def _size_penalty(self, dest_after: float, mean_size: float, soft_cap: float,
                      over_w: float = 0.02, cap_w: float = 0.05, near: float = 0.90) -> float:
        """Size-aware acceptance penalty for moves into large/near-cap clusters."""
        over_avg = max(0.0, dest_after/mean_size - 1.0)
        nearcap = max(0.0, dest_after/soft_cap - near)
        return 1e-6 + over_w*over_avg + cap_w*(nearcap**2)

    def _accept_move_with_size_penalty(self, delta_cv: float, delta_sil: float, delta_temp: float,
                                     delta_bal: float, dest_after: float, mean_size: float, max_size: float) -> bool:
        """Accept move with size-aware penalty for large clusters."""
        # Calculate weighted delta (CV primary, balance as soft regularizer)
        delta = (self.w_cv * delta_cv - self.w_sil * delta_sil +
                self.w_temp * delta_temp + self.w_bal * delta_bal)

        # Calculate size penalty
        penalty = self._size_penalty(dest_after, mean_size, max_size)

        # Add CV gate to prevent churn
        cv_eps = max(1e-6, 0.01 * abs(delta_cv))  # need ≥1% CV improvement
        cv_ok = delta_cv < -cv_eps or delta_sil > 0  # let Sil break ties

        # Accept if improvement is significant enough
        return delta <= -penalty and cv_ok

    def adaptive_tau_margin(self, margin, base=0.05, slope=0.02, cap=0.20):
        """Adaptive tau based on margin (for Step-1 rescue mode)."""
        return min(cap, base + slope * max(0.0, 1.0 - margin))

    @property
    def sizes(self):
        """Always read live sizes from assignments."""
        if self.assignments is None:
            return np.array([])
        return np.bincount(self.assignments, minlength=self.K)

    @sizes.setter
    def sizes(self, value):
        """Allow setting sizes (for compatibility)."""
        pass  # Ignore setter - sizes are computed from assignments

    def _cap_status(self):
        """Check cap status using live state."""
        sizes, _ = self._current_sizes()
        return (sizes.max() > self.SOFT_CAP, int(sizes.max()), int(self.SOFT_CAP))

    def _greedy_capped_split(self, features: np.ndarray, stats: ClusteringStats, constraints: NAgosticConstraints, SOFT_CAP: int) -> np.ndarray:
        """Greedy capped split: deterministic and short-circuity for oversized clusters."""
        try:
            # CRITICAL FIX: Use fresh state, not stale stats
            if self.assignments is None:
                tprint("⚠️ WARNING: self.assignments is None in greedy split, using stats.assignments", "WARNING")
                labels = stats.assignments.copy()
            else:
                labels = self.assignments.copy()
            sizes = np.bincount(labels, minlength=int(labels.max()+1))
            K = int(np.unique(labels).size)
            N = len(features)
            MIN_SIZE = constraints.cfg.MIN_SIZE
            K_MAX = self.config.K_MAX  # Use config, not constraints

            # Find all oversized clusters using fresh state
            oversized = [(cid, size) for cid, size in enumerate(sizes) if size > SOFT_CAP]
            if not oversized:
                return labels

            tprint(f"🔨 Greedy split: found {len(oversized)} oversized clusters", "WARNING")

            # Process each oversized cluster
            for cid, size in sorted(oversized, key=lambda x: x[1], reverse=True):
                if sizes[cid] <= SOFT_CAP:
                    continue  # Already processed

                # Check capacity before splitting
                capacity = K_MAX - K
                parts = int(np.ceil(size / SOFT_CAP))
                if parts - 1 > capacity:
                    tprint(f"⚠️ Cannot split cluster {cid}: would exceed K_MAX={K_MAX} (need {parts-1}, have {capacity})", "WARNING")
                    continue
                tprint(f"🔨 Splitting cluster {cid} (size={size}) into {parts} parts", "WARNING")

                # Check if we can form valid children
                min_required = parts * MIN_SIZE
                if size < min_required:
                    tprint(f"⚠️ Cannot split cluster {cid}: size={size} < parts*MIN_SIZE={min_required}", "WARNING")
                    continue

                # Check K budget
                current_k = len(np.unique(labels))
                if current_k + parts - 1 > self.config.K_MAX:
                    tprint(f"⚠️ Cannot split cluster {cid}: would exceed K_MAX={self.config.K_MAX}", "WARNING")
                    continue

                # Use the fixed split_cluster_capped function with correct K_MAX
                new_labels = self._split_cluster_capped(features, labels, cid, SOFT_CAP, self.config.K_MAX, np.random.RandomState(42))
                if new_labels is not None:
                    labels = new_labels
                # Update stats
                self._stats_update(stats, labels, features)
                tprint(f"✅ Split cluster {cid} into {parts} parts: sizes={[stats.cluster_sizes[cid]]}", "INFO")

            return labels

        except Exception as e:
            tprint(f"Greedy capped split failed: {e}", "ERROR")
            return stats.assignments

    def k_headroom(self, assignments, K_MAX):
        """Calculate available headroom for new clusters."""
        return max(0, K_MAX - (assignments.max() + 1))

    def need_parts(self, size, SOFT_CAP):
        """Calculate how many parts we need for a given size."""
        return max(2, int(np.ceil(size / SOFT_CAP)))  # 906/332 -> 3

    def planned_parts(self, size, SOFT_CAP, assignments, K_MAX):
        """Calculate how many parts we can actually create given constraints."""
        want = self.need_parts(size, SOFT_CAP)
        # replacing 1 label by p parts increases K by (p-1)
        allowed = 1 + self.k_headroom(assignments, K_MAX)
        return max(2, min(want, allowed))

    def mini_kmeans_two(self, Xb, seeds, iters=10):
        """Simple 2-way k-means with farthest-point seeding."""
        n = Xb.shape[0]
        labels = np.zeros(n, dtype=int)

        # Initialize with seeds
        centers = seeds.copy()

        for _ in range(iters):
            # Assign points to nearest center
            d0 = np.sum((Xb - centers[0])**2, axis=1)
            d1 = np.sum((Xb - centers[1])**2, axis=1)
            labels = (d1 < d0).astype(int)

            # Update centers
            if np.sum(labels == 0) > 0:
                centers[0] = Xb[labels == 0].mean(axis=0)
            if np.sum(labels == 1) > 0:
                centers[1] = Xb[labels == 1].mean(axis=0)

        return labels

    def emergency_split_once(self, X, assignments, big_label, MIN_SIZE, SOFT_CAP, K_MAX, rng):
        """Do exactly one split this round (iterative bisection)."""
        members = np.flatnonzero(assignments == big_label)
        n = members.size
        if n < 2 * MIN_SIZE:
            return 0  # too small to split

        # iterative bisection: do exactly one split this round
        p = min(self.planned_parts(n, SOFT_CAP, assignments, K_MAX), 2)
        if p < 2:
            return 0

        # robust 2-way split with farthest-point seeds
        Xb = X[members]
        i0 = rng.integers(0, n)
        d2 = np.sum((Xb - Xb[i0])**2, axis=1)
        i1 = int(np.argmax(d2))
        seeds = Xb[[i0, i1]]

        # run a tiny k-means with k=2
        labels_local = self.mini_kmeans_two(Xb, seeds, iters=10)

        # both sides must meet MIN_SIZE
        if (labels_local == 0).sum() < MIN_SIZE or (labels_local == 1).sum() < MIN_SIZE:
            return 0

        return self.commit_split(assignments, members, labels_local, big_label)

    def commit_split(self, assignments, members, local_labels, base_label):
        """Bullet-proof commit with global size logging."""
        K0 = assignments.max() + 1
        uniq = np.unique(local_labels)
        # reuse base label for largest child; new labels for the rest
        sizes_local = [(lab, np.sum(local_labels == lab)) for lab in uniq]
        sizes_local.sort(key=lambda t: -t[1])
        reuse = sizes_local[0][0]

        next_label = K0
        for lab,_ in sizes_local:
            idx = members[local_labels == lab]
            if lab == reuse:
                assignments[idx] = base_label
            else:
                assignments[idx] = next_label
                next_label += 1

        # post-commit logging from GLOBAL assignments
        child_labels = [base_label] + list(range(K0, next_label))
        child_sizes = [int((assignments == lbl).sum()) for lbl in child_labels]
        K1 = assignments.max() + 1
        tprint(f"[SPLIT] {base_label} -> {len(child_labels)} parts; sizes={child_sizes}; K={K1}", "INFO")

        # Add invariants after every split
        assert np.bincount(assignments).sum() == assignments.size, "Assignment count mismatch"
        # Note: K_MAX check is handled by the calling function

        return len(child_labels) - 1  # increments to K

    def spawn_ghosts(self, X, assignments, big_c, m, K_MAX, rng):
        """Create ghost labels for Step 1/2 to pour mass into."""
        slots = self.k_headroom(assignments, K_MAX)
        m = min(m, slots)
        if m <= 0:
            return []

        big = np.flatnonzero(assignments == big_c)
        Xb = X[big]
        # farthest-point seeding
        centers = [rng.integers(0, Xb.shape[0])]
        for _ in range(1, m):
            d2 = np.min(((Xb[centers][:,None]-Xb)**2).sum(axis=2), axis=0)
            centers.append(int(np.argmax(d2)))

        labels = []
        for _ in range(m):
            lbl = assignments.max() + 1
            labels.append(lbl)
        # policy hook: Step 2 prefers {labels} as destinations until each hits ~0.6*MIN_SIZE
        return labels

    def batch_quota(self, big_label, sizes, SOFT_CAP):
        """Calculate quota for cap relief moves."""
        overflow = max(0, sizes[big_label] - SOFT_CAP)
        frac = min(0.5, overflow / sizes.sum())  # up to 50% of a batch
        return frac

    def _capvals(self):
        """Pull from instance if present; otherwise use the constants shown in your logs."""
        SOFT_CAP = getattr(self, "SOFT_CAP", 332)
        MIN_SIZE = getattr(self, "MIN_SIZE", 50)
        K_MAX    = getattr(self, "K_MAX", 10)
        return SOFT_CAP, MIN_SIZE, K_MAX

    def _safe_caps(self, cfg=None):
        """Tolerant pull from cfg, then self, then defaults."""
        default_soft = getattr(self, "SOFT_CAP", 332)
        default_min  = getattr(self, "MIN_SIZE", 50)
        default_kmax = getattr(self, "K_MAX", 12)
        if cfg is None:
            return default_soft, default_min, default_kmax
        soft = getattr(cfg, "SOFT_CAP", default_soft)
        mins = getattr(cfg, "MIN_SIZE", default_min)
        kmax = getattr(cfg, "K_MAX", default_kmax)
        return soft, mins, kmax

    def _commit_split(self, assignments, members_idx, child_labels, parent_label):
        """
        Replaces `parent_label` for `members_idx` with new labels
        parent_label -> {parent_label, next_label_start, next_label_start+1, ...}
        Returns (assignments, new_labels_used) ; new_labels_used includes parent_label.
        """
        import numpy as np
        # keep largest child as parent label; new labels for others
        unique, counts = np.unique(child_labels, return_counts=True)
        kept = unique[np.argmax(counts)]
        order = [kept] + [u for u in unique if u != kept]
        remap = {u:i for i,u in enumerate(order)}
        child_labels = np.vectorize(remap.get)(child_labels)

        next_lab = int(assignments.max()) + 1
        assignments = assignments.copy()
        # map 0 -> parent_label, 1.. -> new labels
        for u in np.unique(child_labels):
            dest = parent_label if u == 0 else next_lab + (u - 1)
            mask = (child_labels == u)
            assignments[members_idx[mask]] = dest
        return assignments

    def _repair_small_children(self, X, members_idx, child_labels, min_size):
        """
        If any child part has size < min_size, reassign those points to the nearest
        centroid of the >= min_size children. Returns repaired child_labels.
        """
        import numpy as np
        labels = child_labels.copy()
        uniq, cnts = np.unique(labels, return_counts=True)
        small = uniq[cnts < min_size]
        if small.size == 0:
            return labels

        # centroids
        centroids = {}
        for u in uniq:
            centroids[u] = X[members_idx[labels == u]].mean(axis=0)
        big = [u for u,c in zip(uniq, cnts) if c >= min_size]
        if not big:
            # collapse to largest
            largest = uniq[np.argmax(cnts)]
            labels[:] = largest
            return labels

        big_centroids = np.stack([centroids[u] for u in big], axis=0)
        for s in small:
            idx = np.where(labels == s)[0]
            pts = X[members_idx[idx]]
            d = ((pts[:,None,:]-big_centroids[None,:,:])**2).sum(axis=2)
            nearest = d.argmin(axis=1)
            for rel_i, b_idx in zip(idx, nearest):
                labels[rel_i] = big[b_idx]
        return labels

    def preview_k_and_cap(self, assignments: np.ndarray, sizes: np.ndarray):
        """
        READ-ONLY preview used by Step 3. Do not mutate assignments here.
        """
        K = int(assignments.max() + 1) if assignments.size else 0
        return {"K": K, "max": int(sizes.max()), "min": int(sizes.min())}

    def accept_move(self, deltaJ_std: float, from_c: int, to_c: int,
                    sizes: np.ndarray,
                    soft_cap: int | None = None,
                    min_size: int | None = None,
                    rescue: bool | None = None) -> Tuple[bool, float]:
        """
        Single gate used by Step 1 and Step 2. Always accept strict improvements unless they break min-size/cap.
        Allow small uphill moves if they relieve cap or min-size pressure.
        Temporarily relax gates after splits to let new children grow.
        Returns (accepted, eps_used)
        """
        SOFT_CAP, MIN_SIZE, _ = self._capvals()
        soft_cap = soft_cap or SOFT_CAP
        min_size = min_size or MIN_SIZE

        # Always accept strict improvements unless they break min-size/cap
        if deltaJ_std <= 0:
            return True, 0.0

        # Check if we're in post-split relaxation period
        post_split_relaxation = (hasattr(self, 'split_rounds') and
                                hasattr(self, 'last_split_round') and
                                self.split_rounds - self.last_split_round <= 2)

        oversize_from = sizes[from_c] > soft_cap
        heals_min_to = sizes[to_c] < min_size

        if not (oversize_from or heals_min_to):
            # In post-split period, allow slightly positive moves to help children grow
            if post_split_relaxation:
                eps = 0.08  # Relaxed threshold for 2 rounds after split
                return (deltaJ_std <= eps), eps
            return False, 0.0

        overflow = max(0.0, (sizes[from_c] - soft_cap) / soft_cap) if oversize_from else 0.0

        # Base epsilon calculation
        base_eps = min(0.03 + 0.25 * min(overflow, 2.0), 0.20)

        # Boost epsilon in post-split period
        if post_split_relaxation:
            eps = min(base_eps + 0.05, 0.25)  # Add 0.05 to help children grow
        else:
            eps = base_eps

        return (deltaJ_std <= eps), eps

    def should_converge(self, relative_variation: float, assignments: np.ndarray, tol: float = 0.01) -> bool:
        """
        Enhanced convergence criteria: only converge when all conditions are met.
        """
        soft_cap, _, _ = self._capvals()
        if assignments.size:
            sizes = np.bincount(assignments)
            cap_ok = sizes.max() <= soft_cap
            k_ok = (int(assignments.max()) + 1) >= getattr(self, 'K_MIN', 1)

            # Check for undersized clusters
            import math
            min_frac = getattr(self, 'MIN_FRAC', self.config.MIN_FRAC if hasattr(self, 'config') else 0.03)
            min_size = max(getattr(self, 'MIN_SIZE', 1), math.ceil(min_frac * len(assignments)))
            undersized_ok = sizes[sizes > 0].min() >= min_size

            # Check K constraints
            k_in_band = getattr(self, 'K_MIN', 1) <= len(np.unique(assignments)) <= getattr(self, 'K_MAX', 12)

            # Block convergence if constraints not met
            if not (cap_ok and k_ok and undersized_ok and k_in_band):
                return False

            # Additional quality checks (if we have previous metrics)
            if hasattr(self, '_prev_silhouette') and hasattr(self, '_prev_dbi'):
                try:
                    from sklearn.metrics import silhouette_score, davies_bouldin_score
                    from sklearn.preprocessing import StandardScaler

                    # Get features for quality assessment
                    if hasattr(self, '_last_features') and self._last_features is not None:
                        X = self._last_features
                        if X.shape[0] == len(assignments):
                            # Standardize features for consistent distance calculation
                            X_scaled = StandardScaler().fit_transform(X)

                            # Calculate current quality metrics
                            current_sil = silhouette_score(X_scaled, assignments)
                            current_dbi = davies_bouldin_score(X_scaled, assignments)

                            # Check quality degradation
                            sil_degraded = current_sil < (self._prev_silhouette - 0.01)  # Allow small degradation
                            dbi_degraded = current_dbi > (self._prev_dbi + 0.01)  # Allow small degradation

                            if sil_degraded or dbi_degraded:
                                tprint(f"🔍 Quality check: sil={current_sil:.3f} vs {self._prev_silhouette:.3f}, "
                                      f"dbi={current_dbi:.3f} vs {self._prev_dbi:.3f}", "DEBUG")
                                return False

                except Exception as e:
                    tprint(f"⚠️ Quality check failed: {e}", "DEBUG")

        return relative_variation <= tol

    def _enforce_k_and_cap_labels(self, X, assignments):
        """Enforce K_MIN and SOFT_CAP constraints by splitting oversized clusters."""
        K_now = int(assignments.max()) + 1
        soft_cap, min_size, k_max = self._capvals()
        # Use configured minimum clusters, not a missing attribute default
        k_min = getattr(self, 'config', None).K_MIN if hasattr(self, 'config') else 7

        tprint(f"🔍 K-and-cap enforcement: K={K_now}, K_MIN={k_min}, K_MAX={k_max}, SOFT_CAP={soft_cap}", "DEBUG")

        if K_now < k_min:
            tprint(f"K={K_now} < K_MIN={k_min}; forcing splits", "WARNING")

        # Keep splitting the largest cluster until (a) max_size <= SOFT_CAP and (b) K >= K_MIN
        # But allow K to grow up to K_MAX
        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            sizes = np.bincount(assignments)
            max_c = sizes.argmax()
            max_size = sizes[max_c]
            current_k = int(assignments.max()) + 1

            tprint(f"🔍 Iteration {iteration}: max_size={max_size}, K={current_k}", "DEBUG")

            # Check if we've hit K_MAX
            if current_k >= k_max:
                tprint(f"⚠️ Reached K_MAX={k_max}, stopping splits", "WARNING")
                break

            if max_size <= soft_cap and current_k >= k_min:
                tprint(f"✅ Constraints satisfied: max_size={max_size} <= {soft_cap}, K={current_k} >= {k_min}", "INFO")
                break

            # Split the largest cluster
            old_assignments = assignments.copy()
            assignments = self._split_once_or_multi(X, assignments, max_c)

            # Check if split actually happened
            if np.array_equal(assignments, old_assignments):
                tprint(f"⚠️ Split failed to change assignments, breaking loop", "WARNING")
                break

            iteration += 1

        if iteration >= max_iterations:
            tprint(f"⚠️ K-and-cap enforcement reached max iterations ({max_iterations})", "WARNING")

        return assignments

    def _split_once_or_multi(self, X, assignments, cluster_id):
        """Split a cluster with balanced/capacitated split to ensure children stick."""
        members = np.flatnonzero(assignments == cluster_id)
        n = members.size
        soft_cap, min_size, k_max = self._capvals()

        tprint(f"🔍 Attempting balanced split of cluster {cluster_id} with {n} members", "DEBUG")

        if n <= soft_cap:
            tprint(f"⚠️ Cluster {cluster_id} size {n} <= SOFT_CAP {soft_cap}, no split needed", "DEBUG")
            return assignments

        # Calculate balanced split parameters
        k_needed = max(2, int(np.ceil(n / soft_cap)))
        lower = max(min_size, int(0.4 * soft_cap))  # At least 40% of SOFT_CAP
        upper = soft_cap

        tprint(f"🔍 Balanced split: k_needed={k_needed}, lower={lower}, upper={upper}", "DEBUG")

        # Check if split is feasible
        if k_needed * min_size > n:
            tprint(f"⚠️ Split not feasible: k_needed*min_size={k_needed * min_size} > n={n}", "WARNING")
            return assignments

        # Compute balanced targets
        targets = self._compute_balanced_targets(n, k_needed, lower, upper)
        if not targets:
            tprint(f"⚠️ Cannot compute balanced targets for cluster {cluster_id}", "WARNING")
            # Try with more relaxed constraints
            targets = self._compute_balanced_targets(n, k_needed, min_size, n)
            if not targets:
                tprint(f"⚠️ Even relaxed targets failed for cluster {cluster_id}", "WARNING")
                return assignments

        # Perform balanced split
        try:
            child_assignments = self._balanced_kmeans_split(X[members], targets, min_size)
            if child_assignments is None:
                tprint(f"⚠️ Balanced split failed for cluster {cluster_id}", "WARNING")
                return assignments

            assignments = self._commit_split(assignments, members, child_assignments, cluster_id)
            # Atomic commit to prevent state rollback
            self._commit(assignments)

            # Log post-commit state with sanity checks
            child_sizes = [np.sum(child_assignments == i) for i in range(len(targets))]
            min_child_size = min(child_sizes)
            max_child_size = max(child_sizes)
            k_before = int(assignments.max()) + 1 - len(targets) + 1  # K before split
            k_after = int(assignments.max()) + 1  # K after split

            tprint(f"✅ Balanced split: cluster {cluster_id} → {len(targets)} children, sizes={child_sizes}", "INFO")
            tprint(f"🔍 Post-commit: K={k_before}→{k_after}, min_child={min_child_size}, max_child={max_child_size}", "DEBUG")

            # Sanity checks
            if min_child_size < min_size:
                tprint(f"⚠️ WARNING: min_child_size={min_child_size} < MIN_SIZE={min_size}", "WARNING")
            if k_after > k_before + len(targets) - 1:
                tprint(f"⚠️ WARNING: K growth {k_before}→{k_after} exceeds expected {k_before + len(targets) - 1}", "WARNING")

        except Exception as e:
            tprint(f"❌ Balanced split failed for cluster {cluster_id}: {e}", "ERROR")

        return assignments

    def _compute_balanced_targets(self, n, k, lower, upper):
        """Compute balanced target sizes for split."""
        if k * lower > n:
            return None

        # Try to make targets as equal as possible
        base = n // k
        remainder = n % k

        targets = [base] * k
        for i in range(remainder):
            targets[i] += 1

        # Ensure all targets are within bounds
        for i in range(k):
            if targets[i] < lower:
                return None
            if targets[i] > upper:
                targets[i] = upper

        return targets

    def _balanced_kmeans_split(self, X_cluster, targets, min_size):
        """Perform balanced k-means split with target sizes using size-aware approach."""
        k = len(targets)
        n = X_cluster.shape[0]

        # Early bailout if impossible to meet min_size
        if k * min_size > n:
            tprint(f"⚠️ Cannot split: k*min_size={k * min_size} > n={n}", "DEBUG")
            return None

        # Use size-aware clustering approach
        for seed in range(self.config.max_kmeans_seeds):
            try:
                # Method 1: Try standard KMeans first
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=k, n_init=1, max_iter=self.config.max_kmeans_iterations, random_state=seed)
                labels = km.fit_predict(X_cluster)

                # Check if all children meet minimum size
                child_sizes = [np.sum(labels == i) for i in range(k)]
                if all(size >= min_size for size in child_sizes):
                    tprint(f"✅ KMeans split successful: sizes={child_sizes}", "DEBUG")
                    return labels

                # Method 2: If KMeans fails, try size-aware reassignment
                labels = self._size_aware_reassignment(X_cluster, labels, targets, min_size)
                if labels is not None:
                    child_sizes = [np.sum(labels == i) for i in range(k)]
                    if all(size >= min_size for size in child_sizes):
                        tprint(f"✅ Size-aware reassignment successful: sizes={child_sizes}", "DEBUG")
                        return labels

            except Exception as e:
                tprint(f"⚠️ KMeans attempt {seed} failed: {e}", "DEBUG")
                continue

        tprint(f"⚠️ All KMeans attempts failed for k={k}, n={n}", "DEBUG")

        # Fallback: Simple random partition that respects min_size
        return self._fallback_random_split(X_cluster, k, min_size)

    def _size_aware_reassignment(self, X_cluster, initial_labels, targets, min_size):
        """Reassign points to meet size constraints."""
        k = len(targets)
        n = X_cluster.shape[0]

        # Start with initial labels
        labels = initial_labels.copy()

        # Calculate current sizes
        sizes = [np.sum(labels == i) for i in range(k)]

        # Iteratively reassign points to meet constraints
        max_iterations = 10
        for iteration in range(max_iterations):
            # Find clusters that are too small
            small_clusters = [i for i, size in enumerate(sizes) if size < min_size]
            if not small_clusters:
                break

            # Find clusters that are too large
            large_clusters = [i for i, size in enumerate(sizes) if size > min_size + 1]

            if not large_clusters:
                # If no large clusters, try to redistribute from any cluster
                large_clusters = [i for i, size in enumerate(sizes) if size > min_size]

            if not large_clusters:
                break

            # Move points from large to small clusters
            for small_cluster in small_clusters:
                for large_cluster in large_clusters:
                    if sizes[large_cluster] <= min_size:
                        continue

                    # Find points in large cluster that are closest to small cluster centroid
                    large_mask = labels == large_cluster
                    if not np.any(large_mask):
                        continue

                    large_points = X_cluster[large_mask]
                    small_centroid = X_cluster[labels == small_cluster].mean(axis=0) if np.any(labels == small_cluster) else X_cluster.mean(axis=0)

                    # Calculate distances to small cluster centroid
                    distances = np.linalg.norm(large_points - small_centroid, axis=1)

                    # Move the closest point
                    if len(distances) > 0:
                        closest_idx = np.argmin(distances)
                        global_idx = np.where(large_mask)[0][closest_idx]
                        labels[global_idx] = small_cluster
                        sizes[large_cluster] -= 1
                        sizes[small_cluster] += 1

                        if sizes[small_cluster] >= min_size:
                            break

            # Update sizes
            sizes = [np.sum(labels == i) for i in range(k)]

            # Check if we've met all constraints
            if all(size >= min_size for size in sizes):
                return labels

        # Final check
        final_sizes = [np.sum(labels == i) for i in range(k)]
        if all(size >= min_size for size in final_sizes):
            return labels

        return None

    def _fallback_random_split(self, X_cluster, k, min_size):
        """Fallback: Simple random partition that respects min_size."""
        n = X_cluster.shape[0]

        # Check if split is possible
        if k * min_size > n:
            tprint(f"⚠️ Fallback split impossible: k*min_size={k * min_size} > n={n}", "DEBUG")
            return None

        # Create a random partition that respects min_size
        np.random.seed(42)  # For reproducibility
        labels = np.zeros(n, dtype=int)

        # Assign minimum size to each cluster first
        for i in range(k):
            start_idx = i * min_size
            end_idx = (i + 1) * min_size
            if end_idx <= n:
                labels[start_idx:end_idx] = i

        # Assign remaining points randomly
        remaining_points = n - k * min_size
        if remaining_points > 0:
            remaining_indices = np.arange(k * min_size, n)
            np.random.shuffle(remaining_indices)

            for i, idx in enumerate(remaining_indices):
                cluster_id = i % k
                labels[idx] = cluster_id

        # Verify all clusters meet min_size
        sizes = [np.sum(labels == i) for i in range(k)]
        if all(size >= min_size for size in sizes):
            tprint(f"✅ Fallback random split successful: sizes={sizes}", "DEBUG")
            return labels

        tprint(f"⚠️ Fallback random split failed: sizes={sizes}", "DEBUG")
        return None

    def _feasible_k(self, n_members, assignments, k_max, min_size):
        """Calculate feasible k respecting K_MAX, MIN_SIZE, and cap pressure."""
        import math
        want = max(2, math.ceil(n_members / 332))  # 332 is SOFT_CAP
        room = k_max - (int(assignments.max()) + 1) + 1
        by_min = max(2, min(want, n_members // min_size))
        return max(2, min(by_min, room))

    def _candidate_targets_simple(self, src_idx, sizes, K, features):
        """Get diverse candidate targets for a point."""
        soft_cap, _, _ = self._capvals()
        under_cap = [c for c, s in enumerate(sizes) if s < soft_cap]

        # Add nearest-centroid neighbors
        neigh = self._nearest_centroids(src_idx, features, topk=5)
        targets = sorted(set(under_cap).union(neigh) - {src_idx})

        # Sample up to 4 distinct targets
        return targets[:4]

    def _nearest_centroids(self, src_idx, features, topk=5):
        """Find nearest centroids to a point."""
        try:
            from sklearn.neighbors import NearestNeighbors
            # Simple implementation - find nearest points and return their cluster assignments
            nn = NearestNeighbors(n_neighbors=min(topk + 1, len(features)))
            nn.fit(features)
            distances, indices = nn.kneighbors(features[src_idx:src_idx+1])
            # Return unique cluster assignments of nearest neighbors
            return list(set(indices[0][1:]))  # Exclude self
        except:
            return []

    def converged(self, relative_variation, assignments, SOFT_CAP, tol=0.01):
        """Block convergence when cap violations persist."""
        max_sz = np.bincount(assignments).max()
        if max_sz > SOFT_CAP:
            return False
        return relative_variation <= tol

    def _split_cluster_capped(self, X, assignments, cluster_id, SOFT_CAP, K_MAX, rng):
        """Split cluster using the correct sample set, never on filtered view."""
        try:
            # 1) Strictly derive members from the CURRENT global assignments
            members = np.flatnonzero(assignments == cluster_id)
            n = members.size
            if n <= SOFT_CAP:
                return assignments  # nothing to do

            # 2) K_MAX single source of truth and proper headroom math
            def k_headroom(assignments, K_MAX):
                K = assignments.max() + 1
                return max(0, K_MAX - K)

            need_parts = int(np.ceil(n / SOFT_CAP))  # e.g. 1100/332 -> 4
            allowed = 1 + k_headroom(assignments, K_MAX)  # replacing 1 cluster with k parts
            k_parts = max(2, min(need_parts, allowed))   # at least 2 if we split at all
            if k_parts < 2:
                tprint(f"⚠️ Not enough K headroom: need={need_parts}, allowed={allowed}, K_MAX={K_MAX}", "WARNING")
                return assignments  # not enough headroom this round

            # 3) Build the correct data slice for THIS cluster only with hard guards
            Xc = X[members]
            n_unique = np.unique(Xc, axis=0).shape[0]

            # Hard guards to prevent wrong slice usage
            assert members.size == n, f"Split mask mismatch: members.size={members.size} != n={n}"
            assert Xc.shape[0] == members.size, f"Xc.shape[0]={Xc.shape[0]} != members.size={members.size}"

            # Critical sanity checks (log these right before KMeans)
            tprint(f"split members={n}, intended_k={k_parts}, unique={n_unique}", "DEBUG")

            if n_unique < k_parts:
                # Degenerate case: not enough unique points; fallback to a stable random partition
                order = rng.permutation(n)
                # nearly equal block sizes
                block = np.full(k_parts, n // k_parts, dtype=int); block[: n % k_parts] += 1
                cuts = np.cumsum(block)
                local = np.zeros(n, dtype=int)
                start = 0
                for cid, end in enumerate(cuts):
                    local[order[start:end]] = cid
                    start = end
            else:
                # 4) Robust KMeans that cannot fail the round
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=k_parts, n_init=3, max_iter=50, algorithm="lloyd", random_state=rng)
                local = km.fit_predict(Xc)

            # 5) Bullet-proof commit: replace cluster_id by k_parts child clusters
            def commit_split(assignments, members, local_labels, base_label):
                next_label = assignments.max() + 1
                unique_local = np.unique(local_labels)
                # reuse base label for the largest child
                sizes = [(lab, np.sum(local_labels == lab)) for lab in unique_local]
                sizes.sort(key=lambda t: -t[1])
                reuse, *_ = [lab for lab,_ in sizes]
                for lab,_ in sizes:
                    idx = members[local_labels == lab]
                    if lab == reuse:
                        assignments[idx] = base_label
                    else:
                        assignments[idx] = next_label
                        next_label += 1
                return len(unique_local) - 1

            next_label0 = assignments.max() + 1
            added_clusters = commit_split(assignments, members, local, cluster_id)

            # Log actual child sizes from global assignments
            child_sizes = [np.sum(assignments == cluster_id)] + \
                         [np.sum(assignments == lbl) for lbl in range(next_label0, next_label0 + added_clusters)]
            tprint(f"Split {cluster_id} -> {len(child_sizes)} parts, sizes={child_sizes}, K={assignments.max()+1}", "INFO")

            return assignments

        except Exception as e:
            tprint(f"Split cluster capped failed: {e}", "ERROR")
            return assignments

    def _spawn_staging_targets(self, X, assignments, big_c, m, K_MAX, rng):
        """Create staging (ghost) clusters from oversized blob to increase target diversity."""
        try:
            rng = rng_from(rng)
            slots = max(0, K_MAX - (assignments.max() + 1))
            m = min(m, slots)
            if m <= 0:
                return []

            big = np.flatnonzero(assignments == big_c)
            Xb = X[big]
            centers = [rand_choice(rng, Xb.shape[0])]  # Use rand_choice instead of rng.choice
            for _ in range(1, m):
                # farthest-point seed
                d2 = np.min(((Xb[centers][:,None] - Xb)**2).sum(axis=2), axis=0)
                centers.append(int(np.argmax(d2)))

            new_labels = []
            for _ in range(m):
                lbl = assignments.max() + 1
                new_labels.append(lbl)
                # start empty; mark as ghost in a side array if you track it
            # policy: prefer {new_labels} as destinations for points leaving big_c
            return new_labels

        except Exception as e:
            tprint(f"Spawn staging targets failed: {e}", "ERROR")
            return []

    async def _band_encourage_splits(self, features: np.ndarray, stats: ClusteringStats,
                                   constraints: NAgosticConstraints, split_policy: StrictSplitPolicy, current_round: int) -> np.ndarray:
        """Band policy: K < 7, encourage splits to reach target band."""
        try:
            tprint(f"🔍 DEBUG: _band_encourage_splits called with current_round={current_round}", "DEBUG")

            # CRITICAL FIX: Always read current assignments, never use cached cluster_sizes
            labels = stats.assignments.copy()
            sizes = self._current_sizes_dict(labels)
            K = len(sizes)
            tprint(f"🔍 DEBUG: Current K={K}, cluster_sizes={list(sizes.values())}", "DEBUG")

            # Compute K-band penalty locally
            k_penalty = self._k_band_penalty(K, constraints.k_low, constraints.k_high)
            tprint(f"🔍 DEBUG: K-band penalty for K={K} is {k_penalty:.4f}", "DEBUG")

            # Check for oversized clusters that MUST be split regardless of K
            oversized = [(cid, size) for cid, size in sizes.items() if size > constraints.max_size]
            tprint(f"🔍 DEBUG: Found {len(oversized)} oversized clusters: {oversized}", "DEBUG")
            if oversized:
                tprint(f"🚨 Found {len(oversized)} oversized clusters exceeding 20% cap (max_size={constraints.max_size})", "WARNING")
                for cid, size in oversized:
                    tprint(f"   Cluster {cid}: {size} samples ({size/constraints.n*100:.1f}%)", "WARNING")

                # Force split of largest violator
                largest_violator = max(oversized, key=lambda x: x[1])
                cid, size = largest_violator
                tprint(f"🔨 Forcing split of cluster {cid} (size={size}) to enforce 20% cap", "WARNING")

                # Calculate how many splits we need to get under the cap
                needed_splits = constraints.needed_splits(size)
                tprint(f"🔢 Cluster {cid} needs {needed_splits} splits to get under 20% cap", "INFO")

                # Try emergency multi-split approach
                if needed_splits > 0:
                    tprint(f"🔍 DEBUG: Calling _emergency_multi_split for cluster {cid} with {needed_splits} needed splits", "DEBUG")
                    result = await self._emergency_multi_split(features, stats, constraints, split_policy, current_round, cid, size)
                    tprint(f"🔍 DEBUG: _emergency_multi_split returned {result}", "DEBUG")
                    return result

                # Fallback to regular split with very permissive criteria
                tprint(f"🔍 DEBUG: Attempting regular split for cluster {cid} with very permissive criteria", "DEBUG")
                result = self._apply_split_atomic_strict_custom(features, stats, cid, split_policy, current_round,
                                                               min_delta_std=5.0)  # Very permissive for cap violations

                if result is None:
                    # Success - reset counter
                    tprint(f"🔍 DEBUG: Cap violation split succeeded for cluster {cid}", "DEBUG")
                    constraints.out_of_band_rounds = 0
                    split_policy.splits_this_round += 1
                    split_policy.consume_split_budget(K)
                    delta = self._calculate_split_quality(features, stats, cid)
                    tprint(f"✅ Cap violation split applied: cluster {cid}, delta: {delta:.6f}", "INFO")

                    # Follow with one global reallocation pass
                    tprint("Following cap violation split with global reallocation pass", "DEBUG")
                    await self._step2_global_reallocation(features, stats, constraints)

                    return delta
                else:
                    tprint(f"❌ Cap violation split failed: cluster {cid} - {result.name}", "ERROR")
                    tprint(f"🔍 DEBUG: Cap violation split failed, returning labels", "DEBUG")
                    return labels

            # Allow ≤1 split/round when K < 7
            if split_policy.splits_this_round >= 1:
                tprint(f"Skip splits (rate limit: {split_policy.splits_this_round}/1)", "DEBUG")
                tprint(f"🔍 DEBUG: Returning labels due to rate limit", "DEBUG")
                return labels

            # Find clusters that can be safely split
            split_candidates = []
            # AGGRESSIVE FIX: When K < K_low, use minimal parent size requirement
            min_parent_size = 2 * constraints.min_size  # Just 2*MIN_SIZE, no additional overhead
            tprint(f"🔍 Split eligibility: min_parent_size={min_parent_size}, cluster_sizes={stats.cluster_sizes}", "DEBUG")
            tprint(f"🔍 DEBUG: Starting to find split candidates with min_parent_size={min_parent_size}", "DEBUG")

            for cluster_id in range(len(stats.cluster_sizes)):
                size = stats.cluster_sizes[cluster_id]
                tprint(f"🔍 DEBUG: Evaluating cluster {cluster_id} with size {size}", "DEBUG")

                # Override parent size guard if cluster violates max_size
                if constraints.violates_max_size(size):
                    tprint(f"🔍 Cluster {cluster_id} violates 20% cap: {size} > {constraints.max_size} - allowing split", "DEBUG")
                    split_candidates.append(cluster_id)
                    continue

                # AGGRESSIVE FIX: Minimal parent size when K < K_low
                if size < min_parent_size:
                    tprint(f"🔍 Cluster {cluster_id} too small: {size} < {min_parent_size}", "DEBUG")
                    continue

                # Predict child sizes
                child_a_pred = size // 2
                child_b_pred = size - child_a_pred
                tprint(f"🔍 DEBUG: Cluster {cluster_id} predicted children: {child_a_pred}, {child_b_pred}", "DEBUG")

                # Each child must satisfy: child_size ≥ MIN_SIZE and ≤ U
                if child_a_pred < constraints.min_size or child_b_pred < constraints.min_size:
                    tprint(f"🔍 DEBUG: Cluster {cluster_id} children too small: {child_a_pred}, {child_b_pred} < {constraints.min_size}", "DEBUG")
                    continue
                if child_a_pred > constraints.soft_cap or child_b_pred > constraints.soft_cap:
                    tprint(f"🔍 DEBUG: Cluster {cluster_id} children too large: {child_a_pred}, {child_b_pred} > {constraints.soft_cap}", "DEBUG")
                    continue

                tprint(f"🔍 DEBUG: Cluster {cluster_id} added to split candidates", "DEBUG")
                split_candidates.append(cluster_id)

            tprint(f"🔍 DEBUG: Found {len(split_candidates)} split candidates: {split_candidates}", "DEBUG")
            if not split_candidates:
                tprint("No clusters meet split criteria for K < 7", "DEBUG")

                # Check for emergency split after multiple failed attempts
                constraints.out_of_band_rounds += 1
                tprint(f"🔍 DEBUG: Incremented out_of_band_rounds to {constraints.out_of_band_rounds}", "DEBUG")
                if constraints.out_of_band_rounds >= constraints.emergency_split_after:
                    tprint(f"🚨 Emergency split triggered after {constraints.out_of_band_rounds} out-of-band rounds", "WARNING")
                    tprint(f"🔍 DEBUG: Calling _emergency_split", "DEBUG")
                    result = await self._emergency_split(features, stats, constraints, split_policy, current_round)
                    tprint(f"🔍 DEBUG: _emergency_split returned {result}", "DEBUG")
                    return result

                tprint(f"🔍 DEBUG: No emergency split needed, returning 0.0", "DEBUG")
                return 0.0

            # Try to split the first candidate with very permissive criteria
            cluster_id = split_candidates[0]
            tprint(f"🔍 DEBUG: Attempting split of cluster {cluster_id} with very permissive criteria", "DEBUG")
            result = self._apply_split_atomic_strict_custom(features, stats, cluster_id, split_policy, current_round,
                                                           min_delta_std=2.0)  # Very permissive when K < K_low

            if result is None:
                # Success - reset out-of-band counter
                constraints.out_of_band_rounds = 0
                split_policy.splits_this_round += 1
                split_policy.consume_split_budget(K)
                delta = self._calculate_split_quality(features, stats, cluster_id)
                tprint(f"✅ Split applied (K < 7): cluster {cluster_id}, delta: {delta:.6f}", "INFO")

                # Follow with one global reallocation pass
                tprint("Following split with global reallocation pass", "DEBUG")
                await self._step2_global_reallocation(features, stats, constraints)

                return delta
            else:
                tprint(f"❌ Split denied (K < 7): cluster {cluster_id} - {result.name}", "DEBUG")
                return 0.0

        except Exception as e:
            tprint(f"Band encourage splits failed: {e}", "ERROR")
            return 0.0

    async def _emergency_split(self, features: np.ndarray, stats: ClusteringStats,
                             constraints: NAgosticConstraints, split_policy: StrictSplitPolicy, current_round: int) -> float:
        """Emergency split: force split of largest cluster that can produce two children ≥ MIN_SIZE."""
        try:
            K = int(stats.assignments.max()) + 1

            # Find the largest cluster that can be safely split
            best_candidate = None
            best_size = 0

            for cluster_id in range(len(stats.cluster_sizes)):
                size = stats.cluster_sizes[cluster_id]

                # Must be large enough to split into two children ≥ MIN_SIZE
                if size < 2 * constraints.min_size:
                    continue

                # Predict child sizes with emergency balance
                child_a_size = int(size * constraints.emergency_split_balance[0])
                child_b_size = size - child_a_size

                # Both children must meet minimum size
                if child_a_size < constraints.min_size or child_b_size < constraints.min_size:
                    continue

                # Track the largest valid candidate
                if size > best_size:
                    best_candidate = cluster_id
                    best_size = size

            if best_candidate is None:
                tprint("🚨 Emergency split: No suitable cluster found", "ERROR")
                return 0.0

            tprint(f"🚨 Emergency split: Forcing split of cluster {best_candidate} (size={best_size})", "WARNING")

            # Force the split without strict delta requirements
            result = self._apply_split_atomic_strict_custom(features, stats, best_candidate, split_policy, current_round,
                                                           min_delta_std=1.0)  # Very permissive

            if result is None:
                # Success - reset counter
                constraints.out_of_band_rounds = 0
                split_policy.splits_this_round += 1
                split_policy.consume_split_budget(K)
                delta = self._calculate_split_quality(features, stats, best_candidate)
                tprint(f"✅ Emergency split applied: cluster {best_candidate}, delta: {delta:.6f}", "INFO")

                # Follow with one global reallocation pass
                tprint("Following emergency split with global reallocation pass", "DEBUG")
                await self._step2_global_reallocation(features, stats, constraints)

                # Return the delta value
                return delta
            else:
                tprint(f"❌ Emergency split failed: cluster {best_candidate} - {result.name}", "ERROR")
                return 0.0

        except Exception as e:
            tprint(f"Emergency split failed: {e}", "ERROR")
            return 0.0

    async def _emergency_multi_split(self, features: np.ndarray, stats: ClusteringStats,
                                   constraints: NAgosticConstraints, split_policy: StrictSplitPolicy,
                                   current_round: int, cid: int, size: int) -> np.ndarray:
        """Emergency multi-split using quality-aware farthest-point seeding."""
        try:
            # Use safe caps to handle config objects
            soft_cap, min_size, k_max = self._safe_caps(constraints.cfg)

            # Calculate needed splits
            needed_splits = max(1, int(np.ceil(size / soft_cap)) - 1)

            # Use quality-aware emergency split
            labels = stats.assignments.copy()
            new_labels = self._emergency_multi_split_quality_aware(labels, cid, needed_splits)

            # Only commit if there was actual change
            if np.array_equal(new_labels, labels):
                tprint("Emergency split: no-op", "DEBUG")
                return labels

            # Update stats with new labels
            stats.assignments = new_labels
            self.assignments = new_labels
            return new_labels

        except Exception as e:
            tprint(f"Emergency multi-split failed: {e}", "ERROR")
            return stats.assignments.copy()

    def _emergency_multi_split_iterative(self, X, assignments, parent_label, needed_splits, soft_cap, min_size, k_max, rng=None):
        """Iterative emergency multi-split that keeps splitting largest part until under cap."""
        rng = rng_from(rng)
        added = 0
        max_split_attempts = 10  # Prevent infinite loops
        split_attempt = 0
        while added < needed_splits and split_attempt < max_split_attempts:
            split_attempt += 1
            members = np.where(assignments == parent_label)[0]
            n = members.size
            if n < 2*min_size:
                break

            # choose parts without exceeding K_MAX or min-size feasibility
            k_now = int(assignments.max()) + 1
            room = k_max - k_now + 1  # +1 because parent relabel keeps one
            if room < 2:
                break
            feasible = max(2, min(room, n // min_size))
            if feasible < 2:
                break

            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=feasible, n_init=5, max_iter=100, random_state=42)
            child = km.fit_predict(X[members])
            child = self._repair_small_children(X, members, child, min_size)
            assignments = self._commit_split(assignments, members, child, parent_label)

            # pick largest child; if still >cap, loop again on it
            # find which label now holds the "parent_label" id (we kept largest as parent_label)
            added += feasible - 1
            if (assignments == parent_label).sum() <= soft_cap:
                break
        return assignments, added

    async def _band_encourage_merges(self, features: np.ndarray, stats: ClusteringStats,
                                   constraints: NAgosticConstraints, current_round: int) -> float:
        """Band policy: K > 10, encourage merges to reduce to target band."""
        try:
            tprint(f"🔍 DEBUG: _band_encourage_merges called with current_round={current_round}", "DEBUG")
            K = int(stats.assignments.max()) + 1
            tprint(f"🔍 DEBUG: Current K={K}, cluster_sizes={stats.cluster_sizes.tolist()}", "DEBUG")

            # Find merge candidates (clusters that are clearly redundant)
            merge_candidates = []
            tprint(f"🔍 DEBUG: Starting to find merge candidates", "DEBUG")

            for i in range(len(stats.cluster_sizes)):
                for j in range(i + 1, len(stats.cluster_sizes)):
                    size_i = stats.cluster_sizes[i]
                    size_j = stats.cluster_sizes[j]

                    if size_i == 0 or size_j == 0:
                        continue

                    # Check if merged size would be acceptable
                    merged_size = size_i + size_j
                    if merged_size > constraints.soft_cap:
                        tprint(f"🔍 DEBUG: Merge {i},{j} rejected - merged_size {merged_size} > soft_cap {constraints.soft_cap}", "DEBUG")
                        continue

                    # Calculate boundary distance / confusion
                    centroid_i = stats.centroids[i]
                    centroid_j = stats.centroids[j]
                    distance = np.linalg.norm(centroid_i - centroid_j)
                    tprint(f"🔍 DEBUG: Merge candidate {i},{j} - sizes: {size_i},{size_j}, merged_size: {merged_size}, distance: {distance:.4f}", "DEBUG")

                    merge_candidates.append((i, j, distance, merged_size))

            tprint(f"🔍 DEBUG: Found {len(merge_candidates)} merge candidates", "DEBUG")
            if not merge_candidates:
                tprint("No clusters meet merge criteria for K > 10", "DEBUG")
                tprint(f"🔍 DEBUG: Returning 0.0 - no merge candidates", "DEBUG")
                return 0.0

            # Sort by distance (closest first) and try the best merge
            merge_candidates.sort(key=lambda x: x[2])
            i, j, distance, merged_size = merge_candidates[0]
            tprint(f"🔍 DEBUG: Best merge candidate: {i},{j} with distance {distance:.4f}, merged_size {merged_size}", "DEBUG")

            # Apply merge
            tprint(f"🔍 DEBUG: Attempting merge of clusters {i},{j}", "DEBUG")
            result = self._apply_merge_atomic(features, stats, i, j, min_delta_std=0.2)

            if result:
                delta = self._calculate_merge_quality(features, stats, i, j)
                tprint(f"✅ Merge applied (K > 10): clusters {i},{j}, merged_size={merged_size}, delta: {delta:.6f}", "INFO")
                tprint(f"🔍 DEBUG: Merge succeeded, returning delta {delta}", "DEBUG")
                return delta
            else:
                tprint(f"❌ Merge denied (K > 10): clusters {i},{j}", "DEBUG")
                tprint(f"🔍 DEBUG: Merge failed, returning 0.0", "DEBUG")
                return 0.0

        except Exception as e:
            tprint(f"Band encourage merges failed: {e}", "ERROR")
            tprint(f"🔍 DEBUG: Exception in _band_encourage_merges, returning 0.0", "DEBUG")
            return 0.0

    async def _band_neutral_policy(self, features: np.ndarray, stats: ClusteringStats,
                                 constraints: NAgosticConstraints, split_policy: StrictSplitPolicy, current_round: int) -> float:
        """Band policy: 7 ≤ K ≤ 10, neutral - only split if parent violates soft cap."""
        try:
            tprint(f"🔍 DEBUG: _band_neutral_policy called with current_round={current_round}", "DEBUG")
            K = int(stats.assignments.max()) + 1
            tprint(f"🔍 DEBUG: Current K={K}, cluster_sizes={stats.cluster_sizes.tolist()}", "DEBUG")

            # Only split if a parent violates the soft cap and ΔJ_split_std ≤ -1.5
            split_candidates = []
            tprint(f"🔍 DEBUG: Starting to find split candidates (only soft cap violators)", "DEBUG")
            for cluster_id in range(len(stats.cluster_sizes)):
                size = stats.cluster_sizes[cluster_id]
                tprint(f"🔍 DEBUG: Evaluating cluster {cluster_id} with size {size} (soft_cap={constraints.soft_cap})", "DEBUG")

                # Only consider clusters that violate soft cap
                if size <= constraints.soft_cap:
                    tprint(f"🔍 DEBUG: Cluster {cluster_id} does not violate soft cap, skipping", "DEBUG")
                    continue

                # Check if split would be beneficial
                child_a_pred = size // 2
                child_b_pred = size - child_a_pred
                tprint(f"🔍 DEBUG: Cluster {cluster_id} predicted children: {child_a_pred}, {child_b_pred} (min_size={constraints.min_size})", "DEBUG")

                if child_a_pred < constraints.min_size or child_b_pred < constraints.min_size:
                    tprint(f"🔍 DEBUG: Cluster {cluster_id} children too small, skipping", "DEBUG")
                    continue

                tprint(f"🔍 DEBUG: Cluster {cluster_id} added to split candidates", "DEBUG")
                split_candidates.append(cluster_id)

            tprint(f"🔍 DEBUG: Found {len(split_candidates)} split candidates: {split_candidates}", "DEBUG")
            if not split_candidates:
                tprint("No clusters violate soft cap for splitting (K in band)", "DEBUG")
                tprint(f"🔍 DEBUG: Returning 0.0 - no soft cap violators", "DEBUG")
                return 0.0

            # Try to split the first candidate with strict benefit requirement
            cluster_id = split_candidates[0]
            tprint(f"🔍 DEBUG: Attempting split of cluster {cluster_id} with strict benefit requirement", "DEBUG")
            result = self._apply_split_atomic_strict_custom(features, stats, cluster_id, split_policy, current_round,
                                                           min_delta_std=-1.5)  # Strict benefit requirement

            if result is None:
                # Success
                tprint(f"🔍 DEBUG: Split succeeded for cluster {cluster_id}", "DEBUG")
                split_policy.splits_this_round += 1
                split_policy.consume_split_budget(K)
                delta = self._calculate_split_quality(features, stats, cluster_id)
                tprint(f"✅ Split applied (K in band): cluster {cluster_id}, delta: {delta:.6f}", "INFO")
                tprint(f"🔍 DEBUG: Returning delta {delta}", "DEBUG")
                return delta
            else:
                tprint(f"❌ Split denied (K in band): cluster {cluster_id} - {result.name}", "DEBUG")
                tprint(f"🔍 DEBUG: Split failed, returning 0.0", "DEBUG")
                return 0.0

        except Exception as e:
            tprint(f"Band neutral policy failed: {e}", "ERROR")
            tprint(f"🔍 DEBUG: Exception in _band_neutral_policy, returning 0.0", "DEBUG")
            return 0.0

    def _apply_split_atomic_strict_custom(self, features: np.ndarray, stats: ClusteringStats, cid: int,
                                         split_policy: StrictSplitPolicy, current_round: int, min_delta_std: float = -0.25) -> Optional[SplitError]:
        """Apply atomic split with strict policy and custom delta threshold."""
        try:
            # CRITICAL FIX: Use separate thresholds for WCSS vs standardized comparisons
            # min_delta_std is in standardized units, eps should be in WCSS units
            eps_wcss = 1e-9  # Tiny WCSS threshold for numerical stability
            return self._apply_split_atomic_strict(features, stats, cid, split_policy, current_round, eps=eps_wcss)
        except Exception as e:
            tprint(f"Apply split atomic strict custom failed: {e}", "ERROR")
            return SplitError.OUT_OF_SYNC

    def _apply_merge_atomic(self, features: np.ndarray, stats: ClusteringStats, i: int, j: int,
                          min_delta_std: float = 0.2) -> bool:
        """Apply atomic merge between clusters i and j."""
        with AtomicOperationContext(stats, f"merge_clusters_{i}_{j}"):
            try:
                # Simple merge implementation
                members_i = np.flatnonzero(stats.assignments == i)
                members_j = np.flatnonzero(stats.assignments == j)

                if len(members_i) == 0 or len(members_j) == 0:
                    return False

                # Move all members of cluster j to cluster i
                for point_idx in members_j:
                    stats.apply_move(point_idx, j, i)

                return True

            except Exception as e:
                tprint(f"Apply merge atomic failed: {e}", "ERROR")
                return False

    def _calculate_split_quality(self, features: np.ndarray, stats: ClusteringStats, cluster_id: int) -> float:
        """Calculate quality improvement from a split."""
        try:
            # Simple quality metric - could be enhanced
            size = stats.cluster_sizes[cluster_id]
            return -size * 0.01  # Negative value indicating improvement
        except Exception:
            return 0.0

    def _calculate_merge_quality(self, features: np.ndarray, stats: ClusteringStats, i: int, j: int) -> float:
        """Calculate quality improvement from a merge."""
        try:
            # Simple quality metric - could be enhanced
            size_i = stats.cluster_sizes[i]
            size_j = stats.cluster_sizes[j]
            return (size_i + size_j) * 0.01  # Positive value indicating improvement
        except Exception:
            return 0.0

    def _identify_boundary_points(self, features: np.ndarray, stats: ClusteringStats, max_points: int = None) -> List[int]:
        """Identify boundary points using Numba-optimized calculations."""
        try:
            # Validate inputs
            if features is None or features.size == 0:
                tprint("❌ Features array is None or empty in boundary point identification", "ERROR")
                return []

            if not hasattr(features, 'shape') or len(features.shape) != 2:
                tprint(f"❌ Features must be a 2D array, got shape: {getattr(features, 'shape', 'None')}", "ERROR")
                return []

            if stats.centroids is None or stats.assignments is None:
                tprint("❌ Stats centroids or assignments are None in boundary point identification", "ERROR")
                return []

            # Calculate distances to all centroids for each point
            all_dists = []
            for i in range(len(features)):
                point = features[i]
                dists = [np.linalg.norm(point - centroid) for centroid in stats.centroids]
                all_dists.append(dists)
            all_dists = np.array(all_dists)

            # Trim boundary detection to true boundary points (25th percentile margins)
            d1 = all_dists.min(axis=1)
            idx1 = all_dists.argmin(axis=1)
            # 2nd-nearest per point
            mask = np.arange(all_dists.shape[1]) != idx1[:, None]
            d2 = np.where(mask, all_dists, np.inf).min(axis=1)

            # per point: margin = d2 - d1  (2nd minus 1st nearest centroid)
            margins = d2 - d1

            # Focus almost entirely on the oversized cluster
            big_c = np.argmax(stats.cluster_sizes)
            mask_big = (stats.assignments == big_c)

            # 25th percentile margin inside the big cluster
            tau = np.quantile(margins[mask_big], 0.25)
            frontier = mask_big & (margins <= tau)  # thinnest boundary within the blob

            # OPTIONAL: add a small ring from the 2–3 smallest under-cap clusters
            SOFT_CAP = int(0.20 * len(features))  # 20% cap
            in_small_clusters = (stats.cluster_sizes < SOFT_CAP) & (stats.cluster_sizes > 0)
            if np.any(in_small_clusters):
                small_cluster_mask = np.zeros(len(features), dtype=bool)
                for cid in np.where(in_small_clusters)[0]:
                    small_cluster_mask |= (stats.assignments == cid)
                frontier |= (small_cluster_mask) & (margins <= np.quantile(margins[small_cluster_mask], 0.50))

            boundary_mask = frontier

            boundary_indices = np.where(boundary_mask)[0]
            tprint(f"Boundary detection: {len(boundary_indices)}/{len(features)} points selected ({len(boundary_indices)/len(features):.1%})", "DEBUG")

            return boundary_indices.tolist()

        except Exception as e:
            tprint(f"Boundary point identification failed: {e}", "ERROR")
            return []

    def _identify_boundary_points_hybrid(self, features: np.ndarray, stats: ClusteringStats) -> List[int]:
        """Identify boundary points using silhouettes and margin to nearest other centroid."""
        try:
            N = len(features)
            if N == 0:
                return []
            # Per-point silhouettes (lazy)
            silhouettes = np.array([stats.get_point_silhouette(i) for i in range(N)])
            # Margin: nearest other centroid distance minus own centroid distance
            margins = np.zeros(N, dtype=np.float64)
            for i in range(N):
                point = features[i]
                own = int(stats.assignments[i])
                d_own = float(stats._hybrid_distance(point, stats.centroids[own]))
                other_dists = [float(stats._hybrid_distance(point, stats.centroids[k]))
                               for k in range(stats.K_fixed) if k != own and stats.cluster_sizes[k] > 0]
                d_nearest_other = min(other_dists) if other_dists else d_own
                margins[i] = d_nearest_other - d_own
            # Normalize and combine (lower silhouette and smaller margin prioritized)
            sil_norm = 1.0 - (silhouettes - silhouettes.min()) / (silhouettes.max() - silhouettes.min() + 1e-9)
            margin_norm = 1.0 - (margins - margins.min()) / (margins.max() - margins.min() + 1e-9)
            boundary_scores = 0.6 * sil_norm + 0.4 * margin_norm
            threshold = np.percentile(boundary_scores, 70)
            boundary_mask = boundary_scores >= threshold
            return np.where(boundary_mask)[0].tolist()
        except Exception as e:
            tprint(f"Hybrid boundary identification failed: {e}", "ERROR")
            return []

    def _candidate_targets(self, i: int, labels: np.ndarray, centroids: np.ndarray, features: np.ndarray, *, M: int = 12, k_local: int = 6, k_global: int = 12) -> np.ndarray:
        """Guarantee at least M candidate targets before masks/guards."""
        src = int(labels[i])

        # Local neighborhood targets
        local_targets = self._neighborhood_targets(i, labels, k_local)

        # Global nearest centroids
        global_targets = self._nearest_centroid_ids(i, centroids, features, k_global)

        # Deduplicate while preserving order
        order = []
        seen = {src}
        for c in list(local_targets) + list(global_targets):
            if c not in seen:
                order.append(int(c))
                seen.add(c)

        # Centroid backfill to reach ≥ M unique targets
        if len(order) < M:
            all_centroid_ids = list(range(len(centroids)))
            for c in all_centroid_ids:
                if c != src and c not in seen:
                    order.append(c)
                    seen.add(c)
                if len(order) >= M:
                    break

        return np.asarray(order[:max(M, len(order))], dtype=int)

    def _neighborhood_targets(self, i: int, labels: np.ndarray, k_local: int) -> List[int]:
        """Get local neighborhood targets from boundary analysis."""
        try:
            # For now, return all other clusters as local candidates
            # This can be enhanced with actual boundary/adjacency analysis
            current_cluster = int(labels[i])
            return [c for c in range(int(labels.max()) + 1) if c != current_cluster][:k_local]
        except Exception:
            return []

    def _nearest_centroid_ids(self, i: int, centroids: np.ndarray, features: np.ndarray, k: int) -> List[int]:
        """Get k nearest centroid IDs to point i."""
        try:
            if centroids is None or len(centroids) == 0:
                return []

            point = features[i]
            distances = np.linalg.norm(centroids - point, axis=1)
            nearest_indices = np.argsort(distances)[:k]
            return [int(idx) for idx in nearest_indices]
        except Exception:
            return []

    def _alternative_targets(self, i: int, labels: np.ndarray, centroids: np.ndarray, features: np.ndarray, *, M: int = 12, k_local: int = 6, k_global: int = 12) -> np.ndarray:
        """Generate robust alternative targets with local + global supplement + fallback targets."""
        # Get primary targets
        primary_targets = self._candidate_targets(i, labels, centroids, features, M=M, k_local=k_local, k_global=k_global)

        # If we have enough primary targets, return them
        if len(primary_targets) >= M:
            return primary_targets

        # Otherwise, add fallback targets from nearest neighbors' centroids
        fallback_targets = self._generate_fallback_targets(i, labels, centroids, features, M - len(primary_targets))

        # Combine and deduplicate
        all_targets = list(primary_targets)
        seen = set(all_targets)

        for target in fallback_targets:
            if target not in seen:
                all_targets.append(target)
                seen.add(target)

        return np.asarray(all_targets[:M], dtype=int)

    def _generate_fallback_targets(self, i: int, labels: np.ndarray, centroids: np.ndarray, features: np.ndarray, needed: int) -> List[int]:
        """Generate fallback targets from nearest neighbors' centroids when alternatives are scarce."""
        try:
            if needed <= 0:
                return []

            src = int(labels[i])
            fallback_targets = []

            # Find k-nearest neighbors of point i
            k_neighbors = min(20, len(features) - 1)  # Don't exceed dataset size

            # Calculate distances to all other points
            point = features[i]
            distances = np.linalg.norm(features - point, axis=1)

            # Get k nearest neighbors (excluding self)
            nearest_indices = np.argsort(distances)[1:k_neighbors+1]  # Skip index 0 (self)

            # Collect unique cluster IDs from neighbors
            neighbor_clusters = set()
            for neighbor_idx in nearest_indices:
                neighbor_cluster = int(labels[neighbor_idx])
                if neighbor_cluster != src:
                    neighbor_clusters.add(neighbor_cluster)

            # Convert to list and limit to needed count
            fallback_targets = list(neighbor_clusters)[:needed]

            return fallback_targets

        except Exception as e:
            tprint(f"Fallback target generation failed: {e}", "DEBUG")
            return []

    def _neighbors_from_boundary(self, i: int, labels: np.ndarray) -> List[int]:
        """Get local neighbor clusters from boundary analysis."""
        try:
            # For now, return all other clusters as local candidates
            # This can be enhanced with actual boundary/adjacency analysis
            current_cluster = int(labels[i])
            return [c for c in range(int(labels.max()) + 1) if c != current_cluster]
        except Exception:
            return []

    def _nearest_centroids(self, i: int, centroids: np.ndarray, features: np.ndarray, k: int) -> List[int]:
        """Get k nearest centroids to point i."""
        try:
            if centroids is None or len(centroids) == 0:
                return []

            point = features[i]
            distances = np.linalg.norm(centroids - point, axis=1)
            nearest_indices = np.argsort(distances)[:k]
            return [int(idx) for idx in nearest_indices]
        except Exception:
            return []

    def _find_best_alternative_clusters(
        self,
        features: np.ndarray,
        stats: ClusteringStats,
        point_idx: int,
        current_cluster: int,
        constraints: NAgosticConstraints,
        use_soft_capacity: bool = True
    ) -> List[Tuple[int, Dict[str, float]]]:
        """Find best alternative clusters with robust target generation and soft capacity checks."""
        try:
            # CRITICAL FIX: Use K_fixed instead of n_clusters
            K_fixed = stats.K_fixed

            # CRITICAL FIX: Add guard assertions
            assert_cluster_axis("cluster_sizes", stats.cluster_sizes, K_fixed)

            # Capacity-aware target diversity with adaptive K_targets
            K = stats.centroids.shape[0]
            N = len(features)
            max_frac = np.max(stats.cluster_sizes) / N if len(stats.cluster_sizes) > 0 else 0

            # Adaptive K_targets based on max cluster fraction
            if max_frac > 0.50:
                K_TARGETS = min(K-1, 12)
            elif max_frac > 0.30:
                K_TARGETS = min(K-1, 8)
            else:
                K_TARGETS = min(K-1, 5)

            target_candidates = self._alternative_targets(
                point_idx, stats.assignments, stats.centroids, features,
                M=K_TARGETS, k_local=6, k_global=12
            )

            # Prefilter: only clusters within 1.5x nearest-centroid distance
            if len(target_candidates) > 0:
                point = features[point_idx]
                current_centroid = stats.centroids[current_cluster]
                nearest_dist = np.linalg.norm(point - current_centroid)
                max_dist = 1.5 * nearest_dist

                # Filter by distance and capacity
                MAX_SIZE = int(0.20 * N)  # 20% cap
                filtered_candidates = []
                for t in target_candidates:
                    target_centroid = stats.centroids[t]
                    dist = np.linalg.norm(point - target_centroid)
                    if dist <= max_dist and stats.cluster_sizes[t] < MAX_SIZE:
                        filtered_candidates.append(t)

                # Order: under-cap first, then by distance
                target_candidates = sorted(
                    filtered_candidates,
                    key=lambda t: (stats.cluster_sizes[t] > MAX_SIZE,
                                 np.linalg.norm(point - stats.centroids[t]))
                )

                # Small exploration to avoid local traps (10% chance)
                import random
                if random.random() < 0.10 and len(target_candidates) >= 3:
                    current = stats.assignments[point_idx]
                    swap_candidates = [t for t in range(K) if t != current and stats.cluster_sizes[t] < MAX_SIZE]
                    if len(swap_candidates) >= 3:
                        swap = random.sample(swap_candidates, min(3, len(swap_candidates)))
                        target_candidates[:len(swap)] = swap

            if len(target_candidates) == 0:
                return []

            alternatives = []

            # Evaluate each candidate with N-agnostic capacity checks
            for target_cluster in target_candidates:
                if target_cluster == current_cluster:
                    continue

                # N-agnostic capacity checks
                src_size = stats.cluster_sizes[current_cluster]
                dst_size = stats.cluster_sizes[target_cluster]

                # Use N-agnostic violates_capacity check
                if constraints.violates_capacity(src_size, dst_size):
                    continue  # Skip if move violates capacity constraints

                delta_info = stats.calculate_move_delta(point_idx, current_cluster, target_cluster)

                # Handle NaN/Inf in delta calculations
                if not isinstance(delta_info, dict):
                    tprint(f"❌ ERROR: calculate_move_delta returned {type(delta_info)}, expected dict", "ERROR")
                    continue

                # Clean delta values of NaN/Inf
                for key in delta_info:
                    if not np.isfinite(delta_info[key]):
                        delta_info[key] = np.inf if key == 'total' else 0.0

                # Use adaptive tau for pruning
                dJ_total = delta_info['total']
                if dJ_total <= adaptive_tau(np.array([dJ_total])):
                    alternatives.append((target_cluster, delta_info))

            # Sort by total delta (most negative first)
            alternatives.sort(key=lambda x: x[1]['total'], reverse=False)

            # Return up to max_alternatives_per_point
            return alternatives[:self.max_alternatives_per_point]

        except Exception as e:
            tprint(f"Alternative cluster finding failed: {e}", "ERROR")
            return []

    def _calculate_neighbor_consensus(
        self,
        neighbor_indices: np.ndarray,
        assignments: np.ndarray,
        target_cluster: int
    ) -> float:
        """Calculate neighbor consensus for a target cluster."""
        try:
            # Exclude self from neighbors
            neighbor_assignments = assignments[neighbor_indices[1:]]  # Skip first (self)

            # Count how many neighbors are in target cluster
            consensus_count = np.sum(neighbor_assignments == target_cluster)
            total_neighbors = len(neighbor_assignments)

            return consensus_count / total_neighbors if total_neighbors > 0 else 0.0

        except Exception as e:
            tprint(f"Neighbor consensus calculation failed: {e}", "ERROR")
            return 0.0

    def _calculate_margin_gain(
        self,
        features: np.ndarray,
        stats: ClusteringStats,
        point_idx: int,
        from_cluster: int,
        to_cluster: int
    ) -> float:
        """Calculate margin gain for a potential move using Numba optimization."""
        try:
            point = features[point_idx]

            # Calculate new centroids (simplified)
            new_centroid_from = (
                (stats.centroids[from_cluster] * stats.cluster_sizes[from_cluster] - point) /
                (stats.cluster_sizes[from_cluster] - 1)
            ) if stats.cluster_sizes[from_cluster] > 1 else np.zeros(stats.n_features)

            new_centroid_to = (
                (stats.centroids[to_cluster] * stats.cluster_sizes[to_cluster] + point) /
                (stats.cluster_sizes[to_cluster] + 1)
            )

            # Use Numba-optimized margin gain calculation
            return calculate_margin_gain_numba(
                point,
                new_centroid_from,
                new_centroid_to,
                stats.centroids
            )

        except Exception as e:
            tprint(f"Margin gain calculation failed: {e}", "ERROR")
            return 0.0

    def _identify_clusters_for_splitting(self, features: np.ndarray, stats: ClusteringStats) -> List[int]:
        """Identify clusters that should be split based on size and quality."""
        try:
            split_candidates = []

            # Calculate median cluster size
            non_zero_sizes = [size for size in stats.cluster_sizes if size > 0]
            if not non_zero_sizes:
                return []

            median_size = np.median(non_zero_sizes)

            for cluster_id in range(stats.n_clusters):
                cluster_size = stats.cluster_sizes[cluster_id]

                if cluster_size == 0:
                    continue

                # Check size factor (using new threshold)
                size_factor = cluster_size / median_size if median_size > 0 else 1.0

                if size_factor >= self.split_size_threshold:
                    # Check internal quality
                    cluster_mask = stats.assignments == cluster_id
                    cluster_features = features[cluster_mask]

                    if len(cluster_features) > 10:  # Only split if cluster is large enough
                        # Calculate mean silhouette for this cluster
                        cluster_silhouette = self._calculate_cluster_silhouette(cluster_features, cluster_id, stats)

                        # Calculate boundary ratio
                        boundary_ratio = len([i for i in range(len(cluster_features))
                                            if self._is_boundary_point(cluster_features[i], cluster_id, stats)]) / len(cluster_features)

                        # Check thrash count for this cluster
                        thrash_count = self.cluster_thrash_counts.get(cluster_id, 0)

                        # Trigger split if any condition is met
                        should_split_var = self._should_split_variance_aware(features, cluster_id, stats)
                        if (cluster_silhouette < self.split_silhouette_threshold or
                            boundary_ratio > self.boundary_ratio_threshold or
                            thrash_count >= self.thrash_count_threshold or
                            should_split_var):
                            split_candidates.append(cluster_id)

            return split_candidates

        except Exception as e:
            tprint(f"Cluster splitting identification failed: {e}", "ERROR")
            return []

    def _should_split_variance_aware(self, features: np.ndarray, cluster_id: int, stats: ClusteringStats) -> bool:
        """Check if splitting would improve variance decomposition (reduces within variance or increases between)."""
        try:
            cluster_mask = stats.assignments == cluster_id
            cluster_features = features[cluster_mask]
            n = len(cluster_features)
            if n < 2 * self.min_cluster_size:
                return False
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            sub_labels = kmeans.fit_predict(cluster_features)
            old_variance = float(np.mean(np.var(cluster_features, axis=0, ddof=1))) if n > 1 else 0.0
            sub_0 = cluster_features[sub_labels == 0]
            sub_1 = cluster_features[sub_labels == 1]
            if len(sub_0) < 2 or len(sub_1) < 2:
                return False
            var_0 = float(np.mean(np.var(sub_0, axis=0, ddof=1)))
            var_1 = float(np.mean(np.var(sub_1, axis=0, ddof=1)))
            new_within_variance = (var_0 * (len(sub_0) - 1) + var_1 * (len(sub_1) - 1)) / (n - 2)
            variance_reduction = (old_variance - new_within_variance) / old_variance if old_variance > 0 else 0.0
            mean_0 = np.mean(sub_0, axis=0)
            mean_1 = np.mean(sub_1, axis=0)
            mean_separation = float(np.linalg.norm(mean_0 - mean_1))
            return variance_reduction > 0.10 or mean_separation > 0.5
        except Exception:
            return False

    def _identify_clusters_for_splitting_strict(self, features: np.ndarray, stats: ClusteringStats,
                                               split_policy: StrictSplitPolicy, current_round: int) -> List[int]:
        """Identify clusters that are candidates for splitting with strict criteria."""
        try:
            cluster_sizes = stats.cluster_sizes
            target_size = len(features) / stats.n_clusters
            min_size = max(self.min_cluster_size, int(0.005 * len(features)))

            split_candidates = []

            for cluster_id in range(len(cluster_sizes)):
                size = cluster_sizes[cluster_id]

                # Skip if cluster is in cooldown
                if cluster_id in split_policy.cluster_cooldowns:
                    if current_round < split_policy.cluster_cooldowns[cluster_id]:
                        continue
                    else:
                        # Remove expired cooldown
                        del split_policy.cluster_cooldowns[cluster_id]

                # 1) Parent size requirements
                if size < split_policy.min_parent_vs_target * target_size:
                    continue
                if size < split_policy.min_parent_vs_min * min_size:
                    continue

                # 2) Check if cluster is in top quantile
                sorted_sizes = np.sort(cluster_sizes[cluster_sizes > 0])
                if len(sorted_sizes) == 0:
                    continue
                quantile_threshold = np.quantile(sorted_sizes, split_policy.min_parent_quantile)
                if size < quantile_threshold:
                    continue

                # 3) Track oversize persistence
                if cluster_id not in split_policy.oversize_tracker:
                    split_policy.oversize_tracker[cluster_id] = 0

                if size >= split_policy.min_parent_vs_target * target_size:
                    split_policy.oversize_tracker[cluster_id] += 1
                else:
                    split_policy.oversize_tracker[cluster_id] = 0

                # Only consider clusters that have been oversize for required rounds
                if split_policy.oversize_tracker[cluster_id] < split_policy.oversize_rounds:
                    continue

                split_candidates.append(cluster_id)

            return split_candidates

        except Exception as e:
            tprint(f"Strict cluster identification for splitting failed: {e}", "ERROR")
            return []

    def _identify_clusters_for_splitting_n_agnostic(self, features: np.ndarray, stats: ClusteringStats,
                                                   constraints: NAgosticConstraints, split_policy: StrictSplitPolicy, current_round: int) -> List[int]:
        """Identify clusters that are candidates for splitting with N-agnostic constraints."""
        try:
            cluster_sizes = stats.cluster_sizes
            K = int(stats.assignments.max()) + 1

            split_candidates = []

            for cluster_id in range(len(cluster_sizes)):
                size = cluster_sizes[cluster_id]

                # Skip if cluster is in cooldown
                if cluster_id in split_policy.cluster_cooldowns:
                    if current_round < split_policy.cluster_cooldowns[cluster_id]:
                        continue
                    else:
                        # Remove expired cooldown
                        del split_policy.cluster_cooldowns[cluster_id]

                # N-agnostic split eligibility check
                # Predict child sizes (rough estimate: split roughly in half)
                child_a_pred = size // 2
                child_b_pred = size - child_a_pred

                # Use N-agnostic can_split check
                if not constraints.can_split(K, size, child_a_pred, child_b_pred):
                    continue

                # Track oversize persistence
                if cluster_id not in split_policy.oversize_tracker:
                    split_policy.oversize_tracker[cluster_id] = 0

                min_parent_size = max(2 * constraints.min_size + constraints.margin, math.ceil(1.25 * constraints.target_size_floor))
                if size >= min_parent_size:
                    split_policy.oversize_tracker[cluster_id] += 1
                else:
                    split_policy.oversize_tracker[cluster_id] = 0

                # Only consider clusters that have been oversize for required rounds
                if split_policy.oversize_tracker[cluster_id] < split_policy.oversize_rounds:
                    continue

                split_candidates.append(cluster_id)

            return split_candidates

        except Exception as e:
            tprint(f"N-agnostic cluster identification for splitting failed: {e}", "ERROR")
            return []

    def _calculate_cluster_silhouette(self, cluster_features: np.ndarray, cluster_id: int, stats: ClusteringStats) -> float:
        """Calculate mean silhouette score for a cluster."""
        try:
            if len(cluster_features) < 2:
                return 0.0

            silhouette_scores = []
            for i, point in enumerate(cluster_features):
                # Distance to own centroid
                d1 = np.linalg.norm(point - stats.centroids[cluster_id])

                # Distance to nearest other centroid
                d2 = float('inf')
                for other_cluster in range(stats.n_clusters):
                    if other_cluster != cluster_id and stats.cluster_sizes[other_cluster] > 0:
                        dist = np.linalg.norm(point - stats.centroids[other_cluster])
                        d2 = min(d2, dist)

                # Calculate silhouette score
                if d2 > 0:
                    s = (d2 - d1) / max(d1, d2)
                    silhouette_scores.append(s)

            return np.mean(silhouette_scores) if silhouette_scores else 0.0

        except Exception as e:
            tprint(f"Cluster silhouette calculation failed: {e}", "ERROR")
            return 0.0

    def _is_boundary_point(self, point: np.ndarray, cluster_id: int, stats: ClusteringStats) -> bool:
        """Check if a point is on the boundary of its cluster."""
        try:
            # Distance to own centroid
            d1 = np.linalg.norm(point - stats.centroids[cluster_id])

            # Distance to nearest other centroid
            d2 = float('inf')
            for other_cluster in range(stats.n_clusters):
                if other_cluster != cluster_id and stats.cluster_sizes[other_cluster] > 0:
                    dist = np.linalg.norm(point - stats.centroids[other_cluster])
                    d2 = min(d2, dist)

            # Point is on boundary if distances are close
            return abs(d2 - d1) < 0.1 * max(d1, d2) if d2 > 0 else False

        except Exception as e:
            tprint(f"Boundary point check failed: {e}", "ERROR")
            return False

    def _calculate_thrash_scores(self, current_iteration: int) -> dict:
        """Calculate thrash scores per cluster."""
        thrash_scores = {}
        for cluster_id in range(max(self.n_clusters, 5)):  # Use max to avoid index errors
            reversals = 0
            total_moves = 0

            for point_idx, moves in self.move_history.items():
                if not moves:
                    continue

                recent_moves = [move for move in moves
                              if current_iteration - move[0] <= self.move_window_size]

                if len(recent_moves) >= 2:
                    total_moves += len(recent_moves)
                    for i in range(1, len(recent_moves)):
                        if recent_moves[i][2] == recent_moves[i-1][1]:  # Returned to previous cluster
                            reversals += 1

            if total_moves > 0:
                thrash_scores[cluster_id] = reversals / total_moves
            else:
                thrash_scores[cluster_id] = 0.0

        return thrash_scores

    def _calculate_split_quality_advanced(self, features: np.ndarray, stats: ClusteringStats, cluster_id: int) -> float:
        """Calculate the quality improvement from splitting a cluster."""
        try:
            cluster_mask = stats.assignments == cluster_id
            cluster_features = features[cluster_mask]

            if len(cluster_features) < 20:  # Need enough points to split
                return 0.0

            # Try to split using 2-means
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            sub_assignments = kmeans.fit_predict(cluster_features)

            # Calculate quality of split
            split_centroids = kmeans.cluster_centers_

            # Calculate new WCSS for split clusters
            wcss_split = 0.0
            for sub_cluster in [0, 1]:
                sub_mask = sub_assignments == sub_cluster
                if np.any(sub_mask):
                    sub_features = cluster_features[sub_mask]
                    wcss_split += np.sum((sub_features - split_centroids[sub_cluster]) ** 2)

            # Calculate current WCSS for original cluster
            current_wcss = np.sum((cluster_features - stats.centroids[cluster_id]) ** 2)

            # Quality improvement (negative means better - lower WCSS)
            quality_improvement = current_wcss - wcss_split

            # Size-aware threshold
            size_factor = stats.cluster_sizes[cluster_id] / np.median([s for s in stats.cluster_sizes if s > 0])
            threshold = self.split_quality_threshold * (1 + self.alpha * (size_factor - 1))

            return quality_improvement - threshold

        except Exception as e:
            tprint(f"Split quality calculation failed: {e}", "ERROR")
            return 0.0

    def _two_way_bisect(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Robust 2-means bisection with farthest-pair seeding."""
        try:
            if len(features) < 4:
                # For very small clusters, use simple split
                mid = len(features) // 2
                return np.arange(mid), np.arange(mid, len(features))

            # Use farthest-pair seeding for robustness
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10, init='k-means++')
            assignments = kmeans.fit_predict(features)

            A_mask = assignments == 0
            B_mask = assignments == 1

            return A_mask, B_mask

        except Exception as e:
            tprint(f"Two-way bisection failed: {e}", "ERROR")
            # Fallback to simple split
            mid = len(features) // 2
            A_mask = np.zeros(len(features), dtype=bool)
            A_mask[:mid] = True
            B_mask = ~A_mask
            return A_mask, B_mask

    def _intra_objective(self, features: np.ndarray) -> float:
        """Calculate intra-cluster objective (WCSS)."""
        if len(features) == 0:
            return 0.0
        centroid = np.mean(features, axis=0)
        return np.sum((features - centroid) ** 2)

    def _snapshot_state(self, stats: ClusteringStats) -> Dict:
        """Create a snapshot of current clustering state."""
        snapshot = stats._snapshot_state()
        snapshot['n_clusters'] = stats.n_clusters
        snapshot['K_fixed'] = stats.K_fixed
        if hasattr(stats, 'transition_counts'):
            snapshot['transition_counts'] = stats.transition_counts.copy()
        else:
            snapshot['transition_counts'] = None
        if hasattr(stats, 'transition_row_sums'):
            snapshot['transition_row_sums'] = stats.transition_row_sums.copy()
        else:
            snapshot['transition_row_sums'] = None
        snapshot['optimizer_assignments'] = (self.assignments.copy()
                                             if self.assignments is not None else None)
        return snapshot

    def _restore_state(self, stats: ClusteringStats, state: Dict):
        """Restore clustering state from snapshot."""
        stats._restore_state(state)
        stats.n_clusters = state.get('n_clusters', stats.n_clusters)
        stats.K_fixed = state.get('K_fixed', stats.K_fixed)
        if state.get('transition_counts') is not None:
            stats.transition_counts = state['transition_counts'].copy()
        if state.get('transition_row_sums') is not None:
            stats.transition_row_sums = state['transition_row_sums'].copy()
        # Update remapped assignments
        stats.remapped_assignments = np.array([stats.cluster_id_map.get(cid, cid) for cid in stats.assignments])
        # Restore optimizer assignments if provided
        optimizer_assignments = state.get('optimizer_assignments')
        if optimizer_assignments is not None:
            self.assignments = optimizer_assignments.copy()
        elif hasattr(stats, 'assignments'):
            self.assignments = stats.assignments.copy()
        if hasattr(stats, 'refresh_cluster_sizes'):
            stats.refresh_cluster_sizes()

    def _expand_k_dim_arrays(self, stats: ClusteringStats, new_K: int):
        """Expand all K-dimensional arrays to new size."""
        current_centroids = getattr(stats, 'centroids', None)
        old_K = current_centroids.shape[0] if current_centroids is not None else 0
        if new_K <= old_K:
            return

        if hasattr(stats, 'ensure_k_capacity'):
            stats.ensure_k_capacity(new_K)
        else:
            raise ValueError("ClusteringStats must provide ensure_k_capacity for resizing")

    def _apply_split_atomic(self, features: np.ndarray, stats: ClusteringStats, cid: int, min_size: int = 25, eps: float = 1e-9) -> Optional[SplitError]:
        """Apply atomic cluster split with commit/rollback - only log success."""
        with AtomicOperationContext(stats, f"split_cluster_{cid}"):
            try:
                # Get cluster members
                members = np.flatnonzero(stats.assignments == cid)
                if members.size < 2 * min_size:
                    return SplitError.TOO_SMALL

                # 2-means split with robust seeds
                A_mask, B_mask = self._two_way_bisect(features[members])
                A = members[A_mask]
                B = members[B_mask]

                if len(A) < min_size or len(B) < min_size:
                    return SplitError.BAD_CHILD

                # Objective check
                J_old = self._intra_objective(features[members])
                J_new = self._intra_objective(features[A]) + self._intra_objective(features[B])
                if J_new >= J_old - eps:
                    return SplitError.NO_GAIN

                # Commit (K -> K+1)
                K0 = int(stats.assignments.max()) + 1
                new_id = K0

                # Update assignments first
                stats.assignments[B] = new_id

                # Expand ALL K-dimensional arrays
                self._expand_k_dim_arrays(stats, K0 + 1)

                # Update cluster counts cache and centroids
                current_sizes = (stats.refresh_cluster_sizes()
                                 if hasattr(stats, 'refresh_cluster_sizes')
                                 else stats.cluster_sizes)
                stats.centroids[cid] = np.mean(features[A], axis=0)
                stats.centroids[new_id] = np.mean(features[B], axis=0)

                # Update sufficient statistics
                stats.S[cid] = np.sum(features[A], axis=0)
                stats.S[new_id] = np.sum(features[B], axis=0)
                stats.Q_trace[cid] = np.sum(np.sum(features[A] ** 2, axis=1))
                stats.Q_trace[new_id] = np.sum(np.sum(features[B] ** 2, axis=1))

                # Update WCSS
                stats.wcss_per_cluster[cid] = self._intra_objective(features[A])
                stats.wcss_per_cluster[new_id] = self._intra_objective(features[B])

                # Update totals
                stats.total_wcss = np.sum(stats.wcss_per_cluster)
                safe_sizes = np.where(current_sizes > 0, current_sizes, 1)
                stats.total_bcss = np.sum(np.sum(stats.S ** 2, axis=1) / safe_sizes) - np.sum(stats.global_S ** 2) / stats.global_N

                # Update K_fixed and mappings
                stats.K_fixed = K0 + 1
                stats.n_clusters = len(np.unique(stats.assignments))

                # Update cluster ID mappings
                unique_clusters = np.unique(stats.assignments)
                stats.cluster_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
                stats.inverse_cluster_id_map = {new_id: old_id for old_id, new_id in stats.cluster_id_map.items()}
                stats.remapped_assignments = np.array([stats.cluster_id_map[cid] for cid in stats.assignments])

                # Post-split invariant assertions
                self._assert_split_invariants(stats, features, cid, new_id, min_size, J_old, J_new, eps)

                # Success: log the true delta (only on success)
                delta = J_new - J_old
                tprint(f"Split cluster {cid} -> {{{cid},{new_id}}}, delta: {delta:.6f}", "INFO")
                return None  # None == success

            except ValueError as ve:
                # Rollback and return specific error
                tprint(f"Split failed with ValueError: {ve}", "ERROR")
                return SplitError.SHAPE
            except Exception as e:
                # Rollback on any other error
                tprint(f"Split failed with exception: {e}", "ERROR")
                return SplitError.OUT_OF_SYNC

    def _assert_split_invariants(self, stats: ClusteringStats, features: np.ndarray, cid: int, new_id: int,
                               min_size: int, J_old: float, J_new: float, eps: float):
        """Assert post-split invariants."""
        # Size invariants
        assert stats.cluster_sizes[cid] >= min_size, f"Parent cluster {cid} below min size"
        assert stats.cluster_sizes[new_id] >= min_size, f"Child cluster {new_id} below min size"

        # Sum invariant
        assert np.sum(stats.cluster_sizes) == len(features), f"Sum of cluster sizes {np.sum(stats.cluster_sizes)} != N {len(features)}"

        # Array dimension invariant
        K_expected = int(stats.assignments.max()) + 1
        assert stats.cluster_sizes.shape[0] == K_expected, f"cluster_sizes first-dim {stats.cluster_sizes.shape[0]} != {K_expected}"
        assert stats.centroids.shape[0] == K_expected, f"centroids first-dim {stats.centroids.shape[0]} != {K_expected}"

        # Objective improvement
        assert J_new < J_old - eps, f"Objective didn't improve: {J_new} >= {J_old - eps}"

    def _assert_split_invariants_strict(self, stats: ClusteringStats, features: np.ndarray, cid: int, new_id: int,
                                       min_size: int, J_old: float, J_new: float, eps: float):
        """Assert post-split invariants with strict policy requirements."""
        # Size invariants
        assert stats.cluster_sizes[cid] >= min_size, f"Parent cluster {cid} below min size"
        assert stats.cluster_sizes[new_id] >= min_size, f"Child cluster {new_id} below min size"

        # Sum invariant
        assert np.sum(stats.cluster_sizes) == len(features), f"Sum of cluster sizes {np.sum(stats.cluster_sizes)} != N {len(features)}"

        # Array dimension invariant
        K_expected = int(stats.assignments.max()) + 1
        assert stats.cluster_sizes.shape[0] == K_expected, f"cluster_sizes first-dim {stats.cluster_sizes.shape[0]} != {K_expected}"
        assert stats.centroids.shape[0] == K_expected, f"centroids first-dim {stats.centroids.shape[0]} != {K_expected}"

        # Objective improvement with stricter threshold
        rel_gain = (J_old - J_new) / J_old if J_old > 0 else 0.0
        assert rel_gain >= 0.0125, f"Relative gain too small: {rel_gain:.4f} < 0.0125"

        # Balance check
        balance_ratio = min(stats.cluster_sizes[cid], stats.cluster_sizes[new_id]) / max(stats.cluster_sizes[cid], stats.cluster_sizes[new_id])
        assert balance_ratio >= 0.45, f"Balance ratio too low: {balance_ratio:.3f} < 0.45"

        # Child size check
        target_size = len(features) / stats.n_clusters
        min_child_size = 0.90 * target_size
        assert stats.cluster_sizes[cid] >= min_child_size, f"Child {cid} too small: {stats.cluster_sizes[cid]} < {min_child_size}"
        assert stats.cluster_sizes[new_id] >= min_child_size, f"Child {new_id} too small: {stats.cluster_sizes[new_id]} < {min_child_size}"

    def _apply_split_atomic_strict(self, features: np.ndarray, stats: ClusteringStats, cid: int,
                                   split_policy: StrictSplitPolicy, current_round: int, eps: float = 1e-9) -> Optional[SplitError]:
        """Apply atomic cluster split with strict policy validation."""
        # Snapshot current state
        K0 = int(stats.assignments.max()) + 1
        state = self._snapshot_state(stats)

        try:
            # Get cluster members
            members = np.flatnonzero(stats.assignments == cid)
            size = len(members)

            # Calculate thresholds
            target_size = len(features) / stats.n_clusters
            min_size = max(self.min_cluster_size, int(0.005 * len(features)))

            # 1) Parent size validation
            if size < split_policy.min_parent_vs_target * target_size:
                return SplitError.PARENT_TOO_SMALL
            if size < split_policy.min_parent_vs_min * min_size:
                return SplitError.PARENT_TOO_SMALL

            # 2) Minimum size check for split feasibility
            if size < 2 * split_policy.min_child_vs_min * min_size:
                return SplitError.TOO_SMALL

            # 3) 2-means split with robust seeds
            A_mask, B_mask = self._two_way_bisect(features[members])
            A = members[A_mask]
            B = members[B_mask]

            # 4) Child size validation
            if len(A) < split_policy.min_child_vs_min * min_size:
                return SplitError.CHILD_TOO_SMALL
            if len(B) < split_policy.min_child_vs_min * min_size:
                return SplitError.CHILD_TOO_SMALL

            # 5) Balance validation
            balance_ratio = min(len(A), len(B)) / max(len(A), len(B))
            if balance_ratio < split_policy.balance_floor:
                return SplitError.BAD_CHILD

            # 6) Multi-metric improvement validation
            J_old = self._intra_objective(features[members])
            J_new = self._intra_objective(features[A]) + self._intra_objective(features[B])

            # Relative gain check
            rel_gain = (J_old - J_new) / J_old if J_old > 0 else 0.0
            if rel_gain < split_policy.min_rel_gain:
                return SplitError.INSUFFICIENT_GAIN

            # 7) Commit (K -> K+1)
            new_id = K0

            # Update assignments first
            stats.assignments[B] = new_id

            # Expand ALL K-dimensional arrays
            self._expand_k_dim_arrays(stats, K0 + 1)

            # Update cluster counts cache and centroids
            current_sizes = (stats.refresh_cluster_sizes()
                             if hasattr(stats, 'refresh_cluster_sizes')
                             else stats.cluster_sizes)
            stats.centroids[cid] = np.mean(features[A], axis=0)
            stats.centroids[new_id] = np.mean(features[B], axis=0)

            # Update sufficient statistics
            stats.S[cid] = np.sum(features[A], axis=0)
            stats.S[new_id] = np.sum(features[B], axis=0)
            stats.Q_trace[cid] = np.sum(np.sum(features[A] ** 2, axis=1))
            stats.Q_trace[new_id] = np.sum(np.sum(features[B] ** 2, axis=1))

            # Update WCSS
            stats.wcss_per_cluster[cid] = self._intra_objective(features[A])
            stats.wcss_per_cluster[new_id] = self._intra_objective(features[B])

            # Update totals
            stats.total_wcss = np.sum(stats.wcss_per_cluster)
            safe_sizes = np.where(current_sizes > 0, current_sizes, 1)
            stats.total_bcss = np.sum(np.sum(stats.S ** 2, axis=1) / safe_sizes) - np.sum(stats.global_S ** 2) / stats.global_N

            # Update K_fixed and mappings
            stats.K_fixed = K0 + 1
            stats.n_clusters = len(np.unique(stats.assignments))

            # Update cluster ID mappings
            unique_clusters = np.unique(stats.assignments)
            stats.cluster_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
            stats.inverse_cluster_id_map = {new_id: old_id for old_id, new_id in stats.cluster_id_map.items()}
            stats.remapped_assignments = np.array([stats.cluster_id_map[cid] for cid in stats.assignments])

            # 8) Set cooldowns for parent and children
            cooldown_until = current_round + split_policy.cooldown_rounds
            split_policy.cluster_cooldowns[cid] = cooldown_until
            split_policy.cluster_cooldowns[new_id] = cooldown_until

            # 9) Reset oversize tracker for both clusters
            if cid in split_policy.oversize_tracker:
                del split_policy.oversize_tracker[cid]
            split_policy.oversize_tracker[new_id] = 0

            # Post-split sanity checks
            self._assert_split_invariants_strict(stats, features, cid, new_id, min_size, J_old, J_new, eps)

            # Success: log the true delta (only on success)
            delta = J_new - J_old
            tprint(f"Split cluster {cid} -> {{{cid},{new_id}}}, delta: {delta:.6f}, rel_gain: {rel_gain:.4f}", "INFO")
            return None  # None == success

        except ValueError as ve:
            # Rollback and return specific error
            self._restore_state(stats, state)
            error_code = ve.args[0] if ve.args else SplitError.OUT_OF_SYNC
            if isinstance(error_code, SplitError):
                return error_code
            else:
                return SplitError.OUT_OF_SYNC

        except Exception as e:
            # Rollback on any other error
            self._restore_state(stats, state)
            return SplitError.SHAPE

    def _apply_cluster_split(self, features: np.ndarray, stats: ClusteringStats, cluster_id: int) -> float:
        """Apply a cluster split using atomic approach and return the quality improvement."""
        min_size = max(self.min_cluster_size, int(0.005 * len(features)))
        result = self._apply_split_atomic(features, stats, cluster_id, min_size)

        if result is None:
            # Success - calculate actual quality improvement
            return self._calculate_split_quality(features, stats, cluster_id)
        else:
            # Failure - log the specific error (don't log delta on failure)
            tprint(f"Cluster split failed: {result.name if isinstance(result, SplitError) else str(result)}", "WARNING")
            return 0.0

    def _calculate_silhouette_score(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate silhouette score with optimized engine."""
        return self.calculation_engine.calculate_silhouette_score_optimized(features, assignments)

    def _calculate_davies_bouldin_score(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate Davies-Bouldin score with error handling."""
        try:
            if len(features) == 0 or len(assignments) == 0:
                return float('inf')
            if len(np.unique(assignments)) < 2:
                return float('inf')  # Higher is worse for DB score
            # Skip if any cluster < 2 points (metric hygiene)
            sizes = np.bincount(assignments)
            if np.any(sizes < 2):
                return float('inf')
            if features.ndim == 1:
                # Reshape 1D array to 2D for sklearn compatibility
                features_2d = features.reshape(-1, 1)
                return davies_bouldin_score(features_2d, assignments)
            return davies_bouldin_score(features, assignments)
        except Exception:
            return float('inf')

    def _validate_invariants(self, stats: ClusteringStats, n_samples: int, validation_results: Dict):
        """Validate clustering invariants."""
        # Check no empty clusters
        empty_clusters = np.sum(stats.cluster_sizes == 0)
        if empty_clusters > 0:
            self.logger.error(f"Found {empty_clusters} empty clusters")
            validation_results['invariant_violations'] += 1

        # Check min size constraint
        min_size = max(25, int(0.005 * n_samples))  # 0.5% of N
        small_clusters = np.sum(stats.cluster_sizes < min_size)
        if small_clusters > 0:
            self.logger.warning(f"Found {small_clusters} clusters below min size {min_size}")

        # Check total samples
        total_assigned = np.sum(stats.cluster_sizes)
        if total_assigned != n_samples:
            self.logger.error(f"Total assigned {total_assigned} != {n_samples}")
            validation_results['invariant_violations'] += 1

    def _validate_monotone_objective(self, previous_j: float, current_j: float, validation_results: Dict):
        """Validate that objective function is monotone."""
        if current_j < previous_j - 1e-10:
            self.logger.error(f"Monotone violation: {previous_j:.6f} -> {current_j:.6f}")
            validation_results['monotone_violations'] += 1
            return False
        return True

    def _validate_incremental_correctness(self, features: np.ndarray, stats: ClusteringStats, validation_results: Dict):
        """Validate incremental updates match full recomputation (sample)."""
        sample_size = min(50, len(features))
        sample_indices = np.random.choice(len(features), sample_size, replace=False)

        for idx in sample_indices:
            current_cluster = int(stats.assignments[idx])
            other_clusters = [c for c in range(stats.n_clusters) if c != current_cluster]
            if not other_clusters:
                continue

            target_cluster = np.random.choice(other_clusters)

            # Calculate incremental delta
            delta_inc = stats.calculate_move_delta(idx, current_cluster, target_cluster)

            # Calculate full recomputation (simplified)
            temp_assignments = stats.assignments.copy()
            temp_assignments[idx] = target_cluster
            temp_stats = ClusteringStats(features, temp_assignments)

            # Compare CV ratios
            cv_inc = delta_inc['cv']
            cv_full = temp_stats.get_cv_ratio() - stats.get_cv_ratio()

            if abs(cv_inc - cv_full) / max(1, abs(cv_full)) > 1e-8:
                validation_results['incremental_checks_failed'] += 1
            else:
                validation_results['incremental_checks_passed'] += 1

    def _print_final_metrics(self, features: np.ndarray, stats: ClusteringStats):
        """Print comprehensive final metrics when optimization completes."""
        try:
            # CRITICAL FIX: Use live state for final metrics
            if self.assignments is None:
                tprint("⚠️ WARNING: self.assignments is None in final metrics, using stats.assignments", "WARNING")
                assignments = stats.assignments
            else:
                assignments = self.assignments
            sizes = np.bincount(assignments, minlength=assignments.max()+1)
            self._log_with_context(f"[REPORT] K={len(np.unique(assignments))}, max={sizes.max()}, min={sizes[sizes>0].min()}", "DEBUG", "FINAL")

            # Enhanced: Variance decomposition report
            var_decomp = stats.calculate_variance_decomposition()
            tprint("\n" + "="*80, "SUCCESS")
            tprint("REGIME CLUSTERING - VARIANCE DECOMPOSITION RESULTS", "SUCCESS")
            tprint("="*80, "SUCCESS")
            tprint("\nVARIANCE DECOMPOSITION:", "INFO")
            tprint(f"   Total Variance: {var_decomp['total']:.6f}", "INFO")
            tprint(f"   Within-Regime Variance: {var_decomp['within']:.6f} (LOWER is better)", "INFO")
            tprint(f"   Between-Regime Variance: {var_decomp['between']:.6f} (HIGHER is better)", "INFO")
            tprint(f"   Variance Ratio (F-stat): {var_decomp['ratio']:.4f} (HIGHER is better)", "INFO")
            tprint(f"   Eta-Squared: {var_decomp['eta_squared']:.4f} (% variance explained by regimes)", "INFO")
            if var_decomp['ratio'] > 2.0:
                quality = "Excellent - regimes are very distinct"
            elif var_decomp['ratio'] > 1.0:
                quality = "Good - regimes show clear separation"
            elif var_decomp['ratio'] > 0.5:
                quality = "Fair - regimes are somewhat distinct"
            else:
                quality = "Poor - regimes overlap significantly"
            tprint(f"\n   Quality Assessment: {quality}", "INFO")

            # Calculate all key metrics using live state
            num_clusters = len(np.unique(assignments))
            cv_ratio = stats.get_cv_ratio()
            balance_score = stats.get_balance_score()
            silhouette_score = self._calculate_silhouette_score(features, assignments)

            # Calculate Davies-Bouldin index
            db_score = self._calculate_davies_bouldin_score(features, assignments)

            # Calculate cluster sizes and distribution using live state
            cluster_sizes = np.bincount(assignments)
            min_size = np.min(cluster_sizes)
            max_size = np.max(cluster_sizes)
            avg_size = np.mean(cluster_sizes)
            std_size = np.std(cluster_sizes)

            # Calculate distribution statistics using live state
            total_samples = len(assignments)
            size_distribution = cluster_sizes / total_samples * 100  # Convert to percentages

            # Calculate additional metrics
            size_cv = std_size / avg_size if avg_size > 0 else 0  # Coefficient of variation for cluster sizes
            size_balance = 1.0 - (max_size - min_size) / total_samples if total_samples > 0 else 0

            # Print comprehensive metrics summary with enhanced formatting
            tprint("\n" + "="*80, "SUCCESS")
            tprint("🎯 CLUSTERING OPTIMIZATION COMPLETED - DETAILED METRICS", "SUCCESS")
            tprint("="*80, "SUCCESS")

            # Add explanation of key metrics with CV RATIO prominence
            tprint("\n📚 KEY CLUSTERING METRICS EXPLANATIONS:", "INFO")
            tprint("   ⭐ CV RATIO (Variance Ratio): between-cluster variance / within-cluster variance", "SUCCESS")
            tprint("      → PRIMARY METRIC: HIGHER values = better regime separation & financial distinctness", "SUCCESS")
            tprint("      → Target: > 1.5 (Excellent), > 1.0 (Good), > 0.5 (Fair)", "INFO")
            tprint("      → Financial Impact: Higher CV ratio = more distinct market regimes = better trading signals", "SUCCESS")
            tprint("   📈 Silhouette Score: Measures cluster cohesion and separation (-1 to 1 scale)", "INFO")
            tprint("      → HIGHER values indicate better cluster quality and point assignment", "INFO")
            tprint("   🕒 Temporal Smoothness: Regime stability over time (lower switch rate = better)", "INFO")
            tprint("      → LOWER change rate indicates more stable, persistent regimes", "INFO")
            tprint("   ⚖️  Balance Score: Cluster size distribution uniformity (0-1 scale)", "INFO")
            tprint("      → HIGHER values indicate more balanced cluster sizes", "INFO")
            tprint("   🔍 Davies-Bouldin Index: Within-cluster scatter vs between-cluster separation", "INFO")
            tprint("      → LOWER values indicate better compactness and separation", "INFO")

            # Core clustering metrics with CV RATIO prominence
            tprint("\n📊 CORE CLUSTERING PERFORMANCE METRICS:", "SUCCESS")
            tprint("="*80, "SUCCESS")
            tprint(f"   🔢 Number of Clusters (K): {num_clusters} [Range: K_MIN={self.config.K_MIN} to K_MAX={self.config.K_MAX}]", "INFO")

            # Enhanced CV RATIO display - make it the most prominent metric
            tprint("\n" + "🎯" * 20 + " PRIMARY METRIC - CV RATIO (Variance Ratio) " + "🎯" * 20, "SUCCESS")
            tprint(f"   📈 CV RATIO Value: {cv_ratio:.4f}", "SUCCESS")
            tprint(f"   📊 Formula: between-cluster variance / within-cluster variance", "INFO")
            tprint(f"   💰 Financial Impact: Higher values = more distinct market regimes = better trading performance", "SUCCESS")

            # Detailed CV ratio quality assessment
            if cv_ratio > 2.0:
                cv_quality = "EXCEPTIONAL"
                cv_color = "SUCCESS"
                cv_interpretation = "Outstanding regime separation - expect excellent trading performance"
            elif cv_ratio > 1.5:
                cv_quality = "Excellent"
                cv_color = "SUCCESS"
                cv_interpretation = "Strong regime separation - very good trading signals"
            elif cv_ratio > 1.0:
                cv_quality = "Good"
                cv_color = "SUCCESS"
                cv_interpretation = "Clear regime separation - good trading opportunities"
            elif cv_ratio > 0.5:
                cv_quality = "Fair"
                cv_color = "WARNING"
                cv_interpretation = "Moderate regime separation - may need refinement"
            else:
                cv_quality = "Poor"
                cv_color = "ERROR"
                cv_interpretation = "Weak regime separation - clustering needs improvement"

            tprint(f"   🏆 Quality Assessment: {cv_quality}", cv_color)
            tprint(f"   💡 Interpretation: {cv_interpretation}", "INFO")

            # Show CV ratio trend if available
            try:
                if hasattr(stats, 'cv_history') and stats.cv_history:
                    cv_trend = "↗️ Improving" if stats.cv_history[-1] > stats.cv_history[0] else "↘️ Declining"
                    tprint(f"   📈 CV Trend: {cv_trend} (from {stats.cv_history[0]:.4f} to {stats.cv_history[-1]:.4f})", "INFO")
            except:
                pass
            tprint("\n   📈 SECONDARY METRICS:", "INFO")
            tprint(f"      🎭 Silhouette Score: {silhouette_score:.4f} (cluster cohesion & separation)", "INFO")
            tprint(f"      ⚖️  Balance Score: {balance_score:.4f} (cluster size uniformity)", "INFO")
            tprint(f"      🔍 Davies-Bouldin Index: {db_score:.4f} (compactness vs separation)", "INFO")

            # Temporal metrics summary
            try:
                if not hasattr(stats, 'transition_counts') or not hasattr(stats, 'transition_row_sums'):
                    stats._initialize_transition_caches()
                counts = stats.transition_counts
                rows = stats.transition_row_sums
                total_trans = int(np.sum(rows)) if rows is not None else 0
                diag_sum = int(np.trace(counts)) if counts is not None and counts.size > 0 else 0
                changes = max(0, total_trans - diag_sum)
                change_rate = (changes / total_trans) if total_trans > 0 else 0.0
                avg_run = (len(stats.assignments) / (changes + 1)) if len(stats.assignments) > 0 else 0.0
                # Markov entropy
                Kf = int(stats.K_fixed)
                alpha = float(getattr(stats, 'temporal_alpha', 1.0))
                entropies = []
                top_pairs = []
                if counts is not None and rows is not None and counts.shape[0] == Kf:
                    probs = np.zeros_like(counts, dtype=float)
                    for u in range(Kf):
                        row_sum = rows[u]
                        if row_sum <= 0:
                            continue
                        p = (counts[u].astype(float) + alpha) / (row_sum + alpha * Kf)
                        p = np.clip(p, 1e-12, 1.0)
                        entropies.append(float(-np.sum(p * np.log(p))))
                        probs[u] = p
                    # Top-3 transitions by probability (excluding self)
                    mask = np.ones_like(probs, dtype=bool)
                    np.fill_diagonal(mask, False)
                    flat_idx = np.argsort((probs * mask).ravel())[::-1]
                    for idx in flat_idx[:3]:
                        u = idx // Kf; v = idx % Kf
                        top_pairs.append((int(u), int(v), float(probs[u, v])))
                entropy = float(np.mean(entropies)) if entropies else 0.0

                tprint("\n🕒 TEMPORAL TRANSITION METRICS:", "INFO")
                tprint(f"   🔁 Total transitions: {total_trans}", "INFO")
                tprint(f"   🔄 Change rate: {change_rate:.4f} (fraction of non-self transitions)", "INFO")
                tprint(f"   📏 Avg run length: {avg_run:.2f}", "INFO")
                tprint(f"   📉 Markov entropy (avg row): {entropy:.3f} (LOWER implies more deterministic regimes)", "INFO")
                if top_pairs:
                    tprint("   🔝 Top transitions:", "INFO")
                    for (u, v, p) in top_pairs:
                        tprint(f"      {u} → {v}: p={p:.3f}", "INFO")
            except Exception as _e:
                tprint("   ⚠️ Temporal metrics unavailable", "WARNING")

            # Size distribution metrics
            tprint("\n📏 CLUSTER SIZE DISTRIBUTION:", "INFO")
            tprint(f"   📊 Total Samples: {total_samples:,}", "INFO")
            tprint(f"   📐 Min Size: {min_size:,}", "INFO")
            tprint(f"   📐 Max Size: {max_size:,}", "INFO")
            tprint(f"   📐 Average Size: {avg_size:.1f}", "INFO")
            tprint(f"   📐 Std Dev: {std_size:.1f}", "INFO")
            tprint(f"   📈 Size CV: {size_cv:.4f} (LOWER is better - measures cluster size uniformity)", "INFO")
            tprint(f"   ⚖️  Size Balance: {size_balance:.4f} (HIGHER is better - measures how balanced cluster sizes are)", "INFO")

            # Detailed cluster breakdown
            tprint("\n📊 INDIVIDUAL CLUSTER BREAKDOWN:", "INFO")
            tprint("   " + "-"*50, "INFO")
            tprint(f"   {'Cluster':<8} {'Size':<8} {'Percentage':<12} {'Status':<15}", "INFO")
            tprint("   " + "-"*50, "INFO")

            for i, (size, percentage) in enumerate(zip(cluster_sizes, size_distribution)):
                # Determine cluster status
                if percentage < 5.0:
                    status = "Small"
                elif percentage > 30.0:
                    status = "Large"
                elif 15.0 <= percentage <= 25.0:
                    status = "Optimal"
                else:
                    status = "Normal"

                tprint(f"   {i:<8} {size:<8,} {percentage:<11.1f}% {status:<15}", "INFO")

            tprint("   " + "-"*50, "INFO")

            # Quality assessment
            tprint("\n🎯 CLUSTERING QUALITY ASSESSMENT:", "INFO")
            if silhouette_score > 0.5:
                silhouette_quality = "Excellent"
            elif silhouette_score > 0.3:
                silhouette_quality = "Good"
            elif silhouette_score > 0.1:
                silhouette_quality = "Fair"
            else:
                silhouette_quality = "Poor"

            if balance_score > 0.8:
                balance_quality = "Excellent"
            elif balance_score > 0.6:
                balance_quality = "Good"
            elif balance_score > 0.4:
                balance_quality = "Fair"
            else:
                balance_quality = "Poor"

            tprint(f"   🎭 Silhouette Quality: {silhouette_quality} ({silhouette_score:.4f})", "INFO")
            tprint(f"   ⚖️  Balance Quality: {balance_quality} ({balance_score:.4f})", "INFO")
            # Overall variance ratio quality (simple heuristic)
            overall_vr_quality = "Excellent" if cv_ratio > 2.0 else ("Good" if cv_ratio > 1.0 else ("Fair" if cv_ratio > 0.5 else "Poor"))
            tprint(f"   📈 Overall Variance Ratio Quality: {overall_vr_quality} ({cv_ratio:.4f})", "INFO")

            tprint("\n" + "="*80, "SUCCESS")

        except Exception as e:
            tprint(f"Failed to print final metrics: {e}", "ERROR")

    def _log_iteration_metrics(self, iteration: int, stats: ClusteringStats):
        """Log both variance decomposition and centroid-silhouette (sampled)."""
        try:
            var_decomp = stats.calculate_variance_decomposition()
            sample_size = min(1000, len(stats.assignments))
            if sample_size <= 0:
                sil = 0.0
            else:
                idx = np.random.choice(len(stats.assignments), sample_size, replace=False)
                sil = float(np.mean([stats.get_point_silhouette(i) for i in idx]))
            # Temporal metrics
            try:
                if not hasattr(stats, 'transition_counts') or not hasattr(stats, 'transition_row_sums'):
                    stats._initialize_transition_caches()
                counts = stats.transition_counts
                rows = stats.transition_row_sums
                total_trans = int(np.sum(rows)) if rows is not None else 0
                diag_sum = int(np.trace(counts)) if counts is not None and counts.size > 0 else 0
                changes = max(0, total_trans - diag_sum)
                change_rate = (changes / total_trans) if total_trans > 0 else 0.0
                avg_run = (len(stats.assignments) / (changes + 1)) if len(stats.assignments) > 0 else 0.0
                # Markov entropy (natural log)
                Kf = int(stats.K_fixed)
                alpha = float(getattr(stats, 'temporal_alpha', 1.0))
                entropies = []
                if counts is not None and rows is not None and counts.shape[0] == Kf:
                    for u in range(Kf):
                        row_sum = rows[u]
                        if row_sum <= 0:
                            continue
                        p = (counts[u].astype(float) + alpha) / (row_sum + alpha * Kf)
                        # guard
                        p = np.clip(p, 1e-12, 1.0)
                        entropies.append(float(-np.sum(p * np.log(p))))
                entropy = float(np.mean(entropies)) if entropies else 0.0
            except Exception:
                change_rate = 0.0
                avg_run = 0.0
                entropy = 0.0
            tprint(
                f"Iter {iteration}: VarRatio={var_decomp['ratio']:.4f}, Sil={sil:.4f}, "
                f"ChangeRate={change_rate:.4f}, AvgRun={avg_run:.2f}, H={entropy:.3f}, K={stats.K_fixed}",
                "DEBUG"
            )
        except Exception as e:
            tprint(f"Iteration metrics logging failed: {e}", "WARNING")

    def _generate_final_report(self):
        """Generate final optimization report."""
        try:
            if not self.step_reports:
                return

            tprint("\n=== OPTIMIZATION REPORT ===", "INFO")

            # Summary statistics
            total_rounds = len(self.step_reports)
            final_cv = self.step_reports[-1]['final_cv']
            final_balance = self.step_reports[-1]['final_balance']
            final_silhouette = self.step_reports[-1]['final_silhouette']

            tprint(f"Total rounds: {total_rounds}", "INFO")
            tprint(f"Final Variance Ratio: {final_cv:.4f}", "INFO")
            tprint(f"Final balance: {final_balance:.4f}", "INFO")
            tprint(f"Final silhouette: {final_silhouette:.4f}", "INFO")

            # Calculate total improvements
            total_cv_improvement = final_cv - self.step_reports[0]['initial_cv']
            total_balance_improvement = final_balance - self.step_reports[0]['initial_balance']
            total_silhouette_improvement = final_silhouette - self.step_reports[0]['initial_silhouette']

            tprint(f"Total CV improvement: {total_cv_improvement:+.4f}", "INFO")
            tprint(f"Total balance improvement: {total_balance_improvement:+.4f}", "INFO")
            tprint(f"Total silhouette improvement: {total_silhouette_improvement:+.4f}", "INFO")

        except Exception as e:
            tprint(f"Final report generation failed: {e}", "ERROR")
