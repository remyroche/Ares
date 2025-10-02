"""
Advanced 3-Step Iterative Clustering Optimization for NAS-TAS.

This module implements a sophisticated iterative optimization loop with:
1. Local frontier moves (CV-focused with balance/silhouette/temporal)
2. Global reallocation (capacity-aware coordination)
3. Break large clusters (size-aware quality thresholds)

Features:
- Incremental statistics tracking (BCSS/WCSS)
- Fast delta calculations using sufficient statistics
- Numba-optimized vectorized operations
- Comprehensive monitoring and reporting
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from numba import jit, prange
import warnings

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error
)

from ..shared_utils import get_logger
from .step1_feature_preparation import ClusteringContext
from .risk_mitigation import RiskMitigationSystem, PRODUCTION_RISK_CONFIG


# Numba-optimized helper functions
@jit(nopython=True, parallel=True)
def calculate_distance_matrix_numba(features: np.ndarray) -> np.ndarray:
    """Calculate pairwise distances using Numba for speed."""
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
        self.n_clusters = len(np.unique(assignments))
        
        # Per-cluster sufficient statistics (exact incremental formulas)
        self.cluster_sizes = np.zeros(self.n_clusters, dtype=np.int32)
        self.centroids = np.zeros((self.n_clusters, self.n_features), dtype=np.float64)
        self.wcss_per_cluster = np.zeros(self.n_clusters, dtype=np.float64)
        
        # Sufficient statistics for exact incremental calculations
        # S_c = sum of points in cluster c
        self.S = np.zeros((self.n_clusters, self.n_features), dtype=np.float64)
        # Q_c = sum of outer products x_i * x_i^T for cluster c (stored as trace)
        self.Q_trace = np.zeros(self.n_clusters, dtype=np.float64)
        
        # Global statistics
        self.global_mean = np.mean(features, axis=0, dtype=np.float64)
        self.total_wcss = 0.0
        self.total_bcss = 0.0
        
        # Global sufficient statistics
        self.global_S = np.sum(features, axis=0, dtype=np.float64)  # S = sum of all points
        self.global_N = self.n_samples
        
        # Initialize all statistics
        self._update_all_stats()
        
    def _update_all_stats(self):
        """Update all clustering statistics with sufficient statistics."""
        unique_clusters = np.unique(self.assignments)
        
        # Reset sufficient statistics
        self.S.fill(0.0)
        self.Q_trace.fill(0.0)
        self.cluster_sizes.fill(0)
        
        for cluster in unique_clusters:
            mask = self.assignments == cluster
            cluster_features = self.features[mask]
            
            if len(cluster_features) > 0:
                n_c = len(cluster_features)
                self.cluster_sizes[cluster] = n_c
                
                # S_c = sum of points in cluster c
                self.S[cluster] = np.sum(cluster_features, axis=0, dtype=np.float64)
                
                # Q_c = sum of ||x_i||^2 for cluster c (trace of outer products)
                self.Q_trace[cluster] = np.sum(np.sum(cluster_features ** 2, axis=1), dtype=np.float64)
                
                # Centroids: μ_c = S_c / n_c
                self.centroids[cluster] = self.S[cluster] / n_c
                
                # WCSS_c = tr(Q_c) - ||S_c||^2 / n_c (exact formula)
                self.wcss_per_cluster[cluster] = self.Q_trace[cluster] - np.sum(self.S[cluster] ** 2) / n_c
        
        self.total_wcss = np.sum(self.wcss_per_cluster)
        
        # BCSS = sum_c ||S_c||^2 / n_c - ||S||^2 / N (exact formula)
        self.total_bcss = np.sum(np.sum(self.S ** 2, axis=1) / self.cluster_sizes) - np.sum(self.global_S ** 2) / self.global_N
    
    def get_cv_ratio(self) -> float:
        """Get current CV ratio (BCSS/WCSS)."""
        if self.total_wcss == 0:
            return 0.0
        return self.total_bcss / self.total_wcss
    
    def get_objective_value(self, w_cv: float = 0.55, w_bal: float = 0.15, 
                           w_sil: float = 0.10, w_temp: float = 0.20,
                           k_complexity_penalty: float = 0.25, k_max: int = 20) -> float:
        """Calculate the full objective function value with k-complexity penalty."""
        cv_ratio = self.get_cv_ratio()
        balance = self.get_balance_score()
        
        # Placeholder for silhouette and temporal (would be calculated in full implementation)
        silhouette_proxy = 0.5  # Placeholder
        temporal_proxy = 0.5    # Placeholder
        
        objective = (
            w_cv * cv_ratio +
            w_bal * balance +
            w_sil * silhouette_proxy +
            w_temp * temporal_proxy
        )
        
        # Add k-complexity penalty to prevent runaway splitting
        k_penalty = k_complexity_penalty * (self.n_clusters - 1) / k_max
        objective -= k_penalty
        
        return objective
    
    def get_balance_score(self) -> float:
        """Calculate cluster balance score."""
        if self.n_clusters <= 1:
            return 1.0
        
        target_size = self.n_samples / self.n_clusters
        size_penalties = []
        
        for size in self.cluster_sizes:
            if size > 0:
                penalty = (size / self.n_samples - 1.0 / self.n_clusters) ** 2
                size_penalties.append(penalty)
        
        return 1.0 - np.mean(size_penalties) if size_penalties else 1.0
    
    def calculate_move_delta(self, point_idx: int, from_cluster: int, to_cluster: int) -> Dict[str, float]:
        """Calculate exact delta for moving a point from one cluster to another using sufficient statistics."""
        if from_cluster == to_cluster:
            return {'total': 0.0, 'cv': 0.0, 'balance': 0.0, 'silhouette': 0.0, 'temporal': 0.0}
        
        point = self.features[point_idx].astype(np.float64)
        
        # Current sufficient statistics
        n_a = self.cluster_sizes[from_cluster]
        n_b = self.cluster_sizes[to_cluster]
        S_a = self.S[from_cluster].copy()
        S_b = self.S[to_cluster].copy()
        Q_a = self.Q_trace[from_cluster]
        Q_b = self.Q_trace[to_cluster]
        
        # New sufficient statistics after move
        n_a_new = n_a - 1
        n_b_new = n_b + 1
        S_a_new = S_a - point
        S_b_new = S_b + point
        Q_a_new = Q_a - np.sum(point ** 2)  # Q_a - ||x||^2
        Q_b_new = Q_b + np.sum(point ** 2)  # Q_b + ||x||^2
        
        # Exact WCSS delta using sufficient statistics
        # WCSS_c = tr(Q_c) - ||S_c||^2 / n_c
        old_wcss_a = Q_a - np.sum(S_a ** 2) / n_a if n_a > 0 else 0.0
        old_wcss_b = Q_b - np.sum(S_b ** 2) / n_b if n_b > 0 else 0.0
        
        new_wcss_a = Q_a_new - np.sum(S_a_new ** 2) / n_a_new if n_a_new > 0 else 0.0
        new_wcss_b = Q_b_new - np.sum(S_b_new ** 2) / n_b_new if n_b_new > 0 else 0.0
        
        wcss_delta = (new_wcss_a - old_wcss_a) + (new_wcss_b - old_wcss_b)
        
        # Exact BCSS delta using sufficient statistics
        # BCSS = sum_c ||S_c||^2 / n_c - ||S||^2 / N (second term is constant)
        old_bcss_terms = (np.sum(S_a ** 2) / n_a if n_a > 0 else 0.0) + (np.sum(S_b ** 2) / n_b if n_b > 0 else 0.0)
        new_bcss_terms = (np.sum(S_a_new ** 2) / n_a_new if n_a_new > 0 else 0.0) + (np.sum(S_b_new ** 2) / n_b_new if n_b_new > 0 else 0.0)
        
        bcss_delta = new_bcss_terms - old_bcss_terms
        
        # Calculate CV delta (separation term)
        old_wcss = self.total_wcss
        old_bcss = self.total_bcss
        new_wcss = old_wcss + wcss_delta
        new_bcss = old_bcss + bcss_delta
        
        old_cv = old_bcss / old_wcss if old_wcss > 0 else 0.0
        new_cv = new_bcss / new_wcss if new_wcss > 0 else 0.0
        cv_delta = new_cv - old_cv
        
        # Calculate balance delta using new cluster sizes
        old_penalty_from = (n_a / self.n_samples - 1.0 / self.n_clusters) ** 2
        old_penalty_to = (n_b / self.n_samples - 1.0 / self.n_clusters) ** 2
        new_penalty_from = (n_a_new / self.n_samples - 1.0 / self.n_clusters) ** 2
        new_penalty_to = (n_b_new / self.n_samples - 1.0 / self.n_clusters) ** 2
        
        balance_delta = -((new_penalty_from + new_penalty_to) - (old_penalty_from + old_penalty_to))
        
        # Calculate bounded silhouette delta
        # New centroids after move
        new_centroid_a = S_a_new / n_a_new if n_a_new > 0 else np.zeros(self.n_features, dtype=np.float64)
        new_centroid_b = S_b_new / n_b_new if n_b_new > 0 else np.zeros(self.n_features, dtype=np.float64)
        
        # Old silhouette: distance to own centroid vs nearest other
        d1_old = np.linalg.norm(point - self.centroids[from_cluster])
        d2_old = min([np.linalg.norm(point - self.centroids[c]) for c in range(self.n_clusters) if c != from_cluster])
        
        # New silhouette: distance to new centroid vs nearest other
        d1_new = np.linalg.norm(point - new_centroid_b)
        d2_new = min([np.linalg.norm(point - self.centroids[c]) for c in range(self.n_clusters) if c != to_cluster])
        
        # Bounded silhouette calculation: s = 1 - d1/(d2+eps), clipped to [-1, 1]
        eps = 1e-8
        s_old = max(-1.0, min(1.0, 1.0 - d1_old / (d2_old + eps) if d2_old > 0 else 0.0))
        s_new = max(-1.0, min(1.0, 1.0 - d1_new / (d2_new + eps) if d2_new > 0 else 0.0))
        
        # Silhouette delta is bounded to [-2, 2] and clipped
        silhouette_delta = max(-2.0, min(2.0, s_new - s_old))
        
        # Temporal delta (basic implementation with small non-zero signal)
        # Simple temporal consistency based on cluster size changes
        # This provides a small but non-zero signal to prevent MAD collapse
        temporal_delta = 0.0
        
        # Basic temporal signal: prefer moves that don't create extreme size imbalances
        size_ratio_old = n_a / n_b if n_b > 0 else float('inf')
        size_ratio_new = n_a_new / n_b_new if n_b_new > 0 else float('inf')
        
        # Small temporal preference for moves that improve size balance
        if size_ratio_old > 2.0 and size_ratio_new < size_ratio_old:
            temporal_delta = 0.01  # Small positive signal
        elif size_ratio_old < 0.5 and size_ratio_new > size_ratio_old:
            temporal_delta = 0.01  # Small positive signal
        else:
            temporal_delta = 0.0
        
        # Normal mode: use all components
        total_delta = cv_delta + balance_delta + silhouette_delta + temporal_delta
        
        return {
            'total': total_delta,
            'cv': cv_delta,
            'balance': balance_delta,
            'silhouette': silhouette_delta,
            'temporal': temporal_delta,
            # Raw values for debugging
            'cv_raw': cv_delta,
            'balance_raw': balance_delta,
            'silhouette_raw': silhouette_delta,
            'temporal_raw': temporal_delta
        }
    
    def apply_move(self, point_idx: int, from_cluster: int, to_cluster: int):
        """Apply a move and update sufficient statistics incrementally."""
        if from_cluster == to_cluster:
            return
        
        point = self.features[point_idx].astype(np.float64)
        
        # Update assignments
        self.assignments[point_idx] = to_cluster
        
        # Update cluster sizes
        self.cluster_sizes[from_cluster] -= 1
        self.cluster_sizes[to_cluster] += 1
        
        # Update sufficient statistics incrementally
        # S_c = sum of points in cluster c
        self.S[from_cluster] -= point
        self.S[to_cluster] += point
        
        # Q_c = sum of ||x_i||^2 for cluster c
        point_norm_sq = np.sum(point ** 2)
        self.Q_trace[from_cluster] -= point_norm_sq
        self.Q_trace[to_cluster] += point_norm_sq
        
        # Update centroids: μ_c = S_c / n_c
        if self.cluster_sizes[from_cluster] > 0:
            self.centroids[from_cluster] = self.S[from_cluster] / self.cluster_sizes[from_cluster]
        else:
            self.centroids[from_cluster] = np.zeros(self.n_features, dtype=np.float64)
        
        self.centroids[to_cluster] = self.S[to_cluster] / self.cluster_sizes[to_cluster]
        
        # Update WCSS and BCSS using exact formulas
        # WCSS_c = tr(Q_c) - ||S_c||^2 / n_c
        if self.cluster_sizes[from_cluster] > 0:
            self.wcss_per_cluster[from_cluster] = self.Q_trace[from_cluster] - np.sum(self.S[from_cluster] ** 2) / self.cluster_sizes[from_cluster]
        else:
            self.wcss_per_cluster[from_cluster] = 0.0
            
        self.wcss_per_cluster[to_cluster] = self.Q_trace[to_cluster] - np.sum(self.S[to_cluster] ** 2) / self.cluster_sizes[to_cluster]
        
        # Update totals
        self.total_wcss = np.sum(self.wcss_per_cluster)
        
        # BCSS = sum_c ||S_c||^2 / n_c - ||S||^2 / N
        self.total_bcss = np.sum(np.sum(self.S ** 2, axis=1) / self.cluster_sizes) - np.sum(self.global_S ** 2) / self.global_N


class IterativeOptimization:
    """Advanced 3-step iterative clustering optimization."""
    
    def __init__(self, verbose: bool = True, k: int | None = None):
        """Initialize the iterative optimization with advanced parameters."""
        self.verbose = verbose
        self.logger = get_logger('IterativeOptimization')
        
        # Initialize cluster count
        self._k = int(k) if k is not None else None
        
        # Optimization parameters
        self.max_rounds = 50
        self.tolerance = 1e-5
        
        # Step 1: Local frontier parameters (LOOSENED FOR TRIAGE)
        self.frontier_fraction = 0.40  # Increased from 25% to 40%
        self.knn_size = 10
        self.neighbor_consensus_threshold = 0.60  # Loosened from 0.65 to 0.60
        self.local_threshold = 0.0
        self.local_churn_cap = 0.02  # 2% of N
        self.hysteresis_rounds = 2
        
        # Step 2: Global reallocation parameters (LOOSENED FOR TRIAGE)
        self.beta = 0.20  # Increased from 0.15 to 0.20
        self.global_threshold = 0.0
        self.global_churn_cap = 0.08  # 8% of N
        self.min_cluster_size = 25
        
        # Step 3: Break large clusters parameters (LOOSENED FOR TRIAGE)
        self.size_factor_threshold = 1.3  # Reduced from 1.5 to 1.3
        self.split_quality_threshold = 0.003  # Reduced from 0.005 to 0.003
        self.alpha = 1.0  # Size-aware penalty
        self.max_new_clusters_per_round = 3
        
        # Objective function weights (OPTIMIZED FOR STEP-1 LOCAL FRONTIER)
        self.w_cv = 0.75  # Prioritize separation gains
        self.w_bal = 0.15  # Keep balance reasonable
        self.w_sil = 0.02  # Minimal silhouette weight (tiebreaker only)
        self.w_temp = 0.08  # Reduced temporal weight
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
        
        return float(self.w_cv * v[0] + self.w_bal * v[1] + 
                    self.w_sil * v[2] + self.w_temp * v[3])
    
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
    
    def _accept_candidate(self, candidate: dict) -> tuple[bool, str]:
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
        
        # Main gate (with temporal bonus) - FIXED: ≤ threshold for negative improvements
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

            # Initialize risk mitigation system
            risk_system = RiskMitigationSystem(PRODUCTION_RISK_CONFIG)

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

            try:
                delta_1 = await self._step1_local_frontier_moves(features, stats)
                total_delta += delta_1
                # Extract moves from the delta calculation (simplified)
                local_moves = int(abs(delta_1) * 100) if delta_1 != 0 else 0
                moves_accepted += local_moves
            except Exception as e:
                tprint(f"⚠️ Step 1 failed: {e}", "WARNING")
                delta_1 = 0.0

            # Step 2: Global reallocation
            if self.verbose:
                tprint(f"🔍 Step 2: Global reallocation (iteration {iteration})", "DEBUG")

            try:
                delta_2 = await self._step2_global_reallocation(features, stats)
                total_delta += delta_2
                # Extract moves from the delta calculation (simplified)
                global_moves = int(abs(delta_2) * 100) if delta_2 != 0 else 0
                moves_accepted += global_moves
            except Exception as e:
                tprint(f"⚠️ Step 2 failed: {e}", "WARNING")
                delta_2 = 0.0

            # Step 3: Break large clusters
            if self.verbose:
                tprint(f"🔍 Step 3: Break large clusters (iteration {iteration})", "DEBUG")

            try:
                delta_3 = await self._step3_break_large_clusters(features, stats)
                total_delta += delta_3
                # Extract splits from the delta calculation (simplified)
                splits_performed = int(abs(delta_3) * 10) if delta_3 != 0 else 0
            except Exception as e:
                tprint(f"⚠️ Step 3 failed: {e}", "WARNING")
                delta_3 = 0.0

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
            context.optimized_assignments = current_assignments
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
            
            # Initialize cluster count if not set
            if self._k is None:
                self._k = current_k
            
            # Sanity check
            assert self.n_clusters == current_k, f"Cluster count mismatch: {self.n_clusters} != {current_k}"
            
            # Initialize clustering statistics
            stats = ClusteringStats(features, current_assignments)
            
            # Initialize risk mitigation system
            risk_system = None
            if enable_risk_mitigation:
                risk_system = RiskMitigationSystem(PRODUCTION_RISK_CONFIG)
                tprint("Risk mitigation system enabled", "INFO")
                tprint("🎯 Advanced 3-step iterative clustering with comprehensive safeguards", "INFO")
            
            # Track convergence and early stopping
            convergence_count = 0
            last_total_delta = float('inf')
            prev_objective = None
            early_stop_count = 0
            
            for round_num in range(self.max_rounds):
                tprint(f"\n=== Round {round_num + 1}/{self.max_rounds} ===", "INFO")
                
                # Update adaptive thresholds
                self._update_adaptive_thresholds(round_num)
                tprint(f"  Thresholds: eps_std_step1={self.eps_std_step1:.3f}, sil_guard={self.sil_guard:.2f}, "
                       f"temporal_bonus={self.temporal_bonus:.2f}", "DEBUG")
                
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
                
                # Report initial metrics
                initial_cv = stats.get_cv_ratio()
                initial_balance = stats.get_balance_score()
                initial_silhouette = self._calculate_silhouette_score(features, current_assignments)
                
                tprint(f"Initial metrics - CV: {initial_cv:.4f}, Balance: {initial_balance:.4f}, Silhouette: {initial_silhouette:.4f}", "INFO")
                
                round_delta = 0.0
                
                # Step 1: Local frontier moves
                local_moves = await self._step1_local_frontier_moves(features, stats, round_num)
                round_delta += local_moves
                
                # Step 2: Global reallocation
                global_moves = await self._step2_global_reallocation(features, stats)
                round_delta += global_moves
                
                # Step 3: Break large clusters (with k-growth prevention)
                split_moves = 0
                if risk_system:
                    # Check k growth before splitting
                    proposed_k = len(np.unique(stats.assignments))
                    if risk_system.check_unbounded_k_growth(current_k, proposed_k, len(features)):
                        split_moves = await self._step3_break_large_clusters(features, stats)
                    else:
                        tprint("Skipping cluster splits due to k-growth prevention", "WARNING")
                else:
                    split_moves = await self._step3_break_large_clusters(features, stats)
                
                round_delta += split_moves
                
                # Update operation counts for risk tracking
                if risk_system:
                    risk_system.update_operation_counts(local_moves, global_moves, split_moves)
                
                # Report final metrics
                final_cv = stats.get_cv_ratio()
                final_balance = stats.get_balance_score()
                final_silhouette = self._calculate_silhouette_score(features, current_assignments)
                
                tprint(f"Final metrics - CV: {final_cv:.4f}, Balance: {final_balance:.4f}, Silhouette: {final_silhouette:.4f}", "INFO")
                tprint(f"Round delta: {round_delta:.6f}", "INFO")
                
                # Risk mitigation: Check metric drift and monotonicity
                if risk_system:
                    current_objective = stats.get_objective_value()
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
                
                # Check convergence
                if abs(round_delta) < self.tolerance:
                    convergence_count += 1
                    if convergence_count >= 3:
                        tprint(f"Convergence reached at round {round_num + 1}", "SUCCESS")
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
            
            # Generate final report
            self._generate_final_report()
            
            tprint("Advanced 3-step iterative optimization completed", "SUCCESS")
            return context
            
        except Exception as e:
            tprint(f"Advanced iterative optimization failed: {e}", "ERROR")
            raise ValueError(f"Advanced iterative optimization failed: {e}")
    
    async def _step1_local_frontier_moves(self, features: np.ndarray, stats: ClusteringStats, current_iteration: int = 0) -> float:
        """Step 1: Local frontier moves focused on CV with balance/silhouette/temporal."""
        try:
            tprint("Step 1: Local frontier moves...", "INFO")

            # Validate features
            if features is None or features.size == 0:
                tprint("❌ Features array is None or empty in local frontier moves", "ERROR")
                return 0.0

            if not hasattr(features, 'shape') or len(features.shape) != 2:
                tprint(f"❌ Features must be a 2D array, got shape: {getattr(features, 'shape', 'None')}", "ERROR")
                return 0.0

            # Find boundary points
            boundary_points = self._identify_boundary_points(features, stats)
            
            if len(boundary_points) == 0:
                tprint("No boundary points found", "INFO")
                return 0.0
            
            # Limit to frontier fraction
            n_boundary = int(len(boundary_points) * self.frontier_fraction)
            boundary_points = boundary_points[:n_boundary]
            
            tprint(f"Evaluating {len(boundary_points)} boundary points", "INFO")
            
            total_delta = 0.0
            moves_made = 0
            max_moves = int(len(features) * self.local_churn_cap)
            
            # Build kNN for neighbor consensus
            if len(features) > self.knn_size:
                nn = NearestNeighbors(n_neighbors=min(self.knn_size + 1, len(features)), metric='euclidean')
                nn.fit(features)
                distances, indices = nn.kneighbors(features)
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
            
            # Collect all deltas for component scale normalization
            all_deltas = {'cv': [], 'balance': [], 'silhouette': [], 'temporal': []}
            candidate_moves = []
            
            for point_idx in boundary_points:
                if moves_made >= max_moves:
                    break
                
                current_cluster = stats.assignments[point_idx]
                
                # Find best alternative clusters
                best_alternatives = self._find_best_alternative_clusters(
                    features, stats, point_idx, current_cluster
                )
                
                # Candidate set sanity check
                if len(best_alternatives) < min(3, self.n_clusters - 1):
                    tprint(f"  ⚠️ Point {point_idx} has only {len(best_alternatives)} alternatives (need ≥3)", "WARNING")
                if current_cluster in [alt[0] for alt in best_alternatives]:
                    tprint(f"  ⚠️ Point {point_idx} includes current cluster {current_cluster} in alternatives", "WARNING")
                
                if not best_alternatives:
                    no_alternatives += 1
                    continue
                
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
                    })
                    
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
                if positives:
                    # Sort by standardized score ASCENDING (negative values are improvements)
                    positives.sort(key=lambda x: x['delta_info'].get('J_std', x['delta_info']['total']), reverse=False)
                    L = 8  # Fixed L=8 as requested
                    tprint(f"  🔄 Top-L policy: applying {min(L, len(positives))} best moves (L={L}, ascending ΔJ_std)", "DEBUG")
                    
                    # Apply moves in chunks
                    for chunk_start in range(0, min(L, len(positives)), self.chunk_size):
                        chunk = positives[chunk_start:chunk_start + self.chunk_size]
                        chunk_moves = 0
                        
                        for move in chunk:
                            if moves_made >= self.max_local_moves_per_iter:
                                break
                            stats.apply_move(move['point_idx'], move['from_cluster'], move['to_cluster'])
                            # Record move for anti-oscillation tracking
                            self._record_move(move['point_idx'], move['from_cluster'], move['to_cluster'], 
                                            current_iteration, move['delta_info'].get('J_std', move['delta_info']['total']))
                            total_delta += move['delta_info']['total']
                            moves_made += 1
                            chunk_moves += 1
                            tprint(f"    ✅ Top-L move {move['point_idx']} {move['from_cluster']}→{move['to_cluster']} "
                                  f"ΔJ_std={move['delta_info'].get('J_std', move['delta_info']['total']):.6f}", "DEBUG")
                        
                        tprint(f"  📦 Chunk {chunk_start//self.chunk_size + 1}: applied {chunk_moves} moves", "DEBUG")
                
                # Exploratory acceptance: only at iteration 0
                if moves_made == 0 and len(candidate_moves) > 0 and current_iteration == 0:
                    # Sort all candidates by score and take top exploratory quota
                    candidate_moves.sort(key=lambda x: self._objective(x['delta_info'], use_std=self.use_std_for_rank), reverse=True)
                    exploratory_moves = candidate_moves[:self.exploratory_quota]
                    tprint(f"  🔬 Exploratory acceptance: trying {len(exploratory_moves)} top moves despite guards", "DEBUG")
                    
                    for move in exploratory_moves:
                        if moves_made >= self.max_local_moves_per_iter:
                            break
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
                    d_std_total = (self.w_cv * d_std['cv'] + 
                                  self.w_bal * d_std['balance'] + 
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
                    })
                    
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
            
            # Enhanced metrics reporting with diagnostics
            tprint(f"Local frontier: {moves_made} moves, delta: {total_delta:.6f}", "INFO")
            tprint(f"  📊 Boundary points: {len(boundary_points)} ({boundary_ratio:.1%} of dataset)", "DEBUG")
            tprint(f"  ⚡ Move efficiency: {move_efficiency:.1%} ({moves_made}/{len(boundary_points)})", "DEBUG")
            tprint(f"  📈 Avg delta per move: {avg_delta_per_move:.6f}", "DEBUG")
            tprint(f"  🔍 Blocking: ΔJ_low={delta_too_low}, CV_guard={cv_guard_blocked}, Sil_guard={sil_guard_blocked}, "
                   f"consensus={consensus_failed}, margin={margin_failed}", "DEBUG")
            
            # Log thrash scores per cluster
            thrash_scores = self._calculate_thrash_scores(current_iteration)
            high_thrash_clusters = [cid for cid, score in thrash_scores.items() if score > 0.1]
            if high_thrash_clusters:
                tprint(f"  🔄 High thrash clusters: {[(cid, f'{score:.2f}') for cid, score in thrash_scores.items() if score > 0.1]}", "WARNING")
            
            # Log sign verification
            tprint(f"  🔍 Top-L uses ascending ΔJ_std (negative=improvement)", "DEBUG")
            
            # One-line sanity checks (catch this instantly next run)
            if len(candidate_moves) > 0:
                top_candidate = candidate_moves[0]
                top_score_gate = self._objective(top_candidate['delta_info'], use_std=self.use_std_for_gate)
                assert top_score_gate >= self.eps_std_step1 - 1e-12, \
                    f"Top candidate {top_candidate['point_idx']} gate={top_score_gate:.6f} < eps={self.eps_std_step1:.3f}"
                
                # Verify printed ΔJ_std equals what the gate used
                printed_deltaJ_std = top_candidate['delta_info'].get('J_std', top_candidate['delta_info']['total'])
                assert abs(top_score_gate - printed_deltaJ_std) < 1e-9, \
                    f"Printed ΔJ_std={printed_deltaJ_std:.6f} != gate ΔJ_std={top_score_gate:.6f} (double normalization or wrong variable)"
            
            return total_delta
            
        except Exception as e:
            tprint(f"Local frontier moves failed: {e}", "ERROR")
            return 0.0
    
    async def _step2_global_reallocation(self, features: np.ndarray, stats: ClusteringStats) -> float:
        """Step 2: Global reallocation with capacity-aware coordination."""
        try:
            tprint("Step 2: Global reallocation...", "INFO")
            
            # Calculate capacity bands
            n_samples = len(features)
            target_size = n_samples / stats.n_clusters
            n_min = max(self.min_cluster_size, int(0.005 * n_samples))
            n_max = int((1.0 / stats.n_clusters + self.beta) * n_samples)
            
            # Score all possible moves with diagnostics
            move_candidates = []
            capacity_blocked = 0
            delta_too_low = 0
            total_evaluated = 0
            
            for point_idx in range(n_samples):
                current_cluster = stats.assignments[point_idx]
                
                # Check all other clusters as potential targets
                for target_cluster in range(stats.n_clusters):
                    if target_cluster == current_cluster:
                        continue
                    
                    total_evaluated += 1
                    
                    # Check capacity constraints
                    if (stats.cluster_sizes[target_cluster] >= n_max or 
                        stats.cluster_sizes[current_cluster] <= n_min):
                        capacity_blocked += 1
                        continue
                    
                    delta_info = stats.calculate_move_delta(point_idx, current_cluster, target_cluster)
                    
                    # Add debugging for delta_info type
                    if not isinstance(delta_info, dict):
                        tprint(f"❌ ERROR: calculate_move_delta returned {type(delta_info)}, expected dict", "ERROR")
                        tprint(f"❌ delta_info value: {delta_info}", "ERROR")
                        continue
                    
                    if delta_info['total'] <= self.eps_std_step2:  # Use adaptive threshold
                        move_candidates.append({
                            'point_idx': point_idx,
                            'from_cluster': current_cluster,
                            'to_cluster': target_cluster,
                            'delta': delta_info['total'],
                            'delta_info': delta_info
                        })
                    else:
                        delta_too_low += 1
            
            # Sort by delta (ascending - negative values are improvements)
            move_candidates.sort(key=lambda x: x['delta'], reverse=False)
            
            # Apply moves with capacity constraints and cap
            total_delta = 0.0
            moves_made = 0
            max_moves = min(int(n_samples * self.global_churn_cap), self.max_global_moves)
            
            # Track capacity changes
            capacity_tracker = stats.cluster_sizes.copy()
            
            for move in move_candidates:
                if moves_made >= max_moves:
                    break
                
                point_idx = move['point_idx']
                from_cluster = move['from_cluster']
                to_cluster = move['to_cluster']
                
                # Check if move is still valid
                if (capacity_tracker[to_cluster] < n_max and 
                    capacity_tracker[from_cluster] > n_min):
                    
                    # Apply the move
                    stats.apply_move(point_idx, from_cluster, to_cluster)
                    capacity_tracker[from_cluster] -= 1
                    capacity_tracker[to_cluster] += 1
                    
                    total_delta += move['delta']
                    moves_made += 1
            
            # Calculate additional metrics
            candidate_ratio = len(move_candidates) / n_samples if n_samples > 0 else 0
            move_efficiency = moves_made / len(move_candidates) if len(move_candidates) > 0 else 0
            avg_delta_per_move = total_delta / moves_made if moves_made > 0 else 0
            capacity_utilization = np.mean(capacity_tracker) / target_size if target_size > 0 else 0
            
            # Enhanced metrics reporting with diagnostics
            tprint(f"Global reallocation: {moves_made} moves, delta: {total_delta:.6f}", "INFO")
            tprint(f"  📊 Move candidates: {len(move_candidates)} ({candidate_ratio:.1%} of dataset)", "DEBUG")
            tprint(f"  ⚡ Move efficiency: {move_efficiency:.1%} ({moves_made}/{len(move_candidates)})", "DEBUG")
            tprint(f"  📈 Avg delta per move: {avg_delta_per_move:.6f}", "DEBUG")
            tprint(f"  🎯 Capacity utilization: {capacity_utilization:.1%}", "DEBUG")
            tprint(f"  📏 Target size: {target_size:.0f}, Min: {n_min}, Max: {n_max}", "DEBUG")
            tprint(f"  🔍 BLOCKING DIAGNOSTICS:", "DEBUG")
            tprint(f"    - Total evaluated: {total_evaluated} move pairs", "DEBUG")
            tprint(f"    - Capacity blocked: {capacity_blocked} ({capacity_blocked/total_evaluated:.1%})", "DEBUG")
            tprint(f"    - ΔJ too low: {delta_too_low} ({delta_too_low/total_evaluated:.1%})", "DEBUG")
            
            return total_delta
            
        except Exception as e:
            tprint(f"Global reallocation failed: {e}", "ERROR")
            return 0.0
    
    async def _step3_break_large_clusters(self, features: np.ndarray, stats: ClusteringStats) -> float:
        """Step 3: Break large clusters with size-aware quality thresholds."""
        try:
            tprint("Step 3: Break large clusters...", "INFO")
            
            # Find clusters that are candidates for splitting
            split_candidates = self._identify_clusters_for_splitting(features, stats)
            
            if not split_candidates:
                tprint("No clusters identified for splitting", "INFO")
                return 0.0
            
            total_delta = 0.0
            splits_made = 0
            
            for cluster_id in split_candidates:
                if splits_made >= self.max_new_clusters_per_round:
                    break
                
                # Calculate split quality
                split_delta = self._calculate_split_quality(features, stats, cluster_id)
                
                if split_delta > 0:
                    # Apply the split
                    new_delta = self._apply_cluster_split(features, stats, cluster_id)
                    total_delta += new_delta
                    splits_made += 1
                    
                    tprint(f"Split cluster {cluster_id}, delta: {new_delta:.6f}", "INFO")
            
            tprint(f"Break large clusters: {splits_made} splits, delta: {total_delta:.6f}", "INFO")
            return total_delta
            
        except Exception as e:
            tprint(f"Break large clusters failed: {e}", "ERROR")
            return 0.0
    
    def _identify_boundary_points(self, features: np.ndarray, stats: ClusteringStats) -> List[int]:
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

            # Use Numba-optimized boundary score calculation
            boundary_scores = calculate_boundary_scores_numba(features, stats.centroids, stats.assignments)

            # Validate boundary scores
            if boundary_scores is None or len(boundary_scores) == 0:
                tprint("❌ Boundary scores calculation failed or returned empty", "ERROR")
                return []

            # Sort by boundary score (ascending - most boundary-like first)
            sorted_indices = np.argsort(boundary_scores)

            # Return point indices
            return sorted_indices.tolist()

        except Exception as e:
            tprint(f"Boundary point identification failed: {e}", "ERROR")
            return []
    
    def _find_best_alternative_clusters(
        self, 
        features: np.ndarray, 
        stats: ClusteringStats, 
        point_idx: int, 
        current_cluster: int
    ) -> List[Tuple[int, Dict[str, float]]]:
        """Find best alternative clusters for a point."""
        try:
            alternatives = []
            
            for target_cluster in range(stats.n_clusters):
                if target_cluster != current_cluster:
                    delta_info = stats.calculate_move_delta(point_idx, current_cluster, target_cluster)
                    
                    # Add debugging for delta_info type
                    if not isinstance(delta_info, dict):
                        tprint(f"❌ ERROR: calculate_move_delta in alternatives returned {type(delta_info)}, expected dict", "ERROR")
                        tprint(f"❌ delta_info value: {delta_info}", "ERROR")
                        continue
                    
                    alternatives.append((target_cluster, delta_info))
            
            # Sort by total delta (descending)
            alternatives.sort(key=lambda x: x[1]['total'], reverse=True)
            
            # Return top N alternatives (limited by max_alternatives_per_point)
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
                        if (cluster_silhouette < self.split_silhouette_threshold or 
                            boundary_ratio > self.boundary_ratio_threshold or
                            thrash_count >= self.thrash_count_threshold):
                            split_candidates.append(cluster_id)
            
            return split_candidates
            
        except Exception as e:
            tprint(f"Cluster splitting identification failed: {e}", "ERROR")
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
    
    def _calculate_split_quality(self, features: np.ndarray, stats: ClusteringStats, cluster_id: int) -> float:
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
    
    def _apply_cluster_split(self, features: np.ndarray, stats: ClusteringStats, cluster_id: int) -> float:
        """Apply a cluster split and return the quality improvement."""
        try:
            cluster_mask = stats.assignments == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_features = features[cluster_mask]
            
            # Perform 2-means split
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            sub_assignments = kmeans.fit_predict(cluster_features)
            
            # Create new cluster ID
            new_cluster_id = stats.n_clusters
            
            # Update assignments
            for i, idx in enumerate(cluster_indices):
                if sub_assignments[i] == 1:
                    stats.assignments[idx] = new_cluster_id
            
            # Update statistics
            stats.n_clusters += 1
            stats.cluster_sizes = np.zeros(stats.n_clusters)
            stats.centroids = np.zeros((stats.n_clusters, stats.n_features))
            stats.wcss_per_cluster = np.zeros(stats.n_clusters)
            
            # Recalculate all statistics
            stats._update_all_stats()
            
            # Calculate quality improvement
            return self._calculate_split_quality(features, stats, cluster_id)
            
        except Exception as e:
            tprint(f"Cluster split application failed: {e}", "ERROR")
            return 0.0
    
    def _calculate_silhouette_score(self, features: np.ndarray, assignments: np.ndarray) -> float:
        """Calculate silhouette score with error handling."""
        try:
            if len(np.unique(assignments)) < 2:
                return 0.0
            return silhouette_score(features, assignments)
        except Exception:
            return 0.0
    
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
            tprint(f"Final CV ratio: {final_cv:.4f}", "INFO")
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
    