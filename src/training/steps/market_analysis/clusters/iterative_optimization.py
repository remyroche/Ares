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
        self.features = features
        self.assignments = assignments
        self.n_samples, self.n_features = features.shape
        self.n_clusters = len(np.unique(assignments))
        
        # Per-cluster statistics
        self.cluster_sizes = np.zeros(self.n_clusters)
        self.centroids = np.zeros((self.n_clusters, self.n_features))
        self.wcss_per_cluster = np.zeros(self.n_clusters)
        
        # Global statistics
        self.global_mean = np.mean(features, axis=0)
        self.total_wcss = 0.0
        self.total_bcss = 0.0
        
        # Initialize all statistics
        self._update_all_stats()
        
    def _update_all_stats(self):
        """Update all clustering statistics."""
        unique_clusters = np.unique(self.assignments)
        
        for cluster in unique_clusters:
            mask = self.assignments == cluster
            cluster_features = self.features[mask]
            
            if len(cluster_features) > 0:
                self.cluster_sizes[cluster] = len(cluster_features)
                self.centroids[cluster] = np.mean(cluster_features, axis=0)
                self.wcss_per_cluster[cluster] = np.sum((cluster_features - self.centroids[cluster]) ** 2)
        
        self.total_wcss = np.sum(self.wcss_per_cluster)
        self.total_bcss = calculate_bcss_incremental(
            self.centroids, self.global_mean, self.cluster_sizes
        )
    
    def get_cv_ratio(self) -> float:
        """Get current CV ratio (BCSS/WCSS)."""
        if self.total_wcss == 0:
            return 0.0
        return self.total_bcss / self.total_wcss
    
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
        """Calculate delta for moving a point from one cluster to another."""
        if from_cluster == to_cluster:
            return {'total': 0.0, 'cv': 0.0, 'balance': 0.0, 'silhouette': 0.0, 'temporal': 0.0}
        
        point = self.features[point_idx]
        
        # Calculate new cluster sizes
        new_sizes = self.cluster_sizes.copy()
        new_sizes[from_cluster] -= 1
        new_sizes[to_cluster] += 1
        
        # Calculate new centroids
        new_centroids = self.centroids.copy()
        if new_sizes[from_cluster] > 0:
            new_centroids[from_cluster] = (
                (self.centroids[from_cluster] * self.cluster_sizes[from_cluster] - point) / 
                new_sizes[from_cluster]
            )
        else:
            new_centroids[from_cluster] = np.zeros(self.n_features)
        
        new_centroids[to_cluster] = (
            (self.centroids[to_cluster] * self.cluster_sizes[to_cluster] + point) / 
            new_sizes[to_cluster]
        )
        
        # Calculate WCSS delta
        old_wcss_from = self.wcss_per_cluster[from_cluster]
        old_wcss_to = self.wcss_per_cluster[to_cluster]
        
        new_wcss_from = np.sum((self.features[self.assignments == from_cluster] - new_centroids[from_cluster]) ** 2)
        new_wcss_to = np.sum((self.features[self.assignments == to_cluster] - new_centroids[to_cluster]) ** 2)
        
        wcss_delta = (new_wcss_from - old_wcss_from) + (new_wcss_to - old_wcss_to)
        
        # Calculate BCSS delta
        old_bcss = self.total_bcss
        new_bcss = calculate_bcss_incremental(new_centroids, self.global_mean, new_sizes)
        bcss_delta = new_bcss - old_bcss
        
        # Calculate CV delta
        old_cv = self.get_cv_ratio()
        new_wcss = self.total_wcss + wcss_delta
        new_cv = new_bcss / new_wcss if new_wcss > 0 else 0.0
        cv_delta = new_cv - old_cv
        
        # Calculate balance delta
        old_balance = self.get_balance_score()
        target_size = self.n_samples / self.n_clusters
        old_penalty_from = (self.cluster_sizes[from_cluster] / self.n_samples - 1.0 / self.n_clusters) ** 2
        old_penalty_to = (self.cluster_sizes[to_cluster] / self.n_samples - 1.0 / self.n_clusters) ** 2
        new_penalty_from = (new_sizes[from_cluster] / self.n_samples - 1.0 / self.n_clusters) ** 2
        new_penalty_to = (new_sizes[to_cluster] / self.n_samples - 1.0 / self.n_clusters) ** 2
        
        balance_delta = -((new_penalty_from + new_penalty_to) - (old_penalty_from + old_penalty_to))
        
        # Calculate local silhouette delta (simplified)
        d1_old = np.linalg.norm(point - self.centroids[from_cluster])
        d2_old = min([np.linalg.norm(point - self.centroids[c]) for c in range(self.n_clusters) if c != from_cluster])
        d1_new = np.linalg.norm(point - new_centroids[to_cluster])
        d2_new = min([np.linalg.norm(point - new_centroids[c]) for c in range(self.n_clusters) if c != to_cluster])
        
        s_old = 1.0 - d1_old / d2_old if d2_old > 0 else 0.0
        s_new = 1.0 - d1_new / d2_new if d2_new > 0 else 0.0
        silhouette_delta = s_new - s_old
        
        # Temporal delta (placeholder - would need temporal data)
        temporal_delta = 0.0
        
        return {
            'total': cv_delta + balance_delta + silhouette_delta + temporal_delta,
            'cv': cv_delta,
            'balance': balance_delta,
            'silhouette': silhouette_delta,
            'temporal': temporal_delta
        }
    
    def apply_move(self, point_idx: int, from_cluster: int, to_cluster: int):
        """Apply a move and update statistics incrementally."""
        if from_cluster == to_cluster:
            return
        
        point = self.features[point_idx]
        
        # Update assignments
        self.assignments[point_idx] = to_cluster
        
        # Update cluster sizes
        self.cluster_sizes[from_cluster] -= 1
        self.cluster_sizes[to_cluster] += 1
        
        # Update centroids
        if self.cluster_sizes[from_cluster] > 0:
            self.centroids[from_cluster] = (
                (self.centroids[from_cluster] * (self.cluster_sizes[from_cluster] + 1) - point) / 
                self.cluster_sizes[from_cluster]
            )
        else:
            self.centroids[from_cluster] = np.zeros(self.n_features)
        
        self.centroids[to_cluster] = (
            (self.centroids[to_cluster] * (self.cluster_sizes[to_cluster] - 1) + point) / 
            self.cluster_sizes[to_cluster]
        )
        
        # Recalculate WCSS and BCSS
        self._update_all_stats()


class IterativeOptimization:
    """Advanced 3-step iterative clustering optimization."""
    
    def __init__(self, verbose: bool = True):
        """Initialize the iterative optimization with advanced parameters."""
        self.verbose = verbose
        self.logger = get_logger('IterativeOptimization')
        
        # Optimization parameters
        self.max_rounds = 50
        self.tolerance = 1e-5
        
        # Step 1: Local frontier parameters
        self.frontier_fraction = 0.25  # q = 25%
        self.knn_size = 10
        self.neighbor_consensus_threshold = 0.65
        self.local_threshold = 0.0
        self.local_churn_cap = 0.02  # 2% of N
        self.hysteresis_rounds = 2
        
        # Step 2: Global reallocation parameters
        self.beta = 0.15  # Capacity buffer
        self.global_threshold = 0.0
        self.global_churn_cap = 0.08  # 8% of N
        self.min_cluster_size = 25
        
        # Step 3: Break large clusters parameters
        self.size_factor_threshold = 1.5  # ρ ≥ 1.5
        self.split_quality_threshold = 0.005  # ΔJ₀ = 0.5%
        self.alpha = 1.0  # Size-aware penalty
        self.max_new_clusters_per_round = 3
        
        # Objective function weights (finance-first)
        self.w_cv = 0.55
        self.w_bal = 0.15
        self.w_sil = 0.10
        self.w_temp = 0.20
        self.lambda_switch = 1e-4  # Small penalty to reduce churn
        
        # Monitoring
        self.step_reports = []
        
    async def execute_optimization_loop(
        self, 
        context: ClusteringContext, 
        config: Any, 
        max_iterations: int = 100
    ) -> ClusteringContext:
        """Execute the advanced 3-step iterative optimization loop."""
        try:
            tprint("Starting advanced 3-step iterative optimization...", "INFO")
            
            features = context.optimized_features
            current_assignments = context.initial_assignments.copy()
            current_k = len(np.unique(current_assignments))
            
            # Initialize clustering statistics
            stats = ClusteringStats(features, current_assignments)
            
            # Track convergence
            convergence_count = 0
            last_total_delta = float('inf')
            
            for round_num in range(self.max_rounds):
                tprint(f"\n=== Round {round_num + 1}/{self.max_rounds} ===", "INFO")
                
                # Report initial metrics
                initial_cv = stats.get_cv_ratio()
                initial_balance = stats.get_balance_score()
                initial_silhouette = self._calculate_silhouette_score(features, current_assignments)
                
                tprint(f"Initial metrics - CV: {initial_cv:.4f}, Balance: {initial_balance:.4f}, Silhouette: {initial_silhouette:.4f}", "INFO")
                
                round_delta = 0.0
                
                # Step 1: Local frontier moves
                local_moves = await self._step1_local_frontier_moves(features, stats)
                round_delta += local_moves
                
                # Step 2: Global reallocation
                global_moves = await self._step2_global_reallocation(features, stats)
                round_delta += global_moves
                
                # Step 3: Break large clusters
                split_moves = await self._step3_break_large_clusters(features, stats)
                round_delta += split_moves
                
                # Report final metrics
                final_cv = stats.get_cv_ratio()
                final_balance = stats.get_balance_score()
                final_silhouette = self._calculate_silhouette_score(features, current_assignments)
                
                tprint(f"Final metrics - CV: {final_cv:.4f}, Balance: {final_balance:.4f}, Silhouette: {final_silhouette:.4f}", "INFO")
                tprint(f"Round delta: {round_delta:.6f}", "INFO")
                
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
    
    async def _step1_local_frontier_moves(self, features: np.ndarray, stats: ClusteringStats) -> float:
        """Step 1: Local frontier moves focused on CV with balance/silhouette/temporal."""
        try:
            tprint("Step 1: Local frontier moves...", "INFO")
            
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
            
            for point_idx in boundary_points:
                if moves_made >= max_moves:
                    break
                
                current_cluster = stats.assignments[point_idx]
                
                # Find best alternative clusters
                best_alternatives = self._find_best_alternative_clusters(
                    features, stats, point_idx, current_cluster
                )
                
                for target_cluster, delta_info in best_alternatives:
                    if delta_info['total'] >= self.local_threshold:
                        # Check neighbor consensus
                        if indices is not None:
                            neighbor_consensus = self._calculate_neighbor_consensus(
                                indices[point_idx], stats.assignments, target_cluster
                            )
                            
                            if neighbor_consensus < self.neighbor_consensus_threshold:
                                continue
                        
                        # Check margin gain
                        margin_gain = self._calculate_margin_gain(
                            features, stats, point_idx, current_cluster, target_cluster
                        )
                        
                        if margin_gain >= 1e-3:  # Minimum margin gain
                            # Apply the move
                            stats.apply_move(point_idx, current_cluster, target_cluster)
                            total_delta += delta_info['total']
                            moves_made += 1
                            break  # Only one move per point per round
            
            tprint(f"Local frontier: {moves_made} moves, delta: {total_delta:.6f}", "INFO")
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
            
            # Score all possible moves
            move_candidates = []
            
            for point_idx in range(n_samples):
                current_cluster = stats.assignments[point_idx]
                
                # Check all other clusters as potential targets
                for target_cluster in range(stats.n_clusters):
                    if target_cluster == current_cluster:
                        continue
                    
                    # Check capacity constraints
                    if (stats.cluster_sizes[target_cluster] >= n_max or 
                        stats.cluster_sizes[current_cluster] <= n_min):
                        continue
                    
                    delta_info = stats.calculate_move_delta(point_idx, current_cluster, target_cluster)
                    
                    if delta_info['total'] >= self.global_threshold:
                        move_candidates.append({
                            'point_idx': point_idx,
                            'from_cluster': current_cluster,
                            'to_cluster': target_cluster,
                            'delta': delta_info['total'],
                            'delta_info': delta_info
                        })
            
            # Sort by delta (descending)
            move_candidates.sort(key=lambda x: x['delta'], reverse=True)
            
            # Apply moves with capacity constraints
            total_delta = 0.0
            moves_made = 0
            max_moves = int(n_samples * self.global_churn_cap)
            
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
            
            tprint(f"Global reallocation: {moves_made} moves, delta: {total_delta:.6f}", "INFO")
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
            # Use Numba-optimized boundary score calculation
            boundary_scores = calculate_boundary_scores_numba(features, stats.centroids, stats.assignments)
            
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
                    alternatives.append((target_cluster, delta_info))
            
            # Sort by total delta (descending)
            alternatives.sort(key=lambda x: x[1]['total'], reverse=True)
            
            # Return top 2 alternatives
            return alternatives[:2]
            
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
                
                # Check size factor
                size_factor = cluster_size / median_size if median_size > 0 else 1.0
                
                if size_factor >= self.size_factor_threshold:
                    # Check internal quality
                    cluster_mask = stats.assignments == cluster_id
                    cluster_features = features[cluster_mask]
                    
                    if len(cluster_features) > 10:  # Only split if cluster is large enough
                        # Calculate internal scatter
                        internal_scatter = np.mean([
                            np.sum((cluster_features[i] - stats.centroids[cluster_id]) ** 2)
                            for i in range(len(cluster_features))
                        ])
                        
                        # Check if internal quality is poor
                        if internal_scatter > np.percentile([
                            np.mean([np.sum((features[stats.assignments == c] - stats.centroids[c]) ** 2)
                                   for c in range(stats.n_clusters) if stats.cluster_sizes[c] > 0])
                        ], 75):  # Top quartile
                            split_candidates.append(cluster_id)
            
            return split_candidates
            
        except Exception as e:
            tprint(f"Cluster splitting identification failed: {e}", "ERROR")
            return []
    
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
    