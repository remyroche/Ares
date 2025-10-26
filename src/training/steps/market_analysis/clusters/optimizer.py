"""
Clustering Optimizer for HDBSCAN Clustering.

This module implements advanced optimization logic including:
- ΔJ objective calculation
- Incremental update functions with Numba JIT compilation
- Neighbor consensus mechanisms
- Hysteresis and churn caps
- Splitting thresholds and k-penalty logic
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from numba import jit, prange
import time
from datetime import datetime

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error
)
from src.utils.common_operations import (
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer
)
from src.utils.common_utilities import (
    calculate_data_quality_metrics, safe_dataframe_operation,
    validate_dataframe_columns, create_summary_statistics
)
from src.utils.math_validation import (
    safe_divide, validate_finite, safe_log, safe_sqrt
)
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations

from ..shared_utils import get_logger
from .iterative_optimization import ClusteringStats

@dataclass
class OptimizerConfig:
    """Configuration for clustering optimizer."""
    # Core optimization parameters
    max_iterations: int = 100
    convergence_tolerance: float = 1e-5
    enable_early_stopping: bool = True

    # ΔJ calculation weights (finance-first approach with enhanced cohesion)
    w_cv: float = 0.50      # CV ratio weight (reduced to make room for cohesion)
    w_bal: float = 0.15     # Balance weight
    w_sil: float = 0.10     # Silhouette weight
    w_temp: float = 0.15    # Temporal consistency weight (reduced)
    w_frag: float = 0.10    # Fragmentation penalty weight (NEW)

    # Enhanced size constraints
    min_cluster_ratio: float = 0.05      # 5% minimum cluster size
    max_cluster_ratio: float = 0.35      # 35% maximum cluster size (more flexible)
    target_cluster_ratio: float = 0.20   # 20% target cluster size
    size_balance_weight: float = 0.20    # Weight for size balance in objective

    # Cohesion and fragmentation parameters
    fragmentation_penalty_threshold: float = 0.5  # Fragmentation score threshold
    cohesion_reward_threshold: float = 0.7        # Cohesion score threshold
    enable_cohesion_optimization: bool = True     # Enable cohesion-based optimization

    # Neighbor consensus parameters
    neighbor_consensus_threshold: float = 0.65
    knn_size: int = 10
    enable_neighbor_consensus: bool = True

    # Hysteresis and churn control
    hysteresis_rounds: int = 2
    local_churn_cap: float = 0.02    # 2% of N
    global_churn_cap: float = 0.08   # 8% of N

    # Splitting parameters
    size_factor_threshold: float = 1.5    # ρ ≥ 1.5
    split_quality_threshold: float = 0.005  # ΔJ₀ = 0.5%
    alpha_penalty: float = 1.0           # Size-aware penalty
    max_new_clusters_per_round: int = 3

    # K-complexity management
    k_complexity_penalty: float = 0.25
    k_max: int = 20
    enable_k_growth_control: bool = True

    # Performance settings
    use_numba_optimization: bool = True
    parallel_processing: bool = False
    memory_optimization: bool = True

class ClusteringOptimizer:
    """
    Advanced clustering optimizer implementing ΔJ objective calculation
    and sophisticated optimization strategies.
    """

    def __init__(self, config: Optional[OptimizerConfig] = None):
        """Initialize the clustering optimizer."""
        self.config = config or OptimizerConfig()
        self.logger = get_logger('ClusteringOptimizer')

        # Optimization state
        self.optimization_history = []
        self.convergence_info = {}
        self.move_history = []

        # Hardware service integration
        try:
            from .hardware_service import HardwareService
            self.hardware_service = HardwareService(verbose=False)  # Less verbose for optimizer
            self.hardware_integration_enabled = True
        except ImportError:
            self.hardware_service = None
            self.hardware_integration_enabled = False

        # Initialize hardware optimizations
        self.hardware_manager = None
        self.memory_optimizer = None
        self.matrix_ops = None

        # Initialize M1 hardware optimizations if available
        if self.config.parallel_processing or self.config.memory_optimization:
            self._initialize_hardware_optimizations()

        # Performance tracking
        self.performance_metrics = {
            "total_optimization_time": 0.0,
            "iteration_count": 0,
            "moves_evaluated": 0,
            "moves_accepted": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "hardware_accelerations": 0,
            "memory_optimizations": 0
        }

        # Numba-compiled functions cache
        self._numba_functions_cache = {}

    def _initialize_hardware_optimizations(self) -> None:
        """Initialize hardware optimizations for clustering optimization."""
        try:
            # Initialize matrix operations with hardware acceleration
            self.matrix_ops = UnifiedMatrixOperations()

            # Get hardware managers for optimization
            self.hardware_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()

            if self.hardware_manager or self.memory_optimizer:
                tprint("🖥️ Hardware optimizations initialized for clustering optimizer", "INFO")
            else:
                tprint("⚠️ Hardware optimizations not available for optimizer, using CPU fallback", "WARNING")

        except Exception as e:
            tprint(f"❌ Hardware initialization for optimizer failed: {e}", "ERROR")
            self.hardware_manager = None
            self.memory_optimizer = None
            self.matrix_ops = None

    async def execute_optimization(
        self,
        context: Any,
        config: Any
    ) -> Any:
        """
        Execute the optimization process.

        Args:
            context: Clustering context with features and assignments
            config: Configuration object

        Returns:
            Updated context with optimized clustering
        """
        try:
            tprint("🚀 Starting advanced clustering optimization...", "INFO")
            start_time = time.time()

            # Apply hardware optimizations if available
            if self.hardware_integration_enabled and self.hardware_service:
                try:
                    # Apply matrix operation optimizations
                    matrix_optimization = self.hardware_service.optimize_matrix_operations()
                    if matrix_optimization.get('success', False):
                        tprint("🧠 Matrix operation optimizations applied", "SUCCESS")
                except Exception as e:
                    tprint(f"⚠️ Hardware matrix optimization failed: {e}", "WARNING")

            # Extract data from context
            features = getattr(context, 'optimized_features', None)
            assignments = getattr(context, 'optimized_assignments', None)

            if features is None or assignments is None:
                tprint("⚠️ No features or assignments found in context", "WARNING")
                return context

            # Apply memory optimization to features if hardware service is available
            if self.hardware_integration_enabled and self.hardware_service:
                try:
                    features, optimization_info = self.hardware_service.optimize_memory(features)
                    if optimization_info.get("hardware_optimization_used", False):
                        self.performance_metrics["memory_optimizations"] += 1
                        tprint("🧠 Memory optimization applied to features", "SUCCESS")
                except Exception as e:
                    tprint(f"⚠️ Memory optimization failed: {e}", "WARNING")

            # Initialize clustering statistics
            stats = ClusteringStats(features, assignments)

            # Execute optimization loop
            optimized_stats = await self._optimization_loop(features, stats, config)

            # Update context with results
            context.optimized_assignments = optimized_stats.assignments
            context.final_k = optimized_stats.n_clusters
            context.optimization_metrics = self._get_optimization_summary()

            # Record total time
            self.performance_metrics["total_optimization_time"] = time.time() - start_time

            tprint(f"✅ Optimization completed in {self.performance_metrics['total_optimization_time']:.2f}s", "SUCCESS")
            return context

        except Exception as e:
            tprint(f"❌ Optimization execution failed: {e}", "ERROR")
            raise ValueError(f"Optimization execution failed: {e}")

    async def _optimization_loop(
        self,
        features: np.ndarray,
        stats: ClusteringStats,
        config: Any
    ) -> ClusteringStats:
        """Main optimization loop implementing the advanced strategy."""

        try:
            n_samples = len(features)
            current_k = stats.n_clusters

            # Track convergence
            convergence_count = 0
            last_total_delta = float('inf')

            # Build kNN structure for neighbor consensus (if enabled)
            nn_structure = None
            if self.config.enable_neighbor_consensus and n_samples > self.config.knn_size:
                nn_structure = await self._build_knn_structure(features)

            for iteration in range(self.config.max_iterations):
                tprint(f"\n🔄 Optimization Iteration {iteration + 1}/{self.config.max_iterations}", "INFO")

                # Calculate current objective value with enhanced metrics
                current_objective = self._calculate_enhanced_objective(
                    stats, features, config
                )

                tprint(f"Current objective: {current_objective:.6f}", "INFO")

                iteration_delta = 0.0

                # Step 1: Evaluate local moves with neighbor consensus
                local_delta = await self._evaluate_local_moves(
                    features, stats, nn_structure, config
                )
                iteration_delta += local_delta

                # Step 2: Evaluate global reallocation
                global_delta = await self._evaluate_global_reallocation(
                    features, stats, config
                )
                iteration_delta += global_delta

                # Step 3: Evaluate cluster splitting (with k-growth control)
                if self.config.enable_k_growth_control:
                    # Check if k-growth is acceptable
                    proposed_k = len(np.unique(stats.assignments))
                    if self._check_k_growth_acceptable(current_k, proposed_k, n_samples):
                        split_delta = await self._evaluate_cluster_splitting(
                            features, stats, config
                        )
                        iteration_delta += split_delta
                    else:
                        tprint("⏸️ Skipping cluster splitting due to k-growth control", "INFO")
                else:
                    split_delta = await self._evaluate_cluster_splitting(
                        features, stats, config
                    )
                    iteration_delta += split_delta

                # Record iteration results
                self.optimization_history.append({
                    'iteration': iteration + 1,
                    'objective': current_objective,
                    'delta': iteration_delta,
                    'n_clusters': stats.n_clusters,
                    'local_delta': local_delta,
                    'global_delta': global_delta,
                    'split_delta': split_delta if 'split_delta' in locals() else 0.0
                })

                # Check convergence
                if abs(iteration_delta) < self.config.convergence_tolerance:
                    convergence_count += 1
                    if convergence_count >= 3:  # Converged for 3 consecutive iterations
                        tprint(f"🎯 Convergence reached at iteration {iteration + 1}", "SUCCESS")
                        self.convergence_info = {
                            'converged': True,
                            'final_iteration': iteration + 1,
                            'final_objective': current_objective,
                            'total_improvement': current_objective - self.optimization_history[0]['objective'] if self.optimization_history else 0
                        }
                        break
                else:
                    convergence_count = 0

                # Early stopping check
                if (self.config.enable_early_stopping and
                    iteration > 10 and
                    abs(iteration_delta) < self.config.convergence_tolerance * 10):
                    tprint(f"🛑 Early stopping at iteration {iteration + 1}", "INFO")
                    break

                last_total_delta = iteration_delta

            # Update performance metrics
            self.performance_metrics["iteration_count"] = len(self.optimization_history)

            return stats

        except Exception as e:
            tprint(f"❌ Optimization loop failed: {e}", "ERROR")
            raise

    async def _evaluate_local_moves(
        self,
        features: np.ndarray,
        stats: ClusteringStats,
        nn_structure: Optional[Any],
        config: Any
    ) -> float:
        """Evaluate local moves with neighbor consensus."""

        try:
            tprint("🔍 Evaluating local moves...", "INFO")

            # Find boundary points for local optimization
            boundary_points = self._identify_boundary_points(features, stats)

            if len(boundary_points) == 0:
                tprint("No boundary points found for local moves", "INFO")
                return 0.0

            # Limit to churn cap
            max_moves = int(len(features) * self.config.local_churn_cap)
            points_to_evaluate = boundary_points[:max_moves]

            total_delta = 0.0
            moves_accepted = 0

            for point_idx in points_to_evaluate:
                current_cluster = stats.assignments[point_idx]

                # Find best alternative clusters
                best_moves = self._find_best_moves_for_point(
                    features, stats, point_idx, current_cluster
                )

                for target_cluster, move_info in best_moves:
                    # Check neighbor consensus if enabled
                    if self.config.enable_neighbor_consensus and nn_structure is not None:
                        consensus_score = self._calculate_neighbor_consensus(
                            nn_structure, stats.assignments, point_idx, target_cluster
                        )

                        if consensus_score < self.config.neighbor_consensus_threshold:
                            continue

                    # Check if move improves objective
                    if move_info['delta'] > 0:
                        # Apply the move
                        stats.apply_move(point_idx, current_cluster, target_cluster)
                        total_delta += move_info['delta']
                        moves_accepted += 1

                        # Record move
                        self.move_history.append({
                            'type': 'local',
                            'point_idx': point_idx,
                            'from_cluster': current_cluster,
                            'to_cluster': target_cluster,
                            'delta': move_info['delta']
                        })

                        break  # Only one move per point per iteration

            tprint(f"Local moves: {moves_accepted} accepted, ΔJ = {total_delta:.6f}", "INFO")
            return total_delta

        except Exception as e:
            tprint(f"❌ Local moves evaluation failed: {e}", "ERROR")
            return 0.0

    async def _evaluate_global_reallocation(
        self,
        features: np.ndarray,
        stats: ClusteringStats,
        config: Any
    ) -> float:
        """Evaluate global reallocation moves."""

        try:
            tprint("🌍 Evaluating global reallocation...", "INFO")

            # Calculate capacity constraints
            n_samples = len(features)
            target_size = n_samples / stats.n_clusters
            n_min = max(25, int(0.005 * n_samples))  # 0.5% of N
            n_max = int((1.0 / stats.n_clusters + 0.15) * n_samples)  # 15% buffer

            # Find all possible moves
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

                    # Calculate move delta
                    move_delta = stats.calculate_move_delta(point_idx, current_cluster, target_cluster)

                    if move_delta['total'] > 0:  # Only consider improving moves
                        move_candidates.append({
                            'point_idx': point_idx,
                            'from_cluster': current_cluster,
                            'to_cluster': target_cluster,
                            'delta': move_delta['total'],
                            'delta_info': move_delta
                        })

            # Sort by delta (descending)
            move_candidates.sort(key=lambda x: x['delta'], reverse=True)

            # Apply moves with churn control
            total_delta = 0.0
            moves_accepted = 0
            max_moves = int(n_samples * self.config.global_churn_cap)

            # Track capacity changes
            capacity_tracker = stats.cluster_sizes.copy()

            for move in move_candidates:
                if moves_accepted >= max_moves:
                    break

                point_idx = move['point_idx']
                from_cluster = move['from_cluster']
                to_cluster = move['to_cluster']

                # Check if move is still valid with current capacities
                if (capacity_tracker[to_cluster] < n_max and
                    capacity_tracker[from_cluster] > n_min):

                    # Apply the move
                    stats.apply_move(point_idx, from_cluster, to_cluster)
                    capacity_tracker[from_cluster] -= 1
                    capacity_tracker[to_cluster] += 1

                    total_delta += move['delta']
                    moves_accepted += 1

                    # Record move
                    self.move_history.append({
                        'type': 'global',
                        'point_idx': point_idx,
                        'from_cluster': from_cluster,
                        'to_cluster': to_cluster,
                        'delta': move['delta']
                    })

            tprint(f"Global reallocation: {moves_accepted} accepted, ΔJ = {total_delta:.6f}", "INFO")
            return total_delta

        except Exception as e:
            tprint(f"❌ Global reallocation evaluation failed: {e}", "ERROR")
            return 0.0

    async def _evaluate_cluster_splitting(
        self,
        features: np.ndarray,
        stats: ClusteringStats,
        config: Any
    ) -> float:
        """Evaluate cluster splitting opportunities."""

        try:
            tprint("✂️ Evaluating cluster splitting...", "INFO")

            # Identify clusters for potential splitting
            split_candidates = self._identify_split_candidates(features, stats)

            if not split_candidates:
                tprint("No clusters identified for splitting", "INFO")
                return 0.0

            total_delta = 0.0
            splits_performed = 0

            for cluster_id in split_candidates:
                if splits_performed >= self.config.max_new_clusters_per_round:
                    break

                # Calculate split quality
                split_delta = self._calculate_split_delta(features, stats, cluster_id)

                if split_delta > self.config.split_quality_threshold:
                    # Perform the split
                    actual_delta = self._perform_cluster_split(features, stats, cluster_id)
                    total_delta += actual_delta
                    splits_performed += 1

                    tprint(f"Split cluster {cluster_id}, ΔJ = {actual_delta:.6f}", "INFO")

                    # Record split
                    self.move_history.append({
                        'type': 'split',
                        'cluster_id': cluster_id,
                        'new_clusters': 2,
                        'delta': actual_delta
                    })

            tprint(f"Cluster splitting: {splits_performed} splits, ΔJ = {total_delta:.6f}", "INFO")
            return total_delta

        except Exception as e:
            tprint(f"❌ Cluster splitting evaluation failed: {e}", "ERROR")
            return 0.0

    def _identify_boundary_points(self, features: np.ndarray, stats: ClusteringStats) -> List[int]:
        """Identify points on cluster boundaries for local optimization."""
        try:
            # Use optimized boundary detection
            if self.config.use_numba_optimization:
                try:
                    # Import the numba function from iterative_optimization
                    from .iterative_optimization import calculate_boundary_scores_numba
                    boundary_scores = calculate_boundary_scores_numba(features, stats.centroids, stats.assignments)
                except ImportError:
                    # Fallback to non-numba implementation
                    boundary_scores = self._calculate_boundary_scores_fallback(features, stats.centroids, stats.assignments)
            else:
                boundary_scores = self._calculate_boundary_scores_fallback(features, stats.centroids, stats.assignments)

            # Sort by boundary score (ascending - most boundary-like first)
            sorted_indices = np.argsort(boundary_scores)

            return sorted_indices.tolist()

        except Exception as e:
            tprint(f"❌ Boundary point identification failed: {e}", "ERROR")
            return []

    def _calculate_boundary_scores_fallback(
        self,
        features: np.ndarray,
        centroids: np.ndarray,
        assignments: np.ndarray
    ) -> np.ndarray:
        """Fallback boundary score calculation without Numba."""
        n = features.shape[0]
        boundary_scores = np.zeros(n)

        for i in range(n):
            point = features[i]
            current_cluster = int(assignments[i])

            # Calculate distances to all centroids
            distances = np.array([np.sqrt(np.sum((point - centroids[c]) ** 2)) for c in range(len(centroids))])

            # Find closest and second closest
            sorted_indices = np.argsort(distances)
            d1 = distances[sorted_indices[0]]  # Distance to own cluster
            d2 = distances[sorted_indices[1]]  # Distance to nearest other cluster

            # Boundary score: difference between d1 and d2
            boundary_scores[i] = d1 - d2

        return boundary_scores

    def _find_best_moves_for_point(
        self,
        features: np.ndarray,
        stats: ClusteringStats,
        point_idx: int,
        current_cluster: int
    ) -> List[Tuple[int, Dict[str, float]]]:
        """Find best alternative clusters for a point."""
        try:
            best_moves = []

            for target_cluster in range(stats.n_clusters):
                if target_cluster != current_cluster:
                    delta_info = stats.calculate_move_delta(point_idx, current_cluster, target_cluster)
                    best_moves.append((target_cluster, delta_info))

            # Sort by total delta (descending)
            best_moves.sort(key=lambda x: x[1]['total'], reverse=True)

            # Return top 2 moves
            return best_moves[:2]

        except Exception as e:
            tprint(f"❌ Best moves finding failed: {e}", "ERROR")
            return []

    async def _build_knn_structure(self, features: np.ndarray) -> Any:
        """Build k-nearest neighbors structure for consensus calculation."""
        try:
            from sklearn.neighbors import NearestNeighbors

            # Use appropriate k size
            k = min(self.config.knn_size + 1, len(features))

            nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
            nn.fit(features)

            return nn

        except Exception as e:
            tprint(f"❌ KNN structure building failed: {e}", "ERROR")
            return None

    def _calculate_neighbor_consensus(
        self,
        nn_structure: Any,
        assignments: np.ndarray,
        point_idx: int,
        target_cluster: int
    ) -> float:
        """Calculate neighbor consensus for a potential move."""
        try:
            # Get k-nearest neighbors (including self)
            distances, indices = nn_structure.kneighbors([nn_structure._fit_X[point_idx]])

            # Exclude self from neighbors
            neighbor_indices = indices[0][1:]  # Skip first (self)
            neighbor_assignments = assignments[neighbor_indices]

            # Count how many neighbors are in target cluster
            consensus_count = np.sum(neighbor_assignments == target_cluster)
            total_neighbors = len(neighbor_assignments)

            return consensus_count / total_neighbors if total_neighbors > 0 else 0.0

        except Exception as e:
            tprint(f"❌ Neighbor consensus calculation failed: {e}", "ERROR")
            return 0.0

    def _identify_split_candidates(self, features: np.ndarray, stats: ClusteringStats) -> List[int]:
        """Identify clusters that are candidates for splitting."""
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

                if size_factor >= self.config.size_factor_threshold:
                    # Check internal quality
                    cluster_mask = stats.assignments == cluster_id
                    cluster_features = features[cluster_mask]

                    if len(cluster_features) > 20:  # Need enough points to split meaningfully
                        # Calculate internal scatter
                        cluster_center = stats.centroids[cluster_id]
                        internal_scatter = np.mean([
                            np.sum((cluster_features[i] - cluster_center) ** 2)
                            for i in range(len(cluster_features))
                        ])

                        # Compare to overall cluster quality distribution
                        all_scatters = []
                        for c in range(stats.n_clusters):
                            c_mask = stats.assignments == c
                            if np.any(c_mask) and stats.cluster_sizes[c] > 0:
                                c_features = features[c_mask]
                                c_center = stats.centroids[c]
                                c_scatter = np.mean([np.sum((c_features[i] - c_center) ** 2) for i in range(len(c_features))])
                                all_scatters.append(c_scatter)

                        if all_scatters:
                            scatter_threshold = np.percentile(all_scatters, 75)  # Top quartile

                            if internal_scatter > scatter_threshold:
                                split_candidates.append(cluster_id)

            return split_candidates

        except Exception as e:
            tprint(f"❌ Split candidate identification failed: {e}", "ERROR")
            return []

    def _calculate_split_delta(self, features: np.ndarray, stats: ClusteringStats, cluster_id: int) -> float:
        """Calculate the objective improvement from splitting a cluster."""
        try:
            cluster_mask = stats.assignments == cluster_id
            cluster_features = features[cluster_mask]

            if len(cluster_features) < 20:  # Need enough points to split
                return 0.0

            # Try to split using 2-means
            from sklearn.cluster import KMeans
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
            threshold = self.config.split_quality_threshold * (1 + self.config.alpha_penalty * (size_factor - 1))

            return quality_improvement - threshold

        except Exception as e:
            tprint(f"❌ Split delta calculation failed: {e}", "ERROR")
            return 0.0

    def _perform_cluster_split(self, features: np.ndarray, stats: ClusteringStats, cluster_id: int) -> float:
        """Perform a cluster split and return the objective improvement."""
        try:
            cluster_mask = stats.assignments == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_features = features[cluster_mask]

            # Perform 2-means split
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            sub_assignments = kmeans.fit_predict(cluster_features)

            # Create new cluster ID
            new_cluster_id = stats.n_clusters

            # Update assignments for second sub-cluster
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

            # Calculate actual objective improvement
            return self._calculate_split_delta(features, stats, cluster_id)

        except Exception as e:
            tprint(f"❌ Cluster split performance failed: {e}", "ERROR")
            return 0.0

    def _check_k_growth_acceptable(self, current_k: int, proposed_k: int, n_samples: int) -> bool:
        """Check if k-growth is within acceptable bounds."""
        try:
            # Calculate k-growth ratio
            if current_k == 0:
                return True

            k_growth_ratio = proposed_k / current_k

            # Base acceptable ratio (more conservative for larger datasets)
            base_ratio = 1.5  # 50% growth
            if n_samples > 1000:
                base_ratio = 1.3  # 30% growth for large datasets
            elif n_samples > 10000:
                base_ratio = 1.2  # 20% growth for very large datasets

            # Apply k-complexity penalty
            complexity_factor = 1 + self.config.k_complexity_penalty * (proposed_k / self.config.k_max)

            acceptable_ratio = base_ratio / complexity_factor

            return k_growth_ratio <= acceptable_ratio

        except Exception as e:
            tprint(f"❌ K-growth check failed: {e}", "ERROR")
            return False

    def _calculate_enhanced_objective(self, stats: ClusteringStats, features: np.ndarray, config: Any) -> float:
        """Calculate enhanced objective function with fragmentation penalties and size constraints."""
        try:
            # Base objective from ClusteringStats
            base_objective = stats.get_objective_value(
                w_cv=self.config.w_cv,
                w_bal=self.config.w_bal,
                w_sil=self.config.w_sil,
                w_temp=self.config.w_temp,
                k_complexity_penalty=self.config.k_complexity_penalty,
                k_max=self.config.k_max
            )
            
            # Calculate fragmentation penalty
            fragmentation_penalty = 0.0
            if self.config.enable_cohesion_optimization:
                fragmentation_penalty = self._calculate_fragmentation_penalty(stats, features)
            
            # Calculate size balance penalty
            size_balance_penalty = self._calculate_size_balance_penalty(stats)
            
            # Calculate cohesion reward
            cohesion_reward = 0.0
            if self.config.enable_cohesion_optimization:
                cohesion_reward = self._calculate_cohesion_reward(stats, features)
            
            # Enhanced objective = base + penalties - rewards
            enhanced_objective = (base_objective + 
                                fragmentation_penalty + 
                                size_balance_penalty - 
                                cohesion_reward)
            
            # Store detailed metrics for monitoring
            if not hasattr(self, '_objective_breakdown'):
                self._objective_breakdown = {}
            
            self._objective_breakdown = {
                'base_objective': base_objective,
                'fragmentation_penalty': fragmentation_penalty,
                'size_balance_penalty': size_balance_penalty,
                'cohesion_reward': cohesion_reward,
                'enhanced_objective': enhanced_objective
            }
            
            return enhanced_objective
            
        except Exception as e:
            tprint(f"❌ Enhanced objective calculation failed: {e}", "ERROR")
            return base_objective if 'base_objective' in locals() else 0.0

    def _calculate_fragmentation_penalty(self, stats: ClusteringStats, features: np.ndarray) -> float:
        """Calculate fragmentation penalty based on cluster cohesion."""
        try:
            total_penalty = 0.0
            n_clusters = stats.n_clusters
            
            for cluster_id in range(n_clusters):
                if stats.cluster_sizes[cluster_id] == 0:
                    continue
                
                # Get cluster points
                cluster_mask = stats.assignments == cluster_id
                cluster_features = features[cluster_mask]
                
                if len(cluster_features) < 2:
                    continue
                
                # Calculate fragmentation score for this cluster
                fragmentation_score = self._calculate_cluster_fragmentation(
                    cluster_features, stats.centroids[cluster_id]
                )
                
                # Apply penalty if fragmentation is high
                if fragmentation_score > self.config.fragmentation_penalty_threshold:
                    penalty = (fragmentation_score - self.config.fragmentation_penalty_threshold) * self.config.w_frag
                    total_penalty += penalty
                    
                    tprint(f"🔧 Cluster {cluster_id} fragmentation penalty: {penalty:.4f}", "DEBUG")
            
            return total_penalty
            
        except Exception as e:
            tprint(f"⚠️ Fragmentation penalty calculation failed: {e}", "WARNING")
            return 0.0

    def _calculate_cluster_fragmentation(self, cluster_features: np.ndarray, centroid: np.ndarray) -> float:
        """Calculate fragmentation score for a single cluster."""
        try:
            if len(cluster_features) < 2:
                return 0.0
            
            # Calculate distances from points to centroid
            distances = np.sqrt(np.sum((cluster_features - centroid) ** 2, axis=1))
            
            # Fragmentation = coefficient of variation of distances
            if np.mean(distances) == 0:
                return 0.0
            
            fragmentation = np.std(distances) / np.mean(distances)
            return min(fragmentation, 1.0)  # Cap at 1.0
            
        except Exception as e:
            tprint(f"⚠️ Cluster fragmentation calculation failed: {e}", "WARNING")
            return 0.0

    def _calculate_size_balance_penalty(self, stats: ClusteringStats) -> float:
        """Calculate penalty for imbalanced cluster sizes."""
        try:
            if stats.n_clusters < 2:
                return 0.0
            
            # Calculate cluster size ratios
            total_samples = len(stats.assignments)
            cluster_ratios = [size / total_samples for size in stats.cluster_sizes if size > 0]
            
            if not cluster_ratios:
                return 0.0
            
            # Calculate size balance metrics
            target_ratio = 1.0 / len(cluster_ratios)  # Equal distribution
            size_variance = np.var(cluster_ratios)
            size_std = np.std(cluster_ratios)
            
            # Penalty based on deviation from target
            penalty = 0.0
            
            # Penalty for clusters that are too large
            for ratio in cluster_ratios:
                if ratio > self.config.max_cluster_ratio:
                    excess = ratio - self.config.max_cluster_ratio
                    penalty += excess * self.config.size_balance_weight
            
            # Penalty for clusters that are too small
            for ratio in cluster_ratios:
                if ratio < self.config.min_cluster_ratio:
                    deficit = self.config.min_cluster_ratio - ratio
                    penalty += deficit * self.config.size_balance_weight
            
            # Additional penalty for high variance (RELAXED)
            if size_std > 0.25:  # RELAXED: Only penalize severe imbalance (was 0.15)
                penalty += size_std * self.config.size_balance_weight * 0.3  # RELAXED: Reduced penalty factor (was 0.5)
            
            return penalty
            
        except Exception as e:
            tprint(f"⚠️ Size balance penalty calculation failed: {e}", "WARNING")
            return 0.0

    def _calculate_cohesion_reward(self, stats: ClusteringStats, features: np.ndarray) -> float:
        """Calculate reward for high cluster cohesion."""
        try:
            total_reward = 0.0
            n_clusters = stats.n_clusters
            
            for cluster_id in range(n_clusters):
                if stats.cluster_sizes[cluster_id] < 2:
                    continue
                
                # Get cluster points
                cluster_mask = stats.assignments == cluster_id
                cluster_features = features[cluster_mask]
                
                # Calculate cohesion score for this cluster
                cohesion_score = self._calculate_cluster_cohesion(
                    cluster_features, stats.centroids[cluster_id]
                )
                
                # Apply reward if cohesion is high
                if cohesion_score > self.config.cohesion_reward_threshold:
                    reward = (cohesion_score - self.config.cohesion_reward_threshold) * self.config.w_frag * 0.5
                    total_reward += reward
                    
                    tprint(f"🔧 Cluster {cluster_id} cohesion reward: {reward:.4f}", "DEBUG")
            
            return total_reward
            
        except Exception as e:
            tprint(f"⚠️ Cohesion reward calculation failed: {e}", "WARNING")
            return 0.0

    def _calculate_cluster_cohesion(self, cluster_features: np.ndarray, centroid: np.ndarray) -> float:
        """Calculate cohesion score for a single cluster."""
        try:
            if len(cluster_features) < 2:
                return 0.0
            
            # Calculate average distance to centroid
            distances = np.sqrt(np.sum((cluster_features - centroid) ** 2, axis=1))
            avg_distance = np.mean(distances)
            
            # Calculate pairwise distances within cluster
            from sklearn.metrics.pairwise import euclidean_distances
            pairwise_distances = euclidean_distances(cluster_features)
            
            # Remove diagonal (self-distances)
            pairwise_distances = pairwise_distances[np.triu_indices_from(pairwise_distances, k=1)]
            
            if len(pairwise_distances) == 0:
                return 0.0
            
            avg_pairwise_distance = np.mean(pairwise_distances)
            
            # Cohesion = ratio of centroid distance to pairwise distance
            # Higher ratio = more cohesive (points close to centroid relative to each other)
            if avg_pairwise_distance == 0:
                return 1.0
            
            cohesion = avg_distance / avg_pairwise_distance
            return min(cohesion, 1.0)  # Cap at 1.0
            
        except Exception as e:
            tprint(f"⚠️ Cluster cohesion calculation failed: {e}", "WARNING")
            return 0.0

    def _get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization performance."""
        try:
            if not self.optimization_history:
                return {"error": "No optimization history available"}

            final_iteration = self.optimization_history[-1]

            return {
                "total_iterations": len(self.optimization_history),
                "final_objective": final_iteration['objective'],
                "total_improvement": (
                    final_iteration['objective'] - self.optimization_history[0]['objective']
                    if len(self.optimization_history) > 1 else 0
                ),
                "final_n_clusters": final_iteration['n_clusters'],
                "converged": self.convergence_info.get('converged', False),
                "moves_evaluated": self.performance_metrics["moves_evaluated"],
                "moves_accepted": self.performance_metrics["moves_accepted"],
                "acceptance_rate": (
                    self.performance_metrics["moves_accepted"] / self.performance_metrics["moves_evaluated"]
                    if self.performance_metrics["moves_evaluated"] > 0 else 0
                )
            }

        except Exception as e:
            return {"error": str(e)}

    def reset_optimizer(self) -> None:
        """Reset optimizer state."""
        try:
            self.optimization_history.clear()
            self.convergence_info.clear()
            self.move_history.clear()

            self.performance_metrics = {
                "total_optimization_time": 0.0,
                "iteration_count": 0,
                "moves_evaluated": 0,
                "moves_accepted": 0,
                "cache_hits": 0,
                "cache_misses": 0
            }

            self._numba_functions_cache.clear()

            tprint("Clustering optimizer reset", "INFO")

        except Exception as e:
            tprint(f"❌ Optimizer reset failed: {e}", "ERROR")
