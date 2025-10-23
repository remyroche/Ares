"""
Risk Mitigation System for Iterative Clustering Optimization.

This module implements precise fixes for the biggest risks in the clustering system:
1. Unbounded k growth via splits
2. Over-churn from global reallocation
3. Metric drift / noisy wins
4. Embedding instability
5. Readiness gates and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import hashlib
import time
from datetime import datetime
import logging
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
import warnings

from .step1_feature_preparation import ClusteringContext
from .shared_utils import get_logger

class ClusteringStats:
    """Simplified ClusteringStats for risk mitigation (avoids circular import)."""

    def __init__(self, features: np.ndarray, assignments: np.ndarray):
        """Initialize with current clustering state."""
        self.features = features
        self.assignments = assignments
        self.n_samples, self.n_features = features.shape

        # CRITICAL FIX: Handle non-consecutive cluster IDs properly
        unique_clusters = np.unique(assignments)
        self.n_clusters = len(unique_clusters)

        # Create mapping from original cluster IDs to consecutive 0-based indices
        self.cluster_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
        self.inverse_cluster_id_map = {new_id: old_id for old_id, new_id in self.cluster_id_map.items()}

        # Remap assignments to consecutive 0-based indices
        self.remapped_assignments = np.array([self.cluster_id_map[cid] for cid in assignments])

        # Calculate basic statistics using remapped assignments
        self.cluster_sizes = np.bincount(self.remapped_assignments, minlength=self.n_clusters)
        self.centroids = np.array([np.mean(features[self.remapped_assignments == i], axis=0)
                                  for i in range(self.n_clusters)])
        self.global_mean = np.mean(features, axis=0)

        # Calculate WCSS and BCSS using remapped assignments
        self.total_wcss = 0.0
        for i in range(self.n_clusters):
            cluster_points = features[self.remapped_assignments == i]
            if len(cluster_points) > 0:
                self.total_wcss += np.sum((cluster_points - self.centroids[i]) ** 2)

        self.total_bcss = np.sum(self.cluster_sizes * np.sum((self.centroids - self.global_mean) ** 2, axis=1))

    def get_cv_ratio(self) -> float:
        """Get current CV ratio (BCSS/WCSS)."""
        if self.total_wcss == 0:
            return 0.0
        return self.total_bcss / self.total_wcss

    def calculate_move_delta(self, point_idx: int, from_cluster: int, to_cluster: int) -> Dict[str, float]:
        """Calculate move delta using remapped cluster IDs."""
        # Map original cluster IDs to consecutive indices
        from_cluster_remapped = self.cluster_id_map.get(from_cluster, from_cluster)
        to_cluster_remapped = self.cluster_id_map.get(to_cluster, to_cluster)

        if from_cluster_remapped == to_cluster_remapped:
            return {'total': 0.0, 'cv': 0.0, 'balance': 0.0, 'silhouette': 0.0, 'temporal': 0.0}

        # Calculate basic delta (simplified for risk mitigation)
        point = self.features[point_idx]

        # Distance-based delta calculation
        from_centroid = self.centroids[from_cluster_remapped]
        to_centroid = self.centroids[to_cluster_remapped]

        old_distance = np.linalg.norm(point - from_centroid)
        new_distance = np.linalg.norm(point - to_centroid)

        distance_delta = new_distance - old_distance

        # Simple balance delta
        old_from_size = self.cluster_sizes[from_cluster_remapped]
        old_to_size = self.cluster_sizes[to_cluster_remapped]

        balance_delta = 0.0  # Simplified for risk mitigation

        return {
            'total': distance_delta,
            'cv': distance_delta,
            'balance': balance_delta,
            'silhouette': 0.0,
            'temporal': 0.0
        }

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

@dataclass
class RiskMitigationConfig:
    """Configuration for risk mitigation system."""

    # Unbounded k growth prevention
    k_complexity_penalty: float = 0.25
    max_new_splits_per_round: int = 3
    max_k_growth_factor: float = 0.1  # 10% of current k

    # Over-churn prevention
    local_churn_cap: float = 0.02  # 2% of N
    global_churn_cap: float = 0.08  # 8% of N
    min_delta_j_threshold: float = 0.0  # ΔJ_point > 0

    # Metric drift prevention
    monotone_tolerance: float = 1e-5
    incremental_audit_frequency: int = 5  # Every 5 rounds
    rollback_on_decrease: bool = True

    # Readiness gates
    min_silhouette: float = 0.2
    max_dbi: float = 2.5
    min_cv_ratio_good: float = 1.5
    min_cv_ratio_excellent: float = 2.0
    min_temporal_acceptable: float = 0.4
    min_temporal_strong: float = 0.6
    min_balance: float = 0.7
    max_churn_per_cycle: float = 0.10  # 10% of N
    min_cluster_size_factor: float = 0.005  # 0.5% of N
    min_cluster_size_absolute: int = 25

    # Validation & stability
    bootstrap_samples: int = 10
    stability_threshold: float = 0.7
    permutation_test_rounds: int = 5

    # Productionization
    max_wall_time: float = 3600.0  # 1 hour
    max_operations: int = 10000
    state_repetition_threshold: int = 3
    convergence_tolerance: float = 1e-5
    max_convergence_cycles: int = 3

@dataclass
class ReadinessGates:
    """Readiness gate status."""

    # Geometry gates
    silhouette_passed: bool = False
    dbi_passed: bool = False

    # Economics gates
    cv_ratio_good: bool = False
    cv_ratio_excellent: bool = False

    # Temporal gates
    temporal_acceptable: bool = False
    temporal_strong: bool = False

    # Balance gates
    balance_passed: bool = False

    # Churn gates
    churn_acceptable: bool = False

    # Monotonicity gates
    monotone_passed: bool = False

    # Size gates
    min_size_passed: bool = False

    # Overall readiness
    overall_ready: bool = False

class RiskMitigationSystem:
    """Comprehensive risk mitigation system for clustering optimization."""

    def __init__(self, config: RiskMitigationConfig = None):
        """Initialize the risk mitigation system."""
        self.config = config or RiskMitigationConfig()
        self.logger = get_logger('RiskMitigationSystem')

        # State tracking
        self.state_history: List[str] = []
        self.objective_history: List[float] = []
        self.operation_counts: Dict[str, int] = {
            'local_moves': 0,
            'global_moves': 0,
            'splits': 0,
            'total_operations': 0
        }

        # Timing and convergence
        self.start_time = time.time()
        self.convergence_cycles = 0
        self.last_objective = None

        # Validation tracking
        self.bootstrap_ari_scores: List[float] = []
        self.permutation_test_results: List[float] = []

    def check_unbounded_k_growth(self, current_k: int, proposed_k: int, n_samples: int) -> bool:
        """Prevent unbounded k growth via splits."""
        # Check max new splits per round
        new_splits = proposed_k - current_k
        if new_splits > self.config.max_new_splits_per_round:
            self.logger.warning(f"Too many new splits: {new_splits} > {self.config.max_new_splits_per_round}")
            return False

        # Check max k growth factor
        max_k_growth = int(self.config.max_k_growth_factor * current_k)
        if new_splits > max_k_growth:
            self.logger.warning(f"K growth too high: {new_splits} > {max_k_growth}")
            return False

        # Check absolute k limit (reasonable upper bound)
        max_absolute_k = min(50, n_samples // 10)  # Max 50 clusters or 1 per 10 samples
        if proposed_k > max_absolute_k:
            self.logger.warning(f"Absolute k limit exceeded: {proposed_k} > {max_absolute_k}")
            return False

        return True

    def apply_k_complexity_penalty(self, objective: float, current_k: int, max_k: int = 20) -> float:
        """Apply k-complexity penalty to prevent runaway splitting."""
        k_penalty = self.config.k_complexity_penalty * (current_k - 1) / max_k
        penalized_objective = objective - k_penalty

        self.logger.debug(f"K-complexity penalty: {k_penalty:.6f}, Objective: {objective:.6f} -> {penalized_objective:.6f}")
        return penalized_objective

    def check_over_churn(self, n_samples: int, local_moves: int, global_moves: int) -> Tuple[bool, str]:
        """Prevent over-churn from global reallocation."""
        local_churn_rate = local_moves / n_samples
        global_churn_rate = global_moves / n_samples
        total_churn_rate = (local_moves + global_moves) / n_samples

        # Check local churn cap
        if local_churn_rate > self.config.local_churn_cap:
            return False, f"Local churn too high: {local_churn_rate:.3f} > {self.config.local_churn_cap}"

        # Check global churn cap
        if global_churn_rate > self.config.global_churn_cap:
            return False, f"Global churn too high: {global_churn_rate:.3f} > {self.config.global_churn_cap}"

        # Check total churn per cycle
        if total_churn_rate > self.config.max_churn_per_cycle:
            return False, f"Total churn too high: {total_churn_rate:.3f} > {self.config.max_churn_per_cycle}"

        return True, "Churn within limits"

    def check_metric_drift(self, current_j: float, previous_j: float) -> Tuple[bool, str]:
        """Enforce monotone J and detect metric drift."""
        if previous_j is None:
            return True, "First iteration"

        # Check monotonicity
        if current_j < previous_j - self.config.monotone_tolerance:
            if self.config.rollback_on_decrease:
                return False, f"Monotone violation: {current_j:.6f} < {previous_j:.6f} (rollback required)"
            else:
                self.logger.warning(f"Monotone violation: {current_j:.6f} < {previous_j:.6f}")

        # Check for convergence
        improvement = current_j - previous_j
        if abs(improvement) < self.config.convergence_tolerance:
            self.convergence_cycles += 1
            if self.convergence_cycles >= self.config.max_convergence_cycles:
                return False, f"Converged: improvement {improvement:.2e} < {self.config.convergence_tolerance} for {self.convergence_cycles} cycles"
        else:
            self.convergence_cycles = 0

        return True, f"Improvement: {improvement:+.6f}"

    def audit_incremental_correctness(self, features: np.ndarray, stats: ClusteringStats) -> bool:
        """Verify incremental ≡ full recomputation on random audits."""
        try:
            # Sample random moves for validation
            sample_size = min(50, len(features))
            sample_indices = np.random.choice(len(features), sample_size, replace=False)

            for idx in sample_indices:
                current_cluster = int(stats.assignments[idx])
                other_clusters = [c for c in range(stats.n_clusters) if c != current_cluster]
                if not other_clusters:
                    continue

                target_cluster = np.random.choice(other_clusters)

                # CRITICAL FIX: Check if this is the risk mitigation ClusteringStats or main ClusteringStats
                if hasattr(stats, 'cluster_id_map'):
                    # This is the risk mitigation ClusteringStats, use remapped IDs
                    current_cluster_remapped = stats.cluster_id_map.get(current_cluster, current_cluster)
                    target_cluster_remapped = stats.cluster_id_map.get(target_cluster, target_cluster)

                    # Calculate incremental delta using remapped IDs
                    delta_inc = stats.calculate_move_delta(idx, current_cluster_remapped, target_cluster_remapped)
                else:
                    # This is the main ClusteringStats, use original IDs
                    delta_inc = stats.calculate_move_delta(idx, current_cluster, target_cluster)

                # Calculate full recomputation
                temp_assignments = stats.assignments.copy()
                temp_assignments[idx] = target_cluster
                temp_stats = ClusteringStats(features, temp_assignments)

                # Compare BCSS/WCSS
                cv_inc = delta_inc['cv']
                cv_full = temp_stats.get_cv_ratio() - stats.get_cv_ratio()

                if abs(cv_inc - cv_full) / max(1, abs(cv_full)) > 1e-8:
                    self.logger.error(f"Incremental audit failed: CV delta mismatch {cv_inc:.6f} vs {cv_full:.6f}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Incremental audit error: {e}")
            return False

    def check_readiness_gates(self, features: np.ndarray, stats: ClusteringStats,
                            assignments: np.ndarray, n_samples: int) -> ReadinessGates:
        """Check all readiness gates."""
        gates = ReadinessGates()

        try:
            tprint(f"🔍 DEBUG: Checking readiness gates - features shape: {features.shape}, assignments shape: {assignments.shape}", "DEBUG")
            tprint(f"🔍 DEBUG: assignments dtype: {assignments.dtype}, unique values: {np.unique(assignments)}", "DEBUG")

            # Geometry gates
            if len(np.unique(assignments)) > 1:
                silhouette = silhouette_score(features, assignments)
                dbi = davies_bouldin_score(features, assignments)

                gates.silhouette_passed = silhouette >= self.config.min_silhouette
                gates.dbi_passed = dbi <= self.config.max_dbi
            else:
                gates.silhouette_passed = False
                gates.dbi_passed = False

            # Economics gates
            cv_ratio = stats.get_cv_ratio()
            gates.cv_ratio_good = cv_ratio >= self.config.min_cv_ratio_good
            gates.cv_ratio_excellent = cv_ratio >= self.config.min_cv_ratio_excellent

            # Temporal gates (placeholder - would need temporal data)
            temporal_score = 0.5  # Placeholder
            gates.temporal_acceptable = temporal_score >= self.config.min_temporal_acceptable
            gates.temporal_strong = temporal_score >= self.config.min_temporal_strong

            # Balance gates
            balance = stats.get_balance_score()
            gates.balance_passed = balance >= self.config.min_balance

            # Churn gates (calculated from operation counts)
            total_churn = self.operation_counts['total_operations'] / n_samples
            gates.churn_acceptable = total_churn <= self.config.max_churn_per_cycle

            # Monotonicity gates
            if len(self.objective_history) >= 2:
                recent_improvement = self.objective_history[-1] - self.objective_history[-2]
                gates.monotone_passed = recent_improvement >= -self.config.monotone_tolerance
            else:
                gates.monotone_passed = True

            # Size gates
            min_size = max(self.config.min_cluster_size_absolute,
                          int(self.config.min_cluster_size_factor * n_samples))
            cluster_sizes = stats.cluster_sizes[stats.cluster_sizes > 0]
            # Ensure cluster_sizes is numeric and handle the boolean logic safely
            min_size_passed = len(cluster_sizes) == 0
            if not min_size_passed and len(cluster_sizes) > 0:
                min_size_passed = int(np.min(cluster_sizes)) >= min_size
            gates.min_size_passed = min_size_passed

            # Overall readiness
            gates.overall_ready = (
                gates.silhouette_passed and gates.dbi_passed and
                gates.cv_ratio_good and gates.balance_passed and
                gates.churn_acceptable and gates.monotone_passed and gates.min_size_passed
            )

        except Exception as e:
            self.logger.error(f"Readiness gate check error: {e}")
            gates.overall_ready = False

        return gates

    def validate_stability(self, features: np.ndarray, assignments: np.ndarray) -> Dict[str, float]:
        """Validate clustering stability using bootstrap and permutation tests."""
        results = {}

        try:
            # Bootstrap stability test
            ari_scores = []
            for _ in range(self.config.bootstrap_samples):
                # Bootstrap sample
                bootstrap_indices = np.random.choice(len(features), len(features), replace=True)
                bootstrap_features = features[bootstrap_indices]
                bootstrap_assignments = assignments[bootstrap_indices]

                # Cluster bootstrap data
                if len(np.unique(bootstrap_assignments)) > 1:
                    kmeans = KMeans(n_clusters=len(np.unique(assignments)), random_state=42)
                    bootstrap_pred = kmeans.fit_predict(bootstrap_features)

                    # Calculate ARI with original assignments
                    ari = adjusted_rand_score(bootstrap_assignments, bootstrap_pred)
                    ari_scores.append(ari)

            results['bootstrap_ari_mean'] = np.mean(ari_scores) if ari_scores else 0.0
            results['bootstrap_ari_std'] = np.std(ari_scores) if ari_scores else 0.0
            results['bootstrap_stable'] = results['bootstrap_ari_mean'] >= self.config.stability_threshold

            # Permutation test
            permutation_scores = []
            for _ in range(self.config.permutation_test_rounds):
                # Shuffle labels
                shuffled_assignments = np.random.permutation(assignments)

                # Calculate objective with shuffled labels
                temp_stats = ClusteringStats(features, shuffled_assignments)
                shuffled_objective = temp_stats.get_objective_value()
                permutation_scores.append(shuffled_objective)

            results['permutation_mean'] = np.mean(permutation_scores)
            results['permutation_std'] = np.std(permutation_scores)

            # Current objective should be significantly higher than permuted
            current_objective = ClusteringStats(features, assignments).get_objective_value()
            results['permutation_test_passed'] = current_objective > results['permutation_mean'] + 2 * results['permutation_std']

        except Exception as e:
            self.logger.error(f"Stability validation error: {e}")
            results = {'bootstrap_ari_mean': 0.0, 'permutation_test_passed': False}

        return results

    def check_state_repetition(self, current_state: str) -> bool:
        """Check for state repetition to prevent infinite loops."""
        self.state_history.append(current_state)

        # Keep only recent history
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-50:]

        # Check for repetition
        if len(self.state_history) >= self.config.state_repetition_threshold:
            recent_states = self.state_history[-self.config.state_repetition_threshold:]
            if len(set(recent_states)) == 1:  # All recent states are identical
                self.logger.warning(f"State repetition detected: {current_state}")
                return False

        return True

    def check_wall_time_budget(self) -> bool:
        """Check if wall time budget is exceeded."""
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.config.max_wall_time:
            self.logger.warning(f"Wall time budget exceeded: {elapsed_time:.1f}s > {self.config.max_wall_time}s")
            return False
        return True

    def check_operations_budget(self) -> bool:
        """Check if operations budget is exceeded."""
        if self.operation_counts['total_operations'] > self.config.max_operations:
            self.logger.warning(f"Operations budget exceeded: {self.operation_counts['total_operations']} > {self.config.max_operations}")
            return False
        return True

    def generate_state_hash(self, k: int, assignments: np.ndarray) -> str:
        """Generate deterministic hash of clustering state."""
        state_string = f"{k}_{assignments.tobytes()}"
        return hashlib.md5(state_string.encode()).hexdigest()[:8]

    def log_cycle_metrics(self, round_num: int, stats: ClusteringStats,
                         features: np.ndarray, assignments: np.ndarray):
        """Log comprehensive cycle metrics."""
        try:
            # Basic metrics
            k = stats.n_clusters
            j = stats.get_objective_value()
            cv_ratio = stats.get_cv_ratio()
            balance = stats.get_balance_score()

            # Quality metrics
            silhouette = 0.0
            dbi = 0.0
            if len(np.unique(assignments)) > 1:
                silhouette = silhouette_score(features, assignments)
                dbi = davies_bouldin_score(features, assignments)

            # Size statistics
            cluster_sizes = stats.cluster_sizes[stats.cluster_sizes > 0]
            size_stats = {
                'min': int(np.min(cluster_sizes)) if len(cluster_sizes) > 0 else 0,
                'median': float(np.median(cluster_sizes)) if len(cluster_sizes) > 0 else 0.0,
                'p95': float(np.percentile(cluster_sizes, 95)) if len(cluster_sizes) > 0 else 0.0
            }

            # Operations summary
            ops_summary = {
                'local_moves': self.operation_counts['local_moves'],
                'global_moves': self.operation_counts['global_moves'],
                'splits': self.operation_counts['splits'],
                'total': self.operation_counts['total_operations']
            }

            # State hash
            state_hash = self.generate_state_hash(k, assignments)

            # Log comprehensive metrics
            self.logger.info(f"Round {round_num} Metrics:")
            self.logger.info(f"  k={k}, J={j:.6f}, CV={cv_ratio:.4f}, Balance={balance:.4f}")
            self.logger.info(f"  Silhouette={silhouette:.4f}, DBI={dbi:.4f}")
            self.logger.info(f"  Operations: {ops_summary}")
            self.logger.info(f"  Sizes: {size_stats}")
            self.logger.info(f"  State hash: {state_hash}")

        except Exception as e:
            self.logger.error(f"Cycle metrics logging error: {e}")

    def should_stop_optimization(self, round_num: int, stats: ClusteringStats,
                               features: np.ndarray, assignments: np.ndarray) -> Tuple[bool, str]:
        """Determine if optimization should stop based on all risk factors."""

        # Check wall time budget
        if not self.check_wall_time_budget():
            return True, "Wall time budget exceeded"

        # Check operations budget
        if not self.check_operations_budget():
            return True, "Operations budget exceeded"

        # Check state repetition
        state_hash = self.generate_state_hash(stats.n_clusters, assignments)
        if not self.check_state_repetition(state_hash):
            return True, "State repetition detected"

        # Check convergence
        if len(self.objective_history) >= 2:
            recent_improvement = self.objective_history[-1] - self.objective_history[-2]
            if abs(recent_improvement) < self.config.convergence_tolerance:
                if self.convergence_cycles >= self.config.max_convergence_cycles:
                    return True, f"Converged: improvement {recent_improvement:.2e} for {self.convergence_cycles} cycles"

        # Check readiness gates
        gates = self.check_readiness_gates(features, stats, assignments, len(features))
        if gates.overall_ready and round_num > 5:  # Allow some warm-up rounds
            return True, "All readiness gates passed"

        return False, "Continue optimization"

    def update_operation_counts(self, local_moves: int, global_moves: int, splits: int):
        """Update operation counts for churn tracking."""
        self.operation_counts['local_moves'] += local_moves
        self.operation_counts['global_moves'] += global_moves
        self.operation_counts['splits'] += splits
        self.operation_counts['total_operations'] += local_moves + global_moves + splits

    def update_objective_history(self, objective: float):
        """Update objective history for convergence tracking."""
        self.objective_history.append(objective)
        if len(self.objective_history) > 100:  # Keep only recent history
            self.objective_history = self.objective_history[-50:]

        self.last_objective = objective

    def apply_risk_mitigation(self, features: np.ndarray, assignments: np.ndarray,
                             total_delta: float, iteration: int) -> Dict[str, Any]:
        """Apply risk mitigation checks and return results."""
        try:
            violations = 0
            warnings = []

            # Check for unbounded k growth
            current_k = len(np.unique(assignments))
            if not self.check_unbounded_k_growth(current_k, current_k, len(features)):
                violations += 1
                warnings.append("Unbounded k growth detected")

            # Check for over-churn (simplified)
            if abs(total_delta) > 0.1:  # High delta might indicate over-churn
                violations += 1
                warnings.append("Potential over-churn detected")

            # Check metric drift
            if len(self.objective_history) >= 2:
                current_j = total_delta
                previous_j = self.objective_history[-1] if self.objective_history else 0.0
                drift_ok, drift_msg = self.check_metric_drift(current_j, previous_j)
                if not drift_ok:
                    violations += 1
                    warnings.append(f"Metric drift: {drift_msg}")

            # Update objective history
            self.update_objective_history(total_delta)

            return {
                "violations": violations,
                "warnings": warnings,
                "total_delta": total_delta,
                "iteration": iteration,
                "status": "success" if violations == 0 else "warning"
            }

        except Exception as e:
            self.logger.error(f"Risk mitigation error: {e}")
            return {
                "violations": 1,
                "warnings": [f"Risk mitigation error: {e}"],
                "total_delta": total_delta,
                "iteration": iteration,
                "status": "error"
            }

# Default configuration for production use
PRODUCTION_RISK_CONFIG = RiskMitigationConfig(
    k_complexity_penalty=0.25,
    max_new_splits_per_round=3,
    local_churn_cap=0.02,
    global_churn_cap=0.08,
    min_delta_j_threshold=0.0,
    monotone_tolerance=1e-5,
    min_silhouette=0.2,
    max_dbi=2.5,
    min_cv_ratio_good=1.5,
    min_cv_ratio_excellent=2.0,
    min_temporal_acceptable=0.4,
    min_temporal_strong=0.6,
    min_balance=0.7,
    max_churn_per_cycle=0.10,
    min_cluster_size_factor=0.005,
    min_cluster_size_absolute=25,
    max_wall_time=3600.0,
    max_operations=10000,
    state_repetition_threshold=3,
    convergence_tolerance=1e-5,
    max_convergence_cycles=3
)
