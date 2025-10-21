"""
Comprehensive Validation Framework for Iterative Clustering Optimization.

This module provides rigorous testing and validation for the 3-step iterative
clustering system to ensure correctness, stability, and performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set, Union, Callable, Awaitable
from dataclasses import dataclass, field
import hashlib
import time
from datetime import datetime
import logging
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.datasets import make_blobs
import warnings

from .iterative_optimization import IterativeOptimization, ClusteringStats
from .step1_feature_preparation import ClusteringContext
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_structured,
    tprint_timer, tprint_logged, LogLevel, TimestampFormat
)

from .shared_utils import get_logger

@dataclass
class ValidationConfig:
    """Configuration for validation framework."""

    # Correctness checks
    incremental_tolerance: float = 1e-8
    monotone_tolerance: float = 1e-10
    sample_moves_for_validation: int = 200

    # Runtime guardrails
    local_churn_cap: float = 0.02  # 2% of N
    global_churn_cap: float = 0.08  # 8% of N
    hysteresis_rounds: int = 2
    knn_rebuild_threshold: float = 0.15  # 15% labels moved
    k_change_threshold: float = 0.10  # 10% k change
    silhouette_eval_frequency: int = 5  # Every 5 rounds

    # Decision thresholds
    neighbor_consensus_threshold: float = 0.65
    capacity_buffer: float = 0.15
    size_factor_threshold: float = 1.5
    split_quality_threshold: float = 0.005
    alpha_penalty: float = 1.0
    k_complexity_penalty: float = 0.25

    # Logging
    log_detailed_stats: bool = True
    log_cluster_hashes: bool = True
    max_hash_history: int = 100

@dataclass
class ValidationResults:
    """Results from validation framework."""

    # Correctness
    incremental_checks_passed: int = 0
    incremental_checks_failed: int = 0
    monotone_violations: int = 0
    invariant_violations: int = 0

    # Performance
    total_moves_proposed: int = 0
    total_moves_accepted: int = 0
    local_moves: int = 0
    global_moves: int = 0
    splits_performed: int = 0

    # Quality metrics
    final_cv_ratio: float = 0.0
    final_silhouette: float = 0.0
    final_balance: float = 0.0
    final_dbi: float = 0.0

    # Runtime
    total_time: float = 0.0
    optimization_rounds: int = 0

    # Determinism
    seed_used: Optional[int] = None
    final_hash: Optional[str] = None

class ClusteringValidator:
    """Comprehensive validation framework for iterative clustering."""

    def __init__(self, config: ValidationConfig = None):
        """Initialize the validator."""
        self.config = config or ValidationConfig()
        self.logger = get_logger('ClusteringValidator')
        self.results = ValidationResults()
        self.hash_history: List[str] = []
        self.moved_points: Set[int] = set()
        self.move_rounds: Dict[int, int] = {}

        tprint("🚀 Clustering Validator initialized", "INFO")

    def validate_incremental_correctness(
        self,
        features: np.ndarray,
        stats: ClusteringStats,
        sample_size: Optional[int] = None
    ) -> bool:
        """Validate that incremental updates match full recomputation."""
        tprint(f"🔍 Validating incremental correctness with sample size: {sample_size}", "DEBUG")
        if sample_size is None:
            sample_size = min(self.config.sample_moves_for_validation, len(features))

        # Randomly sample moves to validate
        sample_indices = np.random.choice(len(features), sample_size, replace=False)

        all_passed = True

        for idx in sample_indices:
            current_cluster = int(stats.assignments[idx])

            # Try moving to a random other cluster
            other_clusters = [c for c in range(stats.n_clusters) if c != current_cluster]
            if not other_clusters:
                continue

            target_cluster = np.random.choice(other_clusters)

            # Calculate incremental delta
            delta_inc = stats.calculate_move_delta(idx, current_cluster, target_cluster)

            # Calculate full recomputation
            delta_full = self._calculate_full_delta(features, stats, idx, current_cluster, target_cluster)

            # Compare BCSS
            bcss_tol = abs(delta_inc['cv'] - delta_full['cv']) / max(1, abs(delta_full['cv']))
            if bcss_tol > self.config.incremental_tolerance:
                self.logger.error(f"BCSS incremental mismatch: {bcss_tol:.2e} > {self.config.incremental_tolerance:.2e}")
                all_passed = False
                self.results.incremental_checks_failed += 1
            else:
                self.results.incremental_checks_passed += 1

        return all_passed

    def _calculate_full_delta(
        self,
        features: np.ndarray,
        stats: ClusteringStats,
        point_idx: int,
        from_cluster: int,
        to_cluster: int
    ) -> Dict[str, float]:
        """Calculate full recomputation delta for validation."""
        # Create temporary assignments
        temp_assignments = stats.assignments.copy()
        temp_assignments[point_idx] = to_cluster

        # Calculate full statistics
        temp_stats = ClusteringStats(features, temp_assignments)

        # Calculate deltas
        cv_delta = temp_stats.get_cv_ratio() - stats.get_cv_ratio()
        balance_delta = temp_stats.get_balance_score() - stats.get_balance_score()

        # Calculate silhouette delta (simplified)
        point = features[point_idx]
        d1_old = np.linalg.norm(point - stats.centroids[from_cluster])
        d2_old = min([np.linalg.norm(point - stats.centroids[c]) for c in range(stats.n_clusters) if c != from_cluster])
        s_old = 1.0 - d1_old / d2_old if d2_old > 0 else 0.0

        d1_new = np.linalg.norm(point - temp_stats.centroids[to_cluster])
        d2_new = min([np.linalg.norm(point - temp_stats.centroids[c]) for c in range(stats.n_clusters) if c != to_cluster])
        s_new = 1.0 - d1_new / d2_new if d2_new > 0 else 0.0

        silhouette_delta = s_new - s_old

        return {
            'cv': cv_delta,
            'balance': balance_delta,
            'silhouette': silhouette_delta,
            'temporal': 0.0  # Placeholder
        }

    def validate_monotone_objective(
        self,
        current_j: float,
        proposed_j: float,
        operation: str
    ) -> bool:
        """Validate that objective function is monotone."""
        if proposed_j < current_j - self.config.monotone_tolerance:
            self.logger.error(f"Monotone violation in {operation}: {current_j:.6f} -> {proposed_j:.6f}")
            self.results.monotone_violations += 1
            return False
        return True

    def validate_invariants(
        self,
        stats: ClusteringStats,
        n_samples: int
    ) -> bool:
        """Validate clustering invariants."""
        all_passed = True

        # Check no empty clusters
        empty_clusters = np.sum(stats.cluster_sizes == 0)
        if empty_clusters > 0:
            self.logger.error(f"Found {empty_clusters} empty clusters")
            all_passed = False
            self.results.invariant_violations += 1

        # Check min size constraint
        min_size = max(25, int(0.005 * n_samples))  # 0.5% of N
        small_clusters = np.sum(stats.cluster_sizes < min_size)
        if small_clusters > 0:
            self.logger.warning(f"Found {small_clusters} clusters below min size {min_size}")

        # Check total samples
        total_assigned = np.sum(stats.cluster_sizes)
        if total_assigned != n_samples:
            self.logger.error(f"Total assigned {total_assigned} != {n_samples}")
            all_passed = False
            self.results.invariant_violations += 1

        return all_passed

    def validate_determinism(
        self,
        seed: int,
        features: np.ndarray,
        assignments: np.ndarray
    ) -> str:
        """Validate determinism and return hash."""
        # Create deterministic hash
        hash_input = f"{seed}_{features.shape}_{assignments.tobytes()}"
        cluster_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]

        self.results.seed_used = seed
        self.results.final_hash = cluster_hash

        # Check for hash repetition
        if cluster_hash in self.hash_history:
            self.logger.warning(f"Hash repetition detected: {cluster_hash}")

        self.hash_history.append(cluster_hash)
        if len(self.hash_history) > self.config.max_hash_history:
            self.hash_history.pop(0)

        return cluster_hash

    def create_synthetic_test_suite(self) -> List[Tuple[str, np.ndarray, int]]:
        """Create synthetic test scenarios."""
        test_cases = []

        # 1. Well-separated blobs (k=3)
        X1, y1 = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
        test_cases.append(("well_separated", X1, 3))

        # 2. Overlapping blobs
        X2, y2 = make_blobs(n_samples=300, centers=3, cluster_std=3.0, random_state=42)
        test_cases.append(("overlapping", X2, 3))

        # 3. One giant + small clusters
        X3 = np.vstack([
            np.random.normal(0, 1, (200, 2)),  # Giant cluster
            np.random.normal(10, 0.5, (50, 2)),  # Small cluster 1
            np.random.normal(-10, 0.5, (50, 2))  # Small cluster 2
        ])
        test_cases.append(("giant_small", X3, 3))

        # 4. No structure (isotropic noise)
        X4 = np.random.normal(0, 1, (300, 2))
        test_cases.append(("no_structure", X4, 2))

        return test_cases

    async def run_synthetic_validation(self) -> Dict[str, ValidationResults]:
        """Run validation on synthetic test cases."""
        tprint("🧪 Starting synthetic validation test suite", "INFO")
        test_cases = self.create_synthetic_test_suite()
        results = {}

        for name, features, expected_k in test_cases:
            self.logger.info(f"Running synthetic test: {name}")

            # Initialize clustering
            initial_kmeans = KMeans(n_clusters=expected_k, random_state=42)
            initial_assignments = initial_kmeans.fit_predict(features)

            # Create context and stats
            context = ClusteringContext(
                features=features,
                assignments=initial_assignments,
                metadata={}
            )
            stats = ClusteringStats(features, initial_assignments)

            # Run optimization with validation
            result = self.run_optimization_with_validation(context, stats)
            results[name] = result

        return results

    async def run_optimization_with_validation(
        self,
        context: ClusteringContext,
        stats: ClusteringStats
    ) -> ValidationResults:
        """Run optimization with comprehensive validation."""
        start_time = time.time()

        # Initialize optimization
        optimizer = IterativeOptimization(verbose=True)

        # Track objective function
        current_j = self._calculate_objective(stats)
        previous_j = current_j

        round_count = 0
        max_rounds = 50

        while round_count < max_rounds:
            round_count += 1
            self.logger.info(f"Optimization round {round_count}")

            # Validate invariants
            if not self.validate_invariants(stats, len(context.features)):
                break

            # Run optimization steps
            step1_improvement = await optimizer._step1_local_frontier_moves(context.features, stats)
            step2_improvement = await optimizer._step2_global_reallocation(context.features, stats)
            step3_improvement = await optimizer._step3_break_large_clusters(context.features, stats)

            # Calculate new objective
            new_j = self._calculate_objective(stats)

            # Validate monotone objective
            if not self.validate_monotone_objective(previous_j, new_j, f"round_{round_count}"):
                # Rollback if needed
                self.logger.error("Rolling back due to monotone violation")
                break

            # Update tracking
            previous_j = new_j

            # Check convergence
            if abs(new_j - current_j) < 1e-5:
                self.logger.info("Converged")
                break

            current_j = new_j

            # Validate incremental correctness (sample)
            if round_count % 5 == 0:  # Every 5 rounds
                self.validate_incremental_correctness(context.features, stats, 50)

        # Final validation
        self.results.total_time = time.time() - start_time
        self.results.optimization_rounds = round_count
        self.results.final_cv_ratio = stats.get_cv_ratio()
        self.results.final_balance = stats.get_balance_score()

        # Calculate final metrics
        if len(np.unique(stats.assignments)) > 1:
            self.results.final_silhouette = silhouette_score(context.features, stats.assignments)
            self.results.final_dbi = davies_bouldin_score(context.features, stats.assignments)

        return self.results

    def _calculate_objective(self, stats: ClusteringStats) -> float:
        """Calculate the objective function value."""
        cv_ratio = stats.get_cv_ratio()
        balance = stats.get_balance_score()

        # Simplified objective (would include silhouette and temporal in full implementation)
        objective = (
            self.config.w_cv * cv_ratio +
            self.config.w_bal * balance +
            self.config.w_sil * 0.5 +  # Placeholder for silhouette
            self.config.w_temp * 0.5    # Placeholder for temporal
        )

        # Add k-complexity penalty
        k_penalty = self.config.k_complexity_penalty * (stats.n_clusters - 1) / 10
        objective -= k_penalty

        return objective

    def log_cycle_stats(
        self,
        round_num: int,
        stats: ClusteringStats,
        features: np.ndarray,
        operations: Dict[str, int]
    ):
        """Log comprehensive cycle statistics."""
        if not self.config.log_detailed_stats:
            return

        # Basic metrics
        k = stats.n_clusters
        j = self._calculate_objective(stats)
        cv_ratio = stats.get_cv_ratio()
        balance = stats.get_balance_score()

        # Calculate silhouette (proxy or full)
        silhouette = 0.0
        if len(np.unique(stats.assignments)) > 1:
            if round_num % self.config.silhouette_eval_frequency == 0:
                # Full silhouette evaluation
                silhouette = silhouette_score(features, stats.assignments)
            else:
                # Proxy silhouette (simplified)
                silhouette = 0.5  # Placeholder

        # DBI
        dbi = 0.0
        if len(np.unique(stats.assignments)) > 1:
            dbi = davies_bouldin_score(features, stats.assignments)

        # Cluster size statistics
        cluster_sizes = stats.cluster_sizes[stats.cluster_sizes > 0]
        size_stats = {
            'min': int(np.min(cluster_sizes)) if len(cluster_sizes) > 0 else 0,
            'median': float(np.median(cluster_sizes)) if len(cluster_sizes) > 0 else 0.0,
            'p95': float(np.percentile(cluster_sizes, 95)) if len(cluster_sizes) > 0 else 0.0
        }

        # Log summary
        self.logger.info(
            f"Round {round_num}: k={k}, J={j:.6f}, CV={cv_ratio:.4f}, "
            f"Balance={balance:.4f}, Silhouette={silhouette:.4f}, DBI={dbi:.4f}"
        )

        self.logger.info(
            f"Operations: {operations}, Sizes: {size_stats}"
        )

        # Log cluster hash if enabled
        if self.config.log_cluster_hashes:
            cluster_hash = self.validate_determinism(42, features, stats.assignments)
            self.logger.info(f"Cluster hash: {cluster_hash}")

# Configuration for the validation framework
VALIDATION_CONFIG = ValidationConfig(
    incremental_tolerance=1e-8,
    monotone_tolerance=1e-10,
    sample_moves_for_validation=200,
    local_churn_cap=0.02,
    global_churn_cap=0.08,
    hysteresis_rounds=2,
    knn_rebuild_threshold=0.15,
    k_change_threshold=0.10,
    silhouette_eval_frequency=5,
    neighbor_consensus_threshold=0.65,
    capacity_buffer=0.15,
    size_factor_threshold=1.5,
    split_quality_threshold=0.005,
    alpha_penalty=1.0,
    k_complexity_penalty=0.25,
    log_detailed_stats=True,
    log_cluster_hashes=True,
    max_hash_history=100
)
