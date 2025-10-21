"""
Clustering Service for NAS-TAS Clustering.

This module provides the main clustering service that interfaces with the ClusteringEngine,
manages initial clustering, and coordinates the iterative optimization loop.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Awaitable
from dataclasses import dataclass
import time
import asyncio

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error
)

from .shared_utils import get_logger
from .step1_feature_preparation import ClusteringContext
from .step2_initial_clustering import InitialClusteringStep
from .iterative_optimization import IterativeOptimization
from .step8_validation import ValidationStep
from .step9_results_consolidation import ResultsConsolidationStep
from .step10_comprehensive_reporting import ComprehensiveReporter
from .risk_mitigation import RiskMitigationSystem
from .validation_framework import ClusteringValidator

@dataclass
class ClusteringResult:
    """Result from clustering service."""
    cluster_assignments: np.ndarray
    n_clusters: int
    metrics: Dict[str, Any]
    optimization_history: Dict[str, Any]
    validation_results: Dict[str, Any]
    execution_time: float
    convergence_status: str

class ClusteringService:
    """
    Main clustering service that interfaces with the ClusteringEngine.

    Responsibilities:
    - Interface with the ClusteringEngine
    - Run initial clustering (k-selection, warm start)
    - Manage iterative optimization loop (local → global → split)
    - Return cluster assignments + metrics
    """

    def __init__(self, verbose: bool = True) -> None:
        """Initialize the clustering service."""
        tprint("🚀 Initializing ClusteringService", "INFO")
        self.verbose = verbose
        self.logger = get_logger('ClusteringService')
        tprint_debug(f"Service verbose mode: {verbose}")

        # Initialize all components
        tprint("🔧 Initializing clustering components", "INFO")
        self.initial_clustering = InitialClusteringStep(verbose=verbose)
        tprint_debug("InitialClusteringStep initialized")
        self.iterative_optimizer = IterativeOptimization(verbose=verbose)
        tprint_debug("IterativeOptimization initialized")
        self.validator = ValidationStep(verbose=verbose)
        tprint_debug("ValidationStep initialized")
        self.results_consolidator = ResultsConsolidationStep(verbose=verbose)
        tprint_debug("ResultsConsolidationStep initialized")
        self.reporter = ComprehensiveReporter(verbose=verbose)
        tprint_debug("ComprehensiveReporter initialized")
        self.risk_mitigator = RiskMitigationSystem()
        tprint_debug("RiskMitigationSystem initialized")
        self.framework_validator = ClusteringValidator()
        tprint_debug("ClusteringValidator initialized")

        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {
            "total_execution_time": 0.0,
            "initial_clustering_time": 0.0,
            "optimization_time": 0.0,
            "validation_time": 0.0,
            "consolidation_time": 0.0,
            "optimization_rounds": 0,
            "convergence_achieved": False
        }
        tprint("✅ ClusteringService initialization completed", "SUCCESS")

    async def run_clustering(
        self,
        features: np.ndarray,
        market_data: pd.DataFrame,
        config: Any
    ) -> ClusteringResult:
        """
        Run the complete clustering pipeline.

        Args:
            features: Feature matrix for clustering
            market_data: Market data for context
            config: Configuration parameters

        Returns:
            ClusteringResult with assignments and metrics
        """
        try:
            start_time = time.time()
            tprint("🚀 Starting Clustering Service", "INFO")

            # Create clustering context
            context = ClusteringContext(
                original_features=features,
                market_data=market_data,
                original_feature_names=getattr(config, 'feature_names', None),
                feature_scores=getattr(config, 'feature_scores', {})
            )

            # Step 1: Initial clustering (k-selection, warm start)
            tprint("📊 Step 1: Initial Clustering", "INFO")
            initial_start = time.time()

            context = await self.initial_clustering.execute(context, config)
            optimal_k = context.optimal_k

            self.performance_metrics["initial_clustering_time"] = time.time() - initial_start

            # Step 2: Iterative optimization loop
            tprint("🔄 Step 2: Iterative Optimization", "INFO")
            optimization_start = time.time()

            # Run iterative optimization with risk mitigation
            context, optimization_history = await self._run_optimization_loop(
                context, config, max_iterations=100
            )

            self.performance_metrics["optimization_time"] = time.time() - optimization_start
            self.performance_metrics["optimization_rounds"] = len(optimization_history.get('rounds', []))

            # Step 3: Validation
            tprint("✅ Step 3: Validation", "INFO")
            validation_start = time.time()

            context = await self.validator.execute(context, config)
            validation_results = context.validation_results

            self.performance_metrics["validation_time"] = time.time() - validation_start

            # Step 4: Results consolidation
            tprint("📋 Step 4: Results Consolidation", "INFO")
            consolidation_start = time.time()

            final_results = await self.results_consolidator.execute(context, config)

            self.performance_metrics["consolidation_time"] = time.time() - consolidation_start

            # Step 5: Generate comprehensive report
            tprint("📊 Step 5: Comprehensive Reporting", "INFO")
            comprehensive_report = self.reporter.generate_comprehensive_report(
                context, final_results, market_data
            )

            # Record total execution time
            total_time = time.time() - start_time
            self.performance_metrics["total_execution_time"] = total_time

            # Determine convergence status
            convergence_status = self._determine_convergence_status(optimization_history)

            # Create final result
            result = ClusteringResult(
                cluster_assignments=context.optimized_assignments,
                n_clusters=optimal_k,
                metrics=final_results,
                optimization_history=optimization_history,
                validation_results=validation_results,
                execution_time=total_time,
                convergence_status=convergence_status
            )

            tprint(f"🎉 Clustering completed in {total_time:.2f}s with {optimal_k} clusters", "SUCCESS")
            return result

        except Exception as e:
            tprint(f"❌ Clustering service failed: {e}", "ERROR")
            raise ValueError(f"Clustering service failed: {e}")

    async def _run_optimization_loop(
        self,
        context: ClusteringContext,
        config: Any,
        max_iterations: int = 100
    ) -> Tuple[ClusteringContext, Dict[str, Any]]:
        """
        Run the iterative optimization loop with risk mitigation.

        Args:
            context: Current clustering context
            config: Configuration parameters
            max_iterations: Maximum optimization iterations

        Returns:
            Updated context and optimization history
        """
        try:
            tprint("🔄 Starting iterative optimization loop", "INFO")

            # Initialize optimization history
            optimization_history = {
                "rounds": [],
                "initial_objective": 0.0,
                "final_objective": 0.0,
                "total_moves": 0,
                "convergence_round": None,
                "risk_violations": 0
            }

            # Get initial objective value
            initial_stats = await self._calculate_clustering_stats(
                context.optimized_features, context.initial_assignments
            )
            optimization_history["initial_objective"] = initial_stats.get_objective_value()

            # Run optimization loop
            for iteration in range(max_iterations):
                round_start = time.time()

                # Run single optimization round
                context, round_results = await self.iterative_optimizer.execute_optimization_round(
                    context, config, iteration
                )

                # Record round results
                round_info = {
                    "iteration": iteration,
                    "execution_time": time.time() - round_start,
                    "objective_value": round_results.get("final_objective", 0.0),
                    "moves_accepted": round_results.get("moves_accepted", 0),
                    "local_moves": round_results.get("local_moves", 0),
                    "global_moves": round_results.get("global_moves", 0),
                    "splits_performed": round_results.get("splits_performed", 0),
                    "cv_ratio": round_results.get("cv_ratio", 0.0),
                    "balance_score": round_results.get("balance_score", 0.0),
                    "silhouette_score": round_results.get("silhouette_score", 0.0)
                }
                optimization_history["rounds"].append(round_info)

                # Update total moves
                optimization_history["total_moves"] += round_results.get("moves_accepted", 0)

                # Check for convergence
                if self._check_convergence(optimization_history, iteration):
                    optimization_history["convergence_round"] = iteration
                    tprint(f"🎯 Convergence achieved at iteration {iteration}", "SUCCESS")
                    break

                # Check risk violations
                if round_results.get("risk_violations", 0) > 0:
                    optimization_history["risk_violations"] += round_results["risk_violations"]

            # Get final objective value
            final_assignments = context.optimized_assignments
            final_stats = await self._calculate_clustering_stats(
                context.optimized_features, final_assignments
            )
            optimization_history["final_objective"] = final_stats.get_objective_value()

            # Print comprehensive final metrics
            self.iterative_optimizer._print_final_metrics(
                context.optimized_features, final_stats
            )

            return context, optimization_history

        except Exception as e:
            tprint(f"❌ Optimization loop failed: {e}", "ERROR")
            raise

    async def _calculate_clustering_stats(self, features: np.ndarray, assignments: np.ndarray):
        """Calculate clustering statistics for objective evaluation."""
        from .risk_mitigation import ClusteringStats
        return ClusteringStats(features, assignments)

    def _check_convergence(self, optimization_history: Dict[str, Any], current_iteration: int) -> bool:
        """Check if optimization has converged."""
        try:
            rounds = optimization_history["rounds"]
            if len(rounds) < 5:  # Need at least 5 rounds to check convergence
                return False

            # Check if objective function has stabilized
            recent_objectives = [r["objective_value"] for r in rounds[-5:]]
            max_recent = max(recent_objectives)
            min_recent = min(recent_objectives)

            # Converged if objective variation is small (< 1% relative change)
            if max_recent > 0:
                relative_variation = (max_recent - min_recent) / max_recent
                return relative_variation < 0.01

            return False

        except Exception as e:
            tprint(f"Convergence check failed: {e}", "WARNING")
            return False

    def _determine_convergence_status(self, optimization_history: Dict[str, Any]) -> str:
        """Determine convergence status from optimization history."""
        if optimization_history.get("convergence_round") is not None:
            return "converged"
        elif optimization_history.get("risk_violations", 0) > 5:
            return "risk_limited"
        else:
            return "max_iterations_reached"

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the clustering service."""
        return {
            "performance_metrics": self.performance_metrics,
            "optimization_efficiency": self.performance_metrics["optimization_rounds"] / max(1, self.performance_metrics["optimization_time"]),
            "convergence_rate": 1.0 if self.performance_metrics["convergence_achieved"] else 0.0
        }

    async def run_initial_clustering_only(
        self,
        features: np.ndarray,
        market_data: pd.DataFrame,
        config: Any
    ) -> Tuple[np.ndarray, int]:
        """
        Run only initial clustering (k-selection, warm start).

        Args:
            features: Feature matrix for clustering
            market_data: Market data for context
            config: Configuration parameters

        Returns:
            Tuple of (cluster_assignments, optimal_k)
        """
        try:
            tprint("🔍 Running initial clustering only", "INFO")
            tprint(f"🔍 DEBUG: features shape: {features.shape}, market_data shape: {market_data.shape}", "DEBUG")

            # Create clustering context
            context = ClusteringContext(
                original_features=features,
                market_data=market_data,
                original_feature_names=getattr(config, 'feature_names', None),
                feature_scores=getattr(config, 'feature_scores', {}),
                optimized_features=features  # Set optimized_features to the input features
            )
            tprint("🔍 DEBUG: ClusteringContext created", "DEBUG")

            # Run initial clustering
            tprint("🔍 DEBUG: About to call initial_clustering.execute", "DEBUG")
            context = await self.initial_clustering.execute(context, config)
            tprint("✅ DEBUG: initial_clustering.execute completed", "DEBUG")

            tprint(f"✅ DEBUG: Returning assignments shape: {context.initial_assignments.shape}, optimal_k: {context.optimal_k}", "DEBUG")
            return context.initial_assignments, context.optimal_k

        except Exception as e:
            tprint(f"❌ Initial clustering failed: {e}", "ERROR")
            raise

    def validate_clustering_quality(self, assignments: np.ndarray, features: np.ndarray) -> Dict[str, Any]:
        """
        Validate clustering quality metrics.

        Args:
            assignments: Cluster assignments
            features: Feature matrix

        Returns:
            Quality metrics dictionary
        """
        try:
            from sklearn.metrics import silhouette_score, davies_bouldin_score

            quality_metrics = {}

            # Silhouette score
            try:
                # Check for valid data before calculating silhouette score
                if len(features) == 0 or len(assignments) == 0:
                    quality_metrics["silhouette_score"] = 0.0
                elif len(np.unique(assignments)) < 2:
                    quality_metrics["silhouette_score"] = 0.0
                elif features.ndim == 1:
                    # Reshape 1D array to 2D for sklearn compatibility
                    features_2d = features.reshape(-1, 1)
                    quality_metrics["silhouette_score"] = float(silhouette_score(features_2d, assignments))
                else:
                    quality_metrics["silhouette_score"] = float(silhouette_score(features, assignments))
            except Exception as e:
                quality_metrics["silhouette_score"] = 0.0
                tprint(f"Silhouette score calculation failed: {e}", "WARNING")

            # Davies-Bouldin index
            try:
                # Check for valid data before calculating Davies-Bouldin score
                if len(features) == 0 or len(assignments) == 0:
                    quality_metrics["davies_bouldin_score"] = float('inf')
                elif len(np.unique(assignments)) < 2:
                    quality_metrics["davies_bouldin_score"] = float('inf')
                elif features.ndim == 1:
                    # Reshape 1D array to 2D for sklearn compatibility
                    features_2d = features.reshape(-1, 1)
                    quality_metrics["davies_bouldin_score"] = float(davies_bouldin_score(features_2d, assignments))
                else:
                    quality_metrics["davies_bouldin_score"] = float(davies_bouldin_score(features, assignments))
            except Exception as e:
                quality_metrics["davies_bouldin_score"] = float('inf')
                tprint(f"Davies-Bouldin score calculation failed: {e}", "WARNING")

            # Cluster balance
            unique, counts = np.unique(assignments, return_counts=True)
            quality_metrics["n_clusters"] = len(unique)
            quality_metrics["min_cluster_size"] = int(np.min(counts))
            quality_metrics["max_cluster_size"] = int(np.max(counts))
            quality_metrics["cluster_balance"] = float(np.std(counts) / np.mean(counts))

            return quality_metrics

        except Exception as e:
            tprint(f"❌ Quality validation failed: {e}", "ERROR")
            return {"error": str(e)}
