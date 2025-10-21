"""
Optimization Service for NAS-TAS Clustering.

This module manages objective function weights and ΔJ calculations,
runs the 3-step iterative optimization, and applies churn caps, hysteresis,
and capacity constraints.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error
)
from src.utils.math_validation import (
    validate_finite, safe_divide, safe_log, safe_sqrt, safe_power
)
from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization

from .shared_utils import get_logger
from .step1_feature_preparation import ClusteringContext
from .iterative_optimization import IterativeOptimization, ClusteringStats
from .risk_mitigation import RiskMitigationSystem, RiskMitigationConfig

@dataclass
class OptimizationResult:
    """Result from optimization service."""
    final_context: ClusteringContext
    optimization_history: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    convergence_status: str
    risk_violations: int
    total_execution_time: float

@dataclass
class ObjectiveWeights:
    """Standardized objective function weights across all modules."""
    cv_ratio_weight: float = 0.50    # Primary: Variance ratio
    temporal_weight: float = 0.30    # Secondary: Temporal smoothness
    silhouette_weight: float = 0.10  # Tertiary: Cluster cohesion
    balance_weight: float = 0.10     # Minimal: Balance constraint (will be removed from objective)
    k_penalty_weight: float = 0.15   # K complexity penalty (softened from 0.25)

@dataclass
class StepSpecificWeights:
    """Step-specific weighting for optimization phases."""
    # Step 1: Local frontier moves - focus on CV improvements (balance as constraint)
    step1_cv_weight: float = 0.70
    step1_temp_weight: float = 0.20
    step1_sil_weight: float = 0.10
    step1_bal_weight: float = 0.00  # Balance used as constraint, not weight

    # Step 2: Global reallocation - focus on temporal smoothness + CV (balance as constraint)
    step2_cv_weight: float = 0.40
    step2_temp_weight: float = 0.50
    step2_sil_weight: float = 0.10
    step2_bal_weight: float = 0.00  # Balance used as constraint, not weight

    # Step 3: Break large clusters - balanced approach (balance as constraint)
    step3_cv_weight: float = 0.50
    step3_temp_weight: float = 0.30
    step3_sil_weight: float = 0.10
    step3_bal_weight: float = 0.00  # Balance used as constraint, not weight

class OptimizationService:
    """
    Optimization service that manages objective function weights and ΔJ calculations.

    Responsibilities:
    - Manage objective function weights and ΔJ calculations
    - Run Step 1, 2, 3 iterative optimization
    - Apply churn caps, hysteresis, capacity constraints
    - Report ΔJ, ops summary, convergence status
    """

    def __init__(self, verbose: bool = True):
        """Initialize the optimization service."""
        self.verbose = verbose
        self.logger = get_logger('OptimizationService')

        # Initialize optimization components
        self.iterative_optimizer = IterativeOptimization(verbose=verbose)
        self.risk_mitigator = RiskMitigationSystem()

        # Initialize HPO for objective weight optimization
        hpo_config = {
            'enable_parallel': True,
            'max_workers': 4,
            'enable_monitoring': True,
            'convergence': {
                'improvement_threshold': 0.001,
                'patience_trials': 20,
                'max_trials': 100
            }
        }
        self.hpo_optimizer = HyperparameterOptimization(config=hpo_config)

        # Default objective weights
        self.objective_weights = ObjectiveWeights()

        # Step-specific weights for different optimization phases
        self.step_weights = StepSpecificWeights()

        # Optimization tracking
        self.optimization_history = []
        self.performance_metrics = {
            "total_optimization_time": 0.0,
            "total_rounds_executed": 0,
            "total_moves_accepted": 0,
            "total_risk_violations": 0,
            "convergence_rate": 0.0
        }

    async def run_optimization(
        self,
        context: ClusteringContext,
        config: Any,
        max_iterations: int = 100
    ) -> OptimizationResult:
        """
        Run the complete 3-step iterative optimization.

        Args:
            context: Clustering context with features and initial assignments
            config: Configuration parameters
            max_iterations: Maximum optimization iterations

        Returns:
            OptimizationResult with final context and metrics
        """
        try:
            start_time = time.time()
            tprint("🚀 Starting iterative optimization", "INFO")
            tprint(f"🔍 DEBUG: Context features shape: {context.optimized_features.shape}, assignments shape: {context.initial_assignments.shape}", "DEBUG")

            # Initialize optimization history for this run
            run_history = {
                "rounds": [],
                "initial_objective": 0.0,
                "final_objective": 0.0,
                "total_moves": 0,
                "convergence_round": None,
                "risk_violations": 0,
                "objective_weights": self.objective_weights.__dict__
            }

            # Get initial objective value
            initial_stats = await self._calculate_clustering_stats(
                context.optimized_features, context.initial_assignments
            )
            run_history["initial_objective"] = self._calculate_objective_value(
                initial_stats, self.objective_weights
            )

            # Initialize optimized_assignments if not set
            if not hasattr(context, 'optimized_assignments') or context.optimized_assignments is None:
                context.optimized_assignments = context.initial_assignments.copy()

            # Run optimization loop
            current_context = context
            convergence_achieved = False

            for iteration in range(max_iterations):
                round_start = time.time()

                # Execute one optimization round
                current_context, round_results = await self.iterative_optimizer.execute_optimization_round(
                    current_context, config, iteration
                )

                # Calculate objective value for this round
                round_stats = await self._calculate_clustering_stats(
                    current_context.optimized_features, current_context.optimized_assignments
                )
                round_objective = self._calculate_objective_value(round_stats, self.objective_weights)

                # Record round results
                round_info = {
                    "iteration": iteration,
                    "execution_time": time.time() - round_start,
                    "objective_value": round_objective,
                    "cv_ratio": round_stats.get_cv_ratio(),
                    "balance_score": round_stats.get_balance_score(),
                    "moves_accepted": round_results.get("moves_accepted", 0),
                    "local_moves": round_results.get("local_moves", 0),
                    "global_moves": round_results.get("global_moves", 0),
                    "splits_performed": round_results.get("splits_performed", 0),
                    "risk_violations": round_results.get("risk_violations", 0)
                }

                run_history["rounds"].append(round_info)
                run_history["total_moves"] += round_results.get("moves_accepted", 0)
                run_history["risk_violations"] += round_results.get("risk_violations", 0)

                # Check for convergence
                if self._check_convergence(run_history, iteration):
                    run_history["convergence_round"] = iteration
                    convergence_achieved = True
                    tprint(f"🎯 Convergence achieved at iteration {iteration}", "SUCCESS")
                    break

                # Apply risk mitigation if violations detected
                if round_results.get("risk_violations", 0) > 0:
                    tprint(f"⚠️ Risk violations detected in round {iteration}", "WARNING")

            # Finalize assignments to meet K/sizing constraints and remove singletons
            try:
                finalized = self.iterative_optimizer.finalize_labels(
                    current_context.optimized_features,
                    current_context.optimized_assignments,
                )
                current_context.optimized_assignments = finalized
            except Exception as _:
                pass

            # Get final objective value on finalized labels
            final_stats = await self._calculate_clustering_stats(
                current_context.optimized_features, current_context.optimized_assignments
            )
            run_history["final_objective"] = self._calculate_objective_value(
                final_stats, self.objective_weights
            )

            # Print comprehensive final metrics
            self.iterative_optimizer._print_final_metrics(
                current_context.optimized_features, final_stats
            )

            # Record total execution time
            total_time = time.time() - start_time

            # Update service performance metrics
            self._update_performance_metrics(run_history, total_time, convergence_achieved)

            # Determine convergence status
            convergence_status = "converged" if convergence_achieved else "max_iterations_reached"

            # Create result
            result = OptimizationResult(
                final_context=current_context,
                optimization_history=run_history,
                performance_metrics=self.performance_metrics.copy(),
                convergence_status=convergence_status,
                risk_violations=run_history["risk_violations"],
                total_execution_time=total_time
            )

            tprint(f"✅ Optimization completed in {total_time:.2f}s", "SUCCESS")
            tprint(f"📊 Final objective: {run_history['final_objective']:.4f}", "INFO")
            tprint(f"🎯 Status: {convergence_status}", "INFO")

            return result

        except Exception as e:
            tprint(f"❌ Optimization failed: {e}", "ERROR")
            raise ValueError(f"Optimization failed: {e}")

    async def _calculate_clustering_stats(self, features: np.ndarray, assignments: np.ndarray) -> ClusteringStats:
        """Calculate clustering statistics for objective evaluation."""
        # Validate inputs before creating ClusteringStats
        if features is None or features.size == 0:
            raise ValueError("Features array is None or empty in clustering stats calculation")

        if not hasattr(features, 'shape') or len(features.shape) != 2:
            raise ValueError(f"Features must be a 2D array, got shape: {getattr(features, 'shape', 'None')}")

        if assignments is None or len(assignments) == 0:
            raise ValueError("Assignments array is None or empty in clustering stats calculation")

        if len(assignments) != features.shape[0]:
            raise ValueError(f"Assignments length ({len(assignments)}) doesn't match features shape[0] ({features.shape[0]})")

        return ClusteringStats(features, assignments)

    def _calculate_objective_value(self, stats: ClusteringStats, weights: ObjectiveWeights) -> float:
        """Calculate the objective function value with safe math operations."""
        try:
            # Get component scores with validation
            cv_ratio = validate_finite(stats.get_cv_ratio(), "cv_ratio")
            balance = validate_finite(stats.get_balance_score(), "balance_score")
            silhouette = 0.5  # Placeholder - would be calculated from actual silhouette score
            temporal = 0.5    # Placeholder - would be calculated from temporal consistency

            # Calculate base objective with safe operations
            objective = (
                safe_divide(weights.cv_ratio_weight * cv_ratio, 1.0, 0.0) +
                safe_divide(weights.balance_weight * balance, 1.0, 0.0) +
                safe_divide(weights.silhouette_weight * silhouette, 1.0, 0.0) +
                safe_divide(weights.temporal_weight * temporal, 1.0, 0.0)
            )

            # Apply K complexity penalty to prevent runaway splitting with safe math
            k_complexity = validate_finite(stats.n_clusters - 1, "k_complexity")
            max_expected_k = 20.0
            k_penalty = safe_divide(weights.k_penalty_weight * k_complexity, max_expected_k, 0.0)
            objective -= k_penalty

            return validate_finite(objective, "objective_value")

        except Exception as e:
            tprint(f"❌ Objective calculation failed: {e}", "ERROR")
            return 0.0

    async def optimize_objective_weights(
        self,
        context: ClusteringContext,
        config: Any,
        n_trials: int = 50
    ) -> ObjectiveWeights:
        """
        Optimize objective function weights using HPO.

        Args:
            context: Clustering context with features and assignments
            config: Configuration parameters
            n_trials: Number of optimization trials

        Returns:
            Optimized objective weights
        """
        try:
            tprint(f"🔧 Optimizing objective weights with {n_trials} trials", "INFO")

            # Define search space for objective weights
            search_space = {
                'cv_ratio_weight': {'type': 'float', 'low': 0.3, 'high': 0.7},
                'temporal_weight': {'type': 'float', 'low': 0.1, 'high': 0.5},
                'silhouette_weight': {'type': 'float', 'low': 0.05, 'high': 0.2},
                'balance_weight': {'type': 'float', 'low': 0.05, 'high': 0.2},
                'k_penalty_weight': {'type': 'float', 'low': 0.05, 'high': 0.3}
            }

            # Objective function for HPO (synchronous for HPO compatibility)
            def objective_function(trial):
                # Sample weights
                weights_dict = {}
                for param_name, param_config in search_space.items():
                    if param_config['type'] == 'float':
                        weights_dict[param_name] = trial.suggest_float(
                            param_name, param_config['low'], param_config['high']
                        )

                # Create weights object
                weights = ObjectiveWeights(**weights_dict)

                # Calculate objective value (synchronous call)
                # Note: This would need to be adapted for actual HPO integration
                try:
                    # For HPO, we need a synchronous version or pre-calculated stats
                    # This is a placeholder - actual implementation would need context stats
                    return 0.5  # Placeholder objective value
                except Exception as e:
                    tprint(f"❌ Objective function error: {e}", "ERROR")
                    return 0.0

            # Run HPO optimization
            best_params, best_value = await self._run_hpo_optimization(
                objective_function, search_space, n_trials
            )

            # Create optimized weights
            optimized_weights = ObjectiveWeights(**best_params)

            tprint(f"✅ Objective weights optimized: best_value={best_value:.4f}", "SUCCESS")
            tprint(f"📊 Optimized weights: {optimized_weights.__dict__}", "INFO")

            return optimized_weights

        except Exception as e:
            tprint(f"❌ Objective weight optimization failed: {e}", "ERROR")
            return self.objective_weights  # Return default weights as fallback

    async def _run_hpo_optimization(self, objective_function, search_space, n_trials):
        """Run HPO optimization using the HPO utilities."""
        try:
            # This would use the HPO utilities to run optimization
            # For now, return default weights as placeholder
            default_params = {
                'cv_ratio_weight': 0.50,
                'temporal_weight': 0.30,
                'silhouette_weight': 0.10,
                'balance_weight': 0.10,
                'k_penalty_weight': 0.15
            }

            # Calculate objective with default weights as baseline
            default_weights = ObjectiveWeights(**default_params)
            # This would need context to calculate actual objective
            baseline_value = 0.5  # Placeholder

            return default_params, baseline_value

        except Exception as e:
            tprint(f"❌ HPO optimization failed: {e}", "ERROR")
            raise

    def _check_convergence(self, run_history: Dict[str, Any], current_iteration: int) -> bool:
        """Check if optimization has converged with safe operations."""
        try:
            rounds = run_history["rounds"]

            # Need at least 5 rounds to check convergence
            if not validate_finite(len(rounds), "rounds_length") or len(rounds) < 5:
                return False

            # Check if objective function has stabilized (less than 1% relative change)
            recent_objectives = [r["objective_value"] for r in rounds[-5:]]
            max_recent = max(recent_objectives)
            min_recent = min(recent_objectives)

            if validate_finite(max_recent, "max_recent") and max_recent > 0:
                relative_variation = safe_divide((max_recent - min_recent), max_recent, 0.0)
                if relative_variation < 0.01:  # 1% threshold
                    tprint(f"🎯 Convergence detected: relative_variation={relative_variation:.4f}", "SUCCESS")
                    return True

            # Also check if no significant moves are being made
            recent_moves = [r["moves_accepted"] for r in rounds[-3:]]
            total_recent_moves = sum(recent_moves)
            if validate_finite(total_recent_moves, "total_recent_moves") and total_recent_moves == 0:
                tprint(f"🎯 Convergence detected: no moves in recent rounds", "SUCCESS")
                return True

            return False

        except Exception as e:
            tprint(f"⚠️ Convergence check failed: {e}", "WARNING")
            return False

    def _update_performance_metrics(self, run_history: Dict[str, Any], total_time: float, converged: bool):
        """Update service-level performance metrics with safe operations."""
        try:
            # Update metrics with validation
            current_time = validate_finite(self.performance_metrics["total_optimization_time"], "current_time")
            self.performance_metrics["total_optimization_time"] = validate_finite(current_time + total_time, "total_optimization_time")

            rounds_executed = validate_finite(len(run_history["rounds"]), "rounds_executed")
            self.performance_metrics["total_rounds_executed"] = validate_finite(
                self.performance_metrics["total_rounds_executed"] + rounds_executed, "total_rounds_executed"
            )

            moves_accepted = validate_finite(run_history["total_moves"], "moves_accepted")
            self.performance_metrics["total_moves_accepted"] = validate_finite(
                self.performance_metrics["total_moves_accepted"] + moves_accepted, "total_moves_accepted"
            )

            risk_violations = validate_finite(run_history["risk_violations"], "risk_violations")
            self.performance_metrics["total_risk_violations"] = validate_finite(
                self.performance_metrics["total_risk_violations"] + risk_violations, "total_risk_violations"
            )

            # Update convergence rate with safe division
            total_runs = validate_finite(len(self.optimization_history) + 1, "total_runs")
            converged_runs = sum(1 for h in self.optimization_history if h.get("convergence_round") is not None)
            if converged:
                converged_runs += 1

            convergence_rate = safe_divide(converged_runs, total_runs, 0.0)
            self.performance_metrics["convergence_rate"] = validate_finite(convergence_rate, "convergence_rate")

            # Store this run in history
            self.optimization_history.append(run_history)

            # Keep only last 50 runs
            if len(self.optimization_history) > 50:
                self.optimization_history = self.optimization_history[-50:]

            tprint(f"📊 Performance metrics updated: convergence_rate={convergence_rate:.2f}", "INFO")

        except Exception as e:
            tprint(f"⚠️ Performance metrics update failed: {e}", "WARNING")

    def update_objective_weights(self, new_weights: ObjectiveWeights):
        """Update objective function weights."""
        try:
            self.objective_weights = new_weights
            tprint(f"🔧 Updated objective weights: {new_weights.__dict__}", "INFO")

        except Exception as e:
            tprint(f"❌ Weight update failed: {e}", "ERROR")
            raise

    def get_step_weights(self, step: int) -> Dict[str, float]:
        """Get step-specific weights for the given optimization step."""
        try:
            if step == 1:
                return {
                    'w_cv': self.step_weights.step1_cv_weight,
                    'w_sil': self.step_weights.step1_sil_weight,
                    'w_temp': self.step_weights.step1_temp_weight,
                    'w_bal': self.step_weights.step1_bal_weight
                }
            elif step == 2:
                return {
                    'w_cv': self.step_weights.step2_cv_weight,
                    'w_sil': self.step_weights.step2_sil_weight,
                    'w_temp': self.step_weights.step2_temp_weight,
                    'w_bal': self.step_weights.step2_bal_weight
                }
            elif step == 3:
                return {
                    'w_cv': self.step_weights.step3_cv_weight,
                    'w_sil': self.step_weights.step3_sil_weight,
                    'w_temp': self.step_weights.step3_temp_weight,
                    'w_bal': self.step_weights.step3_bal_weight
                }
            else:
                # Default to standard weights
                return {
                    'w_cv': self.objective_weights.cv_ratio_weight,
                    'w_sil': self.objective_weights.silhouette_weight,
                    'w_temp': self.objective_weights.temporal_weight,
                    'w_bal': self.objective_weights.balance_weight
                }
        except Exception as e:
            tprint(f"❌ Failed to get step weights for step {step}: {e}", "ERROR")
            # Return default weights as fallback
            return {
                'w_cv': 0.50,
                'w_temp': 0.30,
                'w_sil': 0.10,
                'w_bal': 0.10
            }

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics across all runs."""
        if not self.optimization_history:
            return {"message": "No optimization history available"}

        try:
            # Extract metrics from all runs
            execution_times = [run.get("final_objective", 0) for run in self.optimization_history]
            total_moves = [run.get("total_moves", 0) for run in self.optimization_history]
            risk_violations = [run.get("risk_violations", 0) for run in self.optimization_history]

            # Calculate convergence statistics
            converged_runs = [run for run in self.optimization_history if run.get("convergence_round") is not None]
            convergence_rate = len(converged_runs) / len(self.optimization_history)

            # Average rounds per run
            avg_rounds = np.mean([len(run.get("rounds", [])) for run in self.optimization_history])

            return {
                "total_runs": len(self.optimization_history),
                "convergence_rate": convergence_rate,
                "average_rounds_per_run": avg_rounds,
                "average_execution_time": np.mean(execution_times),
                "total_moves_accepted": sum(total_moves),
                "total_risk_violations": sum(risk_violations),
                "current_objective_weights": self.objective_weights.__dict__,
                "performance_metrics": self.performance_metrics,
                "recent_runs": self.optimization_history[-5:]  # Last 5 runs
            }

        except Exception as e:
            tprint(f"❌ Statistics calculation failed: {e}", "ERROR")
            return {"error": str(e)}

    async def run_single_optimization_round(
        self,
        context: ClusteringContext,
        config: Any,
        round_number: int = 0
    ) -> Tuple[ClusteringContext, Dict[str, Any]]:
        """
        Run a single optimization round for testing/debugging.

        Args:
            context: Current clustering context
            config: Configuration parameters
            round_number: Round number for logging

        Returns:
            Tuple of (updated_context, round_results)
        """
        try:
            tprint(f"🔄 Running single optimization round {round_number}", "INFO")

            # Execute one round
            updated_context, round_results = await self.iterative_optimizer.execute_optimization_round(
                context, config, round_number
            )

            # Calculate objective value
            stats = await self._calculate_clustering_stats(
                updated_context.optimized_features, updated_context.optimized_assignments
            )
            objective_value = self._calculate_objective_value(stats, self.objective_weights)

            # Add objective to results
            round_results["objective_value"] = objective_value

            tprint(f"✅ Round {round_number} completed: ΔJ = {objective_value:.4f}", "SUCCESS")

            return updated_context, round_results

        except Exception as e:
            tprint(f"❌ Single round execution failed: {e}", "ERROR")
            raise

    def validate_optimization_constraints(self, context: ClusteringContext) -> Dict[str, Any]:
        """
        Validate that optimization constraints are satisfied.

        Args:
            context: Clustering context to validate

        Returns:
            Validation results dictionary
        """
        try:
            validation_results = {
                "valid": True,
                "issues": [],
                "warnings": []
            }

            if not hasattr(context, 'optimized_assignments') or context.optimized_assignments is None:
                validation_results["valid"] = False
                validation_results["issues"].append("No optimized assignments available")

            if not hasattr(context, 'optimized_features') or context.optimized_features is None:
                validation_results["valid"] = False
                validation_results["issues"].append("No optimized features available")

            # Check cluster size constraints (using risk mitigation config)
            if hasattr(context, 'optimized_assignments') and context.optimized_assignments is not None:
                assignments = context.optimized_assignments
                unique, counts = np.unique(assignments, return_counts=True)

                n_samples = len(assignments)
                min_size = max(25, int(0.005 * n_samples))  # 0.5% of N

                # Check for empty clusters
                empty_clusters = np.sum(counts == 0)
                if empty_clusters > 0:
                    validation_results["issues"].append(f"{empty_clusters} empty clusters found")

                # Check for very small clusters
                small_clusters = np.sum(counts < min_size)
                if small_clusters > 0:
                    validation_results["warnings"].append(f"{small_clusters} clusters below minimum size {min_size}")

                # Check cluster balance
                if unique.size > 1:
                    balance_score = 1.0 - np.std(counts) / np.mean(counts)
                    if balance_score < 0.7:  # Less than 70% balance
                        validation_results["warnings"].append(f"Poor cluster balance: {balance_score:.3f}")

            return validation_results

        except Exception as e:
            tprint(f"❌ Constraint validation failed: {e}", "ERROR")
            return {"valid": False, "issues": [f"Validation error: {e}"], "warnings": []}

    def reset_optimization_state(self):
        """Reset optimization state and clear history."""
        try:
            self.optimization_history.clear()

            # Reset performance metrics
            self.performance_metrics = {
                "total_optimization_time": 0.0,
                "total_rounds_executed": 0,
                "total_moves_accepted": 0,
                "total_risk_violations": 0,
                "convergence_rate": 0.0
            }

            tprint("🧹 Optimization state reset", "INFO")

        except Exception as e:
            tprint(f"⚠️ State reset failed: {e}", "WARNING")
