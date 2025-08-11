# src/training/optimization/adaptive_trial_allocator.py

"""
Adaptive Trial Allocator for intelligent trial distribution based on parameter importance.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    warning,
)


@dataclass
class TrialAllocationConfig:
    """Configuration for adaptive trial allocation."""

    total_trials: int = 500
    min_trials_per_parameter: int = 10
    max_trials_per_parameter: int = 100
    importance_weight: float = 0.6
    performance_weight: float = 0.4
    dynamic_allocation: bool = True
    reallocation_threshold: float = 0.1


class AdaptiveTrialAllocator:
    """
    Allocates trials based on parameter importance and performance.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize adaptive trial allocator."""
        self.config = config
        self.logger = system_logger.getChild("AdaptiveTrialAllocator")
        self.allocation_config = TrialAllocationConfig(
            **config.get("trial_allocation_config", {}),
        )

        # Track allocation history
        self.allocation_history = []
        self.parameter_performance = defaultdict(list)
        self.parameter_importance = {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="parameter importance calculation",
    )
    def calculate_parameter_importance(
        self,
        parameters: dict[str, Any],
    ) -> dict[str, float]:
        """Calculate parameter importance based on various factors."""
        try:
            importance_scores = {}

            for param_path in parameters:
                # Base importance based on parameter category
                base_importance = self._get_base_importance(param_path)

                # Performance-based importance (if available)
                performance_importance = self._get_performance_importance(param_path)

                # Sensitivity-based importance
                sensitivity_importance = self._get_sensitivity_importance(param_path)

                # Combine importance scores
                total_importance = (
                    base_importance * 0.4
                    + performance_importance * 0.3
                    + sensitivity_importance * 0.3
                )

                importance_scores[param_path] = min(total_importance, 1.0)

            # Normalize importance scores
            if importance_scores:
                max_importance = max(importance_scores.values())
                if max_importance > 0:
                    importance_scores = {
                        k: v / max_importance for k, v in importance_scores.items()
                    }

            self.parameter_importance = importance_scores
            self.logger.info(
                f"Calculated importance for {len(importance_scores)} parameters",
            )
            return importance_scores

        except Exception:
            self.print(error("Error calculating parameter importance: {e}"))
            return {}

    def _get_base_importance(self, param_path: str) -> float:
        """Get base importance based on parameter category."""
        try:
            # Critical parameters get highest importance
            critical_params = [
                "confidence_thresholds.base_entry_threshold",
                "confidence_thresholds.position_close_threshold",
                "position_sizing_parameters.kelly_multiplier",
                "position_sizing_parameters.max_position_size",
                "stop_loss_parameters.stop_loss_atr_multiplier",
            ]

            # Important parameters get medium importance
            important_params = [
                "volatility_parameters.volatility_multiplier",
                "profit_taking_parameters.pt1_target_atr_multiplier",
                "ensemble_parameters.ensemble_method",
                "cooldown_parameters.base_cooldown_minutes",
            ]

            if param_path in critical_params:
                return 1.0
            if param_path in important_params:
                return 0.7
            if "threshold" in param_path.lower():
                return 0.6
            if "multiplier" in param_path.lower():
                return 0.5
            return 0.3

        except Exception:
            self.print(warning("Error getting base importance for {param_path}: {e}"))
            return 0.3

    def _get_performance_importance(self, param_path: str) -> float:
        """Get performance-based importance."""
        try:
            if param_path in self.parameter_performance:
                performances = self.parameter_performance[param_path]
                if performances:
                    # Higher variance in performance = higher importance
                    variance = np.var(performances)
                    return min(variance * 10, 1.0)  # Scale variance

            return 0.5  # Default importance

        except Exception as e:
            self.logger.warning(
                f"Error getting performance importance for {param_path}: {e}",
            )
            return 0.5

    def _get_sensitivity_importance(self, param_path: str) -> float:
        """Get sensitivity-based importance."""
        try:
            # Parameters that affect multiple components get higher importance
            if "confidence" in param_path.lower():
                return 0.8  # Confidence affects many decisions
            if "sizing" in param_path.lower() or "position" in param_path.lower():
                return 0.7  # Sizing affects risk and returns
            if "risk" in param_path.lower() or "stop_loss" in param_path.lower():
                return 0.6  # Risk parameters are important
            if "ensemble" in param_path.lower():
                return 0.5  # Ensemble parameters affect model combination
            return 0.3

        except Exception as e:
            self.logger.warning(
                f"Error getting sensitivity importance for {param_path}: {e}",
            )
            return 0.3

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="trial allocation",
    )
    async def allocate_trials_adaptively(
        self,
        parameters: dict[str, Any],
    ) -> dict[str, int]:
        """Allocate trials based on parameter importance."""
        try:
            # Calculate parameter importance
            importance_scores = self.calculate_parameter_importance(parameters)

            if not importance_scores:
                # Fallback to equal allocation
                equal_trials = self.allocation_config.total_trials // len(parameters)
                return dict.fromkeys(parameters, equal_trials)

            # Allocate trials proportionally to importance
            total_importance = sum(importance_scores.values())
            allocated_trials = {}

            for param, importance in importance_scores.items():
                # Calculate proportional allocation
                proportional_trials = int(
                    (importance / total_importance)
                    * self.allocation_config.total_trials,
                )

                # Apply min/max constraints
                trials = max(
                    self.allocation_config.min_trials_per_parameter,
                    min(
                        proportional_trials,
                        self.allocation_config.max_trials_per_parameter,
                    ),
                )

                allocated_trials[param] = trials

            # Ensure total trials constraint
            total_allocated = sum(allocated_trials.values())
            if total_allocated != self.allocation_config.total_trials:
                # Adjust allocation to match total
                adjustment_factor = (
                    self.allocation_config.total_trials / total_allocated
                )
                for param in allocated_trials:
                    allocated_trials[param] = int(
                        allocated_trials[param] * adjustment_factor,
                    )

            # Store allocation for history
            self.allocation_history.append(
                {
                    "timestamp": pd.Timestamp.now(),
                    "allocation": allocated_trials.copy(),
                    "importance_scores": importance_scores.copy(),
                },
            )

            self.logger.info(
                f"Allocated {self.allocation_config.total_trials} trials across {len(allocated_trials)} parameters",
            )
            return allocated_trials

        except Exception:
            self.print(error("Error allocating trials adaptively: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="performance tracking",
    )
    def track_parameter_performance(self, param_path: str, performance: float) -> bool:
        """Track performance for a specific parameter."""
        try:
            self.parameter_performance[param_path].append(performance)

            # Keep only recent performance data (last 20 trials)
            if len(self.parameter_performance[param_path]) > 20:
                self.parameter_performance[param_path] = self.parameter_performance[
                    param_path
                ][-20:]

            return True

        except Exception:
            self.print(warning("Error tracking performance for {param_path}: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="dynamic reallocation",
    )
    async def check_dynamic_reallocation(
        self,
        current_allocation: dict[str, int],
    ) -> bool:
        """Check if dynamic reallocation is needed."""
        try:
            if not self.allocation_config.dynamic_allocation:
                return False

            if len(self.allocation_history) < 2:
                return False

            # Compare current allocation with previous
            previous_allocation = self.allocation_history[-2]["allocation"]

            # Calculate allocation change
            total_change = 0
            for param in current_allocation:
                if param in previous_allocation:
                    change = abs(current_allocation[param] - previous_allocation[param])
                    total_change += change

            # Check if change exceeds threshold
            total_trials = sum(current_allocation.values())
            change_ratio = total_change / total_trials

            if change_ratio > self.allocation_config.reallocation_threshold:
                self.logger.info(
                    f"Dynamic reallocation triggered (change ratio: {change_ratio:.3f})",
                )
                return True

            return False

        except Exception:
            self.print(warning("Error checking dynamic reallocation: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="optimal allocation calculation",
    )
    def calculate_optimal_allocation(
        self,
        parameters: dict[str, Any],
    ) -> dict[str, int]:
        """Calculate optimal trial allocation based on historical performance."""
        try:
            # Get importance scores
            importance_scores = self.calculate_parameter_importance(parameters)

            # Calculate performance-based weights
            performance_weights = {}
            for param in parameters:
                if (
                    param in self.parameter_performance
                    and self.parameter_performance[param]
                ):
                    # Higher variance = more trials needed
                    variance = np.var(self.parameter_performance[param])
                    performance_weights[param] = min(variance * 5, 1.0)
                else:
                    performance_weights[param] = 0.5

            # Combine importance and performance
            combined_weights = {}
            for param in parameters:
                importance = importance_scores.get(param, 0.5)
                performance = performance_weights.get(param, 0.5)

                combined_weights[param] = (
                    importance * self.allocation_config.importance_weight
                    + performance * self.allocation_config.performance_weight
                )

            # Allocate trials
            total_weight = sum(combined_weights.values())
            allocated_trials = {}

            for param, weight in combined_weights.items():
                trials = int(
                    (weight / total_weight) * self.allocation_config.total_trials,
                )
                trials = max(self.allocation_config.min_trials_per_parameter, trials)
                trials = min(trials, self.allocation_config.max_trials_per_parameter)
                allocated_trials[param] = trials

            return allocated_trials

        except Exception:
            self.print(error("Error calculating optimal allocation: {e}"))
            return {}

    def get_allocation_statistics(self) -> dict[str, Any]:
        """Get allocation statistics."""
        try:
            if not self.allocation_history:
                return {"message": "No allocation history available"}

            latest_allocation = self.allocation_history[-1]["allocation"]

            return {
                "total_parameters": len(latest_allocation),
                "total_trials": sum(latest_allocation.values()),
                "average_trials_per_parameter": np.mean(
                    list(latest_allocation.values()),
                ),
                "min_trials_per_parameter": min(latest_allocation.values()),
                "max_trials_per_parameter": max(latest_allocation.values()),
                "allocation_history_length": len(self.allocation_history),
                "tracked_parameters": len(self.parameter_performance),
            }

        except Exception:
            self.print(error("Error getting allocation statistics: {e}"))
            return {}

    def get_parameter_performance_summary(self) -> dict[str, Any]:
        """Get parameter performance summary."""
        try:
            summary = {}

            for param, performances in self.parameter_performance.items():
                if performances:
                    summary[param] = {
                        "mean_performance": np.mean(performances),
                        "std_performance": np.std(performances),
                        "min_performance": np.min(performances),
                        "max_performance": np.max(performances),
                        "num_trials": len(performances),
                    }

            return summary

        except Exception:
            self.print(error("Error getting performance summary: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="allocation validation",
    )
    def validate_allocation(self, allocation: dict[str, int]) -> bool:
        """Validate trial allocation."""
        try:
            # Check total trials
            total_trials = sum(allocation.values())
            if total_trials != self.allocation_config.total_trials:
                self.logger.warning(
                    f"Total trials mismatch: {total_trials} vs {self.allocation_config.total_trials}",
                )
                return False

            # Check min/max constraints
            for trials in allocation.values():
                if trials < self.allocation_config.min_trials_per_parameter:
                    self.print(warning("Too few trials for {param}: {trials}"))
                    return False
                if trials > self.allocation_config.max_trials_per_parameter:
                    self.print(warning("Too many trials for {param}: {trials}"))
                    return False

            return True

        except Exception:
            self.print(error("Error validating allocation: {e}"))
            return False
