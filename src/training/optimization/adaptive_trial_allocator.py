# src/training/optimization/adaptive_trial_allocator.py

"""Adaptive Trial Allocator for intelligent trial distribution based on parameter importance."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
    """Allocates trials based on parameter importance and performance."""

    def __init__(self, config: Dict[str, Any]) -> None:
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
        parameters: Dict[str, Any],
    ) -> Dict[str, float]:
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

        except Exception as e:
            self.logger.error(error(f"Error calculating parameter importance: {e}"))
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

        except Exception as e:
            self.logger.warning(
                warning(f"Error getting base importance for {param_path}: {e}")
            )
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
    def allocate_trials_adaptively(
        self,
        parameters: Dict[str, Any],
    ) -> Dict[str, int]:
        """Allocate trials based on parameter importance."""
        try:
            # Calculate parameter importance
            importance_scores = self.calculate_parameter_importance(parameters)

            # Get importance scores
            importance_scores = self.calculate_parameter_importance(parameters)

            # Calculate trial allocation
            total_trials = self.allocation_config.total_trials
            min_trials = self.allocation_config.min_trials_per_parameter
            max_trials = self.allocation_config.max_trials_per_parameter

            allocation = {}
            remaining_trials = total_trials

            # Allocate trials based on importance
            for param_path, importance in importance_scores.items():
                # Calculate trials for this parameter
                trials = int(
                    importance * total_trials * 0.8
                )  # Reserve 20% for dynamic allocation
                trials = max(min_trials, min(trials, max_trials))
                trials = min(trials, remaining_trials)

                allocation[param_path] = trials
                remaining_trials -= trials

            # Distribute remaining trials
            if remaining_trials > 0:
                # Give remaining trials to highest importance parameters
                sorted_params = sorted(
                    importance_scores.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )

                for param_path, _ in sorted_params:
                    if remaining_trials <= 0:
                        break

                    additional_trials = min(
                        remaining_trials,
                        max_trials - allocation.get(param_path, 0),
                    )
                    allocation[param_path] = (
                        allocation.get(param_path, 0) + additional_trials
                    )
                    remaining_trials -= additional_trials

            # Record allocation
            self.allocation_history.append(
                {
                    "timestamp": pd.Timestamp.now(),
                    "allocation": allocation.copy(),
                    "importance_scores": importance_scores.copy(),
                    "total_trials": total_trials,
                }
            )

            self.logger.info(
                f"Allocated {total_trials} trials across {len(allocation)} parameters",
            )
            return allocation

        except Exception as e:
            self.logger.error(error(f"Error allocating trials: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance tracking",
    )
    def track_parameter_performance(
        self,
        param_path: str,
        performance: float,
    ) -> None:
        """Track performance for a specific parameter."""
        try:
            self.parameter_performance[param_path].append(performance)

            # Keep only recent performance data (last 100 measurements)
            if len(self.parameter_performance[param_path]) > 100:
                self.parameter_performance[param_path] = self.parameter_performance[
                    param_path
                ][-100:]

            self.logger.debug(f"Tracked performance for {param_path}: {performance}")

        except Exception as e:
            self.logger.error(error(f"Error tracking parameter performance: {e}"))

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="dynamic reallocation check",
    )
    def should_reallocate_trials(self, current_allocation: Dict[str, int]) -> bool:
        """Check if dynamic reallocation is needed."""
        try:
            if not self.allocation_config.dynamic_allocation:
                return False

            # Check if we have enough performance data
            if len(self.allocation_history) < 2:
                return False

            # Compare current allocation with previous
            previous_allocation = self.allocation_history[-2]["allocation"]

            # Calculate allocation difference
            total_diff = 0
            for param_path in current_allocation:
                current_trials = current_allocation.get(param_path, 0)
                previous_trials = previous_allocation.get(param_path, 0)
                total_diff += abs(current_trials - previous_trials)

            # Check if difference exceeds threshold
            threshold = self.allocation_config.reallocation_threshold * sum(
                current_allocation.values()
            )

            return total_diff > threshold

        except Exception as e:
            self.logger.error(error(f"Error checking reallocation: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="optimal allocation calculation",
    )
    def calculate_optimal_allocation(
        self,
        parameters: Dict[str, Any],
    ) -> Dict[str, int]:
        """Calculate optimal trial allocation based on historical performance."""
        try:
            # Get importance scores
            importance_scores = self.calculate_parameter_importance(parameters)

            # Calculate optimal allocation using historical data
            total_trials = self.allocation_config.total_trials
            allocation = {}

            # Use performance data to adjust allocation
            for param_path, importance in importance_scores.items():
                base_trials = int(importance * total_trials * 0.6)

                # Adjust based on performance variance
                if param_path in self.parameter_performance:
                    performances = self.parameter_performance[param_path]
                    if performances:
                        variance = np.var(performances)
                        # More variance = more trials needed
                        variance_adjustment = min(variance * 5, 0.5)
                        base_trials = int(base_trials * (1 + variance_adjustment))

                allocation[param_path] = max(
                    self.allocation_config.min_trials_per_parameter,
                    min(base_trials, self.allocation_config.max_trials_per_parameter),
                )

            return allocation

        except Exception as e:
            self.logger.error(error(f"Error calculating optimal allocation: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="allocation statistics",
    )
    def get_allocation_statistics(self) -> Optional[Dict[str, Any]]:
        """Get allocation statistics."""
        try:
            if not self.allocation_history:
                return {"message": "No allocation history available"}

            summary = {}

            # Calculate statistics
            total_allocations = len(self.allocation_history)
            avg_trials_per_allocation = np.mean(
                [
                    sum(allocation["allocation"].values())
                    for allocation in self.allocation_history
                ]
            )

            # Parameter usage statistics
            param_usage = defaultdict(int)
            for allocation in self.allocation_history:
                for param_path in allocation["allocation"]:
                    param_usage[param_path] += 1

            summary.update(
                {
                    "total_allocations": total_allocations,
                    "avg_trials_per_allocation": avg_trials_per_allocation,
                    "most_used_parameters": dict(
                        sorted(param_usage.items(), key=lambda x: x[1], reverse=True)[
                            :10
                        ]
                    ),
                    "parameter_performance_count": len(self.parameter_performance),
                }
            )

            return summary

        except Exception as e:
            self.logger.error(error(f"Error getting allocation statistics: {e}"))
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="parameter performance summary",
    )
    def get_parameter_performance_summary(self) -> Optional[Dict[str, Any]]:
        """Get parameter performance summary."""
        try:
            summary = {}

            for param_path, performances in self.parameter_performance.items():
                if performances:
                    summary[param_path] = {
                        "count": len(performances),
                        "mean": np.mean(performances),
                        "std": np.std(performances),
                        "min": np.min(performances),
                        "max": np.max(performances),
                        "variance": np.var(performances),
                    }

            return summary

        except Exception as e:
            self.logger.error(
                error(f"Error getting parameter performance summary: {e}")
            )
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="trial allocation validation",
    )
    def validate_trial_allocation(self, allocation: Dict[str, int]) -> bool:
        """Validate trial allocation."""
        try:
            # Check total trials
            total_trials = sum(allocation.values())

            if total_trials > self.allocation_config.total_trials:
                self.logger.warning(
                    f"Allocation exceeds total trials: {total_trials} > {self.allocation_config.total_trials}",
                )
                return False

            # Check minimum trials per parameter
            for param_path, trials in allocation.items():
                if trials < self.allocation_config.min_trials_per_parameter:
                    self.logger.warning(
                        f"Parameter {param_path} has insufficient trials: {trials} < {self.allocation_config.min_trials_per_parameter}",
                    )
                    return False

            return True

        except Exception as e:
            self.logger.error(error(f"Error validating trial allocation: {e}"))
            return False

    def reset_allocation_history(self) -> None:
        """Reset allocation history."""
        self.allocation_history.clear()
        self.parameter_performance.clear()
        self.parameter_importance.clear()
        self.logger.info("Reset allocation history")


def create_adaptive_trial_allocator(
    config: Optional[Dict[str, Any]] = None,
) -> AdaptiveTrialAllocator:
    """Create an adaptive trial allocator instance.

    Args:
        config: Optional configuration dictionary

    Returns:
        AdaptiveTrialAllocator instance

    """
    if config is None:
        config = {}

    return AdaptiveTrialAllocator(config)
