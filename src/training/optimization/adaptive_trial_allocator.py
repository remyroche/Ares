# src/training/optimization/adaptive_trial_allocator.py

"""Adaptive Trial Allocator for intelligent trial distribution based on parameter importance."""

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
    """Allocates trials based on parameter importance and performance."""

    def __init__(self, config: dict[str, Any]) -> None:
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

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="trial allocation",
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="performance tracking",
    )
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
