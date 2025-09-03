"""
Online Learning Manager Module.

This module manages online learning for model weighting based on performance,
adapting model weights dynamically based on their recent performance.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from src.core.decorators import handles_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import error


class OnlineLearningManager:
    """Manages online learning for model weighting based on performance."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize online learning manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("OnlineLearningManager")
        self.model_performances: Dict[str, List[float]] = defaultdict(list)
        self.model_weights: Dict[str, float] = {}
        self.learning_rate: float = config.get("learning_rate", 0.01)
        self.min_weight: float = config.get("min_weight", 0.1)
        self.max_weight: float = config.get("max_weight", 0.8)
        self.performance_window: int = config.get("performance_window", 100)

    @handles_errors(
        exceptions=(ValueError, TypeError, KeyError, ZeroDivisionError),
        default_return=None,
    )
    async def update_model_performance(self, model_id: str, performance: float) -> None:
        """
        Update model performance and recalculate weights.
        
        Args:
            model_id: Identifier for the model
            performance: Performance metric value
        """
        try:
            self.model_performances[model_id].append(performance)

            # Keep only recent performances
            if len(self.model_performances[model_id]) > self.performance_window:
                self.model_performances[model_id] = self.model_performances[model_id][-self.performance_window:]

            # Recalculate weights based on recent performance
            await self._recalculate_weights()

            self.logger.info(f"Updated performance for model {model_id}: {performance}")

        except Exception as e:
            self.logger.error(error(f"Error updating model performance: {e}"))

    @handles_errors(
        exceptions=(ValueError, TypeError, KeyError, ZeroDivisionError),
        default_return=None,
    )
    async def _recalculate_weights(self) -> None:
        """Recalculate model weights based on recent performance."""
        try:
            if not self.model_performances:
                return

            # Calculate average performance for each model
            avg_performances = {}
            for model_id, performances in self.model_performances.items():
                if performances:
                    avg_performances[model_id] = sum(performances) / len(performances)

            if not avg_performances:
                return

            # Normalize performances to calculate weights
            total_performance = sum(avg_performances.values())
            if total_performance > 0:
                for model_id, avg_perf in avg_performances.items():
                    # Calculate raw weight
                    raw_weight = avg_perf / total_performance

                    # Apply learning rate for gradual adjustment
                    current_weight = self.model_weights.get(model_id, raw_weight)
                    new_weight = current_weight + self.learning_rate * (raw_weight - current_weight)

                    # Apply min/max constraints
                    new_weight = max(self.min_weight, min(self.max_weight, new_weight))
                    self.model_weights[model_id] = new_weight

                # Renormalize to ensure weights sum to 1
                total_weight = sum(self.model_weights.values())
                if total_weight > 0:
                    for model_id in self.model_weights:
                        self.model_weights[model_id] /= total_weight

                self.logger.debug(f"Updated model weights: {self.model_weights}")

        except Exception as e:
            self.logger.error(error(f"Error recalculating weights: {e}"))

    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.model_weights.copy()

    def get_model_performances(self) -> Dict[str, List[float]]:
        """Get model performance history."""
        return {k: v.copy() for k, v in self.model_performances.items()}

    def reset_model(self, model_id: str) -> None:
        """
        Reset performance history for a specific model.
        
        Args:
            model_id: Identifier for the model to reset
        """
        if model_id in self.model_performances:
            self.model_performances[model_id] = []
            self.logger.info(f"Reset performance history for model {model_id}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the online learning system."""
        stats = {
            "num_models": len(self.model_performances),
            "model_weights": self.model_weights.copy(),
            "learning_rate": self.learning_rate,
            "performance_window": self.performance_window,
            "model_stats": {}
        }
        
        for model_id, performances in self.model_performances.items():
            if performances:
                stats["model_stats"][model_id] = {
                    "num_observations": len(performances),
                    "avg_performance": sum(performances) / len(performances),
                    "recent_performance": performances[-1] if performances else None,
                    "weight": self.model_weights.get(model_id, 0.0)
                }
        
        return stats