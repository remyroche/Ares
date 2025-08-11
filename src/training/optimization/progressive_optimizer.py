# src/training/optimization/progressive_optimizer.py

"""
Progressive Optimizer for efficient parameter optimization using tiered approach.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import optuna
import pandas as pd

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
    warning,
)


class OptimizationTier(Enum):
    """Enum for optimization tiers."""

    TIER_1_CRITICAL = "tier_1_critical"
    TIER_2_IMPORTANT = "tier_2_important"
    TIER_3_ADVANCED = "tier_3_advanced"


@dataclass
class ProgressiveConfig:
    """Configuration for progressive optimization."""

    # Tier-specific configurations
    tier1_trials: int = 100
    tier2_trials: int = 80
    tier3_trials: int = 60

    tier1_timeout_minutes: int = 30
    tier2_timeout_minutes: int = 90
    tier3_timeout_minutes: int = 180

    # Progressive settings
    enable_progressive_optimization: bool = True
    use_previous_results: bool = True
    adaptive_timeout: bool = True
    convergence_threshold: float = 0.01


class ProgressiveOptimizer:
    """
    Implements progressive optimization strategy for efficiency.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize progressive optimizer."""
        self.config = config
        self.logger = system_logger.getChild("ProgressiveOptimizer")
        self.progressive_config = ProgressiveConfig(
            **config.get("progressive_config", {}),
        )

        # Tier definitions
        self.tier1_critical = [
            "confidence_thresholds.base_entry_threshold",
            "confidence_thresholds.position_close_threshold",
            "position_sizing_parameters.kelly_multiplier",
            "position_sizing_parameters.max_position_size",
            "stop_loss_parameters.stop_loss_atr_multiplier",
        ]

        self.tier2_important = [
            "volatility_parameters.volatility_multiplier",
            "profit_taking_parameters.pt1_target_atr_multiplier",
            "ensemble_parameters.ensemble_method",
            "cooldown_parameters.base_cooldown_minutes",
            "drawdown_parameters.warning_drawdown_threshold",
        ]

        self.tier3_advanced = [
            "market_regime_parameters.regime_specific_constraints",
            "optimization_parameters.secondary_objectives",
            "feature_engineering_parameters.feature_selection_threshold",
            "monitoring_parameters.performance_alert_threshold",
        ]

        # Track optimization progress
        self.optimization_history = []
        self.tier_results = {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="tier 1 optimization",
    )
    async def optimize_tier1_parameters(
        self,
        initial_params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Optimize critical parameters first (10% of time)."""
        try:
            self.logger.info("Starting Tier 1 (Critical) optimization...")
            start_time = time.time()

            # Create objective function for tier 1
            def tier1_objective(trial):
                params = {}

                # Suggest critical parameters
                params["confidence_thresholds.base_entry_threshold"] = (
                    trial.suggest_float("base_entry_threshold", 0.5, 0.9)
                )
                params["confidence_thresholds.position_close_threshold"] = (
                    trial.suggest_float("position_close_threshold", 0.2, 0.6)
                )
                params["position_sizing_parameters.kelly_multiplier"] = (
                    trial.suggest_float("kelly_multiplier", 0.1, 0.5)
                )
                params["position_sizing_parameters.max_position_size"] = (
                    trial.suggest_float("max_position_size", 0.1, 0.4)
                )
                params["stop_loss_parameters.stop_loss_atr_multiplier"] = (
                    trial.suggest_float("stop_loss_atr_multiplier", 1.0, 4.0)
                )

                # Evaluate performance
                return self._evaluate_tier1_performance(params)

            # Create study with warm start if available
            study_name = f"tier1_optimization_{int(time.time())}"
            study = optuna.create_study(
                study_name=study_name,
                direction="maximize",
                storage=None,
            )

            # Add warm start if available
            if initial_params and self.progressive_config.use_previous_results:
                study.enqueue_trial(initial_params)
                self.logger.info("Added warm start trial for Tier 1")

            # Run optimization with timeout
            self.progressive_config.tier1_timeout_minutes * 60
            study.optimize(
                tier1_objective,
                n_trials=self.progressive_config.tier1_trials,
            )

            # Store results
            tier1_results = {
                "best_params": study.best_params,
                "best_value": study.best_value,
                "optimization_time": time.time() - start_time,
                "n_trials": len(study.trials),
                "tier": OptimizationTier.TIER_1_CRITICAL.value,
            }

            self.tier_results[OptimizationTier.TIER_1_CRITICAL.value] = tier1_results
            self.logger.info(
                f"Tier 1 optimization completed in {tier1_results['optimization_time']:.2f}s",
            )

            return tier1_results

        except Exception:
            self.print(error("Error in Tier 1 optimization: {e}"))
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="tier 2 optimization",
    )
    async def optimize_tier2_parameters(
        self,
        tier1_results: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Optimize secondary parameters (30% of time)."""
        try:
            self.logger.info("Starting Tier 2 (Important) optimization...")
            start_time = time.time()

            # Use tier 1 results as initial parameters
            initial_params = {}
            if tier1_results and self.progressive_config.use_previous_results:
                initial_params.update(tier1_results.get("best_params", {}))

            def tier2_objective(trial):
                params = initial_params.copy()

                # Suggest important parameters
                params["volatility_parameters.volatility_multiplier"] = (
                    trial.suggest_float("volatility_multiplier", 0.5, 2.0)
                )
                params["profit_taking_parameters.pt1_target_atr_multiplier"] = (
                    trial.suggest_float("pt1_target_atr_multiplier", 1.5, 4.0)
                )
                params["ensemble_parameters.ensemble_method"] = (
                    trial.suggest_categorical(
                        "ensemble_method",
                        ["confidence_weighted", "majority_vote", "weighted_average"],
                    )
                )
                params["cooldown_parameters.base_cooldown_minutes"] = trial.suggest_int(
                    "base_cooldown_minutes",
                    15,
                    120,
                )
                params["drawdown_parameters.warning_drawdown_threshold"] = (
                    trial.suggest_float("warning_drawdown_threshold", 0.05, 0.25)
                )

                # Evaluate performance
                return self._evaluate_tier2_performance(params)

            # Create study
            study_name = f"tier2_optimization_{int(time.time())}"
            study = optuna.create_study(
                study_name=study_name,
                direction="maximize",
                storage=None,
            )

            # Run optimization
            study.optimize(
                tier2_objective,
                n_trials=self.progressive_config.tier2_trials,
            )

            # Store results
            tier2_results = {
                "best_params": study.best_params,
                "best_value": study.best_value,
                "optimization_time": time.time() - start_time,
                "n_trials": len(study.trials),
                "tier": OptimizationTier.TIER_2_IMPORTANT.value,
            }

            self.tier_results[OptimizationTier.TIER_2_IMPORTANT.value] = tier2_results
            self.logger.info(
                f"Tier 2 optimization completed in {tier2_results['optimization_time']:.2f}s",
            )

            return tier2_results

        except Exception:
            self.print(error("Error in Tier 2 optimization: {e}"))
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="tier 3 optimization",
    )
    async def optimize_tier3_parameters(
        self,
        tier2_results: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Optimize advanced parameters (60% of time)."""
        try:
            self.logger.info("Starting Tier 3 (Advanced) optimization...")
            start_time = time.time()

            # Use tier 2 results as initial parameters
            initial_params = {}
            if tier2_results and self.progressive_config.use_previous_results:
                initial_params.update(tier2_results.get("best_params", {}))

            def tier3_objective(trial):
                params = initial_params.copy()

                # Suggest advanced parameters
                params["feature_engineering_parameters.feature_selection_threshold"] = (
                    trial.suggest_float("feature_selection_threshold", 0.001, 0.05)
                )
                params["monitoring_parameters.performance_alert_threshold"] = (
                    trial.suggest_float("performance_alert_threshold", 0.05, 0.2)
                )
                params["optimization_parameters.min_trades_for_optimization"] = (
                    trial.suggest_int("min_trades_for_optimization", 5, 20)
                )

                # Evaluate performance
                return self._evaluate_tier3_performance(params)

            # Create study
            study_name = f"tier3_optimization_{int(time.time())}"
            study = optuna.create_study(
                study_name=study_name,
                direction="maximize",
                storage=None,
            )

            # Run optimization
            study.optimize(
                tier3_objective,
                n_trials=self.progressive_config.tier3_trials,
            )

            # Store results
            tier3_results = {
                "best_params": study.best_params,
                "best_value": study.best_value,
                "optimization_time": time.time() - start_time,
                "n_trials": len(study.trials),
                "tier": OptimizationTier.TIER_3_ADVANCED.value,
            }

            self.tier_results[OptimizationTier.TIER_3_ADVANCED.value] = tier3_results
            self.logger.info(
                f"Tier 3 optimization completed in {tier3_results['optimization_time']:.2f}s",
            )

            return tier3_results

        except Exception:
            self.print(error("Error in Tier 3 optimization: {e}"))
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="progressive optimization execution",
    )
    async def run_progressive_optimization(
        self,
        initial_params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Run optimization in progressive stages."""
        try:
            self.logger.info("Starting progressive optimization...")
            total_start_time = time.time()

            # Stage 1: Quick coarse optimization (10% of time)
            self.logger.info("Stage 1: Tier 1 (Critical) optimization")
            tier1_results = await self.optimize_tier1_parameters(initial_params)

            if not tier1_results:
                self.print(failed("Tier 1 optimization failed"))
                return None

            # Stage 2: Medium optimization (30% of time)
            self.logger.info("Stage 2: Tier 2 (Important) optimization")
            tier2_results = await self.optimize_tier2_parameters(tier1_results)

            if not tier2_results:
                self.print(failed("Tier 2 optimization failed, using Tier 1 results"))
                tier2_results = tier1_results

            # Stage 3: Fine optimization (60% of time)
            self.logger.info("Stage 3: Tier 3 (Advanced) optimization")
            tier3_results = await self.optimize_tier3_parameters(tier2_results)

            if not tier3_results:
                self.print(failed("Tier 3 optimization failed, using Tier 2 results"))
                tier3_results = tier2_results

            # Combine all results
            combined_results = self._combine_progressive_results(
                tier1_results,
                tier2_results,
                tier3_results,
            )
            combined_results["total_optimization_time"] = time.time() - total_start_time

            # Store in history
            self.optimization_history.append(
                {
                    "timestamp": pd.Timestamp.now(),
                    "results": combined_results.copy(),
                    "tier_results": self.tier_results.copy(),
                },
            )

            self.logger.info(
                f"Progressive optimization completed in {combined_results['total_optimization_time']:.2f}s",
            )
            return combined_results

        except Exception:
            self.print(error("Error in progressive optimization: {e}"))
            return None

    def _combine_progressive_results(
        self,
        tier1_results: dict[str, Any],
        tier2_results: dict[str, Any],
        tier3_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Combine results from all tiers."""
        try:
            combined_results = {
                "best_params": {},
                "best_value": 0.0,
                "optimization_history": [],
                "tier_results": {
                    "tier1": tier1_results,
                    "tier2": tier2_results,
                    "tier3": tier3_results,
                },
                "total_trials": 0,
            }

            # Combine best parameters from all tiers
            for tier_results in [tier1_results, tier2_results, tier3_results]:
                if tier_results and "best_params" in tier_results:
                    combined_results["best_params"].update(tier_results["best_params"])
                    combined_results["total_trials"] += tier_results.get("n_trials", 0)

            # Calculate weighted best value
            total_value = 0.0
            total_weight = 0.0

            for tier_results in [tier1_results, tier2_results, tier3_results]:
                if tier_results and "best_value" in tier_results:
                    weight = 1.0  # Equal weight for now
                    total_value += tier_results["best_value"] * weight
                    total_weight += weight

            if total_weight > 0:
                combined_results["best_value"] = total_value / total_weight

            return combined_results

        except Exception:
            self.print(error("Error combining progressive results: {e}"))
            return {}

    def _evaluate_tier1_performance(self, params: dict[str, Any]) -> float:
        """Evaluate Tier 1 performance (placeholder for actual evaluation)."""
        try:
            # Simulate performance based on critical parameters
            performance = 0.0

            # Entry threshold evaluation
            entry_threshold = params.get(
                "confidence_thresholds.base_entry_threshold",
                0.7,
            )
            if 0.6 <= entry_threshold <= 0.8:
                performance += 0.3
            else:
                performance += 0.1

            # Position sizing evaluation
            kelly_multiplier = params.get(
                "position_sizing_parameters.kelly_multiplier",
                0.25,
            )
            if 0.2 <= kelly_multiplier <= 0.4:
                performance += 0.3
            else:
                performance += 0.1

            # Stop loss evaluation
            stop_loss_multiplier = params.get(
                "stop_loss_parameters.stop_loss_atr_multiplier",
                2.0,
            )
            if 1.5 <= stop_loss_multiplier <= 3.0:
                performance += 0.4
            else:
                performance += 0.1

            return min(performance, 1.0)

        except Exception:
            self.print(warning("Error evaluating Tier 1 performance: {e}"))
            return 0.0

    def _evaluate_tier2_performance(self, params: dict[str, Any]) -> float:
        """Evaluate Tier 2 performance (placeholder for actual evaluation)."""
        try:
            # Simulate performance based on important parameters
            performance = 0.0

            # Volatility multiplier evaluation
            volatility_multiplier = params.get(
                "volatility_parameters.volatility_multiplier",
                1.0,
            )
            if 0.8 <= volatility_multiplier <= 1.5:
                performance += 0.3
            else:
                performance += 0.1

            # Profit taking evaluation
            pt_multiplier = params.get(
                "profit_taking_parameters.pt1_target_atr_multiplier",
                2.5,
            )
            if 2.0 <= pt_multiplier <= 3.5:
                performance += 0.3
            else:
                performance += 0.1

            # Ensemble method evaluation
            ensemble_method = params.get(
                "ensemble_parameters.ensemble_method",
                "confidence_weighted",
            )
            if ensemble_method in ["confidence_weighted", "weighted_average"]:
                performance += 0.4
            else:
                performance += 0.2

            return min(performance, 1.0)

        except Exception:
            self.print(warning("Error evaluating Tier 2 performance: {e}"))
            return 0.0

    def _evaluate_tier3_performance(self, params: dict[str, Any]) -> float:
        """Evaluate Tier 3 performance (placeholder for actual evaluation)."""
        try:
            # Simulate performance based on advanced parameters
            performance = 0.0

            # Feature selection threshold evaluation
            feature_threshold = params.get(
                "feature_engineering_parameters.feature_selection_threshold",
                0.01,
            )
            if 0.005 <= feature_threshold <= 0.02:
                performance += 0.4
            else:
                performance += 0.1

            # Performance alert threshold evaluation
            alert_threshold = params.get(
                "monitoring_parameters.performance_alert_threshold",
                0.1,
            )
            if 0.05 <= alert_threshold <= 0.15:
                performance += 0.3
            else:
                performance += 0.1

            # Min trades evaluation
            min_trades = params.get(
                "optimization_parameters.min_trades_for_optimization",
                10,
            )
            if 8 <= min_trades <= 15:
                performance += 0.3
            else:
                performance += 0.1

            return min(performance, 1.0)

        except Exception:
            self.print(warning("Error evaluating Tier 3 performance: {e}"))
            return 0.0

    def get_progressive_statistics(self) -> dict[str, Any]:
        """Get progressive optimization statistics."""
        try:
            if not self.optimization_history:
                return {"message": "No progressive optimization history available"}

            latest_optimization = self.optimization_history[-1]

            stats = {
                "total_optimizations": len(self.optimization_history),
                "latest_optimization_time": latest_optimization[
                    "timestamp"
                ].isoformat(),
                "tier_results": {},
                "total_optimization_time": latest_optimization["results"].get(
                    "total_optimization_time",
                    0,
                ),
                "total_trials": latest_optimization["results"].get("total_trials", 0),
            }

            # Add tier-specific statistics
            for tier_name, tier_results in (
                latest_optimization["results"].get("tier_results", {}).items()
            ):
                if tier_results:
                    stats["tier_results"][tier_name] = {
                        "best_value": tier_results.get("best_value", 0.0),
                        "optimization_time": tier_results.get("optimization_time", 0),
                        "n_trials": tier_results.get("n_trials", 0),
                    }

            return stats

        except Exception:
            self.print(error("Error getting progressive statistics: {e}"))
            return {}
