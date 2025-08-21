# src/training/steps/step12_final_parameters_optimization/sr_optuna_optimization.py

"""
S/R Parameter Optimization with Optuna (clean implementation)

Provides a stable SROptunaOptimizer with async initialization of required
components and an Optuna-based optimization loop. Heavy integrations are
stubbed via synthetic scoring so the optimizer runs reliably.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd

from src.tactician.sr_breakout_predictor import setup_sr_breakout_predictor
from src.tactician.sr_weight_optimizer import SRWeightOptimizer
from src.utils.logger import system_logger

# Configure Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class SROptimizationResult:
    """Result of S/R parameter optimization."""

    # Optimized parameters
    strength_score_weights: dict[str, float]
    level_detection_params: dict[str, Any]
    breakout_thresholds: dict[str, float]
    zone_multipliers: dict[str, float]
    confidence_thresholds: dict[str, float]

    # Performance metrics
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_return: float
    signal_clarity: float
    noise_reduction: float

    # Optimization metadata
    optimization_score: float
    n_trials: int
    optimization_time: float
    study_name: str
    best_trial_number: int


class SROptunaOptimizer:
    """
    Comprehensive S/R parameter optimizer using Optuna (reliable minimal core).
    """

    def __init__(
        self,
        config: dict[str, Any],
        storage_url: str = "sqlite:///sr_optuna_studies.db",
        study_name_prefix: str = "sr_optimization",
    ) -> None:
        self.config = config
        self.storage_url = storage_url
        self.study_name_prefix = study_name_prefix
        self.logger = system_logger.getChild("SROptunaOptimizer")

        # S/R specific configuration
        self.sr_config = config.get("sr_optimization", {})
        self.objective_weights: dict[str, float] = self.sr_config.get(
            "objective_weights",
            {"sharpe_ratio": 0.4, "win_rate": 0.3, "signal_clarity": 0.3},
        )

        # Optimization parameters
        self.n_trials = int(self.sr_config.get("n_trials", 100))
        self.cv_folds = int(self.sr_config.get("cv_folds", 5))
        self.early_stopping_patience = int(
            self.sr_config.get("early_stopping_patience", 20),
        )
        self.subsample_fraction = float(self.sr_config.get("subsample_fraction", 0.7))

        # Components
        self.sr_predictor: Any | None = None
        self.weight_optimizer: SRWeightOptimizer | None = None

    async def initialize(self) -> bool:
        """Initialize the optimizer components asynchronously."""
        try:
            self.logger.info("Initializing S/R Optuna Optimizer...")

            # Initialize SR predictor
            self.sr_predictor = await setup_sr_breakout_predictor(self.config)
            if not self.sr_predictor:
                self.logger.error("Failed to initialize SR predictor")
                return False

            # Initialize SR weight optimizer
            self.weight_optimizer = SRWeightOptimizer(self.config)
            ok = await self.weight_optimizer.initialize()
            if not ok:
                self.logger.error("Failed to initialize SR weight optimizer")
                return False

            self.logger.info("✅ S/R Optuna Optimizer initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing S/R optimizer: {e}")
            return False

    # --- Parameter Spaces ---
    def _get_strength_score_space(self, trial: optuna.Trial) -> dict[str, float]:
        return {
            "touch_count": trial.suggest_float("touch_count", 0.1, 0.5),
            "total_volume": trial.suggest_float("total_volume", 0.1, 0.4),
            "level_age": trial.suggest_float("level_age", 0.1, 0.4),
            "bounce_rate": trial.suggest_float("bounce_rate", 0.1, 0.4),
            "isolation_score": trial.suggest_float("isolation_score", 0.05, 0.3),
        }

    def _get_level_detection_space(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "min_touch_count": trial.suggest_int("min_touch_count", 2, 10),
            "min_level_age_hours": trial.suggest_int("min_level_age_hours", 1, 48),
            "price_tolerance_pct": trial.suggest_float("price_tolerance_pct", 0.1, 2.0),
            "volume_threshold": trial.suggest_float("volume_threshold", 0.5, 2.0),
            "strength_threshold": trial.suggest_float("strength_threshold", 0.3, 0.8),
        }

    def _get_breakout_space(self, trial: optuna.Trial) -> dict[str, float]:
        return {
            "breakout_threshold": trial.suggest_float("breakout_threshold", 0.6, 0.9),
            "confirmation_periods": trial.suggest_int("confirmation_periods", 1, 5),
            "volume_confirmation": trial.suggest_float("volume_confirmation", 1.2, 3.0),
            "momentum_threshold": trial.suggest_float("momentum_threshold", 0.1, 0.5),
            "false_breakout_filter": trial.suggest_float("false_breakout_filter", 0.1, 0.3),
        }

    def _get_zone_multiplier_space(self, trial: optuna.Trial) -> dict[str, float]:
        return {
            "support_zone_multiplier": trial.suggest_float(
                "support_zone_multiplier",
                0.8,
                1.5,
            ),
            "resistance_zone_multiplier": trial.suggest_float(
                "resistance_zone_multiplier",
                0.8,
                1.5,
            ),
            "sr_zone_threshold": trial.suggest_float("sr_zone_threshold", 0.6, 0.9),
            "zone_expansion_factor": trial.suggest_float("zone_expansion_factor", 1.0, 2.0),
            "zone_contraction_factor": trial.suggest_float("zone_contraction_factor", 0.5, 1.0),
        }

    def _get_confidence_thresholds_space(self, trial: optuna.Trial) -> dict[str, float]:
        return {
            "min_sr_confidence": trial.suggest_float("min_sr_confidence", 0.5, 0.8),
            "high_confidence_threshold": trial.suggest_float(
                "high_confidence_threshold",
                0.7,
                0.9,
            ),
        }

    def _combine_params(self, trial: optuna.Trial) -> dict[str, Any]:
        params: dict[str, Any] = {}
        params.update(self._get_strength_score_space(trial))
        params.update(self._get_level_detection_space(trial))
        params.update(self._get_breakout_space(trial))
        params.update(self._get_zone_multiplier_space(trial))
        params.update(self._get_confidence_thresholds_space(trial))
        return params

    def _synthetic_score(self, params: dict[str, Any]) -> tuple[float, dict[str, float]]:
        """Create a stable synthetic score and metrics based on parameter coherence."""
        # Normalize and balance strength weights
        weights = np.array(
            [
                float(params["touch_count"]),
                float(params["total_volume"]),
                float(params["level_age"]),
                float(params["bounce_rate"]),
                float(params["isolation_score"]),
            ]
        )
        weights = weights / (weights.sum() + 1e-9)
        weight_balance = 1.0 - float(np.std(weights))  # higher is better

        # Sanity on thresholds
        threshold_penalty = 0.0
        if params["breakout_threshold"] < 0.6 or params["breakout_threshold"] > 0.95:
            threshold_penalty += 0.05
        if params["min_touch_count"] < 1:
            threshold_penalty += 0.1

        # Derived metrics
        sharpe_ratio = max(0.0, weight_balance - threshold_penalty)
        win_rate = 0.5 + 0.4 * weight_balance
        profit_factor = 1.0 + 2.0 * weight_balance
        max_drawdown = max(0.0, 0.3 - 0.25 * weight_balance)
        signal_clarity = weight_balance
        noise_reduction = 0.5 + 0.5 * weight_balance
        total_return = win_rate * profit_factor - max_drawdown

        # Objective
        score = (
            self.objective_weights.get("sharpe_ratio", 0.4) * sharpe_ratio
            + self.objective_weights.get("win_rate", 0.3) * win_rate
            + self.objective_weights.get("signal_clarity", 0.3) * signal_clarity
        )

        metrics = {
            "sharpe_ratio": float(sharpe_ratio),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "max_drawdown": float(max_drawdown),
            "signal_clarity": float(signal_clarity),
            "noise_reduction": float(noise_reduction),
            "total_return": float(total_return),
        }
        return float(score), metrics

    def _summarize_study(self, study: optuna.Study) -> dict[str, Any]:
        pruned = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete = study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
        )
        return {
            "study_name": study.study_name,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "total_trials": len(study.trials),
            "n_completed": len(complete),
            "n_pruned": len(pruned),
        }

    def optimize(
        self,
        n_trials: int | None = None,
        n_jobs: int = -1,
        early_stopping_patience: int | None = None,
        subsample_fraction: float | None = None,
    ) -> dict[str, Any]:
        """Run the S/R parameter optimization using Optuna."""
        n_trials_final = int(n_trials or self.n_trials)
        patience = early_stopping_patience if early_stopping_patience is not None else self.early_stopping_patience

        study_name = f"{self.study_name_prefix}_sr_params"
        study = optuna.create_study(
            storage=self.storage_url,
            study_name=study_name,
            direction="maximize",
            pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=n_trials_final),
            sampler=optuna.samplers.TPESampler(seed=42),
            load_if_exists=True,
        )

        def objective(trial: optuna.Trial) -> float:
            params = self._combine_params(trial)
            score, _metrics = self._synthetic_score(params)
            return float(score)

        callbacks = []
        if patience:
            callbacks.append(
                optuna.callbacks.EarlyStoppingCallback(patience, "maximize"),
            )

        self.logger.info(
            f"Starting S/R optimization with {n_trials_final} trials (jobs={n_jobs})",
        )
        start_time = time.time()
        study.optimize(objective, n_trials=n_trials_final, n_jobs=n_jobs, callbacks=callbacks)
        elapsed = time.time() - start_time
        self.logger.info(
            f"S/R optimization finished in {elapsed:.2f}s with {len(study.trials)} trials",
        )
        return self._summarize_study(study)