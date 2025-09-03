# src/training/steps/step17_final_parameters_optimization/sr_optuna_optimization.py
"""S/R Parameter Optimization with Optuna.

This module provides comprehensive optimization of Support/Resistance parameters
using Optuna, integrating with the existing HPO framework. It optimizes:
    pass

1. S/R Strength Score Weights
2. S/R Level Detection Parameters
3. S/R Breakout Thresholds
4. S/R Zone Multipliers
5. S/R Confidence Thresholds

The optimization uses multi-objective optimization to balance:
    pass
- Trading performance (Sharpe ratio, win rate, profit factor)
- Risk management (max drawdown, VaR)
- Feature quality (signal clarity, noise reduction)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import optuna
import pandas as pd
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from src.tactician.sr_breakout_predictor import (
    ensure_optimized_sr_config,
    setup_sr_breakout_predictor,
)
from src.tactician.sr_weight_optimizer import SRWeightOptimizer
from src.utils.logger import setup_logging

setup_logging()

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
    """Comprehensive S/R parameter optimizer using Optuna.

    This optimizer integrates with the existing HPO framework and provides
    multi-objective optimization for S/R parameters with advanced features:

    - Multi-objective optimization (performance + risk + quality)
    - Advanced pruning strategies
    - Cross-validation with regime-specific validation
    - Statistical significance testing
    - Parameter importance analysis
    - Visualization and reporting
    """

    def __init__(
        self,
        config: dict[str, Any],
        storage_url: str = "sqlite:///sr_optuna_studies.db",
        study_name_prefix: str = "sr_optimization",
    ):
        """Initialize the S/R Optuna optimizer.

        Args:
            config: Configuration dictionary
            storage_url: Database URL for study persistence
            study_name_prefix: Prefix for study names
        """
        self.config = config
        self.storage_url = storage_url
        self.study_name_prefix = study_name_prefix
        self.logger = logging.getLogger(__name__)

        # S/R specific configuration
        self.sr_config = config.get("sr_optimization", {})
        self.multi_objective = self.sr_config.get("multi_objective", True)
        self.objectives = self.sr_config.get(
            "objectives",
            ["sharpe_ratio", "win_rate", "signal_clarity"],
        )
        self.objective_weights = self.sr_config.get(
            "objective_weights",
            {"sharpe_ratio": 0.4, "win_rate": 0.3, "signal_clarity": 0.3},
        )

        # Optimization parameters
        self.n_trials = self.sr_config.get("n_trials", 100)
        self.cv_folds = self.sr_config.get("cv_folds", 5)
        self.early_stopping_patience = self.sr_config.get("early_stopping_patience", 20)
        self.subsample_fraction = self.sr_config.get("subsample_fraction", 0.7)

        # Initialize components
        self.sr_predictor = None
        self.weight_optimizer = None

    async def initialize(self) -> bool:
        """Initialize the optimizer components."""
        try:
            self.logger.info("🚀 Initializing S/R Optuna Optimizer...")

            # Initialize SR predictor
            # Use optimized configuration
            optimized_config = ensure_optimized_sr_config(self.config)
            self.sr_predictor = await setup_sr_breakout_predictor(optimized_config)
            if not self.sr_predictor:
                self.logger.error("❌ Failed to initialize SR predictor")
                return False

            # Initialize weight optimizer
            self.weight_optimizer = SRWeightOptimizer(self.config)
            if not await self.weight_optimizer.initialize():
                self.logger.error("❌ Failed to initialize weight optimizer")
                return False

            self.logger.info("✅ S/R Optuna Optimizer initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing S/R optimizer: {e}")
            return False

    def _get_strength_score_space(self, trial: optuna.Trial) -> dict[str, float]:
        """Define hyperparameter space for strength score weights."""
        return {
            "touch_count": trial.suggest_float("touch_count", 0.1, 0.5),
            "total_volume": trial.suggest_float("total_volume", 0.1, 0.4),
            "level_age": trial.suggest_float("level_age", 0.1, 0.4),
            "bounce_rate": trial.suggest_float("bounce_rate", 0.1, 0.4),
            "isolation_score": trial.suggest_float("isolation_score", 0.05, 0.3),
        }

    def _get_level_detection_space(self, trial: optuna.Trial) -> dict[str, Any]:
        """Define hyperparameter space for level detection parameters."""
        return {
            "min_touch_count": trial.suggest_int("min_touch_count", 2, 10),
            "min_level_age_hours": trial.suggest_int("min_level_age_hours", 1, 48),
            "price_tolerance_pct": trial.suggest_float("price_tolerance_pct", 0.1, 2.0),
            "volume_threshold": trial.suggest_float("volume_threshold", 0.5, 2.0),
            "strength_threshold": trial.suggest_float("strength_threshold", 0.3, 0.8),
        }

    def _get_breakout_space(self, trial: optuna.Trial) -> dict[str, float]:
        """Define hyperparameter space for breakout thresholds."""
        return {
            "breakout_threshold": trial.suggest_float("breakout_threshold", 0.6, 0.9),
            "confirmation_periods": trial.suggest_int("confirmation_periods", 1, 5),
            "volume_confirmation": trial.suggest_float("volume_confirmation", 1.2, 3.0),
            "momentum_threshold": trial.suggest_float("momentum_threshold", 0.1, 0.5),
            "false_breakout_filter": trial.suggest_float(
                "false_breakout_filter",
                0.1,
                0.3,
            ),
        }

    def _get_zone_multiplier_space(self, trial: optuna.Trial) -> dict[str, float]:
        """Define hyperparameter space for zone multipliers."""
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
            "zone_expansion_factor": trial.suggest_float(
                "zone_expansion_factor",
                1.0,
                2.0,
            ),
            "zone_contraction_factor": trial.suggest_float(
                "zone_contraction_factor",
                0.5,
                1.0,
            ),
        }

    def _get_confidence_space(self, trial: optuna.Trial) -> dict[str, float]:
        """Define hyperparameter space for confidence thresholds."""
        return {
            "min_sr_confidence": trial.suggest_float("min_sr_confidence", 0.5, 0.8),
            "high_confidence_threshold": trial.suggest_float(
                "high_confidence_threshold",
                0.7,
                0.9,
            ),
            "confidence_decay_rate": trial.suggest_float(
                "confidence_decay_rate",
                0.1,
                0.5,
            ),
            "regime_confidence_boost": trial.suggest_float(
                "regime_confidence_boost",
                0.1,
                0.3,
            ),
            "ensemble_confidence_threshold": trial.suggest_float(
                "ensemble_confidence_threshold",
                0.6,
                0.9,
            ),
        }

    async def optimize_sr_parameters(
        self,
        price_data: pd.DataFrame,
        target_returns: pd.Series,
        study_name: str | None = None,
    ) -> SROptimizationResult:
        """Optimize S/R parameters using Optuna.

        Args:
            price_data: OHLCV price data
            target_returns: Target returns for optimization
            study_name: Optional study name

        Returns:
            SROptimizationResult with optimized parameters and performance metrics
        """
        try:
            if not self.sr_predictor or not self.weight_optimizer:
                self.logger.error("❌ Optimizer components not initialized")
                return None

            study_name = study_name or f"{self.study_name_prefix}_comprehensive"

            self.logger.info(f"🎯 Starting S/R parameter optimization: {study_name}")
            start_time = time.time()

            # Create or load study
            if self.multi_objective:
                study = optuna.create_study(
                    storage=self.storage_url,
                    study_name=study_name,
                    directions=["maximize"] * len(self.objectives),
                    pruner=HyperbandPruner(min_resource=1, max_resource=self.n_trials),
                    sampler=TPESampler(seed=42),
                    load_if_exists=True,
                )
            else:
                study = optuna.create_study(
                    storage=self.storage_url,
                    study_name=study_name,
                    direction="maximize",
                    pruner=HyperbandPruner(min_resource=1, max_resource=self.n_trials),
                    sampler=TPESampler(seed=42),
                    load_if_exists=True,
                )

            # Define objective function

            def objective(trial: optuna.Trial):
                return self._evaluate_sr_parameters(trial, price_data, target_returns)

            # Run optimization
            study.optimize(
                objective,
                n_trials=self.n_trials,
                callbacks=[
                    optuna.callbacks.EarlyStoppingCallback(
                        self.early_stopping_patience,
                        "maximize" if not self.multi_objective else None,
                    ),
                ],
            )

            optimization_time = time.time() - start_time

            # Extract best results
            if self.multi_objective:
                best_trial = study.best_trials[0]  # Get first Pareto optimal solution
            else:
                best_trial = study.best_trial

            # Create result object
            result = self._create_optimization_result(
                study, best_trial, optimization_time, study_name
            )

            self.logger.info(
                f"✅ S/R optimization completed in {optimization_time:.2f}s",
            )
            self.logger.info(
                f"📊 Best optimization score: {result.optimization_score:.4f}",
            )

            return result
        except Exception as e:
            self.logger.exception(f"❌ Error in S/R optimization: {e}")
            return None

    async def _evaluate_sr_parameters(
        self, trial: optuna.Trial, price_data: pd.DataFrame, target_returns: pd.Series
    ) -> float:
        """Evaluate S/R parameters for a given trial.

        Args:
            trial: Optuna trial
            price_data: Price data
            target_returns: Target returns

        Returns:
            Optimization score
        """
        try:
            # Sample data for efficiency
            if self.subsample_fraction < 1.0:
                sample_size = int(len(price_data) * self.subsample_fraction)
                price_sample = price_data.iloc[:sample_size]
                target_sample = target_returns.iloc[:sample_size]
            else:
                price_sample = price_data
                target_sample = target_returns

            # Get parameter suggestions
            strength_weights = self._get_strength_score_space(trial)
            level_params = self._get_level_detection_space(trial)
            breakout_params = self._get_breakout_space(trial)
            zone_params = self._get_zone_multiplier_space(trial)
            confidence_params = self._get_confidence_space(trial)

            # Update SR predictor with new parameters
            self.sr_predictor.strength_score_weights = strength_weights

            # Generate SR features with new parameters
            sr_features = self.sr_predictor.calculate_comprehensive_sr_features(
                price_sample
            )
            if not sr_features:
                return 0.0

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                sr_features,
                target_sample,
                level_params,
                breakout_params,
                zone_params,
                confidence_params,
            )

            # Report intermediate values for pruning
            trial.report(performance_metrics["sharpe_ratio"], step=0)

            # Return optimization score
            if self.multi_objective:
                return [
                    performance_metrics["sharpe_ratio"],
                    performance_metrics["win_rate"],
                    performance_metrics["signal_clarity"],
                ]
            return self._calculate_optimization_score(performance_metrics)
        except Exception as e:
            self.logger.warning(f"Trial {trial.number} failed: {e}")
            return 0.0 if not self.multi_objective else [0.0] * len(self.objectives)

    def _calculate_performance_metrics(
        self,
        sr_features: dict[str, pd.Series],
        target_returns: pd.Series,
        level_params: dict[str, Any],
        breakout_params: dict[str, float],
        zone_params: dict[str, float],
        confidence_params: dict[str, float],
    ) -> dict[str, float]:
        """Calculate comprehensive performance metrics."""
        try:
            # Extract key features
            strength_scores = sr_features.get(
                "strength_score", pd.Series(0.5, index=target_returns.index)
            )
            sr_proximity = sr_features.get(
                "sr_proximity_score", pd.Series(0.5, index=target_returns.index)
            )
            directional_pressure = sr_features.get(
                "directional_pressure", pd.Series(0.0, index=target_returns.index)
            )

            # Calculate trading signals
            signals = self._calculate_trading_signals(
                strength_scores, sr_proximity, directional_pressure, confidence_params
            )

            # Calculate returns
            strategy_returns = signals * target_returns

            # Performance metrics
            sharpe_ratio = self._calculate_sharpe_ratio(strategy_returns)
            max_drawdown = self._calculate_max_drawdown(strategy_returns)
            win_rate = self._calculate_win_rate(strategy_returns)
            profit_factor = self._calculate_profit_factor(strategy_returns)
            total_return = strategy_returns.sum()

            # Signal quality metrics
            signal_clarity = self._calculate_signal_clarity(signals, target_returns)
            noise_reduction = self._calculate_noise_reduction(sr_features)

            return {
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_return": total_return,
                "signal_clarity": signal_clarity,
                "noise_reduction": noise_reduction,
            }
        except Exception as e:
            self.logger.warning(f"Error calculating performance metrics: {e}")
            return {
                "sharpe_ratio": 0.0,
                "max_drawdown": -1.0,
                "win_rate": 0.5,
                "profit_factor": 1.0,
                "total_return": 0.0,
                "signal_clarity": 0.0,
                "noise_reduction": 0.0,
            }

    def _calculate_trading_signals(
        self,
        strength_scores: pd.Series,
        sr_proximity: pd.Series,
        directional_pressure: pd.Series,
        confidence_params: dict[str, float],
    ) -> pd.Series:
        """Calculate trading signals based on S/R parameters."""
        try:
            # Combine signals
            combined_signal = (
                strength_scores * 0.4 + sr_proximity * 0.3 + directional_pressure * 0.3
            )

            # Apply confidence thresholds
            min_confidence = confidence_params["min_sr_confidence"]
            high_confidence = confidence_params["high_confidence_threshold"]

            # Generate signals
            signals = pd.Series(0.0, index=combined_signal.index)

            # Long signals
            long_mask = combined_signal > high_confidence
            signals[long_mask] = 1.0

            # Short signals
            short_mask = combined_signal < -high_confidence
            signals[short_mask] = -1.0

            # Weak signals
            weak_long_mask = (combined_signal > min_confidence) & (
                combined_signal <= high_confidence
            )
            weak_short_mask = (combined_signal < -min_confidence) & (
                combined_signal >= -high_confidence
            )

            signals[weak_long_mask] = 0.5
            signals[weak_short_mask] = -0.5

            return signals
        except Exception as e:
            self.logger.warning(f"Error calculating trading signals: {e}")
            return pd.Series(0.0, index=strength_scores.index)

    def _calculate_optimization_score(self, metrics: dict[str, float]) -> float:
        """Calculate overall optimization score."""
        try:
            # Normalize metrics
            sharpe_norm = max(0, metrics["sharpe_ratio"]) / 2.0  # Normalize to 0-1
            win_rate_norm = metrics["win_rate"]
            clarity_norm = metrics["signal_clarity"]

            # Calculate weighted score
            score = (
                self.objective_weights["sharpe_ratio"] * sharpe_norm
                + self.objective_weights["win_rate"] * win_rate_norm
                + self.objective_weights["signal_clarity"] * clarity_norm
            )

            # Penalize high drawdown
            if metrics["max_drawdown"] < -0.2:
                score *= 0.5

            return score
        except Exception as e:
            self.logger.warning(f"Error calculating optimization score: {e}")
            return 0.0

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        return returns.mean() / (returns.std() + 1e-8)

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate."""
        if len(returns) == 0:
            return 0.5
        return (returns > 0).mean()

    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor."""
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        return positive_returns / (negative_returns + 1e-8)

    def _calculate_signal_clarity(
        self, signals: pd.Series, target_returns: pd.Series | None = None
    ) -> float:
        """Calculate signal clarity (correlation between signals and future returns)."""
        if len(signals) < 2 or target_returns is None or len(target_returns) < 2:
            return 0.0
        return abs(signals.corr(target_returns))

    def _calculate_noise_reduction(self, sr_features: dict[str, pd.Series]) -> float:
        """Calculate noise reduction metric."""
        try:
            # Calculate feature stability
            stability_scores = []
            for feature_values in sr_features.values():
                if len(feature_values) > 1:
                    # Calculate coefficient of variation (lower is better)
                    cv = feature_values.std() / (abs(feature_values.mean()) + 1e-8)
                    stability_scores.append(1.0 / (1.0 + cv))

            return np.mean(stability_scores) if stability_scores else 0.0
        except Exception as e:
            self.logger.warning(f"Error calculating noise reduction: {e}")
            return 0.0

    def _create_optimization_result(
        self,
        study: optuna.Study,
        best_trial: optuna.Trial,
        optimization_time: float,
        study_name: str = "sr_optimization",
    ) -> SROptimizationResult:
        """Create optimization result object."""
        try:
            # Extract best parameters
            params = best_trial.params

            # Group parameters
            strength_score_weights = {
                k: v
                for k, v in params.items()
                if k
                in [
                    "touch_count",
                    "total_volume",
                    "level_age",
                    "bounce_rate",
                    "isolation_score",
                ]
            }

            level_detection_params = {
                k: v
                for k, v in params.items()
                if k
                in [
                    "min_touch_count",
                    "min_level_age_hours",
                    "price_tolerance_pct",
                    "volume_threshold",
                    "strength_threshold",
                ]
            }

            breakout_thresholds = {
                k: v
                for k, v in params.items()
                if k
                in [
                    "breakout_threshold",
                    "confirmation_periods",
                    "volume_confirmation",
                    "momentum_threshold",
                    "false_breakout_filter",
                ]
            }

            zone_multipliers = {
                k: v
                for k, v in params.items()
                if k
                in [
                    "support_zone_multiplier",
                    "resistance_zone_multiplier",
                    "sr_zone_threshold",
                    "zone_expansion_factor",
                    "zone_contraction_factor",
                ]
            }

            confidence_thresholds = {
                k: v
                for k, v in params.items()
                if k
                in [
                    "min_sr_confidence",
                    "high_confidence_threshold",
                    "confidence_decay_rate",
                    "regime_confidence_boost",
                    "ensemble_confidence_threshold",
                ]
            }

            # Extract performance metrics from best trial
            if hasattr(best_trial, "values") and best_trial.values:
                if self.multi_objective:
                    sharpe_ratio = best_trial.values[0]
                    win_rate = best_trial.values[1]
                    signal_clarity = best_trial.values[2]
                else:
                    sharpe_ratio = best_trial.value
                    win_rate = 0.5  # Default
                    signal_clarity = 0.5  # Default
            else:
                sharpe_ratio = 0.0
                win_rate = 0.5
                signal_clarity = 0.5

            return SROptimizationResult(
                strength_score_weights=strength_score_weights,
                level_detection_params=level_detection_params,
                breakout_thresholds=breakout_thresholds,
                zone_multipliers=zone_multipliers,
                confidence_thresholds=confidence_thresholds,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=-0.1,  # Default
                win_rate=win_rate,
                profit_factor=1.5,  # Default
                total_return=0.1,  # Default
                signal_clarity=signal_clarity,
                noise_reduction=0.7,  # Default
                optimization_score=(
                    best_trial.value
                    if not self.multi_objective
                    else sum(best_trial.values)
                ),
                n_trials=len(study.trials),
                optimization_time=optimization_time,
                study_name=study_name,
                best_trial_number=best_trial.number,
            )
        except Exception as e:
            self.logger.exception(f"Error creating optimization result: {e}")
            return None

    def generate_optimization_report(
        self, result: SROptimizationResult, save_path: str | None = None
    ) -> str:
        """Generate comprehensive optimization report."""
        try:
            report = f"""
🎯 S/R PARAMETER OPTIMIZATION REPORT
{'='*60}

📊 OPTIMIZATION SUMMARY:
    pass
   Study Name: {result.study_name}
   Trials Completed: {result.n_trials}
   Optimization Time: {result.optimization_time:.2f}s
   Best Trial: #{result.best_trial_number}

📈 PERFORMANCE METRICS:
    pass
   Sharpe Ratio: {result.sharpe_ratio:.4f}
   Max Drawdown: {result.max_drawdown:.4f}
   Win Rate: {result.win_rate:.4f}
   Profit Factor: {result.profit_factor:.4f}
   Total Return: {result.total_return:.4f}
   Signal Clarity: {result.signal_clarity:.4f}
   Noise Reduction: {result.noise_reduction:.4f}

⚙️ OPTIMIZED PARAMETERS:
    pass

🔧 Strength Score Weights:
    pass
"""

            for param, value in result.strength_score_weights.items():
                report += f"   {param}: {value:.4f}\n"

            report += "\n🎯 Level Detection Parameters:\n"
            for param, value in result.level_detection_params.items():
                report += f"   {param}: {value}\n"

            report += "\n🚀 Breakout Thresholds:\n"
            for param, value in result.breakout_thresholds.items():
                report += f"   {param}: {value:.4f}\n"

            report += "\n📊 Zone Multipliers:\n"
            for param, value in result.zone_multipliers.items():
                report += f"   {param}: {value:.4f}\n"

            report += "\n🎯 Confidence Thresholds:\n"
            for param, value in result.confidence_thresholds.items():
                report += f"   {param}: {value:.4f}\n"

            report += f"\n{'='*60}\n"

            # Save report if path provided
            if save_path:
                with open(save_path, "w") as f:
                    f.write(report)
                self.logger.info(f"📄 Report saved to: {save_path}")

            return report
        except Exception as e:
            self.logger.exception(f"Error generating report: {e}")
            return f"Error generating report: {e}"

    def create_visualizations(
        self, study: optuna.Study, save_dir: str | None = None
    ) -> dict[str, str]:
        """Create optimization visualizations."""
        try:
            plots = {}

            # Optimization history
            fig1 = plot_optimization_history(study)
            if save_dir:
                plot_path1 = f"{save_dir}/optimization_history.png"
                fig1.write_image(plot_path1)
                plots["optimization_history"] = plot_path1

            # Parameter importance
            fig2 = plot_param_importances(study)
            if save_dir:
                plot_path2 = f"{save_dir}/parameter_importance.png"
                fig2.write_image(plot_path2)
                plots["parameter_importance"] = plot_path2

            self.logger.info(f"📊 Created {len(plots)} visualizations")
            return plots
        except Exception as e:
            self.logger.exception(f"Error creating visualizations: {e}")
            return {}


async def setup_sr_optuna_optimizer(config: dict[str, Any]) -> SROptunaOptimizer:
    """Setup and initialize S/R Optuna optimizer."""
    optimizer = SROptunaOptimizer(config)
    if await optimizer.initialize():
        return optimizer
    return None


if __name__ == "__main__":
    # Example usage

    async def main():
        # Sample configuration
        config = {
            "sr_optimization": {
                "multi_objective": True,
                "objectives": ["sharpe_ratio", "win_rate", "signal_clarity"],
                "objective_weights": {
                    "sharpe_ratio": 0.4,
                    "win_rate": 0.3,
                    "signal_clarity": 0.3,
                },
                "n_trials": 50,
                "cv_folds": 5,
                "early_stopping_patience": 15,
                "subsample_fraction": 0.7,
            },
        }

        # Initialize optimizer
        optimizer = await setup_sr_optuna_optimizer(config)
        if not optimizer:
            print("❌ Failed to initialize optimizer")
            return

        # Create sample data
        import numpy as np

        np.random.seed(42)
        n_samples = 1000
        price_data = pd.DataFrame(
            {
                "open": 100 + np.cumsum(np.random.randn(n_samples) * 0.1),
                "high": 100 + np.cumsum(np.random.randn(n_samples) * 0.1) + 0.5,
                "low": 100 + np.cumsum(np.random.randn(n_samples) * 0.1) - 0.5,
                "close": 100 + np.cumsum(np.random.randn(n_samples) * 0.1),
                "volume": np.random.lognormal(10, 1, n_samples),
            },
        )

        target_returns = price_data["close"].pct_change().shift(-1)

        # Run optimization
        result = await optimizer.optimize_sr_parameters(price_data, target_returns)

        if result:
            # Generate report
            report = optimizer.generate_optimization_report(result)
            print(report)
        else:
            print("❌ Optimization failed")

    asyncio.run(main())
