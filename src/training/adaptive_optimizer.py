# src/training/adaptive_optimizer.py

from typing import Any

import numpy as np
import optuna
import pandas as pd

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class MarketRegime:
    """Represents a market regime with specific characteristics."""

    def __init__(
        self,
        name: str,
        volatility: float,
        trend_strength: float,
        regime_type: str,
        optimal_params: dict[str, Any],
    ) -> None:
        self.name = name
        self.volatility = volatility
        self.trend_strength = trend_strength
        self.regime_type = regime_type
        self.optimal_params = optimal_params
        self.confidence = 0.0


class AdaptiveOptimizer:
    """Adaptive hyperparameter optimizer that adjusts parameters based on market regime detection."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("AdaptiveOptimizer")

        # Regime detection configuration
        self.regime_detection_config = config.get(
            "regime_detection",
            {
                "lookback_window": 50,
                "volatility_threshold": 0.02,
                "trend_threshold": 0.01,
            },
        )

        # Regime-specific optimization
        self.regime_optimizers = {}
        self.current_regime = None
        self.regime_history = []
        self.regime_performance = {}

        # Initialize regime templates
        self._initialize_regime_templates()

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="market regime detection",
    )
    def _calculate_regime_features(self, market_data: pd.DataFrame) -> dict[str, float]:
        """Calculate features for regime detection."""
        # Calculate volatility (rolling standard deviation of returns)
        returns = market_data["close"].pct_change().dropna()
        volatility = returns.rolling(window=20).std().iloc[-1]

        # Calculate trend strength
        high_low_diff = market_data["high"] - market_data["low"]
        close_change = market_data["close"].diff()
        trend_strength = (
            abs(close_change).rolling(window=14).mean().iloc[-1]
            / high_low_diff.rolling(window=14).mean().iloc[-1]
        )

        # Calculate momentum
        momentum_short = market_data["close"].pct_change(5).iloc[-1]
        momentum_long = market_data["close"].pct_change(20).iloc[-1]

        return {
            "volatility": volatility,
            "trend_strength": trend_strength,
            "momentum_short": momentum_short,
            "momentum_long": momentum_long,
        }

    def _classify_regime(self, features: dict[str, float]) -> str:
        """Classify market regime based on features."""
        volatility = features["volatility"]
        trend_strength = features["trend_strength"]
        momentum_short = features["momentum_short"]
        momentum_long = features["momentum_long"]

        # Classification logic
        if trend_strength > self.regime_detection_config["trend_threshold"]:
            if momentum_short > 0 and momentum_long > 0:
                return "bull"
            return "bear"
        if volatility < 0.008:  # Low volatility
            return "candle"
        if trend_strength < 0.1:  # Very low trend strength
            return "sr"
        return "sideways"

    def _calculate_regime_confidence(
        self,
        features: dict[str, float],
        regime_type: str,
    ) -> float:
        """Calculate confidence in regime classification."""
        confidence = 0.7

        if regime_type == "volatile":
            if features["volatility"] > 0.04:
                confidence += 0.2
        elif regime_type.startswith("trending"):
            if features["trend_strength"] > 0.6:
                confidence += 0.2
        elif regime_type == "ranging":
            if features["volatility"] < 0.015 and features["trend_strength"] < 0.4:
                confidence += 0.2

        return min(1.0, confidence)

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="adaptive optimization",
    )
    def _update_regime_performance(self, regime_name: str, results: dict[str, Any]) -> None:
        """Update performance tracking for regime."""
        if regime_name not in self.regime_performance:
            self.regime_performance[regime_name] = []

        self.regime_performance[regime_name].append(
            {
                "timestamp": pd.Timestamp.now(),
                "score": results.get("best_score", 0),
                "params": results.get("best_params", {}),
            },
        )


class RegimeSpecificOptimizer:
    """Optimizer specialized for a specific market regime."""

    def __init__(self, regime: MarketRegime, config: dict[str, Any]) -> None:
        self.regime = regime
        self.config = config
        self.logger = system_logger.getChild(f"RegimeOptimizer_{regime.name}")

        # Regime-specific constraints
        self.constraints = self._get_regime_constraints(regime)

    def _get_regime_constraints(self, regime: MarketRegime) -> dict[str, Any]:
        """Get optimization constraints for specific regime."""
        if regime.regime_type == "trending":
            return {
                "tp_multiplier_range": (2.0, 5.0),
                "sl_multiplier_range": (1.0, 2.5),
                "position_size_range": (0.08, 0.25),
            }
        if regime.regime_type == "ranging":
            return {
                "tp_multiplier_range": (1.5, 3.0),
                "sl_multiplier_range": (0.8, 1.5),
                "position_size_range": (0.05, 0.15),
            }
        if regime.regime_type == "volatile":
            return {
                "tp_multiplier_range": (3.0, 6.0),
                "sl_multiplier_range": (1.5, 3.0),
                "position_size_range": (0.03, 0.12),
            }
        return {
            "tp_multiplier_range": (1.5, 4.0),
            "sl_multiplier_range": (1.0, 2.0),
            "position_size_range": (0.05, 0.20),
        }

    def run_optimization(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """Run optimization for specific regime."""
        # Create study
        study = optuna.create_study(direction="maximize")

        # Run optimization
        study.optimize(objective, n_trials=50, show_progress_bar=False)

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "regime_confidence": self.regime.confidence,
        }

    def _regime_objective(
        self,
        trial: optuna.trial.Trial,
        market_data: pd.DataFrame,
    ) -> float:
        """Objective function for regime-specific optimization."""
        # Suggest parameters within regime constraints
        params = self._suggest_regime_parameters(trial)

        # Evaluate parameters
        return self._evaluate_regime_parameters(params, market_data)

    def _suggest_regime_parameters(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        """Suggest parameters within regime-specific constraints."""
        params = {}

        # Trading parameters with regime-specific ranges
        tp_range = self.constraints["tp_multiplier_range"]
        sl_range = self.constraints["sl_multiplier_range"]
        pos_range = self.constraints["position_size_range"]

        params["tp_multiplier"] = trial.suggest_float(
            "tp_multiplier",
            tp_range[0],
            tp_range[1],
        )
        params["sl_multiplier"] = trial.suggest_float(
            "sl_multiplier",
            sl_range[0],
            sl_range[1],
        )
        params["position_size"] = trial.suggest_float(
            "position_size",
            pos_range[0],
            pos_range[1],
        )

        # Model parameters
        params["learning_rate"] = trial.suggest_float(
            "learning_rate",
            1e-4,
            1e-1,
            log=True,
        )
        params["max_depth"] = trial.suggest_int("max_depth", 3, 12)

        return params

    def _evaluate_regime_parameters(
        self,
        params: dict[str, Any],
        market_data: pd.DataFrame,
    ) -> float:
        """Evaluate parameters for specific regime."""
        # Mock evaluation - would integrate with your backtesting
        base_score = 0.5

        # Adjust score based on regime-specific criteria
        if self.regime.regime_type == "trending":
            if params["tp_multiplier"] > params["sl_multiplier"] * 1.5:
                base_score += 0.2
        elif self.regime.regime_type == "ranging":
            if 1.5 <= params["tp_multiplier"] <= 2.5:
                base_score += 0.15
        elif self.regime.regime_type == "support_resistance":
            if 1.8 <= params["tp_multiplier"] <= 3.0:
                base_score += 0.15
        elif self.regime.regime_type == "pattern_based":
            if params["tp_multiplier"] <= 2.0:
                base_score += 0.1

        # Add noise for realistic evaluation
        noise = np.random.normal(0, 0.1)
        return max(0, min(1, base_score + noise))
