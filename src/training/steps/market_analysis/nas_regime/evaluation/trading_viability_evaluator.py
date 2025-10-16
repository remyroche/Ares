"""
Trading Viability Evaluator for NAS Regime Detection.

This module provides trading viability evaluation for regime detection results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TradingViabilityMetrics:
    """Trading viability metrics."""
    signal_quality: float
    regime_stability: float
    prediction_accuracy: float
    trading_viability: float

class TradingViabilityEvaluator:
    """Evaluates trading viability of regime detection results."""

    def __init__(self, viability_threshold: float = 0.6):
        """Initialize the trading viability evaluator."""
        self.viability_threshold = viability_threshold
        self.logger = logging.getLogger(__name__)

    def evaluate_regime_trading_viability(
        self,
        regime_predictions: np.ndarray,
        market_data: pd.DataFrame,
        actual_regimes: Optional[np.ndarray] = None
    ) -> TradingViabilityMetrics:
        """Evaluate trading viability of regime predictions."""
        try:
            # Calculate signal quality
            signal_quality = self._calculate_signal_quality(regime_predictions)

            # Calculate regime stability
            regime_stability = self._calculate_regime_stability(regime_predictions)

            # Calculate prediction accuracy if actual regimes available
            prediction_accuracy = 0.0
            if actual_regimes is not None:
                prediction_accuracy = self._calculate_prediction_accuracy(
                    regime_predictions, actual_regimes
                )

            # Calculate overall trading viability
            trading_viability = self._calculate_trading_viability(
                signal_quality, regime_stability, prediction_accuracy
            )

            return TradingViabilityMetrics(
                signal_quality=signal_quality,
                regime_stability=regime_stability,
                prediction_accuracy=prediction_accuracy,
                trading_viability=trading_viability
            )

        except Exception as e:
            self.logger.error(f"Error evaluating trading viability: {e}")
            return TradingViabilityMetrics(0, 0, 0, 0)

    def _calculate_signal_quality(self, predictions: np.ndarray) -> float:
        """Calculate signal quality based on prediction consistency."""
        if len(predictions) == 0:
            return 0.0

        # Calculate consistency (lower variance = higher quality)
        consistency = 1.0 / (1.0 + np.var(predictions))

        # Calculate signal strength (distance from neutral)
        signal_strength = np.mean(np.abs(predictions - 0.5))

        return (consistency + signal_strength) / 2.0

    def _calculate_regime_stability(self, predictions: np.ndarray) -> float:
        """Calculate regime stability."""
        if len(predictions) < 2:
            return 0.0

        # Calculate regime change frequency
        changes = np.sum(np.diff(predictions) != 0)
        stability = 1.0 - (changes / len(predictions))

        return max(0.0, stability)

    def _calculate_prediction_accuracy(
        self,
        predictions: np.ndarray,
        actual: np.ndarray
    ) -> float:
        """Calculate prediction accuracy."""
        if len(predictions) != len(actual):
            return 0.0

        # Calculate accuracy
        accuracy = np.mean(predictions == actual)
        return accuracy

    def _calculate_trading_viability(
        self,
        signal_quality: float,
        regime_stability: float,
        prediction_accuracy: float
    ) -> float:
        """Calculate overall trading viability score."""
        # Weighted combination
        trading_viability = (
            0.4 * signal_quality +
            0.3 * regime_stability +
            0.3 * prediction_accuracy
        )

        return trading_viability
