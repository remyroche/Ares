"""
Transition Regime Handler

Handles market transitions (cluster -1) by analyzing intensity scores and
predicting emerging trends using step9_5 and step10 models.
"""

from typing import Any
import logging

from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TransitionType(Enum):
    """Types of market transitions."""

    TREND_EMERGENCE = "trend_emergence"
    TREND_CONTINUATION = "trend_continuation"
    TREND_REVERSAL = "trend_reversal"
    RANGE_BREAKOUT = "range_breakout"
    VOLATILITY_SPIKE = "volatility_spike"
    UNCLEAR_TRANSITION = "unclear_transition"


@dataclass
class TransitionAnalysis:
    """Analysis results for market transitions."""

    transition_type: TransitionType
    primary_regime: str
    secondary_regime: str | None
    tertiary_regime: str | None
    regime_weights: dict[str, float]
    confidence_score: float
    predicted_direction: str | None  # "bull", "bear", "sideways"
    step9_5_prediction: dict[str, Any] | None
    step10_prediction: dict[str, Any] | None
    intensity_threshold_met: bool


class TransitionRegimeHandler:
    """
    Handles market transitions by analyzing intensity scores and model predictions.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.intensity_threshold = config.get("transition_intensity_threshold", 0.3)
        self.min_combined_intensity = config.get("min_combined_intensity", 0.6)
        self.max_regimes_to_consider = config.get("max_regimes_to_consider", 3)
        self.step9_5_model = None
        self.step10_model = None

    def set_models(self, step9_5_model, step10_model):
        """Set the step9_5 and step10 models for transition prediction."""
        self.step9_5_model = step9_5_model
        self.step10_model = step10_model

    def analyze_transition(
        self, intensity_scores: dict[str, float],
        current_features: pd.DataFrame, historical_intensities: pd.DataFrame | None = None,
    ) -> TransitionAnalysis:
        """
        Analyze a market transition to determine the best trading approach.

        Args:
            intensity_scores: Current intensity scores for all regimes
            current_features: Current market features for model prediction
            historical_intensities: Historical intensity scores for trend analysis

        Returns:
            TransitionAnalysis with regime weights and predictions
        """
        logger.info("🔍 Analyzing market transition...")

        # 1. Find the strongest regimes (even if below threshold)
        regime_intensities = self._get_top_regime_intensities(intensity_scores)

        # 2. Check if combined intensity is sufficient
        combined_intensity = sum(regime_intensities.values())
        intensity_threshold_met = combined_intensity >= self.min_combined_intensity

        # 3. Get step9_5 predictions for regime transition
        step9_5_prediction = self._get_step9_5_prediction(current_features)

        # 4. Get step10 predictions for path classification
        step10_prediction = self._get_step10_prediction(current_features)

        # 5. Determine transition type and regime weights
        transition_type, regime_weights, confidence = self._determine_transition_type(
            regime_intensities, step9_5_prediction,
            step10_prediction, historical_intensities,
        )

        # 6. Extract primary regimes
        sorted_regimes = sorted(
            regime_weights.items(),
            key=lambda x: x[1],
            reverse=True)
        primary_regime = sorted_regimes[0][0] if sorted_regimes else None
        secondary_regime = sorted_regimes[1][0] if len(sorted_regimes) > 1 else None
        tertiary_regime = sorted_regimes[2][0] if len(sorted_regimes) > 2 else None

        # 7. Predict market direction
        predicted_direction = self._predict_market_direction(
            regime_weights, step9_5_prediction,
            step10_prediction)

        analysis = TransitionAnalysis(
            transition_type=transition_type,
            primary_regime=primary_regime,
            secondary_regime=secondary_regime,
            tertiary_regime=tertiary_regime,
            regime_weights=regime_weights,
            confidence_score=confidence,
            predicted_direction=predicted_direction,
            step9_5_prediction=step9_5_prediction,
            step10_prediction=step10_prediction,
            intensity_threshold_met=intensity_threshold_met,
        )

        logger.info(
            f"📊 Transition Analysis: {transition_type.value} | "
            f"Primary: {primary_regime} | Confidence: {confidence:.2f}",
        )

        return analysis

    def _get_top_regime_intensities(
        self, intensity_scores: dict[str, float],
    ) -> dict[str, float]:
        """Get the top regime intensities = sorted by strength."""
        # Filter out cluster -1 (current transition state)
        filtered_scores = {
            k: v
            for k, v in intensity_scores.items()
            if k != "intensity_cluster_-1" and v > 0
        }

        # Sort by intensity and take top N
        sorted_scores = sorted(
            filtered_scores.items(),
            key=lambda x: x[1],
            reverse=True)
        top_scores = sorted_scores[: self.max_regimes_to_consider]

        return dict(top_scores)

    def _get_step9_5_prediction(
        self, features: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """Get step9_5 prediction for regime transition."""
        if self.step9_5_model is None:
            logger.warning("Step9_5 model not available for transition prediction")
            return None

        try:
            # Get regime transition prediction
            prediction = self.step9_5_model.predict(features)

            # Extract relevant predictions
            return {
                "next_regime_probability": prediction.get("regime_probabilities", {}),
                "price_direction": prediction.get("price_direction", "unknown"),
                "tpsl_prediction": prediction.get("tpsl_prediction", {}),
                "time_to_target": prediction.get("time_to_target", None),
            }
        except Exception as e:
            logger.exception(f"Error getting step9_5 prediction: {e}")
            return None

    def _get_step10_prediction(
        self, features: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """Get step10 prediction for path classification."""
        if self.step10_model is None:
            logger.warning("Step10 model not available for transition prediction")
            return None

        try:
            # Get path classification prediction
            prediction = self.step10_model.predict(features)

            return {
                "path_class": prediction.get("path_class", "unknown"),
                "path_probability": prediction.get("path_probability", 0.0),
                "horizon_predictions": prediction.get("horizon_predictions", {}),
                "reliability_score": prediction.get("reliability_score", 0.0),
            }
        except Exception as e:
            logger.exception(f"Error getting step10 prediction: {e}")
            return None

    def _determine_transition_type(
        self, regime_intensities: dict[str, float],
        step9_5_prediction: dict[str, Any] | None,
        step10_prediction: dict[str, Any] | None,
        historical_intensities: pd.DataFrame | None = None) -> tuple[TransitionType, dict[str, float], float]:
        """Determine the type of transition and calculate regime weights."""

        # Base weights on intensity scores
        total_intensity = sum(regime_intensities.values())
        if total_intensity == 0:
            return TransitionType.UNCLEAR_TRANSITION, {}, 0.0

        # Normalize weights
        base_weights = {k: v / total_intensity for k, v in regime_intensities.items()}

        # Adjust weights based on step9_5 prediction
        if step9_5_prediction:
            base_weights = self._adjust_weights_with_step9_5(
                base_weights, step9_5_prediction,
            )

        # Adjust weights based on step10 prediction
        if step10_prediction:
            base_weights = self._adjust_weights_with_step10(
                base_weights, step10_prediction,
            )

        # Determine transition type
        transition_type = self._classify_transition_type(
            base_weights, step9_5_prediction,
            step10_prediction)

        # Calculate confidence score
        confidence = self._calculate_confidence(
            base_weights, step9_5_prediction,
            step10_prediction)

        return transition_type, base_weights, confidence

    def _adjust_weights_with_step9_5(
        self, weights: dict[str, float],
        prediction: dict[str, Any],
    ) -> dict[str, float]:
        """Adjust regime weights based on step9_5 predictions."""
        adjusted_weights = weights.copy()

        # Boost weights for regimes predicted by step9_5
        next_regime_probs = prediction.get("next_regime_probability", {})
        for regime, prob in next_regime_probs.items():
            if regime in adjusted_weights:
                # Boost weight by prediction probability
                boost_factor = 1.0 + (prob * 0.5)  # Max 50% boost
                adjusted_weights[regime] *= boost_factor

        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {
                k: v / total_weight for k, v in adjusted_weights.items()
            }

        return adjusted_weights

    def _adjust_weights_with_step10(
        self, weights: dict[str, float],
        prediction: dict[str, Any],
    ) -> dict[str, float]:
        """Adjust regime weights based on step10 path classification."""
        adjusted_weights = weights.copy()

        path_class = prediction.get("path_class", "unknown")
        path_prob = prediction.get("path_probability", 0.0)

        # Boost weights based on path class
        if path_class == "beginning_of_trend":
            # Boost trend-related regimes
            for regime in adjusted_weights:
                if "TREND" in regime or "BULL" in regime or "BEAR" in regime:
                    adjusted_weights[regime] *= 1.0 + path_prob * 0.3

        elif path_class == "continuation":
            # Boost current dominant regime
            dominant_regime = max(adjusted_weights.items(), key=lambda x: x[1])[0]
            adjusted_weights[dominant_regime] *= 1.0 + path_prob * 0.2

        elif path_class == "reversal":
            # Boost opposite trend regimes
            for regime in adjusted_weights:
                if (
                    "BULL" in regime
                    and "BEAR" not in regime
                    or "BEAR" in regime
                    and "BULL" not in regime
                ):
                    adjusted_weights[regime] *= 1.0 - path_prob * 0.2

        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {
                k: v / total_weight for k , v in adjusted_weights.items()
            }

        return adjusted_weights

    def _classify_transition_type(
        self, weights: dict[str, float],
        step9_5_prediction: dict[str, Any] | None,
        step10_prediction: dict[str, Any] | None,
    ) -> TransitionType:
        """Classify the type of transition based on weights and predictions."""

        # Get top regime
        top_regime = max(weights.items(), key=lambda x: x[1])[0] if weights else None

        if not top_regime:
            return TransitionType.UNCLEAR_TRANSITION

        # Check step10 path classification
        if step10_prediction:
            path_class = step10_prediction.get("path_class", "unknown")

            if path_class == "beginning_of_trend":
                if (
                    "TREND" in top_regime
                    or "BULL" in top_regime
                    or "BEAR" in top_regime
                ):
                    return TransitionType.TREND_EMERGENCE
                return TransitionType.RANGE_BREAKOUT

            if path_class == "reversal":
                return TransitionType.TREND_REVERSAL

            if path_class == "continuation":
                return TransitionType.TREND_CONTINUATION

        # Fallback classification based on regime type
        if "VOLATILITY" in top_regime:
            return TransitionType.VOLATILITY_SPIKE
        if "TREND" in top_regime or "BULL" in top_regime or "BEAR" in top_regime:
            return TransitionType.TREND_EMERGENCE
        return TransitionType.RANGE_BREAKOUT

    def _predict_market_direction(
        self, weights: dict[str, float],
        step9_5_prediction: dict[str, Any] | None,
        step10_prediction: dict[str, Any] | None,
    ) -> str | None:
        """Predict overall market direction."""

        # Use step9_5 price direction if available
        if step9_5_prediction:
            direction = step9_5_prediction.get("price_direction", "unknown")
            if direction != "unknown":
                return direction

        # Fallback: analyze regime weights
        bull_weight = sum(w for regime, w in weights.items() if "BULL" in regime)
        bear_weight = sum(w for regime, w in weights.items() if "BEAR" in regime)

        if bull_weight > bear_weight + 0.1:
            return "bull"
        if bear_weight > bull_weight + 0.1:
            return "bear"
        return "sideways"

    def _calculate_confidence(
        self, weights: dict[str, float],
        step9_5_prediction: dict[str, Any] | None,
        step10_prediction: dict[str, Any] | None,
    ) -> float:
        """Calculate confidence score for the transition analysis."""

        confidence_factors = []

        # Factor 1: Weight concentration (higher is better)
        if weights:
            max_weight = max(weights.values())
            confidence_factors.append(max_weight)

        # Factor 2: Step9_5 prediction confidence
        if step9_5_prediction:
            # Use TPSL prediction confidence if available
            tpsl_conf = step9_5_prediction.get("tpsl_prediction", {}).get(
                "confidence",
                0.5,
            )
            confidence_factors.append(tpsl_conf)

        # Factor 3: Step10 reliability score
        if step10_prediction:
            reliability = step10_prediction.get("reliability_score", 0.5)
            confidence_factors.append(reliability)

        # Factor 4: Path probability
        if step10_prediction:
            path_prob = step10_prediction.get("path_probability", 0.5)
            confidence_factors.append(path_prob)

        # Average confidence factors
        if confidence_factors:
            return np.mean(confidence_factors)
        return 0.5

    def get_trading_recommendation(
        self, analysis: TransitionAnalysis,
    ) -> dict[str, Any]:
        """Get trading recommendations based on transition analysis."""

        recommendation = {
            "action": "HOLD",  # Default action
            "position_size": 0.0,
            "stop_loss": None, "take_profit": None,
            "confidence": analysis.confidence_score, "reasoning": [],
        }

        # Only trade if intensity threshold is met
        if not analysis.intensity_threshold_met:
            recommendation["reasoning"].append(
                "Insufficient regime intensity for trading",
            )
            return recommendation

        # Determine action based on transition type
        if analysis.transition_type == TransitionType.TREND_EMERGENCE:
            if analysis.predicted_direction == "bull":
                recommendation["action"] = "BUY"
                recommendation["position_size"] = 0.7  # Aggressive for trend emergence
                recommendation["reasoning"].append("Trend emergence detected - bullish")
            elif analysis.predicted_direction == "bear":
                recommendation["action"] = "SELL"
                recommendation["position_size"] = 0.7
                recommendation["reasoning"].append("Trend emergence detected - bearish")

        elif analysis.transition_type == TransitionType.TREND_CONTINUATION:
            if analysis.predicted_direction == "bull":
                recommendation["action"] = "BUY"
                recommendation["position_size"] = 0.5  # Moderate for continuation
                recommendation["reasoning"].append("Trend continuation detected")
            elif analysis.predicted_direction == "bear":
                recommendation["action"] = "SELL"
                recommendation["position_size"] = 0.5
                recommendation["reasoning"].append("Trend continuation detected")

        elif analysis.transition_type == TransitionType.RANGE_BREAKOUT:
            recommendation["action"] = "WAIT"
            recommendation["reasoning"].append(
                "Range breakout detected - wait for confirmation",
            )

        elif analysis.transition_type == TransitionType.VOLATILITY_SPIKE:
            recommendation["action"] = "REDUCE"
            recommendation["position_size"] = 0.2  # Very conservative
            recommendation["reasoning"].append(
                "High volatility detected - reduce exposure",
            )

        # Adjust position size based on confidence
        if analysis.confidence_score < 0.6:
            recommendation["position_size"] *= 0.5
            recommendation["reasoning"].append("Low confidence - reduced position size")

        return recommendation
