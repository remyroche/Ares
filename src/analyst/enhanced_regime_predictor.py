#!/usr/bin/env python3
"""Enhanced Regime Predictor.

This module provides advanced regime change prediction capabilities by integrating:
- Probability-based regime change detection
- Adaptive regime boundaries
- Regime persistence modeling
- Multi-signal regime change detection
- Confidence scoring for predictions
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import expon, gamma, weibull_min
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils.centralized_decorators import handle_errors, with_tracing_span
from src.utils.logger import system_logger


class EnhancedRegimePredictor:
    """Enhanced regime predictor with advanced change detection capabilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        self.logger = system_logger.getChild("EnhancedRegimePredictor")

        # Configuration parameters
        self.stability_threshold = self.config.get("stability_threshold", 0.1)
        self.min_persistence = self.config.get("min_persistence", 3)
        self.entropy_percentile = self.config.get("entropy_percentile", 75)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)

        # Model state
        self.regime_boundaries = None
        self.persistence_model = None
        self.transition_matrix = None
        self.boundary_scaler = None

    @with_tracing_span("enhanced_regime_predictor.predict_regime_changes")
    @handle_errors(
        exceptions=(Exception,),
        default_return={
            "success": False,
            "predictions": [],
            "error": "Prediction failed",
        },
        context="enhanced_regime_prediction",
    )
    def predict_regime_changes(
        self, features: pd.DataFrame, hmm_probs: np.ndarray, hmm_states: np.ndarray
    ) -> Dict[str, Any]:
        """Predict regime changes using enhanced multi-signal approach.

        Args:
            features: Feature DataFrame
            hmm_probs: HMM state probabilities
            hmm_states: HMM state sequence

        Returns:
            Dictionary with regime change predictions and confidence scores
        """
        try:
            self.logger.info("🔮 Predicting regime changes with enhanced model...")

            # Calculate regime stability and entropy
            regime_stability = self._calculate_regime_stability(hmm_probs)
            regime_entropy = self._calculate_regime_entropy(hmm_probs)

            # Detect regime changes using multiple signals
            regime_changes = self._detect_regime_changes_multi_signal(
                hmm_states, regime_stability, regime_entropy
            )

            # Calculate transition probabilities
            transition_probs = self._calculate_transition_probabilities(
                hmm_probs, regime_changes
            )

            # Calculate confidence scores
            confidence_scores = self._calculate_prediction_confidence(
                regime_stability, regime_entropy, transition_probs
            )

            # Apply persistence model if available
            if self.persistence_model:
                persistence_adjustments = self._apply_persistence_model(
                    regime_changes, hmm_states
                )
                confidence_scores *= persistence_adjustments

            # Create prediction events
            predictions = self._create_prediction_events(
                regime_changes, hmm_states, transition_probs, confidence_scores
            )

            # Filter predictions by confidence threshold
            high_confidence_predictions = [
                pred
                for pred in predictions
                if pred["confidence"] >= self.confidence_threshold
            ]

            self.logger.info(
                f"✅ Predicted {len(high_confidence_predictions)} high-confidence regime changes"
            )

            return {
                "success": True,
                "predictions": high_confidence_predictions,
                "all_predictions": predictions,
                "regime_stability": regime_stability.tolist(),
                "regime_entropy": regime_entropy.tolist(),
                "transition_probabilities": transition_probs.tolist(),
                "confidence_scores": confidence_scores.tolist(),
            }

        except Exception as e:
            self.logger.exception(f"❌ Enhanced regime prediction failed: {e}")
            return {"success": False, "predictions": [], "error": str(e)}

    @handle_errors(
        exceptions=(Exception,),
        default_return=np.zeros(0, dtype=float),
        context="calculate_regime_stability",
    )
    def _calculate_regime_stability(self, hmm_probs: np.ndarray) -> np.ndarray:
        """Calculate regime stability (max probability for each timepoint)."""
        try:
            return np.max(hmm_probs, axis=1)
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating regime stability: {e}")
            return np.zeros(len(hmm_probs))

    @handle_errors(
        exceptions=(Exception,),
        default_return=np.zeros(0, dtype=float),
        context="calculate_regime_entropy",
    )
    def _calculate_regime_entropy(self, hmm_probs: np.ndarray) -> np.ndarray:
        """Calculate regime entropy (uncertainty measure)."""
        try:
            eps = 1e-10
            entropy = -np.sum(hmm_probs * np.log(hmm_probs + eps), axis=1)
            return entropy
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating regime entropy: {e}")
            return np.zeros(len(hmm_probs))

    @handle_errors(
        exceptions=(Exception,),
        default_return=np.zeros(0, dtype=bool),
        context="detect_regime_changes_multi_signal",
    )
    def _detect_regime_changes_multi_signal(
        self, hmm_states: np.ndarray, stability: np.ndarray, entropy: np.ndarray
    ) -> np.ndarray:
        """Detect regime changes using multiple signals."""
        try:
            changes = np.zeros(len(hmm_states), dtype=bool)

            # Signal 1: State transitions
            state_changes = np.diff(hmm_states, prepend=hmm_states[0]) != 0

            # Signal 2: Stability drops
            stability_threshold = np.percentile(stability, 25)
            stability_changes = stability < stability_threshold

            # Signal 3: High entropy
            entropy_threshold = np.percentile(entropy, self.entropy_percentile)
            entropy_changes = entropy > entropy_threshold

            # Signal 4: Stability acceleration (rate of change)
            stability_acceleration = np.diff(stability, prepend=stability[0])
            acceleration_threshold = np.percentile(stability_acceleration, 25)
            acceleration_changes = stability_acceleration < acceleration_threshold

            # Combine signals with weighted approach
            for i in range(1, len(hmm_states)):
                signal_score = 0

                if state_changes[i]:
                    signal_score += 0.4  # State change is most important
                if stability_changes[i]:
                    signal_score += 0.3  # Stability drop
                if entropy_changes[i]:
                    signal_score += 0.2  # High entropy
                if acceleration_changes[i]:
                    signal_score += 0.1  # Stability acceleration

                # Require minimum signal score and persistence
                if signal_score >= 0.5 and i >= self.min_persistence:
                    changes[i] = True

            return changes

        except Exception as e:
            self.logger.warning(f"⚠️ Error in multi-signal regime change detection: {e}")
            return np.zeros(len(hmm_states), dtype=bool)

    @handle_errors(
        exceptions=(Exception,),
        default_return=np.zeros(0, dtype=float),
        context="calculate_transition_probabilities",
    )
    def _calculate_transition_probabilities(
        self, hmm_probs: np.ndarray, regime_changes: np.ndarray
    ) -> np.ndarray:
        """Calculate transition probabilities for regime changes."""
        try:
            transition_probs = np.zeros(len(regime_changes))

            for i in range(len(regime_changes)):
                if regime_changes[i] and i < len(hmm_probs) - 1:
                    # Calculate probability change magnitude
                    prob_change = np.abs(hmm_probs[i + 1] - hmm_probs[i])
                    max_change = np.max(prob_change)

                    # Normalize to probability
                    transition_probs[i] = min(max_change * 5, 1.0)  # Scale and cap

            return transition_probs

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating transition probabilities: {e}")
            return np.zeros(len(regime_changes), dtype=float)

    @handle_errors(
        exceptions=(Exception,),
        default_return=np.zeros(0, dtype=float),
        context="calculate_prediction_confidence",
    )
    def _calculate_prediction_confidence(
        self, stability: np.ndarray, entropy: np.ndarray, transition_probs: np.ndarray
    ) -> np.ndarray:
        """Calculate confidence scores for regime change predictions."""
        try:
            confidence_scores = np.zeros(len(stability))

            for i in range(len(stability)):
                # Base confidence from stability
                stability_confidence = stability[i]

                # Entropy penalty (high entropy reduces confidence)
                entropy_penalty = (
                    entropy[i] / np.max(entropy) if np.max(entropy) > 0 else 0
                )

                # Transition probability boost
                transition_boost = (
                    transition_probs[i] if i < len(transition_probs) else 0
                )

                # Combined confidence score
                confidence = (
                    stability_confidence * 0.4
                    + (1 - entropy_penalty) * 0.3
                    + transition_boost * 0.3
                )

                confidence_scores[i] = np.clip(confidence, 0, 1)

            return confidence_scores

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating prediction confidence: {e}")
            return np.zeros(len(stability), dtype=float)

    @handle_errors(
        exceptions=(Exception,),
        default_return=np.ones(0, dtype=float),
        context="apply_persistence_model",
    )
    def _apply_persistence_model(
        self, regime_changes: np.ndarray, hmm_states: np.ndarray
    ) -> np.ndarray:
        """Apply persistence model to adjust confidence scores."""
        try:
            if not self.persistence_model:
                return np.ones(len(regime_changes), dtype=float)

            adjustments = np.ones(len(regime_changes), dtype=float)

            # Calculate current regime durations
            durations = self._calculate_regime_durations(hmm_states)

            # Get survival function from persistence model
            survival_func = self.persistence_model.get("survival_function")
            if survival_func:
                for i in range(len(regime_changes)):
                    if regime_changes[i] and i < len(durations):
                        current_duration = durations[i]

                        # Calculate survival probability
                        survival_prob = survival_func(current_duration)

                        # Adjust confidence based on survival probability
                        # Higher survival probability means regime should persist longer
                        # So we reduce confidence for early transitions
                        adjustments[i] = 1 - survival_prob

            return adjustments

        except Exception as e:
            self.logger.warning(f"⚠️ Error applying persistence model: {e}")
            return np.ones(len(regime_changes), dtype=float)

    @handle_errors(
        exceptions=(Exception,), default_return=[], context="create_prediction_events"
    )
    def _create_prediction_events(
        self,
        regime_changes: np.ndarray,
        hmm_states: np.ndarray,
        transition_probs: np.ndarray,
        confidence_scores: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Create detailed prediction events."""
        try:
            events = []

            for i in range(len(regime_changes)):
                if regime_changes[i] and i < len(hmm_states) - 1:
                    event = {
                        "timestamp_index": i,
                        "from_state": int(hmm_states[i]),
                        "to_state": int(hmm_states[i + 1]),
                        "transition_probability": float(transition_probs[i]),
                        "confidence": float(confidence_scores[i]),
                        "prediction_type": "regime_change",
                        "prediction_horizon": 1,  # Next bar
                        "prediction_metadata": {
                            "method": "enhanced_multi_signal",
                            "signals_used": [
                                "state_transition",
                                "stability",
                                "entropy",
                                "acceleration",
                            ],
                        },
                    }
                    events.append(event)

            return events

        except Exception as e:
            self.logger.warning(f"⚠️ Error creating prediction events: {e}")
            return []

    @handle_errors(
        exceptions=(Exception,),
        default_return=np.zeros(0, dtype=int),
        context="calculate_regime_durations",
    )
    def _calculate_regime_durations(self, states: np.ndarray) -> np.ndarray:
        """Calculate how long each regime persists."""
        try:
            durations = np.zeros(len(states), dtype=int)
            current_state = states[0]
            current_duration = 1

            for i in range(1, len(states)):
                if states[i] == current_state:
                    current_duration += 1
                else:
                    # Update durations for the previous regime
                    for j in range(i - current_duration, i):
                        durations[j] = current_duration
                    current_state = states[i]
                    current_duration = 1

            # Handle the last regime
            for j in range(len(states) - current_duration, len(states)):
                durations[j] = current_duration

            return durations

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating regime durations: {e}")
            return np.zeros(len(states), dtype=int)

    @with_tracing_span("enhanced_regime_predictor.fit_persistence_model")
    @handle_errors(
        exceptions=(Exception,), default_return=False, context="fit_persistence_model"
    )
    def fit_persistence_model(self, regime_sequence: np.ndarray) -> bool:
        """Fit regime persistence model using statistical distributions."""
        try:
            self.logger.info("📊 Fitting regime persistence model...")

            # Calculate regime durations
            durations = self._calculate_regime_durations(regime_sequence)
            unique_durations = np.unique(durations)

            if len(unique_durations) < 3:
                self.logger.warning("⚠️ Insufficient regime duration data for modeling")
                return False

            # Fit multiple distributions
            distribution_fits = {}

            # Weibull distribution
            try:
                shape, loc, scale = weibull_min.fit(durations)
                distribution_fits["weibull"] = {
                    "shape": float(shape),
                    "scale": float(scale),
                    "mean_duration": float(scale * np.exp(1 / shape)),
                    "survival_function": lambda t: weibull_min.sf(t, shape, loc, scale),
                    "aic": self._calculate_aic(
                        durations, weibull_min.pdf, shape, loc, scale
                    ),
                }
            except Exception as e:
                self.logger.warning(f"⚠️ Weibull fit failed: {e}")

            # Exponential distribution
            try:
                loc, scale = expon.fit(durations)
                distribution_fits["exponential"] = {
                    "scale": float(scale),
                    "mean_duration": float(scale),
                    "survival_function": lambda t: expon.sf(t, loc, scale),
                    "aic": self._calculate_aic(durations, expon.pdf, loc, scale),
                }
            except Exception as e:
                self.logger.warning(f"⚠️ Exponential fit failed: {e}")

            # Gamma distribution
            try:
                shape, loc, scale = gamma.fit(durations)
                distribution_fits["gamma"] = {
                    "shape": float(shape),
                    "scale": float(scale),
                    "mean_duration": float(shape * scale),
                    "survival_function": lambda t: gamma.sf(t, shape, loc, scale),
                    "aic": self._calculate_aic(durations, gamma.pdf, shape, loc, scale),
                }
            except Exception as e:
                self.logger.warning(f"⚠️ Gamma fit failed: {e}")

            # Select best fitting distribution
            best_distribution = None
            best_aic = float("inf")

            for dist_name, dist_params in distribution_fits.items():
                if dist_params["aic"] < best_aic:
                    best_aic = dist_params["aic"]
                    best_distribution = dist_name

            if best_distribution:
                self.persistence_model = distribution_fits[best_distribution]
                self.persistence_model["distribution_type"] = best_distribution

                # Calculate persistence statistics
                self.persistence_model["statistics"] = {
                    "mean_duration": float(np.mean(durations)),
                    "median_duration": float(np.median(durations)),
                    "std_duration": float(np.std(durations)),
                    "min_duration": int(np.min(durations)),
                    "max_duration": int(np.max(durations)),
                }

                self.logger.info(f"✅ Fitted {best_distribution} persistence model")
                return True
            else:
                self.logger.warning("⚠️ No valid persistence model could be fitted")
                return False

        except Exception as e:
            self.logger.exception(f"❌ Error fitting persistence model: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,), default_return=float("inf"), context="calculate_aic"
    )
    def _calculate_aic(self, data: np.ndarray, pdf_func, *params) -> float:
        """Calculate Akaike Information Criterion for distribution fitting."""
        try:
            log_likelihood = np.sum(np.log(pdf_func(data, *params) + 1e-10))
            k = len(params)
            aic = 2 * k - 2 * log_likelihood
            return aic
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating AIC: {e}")
            return float("inf")

    @with_tracing_span("enhanced_regime_predictor.fit_adaptive_boundaries")
    @handle_errors(
        exceptions=(Exception,), default_return=False, context="fit_adaptive_boundaries"
    )
    def fit_adaptive_boundaries(self, features: pd.DataFrame) -> bool:
        """Fit adaptive regime boundaries using clustering."""
        try:
            self.logger.info("🔧 Fitting adaptive regime boundaries...")

            # Extract regime characteristics
            regime_features = self._extract_regime_characteristics(features)

            if regime_features.empty:
                self.logger.warning("⚠️ No regime characteristics available")
                return False

            # Scale features
            self.boundary_scaler = StandardScaler()
            scaled_features = self.boundary_scaler.fit_transform(regime_features)

            # Use DBSCAN for adaptive boundary detection
            self.regime_boundaries = DBSCAN(eps=0.1, min_samples=5)
            boundary_labels = self.regime_boundaries.fit_predict(scaled_features)

            # Calculate boundary statistics
            unique_boundaries = np.unique(boundary_labels[boundary_labels >= 0])
            boundary_stats = {}

            for boundary_id in unique_boundaries:
                boundary_mask = boundary_labels == boundary_id
                boundary_features = regime_features[boundary_mask]

                boundary_stats[f"boundary_{boundary_id}"] = {
                    "size": int(np.sum(boundary_mask)),
                    "characteristics": boundary_features.mean().to_dict(),
                    "volatility": float(boundary_features.std().mean()),
                }

            self.logger.info(
                f"✅ Fitted {len(unique_boundaries)} adaptive regime boundaries"
            )
            return True

        except Exception as e:
            self.logger.exception(f"❌ Error fitting adaptive boundaries: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="extract_regime_characteristics",
    )
    def _extract_regime_characteristics(self, features: pd.DataFrame) -> pd.DataFrame:
        """Extract regime characteristics for boundary calculation."""
        try:
            characteristics = pd.DataFrame()

            # Key regime characteristics
            key_features = [
                "price_momentum_10",
                "volatility_20",
                "volume_ratio_10",
                "rsi",
                "adx",
                "bb_position",
                "atr_normalized",
            ]

            for feature in key_features:
                if feature in features.columns:
                    # Calculate rolling statistics
                    characteristics[f"{feature}_mean"] = (
                        features[feature].rolling(20).mean()
                    )
                    characteristics[f"{feature}_std"] = (
                        features[feature].rolling(20).std()
                    )
                    characteristics[f"{feature}_trend"] = features[feature].diff(10)

            # Add regime interaction features
            if (
                "price_momentum_10" in features.columns
                and "volatility_20" in features.columns
            ):
                characteristics["momentum_volatility_ratio"] = features[
                    "price_momentum_10"
                ] / (features["volatility_20"] + 1e-8)

            # Remove NaN values
            characteristics = characteristics.dropna()

            return characteristics

        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting regime characteristics: {e}")
            return pd.DataFrame()

    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of the fitted models."""
        summary = {
            "persistence_model": None,
            "adaptive_boundaries": None,
            "configuration": {
                "stability_threshold": self.stability_threshold,
                "min_persistence": self.min_persistence,
                "entropy_percentile": self.entropy_percentile,
                "confidence_threshold": self.confidence_threshold,
            },
        }

        if self.persistence_model:
            summary["persistence_model"] = {
                "distribution_type": self.persistence_model.get("distribution_type"),
                "mean_duration": self.persistence_model.get("mean_duration"),
                "statistics": self.persistence_model.get("statistics", {}),
            }

        if self.regime_boundaries:
            summary["adaptive_boundaries"] = {
                "n_boundaries": (
                    len(self.regime_boundaries.labels_)
                    if hasattr(self.regime_boundaries, "labels_")
                    else 0
                ),
                "eps": self.regime_boundaries.eps,
                "min_samples": self.regime_boundaries.min_samples,
            }

        return summary
