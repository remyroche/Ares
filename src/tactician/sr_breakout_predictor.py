# src/tactician/sr_breakout_predictor.py

# from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier  # Temporarily commented due to syntax errors
from src.utils.logger import system_logger
from typing import Any
import numpy as np
import pandas as pd
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.centralized_decorators import validate_data_quality

class SRBreakoutPredictor:
    """
    SR Breakout Predictor responsible for predicting support/resistance breakouts.
    This module handles all SR breakout prediction logic and feature engineering.
    Centralized S/R detection using multiple methods:
    - Fractal analysis for swing highs/lows
    - Volume-weighted price levels
    - Traditional pivot points (fallback)
    - ATR-based activation ranges
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize SR breakout predictor.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("SRBreakoutPredictor")

        # SR predictor state
        self.is_initialized: bool = False
        self.sr_predictions: dict[str, Any] = {}

        # Configuration
        self.sr_config: dict[str, Any] = self.config.get("sr_breakout_predictor", {})
        self.enable_sr_breakout_tactics: bool = self.sr_config.get(
            "enable_sr_breakout_tactics",
            True,
        )
        self.sr_proximity_threshold: float = self.sr_config.get(
            "sr_proximity_threshold",
            0.02,
        )
        self.breakout_confidence_threshold: float = self.sr_config.get(
            "breakout_confidence_threshold",
            0.6,
        )
        self.sr_detection_method: str = self.sr_config.get(
            "sr_detection_method",
            "fractal",
        )
        self.min_sr_strength: float = self.sr_config.get(
            "min_sr_strength",
            0.3,
        )
        self.max_sr_levels: int = self.sr_config.get(
            "max_sr_levels",
            10,
        )
        self.sr_lookback_periods: int = self.sr_config.get(
            "sr_lookback_periods",
            100,
        )
        self.volume_weight: float = self.sr_config.get(
            "volume_weight",
            0.7,
        )
        self.price_weight: float = self.sr_config.get(
            "price_weight",
            0.3,
        )
        self.atr_multiplier: float = self.sr_config.get(
            "atr_multiplier",
            1.5,
        )
        self.breakout_confirmation_periods: int = self.sr_config.get(
            "breakout_confirmation_periods",
            3,
        )
        self.false_breakout_filter: bool = self.sr_config.get(
            "false_breakout_filter",
            True,
        )

        # Zone multipliers
        self.support_zone_multiplier: float = self.sr_config.get(
            "support_zone_multiplier",
            0.8,
        )
        self.resistance_zone_multiplier: float = self.sr_config.get(
            "resistance_zone_multiplier",
            1.2,
        )
        self.sr_zone_threshold: float = self.sr_config.get(
            "sr_zone_threshold",
            0.01,
        )
        self.zone_expansion_factor: float = self.sr_config.get(
            "zone_expansion_factor",
            1.1,
        )
        self.zone_contraction_factor: float = self.sr_config.get(
            "zone_contraction_factor",
            0.9,
        )

        # Confidence thresholds
        self.min_sr_confidence: float = self.sr_config.get(
            "min_sr_confidence",
            0.4,
        )
        self.high_confidence_threshold: float = self.sr_config.get(
            "high_confidence_threshold",
            0.8,
        )
        self.confidence_decay_rate: float = self.sr_config.get(
            "confidence_decay_rate",
            0.95,
        )
        self.regime_confidence_boost: float = self.sr_config.get(
            "regime_confidence_boost",
            0.1,
        )
        self.ensemble_confidence_threshold: float = self.sr_config.get(
            "ensemble_confidence_threshold",
            0.7,
        )

        # Feature calculation parameters
        self.feature_config: dict[str, Any] = self.sr_config.get(
            "feature_calculation",
            {},
        )
        self.enable_comprehensive_features: bool = self.feature_config.get(
            "enable_comprehensive_features",
            True,
        )
        self.strength_score_weights: dict[str, float] = self.feature_config.get(
            "strength_score_weights",
            {
                "touch_count": 0.3,
                "total_volume": 0.2,
                "level_age": 0.2,
                "bounce_rate": 0.2,
                "isolation_score": 0.1,
            },
        )

        # LM Model Selection Configuration
        self.lm_config: dict[str, Any] = self.sr_config.get("lm_model_selection", {})
        self.enable_specialist_models: bool = self.lm_config.get(
            "enable_specialist_models",
            True,
        )
        self.sr_proximity_trigger_base: float = self.lm_config.get(
            "sr_proximity_trigger_base",
            0.006,
        )  # 0.6% base proximity
        self.sr_proximity_trigger_min: float = self.lm_config.get(
            "sr_proximity_trigger_min",
            0.003,
        )  # 0.3% minimum proximity
        self.sr_proximity_trigger_max: float = self.lm_config.get(
            "sr_proximity_trigger_max",
            0.015,
        )  # 1.5% maximum proximity
        self.proximity_decay_rate: float = self.lm_config.get(
            "proximity_decay_rate",
            0.98,
        )
        self.proximity_boost_factor: float = self.lm_config.get(
            "proximity_boost_factor",
            1.2,
        )

        # Model ensemble configuration
        self.ensemble_config: dict[str, Any] = self.sr_config.get("ensemble_config", {})
        self.enable_ensemble: bool = self.ensemble_config.get(
            "enable_ensemble",
            True,
        )
        self.ensemble_method: str = self.ensemble_config.get(
            "ensemble_method",
            "weighted_average",
        )
        self.model_weights: dict[str, float] = self.ensemble_config.get(
            "model_weights",
            {
                "fractal": 0.4,
                "volume": 0.3,
                "pivot": 0.2,
                "atr": 0.1,
            },
        )

        # Performance tracking
        self.performance_metrics: dict[str, Any] = {}
        self.prediction_history: list[dict[str, Any]] = []

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid SR breakout predictor configuration"),
            AttributeError: (False, "Missing required SR parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="SR breakout predictor initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the SR breakout predictor."""
        self.logger.info("Initializing SR breakout predictor...")

        try:
            # Validate configuration
            if not self._validate_configuration():
                return False

            # Initialize components
            if not await self._initialize_components():
                return False

            self.is_initialized = True
            self.logger.info("✅ SR breakout predictor initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize SR breakout predictor: {e}")
            return False

    def _validate_configuration(self) -> bool:
        """Validate SR breakout predictor configuration."""
        try:
            required_keys = [
                "sr_proximity_threshold",
                "breakout_confidence_threshold",
                "min_sr_strength",
                "max_sr_levels",
            ]
            for key in required_keys:
                if not hasattr(self, key):
                    self.logger.error(f"Missing required configuration key: {key}")
                    return False

            # Validate values
            if self.sr_proximity_threshold <= 0:
                self.logger.error("Invalid sr_proximity_threshold")
                return False

            if self.breakout_confidence_threshold <= 0 or self.breakout_confidence_threshold >= 1:
                self.logger.error("Invalid breakout_confidence_threshold")
                return False

            if self.min_sr_strength <= 0 or self.min_sr_strength >= 1:
                self.logger.error("Invalid min_sr_strength")
                return False

            if self.max_sr_levels <= 0:
                self.logger.error("Invalid max_sr_levels")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    async def _initialize_components(self) -> bool:
        """Initialize SR breakout predictor components."""
        try:
            # Initialize regime classifier if needed
            if hasattr(self, "regime_classifier"):
                await self.regime_classifier.initialize()

            self.logger.info("✅ SR breakout predictor components initialized")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            return False

    @validate_data_quality(
        required_columns=["open", "high", "low", "close", "volume"],
        min_rows=50,
        max_null_ratio=0.1,
        check_duplicates=True,
        check_timestamps=True,
        context="SR breakout prediction input validation"
    )
    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid input data for SR breakout prediction"),
            AttributeError: (None, "Predictor not properly initialized"),
        },
        default_return=None,
        context="SR breakout prediction",
    )
    async def predict_sr_breakouts(
        self,
        market_data: pd.DataFrame,
        current_price: float,
    ) -> dict[str, Any]:
        """
        Predict support/resistance breakouts.

        Args:
            market_data: Market data DataFrame
            current_price: Current market price

        Returns:
            dict[str, Any]: SR breakout predictions
        """
        if not self.is_initialized:
            self.logger.error("SR breakout predictor not initialized")
            return {}

        try:
            self.logger.info("Predicting SR breakouts...")

            # Detect support and resistance levels
            support_levels = await self._detect_support_levels(market_data)
            resistance_levels = await self._detect_resistance_levels(market_data)

            # Calculate breakout probabilities
            breakout_probabilities = await self._calculate_breakout_probabilities(
                support_levels, resistance_levels, current_price,
            )

            # Calculate confidence scores
            confidence_scores = await self._calculate_confidence_scores(
                support_levels, resistance_levels, market_data,
            )

            # Generate SR features
            sr_features = await self._generate_sr_features(
                support_levels, resistance_levels, market_data,
            )

            # Create predictions
            predictions = {
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "breakout_probabilities": breakout_probabilities,
                "confidence_scores": confidence_scores,
                "sr_features": sr_features,
                "current_price": current_price,
                "timestamp": pd.Timestamp.now(),
            }

            # Store predictions
            self.sr_predictions = predictions

            # Update performance metrics
            self._update_performance_metrics(predictions)

            self.logger.info("✅ SR breakout predictions generated")
            return predictions

        except Exception as e:
            self.logger.error(f"Error predicting SR breakouts: {e}")
            return {}

    async def _detect_support_levels(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect support levels using configured method."""
        try:
            if self.sr_detection_method == "fractal":
                return await self._detect_fractal_support_levels(market_data)
            elif self.sr_detection_method == "volume":
                return await self._detect_volume_support_levels(market_data)
            elif self.sr_detection_method == "pivot":
                return await self._detect_pivot_support_levels(market_data)
            elif self.sr_detection_method == "atr":
                return await self._detect_atr_support_levels(market_data)
            else:
                self.logger.warning(f"Unknown SR detection method: {self.sr_detection_method}")
                return await self._detect_fractal_support_levels(market_data)

        except Exception as e:
            self.logger.error(f"Error detecting support levels: {e}")
            return []

    async def _detect_resistance_levels(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect resistance levels using configured method."""
        try:
            if self.sr_detection_method == "fractal":
                return await self._detect_fractal_resistance_levels(market_data)
            elif self.sr_detection_method == "volume":
                return await self._detect_volume_resistance_levels(market_data)
            elif self.sr_detection_method == "pivot":
                return await self._detect_pivot_resistance_levels(market_data)
            elif self.sr_detection_method == "atr":
                return await self._detect_atr_resistance_levels(market_data)
            else:
                self.logger.warning(f"Unknown SR detection method: {self.sr_detection_method}")
                return await self._detect_fractal_resistance_levels(market_data)

        except Exception as e:
            self.logger.error(f"Error detecting resistance levels: {e}")
            return []

    async def _detect_fractal_support_levels(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect support levels using fractal analysis."""
        try:
            # Implement fractal-based support level detection
            # This is a simplified implementation
            support_levels = []

            # Find local minima in price data
            low_prices = market_data['low'].rolling(window=5, center=True).min()

            # Identify significant support levels
            for i in range(2, len(market_data) - 2):
                if (market_data['low'].iloc[i] == low_prices.iloc[i] and
                    market_data['low'].iloc[i] < market_data['low'].iloc[i-1] and
                    market_data['low'].iloc[i] < market_data['low'].iloc[i+1]):

                    support_level = {
                        "price": market_data['low'].iloc[i],
                        "strength": self._calculate_level_strength(market_data, i, "support"),
                        "timestamp": market_data.index[i],
                        "method": "fractal",
                        "confidence": 0.7,
                    }
                    support_levels.append(support_level)

            return support_levels[:self.max_sr_levels]

        except Exception as e:
            self.logger.error(f"Error in fractal support detection: {e}")
            return []

    async def _detect_fractal_resistance_levels(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect resistance levels using fractal analysis."""
        try:
            # Implement fractal-based resistance level detection
            resistance_levels = []

            # Find local maxima in price data
            high_prices = market_data['high'].rolling(window=5, center=True).max()

            # Identify significant resistance levels
            for i in range(2, len(market_data) - 2):
                if (market_data['high'].iloc[i] == high_prices.iloc[i] and
                    market_data['high'].iloc[i] > market_data['high'].iloc[i-1] and
                    market_data['high'].iloc[i] > market_data['high'].iloc[i+1]):

                    resistance_level = {
                        "price": market_data['high'].iloc[i],
                        "strength": self._calculate_level_strength(market_data, i, "resistance"),
                        "timestamp": market_data.index[i],
                        "method": "fractal",
                        "confidence": 0.7,
                    }
                    resistance_levels.append(resistance_level)

            return resistance_levels[:self.max_sr_levels]

        except Exception as e:
            self.logger.error(f"Error in fractal resistance detection: {e}")
            return []

    async def _detect_volume_support_levels(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect support levels using volume-weighted analysis."""
        try:
            # Implement volume-weighted support level detection
            support_levels = []

            # Calculate volume-weighted average price
            vwap = (market_data['close'] * market_data['volume']).cumsum() / market_data['volume'].cumsum()

            # Find support levels near VWAP
            for i in range(len(market_data)):
                if market_data['low'].iloc[i] <= vwap.iloc[i] * 1.01:  # Within 1% of VWAP
                    support_level = {
                        "price": market_data['low'].iloc[i],
                        "strength": self._calculate_level_strength(market_data, i, "support"),
                        "timestamp": market_data.index[i],
                        "method": "volume",
                        "confidence": 0.6,
                    }
                    support_levels.append(support_level)

            return support_levels[:self.max_sr_levels]

        except Exception as e:
            self.logger.error(f"Error in volume support detection: {e}")
            return []

    async def _detect_volume_resistance_levels(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect resistance levels using volume-weighted analysis."""
        try:
            # Implement volume-weighted resistance level detection
            resistance_levels = []

            # Calculate volume-weighted average price
            vwap = (market_data['close'] * market_data['volume']).cumsum() / market_data['volume'].cumsum()

            # Find resistance levels near VWAP
            for i in range(len(market_data)):
                if market_data['high'].iloc[i] >= vwap.iloc[i] * 0.99:  # Within 1% of VWAP
                    resistance_level = {
                        "price": market_data['high'].iloc[i],
                        "strength": self._calculate_level_strength(market_data, i, "resistance"),
                        "timestamp": market_data.index[i],
                        "method": "volume",
                        "confidence": 0.6,
                    }
                    resistance_levels.append(resistance_level)

            return resistance_levels[:self.max_sr_levels]

        except Exception as e:
            self.logger.error(f"Error in volume resistance detection: {e}")
            return []

    async def _detect_pivot_support_levels(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect support levels using pivot point analysis."""
        try:
            # Implement pivot point support level detection
            support_levels = []

            # Calculate pivot points
            pivot = (market_data['high'] + market_data['low'] + market_data['close']) / 3
            s1 = 2 * pivot - market_data['high']
            pivot - (market_data['high'] - market_data['low'])

            # Find support levels
            for i in range(len(market_data)):
                support_level = {
                    "price": s1.iloc[i],
                    "strength": self._calculate_level_strength(market_data, i, "support"),
                    "timestamp": market_data.index[i],
                    "method": "pivot",
                    "confidence": 0.5,
                }
                support_levels.append(support_level)

            return support_levels[:self.max_sr_levels]

        except Exception as e:
            self.logger.error(f"Error in pivot support detection: {e}")
            return []

    async def _detect_pivot_resistance_levels(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect resistance levels using pivot point analysis."""
        try:
            # Implement pivot point resistance level detection
            resistance_levels = []

            # Calculate pivot points
            pivot = (market_data['high'] + market_data['low'] + market_data['close']) / 3
            r1 = 2 * pivot - market_data['low']
            pivot + (market_data['high'] - market_data['low'])

            # Find resistance levels
            for i in range(len(market_data)):
                resistance_level = {
                    "price": r1.iloc[i],
                    "strength": self._calculate_level_strength(market_data, i, "resistance"),
                    "timestamp": market_data.index[i],
                    "method": "pivot",
                    "confidence": 0.5,
                }
                resistance_levels.append(resistance_level)

            return resistance_levels[:self.max_sr_levels]

        except Exception as e:
            self.logger.error(f"Error in pivot resistance detection: {e}")
            return []

    async def _detect_atr_support_levels(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect support levels using ATR-based analysis."""
        try:
            # Implement ATR-based support level detection
            support_levels = []

            # Calculate ATR
            high_low = market_data['high'] - market_data['low']
            high_close = np.abs(market_data['high'] - market_data['close'].shift())
            low_close = np.abs(market_data['low'] - market_data['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=14).mean()

            # Find support levels
            for i in range(len(market_data)):
                support_level = {
                    "price": market_data['close'].iloc[i] - (atr.iloc[i] * self.atr_multiplier),
                    "strength": self._calculate_level_strength(market_data, i, "support"),
                    "timestamp": market_data.index[i],
                    "method": "atr",
                    "confidence": 0.4,
                }
                support_levels.append(support_level)

            return support_levels[:self.max_sr_levels]

        except Exception as e:
            self.logger.error(f"Error in ATR support detection: {e}")
            return []

    async def _detect_atr_resistance_levels(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect resistance levels using ATR-based analysis."""
        try:
            # Implement ATR-based resistance level detection
            resistance_levels = []

            # Calculate ATR
            high_low = market_data['high'] - market_data['low']
            high_close = np.abs(market_data['high'] - market_data['close'].shift())
            low_close = np.abs(market_data['low'] - market_data['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=14).mean()

            # Find resistance levels
            for i in range(len(market_data)):
                resistance_level = {
                    "price": market_data['close'].iloc[i] + (atr.iloc[i] * self.atr_multiplier),
                    "strength": self._calculate_level_strength(market_data, i, "resistance"),
                    "timestamp": market_data.index[i],
                    "method": "atr",
                    "confidence": 0.4,
                }
                resistance_levels.append(resistance_level)

            return resistance_levels[:self.max_sr_levels]

        except Exception as e:
            self.logger.error(f"Error in ATR resistance detection: {e}")
            return []

    def _calculate_level_strength(self, market_data: pd.DataFrame, index: int, level_type: str) -> float:
        """Calculate the strength of a support/resistance level."""
        try:
            # Base strength calculation
            base_strength = 0.5

            # Volume factor
            volume_factor = min(market_data['volume'].iloc[index] / market_data['volume'].mean(), 2.0)
            base_strength *= (0.5 + 0.5 * volume_factor)

            # Price movement factor
            if level_type == "support":
                price_factor = 1.0 - (market_data['low'].iloc[index] - market_data['close'].iloc[index]) / market_data['close'].iloc[index]
            else:  # resistance
                price_factor = 1.0 - (market_data['close'].iloc[index] - market_data['high'].iloc[index]) / market_data['close'].iloc[index]

            base_strength *= max(0.1, price_factor)

            return min(1.0, max(0.0, base_strength))

        except Exception as e:
            self.logger.error(f"Error calculating level strength: {e}")
            return 0.5

    async def _calculate_breakout_probabilities(
        self,
        support_levels: list[dict[str, Any]],
        resistance_levels: list[dict[str, Any]],
        current_price: float,
    ) -> dict[str, float]:
        """Calculate breakout probabilities for support and resistance levels."""
        try:
            probabilities = {}

            # Calculate support breakout probabilities
            for i, level in enumerate(support_levels):
                distance = (current_price - level["price"]) / current_price
                if distance < 0:  # Price below support
                    prob = min(0.9, abs(distance) / self.sr_proximity_threshold)
                    probabilities[f"support_breakout_{i}"] = prob
                else:
                    probabilities[f"support_breakout_{i}"] = 0.0

            # Calculate resistance breakout probabilities
            for i, level in enumerate(resistance_levels):
                distance = (level["price"] - current_price) / current_price
                if distance < 0:  # Price above resistance
                    prob = min(0.9, abs(distance) / self.sr_proximity_threshold)
                    probabilities[f"resistance_breakout_{i}"] = prob
                else:
                    probabilities[f"resistance_breakout_{i}"] = 0.0

            return probabilities

        except Exception as e:
            self.logger.error(f"Error calculating breakout probabilities: {e}")
            return {}

    async def _calculate_confidence_scores(
        self,
        support_levels: list[dict[str, Any]],
        resistance_levels: list[dict[str, Any]],
        market_data: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate confidence scores for support and resistance levels."""
        try:
            confidence_scores = {}

            # Calculate support confidence scores
            for i, level in enumerate(support_levels):
                confidence = level.get("confidence", 0.5) * level.get("strength", 0.5)
                confidence_scores[f"support_confidence_{i}"] = confidence

            # Calculate resistance confidence scores
            for i, level in enumerate(resistance_levels):
                confidence = level.get("confidence", 0.5) * level.get("strength", 0.5)
                confidence_scores[f"resistance_confidence_{i}"] = confidence

            return confidence_scores

        except Exception as e:
            self.logger.error(f"Error calculating confidence scores: {e}")
            return {}

    async def _generate_sr_features(
        self,
        support_levels: list[dict[str, Any]],
        resistance_levels: list[dict[str, Any]],
        market_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Generate SR-related features for machine learning."""
        try:
            features = {}

            # Calculate proximity to nearest support and resistance
            if support_levels:
                nearest_support = min(support_levels, key=lambda x: abs(x["price"] - market_data['close'].iloc[-1]))
                features["support_proximity"] = abs(nearest_support["price"] - market_data['close'].iloc[-1]) / market_data['close'].iloc[-1]
                features["support_strength"] = nearest_support.get("strength", 0.5)
            else:
                features["support_proximity"] = 1.0
                features["support_strength"] = 0.0

            if resistance_levels:
                nearest_resistance = min(resistance_levels, key=lambda x: abs(x["price"] - market_data['close'].iloc[-1]))
                features["resistance_proximity"] = abs(nearest_resistance["price"] - market_data['close'].iloc[-1]) / market_data['close'].iloc[-1]
                features["resistance_strength"] = nearest_resistance.get("strength", 0.5)
            else:
                features["resistance_proximity"] = 1.0
                features["resistance_strength"] = 0.0

            # Calculate SR zone features
            features["sr_zone_width"] = features["resistance_proximity"] + features["support_proximity"]
            features["sr_zone_center"] = (features["resistance_proximity"] - features["support_proximity"]) / 2

            # Calculate level count features
            features["support_level_count"] = len(support_levels)
            features["resistance_level_count"] = len(resistance_levels)
            features["total_sr_levels"] = len(support_levels) + len(resistance_levels)

            return features

        except Exception as e:
            self.logger.error(f"Error generating SR features: {e}")
            return {}

    def _update_performance_metrics(self, predictions: dict[str, Any]) -> None:
        """Update performance metrics for SR breakout predictions."""
        try:
            # Store prediction in history
            self.prediction_history.append(predictions)

            # Keep only recent predictions
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]

            # Calculate basic metrics
            self.performance_metrics["total_predictions"] = len(self.prediction_history)
            self.performance_metrics["last_prediction_time"] = predictions.get("timestamp", pd.Timestamp.now())

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    async def calculate_sr_features(
        self,
        market_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Calculate SR-related features.

        Args:
            market_data: Market data DataFrame

        Returns:
            dict[str, Any]: SR features
        """
        if not self.is_initialized:
            self.logger.error("SR breakout predictor not initialized")
            return {}

        try:
            # Get current price
            current_price = market_data['close'].iloc[-1]

            # Generate predictions to get features
            predictions = await self.predict_sr_breakouts(market_data, current_price)

            # Extract features
            features = predictions.get("sr_features", {})

            self.logger.info("✅ SR features calculated")
            return features

        except Exception as e:
            self.logger.error(f"Error calculating SR features: {e}")
            return {}

    async def stop(self) -> None:
        """Stop the SR breakout predictor."""
        try:
            self.logger.info("Stopping SR breakout predictor...")
            self.is_initialized = False
            self.logger.info("✅ SR breakout predictor stopped successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to stop SR breakout predictor: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="SR breakout predictor cleanup",
    )
    async def cleanup(self) -> None:
        """Cleanup SR breakout predictor resources."""
        try:
            self.logger.info("Cleaning up SR breakout predictor...")
            await self.stop()
            self.sr_predictions.clear()
            self.prediction_history.clear()
            self.performance_metrics.clear()
            self.logger.info("✅ SR breakout predictor cleanup completed")
        except Exception as e:
            self.logger.error(f"Error cleaning up SR breakout predictor: {e}")


async def setup_sr_breakout_predictor(
    config: dict[str, Any] | None = None,
) -> SRBreakoutPredictor | None:
    """
    Setup and return a configured SRBreakoutPredictor instance.

    Args:
        config: Configuration dictionary

    Returns:
        SRBreakoutPredictor: Configured SR breakout predictor instance
    """
    try:
        predictor = SRBreakoutPredictor(config or {})
        if await predictor.initialize():
            return predictor
        return None
    except Exception as e:
        system_logger.exception(f"Failed to setup SR Breakout Predictor: {e}")
        return None
