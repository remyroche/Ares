# src/tactician/sr_breakout_predictor.py

# from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier  # Temporarily commented due to syntax errors
from src.utils.logger import system_logger
from typing import Any
import numpy as np
import pandas as pd
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.centralized_decorators import validate_data_quality

# Enhanced S/R analysis capabilities integrated directly
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Enhanced S/R types and data structures
class SRType(Enum):
    """Support/Resistance level types."""
    PIVOT = "pivot"
    VOLUME = "volume"
    FIBONACCI = "fibonacci"
    PSYCHOLOGICAL = "psychological"
    FRACTAL = "fractal"
    ATR = "atr"
    COMPOSITE = "composite"

@dataclass
class SRLevel:
    """Support/Resistance level data structure."""
    price: float
    level_type: SRType
    strength: float
    confidence: float
    touches: int
    volume: float
    age: int
    timestamp: pd.Timestamp
    method: str
    proximity: float = 0.0
    breakout_probability: float = 0.0
    last_touch: Optional[pd.Timestamp] = None
    volume_profile: float = 0.0
    psychological_weight: float = 0.0
    fractal_quality: float = 0.0
    composite_score: float = 0.0

@dataclass
class SRBreakoutEvent:
    """S/R breakout event data structure."""
    level: SRLevel
    breakout_type: str  # "support_break" or "resistance_break"
    confidence: float
    volume_confirmation: float
    price_momentum: float
    timestamp: pd.Timestamp
    trigger_features: Dict[str, float]

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
        Initialize SR breakout predictor with enhanced centralized S/R logic.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("SRBreakoutPredictor")

        # SR predictor state
        self.is_initialized: bool = False
        self.sr_predictions: dict[str, Any] = {}
        self.sr_levels: List[SRLevel] = []
        self.breakout_events: List[SRBreakoutEvent] = []
        self.sr_quality_metrics: Dict[str, float] = {}

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
            "composite",  # Enhanced to use composite method
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
        
        # Enhanced S/R configuration
        self.enable_composite_sr: bool = self.sr_config.get("enable_composite_sr", True)
        self.enable_volume_profile: bool = self.sr_config.get("enable_volume_profile", True)
        self.enable_psychological_levels: bool = self.sr_config.get("enable_psychological_levels", True)
        self.enable_fractal_analysis: bool = self.sr_config.get("enable_fractal_analysis", True)
        self.enable_breakout_prediction: bool = self.sr_config.get("enable_breakout_prediction", True)
        
        # S/R analysis parameters
        self.volume_threshold: float = self.sr_config.get("volume_threshold", 1.5)
        self.psychological_levels: List[float] = self.sr_config.get("psychological_levels", [])
        self.fractal_window: int = self.sr_config.get("fractal_window", 20)
        self.strength_decay_factor: float = self.sr_config.get("strength_decay_factor", 0.95)
        
        # Breakout prediction model
        self.breakout_model: Optional[RandomForestClassifier] = None
        self.breakout_model_trained: bool = False
        
        # Centralized S/R state
        self.sr_analysis_state = {
            "last_sr_analysis": None,
            "sr_detection_count": 0,
            "sr_quality_scores": {},
            "sr_redundancy_metrics": {},
            "sr_integration_status": {},
            "breakout_prediction_accuracy": {}
        }
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
        
        # Enhanced S/R analysis capabilities
        self.sr_levels_cache: dict[str, List[SRLevel]] = {}
        self.sr_analysis_history: list[dict[str, Any]] = []
        self.sr_quality_metrics: dict[str, float] = {}
        
        # Centralized S/R analysis state
        self.sr_analysis_state = {
            "last_analysis_time": None,
            "analysis_count": 0,
            "quality_scores": {},
            "redundancy_metrics": {},
            "feature_integration_status": {}
        }

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
        """Detect support levels using enhanced multi-method analysis."""
        try:
            self.logger.info("🔍 Detecting support levels using enhanced analysis...")
            
            # Use enhanced S/R analysis
            sr_levels = await self._analyze_enhanced_sr_levels(market_data, "support")
            
            # Convert SRLevel objects to dict format for compatibility
            support_levels = []
            for level in sr_levels:
                support_levels.append({
                    "price": level.price,
                    "strength": level.strength,
                    "timestamp": level.timestamp,
                    "method": level.method,
                    "confidence": level.confidence,
                    "touches": level.touches,
                    "volume": level.volume,
                    "age": level.age,
                    "proximity": level.proximity,
                    "breakout_probability": level.breakout_probability
                })
            
            self.logger.info(f"✅ Detected {len(support_levels)} support levels")
            return support_levels

        except Exception as e:
            self.logger.error(f"Error detecting support levels: {e}")
            return []

    async def _detect_resistance_levels(self, market_data: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect resistance levels using enhanced multi-method analysis."""
        try:
            self.logger.info("🔍 Detecting resistance levels using enhanced analysis...")
            
            # Use enhanced S/R analysis
            sr_levels = await self._analyze_enhanced_sr_levels(market_data, "resistance")
            
            # Convert SRLevel objects to dict format for compatibility
            resistance_levels = []
            for level in sr_levels:
                resistance_levels.append({
                    "price": level.price,
                    "strength": level.strength,
                    "timestamp": level.timestamp,
                    "method": level.method,
                    "confidence": level.confidence,
                    "touches": level.touches,
                    "volume": level.volume,
                    "age": level.age,
                    "proximity": level.proximity,
                    "breakout_probability": level.breakout_probability
                })
            
            self.logger.info(f"✅ Detected {len(resistance_levels)} resistance levels")
            return resistance_levels

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

    # === ENHANCED S/R ANALYSIS METHODS ===
    
    async def _analyze_enhanced_sr_levels(self, market_data: pd.DataFrame, level_type: str) -> List[SRLevel]:
        """Analyze S/R levels using multiple methods and combine results."""
        try:
            self.logger.info(f"🔍 Analyzing {level_type} levels using enhanced multi-method approach...")
            
            # Collect levels from different methods
            all_levels = []
            
            # 1. Pivot-based levels
            pivot_levels = await self._detect_pivot_levels(market_data, level_type)
            all_levels.extend(pivot_levels)
            
            # 2. Volume-weighted levels
            volume_levels = await self._detect_volume_levels(market_data, level_type)
            all_levels.extend(volume_levels)
            
            # 3. Fractal levels
            fractal_levels = await self._detect_fractal_levels(market_data, level_type)
            all_levels.extend(fractal_levels)
            
            # 4. Fibonacci levels
            fibonacci_levels = await self._detect_fibonacci_levels(market_data, level_type)
            all_levels.extend(fibonacci_levels)
            
            # 5. Psychological levels
            psychological_levels = await self._detect_psychological_levels(market_data, level_type)
            all_levels.extend(psychological_levels)
            
            # 6. ATR-based levels
            atr_levels = await self._detect_atr_levels(market_data, level_type)
            all_levels.extend(atr_levels)
            
            # Cluster and merge similar levels
            merged_levels = self._cluster_and_merge_levels(all_levels, market_data)
            
            # Calculate proximity and breakout probabilities
            current_price = market_data['close'].iloc[-1]
            for level in merged_levels:
                level.proximity = abs(level.price - current_price) / current_price
                level.breakout_probability = self._calculate_breakout_probability(level, current_price)
            
            # Sort by strength and confidence
            merged_levels.sort(key=lambda x: (x.strength * x.confidence), reverse=True)
            
            # Limit to max levels
            final_levels = merged_levels[:self.max_sr_levels]
            
            self.logger.info(f"✅ Enhanced {level_type} analysis completed: {len(final_levels)} levels")
            return final_levels
            
        except Exception as e:
            self.logger.error(f"Error in enhanced S/R analysis: {e}")
            return []

    async def _detect_pivot_levels(self, market_data: pd.DataFrame, level_type: str) -> List[SRLevel]:
        """Detect pivot-based S/R levels."""
        try:
            levels = []
            
            # Calculate pivot points
            pivot = (market_data['high'] + market_data['low'] + market_data['close']) / 3
            
            if level_type == "support":
                s1 = 2 * pivot - market_data['high']
                s2 = pivot - (market_data['high'] - market_data['low'])
                s3 = market_data['low'] - 2 * (market_data['high'] - pivot)
                
                for i, price in enumerate([s1, s2, s3]):
                    if not pd.isna(price):
                        level = SRLevel(
                            price=float(price.iloc[-1]),
                            level_type=SRType.PIVOT,
                            strength=self._calculate_pivot_strength(market_data, price, "support"),
                            confidence=0.6,
                            touches=self._count_touches(market_data, price, "support"),
                            volume=float(market_data['volume'].iloc[-1]),
                            age=len(market_data),
                            timestamp=market_data.index[-1],
                            method=f"pivot_s{i+1}"
                        )
                        levels.append(level)
            else:  # resistance
                r1 = 2 * pivot - market_data['low']
                r2 = pivot + (market_data['high'] - market_data['low'])
                r3 = market_data['high'] + 2 * (pivot - market_data['low'])
                
                for i, price in enumerate([r1, r2, r3]):
                    if not pd.isna(price):
                        level = SRLevel(
                            price=float(price.iloc[-1]),
                            level_type=SRType.PIVOT,
                            strength=self._calculate_pivot_strength(market_data, price, "resistance"),
                            confidence=0.6,
                            touches=self._count_touches(market_data, price, "resistance"),
                            volume=float(market_data['volume'].iloc[-1]),
                            age=len(market_data),
                            timestamp=market_data.index[-1],
                            method=f"pivot_r{i+1}"
                        )
                        levels.append(level)
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error detecting pivot levels: {e}")
            return []

    async def _detect_volume_levels(self, market_data: pd.DataFrame, level_type: str) -> List[SRLevel]:
        """Detect volume-weighted S/R levels."""
        try:
            levels = []
            
            # Calculate VWAP
            vwap = (market_data['close'] * market_data['volume']).cumsum() / market_data['volume'].cumsum()
            
            # Find high-volume price levels
            volume_threshold = market_data['volume'].quantile(0.8)
            high_volume_mask = market_data['volume'] > volume_threshold
            
            if level_type == "support":
                # Find support levels near high volume areas
                for i in range(len(market_data)):
                    if high_volume_mask.iloc[i]:
                        price = market_data['low'].iloc[i]
                        level = SRLevel(
                            price=float(price),
                            level_type=SRType.VOLUME,
                            strength=self._calculate_volume_strength(market_data, price, "support"),
                            confidence=0.7,
                            touches=self._count_touches(market_data, price, "support"),
                            volume=float(market_data['volume'].iloc[i]),
                            age=len(market_data) - i,
                            timestamp=market_data.index[i],
                            method="volume_support"
                        )
                        levels.append(level)
            else:  # resistance
                # Find resistance levels near high volume areas
                for i in range(len(market_data)):
                    if high_volume_mask.iloc[i]:
                        price = market_data['high'].iloc[i]
                        level = SRLevel(
                            price=float(price),
                            level_type=SRType.VOLUME,
                            strength=self._calculate_volume_strength(market_data, price, "resistance"),
                            confidence=0.7,
                            touches=self._count_touches(market_data, price, "resistance"),
                            volume=float(market_data['volume'].iloc[i]),
                            age=len(market_data) - i,
                            timestamp=market_data.index[i],
                            method="volume_resistance"
                        )
                        levels.append(level)
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error detecting volume levels: {e}")
            return []

    async def _detect_fractal_levels(self, market_data: pd.DataFrame, level_type: str) -> List[SRLevel]:
        """Detect fractal-based S/R levels."""
        try:
            levels = []
            window = 5
            
            if level_type == "support":
                # Find local minima
                for i in range(window, len(market_data) - window):
                    if all(market_data['low'].iloc[i] <= market_data['low'].iloc[i-window:i]) and \
                       all(market_data['low'].iloc[i] <= market_data['low'].iloc[i+1:i+window+1]):
                        
                        price = market_data['low'].iloc[i]
                        level = SRLevel(
                            price=float(price),
                            level_type=SRType.FRACTAL,
                            strength=self._calculate_fractal_strength(market_data, i, "support"),
                            confidence=0.8,
                            touches=self._count_touches(market_data, price, "support"),
                            volume=float(market_data['volume'].iloc[i]),
                            age=len(market_data) - i,
                            timestamp=market_data.index[i],
                            method="fractal_support"
                        )
                        levels.append(level)
            else:  # resistance
                # Find local maxima
                for i in range(window, len(market_data) - window):
                    if all(market_data['high'].iloc[i] >= market_data['high'].iloc[i-window:i]) and \
                       all(market_data['high'].iloc[i] >= market_data['high'].iloc[i+1:i+window+1]):
                        
                        price = market_data['high'].iloc[i]
                        level = SRLevel(
                            price=float(price),
                            level_type=SRType.FRACTAL,
                            strength=self._calculate_fractal_strength(market_data, i, "resistance"),
                            confidence=0.8,
                            touches=self._count_touches(market_data, price, "resistance"),
                            volume=float(market_data['volume'].iloc[i]),
                            age=len(market_data) - i,
                            timestamp=market_data.index[i],
                            method="fractal_resistance"
                        )
                        levels.append(level)
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error detecting fractal levels: {e}")
            return []

    async def _detect_fibonacci_levels(self, market_data: pd.DataFrame, level_type: str) -> List[SRLevel]:
        """Detect Fibonacci retracement levels."""
        try:
            levels = []
            
            # Find swing high and low
            swing_high = market_data['high'].max()
            swing_low = market_data['low'].min()
            price_range = swing_high - swing_low
            
            # Fibonacci ratios
            fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
            
            if level_type == "support":
                for ratio in fib_ratios:
                    price = swing_high - (price_range * ratio)
                    level = SRLevel(
                        price=float(price),
                        level_type=SRType.FIBONACCI,
                        strength=0.6,
                        confidence=0.5,
                        touches=self._count_touches(market_data, price, "support"),
                        volume=float(market_data['volume'].iloc[-1]),
                        age=len(market_data),
                        timestamp=market_data.index[-1],
                        method=f"fibonacci_{ratio}"
                    )
                    levels.append(level)
            else:  # resistance
                for ratio in fib_ratios:
                    price = swing_low + (price_range * ratio)
                    level = SRLevel(
                        price=float(price),
                        level_type=SRType.FIBONACCI,
                        strength=0.6,
                        confidence=0.5,
                        touches=self._count_touches(market_data, price, "resistance"),
                        volume=float(market_data['volume'].iloc[-1]),
                        age=len(market_data),
                        timestamp=market_data.index[-1],
                        method=f"fibonacci_{ratio}"
                    )
                    levels.append(level)
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error detecting Fibonacci levels: {e}")
            return []

    async def _detect_psychological_levels(self, market_data: pd.DataFrame, level_type: str) -> List[SRLevel]:
        """Detect psychological S/R levels (round numbers)."""
        try:
            levels = []
            current_price = market_data['close'].iloc[-1]
            
            # Find nearby round numbers
            if level_type == "support":
                # Round down to nearest psychological level
                base_price = int(current_price / 100) * 100
                psychological_levels = [base_price, base_price - 100, base_price - 200]
            else:  # resistance
                # Round up to nearest psychological level
                base_price = int(current_price / 100) * 100 + 100
                psychological_levels = [base_price, base_price + 100, base_price + 200]
            
            for price in psychological_levels:
                if price > 0:
                    level = SRLevel(
                        price=float(price),
                        level_type=SRType.PSYCHOLOGICAL,
                        strength=0.5,
                        confidence=0.4,
                        touches=self._count_touches(market_data, price, level_type),
                        volume=float(market_data['volume'].iloc[-1]),
                        age=len(market_data),
                        timestamp=market_data.index[-1],
                        method="psychological"
                    )
                    levels.append(level)
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error detecting psychological levels: {e}")
            return []

    async def _detect_atr_levels(self, market_data: pd.DataFrame, level_type: str) -> List[SRLevel]:
        """Detect ATR-based S/R levels."""
        try:
            levels = []
            current_price = market_data['close'].iloc[-1]
            
            # Calculate ATR
            atr = self._calculate_atr(market_data)
            current_atr = atr.iloc[-1]
            
            if level_type == "support":
                # Support levels below current price
                for multiplier in [1, 2, 3]:
                    price = current_price - (current_atr * multiplier)
                    level = SRLevel(
                        price=float(price),
                        level_type=SRType.ATR,
                        strength=0.4,
                        confidence=0.3,
                        touches=self._count_touches(market_data, price, "support"),
                        volume=float(market_data['volume'].iloc[-1]),
                        age=len(market_data),
                        timestamp=market_data.index[-1],
                        method=f"atr_support_{multiplier}x"
                    )
                    levels.append(level)
            else:  # resistance
                # Resistance levels above current price
                for multiplier in [1, 2, 3]:
                    price = current_price + (current_atr * multiplier)
                    level = SRLevel(
                        price=float(price),
                        level_type=SRType.ATR,
                        strength=0.4,
                        confidence=0.3,
                        touches=self._count_touches(market_data, price, "resistance"),
                        volume=float(market_data['volume'].iloc[-1]),
                        age=len(market_data),
                        timestamp=market_data.index[-1],
                        method=f"atr_resistance_{multiplier}x"
                    )
                    levels.append(level)
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error detecting ATR levels: {e}")
            return []

    def _cluster_and_merge_levels(self, levels: List[SRLevel], market_data: pd.DataFrame) -> List[SRLevel]:
        """Cluster similar S/R levels and merge them."""
        try:
            if not levels:
                return []
            
            # Prepare data for clustering
            prices = np.array([[level.price] for level in levels])
            
            # Use DBSCAN to cluster similar levels
            scaler = StandardScaler()
            prices_scaled = scaler.fit_transform(prices)
            
            # Cluster with DBSCAN
            clustering = DBSCAN(eps=0.1, min_samples=1).fit(prices_scaled)
            labels = clustering.labels_
            
            # Merge levels in the same cluster
            merged_levels = []
            unique_labels = set(labels)
            
            for label in unique_labels:
                cluster_levels = [levels[i] for i in range(len(levels)) if labels[i] == label]
                
                if len(cluster_levels) == 1:
                    merged_levels.append(cluster_levels[0])
                else:
                    # Merge multiple levels in the same cluster
                    merged_level = self._merge_cluster_levels(cluster_levels)
                    merged_levels.append(merged_level)
            
            return merged_levels
            
        except Exception as e:
            self.logger.error(f"Error clustering and merging levels: {e}")
            return levels

    def _merge_cluster_levels(self, cluster_levels: List[SRLevel]) -> SRLevel:
        """Merge multiple S/R levels in the same cluster."""
        try:
            # Weighted average price based on strength and confidence
            total_weight = sum(level.strength * level.confidence for level in cluster_levels)
            
            if total_weight == 0:
                # Fallback to simple average
                avg_price = sum(level.price for level in cluster_levels) / len(cluster_levels)
                avg_strength = sum(level.strength for level in cluster_levels) / len(cluster_levels)
                avg_confidence = sum(level.confidence for level in cluster_levels) / len(cluster_levels)
            else:
                # Weighted average
                avg_price = sum(level.price * level.strength * level.confidence for level in cluster_levels) / total_weight
                avg_strength = sum(level.strength * level.strength * level.confidence for level in cluster_levels) / total_weight
                avg_confidence = sum(level.confidence * level.strength * level.confidence for level in cluster_levels) / total_weight
            
            # Use the most recent timestamp and highest volume
            latest_timestamp = max(level.timestamp for level in cluster_levels)
            max_volume = max(level.volume for level in cluster_levels)
            total_touches = sum(level.touches for level in cluster_levels)
            
            # Determine the most common level type
            level_types = [level.level_type for level in cluster_levels]
            most_common_type = max(set(level_types), key=level_types.count)
            
            return SRLevel(
                price=avg_price,
                level_type=most_common_type,
                strength=min(1.0, avg_strength * 1.2),  # Boost strength for merged levels
                confidence=min(1.0, avg_confidence * 1.1),  # Slight boost to confidence
                touches=total_touches,
                volume=max_volume,
                age=min(level.age for level in cluster_levels),
                timestamp=latest_timestamp,
                method="merged"
            )
            
        except Exception as e:
            self.logger.error(f"Error merging cluster levels: {e}")
            return cluster_levels[0] if cluster_levels else None

    def _calculate_breakout_probability(self, level: SRLevel, current_price: float) -> float:
        """Calculate breakout probability for an S/R level."""
        try:
            # Base probability on proximity and strength
            proximity_factor = 1.0 - min(level.proximity / 0.1, 1.0)  # Higher proximity = higher probability
            strength_factor = level.strength
            confidence_factor = level.confidence
            
            # Combine factors
            probability = (proximity_factor * 0.4 + strength_factor * 0.3 + confidence_factor * 0.3)
            
            return min(1.0, max(0.0, probability))
            
        except Exception as e:
            self.logger.error(f"Error calculating breakout probability: {e}")
            return 0.5

    # === HELPER METHODS FOR ENHANCED S/R ANALYSIS ===
    
    def _calculate_pivot_strength(self, market_data: pd.DataFrame, pivot_price: pd.Series, level_type: str) -> float:
        """Calculate strength of pivot-based S/R level."""
        try:
            # Count touches and bounces
            touches = self._count_touches(market_data, pivot_price, level_type)
            bounces = self._count_bounces(market_data, pivot_price, level_type)
            
            # Calculate strength based on touches and bounces
            strength = min(1.0, (touches * 0.3 + bounces * 0.7) / 10)
            return strength
            
        except Exception as e:
            self.logger.error(f"Error calculating pivot strength: {e}")
            return 0.5

    def _calculate_volume_strength(self, market_data: pd.DataFrame, price: float, level_type: str) -> float:
        """Calculate strength of volume-based S/R level."""
        try:
            # Find volume at this price level
            price_tolerance = price * 0.001  # 0.1% tolerance
            
            if level_type == "support":
                volume_at_level = market_data[
                    (market_data['low'] >= price - price_tolerance) & 
                    (market_data['low'] <= price + price_tolerance)
                ]['volume'].sum()
            else:  # resistance
                volume_at_level = market_data[
                    (market_data['high'] >= price - price_tolerance) & 
                    (market_data['high'] <= price + price_tolerance)
                ]['volume'].sum()
            
            # Normalize by average volume
            avg_volume = market_data['volume'].mean()
            strength = min(1.0, volume_at_level / (avg_volume * 10))
            
            return strength
            
        except Exception as e:
            self.logger.error(f"Error calculating volume strength: {e}")
            return 0.5

    def _calculate_fractal_strength(self, market_data: pd.DataFrame, index: int, level_type: str) -> float:
        """Calculate strength of fractal-based S/R level."""
        try:
            # Calculate strength based on the sharpness of the fractal
            window = 5
            
            if level_type == "support":
                # Calculate how much the low stands out
                current_low = market_data['low'].iloc[index]
                surrounding_lows = market_data['low'].iloc[max(0, index-window):index].tolist() + \
                                  market_data['low'].iloc[index+1:min(len(market_data), index+window+1)].tolist()
                
                if surrounding_lows:
                    avg_surrounding = sum(surrounding_lows) / len(surrounding_lows)
                    strength = min(1.0, (avg_surrounding - current_low) / current_low * 10)
                else:
                    strength = 0.5
            else:  # resistance
                # Calculate how much the high stands out
                current_high = market_data['high'].iloc[index]
                surrounding_highs = market_data['high'].iloc[max(0, index-window):index].tolist() + \
                                   market_data['high'].iloc[index+1:min(len(market_data), index+window+1)].tolist()
                
                if surrounding_highs:
                    avg_surrounding = sum(surrounding_highs) / len(surrounding_highs)
                    strength = min(1.0, (current_high - avg_surrounding) / current_high * 10)
                else:
                    strength = 0.5
            
            return max(0.1, strength)
            
        except Exception as e:
            self.logger.error(f"Error calculating fractal strength: {e}")
            return 0.5

    def _count_touches(self, market_data: pd.DataFrame, price: float, level_type: str) -> int:
        """Count how many times price touched the S/R level."""
        try:
            tolerance = price * 0.002  # 0.2% tolerance
            
            if level_type == "support":
                touches = ((market_data['low'] >= price - tolerance) & 
                          (market_data['low'] <= price + tolerance)).sum()
            else:  # resistance
                touches = ((market_data['high'] >= price - tolerance) & 
                          (market_data['high'] <= price + tolerance)).sum()
            
            return int(touches)
            
        except Exception as e:
            self.logger.error(f"Error counting touches: {e}")
            return 0

    def _count_bounces(self, market_data: pd.DataFrame, price: float, level_type: str) -> int:
        """Count how many times price bounced off the S/R level."""
        try:
            tolerance = price * 0.002  # 0.2% tolerance
            bounces = 0
            
            for i in range(1, len(market_data)):
                if level_type == "support":
                    # Check if price touched support and then moved up
                    if (market_data['low'].iloc[i] <= price + tolerance and 
                        market_data['close'].iloc[i] > market_data['close'].iloc[i-1]):
                        bounces += 1
                else:  # resistance
                    # Check if price touched resistance and then moved down
                    if (market_data['high'].iloc[i] >= price - tolerance and 
                        market_data['close'].iloc[i] < market_data['close'].iloc[i-1]):
                        bounces += 1
            
            return bounces
            
        except Exception as e:
            self.logger.error(f"Error counting bounces: {e}")
            return 0

    def _calculate_atr(self, market_data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        try:
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=window).mean()
            
            return atr
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return pd.Series([0] * len(market_data))

    # === CENTRALIZED S/R ANALYSIS METHODS ===
    
    async def analyze_centralized_sr_levels(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """
        Perform comprehensive centralized S/R analysis using multiple methods.
        Enhanced with advanced S/R detection, quality control, and redundancy elimination.
        
        Args:
            market_data: Market data DataFrame
            
        Returns:
            dict[str, Any]: Comprehensive S/R analysis results
        """
        try:
            self.logger.info("🔍 Performing comprehensive centralized S/R analysis...")
            
            # Update analysis state
            self.sr_analysis_state["last_analysis_time"] = pd.Timestamp.now()
            self.sr_analysis_state["analysis_count"] += 1
            
            # Enhanced multi-method S/R analysis with quality control
            support_levels = await self._analyze_enhanced_sr_levels(market_data, "support")
            resistance_levels = await self._analyze_enhanced_sr_levels(market_data, "resistance")
            
            # Apply advanced S/R detection methods
            if self.enable_composite_sr:
                support_levels = await self._create_composite_sr_levels(support_levels)
                resistance_levels = await self._create_composite_sr_levels(resistance_levels)
            
            # Enhanced quality metrics calculation
            quality_metrics = await self._calculate_sr_quality_metrics(support_levels, resistance_levels, market_data)
            
            # Advanced redundancy elimination
            redundancy_metrics = await self._calculate_sr_redundancy_metrics(support_levels + resistance_levels)
            
            # Generate comprehensive features with integration status
            sr_features = await self._generate_sr_features(support_levels + resistance_levels, market_data)
            
            # Detect breakout events
            breakout_events = await self._detect_breakout_events(support_levels + resistance_levels, market_data)
            
            # Create comprehensive analysis results
            analysis_results = {
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "quality_metrics": quality_metrics,
                "redundancy_metrics": redundancy_metrics,
                "sr_features": sr_features,
                "breakout_events": breakout_events,
                "analysis_timestamp": pd.Timestamp.now(),
                "analysis_id": self.sr_analysis_state["analysis_count"],
                "integration_status": {
                    "feature_engineering_ready": True,
                    "analyst_component_ready": True,
                    "quality_control_passed": quality_metrics.get("overall_quality", 0) > 0.6,
                    "redundancy_eliminated": redundancy_metrics.get("redundancy_score", 0) < 0.3
                }
            }
            
            # Enhanced caching with quality control
            cache_key = f"sr_analysis_{market_data.index[-1]}"
            if quality_metrics.get("overall_quality", 0) > 0.5:
                self.sr_levels_cache[cache_key] = support_levels + resistance_levels
                self.sr_analysis_history.append(analysis_results)
            
            # Update state with enhanced metrics
            self.sr_quality_metrics.update(quality_metrics)
            self.sr_analysis_state["quality_scores"] = quality_metrics
            self.sr_analysis_state["redundancy_metrics"] = redundancy_metrics
            self.sr_analysis_state["feature_integration_status"] = analysis_results["integration_status"]
            
            self.logger.info("✅ Enhanced comprehensive centralized S/R analysis completed")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in centralized S/R analysis: {e}")
            return {}

    # === ENHANCED CENTRALIZED S/R ANALYSIS METHODS ===
    
    async def get_centralized_sr_features(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """
        Get centralized S/R features for feature engineering integration.
        
        Args:
            market_data: Market data DataFrame
            
        Returns:
            dict[str, Any]: Centralized S/R features
        """
        try:
            # Perform centralized S/R analysis
            analysis_results = await self.analyze_centralized_sr_levels(market_data)
            
            # Extract features
            sr_features = analysis_results.get("sr_features", {})
            
            # Add quality metrics
            quality_metrics = analysis_results.get("quality_metrics", {})
            sr_features.update({
                f"sr_quality_{key}": value for key, value in quality_metrics.items()
            })
            
            # Add redundancy metrics
            redundancy_metrics = analysis_results.get("redundancy_metrics", {})
            sr_features.update({
                f"sr_redundancy_{key}": value for key, value in redundancy_metrics.items()
            })
            
            self.logger.info("✅ Centralized S/R features prepared for feature engineering")
            return sr_features
            
        except Exception as e:
            self.logger.error(f"Error getting centralized S/R features: {e}")
            return {}

    async def get_sr_breakout_predictions(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """
        Get S/R breakout predictions with enhanced analysis.
        
        Args:
            market_data: Market data DataFrame
            
        Returns:
            dict[str, Any]: S/R breakout predictions
        """
        try:
            # Perform centralized S/R analysis
            analysis_results = await self.analyze_centralized_sr_levels(market_data)
            
            # Get current price
            current_price = market_data['close'].iloc[-1]
            
            # Calculate breakout probabilities
            support_levels = analysis_results.get("support_levels", [])
            resistance_levels = analysis_results.get("resistance_levels", [])
            
            breakout_predictions = {
                "support_breakouts": [],
                "resistance_breakouts": [],
                "current_price": current_price,
                "analysis_timestamp": pd.Timestamp.now()
            }
            
            # Analyze support breakouts
            for level in support_levels:
                if current_price < level.price:
                    breakout_prob = self._calculate_breakout_probability(level, current_price)
                    breakout_predictions["support_breakouts"].append({
                        "level": level.price,
                        "probability": breakout_prob,
                        "strength": level.strength,
                        "confidence": level.confidence,
                        "method": level.method
                    })
            
            # Analyze resistance breakouts
            for level in resistance_levels:
                if current_price > level.price:
                    breakout_prob = self._calculate_breakout_probability(level, current_price)
                    breakout_predictions["resistance_breakouts"].append({
                        "level": level.price,
                        "probability": breakout_prob,
                        "strength": level.strength,
                        "confidence": level.confidence,
                        "method": level.method
                    })
            
            self.logger.info("✅ S/R breakout predictions generated")
            return breakout_predictions
            
        except Exception as e:
            self.logger.error(f"Error getting S/R breakout predictions: {e}")
            return {}

    def _calculate_sr_quality_metrics(self, support_levels: List[SRLevel], resistance_levels: List[SRLevel], market_data: pd.DataFrame) -> dict[str, float]:
        """Calculate quality metrics for S/R levels."""
        try:
            metrics = {}
            
            # Coverage metrics
            total_levels = len(support_levels) + len(resistance_levels)
            metrics["total_levels"] = total_levels
            metrics["support_coverage"] = len(support_levels) / max(total_levels, 1)
            metrics["resistance_coverage"] = len(resistance_levels) / max(total_levels, 1)
            
            # Strength metrics
            if support_levels:
                metrics["avg_support_strength"] = sum(level.strength for level in support_levels) / len(support_levels)
                metrics["max_support_strength"] = max(level.strength for level in support_levels)
            else:
                metrics["avg_support_strength"] = 0.0
                metrics["max_support_strength"] = 0.0
                
            if resistance_levels:
                metrics["avg_resistance_strength"] = sum(level.strength for level in resistance_levels) / len(resistance_levels)
                metrics["max_resistance_strength"] = max(level.strength for level in resistance_levels)
            else:
                metrics["avg_resistance_strength"] = 0.0
                metrics["max_resistance_strength"] = 0.0
            
            # Confidence metrics
            if support_levels:
                metrics["avg_support_confidence"] = sum(level.confidence for level in support_levels) / len(support_levels)
            else:
                metrics["avg_support_confidence"] = 0.0
                
            if resistance_levels:
                metrics["avg_resistance_confidence"] = sum(level.confidence for level in resistance_levels) / len(resistance_levels)
            else:
                metrics["avg_resistance_confidence"] = 0.0
            
            # Distribution metrics
            current_price = market_data['close'].iloc[-1]
            price_range = market_data['high'].max() - market_data['low'].min()
            
            if support_levels:
                support_distances = [abs(level.price - current_price) / price_range for level in support_levels]
                metrics["support_distance_variance"] = np.var(support_distances) if len(support_distances) > 1 else 0.0
            else:
                metrics["support_distance_variance"] = 0.0
                
            if resistance_levels:
                resistance_distances = [abs(level.price - current_price) / price_range for level in resistance_levels]
                metrics["resistance_distance_variance"] = np.var(resistance_distances) if len(resistance_distances) > 1 else 0.0
            else:
                metrics["resistance_distance_variance"] = 0.0
            
            # Overall quality score
            quality_factors = [
                metrics["avg_support_strength"],
                metrics["avg_resistance_strength"],
                metrics["avg_support_confidence"],
                metrics["avg_resistance_confidence"],
                1.0 - metrics["support_distance_variance"],
                1.0 - metrics["resistance_distance_variance"]
            ]
            metrics["overall_quality_score"] = sum(quality_factors) / len(quality_factors)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating S/R quality metrics: {e}")
            return {}

    def _eliminate_sr_redundancy(self, support_levels: List[SRLevel], resistance_levels: List[SRLevel]) -> dict[str, Any]:
        """Eliminate redundant S/R levels and calculate redundancy metrics."""
        try:
            metrics = {}
            
            # Combine all levels for redundancy analysis
            all_levels = support_levels + resistance_levels
            
            # Find redundant levels (similar prices)
            redundant_groups = []
            processed_indices = set()
            
            for i, level1 in enumerate(all_levels):
                if i in processed_indices:
                    continue
                    
                redundant_group = [level1]
                processed_indices.add(i)
                
                for j, level2 in enumerate(all_levels[i+1:], i+1):
                    if j in processed_indices:
                        continue
                        
                    # Check if levels are similar (within 0.5% of price)
                    price_diff = abs(level1.price - level2.price) / level1.price
                    if price_diff < 0.005:  # 0.5% threshold
                        redundant_group.append(level2)
                        processed_indices.add(j)
                
                if len(redundant_group) > 1:
                    redundant_groups.append(redundant_group)
            
            # Merge redundant levels
            merged_levels = []
            for group in redundant_groups:
                if len(group) > 1:
                    merged_level = self._merge_redundant_levels(group)
                    merged_levels.append(merged_level)
                else:
                    merged_levels.append(group[0])
            
            # Calculate redundancy metrics
            metrics["original_levels"] = len(all_levels)
            metrics["merged_levels"] = len(merged_levels)
            metrics["redundant_groups"] = len(redundant_groups)
            metrics["redundancy_ratio"] = (len(all_levels) - len(merged_levels)) / max(len(all_levels), 1)
            
            # Update levels with merged results
            support_levels[:] = [level for level in merged_levels if level in support_levels]
            resistance_levels[:] = [level for level in merged_levels if level in resistance_levels]
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error eliminating S/R redundancy: {e}")
            return {}

    def _merge_redundant_levels(self, redundant_group: List[SRLevel]) -> SRLevel:
        """Merge redundant S/R levels into a single level."""
        try:
            # Weighted average based on strength and confidence
            total_weight = sum(level.strength * level.confidence for level in redundant_group)
            
            if total_weight == 0:
                # Fallback to simple average
                avg_price = sum(level.price for level in redundant_group) / len(redundant_group)
                avg_strength = sum(level.strength for level in redundant_group) / len(redundant_group)
                avg_confidence = sum(level.confidence for level in redundant_group) / len(redundant_group)
            else:
                # Weighted average
                avg_price = sum(level.price * level.strength * level.confidence for level in redundant_group) / total_weight
                avg_strength = sum(level.strength * level.strength * level.confidence for level in redundant_group) / total_weight
                avg_confidence = sum(level.confidence * level.strength * level.confidence for level in redundant_group) / total_weight
            
            # Use the most recent timestamp and highest volume
            latest_timestamp = max(level.timestamp for level in redundant_group)
            max_volume = max(level.volume for level in redundant_group)
            total_touches = sum(level.touches for level in redundant_group)
            
            # Determine the most common level type
            level_types = [level.level_type for level in redundant_group]
            most_common_type = max(set(level_types), key=level_types.count)
            
            return SRLevel(
                price=avg_price,
                level_type=most_common_type,
                strength=min(1.0, avg_strength * 1.1),  # Slight boost for merged levels
                confidence=min(1.0, avg_confidence * 1.05),  # Slight boost to confidence
                touches=total_touches,
                volume=max_volume,
                age=min(level.age for level in redundant_group),
                timestamp=latest_timestamp,
                method="merged_redundant"
            )
            
        except Exception as e:
            self.logger.error(f"Error merging redundant levels: {e}")
            return redundant_group[0] if redundant_group else None

    def _generate_comprehensive_sr_features(self, support_levels: List[SRLevel], resistance_levels: List[SRLevel], market_data: pd.DataFrame) -> dict[str, Any]:
        """Generate comprehensive S/R features for feature engineering integration."""
        try:
            features = {}
            current_price = market_data['close'].iloc[-1]
            
            # Basic proximity features
            if support_levels:
                nearest_support = min(support_levels, key=lambda x: abs(x.price - current_price))
                features["nearest_support_price"] = nearest_support.price
                features["nearest_support_distance"] = abs(nearest_support.price - current_price) / current_price
                features["nearest_support_strength"] = nearest_support.strength
                features["nearest_support_confidence"] = nearest_support.confidence
            else:
                features["nearest_support_price"] = current_price * 0.9
                features["nearest_support_distance"] = 0.1
                features["nearest_support_strength"] = 0.0
                features["nearest_support_confidence"] = 0.0
                
            if resistance_levels:
                nearest_resistance = min(resistance_levels, key=lambda x: abs(x.price - current_price))
                features["nearest_resistance_price"] = nearest_resistance.price
                features["nearest_resistance_distance"] = abs(nearest_resistance.price - current_price) / current_price
                features["nearest_resistance_strength"] = nearest_resistance.strength
                features["nearest_resistance_confidence"] = nearest_resistance.confidence
            else:
                features["nearest_resistance_price"] = current_price * 1.1
                features["nearest_resistance_distance"] = 0.1
                features["nearest_resistance_strength"] = 0.0
                features["nearest_resistance_confidence"] = 0.0
            
            # SR zone features
            features["sr_zone_width"] = features["nearest_resistance_distance"] + features["nearest_support_distance"]
            features["sr_zone_center"] = (features["nearest_resistance_distance"] - features["nearest_support_distance"]) / 2
            features["sr_zone_asymmetry"] = abs(features["nearest_resistance_distance"] - features["nearest_support_distance"])
            
            # Level count features
            features["support_level_count"] = len(support_levels)
            features["resistance_level_count"] = len(resistance_levels)
            features["total_sr_levels"] = len(support_levels) + len(resistance_levels)
            
            # Strength aggregation features
            if support_levels:
                features["avg_support_strength"] = sum(level.strength for level in support_levels) / len(support_levels)
                features["max_support_strength"] = max(level.strength for level in support_levels)
                features["support_strength_variance"] = np.var([level.strength for level in support_levels])
            else:
                features["avg_support_strength"] = 0.0
                features["max_support_strength"] = 0.0
                features["support_strength_variance"] = 0.0
                
            if resistance_levels:
                features["avg_resistance_strength"] = sum(level.strength for level in resistance_levels) / len(resistance_levels)
                features["max_resistance_strength"] = max(level.strength for level in resistance_levels)
                features["resistance_strength_variance"] = np.var([level.strength for level in resistance_levels])
            else:
                features["avg_resistance_strength"] = 0.0
                features["max_resistance_strength"] = 0.0
                features["resistance_strength_variance"] = 0.0
            
            # Method distribution features
            method_counts = {}
            for level in support_levels + resistance_levels:
                method = level.method
                method_counts[method] = method_counts.get(method, 0) + 1
            
            for method in ["pivot", "volume", "fractal", "fibonacci", "psychological", "atr"]:
                features[f"sr_method_{method}_count"] = method_counts.get(method, 0)
            
            # Breakout probability features
            if support_levels:
                support_breakout_probs = [level.breakout_probability for level in support_levels]
                features["avg_support_breakout_prob"] = sum(support_breakout_probs) / len(support_breakout_probs)
                features["max_support_breakout_prob"] = max(support_breakout_probs)
            else:
                features["avg_support_breakout_prob"] = 0.0
                features["max_support_breakout_prob"] = 0.0
                
            if resistance_levels:
                resistance_breakout_probs = [level.breakout_probability for level in resistance_levels]
                features["avg_resistance_breakout_prob"] = sum(resistance_breakout_probs) / len(resistance_breakout_probs)
                features["max_resistance_breakout_prob"] = max(resistance_breakout_probs)
            else:
                features["avg_resistance_breakout_prob"] = 0.0
                features["max_resistance_breakout_prob"] = 0.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive S/R features: {e}")
            return {}

    async def get_sr_features_for_engineering(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """
        Get S/R features for feature engineering integration.
        
        Args:
            market_data: Market data DataFrame
            
        Returns:
            dict[str, Any]: S/R features ready for feature engineering
        """
        try:
            # Perform centralized S/R analysis
            analysis_results = await self.analyze_centralized_sr_levels(market_data)
            
            # Extract features
            sr_features = analysis_results.get("sr_features", {})
            
            # Add quality metrics
            quality_metrics = analysis_results.get("quality_metrics", {})
            sr_features.update({
                f"sr_quality_{key}": value for key, value in quality_metrics.items()
            })
            
            # Add redundancy metrics
            redundancy_metrics = analysis_results.get("redundancy_metrics", {})
            sr_features.update({
                f"sr_redundancy_{key}": value for key, value in redundancy_metrics.items()
            })
            
            self.logger.info("✅ S/R features prepared for feature engineering")
            return sr_features
            
        except Exception as e:
            self.logger.error(f"Error getting S/R features for engineering: {e}")
            return {}

    # Enhanced Centralized S/R Analysis Methods
    
    @handle_errors(
        exceptions=(Exception,),
        default_return={"sr_levels": [], "quality_metrics": {}, "redundancy_metrics": {}},
        context="centralized S/R analysis",
    )
    async def analyze_centralized_sr_levels(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive centralized S/R analysis using multiple methods.
        
        Args:
            market_data: Market data DataFrame
            
        Returns:
            Dict[str, Any]: Comprehensive S/R analysis results
        """
        try:
            self.logger.info("🔍 Starting centralized S/R analysis...")
            
            # Initialize results
            analysis_results = {
                "sr_levels": [],
                "quality_metrics": {},
                "redundancy_metrics": {},
                "sr_features": {},
                "breakout_events": []
            }
            
            # Perform different S/R detection methods
            if self.enable_fractal_analysis:
                fractal_levels = await self._detect_fractal_sr_levels(market_data)
                analysis_results["sr_levels"].extend(fractal_levels)
            
            if self.enable_volume_profile:
                volume_levels = await self._detect_volume_sr_levels(market_data)
                analysis_results["sr_levels"].extend(volume_levels)
            
            if self.enable_psychological_levels:
                psychological_levels = await self._detect_psychological_sr_levels(market_data)
                analysis_results["sr_levels"].extend(psychological_levels)
            
            # Perform composite analysis
            if self.enable_composite_sr:
                composite_levels = await self._create_composite_sr_levels(analysis_results["sr_levels"])
                analysis_results["sr_levels"] = composite_levels
            
            # Calculate quality metrics
            analysis_results["quality_metrics"] = await self._calculate_sr_quality_metrics(
                analysis_results["sr_levels"], market_data
            )
            
            # Calculate redundancy metrics
            analysis_results["redundancy_metrics"] = await self._calculate_sr_redundancy_metrics(
                analysis_results["sr_levels"]
            )
            
            # Generate S/R features
            analysis_results["sr_features"] = await self._generate_sr_features(
                analysis_results["sr_levels"], market_data
            )
            
            # Detect breakout events
            if self.enable_breakout_prediction:
                analysis_results["breakout_events"] = await self._detect_breakout_events(
                    analysis_results["sr_levels"], market_data
                )
            
            # Update state
            self.sr_levels = analysis_results["sr_levels"]
            self.sr_analysis_state["last_sr_analysis"] = pd.Timestamp.now()
            self.sr_analysis_state["sr_detection_count"] += 1
            
            self.logger.info(f"✅ Centralized S/R analysis completed: {len(analysis_results['sr_levels'])} levels detected")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in centralized S/R analysis: {e}")
            return {"sr_levels": [], "quality_metrics": {}, "redundancy_metrics": {}}

    async def _detect_fractal_sr_levels(self, market_data: pd.DataFrame) -> List[SRLevel]:
        """Detect S/R levels using fractal analysis."""
        try:
            levels = []
            high = market_data['high'].values
            low = market_data['low'].values
            close = market_data['close'].values
            volume = market_data['volume'].values
            timestamps = market_data.index
            
            # Detect swing highs and lows
            for i in range(self.fractal_window, len(high) - self.fractal_window):
                # Swing high detection
                if all(high[i] >= high[j] for j in range(i - self.fractal_window, i)) and \
                   all(high[i] >= high[j] for j in range(i + 1, i + self.fractal_window + 1)):
                    
                    level = SRLevel(
                        price=high[i],
                        level_type=SRType.FRACTAL,
                        strength=0.5,  # Base strength
                        confidence=0.6,
                        touches=1,
                        volume=volume[i],
                        age=0,
                        timestamp=timestamps[i],
                        method="fractal_high",
                        fractal_quality=0.8
                    )
                    levels.append(level)
                
                # Swing low detection
                if all(low[i] <= low[j] for j in range(i - self.fractal_window, i)) and \
                   all(low[i] <= low[j] for j in range(i + 1, i + self.fractal_window + 1)):
                    
                    level = SRLevel(
                        price=low[i],
                        level_type=SRType.FRACTAL,
                        strength=0.5,  # Base strength
                        confidence=0.6,
                        touches=1,
                        volume=volume[i],
                        age=0,
                        timestamp=timestamps[i],
                        method="fractal_low",
                        fractal_quality=0.8
                    )
                    levels.append(level)
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error in fractal S/R detection: {e}")
            return []

    async def _detect_volume_sr_levels(self, market_data: pd.DataFrame) -> List[SRLevel]:
        """Detect S/R levels using volume profile analysis."""
        try:
            levels = []
            close = market_data['close'].values
            volume = market_data['volume'].values
            timestamps = market_data.index
            
            # Calculate volume-weighted average price
            vwap = np.sum(close * volume) / np.sum(volume)
            
            # Find high volume price levels
            volume_threshold = np.mean(volume) * self.volume_threshold
            
            for i in range(len(close)):
                if volume[i] > volume_threshold:
                    level = SRLevel(
                        price=close[i],
                        level_type=SRType.VOLUME,
                        strength=min(volume[i] / volume_threshold, 2.0),
                        confidence=0.7,
                        touches=1,
                        volume=volume[i],
                        age=0,
                        timestamp=timestamps[i],
                        method="volume_profile",
                        volume_profile=volume[i] / np.mean(volume)
                    )
                    levels.append(level)
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error in volume S/R detection: {e}")
            return []

    async def _detect_psychological_sr_levels(self, market_data: pd.DataFrame) -> List[SRLevel]:
        """Detect psychological S/R levels."""
        try:
            levels = []
            close = market_data['close'].values
            timestamps = market_data.index
            
            # Common psychological levels
            if not self.psychological_levels:
                self.psychological_levels = [0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 50.0, 100.0]
            
            for i in range(len(close)):
                for psych_level in self.psychological_levels:
                    proximity = abs(close[i] - psych_level) / psych_level
                    if proximity < 0.01:  # Within 1% of psychological level
                        level = SRLevel(
                            price=psych_level,
                            level_type=SRType.PSYCHOLOGICAL,
                            strength=0.6,
                            confidence=0.8,
                            touches=1,
                            volume=0.0,
                            age=0,
                            timestamp=timestamps[i],
                            method="psychological",
                            psychological_weight=1.0 - proximity
                        )
                        levels.append(level)
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error in psychological S/R detection: {e}")
            return []

    async def _create_composite_sr_levels(self, individual_levels: List[SRLevel]) -> List[SRLevel]:
        """Create composite S/R levels by combining individual detections."""
        try:
            if not individual_levels:
                return []
            
            # Group levels by proximity
            grouped_levels = []
            used_indices = set()
            
            for i, level1 in enumerate(individual_levels):
                if i in used_indices:
                    continue
                
                group = [level1]
                used_indices.add(i)
                
                for j, level2 in enumerate(individual_levels[i+1:], i+1):
                    if j in used_indices:
                        continue
                    
                    # Check proximity
                    proximity = abs(level1.price - level2.price) / level1.price
                    if proximity < 0.02:  # Within 2%
                        group.append(level2)
                        used_indices.add(j)
                
                if len(group) > 1:
                    # Create composite level
                    composite_price = np.mean([level.price for level in group])
                    composite_strength = np.mean([level.strength for level in group])
                    composite_confidence = np.mean([level.confidence for level in group])
                    total_touches = sum(level.touches for level in group)
                    total_volume = sum(level.volume for level in group)
                    
                    composite_level = SRLevel(
                        price=composite_price,
                        level_type=SRType.COMPOSITE,
                        strength=composite_strength * 1.2,  # Boost for composite
                        confidence=composite_confidence * 1.1,
                        touches=total_touches,
                        volume=total_volume,
                        age=0,
                        timestamp=group[0].timestamp,
                        method="composite",
                        composite_score=len(group) * 0.1
                    )
                    grouped_levels.append(composite_level)
                else:
                    grouped_levels.append(level1)
            
            return grouped_levels
            
        except Exception as e:
            self.logger.error(f"Error creating composite S/R levels: {e}")
            return individual_levels

    async def _calculate_sr_quality_metrics(self, sr_levels: List[SRLevel], market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate quality metrics for S/R levels."""
        try:
            if not sr_levels:
                return {}
            
            metrics = {}
            
            # Strength distribution
            strengths = [level.strength for level in sr_levels]
            metrics["avg_strength"] = np.mean(strengths)
            metrics["strength_std"] = np.std(strengths)
            metrics["max_strength"] = np.max(strengths)
            
            # Confidence distribution
            confidences = [level.confidence for level in sr_levels]
            metrics["avg_confidence"] = np.mean(confidences)
            metrics["confidence_std"] = np.std(confidences)
            
            # Method distribution
            method_counts = {}
            for level in sr_levels:
                method = level.method
                method_counts[method] = method_counts.get(method, 0) + 1
            
            for method, count in method_counts.items():
                metrics[f"method_{method}_ratio"] = count / len(sr_levels)
            
            # Volume profile quality
            volume_profiles = [level.volume_profile for level in sr_levels if level.volume_profile > 0]
            if volume_profiles:
                metrics["avg_volume_profile"] = np.mean(volume_profiles)
                metrics["volume_profile_std"] = np.std(volume_profiles)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating S/R quality metrics: {e}")
            return {}

    async def _calculate_sr_redundancy_metrics(self, sr_levels: List[SRLevel]) -> Dict[str, float]:
        """Calculate redundancy metrics for S/R levels."""
        try:
            if len(sr_levels) < 2:
                return {"redundancy_score": 0.0}
            
            # Calculate proximity matrix
            prices = [level.price for level in sr_levels]
            proximity_matrix = np.zeros((len(prices), len(prices)))
            
            for i in range(len(prices)):
                for j in range(i+1, len(prices)):
                    proximity = abs(prices[i] - prices[j]) / prices[i]
                    proximity_matrix[i, j] = proximity
                    proximity_matrix[j, i] = proximity
            
            # Calculate redundancy score
            close_pairs = np.sum(proximity_matrix < 0.02) / 2  # Pairs within 2%
            total_pairs = len(prices) * (len(prices) - 1) / 2
            redundancy_score = close_pairs / total_pairs if total_pairs > 0 else 0.0
            
            return {
                "redundancy_score": redundancy_score,
                "close_pairs_ratio": close_pairs / len(sr_levels) if sr_levels else 0.0,
                "avg_proximity": np.mean(proximity_matrix[proximity_matrix > 0])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating S/R redundancy metrics: {e}")
            return {"redundancy_score": 0.0}

    async def _generate_sr_features(self, sr_levels: List[SRLevel], market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Generate comprehensive, non-redundant S/R features for machine learning.
        Enhanced with advanced feature categories and redundancy elimination.
        """
        try:
            features = {}
            
            if not sr_levels:
                # Return comprehensive default features
                default_features = [
                    "sr_level_count", "avg_sr_strength", "sr_strength_std", 
                    "avg_sr_confidence", "sr_confidence_std", "sr_redundancy_score",
                    "sr_quality_score", "sr_composite_score", "sr_breakout_probability",
                    "sr_volume_profile", "sr_psychological_weight", "sr_fractal_quality"
                ]
                for feature_name in default_features:
                    features[feature_name] = 0.0
                return features
            
            # === BASIC S/R FEATURES ===
            features["sr_level_count"] = len(sr_levels)
            features["avg_sr_strength"] = np.mean([level.strength for level in sr_levels])
            features["sr_strength_std"] = np.std([level.strength for level in sr_levels])
            features["avg_sr_confidence"] = np.mean([level.confidence for level in sr_levels])
            features["sr_confidence_std"] = np.std([level.confidence for level in sr_levels])
            
            # === METHOD-SPECIFIC FEATURES ===
            method_counts = {}
            method_strengths = {}
            for level in sr_levels:
                method = level.method
                method_counts[method] = method_counts.get(method, 0) + 1
                if method not in method_strengths:
                    method_strengths[method] = []
                method_strengths[method].append(level.strength)
            
            for method in ["fractal", "volume", "psychological", "composite", "pivot", "fibonacci", "atr"]:
                count = method_counts.get(method, 0)
                features[f"sr_method_{method}_count"] = count
                features[f"sr_method_{method}_ratio"] = count / len(sr_levels) if sr_levels else 0.0
                features[f"sr_method_{method}_avg_strength"] = np.mean(method_strengths.get(method, [0.0]))
            
            # === PROXIMITY AND POSITIONING FEATURES ===
            current_price = market_data['close'].iloc[-1]
            proximities = [abs(current_price - level.price) / current_price for level in sr_levels]
            features["min_sr_proximity"] = min(proximities) if proximities else 1.0
            features["avg_sr_proximity"] = np.mean(proximities) if proximities else 1.0
            features["sr_proximity_std"] = np.std(proximities) if proximities else 0.0
            
            # === ADVANCED S/R FEATURES ===
            features["sr_quality_score"] = np.mean([level.composite_score for level in sr_levels])
            features["sr_composite_score"] = np.mean([level.composite_score for level in sr_levels])
            features["sr_breakout_probability"] = np.mean([level.breakout_probability for level in sr_levels])
            features["sr_volume_profile"] = np.mean([level.volume_profile for level in sr_levels])
            features["sr_psychological_weight"] = np.mean([level.psychological_weight for level in sr_levels])
            features["sr_fractal_quality"] = np.mean([level.fractal_quality for level in sr_levels])
            
            # === AGE AND PERSISTENCE FEATURES ===
            ages = [level.age for level in sr_levels]
            features["avg_sr_age"] = np.mean(ages) if ages else 0.0
            features["sr_age_std"] = np.std(ages) if ages else 0.0
            features["sr_persistence_score"] = np.mean([level.touches for level in sr_levels])
            
            # === VOLUME AND TOUCH FEATURES ===
            features["avg_sr_touches"] = np.mean([level.touches for level in sr_levels])
            features["sr_touches_std"] = np.std([level.touches for level in sr_levels])
            features["avg_sr_volume"] = np.mean([level.volume for level in sr_levels])
            
            # === REDUNDANCY ELIMINATION FEATURES ===
            # Calculate feature correlation to identify redundancy
            feature_matrix = np.array([
                [level.strength, level.confidence, level.touches, level.age, level.volume_profile]
                for level in sr_levels
            ])
            
            if len(feature_matrix) > 1:
                # Calculate correlation matrix
                corr_matrix = np.corrcoef(feature_matrix.T)
                # Average correlation (excluding diagonal)
                avg_correlation = (np.sum(corr_matrix) - np.trace(corr_matrix)) / (len(corr_matrix) ** 2 - len(corr_matrix))
                features["sr_feature_correlation"] = avg_correlation
            else:
                features["sr_feature_correlation"] = 0.0
            
            # === INTEGRATION STATUS FEATURES ===
            features["sr_feature_engineering_ready"] = 1.0
            features["sr_analyst_component_ready"] = 1.0
            features["sr_quality_control_passed"] = 1.0 if features["sr_quality_score"] > 0.6 else 0.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error generating S/R features: {e}")
            return {}

    async def _detect_breakout_events(self, sr_levels: List[SRLevel], market_data: pd.DataFrame) -> List[SRBreakoutEvent]:
        """Detect S/R breakout events."""
        try:
            events = []
            close = market_data['close'].values
            volume = market_data['volume'].values
            timestamps = market_data.index
            
            for level in sr_levels:
                # Check for breakouts
                for i in range(len(close)):
                    price = close[i]
                    level_price = level.price
                    
                    # Support break
                    if price < level_price * 0.98:  # 2% below support
                        event = SRBreakoutEvent(
                            level=level,
                            breakout_type="support_break",
                            confidence=level.confidence,
                            volume_confirmation=volume[i] / np.mean(volume),
                            price_momentum=(price - level_price) / level_price,
                            timestamp=timestamps[i],
                            trigger_features={
                                "price_momentum": (price - level_price) / level_price,
                                "volume_confirmation": volume[i] / np.mean(volume),
                                "level_strength": level.strength
                            }
                        )
                        events.append(event)
                        break
                    
                    # Resistance break
                    elif price > level_price * 1.02:  # 2% above resistance
                        event = SRBreakoutEvent(
                            level=level,
                            breakout_type="resistance_break",
                            confidence=level.confidence,
                            volume_confirmation=volume[i] / np.mean(volume),
                            price_momentum=(price - level_price) / level_price,
                            timestamp=timestamps[i],
                            trigger_features={
                                "price_momentum": (price - level_price) / level_price,
                                "volume_confirmation": volume[i] / np.mean(volume),
                                "level_strength": level.strength
                            }
                        )
                        events.append(event)
                        break
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error detecting breakout events: {e}")
            return []

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
            self.sr_levels_cache.clear()
            self.sr_analysis_history.clear()
            self.sr_quality_metrics.clear()
            self.sr_analysis_state.clear()
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
