# src/analyst/advanced_feature_engineering.py

"""
Advanced Feature Engineering for enhanced financial performance.
Implements sophisticated market microstructure features, regime detection,
and adaptive indicators for improved prediction accuracy.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)


class CandlestickPatternAnalyzer:
    """
    Comprehensive candlestick pattern analyzer implementing all major patterns
    for enhanced feature engineering and ML model training.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("CandlestickPatternAnalyzer")

        # Pattern detection parameters
        self.pattern_config = config.get("candlestick_patterns", {})
        self.doji_threshold = self.pattern_config.get("doji_threshold", 0.1)
        self.hammer_ratio = self.pattern_config.get("hammer_ratio", 0.3)
        self.shadow_ratio = self.pattern_config.get("shadow_ratio", 2.0)
        self.engulfing_ratio = self.pattern_config.get("engulfing_ratio", 1.1)
        self.tweezer_threshold = self.pattern_config.get("tweezer_threshold", 0.02)
        self.marubozu_threshold = self.pattern_config.get("marubozu_threshold", 0.1)

        self.is_initialized = False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="candlestick pattern analyzer initialization",
    )
    async def initialize(self) -> bool:
        """Initialize candlestick pattern analyzer."""
        try:
            self.logger.info("🚀 Initializing candlestick pattern analyzer...")
            self.is_initialized = True
            self.logger.info("✅ Candlestick pattern analyzer initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(
                f"❌ Error initializing candlestick pattern analyzer: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return={},
        context="candlestick pattern analysis",
    )
    async def analyze_patterns(self, price_data: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze candlestick patterns and return features for ML training.

        Args:
            price_data: OHLCV price data

        Returns:
            Dictionary containing candlestick pattern features
        """
        try:
            if not self.is_initialized:
                self.print(
                    initialization_error("Candlestick pattern analyzer not initialized")
                )
                return {}

            if price_data.empty or len(price_data) < 3:
                self.print(warning("Insufficient data for pattern analysis"))
                return {}

            # Prepare data with calculated metrics
            df = self._prepare_candlestick_data(price_data)

            # Analyze all patterns
            patterns = {
                "engulfing_patterns": self._detect_engulfing_patterns(df),
                "hammer_hanging_man": self._detect_hammer_hanging_man(df),
                "shooting_star_inverted_hammer": self._detect_shooting_star_inverted_hammer(
                    df,
                ),
                "tweezer_patterns": self._detect_tweezer_patterns(df),
                "marubozu_patterns": self._detect_marubozu_patterns(df),
                "three_methods_patterns": self._detect_three_methods_patterns(df),
                "doji_patterns": self._detect_doji_patterns(df),
                "spinning_top_patterns": self._detect_spinning_top_patterns(df),
            }

            # Convert patterns to ML features
            features = self._convert_patterns_to_features(patterns, df)

            self.logger.info(f"✅ Analyzed {len(patterns)} pattern categories")
            return features

        except Exception as e:
            self.print(error("Error analyzing candlestick patterns: {e}"))
            return {}

    def _prepare_candlestick_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with candlestick metrics using price differences."""
        try:
            df = df.copy()

            # Calculate basic candlestick metrics using price differences
            df["body_size"] = abs(df["close"].diff() - df["open"].diff())
            df["upper_shadow"] = df["high"].diff() - np.maximum(
                df["open"].diff(), df["close"].diff()
            )
            df["lower_shadow"] = (
                np.minimum(df["open"].diff(), df["close"].diff()) - df["low"].diff()
            )
            df["total_range"] = df["high"].diff() - df["low"].diff()
            df["body_ratio"] = df["body_size"] / df["total_range"].replace(0, 1)
            df["is_bullish"] = df["close"].diff() > df["open"].diff()

            # Calculate moving averages for context
            df["avg_body_size"] = df["body_size"].rolling(window=20).mean()
            df["avg_range"] = df["total_range"].rolling(window=20).mean()

            return df.dropna()

        except Exception as e:
            self.print(error("Error preparing candlestick data: {e}"))
            return pd.DataFrame()

    def _detect_engulfing_patterns(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect bullish and bearish engulfing patterns."""
        patterns = []

        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i - 1]

            # Bullish engulfing
            if (
                current["is_bullish"]
                and not previous["is_bullish"]
                and current["open"] < previous["close"]
                and current["close"] > previous["open"]
                and current["body_size"] > previous["body_size"] * self.engulfing_ratio
            ):
                patterns.append(
                    {
                        "index": i,
                        "pattern": "bullish_engulfing",
                        "confidence": min(
                            current["body_size"] / previous["body_size"],
                            2.0,
                        ),
                        "is_bullish": True,
                    },
                )

            # Bearish engulfing
            elif (
                not current["is_bullish"]
                and previous["is_bullish"]
                and current["open"] > previous["close"]
                and current["close"] < previous["open"]
                and current["body_size"] > previous["body_size"] * self.engulfing_ratio
            ):
                patterns.append(
                    {
                        "index": i,
                        "pattern": "bearish_engulfing",
                        "confidence": min(
                            current["body_size"] / previous["body_size"],
                            2.0,
                        ),
                        "is_bullish": False,
                    },
                )

        return patterns

    def _detect_hammer_hanging_man(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect hammer and hanging man patterns."""
        patterns = []

        for i in range(len(df)):
            row = df.iloc[i]

            # Hammer pattern (bullish reversal)
            if (
                row["body_ratio"] <= self.hammer_ratio
                and row["lower_shadow"] > row["body_size"] * self.shadow_ratio
                and row["upper_shadow"] < row["body_size"] * 0.5
            ):
                patterns.append(
                    {
                        "index": i,
                        "pattern": "hammer",
                        "confidence": min(row["lower_shadow"] / row["body_size"], 3.0),
                        "is_bullish": True,
                    },
                )

            # Hanging man pattern (bearish reversal)
            elif (
                row["body_ratio"] <= self.hammer_ratio
                and row["lower_shadow"] > row["body_size"] * self.shadow_ratio
                and row["upper_shadow"] < row["body_size"] * 0.5
                and i > 0
                and df.iloc[i - 1]["close"] > row["close"]
            ):
                patterns.append(
                    {
                        "index": i,
                        "pattern": "hanging_man",
                        "confidence": min(row["lower_shadow"] / row["body_size"], 3.0),
                        "is_bullish": False,
                    },
                )

        return patterns

    def _detect_shooting_star_inverted_hammer(
        self,
        df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Detect shooting star and inverted hammer patterns."""
        patterns = []

        for i in range(len(df)):
            row = df.iloc[i]

            # Shooting star pattern (bearish reversal)
            if (
                row["body_ratio"] <= self.hammer_ratio
                and row["upper_shadow"] > row["body_size"] * self.shadow_ratio
                and row["lower_shadow"] < row["body_size"] * 0.5
            ):
                patterns.append(
                    {
                        "index": i,
                        "pattern": "shooting_star",
                        "confidence": min(row["upper_shadow"] / row["body_size"], 3.0),
                        "is_bullish": False,
                    },
                )

            # Inverted hammer pattern (bullish reversal)
            elif (
                row["body_ratio"] <= self.hammer_ratio
                and row["upper_shadow"] > row["body_size"] * self.shadow_ratio
                and row["lower_shadow"] < row["body_size"] * 0.5
                and i > 0
                and df.iloc[i - 1]["close"] < row["close"]
            ):
                patterns.append(
                    {
                        "index": i,
                        "pattern": "inverted_hammer",
                        "confidence": min(row["upper_shadow"] / row["body_size"], 3.0),
                        "is_bullish": True,
                    },
                )

        return patterns

    def _detect_tweezer_patterns(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect tweezer tops and bottoms patterns."""
        patterns = []

        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i - 1]

            # Tweezer tops (bearish reversal)
            if (
                abs(current["high"] - previous["high"])
                <= self.tweezer_threshold * current["high"]
                and current["high"] > current["close"]
                and previous["high"] > previous["close"]
            ):
                patterns.append(
                    {
                        "index": i,
                        "pattern": "tweezer_top",
                        "confidence": 1.0
                        - abs(current["high"] - previous["high"]) / current["high"],
                        "is_bullish": False,
                    },
                )

            # Tweezer bottoms (bullish reversal)
            elif (
                abs(current["low"] - previous["low"])
                <= self.tweezer_threshold * current["low"]
                and current["low"] < current["open"]
                and previous["low"] < previous["open"]
            ):
                patterns.append(
                    {
                        "index": i,
                        "pattern": "tweezer_bottom",
                        "confidence": 1.0
                        - abs(current["low"] - previous["low"]) / current["low"],
                        "is_bullish": True,
                    },
                )

        return patterns

    def _detect_marubozu_patterns(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect bullish and bearish marubozu patterns."""
        patterns = []

        for i in range(len(df)):
            row = df.iloc[i]

            # Marubozu (no shadows or very small shadows)
            if (
                row["upper_shadow"] < row["total_range"] * self.marubozu_threshold
                and row["lower_shadow"] < row["total_range"] * self.marubozu_threshold
            ):
                pattern_type = (
                    "bullish_marubozu" if row["is_bullish"] else "bearish_marubozu"
                )
                confidence = (
                    1.0
                    - (row["upper_shadow"] + row["lower_shadow"]) / row["total_range"]
                )

                patterns.append(
                    {
                        "index": i,
                        "pattern": pattern_type,
                        "confidence": confidence,
                        "is_bullish": row["is_bullish"],
                    },
                )

        return patterns

    def _detect_three_methods_patterns(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect rising and falling three methods patterns."""
        patterns = []

        # Look for 5-candle patterns
        for i in range(4, len(df)):
            # Rising three methods (bullish continuation)
            if self._is_rising_three_methods(df, i):
                patterns.append(
                    {
                        "index": i,
                        "pattern": "rising_three_methods",
                        "confidence": 0.8,
                        "is_bullish": True,
                    },
                )

            # Falling three methods (bearish continuation)
            elif self._is_falling_three_methods(df, i):
                patterns.append(
                    {
                        "index": i,
                        "pattern": "falling_three_methods",
                        "confidence": 0.8,
                        "is_bullish": False,
                    },
                )

        return patterns

    def _is_rising_three_methods(self, df: pd.DataFrame, index: int) -> bool:
        """Check if the 5-candle pattern is a rising three methods."""
        if index < 4:
            return False

        candles = [df.iloc[i] for i in range(index - 4, index + 1)]

        # First candle should be a long bullish candle
        if not (
            candles[0]["is_bullish"]
            and candles[0]["body_size"] > candles[0]["avg_body_size"]
        ):
            return False

        # Next three candles should be small bearish candles within the range of the first
        for i in range(1, 4):
            if (
                candles[i]["is_bullish"]
                or candles[i]["high"] > candles[0]["high"]
                or candles[i]["low"] < candles[0]["low"]
            ):
                return False

        # Last candle should be a long bullish candle closing above the first
        return (
            candles[4]["is_bullish"]
            and candles[4]["close"] > candles[0]["close"]
            and candles[4]["body_size"] > candles[4]["avg_body_size"]
        )

    def _is_falling_three_methods(self, df: pd.DataFrame, index: int) -> bool:
        """Check if the 5-candle pattern is a falling three methods."""
        if index < 4:
            return False

        candles = [df.iloc[i] for i in range(index - 4, index + 1)]

        # First candle should be a long bearish candle
        if not (
            not candles[0]["is_bullish"]
            and candles[0]["body_size"] > candles[0]["avg_body_size"]
        ):
            return False

        # Next three candles should be small bullish candles within the range of the first
        for i in range(1, 4):
            if (
                not candles[i]["is_bullish"]
                or candles[i]["high"] > candles[0]["high"]
                or candles[i]["low"] < candles[0]["low"]
            ):
                return False

        # Last candle should be a long bearish candle closing below the first
        return (
            not candles[4]["is_bullish"]
            and candles[4]["close"] < candles[0]["close"]
            and candles[4]["body_size"] > candles[4]["avg_body_size"]
        )

    def _detect_doji_patterns(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect doji patterns."""
        patterns = []

        for i in range(len(df)):
            row = df.iloc[i]

            # Doji pattern (very small body)
            if row["body_ratio"] <= self.doji_threshold:
                patterns.append(
                    {
                        "index": i,
                        "pattern": "doji",
                        "confidence": 1.0 - row["body_ratio"],
                        "is_bullish": None,  # Doji is neutral
                    },
                )

        return patterns

    def _detect_spinning_top_patterns(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Detect spinning top patterns."""
        patterns = []

        for i in range(len(df)):
            row = df.iloc[i]

            # Spinning top (small body, equal shadows)
            if (
                row["body_ratio"] <= 0.3
                and abs(row["upper_shadow"] - row["lower_shadow"])
                < 0.2 * row["total_range"]
                and row["upper_shadow"] > 0.1 * row["total_range"]
                and row["lower_shadow"] > 0.1 * row["total_range"]
            ):
                patterns.append(
                    {
                        "index": i,
                        "pattern": "spinning_top",
                        "confidence": 0.7,
                        "is_bullish": None,  # Spinning top is neutral
                    },
                )

        return patterns

    def _calculate_pattern_type_features(
        self,
        patterns: dict[str, list[dict[str, Any]]],
    ) -> dict[str, float]:
        """Calculate pattern type features (count and presence)."""
        features = {}

        # Pattern presence features (binary)
        pattern_types = [
            "engulfing_patterns",
            "hammer_hanging_man",
            "shooting_star_inverted_hammer",
            "tweezer_patterns",
            "marubozu_patterns",
            "three_methods_patterns",
            "doji_patterns",
            "spinning_top_patterns",
        ]

        for pattern_type in pattern_types:
            pattern_list = patterns.get(pattern_type, [])
            features[f"{pattern_type}_count"] = len(pattern_list)
            features[f"{pattern_type}_present"] = 1.0 if pattern_list else 0.0

        return features

    def _calculate_specific_pattern_features(
        self,
        patterns: dict[str, list[dict[str, Any]]],
    ) -> dict[str, float]:
        """Calculate specific pattern features (count and presence)."""
        features = {}

        # Specific pattern features
        specific_patterns = [
            "bullish_engulfing",
            "bearish_engulfing",
            "hammer",
            "hanging_man",
            "shooting_star",
            "inverted_hammer",
            "tweezer_top",
            "tweezer_bottom",
            "bullish_marubozu",
            "bearish_marubozu",
            "rising_three_methods",
            "falling_three_methods",
            "doji",
            "spinning_top",
        ]

        for pattern in specific_patterns:
            count = sum(
                1
                for pattern_list in patterns.values()
                for p in pattern_list
                if p.get("pattern") == pattern
            )
            features[f"{pattern}_count"] = count
            features[f"{pattern}_present"] = 1.0 if count > 0 else 0.0

        return features

    def _calculate_pattern_density_features(
        self,
        patterns: dict[str, list[dict[str, Any]]],
        df: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate pattern density features."""
        features = {}

        # Pattern density features
        total_patterns = sum(len(pattern_list) for pattern_list in patterns.values())
        features["total_patterns"] = total_patterns
        features["pattern_density"] = total_patterns / len(df) if len(df) > 0 else 0.0

        return features

    def _calculate_bullish_bearish_features(
        self,
        patterns: dict[str, list[dict[str, Any]]],
    ) -> dict[str, float]:
        """Calculate bullish vs bearish pattern features."""
        features = {}

        # Bullish vs bearish pattern ratio
        bullish_patterns = sum(
            1
            for pattern_list in patterns.values()
            for p in pattern_list
            if p.get("is_bullish") is True
        )
        bearish_patterns = sum(
            1
            for pattern_list in patterns.values()
            for p in pattern_list
            if p.get("is_bullish") is False
        )

        features["bullish_patterns"] = bullish_patterns
        features["bearish_patterns"] = bearish_patterns
        features["bullish_bearish_ratio"] = bullish_patterns / (bearish_patterns + 1e-8)

        return features

    def _calculate_recent_pattern_features(
        self,
        patterns: dict[str, list[dict[str, Any]]],
        df: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate recent pattern features (last 5 candles)."""
        features = {}

        # Recent pattern features (last 5 candles)
        recent_patterns = []
        for pattern_list in patterns.values():
            recent_patterns.extend(
                [p for p in pattern_list if p.get("index", 0) >= len(df) - 5],
            )

        features["recent_patterns_count"] = len(recent_patterns)
        features["recent_bullish_patterns"] = sum(
            1 for p in recent_patterns if p.get("is_bullish") is True
        )
        features["recent_bearish_patterns"] = sum(
            1 for p in recent_patterns if p.get("is_bullish") is False
        )

        return features

    def _calculate_pattern_confidence_features(
        self,
        patterns: dict[str, list[dict[str, Any]]],
    ) -> dict[str, float]:
        """Calculate pattern confidence features."""
        features = {}

        # Pattern confidence features
        if patterns:
            all_confidences = [
                p.get("confidence", 0.0)
                for pattern_list in patterns.values()
                for p in pattern_list
            ]
            features["avg_pattern_confidence"] = (
                np.mean(all_confidences) if all_confidences else 0.0
            )
            features["max_pattern_confidence"] = (
                np.max(all_confidences) if all_confidences else 0.0
            )
            features["pattern_confidence_std"] = (
                np.std(all_confidences) if all_confidences else 0.0
            )
        else:
            features["avg_pattern_confidence"] = 0.0
            features["max_pattern_confidence"] = 0.0
            features["pattern_confidence_std"] = 0.0

        return features

    def _convert_patterns_to_features(
        self,
        patterns: dict[str, list[dict[str, Any]]],
        df: pd.DataFrame,
    ) -> dict[str, float]:
        """Convert pattern analysis to ML features."""
        try:
            features = {}

            # Calculate different types of pattern features
            features.update(self._calculate_pattern_type_features(patterns))
            features.update(self._calculate_specific_pattern_features(patterns))
            features.update(self._calculate_pattern_density_features(patterns, df))
            features.update(self._calculate_bullish_bearish_features(patterns))
            features.update(self._calculate_recent_pattern_features(patterns, df))
            features.update(self._calculate_pattern_confidence_features(patterns))

            return features

        except Exception as e:
            self.print(error("Error converting patterns to features: {e}"))
            return {}


class FeatureInteractionEngine:
    """
    Engine for creating feature interaction terms to capture complex market dynamics.
    Focuses on creating meaningful interactions between normalized features like
    spread and volume metrics, with explicit support for lagged relationships.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("FeatureInteractionEngine")
        
        # Interaction configuration
        self.interaction_config = config.get("feature_interactions", {})
        self.enable_interactions = self.interaction_config.get("enable_interactions", True)
        self.max_interactions = self.interaction_config.get("max_interactions", 20)
        self.interaction_threshold = self.interaction_config.get("interaction_threshold", 0.1)
        
        # Lag configuration for causality-aware interactions
        self.lag_config = self.interaction_config.get("lag_config", {})
        self.max_lag = self.lag_config.get("max_lag", 5)  # Maximum lag to test
        self.enable_lagged_interactions = self.lag_config.get("enable_lagged_interactions", True)
        self.causality_test_lags = self.lag_config.get("causality_test_lags", [1, 2, 3, 5])  # Specific lags to test
        
        # Define feature groups for interactions
        self.spread_features = [
            "spread_liquidity", "spread_liquidity_bps", "spread_liquidity_z_score",
            "spread_liquidity_change", "spread_liquidity_pct_change", "bid_ask_spread"
        ]
        
        self.volume_features = [
            "volume_roc", "volume_pct_change", "volume_z_score", "volume_liquidity",
            "volume_ma_ratio", "volume_log_diff", "volume_liquidity_z_score"
        ]
        
        self.volatility_features = [
            "realized_volatility", "parkinson_volatility", "garman_klass_volatility",
            "realized_volatility_z_score", "volatility_regime", "volatility_percentile"
        ]
        
        self.momentum_features = [
            "momentum_5", "momentum_10", "momentum_20", "momentum_50",
            "momentum_z_score", "momentum_acceleration", "momentum_strength"
        ]
        
        self.liquidity_features = [
            "price_impact", "kyle_lambda", "amihud_illiquidity", "liquidity_regime",
            "liquidity_percentile", "liquidity_stress", "liquidity_health"
        ]
        
        # Causality-aware feature pairs (predictor -> target)
        self.causality_pairs = [
            # Spread changes predict volume changes
            ("spread_liquidity_change", "volume_roc"),
            ("spread_liquidity_pct_change", "volume_pct_change"),
            ("bid_ask_spread", "volume_liquidity"),
            
            # Volume changes predict price impact
            ("volume_roc", "price_impact"),
            ("volume_pct_change", "kyle_lambda"),
            ("volume_liquidity", "amihud_illiquidity"),
            
            # Volatility changes predict momentum
            ("realized_volatility", "momentum_5"),
            ("parkinson_volatility", "momentum_acceleration"),
            ("volatility_regime", "momentum_strength"),
            
            # Momentum changes predict liquidity
            ("momentum_5", "liquidity_stress"),
            ("momentum_acceleration", "liquidity_health"),
            ("momentum_strength", "liquidity_percentile"),
        ]
        
        self.is_initialized = False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="feature interaction engine initialization",
    )
    async def initialize(self) -> bool:
        """Initialize feature interaction engine."""
        try:
            self.logger.info("🚀 Initializing feature interaction engine...")
            self.is_initialized = True
            self.logger.info("✅ Feature interaction engine initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(
                f"❌ Error initializing feature interaction engine: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return={},
        context="feature interaction generation",
    )
    async def generate_interactions(self, features: dict[str, Any]) -> dict[str, Any]:
        """
        Generate feature interaction terms from normalized features.
        
        Args:
            features: Dictionary of normalized features
            
        Returns:
            Dictionary containing original features plus interaction terms
        """
        try:
            if not self.is_initialized:
                self.print(
                    initialization_error("Feature interaction engine not initialized")
                )
                return features

            if not self.enable_interactions:
                return features

            self.logger.info("🔗 Generating feature interactions...")
            
            # Start with original features
            interaction_features = features.copy()
            
            # Generate concurrent interactions (t=0)
            concurrent_interactions = self._generate_concurrent_interactions(features)
            interaction_features.update(concurrent_interactions)
            
            # Generate lagged interactions for causality
            if self.enable_lagged_interactions:
                lagged_interactions = self._generate_lagged_interactions(features)
                interaction_features.update(lagged_interactions)
            
            # Generate causality-aware interactions
            causality_interactions = self._generate_causality_interactions(features)
            interaction_features.update(causality_interactions)
            
            # Filter interactions based on significance
            filtered_interactions = self._filter_significant_interactions(interaction_features)
            
            self.logger.info(f"✅ Generated {len(filtered_interactions) - len(features)} interaction terms")
            return filtered_interactions

        except Exception as e:
            self.print(error("Error generating feature interactions: {e}"))
            return features

    def _generate_concurrent_interactions(self, features: dict[str, Any]) -> dict[str, float]:
        """Generate concurrent (t=0) feature interactions."""
        interactions = {}
        
        try:
            # Generate spread-volume interactions (primary focus)
            spread_volume_interactions = self._generate_spread_volume_interactions(features)
            interactions.update(spread_volume_interactions)
            
            # Generate volatility-momentum interactions
            volatility_momentum_interactions = self._generate_volatility_momentum_interactions(features)
            interactions.update(volatility_momentum_interactions)
            
            # Generate liquidity-pressure interactions
            liquidity_pressure_interactions = self._generate_liquidity_pressure_interactions(features)
            interactions.update(liquidity_pressure_interactions)
            
            # Generate cross-regime interactions
            cross_regime_interactions = self._generate_cross_regime_interactions(features)
            interactions.update(cross_regime_interactions)
            
            return interactions
            
        except Exception as e:
            self.logger.warning(f"Error generating concurrent interactions: {e}")
            return interactions

    def _generate_lagged_interactions(self, features: dict[str, Any]) -> dict[str, float]:
        """Generate lagged interactions for causality testing."""
        interactions = {}
        
        try:
            # Test causality pairs with different lags
            for predictor, target in self.causality_pairs:
                if predictor in features and target in features:
                    predictor_val = features.get(predictor, 0.0)
                    target_val = features.get(target, 0.0)
                    
                    if isinstance(predictor_val, (int, float)) and isinstance(target_val, (int, float)):
                        # Test different lag combinations
                        for lag in self.causality_test_lags:
                            # Predictor(t-lag) * Target(t) - tests if predictor leads target
                            lagged_name = f"{predictor}_lag{lag}_x_{target}"
                            lagged_value = predictor_val * target_val  # Simplified for now
                            interactions[lagged_name] = lagged_value
                            
                            # Target(t-lag) * Predictor(t) - tests if target leads predictor
                            reverse_lagged_name = f"{target}_lag{lag}_x_{predictor}"
                            reverse_lagged_value = target_val * predictor_val  # Simplified for now
                            interactions[reverse_lagged_name] = reverse_lagged_value
                            
                            # Conditional lagged interactions
                            if abs(predictor_val) > self.interaction_threshold and abs(target_val) > self.interaction_threshold:
                                conditional_name = f"{predictor}_lag{lag}_conditional_x_{target}"
                                conditional_value = predictor_val * target_val * 1.5
                                interactions[conditional_name] = conditional_value
            
            return interactions
            
        except Exception as e:
            self.logger.warning(f"Error generating lagged interactions: {e}")
            return interactions

    def _generate_causality_interactions(self, features: dict[str, Any]) -> dict[str, float]:
        """Generate causality-aware interactions with specific market logic."""
        interactions = {}
        
        try:
            # Spread changes predicting volume changes (market microstructure causality)
            if "spread_liquidity_change" in features and "volume_roc" in features:
                spread_change = features.get("spread_liquidity_change", 0.0)
                volume_roc = features.get("volume_roc", 0.0)
                
                # Widening spreads often predict volume increases (liquidity search)
                spread_volume_causality = f"spread_change_predicts_volume"
                causality_value = spread_change * volume_roc
                interactions[spread_volume_causality] = causality_value
                
                # Amplify when spread is widening and volume is increasing
                if spread_change > 0 and volume_roc > 0:
                    amplified_name = f"spread_widening_volume_increase"
                    amplified_value = spread_change * volume_roc * 2.0
                    interactions[amplified_name] = amplified_value
            
            # Volume changes predicting price impact (market impact causality)
            if "volume_roc" in features and "price_impact" in features:
                volume_roc = features.get("volume_roc", 0.0)
                price_impact = features.get("price_impact", 0.0)
                
                # High volume often predicts higher price impact
                volume_impact_causality = f"volume_predicts_price_impact"
                causality_value = volume_roc * price_impact
                interactions[volume_impact_causality] = causality_value
            
            # Volatility changes predicting momentum (regime causality)
            if "realized_volatility" in features and "momentum_5" in features:
                volatility = features.get("realized_volatility", 0.0)
                momentum = features.get("momentum_5", 0.0)
                
                # High volatility often predicts momentum breakdown
                vol_momentum_causality = f"volatility_predicts_momentum"
                causality_value = volatility * momentum
                interactions[vol_momentum_causality] = causality_value
                
                # Volatility-momentum divergence (when they move in opposite directions)
                if volatility * momentum < 0:
                    divergence_name = f"volatility_momentum_divergence"
                    divergence_value = abs(volatility) * abs(momentum) * -1
                    interactions[divergence_name] = divergence_value
            
            # Momentum changes predicting liquidity stress (flow causality)
            if "momentum_5" in features and "liquidity_stress" in features:
                momentum = features.get("momentum_5", 0.0)
                liquidity_stress = features.get("liquidity_stress", 0.0)
                
                # Strong momentum often predicts liquidity stress
                momentum_liquidity_causality = f"momentum_predicts_liquidity_stress"
                causality_value = momentum * liquidity_stress
                interactions[momentum_liquidity_causality] = causality_value
            
            return interactions
            
        except Exception as e:
            self.logger.warning(f"Error generating causality interactions: {e}")
            return interactions

    def _generate_spread_volume_interactions(self, features: dict[str, Any]) -> dict[str, float]:
        """Generate spread-volume interaction terms."""
        interactions = {}
        
        try:
            # Get available spread and volume features
            available_spreads = [f for f in self.spread_features if f in features]
            available_volumes = [f for f in self.volume_features if f in features]
            
            if not available_spreads or not available_volumes:
                return interactions
            
            # Create spread-volume interactions
            for spread_feature in available_spreads:
                for volume_feature in available_volumes:
                    spread_val = features.get(spread_feature, 0.0)
                    volume_val = features.get(volume_feature, 0.0)
                    
                    if isinstance(spread_val, (int, float)) and isinstance(volume_val, (int, float)):
                        # Main interaction: spread * volume_roc
                        interaction_name = f"{spread_feature}_x_{volume_feature}"
                        interaction_value = spread_val * volume_val
                        interactions[interaction_name] = interaction_value
                        
                        # Additional interaction: spread * volume_roc * volatility (if available)
                        volatility_features = [f for f in self.volatility_features if f in features]
                        if volatility_features:
                            vol_feature = volatility_features[0]  # Use first available
                            vol_val = features.get(vol_feature, 0.0)
                            if isinstance(vol_val, (int, float)):
                                triple_interaction_name = f"{spread_feature}_x_{volume_feature}_x_{vol_feature}"
                                triple_interaction_value = spread_val * volume_val * vol_val
                                interactions[triple_interaction_name] = triple_interaction_value
                        
                        # Ratio interaction: spread / volume (when volume is significant)
                        if abs(volume_val) > 1e-6:
                            ratio_name = f"{spread_feature}_div_{volume_feature}"
                            ratio_value = spread_val / (abs(volume_val) + 1e-8)
                            interactions[ratio_name] = ratio_value
                        
                        # Conditional interaction: spread * volume only when both are significant
                        if abs(spread_val) > self.interaction_threshold and abs(volume_val) > self.interaction_threshold:
                            conditional_name = f"{spread_feature}_conditional_x_{volume_feature}"
                            conditional_value = spread_val * volume_val * 2.0  # Amplify significant interactions
                            interactions[conditional_name] = conditional_value
            
            return interactions
            
        except Exception as e:
            self.logger.warning(f"Error generating spread-volume interactions: {e}")
            return interactions

    def _generate_volatility_momentum_interactions(self, features: dict[str, Any]) -> dict[str, float]:
        """Generate volatility-momentum interaction terms."""
        interactions = {}
        
        try:
            available_volatility = [f for f in self.volatility_features if f in features]
            available_momentum = [f for f in self.momentum_features if f in features]
            
            if not available_volatility or not available_momentum:
                return interactions
            
            for vol_feature in available_volatility:
                for mom_feature in available_momentum:
                    vol_val = features.get(vol_feature, 0.0)
                    mom_val = features.get(mom_feature, 0.0)
                    
                    if isinstance(vol_val, (int, float)) and isinstance(mom_val, (int, float)):
                        # Volatility-momentum interaction
                        interaction_name = f"{vol_feature}_x_{mom_feature}"
                        interaction_value = vol_val * mom_val
                        interactions[interaction_name] = interaction_value
                        
                        # Volatility-momentum divergence (when they move in opposite directions)
                        if vol_val * mom_val < 0:
                            divergence_name = f"{vol_feature}_divergence_{mom_feature}"
                            divergence_value = abs(vol_val) * abs(mom_val) * -1  # Negative for divergence
                            interactions[divergence_name] = divergence_value
            
            return interactions
            
        except Exception as e:
            self.logger.warning(f"Error generating volatility-momentum interactions: {e}")
            return interactions

    def _generate_liquidity_pressure_interactions(self, features: dict[str, Any]) -> dict[str, float]:
        """Generate liquidity-pressure interaction terms."""
        interactions = {}
        
        try:
            available_liquidity = [f for f in self.liquidity_features if f in features]
            available_volumes = [f for f in self.volume_features if f in features]
            
            if not available_liquidity or not available_volumes:
                return interactions
            
            for liq_feature in available_liquidity:
                for vol_feature in available_volumes:
                    liq_val = features.get(liq_feature, 0.0)
                    vol_val = features.get(vol_feature, 0.0)
                    
                    if isinstance(liq_val, (int, float)) and isinstance(vol_val, (int, float)):
                        # Liquidity-pressure interaction
                        interaction_name = f"{liq_feature}_x_{vol_feature}"
                        interaction_value = liq_val * vol_val
                        interactions[interaction_name] = interaction_value
                        
                        # Liquidity stress amplification (when both are high)
                        if abs(liq_val) > self.interaction_threshold and abs(vol_val) > self.interaction_threshold:
                            stress_name = f"{liq_feature}_stress_{vol_feature}"
                            stress_value = liq_val * vol_val * 1.5  # Amplify stress conditions
                            interactions[stress_name] = stress_value
            
            return interactions
            
        except Exception as e:
            self.logger.warning(f"Error generating liquidity-pressure interactions: {e}")
            return interactions

    def _generate_cross_regime_interactions(self, features: dict[str, Any]) -> dict[str, float]:
        """Generate cross-regime interaction terms."""
        interactions = {}
        
        try:
            # Look for regime-related features
            regime_features = [f for f in features.keys() if 'regime' in f.lower()]
            z_score_features = [f for f in features.keys() if 'z_score' in f.lower()]
            
            if not regime_features or not z_score_features:
                return interactions
            
            for regime_feature in regime_features:
                for z_score_feature in z_score_features:
                    regime_val = features.get(regime_feature, 0.0)
                    z_score_val = features.get(z_score_feature, 0.0)
                    
                    if isinstance(regime_val, (int, float)) and isinstance(z_score_val, (int, float)):
                        # Regime-zscore interaction
                        interaction_name = f"{regime_feature}_x_{z_score_feature}"
                        interaction_value = regime_val * z_score_val
                        interactions[interaction_name] = interaction_value
                        
                        # Extreme regime conditions
                        if abs(z_score_val) > 2.0:  # More than 2 standard deviations
                            extreme_name = f"{regime_feature}_extreme_{z_score_feature}"
                            extreme_value = regime_val * z_score_val * 2.0  # Amplify extreme conditions
                            interactions[extreme_name] = extreme_value
            
            return interactions
            
        except Exception as e:
            self.logger.warning(f"Error generating cross-regime interactions: {e}")
            return interactions

    def _filter_significant_interactions(self, features: dict[str, Any]) -> dict[str, Any]:
        """Filter interactions based on significance threshold."""
        try:
            if len(features) <= self.max_interactions:
                return features
            
            # Separate original features from interactions
            original_features = {}
            interaction_features = {}
            
            for key, value in features.items():
                if '_x_' in key or '_div_' in key or '_conditional_' in key or '_divergence_' in key or '_stress_' in key or '_extreme_' in key or '_lag' in key or '_predicts_' in key:
                    interaction_features[key] = value
                else:
                    original_features[key] = value
            
            # Sort interactions by absolute value
            sorted_interactions = sorted(
                interaction_features.items(),
                key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
                reverse=True
            )
            
            # Keep top interactions
            max_interaction_count = self.max_interactions - len(original_features)
            selected_interactions = dict(sorted_interactions[:max_interaction_count])
            
            # Combine original features with selected interactions
            filtered_features = {**original_features, **selected_interactions}
            
            return filtered_features
            
        except Exception as e:
            self.logger.warning(f"Error filtering interactions: {e}")
            return features

    def print(self, message: str) -> None:
        """Print message with proper formatting."""
        print(message)


class AdvancedFeatureEngineering:
    """
    Advanced feature engineering with market microstructure analysis,
    regime detection, and adaptive indicators for improved performance.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("AdvancedFeatureEngineering")

        # Configuration
        self.feature_config = config.get("advanced_features", {})
        self.enable_volatility_modeling = self.feature_config.get(
            "enable_volatility_regime_modeling",
            True,
        )
        self.enable_correlation_analysis = self.feature_config.get(
            "enable_correlation_analysis",
            True,
        )
        self.enable_momentum_analysis = self.feature_config.get(
            "enable_momentum_analysis",
            True,
        )
        self.enable_liquidity_analysis = self.feature_config.get(
            "enable_liquidity_analysis",
            True,
        )
        self.enable_candlestick_patterns = self.feature_config.get(
            "enable_candlestick_patterns",
            True,
        )
        self.enable_multi_timeframe = self.feature_config.get(
            "enable_multi_timeframe",
            True,
        )
        self.enable_meta_labeling = self.feature_config.get(
            "enable_meta_labeling",
            True,
        )

        # Multi-timeframe configuration
        self.timeframes = ["1m", "5m", "15m", "30m"]
        self.timeframe_features = {}

        # Meta-labeling configuration
        self.analyst_labels = [
            "STRONG_TREND_CONTINUATION",
            "EXHAUSTION_REVERSAL",
            "RANGE_MEAN_REVERSION",
            "BREAKOUT_FAILURE",
            "BREAKOUT_SUCCESS",
            "NO_SETUP",
            "VOLATILITY_COMPRESSION",
            "ABSORPTION_AT_LEVEL",
            "FLAG_FORMATION",
            "TRENDING_RANGE",
            "MOVING_AVERAGE_BOUNCE",
            "HEAD_AND_SHOULDERS",
            "DOUBLE_TOP_BOTTOM",
            "CLIMACTIC_REVERSAL",
            "VOLATILITY_EXPANSION",
            "MOMENTUM_IGNITION",
            "GRADUAL_MOMENTUM_FADE",
            "TRIANGLE_FORMATION",
            "RECTANGLE_FORMATION",
            "LIQUIDITY_GRAB",
        ]

        self.tactician_labels = [
            "LOWEST_PRICE_NEXT_1m",
            "HIGHEST_PRICE_NEXT_1m",
            "LIMIT_ORDER_RETURN",
            "VWAP_REVERSION_ENTRY",
            "MARKET_ORDER_NOW",
            "CHASE_MICRO_BREAKOUT",
            "MAX_ADVERSE_EXCURSION_RETURN",
            "ORDERBOOK_IMBALANCE_FLIP",
            "AGGRESSIVE_TAKER_SPIKE",
            "ABORT_ENTRY_SIGNAL",
        ]

        # Initialize components
        self.volatility_model = None
        self.correlation_analyzer = None
        self.momentum_analyzer = None
        self.liquidity_analyzer = None
        self.candlestick_analyzer = None
        self.feature_interaction_engine = FeatureInteractionEngine(config)

        self.is_initialized = False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="advanced feature engineering initialization",
    )
    async def initialize(self) -> bool:
        """Initialize advanced feature engineering components."""
        try:
            self.logger.info("🚀 Initializing advanced feature engineering...")

            # Initialize volatility modeling
            if self.enable_volatility_modeling:
                self.volatility_model = VolatilityRegimeModel(self.config)
                await self.volatility_model.initialize()

            # Initialize correlation analysis
            if self.enable_correlation_analysis:
                self.correlation_analyzer = CorrelationAnalyzer(self.config)
                await self.correlation_analyzer.initialize()

            # Initialize momentum analysis
            if self.enable_momentum_analysis:
                self.momentum_analyzer = MomentumAnalyzer(self.config)
                await self.momentum_analyzer.initialize()

            # Initialize liquidity analysis
            if self.enable_liquidity_analysis:
                self.liquidity_analyzer = LiquidityAnalyzer(self.config)
                await self.liquidity_analyzer.initialize()

            # Initialize candlestick pattern analyzer
            if self.enable_candlestick_patterns:
                self.candlestick_analyzer = CandlestickPatternAnalyzer(self.config)
                await self.candlestick_analyzer.initialize()

            # Initialize feature interaction engine
            await self.feature_interaction_engine.initialize()

            # Initialize meta-labeling system
            if self.enable_meta_labeling:
                from src.analyst.meta_labeling_system import MetaLabelingSystem

                self.meta_labeling_system = MetaLabelingSystem(self.config)
                await self.meta_labeling_system.initialize()

            self.is_initialized = True
            self.logger.info("✅ Advanced feature engineering initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Error initializing advanced feature engineering: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="advanced feature engineering",
    )
    async def engineer_features(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Engineer advanced features for improved prediction accuracy.

        Args:
            price_data: OHLCV price data
            volume_data: Volume and trade flow data
            order_flow_data: Order book and flow data (optional)

        Returns:
            Dictionary containing engineered features
        """
        try:
            if not self.is_initialized:
                self.print(
                    initialization_error("Advanced feature engineering not initialized")
                )
                return {}

            features = {}

            # Market microstructure features
            microstructure_features = await self._engineer_microstructure_features(
                price_data,
                volume_data,
                order_flow_data,
            )
            features.update(microstructure_features)

            # Volatility regime features
            if self.volatility_model:
                volatility_features = await self.volatility_model.model_volatility(
                    price_data,
                )
                features.update(volatility_features)

            # Correlation analysis features
            if self.correlation_analyzer:
                correlation_features = (
                    await self.correlation_analyzer.analyze_correlations(price_data)
                )
                features.update(correlation_features)

            # Momentum analysis features
            if self.momentum_analyzer:
                momentum_features = await self.momentum_analyzer.analyze_momentum(
                    price_data,
                )
                features.update(momentum_features)

            # Liquidity analysis features
            if self.liquidity_analyzer:
                liquidity_features = await self.liquidity_analyzer.analyze_liquidity(
                    price_data,
                    volume_data,
                    order_flow_data,
                )
                features.update(liquidity_features)

            # Candlestick pattern features
            if self.candlestick_analyzer:
                candlestick_features = await self.candlestick_analyzer.analyze_patterns(
                    price_data,
                )
                features.update(candlestick_features)

            # Adaptive indicators
            adaptive_features = self._engineer_adaptive_indicators(price_data)
            features.update(adaptive_features)

            # Feature selection and dimensionality reduction
            selected_features = self._select_optimal_features(features)

            # Add multi-timeframe features if enabled
            if self.enable_multi_timeframe:
                multi_timeframe_features = (
                    await self._engineer_multi_timeframe_features(
                        price_data,
                        volume_data,
                        order_flow_data,
                    )
                )
                selected_features.update(multi_timeframe_features)

            # Add meta-labeling if enabled
            if self.enable_meta_labeling:
                meta_labels = await self._generate_meta_labels(
                    price_data,
                    volume_data,
                    order_flow_data,
                )
                selected_features.update(meta_labels)

            # Generate feature interactions
            interaction_features = await self.feature_interaction_engine.generate_interactions(selected_features)
            original_feature_count = len(selected_features)
            selected_features.update(interaction_features)
            interaction_count = len(selected_features) - original_feature_count

            self.logger.info(
                f"✅ Engineered {len(selected_features)} advanced features (including {interaction_count} interaction terms)",
            )
            try:
                self.logger.info(f"🧾 Feature list ({len(selected_features)}): {sorted(list(selected_features.keys()))}")
            except Exception as e:
                self.logger.warning(f"Failed to log feature list: {e}")
            return selected_features

        except Exception as e:
            self.print(error("Error engineering advanced features: {e}"))
            return {}

    async def _engineer_multi_timeframe_features(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Engineer features across multiple timeframes (1m, 5m, 15m, 30m).

        Args:
            price_data: OHLCV price data
            volume_data: Volume data
            order_flow_data: Order flow data (optional)

        Returns:
            Dictionary containing multi-timeframe features
        """
        try:
            features = {}

            # Resample data to different timeframes
            for timeframe in self.timeframes:
                tf_features = await self._calculate_timeframe_features(
                    price_data,
                    volume_data,
                    order_flow_data,
                    timeframe,
                )
                features.update(tf_features)

            self.logger.info(
                f"✅ Engineered multi-timeframe features for {len(self.timeframes)} timeframes",
            )
            return features

        except Exception as e:
            self.print(error("Error engineering multi-timeframe features: {e}"))
            return {}

    async def _calculate_timeframe_features(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None,
        timeframe: str,
    ) -> dict[str, Any]:
        """Calculate features for a specific timeframe."""
        try:
            # Resample data to timeframe
            resampled_price = self._resample_to_timeframe(price_data, timeframe)
            resampled_volume = self._resample_to_timeframe(volume_data, timeframe)

            # Calculate timeframe-specific features
            features = {}

            # Technical indicators for this timeframe
            features.update(
                self._calculate_technical_indicators(resampled_price, timeframe),
            )

            # Volume analysis for this timeframe
            features.update(
                self._calculate_volume_analysis(resampled_volume, timeframe),
            )

            # Volatility analysis for this timeframe
            features.update(
                self._calculate_volatility_analysis(resampled_price, timeframe),
            )

            # Momentum analysis for this timeframe
            features.update(
                self._calculate_momentum_analysis(resampled_price, timeframe),
            )

            return features

        except Exception as e:
            self.print(error("Error calculating {timeframe} features: {e}"))
            return {}

    def _resample_to_timeframe(
        self,
        data: pd.DataFrame,
        timeframe: str,
    ) -> pd.DataFrame:
        """Resample data to specified timeframe."""
        try:
            # Convert timeframe string to pandas offset
            timeframe_map = {"1m": "1T", "5m": "5T", "15m": "15T", "30m": "30T"}

            offset = timeframe_map.get(timeframe, "1T")

            # Resample OHLCV data
            if (
                "open" in data.columns
                and "high" in data.columns
                and "low" in data.columns
                and "close" in data.columns
            ):
                resampled = data.resample(offset).agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    },
                )
            else:
                # For other data types, use mean
                resampled = data.resample(offset).mean()

            return resampled.dropna()

        except Exception as e:
            self.print(error("Error resampling to {timeframe}: {e}"))
            return data

    def _calculate_technical_indicators(
        self,
        data: pd.DataFrame,
        timeframe: str,
    ) -> dict[str, float]:
        """Calculate technical indicators for a specific timeframe."""
        try:
            features = {}

            # Moving averages
            features[f"{timeframe}_sma_20"] = (
                data["close"].rolling(20).mean().iloc[-1]
                if len(data) >= 20
                else data["close"].iloc[-1]
            )
            features[f"{timeframe}_ema_12"] = data["close"].ewm(span=12).mean().iloc[-1]
            features[f"{timeframe}_ema_26"] = data["close"].ewm(span=26).mean().iloc[-1]

            # RSI
            delta = data["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features[f"{timeframe}_rsi"] = (
                (100 - (100 / (1 + rs))).iloc[-1] if not rs.empty else 50
            )

            # MACD
            ema12 = data["close"].ewm(span=12).mean()
            ema26 = data["close"].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            features[f"{timeframe}_macd"] = macd.iloc[-1] if not macd.empty else 0
            features[f"{timeframe}_macd_signal"] = (
                signal.iloc[-1] if not signal.empty else 0
            )
            features[f"{timeframe}_macd_histogram"] = (
                (macd - signal).iloc[-1] if not macd.empty else 0
            )

            # Bollinger Bands
            sma20 = data["close"].rolling(20).mean()
            std20 = data["close"].rolling(20).std()
            features[f"{timeframe}_bb_upper"] = (
                (sma20 + (std20 * 2)).iloc[-1]
                if not sma20.empty
                else data["close"].iloc[-1]
            )
            features[f"{timeframe}_bb_lower"] = (
                (sma20 - (std20 * 2)).iloc[-1]
                if not sma20.empty
                else data["close"].iloc[-1]
            )
            features[f"{timeframe}_bb_position"] = (
                (
                    (data["close"].iloc[-1] - features[f"{timeframe}_bb_lower"])
                    / (
                        features[f"{timeframe}_bb_upper"]
                        - features[f"{timeframe}_bb_lower"]
                    )
                )
                if features[f"{timeframe}_bb_upper"]
                != features[f"{timeframe}_bb_lower"]
                else 0.5
            )

            return features

        except Exception as e:
            self.logger.exception(
                f"Error calculating technical indicators for {timeframe}: {e}",
            )
            return {}

    def _calculate_volume_analysis(
        self,
        data: pd.DataFrame,
        timeframe: str,
    ) -> dict[str, float]:
        """Calculate volume analysis for a specific timeframe."""
        try:
            features = {}

            if "volume" in data.columns:
                features[f"{timeframe}_volume_sma"] = (
                    data["volume"].rolling(20).mean().iloc[-1]
                    if len(data) >= 20
                    else data["volume"].iloc[-1]
                )
                features[f"{timeframe}_volume_ratio"] = (
                    data["volume"].iloc[-1] / features[f"{timeframe}_volume_sma"]
                    if features[f"{timeframe}_volume_sma"] > 0
                    else 1.0
                )
                features[f"{timeframe}_volume_trend"] = (
                    data["volume"].rolling(10).mean().diff().iloc[-1]
                    if len(data) >= 10
                    else 0
                )

            return features

        except Exception as e:
            self.logger.exception(
                f"Error calculating volume analysis for {timeframe}: {e}",
            )
            return {}

    def _calculate_volatility_analysis(
        self,
        data: pd.DataFrame,
        timeframe: str,
    ) -> dict[str, float]:
        """Calculate volatility analysis for a specific timeframe."""
        try:
            features = {}

            # ATR
            high_low = data["high"] - data["low"]
            high_close = np.abs(data["high"] - data["close"].shift())
            low_close = np.abs(data["low"] - data["close"].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            features[f"{timeframe}_atr"] = (
                true_range.rolling(14).mean().iloc[-1]
                if len(data) >= 14
                else true_range.iloc[-1]
            )

            # Volatility
            returns = data["close"].pct_change()
            features[f"{timeframe}_volatility"] = (
                returns.rolling(20).std().iloc[-1] if len(data) >= 20 else returns.std()
            )

            return features

        except Exception as e:
            self.logger.exception(
                f"Error calculating volatility analysis for {timeframe}: {e}",
            )
            return {}

    def _calculate_momentum_analysis(
        self,
        data: pd.DataFrame,
        timeframe: str,
    ) -> dict[str, float]:
        """Calculate momentum analysis for a specific timeframe."""
        try:
            features = {}

            # Price momentum
            features[f"{timeframe}_price_momentum"] = (
                data["close"].pct_change(5).iloc[-1] if len(data) >= 5 else 0
            )
            features[f"{timeframe}_price_acceleration"] = (
                data["close"].pct_change(5).diff().iloc[-1] if len(data) >= 5 else 0
            )

            # Volume momentum
            if "volume" in data.columns:
                features[f"{timeframe}_volume_momentum"] = (
                    data["volume"].pct_change(5).iloc[-1] if len(data) >= 5 else 0
                )

            return features

        except Exception as e:
            self.logger.exception(
                f"Error calculating momentum analysis for {timeframe}: {e}",
            )
            return {}

    async def _generate_meta_labels(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Generate path-dependent meta-labels for analyst and tactician models.

        Args:
            price_data: OHLCV price data
            volume_data: Volume data
            order_flow_data: Order flow data (optional)

        Returns:
            Dictionary containing meta-labels
        """
        try:
            labels = {}

            # Generate analyst labels (setup model)
            analyst_labels = await self._generate_analyst_labels(
                price_data,
                volume_data,
                order_flow_data,
            )
            labels.update(analyst_labels)

            # Generate tactician labels (entry model)
            tactician_labels = await self._generate_tactician_labels(
                price_data,
                volume_data,
                order_flow_data,
            )
            labels.update(tactician_labels)

            self.logger.info(f"✅ Generated {len(labels)} meta-labels")
            return labels

        except Exception as e:
            self.print(error("Error generating meta-labels: {e}"))
            return {}

    async def _generate_analyst_labels(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Generate analyst labels for setup identification."""
        try:
            if hasattr(self, "meta_labeling_system") and self.meta_labeling_system:
                # Use the meta-labeling system for pattern detection
                pattern_features = (
                    await self.meta_labeling_system._calculate_pattern_features(
                        price_data,
                        volume_data,
                    )
                )

                labels = {}
                labels.update(
                    self.meta_labeling_system._detect_strong_trend_continuation(
                        price_data,
                        pattern_features,
                    ),
                )
                labels.update(
                    self.meta_labeling_system._detect_exhaustion_reversal(
                        price_data,
                        pattern_features,
                    ),
                )
                labels.update(
                    self.meta_labeling_system._detect_range_mean_reversion(
                        price_data,
                        pattern_features,
                    ),
                )
                labels.update(
                    self.meta_labeling_system._detect_breakout_patterns(
                        price_data,
                        pattern_features,
                    ),
                )
                labels.update(
                    self.meta_labeling_system._detect_volatility_patterns(
                        price_data,
                        pattern_features,
                    ),
                )
                labels.update(
                    self.meta_labeling_system._detect_chart_patterns(
                        price_data,
                        pattern_features,
                    ),
                )
                labels.update(
                    self.meta_labeling_system._detect_momentum_patterns(
                        price_data,
                        pattern_features,
                    ),
                )

                # Add NO_SETUP if no other patterns detected
                if not any(labels.values()):
                    labels.update(self.meta_labeling_system.generate_no_setup_label())

                return labels
            # Fallback to basic labels
            return {"NO_SETUP": 1}

        except Exception as e:
            self.print(error("Error generating analyst labels: {e}"))
            return {"NO_SETUP": 1}

    async def _generate_tactician_labels(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Generate tactician labels for entry optimization."""
        try:
            if hasattr(self, "meta_labeling_system") and self.meta_labeling_system:
                # Use the meta-labeling system for entry prediction
                entry_features = (
                    await self.meta_labeling_system._calculate_entry_features(
                        price_data,
                        volume_data,
                        order_flow_data,
                    )
                )
                pattern_features = (
                    await self.meta_labeling_system._calculate_pattern_features(
                        price_data,
                        volume_data,
                    )
                )

                labels = {}
                labels.update(
                    self.meta_labeling_system._predict_price_extremes(
                        price_data,
                        entry_features,
                    ),
                )
                labels.update(
                    self.meta_labeling_system._predict_order_returns(
                        price_data,
                        entry_features,
                    ),
                )
                labels.update(
                    self.meta_labeling_system._detect_entry_signals(
                        price_data,
                        volume_data,
                        order_flow_data,
                    ),
                )
                labels.update(
                    self.meta_labeling_system._predict_adverse_excursion(
                        price_data,
                        entry_features,
                    ),
                )
                labels.update(
                    self.meta_labeling_system.generate_abort_entry_signal(
                        pattern_features,
                    ),
                )

                return labels
            # Fallback to basic labels
            return {
                "LOWEST_PRICE_NEXT_1m": price_data["close"].iloc[-1],
                "HIGHEST_PRICE_NEXT_1m": price_data["close"].iloc[-1],
                "LIMIT_ORDER_RETURN": 0.001,
                "ABORT_ENTRY_SIGNAL": 0,
            }

        except Exception as e:
            self.print(error("Error generating tactician labels: {e}"))
            return {
                "LOWEST_PRICE_NEXT_1m": price_data["close"].iloc[-1],
                "HIGHEST_PRICE_NEXT_1m": price_data["close"].iloc[-1],
                "LIMIT_ORDER_RETURN": 0.001,
                "ABORT_ENTRY_SIGNAL": 0,
            }

    async def _engineer_microstructure_features(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Engineer market microstructure features."""
        try:
            features = {}

            # Price impact analysis
            features.update(self._calculate_price_impact(price_data, volume_data))

            # Order flow imbalance
            if order_flow_data is not None:
                features.update(self._calculate_order_flow_imbalance(order_flow_data))

            # Volume profile analysis
            features.update(self._calculate_volume_profile(price_data, volume_data))

            # Bid-ask spread analysis
            if order_flow_data is not None:
                features.update(self._calculate_spread_analysis(order_flow_data))

            # Market depth analysis
            if order_flow_data is not None:
                features.update(self._calculate_market_depth(order_flow_data))

            return features

        except Exception as e:
            self.print(error("Error engineering microstructure features: {e}"))
            return {}

    def _calculate_price_impact(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate price impact metrics."""
        try:
            # Calculate price changes
            price_changes = price_data["close"].pct_change()

            # Calculate volume-weighted price impact
            volume_weighted_impact = (
                (price_changes * volume_data["volume"]).rolling(20).mean()
            )

            # Calculate Kyle's lambda (price impact parameter)
            kyle_lambda = (
                np.abs(price_changes).rolling(50).mean()
                / volume_data["volume"].rolling(50).mean()
            )

            # Calculate Amihud illiquidity measure
            amihud_illiquidity = np.abs(price_changes) / volume_data["volume"]
            amihud_illiquidity = amihud_illiquidity.rolling(20).mean()

            return {
                "price_impact": volume_weighted_impact.iloc[-1]
                if not volume_weighted_impact.empty
                else 0.0,
                "kyle_lambda": kyle_lambda.iloc[-1] if not kyle_lambda.empty else 0.0,
                "amihud_illiquidity": amihud_illiquidity.iloc[-1]
                if not amihud_illiquidity.empty
                else 0.0,
            }

        except Exception as e:
            self.print(error("Error calculating price impact: {e}"))
            return {}

    def _calculate_order_flow_imbalance(
        self,
        order_flow_data: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate order flow imbalance metrics."""
        try:
            # Calculate buy/sell pressure
            buy_volume = order_flow_data.get("buy_volume", pd.Series(0))
            sell_volume = order_flow_data.get("sell_volume", pd.Series(0))

            # Order flow imbalance
            total_volume = buy_volume + sell_volume
            imbalance = (buy_volume - sell_volume) / total_volume
            imbalance = imbalance.rolling(20).mean()

            # Large order detection
            avg_volume = total_volume.rolling(50).mean()
            large_order_ratio = (total_volume > 2 * avg_volume).rolling(20).mean()

            return {
                "order_flow_imbalance": imbalance.iloc[-1]
                if not imbalance.empty
                else 0.0,
                "large_order_ratio": large_order_ratio.iloc[-1]
                if not large_order_ratio.empty
                else 0.0,
            }

        except Exception as e:
            self.print(error("Error calculating order flow imbalance: {e}"))
            return {}

    def _calculate_volume_profile(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate volume profile metrics."""
        try:
            # Volume-weighted average price (VWAP)
            vwap = (price_data["close"] * volume_data["volume"]).rolling(
                20,
            ).sum() / volume_data["volume"].rolling(20).sum()

            # Volume price trend (VPT)
            vpt = (volume_data["volume"] * price_data["close"].pct_change()).cumsum()

            # Volume rate of change
            volume_roc = volume_data["volume"].pct_change(5)

            # Volume moving average ratio
            volume_ma_ratio = (
                volume_data["volume"] / volume_data["volume"].rolling(20).mean()
            )

            return {
                "vwap": vwap.iloc[-1]
                if not vwap.empty
                else price_data["close"].iloc[-1],
                "vpt": vpt.iloc[-1] if not vpt.empty else 0.0,
                "volume_roc": volume_roc.iloc[-1] if not volume_roc.empty else 0.0,
                "volume_ma_ratio": volume_ma_ratio.iloc[-1]
                if not volume_ma_ratio.empty
                else 1.0,
            }

        except Exception as e:
            self.print(error("Error calculating volume profile: {e}"))
            return {}

    def _engineer_adaptive_indicators(
        self,
        price_data: pd.DataFrame,
    ) -> dict[str, float]:
        """Engineer adaptive technical indicators."""
        try:
            features = {}

            # Adaptive moving averages
            features.update(self._calculate_adaptive_moving_averages(price_data))

            # Adaptive RSI
            features.update(self._calculate_adaptive_rsi(price_data))

            # Adaptive Bollinger Bands
            features.update(self._calculate_adaptive_bollinger_bands(price_data))

            # Adaptive MACD
            features.update(self._calculate_adaptive_macd(price_data))

            return features

        except Exception as e:
            self.print(error("Error engineering adaptive indicators: {e}"))
            return {}

    def _calculate_adaptive_moving_averages(
        self,
        price_data: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate adaptive moving averages based on volatility."""
        try:
            # Calculate volatility
            returns = price_data["close"].pct_change()
            volatility = returns.rolling(20).std()

            # Adaptive periods based on volatility
            base_period = 20
            volatility_factor = volatility / volatility.rolling(100).mean()
            adaptive_period = (base_period * volatility_factor).clip(5, 50)

            # Adaptive SMA
            adaptive_sma = (
                price_data["close"].rolling(window=adaptive_period.astype(int)).mean()
            )

            # Adaptive EMA
            adaptive_alpha = 2 / (adaptive_period + 1)
            adaptive_ema = price_data["close"].ewm(alpha=adaptive_alpha).mean()

            return {
                "adaptive_sma": adaptive_sma.iloc[-1]
                if not adaptive_sma.empty
                else price_data["close"].iloc[-1],
                "adaptive_ema": adaptive_ema.iloc[-1]
                if not adaptive_ema.empty
                else price_data["close"].iloc[-1],
                "adaptive_period": adaptive_period.iloc[-1]
                if not adaptive_period.empty
                else base_period,
            }

        except Exception as e:
            self.print(error("Error calculating adaptive moving averages: {e}"))
            return {}

    def _select_optimal_features(self, features: dict[str, Any]) -> dict[str, float]:
        """Select optimal features using feature importance and correlation analysis."""
        try:
            # Convert to DataFrame for analysis
            feature_df = pd.DataFrame([features])

            # Remove NaN values
            feature_df = feature_df.dropna(axis=1)

            # Remove constant features
            feature_df = feature_df.loc[:, feature_df.std() > 0]

            # Remove highly correlated features
            if len(feature_df.columns) > 1:
                correlation_matrix = feature_df.corr()
                upper_triangle = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                high_correlation = np.abs(correlation_matrix) > 0.95
                high_correlation = high_correlation & upper_triangle

                to_drop = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i + 1, len(correlation_matrix.columns)):
                        if high_correlation.iloc[i, j]:
                            to_drop.append(correlation_matrix.columns[j])

                feature_df = feature_df.drop(columns=list(set(to_drop)))

            return feature_df.iloc[0].to_dict()

        except Exception as e:
            self.logger.error(f"Error selecting optimal features: {e}")
            return features

    def _calculate_adaptive_rsi(self, price_data: pd.DataFrame) -> dict[str, float]:
        """Calculate an adaptive RSI using standard RSI with volatility-aware smoothing."""
        try:
            close = price_data["close"].astype(float)
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = (100 - (100 / (1 + rs))).fillna(50)
            # Volatility-aware smoothing
            vol = close.pct_change().rolling(20, min_periods=1).std().fillna(0)
            weight = (1 / (1 + vol * 100)).clip(0, 1)
            smoothed = (weight * rsi.rolling(5, min_periods=1).mean() + (1 - weight) * rsi.rolling(20, min_periods=1).mean()).fillna(method="ffill").fillna(50)
            return {"adaptive_rsi": float(smoothed.iloc[-1])}
        except Exception as e:
            self.logger.warning(f"Failed to calculate adaptive RSI: {e}")
            return {"adaptive_rsi": 50.0}

    def _calculate_adaptive_bollinger_bands(self, price_data: pd.DataFrame) -> dict[str, float]:
        """Calculate adaptive Bollinger Bands and position based on volatility."""
        try:
            close = price_data["close"].astype(float)
            sma = close.rolling(20, min_periods=1).mean()
            std = close.rolling(20, min_periods=1).std().fillna(0)
            # Adjust band width by recent volatility percentile
            vol = close.pct_change().rolling(20, min_periods=1).std().fillna(0)
            vol_pct = (vol.rank(pct=True)).fillna(0.5)
            width_mult = 1.5 + vol_pct  # 1.5..2.5x
            upper = sma + width_mult * std
            lower = sma - width_mult * std
            denom = (upper - lower).replace(0, np.nan)
            pos = ((close - lower) / denom).clip(0, 1).fillna(0.5)
            return {
                "adaptive_bb_upper": float(upper.iloc[-1] if not upper.empty else close.iloc[-1]),
                "adaptive_bb_lower": float(lower.iloc[-1] if not lower.empty else close.iloc[-1]),
                "adaptive_bb_position": float(pos.iloc[-1] if not pos.empty else 0.5),
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate adaptive Bollinger Bands: {e}")
            return {"adaptive_bb_upper": float("nan"), "adaptive_bb_lower": float("nan"), "adaptive_bb_position": 0.5}

    def _calculate_adaptive_macd(self, price_data: pd.DataFrame) -> dict[str, float]:
        """Calculate an adaptive MACD with volatility-aware blending of spans."""
        try:
            close = price_data["close"].astype(float)
            # Standard MACD components
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            hist = macd - signal
            # Volatility-aware adjustment: blend with a slower set during high vol
            ema_fast = close.ewm(span=8, adjust=False).mean()
            ema_slow = close.ewm(span=34, adjust=False).mean()
            macd_alt = ema_fast - ema_slow
            signal_alt = macd_alt.ewm(span=9, adjust=False).mean()
            vol = close.pct_change().rolling(20, min_periods=1).std().fillna(0)
            alpha = (1 / (1 + vol * 100)).clip(0, 1)  # more weight to slower during high vol
            macd_adapt = (alpha * macd + (1 - alpha) * macd_alt).iloc[-1]
            signal_adapt = (alpha * signal + (1 - alpha) * signal_alt).iloc[-1]
            hist_adapt = macd_adapt - signal_adapt
            return {
                "adaptive_macd": float(macd_adapt),
                "adaptive_macd_signal": float(signal_adapt),
                "adaptive_macd_histogram": float(hist_adapt),
            }
        except Exception as e:
            self.logger.warning(f"Failed to calculate adaptive MACD: {e}")
            return {"adaptive_macd": 0.0, "adaptive_macd_signal": 0.0, "adaptive_macd_histogram": 0.0}


class VolatilityRegimeModel:
    """Model volatility regimes using GARCH and other methods."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VolatilityRegimeModel")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize volatility model."""
        try:
            self.is_initialized = True
            return True
        except Exception as e:
            self.print(initialization_error("Error initializing volatility model: {e}"))
            return False

    async def model_volatility(self, price_data: pd.DataFrame) -> dict[str, float]:
        """Model volatility regimes."""
        try:
            returns = price_data["close"].pct_change().dropna()

            # Calculate various volatility measures
            realized_vol = returns.rolling(20).std()
            parkinson_vol = self._calculate_parkinson_volatility(price_data)
            garman_klass_vol = self._calculate_garman_klass_volatility(price_data)

            # Volatility regime classification
            vol_percentile = realized_vol.rank(pct=True).iloc[-1]

            if vol_percentile > 0.8:
                vol_regime = "high"
            elif vol_percentile < 0.2:
                vol_regime = "low"
            else:
                vol_regime = "medium"

            return {
                "realized_volatility": realized_vol.iloc[-1]
                if not realized_vol.empty
                else 0.0,
                "parkinson_volatility": parkinson_vol.iloc[-1]
                if not parkinson_vol.empty
                else 0.0,
                "garman_klass_volatility": garman_klass_vol.iloc[-1]
                if not garman_klass_vol.empty
                else 0.0,
                "volatility_regime": vol_regime,
                "volatility_percentile": vol_percentile,
            }

        except Exception as e:
            self.print(error("Error modeling volatility: {e}"))
            return {}

    def _calculate_parkinson_volatility(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate Parkinson volatility estimator."""
        try:
            high_low_ratio = np.log(price_data["high"] / price_data["low"]) ** 2
            parkinson_vol = np.sqrt(high_low_ratio / (4 * np.log(2)))
            return parkinson_vol.rolling(20).mean()
        except Exception:
            return pd.Series()

    def _calculate_garman_klass_volatility(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate Garman-Klass volatility estimator."""
        try:
            c = np.log(price_data["close"] / price_data["close"].shift(1))
            h = np.log(price_data["high"] / price_data["close"].shift(1))
            l = np.log(price_data["low"] / price_data["close"].shift(1))

            gk_vol = np.sqrt(0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * c**2)
            return gk_vol.rolling(20).mean()
        except Exception:
            return pd.Series()


class CorrelationAnalyzer:
    """Analyze correlations between different assets and timeframes."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("CorrelationAnalyzer")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize correlation analyzer."""
        try:
            self.is_initialized = True
            return True
        except Exception as e:
            self.print(
                initialization_error("Error initializing correlation analyzer: {e}")
            )
            return False

    async def analyze_correlations(self, price_data: pd.DataFrame) -> dict[str, float]:
        """Analyze correlations."""
        try:
            returns = price_data["close"].pct_change().dropna()

            # Rolling correlations
            corr_5 = returns.rolling(5).corr(returns.shift(1))
            corr_20 = returns.rolling(20).corr(returns.shift(1))

            # Cross-timeframe correlations
            returns_5m = returns.resample("5T").last()
            returns_1h = returns.resample("1H").last()

            cross_corr = (
                returns_5m.corr(returns_1h)
                if len(returns_5m) > 1 and len(returns_1h) > 1
                else 0.0
            )

            return {
                "autocorrelation_5": corr_5.iloc[-1] if not corr_5.empty else 0.0,
                "autocorrelation_20": corr_20.iloc[-1] if not corr_20.empty else 0.0,
                "cross_timeframe_correlation": cross_corr,
            }

        except Exception as e:
            self.print(error("Error analyzing correlations: {e}"))
            return {}


class MomentumAnalyzer:
    """Analyze momentum patterns and signals."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("MomentumAnalyzer")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize momentum analyzer."""
        try:
            self.is_initialized = True
            return True
        except Exception as e:
            self.print(
                initialization_error("Error initializing momentum analyzer: {e}")
            )
            return False

    async def analyze_momentum(self, price_data: pd.DataFrame) -> dict[str, float]:
        """Analyze momentum patterns."""
        try:
            returns = price_data["close"].pct_change().dropna()

            # Momentum indicators
            momentum_5 = returns.rolling(5).mean()
            momentum_20 = returns.rolling(20).mean()
            momentum_50 = returns.rolling(50).mean()

            # Momentum acceleration
            momentum_accel = momentum_5 - momentum_20

            # Momentum strength
            momentum_strength = momentum_5 / momentum_20.std()

            # Momentum divergence
            price_momentum = price_data["close"].pct_change(5)
            volume_momentum = (
                price_data["volume"].pct_change(5)
                if "volume" in price_data.columns
                else pd.Series(0)
            )
            momentum_divergence = price_momentum - volume_momentum

            return {
                "momentum_5": momentum_5.iloc[-1] if not momentum_5.empty else 0.0,
                "momentum_20": momentum_20.iloc[-1] if not momentum_20.empty else 0.0,
                "momentum_50": momentum_50.iloc[-1] if not momentum_50.empty else 0.0,
                "momentum_acceleration": momentum_accel.iloc[-1]
                if not momentum_accel.empty
                else 0.0,
                "momentum_strength": momentum_strength.iloc[-1]
                if not momentum_strength.empty
                else 0.0,
                "momentum_divergence": momentum_divergence.iloc[-1]
                if not momentum_divergence.empty
                else 0.0,
            }

        except Exception as e:
            self.print(error("Error analyzing momentum: {e}"))
            return {}


class LiquidityAnalyzer:
    """Analyze liquidity conditions and market depth."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("LiquidityAnalyzer")
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize liquidity analyzer."""
        try:
            self.is_initialized = True
            return True
        except Exception as e:
            self.print(
                initialization_error("Error initializing liquidity analyzer: {e}")
            )
            return False

    async def analyze_liquidity(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Analyze liquidity conditions."""
        try:
            # Volume-based liquidity measures
            avg_volume = volume_data["volume"].rolling(20).mean()
            volume_liquidity = volume_data["volume"] / avg_volume

            # Price-based liquidity measures
            price_changes = price_data["close"].pct_change()
            price_impact = np.abs(price_changes) / volume_data["volume"]
            price_impact = price_impact.rolling(20).mean()

            # Spread-based liquidity (if order flow data available)
            spread_liquidity = 0.0
            if order_flow_data is not None and "spread" in order_flow_data.columns:
                spread_liquidity = order_flow_data["spread"].rolling(20).mean().iloc[-1]

            # Liquidity regime classification
            liquidity_percentile = volume_liquidity.rank(pct=True).iloc[-1]

            if liquidity_percentile > 0.8:
                liquidity_regime = "high"
            elif liquidity_percentile < 0.2:
                liquidity_regime = "low"
            else:
                liquidity_regime = "medium"

            return {
                "volume_liquidity": volume_liquidity.iloc[-1]
                if not volume_liquidity.empty
                else 1.0,
                "price_impact": price_impact.iloc[-1]
                if not price_impact.empty
                else 0.0,
                "spread_liquidity": spread_liquidity,
                "liquidity_regime": liquidity_regime,
                "liquidity_percentile": liquidity_percentile,
            }

        except Exception as e:
            self.print(error("Error analyzing liquidity: {e}"))
            return {}
