# src/training/steps/vectorized_advanced_feature_engineering.py

"""
Vectorized Advanced Feature Engineering for enhanced financial performance.
Implements sophisticated market microstructure features, regime detection,
and adaptive indicators for improved prediction accuracy with vectorized operations.
"""

import numpy as np
import pandas as pd
import pywt
from typing import Any, Dict, List, Optional, Tuple
from datetime import timedelta

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class VectorizedCandlestickPatternAnalyzer:
    """
    Comprehensive candlestick pattern analyzer implementing all major patterns
    for enhanced feature engineering and ML model training with vectorized operations.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("VectorizedCandlestickPatternAnalyzer")

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
            self.logger.info("🚀 Initializing vectorized candlestick pattern analyzer...")
            self.is_initialized = True
            self.logger.info("✅ Vectorized candlestick pattern analyzer initialized successfully")
            return True
        except Exception as e:
            self.logger.error(
                f"❌ Error initializing vectorized candlestick pattern analyzer: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return={},
        context="candlestick pattern analysis",
    )
    async def analyze_patterns(self, price_data: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze candlestick patterns and return features for ML training using vectorized operations.

        Args:
            price_data: OHLCV price data

        Returns:
            Dictionary containing candlestick pattern features
        """
        try:
            if not self.is_initialized:
                self.logger.error("Candlestick pattern analyzer not initialized")
                return {}

            if price_data.empty or len(price_data) < 3:
                self.logger.warning("Insufficient data for pattern analysis")
                return {}

            # Prepare data with calculated metrics using vectorized operations
            df = self._prepare_candlestick_data_vectorized(price_data)

            # Analyze all patterns using vectorized operations
            patterns = {
                "engulfing_patterns": self._detect_engulfing_patterns_vectorized(df),
                "hammer_hanging_man": self._detect_hammer_hanging_man_vectorized(df),
                "shooting_star_inverted_hammer": self._detect_shooting_star_inverted_hammer_vectorized(df),
                "tweezer_patterns": self._detect_tweezer_patterns_vectorized(df),
                "marubozu_patterns": self._detect_marubozu_patterns_vectorized(df),
                "three_methods_patterns": self._detect_three_methods_patterns_vectorized(df),
                "doji_patterns": self._detect_doji_patterns_vectorized(df),
                "spinning_top_patterns": self._detect_spinning_top_patterns_vectorized(df),
            }

            # Convert patterns to ML features using vectorized operations
            features = self._convert_patterns_to_features_vectorized(patterns, df)

            self.logger.info(f"✅ Analyzed {len(patterns)} pattern categories using vectorized operations")
            return features

        except Exception as e:
            self.logger.error(f"Error analyzing candlestick patterns: {e}")
            return {}

    def _prepare_candlestick_data_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with candlestick metrics using vectorized operations."""
        try:
            df = df.copy()

            # Calculate basic candlestick metrics using vectorized operations
            df["body_size"] = np.abs(df["close"] - df["open"])
            df["upper_shadow"] = df["high"] - np.maximum(df["open"], df["close"])
            df["lower_shadow"] = np.minimum(df["open"], df["close"]) - df["low"]
            df["total_range"] = df["high"] - df["low"]
            df["body_ratio"] = df["body_size"] / df["total_range"].replace(0, 1)
            df["is_bullish"] = df["close"] > df["open"]

            # Calculate moving averages for context using vectorized operations
            df["avg_body_size"] = df["body_size"].rolling(window=20).mean()
            df["avg_range"] = df["total_range"].rolling(window=20).mean()

            return df.dropna()

        except Exception as e:
            self.logger.error(f"Error preparing candlestick data: {e}")
            return pd.DataFrame()

    def _detect_engulfing_patterns_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect bullish and bearish engulfing patterns using vectorized operations."""
        try:
            # Vectorized calculations
            current_body_size = df["body_size"].values
            previous_body_size = df["body_size"].shift(1).values
            current_is_bullish = df["is_bullish"].values
            previous_is_bullish = df["is_bullish"].shift(1).values
            current_open = df["open"].values
            current_close = df["close"].values
            previous_open = df["open"].shift(1).values
            previous_close = df["close"].shift(1).values

            # Bullish engulfing conditions
            bullish_engulfing = (
                current_is_bullish &
                ~previous_is_bullish &
                (current_open < previous_close) &
                (current_close > previous_open) &
                (current_body_size > previous_body_size * self.engulfing_ratio)
            )

            # Bearish engulfing conditions
            bearish_engulfing = (
                ~current_is_bullish &
                previous_is_bullish &
                (current_open > previous_close) &
                (current_close < previous_open) &
                (current_body_size > previous_body_size * self.engulfing_ratio)
            )

            # Calculate confidence scores
            bullish_confidence = np.where(
                bullish_engulfing,
                np.minimum(current_body_size / (previous_body_size + 1e-8), 2.0),
                0.0
            )
            bearish_confidence = np.where(
                bearish_engulfing,
                np.minimum(current_body_size / (previous_body_size + 1e-8), 2.0),
                0.0
            )

            return {
                "bullish_engulfing": bullish_engulfing,
                "bearish_engulfing": bearish_engulfing,
                "bullish_confidence": bullish_confidence,
                "bearish_confidence": bearish_confidence,
            }

        except Exception as e:
            self.logger.error(f"Error detecting engulfing patterns: {e}")
            return {}

    def _detect_hammer_hanging_man_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect hammer and hanging man patterns using vectorized operations."""
        try:
            # Vectorized calculations
            body_ratio = df["body_ratio"].values
            lower_shadow = df["lower_shadow"].values
            upper_shadow = df["upper_shadow"].values
            body_size = df["body_size"].values
            is_bullish = df["is_bullish"].values
            close_prices = df["close"].values

            # Hammer pattern conditions
            hammer_conditions = (
                (body_ratio <= self.hammer_ratio) &
                (lower_shadow > body_size * self.shadow_ratio) &
                (upper_shadow < body_size * 0.5)
            )

            # Hanging man pattern conditions (need previous close for context)
            previous_close = df["close"].shift(1).values
            hanging_man_conditions = (
                (body_ratio <= self.hammer_ratio) &
                (lower_shadow > body_size * self.shadow_ratio) &
                (upper_shadow < body_size * 0.5) &
                (previous_close > close_prices)
            )

            # Calculate confidence scores
            hammer_confidence = np.where(
                hammer_conditions,
                np.minimum(lower_shadow / (body_size + 1e-8), 3.0),
                0.0
            )
            hanging_man_confidence = np.where(
                hanging_man_conditions,
                np.minimum(lower_shadow / (body_size + 1e-8), 3.0),
                0.0
            )

            return {
                "hammer": hammer_conditions,
                "hanging_man": hanging_man_conditions,
                "hammer_confidence": hammer_confidence,
                "hanging_man_confidence": hanging_man_confidence,
            }

        except Exception as e:
            self.logger.error(f"Error detecting hammer/hanging man patterns: {e}")
            return {}

    def _detect_shooting_star_inverted_hammer_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect shooting star and inverted hammer patterns using vectorized operations."""
        try:
            # Vectorized calculations
            body_ratio = df["body_ratio"].values
            upper_shadow = df["upper_shadow"].values
            lower_shadow = df["lower_shadow"].values
            body_size = df["body_size"].values
            close_prices = df["close"].values
            previous_close = df["close"].shift(1).values

            # Shooting star pattern conditions
            shooting_star_conditions = (
                (body_ratio <= self.hammer_ratio) &
                (upper_shadow > body_size * self.shadow_ratio) &
                (lower_shadow < body_size * 0.5)
            )

            # Inverted hammer pattern conditions
            inverted_hammer_conditions = (
                (body_ratio <= self.hammer_ratio) &
                (upper_shadow > body_size * self.shadow_ratio) &
                (lower_shadow < body_size * 0.5) &
                (previous_close < close_prices)
            )

            # Calculate confidence scores
            shooting_star_confidence = np.where(
                shooting_star_conditions,
                np.minimum(upper_shadow / (body_size + 1e-8), 3.0),
                0.0
            )
            inverted_hammer_confidence = np.where(
                inverted_hammer_conditions,
                np.minimum(upper_shadow / (body_size + 1e-8), 3.0),
                0.0
            )

            return {
                "shooting_star": shooting_star_conditions,
                "inverted_hammer": inverted_hammer_conditions,
                "shooting_star_confidence": shooting_star_confidence,
                "inverted_hammer_confidence": inverted_hammer_confidence,
            }

        except Exception as e:
            self.logger.error(f"Error detecting shooting star/inverted hammer patterns: {e}")
            return {}

    def _detect_tweezer_patterns_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect tweezer tops and bottoms patterns using vectorized operations."""
        try:
            # Vectorized calculations
            current_high = df["high"].values
            current_low = df["low"].values
            current_close = df["close"].values
            current_open = df["open"].values
            previous_high = df["high"].shift(1).values
            previous_low = df["low"].shift(1).values
            previous_close = df["close"].shift(1).values
            previous_open = df["open"].shift(1).values

            # Tweezer tops conditions
            tweezer_top_conditions = (
                (np.abs(current_high - previous_high) <= self.tweezer_threshold * current_high) &
                (current_high > current_close) &
                (previous_high > previous_close)
            )

            # Tweezer bottoms conditions
            tweezer_bottom_conditions = (
                (np.abs(current_low - previous_low) <= self.tweezer_threshold * current_low) &
                (current_low < current_open) &
                (previous_low < previous_open)
            )

            # Calculate confidence scores
            tweezer_top_confidence = np.where(
                tweezer_top_conditions,
                1.0 - np.abs(current_high - previous_high) / (current_high + 1e-8),
                0.0
            )
            tweezer_bottom_confidence = np.where(
                tweezer_bottom_conditions,
                1.0 - np.abs(current_low - previous_low) / (current_low + 1e-8),
                0.0
            )

            return {
                "tweezer_top": tweezer_top_conditions,
                "tweezer_bottom": tweezer_bottom_conditions,
                "tweezer_top_confidence": tweezer_top_confidence,
                "tweezer_bottom_confidence": tweezer_bottom_confidence,
            }

        except Exception as e:
            self.logger.error(f"Error detecting tweezer patterns: {e}")
            return {}

    def _detect_marubozu_patterns_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect bullish and bearish marubozu patterns using vectorized operations."""
        try:
            # Vectorized calculations
            upper_shadow = df["upper_shadow"].values
            lower_shadow = df["lower_shadow"].values
            total_range = df["total_range"].values
            is_bullish = df["is_bullish"].values

            # Marubozu conditions (no shadows or very small shadows)
            marubozu_conditions = (
                (upper_shadow < total_range * self.marubozu_threshold) &
                (lower_shadow < total_range * self.marubozu_threshold)
            )

            # Separate bullish and bearish marubozu
            bullish_marubozu = marubozu_conditions & is_bullish
            bearish_marubozu = marubozu_conditions & ~is_bullish

            # Calculate confidence scores
            marubozu_confidence = np.where(
                marubozu_conditions,
                1.0 - (upper_shadow + lower_shadow) / (total_range + 1e-8),
                0.0
            )

            return {
                "bullish_marubozu": bullish_marubozu,
                "bearish_marubozu": bearish_marubozu,
                "marubozu_confidence": marubozu_confidence,
            }

        except Exception as e:
            self.logger.error(f"Error detecting marubozu patterns: {e}")
            return {}

    def _detect_three_methods_patterns_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect rising and falling three methods patterns using vectorized operations."""
        try:
            # This is a complex pattern that requires looking at 5 candles
            # For vectorized implementation, we'll use rolling windows
            window_size = 5
            patterns = {
                "rising_three_methods": np.zeros(len(df), dtype=bool),
                "falling_three_methods": np.zeros(len(df), dtype=bool),
                "rising_three_methods_confidence": np.zeros(len(df), dtype=float),
                "falling_three_methods_confidence": np.zeros(len(df), dtype=float),
            }

            # Process each window
            for i in range(window_size - 1, len(df)):
                window = df.iloc[i - window_size + 1:i + 1]
                
                if self._is_rising_three_methods_vectorized(window):
                    patterns["rising_three_methods"][i] = True
                    patterns["rising_three_methods_confidence"][i] = 0.8
                
                if self._is_falling_three_methods_vectorized(window):
                    patterns["falling_three_methods"][i] = True
                    patterns["falling_three_methods_confidence"][i] = 0.8

            return patterns

        except Exception as e:
            self.logger.error(f"Error detecting three methods patterns: {e}")
            return {}

    def _is_rising_three_methods_vectorized(self, window: pd.DataFrame) -> bool:
        """Check if the 5-candle pattern is a rising three methods using vectorized operations."""
        try:
            if len(window) != 5:
                return False

            candles = window.values
            body_sizes = np.abs(candles[:, 3] - candles[:, 0])  # close - open
            highs = candles[:, 1]
            lows = candles[:, 2]
            closes = candles[:, 3]
            opens = candles[:, 0]
            is_bullish = closes > opens

            # First candle should be a long bullish candle
            if not (is_bullish[0] and body_sizes[0] > np.mean(body_sizes)):
                return False

            # Next three candles should be small bearish candles within the range of the first
            for i in range(1, 4):
                if (is_bullish[i] or highs[i] > highs[0] or lows[i] < lows[0]):
                    return False

            # Last candle should be a long bullish candle closing above the first
            if not (is_bullish[4] and closes[4] > closes[0] and body_sizes[4] > np.mean(body_sizes)):
                return False

            return True

        except Exception:
            return False

    def _is_falling_three_methods_vectorized(self, window: pd.DataFrame) -> bool:
        """Check if the 5-candle pattern is a falling three methods using vectorized operations."""
        try:
            if len(window) != 5:
                return False

            candles = window.values
            body_sizes = np.abs(candles[:, 3] - candles[:, 0])  # close - open
            highs = candles[:, 1]
            lows = candles[:, 2]
            closes = candles[:, 3]
            opens = candles[:, 0]
            is_bullish = closes > opens

            # First candle should be a long bearish candle
            if not (~is_bullish[0] and body_sizes[0] > np.mean(body_sizes)):
                return False

            # Next three candles should be small bullish candles within the range of the first
            for i in range(1, 4):
                if (~is_bullish[i] or highs[i] > highs[0] or lows[i] < lows[0]):
                    return False

            # Last candle should be a long bearish candle closing below the first
            if not (~is_bullish[4] and closes[4] < closes[0] and body_sizes[4] > np.mean(body_sizes)):
                return False

            return True

        except Exception:
            return False

    def _detect_doji_patterns_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect doji patterns using vectorized operations."""
        try:
            # Vectorized calculations
            body_ratio = df["body_ratio"].values

            # Doji pattern conditions (very small body)
            doji_conditions = body_ratio <= self.doji_threshold

            # Calculate confidence scores
            doji_confidence = np.where(doji_conditions, 1.0 - body_ratio, 0.0)

            return {
                "doji": doji_conditions,
                "doji_confidence": doji_confidence,
            }

        except Exception as e:
            self.logger.error(f"Error detecting doji patterns: {e}")
            return {}

    def _detect_spinning_top_patterns_vectorized(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Detect spinning top patterns using vectorized operations."""
        try:
            # Vectorized calculations
            body_ratio = df["body_ratio"].values
            upper_shadow = df["upper_shadow"].values
            lower_shadow = df["lower_shadow"].values
            total_range = df["total_range"].values

            # Spinning top conditions (small body, equal shadows)
            spinning_top_conditions = (
                (body_ratio <= 0.3) &
                (np.abs(upper_shadow - lower_shadow) < 0.2 * total_range) &
                (upper_shadow > 0.1 * total_range) &
                (lower_shadow > 0.1 * total_range)
            )

            # Calculate confidence scores
            spinning_top_confidence = np.where(spinning_top_conditions, 0.7, 0.0)

            return {
                "spinning_top": spinning_top_conditions,
                "spinning_top_confidence": spinning_top_confidence,
            }

        except Exception as e:
            self.logger.error(f"Error detecting spinning top patterns: {e}")
            return {}

    def _convert_patterns_to_features_vectorized(
        self,
        patterns: dict[str, dict[str, np.ndarray]],
        df: pd.DataFrame,
    ) -> dict[str, float]:
        """Convert pattern analysis to ML features using vectorized operations."""
        try:
            features = {}

            # Calculate pattern type features (count and presence)
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
                if pattern_type in patterns:
                    pattern_data = patterns[pattern_type]
                    # Count patterns
                    pattern_count = sum(
                        np.sum(pattern_data.get(key, np.zeros(len(df), dtype=bool)))
                        for key in pattern_data.keys()
                        if isinstance(pattern_data[key], np.ndarray) and pattern_data[key].dtype == bool
                    )
                    features[f"{pattern_type}_count"] = pattern_count
                    features[f"{pattern_type}_present"] = 1.0 if pattern_count > 0 else 0.0

            # Calculate specific pattern features
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
                pattern_count = 0
                for pattern_data in patterns.values():
                    if pattern in pattern_data:
                        pattern_count += np.sum(pattern_data[pattern])
                
                features[f"{pattern}_count"] = pattern_count
                features[f"{pattern}_present"] = 1.0 if pattern_count > 0 else 0.0

            # Calculate pattern density features
            total_patterns = sum(
                features.get(f"{pt}_count", 0) for pt in pattern_types
            )
            features["total_patterns"] = total_patterns
            features["pattern_density"] = total_patterns / len(df) if len(df) > 0 else 0.0

            # Calculate bullish vs bearish pattern features
            bullish_patterns = sum(
                features.get(f"{pattern}_count", 0)
                for pattern in ["bullish_engulfing", "hammer", "inverted_hammer", "tweezer_bottom", "bullish_marubozu", "rising_three_methods"]
            )
            bearish_patterns = sum(
                features.get(f"{pattern}_count", 0)
                for pattern in ["bearish_engulfing", "hanging_man", "shooting_star", "tweezer_top", "bearish_marubozu", "falling_three_methods"]
            )

            features["bullish_patterns"] = bullish_patterns
            features["bearish_patterns"] = bearish_patterns
            features["bullish_bearish_ratio"] = bullish_patterns / (bearish_patterns + 1e-8)

            # Calculate recent pattern features (last 5 candles)
            recent_patterns = 0
            recent_bullish_patterns = 0
            recent_bearish_patterns = 0

            for pattern_data in patterns.values():
                for key, pattern_array in pattern_data.items():
                    if isinstance(pattern_array, np.ndarray) and pattern_array.dtype == bool:
                        if len(pattern_array) >= 5:
                            recent_count = np.sum(pattern_array[-5:])
                            recent_patterns += recent_count
                            
                            if any(bullish in key for bullish in ["bullish", "hammer", "inverted_hammer", "tweezer_bottom", "rising"]):
                                recent_bullish_patterns += recent_count
                            elif any(bearish in key for bearish in ["bearish", "hanging_man", "shooting_star", "tweezer_top", "falling"]):
                                recent_bearish_patterns += recent_count

            features["recent_patterns_count"] = recent_patterns
            features["recent_bullish_patterns"] = recent_bullish_patterns
            features["recent_bearish_patterns"] = recent_bearish_patterns

            # Calculate pattern confidence features
            all_confidences = []
            for pattern_data in patterns.values():
                for key, confidence_array in pattern_data.items():
                    if "confidence" in key and isinstance(confidence_array, np.ndarray):
                        all_confidences.extend(confidence_array[confidence_array > 0])

            if all_confidences:
                features["avg_pattern_confidence"] = np.mean(all_confidences)
                features["max_pattern_confidence"] = np.max(all_confidences)
                features["pattern_confidence_std"] = np.std(all_confidences)
            else:
                features["avg_pattern_confidence"] = 0.0
                features["max_pattern_confidence"] = 0.0
                features["pattern_confidence_std"] = 0.0

            return features

        except Exception as e:
            self.logger.error(f"Error converting patterns to features: {e}")
            return {}
