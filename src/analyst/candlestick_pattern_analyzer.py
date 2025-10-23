from src.utils.tprint import tprint
import warnings

from .core.decorators import handles_errors
"""Candlestick pattern analyzer for advanced feature engineering."""

import numpy as np
import pandas as pd
from typing import Any, Dict, List

from ...utils.logger import system_logger
from src.core.error_classes import execution_error, initialization_error
import logging

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:

    cp = None

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

    @handles_errors(
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

    @handles_errors(
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
                self.print_message(
                    initialization_error("Candlestick pattern analyzer not initialized")
                )
                return {}

            if price_data.empty or len(price_data) < 3:
                self.logger.warning("Insufficient data for pattern analysis")
                return {}

            # Prepare data with calculated metrics
            df = self._prepare_candlestick_data(price_data)

            # Analyze all patterns
            patterns = {
                "engulfing_patterns": self._detect_engulfing_patterns(df),
                "hammer_hanging_man": self._detect_hammer_hanging_man(df),
                "shooting_star_inverted_hammer": self._detect_shooting_star_inverted_hammer(df),
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

        except (ValueError, TypeError) as e:
            self.logger.debug(f"Error in {self.__class__.__name__}: {e}")
            self.logger.error("Error analyzing candlestick patterns: {e}")
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

        except (ValueError, TypeError, IndexError) as e:
            self.logger.debug(f"Error in {self.__class__.__name__}: {e}")
            self.logger.error("Error preparing candlestick data: {e}")
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

        except (AttributeError, TypeError) as e:
            self.logger.debug(f"Error in {self.__class__.__name__}: {e}")
            self.logger.error("Error converting patterns to features: {e}")
            return {}

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

    def print_message(self, message: str) -> None:
        """Print message with proper formatting."""
        tprint(message)

def initialization_error(message: str) -> str:
    """Format initialization error message."""
    return f"❌ {message}"

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _vectorbt_apply_operation(self, data: pd.Series, func,
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)

        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
