# src/analyst/meta_labeling_system.py

"""
Meta-Labeling System for Path-Dependent Trading Signals
Implements comprehensive pattern detection for analyst and tactician models.
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


class MetaLabelingSystem:
    """
    Comprehensive meta-labeling system for path-dependent trading signals.
    Implements both analyst labels (setup identification) and tactician labels (entry optimization).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("MetaLabelingSystem")

        # Configuration
        self.labeling_config = config.get("meta_labeling", {})
        self.enable_analyst_labels = self.labeling_config.get(
            "enable_analyst_labels",
            True,
        )
        self.enable_tactician_labels = self.labeling_config.get(
            "enable_tactician_labels",
            True,
        )

        # Pattern detection parameters
        self.pattern_config = self.labeling_config.get("pattern_detection", {})
        self.volatility_threshold = self.pattern_config.get(
            "volatility_threshold",
            0.02,
        )
        self.momentum_threshold = self.pattern_config.get("momentum_threshold", 0.01)
        self.volume_threshold = self.pattern_config.get("volume_threshold", 1.5)

        # Entry prediction parameters
        self.entry_config = self.labeling_config.get("entry_prediction", {})
        self.prediction_horizon = self.entry_config.get(
            "prediction_horizon",
            5,
        )  # minutes
        self.max_adverse_excursion = self.entry_config.get(
            "max_adverse_excursion",
            0.02,
        )

        self.is_initialized = False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="meta labeling system initialization",
    )
    async def initialize(self) -> bool:
        """Initialize meta-labeling system."""
        try:
            self.logger.info("🚀 Initializing meta-labeling system...")
            self.is_initialized = True
            self.logger.info("✅ Meta-labeling system initialized successfully")
            return True
        except Exception as e:
            self.print(
                initialization_error("❌ Error initializing meta-labeling system: {e}")
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError, IndexError),
        default_return={},
        context="pattern features calculation",
    )
    async def _calculate_pattern_features(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Calculate comprehensive pattern features for label generation."""
        try:
            if price_data.empty:
                self.logger.warning(
                    "Empty price data provided for pattern feature calculation",
                )
                return {}

            if volume_data.empty:
                self.logger.warning(
                    "Empty volume data provided for pattern feature calculation",
                )
                return {}

            # Validate required columns
            required_price_columns = ["open", "high", "low", "close"]
            missing_price_columns = [
                col for col in required_price_columns if col not in price_data.columns
            ]
            if missing_price_columns:
                self.logger.error(
                    f"Missing required price columns: {missing_price_columns}",
                )
                return {}

            required_volume_columns = ["volume"]
            missing_volume_columns = [
                col for col in required_volume_columns if col not in volume_data.columns
            ]
            if missing_volume_columns:
                self.logger.error(
                    f"Missing required volume columns: {missing_volume_columns}",
                )
                return {}

            features = {}

            # Technical indicators with error handling
            try:
                features.update(self._calculate_technical_indicators(price_data))
            except Exception as e:
                self.print(error("Error calculating technical indicators: {e}"))

            # Volume analysis with error handling
            try:
                features.update(self._calculate_volume_features(volume_data))
            except Exception as e:
                self.print(error("Error calculating volume features: {e}"))

            # Price action patterns with error handling
            try:
                features.update(self._calculate_price_action_patterns(price_data))
            except Exception as e:
                self.print(error("Error calculating price action patterns: {e}"))

            # Volatility patterns with error handling
            try:
                features.update(self._calculate_volatility_patterns(price_data))
            except Exception as e:
                self.print(error("Error calculating volatility patterns: {e}"))

            # Momentum patterns with error handling
            try:
                features.update(self._calculate_momentum_patterns(price_data))
            except Exception as e:
                self.print(error("Error calculating momentum patterns: {e}"))

            return features

        except Exception as e:
            self.logger.exception(
                f"Unexpected error in pattern feature calculation: {e}",
            )
            return {}

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> dict[str, float]:
        """Calculate technical indicators for pattern detection."""
        try:
            features = {}

            # Moving averages
            features["sma_20"] = (
                data["close"].rolling(20).mean().iloc[-1]
                if len(data) >= 20
                else data["close"].iloc[-1]
            )
            features["sma_50"] = (
                data["close"].rolling(50).mean().iloc[-1]
                if len(data) >= 50
                else data["close"].iloc[-1]
            )
            features["ema_12"] = data["close"].ewm(span=12).mean().iloc[-1]
            features["ema_26"] = data["close"].ewm(span=26).mean().iloc[-1]

            # RSI
            delta = data["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features["rsi"] = (100 - (100 / (1 + rs))).iloc[-1] if not rs.empty else 50

            # MACD
            ema12 = data["close"].ewm(span=12).mean()
            ema26 = data["close"].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            features["macd"] = macd.iloc[-1] if not macd.empty else 0
            features["macd_signal"] = signal.iloc[-1] if not signal.empty else 0
            features["macd_histogram"] = (
                (macd - signal).iloc[-1] if not macd.empty else 0
            )

            # Bollinger Bands
            sma20 = data["close"].rolling(20).mean()
            std20 = data["close"].rolling(20).std()
            features["bb_upper"] = (
                (sma20 + (std20 * 2)).iloc[-1]
                if not sma20.empty
                else data["close"].iloc[-1]
            )
            features["bb_lower"] = (
                (sma20 - (std20 * 2)).iloc[-1]
                if not sma20.empty
                else data["close"].iloc[-1]
            )
            features["bb_position"] = (
                (
                    (data["close"].iloc[-1] - features["bb_lower"])
                    / (features["bb_upper"] - features["bb_lower"])
                )
                if features["bb_upper"] != features["bb_lower"]
                else 0.5
            )

            # ATR
            high_low = data["high"] - data["low"]
            high_close = np.abs(data["high"] - data["close"].shift())
            low_close = np.abs(data["low"] - data["close"].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            features["atr"] = (
                true_range.rolling(14).mean().iloc[-1]
                if len(data) >= 14
                else true_range.iloc[-1]
            )

            return features

        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return {}

    def _calculate_volume_features(self, data: pd.DataFrame) -> dict[str, float]:
        """Calculate volume-based features."""
        try:
            features = {}

            if "volume" in data.columns:
                features["volume_sma"] = (
                    data["volume"].rolling(20).mean().iloc[-1]
                    if len(data) >= 20
                    else data["volume"].iloc[-1]
                )
                features["volume_ratio"] = (
                    data["volume"].iloc[-1] / features["volume_sma"]
                    if features["volume_sma"] > 0
                    else 1.0
                )
                features["volume_trend"] = (
                    data["volume"].rolling(10).mean().diff().iloc[-1]
                    if len(data) >= 10
                    else 0
                )

                # VWAP
                vwap = (data["close"] * data["volume"]).rolling(20).sum() / data[
                    "volume"
                ].rolling(20).sum()
                features["vwap"] = (
                    vwap.iloc[-1] if not vwap.empty else data["close"].iloc[-1]
                )
                features["price_vwap_ratio"] = (
                    data["close"].iloc[-1] / features["vwap"]
                    if features["vwap"] > 0
                    else 1.0
                )

            return features

        except Exception as e:
            self.logger.error(f"Error calculating volume features: {e}")
            return {}

    def _calculate_price_action_patterns(self, data: pd.DataFrame) -> dict[str, float]:
        """Calculate price action pattern features."""
        try:
            features = {}

            # Support and resistance levels
            features["recent_high"] = (
                data["high"].rolling(20).max().iloc[-1]
                if len(data) >= 20
                else data["high"].iloc[-1]
            )
            features["recent_low"] = (
                data["low"].rolling(20).min().iloc[-1]
                if len(data) >= 20
                else data["low"].iloc[-1]
            )
            features["price_position"] = (
                (data["close"].iloc[-1] - features["recent_low"])
                / (features["recent_high"] - features["recent_low"])
                if features["recent_high"] != features["recent_low"]
                else 0.5
            )

            # Price momentum
            features["price_momentum_5"] = (
                data["close"].pct_change(5).iloc[-1] if len(data) >= 5 else 0
            )
            features["price_momentum_10"] = (
                data["close"].pct_change(10).iloc[-1] if len(data) >= 10 else 0
            )
            features["price_acceleration"] = (
                data["close"].pct_change(5).diff().iloc[-1] if len(data) >= 5 else 0
            )

            # Candlestick patterns
            features["body_size"] = (
                np.abs(data["close"].iloc[-1] - data["open"].iloc[-1])
                / data["close"].iloc[-1]
            )
            features["upper_shadow"] = (
                data["high"].iloc[-1]
                - np.maximum(data["open"].iloc[-1], data["close"].iloc[-1])
            ) / data["close"].iloc[-1]
            features["lower_shadow"] = (
                np.minimum(data["open"].iloc[-1], data["close"].iloc[-1])
                - data["low"].iloc[-1]
            ) / data["close"].iloc[-1]

            return features

        except Exception as e:
            self.print(error("Error calculating price action patterns: {e}"))
            return {}

    def _calculate_volatility_patterns(self, data: pd.DataFrame) -> dict[str, float]:
        """Calculate volatility pattern features."""
        try:
            features = {}

            # Volatility measures
            returns = data["close"].pct_change()
            features["volatility_20"] = (
                returns.rolling(20).std().iloc[-1] if len(data) >= 20 else returns.std()
            )
            features["volatility_10"] = (
                returns.rolling(10).std().iloc[-1] if len(data) >= 10 else returns.std()
            )
            features["volatility_ratio"] = (
                features["volatility_10"] / features["volatility_20"]
                if features["volatility_20"] > 0
                else 1.0
            )

            # Volatility regime
            features["volatility_regime"] = (
                1 if features["volatility_20"] > self.volatility_threshold else 0
            )

            # Bollinger Band width
            sma20 = data["close"].rolling(20).mean()
            std20 = data["close"].rolling(20).std()
            bb_width = (sma20 + (std20 * 2)) - (sma20 - (std20 * 2))
            features["bb_width"] = (
                bb_width.iloc[-1] / sma20.iloc[-1]
                if not sma20.empty and sma20.iloc[-1] > 0
                else 0
            )

            return features

        except Exception as e:
            self.print(error("Error calculating volatility patterns: {e}"))
            return {}

    def _calculate_momentum_patterns(self, data: pd.DataFrame) -> dict[str, float]:
        """Calculate momentum pattern features."""
        try:
            features = {}

            # RSI momentum
            features["rsi_momentum"] = (
                data["close"].pct_change(5).iloc[-1] if len(data) >= 5 else 0
            )

            # MACD momentum
            ema12 = data["close"].ewm(span=12).mean()
            ema26 = data["close"].ewm(span=26).mean()
            macd = ema12 - ema26
            features["macd_momentum"] = macd.diff().iloc[-1] if not macd.empty else 0

            # Price momentum
            features["momentum_regime"] = (
                1 if abs(features["rsi_momentum"]) > self.momentum_threshold else 0
            )

            return features

        except Exception as e:
            self.print(error("Error calculating momentum patterns: {e}"))
            return {}

    def _detect_additional_analyst_patterns(
        self,
        data: pd.DataFrame,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Detect additional analyst patterns requested by users.

        Patterns added:
        - LIQUIDITY_GRAB
        - ABSORPTION_AT_LEVEL
        - TRENDING_RANGE
        - MOVING_AVERAGE_BOUNCE
        - HEAD_AND_SHOULDERS (proxy)
        - DOUBLE_TOP_BOTTOM (proxy)
        - CLIMACTIC_REVERSAL
        """
        try:
            patterns: dict[str, Any] = {}

            close = data["close"].astype(float)
            open_ = data.get("open", close).astype(float)
            high = data.get("high", close).astype(float)
            low = data.get("low", close).astype(float)

            # Common helpers
            bb_position = features.get("bb_position", 0.5)
            bb_width = features.get("bb_width", 0.1)
            volume_ratio = features.get("volume_ratio", 1.0)
            vol_ratio_threshold = max(1.0, float(self.volume_threshold))
            momentum_5 = features.get("price_momentum_5", 0.0)
            momentum_10 = features.get("price_momentum_10", 0.0)

            # LIQUIDITY_GRAB: strong volume spike at extremes with immediate mean-reversion signature
            near_extreme = (bb_position <= 0.1) or (bb_position >= 0.9)
            volume_extreme = volume_ratio >= (2.0 * vol_ratio_threshold)
            mean_reversion_signal = abs(momentum_5) < self.momentum_threshold and bb_width <= 0.06
            patterns["LIQUIDITY_GRAB"] = 1 if (near_extreme and volume_extreme and mean_reversion_signal) else 0

            # ABSORPTION_AT_LEVEL: high volume with narrow range and small net change
            true_range = float((high.iloc[-1] - low.iloc[-1]) / max(1e-12, close.iloc[-1])) if len(close) else 0.0
            small_body = abs(float((close.iloc[-1] - open_.iloc[-1]) / max(1e-12, close.iloc[-1]))) < 0.002
            narrow_range = (bb_width <= 0.03) or (true_range <= 0.002)
            patterns["ABSORPTION_AT_LEVEL"] = 1 if (volume_ratio >= vol_ratio_threshold and narrow_range and small_body) else 0

            # TRENDING_RANGE: low volatility range with directional drift present
            low_vol_range = bb_width <= 0.05
            mild_drift = 0.005 <= abs(momentum_10) <= 0.02
            patterns["TRENDING_RANGE"] = 1 if (low_vol_range and mild_drift) else 0

            # MOVING_AVERAGE_BOUNCE: price near SMA20
            sma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else close.iloc[-1]
            dist_ma = abs(float(close.iloc[-1] - sma20) / max(1e-12, close.iloc[-1])) if len(close) else 1.0
            patterns["MOVING_AVERAGE_BOUNCE"] = 1 if (dist_ma <= 0.0015 and abs(momentum_5) <= 0.01) else 0

            # HEAD_AND_SHOULDERS (proxy): price near recent high with weakening momentum and RSI < 50
            recent_high = float(close.rolling(30, min_periods=1).max().iloc[-1]) if len(close) else float(close.iloc[-1] if len(close) else 0)
            near_recent_high = abs(float(close.iloc[-1] - recent_high) / max(1e-12, recent_high)) <= 0.003 if recent_high else False
            rsi = features.get("rsi", 50.0)
            weakening = momentum_5 < 0 and (abs(momentum_5) < abs(momentum_10))
            patterns["HEAD_AND_SHOULDERS"] = 1 if (near_recent_high and weakening and rsi < 50) else 0

            # DOUBLE_TOP_BOTTOM (proxy): price near recent extrema with reversal momentum
            recent_low = float(close.rolling(30, min_periods=1).min().iloc[-1]) if len(close) else float(close.iloc[-1] if len(close) else 0)
            near_top = abs(float(close.iloc[-1] - recent_high) / max(1e-12, recent_high)) <= 0.002 if recent_high else False
            near_bottom = abs(float(close.iloc[-1] - recent_low) / max(1e-12, recent_low)) <= 0.002 if recent_low else False
            reversal = (near_top and momentum_5 < 0) or (near_bottom and momentum_5 > 0)
            patterns["DOUBLE_TOP_BOTTOM"] = 1 if reversal else 0

            # CLIMACTIC_REVERSAL: very high volume, large range candle, and momentum sign flip proxy
            large_body = abs(float((close.iloc[-1] - open_.iloc[-1]) / max(1e-12, close.iloc[-1]))) >= 0.01 if len(close) else False
            momentum_flip_proxy = abs(momentum_5) >= (self.momentum_threshold * 2)
            patterns["CLIMACTIC_REVERSAL"] = 1 if (volume_extreme and large_body and momentum_flip_proxy) else 0

            return patterns

        except Exception as e:
            self.print(error(f"Error detecting additional analyst patterns: {e}"))
            return {
                "LIQUIDITY_GRAB": 0,
                "ABSORPTION_AT_LEVEL": 0,
                "TRENDING_RANGE": 0,
                "MOVING_AVERAGE_BOUNCE": 0,
                "HEAD_AND_SHOULDERS": 0,
                "DOUBLE_TOP_BOTTOM": 0,
                "CLIMACTIC_REVERSAL": 0,
            }

    # Analyst Label Detection Methods

    def _detect_strong_trend_continuation(
        self,
        data: pd.DataFrame,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Detect STRONG_TREND_CONTINUATION pattern."""
        try:
            # Strong trend continuation: healthy pullback within established trend
            trend_strength = features.get("price_momentum_10", 0)
            rsi = features.get("rsi", 50)
            bb_position = features.get("bb_position", 0.5)

            # Conditions for strong trend continuation
            is_uptrend = trend_strength > 0.02  # Strong upward momentum
            is_pullback = 0.3 < bb_position < 0.7  # Price in middle of BB
            is_healthy_rsi = 40 < rsi < 70  # Not overbought/oversold

            strong_trend_continuation = is_uptrend and is_pullback and is_healthy_rsi

            return {
                "STRONG_TREND_CONTINUATION": 1 if strong_trend_continuation else 0,
                "strong_trend_confidence": min(abs(trend_strength) * 10, 1.0)
                if strong_trend_continuation
                else 0,
            }

        except Exception as e:
            self.print(error("Error detecting strong trend continuation: {e}"))
            return {"STRONG_TREND_CONTINUATION": 0, "strong_trend_confidence": 0}

    def _detect_exhaustion_reversal(
        self,
        data: pd.DataFrame,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Detect EXHAUSTION_REVERSAL pattern."""
        try:
            # Exhaustion reversal: overextended trend showing weakness
            rsi = features.get("rsi", 50)
            bb_position = features.get("bb_position", 0.5)
            volume_ratio = features.get("volume_ratio", 1.0)
            momentum = features.get("price_momentum_5", 0)

            # Conditions for exhaustion reversal
            is_overbought = rsi > 70 or bb_position > 0.8
            is_weakening = momentum < 0  # Momentum turning negative
            is_high_volume = volume_ratio > self.volume_threshold

            exhaustion_reversal = is_overbought and is_weakening and is_high_volume

            return {
                "EXHAUSTION_REVERSAL": 1 if exhaustion_reversal else 0,
                "exhaustion_confidence": min((rsi - 70) / 30, 1.0)
                if exhaustion_reversal
                else 0,
            }

        except Exception as e:
            self.print(error("Error detecting exhaustion reversal: {e}"))
            return {"EXHAUSTION_REVERSAL": 0, "exhaustion_confidence": 0}

    def _detect_range_mean_reversion(
        self,
        data: pd.DataFrame,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Detect RANGE_MEAN_REVERSION pattern."""
        try:
            # Range mean reversion: setup at edge of sideways range
            bb_position = features.get("bb_position", 0.5)
            volatility = features.get("volatility_20", 0)
            features.get("price_position", 0.5)

            # Conditions for range mean reversion
            is_at_edge = bb_position < 0.2 or bb_position > 0.8
            is_low_volatility = volatility < self.volatility_threshold
            is_sideways = abs(features.get("price_momentum_10", 0)) < 0.01

            range_mean_reversion = is_at_edge and is_low_volatility and is_sideways

            return {
                "RANGE_MEAN_REVERSION": 1 if range_mean_reversion else 0,
                "range_reversion_confidence": min(abs(bb_position - 0.5) * 2, 1.0)
                if range_mean_reversion
                else 0,
            }

        except Exception as e:
            self.print(error("Error detecting range mean reversion: {e}"))
            return {"RANGE_MEAN_REVERSION": 0, "range_reversion_confidence": 0}

    def _detect_breakout_patterns(
        self,
        data: pd.DataFrame,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Detect BREAKOUT_SUCCESS and BREAKOUT_FAILURE patterns."""
        try:
            bb_position = features.get("bb_position", 0.5)
            volume_ratio = features.get("volume_ratio", 1.0)
            momentum = features.get("price_momentum_5", 0)
            recent_high = features.get("recent_high", data["close"].iloc[-1])
            recent_low = features.get("recent_low", data["close"].iloc[-1])
            current_price = data["close"].iloc[-1]

            # Breakout success: price breaks level and continues
            is_breakout_up = current_price > recent_high and momentum > 0.01
            is_breakout_down = current_price < recent_low and momentum < -0.01
            is_high_volume = volume_ratio > self.volume_threshold

            breakout_success = (is_breakout_up or is_breakout_down) and is_high_volume

            # Breakout failure: price breaks level but reverses
            is_failed_breakout = (bb_position > 0.8 or bb_position < 0.2) and abs(
                momentum,
            ) < 0.005

            return {
                "BREAKOUT_SUCCESS": 1 if breakout_success else 0,
                "BREAKOUT_FAILURE": 1 if is_failed_breakout else 0,
                "breakout_confidence": min(volume_ratio / 2, 1.0)
                if breakout_success
                else 0,
            }

        except Exception as e:
            self.print(error("Error detecting breakout patterns: {e}"))
            return {
                "BREAKOUT_SUCCESS": 0,
                "BREAKOUT_FAILURE": 0,
                "breakout_confidence": 0,
            }

    def _detect_volatility_patterns(
        self,
        data: pd.DataFrame,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Detect VOLATILITY_COMPRESSION and VOLATILITY_EXPANSION patterns."""
        try:
            volatility_ratio = features.get("volatility_ratio", 1.0)
            bb_width = features.get("bb_width", 0.1)
            volume_ratio = features.get("volume_ratio", 1.0)

            # Volatility compression: BB width narrowing
            is_compression = bb_width < 0.05 and volatility_ratio < 0.8

            # Volatility expansion: sudden increase in volatility
            is_expansion = (
                volatility_ratio > 1.5 and volume_ratio > self.volume_threshold
            )

            return {
                "VOLATILITY_COMPRESSION": 1 if is_compression else 0,
                "VOLATILITY_EXPANSION": 1 if is_expansion else 0,
                "volatility_confidence": min(volatility_ratio, 1.0),
            }

        except Exception as e:
            self.print(error("Error detecting volatility patterns: {e}"))
            return {
                "VOLATILITY_COMPRESSION": 0,
                "VOLATILITY_EXPANSION": 0,
                "volatility_confidence": 0,
            }

    def _detect_chart_patterns(
        self,
        data: pd.DataFrame,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Detect various chart patterns."""
        try:
            patterns = {}

            # Flag formation: sharp move followed by tight pullback
            momentum = features.get("price_momentum_5", 0)
            volatility = features.get("volatility_10", 0)
            is_flag = abs(momentum) > 0.02 and volatility < self.volatility_threshold
            patterns["FLAG_FORMATION"] = 1 if is_flag else 0

            # Triangle formation: narrowing price range
            bb_width = features.get("bb_width", 0.1)
            is_triangle = bb_width < 0.03
            patterns["TRIANGLE_FORMATION"] = 1 if is_triangle else 0

            # Rectangle formation: horizontal consolidation
            price_position = features.get("price_position", 0.5)
            is_rectangle = 0.3 < price_position < 0.7 and bb_width < 0.05
            patterns["RECTANGLE_FORMATION"] = 1 if is_rectangle else 0

            return patterns

        except Exception as e:
            self.print(error("Error detecting chart patterns: {e}"))
            return {
                "FLAG_FORMATION": 0,
                "TRIANGLE_FORMATION": 0,
                "RECTANGLE_FORMATION": 0,
            }

    def _detect_momentum_patterns(
        self,
        data: pd.DataFrame,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Detect momentum-related patterns."""
        try:
            patterns = {}

            # Momentum ignition: momentum indicators breaking out
            rsi = features.get("rsi", 50)
            macd = features.get("macd", 0)
            momentum = features.get("price_momentum_5", 0)

            is_momentum_ignition = (
                (rsi > 60 or rsi < 40)
                and abs(macd) > 0.001
                and abs(momentum) > self.momentum_threshold
            )
            patterns["MOMENTUM_IGNITION"] = 1 if is_momentum_ignition else 0

            # Gradual momentum fade: declining momentum
            momentum_10 = features.get("price_momentum_10", 0)
            is_fade = (
                abs(momentum) < abs(momentum_10)
                and abs(momentum) < self.momentum_threshold
            )
            patterns["GRADUAL_MOMENTUM_FADE"] = 1 if is_fade else 0

            return patterns

        except Exception as e:
            self.print(error("Error detecting momentum patterns: {e}"))
            return {"MOMENTUM_IGNITION": 0, "GRADUAL_MOMENTUM_FADE": 0}

    # Tactician Label Detection Methods

    async def _calculate_entry_features(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Calculate features specific to entry optimization."""
        try:
            features = {}

            # Micro-price action
            features["price_change_1m"] = (
                price_data["close"].pct_change(1).iloc[-1]
                if len(price_data) >= 1
                else 0
            )
            features["price_change_5m"] = (
                price_data["close"].pct_change(5).iloc[-1]
                if len(price_data) >= 5
                else 0
            )

            # Volume analysis
            if "volume" in volume_data.columns:
                features["volume_spike"] = (
                    volume_data["volume"].iloc[-1]
                    / volume_data["volume"].rolling(10).mean().iloc[-1]
                    if len(volume_data) >= 10
                    else 1.0
                )

            # Order flow analysis
            if order_flow_data is not None:
                features["order_imbalance"] = self._calculate_order_imbalance(
                    order_flow_data,
                )

            return features

        except Exception as e:
            self.print(error("Error calculating entry features: {e}"))
            return {}

    def _calculate_order_imbalance(self, order_flow_data: pd.DataFrame) -> float:
        """Calculate order book imbalance."""
        try:
            if (
                "bid_volume" in order_flow_data.columns
                and "ask_volume" in order_flow_data.columns
            ):
                bid_vol = order_flow_data["bid_volume"].iloc[-1]
                ask_vol = order_flow_data["ask_volume"].iloc[-1]
                total_vol = bid_vol + ask_vol
                return (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0
            return 0
        except Exception as e:
            self.print(error("Error calculating order imbalance: {e}"))
            return 0

    def _predict_price_extremes(
        self,
        data: pd.DataFrame,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Predict LOWEST_PRICE_NEXT_1m and HIGHEST_PRICE_NEXT_1m."""
        try:
            current_price = data["close"].iloc[-1]
            volatility = features.get("volatility_20", 0.01)
            momentum = features.get("price_momentum_5", 0)

            # Simple prediction based on current momentum and volatility
            price_change = momentum * self.prediction_horizon
            volatility_impact = volatility * np.sqrt(self.prediction_horizon)

            lowest_price = current_price * (1 + price_change - volatility_impact)
            highest_price = current_price * (1 + price_change + volatility_impact)

            return {
                "LOWEST_PRICE_NEXT_1m": lowest_price,
                "HIGHEST_PRICE_NEXT_1m": highest_price,
                "price_extreme_confidence": min(abs(momentum) * 10, 1.0),
            }

        except Exception as e:
            self.print(error("Error predicting price extremes: {e}"))
            return {
                "LOWEST_PRICE_NEXT_1m": data["close"].iloc[-1],
                "HIGHEST_PRICE_NEXT_1m": data["close"].iloc[-1],
                "price_extreme_confidence": 0,
            }

    def _predict_order_returns(
        self,
        data: pd.DataFrame,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Predict LIMIT_ORDER_RETURN."""
        try:
            data["close"].iloc[-1]
            volatility = features.get("volatility_20", 0.01)
            momentum = features.get("price_momentum_5", 0)

            # Predict optimal limit order return
            expected_return = abs(momentum) * 0.5  # Conservative estimate
            volatility_adjustment = volatility * 0.3

            limit_order_return = expected_return - volatility_adjustment

            return {
                "LIMIT_ORDER_RETURN": max(limit_order_return, 0.001),  # Minimum return
                "limit_order_confidence": min(abs(momentum) * 5, 1.0),
            }

        except Exception as e:
            self.print(error("Error predicting order returns: {e}"))
            return {"LIMIT_ORDER_RETURN": 0.001, "limit_order_confidence": 0}

    def _detect_entry_signals(
        self,
        data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Detect various entry signals."""
        try:
            signals = {}

            # VWAP reversion entry
            features = {}
            vwap = float(data["close"].iloc[-1])
            current_price = float(data["close"].iloc[-1])
            price_vwap_ratio = 1.0

            is_vwap_reversion = abs(price_vwap_ratio - 1.0) < 0.01
            signals["VWAP_REVERSION_ENTRY"] = 1 if is_vwap_reversion else 0

            # Market order now: aggressive momentum
            momentum = 0.0
            volume_spike = 1.0

            is_market_order = (
                abs(momentum) > self.momentum_threshold * 2
                and volume_spike > self.volume_threshold
            )
            signals["MARKET_ORDER_NOW"] = 1 if is_market_order else 0

            # Chase micro breakout
            recent_high = current_price
            is_micro_breakout = False
            signals["CHASE_MICRO_BREAKOUT"] = 1 if is_micro_breakout else 0

            # Order book imbalance flip
            order_imbalance = 0.0
            is_imbalance_flip = False
            signals["ORDERBOOK_IMBALANCE_FLIP"] = 1 if is_imbalance_flip else 0

            # Aggressive taker spike
            volume_ratio = features.get("volume_ratio", 1.0)
            is_taker_spike = volume_ratio > self.volume_threshold * 2
            signals["AGGRESSIVE_TAKER_SPIKE"] = 1 if is_taker_spike else 0

            return signals

        except Exception as e:
            self.print(error("Error detecting entry signals: {e}"))
            return {
                "VWAP_REVERSION_ENTRY": 0,
                "MARKET_ORDER_NOW": 0,
                "CHASE_MICRO_BREAKOUT": 0,
                "ORDERBOOK_IMBALANCE_FLIP": 0,
                "AGGRESSIVE_TAKER_SPIKE": 0,
            }

    def _predict_adverse_excursion(
        self,
        data: pd.DataFrame,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """Predict MAX_ADVERSE_EXCURSION_RETURN."""
        try:
            volatility = features.get("volatility_20", 0.01)
            momentum = features.get("price_momentum_5", 0)

            # Predict worst-case adverse move
            base_adverse = volatility * np.sqrt(self.prediction_horizon)
            momentum_adjustment = abs(momentum) * 0.5

            max_adverse_excursion = base_adverse + momentum_adjustment

            return {
                "MAX_ADVERSE_EXCURSION_RETURN": min(
                    max_adverse_excursion,
                    self.max_adverse_excursion,
                ),
                "adverse_excursion_confidence": min(volatility * 50, 1.0),
            }

        except Exception as e:
            self.print(error("Error predicting adverse excursion: {e}"))
            return {
                "MAX_ADVERSE_EXCURSION_RETURN": 0.01,
                "adverse_excursion_confidence": 0,
            }

    def generate_no_setup_label(self) -> dict[str, Any]:
        """Generate NO_SETUP label when no other patterns are detected."""
        return {"NO_SETUP": 1, "no_setup_confidence": 1.0}

    def generate_abort_entry_signal(self, features: dict[str, Any]) -> dict[str, Any]:
        """Generate ABORT_ENTRY_SIGNAL when conditions deteriorate."""
        try:
            # Conditions that would abort entry
            volatility = features.get("volatility_20", 0)
            momentum = features.get("price_momentum_5", 0)
            volume_ratio = features.get("volume_ratio", 1.0)

            # Abort conditions
            is_high_volatility = volatility > self.volatility_threshold * 2
            is_negative_momentum = momentum < -self.momentum_threshold
            is_low_volume = volume_ratio < 0.5

            should_abort = is_high_volatility or is_negative_momentum or is_low_volume

            return {
                "ABORT_ENTRY_SIGNAL": 1 if should_abort else 0,
                "abort_confidence": min(volatility * 20, 1.0) if should_abort else 0,
            }

        except Exception as e:
            self.print(error("Error generating abort signal: {e}"))
            return {"ABORT_ENTRY_SIGNAL": 0, "abort_confidence": 0}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return={},
        context="analyst labels generation",
    )
    async def generate_analyst_labels(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        timeframe: str = "30m",
    ) -> dict[str, Any]:
        """
        Generate analyst labels for setup identification (multi-timeframe).

        Args:
            price_data: OHLCV price data
            volume_data: Volume data
            timeframe: Timeframe for analysis

        Returns:
            Dict containing analyst labels and confidence scores
        """
        try:
            if not self.is_initialized:
                self.print(initialization_error("Meta-labeling system not initialized"))
                return {}

            # Calculate pattern features
            features = await self._calculate_pattern_features(price_data, volume_data)

            # Generate analyst-specific labels
            analyst_labels = {}

            # Strong trend continuation
            trend_continuation = self._detect_strong_trend_continuation(
                price_data,
                features,
            )
            analyst_labels.update(trend_continuation)

            # Exhaustion reversal
            exhaustion_reversal = self._detect_exhaustion_reversal(price_data, features)
            analyst_labels.update(exhaustion_reversal)

            # Range mean reversion
            range_reversion = self._detect_range_mean_reversion(price_data, features)
            analyst_labels.update(range_reversion)

            # Breakout patterns
            breakout_patterns = self._detect_breakout_patterns(price_data, features)
            analyst_labels.update(breakout_patterns)

            # Volatility patterns
            volatility_patterns = self._detect_volatility_patterns(price_data, features)
            analyst_labels.update(volatility_patterns)

            # Chart patterns
            chart_patterns = self._detect_chart_patterns(price_data, features)
            analyst_labels.update(chart_patterns)

            # Momentum patterns
            momentum_patterns = self._detect_momentum_patterns(price_data, features)
            analyst_labels.update(momentum_patterns)

            # Additional analyst patterns requested by user
            additional_patterns = self._detect_additional_analyst_patterns(price_data, features)
            analyst_labels.update(additional_patterns)

            # If no patterns detected, generate NO_SETUP label
            if not any(analyst_labels.values()):
                no_setup = self.generate_no_setup_label()
                analyst_labels.update(no_setup)

            # Add metadata
            analyst_labels.update(
                {
                    "timeframe": timeframe,
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "features_used": list(features.keys()),
                    "label_count": len(
                        [
                            v
                            for v in analyst_labels.values()
                            if isinstance(v, int | float) and v > 0
                        ],
                    ),
                },
            )

            self.logger.info(
                f"Generated {analyst_labels.get('label_count', 0)} analyst labels for {timeframe}",
            )
            return analyst_labels

        except Exception as e:
            self.print(error("Error generating analyst labels: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return={},
        context="tactician labels generation",
    )
    async def generate_tactician_labels(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
        timeframe: str = "1m",
    ) -> dict[str, Any]:
        """
        Generate tactician labels for entry optimization (1m timeframe).

        Args:
            price_data: OHLCV price data
            volume_data: Volume data
            order_flow_data: Optional order flow data
            timeframe: Timeframe for analysis (typically 1m)

        Returns:
            Dict containing tactician labels and confidence scores
        """
        try:
            if not self.is_initialized:
                self.print(initialization_error("Meta-labeling system not initialized"))
                return {}

            # Calculate entry features
            entry_features = await self._calculate_entry_features(
                price_data,
                volume_data,
                order_flow_data,
            )

            # Generate tactician-specific labels
            tactician_labels = {}

            # Entry signals
            entry_signals = self._detect_entry_signals(
                price_data,
                volume_data,
                order_flow_data,
            )
            tactician_labels.update(entry_signals)

            # Price extremes prediction
            price_extremes = self._predict_price_extremes(price_data, entry_features)
            tactician_labels.update(price_extremes)

            # Order returns prediction
            order_returns = self._predict_order_returns(price_data, entry_features)
            tactician_labels.update(order_returns)

            # Adverse excursion prediction
            adverse_excursion = self._predict_adverse_excursion(
                price_data,
                entry_features,
            )
            tactician_labels.update(adverse_excursion)

            # Abort entry signal
            abort_signal = self.generate_abort_entry_signal(entry_features)
            tactician_labels.update(abort_signal)

            # Add metadata
            tactician_labels.update(
                {
                    "timeframe": timeframe,
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "features_used": list(entry_features.keys()),
                    "signal_count": len(
                        [
                            v
                            for v in tactician_labels.values()
                            if isinstance(v, int | float) and v > 0
                        ],
                    ),
                },
            )

            # Single consolidated log with compact summaries for all tactician labels
            try:
                summaries: dict[str, dict[str, int | float | str]] = {}
                for k, v in tactician_labels.items():
                    if k in ("timeframe", "timestamp", "features_used", "signal_count"):
                        continue
                    try:
                        if isinstance(v, (np.ndarray, pd.Series)):
                            arr = v if isinstance(v, np.ndarray) else v.to_numpy()
                            summaries[k] = {
                                "nonzero": int(np.count_nonzero(arr)),
                                "len": int(arr.size),
                            }
                        elif isinstance(v, (int, float)):
                            summaries[k] = {"value": float(v)}
                        else:
                            summaries[k] = {"type": type(v).__name__}
                    except Exception:
                        summaries[k] = {"summary": "unavailable"}
                self.logger.info({
                    "msg": f"Tactician labels summary for {timeframe}",
                    "timeframe": timeframe,
                    "signal_count": tactician_labels.get("signal_count", 0),
                    "labels": summaries,
                })
            except Exception as e:
                self.logger.warning(f"Failed to log tactician label summary: {e}")
            return tactician_labels

        except Exception as e:
            self.print(error("Error generating tactician labels: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return={},
        context="combined labels generation",
    )
    async def generate_combined_labels(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        order_flow_data: pd.DataFrame | None = None,
        analyst_timeframe: str = "30m",
        tactician_timeframe: str = "1m",
    ) -> dict[str, Any]:
        """
        Generate combined analyst and tactician labels.

        Args:
            price_data: OHLCV price data
            volume_data: Volume data
            order_flow_data: Optional order flow data
            analyst_timeframe: Timeframe for analyst analysis
            tactician_timeframe: Timeframe for tactician analysis

        Returns:
            Dict containing combined labels
        """
        try:
            # Generate analyst labels
            analyst_labels = await self.generate_analyst_labels(
                price_data,
                volume_data,
                analyst_timeframe,
            )

            # Generate tactician labels
            tactician_labels = await self.generate_tactician_labels(
                price_data,
                volume_data,
                order_flow_data,
                tactician_timeframe,
            )

            # Combine labels
            combined_labels = {
                "analyst_labels": analyst_labels,
                "tactician_labels": tactician_labels,
                "combined_timestamp": pd.Timestamp.now().isoformat(),
                "total_labels": (
                    analyst_labels.get("label_count", 0)
                    + tactician_labels.get("signal_count", 0)
                ),
            }

            self.logger.info(
                f"Generated {combined_labels['total_labels']} combined labels",
            )
            return combined_labels

        except Exception as e:
            self.print(error("Error generating combined labels: {e}"))
            return {}

    def get_system_info(self) -> dict[str, Any]:
        """Get meta-labeling system information."""
        return {
            "is_initialized": self.is_initialized,
            "enable_analyst_labels": self.enable_analyst_labels,
            "enable_tactician_labels": self.enable_tactician_labels,
            "volatility_threshold": self.volatility_threshold,
            "momentum_threshold": self.momentum_threshold,
            "volume_threshold": self.volume_threshold,
            "prediction_horizon": self.prediction_horizon,
            "max_adverse_excursion": self.max_adverse_excursion,
            "description": "Meta-labeling system for path-dependent trading signals",
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="meta labeling system cleanup",
    )
    async def stop(self) -> None:
        """Stop the meta-labeling system."""
        self.logger.info("🛑 Stopping Meta-Labeling System...")
        try:
            self.is_initialized = False
            self.logger.info("✅ Meta-Labeling System stopped successfully")
        except Exception as e:
            self.print(error("Error stopping meta-labeling system: {e}"))
