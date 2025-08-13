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
        # Centralized thresholds with defaults
        self.volatility_threshold = self.pattern_config.get("volatility_threshold", 0.02)
        self.momentum_threshold = self.pattern_config.get("momentum_threshold", 0.01)
        self.volume_threshold = self.pattern_config.get("volume_threshold", 1.5)
        # Bollinger thresholds
        self.bb_edge_low = self.pattern_config.get("bb_edge_low", 0.2)
        self.bb_edge_high = self.pattern_config.get("bb_edge_high", 0.8)
        self.bb_mid_low = self.pattern_config.get("bb_mid_low", 0.3)
        self.bb_mid_high = self.pattern_config.get("bb_mid_high", 0.7)
        self.bb_width_compression = self.pattern_config.get("bb_width_compression", 0.05)
        self.bb_width_triangle = self.pattern_config.get("bb_width_triangle", 0.03)
        # Momentum thresholds
        self.trend_momentum_strong = self.pattern_config.get("trend_momentum_strong", 0.02)
        self.breakout_momentum = self.pattern_config.get("breakout_momentum", 0.01)
        self.failed_break_momentum = self.pattern_config.get("failed_break_momentum", 0.005)
        # RSI thresholds
        self.rsi_overbought = self.pattern_config.get("rsi_overbought", 70)
        self.rsi_oversold = self.pattern_config.get("rsi_oversold", 30)
        self.rsi_momentum_hi = self.pattern_config.get("rsi_momentum_hi", 60)
        self.rsi_momentum_lo = self.pattern_config.get("rsi_momentum_lo", 40)
        # Volume ratios
        self.absorption_volume_spike = self.pattern_config.get("absorption_volume_spike", 1.5)
        self.stop_hunt_volume_spike = self.pattern_config.get("stop_hunt_volume_spike", 2.0)
        self.ignition_volume_spike = self.pattern_config.get("ignition_volume_spike", 3.0)
        # Lookbacks
        self.lookback_breakout = self.pattern_config.get("lookback_breakout", 20)
        self.lookback_stop_hunt = self.pattern_config.get("lookback_stop_hunt", 15)
        self.lookback_price_position = self.pattern_config.get("lookback_price_position", 20)
        self.lookback_poc_bins = self.pattern_config.get("lookback_poc_bins", 20)
        self.lookback_poc_window = self.pattern_config.get("lookback_poc_window", 1440)
        # Support/Resistance parameters
        self.sr_lookback = self.pattern_config.get("sr_lookback", 50)
        self.sr_near_pct = self.pattern_config.get("sr_near_pct", 0.003)  # 0.3%
        self.sr_break_pct = self.pattern_config.get("sr_break_pct", 0.0005)
        # Liquidity shift parameters
        self.liquidity_drain_ratio = self.pattern_config.get("liquidity_drain_ratio", 0.6)
        self.wall_removal_drop = self.pattern_config.get("wall_removal_drop", 0.5)
        self.stacking_increase = self.pattern_config.get("stacking_increase", 0.5)
        # Volume profile dynamics
        self.poc_shift_threshold = self.pattern_config.get("poc_shift_threshold", 0.002)
        self.hvn_percentile = self.pattern_config.get("hvn_percentile", 80)
        # Ignition range thresholds
        self.ignition_narrow_env = self.pattern_config.get("ignition_narrow_env", 0.02)
        # Micro breakout window
        self.micro_breakout_window = self.pattern_config.get("micro_breakout_window", 3)

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

        # Simple state for evolving patterns
        self.state: dict[str, Any] = {
            "compression_streak": 0,
            "trend_cont_streak": 0,
        }

        # Canonical label list (single source of truth)
        self.all_labels: list[str] = [
            # Trend/price action
            "STRONG_TREND_CONTINUATION","EXHAUSTION_REVERSAL","RANGE_MEAN_REVERSION",
            "BREAKOUT_SUCCESS","BREAKOUT_FAILURE","FLAG_FORMATION","TRIANGLE_FORMATION","RECTANGLE_FORMATION",
            "MOMENTUM_IGNITION","IGNITION_BAR","MICRO_MOMENTUM_DIVERGENCE",
            # Volatility
            "VOLATILITY_COMPRESSION","VOLATILITY_EXPANSION","FLASH_CRASH_PATTERN",
            # Volume profile & auction
            "PRICE_AT_POC","PRICE_REJECTING_VAH","PRICE_REJECTING_VAL","LVN_TRANSIT","POC_SHIFT","HIGH_VOLUME_NODE_REJECTION",
            # S/R related
            "SR_TOUCH","SR_BOUNCE","SR_BREAK","SR_FAKE_BREAK",
            # Liquidity & market depth
            "BID_ASK_COMPRESSION","ICE_BERG_ORDERS","LIQUIDITY_DRAIN","BID_WALL_REMOVAL","OFFER_STACKING",
            # Traps & false signals
            "BULL_TRAP","BEAR_TRAP","FAKE_BREAKOUT",
            # Regime transitions
            "VOLATILITY_REGIME_CHANGE","TREND_TO_RANGE_TRANSITION",
            # Risk/conviction
            "CHOP_WARNING","FAKE_OUT_RISK_HIGH","LOW_CONVICTION_SETUP","HIGH_CONVICTION_SETUP",
            # Sentiment/momentum refinements
            "MOMENTUM_ACCELERATION","MOMENTUM_DECELERATION","CAPITULATION_SELLING","EUPHORIC_BUYING",
            # Order flow/stop dynamics
            "STOP_HUNT_BELOW_LOW","STOP_HUNT_ABOVE_HIGH","PASSIVE_ABSORPTION_BID","PASSIVE_ABSORPTION_ASK",
        ]

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
            self.logger.exception(
                f"Error initializing meta-labeling system: {e}",
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
                self.logger.exception(f"Error calculating technical indicators: {e}")

            # Volume analysis with error handling
            try:
                features.update(self._calculate_volume_features(volume_data))
            except Exception as e:
                self.logger.exception(f"Error calculating volume features: {e}")

            # Price action patterns with error handling
            try:
                features.update(self._calculate_price_action_patterns(price_data))
            except Exception as e:
                self.logger.exception(f"Error calculating price action patterns: {e}")

            # Volatility patterns with error handling
            try:
                features.update(self._calculate_volatility_patterns(price_data))
            except Exception as e:
                self.logger.exception(f"Error calculating volatility patterns: {e}")

            # Momentum patterns with error handling
            try:
                features.update(self._calculate_momentum_patterns(price_data))
            except Exception as e:
                self.logger.exception(f"Error calculating momentum patterns: {e}")

            # Support/Resistance levels
            try:
                features.update(self._calculate_sr_levels(price_data))
            except Exception as e:
                self.logger.exception(f"Error calculating S/R levels: {e}")

            # Expose current values for entry and meta usage
            try:
                features["current_price"] = float(price_data["close"].iloc[-1])
            except Exception:
                pass
            try:
                if "volume" in volume_data.columns:
                    vol_ma10 = volume_data["volume"].rolling(10, min_periods=1).mean().iloc[-1]
                    last_vol = volume_data["volume"].iloc[-1]
                    features["volume_spike"] = float(last_vol / vol_ma10) if vol_ma10 else 1.0
            except Exception:
                pass

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
            # Handle division by zero in RSI: if loss==0, set rs to a large finite value
            rs = (gain / loss.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            rs = rs.clip(upper=1e6)  # cap extreme ratios to avoid inf
            rsi_series = 100 - (100 / (1 + rs))
            if rsi_series.replace([np.inf, -np.inf], np.nan).notna().any():
                features["rsi"] = float(rsi_series.replace([np.inf, -np.inf], np.nan).fillna(100.0).iloc[-1])
            else:
                features["rsi"] = 50

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
            # Extended ATR windows
            try:
                features["atr_20"] = (
                    true_range.rolling(20).mean().iloc[-1]
                    if len(data) >= 20
                    else features["atr"]
                )
                features["atr_100"] = (
                    true_range.rolling(100).mean().iloc[-1]
                    if len(data) >= 100
                    else features["atr_20"]
                )
            except Exception:
                features.setdefault("atr_20", features.get("atr", 0.0))
                features.setdefault("atr_100", features.get("atr_20", 0.0))

            # ADX(14)
            try:
                up_move = data["high"].diff()
                down_move = -data["low"].diff()
                plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move
                minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move
                atr14 = true_range.rolling(14).mean()
                plus_di = 100 * (plus_dm.rolling(14).mean() / atr14.replace(0, np.nan))
                minus_di = 100 * (minus_dm.rolling(14).mean() / atr14.replace(0, np.nan))
                dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
                adx = dx.rolling(14).mean().replace([np.inf, -np.inf], np.nan).fillna(20)
                features["adx"] = float(adx.iloc[-1])
            except Exception:
                features["adx"] = 20.0

            return features

        except Exception as e:
            self.logger.exception(f"Error calculating technical indicators: {e}")
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
            self.logger.exception(f"Error calculating volume features: {e}")
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
            self.logger.exception(f"Error calculating price action patterns: {e}")
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
            self.logger.exception(f"Error calculating volatility patterns: {e}")
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
            self.logger.exception(f"Error calculating momentum patterns: {e}")
            return {"MOMENTUM_IGNITION": 0}

    def _detect_additional_analyst_patterns(
        self,
        data: pd.DataFrame,
        features: dict[str, Any],
        order_flow_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Detect additional analyst patterns requested by user.

        Uses the centralized 'features' dict; no re-computation.
        """
        try:
            patterns: dict[str, int] = {}

            # Aliases
            close = data["close"]
            high = data["high"]
            low = data["low"]
            open_ = data["open"]
            vol = data.get("volume", pd.Series(index=data.index, dtype=float))

            # Derived helpers from features
            bb_pos = float(features.get("bb_position", 0.5))
            vol_ratio = float(features.get("volume_ratio", 1.0))
            vol_20 = float(features.get("volatility_20", 0.0))
            rng10 = float((high.tail(10).max() - low.tail(10).min()) / max(1e-12, close.iloc[-1])) if len(close) >= 10 else 0.0
            momentum_5 = float(features.get("price_momentum_5", 0.0))
            momentum_10 = float(features.get("price_momentum_10", 0.0))

            # 1) ORDERBOOK_IMBALANCE_STRONG_* (requires order_flow_data; fallback heuristics set 0)
            patterns["ORDERBOOK_IMBALANCE_STRONG_BID"] = 0
            patterns["ORDERBOOK_IMBALANCE_STRONG_ASK"] = 0

            # 2) PASSIVE_ABSORPTION_* (proxy: large volume delta with small progress)
            try:
                # Price progress proxy over last N bars
                N = 5
                progress = abs(float(close.iloc[-1] - close.iloc[-N])) / max(1e-12, float(close.iloc[-N])) if len(close) >= N else 0.0
                vol_spike = vol_ratio > max(1.5, self.volume_threshold)
                patterns["PASSIVE_ABSORPTION_BID"] = 1 if (vol_spike and progress < 0.001 and momentum_5 > 0) else 0
                patterns["PASSIVE_ABSORPTION_ASK"] = 1 if (vol_spike and progress < 0.001 and momentum_5 < 0) else 0
            except Exception:
                patterns["PASSIVE_ABSORPTION_BID"] = 0
                patterns["PASSIVE_ABSORPTION_ASK"] = 0

            # 3) STOP_HUNT_* (poke beyond prior extreme and close back inside)
            try:
                M = 15
                prior_recent_low = float(low.rolling(M, min_periods=1).min().shift(1).iloc[-1]) if len(low) >= 2 else float(low.iloc[-1])
                prior_recent_high = float(high.rolling(M, min_periods=1).max().shift(1).iloc[-1]) if len(high) >= 2 else float(high.iloc[-1])
                broke_low = float(low.iloc[-1]) < prior_recent_low and float(close.iloc[-1]) > prior_recent_low
                broke_high = float(high.iloc[-1]) > prior_recent_high and float(close.iloc[-1]) < prior_recent_high
                vol_spike2 = vol_ratio > 2.0
                patterns["STOP_HUNT_BELOW_LOW"] = 1 if (broke_low and vol_spike2) else 0
                patterns["STOP_HUNT_ABOVE_HIGH"] = 1 if (broke_high and vol_spike2) else 0
            except Exception:
                patterns["STOP_HUNT_BELOW_LOW"] = 0
                patterns["STOP_HUNT_ABOVE_HIGH"] = 0

            # 4) PRICE_AT_POC / REJECTING_VAH/VAL / LVN_TRANSIT (session profile proxy)
            # Simple proxy without full profile: use rolling POC as 20-bin histogram on last 1d
            try:
                window = min(1440, len(close))
                if window >= 50:
                    segment = close.tail(window).values
                    bins = 20
                    hist, edges = np.histogram(segment, bins=bins)
                    poc_idx = int(np.argmax(hist))
                    poc = float((edges[poc_idx] + edges[poc_idx + 1]) / 2.0)
                    current = float(close.iloc[-1])
                    within = abs(current - poc) / max(1e-12, current) <= 0.0005
                    patterns["PRICE_AT_POC"] = 1 if within else 0
                    # Define VA as central 70% of mass
                    cdf = np.cumsum(hist) / max(1, np.sum(hist))
                    vah_idx = int(np.searchsorted(cdf, 0.85))
                    val_idx = int(np.searchsorted(cdf, 0.15))
                    vah = float(edges[min(vah_idx, bins - 1)])
                    val = float(edges[max(val_idx, 0)])
                    # Reject VAH/VAL: intrabar breach and close back inside
                    rejecting_vah = (float(high.iloc[-1]) > vah) and (float(close.iloc[-1]) < vah)
                    rejecting_val = (float(low.iloc[-1]) < val) and (float(close.iloc[-1]) > val)
                    patterns["PRICE_REJECTING_VAH"] = 1 if rejecting_vah else 0
                    patterns["PRICE_REJECTING_VAL"] = 1 if rejecting_val else 0
                    # LVN transit: current in low histogram bin region
                    lvn_threshold = np.percentile(hist, 20)
                    current_bin = int(np.searchsorted(edges, current) - 1)
                    patterns["LVN_TRANSIT"] = 1 if (0 <= current_bin < len(hist) and hist[current_bin] <= lvn_threshold) else 0
                else:
                    patterns["PRICE_AT_POC"] = 0
                    patterns["PRICE_REJECTING_VAH"] = 0
                    patterns["PRICE_REJECTING_VAL"] = 0
                    patterns["LVN_TRANSIT"] = 0
            except Exception:
                patterns["PRICE_AT_POC"] = 0
                patterns["PRICE_REJECTING_VAH"] = 0
                patterns["PRICE_REJECTING_VAL"] = 0
                patterns["LVN_TRANSIT"] = 0

            # 5) IGNITION_BAR
            try:
                rng = float((high.iloc[-1] - low.iloc[-1]) / max(1e-12, close.iloc[-1])) if len(close) else 0.0
                avg_rng = float((high - low).tail(20).mean() / max(1e-12, close.iloc[-1])) if len(close) >= 20 else rng
                vol_spike3 = vol_ratio > 3.0
                patterns["IGNITION_BAR"] = 1 if (vol_spike3 and rng > 2 * avg_rng and rng10 < 0.02) else 0
            except Exception:
                patterns["IGNITION_BAR"] = 0

            # 6) MICRO_MOMENTUM_DIVERGENCE
            try:
                made_higher_high = len(high) >= 10 and float(high.iloc[-1]) > float(high.iloc[-10:-1].max())
                # Compute a short RSI window to detect micro divergence
                delta = close.diff()
                gain = delta.where(delta > 0, 0.0).rolling(5, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0.0)).rolling(5, min_periods=1).mean()
                with np.errstate(divide='ignore', invalid='ignore'):
                    rs = gain / loss.replace(0, np.nan)
                    rsi_short_series = 100 - (100 / (1 + rs))
                rsi_short_series = rsi_short_series.fillna(50)
                rsi_short_now = float(rsi_short_series.iloc[-1]) if len(rsi_short_series) else 50.0
                rsi_window_max = float(
                    rsi_short_series.rolling(10, min_periods=2).max().shift(1).iloc[-1]
                ) if len(rsi_short_series) >= 2 else 50.0
                patterns["MICRO_MOMENTUM_DIVERGENCE"] = 1 if (made_higher_high and rsi_short_now < rsi_window_max) else 0
            except Exception:
                patterns["MICRO_MOMENTUM_DIVERGENCE"] = 0

            # 7) S/R related labels (using rolling extremes as SR proxies)
            try:
                sr_res = float(features.get("resistance_level", high.max()))
                sr_sup = float(features.get("support_level", low.min()))
                cp = float(close.iloc[-1])
                near_res = abs(cp - sr_res) / max(1e-12, cp) <= self.sr_near_pct
                near_sup = abs(cp - sr_sup) / max(1e-12, cp) <= self.sr_near_pct
                prev_cp = float(close.iloc[-2]) if len(close) >= 2 else cp
                broke_res = (prev_cp <= sr_res) and (cp > sr_res * (1 + self.sr_break_pct))
                broke_sup = (prev_cp >= sr_sup) and (cp < sr_sup * (1 - self.sr_break_pct))
                bounce_res = (near_res and cp < sr_res)
                bounce_sup = (near_sup and cp > sr_sup)
                patterns["SR_TOUCH"] = 1 if (near_res or near_sup) else 0
                patterns["SR_BOUNCE"] = 1 if (bounce_res or bounce_sup) else 0
                patterns["SR_BREAK"] = 1 if (broke_res or broke_sup) else 0
                # SR_FAKE_BREAK: pierce and close back inside within last 3 bars
                recent = close.tail(3)
                fake_up = (recent.max() > sr_res * (1 + self.sr_break_pct)) and (cp < sr_res)
                fake_dn = (recent.min() < sr_sup * (1 - self.sr_break_pct)) and (cp > sr_sup)
                patterns["SR_FAKE_BREAK"] = 1 if (fake_up or fake_dn) else 0
            except Exception:
                patterns["SR_TOUCH"] = patterns.get("SR_TOUCH", 0)
                patterns["SR_BOUNCE"] = patterns.get("SR_BOUNCE", 0)
                patterns["SR_BREAK"] = patterns.get("SR_BREAK", 0)
                patterns["SR_FAKE_BREAK"] = patterns.get("SR_FAKE_BREAK", 0)

            # 8) Liquidity shifts & market depth (requires order_flow_data; use proxies if limited)
            try:
                # LIQUIDITY_DRAIN: total depth falls below x% of its rolling mean
                if order_flow_data is not None and {"bid_depth", "ask_depth"}.issubset(order_flow_data.columns):
                    bd = order_flow_data["bid_depth"].reindex(close.index, method="ffill").fillna(0)
                    ad = order_flow_data["ask_depth"].reindex(close.index, method="ffill").fillna(0)
                    tot = bd + ad
                    ma = tot.rolling(20, min_periods=5).mean()
                    drain = float(tot.iloc[-1] / max(1e-12, ma.iloc[-1])) < self.liquidity_drain_ratio
                    patterns["LIQUIDITY_DRAIN"] = 1 if drain else 0
                    # BID_WALL_REMOVAL: large drop in bid wall size
                    if "bid_size" in order_flow_data.columns:
                        bs = order_flow_data["bid_size"].reindex(close.index, method="ffill").fillna(0)
                        drop = (bs.diff().iloc[-1] < -abs(bs.rolling(10, min_periods=3).mean().iloc[-1] * self.wall_removal_drop))
                        patterns["BID_WALL_REMOVAL"] = 1 if drop else 0
                    else:
                        patterns["BID_WALL_REMOVAL"] = 0
                    # OFFER_STACKING: rapid increase in ask depth
                    if "ask_size" in order_flow_data.columns:
                        aS = order_flow_data["ask_size"].reindex(close.index, method="ffill").fillna(0)
                        inc = (aS.diff().iloc[-1] > abs(aS.rolling(10, min_periods=3).mean().iloc[-1] * self.stacking_increase))
                        patterns["OFFER_STACKING"] = 1 if inc else 0
                    else:
                        patterns["OFFER_STACKING"] = 0
                else:
                    patterns.setdefault("LIQUIDITY_DRAIN", 0)
                    patterns.setdefault("BID_WALL_REMOVAL", 0)
                    patterns.setdefault("OFFER_STACKING", 0)
            except Exception:
                patterns["LIQUIDITY_DRAIN"] = patterns.get("LIQUIDITY_DRAIN", 0)
                patterns["BID_WALL_REMOVAL"] = patterns.get("BID_WALL_REMOVAL", 0)
                patterns["OFFER_STACKING"] = patterns.get("OFFER_STACKING", 0)

            # 9) False signals & traps
            try:
                recent_high = float(high.rolling(self.lookback_breakout, min_periods=5).max().shift(1).iloc[-1]) if len(high) else float(high.iloc[-1])
                recent_low = float(low.rolling(self.lookback_breakout, min_periods=5).min().shift(1).iloc[-1]) if len(low) else float(low.iloc[-1])
                broke_up = float(close.iloc[-1]) > recent_high
                broke_dn = float(close.iloc[-1]) < recent_low
                mom = float(features.get("price_momentum_5", 0.0))
                # Trap if break occurs but momentum negative (bull trap) or positive (bear trap) and reversal fast
                rev5 = float(close.pct_change(5).iloc[-1]) if len(close) >= 6 else 0.0
                patterns["BULL_TRAP"] = 1 if (broke_up and mom < 0 and rev5 < 0) else 0
                patterns["BEAR_TRAP"] = 1 if (broke_dn and mom > 0 and rev5 > 0) else 0
                # Fake breakout: break then bb_position retreats to mid within 5 bars
                bb_pos = float(features.get("bb_position", 0.5))
                patterns["FAKE_BREAKOUT"] = 1 if ((broke_up or broke_dn) and 0.3 < bb_pos < 0.7) else 0
            except Exception:
                patterns["BULL_TRAP"] = patterns.get("BULL_TRAP", 0)
                patterns["BEAR_TRAP"] = patterns.get("BEAR_TRAP", 0)
                patterns["FAKE_BREAKOUT"] = patterns.get("FAKE_BREAKOUT", 0)

            # 10) Volume profile & auction dynamics extensions
            try:
                # POC_SHIFT: change in POC level vs prior window
                window = min(self.lookback_poc_window, len(close))
                if window >= 50:
                    seg = close.tail(window).values
                    hist, edges = np.histogram(seg, bins=self.lookback_poc_bins)
                    poc_idx = int(np.argmax(hist))
                    poc = float((edges[poc_idx] + edges[poc_idx + 1]) / 2.0)
                    # Prior window
                    prev_seg = close.tail(window + 50).head(window).values if len(close) >= window + 50 else seg
                    prev_hist, prev_edges = np.histogram(prev_seg, bins=self.lookback_poc_bins)
                    prev_poc_idx = int(np.argmax(prev_hist))
                    prev_poc = float((prev_edges[prev_poc_idx] + prev_edges[prev_poc_idx + 1]) / 2.0)
                    patterns["POC_SHIFT"] = 1 if (abs(poc - prev_poc) / max(1e-12, prev_poc) >= self.poc_shift_threshold) else 0
                    # HVN rejection: current price near HVN bin edge then reverse
                    hvn_cut = np.percentile(hist, self.hvn_percentile)
                    current = float(close.iloc[-1])
                    cur_bin = int(np.searchsorted(edges, current) - 1)
                    hvn_hit = 0 <= cur_bin < len(hist) and hist[cur_bin] >= hvn_cut
                    patterns["HIGH_VOLUME_NODE_REJECTION"] = 1 if (hvn_hit and (abs(float(close.pct_change(3).iloc[-1]) if len(close) >= 4 else 0.0) > 0)) else 0
                else:
                    patterns.setdefault("POC_SHIFT", 0)
                    patterns.setdefault("HIGH_VOLUME_NODE_REJECTION", 0)
            except Exception:
                patterns["POC_SHIFT"] = patterns.get("POC_SHIFT", 0)
                patterns["HIGH_VOLUME_NODE_REJECTION"] = patterns.get("HIGH_VOLUME_NODE_REJECTION", 0)

            # 11) Market regime transitions
            try:
                vol20 = float(features.get("volatility_20", 0.0))
                vol10 = float(features.get("volatility_10", 0.0))
                prev_regime = int(self.state.get("vol_regime", 0))
                curr_regime = 1 if vol20 > self.volatility_threshold else 0
                patterns["VOLATILITY_REGIME_CHANGE"] = 1 if curr_regime != prev_regime else 0
                self.state["vol_regime"] = curr_regime
                # Trend-to-range transition: ADX falls and bb_width expands into mid
                adx = float(features.get("adx", 20.0))
                bb_width = float(features.get("bb_width", 0.0))
                patterns["TREND_TO_RANGE_TRANSITION"] = 1 if (adx < 20 and bb_width > self.bb_width_compression) else 0
            except Exception:
                patterns["VOLATILITY_REGIME_CHANGE"] = patterns.get("VOLATILITY_REGIME_CHANGE", 0)
                patterns["TREND_TO_RANGE_TRANSITION"] = patterns.get("TREND_TO_RANGE_TRANSITION", 0)

            return patterns

        except Exception as e:
            self.logger.exception(f"Error detecting additional analyst patterns: {e}")
            return {
                "ORDERBOOK_IMBALANCE_STRONG_BID": 0,
                "ORDERBOOK_IMBALANCE_STRONG_ASK": 0,
                "PASSIVE_ABSORPTION_BID": 0,
                "PASSIVE_ABSORPTION_ASK": 0,
                "STOP_HUNT_BELOW_LOW": 0,
                "STOP_HUNT_ABOVE_HIGH": 0,
                "PRICE_AT_POC": 0,
                "PRICE_REJECTING_VAH": 0,
                "PRICE_REJECTING_VAL": 0,
                "LVN_TRANSIT": 0,
                "IGNITION_BAR": 0,
                "MICRO_MOMENTUM_DIVERGENCE": 0,
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
            is_uptrend = trend_strength > self.trend_momentum_strong
            is_pullback = self.bb_mid_low < bb_position < self.bb_mid_high
            is_healthy_rsi = (self.rsi_oversold + 10) < rsi < (self.rsi_overbought)

            strong_trend_continuation = is_uptrend and is_pullback and is_healthy_rsi

            return {
                "STRONG_TREND_CONTINUATION": 1 if strong_trend_continuation else 0,
                "strong_trend_confidence": min(abs(trend_strength) * 10, 1.0)
                if strong_trend_continuation
                else 0,
            }

        except Exception as e:
            self.logger.exception(f"Error detecting strong trend continuation: {e}")
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
            is_overbought = rsi > self.rsi_overbought or bb_position > self.bb_edge_high
            is_weakening = momentum < 0
            is_high_volume = volume_ratio > self.volume_threshold

            exhaustion_reversal = is_overbought and is_weakening and is_high_volume

            return {
                "EXHAUSTION_REVERSAL": 1 if exhaustion_reversal else 0,
                "exhaustion_confidence": min((rsi - 70) / 30, 1.0)
                if exhaustion_reversal
                else 0,
            }

        except Exception as e:
            self.logger.exception(f"Error detecting exhaustion reversal: {e}")
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
            is_at_edge = bb_position < self.bb_edge_low or bb_position > self.bb_edge_high
            is_low_volatility = volatility < self.volatility_threshold
            is_sideways = abs(features.get("price_momentum_10", 0)) < self.momentum_threshold

            range_mean_reversion = is_at_edge and is_low_volatility and is_sideways

            return {
                "RANGE_MEAN_REVERSION": 1 if range_mean_reversion else 0,
                "range_reversion_confidence": min(abs(bb_position - 0.5) * 2, 1.0)
                if range_mean_reversion
                else 0,
            }

        except Exception as e:
            self.logger.exception(f"Error detecting range mean reversion: {e}")
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
            current_price = float(data["close"].iloc[-1])

            # Use rolling highs/lows excluding current bar to avoid lookahead
            rolling_high = data["high"].rolling(20, min_periods=1).max().shift(1)
            rolling_low = data["low"].rolling(20, min_periods=1).min().shift(1)
            recent_high = (
                float(rolling_high.iloc[-1]) if not rolling_high.empty else current_price
            )
            recent_low = (
                float(rolling_low.iloc[-1]) if not rolling_low.empty else current_price
            )

            # Breakout success: price breaks prior level and continues with momentum and volume
            is_breakout_up = current_price > recent_high and momentum > self.breakout_momentum
            is_breakout_down = current_price < recent_low and momentum < -self.breakout_momentum
            is_high_volume = volume_ratio > self.volume_threshold
            breakout_success = (is_breakout_up or is_breakout_down) and is_high_volume

            # Breakout failure: price pushes to BB extremes but lacks follow-through
            is_failed_breakout = (bb_position > self.bb_edge_high or bb_position < self.bb_edge_low) and abs(momentum) < self.failed_break_momentum

            return {
                "BREAKOUT_SUCCESS": 1 if breakout_success else 0,
                "BREAKOUT_FAILURE": 1 if is_failed_breakout else 0,
                "breakout_confidence": min(volume_ratio / 2, 1.0) if breakout_success else 0,
            }

        except Exception as e:
            self.logger.exception(f"Error detecting breakout patterns: {e}")
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
            is_compression = bb_width < self.bb_width_compression and volatility_ratio < 0.8

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
            self.logger.exception(f"Error detecting volatility patterns: {e}")
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
            is_triangle = bb_width < self.bb_width_triangle
            patterns["TRIANGLE_FORMATION"] = 1 if is_triangle else 0

            # Rectangle formation: horizontal consolidation
            price_position = features.get("price_position", 0.5)
            is_rectangle = 0.3 < price_position < 0.7 and bb_width < 0.05
            patterns["RECTANGLE_FORMATION"] = 1 if is_rectangle else 0

            return patterns

        except Exception as e:
            self.logger.exception(f"Error detecting chart patterns: {e}")
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
            macd_hist = features.get("macd_histogram", 0)
            momentum = features.get("price_momentum_5", 0)

            is_momentum_ignition = (
                (rsi > self.rsi_momentum_hi or rsi < self.rsi_momentum_lo)
                and abs(macd_hist) > 0.001
                and abs(momentum) > self.momentum_threshold
            )
            patterns["MOMENTUM_IGNITION"] = 1 if is_momentum_ignition else 0

            return patterns

        except Exception as e:
            self.logger.exception(f"Error detecting momentum patterns: {e}")
            return {"MOMENTUM_IGNITION": 0}

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
            self.logger.exception(f"Error calculating entry features: {e}")
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
            self.logger.exception(f"Error calculating order imbalance: {e}")
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
            self.logger.exception(f"Error predicting price extremes: {e}")
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
            self.logger.exception(f"Error predicting order returns: {e}")
            return {"LIMIT_ORDER_RETURN": 0.001, "limit_order_confidence": 0}

    def _detect_entry_signals(
        self,
        features: dict[str, Any],
        order_flow_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Detect various entry signals."""
        try:
            signals = {}

            # VWAP reversion entry (use rolling VWAP from features if available)
            current_price = float(features.get("current_price", 0.0) or features.get("close", 0.0) or 0.0)
            vwap = float(features.get("vwap", current_price))
            price_vwap_ratio = current_price / vwap if vwap != 0 else 1.0
            is_vwap_reversion = abs(price_vwap_ratio - 1.0) < 0.01
            signals["VWAP_REVERSION_ENTRY"] = 1 if is_vwap_reversion else 0

            # Market order now: aggressive momentum with volume spike
            momentum = float(features.get("price_momentum_5", 0.0))
            volume_spike = float(features.get("volume_spike", features.get("volume_ratio", 1.0)))
            is_market_order = (
                abs(momentum) > self.momentum_threshold * 2
                and volume_spike > self.volume_threshold
            )
            signals["MARKET_ORDER_NOW"] = 1 if is_market_order else 0

            # Chase micro breakout: break last N bars' high/low
            prev_high = float(features.get("recent_high", current_price))
            prev_low = float(features.get("recent_low", current_price))
            is_micro_breakout = (current_price > prev_high) or (current_price < prev_low)
            signals["CHASE_MICRO_BREAKOUT"] = 1 if is_micro_breakout else 0

            # Order book imbalance flip (requires order flow)
            is_imbalance_flip = False
            if order_flow_data is not None and {"bid_volume", "ask_volume"}.issubset(order_flow_data.columns):
                try:
                    last_b = float(order_flow_data["bid_volume"].iloc[-1])
                    last_a = float(order_flow_data["ask_volume"].iloc[-1])
                    prev_b = float(order_flow_data["bid_volume"].iloc[-2]) if len(order_flow_data) >= 2 else last_b
                    prev_a = float(order_flow_data["ask_volume"].iloc[-2]) if len(order_flow_data) >= 2 else last_a
                    prev_imb = (prev_b - prev_a) / max(1e-12, (prev_b + prev_a))
                    last_imb = (last_b - last_a) / max(1e-12, (last_b + last_a))
                    is_imbalance_flip = (prev_imb > 0 and last_imb < 0) or (prev_imb < 0 and last_imb > 0)
                except Exception:
                    is_imbalance_flip = False
            signals["ORDERBOOK_IMBALANCE_FLIP"] = 1 if is_imbalance_flip else 0

            # Aggressive taker spike
            is_taker_spike = volume_spike > (self.volume_threshold * 2)
            signals["AGGRESSIVE_TAKER_SPIKE"] = 1 if is_taker_spike else 0

            return signals

        except Exception as e:
            self.logger.exception(f"Error detecting entry signals: {e}")
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
            self.logger.exception(f"Error predicting adverse excursion: {e}")
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
            self.logger.exception(f"Error generating abort signal: {e}")
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
        order_flow_data: pd.DataFrame | None = None,
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
                self.logger.error("Meta-labeling system not initialized")
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
            additional_patterns = self._detect_additional_analyst_patterns(price_data, features, order_flow_data)
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
                    "compression_streak": int(self.state.get("compression_streak", 0)),
                    "label_count": len(
                        [
                            v
                            for v in analyst_labels.values()
                            if isinstance(v, int | float) and v > 0
                        ],
                    ),
                },
            )

            # Compute intensities and activations
            label_keys = [k for k, v in analyst_labels.items() if isinstance(v, (int, float)) and k.isupper() and k not in ("label_count",)]
            intensities = self.compute_intensity_scores(price_data, volume_data, features, label_keys)
            if not hasattr(self, "activation_thresholds"):
                self._init_thresholds_and_reliability()
            activations = {k: 1 if intensities.get(k, 0.0) >= self.activation_thresholds.get(k, self.default_activation_threshold) else 0 for k in label_keys}
            analyst_labels.update({f"intensity_{k}": float(intensities.get(k, 0.0)) for k in label_keys})
            analyst_labels.update({f"active_{k}": int(activations.get(k, 0)) for k in label_keys})

            self.logger.info(
                f"Generated {analyst_labels.get('label_count', 0)} analyst labels for {timeframe}",
            )
            return analyst_labels

        except Exception as e:
            self.logger.exception(f"Error generating analyst labels: {e}")
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
                self.logger.error("Meta-labeling system not initialized")
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
                entry_features,
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
            self.logger.exception(f"Error generating tactician labels: {e}")
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
                order_flow_data,
            )

            # Generate tactician labels
            tactician_labels = await self.generate_tactician_labels(
                price_data,
                volume_data,
                order_flow_data,
                tactician_timeframe,
            )

            # Combine labels and compute MoE-based weights placeholder
            label_keys = [k for k, v in analyst_labels.items() if k.isupper() and isinstance(v, (int, float))]
            intensities = {k: analyst_labels.get(f"intensity_{k}", 0.0) for k in label_keys}
            moe_confidences: dict[str, float] = {k: 0.5 for k in label_keys}  # placeholder; replace with real MoE output
            weights = self.compute_label_weights(intensities, moe_confidences)

            avg_conf = float(np.mean(list(weights.values()))) if weights else 0.0
            combined_labels = {
                "analyst_labels": analyst_labels,
                "tactician_labels": tactician_labels,
                "weights": weights,
                "combined_confidence": avg_conf,
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
            self.logger.exception(f"Error generating combined labels: {e}")
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
            self.logger.exception(f"Error stopping meta-labeling system: {e}")

    # ===== Intensity Scoring and Thresholding Utilities =====

    def _linear_scale(self, value: float, vmin: float, vmax: float) -> float:
        if vmax == vmin:
            return 0.0
        return float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))

    def _percentile_rank(self, series: pd.Series, value: float) -> float:
        try:
            s = series.replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                return 0.0
            prc = float((s <= value).mean())
            return float(np.clip(prc, 0.0, 1.0))
        except Exception:
            return 0.0

    def _compute_label_intensity(
        self,
        label_name: str,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        features: dict[str, Any],
    ) -> float:
        """Compute a 0-1 intensity score per label using bounded linear scaling and percentile ranking for unbounded metrics."""
        try:
            close = price_data["close"]
            volume = volume_data["volume"] if "volume" in volume_data.columns else pd.Series(index=close.index, dtype=float)
            # Common series
            vol_ratio_series = (volume / volume.rolling(20, min_periods=5).mean()).replace([np.inf, -np.inf], np.nan)
            momentum5_series = close.pct_change(5).rolling(5, min_periods=1).sum()

            if label_name == "STRONG_TREND_CONTINUATION":
                trend = float(features.get("price_momentum_10", 0.0))
                bb_pos = float(features.get("bb_position", 0.5))
                rsi = float(features.get("rsi", 50.0))
                part_trend = self._percentile_rank(close.pct_change(10), trend)
                part_bb = 1.0 - abs(bb_pos - 0.5) / 0.5
                part_rsi = 1.0 - abs((rsi - (self.rsi_oversold + self.rsi_overbought) / 2) / 50.0)
                return float(np.clip((part_trend + part_bb + part_rsi) / 3.0, 0.0, 1.0))

            if label_name == "EXHAUSTION_REVERSAL":
                rsi = float(features.get("rsi", 50.0))
                bb_pos = float(features.get("bb_position", 0.5))
                momentum = float(features.get("price_momentum_5", 0.0))
                vol_ratio = float(features.get("volume_ratio", 1.0))
                part_rsi = self._linear_scale(rsi, self.rsi_overbought, 100.0)
                part_bb = self._linear_scale(bb_pos, self.bb_edge_high, 1.0)
                part_mom = self._percentile_rank(-momentum5_series, -momentum)
                part_vol = self._percentile_rank(vol_ratio_series, vol_ratio)
                return float(np.clip((part_rsi + part_bb + part_mom + part_vol) / 4.0, 0.0, 1.0))

            if label_name == "RANGE_MEAN_REVERSION":
                bb_pos = float(features.get("bb_position", 0.5))
                vol20 = float(features.get("volatility_20", 0.0))
                mom10 = float(features.get("price_momentum_10", 0.0))
                part_edge = max(self._linear_scale(self.bb_edge_low - bb_pos, 0.0, self.bb_edge_low), self._linear_scale(bb_pos - self.bb_edge_high, 0.0, 1.0 - self.bb_edge_high))
                part_vol = 1.0 - self._percentile_rank(close.pct_change().rolling(20).std(), vol20)
                part_sideways = 1.0 - self._percentile_rank(abs(close.pct_change(10)), abs(mom10))
                return float(np.clip((part_edge + part_vol + part_sideways) / 3.0, 0.0, 1.0))

            if label_name in ("BREAKOUT_SUCCESS", "BREAKOUT_FAILURE"):
                momentum = float(features.get("price_momentum_5", 0.0))
                vol_ratio = float(features.get("volume_ratio", 1.0))
                part_mom = self._percentile_rank(abs(momentum5_series), abs(momentum))
                part_vol = self._percentile_rank(vol_ratio_series, vol_ratio)
                base = float(np.clip((part_mom + part_vol) / 2.0, 0.0, 1.0))
                return base

            if label_name in ("VOLATILITY_COMPRESSION", "VOLATILITY_EXPANSION"):
                bb_width = float(features.get("bb_width", 0.0))
                vol_ratio = float(features.get("volatility_ratio", 1.0))
                if label_name == "VOLATILITY_COMPRESSION":
                    part_width = 1.0 - self._percentile_rank(price_data["close"].rolling(20).std(), bb_width)
                    part_vr = 1.0 - self._linear_scale(vol_ratio, 1.0, 2.0)
                else:
                    part_width = self._percentile_rank(price_data["close"].rolling(20).std(), bb_width)
                    part_vr = self._linear_scale(vol_ratio, 1.0, 2.0)
                return float(np.clip((part_width + part_vr) / 2.0, 0.0, 1.0))

            if label_name == "MOMENTUM_IGNITION":
                rsi = float(features.get("rsi", 50.0))
                macd_hist = float(features.get("macd_histogram", 0.0))
                momentum = float(features.get("price_momentum_5", 0.0))
                part_rsi = max(self._linear_scale(rsi, self.rsi_momentum_hi, 100.0), self._linear_scale(100.0 - rsi, self.rsi_momentum_lo, 100.0))
                part_macd = self._percentile_rank(price_data["close"].ewm(span=12).mean() - price_data["close"].ewm(span=26).mean(), macd_hist)
                part_mom = self._percentile_rank(abs(momentum5_series), abs(momentum))
                return float(np.clip((part_rsi + part_macd + part_mom) / 3.0, 0.0, 1.0))

            if label_name == "CAPITULATION_SELLING":
                p1 = float(close.pct_change(1).iloc[-1]) if len(close) >= 2 else 0.0
                p3 = float(close.pct_change(3).iloc[-1]) if len(close) >= 4 else 0.0
                rsi = float(features.get("rsi", 50.0))
                volr = float(features.get("volume_ratio", 1.0))
                part_drop = max(self._linear_scale(-p1, 0.0, 0.03), self._linear_scale(-p3, 0.0, 0.05))
                part_vol = self._percentile_rank(vol_ratio_series, volr)
                part_rsi = self._linear_scale(30.0 - rsi, 0.0, 30.0)
                return float(np.clip((part_drop + part_vol + part_rsi) / 3.0, 0.0, 1.0))

            if label_name == "EUPHORIC_BUYING":
                p1 = float(close.pct_change(1).iloc[-1]) if len(close) >= 2 else 0.0
                p3 = float(close.pct_change(3).iloc[-1]) if len(close) >= 4 else 0.0
                rsi = float(features.get("rsi", 50.0))
                volr = float(features.get("volume_ratio", 1.0))
                part_up = max(self._linear_scale(p1, 0.0, 0.03), self._linear_scale(p3, 0.0, 0.05))
                part_vol = self._percentile_rank(vol_ratio_series, volr)
                part_rsi = self._linear_scale(rsi - 70.0, 0.0, 30.0)
                return float(np.clip((part_up + part_vol + part_rsi) / 3.0, 0.0, 1.0))

            if label_name == "DULL_MARKET":
                atr20 = float(features.get("atr_20", features.get("atr", 0.0)))
                atr100 = float(features.get("atr_100", max(atr20, 1e-12)))
                ratio = atr20 / max(atr100, 1e-12)
                return float(np.clip(1.0 - ratio, 0.0, 1.0))

            if label_name == "FAILED_RETEST":
                # Use proximity to prior extremes and reversal size
                h = price_data["high"].rolling(50, min_periods=5).max().shift(1)
                l = price_data["low"].rolling(50, min_periods=5).min().shift(1)
                cp = float(close.iloc[-1])
                ph = float(h.iloc[-1]) if not h.empty else cp
                pl = float(l.iloc[-1]) if not l.empty else cp
                prox_up = 1.0 - min(abs(cp - ph) / max(ph, 1e-12) / 0.0015, 1.0)
                prox_dn = 1.0 - min(abs(cp - pl) / max(pl, 1e-12) / 0.0015, 1.0)
                return float(np.clip(max(prox_up, prox_dn), 0.0, 1.0))

            # Fallback: 1.0 if label present else 0.0
            return 1.0 if features.get(label_name, 0) else 0.0
        except Exception:
            return 0.0

    def compute_intensity_scores(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        features: dict[str, Any],
        label_names: list[str],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for name in label_names:
            scores[name] = self._compute_label_intensity(name, price_data, volume_data, features)
        return scores

    # Activation thresholds and reliability
    def _init_thresholds_and_reliability(self) -> None:
        self.activation_thresholds: dict[str, float] = {}
        self.reliability_scores: dict[str, float] = {}
        self.default_activation_threshold: float = float(self.pattern_config.get("default_activation_threshold", 0.5))

    def set_activation_threshold(self, label: str, threshold: float) -> None:
        if not hasattr(self, "activation_thresholds"):
            self._init_thresholds_and_reliability()
        self.activation_thresholds[label] = float(np.clip(threshold, 0.0, 1.0))

    def set_reliability_score(self, label: str, reliability: float) -> None:
        if not hasattr(self, "reliability_scores"):
            self._init_thresholds_and_reliability()
        self.reliability_scores[label] = float(np.clip(reliability, 0.0, 1.0))

    def compute_label_weights(
        self,
        intensity_scores: dict[str, float],
        moe_confidences: dict[str, float] | None = None,
    ) -> dict[str, float]:
        if not hasattr(self, "reliability_scores"):
            self._init_thresholds_and_reliability()
        weights: dict[str, float] = {}
        for label, intensity in intensity_scores.items():
            conf = float(np.clip((moe_confidences or {}).get(label, 0.5), 0.0, 1.0))
            rel = float(np.clip(self.reliability_scores.get(label, 1.0), 0.0, 1.0))
            weights[label] = float(np.clip(intensity * conf * rel, 0.0, 1.0))
        return weights

    def fit_activation_thresholds(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        triple_barrier_df: pd.DataFrame,
        label_names: list[str],
        thresholds_grid: list[float] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Derive activation thresholds per label using triple-barrier outcomes.

        Returns a dict per label with stats: threshold, hit_rate, frequency, profit_factor, avg_return.
        """
        try:
            if thresholds_grid is None:
                thresholds_grid = [round(x, 2) for x in np.linspace(0.1, 0.9, 9)]
            # Prepare outcomes and proxy returns if needed
            labels_series = triple_barrier_df.get("label", pd.Series(index=price_data.index, dtype=float)).reindex(price_data.index).fillna(0).astype(int)
            # If available, use realized return; else next-bar return as proxy
            if "tb_return" in triple_barrier_df.columns:
                ret_series = triple_barrier_df["tb_return"].reindex(price_data.index).fillna(0.0)
            else:
                ret_series = price_data["close"].pct_change().shift(-1).reindex(price_data.index).fillna(0.0)

            best: dict[str, dict[str, float]] = {}
            # Iterate bars to compute intensities
            for label in label_names:
                # Score per bar
                scores: list[float] = []
                for i in range(len(price_data)):
                    p_slice = price_data.iloc[: i + 1]
                    v_slice = volume_data.iloc[: i + 1]
                    feats = {}
                    try:
                        feats.update(self._calculate_technical_indicators(p_slice))
                        feats.update(self._calculate_volume_features(v_slice))
                        feats.update(self._calculate_price_action_patterns(p_slice))
                        feats.update(self._calculate_volatility_patterns(p_slice))
                        feats.update(self._calculate_momentum_patterns(p_slice))
                    except Exception:
                        pass
                    scores.append(self._compute_label_intensity(label, p_slice, v_slice, feats))
                scores_series = pd.Series(scores, index=price_data.index)

                # Evaluate thresholds
                best_threshold = self.default_activation_threshold
                best_pf = -np.inf
                best_stats = {"threshold": best_threshold, "hit_rate": 0.0, "frequency": 0.0, "profit_factor": 0.0, "avg_return": 0.0}
                for thr in thresholds_grid:
                    triggers = scores_series >= thr
                    freq = float(triggers.mean())
                    if freq <= 0:
                        continue
                    y = labels_series[triggers]
                    r = ret_series[triggers]
                    wins = r[y == 1]
                    losses = r[y == -1]
                    hit_rate = float((y == 1).mean()) if len(y) else 0.0
                    gross = float(wins.clip(lower=0).sum())
                    loss = float(-losses.clip(upper=0).sum())
                    profit_factor = (gross / loss) if loss > 0 else (gross if gross > 0 else 0.0)
                    avg_return = float(r.mean()) if len(r) else 0.0
                    if profit_factor > best_pf and freq > 0.001:
                        best_pf = profit_factor
                        best_threshold = float(thr)
                        best_stats = {
                            "threshold": best_threshold,
                            "hit_rate": float(hit_rate),
                            "frequency": float(freq),
                            "profit_factor": float(profit_factor),
                            "avg_return": float(avg_return),
                        }
                self.set_activation_threshold(label, best_threshold)
                best[label] = best_stats
            return best
        except Exception as e:
            self.logger.warning(f"Activation threshold fitting failed: {e}")
            return {}

    def _calculate_sr_levels(self, data: pd.DataFrame) -> dict[str, float]:
        """Compute simple S/R levels using rolling extremes as proxies and their distances."""
        out: dict[str, float] = {}
        look = max(5, int(self.sr_lookback))
        high_roll = data["high"].rolling(look, min_periods=5).max().shift(1)
        low_roll = data["low"].rolling(look, min_periods=5).min().shift(1)
        try:
            out["resistance_level"] = float(high_roll.iloc[-1]) if not high_roll.empty else float(data["high"].iloc[-1])
            out["support_level"] = float(low_roll.iloc[-1]) if not low_roll.empty else float(data["low"].iloc[-1])
            cp = float(data["close"].iloc[-1])
            out["dist_to_resistance_pct"] = float(abs(cp - out["resistance_level"]) / max(1e-12, cp))
            out["dist_to_support_pct"] = float(abs(cp - out["support_level"]) / max(1e-12, cp))
        except Exception:
            pass
        return out
