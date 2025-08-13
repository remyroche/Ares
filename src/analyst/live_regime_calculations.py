# src/analyst/live_regime_calculations.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


@dataclass(frozen=True)
class RegimeSummary:
    trend: str  # "BULL_TREND" | "BEAR_TREND" | "SIDEWAYS_RANGE"
    trend_confidence: float
    volatility_regime: str  # "LOW" | "NORMAL" | "HIGH" | "EXTREME"
    sr_context: str  # "NEAR_SUPPORT" | "NEAR_RESISTANCE" | "MID_RANGE"


class LiveRegimeCalculator:
    """
    Compute live-trading features once and derive unified regime flags.

    This consolidates core indicators used across modules (trend, volatility,
    momentum, volume, S/R proximity) into a single, fast computation path.

    Expected input: DataFrame with columns ['open','high','low','close','volume']
    indexed by a DatetimeIndex (ascending).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.logger = system_logger.getChild("LiveRegimeCalculator")

        # Thresholds (aligned with existing modules defaults where possible)
        analyst_cfg = (
            self.config.get("analyst", {}).get("unified_regime_classifier", {})
        )
        pattern_cfg = (
            self.config.get("meta_labeling", {}).get("pattern_detection", {})
        )

        # Trend/ADX thresholds
        self.adx_sideways_threshold: float = float(
            analyst_cfg.get("adx_sideways_threshold", 18.0)
        )
        self.adx_trend_threshold: float = float(
            analyst_cfg.get("adx_trend_threshold", 25.0)
        )
        self.ema_fast: int = int(pattern_cfg.get("ema_fast", 21))
        self.ema_slow: int = int(pattern_cfg.get("ema_slow", 55))
        self.ema_sep_min_ratio: float = float(pattern_cfg.get("ema_sep_min_ratio", 0.0))

        # Volatility
        self.volatility_threshold: float = float(
            analyst_cfg.get("volatility_threshold", 0.020)
        )
        # For MarketHealth-like regime classification bands
        self.vol_regime_bands: Tuple[float, float, float] = (
            float(pattern_cfg.get("vol_low_max", 0.02)),
            float(pattern_cfg.get("vol_normal_max", 0.04)),
            float(pattern_cfg.get("vol_high_max", 0.08)),
        )

        # Momentum / Breakout
        self.momentum_threshold: float = float(
            pattern_cfg.get("momentum_threshold", 0.01)
        )
        self.breakout_momentum: float = float(
            pattern_cfg.get("breakout_momentum", 0.01)
        )

        # Volume
        self.volume_threshold: float = float(pattern_cfg.get("volume_threshold", 1.5))

        # S/R
        self.sr_lookback: int = int(pattern_cfg.get("sr_lookback", 50))
        self.sr_near_pct: float = float(pattern_cfg.get("sr_near_pct", 0.003))

        # Bollinger compression heuristic
        self.bb_width_compression: float = float(
            pattern_cfg.get("bb_width_compression", 0.05)
        )

    # --------------------------- Public API ---------------------------

    @handle_errors(exceptions=(Exception,), default_return={}, context="calculate_features")
    def calculate_features(self, ohlcv: pd.DataFrame) -> dict[str, float]:
        """
        Compute a unified set of features on the latest bar.

        Returns a flat dict of feature_name -> float for the last index.
        Missing values are filled conservatively where reasonable.
        """
        if not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty:
            return {}
        for col in ("open", "high", "low", "close", "volume"):
            if col not in ohlcv.columns:
                return {}

        data = ohlcv.copy()
        # Ensure ascending chronological order
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()

        features: Dict[str, float] = {}

        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        last_close = float(close.iloc[-1])
        features["current_price"] = last_close

        # Price returns and volatility
        log_returns = np.log(close / close.shift(1))
        pct_returns = close.pct_change()
        vol20 = log_returns.rolling(20, min_periods=5).std()
        vol10 = log_returns.rolling(10, min_periods=5).std()
        features["log_return"] = float(log_returns.iloc[-1]) if not log_returns.empty else 0.0
        features["price_change"] = float(pct_returns.iloc[-1]) if not pct_returns.empty else 0.0
        features["volatility_20"] = float(vol20.iloc[-1]) if not vol20.empty else 0.0
        features["volatility_10"] = float(vol10.iloc[-1]) if not vol10.empty else 0.0
        features["volatility_ratio"] = (
            float(features["volatility_10"] / features["volatility_20"]) if features["volatility_20"] > 0 else 1.0
        )

        # Volume metrics
        vol_ma20 = volume.rolling(20, min_periods=1).mean()
        features["volume_sma_20"] = float(vol_ma20.iloc[-1]) if not vol_ma20.empty else float(volume.iloc[-1])
        features["volume_ratio"] = (
            float(volume.iloc[-1] / features["volume_sma_20"]) if features["volume_sma_20"] > 0 else 1.0
        )

        # Moving averages and EMA band
        sma20 = close.rolling(20, min_periods=1).mean()
        sma50 = close.rolling(50, min_periods=1).mean()
        ema_fast = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.ema_slow, adjust=False).mean()
        features["sma_20"] = float(sma20.iloc[-1])
        features["sma_50"] = float(sma50.iloc[-1])
        features[f"ema_{self.ema_fast}"] = float(ema_fast.iloc[-1])
        features[f"ema_{self.ema_slow}"] = float(ema_slow.iloc[-1])

        # RSI(14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        rs = rs.replace([np.inf, -np.inf], np.nan).clip(upper=1e6)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        features["rsi"] = float(rsi.fillna(50.0).iloc[-1])

        # MACD (12/26, signal 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        features["macd"] = float(macd.iloc[-1]) if not macd.empty else 0.0
        features["macd_signal"] = float(macd_signal.iloc[-1]) if not macd_signal.empty else 0.0
        features["macd_histogram"] = float((macd - macd_signal).iloc[-1]) if not macd.empty else 0.0

        # Bollinger Bands (20, 2)
        bb_mid = sma20
        bb_std = close.rolling(20, min_periods=1).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        features["bb_upper"] = float(bb_upper.iloc[-1]) if not bb_upper.empty else last_close
        features["bb_lower"] = float(bb_lower.iloc[-1]) if not bb_lower.empty else last_close
        denom = max(features["bb_upper"] - features["bb_lower"], 1e-12)
        features["bb_position"] = float((last_close - features["bb_lower"]) / denom)
        features["bb_width"] = float((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / max(bb_mid.iloc[-1], 1e-12))

        # ATR(14), plus extended ATR windows
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = true_range.rolling(14, min_periods=1).mean()
        atr20 = true_range.rolling(20, min_periods=1).mean()
        atr100 = true_range.rolling(100, min_periods=1).mean()
        features["atr"] = float(atr14.iloc[-1]) if not atr14.empty else float(true_range.iloc[-1])
        features["atr_20"] = float(atr20.iloc[-1]) if not atr20.empty else features["atr"]
        features["atr_100"] = float(atr100.iloc[-1]) if not atr100.empty else features["atr_20"]
        # ATR normalization
        price_abs_change = close.diff().abs()
        atr_den = float(price_abs_change.iloc[-1]) if not price_abs_change.empty else max(last_close, 1.0)
        features["atr_normalized"] = float(np.clip(features["atr"] / max(atr_den, 1e-8), 0.0, 10.0))

        # ADX(14)
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move
        minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move
        atr_for_adx = true_range.rolling(14, min_periods=1).mean().replace(0, np.nan)
        plus_di = 100.0 * (plus_dm.rolling(14, min_periods=1).mean() / atr_for_adx)
        minus_di = 100.0 * (minus_dm.rolling(14, min_periods=1).mean() / atr_for_adx)
        dx = 100.0 * (np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.rolling(14, min_periods=1).mean().replace([np.inf, -np.inf], np.nan).fillna(25.0)
        features["adx"] = float(adx.iloc[-1])

        # Price action windows
        recent_high = high.rolling(20, min_periods=1).max()
        recent_low = low.rolling(20, min_periods=1).min()
        features["recent_high"] = float(recent_high.iloc[-1])
        features["recent_low"] = float(recent_low.iloc[-1])
        denom_pr = max(features["recent_high"] - features["recent_low"], 1e-12)
        features["price_position"] = float((last_close - features["recent_low"]) / denom_pr)

        # Momentum
        features["price_momentum_5"] = float(close.pct_change(5).iloc[-1]) if len(close) >= 5 else 0.0
        features["price_momentum_10"] = float(close.pct_change(10).iloc[-1]) if len(close) >= 10 else 0.0
        features["price_acceleration"] = float(close.pct_change(5).diff().iloc[-1]) if len(close) >= 6 else 0.0

        # Simple S/R proxy and distances
        high_roll = high.rolling(max(5, self.sr_lookback), min_periods=5).max().shift(1)
        low_roll = low.rolling(max(5, self.sr_lookback), min_periods=5).min().shift(1)
        resistance = float(high_roll.iloc[-1]) if not high_roll.empty else float(high.iloc[-1])
        support = float(low_roll.iloc[-1]) if not low_roll.empty else float(low.iloc[-1])
        features["resistance_level"] = resistance
        features["support_level"] = support
        features["dist_to_resistance_pct"] = float(abs(last_close - resistance) / max(1e-12, last_close))
        features["dist_to_support_pct"] = float(abs(last_close - support) / max(1e-12, last_close))

        return features

    @handle_errors(exceptions=(Exception,), default_return={}, context="calculate_regime_flags")
    def calculate_regime_flags(self, features: dict[str, float]) -> dict[str, Any]:
        """
        Derive unified regime flags from computed features.
        """
        if not features:
            return {}

        # Trend regime via EMA band + ADX gate
        ema_fast_val = float(features.get(f"ema_{self.ema_fast}", 0.0))
        ema_slow_val = float(features.get(f"ema_{self.ema_slow}", 0.0))
        adx_val = float(features.get("adx", 25.0))
        close_val = float(features.get("current_price", 0.0))
        base_level = max(float(features.get("sma_50", close_val)), 1e-8)
        ema_sep_norm = abs(ema_fast_val - ema_slow_val) / base_level

        meets_sideways = adx_val < self.adx_sideways_threshold
        meets_trend = (adx_val >= self.adx_trend_threshold) and (
            ema_sep_norm >= max(self.ema_sep_min_ratio, 0.0)
        )

        is_bull = (ema_fast_val > ema_slow_val) and not meets_sideways and meets_trend
        is_bear = (ema_fast_val < ema_slow_val) and not meets_sideways and meets_trend
        is_sideways = not is_bull and not is_bear

        # Trend confidence (mirrors simple rules style)
        denom_sw = max(self.adx_sideways_threshold, 1e-6)
        conf_sideways = float(np.clip((self.adx_sideways_threshold - adx_val) / denom_sw, 0.2, 1.0))
        denom_tr = max(self.adx_trend_threshold - self.adx_sideways_threshold, 1e-6)
        adx_component = float(np.clip((adx_val - self.adx_sideways_threshold) / denom_tr, 0.0, 1.0))
        sep_component = float(np.clip(ema_sep_norm * 10.0, 0.0, 1.0))
        conf_trend = float(np.clip(0.5 * adx_component + 0.5 * sep_component, 0.2, 1.0))
        trend_conf = conf_sideways if is_sideways else conf_trend

        # Volatility regime classification bands (MarketHealth-like)
        vol20 = float(features.get("volatility_20", 0.0))
        low_max, normal_max, high_max = self.vol_regime_bands
        if vol20 <= low_max:
            vol_regime = "LOW"
        elif vol20 <= normal_max:
            vol_regime = "NORMAL"
        elif vol20 <= high_max:
            vol_regime = "HIGH"
        else:
            vol_regime = "EXTREME"

        # SR proximity
        dist_sup = float(features.get("dist_to_support_pct", 1.0))
        dist_res = float(features.get("dist_to_resistance_pct", 1.0))
        near_support = dist_sup <= self.sr_near_pct
        near_resistance = dist_res <= self.sr_near_pct
        if near_support and not near_resistance:
            sr_context = "NEAR_SUPPORT"
        elif near_resistance and not near_support:
            sr_context = "NEAR_RESISTANCE"
        else:
            sr_context = "MID_RANGE"

        # Breakout heuristics
        recent_high = float(features.get("recent_high", 0.0))
        recent_low = float(features.get("recent_low", 0.0))
        momentum_5 = float(features.get("price_momentum_5", 0.0))
        volume_ratio = float(features.get("volume_ratio", 1.0))
        is_breakout_up = (close_val > max(recent_high, 1e-12)) and (momentum_5 > self.breakout_momentum) and (
            volume_ratio > self.volume_threshold
        )
        is_breakout_down = (close_val < max(recent_low, 1e-12)) and (momentum_5 < -self.breakout_momentum) and (
            volume_ratio > self.volume_threshold
        )

        # Compression/Expansion
        bb_width = float(features.get("bb_width", 0.0))
        vol_ratio = float(features.get("volatility_ratio", 1.0))
        is_compression = (bb_width < self.bb_width_compression) and (vol_ratio < 0.8)
        is_expansion = (vol_ratio > 1.5) and (volume_ratio > self.volume_threshold)

        # Consolidate flags
        flags: Dict[str, Any] = {
            "is_bull_trend": bool(is_bull),
            "is_bear_trend": bool(is_bear),
            "is_sideways": bool(is_sideways),
            "trend_confidence": float(trend_conf),
            "volatility_regime": vol_regime,
            "is_high_volatility": bool(vol20 > self.volatility_threshold),
            "near_support": bool(near_support),
            "near_resistance": bool(near_resistance),
            "is_breakout_up": bool(is_breakout_up),
            "is_breakout_down": bool(is_breakout_down),
            "is_volatility_compression": bool(is_compression),
            "is_volatility_expansion": bool(is_expansion),
        }

        # Summary object
        trend_label = "SIDEWAYS_RANGE"
        if is_bull:
            trend_label = "BULL_TREND"
        elif is_bear:
            trend_label = "BEAR_TREND"

        flags["summary"] = RegimeSummary(
            trend=trend_label,
            trend_confidence=float(trend_conf),
            volatility_regime=vol_regime,
            sr_context=sr_context,
        )

        return flags

    @handle_errors(exceptions=(Exception,), default_return={}, context="build_snapshot")
    def build_snapshot(self, ohlcv: pd.DataFrame) -> dict[str, Any]:
        """
        Compute features and unified regime flags in one pass for live trading.

        Returns a dict with keys: 'features', 'flags', and convenience keys
        duplicated at the top level for quick access.
        """
        feats = self.calculate_features(ohlcv)
        if not feats:
            return {}
        flags = self.calculate_regime_flags(feats)
        if not flags:
            return {"features": feats}

        # Flatten key conveniences
        summary: RegimeSummary = flags.get("summary")  # type: ignore[assignment]
        out: dict[str, Any] = {
            "features": feats,
            "flags": flags,
            "trend": summary.trend if summary else None,
            "trend_confidence": summary.trend_confidence if summary else None,
            "volatility_regime": summary.volatility_regime if summary else None,
            "sr_context": summary.sr_context if summary else None,
        }
        return out


__all__ = ["LiveRegimeCalculator", "RegimeSummary"]