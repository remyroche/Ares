# src/analyst/advanced_feature_engineering.py

"""
Advanced Feature Engineering components.
Provides minimal, syntactically-correct implementations to unblock the pipeline
while preserving public APIs used elsewhere in the codebase.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import error, warning


class CandlestickPatternAnalyzer:
    """
    Minimal candlestick pattern analyzer with a few common detections.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("CandlestickPatternAnalyzer")

        pattern_config = config.get("candlestick_patterns", {})
        self.doji_threshold = float(pattern_config.get("doji_threshold", 0.1))
        self.hammer_ratio = float(pattern_config.get("hammer_ratio", 0.3))
        self.shadow_ratio = float(pattern_config.get("shadow_ratio", 2.0))
        self.engulfing_ratio = float(pattern_config.get("engulfing_ratio", 1.1))
        self.tweezer_threshold = float(pattern_config.get("tweezer_threshold", 0.02))
        self.marubozu_threshold = float(pattern_config.get("marubozu_threshold", 0.1))

        self.is_initialized = False

    @handle_errors(exceptions=(Exception,), default_return=False, context="candlestick analyzer init")
    async def initialize(self) -> bool:
        self.logger.info("Initializing candlestick pattern analyzer...")
        self.is_initialized = True
        self.logger.info("Candlestick pattern analyzer initialized")
        return True

    @handle_errors(exceptions=(Exception,), default_return={}, context="candlestick analyze")
    async def analyze_patterns(self, price_data: pd.DataFrame) -> dict[str, Any]:
        if price_data is None or not isinstance(price_data, pd.DataFrame):
            self.logger.error("Price data must be a pandas DataFrame")
            return {}
        if not self.is_initialized:
            self.logger.error("Candlestick pattern analyzer not initialized")
            return {}
        if price_data.empty or len(price_data) < 3:
            self.logger.warning("Insufficient data for pattern analysis")
            return {}

        df = self._prepare_candlestick_data(price_data)
        if df.empty:
            return {}

        patterns = {
            "engulfing_patterns": self._detect_engulfing_patterns(df),
            "hammer_hanging_man": self._detect_hammer_hanging_man(df),
            "shooting_star_inverted_hammer": self._detect_shooting_star_inverted_hammer(df),
            "tweezer_patterns": self._detect_tweezer_patterns(df),
            "marubozu_patterns": self._detect_marubozu_patterns(df),
            "doji_patterns": self._detect_doji_patterns(df),
            "spinning_top_patterns": self._detect_spinning_top_patterns(df),
        }
        return self._convert_patterns_to_features(patterns, df)

    def _prepare_candlestick_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            required_columns = ["open", "high", "low", "close"]
            if df is None or df.empty or not all(col in df.columns for col in required_columns):
                return pd.DataFrame()

            out = df.copy()
            out["body_size"] = (out["close"] - out["open"]).abs()
            out["upper_shadow"] = out["high"] - out[["open", "close"]].max(axis=1)
            out["lower_shadow"] = out[["open", "close"]].min(axis=1) - out["low"]
            out["total_range"] = (out["high"] - out["low"]).replace(0, np.nan)
            out["body_ratio"] = (out["body_size"] / out["total_range"]).fillna(0.0)
            out["is_bullish"] = out["close"] > out["open"]
            out["avg_body_size"] = out["body_size"].rolling(window=20, min_periods=1).mean()
            out["avg_range"] = out["total_range"].rolling(window=20, min_periods=1).mean()
            return out
        except Exception as e:
            self.logger.error(f"Error preparing candlestick data: {e}")
            return pd.DataFrame()

    def _detect_engulfing_patterns(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        patterns: list[dict[str, Any]] = []
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i - 1]
            if (
                current["is_bullish"]
                and not previous["is_bullish"]
                and current["open"] < previous["close"]
                and current["close"] > previous["open"]
                and current["body_size"] > previous["body_size"] * self.engulfing_ratio
            ):
                patterns.append({"index": i, "pattern": "bullish_engulfing", "is_bullish": True})
            elif (
                (not current["is_bullish"]) and previous["is_bullish"]
                and current["open"] > previous["close"] and current["close"] < previous["open"]
                and current["body_size"] > previous["body_size"] * self.engulfing_ratio
            ):
                patterns.append({"index": i, "pattern": "bearish_engulfing", "is_bullish": False})
        return patterns

    def _detect_hammer_hanging_man(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        patterns: list[dict[str, Any]] = []
        for i in range(len(df)):
            row = df.iloc[i]
            is_small_body = row["body_ratio"] <= self.hammer_ratio
            long_lower = row["lower_shadow"] >= self.shadow_ratio * row["body_size"]
            long_upper = row["upper_shadow"] >= self.shadow_ratio * row["body_size"]
            if is_small_body and long_lower and row["is_bullish"]:
                patterns.append({"index": i, "pattern": "hammer", "is_bullish": True})
            if is_small_body and long_upper and (not row["is_bullish"]):
                patterns.append({"index": i, "pattern": "hanging_man", "is_bullish": False})
        return patterns

    def _detect_shooting_star_inverted_hammer(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        patterns: list[dict[str, Any]] = []
        for i in range(len(df)):
            row = df.iloc[i]
            is_small_body = row["body_ratio"] <= self.hammer_ratio
            long_upper = row["upper_shadow"] >= self.shadow_ratio * row["body_size"]
            long_lower = row["lower_shadow"] >= self.shadow_ratio * row["body_size"]
            if is_small_body and long_upper and (not row["is_bullish"]):
                patterns.append({"index": i, "pattern": "shooting_star", "is_bullish": False})
            if is_small_body and long_lower and row["is_bullish"]:
                patterns.append({"index": i, "pattern": "inverted_hammer", "is_bullish": True})
        return patterns

    def _detect_tweezer_patterns(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        patterns: list[dict[str, Any]] = []
        for i in range(1, len(df)):
            a = df.iloc[i - 1]
            b = df.iloc i
            if abs(a["high"] - b["high"]) <= self.tweezer_threshold * a["avg_range"]:
                patterns.append({"index": i, "pattern": "tweezer_top", "is_bullish": False})
            if abs(a["low"] - b["low"]) <= self.tweezer_threshold * a["avg_range"]:
                patterns.append({"index": i, "pattern": "tweezer_bottom", "is_bullish": True})
        return patterns

    def _detect_marubozu_patterns(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        patterns: list[dict[str, Any]] = []
        for i in range(len(df)):
            row = df.iloc[i]
            if row["body_ratio"] >= 1.0 - self.marubozu_threshold:
                name = "marubozu_bull" if row["is_bullish"] else "marubozu_bear"
                patterns.append({"index": i, "pattern": name, "is_bullish": bool(row["is_bullish"])})
        return patterns

    def _detect_doji_patterns(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        patterns: list[dict[str, Any]] = []
        small = df["body_ratio"] <= self.doji_threshold
        for i, is_small in enumerate(small.values.tolist()):
            if is_small:
                patterns.append({"index": i, "pattern": "doji", "is_bullish": False})
        return patterns

    def _detect_spinning_top_patterns(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        patterns: list[dict[str, Any]] = []
        mid = (df["body_ratio"] > self.doji_threshold) & (df["body_ratio"] < 0.5)
        for i, cond in enumerate(mid.values.tolist()):
            if cond:
                patterns.append({"index": i, "pattern": "spinning_top", "is_bullish": False})
        return patterns

    def _convert_patterns_to_features(self, patterns: dict[str, list[dict[str, Any]]], df: pd.DataFrame) -> dict[str, Any]:
        features: dict[str, Any] = {}
        total_rows = int(len(df))
        for name, lst in patterns.items():
            count = len(lst)
            features[f"{name}_count"] = count
            features[f"{name}_ratio"] = (count / total_rows) if total_rows else 0.0
        return features


class AdvancedFeatureEngineering:
    """
    Minimal Advanced Feature Engineering implementation.
    Exposes methods referenced by orchestrators/pipelines:
      - initialize()
      - engineer_features(...)
      - generate_features(...)
      - _engineer_multi_timeframe_features(...)
      - get_feature_statistics()
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("AdvancedFeatureEngineering")
        self.is_initialized = False
        self._feature_stats: dict[str, Any] = {}

    @handle_errors(exceptions=(Exception,), default_return=False, context="advanced fe init")
    async def initialize(self) -> bool:
        self.logger.info("Initializing AdvancedFeatureEngineering...")
        self.is_initialized = True
        self.logger.info("AdvancedFeatureEngineering initialized")
        return True

    @handle_errors(exceptions=(Exception,), default_return=pd.DataFrame(), context="engineer features")
    async def engineer_features(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        order_flow_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if price_data is None or not isinstance(price_data, pd.DataFrame) or price_data.empty:
            return pd.DataFrame()
        df = price_data.copy()
        df = self._basic_features(df)
        self._feature_stats = {"num_rows": int(len(df)), "num_features": int(len(df.columns))}
        return df

    def generate_features(self, df: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        out = self._basic_features(df.copy())
        self._feature_stats = {"num_rows": int(len(out)), "num_features": int(len(out.columns))}
        return out

    def get_feature_statistics(self) -> dict[str, Any]:
        return dict(self._feature_stats)

    @handle_errors(exceptions=(Exception,), default_return={}, context="multi-timeframe features")
    async def _engineer_multi_timeframe_features(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        order_flow_data: Optional[pd.DataFrame] = None,
    ) -> dict[str, Any]:
        if price_data is None or price_data.empty:
            return {}
        close = price_data["close"].astype(float)
        return {
            "mtf_ret_1": float(close.pct_change(1).fillna(0.0).tail(1).values[0]),
            "mtf_ret_5": float(close.pct_change(5).fillna(0.0).tail(1).values[0]) if len(close) >= 6 else 0.0,
        }

    def _basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not {"open", "high", "low", "close"}.issubset(df.columns):
            return df
        out = df.copy()
        out["ret_1"] = out["close"].pct_change().fillna(0.0)
        out["hl_range"] = (out["high"] - out["low"]).astype(float)
        out["volatility_10"] = out["ret_1"].rolling(10, min_periods=1).std().fillna(0.0)
        out["ma_10"] = out["close"].rolling(10, min_periods=1).mean().fillna(method="bfill").fillna(0.0)
        out["zscore_10"] = (out["close"] - out["ma_10"]) / (out["volatility_10"].replace(0, np.nan))
        out["zscore_10"] = out["zscore_10"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return out
