# src/analyst/market_health_analyzer.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from src.utils.logger import system_logger
from src.utils.error_handler import (
    handle_errors,
    handle_data_processing_errors,
)


class GeneralMarketAnalystModule:
    """
    General market health analysis module that provides comprehensive market health metrics.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = system_logger.getChild("MarketHealthAnalyzer")

    @handle_errors(
        exceptions=(ValueError, TypeError, KeyError),
        default_return=0.0,
        context="get_market_health_score",
    )
    def get_market_health_score(self, klines_df: pd.DataFrame) -> float:
        """
        Calculates a comprehensive market health score based on multiple factors.

        Args:
            klines_df: DataFrame with OHLCV data

        Returns:
            float: Market health score between 0 and 100
        """
        if klines_df.empty:
            self.logger.warning("Empty DataFrame provided for market health analysis")
            return 0.0

        try:
            # Calculate individual health metrics
            volatility_score = self._calculate_volatility_score(klines_df)
            volume_score = self._calculate_volume_score(klines_df)
            trend_score = self._calculate_trend_score(klines_df)
            liquidity_score = self._calculate_liquidity_score(klines_df)

            # Weighted combination of scores
            weights = self.config.get(
                "health_score_weights",
                {"volatility": 0.25, "volume": 0.25, "trend": 0.25, "liquidity": 0.25},
            )

            health_score = (
                volatility_score * weights["volatility"]
                + volume_score * weights["volume"]
                + trend_score * weights["trend"]
                + liquidity_score * weights["liquidity"]
            )

            self.logger.info(
                f"Market health score: {health_score:.2f} "
                f"(Vol: {volatility_score:.2f}, Vol: {volume_score:.2f}, "
                f"Trend: {trend_score:.2f}, Liq: {liquidity_score:.2f})"
            )

            return max(0.0, min(100.0, health_score))

        except Exception as e:
            self.logger.error(f"Error calculating market health score: {e}")
            return 0.0

    @handle_data_processing_errors(
        default_return=0.0, context="calculate_volatility_score"
    )
    def _calculate_volatility_score(self, df: pd.DataFrame) -> float:
        """Calculate volatility-based health score."""
        try:
            # Calculate True Range
            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift(1))
            low_close = np.abs(df["low"] - df["close"].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(
                axis=1
            )

            # Calculate ATR
            atr = true_range.rolling(window=14).mean()
            current_atr = atr.iloc[-1]
            avg_atr = atr.mean()

            # Volatility score based on ATR ratio
            if avg_atr > 0:
                volatility_ratio = current_atr / avg_atr
                # Optimal volatility is around 1.0, penalize extremes
                if 0.5 <= volatility_ratio <= 2.0:
                    score = 100.0 - abs(volatility_ratio - 1.0) * 50
                else:
                    score = max(0.0, 100.0 - abs(volatility_ratio - 1.0) * 100)
                return max(0.0, min(100.0, score))
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating volatility score: {e}")
            return 0.0

    @handle_data_processing_errors(default_return=0.0, context="calculate_volume_score")
    def _calculate_volume_score(self, df: pd.DataFrame) -> float:
        """Calculate volume-based health score."""
        try:
            if "volume" not in df.columns:
                return 0.0

            # Calculate volume metrics
            current_volume = df["volume"].iloc[-1]
            avg_volume = df["volume"].rolling(window=20).mean().iloc[-1]
            # volume_std = df["volume"].rolling(window=20).std().iloc[-1]

            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                # Higher volume is generally better, but not extreme
                if volume_ratio >= 1.0:
                    score = min(100.0, 50.0 + (volume_ratio - 1.0) * 25)
                else:
                    score = max(0.0, volume_ratio * 50)
                return max(0.0, min(100.0, score))
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating volume score: {e}")
            return 0.0

    @handle_data_processing_errors(default_return=0.0, context="calculate_trend_score")
    def _calculate_trend_score(self, df: pd.DataFrame) -> float:
        """Calculate trend-based health score."""
        try:
            # Calculate moving averages
            sma_20 = df["close"].rolling(window=20).mean()
            sma_50 = df["close"].rolling(window=50).mean()

            current_price = df["close"].iloc[-1]
            current_sma_20 = sma_20.iloc[-1]
            current_sma_50 = sma_50.iloc[-1]

            # Trend strength based on price position relative to MAs
            if not pd.isna(current_sma_20) and not pd.isna(current_sma_50):
                # Price above both MAs = strong uptrend
                if current_price > current_sma_20 > current_sma_50:
                    score = 100.0
                # Price below both MAs = strong downtrend
                elif current_price < current_sma_20 < current_sma_50:
                    score = 100.0
                # Mixed signals = weaker trend
                else:
                    score = 50.0
                return score
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating trend score: {e}")
            return 0.0

    @handle_data_processing_errors(
        default_return=0.0, context="calculate_liquidity_score"
    )
    def _calculate_liquidity_score(self, df: pd.DataFrame) -> float:
        """Calculate liquidity-based health score."""
        try:
            if "volume" not in df.columns:
                return 0.0

            # Calculate average volume and price
            avg_volume = df["volume"].mean()
            avg_price = df["close"].mean()

            if avg_price > 0:
                # Liquidity score based on volume/price ratio
                liquidity_ratio = avg_volume / avg_price
                # Higher liquidity is better, but with diminishing returns
                score = min(
                    100.0, liquidity_ratio * 1000
                )  # Adjust multiplier as needed
                return max(0.0, min(100.0, score))
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating liquidity score: {e}")
            return 0.0

    @handle_errors(
        exceptions=(ValueError, TypeError, KeyError),
        default_return={},
        context="analyze_market_health",
    )
    def analyze_market_health(self, klines_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive market health analysis.

        Args:
            klines_df: DataFrame with OHLCV data

        Returns:
            Dict: Comprehensive market health metrics
        """
        if klines_df.empty:
            return {}

        try:
            health_score = self.get_market_health_score(klines_df)

            # Additional health metrics
            volatility_metrics = self._get_volatility_metrics(klines_df)
            volume_metrics = self._get_volume_metrics(klines_df)
            trend_metrics = self._get_trend_metrics(klines_df)

            return {
                "health_score": health_score,
                "volatility_metrics": volatility_metrics,
                "volume_metrics": volume_metrics,
                "trend_metrics": trend_metrics,
                "timestamp": pd.Timestamp.now(),
            }

        except Exception as e:
            self.logger.error(f"Error in market health analysis: {e}")
            return {}

    @handle_data_processing_errors(default_return={}, context="get_volatility_metrics")
    def _get_volatility_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed volatility metrics."""
        try:
            # Calculate True Range and ATR
            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift(1))
            low_close = np.abs(df["low"] - df["close"].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(
                axis=1
            )
            atr = true_range.rolling(window=14).mean()

            return {
                "current_atr": atr.iloc[-1] if not atr.empty else 0.0,
                "avg_atr": atr.mean() if not atr.empty else 0.0,
                "atr_ratio": (atr.iloc[-1] / atr.mean())
                if not atr.empty and atr.mean() > 0
                else 0.0,
            }
        except Exception as e:
            self.logger.error(f"Error calculating volatility metrics: {e}")
            return {}

    @handle_data_processing_errors(default_return={}, context="get_volume_metrics")
    def _get_volume_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed volume metrics."""
        try:
            if "volume" not in df.columns:
                return {}

            current_volume = df["volume"].iloc[-1]
            avg_volume = df["volume"].rolling(window=20).mean().iloc[-1]

            return {
                "current_volume": current_volume,
                "avg_volume": avg_volume,
                "volume_ratio": (current_volume / avg_volume)
                if avg_volume > 0
                else 0.0,
            }
        except Exception as e:
            self.logger.error(f"Error calculating volume metrics: {e}")
            return {}

    @handle_data_processing_errors(default_return={}, context="get_trend_metrics")
    def _get_trend_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed trend metrics."""
        try:
            sma_20 = df["close"].rolling(window=20).mean()
            sma_50 = df["close"].rolling(window=50).mean()

            current_price = df["close"].iloc[-1]
            current_sma_20 = sma_20.iloc[-1]
            current_sma_50 = sma_50.iloc[-1]

            return {
                "current_price": current_price,
                "sma_20": current_sma_20,
                "sma_50": current_sma_50,
                "price_vs_sma_20": ((current_price / current_sma_20) - 1) * 100
                if current_sma_20 > 0
                else 0.0,
                "price_vs_sma_50": ((current_price / current_sma_50) - 1) * 100
                if current_sma_50 > 0
                else 0.0,
            }
        except Exception as e:
            self.logger.error(f"Error calculating trend metrics: {e}")
            return {}
