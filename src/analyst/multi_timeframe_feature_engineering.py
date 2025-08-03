# src/analyst/multi_timeframe_feature_engineering.py

"""
Multi-Timeframe Feature Engineering System

This module provides timeframe-specific feature engineering that adapts indicators
to the defined timeframes:

- Execution: 1m & 5m (combined for ultra-short-term execution)
- Tactical: 15m (for tactical decision making)
- Strategic: 1h (for macro trend analysis)

The system ensures that indicators are calculated with appropriate parameters
for each timeframe's characteristics and trading style.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyst.feature_engineering import FeatureEngineeringEngine
from src.config import CONFIG
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class MultiTimeframeFeatureEngineering:
    """
    Multi-timeframe feature engineering system that adapts indicators to specific timeframes.

    This system ensures that technical indicators, volume analysis, and other features
    are calculated with appropriate parameters for each timeframe's characteristics.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the multi-timeframe feature engineering system.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("MultiTimeframeFeatureEngineering")

        # Initialize base feature engineering engine
        self.base_feature_engine = FeatureEngineeringEngine(config)

        # Timeframe definitions and purposes
        self.timeframes = CONFIG.get("TIMEFRAMES", {})
        self.timeframe_sets = CONFIG.get("TIMEFRAME_SETS", {})

        # Multi-timeframe configuration
        self.mtf_config = config.get("multi_timeframe_feature_engineering", {})
        self.enable_mtf_features = self.mtf_config.get("enable_mtf_features", True)
        self.enable_timeframe_adaptation = self.mtf_config.get(
            "enable_timeframe_adaptation",
            True,
        )

        # Timeframe-specific parameter mappings
        self.timeframe_parameters = self._initialize_timeframe_parameters()

        # Feature cache for performance
        self.feature_cache: dict[str, pd.DataFrame] = {}
        self.cache_duration = timedelta(minutes=5)
        self.last_cache_cleanup = datetime.now()

        self.logger.info("🚀 Initialized MultiTimeframeFeatureEngineering")
        self.logger.info(f"📊 Timeframe adaptation: {self.enable_timeframe_adaptation}")

    def _initialize_timeframe_parameters(self) -> dict[str, dict[str, Any]]:
        """
        Initialize timeframe-specific parameters for indicators.

        Returns:
            Dictionary with timeframe-specific parameter mappings
        """
        return {
            # Execution timeframes (1m & 5m)
            "1m": {
                "description": "Ultra-short-term execution",
                "trading_style": "scalping",
                "indicators": {
                    "rsi": {
                        "length": 7,
                        "description": "Fast RSI for immediate signals",
                    },
                    "macd": {
                        "fast": 6,
                        "slow": 12,
                        "signal": 4,
                        "description": "Fast MACD for scalping",
                    },
                    "bbands": {"length": 10, "description": "Tight Bollinger Bands"},
                    "atr": {"length": 7, "description": "Short ATR for tight stops"},
                    "adx": {"length": 7, "description": "Fast ADX for momentum"},
                    "stoch": {"length": 7, "description": "Fast Stochastic"},
                    "sma": {
                        "lengths": [5, 10, 20],
                        "description": "Short-term moving averages",
                    },
                    "ema": {
                        "lengths": [3, 7, 14],
                        "description": "Fast exponential averages",
                    },
                    "volume": {
                        "sma_length": 10,
                        "description": "Short volume analysis",
                    },
                    "volatility": {
                        "window": 10,
                        "description": "Short-term volatility",
                    },
                },
            },
            "5m": {
                "description": "Short-term execution",
                "trading_style": "scalping",
                "indicators": {
                    "rsi": {"length": 10, "description": "Medium-fast RSI"},
                    "macd": {
                        "fast": 8,
                        "slow": 16,
                        "signal": 6,
                        "description": "Medium MACD",
                    },
                    "bbands": {"length": 15, "description": "Medium Bollinger Bands"},
                    "atr": {"length": 10, "description": "Medium ATR"},
                    "adx": {"length": 10, "description": "Medium ADX"},
                    "stoch": {"length": 10, "description": "Medium Stochastic"},
                    "sma": {
                        "lengths": [8, 15, 30],
                        "description": "Medium-term moving averages",
                    },
                    "ema": {
                        "lengths": [5, 10, 20],
                        "description": "Medium exponential averages",
                    },
                    "volume": {
                        "sma_length": 15,
                        "description": "Medium volume analysis",
                    },
                    "volatility": {
                        "window": 15,
                        "description": "Medium-term volatility",
                    },
                },
            },
            # Tactical timeframe (15m)
            "15m": {
                "description": "Tactical decision making",
                "trading_style": "intraday_swing",
                "indicators": {
                    "rsi": {"length": 14, "description": "Standard RSI"},
                    "macd": {
                        "fast": 12,
                        "slow": 26,
                        "signal": 9,
                        "description": "Standard MACD",
                    },
                    "bbands": {"length": 20, "description": "Standard Bollinger Bands"},
                    "atr": {"length": 14, "description": "Standard ATR"},
                    "adx": {"length": 14, "description": "Standard ADX"},
                    "stoch": {"length": 14, "description": "Standard Stochastic"},
                    "sma": {
                        "lengths": [10, 20, 50],
                        "description": "Standard moving averages",
                    },
                    "ema": {
                        "lengths": [7, 14, 30],
                        "description": "Standard exponential averages",
                    },
                    "volume": {
                        "sma_length": 20,
                        "description": "Standard volume analysis",
                    },
                    "volatility": {"window": 20, "description": "Standard volatility"},
                },
            },
            # Strategic timeframe (1h)
            "1h": {
                "description": "Strategic macro trend analysis",
                "trading_style": "swing_trading",
                "indicators": {
                    "rsi": {
                        "length": 21,
                        "description": "Long RSI for trend confirmation",
                    },
                    "macd": {
                        "fast": 16,
                        "slow": 32,
                        "signal": 12,
                        "description": "Long MACD for trends",
                    },
                    "bbands": {"length": 30, "description": "Wide Bollinger Bands"},
                    "atr": {
                        "length": 21,
                        "description": "Long ATR for position sizing",
                    },
                    "adx": {"length": 21, "description": "Long ADX for trend strength"},
                    "stoch": {"length": 21, "description": "Long Stochastic"},
                    "sma": {
                        "lengths": [20, 50, 100],
                        "description": "Long-term moving averages",
                    },
                    "ema": {
                        "lengths": [14, 30, 60],
                        "description": "Long exponential averages",
                    },
                    "volume": {"sma_length": 30, "description": "Long volume analysis"},
                    "volatility": {"window": 30, "description": "Long-term volatility"},
                },
            },
            # Additional timeframes
            "4h": {
                "description": "Medium-term trend analysis",
                "trading_style": "position_trading",
                "indicators": {
                    "rsi": {"length": 28, "description": "Very long RSI"},
                    "macd": {
                        "fast": 20,
                        "slow": 40,
                        "signal": 15,
                        "description": "Very long MACD",
                    },
                    "bbands": {
                        "length": 40,
                        "description": "Very wide Bollinger Bands",
                    },
                    "atr": {"length": 28, "description": "Very long ATR"},
                    "adx": {"length": 28, "description": "Very long ADX"},
                    "stoch": {"length": 28, "description": "Very long Stochastic"},
                    "sma": {
                        "lengths": [30, 60, 120],
                        "description": "Very long-term moving averages",
                    },
                    "ema": {
                        "lengths": [20, 40, 80],
                        "description": "Very long exponential averages",
                    },
                    "volume": {
                        "sma_length": 40,
                        "description": "Very long volume analysis",
                    },
                    "volatility": {
                        "window": 40,
                        "description": "Very long-term volatility",
                    },
                },
            },
            "1d": {
                "description": "Long-term trend analysis",
                "trading_style": "position_trading",
                "indicators": {
                    "rsi": {"length": 35, "description": "Ultra-long RSI"},
                    "macd": {
                        "fast": 24,
                        "slow": 48,
                        "signal": 18,
                        "description": "Ultra-long MACD",
                    },
                    "bbands": {
                        "length": 50,
                        "description": "Ultra-wide Bollinger Bands",
                    },
                    "atr": {"length": 35, "description": "Ultra-long ATR"},
                    "adx": {"length": 35, "description": "Ultra-long ADX"},
                    "stoch": {"length": 35, "description": "Ultra-long Stochastic"},
                    "sma": {
                        "lengths": [40, 80, 160],
                        "description": "Ultra-long-term moving averages",
                    },
                    "ema": {
                        "lengths": [30, 60, 120],
                        "description": "Ultra-long exponential averages",
                    },
                    "volume": {
                        "sma_length": 50,
                        "description": "Ultra-long volume analysis",
                    },
                    "volatility": {
                        "window": 50,
                        "description": "Ultra-long-term volatility",
                    },
                },
            },
        }

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=pd.DataFrame(),
        context="multi-timeframe feature generation",
    )
    async def generate_multi_timeframe_features(
        self,
        data_dict: dict[str, pd.DataFrame],
        agg_trades_dict: dict[str, pd.DataFrame] | None = None,
        futures_dict: dict[str, pd.DataFrame] | None = None,
        sr_levels: list[float] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Generate features for multiple timeframes with appropriate adaptations.

        Args:
            data_dict: Dictionary with timeframe -> DataFrame mapping
            agg_trades_dict: Optional dictionary with timeframe -> agg_trades mapping
            futures_dict: Optional dictionary with timeframe -> futures mapping
            sr_levels: Optional support/resistance levels

        Returns:
            Dictionary with timeframe -> features DataFrame mapping
        """
        try:
            self.logger.info("🎯 Generating multi-timeframe features...")

            # Clean cache if needed
            self._clean_cache()

            features_dict = {}

            for timeframe, data in data_dict.items():
                if data.empty:
                    self.logger.warning(f"Empty data for {timeframe}, skipping")
                    continue

                self.logger.info(f"📊 Generating features for {timeframe}...")

                # Get timeframe-specific parameters
                tf_params = self.timeframe_parameters.get(timeframe, {})

                # Generate base features first
                base_features = await self._generate_base_features(
                    timeframe,
                    data,
                    agg_trades_dict.get(timeframe) if agg_trades_dict else None,
                    futures_dict.get(timeframe) if futures_dict else None,
                    sr_levels,
                )

                # Generate timeframe-specific features
                tf_features = await self._generate_timeframe_specific_features(
                    timeframe,
                    base_features,
                    tf_params,
                )

                # Add timeframe metadata
                tf_features = self._add_timeframe_metadata(
                    tf_features,
                    timeframe,
                    tf_params,
                )

                features_dict[timeframe] = tf_features

                # Cache features
                self._cache_features(timeframe, tf_features)

                self.logger.info(
                    f"✅ Generated {len(tf_features.columns)} features for {timeframe}",
                )

            return features_dict

        except Exception as e:
            self.logger.error(f"Error generating multi-timeframe features: {e}")
            return {}

    async def _generate_base_features(
        self,
        timeframe: str,
        data: pd.DataFrame,
        agg_trades: pd.DataFrame | None = None,
        futures: pd.DataFrame | None = None,
        sr_levels: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Generate base features using the standard feature engineering engine.

        Args:
            timeframe: Timeframe identifier
            data: OHLCV data
            agg_trades: Optional aggregated trades data
            futures: Optional futures data
            sr_levels: Optional support/resistance levels

        Returns:
            DataFrame with base features
        """
        try:
            # Use base feature engineering engine
            base_features = self.base_feature_engine.generate_all_features(
                klines_df=data,
                agg_trades_df=agg_trades or pd.DataFrame(),
                futures_df=futures or pd.DataFrame(),
                sr_levels=sr_levels or [],
            )

            return base_features

        except Exception as e:
            self.logger.error(f"Error generating base features for {timeframe}: {e}")
            return data.copy()

    async def _generate_timeframe_specific_features(
        self,
        timeframe: str,
        base_features: pd.DataFrame,
        tf_params: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Generate timeframe-specific features with adapted parameters.

        Args:
            timeframe: Timeframe identifier
            base_features: Base features DataFrame
            tf_params: Timeframe-specific parameters

        Returns:
            DataFrame with timeframe-specific features
        """
        try:
            features = base_features.copy()

            # Get indicator parameters for this timeframe
            indicator_params = tf_params.get("indicators", {})

            # Calculate timeframe-specific technical indicators
            features = self._calculate_timeframe_technical_indicators(
                features,
                timeframe,
                indicator_params,
            )

            # Calculate timeframe-specific volume indicators
            features = self._calculate_timeframe_volume_indicators(
                features,
                timeframe,
                indicator_params,
            )

            # Calculate timeframe-specific volatility indicators
            features = self._calculate_timeframe_volatility_indicators(
                features,
                timeframe,
                indicator_params,
            )

            # Calculate timeframe-specific momentum indicators
            features = self._calculate_timeframe_momentum_indicators(
                features,
                timeframe,
                indicator_params,
            )

            # Calculate timeframe-specific trend indicators
            features = self._calculate_timeframe_trend_indicators(
                features,
                timeframe,
                indicator_params,
            )

            return features

        except Exception as e:
            self.logger.error(
                f"Error generating timeframe-specific features for {timeframe}: {e}",
            )
            return base_features

    def _calculate_timeframe_technical_indicators(
        self,
        df: pd.DataFrame,
        timeframe: str,
        indicator_params: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Calculate timeframe-specific technical indicators.

        Args:
            df: Features DataFrame
            timeframe: Timeframe identifier
            indicator_params: Indicator parameters

        Returns:
            DataFrame with technical indicators
        """
        try:
            # RSI
            rsi_params = indicator_params.get("rsi", {})
            rsi_length = rsi_params.get("length", 14)
            df[f"rsi_{timeframe}"] = df.ta.rsi(close=df["close"], length=rsi_length)

            # MACD
            macd_params = indicator_params.get("macd", {})
            fast = macd_params.get("fast", 12)
            slow = macd_params.get("slow", 26)
            signal = macd_params.get("signal", 9)
            macd_result = df.ta.macd(
                close=df["close"],
                fast=fast,
                slow=slow,
                signal=signal,
            )
            if macd_result is not None:
                df[f"macd_{timeframe}"] = macd_result.iloc[:, 0]
                df[f"macd_signal_{timeframe}"] = macd_result.iloc[:, 1]
                df[f"macd_hist_{timeframe}"] = macd_result.iloc[:, 2]

            # Bollinger Bands
            bb_params = indicator_params.get("bbands", {})
            bb_length = bb_params.get("length", 20)
            bb_result = df.ta.bbands(close=df["close"], length=bb_length)
            if bb_result is not None:
                df[f"bb_upper_{timeframe}"] = bb_result.iloc[:, 0]
                df[f"bb_middle_{timeframe}"] = bb_result.iloc[:, 1]
                df[f"bb_lower_{timeframe}"] = bb_result.iloc[:, 2]
                df[f"bb_width_{timeframe}"] = bb_result.iloc[:, 3]
                df[f"bb_percent_{timeframe}"] = bb_result.iloc[:, 4]

            # ATR
            atr_params = indicator_params.get("atr", {})
            atr_length = atr_params.get("length", 14)
            df[f"atr_{timeframe}"] = df.ta.atr(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                length=atr_length,
            )

            # ADX
            adx_params = indicator_params.get("adx", {})
            adx_length = adx_params.get("length", 14)
            adx_result = df.ta.adx(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                length=adx_length,
            )
            if adx_result is not None:
                df[f"adx_{timeframe}"] = adx_result.iloc[:, 0]
                df[f"dmp_{timeframe}"] = adx_result.iloc[:, 1]
                df[f"dmn_{timeframe}"] = adx_result.iloc[:, 2]

            # Stochastic
            stoch_params = indicator_params.get("stoch", {})
            stoch_length = stoch_params.get("length", 14)
            stoch_result = df.ta.stoch(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                length=stoch_length,
            )
            if stoch_result is not None:
                df[f"stoch_k_{timeframe}"] = stoch_result.iloc[:, 0]
                df[f"stoch_d_{timeframe}"] = stoch_result.iloc[:, 1]

            return df

        except Exception as e:
            self.logger.error(
                f"Error calculating technical indicators for {timeframe}: {e}",
            )
            return df

    def _calculate_timeframe_volume_indicators(
        self,
        df: pd.DataFrame,
        timeframe: str,
        indicator_params: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Calculate timeframe-specific volume indicators.

        Args:
            df: Features DataFrame
            timeframe: Timeframe identifier
            indicator_params: Indicator parameters

        Returns:
            DataFrame with volume indicators
        """
        try:
            # Volume SMA
            volume_params = indicator_params.get("volume", {})
            volume_sma_length = volume_params.get("sma_length", 20)
            df[f"volume_sma_{timeframe}"] = (
                df["volume"].rolling(volume_sma_length).mean()
            )

            # Volume ratio
            df[f"volume_ratio_{timeframe}"] = (
                df["volume"] / df[f"volume_sma_{timeframe}"]
            )

            # Volume momentum
            df[f"volume_momentum_{timeframe}"] = df["volume"].pct_change(5)

            # OBV (On Balance Volume)
            df[f"obv_{timeframe}"] = df.ta.obv(close=df["close"], volume=df["volume"])

            # VWAP
            df[f"vwap_{timeframe}"] = df.ta.vwap(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                volume=df["volume"],
            )

            return df

        except Exception as e:
            self.logger.error(
                f"Error calculating volume indicators for {timeframe}: {e}",
            )
            return df

    def _calculate_timeframe_volatility_indicators(
        self,
        df: pd.DataFrame,
        timeframe: str,
        indicator_params: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Calculate timeframe-specific volatility indicators.

        Args:
            df: Features DataFrame
            timeframe: Timeframe identifier
            indicator_params: Indicator parameters

        Returns:
            DataFrame with volatility indicators
        """
        try:
            # Volatility window
            vol_params = indicator_params.get("volatility", {})
            vol_window = vol_params.get("window", 20)

            # Rolling volatility
            returns = df["close"].pct_change()
            df[f"volatility_{timeframe}"] = returns.rolling(vol_window).std()

            # Realized volatility
            df[f"realized_volatility_{timeframe}"] = returns.rolling(vol_window).apply(
                lambda x: np.sqrt(np.sum(x**2)),
            )

            # Volatility ratio
            df[f"volatility_ratio_{timeframe}"] = (
                df[f"volatility_{timeframe}"]
                / df[f"volatility_{timeframe}"].rolling(vol_window * 2).mean()
            )

            return df

        except Exception as e:
            self.logger.error(
                f"Error calculating volatility indicators for {timeframe}: {e}",
            )
            return df

    def _calculate_timeframe_momentum_indicators(
        self,
        df: pd.DataFrame,
        timeframe: str,
        indicator_params: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Calculate timeframe-specific momentum indicators.

        Args:
            df: Features DataFrame
            timeframe: Timeframe identifier
            indicator_params: Indicator parameters

        Returns:
            DataFrame with momentum indicators
        """
        try:
            # Price momentum
            df[f"price_momentum_{timeframe}"] = df["close"].pct_change(5)
            df[f"price_acceleration_{timeframe}"] = df[
                f"price_momentum_{timeframe}"
            ].diff()

            # Rate of change
            df[f"roc_{timeframe}"] = df.ta.roc(close=df["close"], length=10)

            # Williams %R
            df[f"williams_r_{timeframe}"] = df.ta.willr(
                high=df["high"],
                low=df["low"],
                close=df["close"],
            )

            # CCI (Commodity Channel Index)
            df[f"cci_{timeframe}"] = df.ta.cci(
                high=df["high"],
                low=df["low"],
                close=df["close"],
            )

            return df

        except Exception as e:
            self.logger.error(
                f"Error calculating momentum indicators for {timeframe}: {e}",
            )
            return df

    def _calculate_timeframe_trend_indicators(
        self,
        df: pd.DataFrame,
        timeframe: str,
        indicator_params: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Calculate timeframe-specific trend indicators.

        Args:
            df: Features DataFrame
            timeframe: Timeframe identifier
            indicator_params: Indicator parameters

        Returns:
            DataFrame with trend indicators
        """
        try:
            # SMA indicators
            sma_params = indicator_params.get("sma", {})
            sma_lengths = sma_params.get("lengths", [10, 20, 50])

            for length in sma_lengths:
                df[f"sma_{length}_{timeframe}"] = df.ta.sma(
                    close=df["close"],
                    length=length,
                )
                df[f"price_vs_sma_{length}_{timeframe}"] = (
                    df["close"] / df[f"sma_{length}_{timeframe}"] - 1
                )

            # EMA indicators
            ema_params = indicator_params.get("ema", {})
            ema_lengths = ema_params.get("lengths", [7, 14, 30])

            for length in ema_lengths:
                df[f"ema_{length}_{timeframe}"] = df.ta.ema(
                    close=df["close"],
                    length=length,
                )
                df[f"price_vs_ema_{length}_{timeframe}"] = (
                    df["close"] / df[f"ema_{length}_{timeframe}"] - 1
                )

            # Trend strength
            df[f"trend_strength_{timeframe}"] = (
                df[f"ema_{ema_lengths[0]}_{timeframe}"]
                - df[f"ema_{ema_lengths[-1]}_{timeframe}"]
            ) / df[f"ema_{ema_lengths[-1]}_{timeframe}"]

            return df

        except Exception as e:
            self.logger.error(
                f"Error calculating trend indicators for {timeframe}: {e}",
            )
            return df

    def _add_timeframe_metadata(
        self,
        df: pd.DataFrame,
        timeframe: str,
        tf_params: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Add timeframe metadata to the features DataFrame.

        Args:
            df: Features DataFrame
            timeframe: Timeframe identifier
            tf_params: Timeframe parameters

        Returns:
            DataFrame with metadata added
        """
        try:
            # Add timeframe information
            df["timeframe"] = timeframe
            df["trading_style"] = tf_params.get("trading_style", "unknown")
            df["timeframe_description"] = tf_params.get("description", "")

            # Add timeframe-specific flags
            if timeframe in ["1m", "5m"]:
                df["is_execution_timeframe"] = True
                df["is_tactical_timeframe"] = False
                df["is_strategic_timeframe"] = False
            elif timeframe == "15m":
                df["is_execution_timeframe"] = False
                df["is_tactical_timeframe"] = True
                df["is_strategic_timeframe"] = False
            elif timeframe == "1h":
                df["is_execution_timeframe"] = False
                df["is_tactical_timeframe"] = False
                df["is_strategic_timeframe"] = True
            else:
                df["is_execution_timeframe"] = False
                df["is_tactical_timeframe"] = False
                df["is_strategic_timeframe"] = False

            return df

        except Exception as e:
            self.logger.error(f"Error adding timeframe metadata for {timeframe}: {e}")
            return df

    def _cache_features(self, timeframe: str, features: pd.DataFrame) -> None:
        """
        Cache features for performance optimization.

        Args:
            timeframe: Timeframe identifier
            features: Features DataFrame
        """
        try:
            cache_key = f"{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            self.feature_cache[cache_key] = features.copy()

            # Limit cache size
            if len(self.feature_cache) > 50:
                oldest_key = min(self.feature_cache.keys())
                del self.feature_cache[oldest_key]

        except Exception as e:
            self.logger.error(f"Error caching features for {timeframe}: {e}")

    def _clean_cache(self) -> None:
        """Clean old entries from the feature cache."""
        try:
            current_time = datetime.now()
            if (current_time - self.last_cache_cleanup) > timedelta(minutes=10):
                keys_to_remove = []

                for key in self.feature_cache.keys():
                    # Extract timestamp from key
                    try:
                        timestamp_str = key.split("_")[-2] + "_" + key.split("_")[-1]
                        cache_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M")

                        if (current_time - cache_time) > self.cache_duration:
                            keys_to_remove.append(key)
                    except:
                        keys_to_remove.append(key)

                for key in keys_to_remove:
                    del self.feature_cache[key]

                self.last_cache_cleanup = current_time

        except Exception as e:
            self.logger.error(f"Error cleaning cache: {e}")

    def get_timeframe_parameters(self, timeframe: str) -> dict[str, Any]:
        """
        Get parameters for a specific timeframe.

        Args:
            timeframe: Timeframe identifier

        Returns:
            Dictionary with timeframe parameters
        """
        return self.timeframe_parameters.get(timeframe, {})

    def get_supported_timeframes(self) -> list[str]:
        """
        Get list of supported timeframes.

        Returns:
            List of supported timeframe identifiers
        """
        return list(self.timeframe_parameters.keys())

    def get_feature_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the multi-timeframe feature engineering system.

        Returns:
            Dictionary with system statistics
        """
        return {
            "supported_timeframes": self.get_supported_timeframes(),
            "cache_size": len(self.feature_cache),
            "enable_mtf_features": self.enable_mtf_features,
            "enable_timeframe_adaptation": self.enable_timeframe_adaptation,
            "last_cache_cleanup": self.last_cache_cleanup.isoformat(),
            "timeframe_parameters_count": len(self.timeframe_parameters),
        }
