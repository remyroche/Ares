# src/analyst/advanced_feature_engineering.py

"""
Advanced Feature Engineering System

This module provides advanced technical analysis features including:

1. Advanced TA: Divergence Detection, Pattern Recognition, Volume Profile Analysis
2. Market Microstructure: Bid/ask imbalances, spread dynamics, market depth, liquidity
3. Volatility Targeting: Dynamic position sizing based on volatility
4. Volume Indicators: VPVR, Point of Control, VWAP, OBV, OBV Divergence
5. Advanced Momentum: ROC, Williams %R, CCI, Money Flow Index
6. S/R Zone Definition: BBands, Keltner Channel, VWAP Std Dev Bands, Volume Profile

This extends the existing feature engineering with advanced market microstructure analysis.
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyst.feature_engineering import FeatureEngineeringEngine
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class AdvancedFeatureEngineering:
    """
    Advanced feature engineering system with market microstructure analysis.

    This class provides advanced technical analysis features including:
    - Divergence detection (price/indicator)
    - Pattern recognition
    - Volume profile analysis
    - Market microstructure analysis
    - Volatility targeting
    - Advanced momentum indicators
    - S/R zone definition
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the advanced feature engineering system.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("AdvancedFeatureEngineering")

        # Initialize base feature engineering engine
        self.base_feature_engine = FeatureEngineeringEngine(config)

        # Advanced feature configuration
        self.advanced_config = config.get("advanced_feature_engineering", {})
        self.enable_divergence_detection = self.advanced_config.get(
            "enable_divergence_detection",
            True,
        )
        self.enable_pattern_recognition = self.advanced_config.get(
            "enable_pattern_recognition",
            True,
        )
        self.enable_volume_profile = self.advanced_config.get(
            "enable_volume_profile",
            True,
        )
        self.enable_market_microstructure = self.advanced_config.get(
            "enable_market_microstructure",
            True,
        )
        self.enable_volatility_targeting = self.advanced_config.get(
            "enable_volatility_targeting",
            True,
        )

        # Pattern recognition parameters
        self.pattern_config = self.advanced_config.get("pattern_recognition", {})
        self.divergence_config = self.advanced_config.get("divergence_detection", {})
        self.volume_profile_config = self.advanced_config.get("volume_profile", {})

        self.logger.info("🚀 Initialized AdvancedFeatureEngineering")
        self.logger.info(f"📊 Advanced features enabled: {self.advanced_config}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=pd.DataFrame(),
        context="advanced feature generation",
    )
    async def generate_advanced_features(
        self,
        klines_df: pd.DataFrame,
        agg_trades_df: pd.DataFrame,
        order_book_df: pd.DataFrame | None = None,
        sr_levels: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Generate advanced features including market microstructure analysis.

        Args:
            klines_df: OHLCV data
            agg_trades_df: Aggregated trades data
            order_book_df: Optional order book data
            sr_levels: Optional support/resistance levels

        Returns:
            DataFrame with advanced features
        """
        try:
            self.logger.info("🎯 Generating advanced features...")

            # Start with base features
            base_features = self.base_feature_engine.generate_all_features(
                klines_df=klines_df,
                agg_trades_df=agg_trades_df,
                futures_df=pd.DataFrame(),  # Skip for now
                sr_levels=sr_levels or [],
            )

            features = base_features.copy()

            # Generate advanced features
            if self.enable_divergence_detection:
                features = self._calculate_divergence_features(features)

            if self.enable_pattern_recognition:
                features = self._calculate_pattern_recognition_features(features)

            if self.enable_volume_profile:
                features = self._calculate_volume_profile_features(
                    features,
                    agg_trades_df,
                )

            if self.enable_market_microstructure:
                features = self._calculate_market_microstructure_features(
                    features,
                    order_book_df,
                )

            if self.enable_volatility_targeting:
                features = self._calculate_volatility_targeting_features(features)

            # Advanced momentum indicators
            features = self._calculate_advanced_momentum_indicators(features)

            # S/R zone definition
            features = self._calculate_sr_zone_features(features)

            self.logger.info(f"✅ Generated {len(features.columns)} advanced features")
            return features

        except Exception as e:
            self.logger.error(f"Error generating advanced features: {e}")
            return klines_df.copy()

    def _calculate_divergence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate divergence detection features.

        Args:
            df: Features DataFrame

        Returns:
            DataFrame with divergence features
        """
        try:
            self.logger.info("🔍 Calculating divergence features...")

            # RSI Divergence
            df = self._calculate_rsi_divergence(df)

            # MACD Divergence
            df = self._calculate_macd_divergence(df)

            # Price/Volume Divergence
            df = self._calculate_price_volume_divergence(df)

            # OBV Divergence
            df = self._calculate_obv_divergence(df)

            return df

        except Exception as e:
            self.logger.error(f"Error calculating divergence features: {e}")
            return df

    def _calculate_rsi_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI divergence.

        Args:
            df: Features DataFrame

        Returns:
            DataFrame with RSI divergence features
        """
        try:
            if "rsi" not in df.columns:
                return df

            # Find peaks and troughs in price and RSI
            price_peaks, _ = find_peaks(df["close"].values, distance=10)
            price_troughs, _ = find_peaks(-df["close"].values, distance=10)
            rsi_peaks, _ = find_peaks(df["rsi"].values, distance=10)
            rsi_troughs, _ = find_peaks(-df["rsi"].values, distance=10)

            # Bullish divergence (price lower lows, RSI higher lows)
            df["rsi_bullish_divergence"] = 0
            if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
                price_trend = (
                    df["close"].iloc[price_troughs[-1]]
                    < df["close"].iloc[price_troughs[-2]]
                )
                rsi_trend = (
                    df["rsi"].iloc[rsi_troughs[-1]] > df["rsi"].iloc[rsi_troughs[-2]]
                )
                df.loc[df.index[-1], "rsi_bullish_divergence"] = (
                    1 if price_trend and rsi_trend else 0
                )

            # Bearish divergence (price higher highs, RSI lower highs)
            df["rsi_bearish_divergence"] = 0
            if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
                price_trend = (
                    df["close"].iloc[price_peaks[-1]]
                    > df["close"].iloc[price_peaks[-2]]
                )
                rsi_trend = (
                    df["rsi"].iloc[rsi_peaks[-1]] < df["rsi"].iloc[rsi_peaks[-2]]
                )
                df.loc[df.index[-1], "rsi_bearish_divergence"] = (
                    1 if price_trend and rsi_trend else 0
                )

            return df

        except Exception as e:
            self.logger.error(f"Error calculating RSI divergence: {e}")
            df["rsi_bullish_divergence"] = 0
            df["rsi_bearish_divergence"] = 0
            return df

    def _calculate_macd_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD divergence.

        Args:
            df: Features DataFrame

        Returns:
            DataFrame with MACD divergence features
        """
        try:
            if "MACD" not in df.columns:
                return df

            # Find peaks and troughs in price and MACD
            price_peaks, _ = find_peaks(df["close"].values, distance=10)
            price_troughs, _ = find_peaks(-df["close"].values, distance=10)
            macd_peaks, _ = find_peaks(df["MACD"].values, distance=10)
            macd_troughs, _ = find_peaks(-df["MACD"].values, distance=10)

            # Bullish divergence
            df["macd_bullish_divergence"] = 0
            if len(price_troughs) >= 2 and len(macd_troughs) >= 2:
                price_trend = (
                    df["close"].iloc[price_troughs[-1]]
                    < df["close"].iloc[price_troughs[-2]]
                )
                macd_trend = (
                    df["MACD"].iloc[macd_troughs[-1]]
                    > df["MACD"].iloc[macd_troughs[-2]]
                )
                df.loc[df.index[-1], "macd_bullish_divergence"] = (
                    1 if price_trend and macd_trend else 0
                )

            # Bearish divergence
            df["macd_bearish_divergence"] = 0
            if len(price_peaks) >= 2 and len(macd_peaks) >= 2:
                price_trend = (
                    df["close"].iloc[price_peaks[-1]]
                    > df["close"].iloc[price_peaks[-2]]
                )
                macd_trend = (
                    df["MACD"].iloc[macd_peaks[-1]] < df["MACD"].iloc[macd_peaks[-2]]
                )
                df.loc[df.index[-1], "macd_bearish_divergence"] = (
                    1 if price_trend and macd_trend else 0
                )

            return df

        except Exception as e:
            self.logger.error(f"Error calculating MACD divergence: {e}")
            df["macd_bullish_divergence"] = 0
            df["macd_bearish_divergence"] = 0
            return df

    def _calculate_price_volume_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price/volume divergence.

        Args:
            df: Features DataFrame

        Returns:
            DataFrame with price/volume divergence features
        """
        try:
            # Price momentum vs volume momentum
            price_momentum = df["close"].pct_change(5)
            volume_momentum = df["volume"].pct_change(5)

            # Divergence when price and volume move in opposite directions
            df["price_volume_divergence"] = np.where(
                (price_momentum > 0) & (volume_momentum < 0),
                1,  # Bullish divergence (price up, volume down)
                np.where(
                    (price_momentum < 0) & (volume_momentum > 0),
                    -1,  # Bearish divergence (price down, volume up)
                    0,  # No divergence
                ),
            )

            return df

        except Exception as e:
            self.logger.error(f"Error calculating price/volume divergence: {e}")
            df["price_volume_divergence"] = 0
            return df

    def _calculate_obv_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate OBV divergence.

        Args:
            df: Features DataFrame

        Returns:
            DataFrame with OBV divergence features
        """
        try:
            if "OBV" not in df.columns:
                return df

            # Find peaks and troughs in price and OBV
            price_peaks, _ = find_peaks(df["close"].values, distance=10)
            price_troughs, _ = find_peaks(-df["close"].values, distance=10)
            obv_peaks, _ = find_peaks(df["OBV"].values, distance=10)
            obv_troughs, _ = find_peaks(-df["OBV"].values, distance=10)

            # Bullish divergence
            df["obv_bullish_divergence"] = 0
            if len(price_troughs) >= 2 and len(obv_troughs) >= 2:
                price_trend = (
                    df["close"].iloc[price_troughs[-1]]
                    < df["close"].iloc[price_troughs[-2]]
                )
                obv_trend = (
                    df["OBV"].iloc[obv_troughs[-1]] > df["OBV"].iloc[obv_troughs[-2]]
                )
                df.loc[df.index[-1], "obv_bullish_divergence"] = (
                    1 if price_trend and obv_trend else 0
                )

            # Bearish divergence
            df["obv_bearish_divergence"] = 0
            if len(price_peaks) >= 2 and len(obv_peaks) >= 2:
                price_trend = (
                    df["close"].iloc[price_peaks[-1]]
                    > df["close"].iloc[price_peaks[-2]]
                )
                obv_trend = (
                    df["OBV"].iloc[obv_peaks[-1]] < df["OBV"].iloc[obv_peaks[-2]]
                )
                df.loc[df.index[-1], "obv_bearish_divergence"] = (
                    1 if price_trend and obv_trend else 0
                )

            return df

        except Exception as e:
            self.logger.error(f"Error calculating OBV divergence: {e}")
            df["obv_bullish_divergence"] = 0
            df["obv_bearish_divergence"] = 0
            return df

    def _calculate_pattern_recognition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pattern recognition features.

        Args:
            df: Features DataFrame

        Returns:
            DataFrame with pattern recognition features
        """
        try:
            self.logger.info("📊 Calculating pattern recognition features...")

            # Double top/bottom
            df = self._calculate_double_patterns(df)

            # Head and shoulders
            df = self._calculate_head_shoulders(df)

            # Triangle patterns
            df = self._calculate_triangle_patterns(df)

            # Flag and pennant patterns
            df = self._calculate_flag_pennant_patterns(df)

            return df

        except Exception as e:
            self.logger.error(f"Error calculating pattern recognition features: {e}")
            return df

    def _calculate_double_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate double top/bottom patterns.

        Args:
            df: Features DataFrame

        Returns:
            DataFrame with double pattern features
        """
        try:
            # Find peaks and troughs
            peaks, _ = find_peaks(df["high"].values, distance=20)
            troughs, _ = find_peaks(-df["low"].values, distance=20)

            df["double_top"] = 0
            df["double_bottom"] = 0

            # Double top detection
            if len(peaks) >= 2:
                peak1, peak2 = peaks[-2], peaks[-1]
                if (
                    abs(df["high"].iloc[peak1] - df["high"].iloc[peak2])
                    / df["high"].iloc[peak1]
                    < 0.02
                ):  # 2% tolerance
                    df.loc[df.index[peak2], "double_top"] = 1

            # Double bottom detection
            if len(troughs) >= 2:
                trough1, trough2 = troughs[-2], troughs[-1]
                if (
                    abs(df["low"].iloc[trough1] - df["low"].iloc[trough2])
                    / df["low"].iloc[trough1]
                    < 0.02
                ):  # 2% tolerance
                    df.loc[df.index[trough2], "double_bottom"] = 1

            return df

        except Exception as e:
            self.logger.error(f"Error calculating double patterns: {e}")
            df["double_top"] = 0
            df["double_bottom"] = 0
            return df

    def _calculate_head_shoulders(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate head and shoulders patterns.

        Args:
            df: Features DataFrame

        Returns:
            DataFrame with head and shoulders features
        """
        try:
            # Find peaks
            peaks, _ = find_peaks(df["high"].values, distance=20)

            df["head_shoulders"] = 0
            df["inverse_head_shoulders"] = 0

            if len(peaks) >= 3:
                # Check for head and shoulders pattern
                left_shoulder = peaks[-3]
                head = peaks[-2]
                right_shoulder = peaks[-1]

                # Head should be higher than shoulders
                if (
                    df["high"].iloc[head] > df["high"].iloc[left_shoulder]
                    and df["high"].iloc[head] > df["high"].iloc[right_shoulder]
                    and abs(
                        df["high"].iloc[left_shoulder]
                        - df["high"].iloc[right_shoulder],
                    )
                    / df["high"].iloc[left_shoulder]
                    < 0.05
                ):
                    df.loc[df.index[right_shoulder], "head_shoulders"] = 1

            return df

        except Exception as e:
            self.logger.error(f"Error calculating head and shoulders: {e}")
            df["head_shoulders"] = 0
            df["inverse_head_shoulders"] = 0
            return df

    def _calculate_triangle_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate triangle patterns.

        Args:
            df: Features DataFrame

        Returns:
            DataFrame with triangle pattern features
        """
        try:
            # Calculate trend lines for triangle detection
            window = 20

            df["ascending_triangle"] = 0
            df["descending_triangle"] = 0
            df["symmetrical_triangle"] = 0

            for i in range(window, len(df)):
                recent_data = df.iloc[i - window : i + 1]

                # Calculate trend lines
                highs = recent_data["high"].values
                lows = recent_data["low"].values
                x = np.arange(len(highs))

                # High trend line
                high_slope, high_intercept, _, _, _ = stats.linregress(x, highs)

                # Low trend line
                low_slope, low_intercept, _, _, _ = stats.linregress(x, lows)

                # Triangle classification
                if high_slope < -0.001 and low_slope > 0.001:  # Ascending triangle
                    df.loc[df.index[i], "ascending_triangle"] = 1
                elif high_slope < -0.001 and low_slope < -0.001:  # Descending triangle
                    df.loc[df.index[i], "descending_triangle"] = 1
                elif abs(high_slope + low_slope) < 0.001:  # Symmetrical triangle
                    df.loc[df.index[i], "symmetrical_triangle"] = 1

            return df

        except Exception as e:
            self.logger.error(f"Error calculating triangle patterns: {e}")
            df["ascending_triangle"] = 0
            df["descending_triangle"] = 0
            df["symmetrical_triangle"] = 0
            return df

    def _calculate_flag_pennant_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate flag and pennant patterns.

        Args:
            df: Features DataFrame

        Returns:
            DataFrame with flag/pennant features
        """
        try:
            window = 10

            df["bull_flag"] = 0
            df["bear_flag"] = 0
            df["pennant"] = 0

            for i in range(window, len(df)):
                recent_data = df.iloc[i - window : i + 1]

                # Check for strong move followed by consolidation
                price_change = (
                    recent_data["close"].iloc[-1] - recent_data["close"].iloc[0]
                ) / recent_data["close"].iloc[0]
                volatility = recent_data["close"].pct_change().std()

                if (
                    abs(price_change) > 0.05 and volatility < 0.02
                ):  # Strong move with low volatility
                    if price_change > 0:
                        df.loc[df.index[i], "bull_flag"] = 1
                    else:
                        df.loc[df.index[i], "bear_flag"] = 1

                # Pennant (symmetrical consolidation)
                if abs(price_change) < 0.02 and volatility < 0.015:
                    df.loc[df.index[i], "pennant"] = 1

            return df

        except Exception as e:
            self.logger.error(f"Error calculating flag/pennant patterns: {e}")
            df["bull_flag"] = 0
            df["bear_flag"] = 0
            df["pennant"] = 0
            return df

    def _calculate_volume_profile_features(
        self,
        df: pd.DataFrame,
        agg_trades_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate volume profile features.

        Args:
            df: Features DataFrame
            agg_trades_df: Aggregated trades data

        Returns:
            DataFrame with volume profile features
        """
        try:
            self.logger.info("📊 Calculating volume profile features...")

            # Enhanced VPVR features using OHLCV data (aligned with SR analyzer)
            df = self._calculate_vpvr_features_ohlcv(df)

            # Additional VPVR features using aggregated trades data
            if not agg_trades_df.empty:
                df = self._calculate_vpvr_features(df, agg_trades_df)

                # Point of Control
                df = self._calculate_point_of_control(df, agg_trades_df)

                # High Volume Nodes (HVN) and Low Volume Nodes (LVN)
                df = self._calculate_volume_nodes(df, agg_trades_df)

                # Cumulative Volume Delta (CVD)
                df = self._calculate_cvd_features(df, agg_trades_df)

            return df

        except Exception as e:
            self.logger.error(f"Error calculating volume profile features: {e}")
            return df

    def _calculate_vpvr_features(
        self,
        df: pd.DataFrame,
        agg_trades_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate Volume Profile Visible Range features using aggregated trades data.

        Args:
            df: Features DataFrame
            agg_trades_df: Aggregated trades data

        Returns:
            DataFrame with VPVR features
        """
        try:
            # Create price bins
            price_range = df["high"].max() - df["low"].min()
            bin_size = price_range / 50  # 50 price bins

            # Calculate volume profile
            agg_trades_df["price_bin"] = pd.cut(agg_trades_df["price"], bins=50)
            volume_profile = agg_trades_df.groupby("price_bin")["quantity"].sum()

            # Find high volume areas
            volume_threshold = volume_profile.quantile(0.75)
            high_volume_prices = volume_profile[volume_profile > volume_threshold].index

            # Calculate distance to high volume areas
            df["distance_to_hvn"] = df["close"].apply(
                lambda x: min(
                    [abs(x - (p.left + p.right) / 2) for p in high_volume_prices],
                )
                if len(high_volume_prices) > 0
                else 0,
            )

            return df

        except Exception as e:
            self.logger.error(f"Error calculating VPVR features: {e}")
            df["distance_to_hvn"] = 0
            return df

    def _calculate_vpvr_features_ohlcv(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate Volume Profile Visible Range features using OHLCV data.
        This method aligns with the SR analyzer implementation.

        Args:
            df: Features DataFrame with OHLCV columns

        Returns:
            DataFrame with VPVR features
        """
        try:
            from src.analyst.data_utils import calculate_volume_profile
            
            # Prepare data for VPVR calculation
            vpvr_data = df.copy()
            vpvr_data.columns = [col.capitalize() for col in vpvr_data.columns]
            
            # Calculate volume profile using data_utils with many bins for maximum detection
            volume_profile_result = calculate_volume_profile(vpvr_data, num_bins=100)
            
            current_price = df['close'].iloc[-1]
            
            # Distance to Point of Control (POC)
            if not pd.isna(volume_profile_result['poc']):
                poc_price = volume_profile_result['poc']
                df['distance_to_poc'] = abs(df['close'] - poc_price) / poc_price
                df['poc_strength'] = 0.9  # POC is very strong
            else:
                df['distance_to_poc'] = 0
                df['poc_strength'] = 0
            
            # Distance to High Volume Nodes (HVNs)
            if volume_profile_result['hvn_levels']:
                df['distance_to_hvn'] = df['close'].apply(
                    lambda x: min([abs(x - hvn) for hvn in volume_profile_result['hvn_levels']])
                )
                df['hvn_strength'] = 0.7  # HVNs are strong
            else:
                df['distance_to_hvn'] = 0
                df['hvn_strength'] = 0
            
            # Distance to Low Volume Nodes (LVNs)
            if volume_profile_result['lvn_levels']:
                df['distance_to_lvn'] = df['close'].apply(
                    lambda x: min([abs(x - lvn) for lvn in volume_profile_result['lvn_levels']])
                )
                df['lvn_strength'] = 0.3  # LVNs are weaker
            else:
                df['distance_to_lvn'] = 0
                df['lvn_strength'] = 0
            
            # VPVR-based support/resistance proximity
            df['near_vpvr_support'] = 0
            df['near_vpvr_resistance'] = 0
            
            for _, row in df.iterrows():
                price = row['close']
                
                # Check if near VPVR support levels
                support_levels = [level for level in volume_profile_result['hvn_levels'] 
                                if level < price]
                if support_levels:
                    nearest_support = min(support_levels, key=lambda x: abs(price - x))
                    if abs(price - nearest_support) / price < 0.02:  # 2% tolerance
                        df.loc[df.index == row.name, 'near_vpvr_support'] = 1
                
                # Check if near VPVR resistance levels
                resistance_levels = [level for level in volume_profile_result['hvn_levels'] 
                                   if level > price]
                if resistance_levels:
                    nearest_resistance = min(resistance_levels, key=lambda x: abs(price - x))
                    if abs(price - nearest_resistance) / price < 0.02:  # 2% tolerance
                        df.loc[df.index == row.name, 'near_vpvr_resistance'] = 1
            
            return df

        except Exception as e:
            self.logger.error(f"Error calculating VPVR features (OHLCV): {e}")
            # Set default values
            df['distance_to_poc'] = 0
            df['poc_strength'] = 0
            df['distance_to_hvn'] = 0
            df['hvn_strength'] = 0
            df['distance_to_lvn'] = 0
            df['lvn_strength'] = 0
            df['near_vpvr_support'] = 0
            df['near_vpvr_resistance'] = 0
            return df

    def _calculate_point_of_control(
        self,
        df: pd.DataFrame,
        agg_trades_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate Point of Control (POC).

        Args:
            df: Features DataFrame
            agg_trades_df: Aggregated trades data

        Returns:
            DataFrame with POC features
        """
        try:
            # Calculate POC as the price level with highest volume
            price_volume = agg_trades_df.groupby("price")["quantity"].sum()
            poc_price = price_volume.idxmax()

            # Distance to POC
            df["distance_to_poc"] = abs(df["close"] - poc_price) / poc_price

            # POC strength (volume at POC relative to average)
            poc_volume = price_volume.max()
            avg_volume = price_volume.mean()
            df["poc_strength"] = poc_volume / avg_volume if avg_volume > 0 else 1

            # Enhanced POC features
            current_price = df["close"].iloc[-1]
            
            # POC position relative to current price
            df["poc_above_current"] = 1 if poc_price > current_price else 0
            df["poc_below_current"] = 1 if poc_price < current_price else 0
            
            # POC dominance (how much stronger POC is compared to other levels)
            volume_sorted = price_volume.sort_values(ascending=False)
            if len(volume_sorted) > 1:
                second_highest_volume = volume_sorted.iloc[1]
                df["poc_dominance"] = poc_volume / second_highest_volume if second_highest_volume > 0 else 1
            else:
                df["poc_dominance"] = 1
            
            # POC stability (how stable the POC has been)
            price_tolerance = poc_price * 0.001  # 0.1% tolerance
            stable_volume = price_volume[
                (price_volume.index >= poc_price - price_tolerance) &
                (price_volume.index <= poc_price + price_tolerance)
            ].sum()
            df["poc_stability"] = stable_volume / poc_volume if poc_volume > 0 else 0

            return df

        except Exception as e:
            self.logger.error(f"Error calculating POC features: {e}")
            df["distance_to_poc"] = 0
            df["poc_strength"] = 1
            df["poc_above_current"] = 0
            df["poc_below_current"] = 0
            df["poc_dominance"] = 1
            df["poc_stability"] = 0
            return df

    def _calculate_volume_nodes(
        self,
        df: pd.DataFrame,
        agg_trades_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate High Volume Nodes (HVN) and Low Volume Nodes (LVN).

        Args:
            df: Features DataFrame
            agg_trades_df: Aggregated trades data

        Returns:
            DataFrame with volume node features
        """
        try:
            # Calculate volume profile
            price_volume = agg_trades_df.groupby("price")["quantity"].sum()

            # Find HVN and LVN using more sophisticated criteria
            volume_threshold_high = price_volume.quantile(0.8)
            volume_threshold_low = price_volume.quantile(0.2)

            hvn_prices = price_volume[price_volume > volume_threshold_high].index
            lvn_prices = price_volume[price_volume < volume_threshold_low].index

            # Distance to nearest HVN and LVN
            df["distance_to_hvn"] = df["close"].apply(
                lambda x: min([abs(x - p) for p in hvn_prices])
                if len(hvn_prices) > 0
                else 0,
            )

            df["distance_to_lvn"] = df["close"].apply(
                lambda x: min([abs(x - p) for p in lvn_prices])
                if len(lvn_prices) > 0
                else 0,
            )

            # Enhanced volume node features
            df["hvn_count"] = len(hvn_prices)
            df["lvn_count"] = len(lvn_prices)
            
            # Volume concentration at current price level
            current_price = df["close"].iloc[-1]
            price_tolerance = current_price * 0.001  # 0.1% tolerance
            
            nearby_volume = price_volume[
                (price_volume.index >= current_price - price_tolerance) &
                (price_volume.index <= current_price + price_tolerance)
            ].sum()
            
            total_volume = price_volume.sum()
            df["volume_concentration"] = nearby_volume / total_volume if total_volume > 0 else 0

            return df

        except Exception as e:
            self.logger.error(f"Error calculating volume nodes: {e}")
            df["distance_to_hvn"] = 0
            df["distance_to_lvn"] = 0
            df["hvn_count"] = 0
            df["lvn_count"] = 0
            df["volume_concentration"] = 0
            return df

    def _calculate_cvd_features(
        self,
        df: pd.DataFrame,
        agg_trades_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate Cumulative Volume Delta (CVD) features.

        Args:
            df: Features DataFrame
            agg_trades_df: Aggregated trades data

        Returns:
            DataFrame with CVD features
        """
        try:
            # Calculate CVD
            agg_trades_df["delta"] = agg_trades_df["quantity"] * np.where(
                agg_trades_df["is_buyer_maker"],
                -1,
                1,
            )

            # Cumulative sum
            cvd = agg_trades_df["delta"].cumsum()

            # Resample to match main DataFrame
            cvd_resampled = cvd.resample("1T").last().reindex(df.index, method="ffill")

            df["cvd"] = cvd_resampled.fillna(0)
            df["cvd_change"] = df["cvd"].diff()
            df["cvd_momentum"] = df["cvd"].diff(5)

            return df

        except Exception as e:
            self.logger.error(f"Error calculating CVD features: {e}")
            df["cvd"] = 0
            df["cvd_change"] = 0
            df["cvd_momentum"] = 0
            return df

    def _calculate_market_microstructure_features(
        self,
        df: pd.DataFrame,
        order_book_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """
        Calculate market microstructure features.

        Args:
            df: Features DataFrame
            order_book_df: Optional order book data

        Returns:
            DataFrame with market microstructure features
        """
        try:
            self.logger.info("🔍 Calculating market microstructure features...")

            # Initialize default values
            df["bid_ask_spread"] = 0.0
            df["order_imbalance"] = 0.0
            df["market_depth"] = 0.0
            df["liquidity_score"] = 0.0

            if order_book_df is not None and not order_book_df.empty:
                # Calculate bid/ask spread
                df = self._calculate_bid_ask_features(df, order_book_df)

                # Calculate order imbalances
                df = self._calculate_order_imbalance_features(df, order_book_df)

                # Calculate market depth
                df = self._calculate_market_depth_features(df, order_book_df)

            return df

        except Exception as e:
            self.logger.error(f"Error calculating market microstructure features: {e}")
            return df

    def _calculate_bid_ask_features(
        self,
        df: pd.DataFrame,
        order_book_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate bid/ask spread features.

        Args:
            df: Features DataFrame
            order_book_df: Order book data

        Returns:
            DataFrame with bid/ask features
        """
        try:
            # Calculate spread
            order_book_df["spread"] = (
                order_book_df["ask_price"] - order_book_df["bid_price"]
            )
            order_book_df["spread_pct"] = (
                order_book_df["spread"] / order_book_df["bid_price"]
            )

            # Resample to match main DataFrame
            spread_resampled = (
                order_book_df["spread_pct"]
                .resample("1T")
                .mean()
                .reindex(df.index, method="ffill")
            )

            df["bid_ask_spread"] = spread_resampled.fillna(0)

            return df

        except Exception as e:
            self.logger.error(f"Error calculating bid/ask features: {e}")
            df["bid_ask_spread"] = 0
            return df

    def _calculate_order_imbalance_features(
        self,
        df: pd.DataFrame,
        order_book_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate order imbalance features.

        Args:
            df: Features DataFrame
            order_book_df: Order book data

        Returns:
            DataFrame with order imbalance features
        """
        try:
            # Calculate order imbalance
            order_book_df["bid_volume"] = order_book_df["bid_quantity"].sum()
            order_book_df["ask_volume"] = order_book_df["ask_quantity"].sum()
            order_book_df["imbalance"] = (
                order_book_df["bid_volume"] - order_book_df["ask_volume"]
            ) / (order_book_df["bid_volume"] + order_book_df["ask_volume"])

            # Resample to match main DataFrame
            imbalance_resampled = (
                order_book_df["imbalance"]
                .resample("1T")
                .mean()
                .reindex(df.index, method="ffill")
            )

            df["order_imbalance"] = imbalance_resampled.fillna(0)

            return df

        except Exception as e:
            self.logger.error(f"Error calculating order imbalance features: {e}")
            df["order_imbalance"] = 0
            return df

    def _calculate_market_depth_features(
        self,
        df: pd.DataFrame,
        order_book_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate market depth features.

        Args:
            df: Features DataFrame
            order_book_df: Order book data

        Returns:
            DataFrame with market depth features
        """
        try:
            # Calculate market depth (total volume within spread)
            order_book_df["market_depth"] = (
                order_book_df["bid_quantity"].sum()
                + order_book_df["ask_quantity"].sum()
            )

            # Resample to match main DataFrame
            depth_resampled = (
                order_book_df["market_depth"]
                .resample("1T")
                .mean()
                .reindex(df.index, method="ffill")
            )

            df["market_depth"] = depth_resampled.fillna(0)

            # Liquidity score (depth relative to price)
            df["liquidity_score"] = df["market_depth"] / df["close"]

            return df

        except Exception as e:
            self.logger.error(f"Error calculating market depth features: {e}")
            df["market_depth"] = 0
            df["liquidity_score"] = 0
            return df

    def _calculate_volatility_targeting_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate volatility targeting features.

        Args:
            df: Features DataFrame

        Returns:
            DataFrame with volatility targeting features
        """
        try:
            self.logger.info("📈 Calculating volatility targeting features...")

            # Calculate rolling volatility
            returns = df["close"].pct_change()
            rolling_vol = returns.rolling(window=20).std()

            # Target volatility (adjustable)
            target_vol = self.advanced_config.get("target_volatility", 0.15)

            # Volatility targeting position sizing
            df["volatility_position_size"] = target_vol / (rolling_vol + 1e-8)

            # Volatility regime
            df["volatility_regime"] = np.where(
                rolling_vol < target_vol * 0.5,
                "low",
                np.where(rolling_vol > target_vol * 1.5, "high", "medium"),
            )

            # Leverage adjustment based on volatility
            df["leverage_adjustment"] = np.where(
                df["volatility_regime"] == "low",
                1.5,  # Higher leverage in low vol
                np.where(
                    df["volatility_regime"] == "high",
                    0.5,
                    1.0,
                ),  # Lower leverage in high vol
            )

            return df

        except Exception as e:
            self.logger.error(f"Error calculating volatility targeting features: {e}")
            df["volatility_position_size"] = 1.0
            df["volatility_regime"] = "medium"
            df["leverage_adjustment"] = 1.0
            return df

    def _calculate_advanced_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced momentum indicators.

        Args:
            df: Features DataFrame

        Returns:
            DataFrame with advanced momentum features
        """
        try:
            self.logger.info("📊 Calculating advanced momentum indicators...")

            # Rate of Change (ROC)
            roc_period = self.advanced_config.get("roc_period", 10)
            df["roc"] = df["close"].pct_change(roc_period)

            # Williams %R
            willr_period = self.advanced_config.get("willr_period", 14)
            df["williams_r"] = df.ta.willr(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                length=willr_period,
            )

            # Commodity Channel Index (CCI)
            cci_period = self.advanced_config.get("cci_period", 20)
            df["cci"] = df.ta.cci(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                length=cci_period,
            )

            # Money Flow Index (MFI)
            mfi_period = self.advanced_config.get("mfi_period", 14)
            df["mfi"] = df.ta.mfi(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                volume=df["volume"],
                length=mfi_period,
            )

            return df

        except Exception as e:
            self.logger.error(f"Error calculating advanced momentum indicators: {e}")
            df["roc"] = 0
            df["williams_r"] = 0
            df["cci"] = 0
            df["mfi"] = 0
            return df

    def _calculate_sr_zone_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate S/R zone definition features.

        Args:
            df: Features DataFrame

        Returns:
            DataFrame with S/R zone features
        """
        try:
            self.logger.info("🎯 Calculating S/R zone features...")

            # Bollinger Bands
            bb_period = self.advanced_config.get("bb_period", 20)
            bb_std = self.advanced_config.get("bb_std", 2)
            bb_result = df.ta.bbands(close=df["close"], length=bb_period, std=bb_std)
            if bb_result is not None:
                df["bb_upper"] = bb_result.iloc[:, 0]
                df["bb_middle"] = bb_result.iloc[:, 1]
                df["bb_lower"] = bb_result.iloc[:, 2]

            # Keltner Channel
            kc_period = self.advanced_config.get("kc_period", 20)
            kc_result = df.ta.kc(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                length=kc_period,
            )
            if kc_result is not None:
                df["kc_upper"] = kc_result.iloc[:, 0]
                df["kc_middle"] = kc_result.iloc[:, 1]
                df["kc_lower"] = kc_result.iloc[:, 2]

            # VWAP Standard Deviation Bands
            if "VWAP" in df.columns:
                vwap_std = df["close"].rolling(window=20).std()
                df["vwap_upper_band"] = df["VWAP"] + (vwap_std * 2)
                df["vwap_lower_band"] = df["VWAP"] - (vwap_std * 2)

            # Support/Resistance zones
            df = self._calculate_sr_zones(df)

            return df

        except Exception as e:
            self.logger.error(f"Error calculating S/R zone features: {e}")
            return df

    def _calculate_sr_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate support/resistance zones.

        Args:
            df: Features DataFrame

        Returns:
            DataFrame with S/R zone features
        """
        try:
            # Find support and resistance levels
            peaks, _ = find_peaks(df["high"].values, distance=20)
            troughs, _ = find_peaks(-df["low"].values, distance=20)

            # Calculate distance to nearest support/resistance
            df["distance_to_resistance"] = df["close"].apply(
                lambda x: min([abs(x - df["high"].iloc[p]) for p in peaks])
                if len(peaks) > 0
                else 0,
            )

            df["distance_to_support"] = df["close"].apply(
                lambda x: min([abs(x - df["low"].iloc[t]) for t in troughs])
                if len(troughs) > 0
                else 0,
            )

            # Zone strength (based on volume at levels)
            df["resistance_strength"] = 0
            df["support_strength"] = 0

            return df

        except Exception as e:
            self.logger.error(f"Error calculating S/R zones: {e}")
            df["distance_to_resistance"] = 0
            df["distance_to_support"] = 0
            df["resistance_strength"] = 0
            df["support_strength"] = 0
            return df

    def get_feature_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the advanced feature engineering system.

        Returns:
            Dictionary with system statistics
        """
        return {
            "enable_divergence_detection": self.enable_divergence_detection,
            "enable_pattern_recognition": self.enable_pattern_recognition,
            "enable_volume_profile": self.enable_volume_profile,
            "enable_market_microstructure": self.enable_market_microstructure,
            "enable_volatility_targeting": self.enable_volatility_targeting,
            "advanced_config": self.advanced_config,
        }

    def get_vpvr_sr_levels(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Get VPVR-based support/resistance levels.
        This method provides the same functionality as the SR analyzer's VPVR detection.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            dict: VPVR S/R levels with POC, HVNs, and LVNs
        """
        try:
            from src.analyst.data_utils import calculate_volume_profile
            
            # Prepare data for VPVR calculation
            vpvr_data = df.copy()
            vpvr_data.columns = [col.capitalize() for col in vpvr_data.columns]
            
            # Calculate volume profile using data_utils with many bins for maximum detection
            volume_profile_result = calculate_volume_profile(vpvr_data, num_bins=100)
            
            current_price = df['close'].iloc[-1]
            
            # Classify levels as support or resistance
            support_levels = []
            resistance_levels = []
            
            # Point of Control (POC)
            if not pd.isna(volume_profile_result['poc']):
                poc_price = volume_profile_result['poc']
                level_type = 'support' if poc_price < current_price else 'resistance'
                level_info = {
                    'price': poc_price,
                    'type': level_type,
                    'strength': 0.9,
                    'method': 'vpvr_poc',
                    'volume_concentration': 1.0  # POC has highest volume
                }
                
                if level_type == 'support':
                    support_levels.append(level_info)
                else:
                    resistance_levels.append(level_info)
            
            # High Volume Nodes (HVNs) with strength information
            if 'hvn_results' in volume_profile_result:
                # Use the new detailed HVN results with strength
                for hvn_result in volume_profile_result['hvn_results']:
                    hvn_price = hvn_result['price']
                    level_type = 'support' if hvn_price < current_price else 'resistance'
                    level_info = {
                        'price': hvn_price,
                        'type': level_type,
                        'strength': hvn_result['strength'],
                        'method': 'vpvr_hvn',
                        'volume_concentration': hvn_result['volume_concentration']
                    }
                    
                    if level_type == 'support':
                        support_levels.append(level_info)
                    else:
                        resistance_levels.append(level_info)
            else:
                # Fallback to old method
                for hvn_price in volume_profile_result['hvn_levels']:
                    level_type = 'support' if hvn_price < current_price else 'resistance'
                    level_info = {
                        'price': hvn_price,
                        'type': level_type,
                        'strength': 0.7,
                        'method': 'vpvr_hvn',
                        'volume_concentration': 0.8
                    }
                    
                    if level_type == 'support':
                        support_levels.append(level_info)
                    else:
                        resistance_levels.append(level_info)
            

            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'poc': volume_profile_result['poc'],
                'hvn_levels': volume_profile_result['hvn_levels'],
                'hvn_results': volume_profile_result.get('hvn_results', []),
                'current_price': current_price
            }
            
        except Exception as e:
            self.logger.error(f"Error getting VPVR S/R levels: {e}")
            return {
                'support_levels': [],
                'resistance_levels': [],
                'poc': None,
                'hvn_levels': [],
                'lvn_levels': [],
                'current_price': None
            }
