import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Any, Optional, Tuple
from src.utils.logger import system_logger
from src.utils.error_handler import (
    handle_errors,
    handle_data_processing_errors,
)


class TechnicalAnalyzer:
    """
    Performs deep technical analysis on market data, calculating a wide range of indicators
    including MAs, VWAP, Volume Profile, and enhanced indicators from ENHANCED_INDICATORS_RECOMMENDATION.md.
    """

    def __init__(self, config: Optional[Dict[Any, Any]] = None):
        self.config = config or {}
        self.logger = system_logger.getChild("TechnicalAnalyzer")

    @handle_errors(
        exceptions=(ValueError, TypeError, KeyError),
        default_return={},
        context="analyze",
    )
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Runs all technical analysis calculations and returns a consolidated dictionary.

        Args:
            df (pd.DataFrame): DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        Returns:
            Dict: A dictionary containing all calculated technical indicators.
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided for technical analysis")
            return {}

        # Ensure correct data types
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Check for NaN values after conversion
        if df[["open", "high", "low", "close", "volume"]].isnull().any().any():
            self.logger.warning("NaN values detected in price data after conversion")

        # --- Standard Indicators (RSI, MACD) ---
        rsi = self._calculate_rsi(df)
        macd = self._calculate_macd(df)

        # --- Moving Averages ---
        mas = self._calculate_moving_averages(df)

        # --- VWAP ---
        vwap = self._calculate_vwap(df)

        # --- Volume Profile ---
        volume_profile = self._calculate_volume_profile(df)

        # --- Enhanced Momentum Indicators ---
        willr = self._calculate_williams_r(df)
        cci = self._calculate_cci(df)
        mfi = self._calculate_mfi(df)
        roc = self._calculate_roc(df)

        # --- Enhanced Trend Indicators ---
        psar = self._calculate_parabolic_sar(df)
        ichimoku = self._calculate_ichimoku_cloud(df)
        supertrend = self._calculate_supertrend(df)

        # --- Enhanced Volatility Indicators ---
        donchian = self._calculate_donchian_channels(df)
        atr_bands = self._calculate_atr_bands(df)

        # --- Enhanced VWAP Variations ---
        vwap_bands = self._calculate_vwap_bands(df)

        current_price = df["close"].iloc[-1]

        return {
            "current_price": current_price,
            "rsi": rsi,
            "macd": macd,
            "moving_averages": mas,
            "vwap": vwap,
            "price_to_vwap_ratio": (current_price / vwap) if vwap and vwap != 0 else 0,
            "volume_profile": volume_profile,
            # Enhanced Momentum Indicators
            "williams_r": willr,
            "cci": cci,
            "mfi": mfi,
            "roc": roc,
            # Enhanced Trend Indicators
            "parabolic_sar": psar,
            "ichimoku": ichimoku,
            "supertrend": supertrend,
            # Enhanced Volatility Indicators
            "donchian_channels": donchian,
            "atr_bands": atr_bands,
            # Enhanced VWAP
            "vwap_bands": vwap_bands,
        }

    @handle_data_processing_errors(default_return=np.nan, context="calculate_rsi")
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculates the Relative Strength Index (RSI)."""
        if len(df) < period:
            self.logger.warning(
                f"Insufficient data for RSI calculation. Need {period}, got {len(df)}"
            )
            return np.nan

        rsi = df.ta.rsi(length=period)
        return rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else np.nan

    @handle_data_processing_errors(default_return={}, context="calculate_macd")
    def _calculate_macd(
        self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Dict:
        """Calculates the Moving Average Convergence Divergence (MACD)."""
        if len(df) < slow:
            self.logger.warning(
                f"Insufficient data for MACD calculation. Need {slow}, got {len(df)}"
            )
            return {}

        macd_df = df.ta.macd(fast=fast, slow=slow, signal=signal)
        if macd_df.empty:
            return {}

        try:
            return {
                "macd": macd_df[f"MACD_{fast}_{slow}_{signal}"].iloc[-1],
                "histogram": macd_df[f"MACDh_{fast}_{slow}_{signal}"].iloc[-1],
                "signal": macd_df[f"MACDs_{fast}_{slow}_{signal}"].iloc[-1],
            }
        except KeyError as e:
            self.logger.error(f"MACD column not found: {e}")
            return {}

    @handle_data_processing_errors(
        default_return={}, context="calculate_moving_averages"
    )
    def _calculate_moving_averages(
        self, df: pd.DataFrame, periods: List[int] = [9, 21, 50, 200]
    ) -> Dict:
        """Calculates multiple Simple Moving Averages (SMAs)."""
        mas = {}
        for period in periods:
            if len(df) < period:
                self.logger.warning(
                    f"Insufficient data for SMA {period}. Need {period}, got {len(df)}"
                )
                continue

            sma = df.ta.sma(length=period)
            if not sma.empty and not pd.isna(sma.iloc[-1]):
                mas[f"sma_{period}"] = sma.iloc[-1]
        return mas

    @handle_data_processing_errors(default_return=np.nan, context="calculate_vwap")
    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculates the Volume-Weighted Average Price (VWAP) for the given period."""
        if "volume" not in df.columns or df["volume"].sum() == 0:
            self.logger.warning("No volume data available for VWAP calculation")
            return np.nan

        try:
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            vwap = (typical_price * df["volume"]).sum() / df["volume"].sum()
            return vwap if not np.isnan(vwap) else np.nan
        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}")
            return np.nan

    @handle_data_processing_errors(
        default_return={}, context="calculate_volume_profile"
    )
    def _calculate_volume_profile(self, df: pd.DataFrame, bins: int = 20) -> Dict:
        """
        Calculates a simple Volume Profile.

        Returns:
            Dict: Containing POC, HVNs, and LVNs.
        """
        try:
            price_range = df["high"].max() - df["low"].min()
            if price_range == 0:
                self.logger.warning("No price range for volume profile calculation")
                return {}

            # Create price bins
            price_bins = pd.cut(df["close"], bins=bins)
            volume_profile = df.groupby(price_bins)["volume"].sum()

            # Find Point of Control (POC)
            poc_bin = volume_profile.idxmax()
            poc_price = poc_bin.mid if hasattr(poc_bin, "mid") else poc_bin

            # Find High Volume Nodes (HVNs) - top 20% of volume
            volume_threshold = volume_profile.quantile(0.8)
            hvn_bins = volume_profile[volume_profile >= volume_threshold]
            hvn_prices = [
                bin.mid if hasattr(bin, "mid") else bin for bin in hvn_bins.index
            ]

            # Find Low Volume Nodes (LVNs) - bottom 20% of volume
            lvn_threshold = volume_profile.quantile(0.2)
            lvn_bins = volume_profile[volume_profile <= lvn_threshold]
            lvn_prices = [
                bin.mid if hasattr(bin, "mid") else bin for bin in lvn_bins.index
            ]

            return {
                "poc": poc_price,
                "hvn_prices": hvn_prices,
                "lvn_prices": lvn_prices,
                "volume_profile": volume_profile.to_dict(),
            }
        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {e}")
            return {}

    @handle_data_processing_errors(
        default_return=np.nan, context="calculate_williams_r"
    )
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculates Williams %R momentum oscillator."""
        if len(df) < period:
            self.logger.warning(
                f"Insufficient data for Williams %R calculation. Need {period}, got {len(df)}"
            )
            return np.nan

        try:
            willr = df.ta.willr(length=period)
            return (
                willr.iloc[-1]
                if not willr.empty and not pd.isna(willr.iloc[-1])
                else np.nan
            )
        except Exception as e:
            self.logger.error(f"Error calculating Williams %R: {e}")
            return np.nan

    @handle_data_processing_errors(default_return=np.nan, context="calculate_cci")
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculates Commodity Channel Index (CCI)."""
        if len(df) < period:
            self.logger.warning(
                f"Insufficient data for CCI calculation. Need {period}, got {len(df)}"
            )
            return np.nan

        try:
            cci = df.ta.cci(length=period)
            return (
                cci.iloc[-1] if not cci.empty and not pd.isna(cci.iloc[-1]) else np.nan
            )
        except Exception as e:
            self.logger.error(f"Error calculating CCI: {e}")
            return np.nan

    @handle_data_processing_errors(default_return=np.nan, context="calculate_mfi")
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculates Money Flow Index (MFI) - Volume-weighted RSI."""
        if len(df) < period:
            self.logger.warning(
                f"Insufficient data for MFI calculation. Need {period}, got {len(df)}"
            )
            return np.nan

        try:
            mfi = df.ta.mfi(length=period)
            return (
                mfi.iloc[-1] if not mfi.empty and not pd.isna(mfi.iloc[-1]) else np.nan
            )
        except Exception as e:
            self.logger.error(f"Error calculating MFI: {e}")
            return np.nan

    @handle_data_processing_errors(default_return=np.nan, context="calculate_roc")
    def _calculate_roc(self, df: pd.DataFrame, period: int = 10) -> float:
        """Calculates Rate of Change (ROC) momentum indicator."""
        if len(df) < period:
            self.logger.warning(
                f"Insufficient data for ROC calculation. Need {period}, got {len(df)}"
            )
            return np.nan

        try:
            roc = df.ta.roc(length=period)
            return (
                roc.iloc[-1] if not roc.empty and not pd.isna(roc.iloc[-1]) else np.nan
            )
        except Exception as e:
            self.logger.error(f"Error calculating ROC: {e}")
            return np.nan

    @handle_data_processing_errors(default_return={}, context="calculate_parabolic_sar")
    def _calculate_parabolic_sar(self, df: pd.DataFrame) -> Dict:
        """Calculates Parabolic SAR trend following indicator."""
        if len(df) < 20:  # Need sufficient data for PSAR
            self.logger.warning(
                f"Insufficient data for Parabolic SAR calculation. Need 20, got {len(df)}"
            )
            return {}

        try:
            psar = df.ta.psar()
            if psar.empty:
                return {}

            current_psar = (
                psar["PSARl_0.02_0.2"].iloc[-1]
                if not pd.isna(psar["PSARl_0.02_0.2"].iloc[-1])
                else psar["PSARs_0.02_0.2"].iloc[-1]
            )
            current_price = df["close"].iloc[-1]

            # Determine trend direction
            trend = "BULLISH" if current_price > current_psar else "BEARISH"

            return {
                "psar_value": current_psar,
                "trend": trend,
                "distance_to_psar": abs(current_price - current_psar) / current_price
                if current_price != 0
                else 0,
            }
        except Exception as e:
            self.logger.error(f"Error calculating Parabolic SAR: {e}")
            return {}

    @handle_data_processing_errors(
        default_return={}, context="calculate_ichimoku_cloud"
    )
    def _calculate_ichimoku_cloud(self, df: pd.DataFrame) -> Dict:
        """Calculates Ichimoku Cloud components."""
        if len(df) < 52:  # Need sufficient data for Ichimoku
            self.logger.warning(
                f"Insufficient data for Ichimoku calculation. Need 52, got {len(df)}"
            )
            return {}

        try:
            ichimoku = df.ta.ichimoku()
            if ichimoku.empty:
                return {}

            current_price = df["close"].iloc[-1]

            # Get latest values
            tenkan_sen = (
                ichimoku["ITS_9"].iloc[-1] if "ITS_9" in ichimoku.columns else np.nan
            )
            kijun_sen = (
                ichimoku["IKS_26"].iloc[-1] if "IKS_26" in ichimoku.columns else np.nan
            )
            senkou_span_a = (
                ichimoku["ISA_9"].iloc[-1] if "ISA_9" in ichimoku.columns else np.nan
            )
            senkou_span_b = (
                ichimoku["ISB_26"].iloc[-1] if "ISB_26" in ichimoku.columns else np.nan
            )

            # Determine cloud position
            cloud_top = (
                max(senkou_span_a, senkou_span_b)
                if not (pd.isna(senkou_span_a) or pd.isna(senkou_span_b))
                else np.nan
            )
            cloud_bottom = (
                min(senkou_span_a, senkou_span_b)
                if not (pd.isna(senkou_span_a) or pd.isna(senkou_span_b))
                else np.nan
            )

            # Determine trend signals
            above_cloud = current_price > cloud_top if not pd.isna(cloud_top) else False
            below_cloud = (
                current_price < cloud_bottom if not pd.isna(cloud_bottom) else False
            )
            in_cloud = not above_cloud and not below_cloud

            tk_cross = None
            if not (pd.isna(tenkan_sen) or pd.isna(kijun_sen)):
                tk_cross = "BULLISH" if tenkan_sen > kijun_sen else "BEARISH"

            return {
                "tenkan_sen": tenkan_sen,
                "kijun_sen": kijun_sen,
                "senkou_span_a": senkou_span_a,
                "senkou_span_b": senkou_span_b,
                "cloud_top": cloud_top,
                "cloud_bottom": cloud_bottom,
                "above_cloud": above_cloud,
                "below_cloud": below_cloud,
                "in_cloud": in_cloud,
                "tk_cross": tk_cross,
            }
        except Exception as e:
            self.logger.error(f"Error calculating Ichimoku Cloud: {e}")
            return {}

    @handle_data_processing_errors(default_return={}, context="calculate_supertrend")
    def _calculate_supertrend(
        self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
    ) -> Dict:
        """Calculates SuperTrend indicator."""
        if len(df) < period:
            self.logger.warning(
                f"Insufficient data for SuperTrend calculation. Need {period}, got {len(df)}"
            )
            return {}

        try:
            # Calculate ATR
            atr = df.ta.atr(length=period)
            if atr.empty:
                return {}

            hl2 = (df["high"] + df["low"]) / 2

            # Calculate basic bands
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)

            # Initialize SuperTrend
            supertrend = pd.Series(index=df.index, dtype=float)
            trend = pd.Series(index=df.index, dtype=int)

            # First value
            supertrend.iloc[0] = lower_band.iloc[0]
            trend.iloc[0] = 1

            # Calculate SuperTrend values
            for i in range(1, len(df)):
                if df["close"].iloc[i] > supertrend.iloc[i - 1]:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    trend.iloc[i] = 1
                elif df["close"].iloc[i] < supertrend.iloc[i - 1]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    trend.iloc[i] = -1
                else:
                    supertrend.iloc[i] = supertrend.iloc[i - 1]
                    trend.iloc[i] = trend.iloc[i - 1]

            current_supertrend = supertrend.iloc[-1]
            current_trend = trend.iloc[-1]
            current_price = df["close"].iloc[-1]

            return {
                "supertrend_value": current_supertrend,
                "trend": "BULLISH" if current_trend == 1 else "BEARISH",
                "distance_to_supertrend": abs(current_price - current_supertrend)
                / current_price
                if current_price != 0
                else 0,
            }
        except Exception as e:
            self.logger.error(f"Error calculating SuperTrend: {e}")
            return {}

    @handle_data_processing_errors(
        default_return={}, context="calculate_donchian_channels"
    )
    def _calculate_donchian_channels(self, df: pd.DataFrame, period: int = 20) -> Dict:
        """Calculates Donchian Channels."""
        if len(df) < period:
            self.logger.warning(
                f"Insufficient data for Donchian Channels calculation. Need {period}, got {len(df)}"
            )
            return {}

        try:
            upper = df["high"].rolling(window=period).max()
            lower = df["low"].rolling(window=period).min()
            middle = (upper + lower) / 2

            current_price = df["close"].iloc[-1]
            upper_value = upper.iloc[-1] if not pd.isna(upper.iloc[-1]) else np.nan
            lower_value = lower.iloc[-1] if not pd.isna(lower.iloc[-1]) else np.nan
            middle_value = middle.iloc[-1] if not pd.isna(middle.iloc[-1]) else np.nan

            # Calculate channel position
            channel_position = None
            if not (pd.isna(upper_value) or pd.isna(lower_value)):
                channel_width = upper_value - lower_value
                if channel_width > 0:
                    channel_position = (current_price - lower_value) / channel_width

            return {
                "upper": upper_value,
                "middle": middle_value,
                "lower": lower_value,
                "channel_position": channel_position,
                "near_upper": channel_position > 0.8
                if channel_position is not None
                else False,
                "near_lower": channel_position < 0.2
                if channel_position is not None
                else False,
            }
        except Exception as e:
            self.logger.error(f"Error calculating Donchian Channels: {e}")
            return {}

    @handle_data_processing_errors(default_return={}, context="calculate_atr_bands")
    def _calculate_atr_bands(
        self, df: pd.DataFrame, period: int = 14, multiplier: float = 2.0
    ) -> Dict:
        """Calculates ATR-based volatility bands."""
        if len(df) < period:
            self.logger.warning(
                f"Insufficient data for ATR Bands calculation. Need {period}, got {len(df)}"
            )
            return {}

        try:
            atr = df.ta.atr(length=period)
            sma = df.ta.sma(length=period)

            if atr.empty or sma.empty:
                return {}

            upper_band = sma + (multiplier * atr)
            lower_band = sma - (multiplier * atr)

            current_price = df["close"].iloc[-1]
            upper_value = (
                upper_band.iloc[-1] if not pd.isna(upper_band.iloc[-1]) else np.nan
            )
            lower_value = (
                lower_band.iloc[-1] if not pd.isna(lower_band.iloc[-1]) else np.nan
            )
            middle_value = sma.iloc[-1] if not pd.isna(sma.iloc[-1]) else np.nan

            return {
                "upper": upper_value,
                "middle": middle_value,
                "lower": lower_value,
                "above_upper": current_price > upper_value
                if not pd.isna(upper_value)
                else False,
                "below_lower": current_price < lower_value
                if not pd.isna(lower_value)
                else False,
            }
        except Exception as e:
            self.logger.error(f"Error calculating ATR Bands: {e}")
            return {}

    @handle_data_processing_errors(default_return={}, context="calculate_vwap_bands")
    def _calculate_vwap_bands(
        self, df: pd.DataFrame, period: int = 20, multiplier: float = 2.0
    ) -> Dict:
        """Calculates VWAP-based bands."""
        if len(df) < period:
            self.logger.warning(
                f"Insufficient data for VWAP Bands calculation. Need {period}, got {len(df)}"
            )
            return {}

        try:
            # Calculate VWAP
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            vwap = (typical_price * df["volume"]).rolling(window=period).sum() / df[
                "volume"
            ].rolling(window=period).sum()

            # Calculate VWAP standard deviation
            vwap_std = typical_price.rolling(window=period).std()

            upper_band = vwap + (multiplier * vwap_std)
            lower_band = vwap - (multiplier * vwap_std)

            current_price = df["close"].iloc[-1]
            vwap_value = vwap.iloc[-1] if not pd.isna(vwap.iloc[-1]) else np.nan
            upper_value = (
                upper_band.iloc[-1] if not pd.isna(upper_band.iloc[-1]) else np.nan
            )
            lower_value = (
                lower_band.iloc[-1] if not pd.isna(lower_band.iloc[-1]) else np.nan
            )

            return {
                "vwap": vwap_value,
                "upper": upper_value,
                "lower": lower_value,
                "above_upper": current_price > upper_value
                if not pd.isna(upper_value)
                else False,
                "below_lower": current_price < lower_value
                if not pd.isna(lower_value)
                else False,
                "distance_from_vwap": (current_price - vwap_value) / vwap_value
                if not pd.isna(vwap_value) and vwap_value != 0
                else 0,
            }
        except Exception as e:
            self.logger.error(f"Error calculating VWAP Bands: {e}")
            return {}
