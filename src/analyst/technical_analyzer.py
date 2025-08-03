from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta

# Initialize pandas_ta properly - try multiple approaches
try:
    ta.extend_pandas()
except Exception:
    # Fallback: manually extend DataFrame
    pd.DataFrame.ta = ta

# Additional fallback: ensure pandas_ta is properly initialized
if not hasattr(pd.DataFrame, "ta"):
    pd.DataFrame.ta = ta

from src.utils.error_handler import (
    handle_data_processing_errors,
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class TechnicalAnalyzer:
    """
    Performs deep technical analysis on market data, calculating a wide range of indicators
    including MAs, VWAP, Volume Profile, and enhanced indicators from ENHANCED_INDICATORS_RECOMMENDATION.md.
    """

    def __init__(self, config: dict[Any, Any] | None = None):
        self.config = config or {}
        self.logger = system_logger.getChild("TechnicalAnalyzer")

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid data format"),
            AttributeError: (None, "Missing required data columns"),
            TypeError: (None, "Invalid data types"),
        },
        default_return=None,
        context="technical analysis",
    )
    def analyze(self, df: pd.DataFrame) -> dict:
        """Perform comprehensive technical analysis with enhanced error handling."""
        try:
            # Validate input data
            self._validate_input_data(df)

            # Perform technical analysis
            results = {}

            # Calculate RSI
            results["rsi"] = self._calculate_rsi(df)

            # Calculate MACD
            results["macd"] = self._calculate_macd(df)

            # Calculate Moving Averages
            results["moving_averages"] = self._calculate_moving_averages(df)

            # Calculate VWAP
            results["vwap"] = self._calculate_vwap(df)

            # Calculate Williams %R
            results["williams_r"] = self._calculate_williams_r(df)

            # Calculate CCI
            results["cci"] = self._calculate_cci(df)

            # Calculate MFI
            results["mfi"] = self._calculate_mfi(df)

            # Calculate ROC
            results["roc"] = self._calculate_roc(df)

            # Calculate Parabolic SAR
            results["parabolic_sar"] = self._calculate_parabolic_sar(df)

            # Calculate Ichimoku Cloud
            results["ichimoku_cloud"] = self._calculate_ichimoku_cloud(df)

            # Calculate SuperTrend
            results["supertrend"] = self._calculate_supertrend(df)

            # Calculate ATR Bands
            results["atr_bands"] = self._calculate_atr_bands(df)

            # Calculate Bollinger Bands
            results["bollinger_bands"] = self._calculate_bollinger_bands(df)

            # Calculate ATR
            results["atr"] = self._calculate_atr(df)

            # Calculate ADX
            results["adx"] = self._calculate_adx(df)

            # Calculate OBV
            results["obv"] = self._calculate_obv(df)

            # Calculate ATR Ratio
            results["atr_ratio"] = self._calculate_atr_ratio(df)

            return results

        except Exception as e:
            self.logger.error(f"Error in technical analysis: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, TypeError, KeyError),
        default_return={},
        context="analyze",
    )
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        print(
            f"[DEBUG] _calculate_rsi: shape={df.shape}, columns={df.columns.tolist()}",
        )
        print(df.head())
        """Calculates the Relative Strength Index (RSI)."""
        if len(df) < period:
            self.logger.warning(
                f"Insufficient data for RSI calculation. Need {period}, got {len(df)}",
            )
            return np.nan

        # Try pandas-ta directly
        try:
            import pandas_ta as ta

            rsi = ta.rsi(df["close"], length=period)
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return np.nan
        return rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else np.nan

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="MACD calculation",
    )
    def _calculate_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> dict:
        print(
            f"[DEBUG] _calculate_macd: shape={df.shape}, columns={df.columns.tolist()}",
        )
        print(df.head())
        """Calculates the Moving Average Convergence Divergence (MACD)."""
        if len(df) < slow:
            self.logger.warning(
                f"Insufficient data for MACD calculation. Need {slow}, got {len(df)}",
            )
            return {}

        # Try pandas-ta directly
        try:
            import pandas_ta as ta

            macd_df = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return {}
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

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="moving averages calculation",
    )
    def _calculate_moving_averages(
        self,
        df: pd.DataFrame,
        periods: list[int] = [9, 21, 50, 200],
    ) -> dict:
        print(
            f"[DEBUG] _calculate_moving_averages: shape={df.shape}, columns={df.columns.tolist()}",
        )
        print(df.head())
        """Calculates multiple Simple Moving Averages (SMAs)."""
        mas = {}
        for period in periods:
            if len(df) < period:
                self.logger.warning(
                    f"Insufficient data for SMA {period}. Need {period}, got {len(df)}",
                )
                continue

            # Try pandas-ta directly
            try:
                import pandas_ta as ta

                sma = ta.sma(df["close"], length=period)
            except Exception as e:
                self.logger.error(f"Error calculating SMA {period}: {e}")
                continue

            if not sma.empty and not pd.isna(sma.iloc[-1]):
                mas[f"sma_{period}"] = sma.iloc[-1]
        return mas

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="VWAP calculation",
    )
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculates the Volume-Weighted Average Price (VWAP) for the given period."""
        if "volume" not in df.columns or df["volume"].sum() == 0:
            self.logger.warning("No volume data available for VWAP calculation")
            return None

        try:
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            vwap = (typical_price * df["volume"]).sum() / df["volume"].sum()
            return vwap if not np.isnan(vwap) else np.nan
        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}")
            return None

    @handle_data_processing_errors(
        default_return={},
        context="calculate_volume_profile",
    )
    def _calculate_volume_profile(self, df: pd.DataFrame, bins: int = 20) -> dict:
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

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="Williams %R calculation",
    )
    def _calculate_williams_r(self, df: pd.DataFrame) -> pd.Series:
        print(
            f"[DEBUG] _calculate_williams_r: shape={df.shape}, columns={df.columns.tolist()}",
        )
        print(df.head())
        """Calculates Williams %R momentum oscillator."""
        if len(df) < 14:
            self.logger.warning(
                f"Insufficient data for Williams %R calculation. Need 14, got {len(df)}",
            )
            return None

        try:
            # Use direct pandas-ta function with correct parameters
            willr = ta.willr(df["high"], df["low"], df["close"], length=14)
            return (
                willr.iloc[-1]
                if not willr.empty and not pd.isna(willr.iloc[-1])
                else np.nan
            )
        except Exception as e:
            self.logger.error(f"Error calculating Williams %R: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="CCI calculation",
    )
    def _calculate_cci(self, df: pd.DataFrame) -> pd.Series:
        print(
            f"[DEBUG] _calculate_cci: shape={df.shape}, columns={df.columns.tolist()}",
        )
        print(df.head())
        """Calculates Commodity Channel Index (CCI)."""
        if len(df) < 20:
            self.logger.warning(
                f"Insufficient data for CCI calculation. Need 20, got {len(df)}",
            )
            return None

        try:
            # Use direct pandas-ta function with correct parameters
            cci = ta.cci(df["high"], df["low"], df["close"], length=20)
            return (
                cci.iloc[-1] if not cci.empty and not pd.isna(cci.iloc[-1]) else np.nan
            )
        except Exception as e:
            self.logger.error(f"Error calculating CCI: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="MFI calculation",
    )
    def _calculate_mfi(self, df: pd.DataFrame) -> pd.Series:
        print(
            f"[DEBUG] _calculate_mfi: shape={df.shape}, columns={df.columns.tolist()}",
        )
        print(df.head())
        """Calculates Money Flow Index (MFI) - Volume-weighted RSI."""
        if len(df) < 14:
            self.logger.warning(
                f"Insufficient data for MFI calculation. Need 14, got {len(df)}",
            )
            return None

        # Debug logging for DataFrame structure
        print(
            f"[DEBUG] Calculating MFI: shape={df.shape}, columns={df.columns.tolist()}",
        )
        print(df.head())
        try:
            # Use direct pandas-ta function with correct parameters
            mfi = ta.mfi(
                df["high"],
                df["low"],
                df["close"],
                df["volume"],
                length=14,
            )
            return (
                mfi.iloc[-1] if not mfi.empty and not pd.isna(mfi.iloc[-1]) else np.nan
            )
        except Exception as e:
            self.logger.error(f"Error calculating MFI: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ROC calculation",
    )
    def _calculate_roc(self, df: pd.DataFrame) -> pd.Series:
        print(
            f"[DEBUG] _calculate_roc: shape={df.shape}, columns={df.columns.tolist()}",
        )
        print(df.head())
        """Calculates Rate of Change (ROC) momentum indicator."""
        if len(df) < 10:
            self.logger.warning(
                f"Insufficient data for ROC calculation. Need 10, got {len(df)}",
            )
            return None

        try:
            # Use direct pandas-ta function with correct parameters
            roc = ta.roc(df["close"], length=10)
            return (
                roc.iloc[-1] if not roc.empty and not pd.isna(roc.iloc[-1]) else np.nan
            )
        except Exception as e:
            self.logger.error(f"Error calculating ROC: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="Parabolic SAR calculation",
    )
    def _calculate_parabolic_sar(self, df: pd.DataFrame) -> pd.Series:
        print(
            f"[DEBUG] _calculate_parabolic_sar: shape={df.shape}, columns={df.columns.tolist()}",
        )
        print(df.head())
        """Calculates Parabolic SAR trend following indicator."""
        if len(df) < 20:  # Need sufficient data for PSAR
            self.logger.warning(
                f"Insufficient data for Parabolic SAR calculation. Need 20, got {len(df)}",
            )
            return None

        try:
            # Use direct pandas-ta function with correct parameters
            psar = ta.psar(df["high"], df["low"], df["close"])
            if psar.empty:
                return None

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
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="Ichimoku Cloud calculation",
    )
    def _calculate_ichimoku_cloud(self, df: pd.DataFrame) -> dict:
        print(
            f"[DEBUG] _calculate_ichimoku_cloud: shape={df.shape}, columns={df.columns.tolist()}",
        )
        print(df.head())
        """Calculates Ichimoku Cloud components."""
        if len(df) < 52:  # Need sufficient data for Ichimoku
            self.logger.warning(
                f"Insufficient data for Ichimoku calculation. Need 52, got {len(df)}",
            )
            return {}

        try:
            # Use direct pandas-ta function with correct parameters
            ichimoku_result = ta.ichimoku(df["high"], df["low"], df["close"])
            
            # Handle the case where ichimoku returns a tuple
            if isinstance(ichimoku_result, tuple):
                ichimoku = ichimoku_result[0]  # First element is the DataFrame
            else:
                ichimoku = ichimoku_result
                
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
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="SuperTrend calculation",
    )
    def _calculate_supertrend(
        self,
        df: pd.DataFrame,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> pd.Series:
        print(
            f"[DEBUG] _calculate_supertrend: shape={df.shape}, columns={df.columns.tolist()}",
        )
        print(df.head())
        """Calculates SuperTrend indicator."""
        if len(df) < period:
            self.logger.warning(
                f"Insufficient data for SuperTrend calculation. Need {period}, got {len(df)}",
            )
            return None

        try:
            # Calculate ATR using direct pandas-ta function
            atr = ta.atr(df["high"], df["low"], df["close"], length=period)
            if atr.empty:
                return None

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
            return None

    @handle_data_processing_errors(
        default_return={},
        context="calculate_donchian_channels",
    )
    def _calculate_donchian_channels(self, df: pd.DataFrame, period: int = 20) -> dict:
        """Calculates Donchian Channels."""
        if len(df) < period:
            self.logger.warning(
                f"Insufficient data for Donchian Channels calculation. Need {period}, got {len(df)}",
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

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ATR Bands calculation",
    )
    def _calculate_atr_bands(
        self,
        df: pd.DataFrame,
        period: int = 14,
        multiplier: float = 2.0,
    ) -> dict:
        print(
            f"[DEBUG] _calculate_atr_bands: shape={df.shape}, columns={df.columns.tolist()}",
        )
        print(df.head())
        """Calculates ATR-based volatility bands."""
        if len(df) < period:
            self.logger.warning(
                f"Insufficient data for ATR Bands calculation. Need {period}, got {len(df)}",
            )
            return None

        try:
            # Use direct pandas-ta functions with correct parameters
            atr = ta.atr(df["high"], df["low"], df["close"], length=period)
            sma = ta.sma(df["close"], length=period)

            if atr.empty or sma.empty:
                return None

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
            return None

    @handle_data_processing_errors(default_return={}, context="calculate_vwap_bands")
    def _calculate_vwap_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        multiplier: float = 2.0,
    ) -> dict:
        """Calculates VWAP-based bands."""
        if len(df) < period:
            self.logger.warning(
                f"Insufficient data for VWAP Bands calculation. Need {period}, got {len(df)}",
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

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="Bollinger Bands calculation",
    )
    def _calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> dict:
        print(
            f"[DEBUG] _calculate_bollinger_bands: shape={df.shape}, columns={df.columns.tolist()}",
        )
        print(df.head())
        """Calculates Bollinger Bands (BBU_20_2.0, BBL_20_2.0)."""
        if len(df) < period:
            self.logger.warning(
                f"Insufficient data for Bollinger Bands calculation. Need {period}, got {len(df)}",
            )
            return None

        try:
            # Try pandas-ta directly if .ta attribute is not available
            try:
                bb = df.ta.bbands(length=period, std=std_dev)
            except AttributeError:
                # Fallback to direct pandas-ta function
                bb = ta.bbands(df["close"], length=period, std=std_dev)
            except Exception:
                # Manual calculation as final fallback
                sma = df["close"].rolling(window=period).mean()
                std = df["close"].rolling(window=period).std()
                bb_upper = sma + (std_dev * std)
                bb_lower = sma - (std_dev * std)
                bb_middle = sma

                bb = pd.DataFrame(
                    {
                        f"BBU_{period}_{std_dev}": bb_upper,
                        f"BBL_{period}_{std_dev}": bb_lower,
                        f"BBM_{period}_{std_dev}": bb_middle,
                    },
                )

            if bb.empty:
                return None

            # Get the latest values
            bb_upper = (
                bb[f"BBU_{period}_{std_dev}"].iloc[-1]
                if f"BBU_{period}_{std_dev}" in bb.columns
                else np.nan
            )
            bb_lower = (
                bb[f"BBL_{period}_{std_dev}"].iloc[-1]
                if f"BBL_{period}_{std_dev}" in bb.columns
                else np.nan
            )
            bb_middle = (
                bb[f"BBM_{period}_{std_dev}"].iloc[-1]
                if f"BBM_{period}_{std_dev}" in bb.columns
                else np.nan
            )

            current_price = df["close"].iloc[-1]

            return {
                "BBU_20_2.0": bb_upper,
                "BBL_20_2.0": bb_lower,
                "BBM_20_2.0": bb_middle,
                "bb_position": (current_price - bb_lower) / (bb_upper - bb_lower)
                if not (pd.isna(bb_upper) or pd.isna(bb_lower))
                and (bb_upper - bb_lower) > 0
                else np.nan,
                "above_upper": current_price > bb_upper
                if not pd.isna(bb_upper)
                else False,
                "below_lower": current_price < bb_lower
                if not pd.isna(bb_lower)
                else False,
            }
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ATR calculation",
    )
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        print(
            f"[DEBUG] _calculate_atr: shape={df.shape}, columns={df.columns.tolist()}",
        )
        print(df.head())
        """Calculates Average True Range (ATR)."""
        if len(df) < period:
            self.logger.warning(
                f"Insufficient data for ATR calculation. Need {period}, got {len(df)}",
            )
            return np.nan

        try:
            # Try pandas-ta directly if .ta attribute is not available
            try:
                atr = df.ta.atr(length=period)
            except AttributeError:
                # Fallback to direct pandas-ta function
                atr = ta.atr(df["high"], df["low"], df["close"], length=period)
            except Exception:
                # Manual ATR calculation as final fallback
                high_low = df["high"] - df["low"]
                high_close = np.abs(df["high"] - df["close"].shift())
                low_close = np.abs(df["low"] - df["close"].shift())

                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(
                    axis=1,
                )
                atr = true_range.rolling(window=period).mean()

            return (
                atr.iloc[-1] if not atr.empty and not pd.isna(atr.iloc[-1]) else np.nan
            )
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return np.nan

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ADX calculation",
    )
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        print(
            f"[DEBUG] _calculate_adx: shape={df.shape}, columns={df.columns.tolist()}",
        )
        print(df.head())
        """Calculates Average Directional Index (ADX)."""
        if len(df) < period:
            self.logger.warning(
                f"Insufficient data for ADX calculation. Need {period}, got {len(df)}",
            )
            return np.nan

        try:
            # Use direct pandas-ta function with correct parameters
            adx = ta.adx(df["high"], df["low"], df["close"], length=period)
        except Exception:
            # Manual ADX calculation as fallback if pandas-ta fails
            # Simplified ADX calculation
            high_diff = df["high"].diff()
            low_diff = df["low"].diff()

            plus_dm = np.where(
                (high_diff > low_diff) & (high_diff > 0),
                high_diff,
                0,
            )
            minus_dm = np.where(
                (low_diff > high_diff) & (low_diff > 0),
                -low_diff,
                0,
            )

            tr = pd.concat(
                [
                    df["high"] - df["low"],
                    np.abs(df["high"] - df["close"].shift()),
                    np.abs(df["low"] - df["close"].shift()),
                ],
                axis=1,
            ).max(axis=1)

            # Smooth the values
            plus_di = (
                100
                * pd.Series(plus_dm).rolling(window=period).mean()
                / tr.rolling(window=period).mean()
            )
            minus_di = (
                100
                * pd.Series(minus_dm).rolling(window=period).mean()
                / tr.rolling(window=period).mean()
            )

            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            dx = pd.Series(dx).fillna(0)  # Fill NaN values
            adx = dx.rolling(window=period).mean()

            # ADX is typically the first column in the result
            adx_column = None
            if hasattr(adx, "columns"):
                for col in adx.columns:
                    if "ADX" in col:
                        adx_column = col
                        break

                if adx_column is None:
                    # If no ADX column found, use the first column
                    adx_column = adx.columns[0]

                return (
                    adx[adx_column].iloc[-1]
                    if not adx.empty and not pd.isna(adx[adx_column].iloc[-1])
                    else np.nan
                )
            # adx is a Series
            return (
                adx.iloc[-1] if not adx.empty and not pd.isna(adx.iloc[-1]) else np.nan
            )
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {e}")
            return np.nan

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="OBV calculation",
    )
    def _calculate_obv(self, df: pd.DataFrame) -> float:
        print(
            f"[DEBUG] _calculate_obv: shape={df.shape}, columns={df.columns.tolist()}",
        )
        print(df.head())
        """Calculates On-Balance Volume (OBV)."""
        if len(df) < 2:
            self.logger.warning(
                f"Insufficient data for OBV calculation. Need 2, got {len(df)}",
            )
            return np.nan

        try:
            # Use direct pandas-ta function with correct parameters
            obv = ta.obv(df["close"], df["volume"])
        except Exception:
            # Manual OBV calculation as fallback if pandas-ta fails
            obv = pd.Series(index=df.index, dtype=float)
            obv.iloc[0] = df["volume"].iloc[0]

            for i in range(1, len(df)):
                if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                    obv.iloc[i] = obv.iloc[i - 1] + df["volume"].iloc[i]
                elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                    obv.iloc[i] = obv.iloc[i - 1] - df["volume"].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i - 1]

        except Exception as e:
            self.logger.error(f"Error calculating OBV: {e}")
            return np.nan

        return (
            obv.iloc[-1] if not obv.empty and not pd.isna(obv.iloc[-1]) else np.nan
        )

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ATR ratio calculation",
    )
    def _calculate_atr_ratio(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculates ATR ratio (ATRr_14) - ATR as a percentage of price."""
        atr_value = self._calculate_atr(df, period)
        if pd.isna(atr_value):
            return np.nan

        current_price = df["close"].iloc[-1]
        if current_price == 0:
            return np.nan

        return (atr_value / current_price) * 100  # Convert to percentage

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="input data validation",
    )
    def _validate_input_data(self, df: pd.DataFrame) -> None:
        """Validate input data for technical analysis."""
        if df is None or df.empty:
            raise ValueError("Input DataFrame is None or empty")

        required_columns = ["open", "high", "low", "close"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
