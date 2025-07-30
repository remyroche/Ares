import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any
from src.utils.logger import system_logger
from src.utils.error_handler import (
    handle_errors,
    handle_data_processing_errors,
    handle_type_conversions,
    error_context,
    ErrorRecoveryStrategies
)

class TechnicalAnalyzer:
    """
    Performs deep technical analysis on market data, calculating a wide range of indicators
    including MAs, VWAP, and Volume Profile.
    """

    def __init__(self, config: Optional[Dict[Any, Any]] = None):
        self.config = config or {}
        self.logger = system_logger.getChild('TechnicalAnalyzer')

    @handle_errors(
        exceptions=(ValueError, TypeError, KeyError),
        default_return={},
        context="analyze"
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
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Check for NaN values after conversion
        if df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
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

        current_price = df['close'].iloc[-1]

        return {
            "current_price": current_price,
            "rsi": rsi,
            "macd": macd,
            "moving_averages": mas,
            "vwap": vwap,
            "price_to_vwap_ratio": (current_price / vwap) if vwap and vwap != 0 else 0,
            "volume_profile": volume_profile
        }

    @handle_data_processing_errors(
        default_return=np.nan,
        context="calculate_rsi"
    )
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculates the Relative Strength Index (RSI)."""
        if len(df) < period:
            self.logger.warning(f"Insufficient data for RSI calculation. Need {period}, got {len(df)}")
            return np.nan
        
        rsi = df.ta.rsi(length=period)
        return rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else np.nan

    @handle_data_processing_errors(
        default_return={},
        context="calculate_macd"
    )
    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculates the Moving Average Convergence Divergence (MACD)."""
        if len(df) < slow:
            self.logger.warning(f"Insufficient data for MACD calculation. Need {slow}, got {len(df)}")
            return {}
        
        macd_df = df.ta.macd(fast=fast, slow=slow, signal=signal)
        if macd_df.empty:
            return {}
        
        try:
            return {
                "macd": macd_df[f'MACD_{fast}_{slow}_{signal}'].iloc[-1],
                "histogram": macd_df[f'MACDh_{fast}_{slow}_{signal}'].iloc[-1],
                "signal": macd_df[f'MACDs_{fast}_{slow}_{signal}'].iloc[-1]
            }
        except KeyError as e:
            self.logger.error(f"MACD column not found: {e}")
            return {}

    @handle_data_processing_errors(
        default_return={},
        context="calculate_moving_averages"
    )
    def _calculate_moving_averages(self, df: pd.DataFrame, periods: List[int] = [9, 21, 50, 200]) -> Dict:
        """Calculates multiple Simple Moving Averages (SMAs)."""
        mas = {}
        for period in periods:
            if len(df) < period:
                self.logger.warning(f"Insufficient data for SMA {period}. Need {period}, got {len(df)}")
                continue
            
            sma = df.ta.sma(length=period)
            if not sma.empty and not pd.isna(sma.iloc[-1]):
                mas[f'sma_{period}'] = sma.iloc[-1]
        return mas

    @handle_data_processing_errors(
        default_return=np.nan,
        context="calculate_vwap"
    )
    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculates the Volume-Weighted Average Price (VWAP) for the given period."""
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            self.logger.warning("No volume data available for VWAP calculation")
            return np.nan
        
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).sum() / df['volume'].sum()
            return vwap if not np.isnan(vwap) else np.nan
        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}")
            return np.nan

    @handle_data_processing_errors(
        default_return={},
        context="calculate_volume_profile"
    )
    def _calculate_volume_profile(self, df: pd.DataFrame, bins: int = 20) -> Dict:
        """
        Calculates a simple Volume Profile.

        Returns:
            Dict: Containing POC, HVNs, and LVNs.
        """
        try:
            price_range = df['high'].max() - df['low'].min()
            if price_range == 0:
                self.logger.warning("No price range for volume profile calculation")
                return {}
            
            bin_size = price_range / bins
            
            # Create price bins
            price_bins = pd.cut(df['close'], bins=bins)
            volume_profile = df.groupby(price_bins)['volume'].sum()
            
            # Find Point of Control (POC)
            poc_bin = volume_profile.idxmax()
            poc_price = poc_bin.mid if hasattr(poc_bin, 'mid') else poc_bin
            
            # Find High Volume Nodes (HVNs) - top 20% of volume
            volume_threshold = volume_profile.quantile(0.8)
            hvn_bins = volume_profile[volume_profile >= volume_threshold]
            hvn_prices = [bin.mid if hasattr(bin, 'mid') else bin for bin in hvn_bins.index]
            
            # Find Low Volume Nodes (LVNs) - bottom 20% of volume
            lvn_threshold = volume_profile.quantile(0.2)
            lvn_bins = volume_profile[volume_profile <= lvn_threshold]
            lvn_prices = [bin.mid if hasattr(bin, 'mid') else bin for bin in lvn_bins.index]
            
            return {
                "poc": poc_price,
                "hvn_prices": hvn_prices,
                "lvn_prices": lvn_prices,
                "volume_profile": volume_profile.to_dict()
            }
        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {e}")
            return {}
