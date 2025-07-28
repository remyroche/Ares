import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List

class TechnicalAnalyzer:
    """
    Performs deep technical analysis on market data, calculating a wide range of indicators
    including MAs, VWAP, and Volume Profile.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Runs all technical analysis calculations and returns a consolidated dictionary.

        Args:
            df (pd.DataFrame): DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        Returns:
            Dict: A dictionary containing all calculated technical indicators.
        """
        if df.empty:
            return {}

        # Ensure correct data types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])

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
            "price_to_vwap_ratio": (current_price / vwap) if vwap else 0,
            "volume_profile": volume_profile
        }

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculates the Relative Strength Index (RSI)."""
        rsi = df.ta.rsi(length=period)
        return rsi.iloc[-1] if not rsi.empty else np.nan

    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculates the Moving Average Convergence Divergence (MACD)."""
        macd_df = df.ta.macd(fast=fast, slow=slow, signal=signal)
        if macd_df.empty:
            return {}
        return {
            "macd": macd_df[f'MACD_{fast}_{slow}_{signal}'].iloc[-1],
            "histogram": macd_df[f'MACDh_{fast}_{slow}_{signal}'].iloc[-1],
            "signal": macd_df[f'MACDs_{fast}_{slow}_{signal}'].iloc[-1]
        }

    def _calculate_moving_averages(self, df: pd.DataFrame, periods: List[int] = [9, 21, 50, 200]) -> Dict:
        """Calculates multiple Simple Moving Averages (SMAs)."""
        mas = {}
        for period in periods:
            sma = df.ta.sma(length=period)
            if not sma.empty:
                mas[f'sma_{period}'] = sma.iloc[-1]
        return mas

    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculates the Volume-Weighted Average Price (VWAP) for the given period."""
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            return np.nan
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).sum() / df['volume'].sum()
        return vwap

    def _calculate_volume_profile(self, df: pd.DataFrame, bins: int = 20) -> Dict:
        """
        Calculates a simple Volume Profile.

        Returns:
            Dict: Containing POC, HVNs, and LVNs.
        """
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / bins
        
        if price_range == 0 or bin_size == 0:
             return {"poc": df['close'].iloc[-1], "hvn": [], "lvn": []}


        # Assign each trade's price to a bin
        binned_prices = ((df['close'] - df['low'].min()) / bin_size).astype(int)
        
        # Group by bin and sum volume
        volume_by_bin = df['volume'].groupby(binned_prices).sum()
        
        # Point of Control (POC)
        poc_bin = volume_by_bin.idxmax()
        poc_price = df['low'].min() + poc_bin * bin_size + (bin_size / 2)

        # High/Low Volume Nodes (simple version)
        mean_volume = volume_by_bin.mean()
        std_volume = volume_by_bin.std()
        
        hvn_prices = []
        lvn_prices = []

        for bin_idx, volume in volume_by_bin.items():
            price = df['low'].min() + bin_idx * bin_size + (bin_size / 2)
            if volume > mean_volume + std_volume:
                hvn_prices.append(price)
            elif volume < mean_volume - (0.5 * std_volume): # More sensitive for LVNs
                lvn_prices.append(price)

        return {
            "poc": poc_price,
            "hvns": sorted(hvn_prices),
            "lvns": sorted(lvn_prices)
        }
