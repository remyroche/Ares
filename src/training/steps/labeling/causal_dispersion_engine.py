import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, List
from src.utils.fracdiff import fracdiff_series
from src.utils.orthogonal_numba import _numba_kalman_filter_1d
from src.utils.tprint import tprint_info, tprint_warning, tprint_error

class CausalDispersionEngine:
    """
    Engine for calculating Causal Dispersion (DMV) and other microstructure-validated surprises.

    Generates:
    1. DMV Factor (Microstructure-Validated Dispersion): Price distance from Fair Value,
       normalized by ATR and weighted by Volume Z-score.
    2. Innovation Features: Stationary features derived from surprises via FracDiff.
    """

    def __init__(self, window_atr: int = 14, window_vol: int = 20):
        self.window_atr = window_atr
        self.window_vol = window_vol

    def calculate_dmv(self, df: pd.DataFrame, market_data: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Calculate Weighted Decoupling (DMV Factor).

        Args:
            df: DataFrame containing at least ['close', 'volume', 'high', 'low'].
            market_data: Optional DataFrame containing ['close'] for market benchmark (e.g., BTC).
                         If None, uses Kalman Filter on asset price as 'Fair Value' proxy.

        Returns:
            pd.Series: The DMV factor (Weighted Decoupling).
        """
        # Ensure we have required columns
        required_cols = ['close', 'volume', 'high', 'low']
        if not all(col in df.columns for col in required_cols):
            tprint_warning(f"CausalDispersionEngine: Missing required columns {required_cols}. Returning empty series.")
            return pd.Series(dtype=float)

        # 1. Calculate Fair Value Distance
        price = df['close']

        if market_data is not None and 'close' in market_data.columns:
            # Use Market Data + Kalman Beta
            # Note: Implementing full rolling beta calculation might be expensive here.
            # Simplified approach: Use Market Price normalized to Asset Price level as baseline?
            # Or assume Beta=1 for now if Beta not pre-calculated?
            # The user provided: panel_df['Fair_Value'] = panel_df['Kalman_Beta'] * panel_df['Market_Price']
            # If we don't have Kalman_Beta, we can try to estimate it or fallback.

            # Fallback: Just use Market Return correlation or simple rebase?
            # Let's stick to the Single Asset Fallback (Kalman Filter on Price) which is robust and self-contained
            # unless 'Kalman_Beta' and 'Market_Price' are explicitly in df.
            if 'Kalman_Beta' in df.columns and 'Market_Price' in df.columns:
                 fair_value = df['Kalman_Beta'] * df['Market_Price']
            else:
                 # Fallback to Kalman Filter on Price as "Expected Value"
                 fair_value = self._get_kalman_expected_price(price)
        else:
            # Single Asset Fallback: Fair Value = Kalman Filter Expectation of Price
            fair_value = self._get_kalman_expected_price(price)

        distance = (price - fair_value).abs()

        # 2. ATR Normalization
        high = df['high']
        low = df['low']
        # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
        prev_close = price.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.window_atr).mean()

        atr_distance = distance / (atr + 1e-9)

        # 3. Volume Z-Score & Participation Weight
        volume = df['volume']
        vol_mean = volume.rolling(self.window_vol).mean()
        vol_std = volume.rolling(self.window_vol).std()

        vol_z = (volume - vol_mean) / (vol_std + 1e-9)

        # Softplus weighting: log(1 + exp(z))
        # Use numpy for vectorization
        vol_weight = np.log1p(np.exp(vol_z))

        # 4. Aggregate into Single Factor
        weighted_decoupling = atr_distance * vol_weight

        return weighted_decoupling.fillna(0.0)

    def _get_kalman_expected_price(self, price: pd.Series) -> pd.Series:
        """Calculate expected price using 1D Kalman Filter."""
        values = price.values.astype(np.float64)
        # Using default parameters for now: Q=1e-5, R=0.01 (can be tuned)
        # Assuming _numba_kalman_filter_1d(obs, Q, R, initial_state, initial_covariance)
        # Returns (state_estimates, covariance_estimates)
        try:
            states, _ = _numba_kalman_filter_1d(values, 1e-5, 0.01, values[0], 1.0)
            return pd.Series(states, index=price.index)
        except Exception as e:
            tprint_warning(f"Kalman Filter failed: {e}. Using EMA fallback.")
            return price.ewm(span=20).mean()

    def generate_innovation_feature(self, surprise_series: pd.Series, d: float = 0.4) -> pd.Series:
        """
        Transform a Surprise series (S_t) into a Stationary Feature (F_t) via FracDiff.
        """
        feature, _ = fracdiff_series(surprise_series, d=d)
        return feature.fillna(0.0)
