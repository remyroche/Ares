"""
De Prado 2026 Causal Protocol - Feature Super-Set (50 Features)
--------------------------------------------------------------
Implements the 50-feature super-set for ETH trading causal discovery.
Divided into Treatment (T), Nuisance (W), and Effect Modifiers (X).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import skew, kurtosis, norm
from src.utils.tprint import tprint_info, tprint_success, tprint_warning

class DePradoCausalFeatures:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.t_features = []
        self.w_features = []
        self.x_features = []

    def generate_all_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Main entry point to generate the 50-feature super-set.
        Returns the DataFrame and a dictionary mapping (T, W, X) to column names.
        """
        if self.verbose:
            tprint_info("🏗️ Generating De Prado 2026 Causal Super-Set (50 Features)...")

        result = pd.DataFrame(index=df.index)
        
        # Treatment (T) - Alpha Drivers (10)
        t_df = self._generate_treatment_features(df)
        self.t_features = t_df.columns.tolist()
        
        # Nuisance (W) - Regime Confounders (30)
        w_df = self._generate_nuisance_features(df)
        self.w_features = w_df.columns.tolist()
        
        # Effect Modifiers (X) - Heterogeneity (10)
        x_df = self._generate_effect_modifiers(df)
        self.x_features = x_df.columns.tolist()
        
        # Combine all
        all_features = pd.concat([t_df, w_df, x_df], axis=1)
        
        # Feature Mapping for ORF
        feature_map = {
            'T': self.t_features,
            'W': self.w_features,
            'X': self.x_features
        }
        
        if self.verbose:
            tprint_success(f"✅ Generated {len(all_features.columns)} causal features (T={len(self.t_features)}, W={len(self.w_features)}, X={len(self.x_features)})")
            
        return all_features, feature_map

    def _generate_treatment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Treatment (T) — The Alpha Drivers (10 Features)"""
        t = pd.DataFrame(index=df.index)
        close = df['close']
        volume = df['volume']
        high = df['high']
        low = df['low']
        
        # 1. Tick-Rule Imbalance: sum(V * sign(delta C))
        delta_c = close.diff()
        t['t_tick_imbalance_10'] = (volume * np.sign(delta_c)).rolling(10).sum()
        
        # 2. Kalman Velocity (1st derivative of Kalman state)
        from .mtf_feature_generation import KalmanFilter1D
        kf = KalmanFilter1D(Q=1e-5, R=0.01, initial_value=close.iloc[0])
        kf_state, _ = kf.filter_series(close)
        t['t_kalman_velocity'] = kf_state.diff()
        
        # 3. Shannon Entropy (Binary path predictability)
        def calc_shannon(x):
            if len(x) < 2: return 0
            p = (np.sign(np.diff(x)) > 0).mean()
            if p <= 0 or p >= 1: return 0
            return -(p * np.log2(p) + (1-p) * np.log2(1-p))
        t['t_shannon_entropy_24'] = close.rolling(24).apply(calc_shannon)
        
        # 4. Approximate Entropy (ApEn) - Micro-regularity proxy
        def quick_apen(x):
            if len(x) < 5: return 0
            # Simplified proxy: std of differences / range
            return np.std(np.diff(x)) / (np.max(x) - np.min(x) + 1e-9)
        t['t_approx_entropy'] = close.rolling(10).apply(quick_apen)
        
        # 5. VWAP Z-Score
        vwap = (close * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
        t['t_vwap_zscore'] = (close - vwap) / (close.rolling(20).std() + 1e-9)
        
        # 6. Hurst Exponent (Local)
        from .mtf_feature_generation import compute_hurst_proxy
        t['t_hurst_local'] = compute_hurst_proxy(close, window=50)
        
        # 7. Volume Force: log(delta C) * log(V)
        t['t_volume_force'] = np.log(delta_c.abs() + 1e-9) * np.log(volume + 1e-9)
        
        # 8. Range Aggression: (H-L)/V intensity
        t['t_range_aggression'] = (high - low) / (volume + 1e-9)
        
        # 9. Acceleration (2nd derivative of Kalman state)
        t['t_kalman_acceleration'] = t['t_kalman_velocity'].diff()
        
        # 10. Momentum Persistence: Z-score of price streaks
        def streak_z(x):
            streaks = np.sign(np.diff(x))
            if len(streaks) == 0: return 0
            return np.mean(streaks) / (np.std(streaks) + 1e-9)
        t['t_momentum_persistence'] = close.rolling(20).apply(streak_z)
        
        return t.fillna(0)

    def _generate_nuisance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nuisance (W) — The 'Regime' Confounders (30 Features)"""
        w = pd.DataFrame(index=df.index)
        close = df['close']
        open_p = df['open']
        high = df['high']
        low = df['low']
        volume = df['volume']
        returns = close.pct_change()
        
        # Volatility Cluster (5)
        from .mtf_feature_generation import (
            compute_yang_zhang_volatility, compute_garman_klass_volatility,
            compute_parkinson_volatility, compute_rogers_satchell_volatility
        )
        w['w_vol_yz'] = compute_yang_zhang_volatility(open_p, high, low, close, window=20)
        w['w_vol_gk'] = compute_garman_klass_volatility(open_p, high, low, close, window=20)
        w['w_vol_pk'] = compute_parkinson_volatility(high, low, window=20)
        w['w_vol_rs'] = compute_rogers_satchell_volatility(open_p, high, low, close, window=20)
        w['w_vol_ewma'] = returns.ewm(span=20).std()
        
        # Friction Cluster (3)
        # Corwin-Schultz Spread Proxy
        high_low_ratio = np.log(high / low)
        w['w_friction_spread'] = high_low_ratio.rolling(2).mean()
        # Amihud Illiquidity
        w['w_friction_amihud'] = returns.abs() / (volume * close + 1e-9)
        # High-Low Pct
        w['w_friction_hl_pct'] = (high - low) / (close + 1e-9)
        
        # Volume Cluster (4)
        w['w_vol_log_v'] = np.log(volume + 1e-9)
        w['w_vol_zscore'] = (volume - volume.rolling(50).mean()) / (volume.rolling(50).std() + 1e-9)
        w['w_vol_accel'] = volume.diff().diff()
        from .mtf_feature_generation import compute_cmf
        w['w_vol_cmf'] = compute_cmf(high, low, close, volume, period=20)
        
        # Geometry Cluster (3)
        # Efficiency Ratio (Lagged to prevent collider bias)
        from .mtf_feature_generation import get_efficiency_ratio
        w['w_geom_er'] = get_efficiency_ratio(close, 14).shift(1)
        # Fractal Dimension Proxy
        w['w_geom_fractal_dim'] = (np.log(high.rolling(20).max() - low.rolling(20).min() + 1e-9) / np.log(20))
        # Fractal Chaos Proxy
        w['w_geom_chaos'] = returns.rolling(20).std() / (returns.abs().rolling(20).mean() + 1e-9)
        
        # Sessionality (5)
        if isinstance(df.index, pd.DatetimeIndex):
            w['w_sess_hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            w['w_sess_hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
            w['w_sess_day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            w['w_sess_day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
            # Session Overlap (dummy for London/NY - rough UTC estimate)
            w['w_sess_overlap'] = ((df.index.hour >= 13) & (df.index.hour <= 16)).astype(int)
        else:
            for c in ['w_sess_hour_sin', 'w_sess_hour_cos', 'w_sess_day_sin', 'w_sess_day_cos', 'w_sess_overlap']:
                w[c] = 0
        
        # Distributional (3)
        w['w_dist_skew'] = returns.rolling(100).skew()
        w['w_dist_kurt'] = returns.rolling(100).kurt()
        w['w_dist_gappiness'] = (open_p - close.shift(1)) / (close.shift(1) + 1e-9)
        
        # Auto-Correlation (5) - Lagged Returns
        for i in range(1, 6):
            w[f'w_autocorr_ret_lag_{i}'] = returns.shift(i)
            
        return w.fillna(0)

    def _generate_effect_modifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Effect Modifiers (X) — The 'Leaves' (10 Features)"""
        x = pd.DataFrame(index=df.index)
        close = df['close']
        volume = df['volume']
        returns = close.pct_change()
        
        # 1. Relative Volatility: 15m Vol / 4h Vol
        vol_15m = returns.rolling(1).std() # placeholder for local
        vol_local = returns.rolling(8).std()
        vol_4h = returns.rolling(96).std()
        x['x_rel_vol'] = vol_local / (vol_4h + 1e-9)
        
        # 2. Relative Volume: 15m Vol / 24h Median
        x['x_rel_volume'] = volume / (volume.rolling(96).median() + 1e-9)
        
        # 3. ADX (14)
        from .mtf_feature_generation import compute_adx
        adx, _, _ = compute_adx(df['high'], df['low'], close, period=14)
        x['x_adx'] = adx
        
        # 4. RSI (8)
        from .mtf_feature_generation import compute_rsi
        x['x_rsi'] = compute_rsi(close, period=8)
        
        # 5. MFI
        from .mtf_feature_generation import compute_mfi
        x['x_mfi'] = compute_mfi(df['high'], df['low'], close, volume, window=14)
        
        # 6. HMM State (Hidden Markov regime label) - Proxy via simple 3-regime vol
        vol_std = returns.rolling(100).std()
        x['x_hmm_proxy'] = pd.qcut(vol_std.rank(method='first'), 3, labels=False, duplicates='drop')
        
        # 7. Time-Since-High: Bars since 24h extreme
        x['x_bars_since_high'] = df.index.to_series().diff().dt.total_seconds().fillna(0).cumsum() # placeholder
        # Actual implementation
        x['x_bars_since_24h_high'] = close.rolling(96).apply(lambda x: 96 - np.argmax(x))
        
        # 8. Time-Since-Low
        x['x_bars_since_24h_low'] = close.rolling(96).apply(lambda x: 96 - np.argmin(x))
        
        # 9. ATR Ratio: Local/Global ATR
        from .mtf_feature_generation import calculate_atr
        atr_local = calculate_atr(df['high'], df['low'], close, window=14)
        atr_global = calculate_atr(df['high'], df['low'], close, window=100)
        x['x_atr_ratio'] = atr_local / (atr_global + 1e-9)
        
        # 10. Efficiency Ratio (Lagged) - already in W, but here as X for heterogeneity
        from .mtf_feature_generation import get_efficiency_ratio
        x['x_geom_er_lagged'] = get_efficiency_ratio(close, 30).shift(1)
        
        return x.fillna(0)
