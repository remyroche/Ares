"""
Layer 3 Specific Feature Generation.

This module provides features specifically designed for the Layer 3 stacking model,
incorporating:
1. Ensemble-based features (Probability, Logit, Momentum)
2. Volume and Shape features
3. Regime and Complexity features (derived from GateModel logic)
"""

import numpy as np
import pandas as pd
import math
import itertools
from scipy.stats import entropy
from typing import List

def _compute_gate_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute regime features based on GateModel logic.

    Includes:
    - Volatility features (rv_short, rv_z_short, etc.)
    - Trend/Momentum features (slope, adx_proxy, snr)
    - Complexity features (choppiness, variance_ratio, entropy)
    - Event features (vol spikes, large candles)
    - Time features (hour_sin/cos, is_weekend)
    """
    features = pd.DataFrame(index=df.index)

    close = df['close']
    high = df['high']
    low = df['low']

    # Log returns
    log_ret = np.log(close / close.shift(1))

    # 1. Volatility Features
    features['rv_short'] = log_ret.rolling(window=12).std() * np.sqrt(12)
    rv_med = log_ret.rolling(window=48).std() * np.sqrt(48)
    features['rv_short_over_med'] = features['rv_short'] / (rv_med + 1e-8)

    tr = (high - low) / close
    atr_short = tr.rolling(window=12).mean()

    rv_long = log_ret.rolling(window=200).std()
    features['rv_z_short'] = (features['rv_short'] - rv_long) / (rv_long + 1e-8)

    # 2. Trend Strength Features
    log_price = np.log(close)
    features['slope_short'] = log_price.diff(12).abs()

    up_move = high.diff()
    down_move = low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smooth = tr.rolling(window=14).sum()
    plus_di = pd.Series(plus_dm).rolling(window=14).sum() / (tr_smooth + 1e-8)
    minus_di = pd.Series(minus_dm).rolling(window=14).sum() / (tr_smooth + 1e-8)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    features['adx_proxy'] = dx.rolling(window=14).mean()

    # 3. Momentum
    features['momentum_short'] = (close.diff(12) / close.shift(12)).abs()
    features['snr'] = features['momentum_short'].abs() / (features['rv_short'] + 1e-8)

    # 4. New Features (Vol Spike, Large Candle, Time)
    rv_mean = features['rv_short'].rolling(window=100, min_periods=20).mean()
    rv_std = features['rv_short'].rolling(window=100, min_periods=20).std()
    rv_z = (features['rv_short'] - rv_mean) / (rv_std + 1e-8)

    is_vol_spike = (rv_z > 2.0).astype(int)

    int_index = pd.Series(np.arange(len(df)), index=df.index)
    last_spike_int_idx = int_index.where(is_vol_spike == 1).ffill()
    features['time_since_last_vol_spike'] = int_index - last_spike_int_idx
    features['time_since_last_vol_spike'] = features['time_since_last_vol_spike'].fillna(1000)

    candle_range = high - low
    is_large_candle = (candle_range > 2.5 * atr_short).astype(int)
    large_candle_int_idx = int_index.where(is_large_candle == 1).ffill()
    features['time_since_last_large_candle'] = int_index - large_candle_int_idx
    features['time_since_last_large_candle'] = features['time_since_last_large_candle'].fillna(1000)

    # 5. Advanced Regime Features
    chop_window = 20
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr_series = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    sum_tr = tr_series.rolling(chop_window).sum()
    max_hi = high.rolling(chop_window).max()
    min_lo = low.rolling(chop_window).min()
    range_hl = max_hi - min_lo

    features['choppiness_index'] = 100 * np.log10(sum_tr / (range_hl + 1e-8)) / np.log10(chop_window)

    vr_window = 50
    r_20 = log_ret.rolling(20).sum()
    r_10 = log_ret.rolling(10).sum()
    var_20 = r_20.rolling(vr_window).var()
    var_10 = r_10.rolling(vr_window).var()
    features['variance_ratio'] = var_20 / (2 * var_10 + 1e-8)

    # Permutation Entropy
    pe_window = 50
    pe_dim = 3
    pe_values = close.values
    pe_n = len(pe_values)

    if pe_n >= pe_window + pe_dim:
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            windows = sliding_window_view(pe_values, window_shape=pe_dim)
        except ImportError:
            shape = (pe_n - pe_dim + 1, pe_dim)
            strides = (pe_values.strides[0], pe_values.strides[0])
            windows = np.lib.stride_tricks.as_strided(pe_values, shape=shape, strides=strides)

        patterns = np.argsort(windows, axis=1)
        perms = list(itertools.permutations(range(pe_dim)))
        perm_to_code = {p: i for i, p in enumerate(perms)}
        codes = np.apply_along_axis(lambda x: perm_to_code[tuple(x)], 1, patterns)

        code_series = pd.Series(codes, index=df.index[pe_dim - 1:])

        def calc_ent(x):
            counts = np.unique(x, return_counts=True)[1]
            probs = counts / counts.sum()
            max_ent = np.log2(math.factorial(pe_dim))
            ent = entropy(probs, base=2)
            return ent / max_ent

        rolling_ent = code_series.rolling(pe_window).apply(calc_ent, raw=True)
        features['permutation_entropy'] = rolling_ent.reindex(df.index)
    else:
        features['permutation_entropy'] = np.nan

    # Time Features
    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    else:
        features['hour_sin'] = 0.0
        features['hour_cos'] = 0.0
        features['is_weekend'] = 0

    return features

def generate_layer3_features(
    df: pd.DataFrame,
    base_model_cols: List[str],
) -> pd.DataFrame:
    """
    Generate features specific to Layer 3 stacking logic.
    These features are derived from base model probabilities, market data,
    and regime/structure analysis.

    Includes:
    - Ensemble Probability (arithmetic mean)
    - Logit Probability & Momentum
    - Volume at Signal (ratio vs 50-bar rolling mean)
    - Candle Shape (current & 4-bar aggregate)
    - Regime/Structure features (Volatility, Trend, Complexity, etc.)

    Args:
        df: DataFrame containing base model columns and market data ('volume', 'high', 'low', 'close')
        base_model_cols: List of column names for base model probabilities

    Returns:
        DataFrame with added feature columns.
    """
    # Create a copy to avoid SettingWithCopy warnings if df is a slice
    df_out = df.copy()

    # 1. Ensemble Probability
    # Use existing if provided (e.g. from ensemble_disagreement), else calculate fallback
    if 'ensemble_prob' not in df_out.columns:
        if base_model_cols:
            # Calculate mean of available base models
            valid_cols = [c for c in base_model_cols if c in df_out.columns]
            if valid_cols:
                df_out['ensemble_prob'] = df_out[valid_cols].mean(axis=1)
            else:
                df_out['ensemble_prob'] = 0.5
        else:
            df_out['ensemble_prob'] = 0.5

    # 2. Logit Probability & Momentum
    # Clip to avoid inf/nan in logit: [0.005, 0.995]
    eps = 0.005
    clipped_prob = df_out['ensemble_prob'].clip(eps, 1.0 - eps)
    df_out['logit_prob'] = np.log(clipped_prob / (1.0 - clipped_prob))

    df_out['logit_momentum_5'] = df_out['logit_prob'] - df_out['logit_prob'].shift(5)
    df_out['logit_momentum_1'] = df_out['logit_prob'] - df_out['logit_prob'].shift(1)

    # 3. Volume at Signal (Ratio vs 50-bar average)
    if 'volume' in df_out.columns:
        vol = df_out['volume'].replace(0, np.nan)
        avg_vol = vol.rolling(window=50, min_periods=1).mean()
        # Ratio: volume / avg_vol
        df_out['vol_at_signal'] = vol / (avg_vol + 1e-8)
        # Fill NaNs/Infs
        df_out['vol_at_signal'] = df_out['vol_at_signal'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    else:
        df_out['vol_at_signal'] = 1.0

    # 4. Candle Shape: (High - Low) / Close
    required_price_cols = ['high', 'low', 'close']
    if all(c in df_out.columns for c in required_price_cols):
        high = df_out['high']
        low = df_out['low']
        close = df_out['close'].replace(0, np.nan)

        # Current bar shape
        df_out['candle_shape'] = (high - low) / close

        # 4-bar aggregated shape (as if it were a single longer bar)
        # Rolling max high, rolling min low over last 4 bars
        roll_high = high.rolling(window=4, min_periods=1).max()
        roll_low = low.rolling(window=4, min_periods=1).min()

        # Normalized by current close
        df_out['candle_shape_4'] = (roll_high - roll_low) / close

        # Cleanup
        for c in ['candle_shape', 'candle_shape_4']:
            df_out[c] = df_out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        df_out['candle_shape'] = 0.0
        df_out['candle_shape_4'] = 0.0

    # 5. Regime Features (from GateModel logic)
    regime_feats = _compute_gate_regime_features(df_out)
    for col in regime_feats.columns:
        df_out[col] = regime_feats[col]

    return df_out
