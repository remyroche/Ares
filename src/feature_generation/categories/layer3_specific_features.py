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

from src.feature_generation.categories.ensemble_disagreement import (
    calculate_ensemble_disagreement_features,
)

def shrink(x, n, prior, k=50):
    """
    De Prado's shrinkage estimator.
    x: Observed mean (e.g., empirical hit rate)
    n: Number of observations
    prior: Prior belief (e.g., 0.5)
    k: Shrinkage factor (speed of convergence)
    """
    # Weight of the prior decreases as n increases
    w = k / (n + k)
    return (1 - w) * x + w * prior

def compute_geometry_features(events_df: pd.DataFrame, window_size: int = 50) -> pd.DataFrame:
    """
    Computes 'Geometry' features based on the path characteristics of PAST trades.

    Includes:
    - Fragility (Avg MAE of winners)
    - Time Geometry (Speed of stops vs targets)
    - Empirical Payoff (Shrinkage-adjusted expectancy)
    """
    # Ensure we are sorted by entry time to prevent lookahead bias
    if 'entry_time' in events_df.columns:
        df = events_df.sort_values('entry_time').copy()
    else:
        df = events_df.copy()

    # Initialize output features
    feats = pd.DataFrame(index=df.index)

    # -------------------------------------------------------------
    # 1. Fragility Geometry (MAE/MFE)
    # -------------------------------------------------------------
    # We only look at CLOSED trades to compute statistics for OPENING trades.
    # We use .shift(1) to strictly enforce "past events only".

    # Normalized MAE (How much heat do we take?)
    # We fillna(0) for MAE because a missing MAE usually implies 0 (immediate win),
    # but strictly checking your data generation logic is safer.
    if 'mae_norm' in df.columns and 'mfe_norm' in df.columns:
        mae_series = df['mae_norm'].fillna(0)
        mfe_series = df['mfe_norm'].fillna(0)

        # Rolling Average MAE (The "Pain" Index)
        # If this is rising, the strategy is getting lucky (surviving deep drawdowns).
        feats['geo_rolling_mae'] = mae_series.rolling(window=window_size).mean().shift(1)

        # Rolling MAE Volatility
        # Inconsistent MAE implies unstable execution.
        feats['geo_mae_volatility'] = mae_series.rolling(window=window_size).std().shift(1)

        # MFE/MAE Ratio (The "Efficiency" Index)
        # High is good. Low implies we risk $1 to make $0.1 momentarily.
        # Add epsilon to avoid div by zero.
        feats['geo_efficiency_ratio'] = (
            mfe_series.rolling(window=window_size).mean() /
            (mae_series.rolling(window=window_size).mean() + 1e-4)
        ).shift(1)

    # -------------------------------------------------------------
    # 2. Time Geometry (Tau)
    # -------------------------------------------------------------
    # Convert timestamps to float seconds or integer ticks for calculation
    # Assuming tau_stop/target are timestamps. If indices, skip conversion.
    if 'tau_stop' in df.columns and 'tau_target' in df.columns and 'entry_time' in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df['entry_time']):
            dur_stop = (df['tau_stop'] - df['entry_time']).dt.total_seconds()
            dur_target = (df['tau_target'] - df['entry_time']).dt.total_seconds()
        else:
            dur_stop = df['tau_stop'] - df['entry_time']
            dur_target = df['tau_target'] - df['entry_time']

        # Rolling Median Duration for Stops vs Targets
        # We use median to be robust against outliers (stuck trades).
        feats['geo_median_time_to_stop'] = dur_stop.rolling(window=window_size, min_periods=10).median().shift(1)
        feats['geo_median_time_to_target'] = dur_target.rolling(window=window_size, min_periods=10).median().shift(1)

        # Time Asymmetry: Are winners faster than losers?
        # > 1.0 implies we hold losers longer (dangerous).
        # < 1.0 implies we cut losers fast (good).
        feats['geo_time_asymmetry'] = (
            feats['geo_median_time_to_stop'] /
            (feats['geo_median_time_to_target'] + 1e-4)
        )

    # -------------------------------------------------------------
    # 3. Empirical Payoff (Shrinkage)
    # -------------------------------------------------------------
    # This implements your specific "expected_normalized_payoff" logic
    # but in a rolling, vectorized way.

    if 'hit_target' in df.columns and 'hit_stop' in df.columns and 'stop_size' in df.columns and 'target_size' in df.columns:
        # Rolling count of hits
        wins = df['hit_target'].rolling(window=window_size).sum().shift(1)
        losses = df['hit_stop'].rolling(window=window_size).sum().shift(1)
        counts = wins + losses

        # Shrinkage-adjusted Win Rate (Probability of Target)
        # Prior = 0.5 (Assumption of random chance)
        raw_win_rate = wins / (counts + 1e-9)
        feats['geo_prob_target_shrunk'] = shrink(raw_win_rate, counts, prior=0.5, k=window_size/2)

        # Shrinkage-adjusted Loss Rate (Probability of Stop)
        raw_loss_rate = losses / (counts + 1e-9)
        feats['geo_prob_stop_shrunk'] = shrink(raw_loss_rate, counts, prior=0.5, k=window_size/2)

        # Implied Payoff
        # We take the CURRENT trade's R:R ratio (target_size / stop_size)
        # and combine it with HISTORICAL probabilities.
        # lambda_rr = Stop / Target (Cost / Reward)
        current_lambda = df['stop_size'] / (df['target_size'] + 1e-9)

        feats['geo_expected_payoff'] = (
            feats['geo_prob_target_shrunk'] -
            (current_lambda * feats['geo_prob_stop_shrunk'])
        )

    return feats.fillna(0)

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
    rv_short = log_ret.rolling(window=12).std() * np.sqrt(12)
    # rv_short, rv_short_over_med, rv_z_short removed from output as per request

    tr = (high - low) / close
    atr_short = tr.rolling(window=12).mean()

    rv_long = log_ret.rolling(window=200).std()
    # rv_z_short removed

    # 2. Trend Strength Features
    log_price = np.log(close)
    features['slope_short'] = log_price.diff(12).abs()

    up_move = high.diff()
    down_move = low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smooth = tr.rolling(window=14).sum()
    plus_di = pd.Series(plus_dm, index=df.index).rolling(window=14).sum() / (tr_smooth + 1e-8)
    minus_di = pd.Series(minus_dm, index=df.index).rolling(window=14).sum() / (tr_smooth + 1e-8)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    features['adx_proxy'] = dx.rolling(window=14).mean()

    # 3. Momentum
    features['momentum_short'] = (close.diff(12) / close.shift(12)).abs()
    features['snr'] = features['momentum_short'].abs() / (rv_short + 1e-8)

    # 4. New Features (Vol Spike, Large Candle, Time)
    rv_mean = rv_short.rolling(window=100, min_periods=20).mean()
    rv_std = rv_short.rolling(window=100, min_periods=20).std()
    rv_z = (rv_short - rv_mean) / (rv_std + 1e-8)

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
    pe_values = close.to_numpy(dtype=float, copy=False)
    pe_n = len(pe_values)

    if pe_n >= pe_window + pe_dim:
        a = pe_values[:-2]
        b = pe_values[1:-1]
        c = pe_values[2:]

        codes = np.empty(pe_n - 2, dtype=np.int8)

        case0 = (a <= b) & (b <= c)  # (0,1,2)
        case1 = (a <= c) & (c < b)   # (0,2,1)
        case2 = (b < a) & (a <= c)   # (1,0,2)
        case3 = (b <= c) & (c < a)   # (1,2,0)
        case4 = (c < a) & (a <= b)   # (2,0,1)

        codes[case0] = 0
        codes[case1] = 1
        codes[case2] = 2
        codes[case3] = 3
        codes[case4] = 4
        codes[~(case0 | case1 | case2 | case3 | case4)] = 5  # (2,1,0)

        one_hot = np.eye(6, dtype=np.int32)[codes.astype(int)]
        csum = np.vstack([np.zeros((1, 6), dtype=np.int32), np.cumsum(one_hot, axis=0)])

        pe_ent = np.full(codes.shape[0], np.nan, dtype=float)
        n_codes = int(codes.shape[0])
        if n_codes >= pe_window:
            end = np.arange(pe_window, n_codes + 1, dtype=int)
            start = end - int(pe_window)
            counts = csum[end] - csum[start]
            probs = counts.astype(float) / float(pe_window)
            logp = np.where(probs > 0.0, np.log2(probs), 0.0)
            ent = -np.sum(probs * logp, axis=1)
            pe_ent[int(pe_window) - 1:] = ent / float(np.log2(6.0))

        features['permutation_entropy'] = pd.Series(pe_ent, index=df.index[pe_dim - 1:]).reindex(df.index)
    else:
        features['permutation_entropy'] = np.nan

    # Time Features
    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
        # Removed: features['hour'] = hour
        features['day_of_week'] = df.index.dayofweek
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    else:
        # Removed: features['hour'] = 0.0
        features['day_of_week'] = 0.0
        features['hour_sin'] = 0.0
        features['hour_cos'] = 0.0
        features['is_weekend'] = 0

    # Efficiency Ratio (Kaufman's)
    er_window = 10
    change = (close - close.shift(er_window)).abs()
    volatility = close.diff().abs().rolling(er_window).sum()
    features['efficiency_ratio'] = change / (volatility + 1e-8)

    return features

def _compute_cross_tf_momentum_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-Timeframe Momentum Agreement Features.

    Measures agreement of momentum across multiple horizons.
    Strong agreement = stronger directional conviction.
    """
    features = pd.DataFrame(index=df.index)
    close = df['close']
    
    # Fill NAs to avoid propagation issues temporarily for calculation
    close_filled = close.ffill().bfill()

    # Momentum at different horizons
    mom_4 = (close_filled / close_filled.shift(4) - 1).fillna(0)      # 1h at 15m
    mom_12 = (close_filled / close_filled.shift(12) - 1).fillna(0)    # 3h at 15m
    # Removed 12h and 24h as per user request
    
    # Sign agreement (how many horizons agree on direction)
    signs = pd.concat([
        np.sign(mom_4),
        np.sign(mom_12)
    ], axis=1)
    
    # Calculate agreement [-1, 1]
    features['momentum_agreement'] = signs.mean(axis=1)
    features['momentum_agreement_abs'] = features['momentum_agreement'].abs()

    # Magnitude-weighted agreement
    magnitudes = pd.concat([mom_4.abs(), mom_12.abs()], axis=1)
    weighted = signs.values * magnitudes.values
    features['momentum_weighted_agreement'] = pd.Series(weighted.sum(axis=1), index=df.index)

    # Trend consistency (rolling sign agreement over last 12 bars)
    features['trend_consistency_12'] = np.sign(mom_12).rolling(12).mean().abs()

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
    - Cross-Timeframe Momentum Agreement
    - Base Model OOF Predictions

    Args:
        df: DataFrame containing base model columns and market data ('volume', 'high', 'low', 'close')
        base_model_cols: List of column names for base model probabilities

    Returns:
        DataFrame with added feature columns.
    """
    # Create a copy to avoid SettingWithCopy warnings if df is a slice
    df_out = df.copy()

    try:
        if 'trend_regime' in df_out.columns:
            tr = df_out['trend_regime'].astype(str).str.lower()
            df_out['trend_regime_is_high'] = (tr == 'high').astype(float)
            df_out['trend_regime_is_low'] = (tr == 'low').astype(float)
        if 'vol_regime' in df_out.columns:
            vr = df_out['vol_regime'].astype(str).str.lower()
            df_out['vol_regime_is_high'] = (vr == 'high').astype(float)
            df_out['vol_regime_is_low'] = (vr == 'low').astype(float)
    except Exception:
        pass

    try:
        if 'volatility_1d' in df_out.columns:
            vol1d = pd.to_numeric(df_out['volatility_1d'], errors='coerce').astype(float)
            q33 = float(vol1d.quantile(0.33)) if vol1d.notna().any() else float('nan')
            q67 = float(vol1d.quantile(0.67)) if vol1d.notna().any() else float('nan')
            if np.isfinite(q33) and np.isfinite(q67) and q67 > q33:
                df_out['vol_bucket_low'] = (vol1d <= q33).astype(float)
                df_out['vol_bucket_mid'] = ((vol1d > q33) & (vol1d <= q67)).astype(float)
                df_out['vol_bucket_high'] = (vol1d > q67).astype(float)
    except Exception:
        pass

    # 1. Ensemble Probability
    # Use existing if provided (e.g. from ensemble_disagreement), else calculate fallback
    valid_cols = [c for c in (base_model_cols or []) if c in df_out.columns]

    if 'ensemble_prob' not in df_out.columns:
        if valid_cols:
            df_out['ensemble_prob'] = df_out[valid_cols].mean(axis=1)
        else:
            df_out['ensemble_prob'] = 0.5
    
    # 1a. Explicitly add Base Model Predictions & Confidence Extremes
    if valid_cols:
        # Root Cause 2: "Excluding base model confidence"
        # We add Max/Min probabilities to capture if *any* model is highly confident.
        df_out['max_base_prob'] = df_out[valid_cols].max(axis=1)
        df_out['min_base_prob'] = df_out[valid_cols].min(axis=1)
        df_out['base_prob_range'] = df_out['max_base_prob'] - df_out['min_base_prob']
    else:
        df_out['max_base_prob'] = 0.5
        df_out['min_base_prob'] = 0.5
        df_out['base_prob_range'] = 0.0

    # 1a. Explicitly add Base Model Predictions as numerical features
    for col in base_model_cols:
        if col in df_out.columns and col != 'ensemble_prob':
            df_out[f"base_pred_{col}"] = pd.to_numeric(df_out[col], errors='coerce').fillna(0.5)

    # 1b. Ensemble Disagreement Features (ens_*)
    # Enhanced disagreement features for meta model following de Prado principles
    disagree_feature_names = [
        "prediction_dispersion",      # Variance of predictions across models
        "confidence_gap",            # Margin between top predictions  
        "uncertainty",               # Normalized entropy (uncertainty measure)
        "prediction_range",         # Range of predictions (max - min)
        "avg_divergence",           # Average pairwise model divergence
        "max_confidence",           # Highest confidence among models
        "disagreement_rate",        # Proportion of models disagreeing on direction
        "snr_internal",             # Mean Probability / Mean Internal Variance
        "snr_consensus",            # Ensemble Mean Probability / StdDev of Model Predictions
        "ensemble_prob",           # Arithmetic mean of probabilities
    ]
    disagree_cols = [f"ens_{k}" for k in disagree_feature_names]
    for col in disagree_cols:
        if col not in df_out.columns:
            df_out[col] = 0.0

    try:
        valid_base_cols = [c for c in (base_model_cols or []) if c in df_out.columns]
        if valid_base_cols:
            df_out[valid_base_cols] = df_out[valid_base_cols].astype(float).fillna(0.5)

            prob_dict = {str(c): df_out[c].astype(float).values for c in valid_base_cols}
            pred_dict = {str(c): (df_out[c].astype(float).values - 0.5) for c in valid_base_cols}

            var_dict = {}
            for c in valid_base_cols:
                var_col = f"{c}_var"
                if var_col in df_out.columns:
                    try:
                        var_dict[str(c)] = pd.to_numeric(df_out[var_col], errors="coerce").astype(float).values
                    except Exception:
                        pass

            # Enhanced disagreement calculation with proper error handling
            disagree = calculate_ensemble_disagreement_features(
                model_predictions=pred_dict,
                model_probabilities=prob_dict,
                model_confidences=None,
                model_variances=var_dict if var_dict else None,
                feature_names=disagree_feature_names,  # Explicitly request all features
                logger=None,
            )

            # Apply disagreement features with proper validation
            for k, col in zip(disagree_feature_names, disagree_cols):
                v = disagree.get(k)
                if isinstance(v, pd.Series) and len(v) == len(df_out):
                    # Apply de Prado-inspired transformations
                    if k == "disagreement_rate":
                        # Transform disagreement rate to agreement strength (inverse)
                        df_out[col] = 1.0 - pd.to_numeric(v.values, errors="coerce")
                    elif k in ["snr_internal", "snr_consensus"]:
                        # Apply log transform to SNR ratios for better scaling
                        snr_vals = pd.to_numeric(v.values, errors="coerce")
                        df_out[col] = np.log1p(np.clip(snr_vals, 0, 100))  # Clip extreme values
                    elif k == "uncertainty":
                        # Keep uncertainty as-is (already normalized)
                        df_out[col] = pd.to_numeric(v.values, errors="coerce")
                    else:
                        df_out[col] = pd.to_numeric(v.values, errors="coerce")
                else:
                    df_out[col] = 0.0
    except Exception:
        pass

    df_out[disagree_cols] = df_out[disagree_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

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

    # 3b. Payoff Asymmetry (Volatility vs Cost)
    # Root Cause 2: "Ignoring volatility / payoff asymmetry"
    # We estimate risk/reward capacity by comparing volatility to an assumed cost.
    # Hardcoded cost proxy (0.003) is used as it's not passed here, but relative magnitude matters most.
    if 'volatility_1d' in df_out.columns:
        vol_1d = pd.to_numeric(df_out['volatility_1d'], errors='coerce').astype(float)
        df_out['volatility_risk_ratio'] = vol_1d / 0.003
        df_out['volatility_risk_ratio'] = df_out['volatility_risk_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    else:
        df_out['volatility_risk_ratio'] = 1.0

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

    # 6. Cross-Timeframe Momentum Agreement
    try:
        if all(c in df_out.columns for c in ['close']):
            mom_feats = _compute_cross_tf_momentum_agreement(df_out)
            for col in mom_feats.columns:
                df_out[col] = mom_feats[col]
    except Exception:
        pass

    # 7. Geometry Features
    try:
        geo_feats = compute_geometry_features(df_out)
        for col in geo_feats.columns:
            df_out[col] = geo_feats[col]
    except Exception:
        pass

    # 9. Regime Interaction Terms (Synergy)
    try:
        if 'ensemble_prob' in df_out.columns:
            ep = df_out['ensemble_prob']
            # Interaction with Trend Regimes
            if 'trend_regime_is_high' in df_out.columns:
                df_out['inter_ep_trend_high'] = ep * df_out['trend_regime_is_high']
            if 'trend_regime_is_low' in df_out.columns:
                df_out['inter_ep_trend_low'] = ep * df_out['trend_regime_is_low']
            
            # Interaction with Volatility Regimes
            if 'vol_regime_is_high' in df_out.columns:
                df_out['inter_ep_vol_high'] = ep * df_out['vol_regime_is_high']
            if 'vol_regime_is_low' in df_out.columns:
                df_out['inter_ep_vol_low'] = ep * df_out['vol_regime_is_low']
            
            # Interaction with Volatility Buckets
            if 'vol_bucket_high' in df_out.columns:
                df_out['inter_ep_vol_bucket_high'] = ep * df_out['vol_bucket_high']
            if 'vol_bucket_low' in df_out.columns:
                df_out['inter_ep_vol_bucket_low'] = ep * df_out['vol_bucket_low']
    except Exception:
        pass

    # 10. Price Position in Range
    try:
        if 'close' in df_out.columns:
            close = df_out['close']
            roll_max = close.rolling(50).max()
            roll_min = close.rolling(50).min()
            range_len = roll_max - roll_min
            df_out['price_position_in_range'] = (close - roll_min) / (range_len + 1e-8)
    except Exception:
        pass

    return df_out
