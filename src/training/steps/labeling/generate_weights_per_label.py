#!/usr/bin/env python3
"""
Generate and assess sample weights for meta-labeling.

This script implements logic to generate sample weights based on:
1. Return Magnitude (log-compressed)
2. Horizon Consistency (correlation of returns across horizons)
3. Uniqueness (De Prado's uniqueness score)

It then assesses these weights by training a probe model (LGBM) and checking
if the weighted training improves learnability (AUC) compared to unweighted.

Usage:
    python src/training/steps/labeling/generate_weights_per_label.py \
        --symbol ETHUSDT --exchange binance --timeframe 15m
"""

import argparse
import logging
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import spearmanr
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import system_logger
try:
    from src.training.steps.labeling.feature_generation_meta_labeling_step import (
        FeatureGenerationMetaLabelingStep,
    )
except ImportError:
    # Fallback for test mode without full env
    FeatureGenerationMetaLabelingStep = None

logger = system_logger.getChild("generate_weights_per_label")

# -------------------------------------------------------------------------
# 1. Mathematical Helpers
# -------------------------------------------------------------------------

def sigmoid_gate(x, threshold, sharpness=10.0, reverse=False):
    """Smooth sigmoid transition."""
    # Clip exponent to avoid overflow
    exponent = np.clip(-sharpness * (x - threshold), -50, 50)
    val = 1.0 / (1.0 + np.exp(exponent))
    return (1.0 - val) if reverse else val

def magnitude_aware_spearman(weights, returns, magnitude_func=np.log1p):
    """
    Vectorized magnitude-aware Spearman correlation.
    Checks if weights rank-order the *magnitude* of returns correctly.
    """
    # 1. Transform returns to magnitude
    abs_rets = np.abs(returns)
    mag_rets = magnitude_func(abs_rets)
    
    # 2. Compute Ranks
    n = len(weights)
    if n < 2:
        return 0.0

    rank_weights = np.argsort(np.argsort(weights)).astype(np.float64)
    rank_mag = np.argsort(np.argsort(mag_rets)).astype(np.float64)
    
    # 3. Center Ranks
    rank_weights -= rank_weights.mean()
    rank_mag -= rank_mag.mean()
    
    # 4. Weighted Covariance
    weighted_cov = np.sum(rank_weights * rank_mag * mag_rets) / (np.sum(mag_rets) + 1e-12)
    
    # 5. Normalize
    std_prod = np.std(rank_weights) * np.std(rank_mag)
    if std_prod == 0:
        return 0.0

    corr = weighted_cov / (std_prod + 1e-12)
    return np.clip(corr, -1.0, 1.0)

# -------------------------------------------------------------------------
# 2. Metric Computation Helpers (Uniqueness, Consistency)
# -------------------------------------------------------------------------

def compute_uniqueness(t_events: pd.DataFrame, close_index: pd.DatetimeIndex) -> pd.Series:
    """
    Compute average uniqueness of each event.

    Args:
        t_events: DataFrame with 't1' (end timestamp) indexed by start timestamp.
        close_index: DatetimeIndex of the underlying price series (bars).

    Returns:
        Series of uniqueness scores (0.0 to 1.0) aligned with t_events.
    """
    if t_events.empty or len(close_index) == 0:
        return pd.Series(1.0, index=t_events.index)

    # Convert timestamps to integer indices for faster counting
    # Map t_events start/end to indices in close_index
    
    # Create a mapping from timestamp to integer index
    idx_map = pd.Series(np.arange(len(close_index)), index=close_index)
    
    # Get start and end indices for each event
    try:
        t0 = t_events.index
        t1 = t_events['t1']

        # Reindex to ensure we match available bars
        valid_mask = t0.isin(close_index) & t1.isin(close_index)
        t0_valid = t0[valid_mask]
        t1_valid = t1[valid_mask]

        if len(t0_valid) == 0:
             return pd.Series(1.0, index=t_events.index)

        start_idx = idx_map.loc[t0_valid].values
        end_idx = idx_map.loc[t1_valid].values

        # Initialize concurrency array (count of active events per bar)
        n_bars = len(close_index)
        concurrency = np.zeros(n_bars, dtype=int)

        # Accumulate concurrency
        for s, e in zip(start_idx, end_idx):
            concurrency[s:e+1] += 1 # Inclusive of end bar

        # Compute uniqueness per event: average(1 / concurrency) over the event window
        uniqueness = np.zeros(len(t0_valid))
        concurrency_inv = 1.0 / np.maximum(concurrency, 1)

        for i, (s, e) in enumerate(zip(start_idx, end_idx)):
            length = e - s + 1
            if length > 0:
                uniqueness[i] = np.sum(concurrency_inv[s:e+1]) / length
            else:
                uniqueness[i] = 1.0 # Instantaneous event

        # Align back to original index
        out = pd.Series(np.nan, index=t_events.index)
        out.loc[t0_valid] = uniqueness
        return out.fillna(1.0) # Default to unique if calculation failed

    except Exception as e:
        logger.warning(f"Uniqueness calculation failed: {e}")
        return pd.Series(1.0, index=t_events.index)

def compute_horizon_consistency(close_series: pd.Series, horizon: int = 12) -> pd.Series:
    """
    Compute consistency of returns across horizons.
    Checks correlation of returns at h vs returns at h-1 and h+1 (conceptually).
    Actually, we'll measure if the return at 'horizon' is consistent with returns at slightly different horizons.
    """
    if len(close_series) < horizon + 5:
        return pd.Series(0.0, index=close_series.index)

    # Calculate returns for horizon, h-2, h+2
    h_main = horizon
    h_short = max(1, horizon - 2)
    h_long = horizon + 2
    
    ret_main = close_series.pct_change(h_main).shift(-h_main)
    ret_short = close_series.pct_change(h_short).shift(-h_short)
    ret_long = close_series.pct_change(h_long).shift(-h_long)

    # Consistency = Agreement in sign and magnitude
    # We can use cosine similarity or correlation.
    # Simple proxy: Sign agreement * magnitude consistency

    # 1. Sign agreement (0 or 1)
    sign_agree = (np.sign(ret_main) == np.sign(ret_short)) & (np.sign(ret_main) == np.sign(ret_long))

    # 2. Magnitude consistency (ratio of smaller/larger)
    # Average magnitude
    mag_avg = (ret_short.abs() + ret_long.abs()) / 2.0
    mag_ratio = (1.0 - (ret_main.abs() - mag_avg).abs() / (ret_main.abs() + 1e-9)).clip(0, 1)

    consistency = sign_agree.astype(float) * mag_ratio
    return consistency.fillna(0.0)

# -------------------------------------------------------------------------
# 3. Weight Generation Logic
# -------------------------------------------------------------------------

def generate_weights_per_label(
    returns, t_events, close_series,
    consistency_scores=None,
    uniqueness_scores=None,
    vol_proxy=None,
    mag_compression=0.5,
    learn_slope=10.0,
    learn_center=0.4,
    uniq_intensity=1.0,
    exp_mag=1.0,
    exp_learn=1.0,
    exp_uniq=1.0,
    **kwargs
):
    """
    Generate sample weights combining magnitude, consistency, and uniqueness.
    """
    # 1. Magnitude Component
    # Log-compress returns to dampen outliers
    abs_rets = np.abs(returns)
    # Mix linear and log based on mag_compression (0=linear, 1=log)
    mag_weight = (1 - mag_compression) * abs_rets + mag_compression * np.log1p(abs_rets)
    # Normalize to mean 1
    mag_weight = mag_weight / (np.nanmean(mag_weight) + 1e-9)

    # 2. Learnability/Consistency Component
    if consistency_scores is None:
        consistency_scores = np.ones_like(returns)

    # Sigmoid gate on consistency
    learn_weight = sigmoid_gate(consistency_scores, threshold=learn_center, sharpness=learn_slope)
    # Normalize
    learn_weight = learn_weight / (np.nanmean(learn_weight) + 1e-9)

    # 3. Uniqueness Component
    if uniqueness_scores is None:
        uniqueness_scores = np.ones_like(returns)

    # Scale uniqueness impact
    # If uniq_intensity > 1, we punish redundancy heavily
    uniq_weight = uniqueness_scores ** uniq_intensity
    uniq_weight = uniq_weight / (np.nanmean(uniq_weight) + 1e-9)

    # 4. Combine
    # Multiplicative combination with exponents
    raw_combined = (mag_weight ** exp_mag) * (learn_weight ** exp_learn) * (uniq_weight ** exp_uniq)

    # Final normalization
    final_weights = raw_combined / (np.nanmean(raw_combined) + 1e-9)

    # Safety clip
    final_weights = np.clip(final_weights, 0.01, 10.0)
    
    return final_weights

# -------------------------------------------------------------------------
# 4. Assessment (ML Probe)
# -------------------------------------------------------------------------

def assess_label_quality_ml(X, y, weights, n_splits=3):
    """
    Train a probe model with the given weights and evaluate its performance.
    """
    # Prepare data
    valid_mask = ~y.isna() & ~X.isna().any(axis=1) & np.isfinite(weights)
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    w_clean = weights[valid_mask]
    
    if len(y_clean) < 100:
        return {'auc': 0.5, 'log_loss': 1.0}

    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    losses = []

    model = lgb.LGBMClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        verbose=-1,
        random_state=42
    )
    
    for train_idx, test_idx in tscv.split(X_clean):
        X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]
        w_train = w_clean[train_idx]

        try:
            # Train WITH weights
            model.fit(X_train, y_train, sample_weight=w_train)

            # Evaluate WITHOUT weights (we want to know if it learned the ground truth better)
            # Or should we evaluate weighted? Standard practice is evaluating on raw/uniform test set.
            preds = model.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, preds)
            ll = log_loss(y_test, preds)
            aucs.append(auc)
            losses.append(ll)
        except Exception as e:
            continue

    if not aucs:
        return {'auc': 0.5, 'log_loss': 1.0}

    return {
        'auc': np.mean(aucs),
        'auc_std': np.std(aucs),
        'log_loss': np.mean(losses)
    }

# -------------------------------------------------------------------------
# 5. Test Data Generation
# -------------------------------------------------------------------------

def _generate_test_data(n_rows=1000):
    """Generate synthetic data for testing logic."""
    dates = pd.date_range(start="2023-01-01", periods=n_rows, freq="15min")
    close = np.random.lognormal(0, 0.01, n_rows).cumprod()
    close = pd.Series(close, index=dates)

    df = pd.DataFrame({'close': close, 'volatility_1d': 0.01}, index=dates)

    # Generate random features
    for i in range(5):
        df[f'feature_{i}'] = np.random.randn(n_rows)

    # Generate realized returns and labels (correlated with feature_0 for signal)
    df['realized_return'] = df['feature_0'] * 0.01 + np.random.randn(n_rows) * 0.005
    df['binary_label'] = (df['realized_return'] > 0).astype(int)

    # Duration
    df['event_duration_bars'] = np.random.randint(1, 20, n_rows)

    return df

# -------------------------------------------------------------------------
# 6. Main Execution
# -------------------------------------------------------------------------

def _load_labeled_data(symbol, exchange, timeframe):
    """Load labeled_data using the step logic."""
    if FeatureGenerationMetaLabelingStep is None:
        return None

    step = FeatureGenerationMetaLabelingStep()
    step.set_context(symbol=symbol, exchange=exchange, timeframe=timeframe)

    # Try different naming conventions
    candidates = [
        f"labeled_data_{symbol}_{timeframe}",
        f"labeled_data_{symbol}_{exchange}_{timeframe}",
    ]

    for artifact_name in candidates:
        try:
            df = step._get_artifact(artifact_name, artifact_type="data", data_category="features")
            if df is not None and not df.empty:
                return df
        except Exception as e:
            pass

    return None

def main():
    parser = argparse.ArgumentParser(description="Generate and assess label weights.")
    parser.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    parser.add_argument("--timeframe", type=str, default="15m", help="Timeframe (e.g. 15m)")
    parser.add_argument("--test-mode", action="store_true", help="Run with synthetic data")

    args = parser.parse_args()

    if args.test_mode:
        print("Running in TEST MODE with synthetic data...")
        df = _generate_test_data(1000)
    else:
        # 1. Load Data
        print(f"Loading data for {args.symbol} {args.timeframe}...")
        df = _load_labeled_data(args.symbol, args.exchange, args.timeframe)
        if df is None:
            print(f"Error: Could not load labeled_data for {args.symbol}. Use --test-mode to verify logic.")
            sys.exit(1)

    print(f"Loaded {len(df)} rows.")

    # 2. Prepare Inputs
    if 'realized_return' not in df.columns:
        print("Error: 'realized_return' not in labeled_data")
        sys.exit(1)

    # Use binary_label if available, or target
    target_col = 'binary_label' if 'binary_label' in df.columns else 'target'
    if target_col not in df.columns:
        print("Error: No binary label column found.")
        sys.exit(1)

    returns = df['realized_return'].fillna(0.0).values
    close_series = df['close']

    # Prepare t_events (approximated from duration if explicit t1 not available)
    # labeled_data usually has 'event_duration_bars'
    if 'event_duration_bars' in df.columns:
        durations = df['event_duration_bars'].fillna(1).astype(int)
        # t1 is index + duration
        # We need integer iloc access
        t_events = pd.DataFrame(index=df.index)
        # Use simple integer offset for now as index might be non-contiguous
        # Map current index time to close_series index
        t_events['t1'] = [
            df.index[min(i + d, len(df)-1)]
            for i, d in enumerate(durations)
        ]
    else:
        # Fallback: assume fixed horizon
        horizon = 12
        t_events = pd.DataFrame(index=df.index)
        t_events['t1'] = [
            df.index[min(i + horizon, len(df)-1)]
            for i in range(len(df))
        ]

    # 3. Pre-calculate Metrics
    print("Computing consistency and uniqueness...")
    consistency_scores = compute_horizon_consistency(close_series).values
    uniqueness_scores = compute_uniqueness(t_events, df.index).values
    vol_proxy = df['volatility_1d'].fillna(0.0).values if 'volatility_1d' in df.columns else np.zeros_like(returns)

    # 4. Generate Weights (using default params or "best" generic params)
    # These params could be tuned via HPO, here we use reasonable defaults
    # favoring uniqueness and learnability
    params = {
        'mag_compression': 0.5,
        'learn_slope': 10.0,
        'learn_center': 0.4,
        'uniq_intensity': 1.0,
        'exp_mag': 1.0,
        'exp_learn': 1.2, # Slight bias to consistency
        'exp_uniq': 1.0,
    }

    print("Generating weights...")
    weights = generate_weights_per_label(
        returns, t_events, close_series,
        consistency_scores=consistency_scores,
        uniqueness_scores=uniqueness_scores,
        vol_proxy=vol_proxy,
        **params
    )
    
    df['generated_sample_weight'] = weights
    
    # 5. Assess with ML
    print("Assessing weights with ML probe...")
    # Prepare features: exclude targets and leaks
    drop_cols = [c for c in df.columns if 'target' in c or 'label' in c or 'return' in c or 'weight' in c or 'future' in c]
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
    y = df[target_col]
    
    # Baseline (Uniform weights)
    uniform_weights = np.ones_like(weights)
    res_base = assess_label_quality_ml(X, y, uniform_weights)
    
    # Weighted
    res_weighted = assess_label_quality_ml(X, y, weights)
    
    print(f"Baseline AUC: {res_base['auc']:.4f}")
    print(f"Weighted AUC: {res_weighted['auc']:.4f}")
    
    improvement = res_weighted['auc'] - res_base['auc']
    print(f"Improvement: {improvement:+.4f}")
    
    # 6. Generate Report
    outcomes_dir = Path("outcomes")
    outcomes_dir.mkdir(exist_ok=True)
    ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = outcomes_dir / f"weight_assessment_{args.symbol}_{ts_str}.md"
    
    with open(report_path, "w") as f:
        f.write(f"# Label Weight Assessment Report\n\n")
        f.write(f"**Date:** {ts_str}\n")
        f.write(f"**Symbol:** {args.symbol}\n")
        f.write(f"**Timeframe:** {args.timeframe}\n\n")

        f.write("## 1. Weight Statistics\n")
        f.write(f"- Mean: {np.mean(weights):.4f}\n")
        f.write(f"- Std: {np.std(weights):.4f}\n")
        f.write(f"- Min: {np.min(weights):.4f}\n")
        f.write(f"- Max: {np.max(weights):.4f}\n\n")

        f.write("## 2. ML Assessment (Probe Model)\n")
        f.write("| Metric | Baseline (Uniform) | Weighted |\n")
        f.write("|---|---|---|\n")
        f.write(f"| AUC | {res_base['auc']:.4f} | {res_weighted['auc']:.4f} |\n")
        f.write(f"| Log Loss | {res_base['log_loss']:.4f} | {res_weighted['log_loss']:.4f} |\n\n")

        f.write("## 3. Interpretation\n")
        if improvement > 0.01:
            f.write(f"✅ **Positive Impact**: Weighted training improved AUC by {improvement:.4f}.\n")
            f.write("The weighting scheme successfully emphasizes more learnable/reliable events.\n")
        elif improvement < -0.01:
            f.write(f"❌ **Negative Impact**: Weighted training degraded AUC by {improvement:.4f}.\n")
            f.write("The weights might be focusing on noise or filtering out valuable signals.\n")
        else:
            f.write(f"⚪ **Neutral Impact**: Minimal change in AUC ({improvement:+.4f}).\n")
            f.write("The weighting scheme didn't significantly alter learnability.\n")

    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
