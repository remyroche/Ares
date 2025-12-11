#!/usr/bin/env python3
"""
Generate and assess sample weights for meta-labeling.

This script implements logic to generate sample weights based on:
1. Return Magnitude (log-compressed)
2. Horizon Consistency (correlation of returns across horizons)
3. Uniqueness (De Prado's uniqueness score)

It then assesses these weights by training a probe model (LGBM) and checking
if the weighted training improves learnability (AUC, Brier, Sharpe, IC) compared to unweighted.

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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score
)

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

def calculate_sharpe_ratio(preds, returns, threshold=0.5):
    """Calculate Sharpe Ratio of a strategy trading on predictions > threshold."""
    signals = (preds > threshold).astype(int)

    strategy_returns = signals * returns

    if np.sum(signals) == 0:
        return 0.0

    mean_ret = np.mean(strategy_returns)
    std_ret = np.std(strategy_returns)

    if std_ret == 0:
        return 0.0

    # Just raw sharpe per trade for comparison
    return mean_ret / std_ret

def compute_ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Include 1.0 in the last bin
        if i == n_bins - 1:
            mask = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            mask = (y_prob >= bin_lower) & (y_prob < bin_upper)

        if np.sum(mask) > 0:
            bin_prob = np.mean(y_prob[mask])
            bin_true = np.mean(y_true[mask])
            bin_weight = np.sum(mask) / len(y_prob)
            ece += bin_weight * np.abs(bin_prob - bin_true)

    return ece

def assess_label_quality_ml(X, y, weights, returns=None, n_splits=3):
    """
    Train a probe model with the given weights and evaluate its performance.

    Includes calibration (Isotonic Regression) on an inner split to ensure
    Brier Score and ECE are meaningful.
    """
    # Prepare data
    if returns is None:
        returns = np.zeros(len(y))

    valid_mask = ~y.isna() & ~X.isna().any(axis=1) & np.isfinite(weights)
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    w_clean = weights[valid_mask]
    r_clean = returns[valid_mask]

    if len(y_clean) < 100:
        return {
            'auc': 0.5, 'log_loss': 1.0, 'brier_score': 0.25, 'ece': 0.0,
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'sharpe_ratio': 0.0, 'information_coefficient': 0.0
        }

    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = {
        'auc': [],
        'log_loss': [],
        'brier_score': [],
        'ece': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'sharpe_ratio': [],
        'information_coefficient': []
    }

    model = lgb.LGBMClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        verbose=-1,
        random_state=42
    )

    for train_idx, test_idx in tscv.split(X_clean):
        # Time-series safe internal split for calibration
        # Split train_idx into fit (70%) and calib (30%)
        n_train = len(train_idx)
        split_point = int(n_train * 0.7)

        # If too small, skip calibration split and just use raw (fallback)
        if split_point < 20 or (n_train - split_point) < 20:
            fit_idx = train_idx
            calib_idx = None
        else:
            fit_idx = train_idx[:split_point]
            calib_idx = train_idx[split_point:]

        X_fit = X_clean.iloc[fit_idx]
        y_fit = y_clean.iloc[fit_idx]
        w_fit = w_clean[fit_idx]

        X_test = X_clean.iloc[test_idx]
        y_test = y_clean.iloc[test_idx]
        r_test = r_clean[test_idx]

        try:
            # 1. Train base model on fit set
            model.fit(X_fit, y_fit, sample_weight=w_fit)

            # 2. Calibrate
            iso_reg = None
            if calib_idx is not None:
                X_cal = X_clean.iloc[calib_idx]
                y_cal = y_clean.iloc[calib_idx]

                # Get raw probabilities on calibration set
                probs_cal = model.predict_proba(X_cal)[:, 1]

                # Fit Isotonic Regression
                iso_reg = IsotonicRegression(out_of_bounds='clip')
                iso_reg.fit(probs_cal, y_cal) # Isotonic doesn't support sample_weight in older sklearn, skipping w for calib

            # 3. Predict on Test
            preds_raw = model.predict_proba(X_test)[:, 1]

            if iso_reg:
                preds_calibrated = iso_reg.predict(preds_raw)
            else:
                preds_calibrated = preds_raw

            # 4. Compute Metrics on Calibrated Probabilities
            preds_binary = (preds_calibrated > 0.5).astype(int)

            metrics['auc'].append(roc_auc_score(y_test, preds_calibrated))
            metrics['log_loss'].append(log_loss(y_test, preds_calibrated))
            metrics['brier_score'].append(brier_score_loss(y_test, preds_calibrated))
            metrics['ece'].append(compute_ece(y_test, preds_calibrated))

            metrics['precision'].append(precision_score(y_test, preds_binary, zero_division=0))
            metrics['recall'].append(recall_score(y_test, preds_binary, zero_division=0))
            metrics['f1'].append(f1_score(y_test, preds_binary, zero_division=0))

            # Financial Metrics
            metrics['sharpe_ratio'].append(calculate_sharpe_ratio(preds_calibrated, r_test))

            # Information Coefficient (Rank correlation between prob and return)
            ic, _ = spearmanr(preds_calibrated, r_test)
            metrics['information_coefficient'].append(ic if np.isfinite(ic) else 0.0)

        except Exception:
            continue

    if not metrics['auc']:
        return {k: 0.0 for k in metrics}

    # Aggregate
    results = {k: np.mean(v) for k, v in metrics.items()}
    results['auc_std'] = np.std(metrics['auc'])

    return results

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
    print("Assessing weights with ML probe (with calibration)...")
    # Prepare features: exclude targets and leaks
    drop_cols = [c for c in df.columns if 'target' in c or 'label' in c or 'return' in c or 'weight' in c or 'future' in c]
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
    y = df[target_col]
    
    # Baseline (Uniform weights)
    uniform_weights = np.ones_like(weights)
    res_base = assess_label_quality_ml(X, y, uniform_weights, returns=returns)
    
    # Weighted
    res_weighted = assess_label_quality_ml(X, y, weights, returns=returns)
    
    auc_imp = res_weighted['auc'] - res_base['auc']
    sharpe_imp = res_weighted['sharpe_ratio'] - res_base['sharpe_ratio']
    ic_imp = res_weighted['information_coefficient'] - res_base['information_coefficient']
    
    print(f"Baseline AUC: {res_base['auc']:.4f} | Weighted AUC: {res_weighted['auc']:.4f} ({auc_imp:+.4f})")
    print(f"Baseline Sharpe: {res_base['sharpe_ratio']:.4f} | Weighted Sharpe: {res_weighted['sharpe_ratio']:.4f} ({sharpe_imp:+.4f})")
    
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

        f.write("## 2. Comprehensive ML Assessment (Probe Model)\n")
        f.write("*Note: Probabilities are calibrated via Isotonic Regression on an inner temporal split before computing Brier/ECE.*\n\n")
        f.write("| Metric | Baseline (Uniform) | Weighted | Delta |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| **AUC** | {res_base['auc']:.4f} | {res_weighted['auc']:.4f} | **{auc_imp:+.4f}** |\n")
        f.write(f"| **Log Loss** | {res_base['log_loss']:.4f} | {res_weighted['log_loss']:.4f} | {(res_weighted['log_loss'] - res_base['log_loss']):+.4f} |\n")
        f.write(f"| **Brier Score** | {res_base['brier_score']:.4f} | {res_weighted['brier_score']:.4f} | {(res_weighted['brier_score'] - res_base['brier_score']):+.4f} |\n")
        f.write(f"| **ECE** | {res_base['ece']:.4f} | {res_weighted['ece']:.4f} | {(res_weighted['ece'] - res_base['ece']):+.4f} |\n")
        f.write(f"| **Sharpe Ratio** | {res_base['sharpe_ratio']:.4f} | {res_weighted['sharpe_ratio']:.4f} | **{sharpe_imp:+.4f}** |\n")
        f.write(f"| **Info. Coeff.** | {res_base['information_coefficient']:.4f} | {res_weighted['information_coefficient']:.4f} | {ic_imp:+.4f} |\n")
        f.write(f"| Precision | {res_base['precision']:.4f} | {res_weighted['precision']:.4f} | {(res_weighted['precision'] - res_base['precision']):+.4f} |\n")
        f.write(f"| Recall | {res_base['recall']:.4f} | {res_weighted['recall']:.4f} | {(res_weighted['recall'] - res_base['recall']):+.4f} |\n")
        f.write(f"| F1-Score | {res_base['f1']:.4f} | {res_weighted['f1']:.4f} | {(res_weighted['f1'] - res_base['f1']):+.4f} |\n\n")

        f.write("## 3. Interpretation\n")

        if auc_imp > 0.005 and sharpe_imp > 0:
            f.write(f"✅ **Strong Positive Impact**: Weights improved both Learnability (AUC) and Profitability (Sharpe).\n")
            f.write("The weighting scheme is effectively highlighting high-quality signal events.\n")
        elif auc_imp > 0.005:
            f.write(f"⚠️ **Mixed Impact**: Improved Learnability (AUC) but not Profitability (Sharpe).\n")
            f.write("The model learns better, but it might be learning economically insignificant events.\n")
        elif sharpe_imp > 0.05:
            f.write(f"⚠️ **Economic Impact**: Improved Profitability (Sharpe) despite flat/lower AUC.\n")
            f.write("The weights are forcing the model to focus on the 'big winners', possibly sacrificing overall accuracy.\n")
        else:
            f.write(f"⚪ **Neutral/Negative Impact**: No significant improvement detected.\n")

    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
