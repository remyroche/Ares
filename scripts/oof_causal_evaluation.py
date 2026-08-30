#!/usr/bin/env python3
"""
OOF Causal Evaluation Script

Evaluates Out-Of-Fold (OOF) predictions against binary and continuous targets.
Calculates classification, regression, calibration, stability, and causal metrics.
Implements De Prado's 2026 Causal Framework elements:
1. Residual Causal Test (HSIC)
2. Cross-Validation Predictability (CVP)
3. Monotonicity of Group Effects (GATES)
4. Regime-Based Residual Variance
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    precision_score, f1_score, average_precision_score,
    brier_score_loss, log_loss, mean_absolute_error, accuracy_score, auc, r2_score
)
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OOFEvaluator")

def compute_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    try:
        excess = np.asarray(returns) - risk_free_rate
        if excess.size == 0:
            return 0.0
        std = float(np.std(excess, ddof=1)) if excess.size > 1 else float(np.std(excess))
        if std <= 1e-12:
            return 0.0
        return float(np.mean(excess) / std)
    except Exception:
        return 0.0

def calculate_hsic(X: np.ndarray, Y: np.ndarray, subsample: int = 2000, random_state: int = 42) -> float:
    """
    Calculate HSIC (Hilbert-Schmidt Independence Criterion).
    Checks if Residuals (Y) are independent of Features (X).
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Handle NaNs
    mask = ~np.isnan(X).any(axis=1) if X.ndim > 1 else ~np.isnan(X)
    mask &= ~np.isnan(Y).any(axis=1) if Y.ndim > 1 else ~np.isnan(Y)
    X = X[mask]
    Y = Y[mask]

    n_samples = X.shape[0]
    if n_samples < 10: return 0.0

    if n_samples > subsample:
        rng = np.random.RandomState(random_state)
        indices = rng.choice(n_samples, subsample, replace=False)
        X = X[indices]
        Y = Y[indices]
        n_samples = subsample

    if X.ndim == 1: X = X.reshape(-1, 1)
    if Y.ndim == 1: Y = Y.reshape(-1, 1)

    # Use RBF kernel by default for non-linear dependence
    K = rbf_kernel(X)
    L = rbf_kernel(Y)

    # Centering matrix H = I - 1/n * 1 * 1^T
    H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples

    Kc = H @ K @ H
    Lc = H @ L @ H

    hsic_value = np.trace(Kc @ Lc) / ((n_samples - 1) ** 2)
    return float(hsic_value)

def qini_coefficient(y_true, y_pred):
    df = pd.DataFrame({'y': y_true, 'score': y_pred})
    df = df.sort_values('score', ascending=False)
    df['cum_y'] = df['y'].cumsum()

    n = len(df)
    total_y = df['y'].sum()
    random_curve = np.linspace(0, total_y, n)

    x_axis = np.arange(len(df))
    area_model = auc(x_axis, df['cum_y'])
    area_random = auc(x_axis, random_curve)

    qini_value = area_model - area_random
    denom = n * total_y if total_y != 0 else 1.0
    return qini_value / denom

def population_stability_index(expected, actual, buckets=10):
    try:
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2: return 0.0

        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

        def sub_psi(e_perc, a_perc):
            if a_perc == 0: a_perc = 0.0001
            if e_perc == 0: e_perc = 0.0001
            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return value

        psi_value = np.sum([sub_psi(expected_percents[i], actual_percents[i]) for i in range(len(expected_percents))])
        return psi_value
    except Exception:
        return 0.0

def causal_calibrated_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            ece += np.abs(y_prob[in_bin].mean() - y_true[in_bin].mean()) * prop_in_bin
    return ece

def prediction_entropy(probs):
    probs = np.clip(probs, 1e-9, 1 - 1e-9)
    return stats.entropy(np.vstack([probs, 1-probs]).T, axis=1).mean()

def prob_sharpe_ratio(sharpe, skew, kurt, n):
    if n <= 1: return 0.0
    denom = np.sqrt(1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2)
    if denom == 0 or np.isnan(denom): return 0.0
    z_stat = (sharpe * np.sqrt(n - 1)) / denom
    return stats.norm.cdf(z_stat)

def calculate_max_drawdown_duration(cumulative_returns):
    high_water_mark = cumulative_returns.cummax()
    drawdown = cumulative_returns - high_water_mark
    is_drawdown = drawdown < 0
    streaks = is_drawdown.ne(is_drawdown.shift()).cumsum()
    durations = streaks[is_drawdown].value_counts()
    return durations.max() if not durations.empty else 0

def detect_regimes(returns, volatility_window=20, trend_window=20):
    vol = returns.rolling(volatility_window).std()
    low_thresh = vol.quantile(0.33)
    high_thresh = vol.quantile(0.66)

    vol_regime = pd.Series(index=returns.index, data='Mid')
    vol_regime[vol <= low_thresh] = 'Low'
    vol_regime[vol > high_thresh] = 'High'

    abs_change = returns.abs().rolling(trend_window).sum()
    net_change = returns.rolling(trend_window).sum().abs()
    er = net_change / abs_change.replace(0, np.inf)

    trend_thresh = er.median()
    trend_regime = pd.Series(index=returns.index, data='Range')
    trend_regime[er > trend_thresh] = 'Trend'

    return vol_regime + "_" + trend_regime

def gates_analysis(y_true, y_pred, buckets=10):
    """
    Monotonicity of Group Effects (GATES).
    Checks if mean outcome increases with predicted decile.
    """
    try:
        df = pd.DataFrame({'true': y_true, 'pred': y_pred})
        df['decile'] = pd.qcut(df['pred'], buckets, labels=False, duplicates='drop')
        grouped = df.groupby('decile')['true'].mean()

        # Calculate Spearman correlation between decile rank and mean outcome
        spearman, _ = stats.spearmanr(grouped.index, grouped.values)

        # Monotonicity Score: % of steps that are increasing
        diffs = np.diff(grouped.values)
        monotonicity_score = np.mean(diffs > 0)

        return spearman, monotonicity_score, grouped.to_dict()
    except Exception as e:
        return 0.0, 0.0, {}

def cvp_test(X, y_true, model_name='Ridge'):
    """
    Cross-Validation Predictability (CVP).
    Compare R2 of X -> Y vs Y -> X.
    Since Y is 1D and X is ND, for Y -> X we predict each feature and average R2?
    Or just predict the first PC of X.
    We'll average R2 of top 5 features predicted by Y.
    """
    try:
        # Standardize
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        scaler_y = StandardScaler()
        y_scaled = y_true.reshape(-1, 1)
        y_scaled = scaler_y.fit_transform(y_scaled).ravel()

        # Model A: X -> Y
        model_a = Ridge()
        model_a.fit(X_scaled, y_scaled)
        y_pred_a = model_a.predict(X_scaled)
        r2_a = r2_score(y_scaled, y_pred_a)

        # Model B: Y -> X (Predict top features)
        # We can't easily define "R2 of vector". We'll take average R2 of predicting each feature.
        # This is rough but indicative.
        r2_b_scores = []
        # Limit to first 10 features to save time if high dim
        limit_feats = min(10, X.shape[1])
        for i in range(limit_feats):
            model_b = Ridge()
            # Predict X_i using Y
            model_b.fit(y_scaled.reshape(-1, 1), X_scaled[:, i])
            x_pred = model_b.predict(y_scaled.reshape(-1, 1))
            r2_b_scores.append(r2_score(X_scaled[:, i], x_pred))

        r2_b = np.mean(r2_b_scores)

        return r2_a, r2_b
    except Exception:
        return 0.0, 0.0

def run_evaluation(predictions_path, targets_path, output_dir, target_col, return_col, fold_col, volume_col, cost, pred_cols, timestamp_col, features_path):
    logger.info(f"Loading predictions from {predictions_path}")
    preds_df = pd.read_csv(predictions_path) if predictions_path.endswith('.csv') else pd.read_parquet(predictions_path)

    if targets_path:
        logger.info(f"Loading targets from {targets_path}")
        targets_df = pd.read_csv(targets_path) if targets_path.endswith('.csv') else pd.read_parquet(targets_path)
        if len(targets_df) == len(preds_df):
             for col in [target_col, return_col, volume_col, timestamp_col]:
                 if col and col not in preds_df.columns and col in targets_df.columns:
                     preds_df[col] = targets_df[col].values
        else:
            preds_df = preds_df.merge(targets_df, left_index=True, right_index=True)

    # Load Features if provided
    features_df = None
    if features_path:
        logger.info(f"Loading features from {features_path}")
        features_df = pd.read_csv(features_path) if features_path.endswith('.csv') else pd.read_parquet(features_path)
        # Align features
        if len(features_df) != len(preds_df):
            logger.warning("Features length mismatch. Attempting index merge.")
            # Assume index alignment if not merged
            features_df = features_df.loc[preds_df.index] if preds_df.index.equals(features_df.index) else features_df
            # If strictly different, this might fail. We assume user provided aligned csv.

    # Sort data
    if timestamp_col and timestamp_col in preds_df.columns:
        preds_df = preds_df.sort_values(timestamp_col)
        if features_df is not None: features_df = features_df.loc[preds_df.index]
    else:
        potential_time_cols = [c for c in preds_df.columns if 'time' in c.lower() or 'date' in c.lower()]
        if potential_time_cols:
            col = potential_time_cols[0]
            preds_df = preds_df.sort_values(col)
            if features_df is not None: features_df = features_df.loc[preds_df.index]
        elif fold_col in preds_df.columns:
             preds_df = preds_df.sort_values(fold_col)
             if features_df is not None: features_df = features_df.loc[preds_df.index]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if fold_col not in preds_df.columns:
        n = len(preds_df)
        preds_df['__fold__'] = pd.qcut(np.arange(n), 5, labels=False)
        fold_col = '__fold__'

    report_lines = []
    report_lines.append(f"# OOF Causal Evaluation Report (De Prado 2026 Framework)")
    report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Predictions File: {predictions_path}")
    if features_path: report_lines.append(f"Features File: {features_path}")
    report_lines.append(f"\n")

    if not pred_cols:
        pred_cols = [c for c in preds_df.columns if 'pred' in c or 'prob' in c or 'score' in c]
        pred_cols = [c for c in pred_cols if c not in [target_col, return_col, fold_col, timestamp_col, volume_col]]
        if not pred_cols:
            logger.error("No prediction columns specified or found.")
            return

    for model_name in pred_cols:
        logger.info(f"Evaluating Model: {model_name}")
        report_lines.append(f"## Model: {model_name}")

        y_prob = preds_df[model_name].values
        mask = ~np.isnan(y_prob) & ~np.isnan(preds_df[target_col]) & ~np.isnan(preds_df[return_col])
        df_clean = preds_df[mask].copy()

        if len(df_clean) == 0: continue

        y_true = df_clean[target_col].values
        y_prob = df_clean[model_name].values
        y_ret = df_clean[return_col].values
        folds = df_clean[fold_col].values

        # --- 1. Basic Metrics ---
        y_pred_bin = (y_prob > 0.5).astype(int)
        prec = precision_score(y_true, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true, y_pred_bin, zero_division=0)
        pr_auc = average_precision_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        ll = log_loss(y_true, y_prob)
        ic, _ = stats.spearmanr(y_prob, y_ret)

        report_lines.append("### Standard Metrics")
        report_lines.append(f"| Metric | Value |")
        report_lines.append(f"|---|---|")
        report_lines.append(f"| Precision | {prec:.4f} |")
        report_lines.append(f"| F1-Score | {f1:.4f} |")
        report_lines.append(f"| PR-AUC | {pr_auc:.4f} |")
        report_lines.append(f"| Brier Score | {brier:.4f} |")
        report_lines.append(f"| Log-Loss | {ll:.4f} |")
        report_lines.append(f"| IC | {ic:.4f} |")

        # --- 2. Causal Metrics (De Prado 2026) ---
        report_lines.append("\n### Causal & Stability Metrics (De Prado 2026)")
        report_lines.append(f"| Metric | Value | Description |")
        report_lines.append(f"|---|---|---|")

        # GATES (Monotonicity)
        spearman_gates, mono_score, _ = gates_analysis(y_true, y_prob)
        report_lines.append(f"| GATES Spearman | {spearman_gates:.4f} | Monotonicity of Group Effects |")
        report_lines.append(f"| GATES Monotonicity | {mono_score:.2%} | % of increasing decile steps |")

        # CVP & HSIC (Requires Features)
        if features_path and features_df is not None:
            # Align features to cleaned data
            X = features_df.loc[df_clean.index].values

            # Residual Causal Test (HSIC)
            residuals = y_true - y_prob
            hsic_score = calculate_hsic(X, residuals)
            # Rough p-value interpretation: 0 is independent. High is dependent.
            # We report raw HSIC.
            report_lines.append(f"| HSIC (Resid vs X) | {hsic_score:.6f} | Residual Independence (>0 implies leakage) |")

            # CVP (Cross-Validation Predictability)
            # Predict Y from X (Model A) vs X from Y (Model B)
            # Note: We use y_prob as proxy for "Model A" performance if it was trained on X?
            # Or we train a fresh simple model here to compare apples-to-apples.
            # We'll train fresh Ridge for both to be fair.
            r2_a, r2_b = cvp_test(X, y_true)
            direction = "X->Y" if r2_a > r2_b else "Y->X"
            report_lines.append(f"| CVP R2 (X->Y) | {r2_a:.4f} | Forward Predictability |")
            report_lines.append(f"| CVP R2 (Y->X) | {r2_b:.4f} | Backward Predictability |")
            report_lines.append(f"| Causal Arrow | {direction} | Inferred Direction |")
        else:
             report_lines.append(f"| HSIC / CVP | N/A | Requires --features argument |")

        # Other Stability
        qini = qini_coefficient(y_ret, y_prob)
        cce = causal_calibrated_error(y_true, y_prob)
        entropy = prediction_entropy(y_prob)

        report_lines.append(f"| Qini Coefficient | {qini:.6f} | Uplift Quality |")
        report_lines.append(f"| CCE (ECE) | {cce:.4f} | Causal Calibrated Error |")
        report_lines.append(f"| Pred Entropy | {entropy:.4f} | Uncertainty |")

        # --- 3. Regime Analysis & Heteroscedasticity ---
        regimes = detect_regimes(pd.Series(y_ret, index=df_clean.index))
        df_clean['regime'] = regimes
        # Residuals
        df_clean['resid'] = df_clean[target_col] - df_clean[model_name]

        report_lines.append("\n### Regime Analysis & Heteroscedasticity")
        report_lines.append(f"| Regime | Count | Sharpe | Resid Var | Bias |")
        report_lines.append(f"|---|---|---|---|---|")

        resid_vars = []

        # Financial Simulation
        signal = (y_prob > 0.5).astype(int)
        strat_ret = signal * y_ret
        turnover = pd.Series(signal, index=df_clean.index).diff().abs().fillna(0)
        costs = turnover * (cost / 2.0)
        net_strat_ret = strat_ret - costs

        for reg in sorted(df_clean['regime'].unique()):
            subset = df_clean[df_clean['regime'] == reg]
            if len(subset) < 10: continue

            reg_sharpe = compute_sharpe_ratio(net_strat_ret[subset.index])
            reg_bias = subset['resid'].mean() * -1 # Bias = Pred - True = -Resid
            reg_var = subset['resid'].var()
            resid_vars.append(reg_var)

            report_lines.append(f"| {reg} | {len(subset)} | {reg_sharpe:.4f} | {reg_var:.6f} | {reg_bias:.4f} |")

        # Heteroscedasticity Check
        if resid_vars:
            var_cv = np.std(resid_vars) / np.mean(resid_vars) if np.mean(resid_vars) != 0 else 0
            report_lines.append(f"\n**Regime Residual CoV:** {var_cv:.4f} (High = Heteroscedastic/Missing Moderator)")

        report_lines.append("\n---\n")

    fname = f"causal_reliability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    out_file = out_path / fname
    with open(out_file, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"Report generated: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OOF Causal Evaluation")
    parser.add_argument("--predictions", required=True, help="Path to predictions file")
    parser.add_argument("--targets", help="Path to targets file")
    parser.add_argument("--features", help="Path to features file (for HSIC/CVP)")
    parser.add_argument("--output", default="outcomes/", help="Output directory")
    parser.add_argument("--target-col", default="target", help="Binary target column")
    parser.add_argument("--return-col", default="returns", help="Continuous return column")
    parser.add_argument("--fold-col", default="fold", help="Fold column")
    parser.add_argument("--volume-col", help="Volume column")
    parser.add_argument("--timestamp-col", help="Timestamp column")
    parser.add_argument("--cost", type=float, default=0.003, help="Transaction cost")
    parser.add_argument("--pred-cols", nargs="+", help="Specific prediction columns")

    args = parser.parse_args()

    run_evaluation(
        args.predictions, args.targets, args.output,
        args.target_col, args.return_col, args.fold_col,
        args.volume_col, args.cost, args.pred_cols, args.timestamp_col, args.features
    )
