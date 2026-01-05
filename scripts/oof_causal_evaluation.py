#!/usr/bin/env python3
"""
OOF Causal Evaluation Script

Evaluates Out-Of-Fold (OOF) predictions against binary and continuous targets.
Calculates classification, regression, calibration, stability, and causal metrics.
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
    brier_score_loss, log_loss, mean_absolute_error, accuracy_score, auc
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OOFEvaluator")

def compute_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Compute the Sharpe ratio given per-period returns.
    """
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

def qini_coefficient(y_true, y_pred, treatment_effect=None):
    """
    Calculates Qini Coefficient (Area between Uplift Curve and Random).
    If treatment_effect is not provided, assumes y_true represents the outcome (return).
    """
    # Sort by score descending
    df = pd.DataFrame({'y': y_true, 'score': y_pred})
    df = df.sort_values('score', ascending=False)

    # Cumulative outcome
    df['cum_y'] = df['y'].cumsum()

    # Random curve (diagonal)
    total_y = df['y'].sum()
    n = len(df)
    random_curve = np.linspace(0, total_y, n)

    # Qini Area
    # Area under model curve - Area under random curve
    # Use sklearn auc for stability across versions
    x_axis = np.arange(len(df))
    area_model = auc(x_axis, df['cum_y'])
    area_random = auc(x_axis, random_curve)

    # Standard Qini usually is Area / N^2 or similar, but often reported as raw area diff.
    # We return area difference normalized by N*TotalY to make it scale-independent roughly.
    # Actually, standard Qini coefficient is often just the Area Difference.
    qini_value = area_model - area_random

    # Normalizing by total_abs_y * N would keep it in range, but definitions vary.
    # We stick to simple Area Diff / (N * Total_Y) to map to [0,1] for perfect sorting if monotonic.
    denom = n * total_y if total_y != 0 else 1.0
    return qini_value / denom

def population_stability_index(expected, actual, buckets=10):
    """
    Calculate PSI between two distributions.
    Handles duplicate bin edges safely.
    """
    try:
        # Use percentiles to find breakpoints
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        # Ensure unique to avoid ValueError in histogram
        breakpoints = np.unique(breakpoints)

        # If distribution is highly concentrated (e.g. all zeros), we can't bucket effectively.
        if len(breakpoints) < 2:
            return 0.0

        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

        def sub_psi(e_perc, a_perc):
            if a_perc == 0: a_perc = 0.0001
            if e_perc == 0: e_perc = 0.0001
            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return value

        psi_value = np.sum([sub_psi(expected_percents[i], actual_percents[i]) for i in range(len(expected_percents))])
        return psi_value
    except Exception as e:
        logger.warning(f"PSI calculation failed: {e}")
        return 0.0

def causal_calibrated_error(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error (ECE), labeled as CCE.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_prob_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_prob_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

def prediction_entropy(probs):
    """Shannon entropy of probabilities."""
    # Clip to avoid log(0)
    probs = np.clip(probs, 1e-9, 1 - 1e-9)
    return stats.entropy(np.vstack([probs, 1-probs]).T, axis=1).mean()

def prob_sharpe_ratio(sharpe, skew, kurt, n):
    """
    Probabilistic Sharpe Ratio (De Prado).
    Target Sharpe is 0.
    """
    if n <= 1: return 0.0
    # denominator
    denom = np.sqrt(1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2)
    if denom == 0 or np.isnan(denom):
        return 0.0

    z_stat = (sharpe * np.sqrt(n - 1)) / denom
    return stats.norm.cdf(z_stat)

def calculate_max_drawdown_duration(cumulative_returns):
    """
    Calculate the longest duration (in bars) of drawdown.
    """
    high_water_mark = cumulative_returns.cummax()
    drawdown = cumulative_returns - high_water_mark

    is_drawdown = drawdown < 0

    # Identify streaks of True
    streaks = is_drawdown.ne(is_drawdown.shift()).cumsum()
    durations = streaks[is_drawdown].value_counts()

    return durations.max() if not durations.empty else 0

def detect_regimes(returns, volatility_window=20, trend_window=20):
    """
    Detect 6 regimes: Low/Mid/High Vol * Trend/Range.
    Requires chronologically sorted returns.
    """
    # 1. Volatility Regimes (Tertiles)
    # Using realized volatility of returns
    vol = returns.rolling(volatility_window).std()

    low_thresh = vol.quantile(0.33)
    high_thresh = vol.quantile(0.66)

    vol_regime = pd.Series(index=returns.index, data='Mid')
    vol_regime[vol <= low_thresh] = 'Low'
    vol_regime[vol > high_thresh] = 'High'

    # 2. Trend Regimes (Efficiency Ratio)
    # ER = abs(sum(r)) / sum(abs(r))
    abs_change = returns.abs().rolling(trend_window).sum()
    net_change = returns.rolling(trend_window).sum().abs()

    er = net_change / abs_change.replace(0, np.inf)

    # Median split
    trend_thresh = er.median()
    trend_regime = pd.Series(index=returns.index, data='Range')
    trend_regime[er > trend_thresh] = 'Trend'

    return vol_regime + "_" + trend_regime

def run_evaluation(predictions_path, targets_path, output_dir, target_col, return_col, fold_col, volume_col, cost, pred_cols, timestamp_col):
    logger.info(f"Loading predictions from {predictions_path}")
    preds_df = pd.read_csv(predictions_path) if predictions_path.endswith('.csv') else pd.read_parquet(predictions_path)

    if targets_path:
        logger.info(f"Loading targets from {targets_path}")
        targets_df = pd.read_csv(targets_path) if targets_path.endswith('.csv') else pd.read_parquet(targets_path)
        if len(targets_df) == len(preds_df):
             if target_col not in preds_df.columns:
                 preds_df[target_col] = targets_df[target_col].values
             if return_col not in preds_df.columns:
                 preds_df[return_col] = targets_df[return_col].values
             if volume_col and volume_col not in preds_df.columns and volume_col in targets_df.columns:
                 preds_df[volume_col] = targets_df[volume_col].values
             if timestamp_col and timestamp_col not in preds_df.columns and timestamp_col in targets_df.columns:
                 preds_df[timestamp_col] = targets_df[timestamp_col].values
        else:
            logger.warning("Length mismatch between preds and targets. Using index merge.")
            preds_df = preds_df.merge(targets_df, left_index=True, right_index=True)

    # Sort data for time-series calculations
    if timestamp_col and timestamp_col in preds_df.columns:
        logger.info(f"Sorting data by timestamp column: {timestamp_col}")
        preds_df = preds_df.sort_values(timestamp_col)
    else:
        # Try to infer time column
        potential_time_cols = [c for c in preds_df.columns if 'time' in c.lower() or 'date' in c.lower()]
        if potential_time_cols:
            col = potential_time_cols[0]
            logger.info(f"Inferred timestamp column: {col}. Sorting data.")
            preds_df = preds_df.sort_values(col)
        elif fold_col in preds_df.columns:
             logger.info(f"Sorting by fold column '{fold_col}' as proxy for time.")
             preds_df = preds_df.sort_values(fold_col)
        else:
            logger.warning("No timestamp or fold column found. Assuming data is already chronological.")

    # Ensure output dir
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Fill defaults if missing
    if fold_col not in preds_df.columns:
        logger.info("No fold column found. Creating 5 chronological folds.")
        n = len(preds_df)
        preds_df['__fold__'] = pd.qcut(np.arange(n), 5, labels=False)
        fold_col = '__fold__'

    report_lines = []
    report_lines.append(f"# OOF Causal Evaluation Report")
    report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Predictions File: {predictions_path}")
    report_lines.append(f"\n")

    # Loop through prediction columns
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

        if len(df_clean) == 0:
            report_lines.append("No valid data points.")
            continue

        y_true = df_clean[target_col].values
        y_prob = df_clean[model_name].values
        y_ret = df_clean[return_col].values
        folds = df_clean[fold_col].values

        y_pred_bin = (y_prob > 0.5).astype(int)

        # --- 1. Basic Metrics ---
        prec = precision_score(y_true, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true, y_pred_bin, zero_division=0)
        pr_auc = average_precision_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        ll = log_loss(y_true, y_prob)

        ic, _ = stats.spearmanr(y_prob, y_ret)
        mae_binary = mean_absolute_error(y_true, y_prob)

        report_lines.append("### Standard Metrics")
        report_lines.append(f"| Metric | Value |")
        report_lines.append(f"|---|---|")
        report_lines.append(f"| Precision | {prec:.4f} |")
        report_lines.append(f"| F1-Score | {f1:.4f} |")
        report_lines.append(f"| PR-AUC | {pr_auc:.4f} |")
        report_lines.append(f"| Brier Score | {brier:.4f} |")
        report_lines.append(f"| Log-Loss | {ll:.4f} |")
        report_lines.append(f"| IC (Rank corr with Returns) | {ic:.4f} |")
        report_lines.append(f"| MAE (vs Binary Target) | {mae_binary:.4f} |")

        # --- 2. Causal / Uplift / Stability ---
        qini = qini_coefficient(y_ret, y_prob)

        unique_folds = np.sort(np.unique(folds))
        if len(unique_folds) > 1:
            fold_first = df_clean[df_clean[fold_col] == unique_folds[0]][model_name]
            fold_last = df_clean[df_clean[fold_col] == unique_folds[-1]][model_name]
            psi_val = population_stability_index(fold_first, fold_last)
            ks_stat, ks_p = stats.ks_2samp(fold_first, fold_last)
        else:
            psi_val = 0.0
            ks_stat, ks_p = 0.0, 1.0

        sign_switch_rate = np.mean(np.sign(y_prob - 0.5) == np.sign(y_ret))
        bias = np.mean(y_prob - y_true)
        entropy = prediction_entropy(y_prob)
        cce = causal_calibrated_error(y_true, y_prob)

        report_lines.append("\n### Causal & Stability Metrics")
        report_lines.append(f"| Metric | Value | Description |")
        report_lines.append(f"|---|---|---|")
        report_lines.append(f"| Qini Coefficient | {qini:.6f} | Cumulative lift vs random |")
        report_lines.append(f"| PSI (First vs Last Fold) | {psi_val:.4f} | Prediction Drift |")
        report_lines.append(f"| K-S Stat | {ks_stat:.4f} | Distribution difference (p={ks_p:.4f}) |")
        report_lines.append(f"| Sign Switch Rate | {sign_switch_rate:.4f} | Sign Accuracy vs Returns |")
        report_lines.append(f"| Prediction Bias | {bias:.4f} | Mean(Pred - True) |")
        report_lines.append(f"| Prediction Entropy | {entropy:.4f} | Certainty (Shannon) |")
        report_lines.append(f"| Causal Calibrated Error (CCE) | {cce:.4f} | Local calibration |")

        # --- 3. Financial Metrics ---
        signal = (y_prob > 0.5).astype(int)

        # Apply cost (round trip divided by 2 per side)
        strat_ret = signal * y_ret

        # Calculate positions and turnover
        pos = pd.Series(signal, index=df_clean.index)
        turnover = pos.diff().abs().fillna(0)

        costs = turnover * (cost / 2.0)
        net_strat_ret = strat_ret - costs

        cum_ret = net_strat_ret.cumsum()

        max_dd_duration = calculate_max_drawdown_duration(cum_ret)

        total_ret = net_strat_ret.sum()
        total_turnover = turnover.sum()
        ret_to_turnover = total_ret / total_turnover if total_turnover > 0 else 0.0

        # HHI of Monthly Returns
        df_clean['net_ret'] = net_strat_ret
        if timestamp_col and timestamp_col in df_clean.columns:
            try:
                ts = pd.to_datetime(df_clean[timestamp_col])
                monthly_ret = df_clean.groupby(ts.dt.to_period('M'))['net_ret'].sum()
            except:
                monthly_ret = df_clean.groupby(pd.qcut(np.arange(len(df_clean)), 12, duplicates='drop'))['net_ret'].sum()
        else:
            monthly_ret = df_clean.groupby(pd.qcut(np.arange(len(df_clean)), 12, duplicates='drop'))['net_ret'].sum()

        pos_months = monthly_ret[monthly_ret > 0]
        if pos_months.sum() > 0:
            shares = pos_months / pos_months.sum()
            hhi = (shares ** 2).sum()
        else:
            hhi = 0.0

        participation = signal.mean()

        capacity_str = "N/A (No Volume)"
        if volume_col and volume_col in df_clean.columns:
            avg_vol_usd = (df_clean[volume_col] * 1.0).mean()
            # Rough heuristic: Capacity = 1% of Avg Vol * (Sharpe / Threshold)
            capacity_est = avg_vol_usd * 0.01 * (ic * 10)
            capacity_str = f"${capacity_est:,.0f}"

        report_lines.append("\n### Financial Metrics (Simulated)")
        report_lines.append(f"| Metric | Value | Logic |")
        report_lines.append(f"|---|---|---|")
        report_lines.append(f"| Max Drawdown Duration | {max_dd_duration} bars | Longest time underwater |")
        report_lines.append(f"| Returns-to-Turnover | {ret_to_turnover:.4f} | Efficiency (Ret / Turnover) |")
        report_lines.append(f"| HHI of Returns | {hhi:.4f} | Concentration of profit (monthly) |")
        report_lines.append(f"| Participation Rate | {participation:.2%} | % Time in market |")
        report_lines.append(f"| Capacity Estimate | {capacity_str} | (Heuristic) |")

        # --- 4. Regime Analysis ---
        regimes = detect_regimes(pd.Series(y_ret, index=df_clean.index))
        df_clean['regime'] = regimes

        report_lines.append("\n### Regime-Specific Analysis")
        report_lines.append(f"| Regime | Count | Sharpe | C-PSR | Bias |")
        report_lines.append(f"|---|---|---|---|---|")

        best_perf = -np.inf
        worst_perf = np.inf

        for reg in sorted(df_clean['regime'].unique()):
            subset = df_clean[df_clean['regime'] == reg]
            if len(subset) < 10: continue

            sub_net_ret = net_strat_ret[subset.index]

            reg_sharpe = compute_sharpe_ratio(sub_net_ret)
            reg_bias = np.mean(subset[model_name] - subset[target_col])

            s_skew = stats.skew(sub_net_ret)
            s_kurt = stats.kurtosis(sub_net_ret)
            c_psr = prob_sharpe_ratio(reg_sharpe, s_skew, s_kurt, len(subset))

            report_lines.append(f"| {reg} | {len(subset)} | {reg_sharpe:.4f} | {c_psr:.4f} | {reg_bias:.4f} |")

            best_perf = max(best_perf, reg_sharpe)
            worst_perf = min(worst_perf, reg_sharpe)

        tr_ratio = worst_perf / best_perf if (best_perf != 0 and abs(best_perf) > 1e-9) else 0.0

        report_lines.append(f"\n**Invariance Test (TR Ratio):** {tr_ratio:.4f} (Worst Sharpe / Best Sharpe)")

        regime_switch = (df_clean['regime'] != df_clean['regime'].shift()) & (df_clean['regime'].shift().notna())
        switch_indices = df_clean[regime_switch].index

        if len(switch_indices) > 0:
            switch_probs = y_prob[regime_switch]
            switch_entropy = prediction_entropy(switch_probs)
            report_lines.append(f"**Regime Transition Entropy:** {switch_entropy:.4f} (Uncertainty at regime shifts)")
        else:
            report_lines.append("**Regime Transition Entropy:** N/A (No switches)")

        report_lines.append("\n---\n")

    fname = f"causal_reliability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    out_file = out_path / fname
    with open(out_file, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"Report generated: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OOF Causal Evaluation")
    parser.add_argument("--predictions", required=True, help="Path to predictions file")
    parser.add_argument("--targets", help="Path to targets file (optional if in preds)")
    parser.add_argument("--output", default="outcomes/", help="Output directory")
    parser.add_argument("--target-col", default="target", help="Binary target column")
    parser.add_argument("--return-col", default="returns", help="Continuous return column")
    parser.add_argument("--fold-col", default="fold", help="Fold column")
    parser.add_argument("--volume-col", help="Volume column for capacity")
    parser.add_argument("--timestamp-col", help="Timestamp column for sorting")
    parser.add_argument("--cost", type=float, default=0.003, help="Transaction cost")
    parser.add_argument("--pred-cols", nargs="+", help="Specific prediction columns to evaluate")

    args = parser.parse_args()

    run_evaluation(
        args.predictions, args.targets, args.output,
        args.target_col, args.return_col, args.fold_col,
        args.volume_col, args.cost, args.pred_cols, args.timestamp_col
    )
