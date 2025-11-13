#!/usr/bin/env python3
"""
Analyst Signal Diagnostics

Usage:
  python scripts/analyst_signal_diagnostics.py \
    --data-path path/to/data.csv \
    --target-col target \
    --time-col datetime \
    --output-dir analysis_output/diagnostics \
    --max-samples 20000

Notes:
- Expects a tabular dataset with numeric features and a regression target.
- If time-col is provided, learning curves and CV will respect temporal ordering.
- Generates plots and a JSON summary with stability metrics.
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def tolerance_accuracy(y_true, y_pred, rel=0.1, min_abs=0.01):
    tol = np.maximum(min_abs, rel * np.std(y_true))
    return float(np.mean(np.abs(y_true - y_pred) <= tol))


def load_data(path: str, target_col: str, time_col: str | None, max_samples: int | None):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data path not found: {path}")
    if p.suffix.lower() in {'.csv'}:
        df = pd.read_csv(p)
    elif p.suffix.lower() in {'.parquet'}:
        df = pd.read_parquet(p)
    elif p.suffix.lower() in {'.feather'}:
        df = pd.read_feather(p)
    else:
        # Try CSV by default
        df = pd.read_csv(p)
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)
    if max_samples and len(df) > max_samples:
        df = df.tail(max_samples)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in data")
    y = df[target_col].astype(float).values
    X = df.drop(columns=[target_col])
    # Keep only numeric
    X = X.select_dtypes(include=[np.number]).copy()
    cols = X.columns.tolist()
    return X.values, y, cols, df[time_col] if time_col and time_col in df.columns else None


def learning_curve_over_time(X, y, n_points=8):
    n = len(y)
    sizes = np.linspace(0.2, 1.0, n_points)
    train_scores, test_scores = [], []
    model = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=64, random_state=42, n_jobs=-1)
    for frac in sizes:
        end = max(int(n * frac), 50)
        X_tr, y_tr = X[:end], y[:end]
        # split last 15% as test
        split = max(int(end * 0.85), end - 1)
        X_train, y_train = X_tr[:split], y_tr[:split]
        X_test, y_test = X_tr[split:], y_tr[split:]
        if len(X_test) < 5:
            train_scores.append(np.nan)
            test_scores.append(np.nan)
            continue
        model.fit(X_train, y_train)
        y_tr_hat = model.predict(X_train)
        y_te_hat = model.predict(X_test)
        train_scores.append(r2_score(y_train, y_tr_hat))
        test_scores.append(r2_score(y_test, y_te_hat))
    return sizes, np.array(train_scores), np.array(test_scores)


def temporal_cv_stability(X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    model = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=64, random_state=42, n_jobs=-1)
    for tr_idx, te_idx in tscv.split(X):
        if len(te_idx) < 5:
            continue
        model.fit(X[tr_idx], y[tr_idx])
        pred = model.predict(X[te_idx])
        scores.append(r2_score(y[te_idx], pred))
    return np.array(scores)


def permutation_importance_stability(X, y, feature_names, n_repeats=5, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    importances = []
    model = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=64, random_state=42, n_jobs=-1)
    for tr_idx, te_idx in tscv.split(X):
        model.fit(X[tr_idx], y[tr_idx])
        result = permutation_importance(model, X[te_idx], y[te_idx], n_repeats=n_repeats, random_state=42, n_jobs=-1)
        importances.append(result.importances_mean)
    if not importances:
        return feature_names, np.array([]), np.array([])
    imp_arr = np.vstack(importances)
    mean_imp = imp_arr.mean(axis=0)
    std_imp = imp_arr.std(axis=0)
    order = np.argsort(-mean_imp)
    return [feature_names[i] for i in order], mean_imp[order], std_imp[order]


def adversarial_validation(X, time_index=None):
    n = len(X)
    y_adv = np.zeros(n)
    y_adv[n//2:] = 1  # early vs late periods
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=200, n_jobs=-1)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr, te in kf.split(Xs):
        clf.fit(Xs[tr], y_adv[tr])
        prob = clf.predict_proba(Xs[te])[:,1]
        # use simple ROC proxy via R2 vs binary labels (bounded insight)
        scores.append(r2_score(y_adv[te], prob))
    return float(np.mean(scores)), float(np.std(scores))


def corr_spectrum_over_time(X, y, feature_names, windows=5):
    n = len(y)
    edges = np.linspace(0, n, windows+1, dtype=int)
    corr_stats = []
    for i in range(windows):
        a, b = edges[i], edges[i+1]
        if b - a < 20:
            continue
        yi = y[a:b]
        Xi = X[a:b]
        cors = []
        for j in range(min(Xi.shape[1], 200)):
            xj = Xi[:, j]
            if np.std(xj) < 1e-9:
                cors.append(0.0)
            else:
                cors.append(np.corrcoef(xj, yi)[0,1])
        cors = np.nan_to_num(np.array(cors))
        corr_stats.append({
            'window': i+1,
            'abs_mean_corr': float(np.mean(np.abs(cors))),
            'max_abs_corr': float(np.max(np.abs(cors)))
        })
    return corr_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-path', required=True)
    ap.add_argument('--target-col', required=True)
    ap.add_argument('--time-col', default=None)
    ap.add_argument('--output-dir', default='analysis_output/diagnostics')
    ap.add_argument('--max-samples', type=int, default=20000)
    args = ap.parse_args()

    X, y, feat_names, time_index = load_data(args.data_path, args.target_col, args.time_col, args.max_samples)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(args.output_dir) / f'diagnostics_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Learning curves
    sizes, tr_scores, te_scores = learning_curve_over_time(X, y)
    plt.figure(figsize=(8,4))
    plt.plot(sizes, tr_scores, label='Train R2')
    plt.plot(sizes, te_scores, label='Test R2')
    plt.xlabel('Fraction of data used (chronological)')
    plt.ylabel('R2')
    plt.title('Learning Curve (Chronological)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'learning_curve.png', dpi=140)
    plt.close()

    # Temporal CV stability
    cv_scores = temporal_cv_stability(X, y, n_splits=5)
    plt.figure(figsize=(6,4))
    plt.bar(range(1, len(cv_scores)+1), cv_scores)
    plt.xlabel('Fold')
    plt.ylabel('R2')
    plt.title('Temporal CV Fold R2')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'temporal_cv_r2.png', dpi=140)
    plt.close()

    # Permutation importance stability
    names_ordered, imp_mean, imp_std = permutation_importance_stability(X, y, feat_names, n_repeats=3, n_splits=3)
    top_k = min(20, len(names_ordered))
    if top_k > 0:
        plt.figure(figsize=(8,6))
        plt.barh(names_ordered[:top_k][::-1], imp_mean[:top_k][::-1], xerr=imp_std[:top_k][::-1], alpha=0.8)
        plt.xlabel('Mean Permutation Importance (CV)')
        plt.title('Top Feature Importance (Permutation, Stability)')
        plt.tight_layout()
        plt.savefig(out_dir / 'permutation_importance_stability.png', dpi=140)
        plt.close()

    # Adversarial validation
    adv_mean, adv_std = adversarial_validation(X)

    # Correlation spectrum over time
    corr_stats = corr_spectrum_over_time(X, y, feat_names, windows=5)

    # Summary JSON
    summary = {
        'timestamp': ts,
        'n_samples': int(len(y)),
        'n_features': int(X.shape[1]),
        'learning_curve': {
            'fractions': sizes.tolist(),
            'train_r2': np.nan_to_num(tr_scores.astype(float), nan=0.0).tolist(),
            'test_r2': np.nan_to_num(te_scores.astype(float), nan=0.0).tolist()
        },
        'temporal_cv': {
            'fold_r2': np.nan_to_num(cv_scores.astype(float), nan=0.0).tolist(),
            'mean_r2': float(np.nanmean(cv_scores)) if len(cv_scores) else None,
            'std_r2': float(np.nanstd(cv_scores)) if len(cv_scores) else None
        },
        'permutation_importance': {
            'feature_names_ordered': names_ordered[:top_k] if top_k else [],
            'mean_importance': imp_mean[:top_k].tolist() if top_k else [],
            'std_importance': imp_std[:top_k].tolist() if top_k else []
        },
        'adversarial_validation': {
            'mean_r2_proxy': adv_mean,
            'std_r2_proxy': adv_std
        },
        'corr_spectrum': corr_stats,
        'notes': [
            'High adversarial separability suggests distribution shift (train vs test).',
            'Low/unstable temporal CV R2 suggests weak or unstable signal.',
            'Learning curve plateau near 0 indicates limited predictive signal or mis-specification.'
        ]
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Diagnostics saved to: {out_dir}")


if __name__ == '__main__':
    main()
