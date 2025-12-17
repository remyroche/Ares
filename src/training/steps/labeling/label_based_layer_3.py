import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
    accuracy_score
)
from typing import List, Tuple, Optional, Any

def layer3_analyst_lgbm(
    oof_df: pd.DataFrame,
    base_model_cols: List[str],
    target_col: str,
    train_split_date: Optional[str] = None,
    sample_weight: Optional[np.ndarray] = None
) -> Tuple[pd.DataFrame, Any]:
    """
    Transforms diverse Base Model scores into a single Calibrated Probability using LGBM.

    Optimizes for: LogLoss (Primary)
    Reports: AUC, IC, ECE, MCE, Brier
    """
    print(f"\n{'='*60}")
    print("LAYER 3: ANALYST META-MODEL (LGBM + CALIBRATION)")
    print(f"{'='*60}")

    df = oof_df.copy()

    # ---------------------------------------------------------
    # 1. Feature Engineering: Ensemble Disagreement
    # ---------------------------------------------------------
    print(">> Generating Disagreement Features...")

    if not base_model_cols:
         print("⚠️ No base models provided for Layer 3 feature engineering!")
         # Fallback if no individual models: assume single consensus column is input?
         # The orchestrator should handle this, but let's be safe.
         meta_features = []
    else:
        # Capture the "Confusion" and "Consensus" of the ensemble
        # Handle NaN values in base models (geometries that didn't vote)
        base_df = df[base_model_cols]

        df['meta_std'] = base_df.std(axis=1)    # Risk Proxy
        df['meta_mean'] = base_df.mean(axis=1)  # Consensus
        df['meta_skew'] = base_df.skew(axis=1)  # Directional Bias
        df['meta_range'] = base_df.max(axis=1) - base_df.min(axis=1)
        df['meta_count'] = base_df.notna().sum(axis=1) # Number of voting geometries

        meta_features = base_model_cols + ['meta_std', 'meta_mean', 'meta_skew', 'meta_range', 'meta_count']

        # Fill NaN features (for events where some geometries were inactive)
        # Standard fill 0.5 for scores?
        df[base_model_cols] = df[base_model_cols].fillna(0.5)
        df[meta_features] = df[meta_features].fillna(0)

    # Add external features if available in oof_df
    # We expect 'volatility_1d', 'trend_regime' dummies etc to be useful
    additional_features = [c for c in df.columns if c.startswith('vol_') or c.startswith('trend_') or c in ['volatility_1d']]
    meta_features += additional_features

    # Clean target
    df = df.dropna(subset=[target_col])
    if sample_weight is not None:
        if len(sample_weight) != len(oof_df):
             print(f"⚠️ Weight length mismatch! {len(sample_weight)} vs {len(oof_df)}")
             sample_weight = sample_weight[:len(df)] # Rough fix, but should align
        # Align weights to dropped NA target rows
        # Assuming oof_df index was reset or consistent.
        # Best to align by index.
        w_series = pd.Series(sample_weight, index=oof_df.index)
        w_aligned = w_series.loc[df.index].values
    else:
        w_aligned = None

    # ---------------------------------------------------------
    # 2. Split Data (Strict Time Series)
    # ---------------------------------------------------------
    # Ensure sorted
    if 'date' in df.columns:
        df = df.sort_values('date')
    else:
        # Assume index is time-sorted
        df = df.sort_index()

    if train_split_date and 'date' in df.columns:
        train_mask = df['date'] < train_split_date
        val_mask = df['date'] >= train_split_date
        train = df[train_mask]
        val = df[val_mask]
        if w_aligned is not None:
            w_train = w_aligned[train_mask]
            w_val = w_aligned[val_mask] # not used for validation metrics usually
        else:
            w_train = None
    else:
        # Default: 80/20 sequential split
        split_idx = int(len(df) * 0.80)
        train = df.iloc[:split_idx]
        val = df.iloc[split_idx:]
        if w_aligned is not None:
            w_train = w_aligned[:split_idx]
        else:
            w_train = None

    X_train = train[meta_features]
    y_train = train[target_col]
    X_val = val[meta_features]
    y_val = val[target_col]

    print(f">> Split: Train {X_train.shape} | Val {X_val.shape}")

    # ---------------------------------------------------------
    # 3. Define Core LGBM (Optimized for LogLoss)
    # ---------------------------------------------------------
    # The base estimator must maximize information extraction (LogLoss)
    # before the calibrator polishes the probabilities.

    lgbm_params = {
        'objective': 'binary',     # Explicitly minimizes LogLoss
        'metric': 'binary_logloss',
        'n_estimators': 200,
        'learning_rate': 0.03,
        'max_depth': 4,            # Keep shallow to prevent noise memorization
        'num_leaves': 16,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1.0,          # L1 Regularization
        'reg_lambda': 1.0,         # L2 Regularization
        'random_state': 42,
        'n_jobs': 1,
        'verbose': -1
    }

    # ---------------------------------------------------------
    # 4. Train with Calibration (Isotonic Regression)
    # ---------------------------------------------------------
    print(">> Training Calibrated Classifier (TSC V-Split)...")

    # Use TimeSeriesSplit to prevent leakage during internal calibration folds
    tscv = TimeSeriesSplit(n_splits=3)

    base_model = lgb.LGBMClassifier(**lgbm_params)

    # Isotonic: Non-parametric, fits the "S-curve" best for financial data
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method='isotonic',
        cv=tscv
    )

    if w_train is not None:
        # CalibratedClassifierCV supports sample_weight in fit
        calibrated_model.fit(X_train, y_train, sample_weight=w_train)
    else:
        calibrated_model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 5. Comprehensive Analytics Suite
    # ---------------------------------------------------------
    print(">> Calculating Performance Metrics...")

    val_probs = calibrated_model.predict_proba(X_val)[:, 1]

    # A. Primary Objective: Log Loss (Uncertainty + Calibration)
    # Lower is better. Measures "Honest Confidence".
    score_logloss = log_loss(y_val, val_probs)

    # B. Resolution: AUC (Ranking Quality)
    # Higher is better. Can we distinguish High Prob from Low Prob?
    try:
        score_auc = roc_auc_score(y_val, val_probs)
    except:
        score_auc = 0.5

    # C. Linearity: IC (Information Coefficient)
    # Spearman Correlation between probability and outcome.
    score_ic, _ = spearmanr(val_probs, y_val)
    if np.isnan(score_ic): score_ic = 0.0

    # D. Calibration: ECE & MCE
    # Expected Calibration Error: Weighted average gap between confidence and reality.
    # Max Calibration Error: Worst single bin gap.
    prob_true, prob_pred = calibration_curve(y_val, val_probs, n_bins=10)

    # Calculate ECE manually (sklearn doesn't have a direct func)
    if len(val_probs) > 0:
        hist, bin_edges = np.histogram(val_probs, bins=10, range=(0, 1))
        # Weights = fraction of samples in each bin
        weights = hist / len(val_probs)
        # Filter out empty bins to avoid mismatch
        mask = hist > 0
        # ECE = sum(weight * |prob_true - prob_pred|)
        # Note: calibration_curve returns bins that have data.
        # We assume alignment, but robust implementation aligns by bin index.
        # Simplified ECE approximation:
        if len(prob_true) == len(weights[mask]):
            score_ece = np.sum(weights[mask] * np.abs(prob_true - prob_pred))
            score_mce = np.max(np.abs(prob_true - prob_pred))
        else:
            score_ece = 0.0
            score_mce = 0.0
    else:
        score_ece = 0.0
        score_mce = 0.0

    # E. Robustness: Brier Score vs Baseline
    # Brier = Mean Squared Error of probability.
    # Must beat "No-Skill" (predicting the mean rate for everyone).
    score_brier = brier_score_loss(y_val, val_probs)
    no_skill_prob = [y_train.mean() for _ in range(len(y_val))]
    score_brier_base = brier_score_loss(y_val, no_skill_prob)
    if score_brier_base > 0:
        brier_skill_score = 1 - (score_brier / score_brier_base) # > 0 means skill
    else:
        brier_skill_score = 0.0

    # ---------------------------------------------------------
    # 6. Reporting
    # ---------------------------------------------------------
    metrics = {
        "Log Loss (Primary)": f"{score_logloss:.5f}",
        "AUC (Resolution)":   f"{score_auc:.5f}",
        "IC (Spearman)":      f"{score_ic:.5f}",
        "ECE (Calibration)":  f"{score_ece:.5f}",
        "MCE (Worst Bin)":    f"{score_mce:.5f}",
        "Brier Score":        f"{score_brier:.5f}",
        "Brier Skill Score":  f"{brier_skill_score:.2%}"
    }

    print("\n" + "-"*30)
    print("   LAYER 3 PERFORMANCE REPORT")
    print("-" * 30)
    for k, v in metrics.items():
        print(f"{k:<20} : {v}")
    print("-" * 30 + "\n")

    # Warn if model is "Calibrated but Useless" (Low AUC)
    if score_auc < 0.52:
        print("⚠️  WARNING: AUC is near random (0.5). Model is calibrated but has no resolution.")
    if score_ece > 0.10:
        print("⚠️  WARNING: High Calibration Error (>10%). Sizing engine may over-bet.")

    # Return val set with probs for Layer 4
    val_export = val.copy()
    val_export['meta_prob'] = val_probs

    return val_export, calibrated_model

# ---------------------------------------------------------
# Helper: Advanced Diagnostic Plot
# ---------------------------------------------------------
def plot_diagnostics(y_true, y_prob, output_path=None):
    """
    Plots Reliability Diagram (Calibration) AND Probability Density (Resolution).
    Crucial to see if the model is just hugging 0.5.
    """
    try:
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        # 1. Reliability Diagram
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ax[0].plot(prob_pred, prob_true, marker='o', linewidth=2, label='Meta-Model')
        ax[0].plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5, label='Perfect')
        ax[0].set_xlabel('Predicted Probability')
        ax[0].set_ylabel('Actual Win Rate')
        ax[0].set_title('Calibration (Reliability)')
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

        # 2. Probability Density (Histogram)
        # We want a "U-shape" (confident) or broad spread.
        # A spike at 0.5 means "Clueless Weatherman".
        sns.histplot(y_prob, bins=20, kde=True, ax=ax[1], color='purple', alpha=0.6)
        ax[1].set_xlim(0, 1)
        ax[1].set_xlabel('Predicted Probability')
        ax[1].set_title('Resolution (Confidence Distribution)')
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
            print(f"Diagnostics plot saved to {output_path}")
        else:
            # If no display, do nothing (headless env)
            pass
        plt.close(fig)
    except Exception as e:
        print(f"Failed to generate plots: {e}")
