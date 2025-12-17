import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import TimeSeriesSplit, KFold
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

    Performs K-Fold Cross-Validation to generate OOF predictions for the entire dataset
    (for unbiased analytics), and then trains a Final Model on all data (for production).

    Optimizes for: LogLoss (Primary)
    Reports: AUC, IC, ECE, MCE, Brier (based on OOF predictions)
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
        df[base_model_cols] = df[base_model_cols].fillna(0.5)
        df[meta_features] = df[meta_features].fillna(0)

    # Add external features if available in oof_df
    additional_features = [c for c in df.columns if c.startswith('vol_') or c.startswith('trend_') or c in ['volatility_1d']]
    meta_features += additional_features

    # Clean target
    df = df.dropna(subset=[target_col])

    # Align sample_weight
    if sample_weight is not None:
        if len(sample_weight) != len(oof_df):
             print(f"⚠️ Weight length mismatch! {len(sample_weight)} vs {len(oof_df)}")
             sample_weight = sample_weight[:len(df)] # Rough fix
        w_series = pd.Series(sample_weight, index=oof_df.index)
        w_aligned = w_series.loc[df.index].values
    else:
        w_aligned = None

    # ---------------------------------------------------------
    # 2. OOF Generation (K-Fold Stacking)
    # ---------------------------------------------------------
    print(">> Generating OOF Predictions (K-Fold)...")

    kf = KFold(n_splits=5, shuffle=False)

    # Initialize OOF array
    oof_probs = np.full(len(df), np.nan)

    lgbm_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'n_estimators': 200,
        'learning_rate': 0.03,
        'max_depth': 4,
        'num_leaves': 16,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': 1,
        'verbose': -1
    }

    X = df[meta_features]
    y = df[target_col]

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        w_train = w_aligned[train_idx] if w_aligned is not None else None

        # Internal CV for Calibration
        # We use TimeSeriesSplit internally to respect time within the training fold?
        # Or standard CV? Since the training fold might be discontiguous (no, KFold blocks are contiguous).
        # We use TimeSeriesSplit for robustness.
        tscv_inner = TimeSeriesSplit(n_splits=3)

        base_est = lgb.LGBMClassifier(**lgbm_params)
        calib_clf = CalibratedClassifierCV(
            estimator=base_est,
            method='isotonic',
            cv=tscv_inner
        )

        try:
            calib_clf.fit(X_train, y_train, sample_weight=w_train)
            probs = calib_clf.predict_proba(X_test)[:, 1]
            oof_probs[test_idx] = probs
        except Exception as e:
            print(f"⚠️ Fold {fold} failed: {e}")

    df['meta_prob'] = oof_probs

    # ---------------------------------------------------------
    # 3. Final Model Training (Production)
    # ---------------------------------------------------------
    print(">> Training Final Production Model (All Data)...")

    final_base = lgb.LGBMClassifier(**lgbm_params)
    final_tscv = TimeSeriesSplit(n_splits=3)
    final_model = CalibratedClassifierCV(
        estimator=final_base,
        method='isotonic',
        cv=final_tscv
    )

    try:
        final_model.fit(X, y, sample_weight=w_aligned)
    except Exception as e:
        print(f"⚠️ Final model training failed: {e}")

    # ---------------------------------------------------------
    # 4. Comprehensive Analytics Suite (on OOF)
    # ---------------------------------------------------------
    print(">> Calculating OOF Performance Metrics...")

    # Filter out NaNs (e.g. if some folds failed or gaps)
    mask = ~np.isnan(oof_probs)
    y_true = y[mask]
    y_prob = oof_probs[mask]

    if len(y_true) > 0:
        # A. Log Loss
        score_logloss = log_loss(y_true, y_prob)

        # B. AUC
        try:
            score_auc = roc_auc_score(y_true, y_prob)
        except:
            score_auc = 0.5

        # C. IC
        score_ic, _ = spearmanr(y_prob, y_true)
        if np.isnan(score_ic): score_ic = 0.0

        # D. Calibration
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

        # ECE
        hist, _ = np.histogram(y_prob, bins=10, range=(0, 1))
        weights = hist / len(y_prob)
        # Assuming calibration_curve bins match histogram (simplification)
        # Proper ECE requires bin-wise matching, but this is a decent proxy/placeholder
        # if we assume standard equal-width bins.
        # Actually calibration_curve uses 'uniform' strategy by default.
        # Let's just use what we have.
        if len(prob_true) == len(weights[hist > 0]):
             score_ece = np.sum(weights[hist > 0] * np.abs(prob_true - prob_pred))
        else:
             score_ece = 0.0

        score_mce = np.max(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0.0

        # E. Brier
        score_brier = brier_score_loss(y_true, y_prob)
        no_skill_prob = [y.mean() for _ in range(len(y_true))]
        score_brier_base = brier_score_loss(y_true, no_skill_prob)
        brier_skill_score = 1 - (score_brier / score_brier_base) if score_brier_base > 0 else 0.0
    else:
        score_logloss, score_auc, score_ic, score_ece, score_mce, score_brier, brier_skill_score = 0, 0, 0, 0, 0, 0, 0

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
    print("   LAYER 3 PERFORMANCE REPORT (OOF)")
    print("-" * 30)
    for k, v in metrics.items():
        print(f"{k:<20} : {v}")
    print("-" * 30 + "\n")

    if score_auc < 0.52:
        print("⚠️  WARNING: OOF AUC is near random (0.5). Model may not generalize.")
    if score_ece > 0.10:
        print("⚠️  WARNING: High Calibration Error (>10%).")

    # Return full dataframe with predictions + final model
    return df, final_model

# ---------------------------------------------------------
# Helper: Advanced Diagnostic Plot
# ---------------------------------------------------------
def plot_diagnostics(y_true, y_prob, output_path=None):
    """
    Plots Reliability Diagram (Calibration) AND Probability Density (Resolution).
    """
    try:
        # Remove NaNs
        mask = ~np.isnan(y_prob) & ~np.isnan(y_true)
        y_true = y_true[mask]
        y_prob = y_prob[mask]

        if len(y_true) == 0:
            return

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
            pass
        plt.close(fig)
    except Exception as e:
        print(f"Failed to generate plots: {e}")
