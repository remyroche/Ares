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
from typing import List, Tuple, Optional, Any, Dict
import copy

def layer3_analyst_lgbm(
    oof_df: pd.DataFrame,
    base_model_cols: List[str],
    target_col: str,
    train_split_date: Optional[str] = None,
    sample_weight: Optional[np.ndarray] = None,
    # New arguments for Scheme comparison
    layer1_weight: Optional[np.ndarray] = None,
    layer2_weight: Optional[np.ndarray] = None,
    net_returns: Optional[np.ndarray] = None
) -> Tuple[pd.DataFrame, Any]:
    """
    Transforms diverse Base Model scores into a single Calibrated Probability using LGBM.

    Performs a comparison of 7 specified weighting schemes using ScoreL3 logic:
      ScoreL3 = 100*(AUC-0.5) + 50*(0.693-LogLoss) - 200*ECE

    Selects the best scheme and trains the final production model.
    """
    print(f"\n{'='*60}")
    print("LAYER 3: ANALYST META-MODEL (LGBM + CALIBRATION) - COMPARATIVE MODE")
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

    # Align Weights (and handle length mismatches robustly)
    valid_idx = df.index

    def align_vector(vec, name):
        if vec is None: return None
        if len(vec) != len(oof_df):
            # Try to match if it's just truncation (e.g. from nan removal)
            if len(vec) >= len(df):
                print(f"⚠️ {name} length mismatch ({len(vec)} vs {len(df)}). Attempting to align via index if Series.")
                if isinstance(vec, pd.Series):
                    return vec.reindex(valid_idx).fillna(0).values
                else:
                    print(f"   Cannot align numpy array safely. Using first {len(df)} elements.")
                    return vec[:len(df)]
            else:
                 print(f"⚠️ {name} too short ({len(vec)} vs {len(df)}). Padding with 1s.")
                 padded = np.ones(len(df))
                 padded[:len(vec)] = vec
                 return padded
        # If lengths match exactly with oof_df, we just filter to valid_idx
        if len(vec) == len(oof_df):
             if isinstance(vec, pd.Series):
                  return vec.loc[valid_idx].values
             else:
                  # Assuming vec corresponds to oof_df rows one-to-one
                  # We need to filter it the same way df was filtered from oof_df
                  # df is oof_df.dropna(subset=[target_col])
                  # So we need the boolean mask
                  mask = oof_df[target_col].notna().values
                  return vec[mask]
        return vec

    # If legacy sample_weight is passed but new components are missing, treat sample_weight as Scheme 3 or 1 (fallback)
    # But ideally, caller provides components.

    w_l1 = align_vector(layer1_weight, "Layer1 Weight")
    w_l2 = align_vector(layer2_weight, "Layer2 Weight")
    ret_vec = align_vector(net_returns, "Net Returns")

    # Fallback/Defaults if missing
    if w_l1 is None: w_l1 = np.ones(len(df))
    if w_l2 is None: w_l2 = np.ones(len(df))
    if ret_vec is None: ret_vec = np.zeros(len(df))

    # Calculate Magnitude Factor: log(1 + |NetReturns|)
    # Use abs() to handle negative returns safely for magnitude importance
    magnitude_log = np.log1p(np.abs(ret_vec))

    # ---------------------------------------------------------
    # 2. Define Weighting Schemes
    # ---------------------------------------------------------
    schemes = {}

    # Scheme 1: target_sample_weight (layer1)
    schemes["S1_L1"] = w_l1

    # Scheme 2: target_sample_weight * final composite weight (layer2)
    schemes["S2_L1_L2"] = w_l1 * w_l2

    # Scheme 3: final composite weight (layer2)
    schemes["S3_L2"] = w_l2

    # Scheme 4: log(1+NetReturns) for magnitude integration
    schemes["S4_Mag"] = magnitude_log

    # Scheme 5: target_sample_weight * log(1+NetReturns)
    schemes["S5_L1_Mag"] = w_l1 * magnitude_log

    # Scheme 6: final composite weight * log(1+NetReturns)
    schemes["S6_L2_Mag"] = w_l2 * magnitude_log

    # Scheme 7: target_sample_weight * final composite weight * log(1+NetReturns)
    schemes["S7_All"] = w_l1 * w_l2 * magnitude_log

    # Normalize all weights to mean=1.0 for stability
    for k in schemes:
        if schemes[k].mean() > 1e-9:
            schemes[k] = schemes[k] / schemes[k].mean()
        else:
            schemes[k] = np.ones_like(schemes[k]) # Fallback for zero weights

    # ---------------------------------------------------------
    # 3. Comparative Evaluation (K-Fold OOF)
    # ---------------------------------------------------------
    print("\n>> Comparing 7 Weighting Schemes...")

    results = []

    best_score = -float('inf')
    best_scheme_name = None
    best_model_artifacts = None # To store OOF preds and Final Model

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

    kf = KFold(n_splits=5, shuffle=False)

    for name, w_vec in schemes.items():
        print(f"   Evaluating {name}...")

        # A. OOF Generation
        oof_probs = np.full(len(df), np.nan)

        try:
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                w_train = w_vec[train_idx]

                # Internal Calibration
                tscv_inner = TimeSeriesSplit(n_splits=3)
                base_est = lgb.LGBMClassifier(**lgbm_params)
                calib_clf = CalibratedClassifierCV(
                    estimator=base_est,
                    method='isotonic',
                    cv=tscv_inner
                )

                calib_clf.fit(X_train, y_train, sample_weight=w_train)
                probs = calib_clf.predict_proba(X_test)[:, 1]
                oof_probs[test_idx] = probs

            # B. Metrics Calculation
            mask = ~np.isnan(oof_probs)
            y_true_eval = y[mask]
            y_prob_eval = oof_probs[mask]

            if len(y_true_eval) == 0:
                raise ValueError("No valid predictions generated.")

            auc = roc_auc_score(y_true_eval, y_prob_eval)
            ll = log_loss(y_true_eval, y_prob_eval)

            # ECE
            prob_true, prob_pred = calibration_curve(y_true_eval, y_prob_eval, n_bins=10)
            hist, _ = np.histogram(y_prob_eval, bins=10, range=(0, 1))
            weights = hist / len(y_prob_eval)
            # Match bins roughly
            if len(prob_true) > 0:
                 # Note: calibration_curve bins might not align perfectly with histogram if empty bins exist
                 # Standard calculation: sum(w_i * |p_i - o_i|)
                 # We'll rely on the fact that calibration_curve returns points for populated bins
                 # But we need the count for *those* bins.
                 # Let's re-calculate manually for correctness
                 ece_sum = 0
                 bin_edges = np.linspace(0, 1, 11)
                 for i in range(10):
                     bin_mask = (y_prob_eval >= bin_edges[i]) & (y_prob_eval < bin_edges[i+1])
                     if i == 9: bin_mask = (y_prob_eval >= bin_edges[i]) & (y_prob_eval <= bin_edges[i+1])

                     if np.sum(bin_mask) > 0:
                         p_mean = np.mean(y_prob_eval[bin_mask])
                         o_mean = np.mean(y_true_eval[bin_mask])
                         ece_sum += (np.sum(bin_mask) / len(y_prob_eval)) * np.abs(p_mean - o_mean)
                 ece = ece_sum
            else:
                 ece = 0.0

            # C. ScoreL3
            # ScoreL3 = 100*(AUC-0.5) + 50*(0.693-LogLoss) - 200*ECE
            score = 100 * (auc - 0.5) + 50 * (0.693 - ll) - 200 * ece

            # Interpretability Rating
            if score < 0: rating = "Toxic"
            elif score < 0.15: rating = "Weak"
            elif score < 0.3: rating = "Good"
            else: rating = "Excellent"

            results.append({
                "Scheme": name,
                "Score": score,
                "AUC": auc,
                "LogLoss": ll,
                "ECE": ece,
                "Rating": rating
            })

            # Check for best
            if score > best_score:
                best_score = score
                best_scheme_name = name
                # Store artifacts for the best model so we don't have to re-train
                best_model_artifacts = {
                    "oof_probs": oof_probs,
                    "w_vec": w_vec
                }

        except Exception as e:
            print(f"⚠️ Scheme {name} failed: {e}")
            results.append({
                "Scheme": name,
                "Score": -999,
                "AUC": 0, "LogLoss": 99, "ECE": 99, "Rating": "Failed"
            })

    # ---------------------------------------------------------
    # 4. Reporting & Selection
    # ---------------------------------------------------------
    results_df = pd.DataFrame(results).sort_values("Score", ascending=False)

    print("\n" + "="*60)
    print("   LAYER 3 WEIGHTING SCHEME COMPARISON")
    print("="*60)
    print(f"{'Scheme':<15} | {'Score':<8} | {'AUC':<6} | {'LogLoss':<8} | {'ECE':<6} | {'Rating'}")
    print("-" * 75)
    for _, row in results_df.iterrows():
        print(f"{row['Scheme']:<15} | {row['Score']:>8.4f} | {row['AUC']:>6.4f} | {row['LogLoss']:>8.4f} | {row['ECE']:>6.4f} | {row['Rating']}")
    print("-" * 75)

    print(f"\n🏆 WINNER: {best_scheme_name} (Score: {best_score:.4f})")

    if best_model_artifacts is None:
        print("❌ Critical Failure: No schemes succeeded.")
        # Fallback to simple unweighted
        return df, None

    # ---------------------------------------------------------
    # 5. Final Model Training (Production) using WINNER
    # ---------------------------------------------------------
    print(f">> Training Final Production Model using {best_scheme_name}...")

    df['meta_prob'] = best_model_artifacts['oof_probs']
    w_best = best_model_artifacts['w_vec']

    final_base = lgb.LGBMClassifier(**lgbm_params)
    final_tscv = TimeSeriesSplit(n_splits=3)
    final_model = CalibratedClassifierCV(
        estimator=final_base,
        method='isotonic',
        cv=final_tscv
    )

    try:
        final_model.fit(X, y, sample_weight=w_best)
    except Exception as e:
        print(f"⚠️ Final model training failed: {e}")
        final_model = None

    # ---------------------------------------------------------
    # 6. Final Diagnostics (on Best OOF)
    # ---------------------------------------------------------
    # Just reusing the print layout from before for consistency
    mask = ~np.isnan(df['meta_prob'])
    y_true = y[mask]
    y_prob = df.loc[mask, 'meta_prob']

    if len(y_true) > 0:
        score_logloss = log_loss(y_true, y_prob)
        try: score_auc = roc_auc_score(y_true, y_prob)
        except: score_auc = 0.5
        score_ic, _ = spearmanr(y_prob, y_true)
        if np.isnan(score_ic): score_ic = 0.0

        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        score_mce = np.max(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0.0

        score_brier = brier_score_loss(y_true, y_prob)

    metrics = {
        "Log Loss": f"{score_logloss:.5f}",
        "AUC":      f"{score_auc:.5f}",
        "IC":       f"{score_ic:.5f}",
        "MCE":      f"{score_mce:.5f}",
        "Brier":    f"{score_brier:.5f}"
    }

    print("\n   WINNER PERFORMANCE (OOF)")
    for k, v in metrics.items():
        print(f"   {k:<10} : {v}")
    print("")

    # Return full dataframe with predictions + final model
    return df, final_model

# ---------------------------------------------------------
# Helper: Advanced Diagnostic Plot (Unchanged)
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
