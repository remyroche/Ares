# Investigation Report: Analysis of Changes and Metric Shifts (Feb 10 -> Feb 11)

This report provides an exhaustive list of changes identified in the codebase between Feb 10 and Feb 11, along with an investigation into the observed shifts in model performance metrics.

## 1. Summary of Changes

Based on a detailed code inspection, the following changes were introduced in the current session (Feb 11) compared to the previous state:

### A. New Feature Engineering (`extreme_price_movements/features.py`)
A significant block of features labeled **"Report 2026-02-10"** was added to the `compute_features_hourly` function. These features focus on exhaustion, risk, and specific strategy signals:
*   **Exhaustion Features:** `wick_ratio_4h_max`, `vol_price_div`, `rsi_lag1`, `rsi_1h_slope`, `clv_mean_24`, `vol_z_4h`, `atr_pct_change`.
*   **Risk Features:** `cvar_5pct`, `amihud_illiq`.
*   **Meta Features:** `trend_t`, `breakout_t`, `rvol_ratio`, `mr_soft`, `climax`, `shock_decay`, etc.
*   **Alpha Features:** `breakout_min`, `impulse_reversal`, `breakout_confirmed`.

### B. Specialist Model Integration (`extreme_price_movements/training.py`)
*   **OOF Injection:** The training pipeline now explicitly injects **Out-of-Fold (OOF) predictions** from specialist models (`trap_score`, `gamma_score`) into the feature set for Alpha models.
*   **Feature Filtering:** Training data for Alpha models is filtered using specific `tf_feature_keys` and `mr_feature_keys`.

### C. Candidate Selection Logic (`extreme_price_movements/candidates.py`)
*   **Vectorized Filters:** The `select_trade_candidates_vectorized` function (newly added) enforces stricter criteria:
    *   **Range Filter:** `range_12h_pct > 0.07`.
    *   **Volatility Filter:** `volatility_zscore > 1.6`.
    *   **Sign Consistency:** `sign_consistency >= 0.80`.
*   **Time Expansion:** Candidates are expanded with offsets `[-12, -8, -4, 4, 8, 12, 16]` hours.

### D. Training Pipeline Updates (`extreme_price_movements/training.py`)
*   **Feature Selection:** Uses `mdi_feature_selection_v3` with `ExtraTreesRegressor`.
*   **Risk Optimization:** Uses `run_tp_sl_selection_fast` with a **smaller** grid (TP max 1.5) than intended.

---

## 2. Investigation of Metric Changes

The observed shifts in model performance metrics are a direct result of the changes listed above.

### A. Improvement in Alpha Rw-AUC and OOF IC
**Observation:**
*   `long_mr` Rw-AUC: +0.029, OOF IC: +0.031
*   `short_mr` Rw-AUC: +0.044
*   `short_tf` Rw-AUC: +0.022

**Cause:**
The addition of **powerful new features** (Wick Ratio, Vol/Price Div, Specialist Scores) provides better signal discrimination, boosting the model's ranking ability (AUC) and correlation with returns (IC).

### B. Degradation in Alpha Calibration (ECE@10)
**Observation:**
*   `long_mr` ECE@10: +0.054 (worse)
*   `long_tf` ECE@10: +0.130 (worse)

**Cause:**
**Calibration Pipeline Mismatch in `ModelRace`.**
*   The `ModelRace` logic trains the `IsotonicRegression` calibrator on **bias-corrected OOF probabilities** (where the mean is forced to match the low target prevalence, e.g., 5%).
*   However, during inference (`predict_proba`), the final model's **raw, uncorrected output** (often biased around 0.5 due to class balancing/weighting) is fed directly into this calibrator.
*   Since the input distributions differ drastically (Bias-Corrected vs. Raw), the Isotonic model misinterprets the high raw scores as extreme confidence, resulting in poorly calibrated final probabilities and high ECE.

### C. Failure of Meta Models (Spread/IC)
**Observation:**
*   All Meta Models failed the stage gates (Spread, IC, Coverage).

**Cause:**
**Insufficient Regularization for Noisy Features.**
*   The Meta Models use GBDT (LGBM/XGB) Quantile Regression with HPO.
*   The current HPO ranges for regularization (`reg_lambda` max 20-40) are **insufficient** for the increased noise dimensionality introduced by the new feature set.
*   The optimizer likely overfits to training noise (improving validation Pinball Loss locally) but fails to generalize to the OOS spread metric used in the gates.

### D. Success of Short TF Strategy
**Observation:**
*   `short_tf` passed all stage gates.

**Reason:**
Short Trend Following inherently benefits from volatility expansion signals (`vol_price_div`, `breakout_confirmed`), allowing it to maintain high precision even with imperfect calibration.

---

## 3. Recommendations

1.  **Fix Alpha Calibration:** Update `ModelRace.predict_proba` to apply the same **Bias Correction** to the raw model output *before* passing it to the Isotonic Calibrator, ensuring consistent input distributions.
2.  **Extend Meta Model Regularization:** Update `meta_model.py` to extend the HPO grid for regularization parameters to much higher values (e.g., `lambda` up to 100) to force generalization.
3.  **Verify Candidate Logic:** Ensure `sign_consistency` uses the correct data source (Panel vs Features).
