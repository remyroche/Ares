# Investigation Report: Analysis of Changes and Metric Shifts (Feb 10 -> Feb 11)

This report provides an exhaustive list of changes identified in the codebase between Feb 10 and Feb 11, along with an investigation into the observed shifts in model performance metrics.

## 1. Summary of Changes

Based on a detailed code inspection, the following changes were introduced in the current session (Feb 11) compared to the previous state:

### A. New Feature Engineering (`extreme_price_movements/features.py`)
A significant block of features labeled **"Report 2026-02-10"** was added to the `compute_features_hourly` function. These features focus on exhaustion, risk, and specific strategy signals:
*   **Exhaustion Features:**
    *   `wick_ratio_4h_max`: Rolling max of wick ratio (detects rejection).
    *   `vol_price_div`: Volume-price divergence (correlation of returns and volume changes).
    *   `rsi_lag1`, `rsi_1h_slope`: RSI momentum and change.
    *   `clv_mean_24`: Mean Close Location Value (skewness proxy).
    *   `vol_z_4h`: Short-term volume z-score.
    *   `atr_pct_change`: Volatility cooling signal.
*   **Risk Features:**
    *   `cvar_5pct`: Conditional Value at Risk proxy (tail risk).
    *   `amihud_illiq`: Amihud illiquidity proxy (price impact per unit volume).
*   **Meta Features (TF & MR):**
    *   `trend_t`, `trend_z_t`, `convexity_t`, `breakout_t`, `rvol_ratio`.
    *   `vw_breakout`, `breakout_soft`, `tail_score`.
    *   `mr_soft`, `mr_potential`, `climax`, `vol_exhaust`, `shock_decay`, `pct_extreme`, `stall`, `mr_failure`.
*   **Alpha Features:**
    *   `breakout_min`, `impulse_reversal`, `impulse_reversal_short`, `breakout_confirmed`, `pct_breakout_t`.

### B. Specialist Model Integration (`extreme_price_movements/training.py`)
*   **OOF Injection:** The training pipeline now explicitly injects **Out-of-Fold (OOF) predictions** from specialist models (`trap_score`, `gamma_score`) into the feature set for Alpha models.
*   **Feature Filtering:** Training data for Alpha models is filtered using specific `tf_feature_keys` and `mr_feature_keys`, while Meta models receive the full feature set plus `pred_logit` and interaction terms.

### C. Candidate Selection Logic (`extreme_price_movements/candidates.py`)
*   **Vectorized Filters:** The `select_trade_candidates_vectorized` function now enforces stricter criteria:
    *   **Range Filter:** `range_12h_pct > 0.07` (12h High/Low range > 7%).
    *   **Volatility Filter:** `volatility_zscore > 1.6`.
    *   **Sign Consistency:** `sign_consistency >= 0.80` (requires 80% of recent returns to align with trend direction).
*   **Time Expansion:** Candidates are expanded with offsets `[-12, -8, -4, 4, 8, 12, 16]` hours around the event to capture context.

### D. Training Pipeline Updates (`extreme_price_movements/training.py`)
*   **Feature Selection:** Now uses `mdi_feature_selection_v3` with `ExtraTreesRegressor`.
*   **Risk Optimization:** `optimize_risk_params` uses `run_tp_sl_selection_fast` with a **smaller** grid (TP max 1.5) than the "Expanded" version described in the implementation summary (TP max 3.0), suggesting the expanded grid might be missing or reverted.

---

## 2. Investigation of Metric Changes

The observed shifts in model performance metrics are a direct result of the changes listed above.

### A. Improvement in Alpha Rw-AUC and OOF IC
**Observation:**
*   `long_mr` Rw-AUC: +0.029, OOF IC: +0.031
*   `short_mr` Rw-AUC: +0.044
*   `short_tf` Rw-AUC: +0.022

**Cause:**
The addition of **powerful new features** (Wick Ratio, Vol/Price Div, Specialist Scores) provides better signal discrimination.
*   `wick_ratio_4h_max` and `vol_price_div` are excellent for detecting exhaustion, which directly improves `short_mr` (Mean Reversion) and `short_tf` (Trend Following breakdown) strategies.
*   `trap_score` and `gamma_score` (Specialist OOFs) likely add high-quality, independent information about trap probability and volatility regimes, boosting the Information Coefficient (IC).

### B. Degradation in Alpha Calibration (ECE@10)
**Observation:**
*   `long_mr` ECE@10: +0.054 (worse)
*   `long_tf` ECE@10: +0.130 (worse)

**Cause:**
**Overconfidence without Regularization.**
*   The new features make the models more confident (pushing probabilities closer to 0 or 1).
*   However, the **"Regularization Enhancement"** (expanded alpha/lambda grids for calibration) described in the implementation summary is **MISSING** from the current code (`meta_model.py` still uses standard `reg_alpha=0.5/1.0`).
*   This leads to models that rank well (high AUC) but are miscalibrated (high ECE), meaning their probability estimates are too extreme compared to actual win rates.

### C. Failure of Meta Models (Spread/IC)
**Observation:**
*   All Meta Models failed the stage gates (Spread, IC, Coverage).

**Cause:**
**Missing Regularization.**
*   The Meta Models are trained on `pred_logit` plus the new "Report 2026-02-10" meta features.
*   Without the "Expanded alpha grid" (up to 20.0) mentioned in the summary, the Meta Models likely **overfit** to the noise in the new features, leading to poor generalization (low IC) and inability to separate top/bottom performers (low Spread).

### D. Success of Short TF Strategy
**Observation:**
*   `short_tf` passed all stage gates.

**Reason:**
Short Trend Following (`short_tf`) inherently benefits from volatility expansion and breakdown signals. The new `vol_price_div` (volume-price divergence) and `breakout_confirmed` features are particularly strong predictors for this regime, allowing it to maintain high precision (Lift@k) even if calibration is imperfect.

---

## 3. Recommendations

1.  **Apply Regularization:** Implement the "Expanded alpha grid" (up to 20.0) in `meta_model.py` to fix Meta Model overfitting and improve Alpha calibration.
2.  **Calibrate Alpha Models:** Re-run calibration tuning (Isotonic or Platt scaling) or apply the missing "Expanded TP/SL Grids" to find more robust risk parameters that might align probabilities better.
3.  **Verify Candidate Logic:** Ensure the `sign_consistency` check in `candidates.py` is working as intended (using correct data source), as it significantly reduces the candidate pool and improves signal quality.
