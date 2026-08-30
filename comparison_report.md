# In-Depth Comparison: Commit `34e0a3ff9` vs `1877b27fb`

This report details the file-by-file changes made in the `extreme_price_movements/` directory between commit `34e0a3ff9` (older) and `1877b27fb` (newer), with a focus on **base models training**.

## **Executive Summary**

The changes represent a significant architectural shift towards a **Tournament-Based Training Pipeline** with enhanced robustness and feature engineering:
1.  **Architecture**: Shifted from single-model training to a **Parallel Tournament** of candidates (XGBoost, LightGBM, ExtraTrees, Quantile) with automatic monotonic constraints.
2.  **Pipeline**: Split into distinct **Base (Alpha)** and **Meta** stages for better resource management and resumability.
3.  **Robustness**: Introduced **Regime-Stratified Purged K-Fold** to combat fold instability across market regimes, and stricter regularization in HPO.
4.  **Features**: Massive expansion of meta features (Trend Quality, Support/Dip Context, "Report 2026-02-10" requests) and optimization using **Parallel Numba Kernels**.
5.  **Gating**: New tail-focused metrics (Top 40% Coverage/Loss, Downside Protection) for Meta Models.

---

## **1. Training Pipeline & Orchestration**

### **`training.py`**
*   **Stage-Gated Logic**: Explicitly separates Alpha and Meta model training. Fails the pipeline if fewer than 50% of models pass gates.
*   **Meta Training**: Now constructs a **Union Dataset** across horizons (H2, H4, H8) instead of intersection, maximizing data usage.
*   **Reporting**: Added detailed JSON reporting (`quality_gate_report`) containing winners, losers, and specific failure reasons (e.g., "Pass_Spread", "Pass_Downside").
*   **Multi-Horizon Support**: Explicitly handles models trained on multiple horizons (`models_by_h`), defaulting to H4 if others are missing.

### **`pipeline_steps.py`**
*   **Two-Stage Execution**:
    *   `run_training_base_step`: Trains Alpha models and saves them to `native/` directory.
    *   `run_training_meta_step`: Loads base models (preferring native format), trains Meta models, and saves final state.
*   **Native Serialization**: Added logic to load models from `native` directory (using `ModelRace.load_native`) which is significantly faster and more storage-efficient than pickling.
*   **Fallback Logic**: If current run artifacts are missing, searches previous runs for base models to allow meta-only retraining.
*   **Memory Management**: Added explicit `gc.collect()` and dataframe deletion after panel construction.

---

## **2. Model Architecture & Definition**

### **`meta_model.py`** (Major Refactor)
*   **Tournament Architecture**: `fit()` now runs a parallel race of multiple candidate models (XGB, LGBM, ExtraTrees, Quantile Regression) with different objectives.
*   **Monotonic Constraints**: Automatically discovers monotonic relationships between features and target, and enforces them on tree models to prevent overfitting.
*   **Guardrails**: Implemented strict "Guardrails" (e.g., minimum IC, Sharpe) to filter candidates before HPO.
*   **Quantile Regression**: Explicit support for quantile objectives (`quantile_alpha`), predicting median or specific tails.
*   **HPO**: Uses Optuna to optimize hyperparameters for the *winning* candidate only.

### **`model_race.py`**
*   **Calibration Overhaul**:
    *   **Isotonic**: Now fits *without* sample weights to target actual prevalence (approx. 0.31) instead of weighted prevalence (0.5), fixing probability calibration.
    *   **Platt Scaling**: Added optional Platt Scaling (Logistic Regression) after Isotonic if it improves Brier Score by > `1e-4`.
*   **Prediction Safety**: `predict_proba` now enforces strict checks on calibration state to prevent using uncalibrated models.
*   **Native IO**: Added `save_native` / `load_native` methods to handle backend-specific formats (e.g., `.ubj` for XGBoost).

### **`model_mr.py` / `model_tf.py`**
*   **Two-Stage Feature Selection**:
    1.  **Stage 1**: Uses `mdi_feature_selection_v3` to select 2x the target number of features.
    2.  **Stage 2**: Refines selection using **`mdi_feature_selection_v4_topk`** (see below).

### **`purged_cv.py`**
*   **`RegimeStratifiedPurgedKFold`**: New CV class.
    *   Ensures each fold has a balanced distribution of Volatility Regimes (Low/Normal/High).
    *   Computes regimes dynamically from 24h rolling volatility if not provided.
    *   Addresses "Fold Robustness" issues where models failed when tested on unseen regimes.

---

## **3. Feature Engineering & Selection**

### **`features.py`**
*   **New Meta Features**: Implemented requests from "Report 2026-02-10":
    *   **Trend**: `trend_t`, `trend_z_t`, `convexity_t`.
    *   **Breakout**: `breakout_t`, `vw_breakout`, `breakout_soft`.
    *   **Mean Reversion**: `mr_soft`, `mr_potential`, `climax`, `shock_decay`.
*   **Context Features**:
    *   **Trend Quality**: `trend_regime_stability`, `trend_strength_vs_reversion`.
    *   **Support/Dip**: `support_quality_score`, `dip_velocity`, `reversion_target_distance`.
*   **Optimization**: Replaced pandas rolling operations with `ff.numba_rolling_*` for massive speedup.

### **`fast_funcs.py`**
*   **Parallel Kernels**: Added `@jit(parallel=True)` implementations for:
    *   Rolling Max, Min, Sum, Mean, Std, Median.
    *   Percent Change, EWMA.
*   **Fused Kernels**:
    *   `numba_atr`: Single-pass High/Low/Close processing.
    *   `numba_zscore`: Single-pass mean/std/zscore calculation with "Assumed Mean" centering for numerical stability on large float values.

### **`feature_selection_extreme_events.py`**
*   **`mdi_feature_selection_v4_topk`**:
    *   Combines **MDI Importance** (70%) with **Decile-Based Ranking Importance** (30%).
    *   **Decile Ranking**: Measures Spearman correlation between feature value deciles and target positive rate. Prioritizes features with strong *monotonic* signal.

### **`config.py`**
*   **Feature Expansion**: Added `_gtXX_` granular score bins (25, 50, 66, 75, 85, 90) and multi-day regime context (`donch_dist_48`, etc.).
*   **Risk**: Reduced `min_tp_sl_ratio` from 1.5 to **1.2**.
*   **Labels**: Reverted `label_soft_alpha_max` to `0.15`.

---

## **4. Optimization & Tuning**

### **`post_race_hpo.py`**
*   **Stricter Regularization**:
    *   **XGBoost**: Increased `min_child_weight` (10→75), `reg_lambda` (5→15).
    *   **LightGBM**: Increased `min_child_samples` (30→75), `lambda_l2` (5→15).
    *   **CatBoost**: Increased `l2_leaf_reg` (5→15).
*   **CV**: Reduced default `n_splits` from 4 to 3.
*   **Precision**: Switched to `float32` for predictions and weights to save memory.

### **`gate_metrics.py`**
*   **Meta Gates**: Added specific logic for **Quantile Meta Models**:
    *   **Coverage**: Must be within 8% of target tau (0.85) on Top 40% predictions.
    *   **Pinball**: Must improve over baseline by 1%.
    *   **Downside**: ES10 of selection must not be >20% worse than baseline.

### **`data_store.py`**
*   **Resumability**: `save_features` now tracks progress in `_resume.json`.
*   **Memory**: Optimizes saving by extracting columns one-by-one per symbol instead of materializing full arrays.
