# Comparison Report: 34e0a3f vs 1877b27

**Focus:** Labeling and Sample Weights in `extreme_price_movements/`

## 1. Labeling Logic (`labeling.py` & `config.py`)

### Stall Exit Implementation
*   **File:** `labeling.py`
*   **Change:** The `_numba_triple_barrier` function now includes a **Stall Exit** mechanism.
*   **Logic:**
    *   It checks trade progress at 50% of the horizon (`stall_ns = limit_ns // 2`).
    *   If the Maximum Favorable Excursion (MFE) has not reached 50% of the activation threshold (`stall_threshold = 0.5 * activation`), the trade is exited immediately at the close price.
    *   The label is set to `0` (neutral/timeout), and the return is recorded based on the exit price.
    *   This prevents holding dead trades that show no momentum in the first half of the intended duration.

### Soft Labels
*   **File:** `config.py`
*   **Change:** `label_use_soft` is set to `True`, and `label_soft_alpha_max` is explicitly set (reverted) to `0.15`.
*   **Impact:** Determines whether the target variable for training is a hard binary class or a soft probability/score, likely to help models learn from near-misses or strong moves.

## 2. Sample Weights (`sample_weights.py` & `training.py`)

### Decisiveness Weighting
*   **File:** `sample_weights.py`
*   **New Function:** `compute_mfe_mae_weights`
*   **Logic:**
    *   Weights samples based on how "decisive" the price movement was relative to barriers.
    *   Calculates normalized excursions: `r_mfe = mfe / tp` and `r_mae = mae / sl`.
    *   Uses the maximum of these (`d = max(r_mfe, r_mae)`) to compute a base weight: `w_base = w_min + (1 - w_min) * clip(d / tau, 0, 1)`.
    *   **Penalties:**
        *   **Touch Margin:** Weights are halved if the price barely touched the barrier (`touch_margin < cost_floor`).
        *   **Timeouts:** Weights are capped at `0.7` for timeout events (`is_timeout=True`), de-emphasizing indeterminate outcomes.

### Integration in Training
*   **File:** `training.py`
*   **Usage:** The `generate_label_datasets` workflow (implied by config usage) now incorporates `mfe_mae_w_min` (default 0.5) from `config.py`, linking the new weighting logic to the training pipeline.

## 3. Model Calibration & Training (`model_race.py` & `training.py`)

### Isotonic Calibration vs. Sample Weights
*   **File:** `model_race.py`
*   **Change:** The `IsotonicRegression` calibration step is now explicitly fitted **without sample weights**.
*   **Reasoning:**
    *   Sample weights are often used to balance classes (upweighting the minority class).
    *   Calibrating with these weights would bias the probabilities towards `0.5` (balanced).
    *   By ignoring weights during calibration, the model's probabilities are mapped to the **actual dataset prevalence** (e.g., ~30% positives), ensuring they represent real-world frequencies.

### Platt Scaling
*   **File:** `model_race.py`
*   **Change:** An optional **Platt Scaling** (Logistic Regression) step has been added *after* Isotonic calibration.
*   **Logic:** It is only retained if it improves the Brier score by at least `1e-4`, providing an extra layer of calibration refinement if needed.

### Meta Model Union Dataset
*   **File:** `training.py`
*   **Change:** `train_meta_models_from_artifacts` now constructs the training set using the **Union** of samples across horizons (H2, H4, H8).
*   **Logic:**
    *   Samples are aligned by `(timestamp, symbol)`.
    *   Missing OOF predictions (e.g., if a sample exists in H4 but not H2) are imputed with **0.5 (neutral)**.
    *   This ensures maximum data usage and robustness against missing horizon data.

### Tail-Weighted Models
*   **File:** `meta_model.py`
*   **Change:** Explicit support for "tailweighted" models.
*   **Logic:**
    *   If a model is flagged as "tailweighted", the system recognizes it uses a transformed target.
    *   However, for scoring and metric calculation (`_write_model_reports`), it uses the original, un-transformed target (`score_y`) to ensure metrics like PnL and Precision are grounded in reality.
    *   Hyperparameter optimization (`_optimize_hpo`) uses **Pinball Loss** (quantile loss, alpha=0.85) on the *original scale* target.

## 4. Configuration & Reporting (`config.py` & others)

### Feature Explosion
*   **File:** `config.py`
*   **Change:** Massive addition of features:
    *   **Multi-horizon aggregates** (e.g., `ret_mean`, `rv_max`).
    *   **Tail-risk features** (e.g., `tail_risk_score`).
    *   **Multi-day regime context** (e.g., `donch_dist_48`, `trend_slope_120h`).
    *   **Report 2026-02-10 features** for TF and MR Meta models.
*   **Impact:** significantly expands the information available to models, particularly for regime awareness and tail event detection.

### Risk Parameters
*   **File:** `config.py`
*   **Change:** `min_tp_sl_ratio` reduced from `1.5` to `1.2`, allowing for trades with slightly lower reward-to-risk ratios if the probability is high enough.

### Reporting
*   **File:** `training.py` & `meta_model.py`
*   **Change:** Enhanced "Stage Gate" reporting and comprehensive Meta Model metrics (Precision@k, PnL/day, Sortino, etc.) to better evaluate model quality beyond just AUC/LogLoss.
