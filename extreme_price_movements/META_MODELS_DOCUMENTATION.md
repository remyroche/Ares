# Meta Models Documentation

This document outlines the architecture, models, targets, and objective functions used for the Meta Models in `extreme_price_movements/`.

## 1. Regression Meta Model (`MetaModel`)

The regression meta model is designed to predict the magnitude and direction of returns, aggregating signals from multiple horizons (H2, H4, H8).

*   **Source File:** `extreme_price_movements/meta_model.py`
*   **Target Calculation:** `extreme_price_movements/training.py` (`compute_meta_target`)

### Models (Candidates)
The model uses a race architecture to select the best performer among the following candidates:
*   **Ridge:** `sklearn.linear_model.Ridge` with `RobustScaler` (alpha=5.0).
*   **ExtraTrees:** `sklearn.ensemble.ExtraTreesRegressor` (Baseline: 300 estimators, max depth 8, min samples leaf 40).
*   **ExtraTrees (Tail-Weighted):** Same as above, but trained with `tail_lambda=2.0`. This applies sample weights that emphasize the tails of the distribution.
*   **XGBoost:** `xgb.XGBRegressor` (Objective: `reg:squarederror`) if `xgboost` is available.

### Target
*   **Weighted Horizon Return:** A composite target constructed from log-returns across 2h, 4h, and 8h horizons:
    $$y_{target} = 0.40 \cdot r_{H2} + 0.35 \cdot r_{H4} + 0.25 \cdot r_{H8}$$
*   **Winsorization:** The target is asymmetrically winsorized:
    *   **Downside:** Hard-clipped at the 5th percentile.
    *   **Upside:** Square-root compressed above the 90th percentile to dampen outliers while preserving rank order.
*   **Transformation:** For tail-weighted candidates, the target is transformed using `sign(y) * log1p(|y|)` to linearize heavy tails.

### Objective Functions
*   **Training Objective:**
    *   Ridge: L2 Loss (MSE + Regularization).
    *   ExtraTrees: Variance Reduction (MSE).
    *   XGBoost: `reg:squarederror` (MSE).
*   **Selection & HPO Objective:** The winner of the race is selected and hyperparameter-tuned to maximize **Spearman IC** (Information Coefficient) on Out-Of-Fold (OOF) predictions.

---

## 2. Classification Meta Model (`MetaClassifierModel`)

The classification meta model predicts the probability of a successful trade outcome based on a multi-barrier logic.

*   **Source File:** `extreme_price_movements/meta_model.py`

### Models (Candidates)
*   **Ridge Classifier:** `sklearn.linear_model.LogisticRegression` with L2 penalty (`C=0.1`, class_weight="balanced", solver="lbfgs").
*   **ExtraTrees Classifier:** `sklearn.ensemble.ExtraTreesClassifier` (300 estimators, max depth 8).
*   **CatBoost Classifier:** `catboost.CatBoostClassifier` (if available).

### Target
*   **Multi-Barrier Binary Label:** A binary label (`1` if successful, `0` otherwise).
*   **Logic:** A sample is positive (`1`) if *any* horizon return exceeds a dynamic Take-Profit (TP) threshold across a grid of (TP, SL) combinations (e.g., TP 1.5%-6.0%, SL 0.5%-2.0%) before hitting the Stop-Loss.
*   **Fallback:** If no barrier logic applies (e.g., missing future data), it defaults to the **Top 30%** by absolute return magnitude.

### Objective Functions
*   **Training Objective:**
    *   Models minimize **Log Loss** (Cross-Entropy) or Gini Impurity.
*   **Selection Objective:** The winner is chosen based on a custom **Composite Score** that balances precision, lift, and risk-adjusted returns:
    $$\text{Score} = \text{PR-AUC}_{\text{lift}} \times \text{Lift}_{@26\%} \times (1 + 0.3 \cdot \text{Sortino})$$
