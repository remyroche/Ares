# Models Trained in Extreme Price Movements Pipeline

## 1. Base Models (Alpha Models)
*   **Type:** Classifier (Ensemble via `ModelRace`: ExtraTrees, XGBoost, LightGBM, CatBoost)
*   **Target:** Binary Classification (Positive = Take Profit, Negative = Stop Loss or Timeout).
    *   Derived from Triple Barrier Method labels (2=TP, 1=TO, 0=SL).
    *   `y_bin = (label == 2)`
*   **Objective Function:** Log Loss (Binary Cross-Entropy).
    *   Evaluated via `calculate_selection_score` (AUC, IC, BSS).
*   **Prediction Output:** Probability of Take Profit (`oof_probs`).

## 2. Meta Models
### Regression Meta Model (`MetaModel`)
*   **Type:** Regressor (ExtraTrees, Ridge, XGBoost, HuberRegressor)
*   **Target:** Risk-normalized Log-Return (`y_target_h`).
    *   Weighted average of returns at H2, H4, H8 horizons.
    *   Normalized by volatility proxy (ATR) and squashed via `asinh`.
*   **Objective Function:**
    *   Evaluation: Composite Score (maximizing Spearman IC on OOF predictions).
    *   Training: MSE (ExtraTrees/XGBoost), Huber Loss (HuberRegressor), or L2-regularized MSE (Ridge).
*   **Prediction Output:** Predicted Score (Return/Utility proxy).

### Classification Meta Model (`MetaClassifierModel`)
*   **Type:** Classifier (LogisticRegression, ExtraTrees, CatBoost)
*   **Target:** Multi-class Labels (0=SL, 1=Timeout, 2=TP).
    *   Derived from risk-unit thresholds (`k_tp * vol`, `k_sl * vol`).
*   **Objective Function:** Log Loss (Multi-class Cross-Entropy).
    *   Evaluated on Utility (Lift, Precision, PnL).
*   **Prediction Output:** Probability distribution over [SL, TO, TP].

## 3. Position Sizer (`RidgePositionSizer`)
*   **Type:** Regressor / Optimizer (Constrained Linear Combiner)
*   **Target:** Net Log Returns (`y_net`).
    *   Target representation selected via race (e.g., `winsorized`, `huber_adv`, `rolling_rank`).
*   **Objective Function:** Huber Loss + L2 Regularization.
    *   Uses asymmetric sample weights (losing trades weighted more).
*   **Prediction Output:** Portfolio Weights / Position Size Signal.

## 4. Exhaustion Models (`ExhaustionModel`)
*   **Type:** Classifier (ExtraTreesClassifier)
*   **Target:** Binary Reversal (1=Reversal, 0=Continuation).
    *   UP Model: Long Reversal (Bottom) during Downtrend.
    *   DOWN Model: Short Reversal (Top) during Uptrend.
*   **Objective Function:** Log Loss / Gini Impurity.
*   **Prediction Output:** Probability of Reversal.

## 5. Specialist Models
### Trap Specialist
*   **Type:** GMM (Gaussian Mixture Model) - Clustering.
*   **Target:** Quality Labels (derived from `compute_quality_labels`).
*   **Objective Function:** Maximize Likelihood (EM Algorithm).
*   **Prediction Output:** Quality Score (mean quality of the assigned cluster).

### Gamma Specialist
*   **Type:** Regressor (ExtraTreesRegressor).
*   **Target:** Realized Volatility Magnitude (`y_gamma`).
*   **Objective Function:** MSE (Mean Squared Error).
*   **Prediction Output:** Predicted Volatility Magnitude.
