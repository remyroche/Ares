# Layer A Predictor Review Report

## A. Layer A mechanics

Layer A constructs a composite sizing score based on three predictive submodels: Edge, Downside, and Uncertainty.

### 1. Edge Model (Model 1)
- **Feature Set:** `model1_edge_feature_keys` from `config.py`. It includes base model OOF properties (mean, std, meta predictions), sign agreement, recent returns (3, 6, 12), EMA properties, regime scores, and impulse/range metrics. Features undergo ElasticNet selection if enabled.
- **Target Definition & Race:** Four targets are evaluated:
  1.  `log_clipped_winsorized_net`: Symmetrically soft-winsorized net returns.
  2.  `rank_style_target`: Fold-local cross-sectional rank of returns mapped to [-1, 1].
  3.  `robust_utility_target`: Averaged policy utility simulated over multiple SL/TP geometries.
  4.  `volatility_normalized_target`: Returns normalized by entry ATR.
  The winning target is chosen by running 2-fold temporal cross-validation for each target and selecting the one with the highest PnL score (`mean_net - 0.5 * std_net` of top decile PnL).
- **Transform(s):** Features are scaled using `PredictionScaler` (replaces NaNs with train fold median, then StandardScaling). The targets themselves are pre-transformed (e.g., `log_clipped_winsorized_net` applies a symmetric soft-plus/log1p transformation).
- **Model Type:** `Ridge(alpha=1.0)`.
- **Training Flow:**
  1. Temporal OOF target race.
  2. The winning target is used for final feature selection (ElasticNet) and fitting the final Ridge model on all provided data.
- **Prediction Output Range:** Continuous. Range depends on the winning target (e.g., [-1, 1] for rank, real values for others).
- **Role inside final score:** Represents the expected positive utility/return. Forms the base of the final score: `Score = Edge - λ*Downside - η*Uncertainty`.

### 2. Downside Model (Model 2)
- **Feature Set:** `model2_downside_feature_keys`. Includes OOF properties, recent returns, regime scores, impulse speed, and wick/rejection features.
- **Target Definition:** `y_downside` (passed as an argument, externally defined).
- **Transform(s):** Uses `_soft_winsorize_downside` which applies soft clipping (tanh) to the tails (0 to 98th percentile). Features are scaled via `PredictionScaler`.
- **Model Type:** `Ridge(alpha=1.0)`.
- **Training Flow:**
  1. 2-fold temporal OOF evaluation.
  2. Final feature selection.
  3. Final fit on all data with the target hard-clipped at the 98th percentile (`y_final = np.clip(y_downside, 0.0, np.percentile(y_downside, 98))`).
- **Prediction Output Range:** Continuous, conceptually positive (representing downside magnitude/risk).
- **Role inside final score:** Represents left-tail risk. Subtracted from Edge: `- λ * downside_pred`.

### 3. Uncertainty Model (Model 3)
- **Feature Set:** `model3_uncertainty_feature_keys`. Includes OOF properties, regime properties, entropy metrics, and importantly, **the OOF predictions from Model 1 and Model 2** (`edge_pred`, `downside_pred`, `edge_minus_downside`, `abs_edge_pred`, `oof_asym_hat`).
- **Target Definition:** The target is the absolute residual of Model 1's OOF predictions against the winning target from the Model 1 target race: `residuals = y_winning_target - model1_oof_pred`.
- **Transform(s):** Fits on the log of the absolute residuals: `y_target = np.log1p(np.abs(residuals))`. Predictions are exponentially transformed back: `np.expm1(pred_log)`.
- **Model Type:** `Ridge(alpha=1.0)`.
- **Training Flow:**
  1. 2-fold temporal OOF eval, fitting *only* on samples where Model 1 OOF predictions are finite.
  2. Final feature selection and fit on the full dataset (where OOF is valid) using the log-transformed residuals.
- **Prediction Output Range:** Continuous, positive (representing absolute error).
- **Role inside final score:** Represents prediction uncertainty or model distrust. Subtracted from the base score: `- η * uncertainty_pred`.

### Score Construction
Final score is a linear combination:
`Score = Edge - (λ * Downside) - (η * Uncertainty)`
(where λ is `lambda_downside` and η is `eta_uncertainty`).


## B. Confirmed issues

1. **Information Leakage in Uncertainty Target Construction (Severity: High)**
   - **Code Evidence:** In `LayerAPredictor._run_model3_oof_eval` and `fit`, Model 3's target is computed as `residuals = self.model1_y_final_ - self.model1_oof_pred_`.
   - **Issue:** `self.model1_y_final_` is constructed using the *entire* dataset. For example, if the winning target is `rank_style_target`, it is rebuilt using `build_rank_target(raw_returns, mode="fold_local")` across the full X vector in `fit`, meaning the rank depends on future data relative to the OOF folds. The OOF predictions (`model1_oof_pred_`) were generated strictly temporally. Subtracting a globally ranked/transformed target from temporally generated OOF predictions mixes different scales and leaks future distribution information into the residuals.
   - **Type:** Correctness issue.

2. **Mixing of Incompatible Target Scales in Final Score (Severity: High)**
   - **Code Evidence:** `Score = Edge - λ*Downside - η*Uncertainty`.
   - **Issue:** The `Edge` scale is completely dynamic based on the target race. It could be `[-1, 1]` (rank), a log-return scale (`log_clipped_winsorized_net`), or an expected utility value. `Downside` has its own external scale (`y_downside`). `Uncertainty` is on the scale of absolute `Edge` residuals. Subtracting them with static `λ=0.5` and `η=0.5` (the defaults) is mathematically arbitrary if `Edge` happens to be a rank target while `Downside` is a raw MAE/ATR value. If the scales don't align perfectly, one component will dominate the score randomly based on the target race winner.
   - **Type:** Design-quality / Correctness issue.

3. **Inconsistent Target Transformations in Downside Model (Severity: Medium)**
   - **Code Evidence:**
     In `_run_model2_oof_eval`:
     Fold fit: `y_tr_down = _soft_winsorize_downside(y_downside[tr_idx], ...)`
     Final fit: `y_final = np.clip(y_downside, 0.0, np.percentile(y_downside, 98))`
   - **Issue:** The OOF predictions (which feed into Model 3 as features) are generated by models trained on *soft-winsorized* targets. The final Downside model is trained on *hard-clipped* targets. This inconsistency means the OOF features seen by Model 3 have different distributional properties than what the final Downside model will produce in live prediction.
   - **Type:** Correctness issue.

4. **Target Race Uses Naive Score That Favors Volatility (Severity: Medium)**
   - **Code Evidence:** `score = mean_net - 0.5 * std_net` in `_run_model1_target_race`.
   - **Issue:** This metric (`top_10_mean_net`) evaluates the raw PnL of the top decile. It does not account for trade duration, capital efficiency, or hit rate strictly. This can favor a target definition that produces highly volatile, lucky extreme predictions in the top decile over a target that produces stable, monotonic predictions across the board.
   - **Type:** Quant / Financial logic issue.

5. **`model1_oof_pred_` Feature Alignment in Final Predict (Severity: Low/Medium)**
   - **Code Evidence:** In `predict_components`, `fd3["edge_pred"] = edge_p`.
   - **Issue:** Model 3 was trained using OOF predictions of Model 1 as features. In production/predict, it uses the actual Model 1 predictions. This is standard stacking practice, but combined with the fact that Model 1's final model is fit on a potentially different target vector (e.g. global rank vs fold rank), the feature distributions will drift between train (OOF) and predict.
   - **Type:** Design-quality issue.


## C. Improvement opportunities

1. **Standardize Target Scales Before Combining:**
   Instead of raw subtraction (`Edge - λ*Downside - η*Uncertainty`), the components should be standardized (e.g., z-scored within the fold or bucket) before blending, OR the models should be forced to predict on a unified, economically comparable scale (like annualized bps or expected PnL).

2. **Remove the Target Race (or constrain it):**
   The target race introduces massive instability in the meaning of the `Edge` score. A simpler, more robust approach is to pick a single, robust target (like `log_clipped_winsorized_net` or `robust_utility_target`) and stick to it. If the race remains, ensure the outputs are scaled to a standard normal distribution before Model 3 residual calculation and final score blending.

3. **Align Downside Training:**
   Use the exact same transformation (`_soft_winsorize_downside`) for both the OOF fold training and the final model training in `Model2Downside`.

4. **Fix Residual Target Leakage:**
   In Model 3's OOF evaluation, compute residuals strictly using the target values as they were defined *during that specific fold*, not against the globally recalculated `self.model1_y_final_`.

5. **Better Economic Metric for Target Race:**
   Change the target race scoring function to use the Sortino ratio or `net_pnl_day` of the top decile, rather than raw `mean_net - 0.5 * std_net`, to ensure capital efficiency is rewarded.


## D. Minimal safe experiments

1. **Ablation: Disable Target Race**
   - **Action:** Hardcode `best_name = "robust_utility_target"` (or another strong default) in `_run_model1_target_race` and bypass the loop.
   - **Rationale:** Tests if the complexity and scale-mixing of the target race is actually degrading out-of-sample performance compared to a static, well-behaved target.

2. **Experiment: Standardize Components Before Scoring**
   - **Action:** Modify `predict_score` (and the corresponding scoring step in `fit`) to z-score `comps["edge"]`, `comps["downside"]`, and `comps["uncertainty"]` individually before applying the λ and η weights.
   - **Rationale:** Ensures that the subtraction operates on mathematically comparable scales, preventing an unexpectedly large variance in Downside or Uncertainty from completely muting the Edge signal.

3. **Experiment: Unify Downside Transform**
   - **Action:** Change the final fit target in `_run_model2_oof_eval` from `np.clip(...)` to use `_soft_winsorize_downside(y_downside, ...)`.
   - **Rationale:** Removes the train/predict distribution skew in Downside features passed to Model 3.