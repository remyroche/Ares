# Changes and Degradation Report (Feb 10 -> Feb 11)

## 1. Summary of Degradation
Several metrics degraded between the Feb 10 state and the current Feb 11 state, specifically:
- **`long_tf` Rw-AUC**: Dropped from 0.618 to 0.547.
- **`long_tf` ECE@10**: Worsened significantly from 0.006 to 0.136 (massive miscalibration).
- **`short_mr` OOF IC**: Worsened from 0.231 to 0.135.
- **Meta Models**: Underperformed, with only 1/4 passing all stage gates.

## 2. Root Cause of ECE Degradation
A critical bug was identified in `extreme_price_movements/model_race.py` concerning the calibration of the final model.

*   **Issue:** The `ModelRace` workflow trains fold models, applies bias correction (to match unweighted prevalence), and then trains an Isotonic Calibrator on these corrected OOF probabilities. However, the *final* model is retrained on the full dataset with sample weights (which biases predictions towards the weighted mean, typically ~0.5). The `predict_proba` method was feeding these *biased* raw probabilities directly into the Isotonic Calibrator, which expects *unbiased* (low prevalence) probabilities.
*   **Impact:** This mismatch caused the model to over-predict probabilities (e.g., predicting ~0.5 when the true probability is ~0.05), leading to the explosion in ECE (Expected Calibration Error) and likely affecting downstream metrics that rely on calibrated scores.
*   **Fix:** The `predict_proba` pipeline has been patched to calculate and apply a bias correction factor to the final model's raw output *before* feeding it to the calibrator.

## 3. Exhaustive List of Changes (Inferred)
Based on code analysis and "Updated 2026-02-10" annotations, the following changes were made:

### Feature Engineering (`extreme_price_movements/features.py`)
*   **New Exhaustion & Risk Features:** Added `wick_ratio_4h_max`, `vol_price_div`, `rsi_lag1`, `rsi_1h_slope`, `cvar_5pct` (Tail Risk), `amihud_illiq` (Liquidity Shock), `clv_mean_24`, `vol_z_4h`.
*   **New Alpha Features:** Added `breakout_min`, `impulse_reversal`, `breakout_t`, `pct_breakout_t`.
*   **Orthogonal Features:** Added `mtf_divergence`, `autocorr_6h`, `path_efficiency`, `hurst_proxy`, `vol_concentration`, `vol_price_diverge`.
*   **Residualised Features:** Added z-scored versions of key signals (`rsi_z`, `dist_ema_fast_z`, etc.) and "surprise" features (`accept_surprise`, `overext_surprise`).
*   **Updated Gate Logic:** The `add_gate_features_panel` and `select_gated_features` functions were updated to dynamically select the best gate thresholds (e.g., `gt66` vs `gt85`) based on performance.
*   **New Helper Features:** `donch_dist`, `pullback`, `excess`, `clv`, `evr`, `progress`, `speed`, `tail_against`, `mfe/mae`.

### Model Training (`extreme_price_movements/training.py`)
*   **MDI Feature Selection:** Implemented `mdi_feature_selection_v3` inside `train_models_from_artifacts` to prune features before training `ModelRace`. This selects the top ~60 features from the expanded set.
*   **Spike Anatomy Models:** Added training for `spike_anatomy_best` and `spike_anatomy_worst` GMM models.
*   **Specialist Injection:** `trap_score` and `gamma_score` OOF predictions are now injected into the Alpha Model feature set.
*   **Meta Model Training:** Updated to use a rank-percentile target (`y_target`) instead of raw returns or binary labels, and added interaction features (`pred_logit` × `vol_z`, etc.).

### Configuration (`extreme_price_movements/config.py`)
*   **Expanded Feature Lists:** `MODEL_FEATURES`, `tf_feature_keys`, `mr_feature_keys`, and `meta_feature_keys` were significantly expanded to include the new features listed above.
*   **Calibration Settings:** Added `ece_top10` to metrics tracking.

### Model Race (`extreme_price_movements/model_race.py`)
*   **Calibration Logic:** Added `IsotonicRegression` on OOF predictions and manual bias correction. (This contained the bug described in Section 2).

## 4. Recommendations
*   **Verify ECE Fix:** Ensure `long_tf` ECE drops back to < 0.01 levels.
*   **Monitor Feature Selection:** The dynamic MDI feature selection might be unstable if the feature set is too large or noisy. Check if `long_tf` features are consistent across folds.
*   **Meta Model Tuning:** The Meta Model degradation might be due to the new features or the calibration issue (since Meta inputs are Alpha OOFs, but if Alpha OOFs were fine in the race, Meta inputs should be fine. However, if Meta uses *final* Alpha predictions for inference, those were broken).
