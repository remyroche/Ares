# Changes Log - HPO Pipeline Stabilization

## 2026-01-02

### `src/training/steps/labeling/label_based_layer_2.py`

1.  **Fixed Syntax Errors**:
    - Corrected malformed keyword arguments in `LGBMClassifier` calls (e.g., changed `objective: 'regression'` to `objective='regression'`, `eval_metric: 'rmse'` to `eval_metric='rmse'`).
    - Corrected syntax in `LAYER2_PROBE_CONSTANTS` dictionary.
    - Fixed lines where multiple statements were incorrectly merged (e.g., `fit(...) preds=...`) by splitting them into separate lines.

2.  **Fixed Probe Feature Alignment / Generation Error**:
    - **Issue**: `self._global_probe_features` was overloaded. It was used as a **data cache** (dict) in `_build_geometry_independent_event_features`, but as a **feature name list** (list of strings) in the HPO orchestration. This caused `TypeError: list indices must be integers...` when the cache logic tried to access it as a dict.
    - **Fix**: Modified `_build_geometry_independent_event_features` to use a dedicated, lazily-initialized `self._probe_data_cache` strictly for caching probe feature data. This isolates the caching mechanism from the feature name list.
    - **Alignment**: Updates `_select_best_geometry_via_race` to use explicit index intersection for aligning probe features, ensuring robustness against index type mismatches.

3.  **Repaired HPO Optimization Methods**:
    - Rewrote `_optimize_focal_loss_params`, `_optimize_lgbm_bce_params`, `_optimize_xgb_params`, and `_optimize_catboost_params`.
    - Restored missing `optuna.create_study` calls.
    - Fixed indentation and logic flow to ensure correct objective function execution and metric reporting.
    - Ensured robust handling of `pruner` configuration.

### `src/training/steps/labeling/orthogonal_label_generation.py`

1.  **Optimized Candidate Construction**:
    - Modified to skip the expensive `get_signal_specific_weights` calculation when `return_raw_candidates=True`.
    - This reduced the candidate generation phase from ~7 minutes to a few seconds, as these weights are not needed for the initial candidate selection race.

### `src/training/steps/labeling/layer3/core.py`

1.  **Fixed Missing Import**:
    - Added `calculate_sample_weights_efficient` to the import from `.utils`.
    - This function was called but not imported, causing `NameError` at Layer 3 startup.

### `src/training/steps/labeling/layer3/model_training.py`

1.  **Fixed Missing Import**:
    - Added `from joblib import Parallel, delayed`.
    - These were used for parallel CV fold training but not imported.

2.  **Fixed predict_proba Array Handling**:
    - Modified `train_probability_head` to handle cases where `predict_proba` returns a 2D array with only 1 column (single class scenario).
    - Now checks `prob.shape[1] >= 2` before accessing `prob[:, 1]`.

3.  **Fixed Single-Class Metrics Computation**:
    - Added try/except around `roc_auc_score` and `log_loss` to handle cases where `y_true` contains only one class.
    - Added `labels=[0, 1]` parameter to `log_loss` for explicit class handling.

### `src/training/steps/labeling/label_based_layer_4.py`

1.  **Fixed Parameter Mismatch in `_train_layer4_oof_extratrees_pnl`**:
    - Added `l3_models_metadata` and `l3_quantile_thresholds` parameters to the function signature.
    - These were passed from `train_layer4_oof` but not accepted by the internal function, causing `TypeError`.

---

## Model Performance Improvements (2026-01-02 07:20)

### `src/training/steps/labeling/layer3/model_training.py`

4.  **Fixed Single-Class CV Fold Handling**:
    - Added check for single-class training/validation sets before each fold.
    - Skips fold and assigns neutral prediction (0.5) if either set has only one class.
    - Wrapped training in try/except to catch calibration failures gracefully.

### `src/training/steps/labeling/orthogonal_label_generation.py`

2.  **Relaxed Balance Gate for Rare-Event Signals**:
    - ~~Extended 3% minimum balance threshold to `VOL_PARTICIPATION`, `RANGE_ATR`, and `TAIL_RISK` families.~~
    - **REVERTED**: Balance gates kept strict per user request.

3.  **Expanded PRICE_CUSUM k Range**:
    - Added lower k values (0.3, 0.5) to base_params.
    - Lower k = more signals (higher recall), Higher k = fewer signals (higher precision).
    - Previous: k ∈ {0.8, 1.2}, Now: k ∈ {0.3, 0.5, 0.8, 1.2}.

4.  **Fixed Probability Head Racing Logic**:
    - Separated LGBM and sklearn model fitting - LogisticRegression doesn't support `eval_set`.
    - Fixed missing `probs` assignment after LGBM fit.
    - Changed eval_set from `(X_tr, y_tr)` to `(X_val, y_val)` for proper early stopping.

### `src/training/steps/labeling/meta_labeling_hpo_sample_weighted.py`

1.  **Skip Layer 5 if Layer 4 is Disabled**:
    - Added `layer5_skipped` flag when Layer 4 is disabled.
    - Layer 5 sizer is not initialized if Layer 4 failed.

---

## Layer 3 Model Training Fixes (2026-01-02 08:50)

### `src/training/steps/labeling/layer3/model_training.py`

1.  **Fixed Alpha Head OOF Collection**:
    - Changed from `fold_preds.extend()` to proper `model_oof[val_idx] = preds` indexing.
    - Predictions are now collected at correct indices for all models.
    - Failed folds now fill with 0.0 (median) instead of being skipped.

2.  **Added StandardScaler for Ridge**:
    - Ridge now uses StandardScaler to prevent coefficient suppression.
    - Alpha increased from 1.0 to 10.0 for event-driven training.

3.  **Added Tweedie Loss Option for LGBM**:
    - Added `tweedie` objective with `tweedie_variance_power=1.2` (per de Prado recommendation).

4.  **Fixed Probability Head OOF Collection**:
    - Same proper indexing fix as alpha head.
    - Added pre-fold single-class checks with skip logic.
    - Handled `predict_proba` output shape edge cases (1D, 2D with 1 col).

5.  **Changed eval_set for Early Stopping**:
    - Changed from `(X_tr, y_tr)` to `(X_val, y_val)` for proper early stopping.

6.  **Added Focal Loss Probability Clipping**:
    - Custom objectives return raw logits - now apply sigmoid and clip to [0, 1].

---

## De Prado Pipeline Improvements (2026-01-02 09:12)

### `src/training/steps/labeling/orthogonal_label_generation.py`

1.  **Horizon Consolidation**:
    - PRICE_CUSUM: [12, 48] (classifier, multiple horizons)
    - All others: [48] only (regressor, single horizon)
    - Expected candidate reduction: ~50%

2.  **Tightened Parameter Sweep**:
    - Reduced `base_params` to single config for non-PRICE_CUSUM families:
      - VOL_CUSUM: 1 config (was 2)
      - LIQ_CUSUM: 1 config (was 2)
      - VOL_PARTICIPATION: 1 config (was 2)
      - RANGE_ATR: 1 config (was 2)
      - TAIL_RISK: 1 metric (was 2)
      - TREND_REGIME: 1 config (was 2)

3.  **Family-Specific Horizon Lookup**:
    - Added logic to use `family_horizons` from config per family.


