# HPO Pipeline Changes - 2026-01-01

## Overview
Comprehensive debugging session to resolve blockers in `meta_labeling_hpo_sample_weighted` pipeline. The goal was to enable successful end-to-end execution of Layers 0-5.

## Critical Bug Fixes

### Layer 2: Feature Selection
- **Issue**: Feature selection discarding threshold was too lenient (40%).
- **Fix**: Updated `lgbm_feature_selection.py` config `bottom_percentile` to `0.50` (discard bottom 50%).

### Layer 2: OOF (Out-of-Fold) Analytics
- **Issue 1 (KeyError)**: Mismatch between features index and label index during OOF due to dropped events.
  - **Fix**: Added robust index intersection logic in `run_oof_analytics` to align `labels` and `weights` with `X_train_final`.
- **Issue 2 (LGBM Deprecation)**: `early_stopping_rounds` passed to `fit()` while also using `callbacks`.
  - **Fix**: Removed explicit `early_stopping_rounds` argument, relying solely on the callback.
- **Issue 3 (IndexError)**: `predict_proba` returned 1D array when training set had single class, causing `[:, 1]` slicing failure.
  - **Fix**: Added shape check (`ndim == 2`) to handle both 1D and 2D probability outputs safely.
- **Issue 4 (Empty Output)**: `individual_geos` dictionary was overwritten in each CV fold, resulting in empty or partial output that failed downstream checks.
  - **Fix**: Implemented accumulation logic (`_individual_geos_accum`) to collect predictions across folds and concatenate them at the end of the OOF process.
- **Issue 5 (TypeError)**: `compute_realized_returns` crashed when `stop_threshold` was `None` (empty `sr_levels`).
  - **Fix**: Added explicit `None` handling for `stop_threshold` in `feature_generation_meta_labeling_step.py`, defaulting to 0.0.

### Layer 3: Meta-Labeling
- **Issue 1 (Key Mismatch)**: `meta_labeling_hpo_sample_weighted.py` tried to unpack `individual_geometries` but Layer 2 returned `individual_geos`.
  - **Fix**: Updated unpacking key to `individual_geos`.
- **Issue 2 (LightGBMError)**: Feature names (geometry UUIDs) containing JSON characters (`{`, `}`,`:`) caused LGBM training to crash.
  - **Fix**: Implemented sanitization of `individual_geos` keys in `meta_labeling_hpo_sample_weighted.py` before passing to Layer 3 (replacing special chars with underscores).
- **Issue 3 (IndexError)**: Similar to Layer 2, Layer 3's probability head training crashed on `predict_proba` shape.
  - **Fix**: Applied the same robust 1D/2D shape check to `train_probability_head` in `layer3/model_training.py`.

## Verification
- **Run 12** successfully navigated all previous failure points.
- Layer 2 correctly trains, predicts, and accumulates OOF data.
- Layer 3 successfully ingests sanitized base model predictions and trains meta-models (Logistic Regression, LGBM).
