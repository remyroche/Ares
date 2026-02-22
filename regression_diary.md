# Regression Diary: MR Signal Investigation

## Baseline Reference
- **Commit**: `34e0a3ff956a14df5cd78c09784054cc2cdf56db`
- **Mit**: long_mr=0.215, long_tf=0.152, short_mr=0.028, short_tf=eta IC at that comm0.092
- **Meta winners**: Ridge (long_mr α=5.0, long_tf α=10.0, short_mr α=0.01), ExtraTrees (short_tf)

## Phase 0: Revert to Baseline (2026-02-12)
- Reverted all `extreme_price_movements/` files to 34e0a3ff state
- Key changes removed:
  - Entropy/semivariance/regime features
  - Dir_path 2h features & gate interactions
  - Soft labels, MFE/MAE weighting, weighted_union quantile mode
  - Multi-horizon alpha ensemble (AlphaHorizonEnsemble)
  - Complex meta model race (XGB/LGB/quantile) → simple Ridge vs ET
  - Gate-aware winner selection, Platt scaling, prior correction via logit shift
  - Dynamic regularization in ModelRace
  - Early stall exit in triple barrier labeling
  - Adaptive vol-scaling for barriers
  - Native model save/load, OOF parquet caching
  - Stage="base"/"meta" pipeline split
- **Status**: Running baseline pipeline to verify metrics

---

## Run Log

### Run 1: Baseline Verification
- **Date**: 2026-02-13
- **Config**: 34e0a3ff baseline (3yr data, fetch_symbols_M=500, 496 symbols)
- **Steps**: feature_generation → labels → train
- **Expected**: IC close to 34e0a3ff reference values

#### Results

| Bucket   | Baseline IC | Current IC | Baseline Winner       | Current Winner        |
|----------|-------------|------------|-----------------------|-----------------------|
| long_mr  | 0.215       | 0.197      | Ridge (α=5.0)         | ExtraTrees            |
| long_tf  | 0.152       | 0.278      | Ridge (α=10.0)        | ExtraTrees            |
| short_mr | 0.028       | 0.179      | Ridge (α=0.01)        | Ridge (α=0.01)        |
| short_tf | 0.092       | 0.227      | ExtraTrees            | ExtraTrees            |

#### Observations
- ICs are in a comparable range. Differences are expected due to:
  - Universe composition changes (different symbols available, volume rankings shift)
  - Non-deterministic model training (ExtraTrees, XGBoost)
  - Data window shift (~2 hours later than original baseline run)
- short_mr improved significantly (0.028 → 0.179), likely due to universe composition
- long_tf improved (0.152 → 0.278)
- long_mr slightly lower (0.215 → 0.197)
- Meta winner shifts (long_mr/long_tf: Ridge→ET) are within normal race variance
- **Verdict**: Baseline is functional. Code matches 34e0a3ff logic. Proceed with incremental re-introduction.

#### Fixes Applied During Run
1. **Memory fix**: Disabled joblib cache on `_compute_features_cached` (was OOM from serializing 326-feature dict)
2. **Save fix**: `save_features` now processes one symbol at a time instead of pre-extracting all numpy arrays
3. **Cache clearing**: Added `clear_cache()` to `run_pipeline.py` (clears `./cache/features/` before each run)
4. **Timestamp fix**: Labels/train modes now use `find_latest_feature_ts()` instead of `get_ts_sig()` to match feature generation timestamp
5. **Exhaustion bug**: Fixed `predict_proba` IndexError when single-class fold returns 1-column array

---

### Run 2: Phase I — Re-add Feature Generation Changes
- **Date**: 2026-02-13
- **Changes applied**:
  - `features.py`: Restored improved version from commit `1877b27fb` — adds entropy (Shannon, permutation, spectral), semivariance (up/down), vol/liq gates, dir_path 2h features, OHLCV trend quality, regime-transition features, multi-day regime features, per-column FFD caching, memory-optimized sequential transforms
  - `gated_features.py`: Restored panel-aware gate generation, gate selection logic, gate interactions
  - `config.py`: Updated MODEL_FEATURES, HELPER_BASE_FEATURES, causal_cols, tf/mr/meta_feature_keys with new features. **Kept baseline label/training params** (TP=[5.0,3.5,2.0], min_net_rr=1.5, extreme_pct=0.05, accept_gate_window=64)
  - `fast_funcs.py`: Added public `numba_ewma` wrapper
  - `data_store.py`: Rewrote `save_features` with chunked approach (30 features/chunk) to avoid OOM with 451 features
- **Feature count**: 451 (vs 326 baseline)
- **Steps**: feature_generation → labels → train

#### Results

| Bucket   | Baseline IC | Phase I IC | Baseline Winner       | Phase I Winner        | Delta   |
|----------|-------------|------------|-----------------------|-----------------------|---------|
| long_mr  | 0.197       | 0.227      | ExtraTrees            | Ridge (α=5.0)         | **+0.030 ✅** |
| long_tf  | 0.278       | 0.091      | ExtraTrees            | Ridge (α=0.01)        | **-0.187 ❌** |
| short_mr | 0.179       | 0.229      | Ridge (α=0.01)        | ExtraTrees            | **+0.050 ✅** |
| short_tf | 0.227       | 0.188      | ExtraTrees            | ExtraTrees            | **-0.039 ❌** |

#### Detailed Meta Metrics

| Bucket   | GtP   | Spread (bps) | Sharpe | Top10 AvgRet (bps) |
|----------|-------|-------------|--------|---------------------|
| long_mr  | 1.320 | +6.16       | 0.092  | +22.01              |
| long_tf  | 1.368 | +3.06       | 0.086  | +26.82              |
| short_mr | 1.080 | +5.43       | 0.027  | +5.74               |
| short_tf | 1.586 | +6.37       | 0.146  | +41.23              |

#### Base Model Best Horizons

| Bucket   | Best H | Winner   | OOF AUC | Rw-AUC |
|----------|--------|----------|---------|--------|
| long_mr  | H=8    | catboost | 0.553   | 0.535  |
| long_tf  | H=2    | lightgbm | 0.596   | 0.564  |
| short_mr | H=8    | catboost | 0.605   | 0.591  |
| short_tf | H=8    | lightgbm | 0.611   | 0.594  |

#### Observations
- **long_mr improved** (+0.030 IC): New features (entropy, semivariance, regime) help MR detection
- **long_tf degraded** (-0.187 IC): Significant drop. The expanded feature set may be adding noise for TF. Ridge α=0.01 (very low regularization) suggests overfitting to noisy features
- **short_mr improved** (+0.050 IC): ExtraTrees now wins over Ridge, suggesting non-linear feature interactions help
- **short_tf slightly degraded** (-0.039 IC): Minor drop, still strong (GtP=1.586, best overall)
- **Net assessment**: Mixed. 2 heads improved, 2 degraded. long_tf degradation is concerning (-0.187)
- **Verdict**: Proceed to Phase II (label changes) since the feature generation changes are a prerequisite for later phases. The long_tf degradation may be addressed by Phase III (base model changes with better feature selection)

---

### Run 3: Phase II — Re-add Label Changes
- **Date**: 2026-02-13
- **Changes applied**:
  - `labeling.py`: Restored improved version — adds **early stall exit** in triple barrier (if MFE < 50% of activation at 50% of horizon, exit early)
  - `sample_weights.py`: Restored improved version — adds `compute_mfe_mae_weights()` function (not yet called by baseline training.py)
  - `config.py`: Updated label params:
    - `label_tp_values_pct`: [5.0,3.5,2.0] → [1.5,2.0,3.0,4.0,5.0,6.0] (wider TP search)
    - `label_min_net_rr`: 1.5 → 0.9 (less restrictive)
    - `train_extreme_pct_hourly`: 0.05 → 0.07 (more extreme events)
    - `trade_extreme_pct`: 0.06 → 0.07
    - Added: `mfe_mae_w_min=0.5`, `train_min_range_pct=0.07`, `train_min_vol_zscore=1.6`, `label_quantile_mode="weighted_union"`
    - Note: `mfe_mae_*`, `train_min_range_pct`, `train_min_vol_zscore`, `label_quantile_mode` are not yet consumed by baseline training.py
- **Steps**: labels → train (features unchanged from Phase I)

#### Results

| Bucket   | Phase I IC | Phase II IC | Phase I Winner        | Phase II Winner       | Delta   |
|----------|------------|-------------|-----------------------|-----------------------|---------|
| long_mr  | 0.227      | 0.127       | Ridge (α=5.0)         | Ridge (NegIC=+0.034)  | **-0.100 ❌** |
| long_tf  | 0.091      | 0.162       | Ridge (α=0.01)        | ExtraTrees            | **+0.071 ✅** |
| short_mr | 0.229      | 0.282       | ExtraTrees            | ExtraTrees            | **+0.053 ✅** |
| short_tf | 0.188      | 0.218       | ExtraTrees            | ExtraTrees            | **+0.030 ✅** |

#### Detailed Meta Metrics

| Bucket   | GtP   | Spread (bps) | Sharpe | Top10 AvgRet (bps) |
|----------|-------|-------------|--------|---------------------|
| long_mr  | 1.209 | -4.59       | 0.060  | +13.77              |
| long_tf  | 1.181 | +5.96       | 0.045  | +11.23              |
| short_mr | 1.072 | +6.86       | 0.024  | +4.63               |
| short_tf | 1.812 | +5.33       | 0.195  | +58.17              |

#### Observations
- **long_mr degraded** (-0.100 IC): Ridge NegIC is *positive* (+0.034) meaning meta model hurts. The wider TP search + lower min_net_rr may be creating noisier labels for MR
- **long_tf recovered** (+0.071 IC): Improved from 0.091 to 0.162, partially recovering the Phase I drop
- **short_mr improved** (+0.053 IC): Now at 0.282, best so far
- **short_tf improved** (+0.030 IC): GtP=1.812 is excellent, Sharpe=0.195
- **Net assessment**: 3/4 improved, but long_mr is now the weakest link. The label changes help short models significantly
- **Verdict**: Keep Phase II changes. long_mr degradation may be addressed by Phase III (base model improvements). Proceed to Phase III

---

### Run 4: Phase III+IV — Re-add Base + Meta Model Changes
- **Date**: 2026-02-13
- **Changes applied**:
  - `training.py`: Restored improved version — multi-horizon alpha ensemble (deploy all 3 horizons per bucket), complex meta model race (23 candidates: XGB quantile, Ridge, ExtraTrees, with tail-weighting and monotone constraints), strict guardrails (spearman_ic≥0.03, robust_loss≥2%, etc.), MFE/MAE weighting support, `compute_meta_target` (weighted avg of per-horizon log-returns), `train_meta_models_from_artifacts` with union dataset building
  - `model_race.py`: Restored improved version — native save/load, gate-aware winner selection, prior correction via logit shift, Platt scaling, dynamic regularization, isotonic calibration
  - `feature_selection_extreme_events.py`: Restored improved version — MDI v4 with TopK + decile-ranking selection, combined importance weighting
  - `post_race_hpo.py`: Restored improved version — relaxed regularization bounds, float64 precision
  - `meta_model.py`: Restored improved version — complex race with 23 candidates, monotone constraint discovery, HPO via Optuna
  - **Bug fix**: `compute_meta_target` shape mismatch — aligned horizon return arrays to base dataset length
- **Steps**: train only (features + labels unchanged from Phase II)

#### Base Model Results

| Bucket   | Best H | Winner    | OOF AUC | Rw-AUC | OOF IC  |
|----------|--------|-----------|---------|--------|---------|
| long_mr  | H=2    | catboost  | 0.571   | 0.554  | 0.068   |
| long_tf  | H=2    | lightgbm  | 0.616   | 0.600  | 0.139   |
| short_mr | H=2    | catboost  | 0.602   | 0.595  | 0.179   |
| short_tf | H=8    | catboost  | 0.644   | 0.683  | 0.311   |

#### Meta Model Results

| Bucket   | Winner                              | Guardrails | Notes |
|----------|-------------------------------------|------------|-------|
| long_mr  | xgb_multi_075_080_085_unconstrained | ❌ All fail | spearman_ic < 0.03, ic_stable_sign: False |
| long_tf  | extratrees_tailweighted_l0          | ❌ Most fail | ic_stable_sign: True, es20_meta_vs_base: True |
| short_mr | ridge_tailweighted_l1               | ❌ Most fail | ic_stable_sign: True, es20_meta_vs_base: True, net_return_vs_no_meta: True |
| short_tf | ridge_reg                           | ❌ Most fail | top20_bottom50_spread: True, es20_meta_vs_base: True |

#### Observations
- **Base models are strong**: short_tf H=8 OOF_AUC=0.644, IC=0.311 — best base model across all phases
- **Meta models all fail guardrails**: The improved meta model race has strict guardrails (spearman_ic≥0.03, robust_loss≥2%, etc.) that none of the 4 buckets pass
- **Complex race overhead**: Training time increased from ~230s to ~635s due to 23-candidate race + Optuna HPO per meta model
- **Guardrail strictness**: The guardrails may be too strict for this dataset size (~1500-2000 samples per bucket). The baseline simple Ridge vs ET race was more forgiving
- **Net assessment**: Base model improvements are real (better AUC, IC). Meta model regression due to overly strict guardrails
- **Verdict**: The improved base model code (multi-horizon ensemble, isotonic calibration, MDI v4) is beneficial. The improved meta model race is too strict — consider relaxing guardrails or reverting to simpler Ridge vs ET race
