# Base Model Regression Review (Last 48 Hours)

## Scope
Review recent changes to base-model training and label generation that may explain:
- lower Lift_k,
- lower ROC_AUC,
- worse Brier calibration,
- and altered PR_AUC behavior.

## High-confidence regression candidates

### 1) Labels are no longer forcibly regenerated before training
**Change:** training now skips label generation when artifacts already exist.

- Previous behavior: always refresh labels before training.
- Current behavior: if label artifacts exist, label generation is skipped.

**Why this can hurt quality:**
- Training may mix a new training stack (feature selection, CV folds, weighting, model race logic) with stale labels built under an older geometry/fee configuration.
- This can produce metric drift, especially calibration and lift, because classifier targets and barrier geometry are no longer guaranteed to match current code/config.

**Evidence in logs:**
- `Label artifacts already exist, skipping label refresh...`

### 2) Base-model CV splits reduced from 3 -> 2 in ModelRace call site
**Change:** base race now uses `ModelRace(..., n_splits=2)`.

**Why this can hurt quality:**
- Fewer OOF folds increase variance and can degrade calibration robustness and winner stability.
- It can over-select models that look good in fewer folds but generalize worse, reducing realized lift/AUC and worsening Brier.

### 3) Label geometry scoring was materially changed (quality weighting / penalties)
**Change set (same period):**
- New edge-term cap and reduced edge weight,
- new balance-band penalty,
- new geometry weighting formula.

**Why this can hurt quality:**
- This changes which TP/SL geometries are selected and therefore changes target construction itself.
- If prior strong-lift regimes were edge-ratio-driven, lowering edge influence can reduce top-decile separation (Lift_k), even if PR-AUC shape changes.

### 4) Universe timing behavior changed in feature step
**Change:** universe selection now passes `ts_sig` instead of `None`.

**Why this can hurt quality:**
- Symbol set can differ materially from prior runs (especially historical runs), changing class balance and event quality.
- This can reduce AUC/lift if previously selected symbols had more favorable signal-to-noise.

## Lower-confidence / secondary contributors

### 5) MDI selector base model changed (shallower ExtraTrees)
- `ExtraTreesRegressor` changed to `max_depth=6` and fewer trees.
- Could weaken feature ranking quality and reduce downstream discriminative power.

### 6) Selection-metric fallback order changed
- Event scoring fallback now includes `range_24h_pct`.
- If this fallback is triggered, sample weighting can shift and alter model calibration / ranking behavior.

### 7) Triple-barrier implementation parallelization changes
- Dynamic `n_jobs` and scalar TP/SL fast path were introduced in label computation.
- Likely neutral semantically, but worth validating deterministic parity in case of subtle alignment/order interactions.

## Prioritized validation plan (fast)

1. **A/B rerun with forced label refresh ON vs OFF** under same `run_id`/timestamp context.
2. **A/B rerun with `n_splits=3` vs `n_splits=2`** for base race only.
3. **A/B rerun with geometry calibration toggles reverted** (`label_edge_term_cap`, balance penalty, edge weight block).
4. Compare per-bucket:
   - ROC_AUC, PR_AUC, Lift@20, Brier/BSS,
   - label composition TP/SL/TO rates,
   - selected model + selected feature sets.

## Suggested rollback order

1. Re-enable mandatory label refresh before base training.
2. Restore base race to 3 CV splits.
3. Temporarily revert geometry reweighting block to prior formula and re-evaluate.
4. Keep diagnostics logging, but gate deployment decisions on refreshed-label runs only.


## Direct answer: labeling and `compare_tbm_parameters.py`

Yes — both are plausible and likely connected.

- `compare_tbm_parameters.py` writes `offline_optimisers/reports/tbm_best_params.csv` and `tbm_geometry_grid.csv`.
- At train time, `apply_offline_optimizer_best_params(...)` auto-loads `tbm_best_params.csv` and injects TBM geometry into runtime config (`barrier_k_tp`, `barrier_sl_base_mult`, `barrier_tp_*`, `barrier_sl_*`, `barrier_atr_window`, `label_horizon_*`).
- Label generation/training then uses those injected settings.

So any objective/pool-selection changes in `compare_tbm_parameters.py` can change deployed barrier geometry, which changes label distributions and therefore downstream Lift/AUC/Brier.

### Specific 48h `compare_tbm_parameters.py` risk points

1. Optuna objective switched to `stage2_score` path and search-space behavior changed (continuous suggestions).
2. Candidate pool is now filtered to top-50% by `stage2_score` before diversity selection.
3. Structural/final selection now preferentially runs on that filtered pool with fallback logic.

These can materially alter the winning `tbm_best_params.csv` row even if code compiles and pipeline runs.

### Coupled failure mode to watch

- If labels are **not refreshed** (current train behavior when artifacts already exist), you can end up with stale labels inconsistent with newly injected TBM params and updated training stack.
- If labels **are** refreshed, but new TBM winner is less calibration-friendly, label composition may shift and degrade Brier/Lift.

This is why the first A/B should isolate:
- old vs new `tbm_best_params.csv`, and
- refresh-labels ON vs OFF.
