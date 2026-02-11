# Change Audit: `34e0a3ff956a14df5cd78c09784054cc2cdf56db` -> `HEAD`

## Scope and size

- **Total files changed:** 1,892
- **Status breakdown:** 1,869 added, 23 modified
- **By top-level path:** `cache/` 1,862 files, `extreme_price_movements/` 28 files, `.jules/` 1 file, `tests/` 1 file
- **Largest bulk change:** `cache/ffd_columns/**` with **1,859** newly added `.npy` files (464 each under `open/`, `high/`, `low/`, `close/`, plus 3 under `default/`)

Artifacts in this folder:
- `change_audit_34e0a3f_to_head_files.txt` — full exhaustive name-status file list
- `change_audit_34e0a3f_to_head_commits.txt` — chronological commit list in range

---

## Commit-by-commit (chronological)

1. `09012f9` Optimize numba_zscore with fused kernel and improve stability
2. `50bc669` Merge PR #3597
3. `1af0e26` Update optimise bucket params path and train fallback loading
4. `ccb360b` Merge PR #3602
5. `15f3a7b` Implement report recommendations: decoupled engine, exhaustion filters, new features
6. `17de4f1` Feature selection update: 5x ElasticNet prescreen + MDI docs
7. `328165b` Finalize report implementation
8. `eb0eec5` Merge PR #3604
9. `0d2355d` Add quantile feature selection module
10. `4fa5ae3` Finalize report implementation
11. `37a4f30` Refactor RSI and ATR with fused parallel numba kernels
12. `ca48f0b` Finalize report implementation
13. `112b4eb` Merge PR #3603
14. `81cd9d8` Merge PR #3605
15. `08208d8` Tail-aware quantile meta-model acceptance criteria
16. `cf4519c` Merge PR #3606
17. `3641cf8` Optimization reporting enhancement (CSV)
18. `7b5348a` Merge PR #3607
19. `b7c5b26` Add TF/MR meta and alpha features
20. `cdc5767` Add model stage gates
21. `fa8e205` Add model stage gates
22. `0887561` Add/optimize TF-MR alpha features
23. `5c96578` Merge PR #3609
24. `15cb39d` Add TF/MR meta and alpha features (optimized)
25. `16440d6` Merge PR #3608
26. `dedb242` Morning commit: quantile and interaction gates (+ bulk cache files)

---

## Exhaustive non-cache code/data changes (28 files under `extreme_price_movements/`, plus 2 utility paths)

### Pipeline orchestration and data flow

- `extreme_price_movements/run_pipeline.py`
  - Added shared bucket params path (`extreme_price_movements/artifacts/models/bucket_params.json`) with legacy fallback copy behavior.
  - Train path now attempts to load bucket params from shared or legacy path.
- `extreme_price_movements/optimise.py`
  - Per-step trial logs are collected and written to consolidated CSV next to JSON output.
- `extreme_price_movements/data_store.py`
  - Pathing and data artifact handling updated (decoupling/report-driven adjustments).
- `extreme_price_movements/pipeline_steps.py`
  - Minor pipeline step wiring update.

### Signal generation and model behavior

- `extreme_price_movements/engine.py`
  - Switched from net-score coupling (`long - short`) to **decoupled independent signal checks** per regime, with explicit conflict suppression (if both long/short fire, take none).
- `extreme_price_movements/meta_model.py`
  - Full rewrite to quantile-focused framework:
    - Quantile GAM-based monotonicity discovery
    - Interaction discovery
    - Quantile-model candidate racing (xgboost/lightgbm)
    - Fold calibration logic and acceptance diagnostics
- `extreme_price_movements/model_race.py`
  - Added race report metrics output for generic meta model race.
- `extreme_price_movements/training.py`
  - Added stage-gate evaluation for alpha and meta models via `gate_metrics.compute_stage_gate_metrics` and reporting summaries.
- `extreme_price_movements/gate_metrics.py` (new)
  - Introduces formal gate pass/fail logic for alpha and quantile meta models.

### Feature engineering and selection

- `extreme_price_movements/features.py`
  - Removed joblib-level feature-cache path for full panel hashing; moved to direct compute + persisted parquet flow.
  - Added per-column FFD incremental cache (`cache/ffd_columns`) with hash-keyed results and cached `d_opt` reuse.
  - Added many new exhaustion/risk/liquidity/TF/MR/alpha features.
  - Added dynamic gated feature family generation + threshold selection.
- `extreme_price_movements/gated_features.py`
  - Added panel-native gated feature generation and dynamic threshold selection helpers.
  - Expanded strict thresholds from `{66,85}` to `{25,50,66,75,85,90}`.
- `extreme_price_movements/config.py`
  - Added many new raw and meta features to feature lists and gate feature names.
- `extreme_price_movements/feature_selection_extreme_events.py`
  - ElasticNet pre-screen generalized and changed from 3x to 5x target feature multiplier.
- `extreme_price_movements/quantile_feature_selection_extreme_events.py` (new)
  - Added quantile-aware feature selection module used by meta modeling.

### Performance kernels/tests

- `extreme_price_movements/fast_funcs.py`
  - RSI and ATR kernels rewritten into fused 1D + parallel matrix variants.
- `extreme_price_movements/tests/test_fast_funcs_indicators.py` (new)
- `extreme_price_movements/tests/test_fast_funcs_zscore.py` (new)
- `extreme_price_movements/tests/test_new_logic.py` modified

### TP/SL optimizer subsystem

- `extreme_price_movements/tpsl_optimiser/metrics_utils.py` (new)
- Modified:
  - `10_tp_sl_calibration.py`
  - `20_loss_limiter_opt.py`
  - `30_profit_exit_opt.py`
  - `40_position_sizing_opt.py`
- These now expose richer metrics and trial tables aligned to step-level reporting.

### Reports/artifacts committed

- `extreme_price_movements/artifacts/models/bucket_params.json` (new)
- `extreme_price_movements/artifacts/models/bucket_params.csv` (new)
- `extreme_price_movements/reports/meta_model_generic_metrics.csv` (new)
- `extreme_price_movements/reports/meta_model_generic_race.csv` (new)
- `extreme_price_movements/reports/20260204_220000/training_report.md` modified

### Other changed files

- `.jules/bolt.md` modified
- `tests/inspect_data_values.py` (new)

---

## Why your metrics likely moved (root-cause hypotheses mapped to observed deltas)

### 1) Why several **Rw-AUC / IC improved**

Most likely contributors:

1. **Much larger feature space + new directional features**
   - New TF/MR/alpha features and exhaustion/risk/liquidity features likely increased ranking power in specific regimes.
2. **Dynamic gate threshold feature selection**
   - Replacing fixed gate thresholds with selected thresholds can raise rank metrics if thresholds better fit current distribution.
3. **Meta model redesign to quantile framework + feature selection changes**
   - Quantile-tail-aware modeling may improve top-tail ranking behavior and Rw-AUC while changing calibration characteristics.
4. **Engine decoupling**
   - Independent long/short signal checks change trade candidate composition (and therefore measured rank/IC distributions).

### 2) Why **ECE worsened across many models**

Most likely contributors:

1. **Target/model objective mismatch**
   - Quantile/ranking-oriented training and feature race objectives optimize ordering/tail utility, not probability calibration.
2. **Expanded and more non-linear feature interactions**
   - More expressive models can improve discrimination while becoming poorly calibrated without explicit post-calibration.
3. **Threshold/gate logic changes**
   - Dynamic thresholded gates alter score distribution shape and class prevalence, which can inflate ECE bins.

### 3) Why **meta stage gates underperformed vs Feb 10**

Most likely contributors:

1. **Stricter or newly explicit pass/fail stage gate framework** now enforced in code.
2. **Meta rewrite changed optimization target** toward quantile-tail metrics and constraints, potentially sacrificing pure IC in some legs.
3. **Monotone + interaction constraints discovery may be over-constraining** for current sample.
4. **Coverage/Spread checks can fail even when some ranking metrics improve** (different objective families).

### 4) Why **short_mr mixed signals** (Rw-AUC up, OOF IC down)

Plausible explanation:

- Changes to gate selection, decoupled signal logic, and new exhaustion features can improve top-tail ranking (AUC-like behavior at selected horizons) but reduce global monotonic correlation (OOF IC), especially if tail performance improves at the expense of mid-distribution ordering.

---

## Immediate validation checks to run next (recommended)

1. **Ablation by family** (turn off one family at a time):
   - New TF/MR alpha features
   - Exhaustion/risk/liquidity features
   - Dynamic gate threshold selection
   - Quantile meta selector/model stack
2. **Calibration-only pass**:
   - Keep model fixed, run isotonic or temperature/binwise recalibration and re-measure ECE.
3. **Re-run with fixed gate thresholds (66/85 only)** as control.
4. **Meta constraints sensitivity**:
   - No constraints vs monotone only vs monotone+interaction.
5. **Engine logic A/B**:
   - Legacy net-score coupling vs decoupled independent-signal logic.

