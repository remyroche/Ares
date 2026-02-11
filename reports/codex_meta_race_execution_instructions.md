# Codex Execution Instructions: Meta Race, Calibration Contract, Horizons, and Targets

Use this as the **exact implementation brief**.

---

## Scope
Implement the items below in:
- `extreme_price_movements/model_race.py`
- `extreme_price_movements/meta_model.py`
- `extreme_price_movements/feature_selection_extreme_events.py`
- `extreme_price_movements/features.py`
- `extreme_price_movements/training.py`
- tests under `extreme_price_movements/tests/`

Do not modify unrelated reports/artifacts unless tests regenerate deterministic outputs.

---

## 1) Calibration bias contract in `ModelRace.predict_proba` (must be explicit + persisted)

### Problem contract
During CV pipeline:
1. fold model trained with sample weights -> raw probs are prevalence-biased
2. fold raw probs corrected to unweighted prevalence domain (odds mapping)
3. isotonic fit on corrected OOF probs

Therefore inference pipeline must always be:
1. final model raw probs (biased)
2. apply **same correction contract** used during OOF
3. pass corrected probs to isotonic
4. output calibrated probs

### Required implementation
In `ModelRace`:
- Add explicit serialized calibration contract object, e.g.:
  - `self.bias_contract_ = {`
    - `"method": "odds_scale"`,
    - `"target_unweighted_prevalence": float`,
    - `"weighted_prevalence": float`,
    - `"factor": float`,
    - `"eps": 1e-9`
  - `}`
- Keep `final_bias_factor_` only as derived convenience; source of truth is `bias_contract_`.
- In `fit()`:
  - compute and store `weighted_prevalence` (training weights), unweighted prevalence, and factor.
  - store contract **before return**.
- In `predict_proba()`:
  - assert `bias_contract_` exists; if missing raise explicit `RuntimeError` with actionable message.
  - apply correction from contract to raw probs **before isotonic**.
  - if isotonic missing, still return bias-corrected probs.
- Add helper function(s):
  - `_build_bias_contract(raw_probs, y_hard, sample_weight)`
  - `_apply_bias_contract(raw_probs, contract)`
- Ensure object is joblib/pickle serializable.

### Required unit test
Add `extreme_price_movements/tests/test_model_race_calibration_contract.py` with synthetic data:
- generate binary outcomes with low base prevalence (e.g., ~5-10%).
- generate non-uniform sample weights that inflate weighted prevalence.
- fit `ModelRace`.
- assert:
  - `bias_contract_` exists and has required keys.
  - mean(corrected raw probs before isotonic) ≈ unweighted prevalence (tight tolerance, e.g. <= 0.02).
  - isotonic output mean does not diverge wildly (e.g. abs(delta_mean)<=0.05).
  - top-bin ECE does not explode when comparing OOF-calibrated path vs full-model predict path.

---

## 2) Re-add Ridge + ExtraTrees in meta race + top-decile calibration metric

### Required implementation
In `meta_model.py`:
- Keep quantile candidates.
- Include non-quantile regressors in race:
  - Ridge
  - ExtraTreesRegressor
- Add per-candidate metric:
  - `top_decile_calibration_gap = |mean(pred_top10) - mean(target_top10)|`
  - optionally add `top_decile_ece` (binning only top decile subset)
- include this metric in race report rows and CSV.

### Feature selection multiplier adjustment
In `feature_selection_extreme_events.py`:
- change ElasticNet prescreen multiplier from 5x to **4x**.
- keep parameterized API but set default and call-sites to 4.

### Additional linear target variant for Ridge/Lasso style
In `meta_model.py` add candidate family using transformed target:
- target:
  - `y_rank = signed_log_demeaned(ret_H)` (monotone with returns but drift-robust)
- sample weights:
  - top-tail emphasis, e.g. `w = 1 + lambda_tail * ramp(rank_percentile, start=0.7, end=1.0)`
- fit Ridge (and optional lasso/elasticnet if available) on `(X, y_rank, w)`
- evaluate in same race framework.

---

## 3) Meta model race redesign: include weighted top-rank regression variants

### Candidate pool requirements
Maintain low-compute race + higher-compute final train.

Race pool must include:
1. quantile regressors (current + regularization improvements)
2. basic regressors (Ridge + ExtraTrees)
3. weighted top-rank regression variants for:
   - Ridge
   - LGBMRegressor
   - ExtraTreesRegressor
   - XGBRegressor

All are regressors (not classifiers).

### Objectives/metrics
Use OOF-based business metrics and top-tail behavior:
- IC / Spearman
- utility at top-k (k in [10%, 30%])
- top-decile calibration gap (from section 2)
- downside control metric (e.g., ES10 on selected tail)

Define a composite race score with explicit weights and log each component.

### Compute policy
- Race mode: small estimators/trees + aggressive early stopping.
- Final mode: larger estimator counts and full training budget.
- Keep HPO bounded with pruning + timeout.

---

## 4) Regularization + `num_parallel_tree`

### Required changes
- XGB default `num_parallel_tree = 10`.
- XGB HPO search range `num_parallel_tree = [5, 500]`.
- Ensure aggressive early stopping and pruning in HPO objective:
  - shorter warmup
  - median pruner enabled
  - cap trial runtime and total timeout

Keep previously requested regularization ranges:
- XGB `reg_alpha` high range up to 20
- XGB `reg_lambda` high range up to 100
- LGBM `lambda_l1` up to 20
- LGBM `lambda_l2` up to 100

---

## 5) Horizon mismatch infrastructure (H=2/4/8 as separate base models)

### Required architecture
In `training.py` and dependent wiring:
- train alpha/base models **separately per horizon H in {2,4,8}**.
- keep per-H feature selection and model artifact.
- do **not** collapse into a single chosen H.
- meta input must consume:
  - `pred_H2`, `pred_H4`, `pred_H8`
  - plus context features

### Data contract
- model bundle should expose per-horizon models explicitly.
- inference path should compute and pass all per-H predictions to meta model.

---

## 6) Gate target misalignment fix

In `features.py`, for gate feature selection target proxy:
- replace current proxy with:
  - `target_proxy = 0.3*ret2 + 0.4*ret4 + 0.3*ret8`
- ensure train mask tail exclusion matches max horizon (8 bars).

---

## 7) Deliverables and acceptance criteria

### Code deliverables
- Updated modules listed in Scope.
- New/updated tests:
  - `test_model_race_calibration_contract.py`
  - adjust existing tests for changed horizon/meta interfaces.

### Functional acceptance
1. `ModelRace.predict_proba` applies persisted bias contract before isotonic.
2. Missing contract raises explicit error.
3. Meta race includes quantile + non-quantile + weighted-top-rank variants.
4. Race report includes top-decile calibration metric.
5. ElasticNet prescreen effectively uses 4x target count.
6. Per-horizon base predictions (H2/H4/H8) flow into meta training/inference.
7. Gate target proxy equals `0.3*ret2 + 0.4*ret4 + 0.3*ret8`.

### Validation checklist (run and paste output)
- `python -m py_compile extreme_price_movements/model_race.py extreme_price_movements/meta_model.py extreme_price_movements/features.py extreme_price_movements/training.py extreme_price_movements/feature_selection_extreme_events.py`
- `PYTHONPATH=. pytest -q extreme_price_movements/tests/test_model_race_calibration_contract.py`
- `PYTHONPATH=. pytest -q extreme_price_movements/tests/test_new_logic.py extreme_price_movements/tests/test_fast_funcs_zscore.py`
- any additional horizon/meta integration tests.

---

## Copy-paste prompt for Codex

Implement the following in `/workspace/Ares`:

1) In `extreme_price_movements/model_race.py`, formalize and persist calibration-bias contract used between fold training and isotonic calibration. During inference, enforce pipeline: raw weighted-model probs -> bias correction mapping -> isotonic. Store contract fields (`method`, weighted/unweighted prevalence, factor, eps), assert presence in `predict_proba`, and ensure serializability.

2) Add `extreme_price_movements/tests/test_model_race_calibration_contract.py` on synthetic skewed-prevalence + weighted data. Validate corrected mean aligns with unweighted prevalence pre-isotonic, isotonic mean is stable, and top-bin ECE doesn’t blow up between OOF/full-model paths.

3) In `extreme_price_movements/meta_model.py`, keep quantile candidates and re-add non-quantile regressors (Ridge + ExtraTreesRegressor), plus L1 QuantileRegressor. Add top-decile calibration metric (`abs(mean(pred_top10)-mean(target_top10))`) to race reporting.

4) In `extreme_price_movements/feature_selection_extreme_events.py`, change ElasticNet prescreen effective multiplier from 5x to 4x (defaults + call-sites).

5) Add weighted top-rank regression variants (Ridge/LGBM/ExtraTrees/XGB): transformed target monotone with returns and tail-emphasized sample weights (e.g., rank ramp >=70%). Keep race low-compute and final training higher-compute.

6) Regularization/HPO updates: XGB default `num_parallel_tree=10`, HPO range `[5,500]`; preserve high regularization ranges (XGB alpha up to 20, lambda up to 100; LGBM L1 up to 20, L2 up to 100). Use aggressive early stopping and pruning.

7) Horizon infrastructure in `extreme_price_movements/training.py`: keep separate base models per H=2/4/8 and pass `(pred_H2,pred_H4,pred_H8)` into meta model with context features; no single-horizon collapse.

8) In `extreme_price_movements/features.py`, set gate target proxy to `0.3*ret2 + 0.4*ret4 + 0.3*ret8`; keep tail exclusion aligned to 8 bars.

9) Run validations:
- `python -m py_compile ...`
- `PYTHONPATH=. pytest -q extreme_price_movements/tests/test_model_race_calibration_contract.py`
- `PYTHONPATH=. pytest -q extreme_price_movements/tests/test_new_logic.py extreme_price_movements/tests/test_fast_funcs_zscore.py`

10) Commit with a clear message and provide a concise change summary + test results.
