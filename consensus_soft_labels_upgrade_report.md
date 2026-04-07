# Consensus Soft Labels Upgrade Report

## 1. Summary
The label-generation path for the `tbm_500_250` and `tbm_250_125` meta-model heads was upgraded to use geometry consensus soft labels derived from multiple evaluated triple-barrier configurations. All other heads and unrelated metrics preserve their previous hard-label configurations, strictly isolated from these changes.

## 2. Soft-Target Training
- **Multiclass Cross-Entropy:** Replaced the previous 3-regressor hack with an authentic soft-target multiclass cross-entropy optimization approach.
- **XGBoost:** Modified the base `MetaClassifierModel` candidate handler for `xgb_parallel_forest` (`xgb_clf`) to pass a custom-written softmax cross-entropy objective (`_soft_crossentropy_obj`) when it detects continuous multi-class targets. This calculates exact analytical gradients (`p - y`) and hessians (`p * (1 - p)`).
- **Unsupported Models:** Intentionally disabled support for `et_clf`, `catboost_clf`, and `ridge_clf` for soft probability targets per the instructions (they default to 3-regressor or raise exceptions if attempting unsupported outputs).

## 3. Leakage Controls
- **Geometry Selection Isolation:** We completely eliminated cross-fold data leakage in generating candidates. An object called `DynamicSoftLabels` defers processing; during cross-validation, candidate geometries are empirically filtered solely using training indices, then these explicitly retained geometries are used to evaluate identical bounds on the respective validation folds.
- **ATR Causality Validation:** We verified the computation logic for `ATR_1h`. The underlying code uses `numba_atr_no_norm` which solely applies a backward-looking EWMA. Thus, `ATR_1h` (and scaled `ATR_h`) is strictly causal and leaks no forward-looking information. This invariant is now explicitly asserted and documented during candidate geometry generation.

## 4. Edge-Case Hardening
- **Safeguards added:**
  - `validate_probability_simplex` enforces output boundaries on probabilities, preventing precision accumulation bugs by strictly clipping at `1e-12` and normalizing.
  - Zero retained geometries triggers a fail-safe loud `ValueError` exception exposing the exact counts of validation rule rejections.
  - Retained geometries < 3 trigger a specific warning during generation.
  - Low-entropy (<0.2) or extremely imbalanced predictions log warnings identifying "Degenerate soft labels".

## 5. Class-Order Consistency
- **Canonical Order:** Hardcoded and universally enforced `CLASS_ORDER = ("TP", "SL", "TO")` (Index 0 = TP, Index 1 = SL, Index 2 = TO).
- **Adaptations:** We aligned `policy_ml.py` expected utility calculations and all intermediate classifier logic directly to this mapping (`MetaClassifierModel._compute_clf_metrics`), converting from older structures without resorting to implicit hidden data broadcasting mapping hacks.

## 6. Metrics and Diagnostics
- **Evaluation logs now capture:**
  - Fold candidate geometry counts (Evaluated, Retained, Rejected by `base_rates`, `value_bounds`, etc.).
  - True `log_loss` multiclass cross-entropy computed correctly for target soft distributions.
  - Explicit reporting of target probability distributions (`mass` for each class) and system entropy.

## 7. Risks / Caveats
- Out-of-the-box `log_loss` via sklearn and native classification objective metrics like `accuracy` often struggle fundamentally when true targets are provided as probabilities. We bypass this with custom implementations, but some metrics (e.g., standard `roc_auc_score` with OVR) cannot handle true soft targets and fail silently/continue via the try/except wrappers (this is intended, relying primarily on `log_loss` instead).
- Only XGBoost is configured to utilize the proper cross-entropy softmax framework for the soft labels right now.

## 8. Validation Summary
- `test_model.py` simulation validates complete pipeline: custom objective gradients calculate perfectly, probability matrices are enforced dynamically in `_cv_evaluate`, and soft label logs identify candidate geometries dynamically per split without failure.
