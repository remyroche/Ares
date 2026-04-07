# Consensus Soft Labels Upgrade Report

## 1. Summary
The label-generation path for the `tbm_500_250` and `tbm_250_125` meta-model heads was upgraded to use geometry consensus soft labels derived from multiple evaluated triple-barrier configurations. All other heads and unrelated metrics preserve their previous hard-label configurations, strictly isolated from these changes.

## 2. Soft-Target Training
- **Multiclass Cross-Entropy:** Replaced the previous 3-regressor hack with an authentic soft-target multiclass cross-entropy optimization approach.
- **XGBoost:** Modified the base `MetaClassifierModel` candidate handler for `xgb_parallel_forest` (`xgb_clf`) to pass a custom-written softmax cross-entropy objective (`_soft_crossentropy_obj`) when it detects continuous multi-class targets. This calculates exact analytical gradients (`prob - y_soft`) in logit space and hessians (`prob * (1 - prob)`) applying numerically stable softmax normalization (`np.exp(preds - np.max(preds))`).
- **Unsupported Models:** Intentionally disabled support for `et_clf`, `catboost_clf`, and `ridge_clf` for soft probability targets per the instructions (they default to 3-regressor or raise exceptions if attempting unsupported outputs).

## 3. Leakage Controls
- **Geometry Selection Isolation:** We completely eliminated cross-fold data leakage in generating candidates. An object called `DynamicSoftLabels` defers processing; during cross-validation, candidate geometries are empirically filtered solely using training indices, then these explicitly retained geometries are used to evaluate identical bounds on the respective validation folds.
- **ATR Causality Validation:** We verified the computation logic for `ATR_1h`. The underlying code uses `numba_atr_no_norm` which solely applies a backward-looking EWMA. Thus, `ATR_1h` (and scaled `ATR_h`) is strictly causal and leaks no forward-looking information. This invariant is now explicitly asserted and documented during candidate geometry generation.

## 4. Edge-Case Hardening
- **Safeguards added:**
  - `validate_probability_simplex` enforces output boundaries on probabilities, preventing precision accumulation bugs by strictly clipping at `1e-12` and normalizing. Runtime assertions (`assert np.allclose(pred.sum(axis=1), 1.0, atol=1e-6)`) enforce output boundaries definitively downstream inside OOF generation and inference wrappers.
  - Zero retained geometries triggers a fail-safe loud `ValueError` exception exposing the exact counts of validation rule rejections.
  - Retained geometries < 3 trigger a specific warning during generation.
  - Target entropy profiles are actively measured: overly confident (mean entropy < 0.2) or overly indecisive (mean entropy > 1.4) label distributions log diagnostic warnings. Extreme class collapses (`mass < 1%`) also warn loudly.

## 5. Consensus Architecture and Geometry Logic
- **Tie Resolution and Horizon Alignment:** The horizon `h` establishes the upper bound limit for timeout `TO = h`. Hit occurrences ensure time to valid excursion `t <= h`. When multiple barriers flag valid hits (a collision), the first-touch chronologically (`t_mfe < t_mae` or reversed) explicitly breaks the tie to accurately emulate sequential touch.
- **Aggregation:** Soft labels form directly from hard voting outcomes of retained geometries, guaranteeing target sets naturally inhabit a complete simplex array without artificial smoothing applied outside of probability consensus fractions.

## 6. Class-Order Consistency
- **Canonical Order:** Hardcoded and universally enforced `CLASS_ORDER = ("TP", "SL", "TO")` (Index 0 = TP, Index 1 = SL, Index 2 = TO).
- **Adaptations:** We aligned `policy_ml.py` expected utility calculations and all intermediate classifier logic directly to this mapping (`MetaClassifierModel._compute_clf_metrics`), converting from older structures without resorting to implicit hidden data broadcasting mapping hacks.

## 7. Metrics and Diagnostics
- **Evaluation logs now capture:**
  - Fold candidate geometry counts (Evaluated, Retained, Rejected by `base_rates`, `value_bounds`, etc.).
  - True `log_loss` multiclass cross-entropy computed correctly for target soft distributions using `1e-12` clipped epsilon guards.
  - Explicit reporting of target probability distributions (`mass` for each class) and system entropy (mean and variance).

## 8. Risks / Caveats
- Out-of-the-box `log_loss` via sklearn and native classification objective metrics like `accuracy` often struggle fundamentally when true targets are provided as probabilities. We bypass this with custom implementations, but some metrics (e.g., standard `roc_auc_score` with OVR) cannot handle true soft targets and fail silently/continue via the try/except wrappers (this is intended, relying primarily on cross-entropy log-loss instead).
- Only XGBoost is configured to utilize the proper cross-entropy softmax framework for the soft labels right now.

## 9. Validation Summary
- `test_model.py` simulation validates complete pipeline: custom objective gradients calculate perfectly, probability matrices are enforced dynamically in `_cv_evaluate`, and soft label logs identify candidate geometries dynamically per split without failure.
