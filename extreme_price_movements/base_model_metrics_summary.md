# Base Model Metrics Summary (Latest Training Run)

This note summarizes the final base-model metrics captured at the end of `training_log.txt`.

## Stage-gate outcome

- Alpha stage-gate result: **0/12 passed** (minimum required: 6).
- All 12 evaluated models were marked `passed=False`.

## Alpha stage-gate metric ranges

Across the 12 evaluated entries in the stage-gate table:

- `PR_AUC`: 0.7426 to 0.7811
- `Brier_Imp`: 0.001675 to 0.009646
- `Lift_k`: 1.0390 to 1.0583
- `CV_Prec_k`: 0.003172 to 0.005645

## Base winner quality (by bucket family)

- `long_tf_*` winner: `xgboost`
  - AUC 0.5019, IC 0.0005, LogLoss 0.6407, PR-AUC 0.7405, Lift@20 1.043, BrierImp -14.2%
- `short_mr_*` winner: `catboost`
  - AUC 0.5045, IC 0.0066, LogLoss 0.6401, PR-AUC 0.7424, Lift@20 1.050, BrierImp -14.1%
- `long_mr_*` winner: `catboost`
  - AUC 0.5075, IC 0.0085, LogLoss 0.6205, PR-AUC 0.7735, Lift@20 1.040, BrierImp -19.6%
- `short_tf_*` winner: `xgboost`
  - AUC 0.5061, IC 0.0071, LogLoss 0.6207, PR-AUC 0.7719, Lift@20 1.034, BrierImp -19.6%

## Interpretation

- Discriminative performance is only marginally above random (AUC values are close to 0.5).
- Lift is positive but small (~1.03 to 1.05).
- Reported Brier improvements are negative in the winner table, indicating weaker probability quality versus reference baseline for this run.
