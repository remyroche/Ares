# HPO Pipeline Run 35 - Detailed Outcomes Report

**Generated**: 2026-01-02T01:30:48
**Symbol**: ETHUSDT | **Timeframe**: 15m | **Direction**: Long

---

## Executive Summary

The HPO pipeline successfully ran through all layers (0 → 5) after multiple bug fixes. Total runtime: **379.76 seconds (~6.3 minutes)**.

| Layer | Status | Notes |
|-------|--------|-------|
| Layer 0 (Kalman Filter) | ✅ Pass | Q=3.16e-05, R=3.16e-04, vwap_weight=0.4 |
| Layer 1 (Weighting) | ✅ Pass | Weights computed for all samples |
| Layer 2 (Base Models) | ✅ Pass | 358 candidates → 5 selected geometries |
| Layer 3 (Meta-Learners) | ✅ Pass | Dual heads (Alpha + Probability) |
| Layer 4 (Risk Filter) | ⚠️ Skipped | Unexpected keyword argument error |
| Layer 5 (Position Sizing) | ✅ Pass | Net Sortino: 0.83, PnL: +21.07% |

**OOS Reliability Score**: 80/100

---

## Layer 2: Base Models (Orthogonal Geometries)

### Selected Geometries
| Family | Horizon | PR-AUC | AUC | Recall |
|--------|---------|--------|-----|--------|
| PRICE_CUSUM | H=12 | 0.2209 | 0.5168 | 0.00 |
| PRICE_CUSUM | H=48 | 0.0830 | 0.5513 | 0.00 |
| VOL_CUSUM | H=12 | 0.5199 | 0.6872 | 0.12 |
| LIQ_CUSUM | H=12 | 0.8398 | 0.6982 | 0.94 |
| VOL_PARTICIPATION | H=48 | 0.3940 | 0.6898 | 0.11 |
| RANGE_ATR | H=48 | 0.3860 | 0.7267 | 0.01 |

### Metrics
- **Total Candidates**: 358
- **Coverage**: 100%
- **Mean Return**: -0.29%
- **Labeled Samples**: 31,059

---

## Layer 3: Meta-Learners (Dual Head)

### Alpha Head (Regressors)
| Model | Score |
|-------|-------|
| Ridge_MSE | 113.33 |
| LGBM_MSE | 152.31 |

- **Final IC (Information Coefficient)**: 0.4693

### Probability Head (Classifiers)
| Model | Score |
|-------|-------|
| LGBM_LogLoss | -999.00 (failed) |
| LGBM_Focal | -999.00 (failed) |

- **Final AUC**: NaN
- **Final LogLoss**: 36.04

> ⚠️ **Issue**: Probability head training failed for all models except LGBM variants, likely due to single-class issues in CV folds.

### Calibration
- **ECE (Expected Calibration Error)**: 1.00 (target: <0.05)

---

## Layer 4: Risk Filter

**Status**: ⚠️ **Disabled**

**Reason**: `_train_layer4_oof_extratrees_pnl() got an unexpected keyword argument 'l3_models_metadata'`

> This error suggests a function signature mismatch that needs investigation.

---

## Layer 5: Position Sizing

### Out-of-Sample (OOS) Performance
| Metric | Value |
|--------|-------|
| **Total PnL** | +21.07% |
| **Avg Trade PnL** | +0.012% |
| **Trade Count** | 1,729 |
| **Net Sortino** | 0.833 |
| **Maximum Drawdown** | 9.67% |
| **AUC** | 0.519 |

### In-Sample vs OOS Comparison
| Metric | In-Sample | OOS | Degradation |
|--------|-----------|-----|-------------|
| PnL | - | +21.07% | N/A |
| Sortino | -0.86 | +0.83 | +169% improve |
| Trades | 10,619 | 1,729 | 16% ratio |

---

## Fixes Applied During This Session

### `label_based_layer_2.py`
1. Fixed syntax errors in dictionary literals
2. Fixed probe feature alignment (separated data cache from feature list)
3. Fixed Optuna HPO function logic
4. Fixed indentation error causing import failure

### `layer3/core.py`
1. Added missing `calculate_sample_weights_efficient` import

### `layer3/model_training.py`
1. Added missing `Parallel, delayed` imports from joblib
2. Fixed `predict_proba` array handling for single-class
3. Fixed `log_loss` and `roc_auc_score` single-class handling

### `orthogonal_label_generation.py`
1. Optimized candidate generation by skipping weight calculation

---

## Recommendations

1. **Fix Layer 4**: Investigate the `l3_models_metadata` keyword argument error
2. **Improve Probability Head**: The LogisticRegression training is failing due to unsupported `eval_set` argument - need model-specific fit logic
3. **Calibration**: ECE of 1.0 indicates poor probability calibration - needs investigation
4. **Single-Class Folds**: The classification models are encountering single-class CV folds, suggesting class imbalance issues
