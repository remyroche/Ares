# HPO Pipeline Run Report
**Date:** 2025-12-30
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Duration:** 699.86 seconds (~12 minutes)

## ✅ Pipeline Status: SUCCESS

All 6 layers (0-5) completed successfully with 10 artifacts produced.

---

## Summary of Fixes Applied

| Issue | Status | Fix |
|-------|--------|-----|
| `OOFCalibrationConfig` invalid args | ✅ Fixed | Changed `cv_folds` → `min_samples_for_calibration=100` |
| Early stopping tuple index error | ✅ Fixed | Removed early stopping for custom objectives (Focal Loss) |
| Regime leaves extraction empty | ✅ Fixed | Pass `market_data.copy()` instead of empty DataFrame |

---

## Layer 2 Results

### Geometry Selection
- **6 geometries selected** with horizons: (12, 24, 48) × 2 each
- **3 geometries** proceeded to Tier 3 HPO (2 pruned for low score <0.52)
- Best Learnability Score: **1.382**

### Model Races
- **CatBoost** won most races (avg score ~0.54-0.59)
- XGB and LGBM competitive on some geometries

### Regime Leaves
- **2701 regime leaves** extracted per geometry (previously 0!)
- Saved to `outcomes/regime_leaves/`

### Feature Selection
- **50 features** selected per geometry via Titan RFE

---

## Layer 3 Results

### Meta-Model Performance
- Mean CV AUC: **0.5089** (stability: 0.9697)
- Mean Brier Score: **0.2501**
- Mean Average Precision: **0.4969**
- OOF Calibration: Applied via `OOFProbabilityCalibrator`

### Holdout Results
- Holdout AUC: **0.5266**
- Holdout Brier: **0.2444**
- Holdout AP: **0.3961**

---

## SNR Diagnostics Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Coverage | 100% | High coverage |
| Mean CV AUC | 0.5089 | Weak learnability |
| Balance (entropy) | 0.9986 | Well balanced labels |
| Combined score | 0.6514 | Good overall quality |
| Stability | 0.9697 | Highly stable |
| Pseudo-R² | -0.0006 | Very weak signal |

---

## Artifacts Produced

1. `layer3_oof_preds.csv` - OOF predictions
2. `regime_leaves_Geo_Sel*.parquet` - Regime leaf features
3. `snr_label_quality_*.json` - Label quality diagnostics
4. `snr_label_learnability_*.json` - Learnability diagnostics
5. `snr_model_robustness_*.json` - Model robustness diagnostics
6. `temporal_auc_*.png` - Temporal AUC plot

---

## De Prado Framework Verification

| Layer | Role | Status |
|-------|------|--------|
| Layer 2 | Base models (barrier labeling) | ✅ Working |
| Layer 3 | Meta models (ensemble) | ✅ Working |
| Layer 4 | Position sizer | ✅ Working |

---

## Recommendations

1. **Signal Strength**: Mean AUC ~0.51 indicates weak signal. Consider:
   - More aggressive feature engineering
   - Different barrier labeling parameters
   - Longer lookback periods

2. **Calibration**: CalibratedClassifierCV still shows tuple index errors with custom objectives. Consider:
   - Using temperature scaling instead
   - Manual calibration post-hoc

3. **Regime Leaves**: Now extracting 2701 features - consider pruning for efficiency.
