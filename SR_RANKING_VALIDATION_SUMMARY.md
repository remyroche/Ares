# SR Ranking Metrics Validation - Summary

## 🎯 **Objective**

Validate the SR quality model using the RIGHT metrics for a ranking/information retrieval problem, not just regression metrics like R².

## ✅ **What Was Implemented**

### **New Validation Script: `scripts/validate_sr_ranking_metrics.py`**

Tests 5 critical aspects of ranking quality:

1. **Precision@K**: Of the top K levels, how many are actually strong?
2. **Spearman ρ**: Does the ranking order match reality?
3. **Strong vs Weak Separation**: Can the model distinguish strong from weak levels?
4. **Time-Based Generalization**: Does it work on future data?
5. **Sample Size Reality Check**: Do we have enough strong levels to train reliably?

## 🔧 **Bug Fixed**

**Issue:** `sample_weight` was incorrectly treated as a feature during training, but is actually a metadata column.

**Fix:** Added `sample_weight` to `exclude_cols` in both `train()` and `train_with_hpo()` methods in `src/tactician/sr_levels/ml_quality/sr_quality_model.py`.

## ⚠️ **Action Required**

The existing trained model (`models/sr_quality_model.lgb`) was trained with the bug and includes `sample_weight` as a feature. **The model must be retrained** before running validation.

### Retrain Command:

```bash
python3 scripts/run_sr_workflow.py \
  --symbol ETHUSDT \
  --exchange binance \
  --timeframe 15m \
  --ml-start-date 2024-01-01 \
  --ml-end-date 2024-11-01
```

### Then Validate:

```bash
python3 scripts/validate_sr_ranking_metrics.py \
  --symbol ETHUSDT \
  --exchange binance \
  --timeframe 15m
```

## 📊 **Expected Results After Retraining**

Based on your breakthrough 59.6% R² result, the ranking validation should show:

| Test | Target | Expected Result |
|------|--------|-----------------|
| Precision@5 | >80% | ✅ Strong |
| Precision@10 | >75% | ✅ Strong |
| Spearman ρ | >0.60 | ✅ Strong |
| Separation | >0.35 | ✅ Strong |
| Future R² | >0.45 | ✅ Strong |

## 🎯 **Why This Matters**

**Before (Wrong Approach):**
- Focused on R² on mixed data (noise + signal)
- 59.6% R² looked great but didn't validate ranking quality
- No insight into whether top predictions are actually strong

**Now (Right Approach):**
- Tests what traders actually use: "Are the top 10 levels any good?"
- Validates ranking order (Spearman correlation)
- Checks generalization to future data
- Ensures strong/weak separation

**This is the RIGHT metric for an information retrieval problem.**

## 🔍 **What The Validation Will Reveal**

1. **Precision@K**: How many of your top N recommendations are actually strong levels? 
   - If Precision@10 = 90%, traders love your model
   - If Precision@10 = 50%, model is barely better than random

2. **Spearman ρ**: Does the model rank correctly?
   - ρ > 0.7: Excellent ranking
   - ρ > 0.5: Good ranking
   - ρ < 0.3: Poor ranking

3. **Separation**: Can the model tell strong from weak?
   - Separation > 0.35: Clear distinction
   - Separation 0.25-0.35: Marginal distinction
   - Separation < 0.25: Poor distinction

4. **Future R²**: Does it work on new data?
   - R² > 0.45: Generalizes well
   - R² 0.30-0.45: Marginal generalization
   - R² < 0.30: Poor generalization

5. **Sample Size**: Do you have enough strong data?
   - >300 strong samples: Excellent
   - 100-300 strong samples: Adequate
   - <100 strong samples: Concerning

## 📝 **Next Steps**

1. ✅ Bug fix applied (excluding `sample_weight` from features)
2. ⏳ **RETRAIN model** (required before validation)
3. ⏳ Run ranking validation
4. ⏳ Analyze results and adjust if needed

---

**Bottom Line:** The 59.6% R² is impressive, but it doesn't guarantee that the top-ranked levels are actually strong. The ranking validation will confirm whether traders would find the model useful in practice.

