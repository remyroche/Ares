# ✅ Training Approach Fixed

## 🔴 **Root Cause of Model Collapse**

The model was predicting ~0.81 for everything due to **triple filtering**:

1. **Top 30% filtering** (filter_percentile=70) → kept only quality >= 70th percentile
2. **Additional filter**: Removed quality < 0.25
3. **Additional filter**: Removed strength < 0.4
4. **Extreme weights**: 0.1x to 3.0x (30x range)

**Result:** Almost all training data in 0.7-1.0 range → model learned to predict mean (~0.81)

## ✅ **Fix Applied**

### **1. NO Hard Filtering**
- Changed `filter_percentile` from `70.0` to `100.0`
- Removed additional quality/strength filters
- Keep ALL data with variance intact

### **2. Gentler Confidence Weights**
```python
# OLD (too extreme - 30x range):
Noise (0-0.3):      0.1x
Weak (0.3-0.5):     0.3x
Medium (0.5-0.7):   0.7x
Strong (0.7-0.85):  1.5x
Critical (0.85+):   3.0x

# NEW (gentle - 6.7x range):
Noise (0-0.3):      0.3x  ← 3x increase
Weak (0.3-0.5):     0.5x  ← 1.7x increase
Medium (0.5-0.7):   0.8x  ← 1.1x increase
Strong (0.7-0.85):  1.2x  ← 0.8x decrease
Critical (0.85+):   2.0x  ← 0.67x decrease
```

### **3. Why This Works**

**Old Approach:** Filter → Narrow range → Model predicts mean
- After filtering: Most data in 0.7-1.0 range
- Limited variance → easy to memorize
- Model collapse: Predict ~0.81 for everything

**New Approach:** Keep all data → Preserve variance → Model learns features
- Full range: 0.0-1.0
- High variance → must learn features to discriminate
- Weights guide learning without removing signal

## 📊 **Expected Results**

With full variance preserved:
- **Spearman ρ**: Should improve from 0.386 to >0.60
- **Separation**: Should improve from 0.004 to >0.25
- **Future R²**: Should improve from -0.538 to >0.30
- **Precision@K**: Should maintain ~100% (already excellent)

## 🎯 **Files Modified**

1. **`src/tactician/sr_levels/ml_quality/sr_quality_model.py`**
   - Changed filtering logic to preserve data
   - Added safeguards against over-filtering

2. **`src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py`**
   - Reduced weight extremes (30x → 6.7x range)

3. **`scripts/run_sr_workflow.py`**
   - Changed `filter_percentile` from 70.0 to 100.0
   - Updated logging messages

4. **`train_sr_quality_model.py`**
   - Removed hard filtering
   - Use weighted data directly

---

**Key Principle:** For ranking problems, **preserve variance** and use **gentle weighting**, don't filter aggressively.

