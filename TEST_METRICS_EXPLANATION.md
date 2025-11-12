# Test Set Metrics - Where to Find Them

**Date**: 2025-11-11 23:06  
**Status**: ⏳ Training in progress - test metrics pending

---

## ❓ YOUR QUESTION

> "Where is the test / walk forward accuracy score?"

The **0.7787 score** you see in the logs is the **cross-validation score during HPO**, not the final test set score.

---

## 📊 TRAINING FLOW & METRICS

### **Phase 1: HPO (Hyperparameter Optimization)** ✅ COMPLETE
**What happens**: Find optimal hyperparameters using TimeSeriesSplit CV

**Metrics shown**:
- LightGBM: Best CV score = **0.7787** (5-fold TimeSeriesSplit)
- CatBoost: Best CV score = **0.7791** (5-fold TimeSeriesSplit)

**These are NOT test set scores!** They are cross-validation scores used to select the best hyperparameters.

---

### **Phase 2: Final Training** ⏳ IN PROGRESS
**What happens**: Retrain models with optimized hyperparameters on train/val/test splits

**Metrics to be shown**:
```
✅ LightGBM trained: 100 iterations
   📊 Train R²: 0.XXXX, RMSE: 0.XXXX
   📊 Val R²: 0.XXXX, RMSE: 0.XXXX
   📊 Test R²: 0.XXXX, RMSE: 0.XXXX  ← THIS IS THE TEST SCORE YOU WANT
   ⚠️  Train-Test Gap: 0.XXXX (X.X%)
   ✅ Good generalization (overfitting ratio < 10%)
```

**Current status**: Training started at 23:03:39, still running

---

### **Phase 3: Report Generation** ⏳ PENDING
**What happens**: Generate comprehensive reports with all metrics

**Files to check**:
- `outcomes/analyst_base_ETHUSDT_15m_long_report_*.md`
- `outcomes/analyst_base_ETHUSDT_15m_long_metrics_*.json`
- `outcomes/analyst_base_ETHUSDT_*_training_report.md`

---

## 🔍 WHY TWO DIFFERENT SCORES?

### **CV Score (HPO Phase)**
- **Purpose**: Select best hyperparameters
- **Method**: 5-fold TimeSeriesSplit cross-validation
- **Data used**: Training + validation sets only
- **Score**: Average across 5 folds
- **Example**: 0.7787 (LightGBM)

### **Test Score (Final Training Phase)**
- **Purpose**: Evaluate true generalization
- **Method**: Single held-out test set (15% of data)
- **Data used**: Test set only (never seen during training or HPO)
- **Score**: Single evaluation on unseen data
- **Example**: Will be shown as "Test R²: 0.XXXX"

---

## 📈 EXPECTED TEST SCORES

Based on the CV scores, here's what to expect:

| Model | CV Score (HPO) | Expected Test R² | Reason |
|-------|----------------|------------------|--------|
| **LightGBM** | 0.7787 | 0.74-0.78 | Test usually 0-5% lower |
| **CatBoost** | 0.7791 | 0.74-0.78 | Test usually 0-5% lower |

**Why lower?**
- Test set is truly unseen data
- CV can be slightly optimistic
- This is CORRECT and expected behavior

---

## 🔎 HOW TO FIND TEST METRICS

### **Option 1: Live Logs** (Real-time)
```bash
# Watch for test metrics as they appear
tail -f logs/unified_*.log | grep -E "Train R²|Val R²|Test R²|Train-Test Gap"
```

### **Option 2: Search Logs** (After completion)
```bash
# Find all test metrics
grep -A 5 "Test R²" logs/unified_20251111_*.log
```

### **Option 3: Reports** (After completion)
Check these files:
- `outcomes/analyst_base_ETHUSDT_15m_long_report_*.md` - Markdown report
- `outcomes/analyst_base_ETHUSDT_15m_long_metrics_*.json` - JSON metrics
- Look for keys: `test_r2`, `test_mse`, `test_mae`, `test_rmse`

---

## ⏱️ TIMELINE

| Time | Event | Status |
|------|-------|--------|
| 22:20:47 | LightGBM HPO Complete | ✅ Done |
| 23:03:36 | CatBoost HPO Complete | ✅ Done |
| 23:03:39 | Final Training Started | ⏳ In Progress |
| 23:0X:XX | Test Metrics Logged | ⏳ Pending |
| 23:0X:XX | Reports Generated | ⏳ Pending |

**Current time**: 23:06  
**Expected completion**: ~5-10 minutes

---

## 🎯 WHAT YOU'LL SEE

### **In Console Logs**:
```
🔧 LightGBM Configuration:
   CPU Threads: 4
   Training samples: 14023
   Features: 71
   📊 Data splits (temporal order preserved):
      Train: 9816 samples (70.0%)
      Val: 2103 samples (15.0%)
      Test: 2104 samples (15.0%)
================================================================================

✅ LightGBM trained: 100 iterations
   📊 Train R²: 0.8234, RMSE: 0.1456
   📊 Val R²: 0.7891, RMSE: 0.1589
   📊 Test R²: 0.7654, RMSE: 0.1678  ← YOUR TEST SCORE
   ⚠️  Train-Test Gap: 0.0580 (7.0%)
   ✅ Good generalization (overfitting ratio < 10%)
```

### **In JSON Report**:
```json
{
  "lightgbm": {
    "train_r2": 0.8234,
    "val_r2": 0.7891,
    "test_r2": 0.7654,  ← YOUR TEST SCORE
    "train_test_r2_gap": 0.0580,
    "overfitting_ratio": 0.0705,
    "generalization_score": 0.9295
  }
}
```

---

## 🚨 IMPORTANT NOTES

### **1. Test Score ≠ CV Score**
- CV score (0.7787) is from HPO phase
- Test score (pending) is from final evaluation
- They will be similar but not identical

### **2. Test Score Will Be Lower**
- This is EXPECTED and CORRECT
- Test set is truly unseen data
- 0-5% lower than CV is normal

### **3. Walk-Forward Validation**
- Currently using TimeSeriesSplit (temporal CV)
- Test set is chronologically after train/val
- This respects temporal ordering

---

## ✅ SUMMARY

**Your Question**: "Where is the test / walk forward accuracy score?"

**Answer**: 
1. The **0.7787** you see is the **CV score from HPO** (not test score)
2. The **test set score** will appear in ~5-10 minutes when final training completes
3. Look for logs with **"Test R²: 0.XXXX"** or check the JSON reports
4. Expected test R²: **0.74-0.78** (slightly lower than CV is normal)

**Status**: ⏳ Training in progress, test metrics pending

---

**Next**: Wait for training to complete, then check:
```bash
# Option 1: Live monitoring
tail -f logs/unified_*.log | grep "Test R²"

# Option 2: After completion
ls -lt outcomes/analyst_base_ETHUSDT_*
```
