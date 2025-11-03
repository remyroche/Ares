# 🚀 Next Steps After Quality Score Fixes

**Status:** ✅ Code fixes applied successfully  
**Date:** November 2, 2025  
**Action Required:** Recollect training data

---

## ⚠️ CRITICAL: Data Recollection Required

The current training data (`data_cache/sr_ml_training/sr_quality_training_data.parquet`) was generated with the **OLD broken formula**. You **MUST** regenerate it.

---

## 📋 Step-by-Step Workflow

### **STEP 1: Backup Old Data (Optional)**
```bash
# Optional: Keep old data for comparison
mkdir -p data_cache/sr_ml_training/backup_old_formula
mv data_cache/sr_ml_training/sr_quality_training_data.parquet \
   data_cache/sr_ml_training/backup_old_formula/
mv data_cache/sr_ml_training/sr_quality_training_data_metadata.json \
   data_cache/sr_ml_training/backup_old_formula/
```

**OR delete old data directly:**
```bash
rm data_cache/sr_ml_training/sr_quality_training_data.parquet
rm data_cache/sr_ml_training/sr_quality_training_data_metadata.json
```

---

### **STEP 2: Recollect Training Data**

Find and run your data collection script. Common locations:

**Option A: Dedicated collection script**
```bash
python3 scripts/collect_sr_training_data.py
```

**Option B: Module execution**
```bash
python3 -m src.tactician.sr_levels.ml_quality.collect_data
```

**Option C: Check your existing scripts**
```bash
# Look for collection scripts
find . -name "*collect*" -name "*.py" | grep -E "(sr|quality|training)"

# Check scripts directory
ls -la scripts/
```

**Option D: Manual collection (if no script exists)**
```python
# Create a collection script
import asyncio
from src.tactician.sr_levels.ml_quality.sr_quality_data_collector import SRQualityDataCollector

async def collect():
    collector = SRQualityDataCollector()
    
    # Adjust parameters as needed
    training_data = await collector.collect_training_data(
        symbol='ETHUSDT',
        exchange='binance',
        start_date='2025-05-01',
        end_date='2025-09-30',
        timeframe='15m',
        forward_days=10,
        sample_freq_days=7
    )
    
    # Save
    output_path = collector.save_training_data(training_data)
    print(f"✅ Training data saved to: {output_path}")
    
    return training_data

# Run
if __name__ == '__main__':
    asyncio.run(collect())
```

---

### **STEP 3: Validate Improvements**

After recollection, validate that the fixes worked:

```bash
python3 validate_quality_score.py
```

**Expected Results:**

✅ **Bounce Strength:**
```
Mean: ~0.60 (was 0.98) ✅
Median: ~0.55 (was 1.00) ✅
Std: ~0.25 (was 0.10) ✅
At max (≥0.95): <10% (was 50%) ✅
```

✅ **Trade Profit:**
```
Mean: ~0.15 (was -0.05) ✅
Median: ~0.00 (was -0.50) ✅
Winning trades: ~50% (was 35%) ✅
Positive expectancy: YES ✅
```

✅ **Quality Score:**
```
Mean: ~0.55 (similar to before)
Std: ~0.26 (was 0.25) ✅
Distribution: More normal ✅
All components contribute: YES ✅
```

✅ **Feature Correlations:**
```
Top correlation: >0.4 (was 0.36) ✅
Strong features (>0.3): 5-7 (was 3) ✅
Weak features (<0.1): <25 (was 28) ✅
```

---

### **STEP 4: Compare Old vs New (Optional)**

If you backed up old data, compare distributions:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load both datasets
old_data = pd.read_parquet('data_cache/sr_ml_training/backup_old_formula/sr_quality_training_data.parquet')
new_data = pd.read_parquet('data_cache/sr_ml_training/sr_quality_training_data.parquet')

# Compare bounce strength
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(old_data['bounce_strength'], bins=50, alpha=0.7, label='Old (Broken)', color='red')
axes[0].set_title('OLD: Bounce Saturated')
axes[0].axvline(old_data['bounce_strength'].mean(), color='darkred', linestyle='--', 
                label=f'Mean: {old_data["bounce_strength"].mean():.3f}')
axes[0].legend()

axes[1].hist(new_data['bounce_strength'], bins=50, alpha=0.7, label='New (Fixed)', color='green')
axes[1].set_title('NEW: Better Spread')
axes[1].axvline(new_data['bounce_strength'].mean(), color='darkgreen', linestyle='--',
                label=f'Mean: {new_data["bounce_strength"].mean():.3f}')
axes[1].legend()

plt.savefig('analysis_output/bounce_comparison_old_vs_new.png', dpi=300)
print("✅ Comparison saved to: analysis_output/bounce_comparison_old_vs_new.png")
```

---

### **STEP 5: Retrain Model**

Once validation passes, retrain the ML model:

```bash
# Find your training script
python3 scripts/train_sr_quality_model.py

# OR
python3 -m src.tactician.sr_levels.ml_quality.train_model

# OR check your existing workflow
```

**What to expect:**
- Better model performance (cleaner labels)
- Stronger feature importance for top features
- Better generalization (less overfitting to noise)

---

### **STEP 6: Evaluate Model Performance**

Compare old vs new model:

**Metrics to track:**
- Training RMSE/MAE
- Validation RMSE/MAE
- Feature importance rankings
- Prediction distribution
- Model confidence scores

**Expected improvements:**
- Lower validation error ✅
- More interpretable feature importance ✅
- Better prediction confidence ✅

---

## 🎯 Validation Checklist

After completing all steps, verify:

- [ ] Old training data backed up or deleted
- [ ] New training data collected with fixed formula
- [ ] `validate_quality_score.py` shows improvements:
  - [ ] bounce_strength mean < 0.8
  - [ ] bounce_strength std > 0.2
  - [ ] trade_profit mean > 0
  - [ ] quality_score variance maintained
  - [ ] Top correlation > 0.4
  - [ ] Fewer weak features
- [ ] Model retrained successfully
- [ ] Model performance improved or maintained

---

## 📊 Before/After Summary

### Code Changes:
| Component | Before | After |
|-----------|--------|-------|
| Bounce window | Full forward window | First 10 bars ✅ |
| Bounce threshold | 2% | 3% ✅ |
| Trade R/R | 2:1 (2% TP) | 1:1 (1% TP) ✅ |
| Quality weights | 0.35/0.35/0.30 | 0.333/0.333/0.333 ✅ |

### Expected Data Improvements:
| Metric | Before | After |
|--------|--------|-------|
| bounce_strength mean | 0.98 ❌ | ~0.60 ✅ |
| trade_profit mean | -0.05 ❌ | ~0.15 ✅ |
| Top correlation | 0.36 ⚠️ | ~0.45 ✅ |
| Strong features | 3 ⚠️ | 5-7 ✅ |

---

## 🔧 Troubleshooting

### Issue: Can't find data collection script
**Solution:** Use the manual collection code in STEP 2, Option D

### Issue: Validation shows no improvement
**Check:**
1. Did you recollect data with the FIXED code?
2. Is the new data actually being used?
3. Check logs for any errors during collection

### Issue: New bounce_strength still saturated
**Debug:**
```python
# Check if early_future is working
import pandas as pd
data = pd.read_parquet('data_cache/sr_ml_training/sr_quality_training_data.parquet')
print(f"Bounce mean: {data['bounce_strength'].mean()}")
print(f"Bounce at max (≥0.95): {(data['bounce_strength'] >= 0.95).sum() / len(data) * 100:.1f}%")

# Should be:
# Mean: ~0.60
# At max: <10%
```

### Issue: Trade profit still negative
**Debug:**
```python
# Check trade profit distribution
import pandas as pd
data = pd.read_parquet('data_cache/sr_ml_training/sr_quality_training_data.parquet')
print(f"Trade profit mean: {data['trade_profit'].mean()}")
print(f"Winning trades: {(data['trade_profit'] > 0).sum() / len(data) * 100:.1f}%")

# Should be:
# Mean: ~0.15
# Winning: ~50%
```

---

## 📞 Quick Reference

**Validation command:**
```bash
python3 validate_quality_score.py
```

**View fixes applied:**
```bash
cat FIXES_APPLIED.md
```

**Check investigation details:**
```bash
cat INVESTIGATION_SUMMARY.md
```

**View critical issues visualization:**
```bash
open analysis_output/quality_issues_summary.png
```

---

## ✅ Success Criteria

Your fixes are successful if validation shows:

1. ✅ bounce_strength mean < 0.8 (not saturated)
2. ✅ bounce_strength std > 0.2 (good variance)
3. ✅ trade_profit mean > 0 (positive expectancy)
4. ✅ quality_score variance maintained (std > 0.25)
5. ✅ Top feature correlation > 0.4 (stronger)
6. ✅ Strong features (>0.3): at least 5
7. ✅ All components contribute to quality score

---

**Last updated:** November 2, 2025  
**Status:** Awaiting data recollection  
**Next action:** Run data collection script!

Good luck! 🚀
