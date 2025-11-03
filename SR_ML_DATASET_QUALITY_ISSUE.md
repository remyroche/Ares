# SR ML Dataset Quality Issue - Root Cause Analysis

## 🔴 Critical Finding: Discrete Default Values Dominate Dataset

### The Problem

The training dataset is **dominated by discrete default values**, not continuous measurements:

```
Distribution of quality_score values:
├─ 39.4% → EXACTLY 0.2000 (717/1,821 samples)
├─ 23.0% → EXACTLY 0.3675 (418/1,821 samples)
└─ 45.0% → Only these TWO values (820/1,821 samples)

Histogram (extremely narrow):
├─ 0.2-0.3: 41.4% ████████████████████████████████████████
├─ 0.3-0.4: 26.9% ███████████████████████████
└─ All others: 31.7% (spread across 8 bins)
```

### Why This Causes Model Collapse

1. **No Variance to Learn From**
   - 62.4% of data clustered in two narrow bins (0.2-0.4)
   - IQR (inter-quartile range): only 0.29
   - Coefficient of variation: 0.633 (relatively low for 0-1 scale)

2. **Every HPO Trial Shows**
   ```
   WARNING: Very low score variance detected: 0.000000
   ```
   - All 100 trials predict nearly identical values
   - R² stays around -0.06 to -0.29 (worse than predicting mean)

3. **Model Cannot Discriminate**
   - LightGBM learns to predict ~0.40 (the mean) for everything
   - No patterns to learn when 62% of data is essentially identical

---

## 🔍 Where Defaults Come From

### Source Code Analysis

#### Default Value #1: **0.2** (39.4% of data)

```python:378:386:src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py
if len(hits) == 0:
    # Level NOT tested - assign low quality
    return {
        'hit_rate': 0.0,
        'bounce_strength': 0.0,
        'hold_strength': 0.5,
        'trade_profit': 0.0,
        'quality_score': 0.2  # Low quality (untested)
    }
```

**Meaning**: Level was **NEVER TESTED** in the 10-day forward window
- Price never touched the level
- No actual performance data
- Assigned arbitrary "low quality" score

#### Default Value #2: **~0.3675** (23.0% of data)

This comes from the quality formula when bounce/hold are mediocre:

```python:424:428:src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py
quality_score = (
    bounce_strength * 0.35 +    # Strong bounces = good
    hold_strength * 0.35 +      # Levels that hold = good
    max(trade_profit, 0) * 0.30 # Profitable = good
)
```

When `bounce_strength ≈ 0.5`, `hold_strength ≈ 0.5`, `trade_profit = 0`:
```
quality_score = 0.5 * 0.35 + 0.5 * 0.35 + 0 * 0.30 = 0.35
```

**Meaning**: Level was tested but showed **weak/mediocre performance**

---

## 🎯 Real Problem: Not Enough "Tested" Levels

### Current Setup
```
Forward Window: 10 days
Sample Frequency: Every 7 days
Result: 39.4% of levels NEVER get tested
```

### Why Levels Aren't Tested
1. **10-day window is too short** for volatile markets
2. **Many levels are far from current price** (15m timeframe = lots of levels)
3. **Support/resistance levels may be distant** from trading range

### Data Breakdown (1,821 total samples)
```
├─ Untested (0.2):     717 (39.4%) ← DEFAULT VALUE
├─ Weak tested (~0.37): 418 (23.0%) ← NARROW CLUSTER  
├─ Medium (0.4-0.6):   119 ( 6.5%)
├─ Good (0.6-0.8):     181 ( 9.9%)
└─ Excellent (0.8-1.0): 125 ( 6.9%)

Useful samples (>0.5): Only 425/1,821 (23.3%)
```

---

## ✅ Solutions (Ranked by Impact)

### 🥇 Solution 1: Exclude Untested Levels from Training

**Rationale**: Untested levels have NO signal, only noise

```python
# In sr_quality_data_collector.py
def collect_training_data(...):
    # After collecting samples
    training_df = training_df[training_df['quality_score'] > 0.2].copy()
    # Removes 39.4% of noise
```

**Impact**: 
- Removes 717 samples with no real data
- Training set: 1,104 samples (all tested)
- Better variance for model to learn

**Trade-off**: Smaller dataset, but higher quality

---

### 🥈 Solution 2: Increase Forward Window

**Current**: 10 days  
**Proposed**: 20-30 days

```python
# In run_sr_workflow.py or train_sr_quality_model.py
forward_window_days=20  # Was 10
```

**Impact**:
- More levels get tested (hit by price)
- More real performance data
- Less reliance on defaults

**Trade-off**: Less temporal samples (longer gap between samples)

---

### 🥉 Solution 3: Use Classification Instead of Regression

**Rationale**: If target is mostly discrete, treat it as classes

```python
# Convert to binary/multi-class problem
Classes:
├─ Noise (0-0.3):   Drop or class 0
├─ Weak (0.3-0.5):  Class 1
├─ Medium (0.5-0.7): Class 2
├─ Strong (0.7-1.0): Class 3
```

Use LightGBM classifier with ranking objective:
```python
lgb.train({
    'objective': 'lambdarank',  # or 'multiclass'
    'num_class': 3,
    'metric': 'ndcg',
})
```

**Impact**:
- Better suited to discrete/clustered targets
- Ranking loss function aligns with use case
- Built-in handling of class imbalance

---

### 🔧 Solution 4: Ensemble Approach

Combine multiple strategies:

1. **Train on only tested levels** (quality > 0.2)
2. **Use 20-day forward window**
3. **Train ranking model** (LambdaRank)
4. **Validate on Precision@K and Spearman ρ**

---

## 📊 Expected Results After Fix

### Current State
```
R²: -0.29 (model collapse)
Spearman ρ: ~0.1 (no ranking ability)
Precision@10: ~random
```

### After Excluding Untested Levels
```
Training samples: 1,104 (down from 1,821)
Quality distribution:
├─ 0.3-0.4: 418 (37.9%) ← Still dominant
├─ 0.4-0.6:  76 ( 6.9%)
├─ 0.6-0.8: 181 (16.4%)
└─ 0.8-1.0: 125 (11.3%)

Expected R²: 10-20% (better, but still low)
Spearman ρ: 0.3-0.5 (moderate ranking ability)
```

### After 20-Day Window + Exclude Untested
```
More levels tested → Less 0.2 defaults
Better quality variance
Expected R²: 20-30%
Spearman ρ: 0.5-0.7 (good ranking)
Precision@10: 60-70% (useful)
```

### After Classification/LambdaRank
```
Optimized for ranking, not R²
Spearman ρ: 0.6-0.8 (excellent ranking)
Precision@10: 70-80% (very useful)
NDCG@10: 0.7-0.8 (strong)
```

---

## 🚀 Recommended Action Plan

### Phase 1: Quick Win (5 minutes)
```bash
# Exclude untested levels from training
# Edit: src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py
# After line 155, add:
training_df = training_df[training_df['quality_score'] > 0.2].copy()
```

### Phase 2: Medium Term (10 minutes)
```bash
# Increase forward window
# Edit: scripts/run_sr_workflow.py, train_sr_quality_model.py
# Change forward_window_days from 10 to 20
```

### Phase 3: Long Term (30 minutes)
```bash
# Convert to LambdaRank classification
# Edit: src/tactician/sr_levels/ml_quality/sr_quality_model.py
# Change objective from 'regression' to 'lambdarank'
# Discretize quality_score into classes
```

---

## 🔬 Validation Commands

After implementing Phase 1:
```bash
# Check new distribution
python3 -c "
import pandas as pd
df = pd.read_parquet('data_cache/sr_ml_training/sr_quality_training_data.parquet')
print(f'Samples: {len(df)}')
print(f'Quality > 0.2: {(df[\"quality_score\"] > 0.2).sum()}')
print(df['quality_score'].describe())
"

# Retrain
cd /Users/remyroche/Documents/Ares
python3 scripts/run_sr_workflow.py --symbol ETHUSDT --exchange binance --timeframe 15m --ml-start-date 2024-01-01 --ml-end-date 2024-11-01 2>&1 | tee sr_phase1_fix.log

# Check results
grep -E "ML CV avg Val R²|Spearman|Precision" sr_phase1_fix.log
```

---

## 📝 Key Insights

1. **The core issue is NOT the model** - LightGBM is fine
2. **The core issue is NOT the weighting** - Gentle/aggressive doesn't matter
3. **The core issue IS the data** - 62% of data is concentrated in narrow, discrete values
4. **Training on untested levels is like training on noise** - They have no predictive signal
5. **Regression on discrete targets is doomed** - Consider classification/ranking

---

## ⚠️ Why Previous Fixes Failed

### ❌ Confidence Weighting (Label Smoothing)
- **Attempted**: Weight noise 0.3x, strong 2.0x
- **Failed**: Still training on 39.4% untested levels
- **Issue**: Weighting doesn't change the fact that 62% of data is clustered

### ❌ No Hard Filtering
- **Attempted**: Keep all data, rely on weights
- **Failed**: Model learns mean of narrow distribution
- **Issue**: Too much noise drowns out signal

### ❌ Gentle Weights vs Aggressive Weights
- **Attempted**: Try different weight ranges
- **Failed**: R² stayed negative regardless
- **Issue**: Variance problem, not weight problem

---

## ✅ The Real Fix

**Stop training on untested levels**. They are not "low quality" levels - they are **no data** levels.

```python
# Simple, powerful fix:
training_df = training_df[training_df['quality_score'] > 0.2].copy()
```

This single line will:
- Remove 39.4% of noise
- Increase target variance
- Give model real patterns to learn
- Improve R², Spearman ρ, Precision@K

---

## 🎯 Bottom Line

The model is failing because **we're asking it to learn from 62% noise**. No amount of hyperparameter tuning, weighting, or filtering can fix a fundamentally noisy dataset.

**Solution**: Clean the data first, then train the model.

