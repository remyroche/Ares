# How to Diagnose Regime Data Issues

## The Problem

Your HPO is giving **identical scores (0.8000)** across all trials. This diagnostic script will identify **exactly why**.

## Run the Diagnostic

```bash
cd /Users/remyroche/Documents/Ares
python scripts/diagnose_regime_data_leakage.py
```

This will analyze your regime data and generate a comprehensive report.

## What the Script Checks

### 1️⃣  Data Structure
- Number of samples and features
- Column types and names
- Timestamp/index information

### 2️⃣  Regime Labels
- Number of unique regimes
- Class distribution (balanced?)
- Regime transitions (how often regimes change)
- Average regime duration

**Red Flags:**
- No regime transitions → Labels are constant
- Very long regimes (>100 samples) → Not enough variation

### 3️⃣  Feature Quality
- Feature variance (are features varying?)
- Zero/low variance features
- NaN values
- Feature ranges

**Red Flags:**
- Many zero-variance features → No information
- All features have low variance → No signal

### 4️⃣  Temporal Alignment
- Are timestamps monotonic?
- Are features and labels from same timestamp?

**Red Flags:**
- Features from SAME timestamp as labels → **DATA LEAKAGE!**
  - You're predicting CURRENT regime, not FUTURE regime
  - Features at time T should predict regime at T+1, not T

### 5️⃣  Prediction Capability
- Train/test performance
- Cross-validation scores and variance
- Feature importance analysis
- Confusion matrix

**Red Flags:**
- CV scores all identical (std < 0.001) → **YOUR PROBLEM!**
- Max feature importance < 0.05 → **NO SIGNAL!**
- Near-perfect accuracy (>95%) → **DATA LEAKAGE!**

## Interpreting Results

### Scenario A: No Feature Signal (Most Common)
```
🔬 Feature Importance Analysis:
  Max importance: 0.0423  🚨 ALL < 5%!
  Features >5% importance: 0/21

🔄 Cross-Validation:
  Fold scores: ['0.8000', '0.8000', '0.8000']
  Std CV: 0.000000  🚨 IDENTICAL!
```

**Diagnosis**: Features have NO predictive power!

**Solutions**:
1. Check feature engineering pipeline
2. Verify features are calculated correctly
3. Add more informative features (momentum, volatility, volume)
4. Check if regime labels make sense

### Scenario B: Data Leakage
```
📊 Performance:
  Train accuracy: 0.9850  ⚠️  VERY HIGH!
  Test accuracy: 0.9800  ⚠️  VERY HIGH!

📋 Confusion Matrix:
  Diagonal percentage: 98.5%  🚨 NEAR-PERFECT!

⚠️  POTENTIAL ISSUE: Features from SAME timestamp as labels!
```

**Diagnosis**: Features using future information!

**Solutions**:
1. Shift labels forward by 1 period
   - Features at time T → predict regime at T+1
2. Use lagged features only
3. Don't include regime_id in features

### Scenario C: Meaningless Labels
```
🔄 Regime Transitions: 5 (0.42% of data)
⏱️  Regime Duration Statistics:
  Mean: 238.4 samples
  
🎯 Regime Statistics:
  Regime 0: 1190 samples (99.6%)
  Regime 1: 5 samples (0.4%)
```

**Diagnosis**: Regimes aren't transitioning!

**Solutions**:
1. Re-run clustering with different parameters
2. Check if market data is too homogeneous
3. Try different timeframe
4. Verify clustering algorithm is working

### Scenario D: Broken Cross-Validation
```
🔄 Cross-Validation:
  Fold scores: ['0.8000', '0.8000', '0.8000']
  Std CV: 0.000000

But feature importances look good:
  Max importance: 0.2500  ✅
```

**Diagnosis**: CV using same fold repeatedly

**Solutions**:
1. Remove `random_state` from CV
2. Check if data is being shuffled identically
3. Verify CV split logic

## Expected Good Output

```
🔄 Cross-Validation:
  Fold scores: ['0.7215', '0.7843', '0.7556']  ✅ Varying!
  Mean CV: 0.7538
  Std CV: 0.025847  ✅ Good variance!

🔬 Feature Importance Analysis:
  Max importance: 0.1823  ✅ Above 0.05!
  Features >5% importance: 8/21  ✅ Multiple features!

📊 Performance:
  Train accuracy: 0.8234
  Test accuracy: 0.7538
  Overfit gap: 0.0696  ✅ Reasonable gap!
```

## Quick Fix Guide

Based on the diagnostic output:

| Problem | Quick Fix |
|---------|-----------|
| 🚨 CV scores identical | Features have no signal - fix feature engineering |
| 🚨 Max importance < 0.05 | Features not predictive - add better features |
| 🚨 Train/test accuracy > 95% | Data leakage - shift labels or use lagged features |
| 🚨 No regime transitions | Re-run clustering or use different data |
| 🚨 Features from same timestamp | Shift labels forward: `labels = labels.shift(-1)` |

## After Running Diagnostic

1. **Identify the issue** from the report
2. **Fix the root cause** (usually feature engineering or labeling)
3. **Re-run the diagnostic** to verify fix
4. **Then run HPO** - scores should now vary!

## Common Fixes

### Fix 1: Shift Labels Forward (Most Common)
```python
# In your data preparation:
df['regime_id_future'] = df['regime_id'].shift(-1)
# Use regime_id_future as target
```

### Fix 2: Add Lagged Features
```python
# Only use past information
df['momentum_lag1'] = df['close'].pct_change(1).shift(1)
df['volume_lag1'] = df['volume'].shift(1)
```

### Fix 3: Verify Feature Calculation
```python
# Check features are actually varying
print(f"Feature variance: {df[feature_cols].var()}")
print(f"Feature correlation with target: {df[feature_cols].corrwith(df['regime_id'])}")
```

## Questions to Ask

After running the diagnostic:

1. **Are CV scores varying?** (std > 0.01)
   - No → Features have no signal
   
2. **Are any features important?** (max > 0.05)
   - No → Feature engineering broken
   
3. **Is accuracy too high?** (>95%)
   - Yes → Data leakage
   
4. **Are regimes transitioning?** (>5% transitions)
   - No → Clustering/labeling issue

## Next Steps

1. ✅ Run `python scripts/diagnose_regime_data_leakage.py`
2. 📋 Read the output carefully
3. 🔧 Fix identified issues
4. 🔄 Re-run diagnostic to verify
5. 🚀 Run HPO again - should work now!

---

**The diagnostic will tell you EXACTLY what's wrong!**

