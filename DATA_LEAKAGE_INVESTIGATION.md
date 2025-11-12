# Data Leakage Investigation

**Date**: 2025-11-12 00:25  
**Issue**: 77% gap between HPO CV R² (0.78) and Test R² (0.01)  
**Status**: Investigation in progress

---

## 🚨 CRITICAL FINDINGS

### **Massive Performance Gap**:
- **HPO Cross-Validation R²**: 0.78 (excellent)
- **Test Set R²**: 0.01 (terrible)
- **Gap**: 77% - This is HUGE and indicates serious issues

---

## 🔍 POTENTIAL CAUSES

### **1. Data Leakage in Features**
Features may contain information from the future that wouldn't be available at prediction time.

**Common sources**:
- Look-ahead bias in technical indicators
- Using future data to calculate current features
- Target leakage (features derived from target)
- Regime probabilities calculated on full dataset

**To Check**:
- Review feature generation pipeline
- Check if features use future data
- Verify regime detection doesn't leak information

---

### **2. Cross-Validation Issues**
HPO cross-validation may not properly simulate test conditions.

**Possible issues**:
- CV folds not respecting temporal order
- Data leakage between folds
- Feature scaling/normalization done before splitting
- Regime features calculated on entire dataset before CV

**To Check**:
- Verify TimeSeriesSplit is used correctly
- Check if preprocessing happens before or after split
- Ensure no information flows from validation to training

---

### **3. Train/Val/Test Distribution Mismatch**
Test set may have different characteristics than train/val.

**Possible issues**:
- Market regime change in test period
- Different volatility/trends
- Temporal drift
- Test period is out-of-sample in time

**To Check**:
- Compare feature distributions across splits
- Check target variable statistics
- Analyze market conditions in each period

---

### **4. Overfitting to CV Folds**
HPO may be overfitting to the specific CV fold structure.

**Possible issues**:
- Too many HPO trials
- HPO finding parameters that work for CV but not test
- CV metric not representative of true performance

---

## 🔬 INVESTIGATION STEPS

### **Step 1: Check Feature Generation**
```python
# Review these files:
- src/feature_generation/
- src/analyst/advanced_feature_engineering.py
- src/training/steps/pre_training/
```

**Look for**:
- Features using `.shift()` with positive values (look-ahead)
- Rolling windows that include future data
- Features calculated on full dataset before splitting

---

### **Step 2: Check Regime Detection**
```python
# Review:
- src/training/steps/market_analysis/rolling_hmm_clustering/
```

**Look for**:
- Regime probabilities calculated on entire dataset
- HMM fitted on full data before train/test split
- Regime features that leak future information

---

### **Step 3: Verify Data Splits**
```python
# Check temporal ordering:
- Are train/val/test truly sequential?
- Is there any shuffle happening?
- Are indices preserved correctly?
```

---

### **Step 4: Compare Distributions**
```python
# Compare feature statistics:
import pandas as pd

# Load data
train_data = ...
val_data = ...
test_data = ...

# Compare distributions
for col in train_data.columns:
    print(f"{col}:")
    print(f"  Train: mean={train_data[col].mean():.4f}, std={train_data[col].std():.4f}")
    print(f"  Val:   mean={val_data[col].mean():.4f}, std={val_data[col].std():.4f}")
    print(f"  Test:  mean={test_data[col].mean():.4f}, std={test_data[col].std():.4f}")
```

---

### **Step 5: Check Target Variable**
```python
# Analyze target distribution
print("Target Statistics:")
print(f"Train: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
print(f"Val:   mean={y_val.mean():.4f}, std={y_val.std():.4f}")
print(f"Test:  mean={y_test.mean():.4f}, std={y_test.std():.4f}")

# Check for look-ahead bias in target
# Target should be future returns, not current
```

---

## 🎯 MOST LIKELY CULPRITS

### **1. Regime Features** (HIGH PROBABILITY)
Regime detection is often done on the entire dataset, which leaks future information into past predictions.

**Fix**: Ensure regime detection is done in a rolling/expanding window fashion.

### **2. Feature Scaling** (MEDIUM PROBABILITY)
If features are normalized using statistics from the entire dataset, this leaks information.

**Fix**: Fit scaler only on training data, then transform val/test.

### **3. Look-Ahead in Technical Indicators** (MEDIUM PROBABILITY)
Some indicators may use future data points.

**Fix**: Review all `.shift()`, `.rolling()`, and `.expanding()` operations.

---

## 📊 DIAGNOSTIC METRICS TO ADD

### **Feature Importance Analysis**:
```python
# Check which features are most important
# If regime features dominate, they may be leaking
feature_importance = model.feature_importance()
top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
```

### **Temporal Consistency Check**:
```python
# Check if model performance degrades over time
# Split test set into chunks and evaluate separately
test_chunks = np.array_split(test_data, 5)
for i, chunk in enumerate(test_chunks):
    score = model.score(chunk)
    print(f"Test chunk {i}: R² = {score:.4f}")
```

---

## 🚨 RED FLAGS TO LOOK FOR

1. **Features with perfect correlation to target** (>0.95)
2. **Regime features that change exactly when target changes**
3. **Features that have future information** (shift with positive values)
4. **Preprocessing done before train/test split**
5. **HMM/clustering fitted on entire dataset**

---

## ✅ NEXT STEPS

1. **Run training with current fixes** (no early stopping, accuracy metric)
2. **Analyze feature importance** from trained models
3. **Check regime feature generation** for leakage
4. **Review feature engineering pipeline** for look-ahead bias
5. **Compare train/val/test distributions** for drift

---

**Status**: Fixes applied (1 & 2 complete), running training now...
