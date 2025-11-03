# 🔧 Ranking & R² Problem - Fixes Applied

**Date:** November 2, 2025  
**Status:** ✅ Fixed

---

## 🔴 **Root Cause Identified**

The model's ranking metrics (Precision@K, Spearman ρ, NDCG) and R² scores were unreliable due to **feature selection bugs** in the prediction pipeline:

### **Problem 1: Incorrect Feature Selection in Validation Script**
**File:** `scripts/validate_sr_ranking_metrics.py`

**Issue:**
- Line 259: Passing **full DataFrame** to model.predict() including non-feature columns like `quality_score`, `date`, `symbol`, etc.
- This caused predictions to fail or use wrong features

**Before:**
```python
X = training_data  # Pass full DataFrame to model.predict()
y = training_data['quality_score']
```

**After:**
```python
# Extract features properly - model expects only feature columns
if model.feature_names is not None:
    features_df = training_data[model.feature_names].copy()
else:
    # Fallback: use columns starting with 'feature_'
    feature_cols = [col for col in training_data.columns if col.startswith('feature_')]
    features_df = training_data[feature_cols].copy()
y = training_data['quality_score'].copy()
```

---

### **Problem 2: Direct Model.predict() Call Bypassing Feature Selection**
**File:** `src/tactician/sr_levels/ml_quality/sr_quality_model.py`

**Issue:**
- Line 536 in `evaluate_ranking()`: Called `self.model.predict(X_test)` directly
- Bypassed the `self.predict()` wrapper which handles proper feature selection
- Could cause errors if X_test has extra columns or wrong column order

**Before:**
```python
def evaluate_ranking(self, X_test: pd.DataFrame, y_true: pd.Series, 
                    k: int = 10, quality_threshold: float = 0.7) -> Dict:
    if self.model is None:
        raise ValueError("No trained model. Train model first.")
    
    # Predict
    y_pred = self.model.predict(X_test)  # ❌ WRONG - bypasses feature selection
```

**After:**
```python
def evaluate_ranking(self, X_test: pd.DataFrame, y_true: pd.Series, 
                    k: int = 10, quality_threshold: float = 0.7) -> Dict:
    if self.model is None:
        raise ValueError("No trained model. Train model first.")
    
    # Predict using self.predict() to ensure proper feature selection
    y_pred = self.predict(X_test)  # ✅ CORRECT - uses feature selection wrapper
```

---

### **Problem 3: Inconsistent Feature Selection in Workflow**
**File:** `scripts/run_sr_workflow.py`

**Issue:**
- Line 684: Used `training_df.filter(like='feature_')` for ranking evaluation
- Should use `model.feature_names` if available for consistency

**Before:**
```python
X_eval = training_df.filter(like='feature_')
y_eval = training_df['quality_score']

ranking_metrics = model.evaluate_ranking(X_eval, y_eval, k=10)
```

**After:**
```python
# Use model's feature_names if available for proper feature selection
if model.feature_names is not None:
    X_eval = training_df[model.feature_names]
    self.logger.info(f"   Using {len(model.feature_names)} features from model")
else:
    X_eval = training_df.filter(like='feature_')
    self.logger.info(f"   Using {len(X_eval.columns)} features (filter by prefix)")

y_eval = training_df['quality_score']

ranking_metrics = model.evaluate_ranking(X_eval, y_eval, k=10)
```

---

## ✅ **What Was Fixed**

### **1. Proper Feature Selection Everywhere**
- All prediction calls now use only the features the model was trained on
- Consistent use of `model.feature_names` when available
- Fallback to `filter(like='feature_')` when feature_names not available

### **2. Fixed evaluate_ranking() Method**
- Changed from `self.model.predict()` to `self.predict()`
- Ensures feature selection, NaN handling, and clipping happen correctly
- More robust against column order changes or extra columns

### **3. Fixed Validation Script**
- Extract features properly before all predictions
- Use `.values` to get numpy arrays from pandas Series consistently
- Better handling of strong/weak masks for data splitting

---

## 📊 **Expected Impact**

With these fixes, the ranking metrics should now be:

### **Before (with bugs):**
- **Spearman ρ**: Unreliable, could be artificially low due to wrong features
- **Precision@K**: Could fail or give incorrect results
- **R²**: Unstable due to feature mismatch
- **Predictions**: Wrong or inconsistent

### **After (fixed):**
- **Spearman ρ**: Should improve to >0.60 (proper ranking correlation)
- **Separation**: Should show clear distinction between strong/weak (>0.25)
- **Future R²**: Should generalize better (>0.30)
- **Precision@K**: Should maintain high values (>75%)
- **Predictions**: Consistent and correct across all scripts

---

## 🎯 **Files Modified**

1. **`src/tactician/sr_levels/ml_quality/sr_quality_model.py`**
   - Line 536: Changed `self.model.predict()` to `self.predict()`
   - Ensures proper feature selection in evaluate_ranking()

2. **`scripts/validate_sr_ranking_metrics.py`**
   - Lines 257-267: Added proper feature extraction using model.feature_names
   - Lines 270-271: Fixed mask creation for strong/weak levels
   - Lines 276-277: Fixed data indexing and numpy conversion
   - Throughout: Fixed pandas indexing and type handling

3. **`scripts/run_sr_workflow.py`**
   - Lines 684-690: Added model.feature_names check for ranking evaluation
   - Added logging for transparency
   - Added traceback on ranking evaluation failure

---

## 🚀 **How to Verify the Fix**

1. **Retrain the model:**
   ```bash
   python scripts/run_sr_workflow.py --symbol ETHUSDT --timeframe 15m --lookback-days 180
   ```

2. **Validate ranking metrics:**
   ```bash
   python scripts/validate_sr_ranking_metrics.py --symbol ETHUSDT --timeframe 15m
   ```

3. **Check the metrics:**
   - Precision@10 should be >75%
   - Spearman ρ should be >0.60
   - Separation (strong - weak) should be >0.25
   - Future R² should be >0.30

---

## 🔑 **Key Principle**

**Always use `model.predict()` wrapper, never call `model.model.predict()` directly!**

The wrapper ensures:
- ✅ Correct feature selection via `model.feature_names`
- ✅ Proper feature ordering
- ✅ NaN handling (fillna)
- ✅ Output clipping to [0, 1] range

---

**Summary:** Feature selection bugs fixed in 3 critical files. Model predictions should now be consistent and ranking metrics reliable.

