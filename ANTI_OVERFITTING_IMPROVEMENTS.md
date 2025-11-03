# Anti-Overfitting Improvements for SR Quality ML Model

**Date:** 2025-11-01  
**Status:** ✅ Implemented & Running with HPO

---

## 🔍 Problem Identified

### Overfitting Evidence
- **Fold 0**: Train R² (52.3%) → Val R² (20.4%) = **31.9% gap** 😱
- **Fold 1**: Train R² (36.5%) → Val R² (9.5%) = **27.0% gap** 😱
- **Average Train-Val Gap**: ~25-30% (indicates severe overfitting)

### Root Causes
1. **Weak Regularization**: `lambda_l1=0.1`, `lambda_l2=0.1` (10x too weak)
2. **Too Complex Model**: `num_leaves=31`, `max_depth=6` (excessive capacity)
3. **Insufficient Data Per Leaf**: `min_data_in_leaf=20` (allows overfitting on small samples)
4. **No Data Quality Filtering**: Training on weak/noisy SR levels

---

## ✅ Solutions Implemented

### 1. **Strong Regularization**
```python
# OLD (too weak)
'lambda_l1': 0.1
'lambda_l2': 0.1

# NEW (10x stronger + HPO optimized)
'lambda_l1': {'type': 'float', 'low': 0.5, 'high': 5.0, 'default': 1.0, 'log': True}
'lambda_l2': {'type': 'float', 'low': 0.5, 'high': 5.0, 'default': 1.0, 'log': True}
```

### 2. **Reduced Model Complexity**
```python
# OLD (too complex)
'num_leaves': 31
'max_depth': 6

# NEW (simpler + HPO search space)
'num_leaves': {'type': 'int', 'low': 10, 'high': 31, 'default': 15}
'max_depth': {'type': 'int', 'low': 3, 'high': 6, 'default': 4}
```

### 3. **Increased Min Data Per Leaf**
```python
# OLD (too permissive)
'min_data_in_leaf': 20

# NEW (requires more evidence + HPO optimized)
'min_data_in_leaf': {'type': 'int', 'low': 30, 'high': 100, 'default': 50}
```

### 4. **Slower Learning Rate**
```python
# OLD
'learning_rate': 0.05

# NEW (more stable + HPO search space)
'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.05, 'default': 0.03, 'log': True}
```

### 5. **Increased Subsampling (Variance Reduction)**
```python
# OLD
'feature_fraction': 0.9
'bagging_fraction': 0.8

# NEW (higher dropout for robustness)
'feature_fraction': {'type': 'float', 'low': 0.6, 'high': 0.9, 'default': 0.7}
'bagging_fraction': {'type': 'float', 'low': 0.6, 'high': 0.9, 'default': 0.7}
```

### 6. **Data Quality Filtering**
```python
# FILTER OUT WEAK SR LEVELS (only train on meaningful levels)
min_quality_threshold = 0.25  # Remove bottom 25% of levels

# Filter by quality_score
if 'quality_score' in training_data.columns:
    training_data = training_data[training_data['quality_score'] >= min_quality_threshold]

# Also filter by strength
if 'feature_strength' in training_data.columns:
    training_data = training_data[training_data['feature_strength'] >= 0.4]
```

### 7. **Hyperparameter Optimization (HPO)**
```python
# NEW METHOD: train_with_hpo()
metrics = model.train_with_hpo(
    training_df,
    target_column='quality_score',
    n_trials=100,  # Test 100 parameter combinations
    n_folds=5,
    method='bayesian'  # Efficient Bayesian optimization
)
```

**HPO Search Space**: 8 parameters with anti-overfitting ranges
- Model complexity: `num_leaves` (10-31), `max_depth` (3-6)
- Strong regularization: `lambda_l1` (0.5-5.0), `lambda_l2` (0.5-5.0)
- Data requirements: `min_data_in_leaf` (30-100)
- Learning rate: (0.01-0.05)
- Subsampling: `feature_fraction` (0.6-0.9), `bagging_fraction` (0.6-0.9)

---

## 📁 Files Modified

### 1. `src/tactician/sr_levels/ml_quality/sr_quality_model.py`
**Changes:**
- ✅ Updated `_get_default_config()` with anti-overfitting defaults
- ✅ Added data filtering in `train()` method  
- ✅ Added new `train_with_hpo()` method for hyperparameter optimization
- ✅ Imported `optimize_hyperparameters` from HPO utilities

**Key Methods:**
```python
def train_with_hpo(self, training_data, n_trials=100, method='bayesian')
    # 1. Filter weak SR levels (quality < 0.25, strength < 0.4)
    # 2. Define anti-overfitting search space
    # 3. Run Bayesian HPO with 100 trials
    # 4. Train final model with optimized parameters
    # 5. Return metrics + HPO results
```

### 2. `scripts/run_sr_workflow.py`
**Changes:**
- ✅ Changed `model.train()` → `model.train_with_hpo()`
- ✅ Added HPO configuration (100 trials, Bayesian method)
- ✅ Added logging of optimized parameters

**Before:**
```python
metrics = model.train(training_df, target_column='quality_score', n_folds=5)
```

**After:**
```python
metrics = model.train_with_hpo(
    training_df,
    target_column='quality_score',
    n_trials=100,
    n_folds=5,
    method='bayesian'
)
```

---

## 🎯 Expected Improvements

### Reduced Overfitting
- **Target**: Reduce train-val gap from ~25-30% to <10%
- **Method**: Stronger regularization + simpler model + quality filtering

### Better Generalization
- **Target**: Improve validation R² from 13.6% to 18-22%
- **Method**: HPO finds optimal balance between bias and variance

### More Stable Predictions
- **Target**: Reduce std of validation R² across folds
- **Method**: Consistent data quality + robust hyperparameters

---

## 📊 Running Status

**Current Run:**
- **Command**: `python scripts/run_sr_workflow.py ... (with HPO)`
- **PID**: 46721
- **Log File**: `sr_hpo_workflow.log`
- **Status**: 🔄 Running (HPO in progress - 100 trials)
- **Monitor**: `tail -f sr_hpo_workflow.log`

**HPO Configuration:**
- Optimization Method: Bayesian (Optuna TPE)
- Number of Trials: 100
- CV Folds: 5 (Time Series Split)
- Objective: Minimize MSE (maximize validation performance)

---

## 🔄 Next Steps

1. **Wait for HPO Completion** (~10-20 minutes for 100 trials)
2. **Review Results**:
   - Check train-val gap (should be <10%)
   - Verify validation R² improvement
   - Examine optimized hyperparameters
3. **Generate SHAP Analysis** with retrained model
4. **Compare Performance**:
   - Before: R²=13.6%, Gap=25-30%
   - After: R²=?, Gap=?

---

## 📚 Technical Details

### Anti-Overfitting Strategy
1. **Regularization**: L1 + L2 penalties on weights
2. **Complexity Control**: Limit tree depth and leaves
3. **Data Requirements**: Require minimum samples per leaf
4. **Ensemble Diversity**: Subsample features and data
5. **Early Stopping**: Prevent excessive training
6. **Quality Filtering**: Train only on meaningful SR levels

### HPO Advantages
- **Automated Tuning**: No manual guessing
- **Bayesian Efficiency**: Smart parameter exploration
- **Cross-Validation**: Robust generalization estimates
- **Objective Optimization**: Directly minimize overfitting metrics

---

## ✨ Key Takeaways

1. **Overfitting is Addressable**: With proper regularization and HPO
2. **Quality > Quantity**: Filtering weak data improves signal/noise
3. **Simpler is Better**: Reduced complexity improves generalization
4. **HPO is Essential**: For finding optimal anti-overfitting balance

**Expected Outcome**: A more generalizable SR quality model that performs consistently on new data! 🚀

