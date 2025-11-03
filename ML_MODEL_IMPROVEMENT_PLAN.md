# SR Quality Model Improvement Plan

## 📊 Current Model Performance Analysis

### **Cross-Validation Results**

| Metric | Value | Assessment |
|--------|-------|------------|
| **Average Val R²** | **0.128** (12.8%) | ⚠️ **LOW** - Model explains only 12.8% of variance |
| **Best Fold R²** | 0.386 (Fold 2) | ✅ Decent |
| **Worst Fold R²** | -0.017 (Fold 4) | ❌ **NEGATIVE** - Worse than baseline! |
| **R² Std Dev** | 0.141 | ⚠️ **HIGH** - Very inconsistent across folds |
| **Avg Val RMSE** | 0.238 | ⚠️ Moderate error |
| **Avg Val MAE** | 0.184 | ⚠️ Moderate error |

### **Fold-by-Fold Performance**

| Fold | Samples | Val R² | Boost Rounds | Assessment |
|------|---------|--------|--------------|------------|
| 0 | 161 | 0.030 | 10 | ❌ Very poor, early stopped |
| 1 | 318 | 0.155 | 74 | ⚠️ Below average |
| **2** | **475** | **0.386** | **102** | ✅ **BEST** |
| 3 | 632 | 0.085 | 13 | ❌ Poor, early stopped |
| 4 | 789 | -0.017 | 4 | ❌ **WORST** - Negative R²! |

---

## 🚨 **Key Issues Identified**

### **1. High Variance Across Folds (σ = 0.14)**
- Fold 2: R² = 0.386 ✅
- Fold 4: R² = -0.017 ❌
- **Range: 0.403 difference!**

**Root Causes:**
- Dataset size varies significantly (161 → 789 samples)
- Temporal patterns may differ across folds
- Possible overfitting in some folds
- Target variable (quality_score) may be unstable

### **2. Low Average R² (12.8%)**
- Model explains less than 13% of variance
- 87% of variance is unexplained
- Suggests poor feature-target relationship

**Root Causes:**
- Quality_score calculation may be noisy
- Missing important features
- Features may not be predictive of future SR performance
- Label quality issues

### **3. Early Stopping in 3 of 5 Folds**
- Fold 0: 10 rounds (stopped early)
- Fold 3: 13 rounds (stopped early)  
- Fold 4: 4 rounds (stopped early)

**Root Causes:**
- Insufficient training data
- Overfitting detected quickly
- Poor data quality

### **4. Negative R² in Fold 4**
- R² = -0.017 means model is **worse than predicting the mean**
- This is a critical failure

**Root Causes:**
- Fold 4 distribution differs significantly from training
- Possible temporal regime shift
- Model memorized training data, doesn't generalize

---

## 🛠️ **Improvement Recommendations**

### **Priority 1: Improve Data Quality & Quantity**

#### **A. Increase Training Data**
```python
# Current: ~946 total samples (too small!)
# Recommendation: 5,000+ samples

# In run_sr_workflow.py, increase training period
self.ml_start_date = (end_dt - timedelta(days=365)).strftime('%Y-%m-%d')  # 1 year instead of 6 months
self.ml_end_date = end_dt.strftime('%Y-%m-%d')

# Increase sampling frequency
ml_sample_freq_days=3  # Every 3 days instead of 7
```

**Expected Impact:**
- 5x more training data
- Better generalization
- More stable R² across folds

#### **B. Improve Quality Score Calculation**
```python
# File: src/tactician/sr_levels/ml_quality.py

# Current quality_score may be too noisy
# Suggestions:
# 1. Use longer forward window (20 days instead of 10)
# 2. Weight recent performance more heavily
# 3. Combine multiple metrics (bounces + breaks + hold time)
# 4. Normalize by market volatility regime

forward_days=20  # Increase from 10
```

**Expected Impact:**
- More stable target variable
- Better signal-to-noise ratio
- Improved R² by 10-15%

---

### **Priority 2: Feature Engineering**

#### **A. Add Missing Critical Features**

Based on LGBM importance, add:

```python
# 1. Time-based features (SR levels decay over time)
features['days_since_formation'] = (current_date - sr_level.first_seen).days
features['recency_score'] = np.exp(-days_since_formation / 30)  # Exponential decay

# 2. Confluence features (multiple methods agreeing)
features['method_confluence'] = count_detection_methods(sr_level)
features['method_agreement_score'] = calculate_method_agreement(sr_level)

# 3. Volume-weighted features
features['volume_weighted_strength'] = strength * normalized_volume
features['volume_consistency'] = std(volume_at_touches) / mean(volume_at_touches)

# 4. Regime-aware features
features['volatility_adjusted_strength'] = strength / market_volatility
features['trend_alignment'] = 1.0 if (is_support and trend_down) else alignment_score

# 5. Historical performance features
features['recent_success_rate'] = successful_bounces / total_tests (last 30 days)
features['win_rate_trend'] = recent_win_rate - historical_win_rate
```

**Expected Impact:**
- Capture temporal patterns
- Better regime awareness
- Improved R² by 15-20%

#### **B. Feature Interactions**

```python
# Top features from LGBM:
# 1. distance_to_current_pct (152)
# 2. approach_velocity (102)
# 3. prominence (81)

# Create interactions:
features['distance_x_velocity'] = distance_to_current_pct * approach_velocity
features['prominence_x_strength'] = prominence * strength
features['distance_x_volatility'] = distance_to_current_pct / market_volatility
```

**Expected Impact:**
- Capture non-linear relationships
- Improved R² by 5-10%

---

### **Priority 3: Model Architecture & Hyperparameters**

#### **A. Hyperparameter Optimization**

Current config is **not optimized**:
```python
# Current (default values):
{
    'num_leaves': 31,           # Too high for small dataset
    'learning_rate': 0.05,      # Could be tuned
    'max_depth': 6,             # Could be optimized
    'min_data_in_leaf': 20,     # Too high for 161-sample fold
    'lambda_l1': 0.1,           # Not optimized
    'lambda_l2': 0.1            # Not optimized
}
```

**Use HPO to optimize:**
```python
from src.utils.ml_common.optimization.hpo_utils import optimize_hyperparameters

search_space = {
    'num_leaves': {'type': 'int', 'low': 15, 'high': 63},
    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.2, 'log': True},
    'max_depth': {'type': 'int', 'low': 3, 'high': 10},
    'min_data_in_leaf': {'type': 'int', 'low': 5, 'high': 50},
    'lambda_l1': {'type': 'float', 'low': 0.0, 'high': 1.0},
    'lambda_l2': {'type': 'float', 'low': 0.0, 'high': 1.0},
    'feature_fraction': {'type': 'float', 'low': 0.6, 'high': 1.0},
    'bagging_fraction': {'type': 'float', 'low': 0.6, 'high': 1.0}
}

# Run Bayesian optimization
results = optimize_hyperparameters(
    model_factory=lambda **params: lgb.LGBMRegressor(**params),
    X=X_train, y=y_train,
    search_space=search_space,
    n_trials=100,
    method='bayesian',
    scoring='r2',
    cv=5
)
```

**Expected Impact:**
- Improved R² by 10-20%
- Better generalization
- Reduced overfitting

#### **B. Model Ensemble**

```python
# Instead of single LGBM, use ensemble:
from sklearn.ensemble import VotingRegressor

models = [
    ('lgbm', lgb.LGBMRegressor(**best_params)),
    ('xgb', xgb.XGBRegressor(**xgb_params)),
    ('catboost', CatBoostRegressor(**cat_params))
]

ensemble = VotingRegressor(estimators=models)
ensemble.fit(X_train, y_train)
```

**Expected Impact:**
- Improved R² by 5-15%
- More robust predictions
- Lower variance

---

### **Priority 4: Cross-Validation Strategy**

#### **A. Use Purged TimeSeriesSplit**

Current CV may have data leakage:
```python
# Current: Standard 5-fold CV (temporal leakage possible)

# Better: Purged TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=5, gap=10)  # 10-period gap between folds
```

**Expected Impact:**
- Prevent temporal leakage
- More realistic validation
- More consistent R² across folds

#### **B. Stratified by Volatility Regime**

```python
# Ensure each fold has samples from all volatility regimes
from sklearn.model_selection import StratifiedKFold

# Create regime bins
regime_labels = pd.cut(training_df['market_volatility'], bins=3, labels=['low', 'med', 'high'])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Expected Impact:**
- Better fold balance
- Reduced variance between folds
- Improved generalization

---

### **Priority 5: Regularization & Overfitting Prevention**

#### **A. Stronger Regularization**

```python
# Current params allow overfitting
{
    'lambda_l1': 0.1,  # Too low
    'lambda_l2': 0.1,  # Too low
    'min_data_in_leaf': 20  # Too low for small folds
}

# Recommended:
{
    'lambda_l1': 0.5,       # Increase L1 (feature selection)
    'lambda_l2': 0.5,       # Increase L2 (weight decay)
    'min_data_in_leaf': 50, # Increase to prevent overfitting
    'min_child_samples': 30,
    'min_split_gain': 0.01,
    'path_smooth': 1.0      # Add path smoothing
}
```

#### **B. Early Stopping with Validation**

```python
# Use proper validation set for early stopping
lgb.train(
    params,
    train_set,
    valid_sets=[train_set, valid_set],
    valid_names=['train', 'valid'],
    num_boost_round=500,
    callbacks=[
        lgb.early_stopping(stopping_rounds=20),
        lgb.log_evaluation(period=10)
    ]
)
```

**Expected Impact:**
- Prevent overfitting
- More stable performance
- Improved R² by 5-10%

---

### **Priority 6: Target Variable Refinement**

#### **A. Multi-Objective Quality Score**

Current `quality_score` might be too simple. Create composite:

```python
def calculate_enhanced_quality_score(sr_level, forward_data, forward_days=20):
    """
    Calculate comprehensive quality score combining multiple aspects.
    """
    # 1. Bounce quality (how well it held)
    bounce_score = calculate_bounce_quality(sr_level, forward_data)
    
    # 2. Hold time (how long it lasted before breaking)
    hold_score = calculate_hold_duration(sr_level, forward_data) / forward_days
    
    # 3. Reaction strength (price action at level)
    reaction_score = calculate_reaction_strength(sr_level, forward_data)
    
    # 4. Predictive value (did it help predict moves?)
    prediction_score = calculate_predictive_value(sr_level, forward_data)
    
    # Weighted composite
    quality_score = (
        bounce_score * 0.3 +
        hold_score * 0.25 +
        reaction_score * 0.25 +
        prediction_score * 0.2
    )
    
    return quality_score
```

**Expected Impact:**
- More meaningful target
- Better signal
- Improved R² by 15-25%

#### **B. Regime-Specific Labels**

```python
# Quality may differ by market regime
def calculate_regime_aware_quality(sr_level, forward_data, regime):
    """
    Different quality metrics for different regimes.
    """
    if regime == 'trending':
        # In trends, SR levels should act as bounce points
        return calculate_bounce_quality(sr_level, forward_data)
    elif regime == 'ranging':
        # In ranges, SR levels should hold frequently
        return calculate_hold_frequency(sr_level, forward_data)
    elif regime == 'volatile':
        # In volatile markets, SR levels should have strong reactions
        return calculate_reaction_strength(sr_level, forward_data)
```

**Expected Impact:**
- More contextual quality assessment
- Better model understanding
- Improved R² by 10-15%

---

## 📈 **Implementation Roadmap**

### **Phase 1: Quick Wins (1-2 hours)**

1. ✅ **Fix SHAP generation** (DONE!)
2. **Increase training data** to 1 year
3. **Optimize hyperparameters** with Bayesian HPO (100 trials)
4. **Add stronger regularization**

**Expected R² improvement:** 0.128 → 0.20-0.25 (+50-100%)

### **Phase 2: Feature Engineering (2-3 hours)**

5. **Add time-based features** (recency, decay)
6. **Add confluence features** (method agreement)
7. **Add volume-weighted features**
8. **Create feature interactions** (top pairs)

**Expected R² improvement:** 0.20-0.25 → 0.30-0.35 (+25-40%)

### **Phase 3: Advanced Improvements (4-6 hours)**

9. **Use Purged TimeSeriesSplit** CV
10. **Create ensemble model** (LGBM + XGB + CatBoost)
11. **Improve quality_score calculation**
12. **Add regime-specific features**

**Expected R² improvement:** 0.30-0.35 → 0.45-0.55 (+30-50%)

---

## 🎯 **Expected Final Performance**

| Metric | Current | After Phase 1 | After Phase 2 | After Phase 3 | Improvement |
|--------|---------|---------------|---------------|---------------|-------------|
| **Avg Val R²** | 0.128 | 0.22 | 0.32 | **0.50** | **+291%** ✅ |
| **R² Std Dev** | 0.141 | 0.10 | 0.08 | **0.05** | **-65%** ✅ |
| **Worst Fold R²** | -0.017 | 0.10 | 0.20 | **0.35** | **Positive!** ✅ |
| **Val RMSE** | 0.238 | 0.20 | 0.18 | **0.15** | **-37%** ✅ |

---

## 📋 **Immediate Action Items**

### **1. Increase Training Data (Easiest)**

```bash
# Run with 1 year of data instead of 6 months
python scripts/run_sr_workflow.py \
    --symbol ETHUSDT \
    --exchange binance \
    --timeframe 15m \
    --ml-start-date 2024-11-01 \
    --ml-end-date 2025-11-01 \
    --ml-sample-freq-days 3  # Sample every 3 days
```

### **2. Optimize LGBM Hyperparameters**

Create script: `scripts/optimize_sr_quality_model.py`

```python
from src.utils.ml_common.optimization.hpo_utils import optimize_hyperparameters
import pandas as pd
import lightgbm as lgb

# Load training data
training_df = pd.read_parquet('data_cache/sr_ml_training/sr_quality_training_data.parquet')

# Prepare features
X = training_df.drop(columns=['quality_score', 'date', 'symbol', 'exchange', 'timeframe'])
y = training_df['quality_score']

# Define search space
search_space = {
    'num_leaves': {'type': 'int', 'low': 15, 'high': 63},
    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.2, 'log': True},
    'max_depth': {'type': 'int', 'low': 3, 'high': 10},
    'min_data_in_leaf': {'type': 'int', 'low': 10, 'high': 100},
    'lambda_l1': {'type': 'float', 'low': 0.0, 'high': 2.0},
    'lambda_l2': {'type': 'float', 'low': 0.0, 'high': 2.0},
    'feature_fraction': {'type': 'float', 'low': 0.6, 'high': 1.0},
    'bagging_fraction': {'type': 'float', 'low': 0.6, 'high': 1.0}
}

# Optimize
results = optimize_hyperparameters(
    model_factory=lambda **p: lgb.LGBMRegressor(**p, objective='regression', n_estimators=500),
    X=X, y=y,
    search_space=search_space,
    n_trials=100,
    method='bayesian',
    scoring='r2',
    cv=5
)

print(f"Best params: {results['best_params']}")
print(f"Best score: {results['best_score']}")
```

### **3. Add Feature Engineering**

File: `src/tactician/sr_levels/ml_quality.py`

Add to feature calculation:

```python
def calculate_enhanced_features(sr_level, market_data, current_date):
    """Enhanced feature set for better prediction."""
    features = {}
    
    # Existing features
    features.update(calculate_basic_features(sr_level))
    
    # NEW: Time-based features
    days_since = (current_date - sr_level.first_seen).days
    features['days_since_formation'] = days_since
    features['recency_score'] = np.exp(-days_since / 30)
    features['age_category'] = 'new' if days_since < 7 else ('medium' if days_since < 30 else 'old')
    
    # NEW: Confluence features
    features['method_count'] = len(sr_level.detection_methods)
    features['method_diversity_score'] = calculate_method_diversity(sr_level)
    
    # NEW: Volume features
    features['volume_consistency'] = calculate_volume_consistency(sr_level, market_data)
    features['volume_trend'] = calculate_volume_trend(sr_level, market_data)
    
    # NEW: Interaction features
    features['dist_x_velocity'] = features['distance_to_current_pct'] * features['approach_velocity']
    features['prominence_x_strength'] = features['prominence'] * features['strength']
    
    return features
```

---

## 🎓 **Best Practices for SR Quality Modeling**

### **1. Target Variable Design**
- ✅ Use **forward-looking** performance (already doing)
- ✅ Weight recent tests more heavily
- ✅ Combine multiple quality aspects (bounces, holds, reactions)
- ✅ Normalize by market conditions

### **2. Feature Selection**
- ✅ Keep features with LGBM importance > 10
- ✅ Add temporal features (time decay, recency)
- ✅ Add regime-aware features (volatility-adjusted)
- ✅ Create top feature interactions

### **3. Model Training**
- ✅ Use proper temporal CV (TimeSeriesSplit with gap)
- ✅ Optimize hyperparameters with Bayesian HPO
- ✅ Use ensemble for robustness
- ✅ Monitor for overfitting

### **4. Validation**
- ✅ Check performance across different regimes
- ✅ Validate on out-of-sample data
- ✅ Test on different symbols/timeframes
- ✅ Monitor production performance

---

## 📊 **Diagnostic Checks**

### **Check 1: Data Distribution**
```python
# Analyze training data
training_df = pd.read_parquet('data_cache/sr_ml_training/sr_quality_training_data.parquet')

print("Quality Score Distribution:")
print(training_df['quality_score'].describe())
print(f"\nSkewness: {training_df['quality_score'].skew():.3f}")
print(f"Kurtosis: {training_df['quality_score'].kurt():.3f}")

# Check for outliers
print(f"\nOutliers (> 3 std): {(abs(training_df['quality_score'] - training_df['quality_score'].mean()) > 3*training_df['quality_score'].std()).sum()}")
```

### **Check 2: Feature Correlations**
```python
# Find redundant features
corr_matrix = training_df.select_dtypes(include=[np.number]).corr()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.9:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

print("Highly correlated features (r > 0.9):")
for f1, f2, corr in high_corr_pairs:
    print(f"  {f1} <-> {f2}: {corr:.3f}")
```

### **Check 3: Feature-Target Correlation**
```python
# Find which features correlate with quality_score
target_corr = training_df.select_dtypes(include=[np.number]).corrwith(training_df['quality_score']).abs().sort_values(ascending=False)

print("\nTop 10 features correlated with quality_score:")
print(target_corr.head(10))
```

---

## 🚀 **Quick Start: Immediate Improvements**

### **Step 1: Run with More Data**
```bash
python scripts/run_sr_workflow.py \
    --symbol ETHUSDT \
    --timeframe 15m \
    --ml-start-date 2024-05-01 \
    --ml-end-date 2025-11-01 \
    --ml-sample-freq-days 3
```

### **Step 2: Optimize Hyperparameters**
```bash
# Create and run optimization script (I can create this)
python scripts/optimize_sr_quality_model.py
```

### **Step 3: Validate Improvements**
```bash
# Re-run workflow with optimized model
python scripts/run_sr_workflow.py --symbol ETHUSDT --timeframe 15m
```

---

## 📈 **Expected Improvement Timeline**

| Phase | Action | Time | R² Before | R² After | Status |
|-------|--------|------|-----------|----------|--------|
| **Current** | Baseline | - | - | 0.128 | ✅ |
| **Quick Fix** | More data + HPO | 1-2 hrs | 0.128 | 0.22 | 🎯 Next |
| **Feature Eng** | Add features | 2-3 hrs | 0.22 | 0.32 | 🎯 |
| **Advanced** | Ensemble + refinement | 4-6 hrs | 0.32 | 0.50+ | 🎯 |

---

## ✅ **Summary**

**SHAP Fix:** ✅ **DONE** - Filters non-numeric columns now

**Model Improvements Needed:**
1. 🎯 **Increase training data** (1 year instead of 6 months)
2. 🎯 **Optimize hyperparameters** (Bayesian HPO with 100 trials)
3. 🎯 **Add time-based features** (recency, decay)
4. 🎯 **Improve quality_score** (multi-objective, regime-aware)
5. 🎯 **Use ensemble** (LGBM + XGB + CatBoost)
6. 🎯 **Fix CV strategy** (Purged TimeSeriesSplit)

**Expected Final R²:** **0.45-0.55** (vs. current 0.128)

**Would you like me to create the hyperparameter optimization script or implement any of these improvements?** 🚀
