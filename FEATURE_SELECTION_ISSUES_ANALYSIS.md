# Feature Selection Issues Analysis

**Date:** 2025-11-09  
**Report:** outcomes/final_feature_selection_outcome_report_report_20251109_161522.md

## Critical Issues Identified

### 1. **Cross-Validation Consistency: 0.0000 (CRITICAL BUG)**

**Problem:**
- Average consistency: 0.0000
- Consistent features: 0
- This indicates NO features are being selected consistently across CV folds

**Root Cause:**
The `_select_features_for_window()` method (line 750-767) uses a **different selection method** than the main feature selection:
- **Main selection:** Uses SHAP/permutation importance (captures interactions)
- **CV window selection:** Uses `mutual_info_regression` with `SelectKBest`
- **Result:** Features selected by SHAP never match features selected by mutual info

**Code Location:**
```python
# Line 754-756 in final_feature_selection.py
selector = SelectKBest(
    score_func=mutual_info_regression,  # ❌ WRONG: Should use same method as main selection
    k=max_window_features
)
```

**Fix Required:**
The CV analysis should use the SAME selection method (SHAP/permutation) as the main feature selection. Currently it's comparing apples to oranges.

---

### 2. **Baseline Performance: 0.82x (Features Underperform)**

**Problem:**
- Improvement ratio: 0.82x (should be > 1.0)
- Selected features avg score: 0.058510
- Baseline avg score: 0.071277
- **This means randomly selected features perform BETTER than our carefully selected ones!**

**Root Cause:**
This is likely caused by:
1. **Overfitting to SHAP importance:** SHAP values might be capturing noise rather than signal
2. **Small sample size:** Only 100 samples after alignment (see error log)
3. **Data leakage in baseline:** The baseline might be using the full dataset while selected features used aligned subset

**Evidence from logs:**
```
📊 After alignment - base_features shape: (3155, 29)
📊 Aligning dataframes using 100 common indices
```

The dataset was reduced from 3155 to 100 samples due to alignment issues. This is **far too small** for reliable feature selection.

---

### 3. **Max Correlation: 1.0000 (Perfect Correlation Detected)**

**Problem:**
- Max correlation: 1.0000
- High correlation pairs: 3
- This indicates duplicate or perfectly correlated features

**Root Cause:**
Despite the `_remove_exact_duplicates()` method, some features are still perfectly correlated. This could be:
1. **Same feature with different names**
2. **Derived features that are mathematically equivalent**
3. **Features with identical values after transformations**

**Evidence:**
Multiple momentum features are selected:
- `momentum_endpoints_sma_20`
- `momentum_14_price_returns`
- `momentum_21_price_returns`
- `vectorbt_momentum_5_price_returns`
- `vectorbt_momentum_20_price_returns`

These might be calculating similar momentum metrics over different windows, leading to high correlation.

---

### 4. **SHAP Analysis: 0 Generated (Missing Interpretability)**

**Problem:**
- SHAP analyses: 0
- This means no SHAP plots or detailed feature importance visualizations were generated

**Root Cause:**
The SHAP analysis generation step is likely failing silently or not being called. Check:
1. SHAP library availability
2. Error handling in SHAP generation code
3. Whether SHAP generation is conditional on some flag

---

### 5. **Too Many Momentum Features (Feature Diversity Issue)**

**Problem:**
5 out of top 20 features are momentum-based:
1. momentum_endpoints_sma_20 (#3)
2. momentum_14_price_returns (#9)
3. momentum_21_price_returns (#10)
4. vectorbt_momentum_5_price_returns (#18)
5. vectorbt_momentum_20_price_returns (#19)

**Root Cause:**
The hierarchical redundancy removal (correlation threshold: 0.85) is not aggressive enough to remove similar momentum features. These features likely have correlation < 0.85 but are still measuring the same underlying phenomenon.

**Impact:**
- Reduced feature diversity
- Potential overfitting to momentum regime
- Poor generalization to non-trending markets

---

## Recommended Fixes

### Priority 1: Fix CV Consistency (CRITICAL)

**File:** `src/training/steps/pre_training/components/final_feature_selection.py`  
**Method:** `_select_features_for_window()` (line 750-767)

**Change:**
```python
def _select_features_for_window(self, X_window: pd.DataFrame, y_window: pd.Series) -> List[str]:
    """Select features using SAME method as main selection."""
    try:
        # Use SHAP/permutation importance (same as main selection)
        if self.config.use_permutation_importance and LGBM_AVAILABLE and SHAP_AVAILABLE:
            model = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=50,  # Fewer for speed
                verbose=-1,
                random_state=42
            )
            model.fit(X_window, y_window)
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_window)
            importances = np.mean(np.abs(shap_values), axis=0)
        else:
            # Fallback to permutation importance
            model = ExtraTreesRegressor(n_estimators=50, random_state=42)
            model.fit(X_window, y_window)
            perm_importance = permutation_importance(
                model, X_window, y_window,
                n_repeats=5,
                random_state=42
            )
            importances = perm_importance.importances_mean
        
        # Select top features
        max_window_features = min(20, len(X_window.columns))
        top_indices = np.argsort(importances)[::-1][:max_window_features]
        selected_features = [X_window.columns[i] for i in top_indices]
        
        return selected_features
        
    except Exception as e:
        self.logger.error(f"Error selecting features for window: {e}")
        return []
```

### Priority 2: Fix Data Alignment Issues

**Problem:** Dataset reduced from 3155 to 100 samples  
**Location:** `feature_generation_final_feature_selection_step.py` lines 650-700

**Fix:**
1. Investigate why alignment is failing
2. Use forward-fill or interpolation instead of dropping misaligned rows
3. Ensure all feature sources use the same timestamp index

### Priority 3: Improve Feature Diversity

**File:** `final_feature_selection.py`  
**Method:** `_reduce_redundancy_hierarchical()`

**Changes:**
1. Lower correlation threshold from 0.85 to 0.75
2. Add feature category diversity constraint (max 2-3 features per category)
3. Implement semantic grouping for momentum features

**Example:**
```python
# Group similar features
feature_groups = {
    'momentum': ['momentum_', 'vectorbt_momentum_'],
    'volume': ['volume_', 'obv', 'cmf'],
    'volatility': ['volatility_', 'atr', 'bbands'],
    # ... etc
}

# Limit features per group
max_per_group = 2
```

### Priority 4: Enable SHAP Analysis

**Check:**
1. Verify SHAP library is installed
2. Add explicit SHAP plot generation
3. Save SHAP summary plots and feature importance plots

### Priority 5: Improve Baseline Comparison

**Fix:**
1. Ensure baseline uses same data subset as selected features
2. Use multiple random seeds for baseline (currently 10 trials)
3. Add statistical significance test (t-test or Mann-Whitney U)

---

## Expected Improvements After Fixes

1. **CV Consistency:** Should increase from 0.0 to 0.4-0.6
2. **Baseline Performance:** Should improve from 0.82x to 1.2-1.5x
3. **Feature Diversity:** Reduce momentum features from 5 to 2-3
4. **Correlation:** Max correlation should drop from 1.0 to < 0.85
5. **SHAP Analysis:** Should generate plots and detailed importance scores

---

## Additional Recommendations

### 1. Add Feature Category Metadata
Track feature categories (momentum, volume, volatility, etc.) to ensure diversity.

### 2. Implement Rolling Window Validation
Instead of fixed time splits, use expanding window validation for time series.

### 3. Add Feature Importance Stability Metric
Track how feature importance changes across different data periods.

### 4. Implement Feature Selection Ensemble
Combine multiple selection methods (SHAP, permutation, mutual info) with voting.

### 5. Add Data Quality Checks
- Minimum sample size requirements (e.g., > 1000 samples)
- Maximum NaN percentage per feature
- Minimum variance requirements

---

## Summary

The feature selection pipeline has **4 critical bugs** that need immediate attention:

1. ❌ **CV consistency is broken** (using wrong selection method)
2. ❌ **Selected features underperform baseline** (data alignment issues)
3. ⚠️ **Too many correlated features** (weak redundancy removal)
4. ⚠️ **Missing SHAP analysis** (no interpretability)

The most critical fix is **Priority 1** (CV consistency), as it indicates a fundamental mismatch between the main selection method and the validation method.
