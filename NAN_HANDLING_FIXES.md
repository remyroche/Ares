# NaN Handling Fixes for VectorizedFeatureSelector

## Problem
The `VectorizedFeatureSelector` class was failing with the following error:
```
Error removing high VIF features: Input X contains NaN.
LinearRegression does not accept missing values encoded as NaN natively.
```

This error occurred because the feature selection methods were trying to use machine learning algorithms (LinearRegression, mutual_info_classif, LightGBM) on data containing NaN values without proper preprocessing.

## Root Cause
The following methods in `src/training/steps/vectorized_labelling_orchestrator.py` were not handling NaN values:

1. `_remove_high_vif_features_vectorized()` - Uses LinearRegression
2. `_remove_low_mutual_info_features_vectorized()` - Uses mutual_info_classif
3. `_remove_low_importance_features_vectorized()` - Uses LightGBM
4. `_remove_correlated_features_vectorized()` - Uses correlation calculation

## Solution
Added comprehensive NaN handling using `sklearn.impute.SimpleImputer` with median strategy:

### 1. Main Feature Selection Method
Added NaN handling at the beginning of `select_optimal_features()`:
```python
# Handle NaN values at the beginning
if numeric_data.isnull().any().any():
    self.logger.info("Handling NaN values in feature selection...")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    numeric_data = pd.DataFrame(
        imputer.fit_transform(numeric_data),
        columns=numeric_data.columns,
        index=numeric_data.index
    )
```

### 2. VIF Calculation Method
Added NaN handling in `_remove_high_vif_features_vectorized()`:
```python
# Handle NaN values by imputing with median
imputer = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(
    imputer.fit_transform(data),
    columns=data.columns,
    index=data.index
)
```

### 3. Mutual Information Method
Added NaN handling in `_remove_low_mutual_info_features_vectorized()`:
```python
# Handle NaN values by imputing with median
imputer = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(
    imputer.fit_transform(data),
    columns=data.columns,
    index=data.index
)
```

### 4. LightGBM Importance Method
Added NaN handling in `_remove_low_importance_features_vectorized()`:
```python
# Handle NaN values by imputing with median
imputer = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(
    imputer.fit_transform(data),
    columns=data.columns,
    index=data.index
)
```

### 5. Correlation Method
Added NaN handling in `_remove_correlated_features_vectorized()`:
```python
# Handle NaN values by imputing with median for correlation calculation
imputer = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(
    imputer.fit_transform(data),
    columns=data.columns,
    index=data.index
)
```

## Verification
Created and ran a test script that:
1. Generated synthetic data with NaN values
2. Tested all feature selection methods
3. Verified that no errors occur with NaN values
4. Confirmed that feature selection completes successfully

## Benefits
- **Robustness**: Feature selection now works with real-world data containing missing values
- **Consistency**: All methods use the same NaN handling strategy (median imputation)
- **Logging**: Added informative logging when NaN values are detected and handled
- **Performance**: Median imputation is computationally efficient and preserves data distribution

## Files Modified
- `src/training/steps/vectorized_labelling_orchestrator.py` - Main fixes
- `NAN_HANDLING_FIXES.md` - This documentation

## Notes
- The `enhanced_coarse_optimizer.py` file already had proper NaN handling in its mutual information calculations
- Used median imputation strategy as it's robust to outliers and preserves data distribution
- All changes maintain backward compatibility 