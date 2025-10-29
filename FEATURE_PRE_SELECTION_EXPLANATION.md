# Feature Pre-Selection in Regime Models Training

## Overview

Before training ML models for regime classification, we perform **feature pre-selection** to reduce dimensionality, improve model performance, and reduce training time. This document explains the feature pre-selection process used in `regime_models_training` and `regime_ensemble_training`.

## Feature Pre-Selection Flow

### 1. Feature Generation Phase

**Location**: `_prepare_training_data()` / `_prepare_training_data_improved()`

**Process**:
1. Generate comprehensive features using the feature bank system
2. Categories included:
   - Momentum features
   - Volatility features
   - Volume features
   - Trend features
   - Oscillator features
   - Returns features
   - Advanced regime features (if available)
3. Add smoothed features (rolling aggregates, EWM) if enabled
   - **Important**: Already-smoothed features (detected by name patterns) are skipped to avoid double-smoothing
   - Detection patterns: `_ma\d+`, `_std\d+`, `_ewm[\d.]+`, `lagged_`, `_lagged`, `_lag\d+`

**Result**: Raw feature matrix `X` with original feature names

### 2. Feature Selection Phase

**Location**: `_run_feature_selection()`

**Method**: Model-based feature selection using `SelectFromModel` with LightGBM

**Algorithm**:
```python
selector = SelectFromModel(
    lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
        class_weight='balanced',
        importance_type='gain',
        verbose=-1
    ),
    threshold='median',  # Use median threshold
    max_features=min(100, features.shape[1])  # Cap at 100 features
)
```

**Selection Criteria**:
- Uses LightGBM's **feature importance** (gain-based)
- Selects features above **median importance threshold**
- For large feature sets (>100 features): caps at 100 features
- For smaller sets: keeps at least 75% of features

**Fallback Strategy**:
If model-based selection fails:
1. **Primary fallback**: Variance-based selection
   - Selects top features by variance
   - Keeps at least 90% of features or minimum 90 features
2. **Ultimate fallback**: Use all features
   - Prevents complete failure
   - Logs warning

### 3. Feature Selection Output

The selection process returns `feature_selection_info` dictionary:

```python
{
    'selection_performed': True,
    'selection_method': 'lightgbm_selectfrommodel',  # or 'variance_fallback'
    'selected_indices': [0, 1, 5, 10, ...],  # Indices of selected features
    'selected_feature_names': ['feature_1', 'feature_2', ...],
    'retained_feature_count': 87,
    'total_feature_count': 245,
    'feature_importances': {  # Dictionary mapping feature names to importance scores
        'feature_1': 0.1234,
        'feature_2': 0.0987,
        ...
    },
    'importance_ranking': [  # Top 20 features ranked by importance
        {'feature': 'feature_1', 'importance': 0.1234, 'rank': 1},
        ...
    ],
    'top_features_preview': 'feature_1 (0.1234), feature_2 (0.0987), ...',
    'selection_time_seconds': 2.345
}
```

### 4. Feature Application Phase

**Location**: `_apply_feature_selection()`

**Process**:
1. Takes original feature matrix `X` and `feature_selection_info`
2. Extracts `selected_indices` from selection info
3. Filters features: `X_selected = X[:, selected_indices]`
4. Updates feature names to match selected features

**Result**: Reduced feature matrix ready for model training

## Why Pre-Select Features?

### Benefits:
1. **Reduced Overfitting**: Fewer features reduce model complexity
2. **Faster Training**: Smaller feature space speeds up training
3. **Better Generalization**: Model focuses on most informative features
4. **Interpretability**: Easier to understand model decisions
5. **Memory Efficiency**: Smaller matrices use less memory

### Feature Selection Strategy:

**Model-Based Selection (Primary)**:
- Uses LightGBM to learn feature importance
- Selects features that contribute most to regime classification
- Accounts for feature interactions through tree-based learning

**Variance-Based Fallback**:
- Selects features with highest variance
- Assumes high-variance features contain more information
- Simple and fast fallback

## Integration Points

### Before Training:
```python
# 1. Generate features
X, feature_names = self._generate_features_with_bank(data)

# 2. Perform feature selection
feature_selection_info = self._run_feature_selection(X, y, feature_names)

# 3. Apply selection
if feature_selection_info.get('selected_indices'):
    X = self._apply_feature_selection(X, feature_selection_info)
    feature_names = feature_selection_info.get('selected_feature_names', feature_names)
```

### During Training:
Feature selection info is passed to `_train_regime_models()`:
- Used for logging and reporting
- Included in final artifacts
- Helps track which features were important

### After Training:
Feature selection info is stored in artifacts:
```python
'artifacts': {
    'regime_models_training_result': {
        'feature_selection': feature_selection_info,
        'selected_feature_names': [...],
        ...
    }
}
```

## Configuration

Feature selection can be controlled via:
- **Selection threshold**: Currently uses 'median' (configurable)
- **Max features**: Capped at 100 for large feature sets
- **Minimum retention**: At least 75-90% of features retained based on dataset size

## Performance Considerations

- **Selection Time**: Typically 1-5 seconds for 200-500 features
- **Memory**: Minimal overhead (creates temporary LightGBM model)
- **Scalability**: Handles feature sets up to several thousand features

## Metrics Tracking

The selection process tracks:
- Number of features retained vs. total
- Selection method used
- Feature importance scores
- Top features preview
- Selection time

This information is logged and included in training artifacts for analysis.
