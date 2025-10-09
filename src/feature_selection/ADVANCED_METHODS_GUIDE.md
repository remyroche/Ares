# Advanced Feature Selection Methods Guide

## 🚀 Overview

This guide covers the advanced feature selection methods implemented using LASSO, RandomForest, and LightGBM with permutation importance and comprehensive validation framework.

## 🧠 Advanced Methods

### 1. LASSO Feature Selection
- **Algorithm**: LASSO with cross-validation for optimal alpha selection
- **Strengths**: Handles multicollinearity, provides sparse solutions, interpretable coefficients
- **Use Cases**: High-dimensional data, when sparsity is desired, linear relationships

### 2. RandomForest Feature Selection
- **Algorithm**: RandomForest with feature importance ranking
- **Strengths**: Handles non-linear relationships, robust to outliers, no feature scaling needed
- **Use Cases**: Non-linear data, mixed data types, when interpretability is important

### 3. LightGBM Feature Selection
- **Algorithm**: LightGBM with feature importance and built-in regularization
- **Strengths**: Fast training, handles categorical features, memory efficient
- **Use Cases**: Large datasets, categorical data, when speed is important

### 4. Ensemble Advanced Selection
- **Algorithm**: Combines all three methods with weighted voting
- **Strengths**: Robust to method biases, captures different aspects, reduces overfitting
- **Use Cases**: When you want the best of all methods, critical applications

## 🔧 Permutation Importance

### Features
- **Cross-validation aware**: Uses CV for robust importance calculation
- **Parallel processing**: Hardware-optimized parallel computation
- **Stability checking**: Validates importance stability across repeats
- **Feature interactions**: Calculates pairwise feature interactions

### Configuration
```python
from src.feature_selection import PermutationConfig, create_permutation_calculator

config = PermutationConfig(
    n_repeats=10,
    scoring='neg_mean_squared_error',
    cv_folds=5,
    enable_parallel=True,
    enable_stability_check=True
)

calculator = create_permutation_calculator(config)
```

## 📊 Validation Framework

### Cross-Validation
- **Multiple strategies**: KFold, TimeSeriesSplit, StratifiedKFold
- **Comprehensive metrics**: Performance, stability, interpretability
- **Hardware optimization**: M1-optimized parallel processing

### Regression Testing
- **Reference tracking**: Saves and compares results over time
- **Performance monitoring**: Detects performance regressions
- **Automatic thresholding**: Configurable regression detection

### Validation Metrics
- **Selection quality**: Variance, correlation, mutual information
- **Stability metrics**: Bootstrap stability, consistency across folds
- **Interpretability**: Feature count, independence, complexity

## 🎯 Quick Start

### Basic Advanced Selection

```python
from src.feature_selection import AdvancedFeatureSelector, AdvancedSelectionConfig

# Configure advanced selection
config = AdvancedSelectionConfig(
    enable_permutation_importance=True,
    enable_hardware_optimization=True,
    cv_folds=5
)

# Create selector
selector = AdvancedFeatureSelector(config)

# Select features
result = selector.select_features(X, y, method='ensemble', n_features=50)

print(f"Selected {len(result['selected_features'])} features")
print(f"Method: {result['method']}")
```

### Individual Method Selection

```python
from src.feature_selection import (
    LASSOFeatureSelector,
    RandomForestFeatureSelector,
    LightGBMFeatureSelector
)

# LASSO selection
lasso_selector = LASSOFeatureSelector()
lasso_result = lasso_selector.select_features(X, y, n_features=30)

# RandomForest selection
rf_selector = RandomForestFeatureSelector()
rf_result = rf_selector.select_features(X, y, n_features=40)

# LightGBM selection
lgb_selector = LightGBMFeatureSelector()
lgb_result = lgb_selector.select_features(X, y, n_features=35)
```

### Method Comparison

```python
# Compare all methods
comparison_result = selector.compare_methods(X, y, n_features=50)

print("Method Comparison Results:")
for method, result in comparison_result['results'].items():
    print(f"{method}: {result['n_selected']} features")
```

## 🔍 Permutation Importance Usage

### Calculate Permutation Importance

```python
from src.feature_selection import PermutationImportanceCalculator, PermutationConfig

# Configure permutation importance
config = PermutationConfig(
    n_repeats=10,
    scoring='neg_mean_squared_error',
    enable_parallel=True
)

calculator = PermutationImportanceCalculator(config)

# Calculate importance for a fitted model
importance_result = calculator.calculate_importance(model, X, y, feature_names)

print("Feature Importance:")
for feature, importance in importance_result['feature_importance'].items():
    print(f"{feature}: {importance['importance_mean']:.4f} ± {importance['importance_std']:.4f}")
```

### Feature Interactions

```python
# Calculate feature interactions
interaction_result = calculator.calculate_feature_interactions(
    model, X, y, top_features=10
)

print("Feature Interactions:")
for (feat1, feat2), importance in interaction_result['interaction_scores'].items():
    print(f"{feat1} × {feat2}: {importance:.4f}")
```

## 📈 Validation Framework Usage

### Cross-Validation

```python
from src.feature_selection import FeatureSelectionValidator, ValidationConfig

# Configure validation
config = ValidationConfig(
    cv_folds=5,
    cv_strategy='kfold',
    enable_regression_testing=True
)

validator = FeatureSelectionValidator(config)

# Define selection function
def my_selection_func(X, y, **kwargs):
    selector = AdvancedFeatureSelector()
    return selector.select_features(X, y, method='ensemble', **kwargs)

# Validate selection method
validation_result = validator.validate_selection_method(
    X, y, my_selection_func, test_name="my_test"
)

print("Validation Results:")
print(f"Overall Success: {validation_result['overall_success']}")
print(f"CV Results: {validation_result['cross_validation']['aggregated_results']}")
```

### Regression Testing

```python
# Run regression test
regression_result = validator.validate_selection_method(
    X, y, my_selection_func, test_name="regression_test"
)

if regression_result['regression_test']['regression_detected']:
    print("⚠️ Regression detected!")
else:
    print("✅ No regression detected")
```

## ⚙️ Configuration Options

### Advanced Selection Configuration

```python
from src.feature_selection import AdvancedSelectionConfig

config = AdvancedSelectionConfig(
    # General settings
    random_state=42,
    n_jobs=-1,
    enable_hardware_optimization=True,
    
    # LASSO settings
    lasso_cv_folds=5,
    lasso_alphas=(0.001, 1.0),
    lasso_n_alphas=100,
    
    # RandomForest settings
    rf_n_estimators=100,
    rf_max_depth=None,
    rf_max_features='sqrt',
    
    # LightGBM settings
    lgb_n_estimators=100,
    lgb_learning_rate=0.1,
    lgb_max_depth=6,
    
    # Validation settings
    cv_folds=5,
    enable_time_series_cv=True,
    
    # Performance settings
    enable_permutation_importance=True,
    permutation_n_repeats=10
)
```

### Permutation Importance Configuration

```python
from src.feature_selection import PermutationConfig

config = PermutationConfig(
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    cv_folds=5,
    enable_parallel=True,
    enable_stability_check=True,
    stability_threshold=0.1
)
```

### Validation Configuration

```python
from src.feature_selection import ValidationConfig

config = ValidationConfig(
    cv_folds=5,
    cv_strategy='kfold',  # 'kfold', 'timeseries', 'stratified'
    test_size=0.2,
    random_state=42,
    enable_regression_testing=True,
    regression_threshold=0.1,
    enable_performance_metrics=True,
    enable_stability_metrics=True,
    enable_interpretability_metrics=True
)
```

## 📊 Performance Monitoring

### Get Performance Statistics

```python
# Get selector performance stats
stats = selector.get_performance_stats()
print("Selector Performance:", stats)

# Get permutation importance stats
perm_stats = calculator.get_performance_stats()
print("Permutation Stats:", perm_stats)

# Get validation stats
validation_stats = validator.cv_framework.metrics_calculator.get_performance_stats()
print("Validation Stats:", validation_stats)
```

### Monitor Selection Quality

```python
# Calculate selection metrics
from src.feature_selection import ValidationMetrics

metrics_calculator = ValidationMetrics()
metrics = metrics_calculator.calculate_selection_metrics(
    X, y, selected_features, feature_names
)

print("Selection Quality Metrics:")
for metric, value in metrics['metrics'].items():
    print(f"{metric}: {value}")
```

## 🔧 Advanced Usage

### Custom Ensemble Weights

```python
# Create ensemble with custom weights
ensemble_selector = EnsembleAdvancedSelector()

# Define custom weights
weights = {
    'lasso': 0.4,
    'random_forest': 0.3,
    'lightgbm': 0.3
}

# Select features with custom weights
result = ensemble_selector.select_features(
    X, y, method='ensemble', weights=weights
)
```

### Custom Scoring Function

```python
from sklearn.metrics import mean_absolute_error

# Define custom scoring function
def custom_scoring(y_true, y_pred):
    return -mean_absolute_error(y_true, y_pred)

# Use custom scoring in permutation importance
calculator = PermutationImportanceCalculator()
result = calculator.calculate_importance(
    model, X, y, custom_scoring=custom_scoring
)
```

### Time Series Cross-Validation

```python
# Configure for time series data
config = ValidationConfig(
    cv_strategy='timeseries',
    cv_folds=5
)

validator = FeatureSelectionValidator(config)
result = validator.validate_selection_method(X, y, selection_func)
```

## 🐛 Troubleshooting

### Common Issues

1. **LightGBM Import Error**
   ```python
   # Install LightGBM
   pip install lightgbm
   ```

2. **Memory Issues with Large Datasets**
   ```python
   # Reduce permutation repeats
   config = PermutationConfig(n_repeats=5)
   
   # Use fewer CV folds
   config = ValidationConfig(cv_folds=3)
   ```

3. **Slow Performance**
   ```python
   # Enable hardware optimization
   config = AdvancedSelectionConfig(enable_hardware_optimization=True)
   
   # Use parallel processing
   config = PermutationConfig(enable_parallel=True, n_jobs=-1)
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
from src.utils.tprint import tprint_debug
tprint_debug("Debug information will be shown")
```

## 📚 Complete Example

```python
import numpy as np
from src.feature_selection import (
    AdvancedFeatureSelector,
    FeatureSelectionValidator,
    AdvancedSelectionConfig,
    ValidationConfig
)

# Generate sample data
X = np.random.rand(1000, 100)
y = np.random.rand(1000)

# Configure advanced selection
selection_config = AdvancedSelectionConfig(
    enable_permutation_importance=True,
    enable_hardware_optimization=True,
    cv_folds=5
)

# Create selector
selector = AdvancedFeatureSelector(selection_config)

# Select features
result = selector.select_features(X, y, method='ensemble', n_features=20)

print(f"Selected {len(result['selected_features'])} features")
print(f"Method: {result['method']}")

# Validate selection
validation_config = ValidationConfig(
    cv_folds=5,
    enable_regression_testing=True
)

validator = FeatureSelectionValidator(validation_config)

def selection_func(X, y, **kwargs):
    return selector.select_features(X, y, **kwargs)

validation_result = validator.validate_selection_method(
    X, y, selection_func, test_name="advanced_test"
)

print(f"Validation Success: {validation_result['overall_success']}")
print(f"CV Results: {validation_result['cross_validation']['aggregated_results']}")
```

## 🎉 Conclusion

The advanced feature selection methods provide state-of-the-art algorithms with comprehensive validation and monitoring capabilities. The permutation importance calculation ensures robust feature ranking, while the validation framework provides confidence in the selection quality.

Key benefits:
- **Multiple algorithms**: LASSO, RandomForest, LightGBM
- **Robust validation**: Cross-validation and regression testing
- **Hardware optimization**: M1-optimized performance
- **Comprehensive metrics**: Quality, stability, interpretability
- **Easy to use**: Simple API with powerful configuration options