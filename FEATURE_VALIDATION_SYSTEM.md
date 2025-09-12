# Feature Validation System

## Overview

A centralized feature validation system has been implemented in `src/feature_engineering/math_validation.py` to automatically detect and warn about problematic features during feature generation. This system addresses the need to immediately identify features that have infinite values, are constant, generate NaN, or are mostly zeros.

## Key Features

✅ **Automatic Detection of:**
- Infinite values (`inf`, `-inf`)
- Constant features (all values identical)
- NaN values
- Excessive zero values (>1% zeros)
- Zero standard deviation
- **Warm-up period exclusion** (first 50 rows excluded by default)

✅ **Multiple Usage Patterns:**
- Individual feature validation
- DataFrame-wide validation
- Automatic decorator-based validation
- Configurable warning levels

✅ **Integration with Existing Code:**
- Works with existing `math_validation.py` functions
- Compatible with current feature engineering classes
- Minimal code changes required

## Usage Examples

### 1. Individual Feature Validation

```python
from src.feature_engineering.math_validation import validate_feature_quality
import numpy as np

# Validate a single feature (excludes first 50 rows by default)
feature_data = np.array([1.0, 2.0, np.nan, 4.0, np.inf])
report = validate_feature_quality(feature_data, "my_feature")

print(f"Issues found: {len(report['issues'])}")
print(f"Excluded rows: {report.get('excluded_first_rows', 0)}")
for issue in report['issues']:
    print(f"- {issue}")

# Custom exclusion settings
report_custom = validate_feature_quality(
    feature_data, "my_feature", 
    exclude_first_rows=100  # Exclude first 100 rows instead of 50
)
```

### 2. DataFrame Validation

```python
from src.feature_engineering.math_validation import validate_features_dataframe
import pandas as pd

# Validate all numeric columns in a DataFrame (excludes first 50 rows by default)
df = pd.DataFrame({
    'good_feature': [1, 2, 3, 4, 5],
    'constant_feature': [1, 1, 1, 1, 1],
    'nan_feature': [1, 2, np.nan, 4, 5]
})

validation_results = validate_features_dataframe(df)
for feature_name, report in validation_results.items():
    if report['issues']:
        print(f"{feature_name}: {len(report['issues'])} issues")
        print(f"  Excluded rows: {report.get('excluded_first_rows', 0)}")

# Custom exclusion settings
validation_results_custom = validate_features_dataframe(
    df, exclude_first_rows=100  # Exclude first 100 rows
)
```

### 3. Automatic Validation with Decorators

```python
from src.feature_engineering.math_validation import validate_generated_features
import pandas as pd

@validate_generated_features
def create_my_features(data: pd.DataFrame) -> pd.DataFrame:
    """This function will automatically validate its output."""
    result = data.copy()
    result['new_feature'] = data['price'] / data['volume']  # Might create inf/NaN
    result['constant_feature'] = 1.0  # Will be flagged as constant
    return result

# When called, this will automatically validate the output and issue warnings
# Example output:
# WARNING: Function create_my_features generated 2 feature quality issues:
#   Feature 'new_feature': 1 issue
#     - Feature 'new_feature' contains 5 infinite values
#   Feature 'constant_feature': 1 issue
#     - Feature 'constant_feature' is constant (all values = 1.0)
enhanced_data = create_my_features(market_data)
```

### 4. Pipeline Validation

```python
from src.feature_engineering.math_validation import validate_feature_pipeline

@validate_feature_pipeline
def my_feature_pipeline(input_data: pd.DataFrame) -> pd.DataFrame:
    """Validates both input and output."""
    # Process data...
    return processed_data
```

### 5. Strict Validation (Raises Exceptions)

```python
from src.feature_engineering.math_validation import strict_feature_validation

@strict_feature_validation
def critical_feature_generation(data: pd.DataFrame) -> pd.DataFrame:
    """Raises exceptions on critical issues instead of just warnings."""
    # Process data...
    return processed_data
```

## Integration with Existing Feature Engineering

The validation system has been integrated into the main feature engineering classes:

### EnhancedFeatureEngineering Class

The main feature generation method now includes automatic validation:

```python
# In step06_enhanced_feature_engineering.py
@validate_generated_features
@inject_utilities('common_ops', 'data_proc', 'math_val', 'm1_memory', 'm1_cpu')
async def create_enhanced_features_with_utilities(self, market_data: pd.DataFrame, ...):
    # This method now automatically validates all generated features
```

### Sophisticated Interactions

```python
@validate_generated_features
@step06_function_validator(validation_level=ValidationLevel.COMPREHENSIVE)
def create_sophisticated_interactions(self, features: pd.DataFrame, ...):
    # This method now automatically validates interaction features
```

## Validation Report Structure

Each validation returns a comprehensive report:

```python
{
    'feature_name': 'my_feature',
    'total_values': 1000,
    'excluded_first_rows': 50,  # NEW: Number of rows excluded from validation
    'valid_values': 950,
    'infinite_values': 10,
    'nan_values': 40,
    'zero_values': 200,
    'constant_feature': False,
    'unique_values': 500,
    'min_value': -5.2,
    'max_value': 12.8,
    'mean_value': 2.1,
    'std_value': 3.4,
    'issues': [
        "Feature 'my_feature' contains 10 infinite values",
        "Feature 'my_feature' contains 40 NaN values",
        "Feature 'my_feature' has 2.1% zero values"  # NEW: triggers at 1%+ threshold
    ],
    'critical_issues': [
        "Feature 'my_feature' contains 10 infinite values",
        "Feature 'my_feature' contains 40 NaN values"
    ],
    'warnings': [
        "Feature 'my_feature' has 2.1% zero values"
    ]
}
```

## Configuration Options

### Validation Levels

- **Basic**: Only critical issues (infinite, NaN)
- **Standard**: Critical + constant features + excessive zeros
- **Comprehensive**: All checks including statistical analysis

### Warning Behavior

- `warn_on_issues=True`: Issues warnings for problems
- `raise_on_critical=True`: Raises exceptions for critical issues
- `validate_input=True`: Validates input DataFrame
- `validate_output=True`: Validates output DataFrame
- `exclude_first_rows=50`: Number of rows to exclude from validation (default: 50)

### Zero Value Threshold

- **New threshold**: 1% (was 50%)
- **Rationale**: More sensitive detection of problematic features
- **Example**: A feature with 1%+ zeros will now trigger a warning

## Best Practices

### 1. Apply to All Feature Generation Functions

```python
# Add this decorator to all feature generation methods
@validate_generated_features
def your_feature_function(data):
    # Your feature generation code
    return enhanced_data
```

### 2. Use Appropriate Validation Level

```python
# For critical features that must be perfect
@strict_feature_validation
def critical_features(data):
    return processed_data

# For exploratory features
@validate_generated_features
def experimental_features(data):
    return processed_data
```

### 3. Handle Validation Results

```python
def process_with_validation(data):
    try:
        result = your_feature_function(data)
        return result
    except FeatureValidationError as e:
        logger.error(f"Feature validation failed: {e}")
        # Handle the error appropriately
        return None
```

## Benefits

1. **Immediate Problem Detection**: Issues are caught as soon as features are generated
2. **Consistent Quality**: All features are validated using the same criteria
3. **Easy Integration**: Simple decorator-based approach requires minimal code changes
4. **Configurable**: Different validation levels for different use cases
5. **Comprehensive Reporting**: Detailed information about feature quality issues
6. **Performance Friendly**: Validation can be disabled in production if needed
7. **Sensitive Detection**: 1% zero threshold catches more problematic features
8. **Warm-up Period Handling**: Excludes first 50 rows to avoid false warnings during initialization
9. **Feature-Specific Reporting**: Every warning clearly identifies which specific feature has issues

## Migration Guide

To add validation to existing feature generation functions:

1. **Import the decorator**:
   ```python
   from src.feature_engineering.math_validation import validate_generated_features
   ```

2. **Add the decorator**:
   ```python
   @validate_generated_features
   def your_existing_function(data):
       # Existing code unchanged
       return result
   ```

3. **Test the integration**:
   - Run your feature generation
   - Check for validation warnings
   - Address any issues found

## Future Enhancements

Potential future improvements to the validation system:

- **Custom Validation Rules**: Allow users to define custom validation criteria
- **Performance Metrics**: Track validation performance impact
- **Batch Validation**: Optimize validation for large datasets
- **Integration with Monitoring**: Connect validation results to monitoring systems
- **Feature Quality Scoring**: Provide overall quality scores for feature sets

## Recent Updates

### Version 2.0 Changes

1. **Zero Threshold Reduced**: From 50% to 1% for more sensitive detection
2. **Warm-up Period Exclusion**: First 50 rows excluded by default to avoid false warnings
3. **Enhanced Configuration**: All functions support `exclude_first_rows` parameter
4. **Backward Compatibility**: Existing code continues to work with new defaults

### Impact of Changes

- **More Sensitive**: Features with just 1%+ zeros now trigger warnings (was 50%+)
- **Fewer False Positives**: Warm-up periods don't cause validation failures
- **Better Production Ready**: Handles real-world data initialization patterns

## Conclusion

The centralized feature validation system provides a robust, easy-to-use solution for ensuring feature quality across the entire feature engineering pipeline. By automatically detecting problematic features, it helps maintain data quality and prevents downstream issues in machine learning models.

The system is designed to be:
- **Non-intrusive**: Minimal impact on existing code
- **Comprehensive**: Covers all common feature quality issues
- **Flexible**: Configurable for different use cases
- **Reliable**: Thoroughly tested and validated
- **Sensitive**: 1% zero threshold catches subtle issues
- **Smart**: Excludes warm-up periods to avoid false warnings
- **Clear**: Every warning identifies the specific feature name

This implementation addresses the original requirement to "immediately generate a warning if we generate features that have infinite values, are constant, generate NaN or are 0" while providing a foundation for more advanced feature quality management. The recent updates make the system even more practical for production use.