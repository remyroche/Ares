# Data Validation Decorators Implementation for ML Prediction Methods

## Overview

This document summarizes the implementation of data validation decorators for ML prediction methods in the `src/tactician/` directory. The decorators ensure data quality, proper formatting, and robust error handling for critical ML prediction operations.

## Implemented Decorators

### 1. SR Breakout Predictor (`src/tactician/sr_breakout_predictor.py`)

**Method Enhanced:** `predict_sr_breakouts()`

```python
@validate_data_quality(
    required_columns=["open", "high", "low", "close", "volume"],
    min_rows=50,
    max_null_ratio=0.1,
    check_duplicates=True,
    check_timestamps=True,
    context="SR breakout prediction input validation"
)
```

**Validation Criteria:**
- **Required Columns:** OHLCV data columns
- **Minimum Rows:** 50 data points for reliable pattern detection
- **Null Ratio:** Maximum 10% null values allowed
- **Duplicates:** Check for duplicate timestamps
- **Timestamps:** Validate timestamp format and sequence

### 2. ML Tactics Manager (`src/tactician/ml_tactics_manager.py`)

**Method Enhanced:** `_validate_tactics_input()`

```python
@validate_data_quality(
    required_columns=None,  # Dict input validation
    min_rows=1,
    max_null_ratio=0.0,
    check_duplicates=False,
    check_timestamps=False,
    context="ML tactics input validation"
)
```

**Validation Criteria:**
- **Input Type:** Dictionary validation (not DataFrame)
- **Null Tolerance:** Zero tolerance for missing required fields
- **Context:** ML tactics input parameter validation

### 3. ML Target Validator (`src/tactician/ml_target_validator.py`)

**Methods Enhanced:**
- `validate_target()`
- `validate_prediction()`

```python
@validate_data_quality(
    required_columns=None,  # Dict input validation
    min_rows=1,
    max_null_ratio=0.0,
    check_duplicates=False,
    check_timestamps=False,
    context="ML target/prediction validation"
)
```

**Validation Criteria:**
- **Input Type:** Dictionary validation for target/prediction data
- **Null Tolerance:** Zero tolerance for missing critical fields
- **Context:** ML target and prediction validation

### 4. ML Target Updater (`src/tactician/ml_target_updater.py`)

**Methods Enhanced:**
- `_generate_target_prediction()`
- `_get_market_data()`

```python
# For target prediction generation
@validate_data_quality(
    required_columns=["open", "high", "low", "close", "volume"],
    min_rows=20,
    max_null_ratio=0.1,
    check_duplicates=True,
    check_timestamps=True,
    context="ML target prediction generation"
)

# For market data retrieval
@validate_data_quality(
    required_columns=["timestamp", "open", "high", "low", "close", "volume"],
    min_rows=1,
    max_null_ratio=0.0,
    check_duplicates=False,
    check_timestamps=True,
    context="ML target updater market data retrieval"
)
```

**Validation Criteria:**
- **Target Prediction:** 20+ data points, 10% null tolerance
- **Market Data:** Complete OHLCV data with timestamps
- **Data Quality:** Duplicate detection and timestamp validation

### 5. Position Sizer (`src/tactician/position_sizer.py`)

**Method Enhanced:** `calculate_position_size()`

```python
@validate_data_quality(
    required_columns=None,  # Dict input validation
    min_rows=1,
    max_null_ratio=0.0,
    check_duplicates=False,
    check_timestamps=False,
    context="position sizing calculation input validation"
)
```

**Validation Criteria:**
- **Input Type:** Dictionary validation for ML predictions
- **Null Tolerance:** Zero tolerance for missing confidence scores
- **Context:** Position sizing calculation input validation

## Validation Features

### 1. **Data Quality Checks**
- **Required Columns:** Ensures all necessary data columns are present
- **Minimum Rows:** Validates sufficient data for reliable predictions
- **Null Ratio:** Prevents processing of incomplete datasets
- **Duplicate Detection:** Identifies and handles duplicate entries
- **Timestamp Validation:** Ensures proper time series data

### 2. **Context-Specific Validation**
- **ML Prediction Methods:** Strict validation for critical prediction inputs
- **Market Data Processing:** OHLCV data validation with appropriate thresholds
- **Dictionary Inputs:** Specialized validation for configuration and parameter dictionaries

### 3. **Error Handling Integration**
- **Graceful Degradation:** Returns appropriate default values on validation failure
- **Detailed Logging:** Comprehensive error messages with context
- **Exception Mapping:** Maps validation errors to specific exception types

## Benefits

### 1. **Data Quality Assurance**
- Prevents processing of invalid or incomplete data
- Ensures reliable ML predictions
- Reduces runtime errors and exceptions

### 2. **Debugging and Monitoring**
- Clear validation failure messages
- Context-specific error reporting
- Performance impact tracking

### 3. **System Reliability**
- Robust error handling for critical ML operations
- Consistent validation across all prediction methods
- Improved system stability

## Implementation Details

### 1. **Decorator Order**
Data validation decorators are applied **before** error handling decorators to ensure:
- Input validation occurs first
- Error handling catches validation failures
- Proper error context is maintained

### 2. **Performance Considerations**
- **Lightweight Validation:** Minimal performance impact
- **Async Support:** Compatible with async methods
- **Conditional Validation:** Only validates when data is present

### 3. **Extensibility**
- **Configurable Parameters:** Easy to adjust validation criteria
- **Context Awareness:** Different validation rules for different contexts
- **Modular Design:** Easy to add new validation rules

## Usage Examples

### DataFrame Validation
```python
@validate_data_quality(
    required_columns=["open", "high", "low", "close", "volume"],
    min_rows=50,
    max_null_ratio=0.1,
    check_duplicates=True,
    check_timestamps=True,
    context="market data validation"
)
async def process_market_data(self, data: pd.DataFrame) -> dict:
    # Method implementation
```

### Dictionary Validation
```python
@validate_data_quality(
    required_columns=None,
    min_rows=1,
    max_null_ratio=0.0,
    check_duplicates=False,
    check_timestamps=False,
    context="configuration validation"
)
async def validate_config(self, config: dict) -> bool:
    # Method implementation
```

## Future Enhancements

### 1. **Advanced Validation Rules**
- **Statistical Validation:** Outlier detection and statistical tests
- **Domain-Specific Rules:** Trading-specific validation criteria
- **Dynamic Thresholds:** Adaptive validation based on market conditions

### 2. **Performance Optimization**
- **Caching:** Cache validation results for repeated data
- **Parallel Validation:** Concurrent validation for large datasets
- **Lazy Validation:** Validate only when data is accessed

### 3. **Monitoring and Analytics**
- **Validation Metrics:** Track validation success/failure rates
- **Performance Monitoring:** Measure validation overhead
- **Alerting:** Notify on validation failures

## Conclusion

The implementation of data validation decorators for ML prediction methods provides:

1. **Enhanced Data Quality:** Ensures reliable inputs for ML predictions
2. **Improved Error Handling:** Graceful handling of validation failures
3. **Better Debugging:** Clear error messages and context
4. **System Reliability:** Robust validation across all prediction methods

These decorators significantly improve the reliability and maintainability of the ML prediction pipeline in the tactician system.