# Enhanced Outlier Handling and Schema Validation Guide

## Overview

This guide covers the enhanced outlier handling and data schema validation framework that provides:

- **Sophisticated Outlier Detection**: Multiple detection methods with severity classification
- **Error Raising Instead of Silent Removal**: Critical outliers trigger exceptions to prevent data corruption
- **Data Schema Validation**: Standard schemas for common data types with custom schema creation
- **Root Cause Analysis**: Detailed outlier information and context for debugging
- **Data Integrity Preservation**: Comprehensive validation before file operations

## Table of Contents

1. [Outlier Detection Methods](#outlier-detection-methods)
2. [Severity Classification](#severity-classification)
3. [Schema Validation](#schema-validation)
4. [Error Handling](#error-handling)
5. [Integration](#integration)
6. [Configuration](#configuration)
7. [Examples](#examples)

## Outlier Detection Methods

### Available Methods

The framework provides multiple outlier detection algorithms:

#### 1. Z-Score Method
- **Description**: Detects outliers based on standard deviations from mean
- **Use Case**: Normal distribution data
- **Threshold**: Typically 2-3 standard deviations
- **Pros**: Simple, fast, widely understood
- **Cons**: Sensitive to extreme values affecting mean/std

#### 2. IQR (Interquartile Range) Method
- **Description**: Uses quartiles to define outlier boundaries
- **Use Case**: Non-normal distributions, robust to extreme values
- **Threshold**: Typically 1.5 * IQR
- **Pros**: Robust, handles skewed data well
- **Cons**: May miss outliers in tails

#### 3. Isolation Forest
- **Description**: Machine learning approach using random partitioning
- **Use Case**: High-dimensional data, complex patterns
- **Threshold**: Contamination parameter (default: 0.1)
- **Pros**: Handles high-dimensional data, no distribution assumptions
- **Cons**: Requires scikit-learn, computationally intensive

#### 4. Local Outlier Factor (LOF)
- **Description**: Density-based outlier detection
- **Use Case**: Clustered data, local density variations
- **Threshold**: LOF score threshold
- **Pros**: Handles clustered data well, local perspective
- **Cons**: Requires scikit-learn, parameter tuning needed

#### 5. Mahalanobis Distance
- **Description**: Multivariate outlier detection using covariance
- **Use Case**: Multivariate data with correlations
- **Threshold**: Chi-squared distribution threshold
- **Pros**: Handles correlations, multivariate
- **Cons**: Requires scipy, sensitive to covariance estimation

### Method Selection Guidelines

```python
# For financial time series data
if data_type == "price_data":
    method = "iqr"  # Robust to market crashes
    
elif data_type == "volume_data":
    method = "zscore"  # Usually normal distribution
    
elif data_type == "feature_matrix":
    method = "isolation_forest"  # High-dimensional
    
elif data_type == "correlated_features":
    method = "mahalanobis"  # Handles correlations
```

## Severity Classification

### Severity Levels

Outliers are classified into four severity levels:

#### 1. LOW
- **Description**: Minor deviations from normal
- **Action**: Log warning, continue processing
- **Example**: Slight price spikes in volatile markets

#### 2. MEDIUM
- **Description**: Moderate deviations requiring attention
- **Action**: Log error, continue with caution
- **Example**: Unusual volume spikes

#### 3. HIGH
- **Description**: Significant deviations indicating problems
- **Action**: Log error, optionally raise exception
- **Example**: Price jumps >10x normal range

#### 4. CRITICAL
- **Description**: Extreme deviations indicating data corruption
- **Action**: Log error, raise exception, stop processing
- **Example**: Negative prices, zero volumes

### Severity Determination

```python
# Z-score severity classification
if max_z_score > threshold * 3:
    severity = OutlierSeverity.CRITICAL
elif max_z_score > threshold * 2:
    severity = OutlierSeverity.HIGH
elif max_z_score > threshold * 1.5:
    severity = OutlierSeverity.MEDIUM
else:
    severity = OutlierSeverity.LOW

# IQR severity classification
if max_distance > threshold * 2:
    severity = OutlierSeverity.CRITICAL
elif max_distance > threshold * 1.5:
    severity = OutlierSeverity.HIGH
elif max_distance > threshold * 1.2:
    severity = OutlierSeverity.MEDIUM
else:
    severity = OutlierSeverity.LOW
```

## Schema Validation

### Standard Schemas

The framework provides pre-defined schemas for common data types:

#### 1. Klines Schema
```python
klines_schema = {
    "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
    "data_types": {
        "timestamp": "int64",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64"
    },
    "constraints": {
        "open": {"min": 0, "not_null": True},
        "high": {"min": 0, "not_null": True},
        "low": {"min": 0, "not_null": True},
        "close": {"min": 0, "not_null": True},
        "volume": {"min": 0, "not_null": True}
    }
}
```

#### 2. Features Schema
```python
features_schema = {
    "required_columns": ["timestamp"],
    "optional_columns": [],  # Features can vary
    "data_types": {"timestamp": "int64"},
    "constraints": {"timestamp": {"not_null": True}}
}
```

#### 3. Labels Schema
```python
labels_schema = {
    "required_columns": ["timestamp", "label"],
    "data_types": {
        "timestamp": "int64",
        "label": "object"
    },
    "constraints": {
        "timestamp": {"not_null": True},
        "label": {"not_null": True}
    }
}
```

### Custom Schema Creation

```python
# Create custom schema for trading signals
trading_signals_schema = handler.create_custom_schema(
    name="trading_signals",
    required_columns=["timestamp", "signal", "confidence"],
    optional_columns=["stop_loss", "take_profit"],
    data_types={
        "timestamp": "int64",
        "signal": "object",  # "buy", "sell", "hold"
        "confidence": "float64",
        "stop_loss": "float64",
        "take_profit": "float64"
    },
    constraints={
        "confidence": {"min": 0.0, "max": 1.0, "not_null": True},
        "stop_loss": {"min": 0.0},
        "take_profit": {"min": 0.0}
    }
)
```

### Constraint Types

#### 1. Data Type Constraints
- **not_null**: Column must not contain null values
- **unique**: Column values must be unique
- **min/max**: Numeric value bounds

#### 2. Value Constraints
```python
constraints = {
    "price": {"min": 0, "max": 1000000},
    "volume": {"min": 0},
    "timestamp": {"not_null": True},
    "id": {"unique": True, "not_null": True}
}
```

## Error Handling

### Error Raising Configuration

```python
# Initialize with error raising enabled
handler = EnhancedOutlierHandler(raise_errors=True, log_details=True)

# Override for specific operations
outliers = handler.detect_outliers(
    data, 
    method="zscore", 
    threshold=2.0, 
    raise_errors=False  # Override default
)
```

### Error Handling Patterns

#### 1. Critical Outlier Handling
```python
try:
    outliers = handler.detect_outliers(data, raise_errors=True)
except ValueError as e:
    logger.error(f"Critical outliers detected: {e}")
    # Handle critical outliers
    # - Stop processing
    # - Alert operators
    # - Log detailed information
```

#### 2. Warning-Only Mode
```python
# Detect outliers without raising exceptions
outliers = handler.detect_outliers(data, raise_errors=False)

# Handle based on severity
critical_outliers = [o for o in outliers if o.severity == OutlierSeverity.CRITICAL]
high_outliers = [o for o in outliers if o.severity == OutlierSeverity.HIGH]

if critical_outliers:
    logger.error(f"Critical outliers: {len(critical_outliers)}")
    # Take action without stopping

if high_outliers:
    logger.warning(f"High severity outliers: {len(high_outliers)}")
    # Monitor closely
```

### Logging and Reporting

#### 1. Detailed Logging
```python
# Enable detailed logging
handler = EnhancedOutlierHandler(log_details=True)

# Outliers are automatically logged with context
# - Detection method used
# - Threshold values
# - Statistical context (mean, std, etc.)
# - Severity classification
```

#### 2. Outlier Reports
```python
# Generate comprehensive report
report = handler.get_outlier_report()

# Report includes:
# - Total outlier groups
# - Severity distribution
# - Column distribution
# - Method distribution
# - Recent outliers with timestamps
```

## Integration

### Data Quality Framework Integration

```python
from src.utils.data_quality_framework import data_quality_framework
from src.utils.enhanced_outlier_handler import enhanced_outlier_handler

# Configure quality framework
cleaning_rules = {
    "outlier_handling": "detect_only",
    "outlier_config": {
        "method": "zscore",
        "threshold": 2.0,
        "severity_threshold": "medium",
        "raise_errors": False
    }
}

# Clean data with outlier detection
cleaned_data = data_quality_framework.clean_data(data, cleaning_rules)
```

### Pipeline Integration

```python
# Integrate with training pipeline
@monitor_pipeline_step(
    stage=PipelineStage.DATA_VALIDATION,
    validation_level=PipelineValidationLevel.STRICT
)
def validate_data_quality(data: pd.DataFrame) -> bool:
    """Validate data quality with outlier detection."""
    
    # Validate schema
    schema_result = enhanced_outlier_handler.validate_data_schema(data, "klines")
    if not schema_result["valid"]:
        logger.error(f"Schema validation failed: {schema_result['errors']}")
        return False
    
    # Detect outliers
    outliers = enhanced_outlier_handler.detect_outliers(
        data, 
        method="iqr", 
        threshold=1.5,
        raise_errors=False
    )
    
    # Log outlier information
    if outliers:
        logger.warning(f"Detected {len(outliers)} outlier groups")
        for outlier in outliers:
            logger.warning(f"  {outlier.column}: {len(outlier.indices)} values, severity={outlier.severity.value}")
    
    return True
```

## Configuration

### Handler Configuration

```python
# Basic configuration
handler = EnhancedOutlierHandler(
    raise_errors=True,      # Raise exceptions for critical/high outliers
    log_details=True        # Log detailed outlier information
)

# Advanced configuration
handler = EnhancedOutlierHandler(
    raise_errors=False,     # Warning-only mode
    log_details=True        # Still log details
)
```

### Method-Specific Configuration

```python
# Z-score configuration
zscore_config = {
    "method": "zscore",
    "threshold": 3.0,       # 3 standard deviations
    "columns": ["price", "volume"]  # Specific columns only
}

# IQR configuration
iqr_config = {
    "method": "iqr",
    "threshold": 1.5,       # 1.5 * IQR
    "columns": None          # All numeric columns
}

# Isolation Forest configuration
iso_forest_config = {
    "method": "isolation_forest",
    "threshold": 0.1,       # Contamination parameter
    "columns": None          # All numeric columns
}
```

### Schema Configuration

```python
# Load custom schemas
custom_schemas = {
    "trading_signals": trading_signals_schema,
    "market_data": market_data_schema,
    "model_predictions": predictions_schema
}

# Register with handler
for name, schema in custom_schemas.items():
    handler.standard_schemas[name] = schema
```

## Examples

### Complete Outlier Detection Example

```python
import pandas as pd
from src.utils.enhanced_outlier_handler import enhanced_outlier_handler

# Load data
data = pd.read_csv("trading_data.csv")

# Configure outlier detection
outlier_config = {
    "method": "iqr",
    "threshold": 1.5,
    "columns": ["price", "volume", "returns"],
    "raise_errors": False
}

# Detect outliers
outliers = enhanced_outlier_handler.detect_outliers(
    data, 
    **outlier_config
)

# Analyze results
if outliers:
    print(f"Detected {len(outliers)} outlier groups")
    
    for outlier in outliers:
        print(f"\nColumn: {outlier.column}")
        print(f"  Count: {len(outlier.indices)}")
        print(f"  Severity: {outlier.severity.value}")
        print(f"  Method: {outlier.method}")
        print(f"  Values: {outlier.values[:5]}...")
        
        if outlier.severity in ["high", "critical"]:
            print(f"  ⚠️  Requires attention!")
else:
    print("No outliers detected")

# Generate report
report = enhanced_outlier_handler.get_outlier_report()
print(f"\nOutlier Report: {report['total_outlier_groups']} groups")
```

### Schema Validation Example

```python
# Validate klines data
klines_data = pd.DataFrame({
    "timestamp": [1640995200000, 1640995260000],
    "open": [50000.0, 50100.0],
    "high": [50200.0, 50300.0],
    "low": [49900.0, 50050.0],
    "close": [50100.0, 50250.0],
    "volume": [1000.0, 1200.0]
})

# Validate against klines schema
validation_result = enhanced_outlier_handler.validate_data_schema(klines_data, "klines")

if validation_result["valid"]:
    print("✅ Data passes schema validation")
else:
    print("❌ Schema validation failed:")
    for error in validation_result["errors"]:
        print(f"  - {error}")
    
    for warning in validation_result["warnings"]:
        print(f"  ⚠️  {warning}")
```

### Custom Schema Example

```python
# Create schema for technical indicators
indicators_schema = enhanced_outlier_handler.create_custom_schema(
    name="technical_indicators",
    required_columns=["timestamp", "rsi", "macd", "bollinger_upper", "bollinger_lower"],
    optional_columns=["volume_sma", "price_sma"],
    data_types={
        "timestamp": "int64",
        "rsi": "float64",
        "macd": "float64",
        "bollinger_upper": "float64",
        "bollinger_lower": "float64",
        "volume_sma": "float64",
        "price_sma": "float64"
    },
    constraints={
        "rsi": {"min": 0, "max": 100, "not_null": True},
        "macd": {"not_null": True},
        "bollinger_upper": {"min": 0, "not_null": True},
        "bollinger_lower": {"min": 0, "not_null": True}
    }
)

# Validate technical indicators data
indicators_data = pd.DataFrame({
    "timestamp": [1640995200000],
    "rsi": [65.5],
    "macd": [0.25],
    "bollinger_upper": [52000.0],
    "bollinger_lower": [48000.0],
    "volume_sma": [1100.0],
    "price_sma": [50000.0]
})

validation_result = enhanced_outlier_handler.validate_data_schema(indicators_data, "technical_indicators")
print(f"Indicators validation: {'✅ Passed' if validation_result['valid'] else '❌ Failed'}")
```

## Best Practices

### 1. Method Selection
- **Start with IQR**: Robust and handles most financial data well
- **Use Z-score for normal distributions**: Price returns, volume ratios
- **Try Isolation Forest for complex patterns**: High-dimensional feature matrices
- **Consider Mahalanobis for correlated features**: Technical indicators

### 2. Threshold Tuning
- **Conservative thresholds**: Start with standard values (1.5 for IQR, 2.0 for Z-score)
- **Domain-specific tuning**: Adjust based on asset volatility and market conditions
- **Iterative refinement**: Monitor false positive/negative rates

### 3. Error Handling Strategy
- **Development**: Enable error raising to catch issues early
- **Production**: Consider warning-only mode for non-critical systems
- **Monitoring**: Always log detailed information for analysis

### 4. Schema Design
- **Start simple**: Begin with basic required columns
- **Add constraints gradually**: Implement validation rules incrementally
- **Document assumptions**: Clearly specify data type and constraint expectations

### 5. Performance Considerations
- **Column selection**: Only analyze relevant columns
- **Batch processing**: Process large datasets in chunks
- **Caching**: Cache schema validation results when possible

## Troubleshooting

### Common Issues

#### 1. No Outliers Detected
```python
# Check data types
print(data.dtypes)

# Verify numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
print(f"Numeric columns: {numeric_cols.tolist()}")

# Check for null values
print(f"Null values: {data.isnull().sum()}")
```

#### 2. Too Many Outliers
```python
# Adjust threshold
outliers = handler.detect_outliers(data, threshold=3.0)  # More conservative

# Use different method
outliers = handler.detect_outliers(data, method="iqr")  # More robust

# Filter by severity
high_severity_only = [o for o in outliers if o.severity in ["high", "critical"]]
```

#### 3. Schema Validation Errors
```python
# Check column names
print(f"Expected: {schema.required_columns}")
print(f"Actual: {data.columns.tolist()}")

# Check data types
for col, expected_type in schema.data_types.items():
    if col in data.columns:
        actual_type = str(data[col].dtype)
        print(f"{col}: expected {expected_type}, got {actual_type}")
```

### Performance Optimization

```python
# Profile outlier detection
import time

start_time = time.time()
outliers = handler.detect_outliers(data, method="isolation_forest")
end_time = time.time()

print(f"Detection time: {end_time - start_time:.2f} seconds")

# Use faster methods for large datasets
if len(data) > 100000:
    method = "iqr"  # Faster than isolation forest
else:
    method = "isolation_forest"  # More accurate for smaller datasets
```

## Conclusion

The enhanced outlier handling and schema validation framework provides:

1. **Robust Outlier Detection**: Multiple methods with severity classification
2. **Data Integrity**: Comprehensive validation before processing
3. **Error Prevention**: Critical outlier detection with configurable error raising
4. **Flexibility**: Custom schemas and constraint definitions
5. **Integration**: Seamless integration with existing data quality frameworks

This framework ensures data quality and prevents data corruption while providing detailed insights for debugging and optimization.