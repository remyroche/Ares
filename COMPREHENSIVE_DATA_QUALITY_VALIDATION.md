# Comprehensive Data Quality Validation System

## Overview

This document describes the comprehensive data quality validation system implemented for Step1, Step1_5, and Step2 of the training pipeline. The system ensures data quality and file structure validation at each step, with special attention to NaN, infinite, and constant values in Step2.

## Key Features

### 🔍 **Comprehensive Validation**
- **File Structure Validation**: Checks file existence, size, and basic properties
- **Data Quality Validation**: Validates DataFrame quality including NaN, infinite, and constant values
- **Feature-Specific Validation**: Special validation for Step2 features with stricter thresholds
- **Detailed Logging**: Comprehensive logging of all quality issues found

### 📊 **Quality Checks**
- **NaN Values**: Detects and reports features with excessive NaN values (0.1% threshold)
- **Infinite Values**: Identifies features with infinite or negative infinite values (1 value threshold)
- **Constant Features**: Flags features with insufficient variation (2+ unique values, except boolean)
- **High Correlations**: Detects highly correlated feature pairs
- **File Integrity**: Validates file structure and basic properties
- **Data Structure**: Validates columns, format, index, and data types at every step

### 🎯 **Step-Specific Validation**

#### **Step1: Data Collection**
- Validates consolidated parquet files (klines, aggtrades)
- Checks file existence and basic data quality
- Validates data completeness and format
- **Data Structure Validation**: Columns, format, index, data types
- **Thresholds**: 
  - Max NaN ratio: 0.1%
  - Max infinite count: 1 value
  - Min unique values: 2 (except boolean)

#### **Step1_5: Data Converter**
- Validates unified data directory structure
- Checks partitioned parquet files
- Validates data quality in unified format
- **Data Structure Validation**: Columns, format, index, data types
- **Thresholds**: Same as Step1

#### **Step2: Feature Engineering**
- **Stricter validation** for feature quality
- Special attention to NaN, infinite, and constant values
- Validates feature correlations
- Ensures minimum feature count requirements
- **Data Structure Validation**: Columns, format, index, data types
- **Thresholds**:
  - Max NaN ratio: 0.1% (stricter)
  - Max infinite count: 1 value
  - Min unique values: 2 (except boolean)
  - Max correlation threshold: 95%

## Implementation

### Data Structure Validation

The system now includes comprehensive data structure validation at every step:

#### **Required Columns Validation**
- **Klines Data**: Validates presence of `timestamp`, `open`, `high`, `low`, `close`, `volume`
- **Aggtrades Data**: Validates presence of `timestamp`, `price`, `quantity`
- **Feature Data**: Validates presence of `timestamp` and minimum feature count

#### **Data Type Validation**
- **Timestamp Columns**: Ensures proper datetime format
- **Numeric Columns**: Validates OHLCV and trade data are numeric
- **Feature Columns**: Validates appropriate data types for features

#### **Data Range Validation**
- **Price Data**: Ensures positive values for OHLCV
- **Volume Data**: Ensures non-negative values
- **Feature Data**: Validates reasonable value ranges

#### **Index and Format Validation**
- **Datetime Index**: Validates proper timestamp formatting
- **Duplicate Columns**: Detects and reports duplicate column names
- **Empty Columns**: Identifies completely empty columns

### Core Components

#### 1. **ComprehensiveDataQualityValidator** (`src/utils/comprehensive_data_quality_validator.py`)
```python
class ComprehensiveDataQualityValidator:
    def validate_step1_data_quality(symbol, exchange, data_dir)
    def validate_step1_5_data_quality(symbol, exchange, data_dir)
    def validate_step2_data_quality(symbol, exchange, data_dir)
```

#### 2. **Data Quality Decorators** (`src/utils/data_quality_decorators.py`)
```python
@validate_step1_quality
@validate_step1_5_quality
@validate_step2_quality
def your_function():
    pass
```

#### 3. **Integration in Pipeline Steps**
- **Step1**: Integrated in `run_step()` function
- **Step1_5**: Integrated in `execute()` method
- **Step2**: Integrated in `run_step()` function

### Usage Examples

#### Basic Validation
```python
from src.utils.comprehensive_data_quality_validator import validate_step2_quality

# Validate Step2 features
result = validate_step2_quality("ETHUSDT", "BINANCE", "data/training")
print(f"Validation passed: {result['validation_passed']}")
```

#### Comprehensive Validation
```python
from src.utils.comprehensive_data_quality_validator import ComprehensiveDataQualityValidator

validator = ComprehensiveDataQualityValidator({
    "max_nan_ratio": 0.1,
    "max_infinite_ratio": 0.05,
    "min_unique_values": 3,
    "min_feature_count": 40
})

# Validate all steps
step1_result = validator.validate_step1_data_quality("ETHUSDT", "BINANCE")
step1_5_result = validator.validate_step1_5_data_quality("ETHUSDT", "BINANCE")
step2_result = validator.validate_step2_data_quality("ETHUSDT", "BINANCE")

# Save comprehensive report
validator.save_validation_report("validation_report.json")
```

#### Feature Quality Logging
```python
from src.utils.data_quality_decorators import log_feature_quality_issues

# Log detailed feature quality issues
log_feature_quality_issues(feature_df, "Training Features")
```

## Validation Results

### Step1 Validation
```json
{
  "step": "step1_data_collection",
  "validation_passed": true,
  "issues": [],
  "file_checks": {
    "klines_BINANCE_ETHUSDT_1m_consolidated.parquet": {
      "exists": true,
      "size_mb": 45.2,
      "issues": []
    }
  },
  "data_quality": {
    "klines_BINANCE_ETHUSDT_1m_consolidated.parquet": {
      "passed": true,
      "shape": [43200, 6],
      "issues": []
    }
  }
}
```

### Step2 Validation (with issues)
```json
{
  "step": "step2_feature_engineering",
  "validation_passed": false,
  "issues": [
    "Features with NaN values: ['rsi', 'macd']",
    "Features with infinite values: ['bollinger_upper']",
    "Constant features found: ['constant_rsi']"
  ],
  "problematic_features": {
    "nan_features": ["rsi", "macd"],
    "infinite_features": ["bollinger_upper"],
    "constant_features": ["constant_rsi"],
    "high_correlation_pairs": [["rsi", "rsi_duplicate"]]
  }
}
```

## Quality Thresholds

### Configurable Parameters
```python
config = {
    "max_nan_ratio": 0.001,         # Maximum allowed ratio of NaN values (0.1%)
    "max_infinite_count": 1,        # Maximum allowed count of infinite values
    "min_unique_values": 2,         # Minimum unique values for non-constant features
    "max_constant_ratio": 0.95,     # Maximum ratio for constant features
    "min_feature_count": 40,        # Minimum required features for Step2
    "max_correlation_threshold": 0.95  # Maximum correlation threshold
}
```

### Step-Specific Thresholds

| Step | Max NaN Ratio | Max Infinite Count | Min Unique Values | Check Correlations | Data Structure Validation |
|------|---------------|-------------------|-------------------|-------------------|---------------------------|
| Step1 | 0.1% | 1 | 2 (except boolean) | No | Yes |
| Step1_5 | 0.1% | 1 | 2 (except boolean) | No | Yes |
| Step2 | 0.1% | 1 | 2 (except boolean) | Yes | Yes |

## Error Handling

### Graceful Degradation
- Validation failures don't stop the pipeline
- Issues are logged as warnings
- Pipeline continues with quality issues noted
- Comprehensive reports are generated

### Fallback Mechanisms
- Legacy validation systems are preserved
- Multiple validation layers for robustness
- Detailed error reporting and logging

## Testing

### Test Script
Run the comprehensive test to validate the system:
```bash
python test_comprehensive_data_quality_validation.py
```

### Test Coverage
- ✅ Step1 data quality validation
- ✅ Step1_5 data quality validation  
- ✅ Step2 feature quality validation
- ✅ NaN, infinite, constant value detection
- ✅ High correlation detection
- ✅ File structure validation
- ✅ Comprehensive reporting

## Integration Points

### Pipeline Integration
1. **Step1**: Validation runs after data collection
2. **Step1_5**: Validation runs after data conversion
3. **Step2**: Validation runs after feature engineering

### Logging Integration
- All validation results are logged to the system logger
- Detailed quality reports are generated
- Issues are categorized and prioritized

### Report Generation
- JSON reports for programmatic access
- Human-readable logs for debugging
- Summary statistics for monitoring

## Best Practices

### For Step1
- Ensure data completeness before validation
- Check for missing files and directories
- Validate basic data structure

### For Step1_5
- Verify unified data directory structure
- Check partitioned file integrity
- Validate data format consistency

### For Step2
- **Critical**: Check for NaN, infinite, and constant values
- Validate feature correlations
- Ensure minimum feature count
- Monitor feature quality metrics

## Monitoring and Alerts

### Quality Metrics
- Feature count and quality statistics
- Data completeness metrics
- Correlation analysis results
- Issue frequency and severity

### Alert Thresholds
- High NaN ratio (>10% for features)
- Infinite values detected
- Too many constant features
- Insufficient feature count

## Future Enhancements

### Planned Features
- Real-time quality monitoring
- Automated issue resolution
- Quality trend analysis
- Performance optimization

### Integration Opportunities
- MLflow integration for experiment tracking
- Grafana dashboards for monitoring
- Automated alerting systems
- Quality-based model selection

## Conclusion

The comprehensive data quality validation system ensures robust data quality throughout the training pipeline. With special attention to NaN, infinite, and constant values in Step2, the system provides:

- **Reliability**: Consistent quality checks across all steps
- **Transparency**: Detailed logging and reporting
- **Flexibility**: Configurable thresholds and validation rules
- **Robustness**: Graceful error handling and fallback mechanisms

This system helps maintain high-quality data for machine learning model training and ensures reliable feature engineering processes.