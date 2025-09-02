# Data Quality and Formatting Framework Guide

## Overview

This guide provides comprehensive documentation for the Data Quality and Formatting Framework, which ensures data consistency, quality, and standardization across all pipeline steps.

## Framework Components

### 1. Data Quality Framework (`src/utils/data_quality_framework.py`)

The Data Quality Framework provides comprehensive data validation, cleaning, and quality assessment capabilities.

#### Key Features

- **Data Validation**: Schema enforcement, data type validation, range checking
- **Quality Scoring**: Completeness, consistency, accuracy, and timeliness metrics
- **Data Cleaning**: Automatic outlier removal, missing value handling, duplicate removal
- **Data Profiling**: Comprehensive data analysis and statistics
- **Quality Policies**: Configurable validation rules and thresholds

#### Core Classes

##### DataQualityLevel Enum
```python
class DataQualityLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
```

##### DataFormat Enum
```python
class DataFormat(Enum):
    KLINES = "klines"
    FEATURES = "features"
    LABELS = "labels"
    PREDICTIONS = "predictions"
    METADATA = "metadata"
    CONFIG = "config"
```

##### DataQualityFramework Class
The main framework class that orchestrates all data quality operations.

#### Usage Examples

##### Basic Data Validation
```python
from src.utils.data_quality_framework import data_quality_framework

# Validate data
validation_results = data_quality_framework.validate_data(data)

# Check if validation passed
if validation_results["overall_passed"]:
    print("Data validation successful")
else:
    print(f"Validation failed: {validation_results['failed_rules']} rules failed")
```

##### Data Quality Scoring
```python
# Calculate overall quality score
quality_score = data_quality_framework.calculate_quality_score(data)
print(f"Data quality score: {quality_score:.2%}")

# Get detailed quality report
quality_report = data_quality_framework.get_quality_report(data)
print(f"Completeness: {quality_report['quality_metrics']['completeness']:.2%}")
```

##### Data Cleaning
```python
# Clean data with default rules
cleaned_data = data_quality_framework.clean_data(data)

# Clean with custom rules
custom_rules = {
    "remove_duplicates": True,
    "handle_missing_values": True,
    "remove_outliers": True,
    "outlier_config": {
        "method": "iqr",
        "threshold": 2.0,
        "severity_threshold": "high"
    }
}
cleaned_data = data_quality_framework.clean_data(data, custom_rules)
```

##### Data Profiling
```python
# Generate comprehensive data profile
profile = data_quality_framework.profile_data(data)

print(f"Data shape: {profile['data_shape']}")
print(f"Memory usage: {profile['memory_usage']} bytes")
print(f"Missing values: {profile['summary']['missing_values']}")
```

### 2. Data Formatting Framework (`src/utils/data_formatting_framework.py`)

The Data Formatting Framework ensures consistent data formats across all pipeline steps.

#### Key Features

- **Standard Formats**: Predefined formats for different data types
- **Column Standardization**: Automatic naming convention enforcement
- **Data Type Standardization**: Consistent numeric and timestamp formats
- **Format Validation**: Ensures data conforms to specified formats
- **Cross-step Consistency**: Maintains format consistency across pipeline steps

#### Standard Data Formats

##### KLINES Format
```python
{
    "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
    "data_types": {
        "timestamp": "int64",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64"
    }
}
```

##### FEATURES Format
```python
{
    "required_columns": ["timestamp"],
    "data_types": {
        "timestamp": "int64"
    }
}
```

##### LABELS Format
```python
{
    "required_columns": ["timestamp", "label"],
    "data_types": {
        "timestamp": "int64",
        "label": "int64",
        "label_probability": "float64"
    }
}
```

#### Usage Examples

##### Format Standardization
```python
from src.utils.data_formatting_framework import data_formatting_framework, DataFormat

# Standardize data to klines format
formatted_klines = data_formatting_framework.standardize_format(data, DataFormat.KLINES)

# Standardize to features format
formatted_features = data_formatting_framework.standardize_format(data, DataFormat.FEATURES)
```

##### Format Validation
```python
# Validate data format
validation_results = data_formatting_framework.validate_data_format(data, DataFormat.KLINES)

if validation_results["valid"]:
    print("Data format is valid")
else:
    print(f"Format issues: {validation_results['issues']}")
```

##### Timestamp Normalization
```python
# Normalize timestamps to unix seconds
normalized_data = data_formatting_framework.normalize_timestamps(
    data, "timestamp", "unix_seconds"
)

# Normalize to ISO string format
iso_data = data_formatting_framework.normalize_timestamps(
    data, "timestamp", "iso_string"
)
```

##### Missing Value Handling
```python
# Handle missing values with forward fill
filled_data = data_formatting_framework.handle_missing_values(data, "forward_fill")

# Handle with median fill
median_filled = data_formatting_framework.handle_missing_values(data, "median")

# Intelligent gap filling
intelligent_filled = data_formatting_framework.handle_missing_values(
    data, "intelligent", symbol="BTCUSDT", exchange="binance"
)
```

### 3. Testing Framework (`test_data_quality_and_formatting.py`)

Comprehensive testing suite that validates all framework functionality.

#### Test Categories

1. **Data Validation Tests**: Schema validation, data type validation
2. **Data Formatting Tests**: Format standardization, column naming
3. **Data Cleaning Tests**: Outlier removal, missing value handling
4. **Data Profiling Tests**: Profile generation, statistics calculation
5. **Quality Scoring Tests**: Score calculation, metric validation
6. **Format Validation Tests**: Format compliance checking
7. **Timestamp Tests**: Normalization and consistency
8. **Missing Value Tests**: Various handling strategies
9. **Data Type Tests**: Type standardization
10. **Cross-step Tests**: Format consistency across steps

#### Running Tests

```bash
# Run all tests
python test_data_quality_and_formatting.py

# Run specific test category
python -m pytest test_data_quality_and_formatting.py::DataQualityAndFormattingTester::test_data_validation
```

## Configuration

### Quality Policies

```python
quality_policies = {
    "strict_validation": True,
    "auto_clean": True,
    "profiling_enabled": True,
    "max_issues_critical": 0,
    "max_issues_high": 5,
    "max_issues_medium": 20,
    "max_issues_low": 100
}
```

### Formatting Policies

```python
formatting_policies = {
    "column_naming_convention": "snake_case",
    "timestamp_format": "unix_seconds",
    "numeric_precision": 8,
    "auto_rename_columns": True,
    "strict_formatting": True,
    "preserve_original": True
}
```

## Integration with Training Pipeline

### Step-Level Integration

Each pipeline step automatically uses the data quality and formatting frameworks:

```python
# Data is automatically validated and formatted
result = await self.execute_step1_data_collection(training_input)

# Quality metrics are automatically collected
quality_metrics = await self._get_step_quality_metrics("step1_data_collection", result)
```

### Pipeline Reporting

Quality and formatting information is automatically included in step reports:

```python
step_report = {
    "step_name": "step1_data_collection",
    "quality_metrics": quality_metrics,
    "format_validation": format_validation_results,
    "data_profile": data_profile
}
```

## Best Practices

### 1. Data Quality

- Always validate data before processing
- Use appropriate quality thresholds for your use case
- Monitor quality metrics over time
- Implement automated quality alerts

### 2. Data Formatting

- Standardize formats early in the pipeline
- Maintain format consistency across steps
- Use predefined format specifications
- Validate formats before and after transformations

### 3. Testing

- Test with realistic data samples
- Validate edge cases and error conditions
- Monitor performance impact
- Regular regression testing

### 4. Monitoring

- Track quality metrics over time
- Monitor format compliance
- Alert on quality degradation
- Regular framework health checks

## Troubleshooting

### Common Issues

1. **Validation Failures**: Check data schema and types
2. **Format Errors**: Verify column names and data types
3. **Performance Issues**: Adjust quality policies and thresholds
4. **Memory Issues**: Optimize data profiling settings

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.getLogger("DataQualityFramework").setLevel(logging.DEBUG)
logging.getLogger("DataFormatting").setLevel(logging.DEBUG)
```

## Performance Considerations

### Optimization Strategies

1. **Lazy Validation**: Only validate when necessary
2. **Caching**: Cache validation results for repeated data
3. **Batch Processing**: Process data in chunks for large datasets
4. **Parallel Processing**: Use parallel validation for independent data

### Memory Management

1. **Streaming**: Process data in streams for large files
2. **Chunking**: Process data in manageable chunks
3. **Cleanup**: Explicitly clean up large data objects
4. **Monitoring**: Monitor memory usage during processing

## Future Enhancements

### Planned Features

1. **Machine Learning Quality Assessment**: AI-powered quality scoring
2. **Real-time Quality Monitoring**: Live quality metrics
3. **Advanced Outlier Detection**: Statistical and ML-based methods
4. **Format Evolution**: Automatic format adaptation
5. **Quality Prediction**: Predictive quality assessment

### Extension Points

The framework is designed to be extensible:

1. **Custom Validation Rules**: Add domain-specific validation
2. **Custom Formats**: Define new data formats
3. **Custom Quality Metrics**: Implement specialized metrics
4. **Integration Hooks**: Connect with external quality tools

## Conclusion

The Data Quality and Formatting Framework provides a robust foundation for ensuring data quality and consistency across the entire training pipeline. By following the guidelines in this document, you can effectively use these tools to maintain high data standards and improve pipeline reliability.

For additional support or feature requests, please refer to the project documentation or contact the development team.