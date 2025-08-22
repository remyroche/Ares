# Comprehensive File Format Validation Implementation

## Overview

This document describes the implementation of comprehensive file format validation for the training pipeline steps 1, 1.5, 2, and 4. The validation ensures data quality and consistency across all pipeline stages.

## Validation Requirements

As requested, the validation includes the following checks:

1. **Type of file** - Validates file format (parquet, CSV, JSON)
2. **Type of strings, boolean values, etc.** - Validates data types of columns
3. **Number of columns** - Ensures expected column count
4. **Column names** - Validates required column names are present
5. **Column completeness (no empty values)** - Checks for null/missing values
6. **Index** - Validates DataFrame index integrity

## Implementation Details

### Core Validation Module

**File**: `src/utils/comprehensive_file_validation.py`

The core validation module provides:

- `ComprehensiveFileValidator` class with configurable validation rules
- Step-specific convenience functions
- Detailed validation reporting with severity levels
- Support for multiple file formats and schemas

### Validation Components

#### 1. File Type Validation
- Checks file existence and size
- Validates file extensions (.parquet, .csv, .json)
- Supports multiple file formats with appropriate loaders

#### 2. Data Type Validation
- Validates expected data types for each column
- Checks for mixed data types in object columns
- Supports flexible type matching (e.g., timestamp can be int64 or datetime64[ns])

#### 3. Column Count Validation
- Ensures minimum required number of columns
- Validates against expected schemas for each step

#### 4. Column Names Validation
- Checks for required columns based on schema
- Detects duplicate column names
- Validates column name patterns

#### 5. Column Completeness Validation
- Calculates null ratios for each column
- Identifies completely empty columns
- Configurable thresholds for acceptable null ratios

#### 6. Index Validation
- Checks for unique index values
- Validates monotonic ordering (for time series data)
- Detects null values in index

### Step-Specific Validation

#### Step 1: Data Collection
- **Files**: `klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet`, `aggtrades_{exchange}_{symbol}_consolidated.parquet`
- **Schema**: klines (OHLCV data)
- **Validation**: Ensures proper OHLCV structure with timestamp

#### Step 1.5: Data Converter
- **Files**: `unified_{exchange}_{symbol}_{timeframe}.parquet`, `unified_{exchange}_{symbol}_{timeframe}_config.json`
- **Schema**: Unified data format
- **Validation**: Checks unified data structure and configuration

#### Step 2: Feature Engineering
- **Files**: `features_{exchange}_{symbol}_{split}.parquet` (train/validation/test)
- **Schema**: features (flexible feature columns)
- **Validation**: Ensures feature data quality and structure

#### Step 4: Processing and Labeling
- **Files**: `{exchange}_{symbol}_labeled_{split}.parquet` (train/validation/test)
- **Schema**: Labeled data with features and targets
- **Validation**: Validates labeled data structure and target presence

## Configuration

The validation system uses a configurable approach with default settings:

```python
{
    "file_types": {
        "parquet": {"extensions": [".parquet"], "required": True},
        "csv": {"extensions": [".csv"], "required": False},
        "json": {"extensions": [".json"], "required": False}
    },
    "data_quality": {
        "max_null_ratio": 0.5,
        "min_rows": 1,
        "max_duplicate_ratio": 0.1,
        "check_data_types": True,
        "check_index": True,
        "check_column_names": True,
        "check_completeness": True
    },
    "expected_schemas": {
        "klines": {
            "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
            "expected_types": {
                "timestamp": ["int64", "datetime64[ns]"],
                "open": ["float64"],
                "high": ["float64"],
                "low": ["float64"],
                "close": ["float64"],
                "volume": ["float64"]
            }
        },
        # ... other schemas
    }
}
```

## Integration with Pipeline Steps

### Step 1 Integration
```python
# In step1_data_collection.py
from src.utils.comprehensive_file_validation import validate_step1_file

# After data collection completion
validation_success = await step._run_comprehensive_validation(
    symbol, exchange, timeframe, data_dir, logger
)
```

### Step 1.5 Integration
```python
# In step1_5_data_converter.py
from src.utils.comprehensive_file_validation import validate_step1_5_file

# After data conversion completion
validation_success = await _run_comprehensive_validation(
    symbol, exchange, timeframe, data_dir
)
```

### Step 2 Integration
```python
# In step2_feature_engineering.py
from src.utils.comprehensive_file_validation import validate_step2_file

# After feature engineering completion
validation_success = await _run_comprehensive_validation(
    symbol, exchange, data_dir, logger
)
```

### Step 4 Integration
```python
# In step4_processing_labeling.py
from src.utils.comprehensive_file_validation import validate_step4_file

# After processing and labeling completion
validation_success = await _run_comprehensive_validation(
    symbol, exchange, data_dir, logger
)
```

## Validation Results

Each validation returns a `FileValidationResult` object containing:

- `is_valid`: Boolean indicating overall validation status
- `file_path`: Path to the validated file
- `file_type`: Detected file type
- `issues`: List of validation issues with severity levels
- `summary`: Detailed validation metrics
- `validation_timestamp`: When validation was performed

### Issue Severity Levels

- **CRITICAL**: File cannot be processed (missing file, empty file)
- **ERROR**: Major issues that prevent proper processing (missing required columns, duplicate index)
- **WARNING**: Issues that may affect quality but don't prevent processing (high null ratios, incorrect data types)
- **INFO**: Informational messages about validation results

## Usage Examples

### Basic Validation
```python
from src.utils.comprehensive_file_validation import validate_step1_file

result = validate_step1_file("data_cache/klines_BINANCE_ETHUSDT_1m_consolidated.parquet")
if result.is_valid:
    print("✅ File validation passed")
else:
    for issue in result.issues:
        print(f"⚠️ {issue.severity.value}: {issue.description}")
```

### Custom Validation
```python
from src.utils.comprehensive_file_validation import ComprehensiveFileValidator

validator = ComprehensiveFileValidator()
result = validator.validate_file_format(
    "my_file.parquet",
    expected_schema="features",
    step_name="custom_step"
)
```

## Testing

A comprehensive test suite is provided in `test_comprehensive_validation.py`:

```bash
python test_comprehensive_validation.py
```

The test suite validates:
- Validator initialization and configuration
- Step-specific validation functions
- Schema validation with sample data
- Error handling for invalid files

## Benefits

1. **Data Quality Assurance**: Ensures consistent data quality across all pipeline steps
2. **Early Error Detection**: Catches issues before they propagate through the pipeline
3. **Comprehensive Reporting**: Detailed validation reports with actionable information
4. **Configurable**: Flexible configuration for different validation requirements
5. **Non-blocking**: Validation issues are logged but don't necessarily stop pipeline execution
6. **Step-specific**: Tailored validation for each pipeline step's requirements

## Future Enhancements

1. **Auto-fix Capabilities**: Automatic correction of common validation issues
2. **Performance Optimization**: Parallel validation for multiple files
3. **Custom Schemas**: User-defined validation schemas
4. **Integration with Monitoring**: Real-time validation monitoring and alerting
5. **Validation History**: Tracking validation results over time

## Conclusion

The comprehensive file format validation system provides robust data quality assurance for the training pipeline. It ensures that each step produces properly formatted, complete, and consistent data for downstream processing, reducing the risk of errors and improving overall pipeline reliability.