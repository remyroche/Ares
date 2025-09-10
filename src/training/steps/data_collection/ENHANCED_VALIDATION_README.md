# Enhanced Data Validation Framework

## Overview

The Enhanced Data Validation Framework provides comprehensive validation during data collection for klines, aggtrades, and futures data. It ensures data quality, consistency, and compliance with downstream processing requirements.

## Features

### ✅ **Real-time Schema Enforcement**
- Validates data structure during API collection
- Enforces field types, constraints, and requirements
- Maps exchange-specific field names to standardized formats

### ✅ **Comprehensive Data Quality Checks**
- **NaN Detection**: Identifies and handles missing values
- **Infinite Value Detection**: Catches infinite values in numeric fields
- **Zero Value Validation**: Configurable zero value handling
- **Negative Value Validation**: Configurable negative value handling
- **Type Conversion**: Automatic type conversion with validation

### ✅ **Time Gap Detection**
- **Klines**: 1.1 minute maximum gap (66 seconds + 5 second tolerance)
- **Aggtrades**: 1 second maximum gap (1 second + 0.1 second tolerance)
- **Futures**: 9 hour maximum gap (32400 seconds + 5 minute tolerance)

### ✅ **Exchange Field Mapping**
- **Binance**: Maps `open_time` → `timestamp`, `p` → `price`, etc.
- **Coinbase**: Maps `price_open` → `open`, `size` → `quantity`, etc.
- **Kraken**: Maps `time` → `timestamp`, `vol` → `volume`, etc.

### ✅ **Downstream Compatibility**
- Ensures data format matches Step02-06 expectations
- Validates against existing pipeline schemas
- Maintains backward compatibility

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Enhanced Data Validation Framework           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Klines        │  │   Aggtrades     │  │   Futures    │ │
│  │   Validator     │  │   Validator     │  │   Validator  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Schema        │  │   Field         │  │   Time Gap   │ │
│  │   Enforcement   │  │   Mapping       │  │   Detection  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Data Quality  │  │   Type          │  │   Validation │ │
│  │   Checks        │  │   Conversion    │  │   Reporting  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Data Schemas

### Klines Schema
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
    },
    "constraints": {
        "open": {"min": 0.0, "allow_zero": False},
        "high": {"min": 0.0, "allow_zero": False},
        "low": {"min": 0.0, "allow_zero": False},
        "close": {"min": 0.0, "allow_zero": False},
        "volume": {"min": 0.0, "allow_zero": True}
    },
    "time_gap_config": {
        "max_gap_seconds": 66.0,
        "tolerance_seconds": 5.0
    }
}
```

### Aggtrades Schema
```python
{
    "required_columns": ["timestamp", "price", "quantity"],
    "data_types": {
        "timestamp": "int64",
        "price": "float64",
        "quantity": "float64"
    },
    "constraints": {
        "price": {"min": 0.0, "allow_zero": False},
        "quantity": {"min": 0.0, "allow_zero": False}
    },
    "time_gap_config": {
        "max_gap_seconds": 1.0,
        "tolerance_seconds": 0.1
    }
}
```

### Futures Schema
```python
{
    "required_columns": ["timestamp", "funding_rate"],
    "data_types": {
        "timestamp": "int64",
        "funding_rate": "float64"
    },
    "constraints": {
        "funding_rate": {"allow_zero": True}
    },
    "time_gap_config": {
        "max_gap_seconds": 32400.0,
        "tolerance_seconds": 300.0
    }
}
```

## Usage Examples

### Basic Usage

```python
from src.training.steps.data_collection.enhanced_data_validation_framework import (
    DataType, get_validator
)

# Get validator for klines data
klines_validator = get_validator(DataType.KLINES)

# Validate a batch of data
raw_klines_data = [
    {
        "open_time": 1640995200000,  # Binance format
        "open": "3000.0",
        "high": "3100.0",
        "low": "2900.0",
        "close": "3050.0",
        "volume": "1000.0"
    }
]

validated_data = klines_validator.validate_batch(raw_klines_data)
print(f"Validated {len(validated_data)} rows")
```

### Enhanced Data Collection

```python
from src.training.steps.data_collection.enhanced_data_collector import (
    EnhancedDataCollectionManager
)

# Initialize collection manager
manager = EnhancedDataCollectionManager("BINANCE", "ETHUSDT", "1m")

# Collect klines data
raw_klines_batch = [...]  # Raw data from API
await manager.collect_klines_batch(raw_klines_batch)

# Collect aggtrades data
raw_aggtrades_batch = [...]  # Raw data from API
await manager.collect_aggtrades_batch(raw_aggtrades_batch)

# Finalize all collections
summary = await manager.finalize_all_collections()
print(f"Overall success rate: {summary['overall_success_rate']:.1f}%")
```

### Complete Pipeline Integration

```python
from src.training.steps.data_collection.enhanced_data_collection_integration import (
    run_enhanced_data_collection_pipeline
)

# Run complete enhanced pipeline
summary = await run_enhanced_data_collection_pipeline(
    symbol="ETHUSDT",
    exchange="BINANCE",
    timeframe="1m",
    force_rerun=False
)

print(f"Pipeline success: {summary['overall_success']}")
```

## Integration with Existing Pipeline

### Step01 Integration

```python
from src.training.steps.data_collection.enhanced_step01_data_collection import (
    run_enhanced_step01_data_collection
)

# Run enhanced Step01
success = await run_enhanced_step01_data_collection(
    symbol="ETHUSDT",
    exchange="BINANCE",
    timeframe="1m",
    force_rerun=True
)
```

### Step01_5 Integration

```python
from src.training.steps.data_collection.enhanced_step01_5_data_converter import (
    run_enhanced_step01_5_data_converter
)

# Run enhanced Step01_5
success = await run_enhanced_step01_5_data_converter(
    symbol="ETHUSDT",
    exchange="BINANCE",
    timeframe="1m",
    force_rerun=True
)
```

## Validation Results

### Validation Summary
```python
{
    "data_type": "klines",
    "total_rows_processed": 1000,
    "valid_rows": 995,
    "invalid_rows": 5,
    "success_rate": 99.5,
    "time_gaps_detected": 0,
    "total_errors": 5,
    "error_breakdown": {
        "critical": 0,
        "high": 2,
        "medium": 2,
        "low": 1
    }
}
```

### Error Types
- **Critical**: Stop processing (missing required fields, type conversion failures)
- **High**: Log error, continue with warning (NaN values, infinite values, negative prices)
- **Medium**: Log warning, continue (zero values, time gaps)
- **Low**: Log info, continue (minor format issues)

## Configuration

### Custom Field Mapping
```python
from src.training.steps.data_collection.enhanced_data_validation_framework import (
    FieldDefinition, DataSchema, DataType
)

# Create custom field definition
custom_field = FieldDefinition(
    name="timestamp",
    dtype="int64",
    source_mapping={
        "binance": "open_time",
        "coinbase": "timestamp",
        "kraken": "time",
        "custom_exchange": "ts"
    }
)
```

### Custom Validation Rules
```python
def custom_price_validator(value):
    """Custom validator for price fields."""
    return 0 < value < 1000000  # Price between 0 and 1M

price_field = FieldDefinition(
    name="price",
    dtype="float64",
    min_value=0.0,
    allow_zero=False,
    custom_validator=custom_price_validator
)
```

## Performance Considerations

### Batch Processing
- Process data in batches for memory efficiency
- Recommended batch size: 1000-10000 rows
- Use async processing for I/O operations

### Memory Optimization
- Validators use minimal memory overhead
- Data is processed row-by-row to avoid memory spikes
- Automatic garbage collection after validation

### Error Handling
- Non-blocking error handling for non-critical issues
- Configurable error severity levels
- Detailed error reporting with context

## Troubleshooting

### Common Issues

1. **Type Conversion Errors**
   ```python
   # Ensure string values can be converted to numeric types
   raw_data = [{"price": "3000.0"}]  # String format
   # Validator will convert to float64 automatically
   ```

2. **Missing Required Fields**
   ```python
   # Add default values for optional fields
   field_def = FieldDefinition(
       name="optional_field",
       dtype="float64",
       required=False,
       default_value=0.0
   )
   ```

3. **Time Gap Detection**
   ```python
   # Adjust tolerance for time gaps
   time_gap_config = TimeGapConfig(
       max_gap_seconds=66.0,
       tolerance_seconds=10.0  # Increase tolerance
   )
   ```

### Debug Mode
```python
import logging
logging.getLogger("EnhancedDataValidation").setLevel(logging.DEBUG)
```

## Testing

### Unit Tests
```python
# Test individual validators
def test_klines_validation():
    validator = get_validator(DataType.KLINES)
    test_data = [{"open_time": 1640995200000, "open": "3000.0", ...}]
    result = validator.validate_batch(test_data)
    assert len(result) == 1
```

### Integration Tests
```python
# Test complete pipeline
async def test_enhanced_pipeline():
    summary = await run_enhanced_data_collection_pipeline(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m"
    )
    assert summary['overall_success'] == True
```

## Migration Guide

### From Existing Pipeline
1. **Replace existing validators** with enhanced validators
2. **Update field mappings** for your exchange
3. **Configure time gap tolerances** based on your data
4. **Test with existing data** to ensure compatibility

### Backward Compatibility
- Enhanced framework maintains compatibility with existing schemas
- Existing data files can be validated without modification
- Gradual migration supported

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review validation error messages
3. Enable debug logging for detailed information
4. Test with sample data to isolate issues

## Future Enhancements

- **Real-time streaming validation** for live data feeds
- **Machine learning-based anomaly detection** for data quality
- **Advanced time series validation** with trend analysis
- **Multi-exchange data normalization** and comparison
- **Automated data quality scoring** and reporting