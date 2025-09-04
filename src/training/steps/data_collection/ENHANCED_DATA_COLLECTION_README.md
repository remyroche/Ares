# Enhanced Data Collection Framework

## Overview

The Enhanced Data Collection Framework provides comprehensive, production-ready data collection with extensive validation, logging, and API-agnostic capabilities. It ensures data quality, consistency, and compliance with downstream processing requirements.

## 🚀 Key Features

### ✅ **Extensive Logging and Printing**
- Comprehensive logging at all levels (DEBUG, INFO, WARNING, ERROR)
- Detailed progress tracking and performance metrics
- Structured logging with context and timestamps
- Real-time validation feedback

### ✅ **Integration with Utils/ Decorators**
- `@handles_errors` for robust error handling
- `@traced` for distributed tracing
- `@memory_efficient` for memory optimization
- `@resource_monitor` for resource tracking
- `@with_enhanced_mlflow_logging` for MLflow integration

### ✅ **Field Mapping for Different Exchanges**
- **Binance**: `open_time` → `timestamp`, `p` → `price`, etc.
- **Coinbase**: `price_open` → `open`, `size` → `quantity`, etc.
- **Kraken**: `time` → `timestamp`, `vol` → `volume`, etc.
- **Gate.io**: `t` → `timestamp`, `o` → `open`, etc.
- **MEXC**: `open_time` → `timestamp`, `p` → `price`, etc.
- **OKX**: `ts` → `timestamp`, `o` → `open`, etc.

### ✅ **Data Qualification with Duplicate Removal**
- Automatic duplicate detection and removal
- Primary key-based deduplication
- Data quality scoring and metrics
- Comprehensive validation reporting

### ✅ **API-Agnostic Data Collection**
- Integration with `exchange/` directory
- Support for multiple exchange APIs
- Standardized data collection interface
- Fallback mechanisms for API failures

### ✅ **Comprehensive Gap Detection**
- **Klines**: 1.1 minute maximum gap (66 seconds + 5 second tolerance)
- **Aggtrades**: 1 second maximum gap (1 second + 0.1 second tolerance)
- **Futures**: 9 hour maximum gap (32400 seconds + 5 minute tolerance)
- Automatic gap filling and reporting

### ✅ **Incremental Downloading**
- Batches start where previous batch ended
- No data loss between batches
- Configurable batch sizes
- Resume capability from last timestamp

## 📁 File Structure

```
src/training/steps/data_collection/
├── enhanced_validation_framework_with_decorators.py    # Core validation framework
├── exchange_field_mappings.py                         # Exchange field mappings
├── enhanced_api_agnostic_data_collector.py            # API-agnostic collector
├── enhanced_data_collection_demo.py                   # Comprehensive demo
├── ENHANCED_DATA_COLLECTION_README.md                 # This documentation
└── ENHANCED_VALIDATION_README.md                      # Detailed validation docs
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Enhanced Data Collection Framework           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Field         │  │   Validation    │  │   API        │ │
│  │   Mapping       │  │   Framework     │  │   Collector  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Gap           │  │   Incremental   │  │   Data       │ │
│  │   Detection     │  │   Downloading   │  │   Quality    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Decorators    │  │   Logging       │  │   Error      │ │
│  │   Integration   │  │   Framework     │  │   Handling   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Basic Usage

```python
from src.training.steps.data_collection.enhanced_validation_framework_with_decorators import (
    DataType, validate_data_batch
)

# Validate klines data
klines_data = [
    {
        "open_time": 1640995200000,  # Binance format
        "open": "3000.0",
        "high": "3100.0",
        "low": "2900.0",
        "close": "3050.0",
        "volume": "1000.0"
    }
]

validated_data = validate_data_batch(DataType.KLINES, klines_data, "BINANCE")
print(f"Validated {len(validated_data)} rows")
```

### API-Agnostic Data Collection

```python
from src.training.steps.data_collection.enhanced_api_agnostic_data_collector import (
    collect_incremental_data, detect_and_fill_gaps
)

# Collect data incrementally
result = await collect_incremental_data(
    exchange="BINANCE",
    symbol="ETHUSDT",
    timeframe="1m",
    data_types=["klines"],
    max_batches=10
)

# Detect and fill gaps
gap_result = await detect_and_fill_gaps(
    exchange="BINANCE",
    symbol="ETHUSDT",
    timeframe="1m",
    data_types=["klines"]
)
```

### Field Mapping

```python
from src.training.steps.data_collection.exchange_field_mappings import get_exchange_mapper

# Get field mapper for Binance
mapper = get_exchange_mapper("binance")

# Map exchange-specific fields to standardized fields
raw_data = {"open_time": 1640995200000, "open": "3000.0"}
mapped_data = mapper.map_fields("klines", raw_data)
print(mapped_data)  # {'timestamp': 1640995200000, 'open': '3000.0'}
```

## 📊 Data Schemas

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

## 🔧 Configuration

### Exchange Field Mappings

The framework supports comprehensive field mappings for all major exchanges:

```python
# Binance
{
    "timestamp": "open_time",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume"
}

# Coinbase
{
    "timestamp": "timestamp",
    "open": "price_open",
    "high": "price_high",
    "low": "price_low",
    "close": "price_close",
    "volume": "volume"
}

# Kraken
{
    "timestamp": "time",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "vol"
}
```

### Validation Rules

```python
# Custom validation rules
def custom_price_validator(value):
    """Custom validator for price fields."""
    return 0 < value < 1000000  # Price between 0 and 1M

field_def = FieldDefinition(
    name="price",
    dtype="float64",
    min_value=0.0,
    allow_zero=False,
    custom_validator=custom_price_validator
)
```

## 📈 Performance Features

### Memory Optimization
- Batch processing with configurable batch sizes
- Memory-efficient data structures
- Automatic garbage collection
- Resource monitoring and limits

### Error Handling
- Non-blocking error handling for non-critical issues
- Configurable error severity levels
- Detailed error reporting with context
- Graceful degradation and fallbacks

### Logging and Monitoring
- Structured logging with context
- Performance metrics and timing
- MLflow integration for experiment tracking
- Real-time validation feedback

## 🔍 Gap Detection

### Automatic Gap Detection
```python
# Detect gaps in existing data
gaps = gap_detector.detect_gaps(data, "klines")

# Get gap summary
summary = gap_detector.get_gap_summary(gaps)
print(f"Total gaps: {summary['total_gaps']}")
print(f"Total gap time: {summary['total_gap_minutes']:.1f} minutes")
```

### Gap Filling
```python
# Fill detected gaps
for gap in gaps:
    success, data, _ = await downloader.download_incremental_batch(
        data_type="klines",
        start_timestamp=gap['start_timestamp'],
        end_timestamp=gap['end_timestamp']
    )
```

## 📊 Data Quality Metrics

### Validation Summary
```python
{
    "data_type": "klines",
    "exchange": "BINANCE",
    "total_rows_processed": 1000,
    "valid_rows": 995,
    "invalid_rows": 5,
    "success_rate": 99.5,
    "time_gaps_detected": 0,
    "duplicates_removed": 2,
    "total_errors": 5,
    "error_breakdown": {
        "critical": 0,
        "high": 2,
        "medium": 2,
        "low": 1
    }
}
```

### Quality Scoring
- **Success Rate**: Percentage of valid rows
- **Gap Detection**: Number and duration of time gaps
- **Duplicate Removal**: Number of duplicates found and removed
- **Error Classification**: Breakdown by severity level

## 🚀 Advanced Features

### Incremental Data Collection
```python
# Collect data incrementally
collector = EnhancedAPIAgnosticDataCollector("BINANCE", "ETHUSDT", "1m")

# Download incremental batches
for batch_num in range(max_batches):
    success, data, next_timestamp = await collector.download_incremental_batch(
        data_type="klines",
        start_timestamp=last_timestamp,
        batch_size=1000
    )
    
    if success:
        last_timestamp = next_timestamp
```

### Period-Based Collection
```python
# Collect data for specific period
start_time = datetime.now() - timedelta(hours=24)
end_time = datetime.now()

result = await collector.collect_data_for_period(
    start_time=start_time,
    end_time=end_time,
    data_types=["klines", "aggtrades"]
)
```

### Batch Management
```python
# Download without erasing previous batches
result = await collector.collect_incremental_data(
    data_types=["klines"],
    max_batches=10,
    preserve_existing=True
)
```

## 🧪 Testing and Demo

### Run Comprehensive Demo
```bash
python src/training/steps/data_collection/enhanced_data_collection_demo.py
```

### Demo Features
- Field mapping validation for all exchanges
- Data validation with error handling
- Data qualification with duplicate removal
- API-agnostic data collection
- Gap detection and filling
- Comprehensive feature integration

## 🔧 Integration

### With Existing Pipeline
```python
# Replace existing validators
from src.training.steps.data_collection.enhanced_validation_framework_with_decorators import get_validator

# Use enhanced validator
validator = get_validator(DataType.KLINES, "BINANCE")
validated_data = validator.validate_batch(raw_data)
```

### With Exchange APIs
```python
# Use API-agnostic collector
from src.training.steps.data_collection.enhanced_api_agnostic_data_collector import (
    EnhancedAPIAgnosticDataCollector
)

collector = EnhancedAPIAgnosticDataCollector("BINANCE", "ETHUSDT", "1m")
result = await collector.collect_incremental_data()
```

## 📋 Migration Guide

### From Existing Framework
1. **Replace validators** with enhanced validators
2. **Update field mappings** for your exchange
3. **Configure time gap tolerances** based on your data
4. **Test with existing data** to ensure compatibility

### Backward Compatibility
- Enhanced framework maintains compatibility with existing schemas
- Existing data files can be validated without modification
- Gradual migration supported

## 🚨 Error Handling

### Error Severity Levels
- **CRITICAL**: Stop processing (missing required fields, type conversion failures)
- **HIGH**: Log error, continue with warning (NaN values, infinite values, negative prices)
- **MEDIUM**: Log warning, continue (zero values, time gaps)
- **LOW**: Log info, continue (minor format issues)

### Error Recovery
```python
try:
    validated_data = validator.validate_batch(raw_data)
except ValueError as e:
    logger.error(f"Critical validation error: {e}")
    # Handle critical errors
except Exception as e:
    logger.warning(f"Non-critical validation error: {e}")
    # Continue with partial data
```

## 📊 Monitoring and Metrics

### Performance Metrics
- Data collection throughput (rows/second)
- Validation success rates
- Gap detection accuracy
- Memory usage and optimization

### Quality Metrics
- Data completeness scores
- Validation error rates
- Gap frequency and duration
- Duplicate detection rates

## 🔮 Future Enhancements

- **Real-time streaming validation** for live data feeds
- **Machine learning-based anomaly detection** for data quality
- **Advanced time series validation** with trend analysis
- **Multi-exchange data normalization** and comparison
- **Automated data quality scoring** and reporting
- **Distributed validation** for large datasets
- **Custom validation rules** via configuration files

## 📞 Support

For issues or questions:
1. Check the troubleshooting section in the validation README
2. Review validation error messages and logs
3. Enable debug logging for detailed information
4. Test with sample data to isolate issues
5. Check exchange API documentation for field mappings

## 📄 License

This enhanced data collection framework is part of the larger trading system and follows the same licensing terms.

---

**🎉 The Enhanced Data Collection Framework provides production-ready, comprehensive data collection with extensive validation, logging, and API-agnostic capabilities. It ensures data quality, consistency, and compliance with downstream processing requirements while maintaining high performance and reliability.**