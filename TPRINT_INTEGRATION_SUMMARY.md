# tprint_data_preview Integration Summary

## Overview

Successfully integrated `tprint_data_preview` from `src/utils/tprint.py` into the exchanges standardization logic. This integration provides helpful data previews during the data processing pipeline, making it easier to debug and understand data transformations.

## Integration Points

### 1. UnifiedExchangeStandardizer (`exchanges/shared/unified_exchange_standardizer.py`)

**Updated Methods:**
- `standardize_data()` - Added `enable_data_preview: bool = True` parameter
- `standardize_to_dataframe()` - Added `enable_data_preview: bool = True` parameter
- `standardize_exchange_ohlcv()` - Added `enable_data_preview: bool = True` parameter

**Preview Points:**
- Raw data preview before processing
- Standardized data preview after conversion
- Final DataFrame preview after optimization

### 2. UnifiedOHLCVStandardizer (`exchanges/shared/unified_ohlcv_standardizer.py`)

**Updated Methods:**
- `standardize_data()` - Added `enable_data_preview: bool = True` parameter

**Preview Points:**
- Raw data preview before processing
- Standardized data preview after conversion

### 3. Klines Adapters

**Updated Adapters:**
- `exchanges/binance/klines_adapter.py`
- `exchanges/bingx/klines_adapter.py`
- `exchanges/okx/klines_adapter.py`
- `exchanges/mexc/klines_adapter.py`
- `exchanges/gateio/klines_adapter.py`
- `exchanges/phemex/klines_adapter.py`

**Updated Methods:**
- `get_klines_data()` - Added `enable_data_preview: bool = True` parameter
- `download_and_process_klines()` - Added `enable_data_preview: bool = True` parameter

**Preview Points:**
- Raw klines data from exchange APIs
- Processed data after pipeline processing

## Features Added

### 1. Data Preview Integration
- **Raw Data Preview**: Shows incoming data before standardization
- **Standardized Data Preview**: Shows data after conversion to unified format
- **Final Data Preview**: Shows processed data ready for use

### 2. Configuration Options
- **Environment Variable**: `ENABLE_DATA_PREVIEW=true` to enable/disable previews globally
- **Method Parameter**: `enable_data_preview=True` to control per-method
- **Log Level Control**: Previews use DEBUG/INFO levels for appropriate visibility

### 3. Smart Preview Features
- **Automatic Data Type Detection**: Handles lists, dicts, DataFrames, numpy arrays
- **Size Limits**: Prevents log pollution with large datasets
- **Metadata Display**: Shows data shape, memory usage, quality metrics
- **Truncation**: Smart truncation of large data for readability

## Usage Examples

### Basic Usage
```python
from exchanges.shared.unified_exchange_standardizer import standardize_exchange_ohlcv

# With data preview (default)
df = standardize_exchange_ohlcv(
    raw_data, 
    "binance", 
    "BTCUSDT", 
    "5m",
    enable_data_preview=True
)

# Without data preview
df = standardize_exchange_ohlcv(
    raw_data, 
    "binance", 
    "BTCUSDT", 
    "5m",
    enable_data_preview=False
)
```

### Klines Adapter Usage
```python
from exchanges.binance.klines_adapter import BinanceKlinesAdapter

adapter = BinanceKlinesAdapter()

# With data preview (default)
data = await adapter.get_klines_data(
    "BTCUSDT", 
    "5m", 
    enable_data_preview=True
)

# Process with preview
processed = await adapter.download_and_process_klines(
    "BTCUSDT", 
    "5m", 
    enable_data_preview=True
)
```

### Environment Control
```bash
# Enable data previews globally
export ENABLE_DATA_PREVIEW=true

# Disable data previews globally
export ENABLE_DATA_PREVIEW=false
```

## Preview Output Examples

### Raw Data Preview
```
[2025-10-21 14:10:13.284] DEBUG: 📊 Raw binance data for BTCUSDT (5m) preview:
[2025-10-21 14:10:13.284] DEBUG:   Type: list
[2025-10-21 14:10:13.284] DEBUG:   Length: 10
[2025-10-21 14:10:13.284] DEBUG:   Memory: 0.00 MB
[2025-10-21 14:10:13.284] DEBUG:   Preview: [{'openTime': 1760969413285, 'open': '50000.00', ...}]
```

### Standardized Data Preview
```
[2025-10-21 14:10:13.284] DEBUG: 📊 Standardized binance data for BTCUSDT (5m) preview:
[2025-10-21 14:10:13.284] DEBUG:   Shape: (10, 15)
[2025-10-21 14:10:13.284] DEBUG:   Dtypes: {'symbol': 'object', 'timestamp': 'datetime64[ns]', ...}
[2025-10-21 14:10:13.284] DEBUG:   Memory: 0.00 MB
[2025-10-21 14:10:13.284] DEBUG:   Sample data (first 3 rows):
[2025-10-21 14:10:13.284] DEBUG:     symbol  timestamp  open  high  low  close  volume
[2025-10-21 14:10:13.284] DEBUG:   0  BTCUSDT 2025-10-21  50000  50050  49950  50025    100
```

## Benefits

### 1. Debugging & Development
- **Visual Data Inspection**: Quickly see data structure and content
- **Transformation Tracking**: Follow data through processing pipeline
- **Quality Assessment**: Identify data quality issues early

### 2. Monitoring & Operations
- **Data Validation**: Verify data correctness during processing
- **Performance Monitoring**: Track data processing efficiency
- **Error Diagnosis**: Identify issues in data transformation

### 3. Documentation & Learning
- **Data Flow Understanding**: Visualize how data moves through system
- **Format Examples**: See actual data structures and formats
- **Integration Examples**: Understand how different exchanges work

## Configuration

### Environment Variables
- `ENABLE_DATA_PREVIEW`: Global enable/disable (default: false)
- `DATA_PREVIEW_MAX_ROWS`: Maximum rows to show (default: 5)
- `DATA_PREVIEW_MAX_COLS`: Maximum columns to show (default: 10)
- `DATA_PREVIEW_LARGE_THRESHOLD`: Large dataset threshold (default: 10000)

### Method Parameters
- `enable_data_preview: bool = True`: Control per-method
- `max_rows: int = None`: Override default row limit
- `max_cols: int = None`: Override default column limit
- `level: LogLevel = LogLevel.DEBUG`: Control log level

## Testing

Created comprehensive test suite:
- `test_tprint_integration.py`: Full integration test with sample data
- `simple_tprint_test.py`: Basic functionality test without dependencies

Test results show:
- ✅ Basic data preview functionality working
- ✅ Integration points properly configured
- ✅ Method signatures updated correctly
- ✅ Environment variable control working

## Files Modified

### Core Standardization
- `exchanges/shared/unified_exchange_standardizer.py`
- `exchanges/shared/unified_ohlcv_standardizer.py`

### Klines Adapters
- `exchanges/binance/klines_adapter.py`
- `exchanges/bingx/klines_adapter.py`
- `exchanges/okx/klines_adapter.py`
- `exchanges/mexc/klines_adapter.py`
- `exchanges/gateio/klines_adapter.py`
- `exchanges/phemex/klines_adapter.py`

### Test Files
- `test_tprint_integration.py`
- `simple_tprint_test.py`

## Conclusion

The `tprint_data_preview` integration is now complete and provides comprehensive data preview capabilities throughout the exchanges standardization logic. This enhancement significantly improves debugging, monitoring, and understanding of data processing workflows while maintaining backward compatibility and performance.

The integration follows the existing codebase patterns and provides flexible configuration options to suit different use cases and environments.