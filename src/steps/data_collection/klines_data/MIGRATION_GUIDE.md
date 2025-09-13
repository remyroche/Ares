# Migration Guide: parquet_utils → klines_parquet

This guide explains how to migrate from `utils/parquet_utils.py` to `steps/data_collection/klines_data/klines_parquet.py` for better integration with historical klines data.

## Why Migrate?

The new `klines_parquet.py` provides:
- ✅ **Seamless access** to data in `historical_data/` directory structure
- ✅ **Backward compatibility** - same API as `parquet_utils.py`
- ✅ **Enhanced functionality** for klines-specific operations
- ✅ **Automatic data discovery** based on symbol and interval
- ✅ **Better error handling** and logging

## Quick Migration

### Before (using parquet_utils)
```python
from src.utils.parquet_utils import get_parquet_utils, safe_read_parquet

# Direct parquet file access
utils = get_parquet_utils()
df = utils.safe_read_parquet("historical_data/binance/ethusdt/raw/ethusdt_1m_2024_09.parquet")
```

### After (using klines_parquet)
```python
from src.steps.data_collection.klines_data import get_parquet_utils, safe_read_parquet

# Same API, but now works with klines data structure
utils = get_parquet_utils()
df = utils.safe_read_parquet("historical_data/binance/ethusdt/raw/ethusdt_1m_2024_09.parquet")

# Or use the enhanced klines manager directly
from src.steps.data_collection.klines_data import get_klines_manager
manager = get_klines_manager()
df = manager.read_data("ETHUSDT", "1m")  # Even simpler!
```

## API Compatibility

All `parquet_utils.py` functions are available with the same signatures:

| Function | Status | Notes |
|----------|--------|-------|
| `get_parquet_utils()` | ✅ Compatible | Returns `KlinesParquetManager` instance |
| `safe_read_parquet()` | ✅ Compatible | Enhanced with klines data support |
| `validate_parquet_file()` | ✅ Compatible | Same functionality |
| `safe_read_parquet_with_dtype_normalization()` | ✅ Compatible | Same functionality |
| `repair_parquet_file()` | ✅ Compatible | Same functionality |
| `harmonize_schema_after_read()` | ✅ Compatible | Same functionality |

## Enhanced Features

### Automatic Data Discovery
```python
# Old way - manual file path construction
file_path = "historical_data/binance/ethusdt/raw/ethusdt_1m_2024_09.parquet"
df = safe_read_parquet(file_path)

# New way - automatic discovery
manager = get_klines_manager()
df = manager.read_data("ETHUSDT", "1m")  # Finds all relevant files automatically
```

### Better Error Handling
```python
# Enhanced error messages and fallback strategies
df = safe_read_parquet("some_file.parquet")
# If file is not found in klines structure, falls back to direct reading
```

### Data Validation and Statistics
```python
manager = get_klines_manager()

# Get comprehensive data info
info = manager.get_data_info("ETHUSDT", "1m")
print(f"Records: {info['total_records']:,}")
print(f"Date range: {info['date_range']}")
print(f"File size: {info['file_size_mb']:.1f} MB")

# Get data statistics
stats = manager.get_data_statistics("ETHUSDT", "1m")
```

## Migration Steps

### Step 1: Update Imports
```python
# Change this:
from src.utils.parquet_utils import get_parquet_utils, safe_read_parquet

# To this:
from src.steps.data_collection.klines_data import get_parquet_utils, safe_read_parquet
```

### Step 2: Consider Enhanced Features
After updating imports, you can optionally use enhanced features:

```python
# Instead of direct file reading, use the manager
manager = get_klines_manager()

# Read data with automatic filtering
df = manager.read_data(
    symbol="ETHUSDT",
    interval="1m",
    start_date=datetime(2024, 9, 1),
    end_date=datetime(2024, 9, 30)
)

# Get data quality metrics
quality = manager.get_data_statistics("ETHUSDT", "1m")
```

### Step 3: Update File Paths (Optional)
If you're using hardcoded file paths, consider switching to the manager approach:

```python
# Before
files = [
    "historical_data/binance/ethusdt/raw/ethusdt_1m_2024_09.parquet",
    "historical_data/binance/ethusdt/raw/ethusdt_1m_2024_10.parquet",
    # ... more files
]

# After
manager = get_klines_manager()
df = manager.read_data("ETHUSDT", "1m")  # Handles all files automatically
```

## Testing Migration

Use the provided test script to verify your migration:

```bash
cd /Users/remyroche/Documents/Ares
python src/steps/data_collection/klines_data/test_klines_structure.py
```

## Rollback Plan

If you need to rollback:
1. Change imports back to `src.utils.parquet_utils`
2. No other code changes needed (API is identical)

## Benefits of Migration

1. **🎯 Better Data Access**: Seamless access to `historical_data/` structure
2. **🔍 Enhanced Discovery**: Automatic file discovery and metadata
3. **📊 Better Monitoring**: Comprehensive logging and statistics
4. **🛡️ Error Resilience**: Better error handling and fallback strategies
5. **📈 Future-Proof**: Built for klines-specific optimizations

## Support

If you encounter issues during migration:
1. Check the test script output
2. Verify your file paths and data structure
3. Review the enhanced error messages for guidance
4. See the README.md for detailed usage examples
