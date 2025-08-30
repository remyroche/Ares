# Structured Data Cache Implementation

## Overview

This document describes the implementation of a structured data cache directory system that organizes data by exchange and asset pairs, replacing the previous flat `data_cache/` structure.

## New Directory Structure

### Before (Flat Structure)
```
data_cache/
├── klines_BINANCE_ETHUSDT_1m_consolidated.parquet
├── aggtrades_BINANCE_ETHUSDT_consolidated.parquet
├── unified_BINANCE_ETHUSDT_1m.parquet
└── ...
```

### After (Structured Structure)
```
data_cache/
├── binance/
│   ├── ethusdt/
│   │   ├── klines_BINANCE_ETHUSDT_1m_consolidated.parquet
│   │   ├── aggtrades_BINANCE_ETHUSDT_consolidated.parquet
│   │   ├── unified/
│   │   │   └── unified_BINANCE_ETHUSDT_1m.parquet
│   │   ├── processed/
│   │   │   └── BINANCE_ETHUSDT_1m_validated_data.parquet
│   │   └── backup_pre_unified/
│   │       └── ...
│   ├── btcusdt/
│   │   └── ...
│   └── adausdt/
│       └── ...
├── coinbase/
│   ├── ethusdt/
│   │   └── ...
│   └── ...
└── ...
```

## Files Modified

### 1. Data Downloaders

#### `backtesting/ares_data_downloader_optimized.py`
- **Modified**: `DownloadConfig` class
  - Added `data_dir: str = None` parameter
- **Modified**: `OptimizedDataDownloader.__init__()` method
  - Updated to use structured directory: `data_cache/exchange/asset/`
  - Added fallback to use provided `data_dir` if specified

#### `backtesting/ares_data_downloader_clean.py`
- **Modified**: `DownloadConfig` class
  - Added `data_dir: str = None` parameter
- **Modified**: `CleanDataDownloader.__init__()` method
  - Updated to use structured directory: `data_cache/exchange/asset/`
  - Added fallback to use provided `data_dir` if specified

### 2. Training Steps

#### `src/training/steps/data_downloader.py`
- **Modified**: `download_all_data_with_consolidation()` function
  - Added `data_dir: str = None` parameter
  - Updated to pass `data_dir` to both optimized and clean downloader configs

#### `src/training/steps/step1_data_collection.py`
- **Modified**: `run_step()` function
  - Changed default `data_dir` from `"data_cache"` to `None`
  - Added logic to construct structured directory: `data_cache/exchange/asset/`
  - Updated file path construction to use structured directory
  - Updated data existence checks to use structured paths

#### `src/training/steps/step1_5_data_converter.py`
- **Modified**: `UnifiedDataConverter.__init__()` method
  - Updated to use structured directory in `execute()` method
- **Modified**: `run_step()` function
  - Changed default `data_dir` from `"data_cache"` to `None`
  - Added logic to construct structured directory: `data_cache/exchange/asset/`

#### `src/training/steps/step2_data_reading.py`
- **Modified**: `read_unified_data()` method
  - Updated unified data path construction to use structured directory
- **Modified**: `execute()` method
  - Updated processed data output path to use structured directory
- **Modified**: `run_step()` and `run_step_enhanced()` functions
  - Changed default `data_dir` from `"data_cache"` to `None`
  - Added logic to construct structured directory: `data_cache/exchange/asset/`
- **Added**: `os` import for path operations

## Key Changes

### 1. Directory Structure
- **Base**: `data_cache/`
- **Exchange Level**: `data_cache/{exchange.lower()}/`
- **Asset Level**: `data_cache/{exchange.lower()}/{symbol.lower()}/`
- **Subdirectories**:
  - `unified/` - For unified data from step1_5
  - `processed/` - For processed data from step2
  - `backup_pre_unified/` - For backups before unification

### 2. Parameterization
- **Dynamic Parameters**: All functions now use exchange and symbol parameters from function arguments
- **No Hardcoded Values**: Removed all hardcoded "BINANCE" and "ETHUSDT" default values
- **Required Parameters**: Symbol and exchange are now required parameters with proper validation
- **Flexible Defaults**: `data_dir` defaults to `None` and is constructed dynamically

### 2. File Naming Convention
- **Klines**: `klines_{EXCHANGE}_{SYMBOL}_{TIMEFRAME}_consolidated.parquet`
- **Aggtrades**: `aggtrades_{EXCHANGE}_{SYMBOL}_consolidated.parquet`
- **Unified**: `unified_{EXCHANGE}_{SYMBOL}_{TIMEFRAME}.parquet`
- **Processed**: `{EXCHANGE}_{SYMBOL}_{TIMEFRAME}_validated_data.parquet`

### 3. Backward Compatibility
- All functions now accept `data_dir=None` as default
- When `data_dir` is `None`, the structured directory is automatically constructed
- When `data_dir` is provided, it's used as-is (allowing custom paths)
- Symbol and exchange parameters are required (no more hardcoded defaults)

### 4. Configuration Updates
- `DownloadConfig` classes now include `data_dir` parameter
- Downloaders automatically create the structured directory structure
- All file operations use the structured paths

## Benefits

### 1. Organization
- **Clear Separation**: Each exchange and asset pair has its own directory
- **Scalability**: Easy to add new exchanges and assets without cluttering
- **Maintenance**: Easier to manage and clean up specific data

### 2. Performance
- **Reduced File System Load**: Fewer files in single directories
- **Faster Searches**: Directory-based organization improves file lookup
- **Better Caching**: OS-level caching works better with organized structures

### 3. Multi-Asset Support
- **Parallel Processing**: Multiple assets can be processed simultaneously
- **Isolation**: Issues with one asset don't affect others
- **Independent Updates**: Each asset can be updated independently

### 4. Debugging and Monitoring
- **Clear Structure**: Easy to identify which files belong to which asset
- **Logging**: Better log organization with structured paths
- **Troubleshooting**: Easier to isolate and fix issues per asset

## Usage Examples

### Basic Usage (Automatic Structured Directory)
```python
# Step 1: Data Collection
await run_step(
    symbol="ETHUSDT",
    exchange="BINANCE",
    timeframe="1m"
    # data_dir=None (default) - will use data_cache/binance/ethusdt/
)

# Step 1.5: Data Conversion
await run_step(
    symbol="ETHUSDT",
    exchange="BINANCE",
    timeframe="1m"
    # data_dir=None (default) - will use data_cache/binance/ethusdt/
)

# Step 2: Data Reading
await run_step(
    symbol="ETHUSDT",
    exchange="BINANCE",
    timeframe="1m"
    # data_dir=None (default) - will use data_cache/binance/ethusdt/
)
```

### Custom Directory Usage
```python
# Use custom directory structure
await run_step(
    symbol="ETHUSDT",
    exchange="BINANCE",
    timeframe="1m",
    data_dir="/custom/path/to/data"
)
```

### Multiple Assets
```python
# Process multiple assets in parallel
assets = [
    ("ETHUSDT", "BINANCE", "1m"),
    ("BTCUSDT", "BINANCE", "1m"),
    ("ADAUSDT", "BINANCE", "1m"),
]

for symbol, exchange, timeframe in assets:
    await run_step(symbol=symbol, exchange=exchange, timeframe=timeframe)
    # Each will use its own structured directory:
    # - data_cache/binance/ethusdt/
    # - data_cache/binance/btcusdt/
    # - data_cache/binance/adausdt/
```

## Testing

Comprehensive test scripts have been created to verify:

1. **Directory Structure Creation**: Ensures proper directory hierarchy
2. **File Path Construction**: Validates correct file naming
3. **Directory Listing**: Confirms proper organization
4. **Path Resolution**: Tests absolute and relative path handling
5. **Multiple Combinations**: Verifies support for different exchange/asset pairs
6. **Parameterization**: Ensures all functions use dynamic parameters instead of hardcoded values

### Running Tests
```bash
# Test directory structure
python3 test_structured_data_cache_simple.py --symbol BTCUSDT --exchange COINBASE --timeframe 5m

# Test with different parameters
python3 test_structured_data_cache_simple.py --symbol ADAUSDT --exchange BINANCE --timeframe 15m
```

## Migration Notes

### For Existing Users
- **Automatic Migration**: Existing code will automatically use the new structure
- **No Breaking Changes**: All function signatures remain compatible
- **Gradual Migration**: Old flat structure can coexist during transition

### For New Implementations
- **Recommended**: Use the new structured approach
- **Best Practice**: Let the system auto-construct directories
- **Custom Paths**: Use `data_dir` parameter for custom requirements

## Future Enhancements

### Potential Improvements
1. **Compression**: Add support for compressed data storage
2. **Versioning**: Implement data versioning within asset directories
3. **Metadata**: Add metadata files for each asset directory
4. **Cleanup**: Automated cleanup of old/unused data
5. **Backup**: Automated backup strategies for structured data

### Monitoring and Logging
1. **Directory Monitoring**: Track directory sizes and growth
2. **Access Logging**: Log data access patterns
3. **Performance Metrics**: Monitor read/write performance
4. **Health Checks**: Automated health checks for data integrity

## Conclusion

The structured data cache implementation provides a robust, scalable, and organized approach to data management. It maintains backward compatibility while offering significant improvements in organization, performance, and maintainability. The implementation is ready for production use and provides a solid foundation for future enhancements.