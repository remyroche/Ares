# Consolidated Data Downloader

This document describes the consolidated data downloader that has been created to streamline the data collection process in the Ares trading system.

## Overview

The consolidated data downloader (`data_downloader.py`) combines all the functionality from the original `backtesting/ares_data_downloader.py` into a single, reusable module that can be called by `training/steps/step1_data_collection.py`.

## Files

### `src/training/steps/data_downloader.py`
The main consolidated data downloader module that contains:

- **`download_all_data(symbol, interval, lookback_years)`**: Main async function that downloads all data types
- **`download_klines_data()`**: Downloads k-line (candlestick) data
- **`download_agg_trades_data()`**: Downloads aggregated trades data  
- **`download_futures_data()`**: Downloads futures data (funding rates)
- **`get_monthly_periods()`**: Helper function for monthly data periods
- **`get_daily_periods()`**: Helper function for daily data periods
- **`download_with_retry()`**: Retry wrapper for API calls

### Updated `src/training/steps/step1_data_collection.py`
The step1 data collection module has been updated to use the consolidated downloader:

- **`fetch_missing_data()`**: Now calls the consolidated downloader instead of individual API calls
- **Import**: Added import for `download_all_data` from the data_downloader module

## Key Features

### 1. **Consolidated Functionality**
- All data downloading logic is now in one place
- Consistent error handling and logging
- Unified retry mechanisms

### 2. **Incremental Downloads**
- Checks for existing data files before downloading
- Only downloads new data since the last update
- Prevents duplicate downloads

### 3. **Multiple Data Types**
- **K-lines**: OHLCV candlestick data
- **Aggregated Trades**: Trade execution data
- **Futures Data**: Funding rates

### 4. **Robust Error Handling**
- Network operation decorators with retry logic
- File operation decorators with permission handling
- Graceful fallbacks when imports fail

### 5. **Flexible Configuration**
- Supports different symbols, intervals, and lookback periods
- Configurable retry attempts and delays
- Environment-aware logging

## Usage

### Direct Usage
```python
from src.training.steps.data_downloader import download_all_data

# Download data for BTCUSDT
results = await download_all_data("BTCUSDT", "1h", 5)
```

### Integration with Step1
The `fetch_missing_data()` function in step1_data_collection.py now automatically uses the consolidated downloader:

```python
# This function now uses the consolidated downloader internally
result = await fetch_missing_data(symbol, start_time_ms, end_time_ms, available_data, logger)
```

## Data Storage

### Cache Directory Structure
```
data_cache/
├── klines_BTCUSDT_1h_2025-01.csv
├── aggtrades_BTCUSDT_2025-01-15.csv
└── futures_BTCUSDT_2025-01.csv
```

### File Naming Convention
- **K-lines**: `klines_{symbol}_{interval}_{YYYY-MM}.csv`
- **Aggregated Trades**: `aggtrades_{symbol}_{YYYY-MM-DD}.csv`
- **Futures**: `futures_{symbol}_{YYYY-MM}.csv`

## Configuration

### Default Settings
```python
CACHE_DIR = "data_cache"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
```

### Environment Variables
The downloader automatically uses configuration from:
- `src.config.CONFIG` for symbol and interval settings
- `src.config.settings` for API credentials
- Falls back to default values if imports fail

## Error Handling

### Network Errors
- Automatic retry with exponential backoff
- Timeout protection for long-running requests
- Graceful degradation when API is unavailable

### File System Errors
- Permission error handling with helpful messages
- Automatic directory creation
- File corruption detection and recovery

### Import Errors
- Mock implementations when dependencies are missing
- Fallback logging when system_logger is unavailable
- Graceful degradation for development environments

## Testing

A test script (`test_data_downloader.py`) is provided to verify the functionality:

```bash
python test_data_downloader.py
```

The test script validates:
1. Direct data downloader functionality
2. Integration with step1_data_collection
3. Error handling and fallback mechanisms

## Migration from Original

### What Changed
- **Before**: Multiple separate download functions in step1
- **After**: Single consolidated downloader with unified interface

### Benefits
- **Reduced Code Duplication**: Single source of truth for data downloading
- **Better Error Handling**: Consistent retry and fallback mechanisms
- **Easier Maintenance**: All download logic in one place
- **Improved Testing**: Can test downloader independently

### Backward Compatibility
- The step1 interface remains the same
- Existing data files are still compatible
- No changes needed to other training steps

## Future Enhancements

### Potential Improvements
1. **Parallel Downloads**: Download multiple data types simultaneously
2. **Compression**: Compress cached data files to save space
3. **Validation**: Add data quality checks and validation
4. **Monitoring**: Add metrics and monitoring for download performance
5. **Caching**: Implement more sophisticated caching strategies

### Configuration Options
1. **Rate Limiting**: Configurable API rate limits
2. **Data Retention**: Automatic cleanup of old data files
3. **Quality Settings**: Configurable data quality thresholds
4. **Notification**: Email/Slack notifications for download failures

## Troubleshooting

### Common Issues

1. **Permission Errors**
   ```
   ERROR: Could not create/access 'data_cache' directory
   SOLUTION: chmod -R u+rw data_cache
   ```

2. **Import Errors**
   ```
   ERROR: Could not import CONFIG or logging utilities
   SOLUTION: Check that src/config.py exists and is properly configured
   ```

3. **API Timeouts**
   ```
   ERROR: Aggregated trades fetch timed out after 30 seconds
   SOLUTION: Check network connection and API availability
   ```

### Debug Mode
Enable debug logging by setting the log level:
```python
import logging
logging.getLogger('DataDownloader').setLevel(logging.DEBUG)
```

## Conclusion

The consolidated data downloader provides a robust, maintainable solution for data collection in the Ares trading system. It combines the best practices from the original implementation while providing a cleaner, more modular architecture that's easier to test and maintain. 