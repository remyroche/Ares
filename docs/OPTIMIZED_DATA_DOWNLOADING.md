# Optimized Data Downloading for MEXC and GATEIO

## Overview

The optimized data downloading system provides enhanced performance for downloading klines, aggregated trades, and futures data from MEXC and GATEIO exchanges. It implements parallel processing, concurrent downloads, and intelligent caching to maximize efficiency while minimizing API usage.

## Key Features

### 1. Incremental Downloads
- **Smart File Detection**: Only downloads data that isn't already present
- **File-based Caching**: Saves data incrementally as it's received
- **Resume Capability**: Can resume interrupted downloads

### 2. Parallel Processing
- **Concurrent Downloads**: Downloads multiple time periods simultaneously
- **Data Type Parallelism**: Downloads klines, aggtrades, and futures in parallel
- **Configurable Concurrency**: Adjustable concurrent download limits

### 3. Optimized File Management
- **Monthly Files for Klines**: `klines_{exchange}_{symbol}_{interval}_{YYYY-MM}.csv`
- **Daily Files for Aggtrades**: `aggtrades_{exchange}_{symbol}_{YYYY-MM-DD}.csv`
- **Daily Files for Futures**: `futures_{exchange}_{symbol}_{YYYY-MM-DD}.csv`

## Usage

### Via Ares Launcher (Recommended)

```bash
# Optimized download for MEXC (default)
python ares_launcher.py load --symbol ETHUSDT --exchange MEXC --interval 1m

# Optimized download for GATEIO
python ares_launcher.py load --symbol ETHUSDT --exchange GATEIO --interval 1m

# Use standard downloader (disable optimization)
python ares_launcher.py load --symbol ETHUSDT --exchange MEXC --no-optimized
```

### Direct Script Usage

```bash
# Optimized downloader directly
python backtesting/ares_data_downloader_optimized.py --symbol ETHUSDT --exchange MEXC --interval 1m --lookback-years 2 --max-concurrent 5

# Test script
python scripts/test_optimized_download.py --symbol ETHUSDT --exchange MEXC --lookback-years 1
```

## Configuration

### DownloadConfig Parameters

```python
@dataclass
class DownloadConfig:
    symbol: str                    # Trading symbol (e.g., "ETHUSDT")
    exchange: str                  # Exchange name (e.g., "MEXC", "GATEIO")
    interval: str                  # K-line interval (e.g., "1m", "5m", "1h")
    lookback_years: int           # Years of historical data to download
    max_concurrent_downloads: int = 5    # Max concurrent download tasks
    max_concurrent_requests: int = 10    # Max concurrent API requests
    chunk_size: int = 1000        # Data processing chunk size
    retry_attempts: int = 3       # Number of retry attempts
    retry_delay: float = 1.0      # Delay between retries (seconds)
    rate_limit_delay: float = 0.1 # Rate limiting delay (seconds)
    memory_threshold: float = 0.8 # Memory usage threshold for cleanup
```

## File Structure

### Generated Files

```
data_cache/
├── klines_MEXC_ETHUSDT_1m_2024-01.csv
├── klines_MEXC_ETHUSDT_1m_2024-02.csv
├── klines_MEXC_ETHUSDT_1m_2024-03.csv
├── aggtrades_MEXC_ETHUSDT_2024-01-01.csv
├── aggtrades_MEXC_ETHUSDT_2024-01-02.csv
├── aggtrades_MEXC_ETHUSDT_2024-01-03.csv
├── futures_MEXC_ETHUSDT_2024-01-01.csv
├── futures_MEXC_ETHUSDT_2024-01-02.csv
└── futures_MEXC_ETHUSDT_2024-01-03.csv
```

### File Naming Convention

- **Klines**: `klines_{exchange}_{symbol}_{interval}_{YYYY-MM}.csv`
- **Aggtrades**: `aggtrades_{exchange}_{symbol}_{YYYY-MM-DD}.csv`
- **Futures**: `futures_{exchange}_{symbol}_{YYYY-MM-DD}.csv`

## Performance Optimizations

### 1. Connection Pooling
- **aiohttp Session**: Reuses HTTP connections
- **Keep-alive**: Maintains persistent connections
- **Connection Limits**: Configurable per-host limits

### 2. Rate Limiting
- **Semaphore Control**: Limits concurrent API requests
- **Exponential Backoff**: Intelligent retry mechanism
- **Exchange-specific Limits**: Respects exchange rate limits

### 3. Memory Management
- **Incremental Processing**: Processes data in chunks
- **Immediate Saving**: Saves files as data is received
- **Memory Monitoring**: Prevents memory overflow

### 4. Error Handling
- **Graceful Degradation**: Continues on partial failures
- **Detailed Logging**: Comprehensive error reporting
- **Retry Logic**: Automatic retry for transient errors

## Exchange-Specific Optimizations

### MEXC Optimizations
- **Concurrent Hour Processing**: Downloads multiple hours simultaneously
- **Optimized API Calls**: Uses larger limits for faster downloads
- **Connection Pooling**: Efficient HTTP connection management

### GATEIO Optimizations
- **Parallel Period Downloads**: Downloads multiple time periods concurrently
- **Rate Limit Compliance**: Respects GATEIO's rate limits
- **Error Recovery**: Robust error handling for GATEIO API

## Monitoring and Statistics

The optimized downloader provides comprehensive statistics:

```python
stats = {
    "klines_downloaded": 24,      # Number of klines files downloaded
    "aggtrades_downloaded": 730,  # Number of aggtrades files downloaded
    "futures_downloaded": 730,    # Number of futures files downloaded
    "total_time": 125.5,          # Total download time in seconds
    "errors": 2,                  # Number of errors encountered
}
```

## Best Practices

### 1. Concurrency Settings
- **Conservative**: Start with 3-5 concurrent downloads
- **Monitor**: Watch for rate limit errors
- **Adjust**: Increase gradually based on performance

### 2. Memory Management
- **Chunk Size**: Use 1000-5000 records per chunk
- **Cleanup**: Monitor memory usage during large downloads
- **Resume**: Use incremental downloads for large datasets

### 3. Error Handling
- **Logging**: Enable detailed logging for troubleshooting
- **Retries**: Use exponential backoff for transient errors
- **Monitoring**: Watch for API rate limit responses

## Troubleshooting

### Common Issues

1. **Rate Limit Errors**
   - Reduce `max_concurrent_requests`
   - Increase `rate_limit_delay`
   - Check exchange-specific limits

2. **Memory Issues**
   - Reduce `chunk_size`
   - Lower `max_concurrent_downloads`
   - Monitor `memory_threshold`

3. **Connection Errors**
   - Check network connectivity
   - Verify API credentials
   - Review firewall settings

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Comparison with Standard Downloader

| Feature | Standard Downloader | Optimized Downloader |
|---------|-------------------|---------------------|
| Parallel Processing | ❌ Sequential | ✅ Concurrent |
| File Caching | ❌ Basic | ✅ Smart Detection |
| Memory Management | ❌ Basic | ✅ Optimized |
| Error Recovery | ❌ Basic | ✅ Robust |
| Performance | ⚠️ Moderate | 🚀 High |
| Resource Usage | ⚠️ High | ✅ Efficient |

## Future Enhancements

1. **Database Integration**: Store metadata in SQLite
2. **Compression**: Compress CSV files for storage efficiency
3. **Validation**: Add data integrity checks
4. **Scheduling**: Automated periodic downloads
5. **Metrics**: Performance monitoring dashboard

## API Reference

### OptimizedDataDownloader

```python
class OptimizedDataDownloader:
    def __init__(self, config: DownloadConfig)
    async def initialize() -> bool
    async def cleanup()
    async def run_optimized_download() -> bool
    async def download_klines_parallel() -> bool
    async def download_aggtrades_parallel() -> bool
    async def download_futures_parallel() -> bool
```

### DownloadConfig

```python
@dataclass
class DownloadConfig:
    symbol: str
    exchange: str
    interval: str
    lookback_years: int
    max_concurrent_downloads: int = 5
    max_concurrent_requests: int = 10
    # ... additional parameters
```

## Examples

### Basic Usage

```python
from backtesting.ares_data_downloader_optimized import OptimizedDataDownloader, DownloadConfig

config = DownloadConfig(
    symbol="ETHUSDT",
    exchange="MEXC",
    interval="1m",
    lookback_years=2,
    max_concurrent_downloads=5
)

downloader = OptimizedDataDownloader(config)
success = await downloader.run_optimized_download()
```

### Custom Configuration

```python
config = DownloadConfig(
    symbol="BTCUSDT",
    exchange="GATEIO",
    interval="5m",
    lookback_years=1,
    max_concurrent_downloads=3,
    max_concurrent_requests=5,
    retry_attempts=5,
    rate_limit_delay=0.2
)
```

This optimized system provides significant performance improvements while maintaining reliability and respecting exchange rate limits. 