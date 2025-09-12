# Append Data Download System

This system ensures that when downloading data from exchanges, new files are created and each batch download appends to existing data without deleting it. This prevents data loss and allows for incremental data collection.

## 🚀 Key Features

- **Batch-based file naming**: Each download creates unique files with timestamps and batch numbers
- **Incremental downloading**: New downloads start where previous ones ended
- **Data consolidation**: Merge multiple batch files into unified datasets
- **Comprehensive monitoring**: Real-time progress tracking and performance metrics
- **Error handling**: Robust error recovery and fallback mechanisms
- **Memory efficient**: Handles large datasets with chunked processing

## 📁 File Structure

```
src/training/steps/data_collection/
├── enhanced_append_data_downloader.py    # Main append downloader
├── data_consolidation_manager.py         # Data consolidation utilities
├── data_download_monitor.py              # Monitoring and logging
├── unified_data_downloader.py            # Updated with append mode
└── standardized_parquet_handler.py       # File handling utilities
```

## 🔧 Usage Examples

### Basic Append Download

```python
from src.training.steps.data_collection.enhanced_append_data_downloader import download_data_with_append

# Download data with append functionality
result = await download_data_with_append(
    symbol="ETHUSDT",
    exchange="BINANCE",
    data_type="klines",
    timeframe="1m",
    max_batches=5
)

if result['success']:
    print(f"Downloaded {result['total_rows']} rows in {result['successful_batches']} batches")
```

### Monitored Download with Progress Tracking

```python
from src.training.steps.data_collection.data_download_monitor import (
    start_download_session,
    update_download_progress,
    end_download_session
)

# Start monitoring session
session = start_download_session(
    session_id="my_session_001",
    symbol="BTCUSDT",
    exchange="BINANCE",
    data_type="klines"
)

# Download with monitoring
downloader = EnhancedAppendDataDownloader()
result = await downloader.download_with_append(
    symbol="BTCUSDT",
    exchange="BINANCE",
    data_type="klines",
    max_batches=3
)

# Update progress for each batch
for batch_result in result['batch_results']:
    update_download_progress(
        session_id=session['session_id'],
        batch_number=batch_result['batch_number'],
        batch_success=batch_result['success'],
        rows_downloaded=batch_result['rows']
    )

# End session
summary = end_download_session(session['session_id'], 'completed')
```

### Data Consolidation

```python
from src.training.steps.data_collection.data_consolidation_manager import consolidate_session_data

# Consolidate all batch files from a session
consolidate_result = await consolidate_session_data(
    symbol="ETHUSDT",
    exchange="BINANCE",
    data_type="klines",
    timeframe="1m",
    session_id="my_session_001",
    remove_originals=False
)

if consolidate_result['success']:
    print(f"Consolidated {consolidate_result['total_rows']} rows")
    print(f"File: {consolidate_result['consolidated_file']}")
```

### List and Manage Data Files

```python
from src.training.steps.data_collection.enhanced_append_data_downloader import list_data_files

# List all available data
all_data = await list_data_files()
print(f"Total files: {all_data['total_files']} ({all_data['total_size_mb']:.2f} MB)")

# List data for specific symbol
eth_data = await list_data_files(symbol="ETHUSDT", exchange="BINANCE")
print(f"ETHUSDT files: {eth_data['total_files']}")
```

## 📊 File Naming Convention

The system uses a standardized naming convention for batch files:

```
{data_type}_{exchange}_{symbol}_{timeframe}_{session_id}_batch_{batch_number}_{timestamp}.parquet
```

Example:
```
klines_BINANCE_ETHUSDT_1m_20240101_120000_batch_001_20240101_120030.parquet
```

## 🔄 Data Flow

1. **Download Phase**: Data is downloaded in batches and saved to unique files
2. **Monitoring Phase**: Progress is tracked and logged in real-time
3. **Consolidation Phase**: Multiple batch files can be merged into unified datasets
4. **Management Phase**: Files can be listed, analyzed, and managed

## 📈 Monitoring and Logging

The system provides comprehensive monitoring capabilities:

- **Real-time progress tracking**: Monitor download progress in real-time
- **Performance metrics**: Track download speeds, memory usage, and quality scores
- **Error tracking**: Log and track errors and warnings
- **Historical data**: Maintain statistics across multiple sessions
- **Alert system**: Get notified of issues and failures

### Monitoring Dashboard

```python
from src.training.steps.data_collection.data_download_monitor import get_monitoring_dashboard

dashboard = get_monitoring_dashboard()
print(f"Total sessions: {dashboard['overview']['total_sessions']}")
print(f"Success rate: {dashboard['overview']['success_rate']:.1f}%")
print(f"Total rows downloaded: {dashboard['overview']['total_rows_downloaded']}")
```

## 🛠️ Configuration

### Environment Variables

```bash
# Data cache directory
export DATA_CACHE_PATH="/path/to/data/cache"

# Monitor file location
export MONITOR_FILE="/path/to/monitor.json"

# Logging level
export LOG_LEVEL="INFO"
```

### Configuration Options

```python
# Initialize with custom settings
downloader = EnhancedAppendDataDownloader(
    data_cache_path="custom_data_cache"
)

# Configure consolidation
consolidation_manager = DataConsolidationManager(
    data_cache_path="custom_data_cache"
)

# Configure monitoring
monitor = DataDownloadMonitor(
    data_cache_path="custom_data_cache",
    monitor_file="custom_monitor.json"
)
```

## 🔧 Advanced Usage

### Custom Batch Processing

```python
# Process files in chunks for large datasets
consolidate_result = await consolidation_manager.consolidate_all_available(
    symbol="ETHUSDT",
    exchange="BINANCE",
    data_type="klines",
    chunk_size=50,  # Process 50 files at a time
    max_memory_mb=2000  # Limit memory usage to 2GB
)
```

### Time Range Consolidation

```python
from datetime import datetime, timedelta

# Consolidate data from specific time range
consolidate_result = await consolidation_manager.consolidate_by_time_range(
    symbol="ETHUSDT",
    exchange="BINANCE",
    data_type="klines",
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now(),
    remove_originals=False
)
```

### Data Quality Monitoring

```python
# The system automatically validates data quality during download
# Quality scores are tracked in monitoring data
dashboard = get_monitoring_dashboard()
quality_scores = dashboard['performance']['quality_scores']
print(f"Recent quality scores: {quality_scores}")
```

## 🚨 Error Handling

The system includes comprehensive error handling:

- **Graceful degradation**: Falls back to standard mode if append mode fails
- **Retry mechanisms**: Automatic retry for transient failures
- **Error logging**: Detailed error information is logged and tracked
- **Recovery options**: Ability to resume failed downloads

### Error Recovery

```python
# Check for failed sessions
dashboard = get_monitoring_dashboard()
failed_sessions = [s for s in dashboard['recent_activity']['sessions'] 
                   if s['status'] == 'failed']

# Retry failed downloads
for session in failed_sessions:
    # Implement retry logic
    pass
```

## 📋 Best Practices

1. **Use unique session IDs**: Ensure each download session has a unique identifier
2. **Monitor memory usage**: Use chunked processing for large datasets
3. **Regular consolidation**: Periodically consolidate batch files to avoid file proliferation
4. **Error monitoring**: Set up alerts for failed downloads
5. **Data validation**: Always validate data quality after downloads
6. **Backup strategy**: Implement backup for critical data files

## 🔍 Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Memory issues**: Reduce batch size or use chunked processing
3. **File permission errors**: Check write permissions for data cache directory
4. **Network timeouts**: Increase timeout settings for exchange APIs

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
downloader = EnhancedAppendDataDownloader()
# ... rest of your code
```

## 📚 API Reference

### EnhancedAppendDataDownloader

- `download_with_append()`: Download data with append functionality
- `consolidate_batches()`: Consolidate multiple batch files
- `list_available_data()`: List all available data files

### DataConsolidationManager

- `consolidate_by_session()`: Consolidate files from a specific session
- `consolidate_by_time_range()`: Consolidate files from a time range
- `consolidate_all_available()`: Consolidate all available files

### DataDownloadMonitor

- `start_session()`: Start monitoring a download session
- `update_batch_progress()`: Update progress for a batch
- `end_session()`: End a monitoring session
- `get_monitoring_summary()`: Get comprehensive monitoring summary

## 🤝 Contributing

When contributing to this system:

1. Follow the existing code style and patterns
2. Add comprehensive error handling
3. Include logging for all operations
4. Write tests for new functionality
5. Update documentation for new features

## 📄 License

This system is part of the Ares trading pipeline and follows the same licensing terms.