# Large Dataset Optimization Guide

## Installation

### Dependencies

The efficiency optimization dependencies are already included in the `pyproject.toml` file. Install them using Poetry:

```bash
# Install all dependencies including efficiency optimizations
poetry install

# Or add specific dependencies if needed
poetry add pyarrow fastparquet memory-profiler
```

### System Requirements

- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **CPU**: 4 cores (8 cores recommended)
- **OS**: Linux/macOS/Windows

## Overview

This guide provides comprehensive strategies for efficiently handling large datasets (2+ years of historical data) in the Ares trading system. The optimizations are designed to work within laptop constraints while maintaining training quality.

## Key Optimization Strategies

### 1. Intelligent Caching with SQLite Storage

**Problem**: Loading 2 years of data repeatedly is time-consuming and memory-intensive.

**Solution**: Multi-level caching system with SQLite database storage.

```python
# Initialize efficiency optimizer
efficiency_optimizer = DataEfficiencyOptimizer(db_manager, symbol, timeframe)

# Load data with intelligent caching
data = await efficiency_optimizer.load_data_with_caching(
    lookback_days=730,  # 2 years
    force_reload=False
)
```

**Benefits**:
- 24-hour cache validity with automatic refresh
- Compressed storage with pickle protocol
- Cache validation to ensure data integrity
- Memory-efficient loading with garbage collection

### 2. Time-Based Data Segmentation

**Problem**: Processing 2 years of data at once overwhelms memory.

**Solution**: Segment data by time periods for progressive processing.

```python
# Segment large datasets by time
segments = efficiency_optimizer.segment_data_by_time(
    data, segment_days=30  # 30-day segments
)

# Process each segment independently
for start_date, end_date, segment_data in segments:
    processed_segment = process_segment(segment_data)
    # Memory cleanup between segments
    if efficiency_optimizer.should_cleanup_memory():
        efficiency_optimizer.cleanup_memory()
```

**Benefits**:
- Configurable segment size (default: 30 days)
- Memory cleanup between segments
- Parallel processing capability
- Progress tracking per segment

### 3. Memory-Efficient Data Processing

**Problem**: Large DataFrames consume excessive memory.

**Solution**: Chunk-based processing with memory optimization.

```python
# Process data in configurable chunks
processed_data = efficiency_optimizer.process_data_in_chunks(
    data, chunk_size=10000  # 10k rows per chunk
)

# Memory optimization for DataFrames
optimized_df = efficiency_optimizer.optimize_dataframe_memory(df)
```

**Memory Optimization Techniques**:
- Downcast numeric types (int64 → int32, float64 → float32)
- Convert low-cardinality object columns to categories
- Automatic garbage collection at memory thresholds
- Memory usage monitoring and alerts

### 4. Database-Backed Feature Storage

**Problem**: Computing features repeatedly is expensive.

**Solution**: Store computed features in SQLite database for fast retrieval.

```python
# Store computed features
efficiency_optimizer.store_features_in_database(
    features_df, feature_type="technical"
)

# Load features for specific time period
cached_features = efficiency_optimizer.load_features_from_database(
    start_date, end_date, feature_names=["rsi", "macd"]
)
```

**Database Schema**:
```sql
CREATE TABLE feature_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    feature_name TEXT NOT NULL,
    feature_value REAL,
    feature_type TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**Benefits**:
- Fast feature retrieval by time period
- Selective feature loading
- Automatic indexing for performance
- Feature type categorization

### 5. Checkpoint and Resume Capabilities

**Problem**: Long training runs can be interrupted.

**Solution**: Automatic checkpointing with resume functionality.

```python
# Create processing checkpoint
efficiency_optimizer.create_processing_checkpoint(
    "training_ETHUSDT",
    {"stage": "feature_engineering", "progress": 0.75}
)

# Resume from checkpoint
checkpoint = efficiency_optimizer.get_latest_checkpoint("training_ETHUSDT")
if checkpoint:
    resume_training_from_checkpoint(checkpoint)
```

**Checkpoint Data**:
- Processing stage and progress
- Last processed timestamp
- Computed features count
- Memory usage statistics

### 6. Progressive Data Processing

**Problem**: Large datasets cause memory spikes.

**Solution**: Progressive loading and processing with memory management.

```python
# Progressive data loading
for segment in data_segments:
    # Load segment
    segment_data = load_segment(segment)
    
    # Process segment
    processed_segment = process_segment(segment_data)
    
    # Store results
    store_segment_results(processed_segment)
    
    # Memory cleanup
    del segment_data, processed_segment
    gc.collect()
```

## Configuration

### Enhanced Training Configuration

```python
CONFIG["ENHANCED_TRAINING"] = {
    "enable_efficiency_optimizations": True,
    "segment_days": 30,                    # Days per segment
    "chunk_size": 10000,                   # Rows per chunk
    "enable_feature_caching": True,        # Cache computed features
    "memory_threshold": 0.8,               # Memory cleanup threshold (80%)
    "cache_expiry_hours": 24,              # Cache validity period
    "database_cleanup_threshold_mb": 1000, # Database cleanup threshold
    "enable_checkpointing": True,          # Enable checkpoint/resume
    "max_segment_size": 50000,            # Maximum rows per segment
}
```

### Model Training Configuration

```python
CONFIG["MODEL_TRAINING"] = {
    "data_retention_days": 730,            # 2 years of data
    "min_data_points": 1000,              # Minimum data points
    "hyperparameter_tuning": {
        "max_trials": 50,                  # Reduced for efficiency
    },
    "coarse_hpo": {
        "n_trials": 20,                   # Reduced for efficiency
    },
}
```

## Usage Examples

### Basic Enhanced Training

```python
from src.training.enhanced_training_manager import EnhancedTrainingManager
from src.database.sqlite_manager import SQLiteManager

# Initialize
db_manager = SQLiteManager()
await db_manager.initialize()

training_manager = EnhancedTrainingManager(db_manager)

# Run enhanced training
session_id = await training_manager.run_full_training(
    symbol="ETHUSDT",
    exchange_name="BINANCE",
    timeframe="1h",
    lookback_days_override=730  # 2 years
)
```

### Custom Configuration

```python
# Update configuration for your needs
CONFIG["ENHANCED_TRAINING"]["segment_days"] = 60  # Larger segments
CONFIG["ENHANCED_TRAINING"]["chunk_size"] = 5000   # Smaller chunks
CONFIG["ENHANCED_TRAINING"]["memory_threshold"] = 0.7  # More aggressive cleanup

# Run with custom settings
training_manager = EnhancedTrainingManager(db_manager)
session_id = await training_manager.run_full_training("ETHUSDT")
```

### Monitoring and Statistics

```python
# Get efficiency statistics
stats = training_manager.get_efficiency_stats()
print(f"Memory usage: {stats['memory_usage_percent']:.1f}%")
print(f"Database size: {stats['database_size_mb']:.1f} MB")
print(f"Feature cache records: {stats['feature_cache_records']}")

# Get database statistics
db_stats = efficiency_optimizer.get_database_stats()
print(f"Raw data records: {db_stats['raw_data_records']}")
print(f"Feature types: {db_stats['feature_types']}")
```

## Performance Optimizations

### Memory Management

1. **Automatic Cleanup**: Triggers at 80% memory usage
2. **Garbage Collection**: Forced cleanup between segments
3. **DataFrame Optimization**: Downcast numeric types
4. **Chunk Processing**: Process data in smaller chunks

### Storage Optimization

1. **Compressed Caching**: Pickle with highest protocol
2. **Database Indexing**: Automatic indexes on timestamp and feature names
3. **Selective Loading**: Load only required features
4. **Automatic Cleanup**: Remove old data based on thresholds

### Processing Optimization

1. **Time Segmentation**: Process data in time-based segments
2. **Feature Caching**: Store computed features in database
3. **Checkpointing**: Save progress for resume capability
4. **Parallel Processing**: Process segments in parallel (future enhancement)

## System Requirements

### Minimum Requirements

- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **CPU**: 4 cores (8 cores recommended)
- **OS**: Linux/macOS/Windows

### Recommended Configuration

- **RAM**: 16GB or more
- **Storage**: 50GB free space (SSD recommended)
- **CPU**: 8 cores or more
- **Network**: Stable internet connection for data download

**Note**: The efficiency optimizations are designed to work within these constraints. For very large datasets (>3 years), consider using cloud resources or distributed processing.

## Troubleshooting

### Memory Issues

```python
# Check memory usage
memory_percent = efficiency_optimizer.get_memory_usage()
if memory_percent > 90:
    print("High memory usage detected")

# Force memory cleanup
efficiency_optimizer.cleanup_memory()

# Reduce chunk size
CONFIG["ENHANCED_TRAINING"]["chunk_size"] = 5000
```

### Storage Issues

```python
# Check database size
stats = efficiency_optimizer.get_database_stats()
if stats['database_size_mb'] > 1000:
    print("Large database detected")

# Clean up old data
efficiency_optimizer.cleanup_old_data(days_to_keep=365)
```

### Performance Issues

```python
# Reduce segment size for better memory management
CONFIG["ENHANCED_TRAINING"]["segment_days"] = 15

# Increase memory threshold
CONFIG["ENHANCED_TRAINING"]["memory_threshold"] = 0.9

# Disable feature caching if storage is limited
CONFIG["ENHANCED_TRAINING"]["enable_feature_caching"] = False
```

## Best Practices

### 1. Start Small

```python
# Test with smaller dataset first
CONFIG["MODEL_TRAINING"]["data_retention_days"] = 90  # 3 months
CONFIG["ENHANCED_TRAINING"]["segment_days"] = 15
```

### 2. Monitor Resources

```python
# Check system resources before starting
import psutil
memory = psutil.virtual_memory()
if memory.percent > 80:
    print("Consider closing other applications")
```

### 3. Use Checkpoints

```python
# Enable checkpointing for long runs
CONFIG["ENHANCED_TRAINING"]["enable_checkpointing"] = True

# Resume from checkpoint if interrupted
await training_manager.resume_training_from_checkpoint("ETHUSDT")
```

### 4. Clean Up Regularly

```python
# Clean up old data periodically
efficiency_optimizer.cleanup_old_data(days_to_keep=365)

# Monitor database size
stats = efficiency_optimizer.get_database_stats()
if stats['database_size_mb'] > 1000:
    print("Consider cleaning up old data")
```

## Future Enhancements

### Planned Features

1. **Parallel Processing**: Process segments in parallel
2. **Distributed Storage**: Support for external databases
3. **Incremental Training**: Train on new data only
4. **Cloud Integration**: Support for cloud storage
5. **Real-time Monitoring**: Web-based monitoring dashboard

### Advanced Optimizations

1. **GPU Acceleration**: Use GPU for feature computation
2. **Streaming Processing**: Process data as it arrives
3. **Compression**: Advanced data compression techniques
4. **Caching Strategies**: Multi-level caching with Redis
5. **Load Balancing**: Distribute processing across machines

## Conclusion

The enhanced training system provides comprehensive optimizations for handling large datasets efficiently on laptop hardware. By implementing intelligent caching, time-based segmentation, memory management, and database-backed storage, you can train models on 2+ years of data while maintaining system stability and performance.

The system is designed to be configurable and adaptable to different hardware constraints and dataset sizes. Start with the default settings and adjust based on your specific requirements and system capabilities. 