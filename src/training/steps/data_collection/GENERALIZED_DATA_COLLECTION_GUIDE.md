# Generalized Data Collection Framework

## Overview

This guide explains how to use the generalized data collection framework that leverages all the comprehensive tools available in BaseStep. The framework provides a standardized approach to data collection operations with:

- Complete BaseStep integration with all comprehensive utilities
- Hardware optimization and memory management
- Advanced logging with tprint integration
- Data quality validation and cleaning
- Model persistence and caching
- ML common utilities integration
- Comprehensive error handling and validation

## Architecture

The generalized data collection framework consists of three main components:

### 1. Enhanced Generalized Data Collector (`enhanced_generalized_data_collector.py`)

The main data collector that inherits from BaseStep and provides comprehensive data collection capabilities:

```python
from src.training.steps.data_collection.enhanced_generalized_data_collector import EnhancedGeneralizedDataCollector

# Create collector
collector = EnhancedGeneralizedDataCollector("my_collection", config)

# Execute collection
result = await collector.execute(config)
```

**Features:**
- Complete BaseStep integration
- Hardware optimization
- Advanced logging
- Data quality validation
- Performance monitoring
- Comprehensive error handling

### 2. Generalized Data Collection Utilities (`generalized_data_collection_utils.py`)

Common utilities and patterns for data collection operations:

```python
from src.training.steps.data_collection.generalized_data_collection_utils import (
    create_standard_collection_config,
    validate_collection_config,
    validate_klines_data,
    validate_data_quality,
    detect_gaps,
    analyze_gap_patterns,
    generate_filename,
    find_latest_file,
    create_performance_tracker
)
```

**Features:**
- Standardized configuration management
- Data validation utilities
- Gap detection and analysis
- File operations utilities
- Performance monitoring utilities

### 3. Refactored Processing Pipeline (`refactored_klines_processing_pipeline.py`)

Example of how to refactor existing data collection steps to use the generalized tools:

```python
from src.training.steps.data_collection.refactored_klines_processing_pipeline import RefactoredKlinesProcessingPipeline

# Create pipeline
pipeline = RefactoredKlinesProcessingPipeline("klines_processing", config)

# Execute processing
result = await pipeline.execute(config)
```

## Usage Examples

### Basic Data Collection

```python
import asyncio
from datetime import datetime, timedelta
from src.training.steps.data_collection.enhanced_generalized_data_collector import (
    collect_data_incremental,
    collect_data_for_period,
    detect_and_fill_gaps
)

async def main():
    # Incremental data collection
    result = await collect_data_incremental(
        exchange="BINANCE",
        symbol="ETHUSDT",
        timeframe="1m",
        data_types=["klines"],
        max_batches=10
    )
    
    # Period data collection
    start_time = datetime.now() - timedelta(days=7)
    end_time = datetime.now()
    
    result = await collect_data_for_period(
        exchange="BINANCE",
        symbol="ETHUSDT",
        timeframe="1m",
        start_time=start_time,
        end_time=end_time
    )
    
    # Gap detection and filling
    gap_result = await detect_and_fill_gaps(
        exchange="BINANCE",
        symbol="ETHUSDT",
        timeframe="1m"
    )

asyncio.run(main())
```

### Custom Data Collection Step

```python
from src.training.steps.base_step import BaseStep
from src.training.steps.data_collection.generalized_data_collection_utils import (
    create_standard_collection_config,
    validate_collection_config,
    validate_klines_data,
    validate_data_quality
)

class MyCustomDataCollector(BaseStep):
    def __init__(self, step_name: str = "my_custom_collector", config: Optional[Dict[str, Any]] = None):
        super().__init__(step_name, config)
        
        # Create standardized configuration
        self.collection_config = create_standard_collection_config(
            exchange=config.get('exchange', 'BINANCE'),
            symbol=config.get('symbol', 'ETHUSDT'),
            timeframe=config.get('timeframe', '1m'),
            **config
        )
        
        # Validate configuration
        is_valid, error_message = validate_collection_config(self.collection_config)
        if not is_valid:
            raise ValueError(f"Invalid configuration: {error_message}")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Use comprehensive BaseStep tools
        self.tprint_step_start("My Custom Data Collection")
        
        # Your custom logic here
        # Access all BaseStep comprehensive utilities:
        # - self.hardware_utils
        # - self.data_quality
        # - self.ml_common
        # - self.math_validation
        # - self.core_decorators
        # - All tprint utilities
        # - All convenience methods
        
        self.tprint_step_end("My Custom Data Collection")
        
        return {'success': True, 'artifacts': ['custom_data']}
```

### Using Generalized Utilities

```python
from src.training.steps.data_collection.generalized_data_collection_utils import (
    validate_klines_data,
    validate_data_quality,
    detect_gaps,
    analyze_gap_patterns,
    create_performance_tracker,
    track_operation
)

# Validate klines data
is_valid, errors = validate_klines_data(data)
if not is_valid:
    print(f"Validation errors: {errors}")

# Validate data quality
df = pd.DataFrame(data)
quality_result = validate_data_quality(df, "klines")
print(f"Quality score: {quality_result['quality_score']}")

# Detect gaps
gaps = detect_gaps(df, "klines")
gap_analysis = analyze_gap_patterns(gaps)
print(f"Found {gap_analysis['total_gaps']} gaps")

# Performance tracking
tracker = create_performance_tracker()
track_operation(tracker, "my_operation", start_time, end_time, success=True)
final_tracker = finalize_performance_tracker(tracker)
```

## Configuration

### Standard Collection Configuration

The framework uses a standardized configuration format:

```python
config = {
    'exchange': 'BINANCE',           # Exchange name
    'symbol': 'ETHUSDT',             # Trading symbol
    'timeframe': '1m',               # Data timeframe
    'data_dir': 'historical_data',   # Data directory
    'collection_mode': 'incremental', # Collection mode
    'data_types': ['klines'],        # Data types to collect
    'max_batches': 10,               # Maximum batches for incremental
    'batch_size': 1000,              # Batch size
    'start_time': datetime,          # Start time for period mode
    'end_time': datetime,            # End time for period mode
    'information': 'klines',         # Information type
    'direction': 'long',             # Direction
    'model': 'Analyst'               # Model type
}
```

### Collection Modes

1. **Incremental**: Collect data incrementally from the last timestamp
2. **Period**: Collect data for a specific time period
3. **Gap Filling**: Detect and fill gaps in existing data

## BaseStep Integration

The generalized data collection framework fully leverages all BaseStep comprehensive tools:

### Hardware Optimization

```python
# Access hardware utilities
if self.hardware_utils:
    optimized_df = self.hardware_utils['optimize_dataframe'](df)
```

### Data Quality Tools

```python
# Access data quality utilities
if self.data_quality:
    cleaner = self._get_data_cleaner()
    if cleaner:
        cleaned_df = cleaner.clean(df)
```

### ML Common Utilities

```python
# Access ML utilities
if self.ml_common:
    optimizer = self._get_ml_optimizer("bayesian")
    cv_validator = self._get_cv_validator("time_series")
```

### Advanced Logging

```python
# Use comprehensive tprint utilities
self.tprint_step_start("Data Collection")
self.tprint_info("Processing data...")
self.tprint_success("Data processed successfully")
self.tprint_performance_summary(metrics)
self.tprint_memory_usage()
self.tprint_hardware_stats()
self.tprint_step_end("Data Collection")
```

### Convenience Methods

```python
# Use BaseStep convenience methods
data = self._safe_json_load("config.json")
result = self._safe_divide(10, 2, default=0)
self._ensure_directory("/path/to/dir")
valid = self._validate_dataframe_columns(df, ["col1", "col2"])
cleaned = self._safe_dataframe_operation(df, "fillna")
```

## Migration Guide

### For Existing Data Collection Steps

1. **Inherit from BaseStep**: Change your class to inherit from `BaseStep`
2. **Use Generalized Utilities**: Replace custom utilities with generalized ones
3. **Leverage BaseStep Tools**: Use comprehensive BaseStep utilities
4. **Standardize Configuration**: Use the standardized configuration format
5. **Add Performance Tracking**: Use the performance tracking utilities

### Example Migration

**Before:**
```python
class MyDataCollector:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def collect_data(self):
        # Custom implementation
        pass
```

**After:**
```python
from src.training.steps.base_step import BaseStep
from src.training.steps.data_collection.generalized_data_collection_utils import (
    create_standard_collection_config,
    validate_collection_config
)

class MyDataCollector(BaseStep):
    def __init__(self, step_name: str = "my_data_collector", config: Optional[Dict[str, Any]] = None):
        super().__init__(step_name, config)
        
        # Use standardized configuration
        self.collection_config = create_standard_collection_config(**config)
        is_valid, error = validate_collection_config(self.collection_config)
        if not is_valid:
            raise ValueError(f"Invalid configuration: {error}")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Use comprehensive BaseStep tools
        self.tprint_step_start("Data Collection")
        
        # Your implementation with BaseStep utilities
        
        self.tprint_step_end("Data Collection")
        return {'success': True, 'artifacts': ['data']}
```

## Best Practices

### 1. Use Standardized Configuration

Always use the standardized configuration format and validation:

```python
config = create_standard_collection_config(**user_config)
is_valid, error = validate_collection_config(config)
if not is_valid:
    raise ValueError(f"Invalid configuration: {error}")
```

### 2. Leverage BaseStep Tools

Use all available BaseStep comprehensive tools:

```python
# Hardware optimization
if self.hardware_utils:
    optimized_data = self.hardware_utils['optimize_dataframe'](data)

# Data quality
if self.data_quality:
    quality_result = self._get_data_quality_assessment(data)

# ML utilities
if self.ml_common:
    optimizer = self._get_ml_optimizer("bayesian")
```

### 3. Use Comprehensive Logging

Use tprint utilities for consistent logging:

```python
self.tprint_step_start("Operation")
self.tprint_info("Processing...")
self.tprint_success("Completed")
self.tprint_performance_summary(metrics)
self.tprint_step_end("Operation")
```

### 4. Implement Performance Tracking

Track performance metrics:

```python
tracker = create_performance_tracker()
track_operation(tracker, "operation", start_time, end_time, success=True)
final_tracker = finalize_performance_tracker(tracker)
```

### 5. Validate Data Quality

Always validate data quality:

```python
is_valid, errors = validate_klines_data(data)
if not is_valid:
    self.tprint_warning(f"Data validation failed: {errors}")

quality_result = validate_data_quality(df, "klines")
if quality_result['quality_score'] < 80:
    self.tprint_warning("Data quality is poor")
```

## Error Handling

The framework provides comprehensive error handling:

```python
try:
    # Your data collection logic
    result = await collect_data()
except Exception as e:
    self.tprint_error(f"Data collection failed: {e}")
    self.tprint_exception(e)
    return {'success': False, 'error': str(e)}
```

## Performance Optimization

The framework includes several performance optimization features:

1. **Hardware Optimization**: Automatic hardware optimization for M1 chips
2. **Memory Management**: Advanced memory management and cleanup
3. **Batch Processing**: Optimized batch processing
4. **Caching**: Smart caching for frequently accessed data
5. **Lazy Loading**: Lazy loading for large datasets

## Monitoring and Metrics

The framework provides comprehensive monitoring:

1. **Performance Metrics**: Track operation performance
2. **Memory Usage**: Monitor memory usage
3. **Quality Scores**: Track data quality scores
4. **Error Tracking**: Track errors and warnings
5. **Hardware Stats**: Monitor hardware utilization

## Conclusion

The generalized data collection framework provides a comprehensive, standardized approach to data collection operations that leverages all BaseStep comprehensive tools. By using this framework, you can:

- Reduce code duplication
- Improve consistency across data collection steps
- Leverage comprehensive BaseStep utilities
- Implement best practices automatically
- Monitor performance and quality
- Handle errors gracefully

For more examples and detailed documentation, see the individual module files and the BaseStep enhancement summary.