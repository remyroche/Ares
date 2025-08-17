# Unified Data Orchestrator Integration Guide

## Overview

The **Unified Data Orchestrator** serves as the **single source of truth** for all data operations in the Ares training pipeline. It provides a centralized, unified approach to data loading, merging, resampling, and multi-timeframe operations with comprehensive quality validation and optimization.

## Key Features

### 🎯 **Single Source of Truth**
- **Centralized Interface**: One orchestrator handles all data operations
- **Intelligent Fallback**: Multiple loading strategies with automatic fallback
- **Consistent API**: Unified interface across all training steps

### 🔄 **Multi-Timeframe Support**
- **Intelligent Resampling**: Automatic upsampling from base timeframes
- **Caching**: Efficient caching of resampled data
- **Quality Preservation**: Maintains data quality during resampling

### 🛡️ **Data Quality & Validation**
- **Automatic Validation**: Comprehensive data quality checks
- **Auto-Repair**: Automatic repair of common data issues
- **Quality Metrics**: Detailed quality reporting

### ⚡ **Performance Optimization**
- **Memory Management**: Efficient memory usage and cleanup
- **Caching Strategy**: Multi-level caching for optimal performance
- **Streaming Support**: Support for large dataset processing

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Unified Data Orchestrator                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Data Sharing    │  │ Unified Data    │  │ Raw Data     │ │
│  │ Manager         │  │ Loader          │  │ Converter    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Resampling      │  │ Quality         │  │ Cache        │ │
│  │ Engine          │  │ Validator       │  │ Manager      │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Integration with Training Pipeline

### 1. **Initialize the Orchestrator**

```python
from src.training.unified_data_orchestrator import (
    get_unified_data_orchestrator,
    initialize_unified_data_orchestrator,
    cleanup_unified_data_orchestrator
)

# In your training manager initialization
async def initialize_training_manager(self):
    # Initialize the unified data orchestrator
    success = await initialize_unified_data_orchestrator(self.config)
    if not success:
        raise Exception("Failed to initialize Unified Data Orchestrator")
    
    # Get the orchestrator instance
    self.data_orchestrator = get_unified_data_orchestrator(self.config)
```

### 2. **Use in Training Steps**

Replace all existing data loading code with the orchestrator:

```python
# OLD WAY - Multiple different data loading approaches
# In step1_7_hmm_regime_discovery.py
df = await data_sharing_manager.get_unified_data(...)
# In other steps
df = await unified_data_loader.load_unified_data(...)
# In feature engineering
df = pd.read_parquet(...)

# NEW WAY - Single unified approach
# In ALL training steps
df = await self.data_orchestrator.get_unified_data(
    symbol=symbol,
    exchange=exchange,
    timeframe=timeframe,
    lookback_days=lookback_days,
    validate_quality=True,
    auto_repair=True
)
```

### 3. **Multi-Timeframe Data Loading**

```python
# Load data for multiple timeframes with intelligent resampling
multi_tf_data = await self.data_orchestrator.get_multi_timeframe_data(
    symbol="BTCUSDT",
    exchange="BINANCE",
    timeframes=["1m", "5m", "15m", "30m", "1h"],
    lookback_days=180,
    validate_quality=True,
    auto_repair=True
)

# Access data for each timeframe
data_1m = multi_tf_data["1m"]
data_5m = multi_tf_data["5m"]
data_15m = multi_tf_data["15m"]
data_30m = multi_tf_data["30m"]
data_1h = multi_tf_data["1h"]
```

## Configuration

### 1. **Orchestrator Configuration**

Add to your configuration:

```yaml
unified_data_orchestrator:
  enable_caching: true
  enable_memory_optimization: true
  enable_quality_validation: true
  enable_auto_repair: true
  
  resampling:
    default_timeframes: ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    cache_size: 100
    
  quality_validation:
    min_data_points: 1000
    max_missing_ratio: 0.1
    max_duplicate_ratio: 0.05
```

### 2. **Integration with Enhanced Training Manager**

Update your enhanced training manager:

```python
class EnhancedTrainingManager:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.data_orchestrator = None
        
    async def initialize(self) -> bool:
        # Initialize unified data orchestrator
        success = await initialize_unified_data_orchestrator(self.config)
        if not success:
            return False
        
        self.data_orchestrator = get_unified_data_orchestrator(self.config)
        return True
    
    async def cleanup(self):
        await cleanup_unified_data_orchestrator()
```

## Migration Guide

### 1. **Step-by-Step Migration**

#### Step 1: Update Imports
```python
# OLD
from src.training.steps.unified_data_loader import UnifiedDataLoader
from src.training.data_sharing_manager import DataSharingManager

# NEW
from src.training.unified_data_orchestrator import get_unified_data_orchestrator
```

#### Step 2: Replace Data Loading
```python
# OLD - Multiple approaches
if hasattr(self, 'data_sharing_manager'):
    df = await self.data_sharing_manager.get_unified_data(...)
elif hasattr(self, 'data_loader'):
    df = await self.data_loader.load_unified_data(...)
else:
    df = pd.read_parquet(...)

# NEW - Single approach
df = await self.data_orchestrator.get_unified_data(
    symbol=symbol,
    exchange=exchange,
    timeframe=timeframe,
    lookback_days=lookback_days
)
```

#### Step 3: Update Multi-Timeframe Loading
```python
# OLD - Manual resampling
df_1m = await load_data("1m")
df_5m = await resample_data(df_1m, "5m")
df_15m = await resample_data(df_1m, "15m")

# NEW - Automatic multi-timeframe loading
multi_tf_data = await self.data_orchestrator.get_multi_timeframe_data(
    symbol=symbol,
    exchange=exchange,
    timeframes=["1m", "5m", "15m"]
)
```

### 2. **Training Step Updates**

#### Update step1_7_hmm_regime_discovery.py
```python
# OLD
from src.training.data_sharing_manager import get_data_sharing_manager
data_sharing_manager = get_data_sharing_manager({})
df = await data_sharing_manager.get_unified_data(...)

# NEW
df = await self.data_orchestrator.get_unified_data(
    symbol=symbol,
    exchange=exchange,
    timeframe=tf,
    lookback_days=actual_lookback_days
)
```

#### Update feature engineering steps
```python
# OLD
from src.training.steps.unified_data_loader import UnifiedDataLoader
loader = UnifiedDataLoader({})
df = await loader.load_unified_data(...)

# NEW
df = await self.data_orchestrator.get_unified_data(
    symbol=symbol,
    exchange=exchange,
    timeframe=timeframe,
    lookback_days=lookback_days
)
```

## Advanced Usage

### 1. **Custom Quality Validation**

```python
# Load data with custom quality settings
df = await self.data_orchestrator.get_unified_data(
    symbol=symbol,
    exchange=exchange,
    timeframe=timeframe,
    validate_quality=True,
    auto_repair=True
)
```

### 2. **Force Reload**

```python
# Force reload from source (bypass cache)
df = await self.data_orchestrator.get_unified_data(
    symbol=symbol,
    exchange=exchange,
    timeframe=timeframe,
    force_reload=True
)
```

### 3. **Custom Timeframes**

```python
# Load custom timeframe combinations
multi_tf_data = await self.data_orchestrator.get_multi_timeframe_data(
    symbol=symbol,
    exchange=exchange,
    timeframes=["1m", "3m", "7m", "15m", "1h"],  # Custom timeframes
    lookback_days=90
)
```

### 4. **Monitoring and Statistics**

```python
# Get orchestrator statistics
stats = self.data_orchestrator.get_stats()
print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache misses: {stats['cache_misses']}")
print(f"Resampling operations: {stats['resampling_operations']}")
print(f"Quality repairs: {stats['quality_repairs']}")

# Get cache information
cache_info = self.data_orchestrator.get_cache_info()
print(f"Resampling cache size: {cache_info['resampling_cache_size']}")
```

## Benefits of Migration

### 1. **Consistency**
- **Single Interface**: All data operations use the same API
- **Consistent Behavior**: Same fallback strategies across all steps
- **Unified Error Handling**: Centralized error handling and logging

### 2. **Performance**
- **Intelligent Caching**: Multi-level caching for optimal performance
- **Memory Optimization**: Efficient memory management and cleanup
- **Reduced Redundancy**: Eliminates duplicate data loading code

### 3. **Quality Assurance**
- **Automatic Validation**: Built-in data quality checks
- **Auto-Repair**: Automatic repair of common data issues
- **Quality Metrics**: Comprehensive quality reporting

### 4. **Maintainability**
- **Centralized Logic**: All data logic in one place
- **Easy Updates**: Changes to data handling affect all steps
- **Better Testing**: Easier to test data operations

### 5. **Reliability**
- **Multiple Fallbacks**: Robust fallback strategies
- **Error Recovery**: Automatic error recovery and retry
- **Data Integrity**: Ensures data integrity across operations

## Troubleshooting

### 1. **Common Issues**

#### Issue: Data not loading
```python
# Check if orchestrator is initialized
if self.data_orchestrator is None:
    await self.initialize()

# Check configuration
print(self.data_orchestrator.config)
```

#### Issue: Memory problems
```python
# Check memory usage
stats = self.data_orchestrator.get_stats()
print(f"Memory cleanups: {stats['memory_cleanups']}")

# Force cleanup
self.data_orchestrator._force_garbage_collection()
```

#### Issue: Cache not working
```python
# Check cache status
cache_info = self.data_orchestrator.get_cache_info()
print(f"Cache size: {cache_info['resampling_cache_size']}")

# Clear cache if needed
self.data_orchestrator.resampling_cache.clear()
```

### 2. **Debugging**

```python
# Enable detailed logging
import logging
logging.getLogger("UnifiedDataOrchestrator").setLevel(logging.DEBUG)

# Check data quality
df = await self.data_orchestrator.get_unified_data(
    symbol=symbol,
    exchange=exchange,
    timeframe=timeframe,
    validate_quality=True,
    auto_repair=False  # Disable auto-repair to see issues
)
```

## Best Practices

### 1. **Initialization**
- Initialize the orchestrator once in your training manager
- Pass the same config instance to all components
- Handle initialization errors gracefully

### 2. **Data Loading**
- Always use the orchestrator for data loading
- Specify appropriate timeframes for your use case
- Enable quality validation for production use

### 3. **Caching**
- Monitor cache hit rates
- Adjust cache sizes based on available memory
- Clear cache periodically for long-running processes

### 4. **Error Handling**
- Always check for None returns
- Handle data quality issues appropriately
- Log data loading statistics for monitoring

### 5. **Performance**
- Use appropriate lookback periods
- Enable memory optimization for large datasets
- Monitor memory usage and cleanup frequency

## Conclusion

The Unified Data Orchestrator provides a robust, efficient, and maintainable solution for all data operations in the Ares training pipeline. By migrating to this single source of truth, you'll gain:

- **Consistency** across all training steps
- **Performance** improvements through intelligent caching
- **Quality assurance** with automatic validation and repair
- **Maintainability** through centralized data logic
- **Reliability** with robust fallback strategies

This migration will significantly improve the overall quality and reliability of your training pipeline while reducing complexity and maintenance overhead.
