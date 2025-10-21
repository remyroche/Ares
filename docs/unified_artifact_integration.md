# Unified Artifact Management System Integration

## Overview

The Unified Artifact Management System provides a seamless integration between three core components:

1. **KlinesParquetManager** - Specialized for klines data with parquet optimization
2. **serialization_utils** - Generic serialization utilities (JSON, Pickle, Parquet)
3. **artifact_manager** - Comprehensive artifact lifecycle management

This integration creates a single, consistent interface for all artifact operations while leveraging the strengths of each component.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Unified Artifact System                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │ KlinesParquet   │  │ Serialization   │  │ Artifact    │  │
│  │ Manager         │  │ Utils           │  │ Manager     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                Enhanced BaseStep                            │
├─────────────────────────────────────────────────────────────┤
│                Step Artifact Manager                        │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Automatic Type Detection
The system automatically detects data types and routes them to the appropriate component:

- **Klines Data**: Automatically uses KlinesParquetManager for optimal storage
- **Generic Data**: Uses artifact_manager with appropriate serialization
- **Metadata**: Uses serialization_utils for JSON/Pickle storage

### 2. Unified Metadata
All components share a consistent metadata format (`UnifiedMetadata`) that includes:

- Core identification (artifact_id, type, step_name)
- Data characteristics (symbol, exchange, interval, direction, model)
- Storage information (location, size, compression)
- Timestamps and data quality metrics
- Component-specific metadata

### 3. Step-Based Workflow Integration
Enhanced BaseStep class provides:

- Automatic context setting
- Step-specific artifact management
- Performance tracking
- Error handling and recovery
- Cleanup operations

## Usage Examples

### Basic Usage

```python
from src.utils.unified_artifact_system import UnifiedArtifactSystem, UnifiedConfig

# Create unified system
config = UnifiedConfig(
    base_dir="my_artifacts",
    enable_klines_optimization=True,
    enable_compression=True,
    enable_caching=True
)

system = UnifiedArtifactSystem(config)

# Set context
system.set_context(
    step_name="data_processing",
    symbol="ETHUSDT",
    exchange="binance",
    interval="1m"
)

# Store klines data (automatically uses KlinesParquetManager)
klines_id = system.store_klines(df, "ETHUSDT", "binance", "1m")

# Store generic data (uses artifact_manager)
generic_id = system.store_artifact(data, "feature_data", "metadata")

# Load data
loaded_klines = system.load_klines("ETHUSDT", "binance", "1m")
loaded_generic = system.load_artifact("feature_data", "metadata")
```

### Enhanced BaseStep Usage

```python
from src.training.enhanced_base_step import EnhancedBaseStep

class DataProcessingStep(EnhancedBaseStep):
    async def _execute_step(self, data):
        # Store input data
        input_id = self.artifacts.store_input(data, "raw_data")
        
        # Process data
        processed_data = self.process_data(data)
        
        # Store intermediate results
        intermediate_id = self.artifacts.store_intermediate(processed_data, "features")
        
        # Store final output
        output_id = self.artifacts.store_output(processed_data, "final_results")
        
        return processed_data

# Create and use step
config = {
    'step_name': 'data_processing',
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'interval': '1m'
}

step = DataProcessingStep(config)
result = await step.execute(data)
```

### Multi-Step Workflow

```python
# Define workflow steps
steps = [
    DataCollectionStep(config1, system),
    FeatureEngineeringStep(config2, system),
    ModelTrainingStep(config3, system)
]

# Execute workflow
for step in steps:
    step.validate_config()
    result = await step.execute(None)
    print(f"Step {step.step_name} completed")
```

## Component Integration Details

### KlinesParquetManager Integration

The system automatically detects klines data based on:
- DataFrame structure (OHLCV columns)
- Context information (symbol, exchange, interval)
- Data characteristics

When klines data is detected:
- Uses specialized parquet optimization
- Applies compression and metadata management
- Maintains data integrity and quality scores

### Serialization Utils Integration

Generic data uses the universal serializer with:
- Automatic format detection
- Multiple serialization backends (JSON, Pickle, Parquet)
- Error handling and fallback mechanisms

### Artifact Manager Integration

The artifact manager provides:
- Comprehensive lifecycle management
- Caching and memory optimization
- Step-based organization
- Performance tracking

## Configuration Options

### UnifiedConfig

```python
@dataclass
class UnifiedConfig:
    base_dir: str = "unified_artifacts"
    enable_klines_optimization: bool = True
    enable_compression: bool = True
    enable_caching: bool = True
    enable_memory_optimization: bool = True
    klines_config: Optional[StorageConfig] = None
    artifact_config: Optional[Dict[str, Any]] = None
    default_serialization_format: str = "auto"
    klines_serialization_format: str = "parquet"
    enforce_metadata_consistency: bool = True
    metadata_version: str = "1.0"
```

### Step Configuration

```python
config = {
    'step_name': 'my_step',
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'interval': '1m',
    'direction': 'long',
    'model': 'Analyst'
}
```

## Performance Monitoring

The system provides comprehensive performance metrics:

```python
# Get system metrics
metrics = system.get_performance_metrics()
print(f"Total operations: {metrics['total_operations']}")
print(f"Cache hit ratio: {metrics['cache_hits'] / metrics['total_operations']}")

# Get step metrics
step_metrics = step.get_execution_summary()
print(f"Success rate: {step_metrics['success_rate']}")
print(f"Average execution time: {step_metrics['average_execution_time']}")
```

## Error Handling

The system provides robust error handling:

- Automatic retry mechanisms
- Graceful degradation
- Detailed error reporting
- Recovery procedures

```python
try:
    result = await step.execute(data)
except Exception as e:
    print(f"Step failed: {e}")
    # Handle error and potentially retry
```

## Best Practices

### 1. Context Setting
Always set context before operations:
```python
system.set_context(step_name, symbol, exchange, interval, direction, model)
```

### 2. Data Type Detection
Let the system automatically detect data types:
```python
# This will automatically use klines manager for OHLCV data
system.store_unified(df, "klines_data", symbol="ETHUSDT", exchange="binance", interval="1m")
```

### 3. Step Organization
Use step-based organization for complex workflows:
```python
# Each step manages its own artifacts
step.artifacts.store_output(data, "result")
step.artifacts.cleanup_step_artifacts()
```

### 4. Performance Optimization
Enable appropriate optimizations:
```python
config = UnifiedConfig(
    enable_compression=True,
    enable_caching=True,
    enable_memory_optimization=True
)
```

### 5. Cleanup
Always cleanup when done:
```python
system.cleanup()
step.cleanup_step()
```

## Migration Guide

### From Individual Components

1. **Replace direct KlinesParquetManager usage**:
   ```python
   # Old
   klines_manager = KlinesParquetManager()
   klines_manager.store_klines(df, symbol, exchange, interval)
   
   # New
   system = UnifiedArtifactSystem()
   system.store_klines(df, symbol, exchange, interval)
   ```

2. **Replace direct artifact_manager usage**:
   ```python
   # Old
   artifact_manager = ArtifactManager(config)
   artifact_manager.save(data, name, "data")
   
   # New
   system = UnifiedArtifactSystem()
   system.store_artifact(data, name, "data")
   ```

3. **Replace direct serialization_utils usage**:
   ```python
   # Old
   safe_serialize(data, filepath, "json")
   
   # New
   system = UnifiedArtifactSystem()
   system.store_artifact(data, name, "metadata")
   ```

### From BaseStep

1. **Inherit from EnhancedBaseStep**:
   ```python
   # Old
   class MyStep(BaseStep):
       def __init__(self, config):
           super().__init__(config)
   
   # New
   class MyStep(EnhancedBaseStep):
       def __init__(self, config, artifact_system=None):
           super().__init__(config, artifact_system)
   ```

2. **Use step artifact manager**:
   ```python
   # Store data
   self.artifacts.store_output(data, "result")
   
   # Load data
   data = self.artifacts.load_input("input_data")
   ```

## Troubleshooting

### Common Issues

1. **Context not set**: Always call `set_context()` before operations
2. **Data type detection fails**: Ensure data has proper structure for klines
3. **Performance issues**: Enable appropriate optimizations in config
4. **Memory issues**: Use memory optimization and cleanup regularly

### Debug Information

Enable debug logging:
```python
import logging
logging.getLogger("UnifiedArtifactSystem").setLevel(logging.DEBUG)
```

Get detailed metrics:
```python
metrics = system.get_performance_metrics()
print(json.dumps(metrics, indent=2))
```

## Future Enhancements

1. **Distributed Storage**: Support for cloud storage backends
2. **Advanced Caching**: Redis-based distributed caching
3. **Real-time Monitoring**: Live performance dashboards
4. **Auto-scaling**: Dynamic resource allocation
5. **Data Versioning**: Git-like versioning for artifacts

## Conclusion

The Unified Artifact Management System provides a powerful, integrated solution for managing artifacts across different data types and workflows. By combining the strengths of specialized components with a unified interface, it simplifies development while maintaining high performance and reliability.

For more examples and detailed API documentation, see the examples in `src/examples/unified_artifact_integration_examples.py`.