# Tactician/Analyst Labeling Integration

This document describes the integration of the tactician and analyst labeling systems into the UnifiedDataDrivenPipeline, replacing the traditional triple barrier labeling approach.

## Overview

The UnifiedDataDrivenPipeline now supports three labeling systems:

1. **Tactician Labeling**: Direction/magnitude based on max favorable/adverse excursion
2. **Analyst Labeling**: "Should we trade?" based on expected PnL > fees + slippage  
3. **Triple Barrier Labeling**: Traditional method (fallback)

## Configuration

### Basic Configuration

```python
from src.training.steps.pre_training.unified_data_driven_pipeline.core.config import create_default_config
from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline import UnifiedDataDrivenPipeline

# Create configuration
config = create_default_config()

# Configure labeling system
config.labeling_system = "tactician_analyst"  # or "triple_barrier"
config.labeling_type = "analyst"  # or "tactician" (only used when labeling_system="tactician_analyst")
config.enable_labeling_optimization = True
config.labeling_quality_threshold = 0.7

# Create pipeline
pipeline = UnifiedDataDrivenPipeline(config)
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `labeling_system` | str | `"tactician_analyst"` | Labeling system to use: `"tactician_analyst"` or `"triple_barrier"` |
| `labeling_type` | str | `"analyst"` | Label type when using tactician_analyst: `"analyst"` or `"tactician"` |
| `enable_labeling_optimization` | bool | `True` | Enable labeling optimization |
| `labeling_quality_threshold` | float | `0.7` | Minimum quality threshold for labels |

## Labeling Systems

### 1. Analyst Labeling

**Purpose**: Binary labels (0/1) indicating whether a trade should be taken based on expected profitability.

**Features**:
- Multi-horizon profit labeling (15m to 150m)
- Volatility-aware target bands
- Transaction cost consideration
- Regime-specific optimization

**Configuration**:
```python
config.labeling_system = "tactician_analyst"
config.labeling_type = "analyst"
```

### 2. Tactician Labeling

**Purpose**: Direction/magnitude labels based on price excursions and entry timing.

**Features**:
- Entry timing optimization (15m timeframe)
- Local maxima/minima detection
- Enhanced entry quality scoring
- Regime-aware labeling

**Configuration**:
```python
config.labeling_system = "tactician_analyst"
config.labeling_type = "tactician"
```

### 3. Triple Barrier Labeling (Fallback)

**Purpose**: Traditional triple barrier method for backward compatibility.

**Configuration**:
```python
config.labeling_system = "triple_barrier"
```

## Implementation Details

### LabelingAdapter Class

The `LabelingAdapter` class handles the switching between different labeling systems:

```python
class LabelingAdapter:
    def __init__(self, config: UnifiedPipelineConfig):
        # Initialize based on configuration
        
    def generate_labels(self, market_data: pd.DataFrame, targets: Optional[pd.Series] = None) -> Dict[str, Any]:
        # Generate labels using configured system
```

### Integration Points

1. **Pipeline Initialization**: The labeling adapter is initialized during pipeline setup
2. **Label Generation**: Labels are generated during the data processing phase
3. **Quality Assessment**: Label quality is assessed and stored in pipeline state
4. **Metadata Storage**: Labeling metadata is preserved for analysis

## Usage Examples

### Example 1: Analyst Labeling Pipeline

```python
from src.training.steps.pre_training.unified_data_driven_pipeline.core.config import create_default_config
from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline import UnifiedDataDrivenPipeline

# Configure for analyst labeling
config = create_default_config()
config.labeling_system = "tactician_analyst"
config.labeling_type = "analyst"
config.labeling_quality_threshold = 0.8

# Create and run pipeline
pipeline = UnifiedDataDrivenPipeline(config)
result = await pipeline.run_pipeline(market_data, pipeline_state)
```

### Example 2: Tactician Labeling Pipeline

```python
# Configure for tactician labeling
config = create_default_config()
config.labeling_system = "tactician_analyst"
config.labeling_type = "tactician"
config.labeling_quality_threshold = 0.7

# Create and run pipeline
pipeline = UnifiedDataDrivenPipeline(config)
result = await pipeline.run_pipeline(market_data, pipeline_state)
```

### Example 3: Fallback to Triple Barrier

```python
# Configure for triple barrier fallback
config = create_default_config()
config.labeling_system = "triple_barrier"

# Create and run pipeline
pipeline = UnifiedDataDrivenPipeline(config)
result = await pipeline.run_pipeline(market_data, pipeline_state)
```

## Dependencies

The tactician/analyst labeling system requires the following modules:

- `src.training.steps.pre_training.profit_labeling.volatility_aware_labeler`
- `src.training.steps.pre_training.tactician_entry_labeler`
- `src.training.steps.pre_training.analyst_profit_labeler`

If these modules are not available, the pipeline will automatically fall back to triple barrier labeling.

## Error Handling

The integration includes comprehensive error handling:

1. **Import Failures**: Graceful fallback to triple barrier labeling
2. **Labeling Failures**: Error messages logged, pipeline continues
3. **Quality Issues**: Warnings logged, quality scores reported
4. **Configuration Errors**: Default values used with warnings

## Performance Considerations

- **Memory Usage**: Tactician/analyst labeling may use more memory than triple barrier
- **Processing Time**: Initial setup may take longer due to additional components
- **Quality Assessment**: Additional computation for quality scoring
- **Caching**: Labeling results are cached for performance

## Migration Guide

### From Triple Barrier to Tactician/Analyst

1. **Update Configuration**:
   ```python
   # Old
   # No specific labeling configuration needed
   
   # New
   config.labeling_system = "tactician_analyst"
   config.labeling_type = "analyst"  # or "tactician"
   ```

2. **Update Pipeline Usage**:
   ```python
   # No changes needed - same interface
   pipeline = UnifiedDataDrivenPipeline(config)
   result = await pipeline.run_pipeline(market_data, pipeline_state)
   ```

3. **Access Labeling Results**:
   ```python
   # Access labeling metadata from pipeline state
   labeling_result = pipeline_state.get('labeling_result', {})
   labeling_quality = pipeline_state.get('labeling_quality', 0.0)
   ```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all required modules are available
2. **Configuration Errors**: Check labeling_system and labeling_type values
3. **Quality Issues**: Adjust labeling_quality_threshold if needed
4. **Performance Issues**: Consider enabling caching and optimization

### Debug Information

Enable debug logging to see detailed labeling information:

```python
import logging
logging.getLogger('src.training.steps.pre_training.unified_data_driven_pipeline').setLevel(logging.DEBUG)
```

## Future Enhancements

1. **Additional Label Types**: Support for more specialized labeling approaches
2. **Dynamic Switching**: Runtime switching between labeling systems
3. **Quality Metrics**: Enhanced quality assessment and reporting
4. **Performance Optimization**: Further optimization of labeling performance

## Conclusion

The tactician/analyst labeling integration provides a flexible and powerful labeling system for the UnifiedDataDrivenPipeline, supporting both traditional and advanced labeling approaches while maintaining backward compatibility and robust error handling.