# Steps 1-7 Compatibility Framework Guide

## Overview

The Steps 1-7 Compatibility Framework provides comprehensive compatibility management between all pipeline steps, ensuring data consistency, schema validation, and proper error handling across the entire data processing pipeline.

## Architecture

### Core Components

1. **Step Contract Validation** - Defines input/output contracts for each step
2. **Data Schema Validation** - Ensures data consistency across steps
3. **Cross-Step Consistency Checks** - Validates data flow between steps
4. **Configuration Compatibility** - Ensures configurations are compatible across steps
5. **Dependency Management** - Manages step dependencies and execution order
6. **Error Propagation** - Handles errors consistently across the pipeline

### Step Definitions

#### Step 1: Data Collection
- **Purpose**: Collect raw data from exchanges
- **Inputs**: Configuration, symbol, exchange, timeframe
- **Outputs**: Klines data, aggtrades data, data paths, metadata
- **Schema**: `klines`, `aggtrades`

#### Step 1.5: Data Converter
- **Purpose**: Convert and unify data formats
- **Inputs**: Klines data, aggtrades data, configuration
- **Outputs**: Unified data, conversion metadata
- **Schema**: `unified`

#### Step 2: Data Reading
- **Purpose**: Read and validate unified data
- **Inputs**: Unified data, configuration
- **Outputs**: Validated data, validation report, quality metrics
- **Schema**: `unified`

#### Step 3: HMM Regime Discovery
- **Purpose**: Discover market regimes using Hidden Markov Models
- **Inputs**: Validated data, configuration
- **Outputs**: Regime labels, HMM model, regime metadata
- **Schema**: `regime_labels`

#### Step 4: Regime Data Splitting
- **Purpose**: Split data by discovered regimes
- **Inputs**: Validated data, regime labels, configuration
- **Outputs**: Regime datasets, splitting metadata
- **Schema**: Multiple regime-specific datasets

#### Step 5: Labeling
- **Purpose**: Apply labels to regime-split data
- **Inputs**: Regime datasets, configuration
- **Outputs**: Labeled datasets, labeling metadata
- **Schema**: Labeled datasets with regime and label information

#### Step 6: Feature Engineering
- **Purpose**: Create features for machine learning
- **Inputs**: Labeled datasets, configuration
- **Outputs**: Feature datasets, feature metadata, feature importance
- **Schema**: Feature-enhanced datasets

#### Step 7: Enhanced Matrix Operations
- **Purpose**: Perform advanced matrix operations and optimizations
- **Inputs**: Feature datasets, configuration
- **Outputs**: Processed datasets, processing metadata, matrix operations
- **Schema**: Final processed datasets

## Data Schemas

### Klines Schema
```yaml
required_columns:
  - timestamp (int64)
  - open (float64)
  - high (float64)
  - low (float64)
  - close (float64)
  - volume (float64)

optional_columns:
  - quote_asset_volume
  - number_of_trades
  - taker_buy_base_asset_volume
  - taker_buy_quote_asset_volume
```

### Aggtrades Schema
```yaml
required_columns:
  - timestamp (int64)
  - price (float64)
  - quantity (float64)

optional_columns:
  - first_trade_id
  - last_trade_id
  - trade_time
  - is_buyer_maker
```

### Unified Schema
```yaml
required_columns:
  - timestamp (int64)
  - open (float64)
  - high (float64)
  - low (float64)
  - close (float64)
  - volume (float64)
  - price (float64)
  - quantity (float64)

optional_columns:
  - regime
  - label
  - split
```

### Regime Labels Schema
```yaml
required_columns:
  - timestamp (int64)
  - regime (int64)

optional_columns:
  - regime_probability
  - regime_confidence
```

## Usage Examples

### Basic Compatibility Check
```python
from src.utils.steps_1_7_compatibility_framework import steps_1_7_compatibility

# Validate step contract
is_valid = steps_1_7_compatibility.validate_step_contract(
    "step1_data_collection",
    inputs={"config": {}, "symbol": "BTCUSDT", "exchange": "binance", "timeframe": "1m"},
    outputs={"klines_data": df, "aggtrades_data": df, "data_paths": {}, "metadata": {}}
)
```

### Cross-Step Consistency Validation
```python
# Validate data consistency across multiple steps
step_data = {
    "step1": klines_df,
    "step2": validated_df,
    "step3": regime_df
}

is_consistent = steps_1_7_compatibility.validate_cross_step_consistency(
    step_data, ["step1", "step2", "step3"]
)
```

### Configuration Compatibility
```python
# Validate configuration compatibility
configs = {
    "step1": {"symbol": "BTCUSDT", "exchange": "binance"},
    "step2": {"symbol": "BTCUSDT", "exchange": "binance"}
}

is_compatible = steps_1_7_compatibility.validate_configuration_compatibility(configs)
```

## Error Handling

### Error Categories
- **VALIDATION**: Data validation errors
- **DEPENDENCY**: Missing dependencies
- **RESOURCE**: Resource constraints
- **CONFIGURATION**: Configuration errors
- **EXECUTION**: Runtime execution errors

### Error Propagation
Errors are automatically categorized and propagated with context:
- Step name where error occurred
- Error type and severity
- Context information
- Suggested recovery actions

## Performance Monitoring

### Quality Metrics
- Data completeness scores
- Schema validation results
- Cross-step consistency metrics
- Configuration compatibility scores
- Error rates and patterns

### Memory Management
- Data lineage tracking
- Memory usage optimization
- Garbage collection coordination
- Resource cleanup

## Testing

### Test Coverage
The framework includes comprehensive tests for:
- Step contract validation
- Data schema validation
- Cross-step consistency
- Configuration compatibility
- Dependency management
- Error propagation
- Performance metrics
- Data quality flow
- Memory management
- Concurrent execution

### Running Tests
```bash
python test_steps_1_7_compatibility.py
```

## Best Practices

### Data Validation
1. Always validate data schemas before processing
2. Check data consistency across steps
3. Monitor data quality metrics
4. Implement proper error handling

### Configuration Management
1. Use consistent configuration across steps
2. Validate configuration compatibility
3. Document configuration requirements
4. Implement configuration validation

### Error Handling
1. Categorize errors appropriately
2. Provide meaningful error messages
3. Implement recovery mechanisms
4. Log errors with context

### Performance
1. Monitor memory usage
2. Track data lineage
3. Optimize data processing
4. Implement caching where appropriate

## Troubleshooting

### Common Issues

#### Schema Validation Failures
- Check required columns are present
- Verify data types match expectations
- Ensure optional columns are properly handled

#### Cross-Step Inconsistencies
- Verify data sources are consistent
- Check timestamp alignment
- Ensure proper data filtering

#### Configuration Conflicts
- Validate common parameters
- Check for conflicting values
- Ensure proper parameter inheritance

### Debug Mode
Enable debug logging for detailed compatibility information:
```python
import logging
logging.getLogger("Steps1_7Compatibility").setLevel(logging.DEBUG)
```

## Integration

### With Existing Pipeline
The framework integrates seamlessly with existing pipeline components:
- Pipeline standards
- Error handling system
- Logging framework
- Model management
- Data validation

### Custom Extensions
Extend the framework for custom requirements:
- Additional step types
- Custom data schemas
- Specialized validation rules
- Enhanced error handling

## Security Considerations

### Data Privacy
- Sensitive data handling
- Access control
- Audit logging
- Encryption requirements

### Configuration Security
- Secure credential storage
- Environment variable usage
- Configuration encryption
- Access restrictions

## Monitoring and Alerting

### Health Checks
- Framework status monitoring
- Performance metrics tracking
- Error rate monitoring
- Resource usage tracking

### Alerts
- Configuration conflicts
- Schema validation failures
- Cross-step inconsistencies
- Performance degradation

## Future Enhancements

### Planned Features
- Real-time compatibility monitoring
- Automated issue resolution
- Enhanced performance optimization
- Advanced error recovery
- Machine learning-based validation

### Extension Points
- Custom validation rules
- Additional data schemas
- Enhanced error handling
- Performance optimizations
- Security enhancements

## Support

### Documentation
- API reference
- Examples and tutorials
- Troubleshooting guides
- Best practices

### Community
- Issue reporting
- Feature requests
- Contributions
- Discussion forums

## Conclusion

The Steps 1-7 Compatibility Framework provides a robust foundation for ensuring data consistency and proper error handling across the entire pipeline. By following the guidelines and best practices outlined in this document, you can build reliable, maintainable, and scalable data processing pipelines.