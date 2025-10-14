# Unified Data-Driven Feature Pipeline

A comprehensive, data-driven feature engineering pipeline that consolidates period optimization, interaction generation, and feature selection into a single, coherent system.

## 🚀 **CONSOLIDATED VERSION**

This pipeline has been **consolidated** to eliminate redundancy and provide a single, comprehensive implementation that integrates all advanced features. See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for migration instructions. This pipeline addresses key challenges in time series feature engineering while preventing leakage and overfitting.

## Key Features

### 🔒 **Leakage Prevention**
- **Purged & Embargoed Walk-Forward CV**: Implements López de Prado's methodology to prevent leakage
- **Strict Time Ordering**: Enforces no train timestamps ≥ any test timestamps
- **Configurable Embargo Windows**: Prevents information leakage between train/test sets

### 🎯 **Multi-Objective Feature Selection**
- **Explicit Objectives**: Optimizes for out-of-sample Sharpe, drawdown, turnover, stability, diversity, mutual information, and profit-centered metrics
- **Pareto Front Analysis**: Finds optimal trade-offs between competing objectives
- **Configurable Weights**: Customize objective importance based on your strategy

### 📊 **Data-Driven Approach**
- **Zero Hardcoded Heuristics**: All decisions based on statistical analysis of actual data
- **Configurable Guardrails**: Lightweight priors/constraints to prevent brittle statistical discovery
- **Adaptive Parameters**: Automatically adjust based on data characteristics

### ⚡ **High Performance**
- **VectorBT Integration**: Optimized rolling operations and matrix computations
- **Parallel Processing**: Multi-core support for large datasets
- **Memory Efficient**: Chunked processing and memory optimization
- **GPU Acceleration**: Optional GPU support for intensive computations

## Quick Start

```python
from src.training.steps.pre_training.unified_data_driven_pipeline import process_features, create_default_config

# Load your data
data = pd.read_csv('your_features.csv')
targets = pd.read_csv('your_targets.csv')['returns']

# Process features with default configuration
result = process_features(data, targets)

print(f"Selected {len(result.selected_features)} features")
print(f"Out-of-sample Sharpe: {result.out_of_sample_sharpe:.3f}")
print(f"Max drawdown: {result.max_drawdown:.3f}")
```

## Architecture

### Core Components

1. **UnifiedDataDrivenPipeline**: Main orchestrator
2. **PurgedEmbargoedWalkForwardCV**: Time series cross-validation
3. **StatisticalAnalysisFramework**: Comprehensive data analysis
4. **MultiObjectiveFeatureSelector**: Feature selection with explicit objectives
5. **VectorBTRollingOptimizer**: High-performance rolling operations

### Data Flow

```
Input Data → Data Validation → Statistical Analysis → Time Series CV → 
Period Optimization → Interaction Generation → Multi-Objective Selection → 
Final Features
```

## Configuration

### Default Configuration
```python
from src.training.steps.pre_training.unified_data_driven_pipeline import create_default_config

config = create_default_config()
```

### High Performance Configuration
```python
from src.training.steps.pre_training.unified_data_driven_pipeline import create_high_performance_config

config = create_high_performance_config()
# Enables GPU acceleration and parallel processing
```

### Memory Efficient Configuration
```python
from src.training.steps.pre_training.unified_data_driven_pipeline import create_memory_efficient_config

config = create_memory_efficient_config()
# Optimized for memory usage
```

### Custom Configuration
```python
from src.training.steps.pre_training.unified_data_driven_pipeline import UnifiedPipelineConfig

config = UnifiedPipelineConfig()
config.feature_selection.multi_objective.max_features = 30
config.feature_selection.multi_objective.objectives = {
    'out_of_sample_sharpe': 0.4,
    'drawdown': 0.3,
    'stability': 0.2,
    'diversity': 0.1
}
```

## Multi-Objective Optimization

The pipeline optimizes for multiple objectives simultaneously:

### Available Objectives

1. **Out-of-Sample Sharpe Ratio**: Risk-adjusted returns
2. **Drawdown**: Maximum drawdown (minimize)
3. **Turnover**: Feature stability (minimize)
4. **Stability**: Jaccard similarity across CV folds
5. **Diversity**: Correlation penalty or DPP
6. **Mutual Information**: Information content with targets
7. **Profit-Centered**: Profit maximization with risk penalty

### Objective Configuration
```python
config.feature_selection.multi_objective.objectives = {
    'out_of_sample_sharpe': 0.25,
    'drawdown': 0.20,
    'turnover': 0.15,
    'stability': 0.15,
    'diversity': 0.10,
    'mutual_information': 0.10,
    'profit_centered': 0.05
}
```

## Time Series Cross-Validation

### Purged & Embargoed Walk-Forward CV

```python
from src.training.steps.pre_training.unified_data_driven_pipeline import create_purged_embargoed_cv

cv = create_purged_embargoed_cv(
    n_splits=5,
    test_size=0.2,
    train_size=0.6,
    purge_fraction=0.1,  # Overlapping test periods
    embargo_fraction=0.05  # Gap between train and test
)

splits = cv.split(data, targets=targets)
```

### Leakage Prevention

The pipeline automatically validates for leakage:
- No train timestamps ≥ any test timestamps
- Embargo window enforcement
- Statistical validation of feature relationships

## Statistical Analysis

### Data Characteristics
```python
from src.training.steps.pre_training.unified_data_driven_pipeline import StatisticalAnalysisFramework

framework = StatisticalAnalysisFramework()
characteristics = framework.analyze_data_characteristics(data)

print(f"Data quality score: {characteristics.data_quality_score:.3f}")
print(f"Average correlation: {characteristics.avg_correlation:.3f}")
print(f"Stability score: {characteristics.stability_score:.3f}")
```

### Pattern Detection
```python
patterns = framework.detect_patterns(data)
print(f"Detected {len(patterns.cyclical_patterns)} cyclical patterns")
print(f"Trend strength: {patterns.trend_strength:.3f}")
```

## Guardrails and Constraints

### Configurable Guardrails
```python
config.period_optimization.guardrails.max_lookback_periods = {
    'price': 252,  # 1 year
    'volatility': 63,  # 3 months
    'momentum': 21,  # 1 month
}

config.interaction_generation.guardrails.feature_costs = {
    'price': 1.0,
    'volatility': 2.0,
    'momentum': 1.5,
}

config.feature_selection.guardrails.correlation_threshold = 0.99
config.feature_selection.guardrails.stability_threshold = 0.8
```

### Domain Sanity Checks
- Price bounds validation
- Volatility bounds validation
- Correlation threshold enforcement
- Stability score requirements

## Performance Monitoring

### Built-in Metrics
```python
pipeline = create_unified_pipeline()
result = pipeline.process(data, targets)

# Get performance statistics
stats = pipeline.get_performance_stats()
print(f"Processing time: {stats['total_processing_time']:.2f}s")
print(f"VectorBT operations: {stats['vectorbt_operations']}")
print(f"Memory usage: {stats['memory_usage_mb']:.1f}MB")
```

### Custom Monitoring
```python
# Enable profiling
config.performance.enable_profiling = True
config.performance.profile_output_dir = "profiles/"

# Enable detailed logging
config.performance.log_level = "DEBUG"
```

## Examples

### Basic Usage
```python
from src.training.steps.pre_training.unified_data_driven_pipeline import process_features

# Simple feature processing
result = process_features(data, targets)
print(f"Selected features: {result.selected_features}")
```

### Advanced Usage
```python
from src.training.steps.pre_training.unified_data_driven_pipeline import create_unified_pipeline, create_high_performance_config

# Create pipeline with custom configuration
config = create_high_performance_config()
pipeline = create_unified_pipeline(config)

# Process with custom feature columns
result = pipeline.process(data, targets, feature_columns=['price', 'volatility', 'momentum'])

# Save results
result.save_result(result, "output/")
```

### Validation Example
```python
from src.training.steps.pre_training.unified_data_driven_pipeline import validate_time_series_splits

# Validate splits for leakage
is_valid = validate_time_series_splits(splits, data)
print(f"Splits are valid: {is_valid}")
```

## Best Practices

### 1. Data Preparation
- Ensure data is time-ordered
- Handle missing values appropriately
- Validate data types and ranges

### 2. Configuration
- Start with default configuration
- Adjust objectives based on your strategy
- Use appropriate CV parameters for your data

### 3. Validation
- Always validate for leakage
- Check objective values make sense
- Monitor performance statistics

### 4. Performance
- Use high-performance config for large datasets
- Enable GPU acceleration if available
- Monitor memory usage

## Troubleshooting

### Common Issues

1. **Leakage Detected**
   - Check data time ordering
   - Verify CV split configuration
   - Ensure no future information leakage

2. **Memory Issues**
   - Use memory-efficient configuration
   - Reduce chunk size
   - Enable memory optimization

3. **Performance Issues**
   - Enable VectorBT optimization
   - Use parallel processing
   - Consider GPU acceleration

4. **No Valid Features Selected**
   - Check objective thresholds
   - Verify data quality
   - Adjust guardrail constraints

### Debug Mode
```python
config.performance.log_level = "DEBUG"
config.performance.enable_profiling = True
```

## API Reference

### Main Classes

- `UnifiedDataDrivenPipeline`: Main pipeline orchestrator
- `PurgedEmbargoedWalkForwardCV`: Time series cross-validation
- `StatisticalAnalysisFramework`: Statistical analysis
- `MultiObjectiveFeatureSelector`: Feature selection
- `UnifiedPipelineConfig`: Configuration management

### Key Functions

- `process_features()`: Simple feature processing
- `create_unified_pipeline()`: Create pipeline instance
- `create_default_config()`: Default configuration
- `validate_time_series_splits()`: Validate CV splits

## Contributing

1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Ensure backward compatibility

## License

This project is part of the Ares Trading System and follows the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the examples
3. Check the API reference
4. Create an issue with detailed information