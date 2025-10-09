# Multi-Timeframe Optimization Guide

## Overview

This guide provides comprehensive instructions for using the enhanced feature lookback optimization and interactive feature generation systems across multiple timeframes (5m, 15m, 60m). The systems now automatically adapt their configuration and performance characteristics based on the target timeframe.

## 🎯 Key Features

### Timeframe-Aware Configuration
- **5m Timeframe**: Fast, high-frequency optimization with reduced parameters
- **15m Timeframe**: Balanced optimization for tactical trading
- **60m Timeframe**: Thorough optimization for strategic trading

### Automatic Parameter Adjustment
- Lookback ranges optimized for each timeframe
- CV folds and optimization time adjusted based on data frequency
- Memory and processing parameters tuned for timeframe characteristics
- Label definition types matched to trading strategies

## 📁 Directory Structure

```
src/training/steps/pre_training/
├── feature_lookback_optimization/
│   ├── timeframe_aware_optimizer.py          # Main timeframe-aware wrapper
│   ├── timeframe_config_loader.py            # Configuration management
│   ├── feature_lookback_optimization_5m_config.yaml    # 5m configuration
│   ├── feature_lookback_optimization_15m_config.yaml   # 15m configuration
│   ├── feature_lookback_optimization_60m_config.yaml   # 60m configuration
│   └── test_timeframe_optimization.py        # Comprehensive tests
└── interaction_feature_generator/
    └── feature_interaction_generation/
        └── timeframe_aware_interactive_generator.py    # Timeframe-aware generator
```

## 🚀 Quick Start

### 1. Basic Usage

```python
from src.training.steps.pre_training.feature_lookback_optimization.timeframe_aware_optimizer import (
    TimeframeAwareFeatureLookbackOptimizer
)

# Initialize optimizer
optimizer = TimeframeAwareFeatureLookbackOptimizer()

# Execute for different timeframes
pipeline_states = [
    {'symbol': 'ETHUSDT', 'exchange': 'binance', 'timeframe': '5m'},
    {'symbol': 'ETHUSDT', 'exchange': 'binance', 'timeframe': '15m'},
    {'symbol': 'ETHUSDT', 'exchange': 'binance', 'timeframe': '60m'}
]

for pipeline_state in pipeline_states:
    result = await optimizer.execute(None, pipeline_state)
    print(f"{pipeline_state['timeframe']}: {result.get('success', False)}")
```

### 2. Interactive Feature Generation

```python
from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.timeframe_aware_interactive_generator import (
    TimeframeAwareInteractiveFeatureGenerator
)

# Initialize generator
generator = TimeframeAwareInteractiveFeatureGenerator()

# Execute for different timeframes
training_input = {'data': your_data, 'targets': your_targets}

for pipeline_state in pipeline_states:
    result = await generator.execute(training_input, pipeline_state)
    print(f"{pipeline_state['timeframe']}: {result.success if hasattr(result, 'success') else 'Unknown'}")
```

## ⚙️ Configuration Details

### 5m Timeframe Configuration
- **Base Period**: 5 minutes
- **Lookback Range**: 3-40 periods (15 minutes - 3.3 hours)
- **CV Folds**: 3 (reduced for speed)
- **Max Optimization Time**: 3 minutes
- **Label Type**: Tactician (high-frequency)
- **Workers**: 8 (more parallelization)
- **Batch Size**: 2000 (larger batches)

### 15m Timeframe Configuration
- **Base Period**: 15 minutes
- **Lookback Range**: 3-40 periods (45 minutes - 10 hours)
- **CV Folds**: 3 (balanced)
- **Max Optimization Time**: 3 minutes
- **Label Type**: Tactician (tactical)
- **Workers**: 6 (balanced)
- **Batch Size**: 1500 (standard)

### 60m Timeframe Configuration
- **Base Period**: 60 minutes
- **Lookback Range**: 2-24 periods (2-24 hours)
- **CV Folds**: 5 (thorough)
- **Max Optimization Time**: 10 minutes
- **Label Type**: Analyst (strategic)
- **Workers**: 4 (thorough processing)
- **Batch Size**: 1000 (smaller batches)

## 🔧 Advanced Usage

### Custom Configuration

```python
from src.training.steps.pre_training.feature_lookback_optimization.timeframe_config_loader import (
    get_timeframe_config_loader
)

# Get configuration loader
loader = get_timeframe_config_loader()

# Get specific configuration
config = loader.get_config_for_timeframe('15m')

# Modify configuration
config['optimized_feature_lookback_optimization']['min_lookback'] = 5
config['optimized_feature_lookback_optimization']['max_lookback'] = 50

# Use modified configuration
optimizer = TimeframeAwareFeatureLookbackOptimizer()
# Configuration will be loaded from the modified config
```

### Validation and Testing

```python
from src.training.steps.pre_training.feature_lookback_optimization.test_timeframe_optimization import (
    run_comprehensive_tests
)

# Run comprehensive tests
results = await run_comprehensive_tests()

# Check results
print(f"Configuration success rate: {results['summary']['configuration_success_rate']:.1f}%")
print(f"Performance success rate: {results['summary']['performance_success_rate']:.1f}%")
```

## 📊 Performance Characteristics

### Expected Performance by Timeframe

| Timeframe | Execution Time | Memory Usage | Features Generated | Optimization Quality |
|-----------|----------------|--------------|-------------------|---------------------|
| 5m        | 1-3 minutes    | 2-4 GB       | 60-80            | High frequency      |
| 15m       | 2-5 minutes    | 3-6 GB       | 100-120          | Balanced            |
| 60m       | 5-15 minutes   | 4-8 GB       | 120-150          | Thorough            |

### Optimization Strategies

#### For 5m Timeframes
- Use aggressive caching
- Enable parallel processing
- Reduce CV folds for speed
- Focus on high-frequency features

#### For 15m Timeframes
- Balanced approach
- Standard caching
- Moderate parallel processing
- Mix of tactical and strategic features

#### For 60m Timeframes
- Thorough optimization
- Conservative caching
- Focus on quality over speed
- Strategic features

## 🧪 Testing

### Running Tests

```bash
# Run comprehensive timeframe tests
python src/training/steps/pre_training/feature_lookback_optimization/test_timeframe_optimization.py

# Run specific timeframe test
python -c "
import asyncio
from src.training.steps.pre_training.feature_lookback_optimization.test_timeframe_optimization import TimeframeOptimizationTester

async def test_15m():
    tester = TimeframeOptimizationTester()
    result = await tester.test_optimization_performance('15m')
    print(result)
    tester.cleanup()

asyncio.run(test_15m())
"
```

### Test Coverage

- Configuration validation for all timeframes
- Performance testing with synthetic data
- Memory usage validation
- Execution time benchmarking
- Error handling verification

## 🔍 Troubleshooting

### Common Issues

#### 1. Configuration Not Found
```
Error: No configuration found for timeframe: 30m
```
**Solution**: Use supported timeframes (5m, 15m, 60m) or add custom configuration.

#### 2. Memory Issues
```
Error: Memory limit exceeded
```
**Solution**: 
- For 5m: Reduce batch size or increase memory
- For 60m: Use smaller datasets or increase memory

#### 3. Slow Performance
```
Warning: Execution time exceeds threshold
```
**Solution**:
- Check hardware resources
- Enable matrix optimization
- Use appropriate timeframe configuration

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use verbose configuration
pipeline_state = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'timeframe': '15m',
    'custom_params': {
        'verbose_logging': True,
        'log_performance': True
    }
}
```

## 📈 Monitoring and Metrics

### Performance Metrics

The system tracks comprehensive metrics:

```python
# Get performance metrics
result = await optimizer.execute(training_input, pipeline_state)

if hasattr(result, 'metadata'):
    metrics = result.metadata
    print(f"Execution time: {metrics.get('execution_time', 0):.3f}s")
    print(f"Memory usage: {metrics.get('memory_usage_mb', 0):.2f} MB")
    print(f"Features optimized: {metrics.get('features_optimized', 0)}")
    print(f"Best IC: {metrics.get('best_ic', 0):.4f}")
```

### Configuration Information

```python
# Get timeframe information
info = optimizer.get_timeframe_info('15m')
print(f"Lookback range: {info['lookback_range']}")
print(f"Base period: {info['base_period_minutes']} minutes")
print(f"CV folds: {info['cv_folds']}")
```

## 🚀 Best Practices

### 1. Timeframe Selection
- **5m**: High-frequency trading, scalping strategies
- **15m**: Tactical trading, swing strategies
- **60m**: Strategic trading, position strategies

### 2. Resource Management
- Monitor memory usage during optimization
- Use appropriate batch sizes for your hardware
- Enable parallel processing for multiple symbols

### 3. Configuration Tuning
- Start with default configurations
- Adjust parameters based on your specific use case
- Test thoroughly before production deployment

### 4. Error Handling
- Always check result success status
- Implement fallback strategies
- Log errors for debugging

## 🔄 Integration with Existing Systems

### Pipeline Integration

```python
# In your pipeline
from src.training.steps.pre_training.feature_lookback_optimization.timeframe_aware_optimizer import (
    TimeframeAwareFeatureLookbackOptimizer
)

# Replace existing optimizer
optimizer = TimeframeAwareFeatureLookbackOptimizer()

# Use in pipeline
result = await optimizer.execute(training_input, pipeline_state)
```

### Component Factory Integration

```python
# Register timeframe-aware components
from src.training.steps.pre_training.components.component_factory import ComponentFactory

# Register the timeframe-aware optimizer
ComponentFactory.register_component(
    'timeframe_aware_feature_lookback_optimization',
    TimeframeAwareFeatureLookbackOptimizer
)
```

## 📚 Additional Resources

### Documentation
- `feature_lookback_optimization/README.md` - Main documentation
- `interaction_feature_generation/README.md` - Interactive features documentation
- Configuration files contain detailed parameter descriptions

### Examples
- `test_timeframe_optimization.py` - Comprehensive test examples
- `timeframe_aware_optimizer.py` - Usage examples in main function
- `timeframe_aware_interactive_generator.py` - Interactive features examples

### Support
- Check logs for detailed error messages
- Use debug mode for troubleshooting
- Run tests to validate configuration

## ✅ Conclusion

The multi-timeframe optimization system provides:

1. **Automatic Configuration**: Optimal settings for each timeframe
2. **Performance Optimization**: Tuned for different trading strategies
3. **Comprehensive Testing**: Validation across all timeframes
4. **Easy Integration**: Drop-in replacement for existing systems
5. **Production Ready**: Robust error handling and monitoring

The system is designed to work seamlessly across 5m, 15m, and 60m timeframes while maintaining the high performance and reliability of the original implementation.