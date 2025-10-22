# Enhanced Profit Labeling System

A comprehensive profit labeling system that integrates advanced tools for efficient data processing, feature generation, feature selection, and hyperparameter optimization.

## Overview

The Enhanced Profit Labeling System provides a complete pipeline for generating high-quality profit labels for machine learning models in trading applications. It integrates multiple utility modules to provide optimal performance and functionality.

## Key Features

- **Efficient Data Loading**: Uses `KlinesParquetManager` for fast data loading
- **Vectorized Computations**: Leverages `VectorBTRollingOptimizer` and `UnifiedVectorizationManager`
- **Feature Generation**: Comprehensive feature bank with multiple categories
- **Feature Selection**: Multiple methods including mRMR, LASSO, RFE, and ensemble approaches
- **Hyperparameter Optimization**: Bayesian TPE optimization with grid search
- **Hardware Optimization**: GPU acceleration and memory optimization
- **Quality Assessment**: Label quality scoring and evaluation
- **Serialization**: Efficient data persistence with multiple formats

## Architecture

```
Enhanced Profit Labeling System
├── Data Loading (KlinesParquetManager)
├── Feature Generation (FeatureBank + VectorBTRollingOptimizer)
├── Feature Selection (mRMR, LASSO, RFE, Ensemble)
├── Label Generation (ConsolidatedProfitLabeler)
├── Hyperparameter Optimization (BayesianTPEOptimizer)
├── Quality Assessment (QualityScoring)
├── Hardware Optimization (UnifiedHardwareManager)
└── Serialization (JSONSerializer, PickleSerializer)
```

## Installation

Ensure all required dependencies are installed:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.training.steps.pre_training.profit_labeling.enhanced_profit_labeling_system import (
    EnhancedProfitLabelingSystem, ProfitLabelingConfig
)

# Create configuration
config = ProfitLabelingConfig(
    symbols=["BTCUSDT"],
    timeframes=["1h"],
    max_features=100
)

# Initialize system
system = EnhancedProfitLabelingSystem(config)

# Run pipeline
results = system.run_full_pipeline()
```

### Advanced Usage

```python
# Advanced configuration with optimization
config = ProfitLabelingConfig(
    symbols=["BTCUSDT", "ETHUSDT"],
    timeframes=["1h", "4h"],
    feature_categories=["volatility", "momentum", "volume", "trend"],
    max_features=500,
    feature_selection_method="ensemble",
    enable_bayesian_optimization=True,
    n_trials=100,
    enable_gpu=False,
    enable_parallel=True
)

system = EnhancedProfitLabelingSystem(config)
results = system.run_full_pipeline()
```

## Configuration

The system supports various configuration presets located in `config/profit_labeling_configs.yaml`:

- **basic**: Simple setup for testing
- **advanced**: Full features for production
- **high_frequency**: For intraday trading
- **research**: For academic research
- **conservative**: Low risk, stable labels
- **aggressive**: High risk, high reward
- **gpu_optimized**: For GPU acceleration
- **memory_efficient**: For limited memory systems
- **multi_asset**: For portfolio-level analysis

### Loading Configuration from File

```python
import yaml

# Load configuration from file
with open('config/profit_labeling_configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)

# Use specific configuration
config_dict = configs['advanced']
config = ProfitLabelingConfig(**config_dict)
```

## Components

### 1. Data Loading

Uses `KlinesParquetManager` for efficient kline data loading:

```python
# Load data manually
data = system.load_data(symbols=["BTCUSDT"], timeframes=["1h"])
```

### 2. Feature Generation

Generates features using the feature bank and VectorBT optimization:

```python
# Generate features
features = system.generate_features(data)
```

### 3. Feature Selection

Selects optimal features using various methods:

```python
# Select features
selected_features = system.select_features(features, labels)
```

### 4. Label Generation

Generates profit labels using the consolidated labeler:

```python
# Generate labels
labels = system.generate_labels(data)
```

### 5. Hyperparameter Optimization

Optimizes hyperparameters using Bayesian TPE:

```python
# Optimize hyperparameters
optimization_results = system.optimize_hyperparameters(features, labels)
```

### 6. Quality Evaluation

Evaluates label quality and performance:

```python
# Evaluate labels
evaluation_results = system.evaluate_labels(features, labels)
```

## Examples

See `examples/usage_examples.py` for comprehensive usage examples:

- Basic usage
- Advanced usage with optimization
- Custom data loading
- Feature selection comparison
- Hyperparameter optimization
- Quality evaluation
- Batch processing
- Custom feature categories
- Performance monitoring

## Performance Optimization

### Hardware Optimization

The system automatically detects and uses available hardware optimizations:

- **CPU Optimization**: Multi-core processing with `EnhancedCPUOptimizer`
- **GPU Acceleration**: CUDA support when available
- **Memory Management**: Efficient memory usage with `UnifiedHardwareManager`

### Parallel Processing

Enable parallel processing for faster execution:

```python
config = ProfitLabelingConfig(
    enable_parallel=True,
    n_jobs=8  # Number of parallel jobs
)
```

### Memory Efficiency

For systems with limited memory:

```python
config = ProfitLabelingConfig(
    memory_efficient=True,
    max_features=100  # Limit feature count
)
```

## Quality Metrics

The system provides comprehensive quality assessment:

- **Label Balance**: Measures class distribution balance
- **Label Stability**: Measures temporal stability
- **Signal-to-Noise Ratio**: Measures label quality
- **Leakage Detection**: Prevents data leakage
- **Noise Gating**: Filters microstructure noise

## Output Format

The system generates structured results:

```python
{
    'config': {...},           # Configuration used
    'timestamp': '...',        # Execution timestamp
    'data': {...},            # Data loading results
    'features': {...},        # Feature generation results
    'labels': {...},          # Label generation results
    'selected_features': {...}, # Feature selection results
    'evaluation': {...},      # Quality evaluation results
    'optimization': {...}     # Hyperparameter optimization results
}
```

## Error Handling

The system includes comprehensive error handling:

- **Graceful Degradation**: Falls back to basic methods when advanced tools unavailable
- **Detailed Logging**: Uses `tprint` for enhanced logging
- **Exception Handling**: Catches and reports errors appropriately
- **Resource Cleanup**: Proper memory management and cleanup

## Dependencies

### Required
- numpy
- pandas
- scikit-learn
- optuna (for Bayesian optimization)
- vectorbt (for vectorized operations)

### Optional
- torch (for GPU acceleration)
- cupy (for CUDA support)
- shap (for explainability)
- lime (for explainability)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Use `memory_efficient=True` and reduce `max_features`
3. **Performance Issues**: Enable parallel processing and hardware optimization
4. **Data Loading Issues**: Check data path and file permissions

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When contributing to the profit labeling system:

1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Ensure backward compatibility
5. Use type hints and docstrings

## License

This project is part of the Ares trading system and follows the same license terms.

## Support

For support and questions:

1. Check the examples in `examples/usage_examples.py`
2. Review the configuration options in `config/profit_labeling_configs.yaml`
3. Examine the test files for usage patterns
4. Check the logs for detailed error information