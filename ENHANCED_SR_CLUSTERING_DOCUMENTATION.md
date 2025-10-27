# Enhanced SR Clustering Module Documentation

## Overview

The Enhanced SR Clustering module is a comprehensive, production-ready implementation for detecting and clustering Support/Resistance (SR) levels in financial time series data. It integrates advanced machine learning techniques, hardware optimization, and robust validation methods to provide highly accurate and efficient SR level detection.

## Key Features

### 🚀 Performance Optimizations
- **VectorBTRollingOptimizer**: High-performance rolling window operations using VectorBT
- **UnifiedVectorizationManager**: Optimized matrix operations and vectorization
- **Hardware Optimization**: M1 chip optimizations for CPU, GPU, and memory
- **Adaptive Optimization**: Dynamic workload optimization based on hardware capabilities

### 🧠 Advanced Machine Learning
- **Multiple Clustering Algorithms**: DBSCAN, HDBSCAN, Spectral, Agglomerative, OPTICS
- **Hyperparameter Optimization**: Bayesian TPE, Hierarchical HPO, Regime-Specific HPO
- **Feature Engineering**: Comprehensive price, volume, time, and technical indicators
- **Dimensionality Reduction**: PCA, ICA, t-SNE, UMAP support

### 🔍 Data Quality & Validation
- **Data Leakage Detection**: Temporal leakage and lookahead bias prevention
- **Purged Cross-Validation**: Time series-specific validation techniques
- **Out-of-Fold/Out-of-Sample**: Robust validation strategies
- **Backtesting Integration**: Real-time performance validation

### 📊 Explainability & Analysis
- **SHAP Integration**: Model interpretability and feature importance
- **LIME Integration**: Local interpretable model explanations
- **Feature Importance**: Comprehensive feature analysis
- **Performance Metrics**: Detailed clustering quality assessment

## Architecture

```
EnhancedSRClustering
├── VectorBTRollingOptimizer (Performance)
├── UnifiedVectorizationManager (Matrix Operations)
├── HPO Components (Optimization)
│   ├── BayesianTPEOptimizer
│   ├── HierarchicalHPO
│   └── RegimeSpecificHPO
├── Hardware Optimization
│   ├── UnifiedHardwareManager
│   ├── AdaptiveOptimizationEngine
│   ├── M1MemoryOptimizer
│   ├── M1CPUOptimizer
│   └── M1GPUManager
├── ML Utilities
│   ├── SHAPLIMEIntegration
│   ├── DataLeakageDetector
│   ├── UnifiedCrossValidation
│   └── TemporalValidation
└── SRBacktestingEngine (Validation)
```

## Configuration

### EnhancedSRClusteringConfig

The configuration system provides comprehensive control over all aspects of the clustering process:

```python
config = EnhancedSRClusteringConfig(
    # Core clustering parameters
    clustering_algorithm=ClusteringAlgorithm.HDBSCAN,
    min_cluster_size=5,
    
    # DBSCAN parameters
    dbscan_eps=0.05,
    dbscan_min_samples=3,
    
    # HDBSCAN parameters
    hdbscan_min_cluster_size=5,
    hdbscan_min_samples=3,
    hdbscan_cluster_selection_epsilon=0.05,
    
    # Feature engineering
    feature_engineering_config={
        'price_features': True,
        'volume_features': True,
        'time_features': True,
        'technical_indicators': True,
        'microstructure_features': True,
        'feature_normalization': 'standard',
        'dimensionality_reduction': 'pca',
        'n_components': 0.8
    },
    
    # Hyperparameter optimization
    hpo_config={
        'optimization_strategy': OptimizationStrategy.BAYESIAN_TPE,
        'n_trials': 50,
        'timeout': 300
    },
    
    # Backtesting validation
    backtesting_config={
        'enabled': True,
        'initial_capital': 10000,
        'commission': 0.001
    },
    
    # Explainability
    explainability_config={
        'shap_enabled': True,
        'lime_enabled': True
    }
)
```

## Usage Examples

### Basic Usage

```python
import asyncio
import pandas as pd
from src.utils.sr_clustering import create_enhanced_sr_clustering

# Create sample data
price_data = pd.DataFrame({
    'open': [100, 101, 102, 101, 100],
    'high': [101, 102, 103, 102, 101],
    'low': [99, 100, 101, 100, 99],
    'close': [100, 101, 102, 101, 100],
    'volume': [1000, 1200, 1100, 1300, 900]
}, index=pd.date_range('2023-01-01', periods=5, freq='1H'))

# Create clustering instance
clustering = create_enhanced_sr_clustering()

# Run clustering
async def main():
    results = await clustering.cluster_sr_levels(price_data)
    print(f"Found {len(results)} clusters")
    for result in results:
        print(f"Cluster: Price={result.centroid_price:.2f}, Quality={result.cluster_quality:.4f}")

asyncio.run(main())
```

### Advanced Usage with Custom Configuration

```python
from src.utils.sr_clustering import (
    EnhancedSRClusteringConfig,
    ClusteringAlgorithm,
    OptimizationStrategy
)

# Create advanced configuration
config = EnhancedSRClusteringConfig(
    clustering_algorithm=ClusteringAlgorithm.HDBSCAN,
    hpo_config={
        'optimization_strategy': OptimizationStrategy.BAYESIAN_TPE,
        'n_trials': 100,
        'timeout': 600
    },
    feature_engineering_config={
        'price_features': True,
        'volume_features': True,
        'time_features': True,
        'technical_indicators': True,
        'microstructure_features': True,
        'feature_normalization': 'robust',
        'dimensionality_reduction': 'umap',
        'n_components': 0.7
    },
    backtesting_config={
        'enabled': True,
        'initial_capital': 50000,
        'commission': 0.0005
    },
    explainability_config={
        'shap_enabled': True,
        'lime_enabled': True
    }
)

# Create clustering instance
clustering = create_enhanced_sr_clustering(config)

# Run clustering
results = await clustering.cluster_sr_levels(price_data)
```

## Clustering Algorithms

### Supported Algorithms

1. **DBSCAN**: Density-based clustering with noise detection
2. **HDBSCAN**: Hierarchical DBSCAN with better parameter selection
3. **Spectral**: Graph-based clustering using spectral analysis
4. **Agglomerative**: Hierarchical clustering with various linkage methods
5. **OPTICS**: Ordering points to identify clustering structure
6. **Hybrid**: Combination of multiple algorithms
7. **Adaptive**: Algorithm selection based on data characteristics

### Algorithm Selection Guidelines

- **DBSCAN**: Good for datasets with clear density clusters and noise
- **HDBSCAN**: Best for datasets with varying cluster densities
- **Spectral**: Effective for non-convex clusters and complex shapes
- **Agglomerative**: Good for hierarchical cluster relationships
- **OPTICS**: Useful for datasets with varying cluster densities
- **Hybrid**: Best for complex datasets requiring multiple approaches
- **Adaptive**: Automatic selection based on data characteristics

## Feature Engineering

### Price Features
- Rolling price statistics (mean, std, min, max)
- OHLC relationships and patterns
- Price momentum and acceleration
- Volatility measures

### Volume Features
- Rolling volume statistics
- Volume-price relationships
- Volume momentum and patterns
- Volume-weighted average price (VWAP)

### Time Features
- Hour of day, day of week, month
- Seasonal patterns and cycles
- Time-based volatility patterns
- Market session indicators

### Technical Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving averages (SMA, EMA)
- Stochastic oscillators

### Market Microstructure Features
- Price impact measures
- Bid-ask spread proxies
- Order flow imbalance
- Market depth indicators

## Hyperparameter Optimization

### Optimization Strategies

1. **Bayesian TPE**: Tree-structured Parzen Estimator for efficient search
2. **Hierarchical HPO**: Multi-level optimization for complex parameter spaces
3. **Regime-Specific HPO**: Different parameters for different market regimes
4. **Adaptive HPO**: Dynamic strategy selection based on data characteristics

### Parameter Spaces

The optimization system automatically defines parameter spaces for each clustering algorithm:

```python
# DBSCAN parameters
param_space = {
    'eps': (0.01, 0.1),
    'min_samples': (3, 20)
}

# HDBSCAN parameters
param_space = {
    'min_cluster_size': (5, 50),
    'min_samples': (3, 20),
    'cluster_selection_epsilon': (0.0, 0.1)
}

# Spectral parameters
param_space = {
    'n_clusters': (2, 20),
    'affinity': ['rbf', 'nearest_neighbors'],
    'gamma': (0.1, 2.0)
}
```

## Validation & Backtesting

### Validation Methods

1. **Purged Cross-Validation**: Time series-specific validation with embargo periods
2. **Out-of-Fold Validation**: Robust cross-validation for time series
3. **Out-of-Sample Validation**: True out-of-sample testing
4. **Temporal Validation**: Time-based validation strategies

### Backtesting Integration

The module integrates with the SRBacktestingEngine for real-time performance validation:

```python
# Backtesting configuration
backtest_config = {
    'start_date': result.first_touch,
    'end_date': result.last_touch,
    'initial_capital': 10000,
    'commission': 0.001
}

# Run backtest
backtest_results = await backtesting_engine.run_backtest(
    price_data=price_data,
    sr_levels=[result.centroid_price],
    config=backtest_config
)
```

### Performance Metrics

- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Total Return**: Overall portfolio return
- **Calmar Ratio**: Return to maximum drawdown ratio

## Explainability & Analysis

### SHAP Integration

SHAP (SHapley Additive exPlanations) provides model interpretability:

```python
# SHAP analysis
shap_values = await shap_lime_integration.calculate_shap_values(
    model=model,
    X=cluster_features,
    feature_names=feature_names
)
```

### LIME Integration

LIME (Local Interpretable Model-agnostic Explanations) provides local explanations:

```python
# LIME analysis
lime_explanations = await shap_lime_integration.calculate_lime_explanations(
    model=model,
    X=cluster_features,
    feature_names=feature_names
)
```

### Feature Importance

Comprehensive feature importance analysis:

```python
# Feature importance
feature_importance = cluster_features.var().sort_values(ascending=False)
result.feature_importance = feature_importance.to_dict()
```

## Performance Monitoring

### Metrics Tracked

- **Execution Time**: Total clustering time
- **Memory Usage**: RAM consumption
- **Clustering Quality**: Silhouette, Calinski-Harabasz, Davies-Bouldin scores
- **Hardware Utilization**: CPU, GPU, memory usage
- **Feature Engineering Time**: Time spent on feature extraction
- **HPO Time**: Hyperparameter optimization duration
- **Backtesting Time**: Validation time

### Performance Logging

```python
# Performance summary
logger.info("=== Enhanced SR Clustering Performance Summary ===")
logger.info(f"Total execution time: {total_time:.2f} seconds")
logger.info(f"Memory usage: {memory_usage:.2f} MB")
logger.info(f"Features processed: {features.shape[0]} samples, {features.shape[1]} features")
logger.info(f"Clusters found: {n_clusters}")
logger.info(f"Average cluster quality: {avg_quality:.4f}")
```

## Error Handling & Robustness

### Data Validation

- **Input Validation**: Comprehensive data format checking
- **Data Leakage Detection**: Temporal leakage and lookahead bias prevention
- **Missing Data Handling**: Robust handling of NaN values
- **Data Quality Checks**: Statistical validation of input data

### Error Recovery

- **Graceful Degradation**: Fallback to simpler algorithms if advanced ones fail
- **Exception Handling**: Comprehensive error catching and logging
- **Resource Management**: Proper cleanup of resources
- **Timeout Handling**: Prevents infinite loops and hanging processes

## Testing

### Test Suite

The module includes comprehensive tests:

```bash
# Run all tests
python test_enhanced_sr_clustering.py

# Run specific test
python -m pytest test_enhanced_sr_clustering.py::test_basic_clustering
```

### Test Coverage

- **Basic Clustering**: Core functionality testing
- **Advanced Clustering**: HPO and advanced features
- **Performance Monitoring**: Performance metrics validation
- **Error Handling**: Robustness testing
- **Integration Testing**: Component integration validation

## Dependencies

### Core Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **scipy**: Scientific computing
- **asyncio**: Asynchronous programming

### Optional Dependencies

- **vectorbt**: High-performance backtesting
- **hdbscan**: Hierarchical DBSCAN clustering
- **umap-learn**: Dimensionality reduction
- **shap**: Model explainability
- **lime**: Local interpretability
- **optuna**: Hyperparameter optimization

### Hardware Dependencies

- **psutil**: System monitoring
- **M1 optimizations**: Apple Silicon specific optimizations

## Best Practices

### Configuration

1. **Start Simple**: Begin with basic configurations and gradually add complexity
2. **Parameter Tuning**: Use HPO for optimal parameter selection
3. **Feature Selection**: Enable only necessary features to avoid overfitting
4. **Validation**: Always use backtesting for validation

### Performance

1. **Hardware Optimization**: Enable hardware optimizations for better performance
2. **Memory Management**: Monitor memory usage for large datasets
3. **Parallel Processing**: Use parallel processing where possible
4. **Caching**: Implement caching for repeated operations

### Data Quality

1. **Data Leakage**: Always check for temporal leakage
2. **Feature Engineering**: Use domain knowledge for feature selection
3. **Validation**: Use multiple validation methods
4. **Monitoring**: Continuously monitor performance metrics

## Troubleshooting

### Common Issues

1. **Import Errors**: Check that all dependencies are installed
2. **Memory Issues**: Reduce dataset size or enable memory optimization
3. **Performance Issues**: Enable hardware optimization and reduce HPO trials
4. **Clustering Failures**: Check data quality and parameter ranges

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Profiling

Use built-in performance monitoring:

```python
# Enable performance monitoring
config.performance_monitoring = True
```

## Future Enhancements

### Planned Features

1. **Deep Learning Integration**: Neural network-based clustering
2. **Real-time Processing**: Streaming data support
3. **Multi-asset Support**: Cross-asset correlation analysis
4. **Advanced Visualization**: Interactive clustering visualizations
5. **Cloud Integration**: Cloud-based processing and storage

### Contributing

1. **Code Style**: Follow PEP 8 guidelines
2. **Testing**: Add tests for new features
3. **Documentation**: Update documentation for changes
4. **Performance**: Optimize for performance and memory usage

## License

This module is part of the ARES trading system and follows the same licensing terms.

## Support

For support and questions:

1. **Documentation**: Check this documentation first
2. **Issues**: Report issues through the project issue tracker
3. **Community**: Join the project community discussions
4. **Professional Support**: Contact the development team for professional support

---

*Last updated: 2024*
