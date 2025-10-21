# Data-Driven Clustering Parameters

This module provides data-driven optimization of clustering parameters, replacing hardcoded values with adaptive, data-driven alternatives that improve clustering quality and economic performance.

## Overview

The clustering pipeline previously used hardcoded parameters that were not optimized for specific datasets or market conditions:

- **Feature Group Weights**: `w_returns=0.50, w_vol=0.30, w_volume=0.20`
- **Regime Merging Thresholds**: `similarity_threshold=0.8, distance_threshold=0.2, p_value_threshold=0.05`
- **Temporal Window Sizes**: `window_size=300, smoothing_window=5`
- **Cluster Validation Thresholds**: `min_silhouette=0.2, max_dbi=2.5`

These hardcoded values are now replaced with data-driven optimization that:

1. **Adapts to data characteristics**: Parameters are optimized based on the specific dataset
2. **Improves clustering quality**: Uses multiple validation metrics to ensure better clustering
3. **Enables economic validation**: Incorporates economic metrics into parameter selection
4. **Provides statistical significance**: Uses permutation testing and bootstrap validation
5. **Supports multiple optimization strategies**: Bayesian TPE, grid search, random search, and adaptive methods

## Architecture

### Core Components

1. **Configuration Classes** (`config/data_driven_config.py`)
   - `DataDrivenClusteringConfig`: Main configuration
   - `FeatureGroupWeightConfig`: Feature weight optimization
   - `RegimeMergingThresholdConfig`: Merging threshold optimization
   - `TemporalWindowConfig`: Temporal window optimization
   - `ClusterValidationThresholdConfig`: Validation threshold optimization

2. **Individual Optimizers** (`optimization/`)
   - `DataDrivenFeatureWeightOptimizer`: Optimizes feature group weights
   - `DataDrivenMergingThresholdOptimizer`: Optimizes regime merging thresholds
   - `DataDrivenTemporalWindowOptimizer`: Optimizes temporal window sizes
   - `DataDrivenValidationThresholdOptimizer`: Optimizes cluster validation thresholds

3. **Main Optimizer** (`optimization/data_driven_clustering_optimizer.py`)
   - `DataDrivenClusteringOptimizer`: Integrates all individual optimizers

4. **Updated Components** (`step1_feature_preparation_data_driven.py`, `similarity_merger_data_driven.py`)
   - Updated feature preparation with data-driven weight optimization
   - Updated similarity merger with data-driven threshold optimization

## Usage

### Basic Usage

```python
from src.training.steps.market_analysis.hdbscan_clustering.config.data_driven_config import DataDrivenClusteringConfig
from src.training.steps.market_analysis.hdbscan_clustering.optimization.data_driven_clustering_optimizer import DataDrivenClusteringOptimizer

# Create configuration
config = DataDrivenClusteringConfig()

# Create optimizer
optimizer = DataDrivenClusteringOptimizer(config)

# Optimize all parameters
result = optimizer.optimize_all_parameters(
    market_data=market_data,
    features=features,
    feature_names=feature_names,
    clustering_func=clustering_function,
    economic_validation_func=economic_validation_function
)

# Get optimized parameters
optimal_parameters = result.optimal_parameters
print(f"Optimal feature weights: {optimal_parameters}")
```

### Individual Component Usage

#### Feature Weight Optimization

```python
from src.training.steps.market_analysis.hdbscan_clustering.optimization.data_driven_feature_weights import DataDrivenFeatureWeightOptimizer
from src.training.steps.market_analysis.hdbscan_clustering.config.data_driven_config import FeatureGroupWeightConfig

# Create optimizer
config = FeatureGroupWeightConfig()
optimizer = DataDrivenFeatureWeightOptimizer(config)

# Optimize weights
result = optimizer.optimize_weights(
    features=features,
    feature_names=feature_names,
    market_data=market_data,
    clustering_func=clustering_function
)

# Get optimal weights
optimal_weights = result.optimal_weights
print(f"Optimal weights: {optimal_weights}")
```

#### Merging Threshold Optimization

```python
from src.training.steps.market_analysis.hdbscan_clustering.optimization.data_driven_merging_thresholds import DataDrivenMergingThresholdOptimizer
from src.training.steps.market_analysis.hdbscan_clustering.config.data_driven_config import RegimeMergingThresholdConfig

# Create optimizer
config = RegimeMergingThresholdConfig()
optimizer = DataDrivenMergingThresholdOptimizer(config)

# Optimize thresholds
result = optimizer.optimize_thresholds(
    cluster_labels=initial_labels,
    features=features,
    merging_func=merging_function
)

# Get optimal thresholds
optimal_thresholds = result.optimal_thresholds
print(f"Optimal thresholds: {optimal_thresholds}")
```

### Integration with Existing Pipeline

#### Updated Feature Preparation

```python
from src.training.steps.market_analysis.clusters.step1_feature_preparation_data_driven import DataDrivenFeaturePreparationStep

# Create data-driven feature preparation step
feature_step = DataDrivenFeaturePreparationStep(
    verbose=True,
    enable_data_driven=True
)

# Execute with data-driven optimization
context = await feature_step.execute(context, config)

# Access optimization results
data_driven_weights = context.data_driven_weights
optimization_results = context.optimization_results
```

#### Updated Similarity Merger

```python
from src.training.steps.market_analysis.hdbscan_clustering.similarity_merger_data_driven import DataDrivenSimilarityMerger

# Create data-driven similarity merger
merger = DataDrivenSimilarityMerger()

# Merge regimes with data-driven thresholds
merged_labels, merging_info = merger.merge_regimes(
    cluster_labels=cluster_labels,
    features=features,
    target_metric='silhouette'
)

# Access optimization results
optimization_results = merger.get_optimization_results()
```

## Configuration

### Feature Group Weight Configuration

```python
from src.training.steps.market_analysis.hdbscan_clustering.config.data_driven_config import FeatureGroupWeightConfig

config = FeatureGroupWeightConfig(
    enable_optimization=True,
    optimization_strategy=OptimizationStrategy.BAYESIAN_TPE,
    n_trials=100,
    primary_metric=ValidationMetric.SILHOUETTE,
    enable_economic_validation=True,
    economic_weight=0.3
)
```

### Merging Threshold Configuration

```python
from src.training.steps.market_analysis.hdbscan_clustering.config.data_driven_config import RegimeMergingThresholdConfig

config = RegimeMergingThresholdConfig(
    enable_optimization=True,
    optimization_strategy=OptimizationStrategy.BAYESIAN_TPE,
    similarity_threshold_range=(0.5, 0.95),
    distance_threshold_range=(0.1, 0.5),
    p_value_threshold_range=(0.01, 0.1),
    n_trials=80
)
```

### Temporal Window Configuration

```python
from src.training.steps.market_analysis.hdbscan_clustering.config.data_driven_config import TemporalWindowConfig

config = TemporalWindowConfig(
    enable_optimization=True,
    optimization_strategy=OptimizationStrategy.BAYESIAN_TPE,
    window_size_range=(50, 500),
    smoothing_window_range=(3, 20),
    enable_volatility_adaptation=True,
    n_trials=60
)
```

### Validation Threshold Configuration

```python
from src.training.steps.market_analysis.hdbscan_clustering.config.data_driven_config import ClusterValidationThresholdConfig

config = ClusterValidationThresholdConfig(
    enable_optimization=True,
    optimization_strategy=OptimizationStrategy.BAYESIAN_TPE,
    min_silhouette_range=(0.1, 0.5),
    max_dbi_range=(1.0, 4.0),
    min_stability_range=(0.5, 0.9),
    enable_permutation_testing=True,
    n_trials=70
)
```

## Optimization Strategies

### 1. Bayesian TPE (Tree-structured Parzen Estimator)
- **Best for**: Complex parameter spaces with many dimensions
- **Advantages**: Efficient exploration, good for continuous parameters
- **Use case**: Default strategy for most optimizations

### 2. Grid Search
- **Best for**: Small parameter spaces with discrete values
- **Advantages**: Exhaustive search, guaranteed to find optimal solution
- **Use case**: When you have few parameters and want complete coverage

### 3. Random Search
- **Best for**: Quick exploration of parameter space
- **Advantages**: Fast, good for initial exploration
- **Use case**: When you need quick results or have limited time

### 4. Adaptive
- **Best for**: When you have domain knowledge about parameter relationships
- **Advantages**: Uses data characteristics to guide optimization
- **Use case**: When you want to incorporate domain expertise

## Validation Metrics

### Primary Metrics
- **Silhouette Score**: Measures cluster separation and cohesion
- **Davies-Bouldin Index**: Measures cluster compactness and separation
- **Calinski-Harabasz Index**: Measures cluster separation
- **Stability Index**: Measures temporal stability of clusters
- **Economic Return**: Measures economic performance of clustering
- **Sharpe Ratio**: Measures risk-adjusted returns

### Statistical Validation
- **Permutation Testing**: Tests statistical significance against random clustering
- **Bootstrap Validation**: Tests stability across bootstrap samples
- **Cross-Validation**: Tests generalization to unseen data

## Economic Validation

The system supports economic validation to ensure that optimized parameters lead to better economic outcomes:

```python
def economic_validation_func(features, cluster_labels, market_data):
    """Example economic validation function."""
    # Calculate returns for each cluster
    cluster_returns = {}
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:  # Skip noise
            continue
        mask = cluster_labels == cluster_id
        cluster_data = market_data[mask]
        if len(cluster_data) > 0:
            returns = cluster_data['close'].pct_change().dropna()
            cluster_returns[cluster_id] = {
                'mean_return': returns.mean(),
                'volatility': returns.std(),
                'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0
            }
    
    # Calculate overall economic score
    if cluster_returns:
        mean_sharpe = np.mean([cr['sharpe_ratio'] for cr in cluster_returns.values()])
        return mean_sharpe
    else:
        return 0.0
```

## Performance Considerations

### Memory Usage
- Feature weight optimization: ~2-4GB for 1000 samples × 100 features
- Merging threshold optimization: ~1-2GB for 1000 samples × 20 features
- Temporal window optimization: ~1-2GB for 1000 samples
- Validation threshold optimization: ~1-2GB for 1000 samples

### Time Complexity
- Feature weight optimization: 5-15 minutes (100 trials)
- Merging threshold optimization: 3-10 minutes (80 trials)
- Temporal window optimization: 2-8 minutes (60 trials)
- Validation threshold optimization: 3-12 minutes (70 trials)

### Optimization Tips
1. **Start with fewer trials**: Use 20-50 trials for initial exploration
2. **Use adaptive strategy**: For quick results with domain knowledge
3. **Enable caching**: Save optimization results for reuse
4. **Parallel processing**: Use multiple cores for faster optimization

## Examples

### Complete Example

See `examples/data_driven_clustering_example.py` for a complete example that demonstrates:

1. Creating sample market data and features
2. Running all optimization components
3. Analyzing results and generating recommendations
4. Calculating performance metrics

### Running the Example

```bash
cd src/training/steps/market_analysis/hdbscan_clustering
python examples/data_driven_clustering_example.py
```

## Migration Guide

### From Hardcoded to Data-Driven

1. **Replace hardcoded weights**:
   ```python
   # Old hardcoded approach
   w_returns, w_vol, w_volume = 0.50, 0.30, 0.20
   
   # New data-driven approach
   optimizer = DataDrivenFeatureWeightOptimizer(config)
   result = optimizer.optimize_weights(features, feature_names, market_data, clustering_func)
   optimal_weights = result.optimal_weights
   ```

2. **Replace hardcoded thresholds**:
   ```python
   # Old hardcoded approach
   similarity_threshold = 0.8
   distance_threshold = 0.2
   p_value_threshold = 0.05
   
   # New data-driven approach
   optimizer = DataDrivenMergingThresholdOptimizer(config)
   result = optimizer.optimize_thresholds(cluster_labels, features, merging_func)
   optimal_thresholds = result.optimal_thresholds
   ```

3. **Replace hardcoded windows**:
   ```python
   # Old hardcoded approach
   window_size = 300
   smoothing_window = 5
   
   # New data-driven approach
   optimizer = DataDrivenTemporalWindowOptimizer(config)
   result = optimizer.optimize_windows(market_data, clustering_func)
   optimal_windows = result.optimal_windows
   ```

## Troubleshooting

### Common Issues

1. **Optimization fails**: Check that clustering function works correctly
2. **Memory errors**: Reduce number of trials or use smaller datasets
3. **Convergence issues**: Try different optimization strategies
4. **Economic validation fails**: Ensure economic validation function is properly implemented

### Debug Mode

Enable detailed logging to debug optimization issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or for specific components
logger = logging.getLogger('DataDrivenFeatureWeightOptimizer')
logger.setLevel(logging.DEBUG)
```

## Future Enhancements

1. **Online Learning**: Update parameters as new data arrives
2. **Multi-Objective Optimization**: Optimize multiple metrics simultaneously
3. **Ensemble Methods**: Combine multiple optimization strategies
4. **Real-time Adaptation**: Adapt parameters based on market conditions
5. **A/B Testing**: Compare different parameter sets in production

## Contributing

When contributing to the data-driven clustering system:

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new functionality
3. Update documentation for new features
4. Ensure backward compatibility when possible
5. Add examples for new optimization strategies

## License

This module is part of the larger clustering system and follows the same license terms.