# HMM Regime Parameter Optimization

## Overview

This system optimizes Step 3 HMM regime discovery parameters to capture **distinct market conditions** rather than predict regime transitions. It uses Optuna for hyperparameter optimization and focuses on market condition differentiation as the primary objective.

## Key Philosophy

✅ **Correct Approach**: Clusters should capture distinct market conditions (volatility, momentum, volume patterns, etc.)
❌ **Incorrect Approach**: Clusters should predict regime transitions

## Features

- **Market Condition Focus**: Optimizes for capturing distinct market conditions
- **Comprehensive Evaluation**: Multiple metrics for cluster quality assessment
- **Flexible Configuration**: Customizable parameter ranges and evaluation weights
- **Visualization**: Detailed optimization results and parameter importance
- **Integration Ready**: Easy to integrate with existing Step 3 pipeline

## Files

1. **`optimize_hmm_regime_parameters.py`** - Main optimization script
2. **`hmm_optimization_config.json`** - Configuration file with parameter ranges
3. **`example_hmm_optimization.py`** - Example usage and demonstrations
4. **`HMM_OPTIMIZATION_README.md`** - This documentation

## Installation

```bash
# Install required dependencies
pip install optuna scikit-learn pandas numpy matplotlib seaborn

# Or add to your requirements.txt
optuna>=3.0.0
scikit-learn>=1.0.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## Quick Start

### 1. Basic Usage

```bash
# Run optimization on your feature data
python optimize_hmm_regime_parameters.py --data_path path/to/your/feature_data.parquet --n_trials 100

# Run with custom configuration
python optimize_hmm_regime_parameters.py --data_path path/to/your/feature_data.parquet --config_path hmm_optimization_config.json --n_trials 150
```

### 2. Example Usage

```bash
# Run the example to see how it works
python example_hmm_optimization.py
```

## Configuration

### Parameter Ranges

The system optimizes several parameter categories:

#### HMM Parameters
- `n_components`: Number of HMM states (2-10)
- `covariance_type`: Covariance structure ('full', 'tied', 'diag', 'spherical')
- `n_iter`: Maximum iterations (50-300)
- `tol`: Convergence tolerance (1e-6 to 1e-2)
- `reg_covar`: Regularization (1e-7 to 1e-2)

#### Clustering Parameters
- `clustering_method`: Algorithm ('kmeans', 'gaussian_mixture')
- `n_clusters`: Number of clusters (3-15)
- `init`: Initialization method for K-means
- `n_init`: Number of initializations
- `max_iter`: Maximum iterations

#### Feature Engineering Parameters
- `use_pca`: Whether to use PCA (True/False)
- `n_pca_components`: Number of PCA components (5-25)
- `feature_selection_method`: Selection strategy ('all', 'variance', 'correlation')
- `scaling_method`: Scaling approach ('standard', 'robust', 'minmax')

### Evaluation Metrics

The optimization uses a weighted combination of metrics:

1. **Market Condition Differentiation** (40% weight)
   - How well clusters differentiate market conditions
   - Measures differences in volatility, momentum, volume patterns

2. **Cluster Quality** (20% weight)
   - Traditional metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
   - Measures cluster separation and cohesion

3. **Market Condition Consistency** (20% weight)
   - How consistent market conditions are within clusters
   - Lower coefficient of variation = better consistency

4. **Cluster Balance** (10% weight)
   - How balanced cluster sizes are
   - Prevents extremely skewed distributions

5. **Market Condition Separation** (10% weight)
   - F-ratio between between-cluster and within-cluster variance
   - Measures how well clusters separate different market conditions

## Usage Examples

### 1. Basic Optimization

```python
from optimize_hmm_regime_parameters import HMMRegimeOptimizer, identify_market_condition_columns
import pandas as pd

# Load your data
data = pd.read_parquet("path/to/your/feature_data.parquet")

# Identify features and market conditions
feature_columns = [col for col in data.columns if col not in ['timestamp', 'composite_cluster_id']]
market_condition_columns = identify_market_condition_columns(data)

# Initialize optimizer
optimizer = HMMRegimeOptimizer()

# Run optimization
results = optimizer.optimize(
    data=data,
    feature_columns=feature_columns,
    market_condition_columns=market_condition_columns,
    n_trials=100
)

print(f"Best Score: {results['best_score']:.4f}")
print(f"Best Parameters: {results['best_params']}")
```

### 2. Custom Configuration

```python
# Custom configuration
config = {
    "evaluation_weights": {
        "market_differentiation": 0.5,  # Emphasize market differentiation
        "cluster_quality": 0.2,
        "market_consistency": 0.2,
        "cluster_balance": 0.05,
        "market_separation": 0.05
    }
}

optimizer = HMMRegimeOptimizer(config)
results = optimizer.optimize(data, feature_columns, market_condition_columns, n_trials=100)
```

### 3. Strategy Comparison

```python
# Test different evaluation strategies
strategies = {
    "market_focused": {"market_differentiation": 0.6, "cluster_quality": 0.1, ...},
    "balanced": {"market_differentiation": 0.3, "cluster_quality": 0.3, ...},
    "quality_focused": {"market_differentiation": 0.2, "cluster_quality": 0.5, ...}
}

for strategy_name, weights in strategies.items():
    config = {"evaluation_weights": weights}
    optimizer = HMMRegimeOptimizer(config)
    results = optimizer.optimize(data, feature_columns, market_condition_columns, n_trials=50)
    print(f"{strategy_name}: {results['best_score']:.4f}")
```

## Integration with Step 3

### 1. Apply Optimized Parameters

After optimization, apply the best parameters to your Step 3:

```python
# Get best parameters
best_params = results['best_params']

# Update your Step 3 configuration
step3_config = {
    "hmm_parameters": {
        "n_components": best_params.get('n_components', 5),
        "covariance_type": best_params.get('covariance_type', 'full'),
        "n_iter": best_params.get('n_iter', 100),
        "tol": best_params.get('tol', 1e-4),
        "reg_covar": best_params.get('reg_covar', 1e-6)
    },
    "clustering_parameters": {
        "method": best_params.get('clustering_method', 'kmeans'),
        "n_clusters": best_params.get('n_clusters', 5)
    },
    "feature_parameters": {
        "use_pca": best_params.get('use_pca', False),
        "scaling_method": best_params.get('scaling_method', 'standard')
    }
}
```

### 2. Validate Results

Use the cluster validation tools to confirm improved quality:

```bash
# Validate the optimized clusters
python test_hmm_cluster_relevance.py --data_path path/to/optimized_cluster_data.parquet
```

## Output Files

The optimization generates several output files:

1. **`optimization_report.md`** - Detailed optimization report
2. **`optimization_results.json`** - Best parameters and optimization history
3. **`optimization_results.png`** - Visualizations of optimization process

## Advanced Usage

### 1. Custom Market Condition Identification

```python
def custom_market_condition_identifier(data: pd.DataFrame) -> List[str]:
    """Custom function to identify market condition columns."""
    market_columns = []
    
    # Your custom logic here
    for col in data.columns:
        if any(keyword in col.lower() for keyword in ['volatility', 'momentum', 'volume']):
            market_columns.append(col)
    
    return market_columns

# Use custom identifier
market_condition_columns = custom_market_condition_identifier(data)
```

### 2. Custom Evaluation Metrics

```python
class CustomHMMOptimizer(HMMRegimeOptimizer):
    def _evaluate_market_condition_capture(self, cluster_data: pd.DataFrame, 
                                         market_condition_columns: List[str]) -> float:
        """Custom evaluation function."""
        # Your custom evaluation logic here
        score = self._calculate_custom_score(cluster_data, market_condition_columns)
        return score
```

### 3. Parallel Optimization

```python
# For large datasets, use parallel optimization
import optuna

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(),
    study_name="parallel_optimization"
)

# Run with multiple workers
study.optimize(objective, n_trials=100, n_jobs=4)
```

## Troubleshooting

### Common Issues

1. **Low Optimization Scores**
   - Increase `n_trials` for more thorough search
   - Adjust evaluation weights to focus on specific aspects
   - Check data quality and feature engineering

2. **Slow Optimization**
   - Reduce parameter search ranges
   - Use fewer trials initially
   - Enable pruning for early termination

3. **Memory Issues**
   - Reduce dataset size for testing
   - Use feature selection to reduce dimensionality
   - Enable garbage collection between trials

### Performance Tips

1. **Start Small**: Begin with 20-50 trials to test the setup
2. **Use Pruning**: Enable MedianPruner for early termination of poor trials
3. **Parallel Processing**: Use multiple workers for faster optimization
4. **Feature Selection**: Reduce feature dimensionality before optimization

## Best Practices

1. **Data Preparation**
   - Ensure clean, preprocessed data
   - Handle missing values appropriately
   - Scale features consistently

2. **Parameter Ranges**
   - Start with reasonable ranges based on domain knowledge
   - Avoid overly broad ranges that waste computation
   - Use log-scale for parameters like tolerances

3. **Evaluation Strategy**
   - Focus on market condition differentiation (40-60% weight)
   - Balance with traditional cluster quality metrics
   - Consider your specific use case when setting weights

4. **Validation**
   - Always validate optimized parameters on test data
   - Use the cluster validation tools to confirm quality
   - Monitor for overfitting to optimization data

## Integration with Pipeline

### 1. Automated Optimization

```python
# Integrate into your pipeline
def optimize_step3_parameters(data: pd.DataFrame) -> Dict[str, Any]:
    """Automatically optimize Step 3 parameters."""
    optimizer = HMMRegimeOptimizer()
    results = optimizer.optimize(data, feature_columns, market_condition_columns)
    return results['best_params']

# Use in pipeline
best_params = optimize_step3_parameters(feature_data)
# Apply to Step 3
apply_optimized_parameters(best_params)
```

### 2. Continuous Optimization

```python
# Set up periodic re-optimization
def periodic_optimization():
    """Re-optimize parameters periodically."""
    new_data = load_recent_data()
    best_params = optimize_step3_parameters(new_data)
    update_pipeline_configuration(best_params)
```

## Conclusion

This optimization system ensures that your HMM regime discovery captures distinct market conditions rather than trying to predict transitions. By focusing on market condition differentiation, you'll get clusters that are more meaningful for trading strategies and risk management.

Remember: **The goal is to identify when the market is in different states (high volatility, trending, mean-reverting, etc.), not to predict when it will change states.**