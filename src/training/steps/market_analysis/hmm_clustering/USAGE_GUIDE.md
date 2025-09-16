# HMM Clustering with Common Utilities - Usage Guide

This guide demonstrates how to use the enhanced HMM clustering module with all available common utilities for comprehensive market analysis.

## Overview

The enhanced HMM clustering implementation integrates with the following common utilities:

- **Hardware Optimization**: M1 GPU, memory, and CPU optimizers
- **Data Operations**: Common operations, utilities, and validation
- **Matrix Operations**: Unified matrix operations for efficient computation
- **ML Common**: Cross-validation, hyperparameter optimization, and validation
- **Serialization**: JSON and pickle serialization utilities
- **Data Management**: Kline parquet handler for market data

## Quick Start

### Basic Usage

```python
from src.training.steps.market_analysis.hmm_clustering.enhanced_hmm_clustering import (
    EnhancedHMMClustering, 
    HMMClusteringConfig
)

# Create configuration
config = HMMClusteringConfig(
    n_components=3,
    covariance_type='full',
    n_iter=100,
    random_state=42,
    use_gpu=True,
    enable_validation=True,
    enable_optimization=True
)

# Create and train model
hmm_clustering = EnhancedHMMClustering(config)
results = hmm_clustering.fit(your_data)

# Get results
print(f"Silhouette Score: {results.silhouette_score:.3f}")
print(f"Training Time: {results.training_time:.2f} seconds")
```

### Complete Integration Example

```python
from src.training.steps.market_analysis.hmm_clustering.integration_example import (
    HMMClusteringIntegration
)

# Create integration instance
integration = HMMClusteringIntegration()

# Load market data
data = integration.load_market_data('path/to/klines.parquet', 'BTCUSDT', '1h')

# Run comprehensive analysis
results = integration.run_comprehensive_analysis(data, optimize_hyperparams=True)

# Generate report
report = integration.generate_report(results)
print(report)

# Save results
integration.save_analysis_results(results, 'analysis_results.json')
```

## Common Utilities Integration

### 1. Hardware Optimization

The HMM clustering automatically uses available hardware optimizations:

```python
# GPU acceleration (M1/M2/M3)
config = HMMClusteringConfig(use_gpu=True)

# Memory optimization
# Automatically applied when memory_optimizer is available

# CPU optimization
# Automatically applied when cpu_optimizer is available
```

### 2. Data Operations

```python
from src.utils.common_operations import (
    safe_dataframe_operation,
    validate_dataframe_columns,
    calculate_data_quality_metrics
)
from src.utils.common_utilities import safe_convert_dtypes
from src.utils.math_validation import safe_divide, safe_log, validate_finite

# Safe DataFrame operations
def process_data(df):
    return safe_dataframe_operation(df, lambda x: x.dropna())

# Validate data quality
quality_metrics = calculate_data_quality_metrics(data)

# Safe mathematical operations
returns = safe_divide(close_prices, close_prices.shift(1))
log_returns = safe_log(returns)
```

### 3. Matrix Operations

```python
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations

# Initialize matrix operations
matrix_ops = UnifiedMatrixOperations()

# Optimize data for clustering
optimized_data = matrix_ops.optimize_for_clustering(features)

# Batch operations
results = matrix_ops.batch_correlation_analysis(feature_matrix)
```

### 4. ML Common Utilities

```python
from src.utils.ml_common.validation.cross_validation import TimeSeriesCrossValidator
from src.utils.ml_common.optimization.hyperparameter_optimization import HyperparameterOptimizer

# Cross-validation
cv_validator = TimeSeriesCrossValidator()
cv_scores = cv_validator.cross_validate(model, X, y)

# Hyperparameter optimization
hpo_optimizer = HyperparameterOptimizer()
best_params = hpo_optimizer.optimize(
    model_class=GaussianHMM,
    param_grid=param_grid,
    X=features,
    cv=cv_validator
)
```

### 5. Serialization

```python
from src.utils.serialization_utils import JSONSerializer, PickleSerializer

# JSON serialization
json_serializer = JSONSerializer()
json_serializer.save(results, 'results.json')

# Pickle serialization
pickle_serializer = PickleSerializer()
pickle_serializer.save(model, 'model.pkl')
```

### 6. Data Management

```python
from src.utils.kline_parquet import KlineParquetHandler

# Load market data
kline_handler = KlineParquetHandler()
data = kline_handler.load_klines('path/to/klines.parquet', 'BTCUSDT', '1h')

# Save processed data
kline_handler.save_klines(data, 'processed_klines.parquet')
```

## Configuration Options

### HMMClusteringConfig

```python
@dataclass
class HMMClusteringConfig:
    n_components: int = 3              # Number of hidden states
    covariance_type: str = 'full'      # Covariance type ('full', 'tied', 'diag')
    n_iter: int = 100                  # Maximum iterations
    random_state: int = 42             # Random seed
    use_gpu: bool = True               # Enable GPU acceleration
    memory_limit_gb: Optional[float] = None  # Memory limit
    enable_validation: bool = True     # Enable validation
    enable_optimization: bool = True   # Enable hyperparameter optimization
    max_retries: int = 3               # Maximum retries
    timeout_seconds: int = 300         # Timeout in seconds
```

## Advanced Usage

### Custom Feature Engineering

```python
def create_custom_features(data):
    """Create custom features using common utilities."""
    features = pd.DataFrame()
    
    # Price features
    features['returns'] = data['close'].pct_change()
    features['log_returns'] = safe_log(data['close'] / data['close'].shift(1))
    
    # Volume features
    features['volume_ratio'] = safe_divide(
        data['volume'], 
        data['volume'].rolling(20).mean()
    )
    
    # Technical indicators
    features['rsi'] = calculate_rsi(data['close'])
    features['macd'] = calculate_macd(data['close'])
    
    return features.dropna()
```

### Hyperparameter Optimization

```python
# Define parameter grid
param_grid = {
    'n_components': [2, 3, 4, 5, 6],
    'covariance_type': ['full', 'tied', 'diag'],
    'n_iter': [50, 100, 200, 500]
}

# Run optimization
best_params = hmm_clustering.optimize_hyperparameters(
    data, 
    param_grid=param_grid
)

# Update configuration
for param, value in best_params.items():
    setattr(hmm_clustering.config, param, value)
```

### Model Persistence

```python
# Save trained model
hmm_clustering.save_model('hmm_model.pkl')

# Load saved model
new_hmm = EnhancedHMMClustering(config)
new_hmm.load_model('hmm_model.pkl')

# Make predictions
predictions = new_hmm.predict(new_data)
probabilities = new_hmm.predict_proba(new_data)
```

### Performance Monitoring

```python
# Get performance summary
summary = hmm_clustering.get_performance_summary()

print(f"Training Metrics: {summary['training_metrics']}")
print(f"Clustering Metrics: {summary['clustering_metrics']}")
print(f"Memory Usage: {summary['memory_usage']}")
print(f"Hardware Info: {summary['hardware_info']}")
```

## Error Handling and Validation

The enhanced HMM clustering includes comprehensive error handling:

```python
try:
    results = hmm_clustering.fit(data)
except ValueError as e:
    print(f"Validation error: {e}")
except ImportError as e:
    print(f"Missing dependency: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Optimization Tips

1. **Use GPU acceleration** when available (M1/M2/M3)
2. **Enable memory optimization** for large datasets
3. **Use appropriate data types** (float32 instead of float64)
4. **Enable hyperparameter optimization** for better results
5. **Use matrix operations** for efficient computation
6. **Monitor memory usage** during training

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce dataset size or enable memory optimization
2. **GPU not available**: Set `use_gpu=False` in config
3. **Convergence issues**: Increase `n_iter` or adjust `covariance_type`
4. **Import errors**: Ensure all dependencies are installed

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
logger = logging.getLogger('EnhancedHMMClustering')
logger.setLevel(logging.DEBUG)
```

## Examples

See the following files for complete examples:

- `enhanced_hmm_clustering.py` - Core implementation
- `integration_example.py` - Complete integration example
- `test_enhanced_hmm.py` - Unit tests

## Dependencies

Required packages:
- numpy
- pandas
- scikit-learn
- hmmlearn
- psutil (optional, for memory monitoring)

Install with:
```bash
pip install numpy pandas scikit-learn hmmlearn psutil
```

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Verify all dependencies are installed
3. Ensure data is properly formatted
4. Check hardware compatibility