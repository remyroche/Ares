# Multi-Stage Feature Selection Pipeline

A reusable, high-performance feature selection pipeline that implements a two-stage approach with VectorBT optimizations and fast-fail error handling.

## Overview

The `MultiStageFeatureSelectionPipeline` class provides a comprehensive solution for feature selection that combines multiple methods in a structured, two-stage approach:

1. **Stage 1**: mRMR + Spearman combination (70% mRMR + 30% Spearman)
2. **Stage 2**: Progressive refinement with RFE using ensemble scoring

## Key Features

- **Fast Fail Error Handling**: No fallback mechanisms - fails immediately if any component is unavailable
- **VectorBT Optimization**: High-performance vectorized operations for large datasets
- **Hardware Acceleration**: M1 memory optimization and adaptive optimization strategies
- **Budget-Aware Selection**: Optional integration with interactive feature generation
- **Bayesian Optimization**: Parameter optimization for enhanced performance
- **Memory Management**: Aggressive cleanup and chunked processing

## Usage

### Basic Usage

```python
from src.training.steps.pre_training.feature_selection import run_multi_stage_feature_selection

# Create your data
X = pd.DataFrame(...)  # Feature matrix
y = pd.Series(...)     # Target variable

# Run feature selection
result = run_multi_stage_feature_selection(
    X=X,
    y=y,
    symbol="BTCUSDT",
    exchange="binance",
    timeframe="15m"
)

print(f"Selected {len(result.selected_features)} features")
print(f"Execution time: {result.execution_time:.2f}s")
```

### Advanced Usage with Custom Configuration

```python
from src.training.steps.pre_training.feature_selection import (
    MultiStageFeatureSelectionPipeline,
    FeatureSelectionConfig
)

# Create custom configuration
config = FeatureSelectionConfig(
    target_features=60,
    enable_vectorbt_optimization=True,
    vectorbt_memory_efficient=True,
    stage1_mrmr_weight=0.8,
    stage1_spearman_weight=0.2,
    rfe_step_size=0.15
)

# Use pipeline class directly
pipeline = MultiStageFeatureSelectionPipeline(config)

try:
    result = pipeline.select_features(X, y, "ETHUSDT", "binance", "5m")
    print(f"Success: {result.success}")
    print(f"Selected features: {result.selected_features}")
finally:
    pipeline.cleanup()
```

## Configuration Options

### Core Parameters

- `target_features`: Number of features to select (default: 60)
- `enable_vectorbt_optimization`: Enable VectorBT optimizations (default: True)
- `vectorbt_memory_efficient`: Use memory-efficient processing (default: True)
- `vectorbt_chunk_size`: Chunk size for processing (default: 1000)

### Stage 1 Parameters

- `stage1_mrmr_weight`: Weight for mRMR scores (default: 0.7)
- `stage1_spearman_weight`: Weight for Spearman scores (default: 0.3)
- `stage1_target_ratio`: Ratio of features to select above target (default: 0.5)

### Stage 2 Parameters

- `rfe_step_size`: Percentage of features to remove per RFE round (default: 0.1)
- `stage2_bootstrap_cv_threshold`: Threshold for enabling bootstrap stability (default: 40)
- `bootstrap_n_samples`: Number of bootstrap samples (default: 100)

### Ensemble Weights

- `lgbm_shap`: LightGBM-SHAP weight (default: 0.4)
- `lasso_ensemble`: LASSO ensemble weight (default: 0.3)
- `rfe`: RFE weight (default: 0.2)
- `bootstrap_stability`: Bootstrap stability weight (default: 0.1)

## Pipeline Stages

### Stage 1: mRMR + Spearman Combination

1. **mRMR Calculation**: Uses VectorBT-optimized mRMR selector
2. **Spearman Correlation**: Calculates Spearman rank correlation
3. **Weighted Combination**: Combines scores with 70% mRMR + 30% Spearman
4. **Feature Selection**: Selects top 50% of features above target

### Stage 2: Progressive Refinement with RFE

1. **Ensemble Scoring**: Uses multiple methods to score features:
   - LightGBM-SHAP (40% weight)
   - LASSO ensemble (30% weight)
   - RFE scores (20% weight)
   - Bootstrap stability (10% weight, when enabled)
2. **Percentage-Based RFE**: Removes 10% of features above target per round
3. **Bootstrap Stability**: Enabled when 40+ features away from target

## Error Handling

The pipeline uses **fast fail** approach:

- **No Fallbacks**: If any required component fails, the entire pipeline fails
- **Early Validation**: Input validation happens before any processing
- **Clear Error Messages**: Detailed error messages for debugging

### Common Error Scenarios

```python
# Empty feature matrix
X_empty = pd.DataFrame()
y = pd.Series([1, 2, 3])
# Raises: ValueError("Input feature matrix X is None or empty")

# Mismatched dimensions
X = pd.DataFrame(np.random.randn(100, 10))
y = pd.Series(np.random.randn(50))
# Raises: ValueError("Feature matrix length doesn't match target length")

# Missing dependencies
# If VectorBT mRMR is not available:
# Raises: RuntimeError("VectorBT mRMR not available - fast fail")
```

## Performance Optimization

### VectorBT Integration

- **Correlation Calculations**: Vectorized correlation matrix computation
- **Rolling Operations**: Optimized rolling window operations
- **Memory Management**: Chunked processing for large datasets

### Hardware Acceleration

- **M1 Memory Optimization**: Advanced memory management for Apple Silicon
- **Adaptive Optimization**: Dynamic strategy selection based on hardware
- **Thread Management**: Optimized thread usage to avoid oversubscription

### Memory Management

- **Aggressive Cleanup**: Automatic memory cleanup after processing
- **Chunked Processing**: Process large datasets in manageable chunks
- **Resource Monitoring**: Real-time memory pressure monitoring

## Dependencies

### Required

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: Machine learning algorithms

### Optional (for enhanced performance)

- `vectorbt`: Vectorized operations
- `lightgbm`: Gradient boosting
- `shap`: SHAP values
- `scipy`: Statistical functions

## Examples

See `examples/multi_stage_feature_selection_example.py` for comprehensive usage examples including:

1. Basic usage with default configuration
2. Custom configuration
3. Error handling demonstration
4. Performance comparison

## Migration from Legacy Code

The legacy `MultiStageFeatureSelector` class has been removed. All code should now use the new `MultiStageFeatureSelectionPipeline`.

### New Code
```python
from src.training.steps.pre_training.feature_selection import run_multi_stage_feature_selection

result = run_multi_stage_feature_selection(X, y, symbol, exchange, timeframe, config)
```

## Best Practices

1. **Always use try-finally**: Ensure proper cleanup of pipeline resources
2. **Configure appropriately**: Set parameters based on your dataset size and requirements
3. **Monitor memory usage**: Use memory-efficient settings for large datasets
4. **Validate inputs**: Ensure data quality before running the pipeline
5. **Handle errors gracefully**: Implement proper error handling in your application

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `vectorbt_chunk_size` or enable `vectorbt_memory_efficient`
2. **Slow performance**: Ensure VectorBT optimization is enabled
3. **Import errors**: Check that all optional dependencies are installed
4. **Configuration errors**: Validate configuration parameters before use

### Debug Mode

Enable debug logging to see detailed pipeline execution:

```python
import logging
logging.getLogger("MultiStageFeatureSelectionPipeline").setLevel(logging.DEBUG)
```