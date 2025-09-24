# CLVSA Architecture Integration with Tree Models

## Overview

This document describes the integration of CLVSA (Contextual Learning with Variable Structure Adaptation) architecture with tree-based models in the codebase. All tree models are now automatically wrapped with CLVSA architecture by default, providing enhanced attention mechanisms and improved performance.

## What is CLVSA?

CLVSA (Contextual Learning with Variable Structure Adaptation) is an advanced architecture that combines:

- **Convolutional layers** for spatial feature extraction
- **LSTM layers** for temporal dependencies modeling
- **Attention mechanisms** for dynamic focus on relevant features
- **Variational components** for uncertainty quantification

For tree models, we use a specialized **Tree CLVSA Wrapper** that provides:

- **Feature attention** using mutual information and tree importance
- **Temporal attention** for time series patterns
- **Regime-aware attention** for market condition adaptation
- **Ensemble attention** for tree-specific optimizations

## Tree Models with CLVSA Integration

The following tree models are automatically wrapped with CLVSA architecture:

### 1. Random Forest
- **Base Model**: `RandomForestRegressor` / `RandomForestClassifier`
- **CLVSA Features**: Feature importance weighting, ensemble attention
- **Default Parameters**: `n_estimators=500`, `max_depth=10`, overfitting prevention

### 2. XGBoost
- **Base Model**: `XGBRegressor` / `XGBClassifier`
- **CLVSA Features**: Gradient-based attention, temporal patterns
- **Default Parameters**: `n_estimators=100`, `learning_rate=0.1`, `max_depth=6`

### 3. LightGBM
- **Base Model**: `LGBMRegressor` / `LGBMClassifier`
- **CLVSA Features**: Leaf-based attention, memory-efficient processing
- **Default Parameters**: `n_estimators=1000`, `learning_rate=0.05`, `max_depth=6`

### 4. CatBoost
- **Base Model**: `CatBoostRegressor` / `CatBoostClassifier`
- **CLVSA Features**: Categorical feature attention, robust scaling
- **Default Parameters**: `iterations=1000`, `learning_rate=0.05`, `depth=6`

### 5. Extra Trees
- **Base Model**: `ExtraTreesRegressor` / `ExtraTreesClassifier`
- **CLVSA Features**: Random split attention, variance-based weighting
- **Default Parameters**: `n_estimators=100`, `max_depth=None`

### 6. Histogram Gradient Boosting
- **Base Model**: `HistGradientBoostingRegressor` / `HistGradientBoostingClassifier`
- **CLVSA Features**: Histogram-based attention, efficient binning
- **Default Parameters**: `max_iter=100`, `max_leaf_nodes=31`

## Usage Examples

### Basic Usage (CLVSA Enabled by Default)

```python
from src.utils.ml_common.models.model_factory import EnhancedModelFactory, ModelConfig, ModelType

# Create model factory
factory = EnhancedModelFactory()

# Create Random Forest with CLVSA (default behavior)
config = ModelConfig(
    model_type=ModelType.RANDOM_FOREST,
    model_name="rf_with_clvsa",
    model_params={
        'n_estimators': 100,
        'max_depth': 8
    }
)

# CLVSA is automatically applied
model = factory.create_model(config)
```

### Advanced CLVSA Configuration

```python
# Custom CLVSA configuration
config = ModelConfig(
    model_type=ModelType.XGBOOST,
    model_name="xgboost_advanced_clvsa",
    model_params={
        'n_estimators': 200,
        'learning_rate': 0.05,
        # CLVSA-specific parameters
        'attention_dim': 128,
        'use_temporal_attention': True,
        'regime_aware': True,
        'ensemble_attention': True,
        'feature_selection_method': 'mutual_info',
        'temporal_window_size': 30,
        'memory_efficient': True
    }
)

model = factory.create_model(config)
```

### Disabling CLVSA

```python
# Disable CLVSA wrapper
config = ModelConfig(
    model_type=ModelType.LIGHTGBM,
    model_name="lightgbm_no_clvsa",
    model_params={
        'use_clvsa': False,  # Disable CLVSA
        'n_estimators': 500,
        'learning_rate': 0.1
    }
)

model = factory.create_model(config)
```

### Direct Tree CLVSA Wrapper Usage

```python
from src.training.steps.model_training.tree_clvsa_wrapper import (
    TreeCLVSAWrapper, TreeCLVSAConfig, create_tree_clvsa_wrapper
)
from sklearn.ensemble import RandomForestRegressor

# Create base model
base_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Create CLVSA configuration
clvsa_config = TreeCLVSAConfig(
    attention_dim=64,
    use_temporal_attention=True,
    regime_aware=True,
    ensemble_attention=True,
    feature_selection_method='mutual_info',
    temporal_window_size=20,
    memory_efficient=True
)

# Wrap with CLVSA
clvsa_model = TreeCLVSAWrapper(base_model, clvsa_config)

# Train with regime information
X, y, regimes = your_data
clvsa_model.fit(X, y, regimes=regimes)

# Make predictions
predictions = clvsa_model.predict(X_test, regimes=regime_test)
```

## CLVSA Features

### 1. Feature Attention
- **Mutual Information**: Measures feature-target relationships
- **Tree Importance**: Uses RandomForest feature importance
- **Correlation-based**: Identifies highly correlated features
- **Automatic Selection**: Best method chosen based on data characteristics

### 2. Temporal Attention
- **Autocorrelation Analysis**: Identifies temporal patterns
- **Rolling Window Variance**: Captures volatility changes
- **Adaptive Window Size**: Automatically adjusts to data length
- **Time Series Optimization**: Specifically designed for financial data

### 3. Regime-Aware Attention
- **Market Regime Detection**: Adapts to different market conditions
- **Regime-Specific Weights**: Different attention for each regime
- **Dynamic Adaptation**: Updates weights based on current regime
- **Economic Significance**: Focuses on economically relevant features

### 4. Ensemble Attention
- **Cross-Validation Stability**: Uses multiple CV folds for robust weights
- **Performance Weighting**: Weights by model performance
- **Tree-Specific Optimization**: Designed for ensemble tree models
- **Memory Efficiency**: Optimized for large datasets

## Configuration Parameters

### TreeCLVSAConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `attention_dim` | int | 64 | Attention dimension for feature weighting |
| `use_temporal_attention` | bool | True | Enable temporal attention patterns |
| `regime_aware` | bool | True | Enable regime-aware attention |
| `attention_dropout` | float | 0.1 | Attention dropout rate |
| `feature_selection_method` | str | 'mutual_info' | Feature selection method |
| `temporal_window_size` | int | 20 | Window size for temporal attention |
| `regime_embedding_dim` | int | 16 | Dimension for regime embeddings |
| `ensemble_attention` | bool | True | Enable ensemble-specific attention |
| `memory_efficient` | bool | True | Use memory-efficient implementations |

### Feature Selection Methods

1. **'mutual_info'**: Mutual information between features and target
2. **'tree_importance'**: RandomForest feature importance
3. **'correlation'**: Correlation-based feature selection

## Performance Benefits

### 1. Enhanced Accuracy
- **Attention Weighting**: Focuses on most relevant features
- **Temporal Patterns**: Captures time series dependencies
- **Regime Adaptation**: Adapts to market conditions
- **Ensemble Optimization**: Leverages tree ensemble strengths

### 2. Improved Interpretability
- **Feature Importance**: Clear attention weights for each feature
- **Temporal Insights**: Understanding of time-based patterns
- **Regime Analysis**: Market condition-specific feature importance
- **Attention Visualization**: Easy to interpret attention patterns

### 3. Robustness
- **Cross-Validation**: Stable attention weights across folds
- **Regime Adaptation**: Handles changing market conditions
- **Memory Efficiency**: Optimized for large datasets
- **Error Handling**: Graceful fallbacks for edge cases

## Integration with Existing Code

### Model Factory Integration
The model factory automatically applies CLVSA to all tree models:

```python
# All these models are automatically wrapped with CLVSA
models = [
    ModelType.RANDOM_FOREST,
    ModelType.LIGHTGBM,
    ModelType.XGBOOST,
    ModelType.CATBOOST,
    ModelType.EXTRA_TREES,
    ModelType.HIST_GRADIENT_BOOSTING
]

for model_type in models:
    config = ModelConfig(model_type=model_type, model_name=f"test_{model_type.value}")
    model = factory.create_model(config)  # Automatically wrapped with CLVSA
```

### Ensemble Integration
CLVSA-wrapped models work seamlessly with ensemble methods:

```python
from sklearn.ensemble import VotingRegressor

# Create multiple CLVSA-wrapped models
rf_model = factory.create_model(ModelConfig(model_type=ModelType.RANDOM_FOREST, model_name="rf"))
xgb_model = factory.create_model(ModelConfig(model_type=ModelType.XGBOOST, model_name="xgb"))
lgb_model = factory.create_model(ModelConfig(model_type=ModelType.LIGHTGBM, model_name="lgb"))

# Create ensemble
ensemble = VotingRegressor([
    ('rf', rf_model),
    ('xgb', xgb_model),
    ('lgb', lgb_model)
])
```

## Monitoring and Analysis

### Attention Weights Analysis
```python
# Get attention weights
attention_weights = model.get_attention_weights()

print("Feature Attention:", attention_weights['feature_attention'])
print("Temporal Attention:", attention_weights['temporal_attention'])
print("Regime Attention:", attention_weights['regime_attention'])
print("Ensemble Attention:", attention_weights['ensemble_attention'])
```

### Performance Monitoring
```python
# Get attention performance metrics
attention_performance = attention_weights['attention_performance']

print("Feature Attention Entropy:", attention_performance['feature_attention_entropy'])
print("Training Time:", attention_performance['training_time'])
print("Regime Attention Count:", attention_performance['regime_attention_count'])
```

## Best Practices

### 1. Data Preparation
- **Feature Scaling**: Use RobustScaler for financial data
- **Regime Labels**: Provide regime information when available
- **Temporal Order**: Maintain temporal order for time series data
- **Missing Values**: Handle missing values before CLVSA processing

### 2. Configuration Tuning
- **Attention Dimension**: Start with 64, increase for complex data
- **Temporal Window**: Adjust based on data frequency
- **Memory Efficiency**: Enable for large datasets
- **Feature Selection**: Use 'mutual_info' for most cases

### 3. Performance Optimization
- **Batch Processing**: Process data in batches for large datasets
- **Memory Management**: Use memory-efficient mode for large data
- **Parallel Processing**: Leverage n_jobs for tree models
- **Early Stopping**: Use early stopping for gradient boosting models

### 4. Monitoring and Debugging
- **Attention Weights**: Monitor attention weight distributions
- **Performance Metrics**: Track training time and accuracy
- **Regime Analysis**: Analyze regime-specific performance
- **Error Handling**: Check for convergence issues

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Enable `memory_efficient=True`
   - Reduce `attention_dim`
   - Use smaller batch sizes

2. **Performance Issues**
   - Check feature selection method
   - Verify regime labels
   - Monitor attention weights

3. **Convergence Issues**
   - Adjust learning rates
   - Check data quality
   - Verify feature scaling

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Create model with debug info
config = ModelConfig(
    model_type=ModelType.RANDOM_FOREST,
    model_name="debug_model",
    model_params={'use_clvsa': True}
)

model = factory.create_model(config)
```

## Future Enhancements

### Planned Features
1. **Multi-timeframe Attention**: Attention across different timeframes
2. **Dynamic Regime Detection**: Automatic regime detection
3. **Attention Visualization**: Interactive attention weight visualization
4. **Hyperparameter Optimization**: Automatic CLVSA parameter tuning

### Research Directions
1. **Attention Mechanisms**: Advanced attention architectures
2. **Regime Modeling**: Improved regime detection methods
3. **Temporal Modeling**: Enhanced temporal attention patterns
4. **Ensemble Methods**: Advanced ensemble attention techniques

## Conclusion

The CLVSA integration with tree models provides a powerful enhancement to the existing machine learning pipeline. By automatically wrapping all tree models with CLVSA architecture, we achieve:

- **Enhanced Performance**: Better accuracy through attention mechanisms
- **Improved Interpretability**: Clear understanding of feature importance
- **Robustness**: Adaptation to different market conditions
- **Ease of Use**: Automatic integration without code changes

The system is designed to be backward compatible, allowing users to disable CLVSA when needed, while providing significant performance improvements when enabled.