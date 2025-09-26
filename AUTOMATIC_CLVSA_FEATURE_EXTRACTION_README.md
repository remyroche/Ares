# Automatic CLVSA Feature Extraction

## Overview

The Automatic CLVSA Feature Extraction system provides seamless integration of advanced attention mechanisms with any machine learning pipeline. This system automatically enhances features using CLVSA (Cross-View Learning with Self-Attention) architecture and caches results for efficient reuse across models.

## Features

### 🔄 Automatic Enhancement
- **Seamless Integration**: Tree-based models are automatically wrapped with CLVSA enhancement
- **Smart Feature Extraction**: Automatically detects and enhances financial data with technical indicators
- **Intelligent Caching**: CVLSA computations are cached and reused across different models
- **Memory Optimization**: Advanced memory management with GPU support and tensor pooling

### 📊 Enhanced Capabilities
- **Multi-Modal Attention**: Cross-view attention between different data modalities
- **Temporal Modeling**: Multi-scale temporal attention for time series data
- **Technical Indicators**: Automatic addition of RSI, MACD, Bollinger Bands, moving averages
- **Feature Expansion**: Original features + CVLSA predictions + technical indicators

### ⚙️ Configurable Options
- **Enhancement Levels**: Basic, Comprehensive, and Advanced configurations
- **Fusion Methods**: Attention-based, weighted average, and stacking fusion
- **Cache Management**: Configurable cache size and memory limits
- **Hardware Acceleration**: GPU support and memory optimization

## Quick Start

### Automatic Enhancement (Recommended)

```python
from src.utils.ml_common.models.model_factory import EnhancedModelFactory, ModelConfig, ModelType

# Create model factory - CLVSA enhancement is enabled by default
factory = EnhancedModelFactory()

# Create any tree model - it will be automatically enhanced with CLVSA
config = ModelConfig(
    model_type=ModelType.RANDOM_FOREST,
    model_name="clvsa_enhanced_rf",
    model_params={
        'n_estimators': 100,
        'max_depth': 10
    }
)

# Model is automatically wrapped with CLVSA
model = factory.create_model(config)

# Train with enhanced features
model.fit(X_train, y_train)

# Predict with enhanced features
predictions = model.predict(X_test)
```

### Using Feature Extractor Directly

```python
from src.utils.ml_common.cvlsa.cvlsa_integration import create_feature_enhancer

# Create automatic feature enhancer
enhancer = create_feature_enhancer(
    auto_detect=True,
    enhancement_level='comprehensive'
)

# Automatically enhance features
X_enhanced = enhancer.fit_transform(X_train, y_train)

# Transform new data with same enhancement
X_test_enhanced = enhancer.transform(X_test)

# Use enhanced features with any model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_enhanced, y_train)
predictions = model.predict(X_test_enhanced)
```

### Advanced Configuration

```python
from src.utils.ml_common.cvlsa.cvlsa_integration import create_automatic_feature_pipeline, EnhancedCVLSAConfig

# Create custom CVLSA configuration
cvlsa_config = EnhancedCVLSAConfig(
    view_embedding_dim=128,
    cross_attention_heads=16,
    temporal_attention_heads=16,
    memory_efficient=False  # Use more memory for better performance
)

# Create feature pipeline with custom config
pipeline = create_automatic_feature_pipeline(cvlsa_config)

# Configure cache
pipeline.set_cache_config(max_cache_size=100, memory_limit_mb=500.0)

# Use with any model
enhanced_features = pipeline.fit_transform(X_train, y_train)
```

## Configuration Options

### Enhancement Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| **Basic** | Lightweight enhancement with smaller models | Quick prototyping |
| **Comprehensive** | Balanced performance and speed | Most use cases |
| **Advanced** | Maximum performance with larger models | High-accuracy requirements |

### Model Factory Configuration

```python
config = ModelConfig(
    model_type=ModelType.XGBOOST,
    model_name="custom_xgb",
    model_params={
        # CLVSA configuration
        'clvsa_config': {
            'attention_dim': 128,
            'fusion_method': 'attention',
            'cvlsa_weight': 0.7,
            'memory_efficient': True,
            'use_temporal_attention': True
        },

        # Model parameters
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.1
    }
)
```

### Feature Enhancer Configuration

```python
enhancer = create_feature_enhancer(
    auto_detect=True,           # Auto-detect market data
    enhancement_level='comprehensive'
)

# Configure cache
enhancer.set_cache_config(
    max_cache_size=100,        # Maximum cached entries
    memory_limit_mb=500.0      # Memory limit in MB
)

# Enable/disable enhancement
enhancer.enable_enhancement(True)
```

## Technical Details

### Feature Enhancement Process

1. **Input Analysis**: Analyze input data to detect financial patterns
2. **Market Data Enhancement**: Add technical indicators (RSI, MACD, Bollinger Bands, etc.)
3. **CVLSA Processing**: Apply cross-view and temporal attention mechanisms
4. **Feature Combination**: Combine original features with CVLSA predictions
5. **Caching**: Store enhanced features for reuse across models

### Cache System

- **Memory Cache**: Fast in-memory storage with LRU eviction
- **Disk Cache**: Persistent storage with compression
- **GPU Memory**: Automatic tensor movement to GPU when beneficial
- **Memory Pooling**: Tensor reuse to reduce allocation overhead

### Performance Optimization

- **Automatic GPU Detection**: Uses CUDA when available and beneficial
- **Memory Pressure Management**: Automatic cleanup when memory limits are reached
- **Background Cleanup**: Continuous cache maintenance
- **Compression**: Optional compression for disk storage

## API Reference

### TreeCLVSAWrapper

```python
class TreeCLVSAWrapper:
    def __init__(self, base_model, config: TreeCLVSAConfig)
    def fit(self, X, y, market_data=None, regimes=None)
    def predict(self, X, market_data=None)
    def predict_proba(self, X, market_data=None)  # For classifiers
    def get_feature_importance(self)
    def get_model_info(self)
    def get_cache_stats(self)
    def clear_cache(self)
    def set_cache_config(self, max_cache_size, memory_limit_mb)
    def enable_cvlsa_enhancement(self, enable)
    def set_fusion_method(self, method)
    def set_cvlsa_weight(self, weight)
    def get_enhancement_summary(self)
```

### CVLSAFeatureExtractor

```python
class CVLSAFeatureExtractor:
    def __init__(self, cvlsa_config=None)
    def fit(self, X, y, market_data=None)
    def transform(self, X, market_data=None)
    def fit_transform(self, X, y, market_data=None)
    def get_feature_importance(self)
    def get_cache_stats(self)
    def clear_cache(self)
    def set_cache_config(self, max_cache_size, memory_limit_mb)
```

### AutomaticCVLSAFeaturePipeline

```python
class AutomaticCVLSAFeaturePipeline:
    def __init__(self, cvlsa_config=None)
    def fit_transform(self, X, y=None, market_data=None)
    def transform(self, X, market_data=None)
    def get_feature_importance(self)
    def get_cache_stats(self)
    def clear_cache(self)
    def set_cache_config(self, max_cache_size, memory_limit_mb)
    def enable_auto_enhancement(self, enable)
```

### CVLSAFeatureEnhancer

```python
class CVLSAFeatureEnhancer:
    def __init__(self, auto_detect=True, enhancement_level='comprehensive')
    def fit_transform(self, X, y=None, market_data=None)
    def transform(self, X, market_data=None)
    def get_enhancement_info(self)
    def enable_enhancement(self, enable)
```

## Examples

### Financial Data Analysis

```python
import pandas as pd
import numpy as np
from src.utils.ml_common.cvlsa.cvlsa_integration import create_feature_enhancer

# Load financial data
market_data = pd.read_csv('stock_data.csv')

# Create feature enhancer
enhancer = create_feature_enhancer(
    auto_detect=True,
    enhancement_level='comprehensive'
)

# Enhance features automatically
X_enhanced = enhancer.fit_transform(
    X_train,
    y_train,
    market_data=market_data
)

print(f"Feature enhancement: {X_train.shape[1]} → {X_enhanced.shape[1]} features")
print(f"Enhancement info: {enhancer.get_enhancement_info()}")
```

### Model Comparison

```python
from src.utils.ml_common.models.model_factory import EnhancedModelFactory, ModelConfig, ModelType

factory = EnhancedModelFactory()

# CLVSA-enhanced model
clvsa_config = ModelConfig(
    model_type=ModelType.RANDOM_FOREST,
    model_name="clvsa_rf",
    model_params={'n_estimators': 100}
)

# Standard model
standard_config = ModelConfig(
    model_type=ModelType.RANDOM_FOREST,
    model_name="standard_rf",
    model_params={
        'use_clvsa': False,  # Disable CLVSA
        'n_estimators': 100
    }
)

clvsa_model = factory.create_model(clvsa_config)
standard_model = factory.create_model(standard_config)

# Compare performance
clvsa_model.fit(X_train, y_train)
standard_model.fit(X_train, y_train)

clvsa_score = clvsa_model.score(X_test, y_test)
standard_score = standard_model.score(X_test, y_test)

print(f"CLVSA Model Score: {clvsa_score:.4f}")
print(f"Standard Model Score: {standard_score:.4f}")
print(f"Improvement: {(clvsa_score - standard_score) * 100:.2f}%")
```

## Best Practices

1. **Use Automatic Enhancement**: Always use CLVSA enhancement unless you have specific reasons not to
2. **Monitor Cache Usage**: Check cache statistics periodically to optimize memory usage
3. **Choose Appropriate Enhancement Level**: Start with 'comprehensive' and adjust based on performance needs
4. **Handle Memory Pressure**: Use memory limits and monitor cache eviction rates
5. **Enable Persistence**: Use disk caching for long-running applications

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce cache size or enable memory-efficient mode
2. **Slow Performance**: Check if GPU is being used, adjust enhancement level
3. **Cache Misses**: Verify data consistency across training and prediction
4. **Feature Dimensionality**: Monitor feature expansion and adjust if needed

### Performance Tips

1. Use GPU acceleration when available
2. Enable memory-efficient processing for large datasets
3. Monitor and optimize cache hit rates
4. Choose appropriate enhancement levels based on data size
5. Use feature selection to reduce dimensionality if needed

## Integration with Existing Pipelines

The CLVSA feature extraction system is designed to integrate seamlessly with existing scikit-learn pipelines:

```python
from sklearn.pipeline import Pipeline
from src.utils.ml_common.cvlsa.cvlsa_integration import CVLSAFeatureEnhancer

# Create pipeline with CLVSA enhancement
pipeline = Pipeline([
    ('feature_enhancement', CVLSAFeatureEnhancer()),
    ('model', RandomForestRegressor())
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)
```

This provides a drop-in replacement for any existing ML pipeline with automatic CLVSA enhancement.