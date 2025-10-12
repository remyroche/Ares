# VectorBT-Enhanced Models Documentation

This document provides comprehensive documentation for the enhanced PatchTST, GRU, and TFT models with VectorBT integration.

## Overview

The VectorBT-enhanced models provide a powerful combination of state-of-the-art time series forecasting models with VectorBT's high-performance backtesting, financial metrics, and feature generation capabilities.

## Enhanced Models

### 1. PatchTST with VectorBT Integration (`src/models/enhanced_patchtst.py`)

**Features:**
- Patch-based time series transformation with transformer architecture
- VectorBT backtesting engine integration
- VectorBT financial metrics calculation
- VectorBT feature generation
- Memory management and performance monitoring

**Key Components:**
- `EnhancedPatchTSTConfig`: Configuration with VectorBT settings
- `EnhancedPatchTSTModel`: Main model class with VectorBT integration
- `PatchEmbedding`: Patch embedding layer
- `MultiHeadAttention`: Multi-head attention mechanism
- `TransformerBlock`: Transformer block with attention and feed-forward

**Usage:**
```python
from src.models.enhanced_patchtst import EnhancedPatchTSTModel, EnhancedPatchTSTConfig

# Create configuration
config = EnhancedPatchTSTConfig(
    lookback_hours=16,
    d_model=96,
    heads=3,
    layers=2,
    enable_vectorbt=True,
    enable_vectorbt_backtesting=True,
    enable_vectorbt_metrics=True,
    enable_vectorbt_features=True
)

# Create and fit model
model = EnhancedPatchTSTModel(config)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Generate VectorBT features
vectorbt_features = model.generate_vectorbt_features(ohlcv_data)

# Run VectorBT backtest
backtest_results = model.run_vectorbt_backtest(signals, prices, timestamps)

# Calculate VectorBT metrics
metrics = model.calculate_vectorbt_metrics(portfolio_values, returns)
```

### 2. GRU with VectorBT Integration (`src/models/patch_gru.py`)

**Features:**
- Gated Recurrent Unit architecture
- VectorBT backtesting and financial metrics
- VectorBT feature generation
- Memory optimization and performance monitoring

**Key Components:**
- `PatchConfig`: Configuration with VectorBT settings
- `BasePatchModel`: Abstract base class with VectorBT functionality
- `SimpleGRU`: GRU implementation with VectorBT integration
- `SimplePatchTST`: PatchTST implementation with VectorBT integration
- `PatchOrchestrator`: Orchestrator for patch model training and prediction

**Usage:**
```python
from src.models.patch_gru import PatchOrchestrator, PatchConfig, ModelType

# Create configuration
config = PatchConfig(
    model_type=ModelType.GRU,
    sequence_length=24,
    horizons=[1, 3],
    enable_vectorbt=True,
    enable_vectorbt_backtesting=True,
    enable_vectorbt_metrics=True,
    enable_vectorbt_features=True
)

# Create and fit model
model = PatchOrchestrator(config)
model.fit(bars_data, targets)

# Make predictions
output = model.predict(bars_data)

# Generate VectorBT features
vectorbt_features = model.generate_vectorbt_features(ohlcv_data)

# Run VectorBT backtest
backtest_results = model.run_vectorbt_backtest(signals, prices, timestamps)
```

### 3. TFT with VectorBT Integration (`src/models/enhanced_tft.py`)

**Features:**
- Temporal Fusion Transformer architecture
- Multi-horizon forecasting capabilities
- VectorBT backtesting and financial metrics
- VectorBT feature generation
- Memory management and performance monitoring

**Key Components:**
- `EnhancedTFTConfig`: Configuration with VectorBT settings
- `EnhancedTFTModel`: Main model class with VectorBT integration
- `TemporalFusionTransformer`: TFT implementation

**Usage:**
```python
from src.models.enhanced_tft import EnhancedTFTModel, EnhancedTFTConfig

# Create configuration
config = EnhancedTFTConfig(
    hidden_size=64,
    sequence_length=24,
    prediction_horizon=1,
    enable_vectorbt=True,
    enable_vectorbt_backtesting=True,
    enable_vectorbt_metrics=True,
    enable_vectorbt_features=True
)

# Create and fit model
model = EnhancedTFTModel(config)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Generate VectorBT features
vectorbt_features = model.generate_vectorbt_features(ohlcv_data)

# Run VectorBT backtest
backtest_results = model.run_vectorbt_backtest(signals, prices, timestamps)
```

### 4. Unified Interface (`src/models/vectorbt_enhanced_models.py`)

**Features:**
- Common interface for all enhanced models
- Unified configuration and API
- VectorBT integration across all models
- Performance monitoring and statistics

**Key Components:**
- `UnifiedModelConfig`: Unified configuration for all models
- `VectorBTEnhancedModelInterface`: Common interface class
- Factory functions for creating models

**Usage:**
```python
from src.models.vectorbt_enhanced_models import (
    create_patchtst_model,
    create_gru_model,
    create_tft_model,
    create_all_models
)

# Create individual models
patchtst_model = create_patchtst_model(sequence_length=24, hidden_size=64)
gru_model = create_gru_model(sequence_length=24, hidden_size=64)
tft_model = create_tft_model(sequence_length=24, hidden_size=64)

# Create all models at once
all_models = create_all_models(sequence_length=24, hidden_size=64)

# Use unified interface
for model_name, model in all_models.items():
    model.fit(X, y)
    predictions = model.predict(X)
    vectorbt_features = model.generate_vectorbt_features(ohlcv_data)
    backtest_results = model.run_vectorbt_backtest(signals, prices, timestamps)
```

## VectorBT Integration Features

### 1. Backtesting Engine

**Capabilities:**
- High-performance portfolio simulation
- Multiple execution modes (CPU, GPU, Parallel, Hybrid)
- Comprehensive performance metrics
- Risk analysis and drawdown assessment
- Memory optimization for large datasets

**Usage:**
```python
# Run backtest
backtest_results = model.run_vectorbt_backtest(
    signals=signals,
    prices=prices,
    timestamps=timestamps,
    mode='cpu'  # or 'gpu', 'parallel', 'hybrid'
)

# Access results
performance_metrics = backtest_results['performance_metrics']
risk_metrics = backtest_results['risk_metrics']
drawdown_analysis = backtest_results['drawdown_analysis']
```

### 2. Financial Metrics

**Capabilities:**
- 50+ financial performance metrics
- Risk-adjusted return calculations
- Drawdown analysis and recovery metrics
- Regime-aware performance analysis
- Benchmark comparison utilities

**Usage:**
```python
# Calculate comprehensive metrics
metrics = model.calculate_vectorbt_metrics(
    portfolio_values=portfolio_values,
    returns=returns,
    benchmark_values=benchmark_values,
    timestamps=timestamps
)

# Access specific metrics
sharpe_ratio = metrics['sharpe_ratio']
max_drawdown = metrics['max_drawdown']
volatility = metrics['volatility']
```

### 3. Feature Generation

**Capabilities:**
- VectorBT-optimized technical indicators
- Volatility, momentum, and trend features
- Memory-efficient processing
- GPU acceleration support
- Batch processing capabilities

**Usage:**
```python
# Generate VectorBT features
vectorbt_features = model.generate_vectorbt_features(ohlcv_data)

# Access generated features
print(f"Generated {vectorbt_features.shape[1]} features")
print(f"Feature names: {list(vectorbt_features.columns)}")
```

### 4. Performance Monitoring

**Capabilities:**
- Memory usage tracking
- Performance statistics
- Operation monitoring
- Cache hit rate analysis
- Error rate tracking

**Usage:**
```python
# Get performance statistics
stats = model.get_vectorbt_stats()

# Access specific stats
memory_usage = stats['memory_usage_gb']
total_operations = stats['total_operations_monitored']
cache_hit_rate = stats['cache_hit_rate']
```

## Configuration Options

### VectorBT Integration Parameters

```python
# Enable/disable VectorBT features
enable_vectorbt: bool = True
enable_vectorbt_backtesting: bool = True
enable_vectorbt_metrics: bool = True
enable_vectorbt_features: bool = True
enable_memory_optimization: bool = True
enable_performance_monitoring: bool = True

# Performance settings
memory_limit_gb: float = 8.0
enable_gpu: bool = False
enable_parallel: bool = True
chunk_size: int = 1000

# VectorBT configurations
vectorbt_backtest_config: Optional[VectorBTBacktestConfig] = None
vectorbt_metrics_config: Optional[FinancialMetricsConfig] = None
```

### Model-Specific Parameters

**PatchTST:**
```python
lookback_hours: int = 16
d_model: int = 96
heads: int = 3
layers: int = 2
export_dims: int = 10
patch_len: int = 16
stride: int = 8
```

**GRU:**
```python
sequence_length: int = 24
horizons: List[int] = [1, 3]
hidden_dim: int = 32
num_layers: int = 1
dropout: float = 0.1
```

**TFT:**
```python
hidden_size: int = 64
sequence_length: int = 24
prediction_horizon: int = 1
lstm_layers: int = 2
attention_heads: int = 4
dropout: float = 0.1
```

## Example Usage

See `examples/vectorbt_enhanced_models_example.py` for comprehensive examples demonstrating:

1. Individual model usage
2. Unified interface usage
3. VectorBT backtesting
4. Financial metrics calculation
5. Feature generation
6. Performance monitoring
7. Advanced configuration options

## Dependencies

**Required:**
- numpy
- pandas
- scikit-learn
- torch (for neural network models)

**Optional (for VectorBT features):**
- vectorbt
- cupy (for GPU acceleration)

**VectorBT Utils (if available):**
- `src.utils.ml_common.vectorbt_backtesting_engine`
- `src.utils.ml_common.vectorbt_financial_metrics`
- `src.feature_generation.core.vectorbt_feature_generator`
- `src.utils.ml_common.vectorbt_memory_manager`
- `src.utils.ml_common.vectorbt_performance_monitor`

## Performance Considerations

1. **Memory Management**: Use `enable_memory_optimization=True` for large datasets
2. **GPU Acceleration**: Set `enable_gpu=True` if CUDA is available
3. **Parallel Processing**: Use `enable_parallel=True` for multi-core systems
4. **Chunk Size**: Adjust `chunk_size` based on available memory
5. **Memory Limit**: Set appropriate `memory_limit_gb` for your system

## Troubleshooting

**Common Issues:**

1. **VectorBT not available**: Some features will be disabled, but models will still work
2. **Memory errors**: Reduce `memory_limit_gb` or `chunk_size`
3. **GPU errors**: Set `enable_gpu=False` if CUDA issues occur
4. **Import errors**: Ensure all VectorBT utils are properly installed

**Debug Mode:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When adding new features or models:

1. Follow the existing pattern for VectorBT integration
2. Add comprehensive configuration options
3. Include performance monitoring
4. Add memory management
5. Update documentation
6. Add example usage

## License

This code is part of the larger project and follows the same licensing terms.