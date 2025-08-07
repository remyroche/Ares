# Wavelet Transform Integration Guide

## Overview

This document provides a comprehensive guide to the complete wavelet transform integration in the trading system. The integration ensures:

1. **All features from `advanced_feature_engineering.py` & `feature_engineering_orchestrator.py`** (except Autoencoder) are integrated into `src/training/steps/vectorized_advanced_feature_engineering.py`
2. **Price differences are used instead of raw prices** for all wavelet analysis
3. **Complete wavelet workflow** is integrated (`wavelet_caching_workflow.py`, `steps/vectorized_advanced_feature_engineering.py`, `steps/precompute_wavelet_features.py`, `steps/backtesting_with_cached_features.py`)
4. **Extensive wavelet techniques** are implemented for labelling and ML training
5. **Live trading integration** uses wavelet features along with other indicators

## Key Components

### 1. Vectorized Advanced Feature Engineering (`src/training/steps/vectorized_advanced_feature_engineering.py`)

This is the core component that integrates all features from the original advanced feature engineering files.

#### Features Integrated:

- **Wavelet Transform Analyzer**: Comprehensive wavelet analysis using price differences
- **Volatility Regime Model**: GARCH and other volatility modeling techniques
- **Correlation Analyzer**: Multi-timeframe correlation analysis
- **Momentum Analyzer**: Momentum indicators and analysis
- **Liquidity Analyzer**: Market liquidity analysis
- **Candlestick Pattern Analyzer**: All major candlestick patterns
- **SR Distance Calculator**: Support/resistance distance calculations
- **Microstructure Features**: Market microstructure analysis
- **Adaptive Indicators**: Adaptive technical indicators

#### Price Differences Usage:

```python
def _prepare_stationary_series(self, price_data: pd.DataFrame) -> dict[str, np.ndarray]:
    """Prepare stationary series for wavelet analysis using price differences."""
    stationary_series = {}
    
    # Price differences (first difference) - primary focus
    price_diff = price_data["close"].diff().dropna().values
    stationary_series["price_diff"] = price_diff
    
    # Second differences (acceleration)
    price_diff_2 = price_data["close"].diff().diff().dropna().values
    stationary_series["price_diff_2"] = price_diff_2
    
    # Use price differences as primary series (not raw prices)
    stationary_series["close"] = price_diff
```

### 2. Wavelet Feature Cache (`WaveletFeatureCache`)

Comprehensive caching system for expensive wavelet calculations:

- **Cache Format**: Parquet files for fast loading
- **Compression**: Snappy compression for efficiency
- **Metadata**: Cache metadata for validation
- **Integrity Validation**: Cache integrity checks
- **Parallel Processing**: Optional parallel caching

### 3. Wavelet Feature Precomputer (`src/training/steps/precompute_wavelet_features.py`)

Pre-computation system for wavelet features:

- **Batch Processing**: Process large datasets efficiently
- **Progress Tracking**: Real-time progress monitoring
- **Parallel Processing**: Multi-worker processing
- **Output Management**: Organized output structure

### 4. Backtesting with Cached Features (`src/training/steps/backtesting_with_cached_features.py`)

Fast backtesting using pre-computed wavelet features:

- **Cache Lookup**: Fast feature retrieval
- **Performance Monitoring**: Backtesting performance metrics
- **Strategy Integration**: Easy strategy integration
- **Multiple Backtests**: Support for multiple backtest configurations

## Complete Workflow Integration

### Workflow Components:

1. **`wavelet_caching_workflow.py`**: Main workflow orchestrator
2. **`steps/vectorized_advanced_feature_engineering.py`**: Core feature engineering
3. **`steps/precompute_wavelet_features.py`**: Feature pre-computation
4. **`steps/backtesting_with_cached_features.py`**: Fast backtesting

### Workflow Steps:

```python
# Step 1: Pre-compute wavelet features
precomputer = WaveletFeaturePrecomputer(config)
await precomputer.precompute_dataset(data_path, symbol)

# Step 2: Run backtesting with cached features
backtester = BacktestingWithCachedFeatures(config)
results = await backtester.run_backtest(price_data, volume_data)

# Step 3: Live trading integration
feature_engineer = VectorizedAdvancedFeatureEngineering(config)
live_features = await feature_engineer.engineer_features(latest_data)
```

## Extensive Wavelet Techniques

### Wavelet Transform Types:

1. **Discrete Wavelet Transform (DWT)**:
   - Multi-level decomposition
   - Energy analysis at each level
   - Boundary effect handling

2. **Continuous Wavelet Transform (CWT)**:
   - Scale-frequency analysis
   - Dynamic scale selection
   - Frequency domain features

3. **Wavelet Packet Analysis**:
   - Complete signal decomposition
   - Optimal basis selection
   - Dimensionality control

4. **Wavelet Denoising**:
   - Signal denoising
   - Threshold selection
   - Quality preservation

5. **Multi-Wavelet Analysis**:
   - Multiple wavelet types
   - Feature selection
   - Ensemble methods

### Feature Categories:

```python
# Wavelet features by category
dwt_features = {k: v for k, v in features.items() if 'dwt' in k.lower()}
cwt_features = {k: v for k, v in features.items() if 'cwt' in k.lower()}
packet_features = {k: v for k, v in features.items() if 'packet' in k.lower()}
denoising_features = {k: v for k, v in features.items() if 'denoising' in k.lower()}
```

## Live Trading Integration

### Real-time Feature Generation:

```python
# Live trading feature generation
live_features = await feature_engineer.engineer_features(
    latest_price_data, latest_volume_data
)

# Wavelet features for live trading
live_wavelet_features = {k: v for k, v in live_features.items() 
                        if 'wavelet' in k.lower()}

# Trading decision based on wavelet energy
energy_features = {k: v for k, v in live_wavelet_features.items() 
                  if 'energy' in k.lower()}
avg_energy = np.mean(list(energy_features.values()))

if avg_energy > threshold:
    # High wavelet energy - potential trading opportunity
    execute_trade()
```

### Integration with Other Indicators:

The wavelet features are integrated with:
- Technical indicators (SMA, EMA, RSI, etc.)
- Volume analysis
- Market microstructure features
- Candlestick patterns
- Support/resistance levels

## Configuration

### Wavelet Configuration:

```python
wavelet_config = {
    "wavelet_type": "db4",
    "decomposition_level": 4,
    "enable_discrete_wavelet": True,
    "enable_continuous_wavelet": True,
    "enable_wavelet_packet": True,
    "enable_denoising": True,
    "max_wavelet_types": 3,
    "enable_stationary_series": True,
    "stationary_transforms": ["price_diff", "returns", "log_returns"],
}
```

### Cache Configuration:

```python
cache_config = {
    "cache_enabled": True,
    "cache_dir": "data/wavelet_cache",
    "cache_format": "parquet",
    "compression": "snappy",
    "validate_cache_integrity": True,
    "cache_expiry_days": 30,
}
```

## Performance Optimization

### Vectorized Operations:

All wavelet operations use vectorized numpy operations for optimal performance:

```python
# Vectorized price difference calculation
price_diff = price_data["close"].diff().values

# Vectorized wavelet analysis
coeffs = pywt.wavedec(price_diff, wavelet_type, level=decomposition_level)

# Vectorized feature extraction
energy = np.sum(coeffs[level] ** 2)
```

### Caching Strategy:

- **Pre-computation**: Expensive calculations done once
- **Fast Loading**: Parquet format for quick access
- **Memory Efficiency**: Compression and chunking
- **Parallel Processing**: Multi-worker support

## Usage Examples

### Basic Usage:

```python
# Initialize feature engineering
feature_engineer = VectorizedAdvancedFeatureEngineering(config)
await feature_engineer.initialize()

# Engineer features
features = await feature_engineer.engineer_features(price_data, volume_data)

# Access wavelet features
wavelet_features = {k: v for k, v in features.items() if 'wavelet' in k.lower()}
```

### Advanced Usage:

```python
# Complete workflow
demo = WaveletIntegrationDemo(config)
await demo.initialize()
await demo.run_complete_demo()

# Live trading
live_features = await feature_engineer.engineer_features(latest_data)
trading_decision = analyze_wavelet_features(live_features)
```

## Testing and Validation

### Demo Script:

Run the comprehensive demo:

```bash
python src/training/wavelet_integration_demo.py
```

This demonstrates:
- Price differences usage
- Complete feature integration
- Wavelet workflow
- Live trading integration
- Extensive wavelet techniques

### Validation Checks:

1. **Feature Count**: Verify all expected features are generated
2. **Price Differences**: Confirm price differences are used instead of raw prices
3. **Cache Performance**: Monitor cache hit rates and performance
4. **Live Trading**: Test real-time feature generation
5. **Wavelet Techniques**: Validate all wavelet transform types

## Troubleshooting

### Common Issues:

1. **Cache Misses**: Check cache configuration and data consistency
2. **Performance Issues**: Verify vectorized operations and parallel processing
3. **Memory Usage**: Monitor memory usage with large datasets
4. **Feature Quality**: Validate wavelet feature selection and quality

### Debugging:

```python
# Enable debug logging
import logging
logging.getLogger("WaveletFeatureCache").setLevel(logging.DEBUG)

# Check cache statistics
cache_stats = wavelet_cache.get_cache_stats()
print(f"Cache stats: {cache_stats}")

# Validate features
feature_count = len(features)
print(f"Generated {feature_count} features")
```

## Conclusion

The wavelet transform integration provides:

✅ **Complete Feature Integration**: All features from advanced_feature_engineering.py and feature_engineering_orchestrator.py are integrated

✅ **Price Differences Usage**: All wavelet analysis uses price differences instead of raw prices

✅ **Complete Workflow**: Full wavelet workflow from pre-computation to live trading

✅ **Extensive Techniques**: Multiple wavelet transform types and analysis methods

✅ **Live Trading Ready**: Real-time feature generation for live trading

✅ **Performance Optimized**: Vectorized operations and efficient caching

This integration ensures that the trading system has access to the most advanced wavelet-based features while maintaining high performance and reliability for both backtesting and live trading scenarios.