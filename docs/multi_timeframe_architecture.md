# Multi-Timeframe Architecture

## Overview

The Ares trading system uses a dual-model architecture with different timeframes for different purposes:

## Model Architecture

### Analyst Model
- **Purpose**: Strategic decisions (should we enter a trade?)
- **Timeframes**: 5m, 15m, 30m
- **Features**: Multi-timeframe technical indicators, autoencoder features
- **Training**: Trained on regime-specific data with comprehensive features

### Tactician Model  
- **Purpose**: Tactical execution (when exactly to enter/exit?)
- **Timeframe**: 1m (high-frequency precision)
- **Features**: 1m technical indicators, autoencoder features, immediate volatility
- **Training**: Trained on 1m data with tactician-specific triple barrier labels

## Feature Engineering

### Analyst Features (5m, 15m, 30m)
```
5m_sma_20, 5m_rsi, 5m_macd, 5m_volatility
15m_sma_20, 15m_rsi, 15m_macd, 15m_volatility  
30m_sma_20, 30m_rsi, 30m_macd, 30m_volatility
```

### Tactician Features (1m)
```
sma_5, sma_10, sma_20, ema_12, ema_26
rsi, macd, macd_signal, bb_position, atr
price_momentum_1, price_momentum_5
volatility, volatility_5, volatility_10
```

### Shared Features
```
autoencoder_feature_1, autoencoder_feature_2, ...
hour_sin, hour_cos, is_asia_session, is_london_session
```

## Configuration

### Enable Multi-timeframe Features
```python
# In config_optuna.py
enable_multi_timeframe_features: bool = True
enable_autoencoder_features: bool = True
```

### Advanced Feature Engineering
```python
# In advanced_feature_engineering.py
self.timeframes = ["5m", "15m", "30m"]  # Analyst only
```

### Tactician Feature Selection
```python
# In step8_tactician_labeling.py
# Excludes analyst timeframes: 5m_, 15m_, 30m_
# Includes 1m features and autoencoder features
```

## Benefits

1. **Strategic Clarity**: Analyst focuses on longer-term market conditions
2. **Tactical Precision**: Tactician executes with 1m precision
3. **Reduced Overfitting**: Models trained on appropriate timeframes for their purpose
4. **Autoencoder Patterns**: Shared pattern recognition across all timeframes
5. **Computational Efficiency**: Each model uses relevant features only