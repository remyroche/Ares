# Multi-Timeframe Feature Engineering System

## Overview

The Multi-Timeframe Feature Engineering system provides timeframe-specific feature engineering that adapts technical indicators and other features to the defined timeframes:

- **Execution**: 1m & 5m (combined for ultra-short-term execution)
- **Tactical**: 15m (for tactical decision making)
- **Strategic**: 1h (for macro trend analysis)

The system ensures that indicators are calculated with appropriate parameters for each timeframe's characteristics and trading style.

## Key Principles

1. **Timeframe-Specific Adaptation**: Indicators are adapted to each timeframe's characteristics
2. **Trading Style Optimization**: Parameters are optimized for the trading style of each timeframe
3. **Performance Optimization**: Caching and efficient processing for real-time use
4. **Consistent Interface**: Unified interface across all timeframes

## Architecture

### Components

1. **MultiTimeframeFeatureEngineering** (`src/analyst/multi_timeframe_feature_engineering.py`)
   - Main orchestrator for multi-timeframe feature generation
   - Manages timeframe-specific parameter mappings
   - Handles feature caching and performance optimization

2. **Base Feature Engineering Engine**
   - Uses existing `FeatureEngineeringEngine` for base features
   - Provides foundation for timeframe-specific adaptations

3. **Timeframe Parameter Mappings**
   - Predefined parameter sets for each timeframe
   - Optimized for trading style and timeframe characteristics

### Timeframe Hierarchy

```
Execution Timeframes (1m, 5m)
├── Ultra-short-term indicators
├── Fast response parameters
└── Scalping optimization

Tactical Timeframe (15m)
├── Intraday swing indicators
├── Balanced parameters
└── Momentum optimization

Strategic Timeframe (1h)
├── Macro trend indicators
├── Long-term parameters
└── Trend confirmation
```

## Timeframe-Specific Parameters

### Execution Timeframes (1m & 5m)

**1m - Ultra-short-term execution:**
```python
"1m": {
    "description": "Ultra-short-term execution",
    "trading_style": "scalping",
    "indicators": {
        "rsi": {"length": 7, "description": "Fast RSI for immediate signals"},
        "macd": {"fast": 6, "slow": 12, "signal": 4, "description": "Fast MACD for scalping"},
        "bbands": {"length": 10, "description": "Tight Bollinger Bands"},
        "atr": {"length": 7, "description": "Short ATR for tight stops"},
        "adx": {"length": 7, "description": "Fast ADX for momentum"},
        "stoch": {"length": 7, "description": "Fast Stochastic"},
        "sma": {"lengths": [5, 10, 20], "description": "Short-term moving averages"},
        "ema": {"lengths": [3, 7, 14], "description": "Fast exponential averages"},
        "volume": {"sma_length": 10, "description": "Short volume analysis"},
        "volatility": {"window": 10, "description": "Short-term volatility"},
    }
}
```

**5m - Short-term execution:**
```python
"5m": {
    "description": "Short-term execution",
    "trading_style": "scalping",
    "indicators": {
        "rsi": {"length": 10, "description": "Medium-fast RSI"},
        "macd": {"fast": 8, "slow": 16, "signal": 6, "description": "Medium MACD"},
        "bbands": {"length": 15, "description": "Medium Bollinger Bands"},
        "atr": {"length": 10, "description": "Medium ATR"},
        "adx": {"length": 10, "description": "Medium ADX"},
        "stoch": {"length": 10, "description": "Medium Stochastic"},
        "sma": {"lengths": [8, 15, 30], "description": "Medium-term moving averages"},
        "ema": {"lengths": [5, 10, 20], "description": "Medium exponential averages"},
        "volume": {"sma_length": 15, "description": "Medium volume analysis"},
        "volatility": {"window": 15, "description": "Medium-term volatility"},
    }
}
```

### Tactical Timeframe (15m)

```python
"15m": {
    "description": "Tactical decision making",
    "trading_style": "intraday_swing",
    "indicators": {
        "rsi": {"length": 14, "description": "Standard RSI"},
        "macd": {"fast": 12, "slow": 26, "signal": 9, "description": "Standard MACD"},
        "bbands": {"length": 20, "description": "Standard Bollinger Bands"},
        "atr": {"length": 14, "description": "Standard ATR"},
        "adx": {"length": 14, "description": "Standard ADX"},
        "stoch": {"length": 14, "description": "Standard Stochastic"},
        "sma": {"lengths": [10, 20, 50], "description": "Standard moving averages"},
        "ema": {"lengths": [7, 14, 30], "description": "Standard exponential averages"},
        "volume": {"sma_length": 20, "description": "Standard volume analysis"},
        "volatility": {"window": 20, "description": "Standard volatility"},
    }
}
```

### Strategic Timeframe (1h)

```python
"1h": {
    "description": "Strategic macro trend analysis",
    "trading_style": "swing_trading",
    "indicators": {
        "rsi": {"length": 21, "description": "Long RSI for trend confirmation"},
        "macd": {"fast": 16, "slow": 32, "signal": 12, "description": "Long MACD for trends"},
        "bbands": {"length": 30, "description": "Wide Bollinger Bands"},
        "atr": {"length": 21, "description": "Long ATR for position sizing"},
        "adx": {"length": 21, "description": "Long ADX for trend strength"},
        "stoch": {"length": 21, "description": "Long Stochastic"},
        "sma": {"lengths": [20, 50, 100], "description": "Long-term moving averages"},
        "ema": {"lengths": [14, 30, 60], "description": "Long exponential averages"},
        "volume": {"sma_length": 30, "description": "Long volume analysis"},
        "volatility": {"window": 30, "description": "Long-term volatility"},
    }
}
```

## Feature Categories

### 1. Technical Indicators

**Timeframe-Specific Adaptations:**
- **RSI**: Shorter periods for execution timeframes, longer for strategic
- **MACD**: Faster parameters for scalping, slower for trend following
- **Bollinger Bands**: Tighter bands for execution, wider for strategic
- **ATR**: Shorter periods for tight stops, longer for position sizing
- **ADX**: Faster for momentum, slower for trend strength
- **Stochastic**: Shorter for immediate signals, longer for confirmation

### 2. Volume Indicators

**Adaptations:**
- **Volume SMA**: Shorter periods for execution, longer for strategic
- **Volume Ratio**: Relative to timeframe-specific average
- **Volume Momentum**: Short-term changes for execution
- **OBV**: On Balance Volume for trend confirmation
- **VWAP**: Volume Weighted Average Price

### 3. Volatility Indicators

**Adaptations:**
- **Rolling Volatility**: Window size adapted to timeframe
- **Realized Volatility**: Cumulative volatility measure
- **Volatility Ratio**: Relative to longer-term average

### 4. Momentum Indicators

**Adaptations:**
- **Price Momentum**: Short-term for execution, long-term for strategic
- **Rate of Change**: Adapted periods for timeframe
- **Williams %R**: Fast for execution, slow for confirmation
- **CCI**: Commodity Channel Index

### 5. Trend Indicators

**Adaptations:**
- **SMA/EMA**: Multiple periods adapted to timeframe
- **Price vs MA**: Relative position indicators
- **Trend Strength**: EMA divergence measure

## Configuration

### Multi-Timeframe Feature Engineering Configuration

```python
"multi_timeframe_feature_engineering": {
    "enable_mtf_features": True,  # Enable multi-timeframe features
    "enable_timeframe_adaptation": True,  # Adapt indicators to timeframes
    "cache_duration_minutes": 5,  # Cache features for 5 minutes
    "enable_feature_caching": True,  # Enable feature caching
    "max_cache_size": 50,  # Maximum cache entries
    "timeframe_adaptation_rules": {
        "execution_timeframes": ["1m", "5m"],  # Ultra-short-term
        "tactical_timeframes": ["15m"],  # Tactical decision making
        "strategic_timeframes": ["1h"],  # Strategic macro trend
        "additional_timeframes": ["4h", "1d"]  # Additional timeframes
    }
}
```

## Usage Examples

### Basic Usage

```python
from src.analyst.multi_timeframe_feature_engineering import MultiTimeframeFeatureEngineering

# Initialize
mtf_feature_engine = MultiTimeframeFeatureEngineering(config)

# Generate features for multiple timeframes
features_dict = await mtf_feature_engine.generate_multi_timeframe_features(
    data_dict={
        "1m": data_1m,
        "5m": data_5m,
        "15m": data_15m,
        "1h": data_1h
    },
    agg_trades_dict=agg_trades_dict,
    futures_dict=futures_dict,
    sr_levels=sr_levels
)

# Access features for specific timeframe
features_1m = features_dict["1m"]
features_1h = features_dict["1h"]
```

### Integration with Existing Systems

```python
# In your trading system
async def get_features_for_timeframe(timeframe: str, data: pd.DataFrame):
    # Get timeframe-specific features
    features_dict = await mtf_feature_engine.generate_multi_timeframe_features(
        data_dict={timeframe: data}
    )
    
    if timeframe in features_dict:
        return features_dict[timeframe]
    else:
        return pd.DataFrame()
```

### Parameter Access

```python
# Get parameters for specific timeframe
tf_params = mtf_feature_engine.get_timeframe_parameters("1m")
rsi_length = tf_params["indicators"]["rsi"]["length"]

# Get supported timeframes
supported_timeframes = mtf_feature_engine.get_supported_timeframes()
```

## Testing

### Test Script

Run the test script to verify the multi-timeframe feature engineering:

```bash
# Basic test
python scripts/test_multi_timeframe_feature_engineering.py --symbol ETHUSDT

# Test specific timeframes
python scripts/test_multi_timeframe_feature_engineering.py --symbol ETHUSDT --timeframes 1m,5m,15m,1h
```

### Test Output Example

```
[14:30:15] INFO: 🚀 Starting full multi-timeframe feature engineering test...
[14:30:16] INFO: ✅ Loaded 500 candles for 1m
[14:30:16] INFO: ✅ Loaded 500 candles for 5m
[14:30:16] INFO: ✅ Loaded 500 candles for 15m
[14:30:16] INFO: ✅ Loaded 500 candles for 1h
[14:30:17] INFO: ⏰ Testing supported timeframes...
[14:30:17] INFO: Supported timeframes: ['1m', '5m', '15m', '1h', '4h', '1d']
[14:30:17] INFO:   1m: Ultra-short-term execution
[14:30:17] INFO:   5m: Short-term execution
[14:30:17] INFO:   15m: Tactical decision making
[14:30:17] INFO:   1h: Strategic macro trend analysis
[14:30:18] INFO: 📈 Testing indicator comparison across timeframes...
[14:30:18] INFO:    RSI Parameters:
[14:30:18] INFO:     1m: length=7 (Fast RSI for immediate signals)
[14:30:18] INFO:     5m: length=10 (Medium-fast RSI)
[14:30:18] INFO:     15m: length=14 (Standard RSI)
[14:30:18] INFO:     1h: length=21 (Long RSI for trend confirmation)
[14:30:19] INFO: 🎯 Testing multi-timeframe feature generation...
[14:30:19] INFO: 📊 Features for 1m:
[14:30:19] INFO:    Shape: (500, 45)
[14:30:19] INFO:    Columns: 45
[14:30:19] INFO:    Timeframe-specific features: 12
[14:30:20] INFO: ✅ Full test completed successfully!
```

## Integration with Multi-Timeframe System

### Combined Usage

```python
from src.analyst.multi_timeframe_feature_engineering import MultiTimeframeFeatureEngineering
from src.analyst.multi_timeframe_regime_integration import MultiTimeframeRegimeIntegration

# Initialize both systems
mtf_feature_engine = MultiTimeframeFeatureEngineering(config)
mtf_regime_integration = MultiTimeframeRegimeIntegration(config)

# Generate features and regime information
features_dict = await mtf_feature_engine.generate_multi_timeframe_features(data_dict)
regime_info = await mtf_regime_integration.get_regime_for_timeframe("5m", data_5m, data_1h)

# Combine for comprehensive analysis
combined_features = features_dict["5m"].copy()
combined_features["regime"] = regime_info["regime"]
combined_features["regime_confidence"] = regime_info["confidence"]
```

## Performance Optimization

### Caching Strategy

- **Feature Cache**: 5-minute cache for generated features
- **Parameter Cache**: Static parameter mappings
- **Memory Management**: Automatic cleanup of old cache entries

### Optimization Features

1. **Efficient Processing**: Vectorized operations for indicator calculation
2. **Memory Management**: Automatic cleanup and memory optimization
3. **Parallel Processing**: Support for concurrent timeframe processing
4. **Caching**: Intelligent caching for repeated calculations

## Benefits

1. **Timeframe-Specific Optimization**: Indicators optimized for each timeframe's characteristics
2. **Trading Style Adaptation**: Parameters adapted to trading style (scalping, swing, position)
3. **Performance Optimization**: Efficient processing and caching
4. **Consistent Interface**: Unified interface across all timeframes
5. **Extensible Design**: Easy to add new timeframes and indicators

## Future Enhancements

1. **Dynamic Parameter Adaptation**: Real-time parameter adjustment based on market conditions
2. **Machine Learning Integration**: ML-based parameter optimization
3. **Advanced Caching**: More sophisticated caching strategies
4. **Performance Monitoring**: Real-time performance metrics
5. **Automated Optimization**: Automatic parameter tuning based on performance

## Troubleshooting

### Common Issues

1. **Missing Data**
   - Ensure sufficient data for indicator calculation
   - Check data quality and timeframe validation

2. **Parameter Errors**
   - Verify timeframe parameter configuration
   - Check indicator parameter validity

3. **Performance Issues**
   - Monitor cache size and cleanup
   - Check memory usage

### Debug Information

```python
# Get system statistics
stats = mtf_feature_engine.get_feature_statistics()
print(f"Supported timeframes: {stats['supported_timeframes']}")
print(f"Cache size: {stats['cache_size']}")
print(f"MTF features enabled: {stats['enable_mtf_features']}")
```

## Conclusion

The Multi-Timeframe Feature Engineering system provides a robust, scalable solution for timeframe-specific feature generation. By adapting indicators to each timeframe's characteristics and trading style, the system ensures optimal performance across all timeframes while maintaining a consistent interface for easy integration with existing systems. 