# Comprehensive Feature List

## Overview

This document provides a comprehensive list of all available features in the unified feature generation system. The system has been enhanced to use **price returns** and **returns-based VWAP** as defaults, with volume features using **volume returns** by default.

## 🎯 Base Calculation Types

The system supports the following base calculation types:

1. **PRICE_RETURNS** - Price returns (percentage changes) - **NEW DEFAULT**
2. **RETURNS_VWAP** - Returns-based Volume Weighted Average Price
3. **PRICE_LEVELS** - Traditional price levels (close, high, low, open)
4. **VOLUME_WEIGHTED** - Volume-weighted calculations
5. **VOLUME_RETURNS** - Volume returns (percentage changes in volume) - **NEW DEFAULT FOR VOLUME**

## 📊 Feature Categories

### 1. 📈 Returns Features
- **SimpleReturnsGenerator** - Simple price returns
- **LogReturnsGenerator** - Logarithmic returns
- **CumulativeReturnsGenerator** - Cumulative returns
- **ReturnVolatilityGenerator** - Volatility of returns
- **ReturnSkewnessGenerator** - Skewness of returns
- **ReturnKurtosisGenerator** - Kurtosis of returns

### 2. 🚀 Momentum Features (Enhanced with Base Calculations)
- **RSIGenerator** - Relative Strength Index
  - Default: `PRICE_RETURNS`
  - Also supports: `RETURNS_VWAP`, `PRICE_LEVELS`
- **MACDGenerator** - Moving Average Convergence Divergence
  - Default: `PRICE_RETURNS`
  - Also supports: `RETURNS_VWAP`, `PRICE_LEVELS`
- **MACDSignalGenerator** - MACD Signal Line
- **MACDHistogramGenerator** - MACD Histogram
- **StochasticGenerator** - Stochastic Oscillator
  - Default: `PRICE_RETURNS`
  - Also supports: `RETURNS_VWAP`, `PRICE_LEVELS`
- **WilliamsRGenerator** - Williams %R
  - Default: `PRICE_RETURNS`
  - Also supports: `RETURNS_VWAP`, `PRICE_LEVELS`
- **ROCGenerator** - Rate of Change
  - Default: `PRICE_RETURNS`
  - Also supports: `RETURNS_VWAP`, `PRICE_LEVELS`
- **MomentumGenerator** - Momentum Indicator
  - Default: `PRICE_RETURNS`
  - Also supports: `RETURNS_VWAP`, `PRICE_LEVELS`

### 3. 📊 Trend Features (Enhanced with Base Calculations)
- **SMAGenerator** - Simple Moving Average
  - Default: `PRICE_RETURNS`
  - Also supports: `RETURNS_VWAP`, `PRICE_LEVELS`
- **EMAGenerator** - Exponential Moving Average
  - Default: `PRICE_RETURNS`
  - Also supports: `RETURNS_VWAP`, `PRICE_LEVELS`

### 4. 📉 Volatility Features (Enhanced with Base Calculations)
- **BollingerBandsGenerator** - Bollinger Bands
  - Default: `PRICE_RETURNS`
  - Also supports: `RETURNS_VWAP`, `PRICE_LEVELS`
  - Band types: upper, lower, middle
- **ATRGenerator** - Average True Range
  - Default: `PRICE_RETURNS`
  - Also supports: `RETURNS_VWAP`, `PRICE_LEVELS`

### 5. 📊 Volume Features (Enhanced with Base Calculations)
- **VolumeMAGenerator** - Volume Moving Average
  - Default: `VOLUME_RETURNS`
  - Also supports: `VOLUME_WEIGHTED`
- **VolumeRatioGenerator** - Volume Ratio
  - Default: `VOLUME_RETURNS`
  - Also supports: `VOLUME_WEIGHTED`
- **OBVGenerator** - On-Balance Volume
- **VWAPGenerator** - Volume Weighted Average Price
  - Default: `PRICE_RETURNS`
  - Also supports: `RETURNS_VWAP`, `PRICE_LEVELS`
- **VolumeROCGenerator** - Volume Rate of Change
- **VPTGenerator** - Volume Price Trend
- **ADLGenerator** - Accumulation/Distribution Line
- **VolumeVolatilityGenerator** - Volume Volatility
- **VolumeSkewnessGenerator** - Volume Skewness

### 6. 🔄 Oscillator Features
- **CCI** - Commodity Channel Index
- **ADX** - Average Directional Index
- **Aroon Oscillator** - Aroon Up/Down
- **Parabolic SAR** - Parabolic Stop and Reverse
- **Ultimate Oscillator** - Ultimate Oscillator
- **KST** - Know Sure Thing
- **APO** - Absolute Price Oscillator
- **CMO** - Chande Momentum Oscillator
- **NATR** - Normalized Average True Range
- **PFE** - Polarized Fractal Efficiency
- **T3** - T3 Moving Average
- **KAMA** - Kaufman's Adaptive Moving Average

### 7. 🎯 Support/Resistance Features
- **Support Level Detection** - Dynamic support levels
- **Resistance Level Detection** - Dynamic resistance levels
- **Pivot Points** - Standard pivot points
- **Fibonacci Retracements** - Fibonacci levels
- **Volume Profile Analysis** - Volume-based levels

### 8. 🕯️ Candlestick Pattern Features
- **Doji Patterns** - Doji, Long-legged Doji, Dragonfly Doji, Gravestone Doji
- **Hammer Patterns** - Hammer, Inverted Hammer
- **Shooting Star Patterns** - Shooting Star, Inverted Hammer
- **Engulfing Patterns** - Bullish/Bearish Engulfing
- **Harami Patterns** - Harami, Harami Cross
- **Morning/Evening Star Patterns** - Morning Star, Evening Star
- **Three White Soldiers/Black Crows** - Three White Soldiers, Three Black Crows
- **Piercing Line/Dark Cloud Cover** - Piercing Line, Dark Cloud Cover

### 9. 🔄 HMM Regime Features
- **Regime Detection** - Hidden Markov Model regime detection
- **Regime Transition Probabilities** - Transition probabilities between regimes
- **Regime-Aware Feature Generation** - Features adapted to market regimes
- **Regime-Based Optimization** - Optimization based on detected regimes

### 10. 🔗 Interaction Features
- **CrossTimeframeInteractionGenerator** - Cross-timeframe feature interactions
- **FeatureRatioGenerator** - Ratios between different features
- **PolynomialFeatureGenerator** - Polynomial transformations of features
- **CorrelationInteractionGenerator** - Correlation-based feature interactions

## ⚡ Enhanced Indicators Summary

The following indicators have been enhanced to support multiple base calculations:

| Indicator | Default | Also Supports |
|-----------|---------|---------------|
| RSI | PRICE_RETURNS | RETURNS_VWAP, PRICE_LEVELS |
| MACD | PRICE_RETURNS | RETURNS_VWAP, PRICE_LEVELS |
| Stochastic | PRICE_RETURNS | RETURNS_VWAP, PRICE_LEVELS |
| Williams %R | PRICE_RETURNS | RETURNS_VWAP, PRICE_LEVELS |
| ROC | PRICE_RETURNS | RETURNS_VWAP, PRICE_LEVELS |
| Momentum | PRICE_RETURNS | RETURNS_VWAP, PRICE_LEVELS |
| SMA | PRICE_RETURNS | RETURNS_VWAP, PRICE_LEVELS |
| EMA | PRICE_RETURNS | RETURNS_VWAP, PRICE_LEVELS |
| Bollinger Bands | PRICE_RETURNS | RETURNS_VWAP, PRICE_LEVELS |
| ATR | PRICE_RETURNS | RETURNS_VWAP, PRICE_LEVELS |
| VWAP | PRICE_RETURNS | RETURNS_VWAP, PRICE_LEVELS |
| Volume MA | VOLUME_RETURNS | VOLUME_WEIGHTED |
| Volume Ratio | VOLUME_RETURNS | VOLUME_WEIGHTED |

## 🔧 Usage Examples

### Basic Usage (New Defaults)
```python
from src.feature_generation import RSIGenerator, VolumeMAGenerator, BaseCalculationType

# RSI with price returns (new default)
rsi = RSIGenerator(period=14)  # Uses PRICE_RETURNS by default

# Volume MA with volume returns (new default)
volume_ma = VolumeMAGenerator(period=20)  # Uses VOLUME_RETURNS by default
```

### Advanced Usage with Different Base Calculations
```python
# RSI with returns VWAP
rsi_vwap = RSIGenerator(
    period=14, 
    base_calculation=BaseCalculationType.RETURNS_VWAP, 
    vwap_period=20
)

# RSI with traditional price levels
rsi_levels = RSIGenerator(
    period=14, 
    base_calculation=BaseCalculationType.PRICE_LEVELS
)

# Volume MA with volume-weighted calculation
volume_ma_weighted = VolumeMAGenerator(
    period=20, 
    base_calculation=BaseCalculationType.VOLUME_WEIGHTED
)
```

### Comprehensive Feature Generation
```python
from src.feature_generation import FeatureBank, BaseCalculationType

# Initialize feature bank
bank = FeatureBank()

# Generate features with different base calculations
features = pd.DataFrame(index=data.index)

# Momentum indicators with price returns (default)
rsi = RSIGenerator(period=14)
macd = MACDGenerator(fast=12, slow=26, signal=9)
stoch = StochasticGenerator(k_period=14, d_period=3)

# Trend indicators with returns VWAP
sma_vwap = SMAGenerator(period=20, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)
ema_vwap = EMAGenerator(period=20, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)

# Volume indicators with volume returns (default)
volume_ma = VolumeMAGenerator(period=20)
volume_ratio = VolumeRatioGenerator(period=20)

# Generate and store features
for generator in [rsi, macd, stoch, sma_vwap, ema_vwap, volume_ma, volume_ratio]:
    features[generator.config.name] = generator.generate(data)

# Store in feature bank
bank.add_features("enhanced_features", features)
```

## 📈 Feature Statistics

- **Total Categories**: 10
- **Enhanced Indicators**: 13
- **Base Calculation Types**: 5
- **Total Individual Features**: 50+
- **Interaction Features**: 4
- **Candlestick Patterns**: 8+
- **Oscillator Features**: 12+

## 🎉 Key Benefits

### 1. **Enhanced Defaults**
- Price-based indicators now use **PRICE_RETURNS** by default
- Volume-based indicators now use **VOLUME_RETURNS** by default
- More sophisticated analysis capabilities

### 2. **Flexible Base Calculations**
- All enhanced indicators support multiple base calculation types
- Easy switching between different calculation methods
- Consistent interface across all indicators

### 3. **Backwards Compatibility**
- Existing code continues to work
- Gradual migration path available
- Legacy support maintained

### 4. **Comprehensive Coverage**
- All major technical indicators covered
- Advanced features like HMM regimes and interactions
- Extensive candlestick pattern recognition

## 🚀 Production Ready

The unified feature generation system is now ready for production use with:

- ✅ Enhanced defaults (price returns, volume returns)
- ✅ Comprehensive feature coverage
- ✅ Flexible base calculations
- ✅ Backwards compatibility
- ✅ Matrix operations integration
- ✅ Hardware acceleration support
- ✅ Feature bank and registry
- ✅ Lookback optimization
- ✅ Interaction features

The system provides a solid foundation for advanced quantitative analysis and trading strategy development.