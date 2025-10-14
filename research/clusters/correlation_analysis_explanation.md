# Correlation and Auto-correlation Analysis for Market Regimes

## Overview

Correlation and auto-correlation analysis are crucial for understanding market regime dynamics and identifying implicit market dimensions. This analysis helps discover temporal dependencies and relationships that characterize different market states.

## Types of Correlation Analysis

### 1. Auto-correlation Analysis
**Definition**: Measures the correlation of a time series with delayed versions of itself.

**Applications in Regime Analysis**:
- **Momentum Persistence**: How long do price movements persist?
- **Mean Reversion**: Do prices tend to revert to their mean?
- **Regime Stability**: How persistent are regime characteristics?

**Implementation**:
```python
# Price auto-correlation at different lags
for lag in [1, 5, 10, 20]:
    returns['autocorr_lag_' + str(lag)] = returns.rolling(50).apply(
        lambda x: x.autocorr(lag)
    )
```

**Economic Significance**:
- **Lag 1**: Immediate momentum/reversal patterns
- **Lag 5-10**: Short-term trend persistence
- **Lag 20+**: Long-term mean reversion tendencies

### 2. Cross-Asset Correlation
**Definition**: Correlation between different financial instruments.

**Applications**:
- **Risk-On/Risk-Off Regimes**: Correlations increase during stress
- **Sector Rotation**: Different sectors lead in different regimes
- **Flight to Quality**: Safe haven correlations during crises

### 3. Volume-Price Correlation
**Definition**: Relationship between trading volume and price movements.

**Regime Implications**:
- **High Correlation**: Trend-following regimes
- **Low/Negative Correlation**: Contrarian or consolidation regimes
- **Changing Correlation**: Regime transition signals

### 4. Cross-Timeframe Correlation
**Definition**: Correlation of the same asset across different timeframes.

**Applications**:
- **Multi-timeframe Alignment**: When different timeframes agree
- **Regime Coherence**: How well regimes align across timeframes
- **Scale Invariance**: Whether patterns repeat across scales

## Implementation in the Framework

### Feature Engineering Integration
```python
# Auto-correlation features
def generate_autocorr_features(returns, lags=[1, 5, 10, 20]):
    features = pd.DataFrame(index=returns.index)
    for lag in lags:
        features[f'autocorr_lag_{lag}'] = returns.rolling(50).apply(
            lambda x: x.autocorr(lag)
        )
    return features

# Rolling correlation features
def generate_rolling_corr_features(data, windows=[20, 50, 100]):
    features = pd.DataFrame(index=data.index)
    for window in windows:
        features[f'vol_price_corr_{window}'] = data['volume'].rolling(window).corr(data['close'])
        features[f'high_low_corr_{window}'] = data['high'].rolling(window).corr(data['low'])
    return features
```

### Dimension Discovery
The framework automatically identifies correlation-based dimensions:

1. **Temporal Correlation Dimension**:
   - Auto-correlation features at various lags
   - Momentum persistence indicators
   - Mean reversion signals

2. **Cross-Asset Correlation Dimension**:
   - Inter-market relationships
   - Sector correlation patterns
   - Risk correlation structures

3. **Volume-Price Correlation Dimension**:
   - Volume confirmation patterns
   - Divergence indicators
   - Liquidity relationship signals

## Economic Interpretation

### Market Regimes and Correlation Patterns

#### 1. Trending Regimes
- **High positive auto-correlation** at short lags (1-5)
- **Strong volume-price correlation**
- **Cross-timeframe alignment**

#### 2. Mean-Reverting Regimes
- **Negative auto-correlation** at short lags
- **Weak or negative volume-price correlation**
- **Cross-timeframe divergence**

#### 3. Volatile/Crisis Regimes
- **Increased cross-asset correlations**
- **Breakdown of normal correlation patterns**
- **High correlation with volatility measures**

#### 4. Consolidation Regimes
- **Low auto-correlation** at all lags
- **Random volume-price relationships**
- **Weak cross-timeframe correlations**

## Feature Groups Identified by the Framework

### Correlation Dimension Features
```python
correlation_features = [
    'autocorr_lag_1', 'autocorr_lag_5', 'autocorr_lag_10', 'autocorr_lag_20',
    'vol_price_corr_20', 'vol_price_corr_50', 'vol_price_corr_100',
    'high_low_corr_20', 'open_close_corr_20',
    'returns_volume_corr_20', 'returns_volatility_corr_20'
]
```

### Analysis Output
The dimension analyzer provides:

1. **Correlation Strength**: How strong are the correlations?
2. **Correlation Stability**: How stable are correlations over time?
3. **Regime Discriminability**: How well do correlations distinguish regimes?
4. **Economic Significance**: Do correlation patterns predict returns?

## Research Applications

### 1. Regime Identification
- Use correlation patterns to identify regime transitions
- Detect when normal relationships break down
- Identify regime-specific correlation structures

### 2. Risk Management
- Monitor correlation changes as regime signals
- Adjust portfolio hedging based on correlation regimes
- Detect systemic risk through correlation spikes

### 3. Trading Strategy Development
- Momentum strategies in high auto-correlation regimes
- Mean reversion strategies in negative auto-correlation regimes
- Volume confirmation in high volume-price correlation regimes

## Integration with Existing Systems

The correlation analysis integrates with your existing feature engineering pipeline:

```python
# Using existing feature generators
from src.feature_engineering.feature_generators import FeatureGenerators

feature_gen = FeatureGenerators()

# Generate correlation-based features
correlation_features = feature_gen.generate_correlation_features(market_data)

# Analyze implicit correlation dimensions
dimension_analyzer = MarketDimensionAnalyzer()
correlation_analysis = dimension_analyzer.analyze_correlation_dimension(correlation_features)
```

## Expected Research Outcomes

1. **Dimension Ranking**: Which correlation patterns are most important for regime identification?
2. **Economic Significance**: Which correlations have the strongest relationship to trading performance?
3. **Regime Stability**: How persistent are correlation-based regime classifications?
4. **Predictive Power**: How well do correlation patterns predict future regime changes?

This correlation analysis provides a systematic way to understand the temporal and cross-sectional relationships that define different market regimes, giving you quantitative insights into market structure and dynamics.