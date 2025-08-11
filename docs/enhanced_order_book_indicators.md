# Enhanced Order Book Indicators

## Overview

The enhanced order book indicators provide sophisticated approximations of real order book data using advanced microstructural analysis. These indicators are designed to capture the essential characteristics of market liquidity and order flow without requiring actual order book data.

## Enhanced Indicators

### 1. **Enhanced Order Flow Imbalance (OFI)**

**Purpose**: Measures the net buying/selling pressure in the market.

**Enhanced Features**:
- **Volume Direction Imbalance**: Analyzes volume patterns in up vs down price movements
- **Trade Size Imbalance**: Considers the distribution of trade sizes
- **Price Impact Asymmetry**: Measures how volume affects price movements
- **Volume-Weighted Pressure**: Combines volume and price direction

**Calculation Components**:
```python
# 1. Volume imbalance based on price direction
up_volume = volume * (price_direction > 0)
down_volume = volume * (price_direction < 0)
volume_direction_imbalance = (up_volume_ma - down_volume_ma) / (up_volume_ma + down_volume_ma)

# 2. Trade size imbalance
trade_size_ratio = volume / trade_size_ma

# 3. Price impact asymmetry
price_impact = abs(price_changes) / volume_ratio
price_impact_ratio = price_impact / price_impact_ma

# 4. Combined OFI
enhanced_ofi = (
    volume_direction_imbalance * 0.4 +
    (trade_size_ratio - 1) * 0.25 +
    (1 - price_impact_ratio) * 0.2 +
    (volume_ratio - 1) * price_direction * 0.15
)
```

**Interpretation**:
- **Positive values**: Net buying pressure
- **Negative values**: Net selling pressure
- **Range**: [-1, 1]
- **Higher absolute values**: Stronger imbalance

### 2. **Enhanced Bid-Ask Spread**

**Purpose**: Estimates the bid-ask spread using multiple microstructural factors.

**Enhanced Features**:
- **Parkinson Volatility**: Uses high-low range for more accurate volatility estimation
- **Volume-Based Liquidity**: Considers volume impact on spread tightness
- **Trade Frequency Impact**: Accounts for trading activity
- **Market Stress Indicators**: Detects periods of market stress

**Calculation Components**:
```python
# 1. Parkinson volatility
log_hl_ratio = log(high / low)
parkinson_vol = sqrt((log_hl_ratio^2).rolling_mean() / (4 * log(2)))

# 2. Volume-based liquidity
liquidity_factor = 1 / volume_ratio

# 3. Trade frequency
frequency_factor = 1 / trade_frequency

# 4. Market stress
stress_factor = volume_weighted_impact.rolling_mean()

# 5. Combined spread
dynamic_spread = base_spread + (
    parkinson_vol * 0.4 +
    liquidity_factor * 0.25 +
    frequency_factor * 0.15 +
    stress_factor * 0.2
)
```

**Interpretation**:
- **Lower values**: Tighter spreads, higher liquidity
- **Higher values**: Wider spreads, lower liquidity
- **Range**: 0.01% - 5%
- **Typical values**: 0.05% - 0.5%

### 3. **Enhanced Market Depth**

**Purpose**: Estimates the volume available at different price levels.

**Enhanced Features**:
- **Volume-Weighted Depth**: Considers volume distribution
- **Price Impact Resistance**: Measures absorption capacity
- **Trade Size Distribution**: Analyzes trade size uniformity
- **Market Resilience**: Tracks volume recovery patterns
- **Depth Asymmetry**: Measures balance between bid/ask sides

**Calculation Components**:
```python
# 1. Volume-weighted depth
volume_z_score = (volume - volume_ma) / volume_std

# 2. Price impact resistance
impact_resistance = 1 / volume_weighted_impact.rolling_mean()

# 3. Trade size uniformity
size_uniformity = 1 / trade_size_cv

# 4. Market resilience
volume_recovery = sum(positive_changes) / sum(abs_changes)

# 5. Depth balance
depth_balance = 1 - abs(directional_volume_ma) / volume_ma

# 6. Combined depth
composite_depth = (
    volume_z_score * 0.3 +
    impact_resistance * 0.25 +
    size_uniformity * 0.2 +
    volume_recovery * 0.15 +
    depth_balance * 0.1
)
```

**Interpretation**:
- **Higher values**: Deeper markets, more volume available
- **Lower values**: Shallow markets, less volume available
- **Units**: Volume units
- **Scale**: Relative to average volume

### 4. **Enhanced Liquidity Score**

**Purpose**: Comprehensive liquidity assessment combining all indicators.

**Enhanced Features**:
- **Multi-Factor Analysis**: Combines all three indicators
- **Market Regime Adjustments**: Adapts to different market conditions
- **Price Stability**: Considers price volatility impact
- **Volume Consistency**: Analyzes trading activity quality
- **Trade Frequency**: Measures market activity levels

**Calculation Components**:
```python
# 1. Normalize all inputs to [0, 1]
ofi_normalized = (ofi + 1) / 2
spread_normalized = 1 - normalize(spread)
depth_normalized = normalize(depth)

# 2. Additional factors
price_stability = 1 - normalize(price_volatility)
volume_consistency = 1 - normalize(volume_cv)
trade_frequency = normalize(frequency)

# 3. Composite score
liquidity_score = (
    ofi_normalized * 0.25 +
    spread_normalized * 0.25 +
    depth_normalized * 0.2 +
    price_stability * 0.15 +
    volume_consistency * 0.1 +
    trade_frequency * 0.05
)

# 4. Regime adjustment
regime_adjustment = f(volatility_regime)
final_score = liquidity_score * regime_adjustment
```

**Interpretation**:
- **Higher values**: Higher liquidity, better trading conditions
- **Lower values**: Lower liquidity, more challenging trading
- **Range**: [0, 1]
- **0.8+**: Excellent liquidity
- **0.6-0.8**: Good liquidity
- **0.4-0.6**: Moderate liquidity
- **<0.4**: Poor liquidity

## Usage Examples

### Basic Usage

```python
from src.training.steps.vectorized_advanced_feature_engineering import VectorizedAdvancedFeatureEngineering

# Initialize feature engineering
fe = VectorizedAdvancedFeatureEngineering(config={})

# Generate microstructure features
features = await fe._engineer_microstructure_features_vectorized(
    price_data=your_price_data,
    volume_data=your_volume_data,
    order_flow_data=your_order_flow_data
)

# Access enhanced indicators
ofi = features["order_flow_imbalance"]
spread = features["bid_ask_spread"]
depth = features["market_depth"]
liquidity_score = features["enhanced_liquidity_score"]
```

### Advanced Analysis

```python
# Analyze liquidity conditions
def analyze_liquidity_conditions(features):
    ofi = features["order_flow_imbalance"]
    spread = features["bid_ask_spread"]
    depth = features["market_depth"]
    liquidity = features["enhanced_liquidity_score"]
    
    # Determine market conditions
    if liquidity > 0.8:
        condition = "Excellent liquidity"
    elif liquidity > 0.6:
        condition = "Good liquidity"
    elif liquidity > 0.4:
        condition = "Moderate liquidity"
    else:
        condition = "Poor liquidity"
    
    # Analyze order flow
    if ofi > 0.5:
        flow = "Strong buying pressure"
    elif ofi < -0.5:
        flow = "Strong selling pressure"
    else:
        flow = "Balanced order flow"
    
    return {
        "condition": condition,
        "order_flow": flow,
        "spread_tightness": "Tight" if spread < 0.1 else "Wide",
        "market_depth": "Deep" if depth > depth.mean() else "Shallow"
    }
```

### Trading Strategy Integration

```python
# Use indicators for position sizing
def calculate_position_size(liquidity_score, ofi, spread):
    # Base position size
    base_size = 1000
    
    # Adjust for liquidity
    liquidity_multiplier = liquidity_score * 2  # 0-2x multiplier
    
    # Adjust for order flow
    flow_multiplier = 1 + (ofi * 0.5)  # 0.5-1.5x multiplier
    
    # Adjust for spread
    spread_multiplier = 1 / (1 + spread)  # Tighter spread = larger position
    
    # Calculate final position size
    position_size = base_size * liquidity_multiplier * flow_multiplier * spread_multiplier
    
    return position_size

# Use for risk management
def assess_market_risk(liquidity_score, depth, spread):
    risk_score = 0
    
    # Low liquidity = higher risk
    if liquidity_score < 0.4:
        risk_score += 3
    elif liquidity_score < 0.6:
        risk_score += 1
    
    # Shallow depth = higher risk
    if depth < depth.quantile(0.25):
        risk_score += 2
    
    # Wide spread = higher risk
    if spread > spread.quantile(0.75):
        risk_score += 2
    
    return risk_score
```

## Configuration

### Default Parameters

```python
# Order Flow Imbalance
ofi_config = {
    "volume_direction_weight": 0.4,
    "trade_size_weight": 0.25,
    "price_impact_weight": 0.2,
    "volume_pressure_weight": 0.15,
    "smoothing_window": 5
}

# Bid-Ask Spread
spread_config = {
    "volatility_weight": 0.4,
    "liquidity_weight": 0.25,
    "frequency_weight": 0.15,
    "stress_weight": 0.2,
    "base_spread": 0.0005,
    "min_spread": 0.0001,
    "max_spread": 0.05
}

# Market Depth
depth_config = {
    "volume_weighted_weight": 0.3,
    "impact_resistance_weight": 0.25,
    "size_uniformity_weight": 0.2,
    "resilience_weight": 0.15,
    "balance_weight": 0.1,
    "min_depth_multiplier": 0.1,
    "max_depth_multiplier": 5.0
}

# Liquidity Score
liquidity_config = {
    "order_flow_weight": 0.25,
    "spread_weight": 0.25,
    "depth_weight": 0.2,
    "stability_weight": 0.15,
    "consistency_weight": 0.1,
    "frequency_weight": 0.05,
    "smoothing_window": 5
}
```

## Performance Considerations

### Computational Efficiency

- **Vectorized Operations**: All calculations use pandas/numpy vectorized operations
- **Rolling Windows**: Optimized rolling window calculations
- **Memory Management**: Efficient memory usage with proper cleanup
- **Caching**: Results can be cached for repeated calculations

### Accuracy vs Speed Trade-offs

- **High Accuracy**: Use longer rolling windows (50+ periods)
- **High Speed**: Use shorter rolling windows (10-20 periods)
- **Balanced**: Use medium windows (20-30 periods)

### Recommended Settings

```python
# For backtesting (high accuracy)
config = {
    "volume_window": 50,
    "volatility_window": 20,
    "smoothing_window": 10
}

# For live trading (high speed)
config = {
    "volume_window": 20,
    "volatility_window": 10,
    "smoothing_window": 5
}
```

## Validation and Testing

### Backtesting Validation

```python
def validate_indicators(historical_data):
    # Calculate indicators
    features = calculate_enhanced_indicators(historical_data)
    
    # Validate ranges
    assert features["order_flow_imbalance"].min() >= -1
    assert features["order_flow_imbalance"].max() <= 1
    assert features["bid_ask_spread"].min() >= 0.0001
    assert features["enhanced_liquidity_score"].min() >= 0
    assert features["enhanced_liquidity_score"].max() <= 1
    
    # Validate correlations
    ofi_corr = features["order_flow_imbalance"].corr(price_returns)
    spread_corr = features["bid_ask_spread"].corr(price_volatility)
    
    print(f"OFI-Price correlation: {ofi_corr:.3f}")
    print(f"Spread-Volatility correlation: {spread_corr:.3f}")
    
    return features
```

### Performance Metrics

```python
def calculate_performance_metrics(features, returns):
    # Liquidity score vs returns
    liquidity_returns = pd.DataFrame({
        'liquidity': features["enhanced_liquidity_score"],
        'returns': returns
    })
    
    # Group by liquidity quintiles
    liquidity_quintiles = pd.qcut(liquidity_returns['liquidity'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    liquidity_returns['quintile'] = liquidity_quintiles
    
    # Calculate returns by quintile
    quintile_returns = liquidity_returns.groupby('quintile')['returns'].agg(['mean', 'std', 'sharpe'])
    
    return quintile_returns
```

## Best Practices

### 1. **Data Quality**
- Ensure clean price and volume data
- Handle missing values appropriately
- Use consistent timeframes

### 2. **Parameter Tuning**
- Adjust weights based on asset characteristics
- Optimize rolling windows for your timeframe
- Test different smoothing parameters

### 3. **Risk Management**
- Use liquidity score for position sizing
- Monitor spread widening for risk signals
- Consider depth for large order execution

### 4. **Market Regime Awareness**
- Adjust expectations in different volatility regimes
- Consider market hours and liquidity patterns
- Monitor for regime changes

## Conclusion

The enhanced order book indicators provide sophisticated approximations of real order book data that can significantly improve trading strategies. By combining multiple microstructural factors, these indicators offer a comprehensive view of market liquidity and order flow dynamics.

Key benefits:
- **No Order Book Data Required**: Works with standard OHLCV data
- **Comprehensive Analysis**: Multiple factors considered
- **Market Regime Adaptive**: Adjusts to different market conditions
- **Vectorized Performance**: Fast computation for real-time use
- **Robust Implementation**: Handles edge cases and data quality issues

These indicators are particularly valuable for:
- **Position Sizing**: Adjust position sizes based on liquidity
- **Risk Management**: Identify periods of poor liquidity
- **Entry/Exit Timing**: Use order flow for timing decisions
- **Market Analysis**: Understand market microstructure dynamics 