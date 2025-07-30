# Volatility Targeting Guide

## Overview

Volatility targeting is a risk management technique that dynamically adjusts position sizes to maintain a consistent level of portfolio volatility. This approach helps stabilize returns and can improve risk-adjusted performance, especially during volatile market conditions.

## How It Works

The basic principle is simple:
- When market volatility is **high** → **reduce** position size
- When market volatility is **low** → **increase** position size

Formula: `Position Size = Target Volatility / Current Volatility × Base Position Size`

## Benefits

1. **Risk Consistency**: Maintains stable risk exposure regardless of market conditions
2. **Better Sharpe Ratios**: Often improves risk-adjusted returns by 10-30%
3. **Downside Protection**: Automatically reduces exposure during volatile periods
4. **Capital Efficiency**: Increases exposure during calm market periods
5. **Emotional Discipline**: Removes emotion from position sizing decisions

## Implementation in Your System

### 1. Configuration

Add volatility targeting settings to your configuration:

```python
# In your config file
VOLATILITY_TARGETING = {
    "enabled": True,
    "target_volatility": 0.15,  # 15% annual target
    "method": "EWMA",  # EWMA, Simple, GARCH, Parkinson, Adaptive
    "lookback_period": 20,
    "max_leverage": 3.0,
    "min_leverage": 0.1,
    "momentum_filter": True,
    "regime_adjustment": True,
    "kelly_enhancement": False
}

# Strategist configuration
STRATEGIST_CONFIG = {
    "volatility_targeting_method": "EWMA",  # Which volatility method to use
    "max_volatility_multiplier": 3.0,
    "min_volatility_multiplier": 0.1,
    "regime_weight": 0.4,
    "ma_weight": 0.2,
    "vwap_weight": 0.15,
    "trend_weight": 0.15,
    "momentum_weight": 0.1,
    "bias_threshold": 0.6
}
```

### 2. Feature Engineering Configuration

```python
# Feature engineering settings
FEATURE_ENGINEERING = {
    "target_volatility": 0.15,
    "simple_vol_period": 20,
    "ewma_span": 20,
    "garch_period": 30,
    "max_leverage": 3.0,
    "min_leverage": 0.1,
    "momentum_period": 10,
    "volatility_window": 24
}
```

### 3. Using the Volatility Targeting Strategy

```python
from src.strategist.volatility_targeting_strategy import (
    VolatilityTargetingStrategy, 
    VolatilityTargetingConfig, 
    VolatilityMethod
)

# Create configuration
config = VolatilityTargetingConfig(
    target_volatility=0.12,  # 12% annual volatility target
    volatility_method=VolatilityMethod.EWMA,
    lookback_period=20,
    max_leverage=2.5,
    min_leverage=0.2,
    momentum_filter=True,
    regime_adjustment=True,
    kelly_enhancement=False
)

# Initialize strategy
vol_strategy = VolatilityTargetingStrategy(config)

# Calculate position multiplier for your asset
multiplier = vol_strategy.calculate_position_multiplier(price_data)

# Apply to your position size
base_position_size = 1000  # USD
adjusted_position_size = base_position_size * multiplier
```

## Volatility Calculation Methods

### 1. Simple Historical Volatility
- Uses rolling standard deviation of returns
- Most straightforward method
- Good for stable market conditions

### 2. EWMA (Exponentially Weighted Moving Average)
- **Recommended for most cases**
- Gives more weight to recent observations
- Responds faster to volatility changes
- More robust during trending markets

### 3. Parkinson Volatility
- Uses high-low price range
- Often more accurate than close-to-close volatility
- Less noisy, especially for intraday data
- Requires high/low price data

### 4. GARCH-like
- Simplified GARCH(1,1) approximation
- Good for volatility clustering
- More sophisticated but computationally heavier

### 5. Adaptive
- Combines multiple methods
- Adjusts weights based on market conditions
- Most sophisticated but may overfit

## Advanced Features

### Momentum Filter
When enabled, reduces exposure during negative price momentum:
- If 10-day momentum < -5%: multiply by 0.7
- If 10-day momentum > +5%: multiply by 1.1
- Otherwise: no adjustment

### Regime Adjustment
Adjusts position size based on volatility regime:
- **High volatility regime** (vol > 1.5x long-term): reduce exposure by 40%
- **Elevated volatility** (vol > 1.2x long-term): reduce exposure by 20%
- **Low volatility regime** (vol < 0.7x long-term): increase exposure by 30%
- **Normal regime**: no adjustment

### Kelly Criterion Enhancement
Uses Kelly criterion to optimize position sizing:
- Estimates optimal fraction based on expected return and volatility
- Capped at 25% Kelly to prevent over-leverage
- Enhances the base volatility targeting multiplier

## Portfolio Application

### Multi-Asset Allocation

```python
# Example: 60/40 stock/bond portfolio with volatility targeting
assets_data = {
    'stocks': stock_price_data,
    'bonds': bond_price_data
}

base_weights = {
    'stocks': 0.6,
    'bonds': 0.4
}

# Generate volatility-targeted allocation
vol_weights = vol_strategy.generate_portfolio_allocation(assets_data, base_weights)
```

### Dynamic Rebalancing

```python
# Check current stats
stats = vol_strategy.get_strategy_stats(price_data)
print(f"Current volatility: {stats['current_volatility']:.1%}")
print(f"Position multiplier: {stats['position_multiplier']:.2f}")

# Rebalance if multiplier has changed significantly
if abs(stats['position_multiplier'] - last_multiplier) > 0.2:
    # Trigger rebalancing
    new_position_size = base_size * stats['position_multiplier']
```

## Example Results

Based on research and backtesting, volatility targeting typically provides:

### Performance Improvements
- **5-15%** improvement in Sharpe ratio
- **10-25%** reduction in maximum drawdown
- **20-40%** reduction in volatility of volatility
- **Better consistency** across different market regimes

### Sample Backtest Results (Hypothetical)
```
Strategy: S&P 500 with 15% Volatility Target (2010-2023)

Metrics                  | Buy & Hold | Vol Targeting | Improvement
-------------------------|------------|---------------|------------
Annual Return            | 12.5%      | 11.8%         | -0.7%
Annual Volatility        | 16.2%      | 15.1%         | -1.1%
Sharpe Ratio            | 0.77       | 0.94          | +22%
Max Drawdown            | -33.7%     | -22.1%        | +34%
Volatility of Volatility | 4.2%       | 2.8%          | +33%
```

## Risk Management

### Position Size Limits
Always use leverage caps to prevent excessive exposure:
- **Conservative**: 0.2x to 2.0x
- **Moderate**: 0.1x to 3.0x
- **Aggressive**: 0.05x to 5.0x

### Monitoring
Key metrics to monitor:
- Current volatility vs. target
- Position multiplier trends
- Regime changes
- Rebalancing frequency

### Circuit Breakers
Implement safeguards:
```python
# Example circuit breaker
if stats['volatility_ratio'] > 3.0:  # Volatility 3x target
    multiplier = min(multiplier, 0.5)  # Cap at 50% exposure
```

## Best Practices

1. **Start Conservative**: Begin with lower target volatility (10-12%)
2. **Use EWMA Method**: Generally the best balance of responsiveness and stability
3. **Enable Momentum Filter**: Helps during trending markets
4. **Monitor Rebalancing**: Don't rebalance too frequently (daily is usually sufficient)
5. **Backtest Thoroughly**: Test on historical data before going live
6. **Combine with Other Signals**: Volatility targeting works best with directional signals

## Integration with Existing System

The volatility targeting is automatically integrated into:

1. **Feature Engineering**: All volatility metrics are calculated
2. **Technical Analysis**: Enhanced indicators inform regime detection
3. **Strategist**: Position sizing is automatically adjusted
4. **Risk Management**: Leverage caps are dynamically applied

Your system will now:
- Calculate multiple volatility measures
- Generate position sizing multipliers
- Adjust leverage based on market conditions
- Improve risk-adjusted returns
- Provide detailed volatility statistics

## Troubleshooting

### Common Issues

1. **Excessive Rebalancing**: Increase minimum threshold for rebalancing
2. **Lag in Adjustments**: Switch to EWMA or reduce lookback period
3. **Over-leverage**: Lower max_leverage parameter
4. **Under-performance**: Enable momentum filter and regime adjustment

### Diagnostics

```python
# Get detailed diagnostics
stats = vol_strategy.get_strategy_stats(price_data)
for key, value in stats.items():
    print(f"{key}: {value}")
```

This will help you monitor and fine-tune your volatility targeting implementation.