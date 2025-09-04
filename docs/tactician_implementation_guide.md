# Tactician Implementation Guide

## Quick Start

The Tactician combines probabilistic scenario analysis with Analyst inputs to make trading decisions. Here's how to use it effectively:

### Basic Usage

```python
from src.tactician.tactician import Tactician

# Initialize with configuration
config = {
    "tactician": {
        "tactics_interval": 30,
        "max_history": 100
    },
    "step17_optimization": {
        "fully_migrated_tactician": {
            "entry_profit_threshold": 0.6,
            "entry_risk_threshold": 0.2,
            "max_position_size": 0.1,
            "max_leverage": 3.0
        }
    }
}

tactician = Tactician(config)
await tactician.initialize()

# Generate predictions
predictions = await tactician.generate_enhanced_predictions(
    market_data=market_df,
    analyst_barriers={"upper_barrier": 0.02, "lower_barrier": -0.01},
    symbol="BTCUSDT",
    timeframe="1m",
    analyst_confidence=0.7
)

# Extract key outputs
entry_signal = predictions["trading_decisions"]["entry_signal"]
position_size = predictions["position_management"]["position_size"]
leverage = predictions["position_management"]["leverage"]
stop_loss = predictions["position_management"]["stop_loss"]
take_profit = predictions["position_management"]["take_profit"]
```

## Understanding the Outputs

### 1. Scenario Predictions
```python
{
    "probabilities": {
        0: 0.05,   # Profit Zone 1 (0.25%)
        1: 0.08,   # Profit Zone 2 (0.5%)
        ...
        16: 0.15   # Neutral
    },
    "predicted_scenario": 5,
    "scenario_name": "Profit Zone 6 (1.5%)",
    "confidence": 0.82,
    "scenario_analysis": {
        "profit_zone_probability": 0.65,
        "risk_zone_probability": 0.20,
        "neutral_probability": 0.15,
        "risk_reward_ratio": 3.25,
        "scenario_dominance": 0.45
    }
}
```

### 2. Trading Decisions
```python
{
    "entry_signal": True,
    "exit_signal": False,
    "direction": "LONG",
    "confidence": 0.78,
    "reasoning": "ENTRY SIGNAL: Strong scenario analysis | Profit: 65% | Risk: 20% | R:R: 3.25",
    "scenario_metrics": {...}
}
```

### 3. Position Management
```python
{
    "position_size": 0.08,      # 8% of capital
    "leverage": 25.0,           # 25x leverage
    "stop_loss": -0.01,         # -1% from entry
    "take_profit": 0.02,        # +2% from entry
    "risk_metrics": {
        "dominance_multiplier": 1.225,
        "ratio_multiplier": 1.5
    }
}
```

## Optimization Strategies

### 1. Conservative Approach
```python
conservative_config = {
    "entry_profit_threshold": 0.7,      # Higher threshold
    "entry_risk_threshold": 0.15,       # Lower risk tolerance
    "entry_confidence_threshold": 0.8,  # Higher confidence required
    "max_position_size": 0.05,          # Smaller positions
    "max_leverage": 2.0                 # Lower leverage
}
```

### 2. Aggressive Approach
```python
aggressive_config = {
    "entry_profit_threshold": 0.5,      # Lower threshold
    "entry_risk_threshold": 0.3,        # Higher risk tolerance
    "entry_confidence_threshold": 0.6,  # Lower confidence acceptable
    "max_position_size": 0.15,          # Larger positions
    "max_leverage": 5.0                 # Higher leverage
}
```

### 3. Balanced Approach
```python
balanced_config = {
    "entry_profit_threshold": 0.6,
    "entry_risk_threshold": 0.2,
    "entry_confidence_threshold": 0.7,
    "max_position_size": 0.1,
    "max_leverage": 3.0
}
```

## Best Practices

### 1. Analyst Integration
- **Higher Analyst Confidence** → More aggressive position sizing
- **Wider Analyst Barriers** → Expect larger moves, adjust accordingly
- **Narrower Analyst Barriers** → More conservative targets

### 2. Market Conditions
- **High Volatility**: Reduce position size and leverage
- **Low Volatility**: Can increase position size within limits
- **Trending Markets**: Trust profit zone predictions more
- **Ranging Markets**: Be cautious with scenario dominance

### 3. Risk Management
```python
# Always verify risk parameters
assert position_size <= config["max_position_size"]
assert leverage <= config["max_leverage"]
assert stop_loss < 0  # Stop loss should be negative
assert take_profit > 0  # Take profit should be positive

# Calculate maximum loss
max_loss = position_size * abs(stop_loss) * leverage
assert max_loss <= 0.02  # Max 2% account risk per trade
```

## Common Patterns

### Pattern 1: High Confidence Entry
```
Conditions:
- Profit zone > 70%
- Risk zone < 15%
- Model confidence > 80%
- Analyst confidence > 70%

Action: Full position with moderate leverage
```

### Pattern 2: Moderate Confidence Entry
```
Conditions:
- Profit zone 60-70%
- Risk zone 15-20%
- Model confidence 70-80%
- Analyst confidence > 60%

Action: Reduced position with minimum leverage
```

### Pattern 3: No Entry
```
Conditions:
- Profit zone < 60% OR
- Risk zone > 20% OR
- Model confidence < 70% OR
- Dominant zone != "profit"

Action: Wait for better setup
```

## Monitoring and Adjustment

### Real-time Monitoring
```python
# Track position performance
current_position = tactician.current_position
if current_position:
    pnl = (current_price - entry_price) / entry_price
    
    # Check exit conditions
    if pnl <= stop_loss or pnl >= take_profit:
        # Execute exit
        pass
```

### Performance Tracking
```python
# Get performance summary
performance = tactician.get_performance_summary()
print(f"Win Rate: {performance['performance_metrics']['win_rate']:.1%}")
print(f"Profit Factor: {performance['performance_metrics']['profit_factor']:.2f}")
```

## Troubleshooting

### Issue: No Entry Signals
- Check if model is trained
- Verify analyst confidence is > 0.5
- Review threshold settings
- Ensure market data quality

### Issue: Too Many False Signals
- Increase entry thresholds
- Require higher scenario dominance
- Add additional filters

### Issue: Poor Performance
- Review win rate and profit factor
- Adjust position sizing
- Optimize thresholds via step17
- Consider market regime changes

## Integration Checklist

- [ ] Initialize Tactician with proper config
- [ ] Ensure scenario predictor is trained
- [ ] Verify analyst inputs are valid
- [ ] Set appropriate risk limits
- [ ] Implement position tracking
- [ ] Monitor performance metrics
- [ ] Regular threshold optimization
- [ ] Logging for debugging