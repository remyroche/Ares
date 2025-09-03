# Tactician Data Flow and Integration

## Complete Data Flow Diagram

```
┌─────────────────────┐
│   Market Data       │
│   (OHLCV)          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  Enhanced Scenario-Based Predictor       │
│  ┌─────────────────────────────────┐    │
│  │ Feature Extraction:             │    │
│  │ - Technical Indicators          │    │
│  │ - Price Patterns               │    │
│  │ - Volume Analysis              │    │
│  └──────────┬──────────────────────┘    │
│             ▼                            │
│  ┌─────────────────────────────────┐    │
│  │ Scenario Prediction:            │    │
│  │ - 17 Scenarios (8P + 8R + 1N)  │    │
│  │ - Probability Distribution      │    │
│  │ - Confidence Score              │    │
│  └─────────────────────────────────┘    │
└──────────┬──────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│                TACTICIAN CORE                         │
│  ┌────────────────────┐  ┌────────────────────┐     │
│  │ Analyst Inputs:    │  │ Scenario Analysis: │     │
│  │ - Barriers         │  │ - Profit Zone Prob │     │
│  │ - Confidence       │  │ - Risk Zone Prob   │     │
│  └────────┬───────────┘  │ - Risk/Reward      │     │
│           │              └────────┬───────────┘      │
│           ▼                       ▼                   │
│  ┌─────────────────────────────────────────────┐     │
│  │         DECISION ENGINE                      │     │
│  │ ┌─────────────────┐  ┌──────────────────┐  │     │
│  │ │ Entry Logic:    │  │ Exit Logic:      │  │     │
│  │ │ - All conditions│  │ - Any condition  │  │     │
│  │ │   must be met   │  │   triggers exit  │  │     │
│  │ └─────────────────┘  └──────────────────┘  │     │
│  └─────────────────────────────────────────────┘     │
└──────────┬───────────────────────────────────────────┘
           │
           ├─────────────────┬──────────────────┐
           ▼                 ▼                  ▼
┌─────────────────┐ ┌──────────────┐  ┌──────────────┐
│ Position Sizer  │ │Leverage Sizer│  │Risk Manager  │
│ ┌─────────────┐ │ │┌────────────┐│  │┌────────────┐│
│ │Dual Conf:   │ │ ││ML Leverage:││  ││Stop Loss:  ││
│ │A * T²       │ │ ││10x - 100x  ││  ││Analyst * M ││
│ └─────────────┘ │ │└────────────┘│  │└────────────┘│
│ ┌─────────────┐ │ │┌────────────┐│  │┌────────────┐│
│ │Size Calc:   │ │ ││Risk Adjust:││  ││Take Profit:││
│ │0.01 - 0.1   │ │ ││Liquidation ││  ││Analyst * M ││
│ └─────────────┘ │ │└────────────┘│  │└────────────┘│
└─────────────────┘ └──────────────┘  └──────────────┘
           │                 │                  │
           └─────────────────┴──────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ FINAL OUTPUT   │
                    │ - Direction    │
                    │ - Position Size│
                    │ - Leverage     │
                    │ - SL/TP Levels │
                    └────────────────┘
```

## Key Integration Points

### 1. Analyst → Tactician
- **Barriers**: Direct input for stop loss and take profit calculations
- **Confidence**: Multiplicative factor in decision making and sizing

### 2. Scenario Predictor → Decision Engine
- **Probabilities**: Zone aggregation for profit/risk assessment
- **Confidence**: Model certainty affects all downstream calculations

### 3. Decision Engine → Position Management
- **Entry/Exit Signals**: Binary decisions gate all position actions
- **Confidence Scores**: Scale position size and leverage

## Calculation Examples

### Example 1: Strong Signal
```
Inputs:
- Analyst Confidence: 0.8
- Analyst Barriers: Upper=2%, Lower=-1%
- Scenario Prediction: 75% profit zone, 15% risk zone
- Model Confidence: 0.85

Calculations:
- Dual Confidence: 0.8 * 0.85² = 0.578
- Entry Signal: YES (all conditions met)
- Position Size: 0.08 (8% of capital)
- Leverage: 25x
- Stop Loss: -1%
- Take Profit: 2%
```

### Example 2: Weak Signal
```
Inputs:
- Analyst Confidence: 0.5
- Analyst Barriers: Upper=1%, Lower=-0.5%
- Scenario Prediction: 45% profit zone, 35% risk zone
- Model Confidence: 0.6

Calculations:
- Dual Confidence: 0.5 * 0.6² = 0.18
- Entry Signal: NO (profit zone < 60%)
- Position Size: 0.01 (minimum)
- Leverage: 10x (minimum)
```

## Configuration Impact

### Entry Thresholds Effect:
- **Higher profit_threshold**: More selective, fewer trades
- **Lower risk_threshold**: More conservative, avoid risky setups
- **Higher confidence_threshold**: Wait for stronger signals

### Sizing Parameters Effect:
- **kelly_multiplier**: Scales theoretical optimal size
- **max_position_size**: Hard cap on exposure
- **leverage_multiplier**: Scales base leverage calculation

## Performance Metrics

The system tracks:
1. **Win Rate**: Percentage of profitable trades
2. **Profit Factor**: Total profit / Total loss
3. **Sharpe Ratio**: Risk-adjusted returns
4. **Max Drawdown**: Largest peak-to-trough decline

These metrics feed back into step17 optimization for continuous improvement.