# Position-Aware Trading Utilities

## Overview

The `position_aware_trading.py` module provides shared utilities for both TAS and NAS regime detection systems to calculate position-aware win rates and trading viability metrics. This ensures accurate performance evaluation for both long and short positions.

## Key Features

- **Position-Aware Win Rates**: Separate win rate calculations for long and short positions
- **Economic Significance**: Position-aware economic significance evaluation
- **Trading Viability**: Position-aware trading viability assessment
- **Shared Utilities**: Used by both TAS and NAS systems for consistency

## Quick Usage

```python
from ..shared_utils.position_aware_trading import (
    PositionAwareTradingAnalyzer, PositionAwareConfig,
    create_position_aware_analyzer, quick_position_aware_analysis
)

# Create analyzer with default configuration
analyzer = create_position_aware_analyzer()

# Quick analysis
result = quick_position_aware_analysis(
    market_data, regime_predictions, position_directions
)

# Access results
overall_win_rate = result['overall_analysis']['overall_win_rate']
long_win_rate = result['overall_analysis']['long_win_rate']
short_win_rate = result['overall_analysis']['short_win_rate']
```

## Position-Aware Win Rate Calculation

### Long Positions
- **Profit Condition**: `returns > minimum_profit_threshold`
- **Example**: If price increases by 0.1%, it's a win for longs

### Short Positions
- **Profit Condition**: `returns < -minimum_profit_threshold`
- **Example**: If price decreases by 0.1%, it's a win for shorts

### Overall Win Rate
- **Profit Condition**: `|returns| > minimum_profit_threshold`
- **Example**: Any directional movement counts as a win

## Configuration

```python
config = PositionAwareConfig(
    minimum_profit_threshold=0.001,  # 0.1% minimum profit
    transaction_cost=0.001,          # 0.1% transaction cost
    position_holding_periods=[1, 5, 10, 20],  # Analysis periods
    risk_free_rate=0.02,             # 2% annual risk-free rate
    win_rate_thresholds={
        'excellent': 0.7,
        'good': 0.6,
        'acceptable': 0.5,
        'poor': 0.4
    }
)
```

## TAS Integration

The TAS regime detector automatically uses position-aware utilities:

```python
from tas_regime.core.tas_regime_detector import TASRegimeDetector

detector = TASRegimeDetector(config)
result = detector.detect_regimes(market_data)

# Position-aware metrics are included in result metadata
position_analysis = result.metadata['position_aware']
```

## NAS Integration

The NAS regime detector also uses the same shared utilities:

```python
from nas_regime.core.perfect_nas_regime_detector import PerfectNASRegimeDetector

detector = PerfectNASRegimeDetector(config)
result = detector.detect_regimes(market_data)

# Position-aware metrics are included in result metadata
position_analysis = result.metadata['adaptive_thresholds']
```

## API Reference

### PositionAwareTradingAnalyzer

#### Methods

- `calculate_position_aware_win_rate(returns, position_directions)`: Calculate position-aware win rates
- `analyze_regime_position_performance(market_data, regime_labels, position_directions)`: Analyze per-regime performance
- `calculate_position_aware_trading_viability(market_data, regime_predictions, position_directions)`: Calculate viability scores
- `get_position_aware_recommendations(position_analysis)`: Get trading recommendations

#### Configuration

- `minimum_profit_threshold`: Minimum return to count as profit (default: 0.001)
- `transaction_cost`: Transaction costs to subtract from returns (default: 0.001)
- `position_holding_periods`: List of periods to analyze (default: [1, 5, 10, 20])
- `risk_free_rate`: Risk-free rate for Sharpe calculations (default: 0.02)

## Benefits

1. **Consistency**: Both TAS and NAS use identical position-aware calculations
2. **Accuracy**: Proper win rate calculation for both long and short positions
3. **Flexibility**: Easy to adjust profit thresholds and transaction costs
4. **Extensibility**: Framework for adding more position-aware metrics

## Example Output

```python
{
    'overall_analysis': {
        'overall_win_rate': 0.65,
        'long_win_rate': 0.72,
        'short_win_rate': 0.58,
        'total_trades': 1000,
        'long_trades': 520,
        'short_trades': 480
    },
    'position_balance_analysis': {
        'position_balance_score': 0.92,
        'diversification_benefit': 0.15,
        'long_short_correlation': 0.23,
        'position_stability': 0.78
    }
}
```

## Error Handling

The utilities include comprehensive error handling with fallback to default values:

- Missing position directions → Infer from returns
- Invalid market data → Use default significance scores
- Calculation errors → Return neutral values (0.5)

This ensures robust operation even with incomplete or noisy data.