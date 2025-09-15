# Dynamic Confidence TPSL Updates

## 🎯 Overview

The Enhanced A/B/C Testing Framework now supports **real-time dynamic TPSL updates** based on confidence scores from analysts and tacticians. This feature allows TPSL levels to be automatically adjusted whenever new confidence scores are released, providing more responsive and adaptive risk management.

## 🚀 Key Features

### Real-Time TPSL Adjustment
- **Automatic Updates**: TPSL levels are automatically recalculated when confidence scores change
- **Threshold-Based**: Only updates when confidence changes exceed a minimum threshold (default: 5%)
- **Position Tracking**: Each position maintains a history of confidence updates and TPSL adjustments
- **Callback System**: Register callbacks to respond to confidence update events

### Confidence Level System
- **High Confidence (≥0.8)**: More aggressive take profit (1.5x), tighter stop loss (0.8x)
- **Medium Confidence (≥0.6)**: Standard TPSL levels (1.0x)
- **Low Confidence (<0.6)**: Conservative take profit (0.8x), wider stop loss (1.2x)

### Configuration Options
- **Update Frequency**: Control how often updates occur (realtime, hourly, daily)
- **Change Threshold**: Minimum confidence change required to trigger TPSL update
- **Weight Configuration**: Adjust the relative importance of analyst vs tactician confidence
- **Multiplier Settings**: Customize TP/SL multipliers for each confidence level

## 📊 Implementation Details

### New Classes and Data Structures

#### `ActivePosition`
```python
@dataclass
class ActivePosition:
    symbol: str
    position_id: str
    entry_price: float
    entry_time: datetime
    position_side: OrderSide
    quantity: float
    current_tp_price: float
    current_sl_price: float
    tpsl_config: TPSLConfig
    last_confidence_update: datetime
    last_analyst_confidence: float
    last_tactician_confidence: float
    confidence_history: List[Tuple[datetime, float, float]]  # (timestamp, analyst_conf, tactician_conf)
    tpsl_update_history: List[Tuple[datetime, float, float]]  # (timestamp, tp_price, sl_price)
```

#### Enhanced `TPSLConfig`
```python
@dataclass
class TPSLConfig:
    # ... existing fields ...
    
    # Dynamic confidence update settings
    enable_dynamic_confidence_updates: bool = True
    confidence_update_frequency: str = "realtime"  # "realtime", "hourly", "daily"
    min_confidence_change_threshold: float = 0.05  # 5% minimum change
```

#### Enhanced `TPSLResult`
```python
@dataclass
class TPSLResult:
    # ... existing fields ...
    
    # Dynamic update tracking
    confidence_updates_count: int = 0
    tpsl_updates_count: int = 0
```

### New Methods

#### `create_position()`
Creates a new position with initial TPSL levels and confidence tracking.

#### `update_confidence_scores()`
Updates confidence scores for all positions of a symbol and recalculates TPSL levels.

#### `close_position()`
Closes a position and creates a TPSL result with update statistics.

#### `register_confidence_update_callback()`
Registers a callback function to be called when confidence scores are updated.

## 🔧 Usage Example

### Basic Setup
```python
from src.training.steps.backtesting.abc_testing.enhanced_abc_testing_framework import (
    TPSLManager, TPSLConfig, TPSLStrategy
)

# Configure dynamic confidence TPSL
tpsl_config = TPSLConfig(
    strategy=TPSLStrategy.CONFIDENCE_BASED,
    take_profit_pct=0.02,
    stop_loss_pct=0.01,
    confidence_threshold_high=0.8,
    confidence_threshold_medium=0.6,
    confidence_threshold_low=0.4,
    high_confidence_tp_multiplier=1.5,
    high_confidence_sl_multiplier=0.8,
    medium_confidence_tp_multiplier=1.0,
    medium_confidence_sl_multiplier=1.0,
    low_confidence_tp_multiplier=0.8,
    low_confidence_sl_multiplier=1.2,
    analyst_confidence_weight=0.6,
    tactician_confidence_weight=0.4,
    enable_dynamic_confidence_updates=True,
    confidence_update_frequency="realtime",
    min_confidence_change_threshold=0.05
)

# Initialize TPSL manager
tpsl_manager = TPSLManager(tpsl_config)
```

### Creating and Managing Positions
```python
# Create a position
position_id = tpsl_manager.create_position(
    symbol="BTCUSDT",
    entry_price=50000.0,
    position_side=OrderSide.BUY,
    quantity=1.0,
    market_data=market_data
)

# Update confidence scores (triggers TPSL recalculation)
tpsl_manager.update_confidence_scores(
    symbol="BTCUSDT",
    analyst_confidence=0.85,  # High confidence
    tactician_confidence=0.75,
    market_data=market_data
)

# Close position
result = tpsl_manager.close_position(
    position_id=position_id,
    exit_price=51000.0,
    exit_reason="take_profit"
)
```

### Callback Registration
```python
def on_confidence_update(symbol, analyst_confidence, tactician_confidence, updated_positions):
    print(f"Confidence updated for {symbol}")
    print(f"Analyst: {analyst_confidence:.2f}, Tactician: {tactician_confidence:.2f}")
    print(f"Updated {len(updated_positions)} positions")

# Register callback
tpsl_manager.register_confidence_update_callback(on_confidence_update)
```

## 📈 Performance Metrics

The enhanced system tracks additional metrics:

- **Confidence Updates per Trade**: Average number of confidence updates per position
- **TPSL Updates per Trade**: Average number of TPSL adjustments per position
- **Confidence Level Distribution**: Time spent in high/medium/low confidence states
- **TPSL Adjustment Frequency**: How often TPSL levels are modified

## 🎯 Benefits

### 1. **Responsive Risk Management**
- TPSL levels automatically adjust to changing market conditions and confidence levels
- More aggressive profit-taking when confidence is high
- More conservative risk management when confidence is low

### 2. **Human-AI Collaboration**
- Integrates human expertise (analyst/tactician confidence) with automated trading
- Allows for real-time adjustment based on human judgment
- Maintains automated execution while incorporating human insights

### 3. **Enhanced Performance Tracking**
- Detailed history of confidence changes and TPSL adjustments
- Ability to analyze the impact of confidence-based decisions
- Comprehensive metrics for strategy evaluation

### 4. **Flexible Configuration**
- Customizable confidence thresholds and multipliers
- Configurable update frequencies and change thresholds
- Support for different weighting schemes for analyst vs tactician confidence

## 🔄 Workflow

1. **Position Creation**: Create position with initial TPSL levels based on current confidence
2. **Confidence Monitoring**: Monitor for new confidence scores from analysts/tacticians
3. **Threshold Check**: Check if confidence change exceeds minimum threshold
4. **TPSL Recalculation**: Recalculate TPSL levels based on new confidence scores
5. **Position Update**: Update position with new TPSL levels
6. **History Tracking**: Record confidence and TPSL update history
7. **Callback Execution**: Execute registered callbacks for confidence updates
8. **Performance Tracking**: Track metrics for analysis and optimization

## 🚀 Future Enhancements

- **Machine Learning Integration**: Use ML models to predict optimal confidence thresholds
- **Multi-Asset Support**: Extend to support confidence updates across multiple assets
- **Advanced Analytics**: Enhanced analytics for confidence-based performance
- **Integration APIs**: APIs for external confidence score providers
- **Backtesting Support**: Historical confidence data for backtesting validation

## 📚 Related Files

- `enhanced_abc_testing_framework.py`: Core implementation of dynamic confidence TPSL
- `dynamic_confidence_tpsl_example.py`: Complete example demonstrating the feature
- `README.md`: Updated documentation with dynamic confidence examples
- `TPSL_STRATEGIES_UPDATE.md`: Summary of TPSL strategy changes

## ✅ Summary

The Dynamic Confidence TPSL Updates feature provides a powerful way to integrate human expertise with automated trading systems. By allowing real-time adjustment of TPSL levels based on analyst and tactician confidence scores, the framework enables more responsive and adaptive risk management while maintaining the benefits of automated execution.

This feature is particularly valuable for:
- **Quantitative Trading Teams**: Integrating human judgment with algorithmic strategies
- **Risk Management**: Providing more nuanced risk controls based on confidence levels
- **Strategy Development**: Testing the impact of confidence-based adjustments
- **Performance Optimization**: Analyzing the effectiveness of confidence-driven decisions