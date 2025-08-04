# Position Monitoring Configuration

## Overview

The Position Monitor provides real-time monitoring of open positions with confidence score re-assessment and position decision logic every 10 seconds. It uses the existing PositionDivisionStrategy for consistent position logic and removes duplicate code.

## Key Features

- **Real-time Monitoring**: Continuously monitors open positions every 10 seconds
- **Confidence Re-assessment**: Re-evaluates confidence scores and position decisions
- **Position Actions**: Supports stay, exit, scale up, scale down, hedge, take profit, stop loss, and full close actions
- **Risk Assessment**: Evaluates risk levels based on position size, leverage, and market conditions
- **Market Condition Analysis**: Assesses market volatility and trend strength
- **Integration**: Uses existing PositionDivisionStrategy for consistent logic

## Configuration

### Basic Configuration

```python
config = {
    "position_monitoring_interval": 10,  # 10 seconds
    "max_assessment_history": 1000,
    "high_risk_threshold": 0.8,
    "medium_risk_threshold": 0.6,
    "low_risk_threshold": 0.3,
    "position_division": {
        # Position division strategy configuration
        "entry_confidence_threshold": 0.7,
        "additional_position_threshold": 0.8,
        "max_positions": 3,
        "take_profit_confidence_decrease": 0.1,
        "take_profit_short_term_decrease": 0.08,
        "stop_loss_confidence_threshold": 0.3,
        "stop_loss_short_term_threshold": 0.24,
        "stop_loss_price_threshold": -0.05,
        "full_close_confidence_threshold": 0.2,
        "full_close_short_term_threshold": 0.16,
        "ml_confidence_weight": 0.6,
        "price_action_weight": 0.2,
        "volume_weight": 0.2,
    }
}
```

### Configuration Parameters

#### Position Monitor Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `position_monitoring_interval` | 10 | Monitoring interval in seconds |
| `max_assessment_history` | 1000 | Maximum number of assessments to keep in history |
| `high_risk_threshold` | 0.8 | Threshold for high risk assessment |
| `medium_risk_threshold` | 0.6 | Threshold for medium risk assessment |
| `low_risk_threshold` | 0.3 | Threshold for low risk assessment |

#### Position Division Strategy Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `entry_confidence_threshold` | 0.7 | Minimum confidence for new position entry |
| `additional_position_threshold` | 0.8 | Confidence threshold for additional positions |
| `max_positions` | 3 | Maximum number of concurrent positions |
| `take_profit_confidence_decrease` | 0.1 | Confidence decrease threshold for take profit |
| `take_profit_short_term_decrease` | 0.08 | Short-term confidence decrease for take profit |
| `stop_loss_confidence_threshold` | 0.3 | Confidence threshold for stop loss |
| `stop_loss_short_term_threshold` | 0.24 | Short-term confidence threshold for stop loss |
| `stop_loss_price_threshold` | -0.05 | Price change threshold for trailing stop loss |
| `full_close_confidence_threshold` | 0.2 | Confidence threshold for full position closure |
| `full_close_short_term_threshold` | 0.16 | Short-term confidence threshold for full closure |

## Position Actions

The Position Monitor can recommend the following actions:

### Action Types

1. **STAY**: Maintain current position (default when conditions are acceptable)
2. **EXIT**: Close the position (low confidence or significant drop)
3. **SCALE_UP**: Increase position size (very high confidence)
4. **SCALE_DOWN**: Reduce position size (high volatility)
5. **HEDGE**: Add hedge position (high risk but good confidence)
6. **TAKE_PROFIT**: Take partial profit (confidence decrease)
7. **STOP_LOSS**: Stop loss (low confidence or trailing stop)
8. **FULL_CLOSE**: Close entire position (very low confidence)

### Action Decision Logic

The monitor uses the following priority order for action decisions:

1. **Full Close** (highest priority): Very low confidence or dramatic drops
2. **Stop Loss**: Low confidence or trailing stop triggered
3. **Take Profit**: Moderate confidence decreases
4. **Scale Up**: Very high confidence opportunities
5. **Hedge**: High risk with acceptable confidence
6. **Scale Down**: High volatility conditions
7. **Stay**: Default when conditions are acceptable

## Integration with Tactician

The Position Monitor is integrated into the Tactician component:

```python
# In tactician initialization
await self._initialize_position_monitor()

# Position monitor is automatically started and runs in background
# It continuously monitors positions and updates assessments
```

### Position Data Structure

Positions should include the following data:

```python
position_data = {
    "symbol": "ETHUSDT",
    "direction": "LONG",
    "entry_price": 1850.0,
    "current_price": 1860.0,
    "position_size": 0.1,
    "leverage": 1.0,
    "entry_confidence": 0.75,
    "entry_timestamp": "2024-01-01T12:00:00",
    "time_in_position_hours": 2.5,
    "market_volatility": 0.15,
    "trend_strength": 0.6,
    "base_confidence": 0.7,
    "short_term_analysis": {
        # Optional short-term analysis data
    }
}
```

## Usage Example

```python
from src.tactician.position_monitor import setup_position_monitor

# Setup position monitor
config = {
    "position_monitoring_interval": 10,
    "position_division": {
        "entry_confidence_threshold": 0.7,
        "stop_loss_confidence_threshold": 0.3,
        # ... other settings
    }
}

position_monitor = await setup_position_monitor(config)

# Add positions to monitor
position_monitor.add_position("pos_001", position_data)

# Monitor runs automatically in background
# Get assessment history
assessments = position_monitor.get_assessment_history(limit=10)

# Get position status
status = position_monitor.get_position_status("pos_001")

# Stop monitoring
await position_monitor.stop_monitoring()
```

## Assessment Results

Each assessment includes:

```python
PositionAssessment(
    position_id="pos_001",
    current_confidence=0.65,
    entry_confidence=0.75,
    confidence_change=-0.10,
    market_conditions="neutral",
    risk_level="medium",
    recommended_action=PositionAction.STAY,
    action_reason="Conditions acceptable - maintain position",
    assessment_timestamp=datetime.now(),
    next_assessment=datetime.now() + timedelta(seconds=10),
    division_analysis={...}  # Full division strategy analysis
)
```

## Testing

Run the test script to see the position monitor in action:

```bash
python test_position_monitor.py
```

This will demonstrate:
- Real-time position monitoring every 10 seconds
- Confidence score re-assessment
- Position action recommendations
- Risk and market condition analysis

## Benefits

1. **Consistent Logic**: Uses existing PositionDivisionStrategy to avoid duplicate code
2. **Real-time Monitoring**: Continuous assessment every 10 seconds
3. **Comprehensive Actions**: Supports all major position management actions
4. **Risk Management**: Integrated risk assessment and management
5. **Market Awareness**: Considers market conditions and volatility
6. **Configurable**: Highly configurable thresholds and parameters
7. **Integration Ready**: Seamlessly integrates with existing trading system

## Monitoring and Logging

The Position Monitor provides detailed logging:

- Position additions/removals
- Assessment results with confidence changes
- Action recommendations with reasons
- Risk level and market condition analysis
- Error handling and recovery

All logs are structured and include timestamps for easy monitoring and debugging. 