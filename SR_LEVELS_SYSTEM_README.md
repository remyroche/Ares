# SR Levels System - Comprehensive Support/Resistance Level Management

## Overview

The SR Levels System is a comprehensive solution for managing Support/Resistance levels in trading strategies. It provides:

1. **SR Level Calculation** based on backtesting data in step2_5
2. **Continuous Updates** during ongoing trading with real-time data
3. **Trading Intelligence Access** to SR levels with comprehensive metadata
4. **Price vs VWAP Comparison** logic for validation
5. **Persistent Storage** and retrieval of SR levels

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SR Levels System                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────────────────────┐  │
│  │ SRLevelsManager │  │     SR Trading Intelligence     │  │
│  │                 │  │                                 │  │
│  │ • Calculate     │  │ • Real-time access             │  │
│  │ • Store         │  │ • Trading decisions            │  │
│  │ • Update        │  │ • Risk assessment              │  │
│  │ • Filter        │  │ • Position recommendations     │  │
│  └─────────────────┘  └─────────────────────────────────┘  │
│           │                           │                    │
│           ▼                           ▼                    │
│  ┌─────────────────┐  ┌─────────────────────────────────┐  │
│  │   SR Breakout   │  │         Trading System          │  │
│  │   Predictor     │  │                                 │  │
│  │                 │  │ • Live trading                  │  │
│  │ • Detection     │  │ • Position management          │  │
│  │ • Analysis      │  │ • Performance tracking          │  │
│  │ • Optimization  │  └─────────────────────────────────┘  │
│  └─────────────────┘                                      │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. SRLevelsManager (`src/tactician/sr_levels_manager.py`)

The core component that manages SR levels throughout their lifecycle. **Fully integrated with SR breakout predictor logic.**

**Features:**
- Calculate SR levels from backtesting data using SR breakout predictor methods
- Continuous updates with live trading data
- Persistent storage with JSON files
- Level deduplication and filtering
- Quality scoring and ranking
- **Direct access to all SR detection methods** (fractal, volume, pivot, ATR)

**Key Methods:**
```python
# Calculate SR levels from backtesting (comprehensive)
await sr_manager.calculate_sr_levels_from_backtest(market_data, timeframe)

# Calculate SR levels with specific method
await sr_manager.calculate_sr_levels_with_method(market_data, "fractal", "both")

# Update levels with live data
await sr_manager.update_levels_with_live_data(price, volume, timestamp)

# Get levels for trading
trading_levels = sr_manager.get_sr_levels_for_trading(current_price)

# Compare price vs VWAP predictions
comparison = sr_manager.compare_price_vs_vwap_predictions(price_levels, vwap_levels)
```

**SR Detection Methods Available:**
- **Fractal Analysis**: Detects swing highs/lows using fractal patterns
- **Volume Analysis**: Volume-weighted price level detection
- **Pivot Analysis**: Traditional pivot point calculations
- **ATR Analysis**: Average True Range based level detection

### 2. SR Trading Intelligence (`src/trading/sr_trading_intelligence.py`)

Provides comprehensive access to SR levels for trading decisions.

**Features:**
- Real-time SR level access
- Trading decision support
- Risk assessment
- Position recommendations
- Performance tracking

**Key Methods:**
```python
# Get comprehensive trading data
data = trading_intelligence.get_sr_levels_for_trading(current_price)

# Update position
await trading_intelligence.update_position("long", entry_price, size, timestamp)

# Close position
await trading_intelligence.close_position(exit_price, timestamp)
```

### 3. SRLevel Class

Individual SR level with comprehensive metadata.

**Properties:**
- `price`: Level price
- `level_type`: "support" or "resistance"
- `method`: Detection method (fractal, volume, pivot, atr)
- `data_source`: "price" or "vwap"
- `strength`: Level strength (0.0 - 1.0)
- `touch_count`: Number of times price touched this level
- `age_hours`: Age of the level in hours
- `bounce_rate`: Rate of successful bounces
- `isolation_score`: How isolated this level is
- `confidence`: Detection confidence
- `metadata`: Additional information

## Installation and Setup

### 1. Dependencies

Ensure you have the required dependencies:
```bash
pip install pandas numpy scikit-learn
```

### 2. Configuration

Copy the configuration file:
```bash
cp config/sr_levels_config.yaml config/
```

Edit the configuration to match your requirements:
```yaml
sr_levels_manager:
  storage_path: "data/sr_levels"
  max_levels: 50
  min_strength: 0.3
```

### 3. Integration with step2_5

The SR levels system is automatically integrated into step2_5_sr_optimization.py. When you run step2_5, it will:

1. Perform SR detection optimization
2. Calculate SR levels from backtesting data
3. Store levels for subsequent steps
4. Generate comprehensive reports

## Usage Examples

### Basic Usage

```python
import asyncio
from src.tactician.sr_levels_manager import create_sr_levels_manager

async def main():
    # Configuration
    config = {
        "sr_levels_manager": {
            "storage_path": "data/sr_levels",
            "max_levels": 50,
            "min_strength": 0.3
        },
        "sr_breakout_predictor": {
            "sr_detection_method": "fractal",
            "max_sr_levels": 20,
            "min_sr_strength": 0.3
        }
    }

    # Initialize manager
    sr_manager = await create_sr_levels_manager(config)

    # Calculate SR levels from backtesting data (comprehensive)
    market_data = load_your_market_data()
    sr_levels = await sr_manager.calculate_sr_levels_from_backtest(market_data, "1m")

    print(f"Found {len(sr_levels['support_levels'])} support levels")
    print(f"Found {len(sr_levels['resistance_levels'])} resistance levels")

    # Calculate SR levels with specific method
    fractal_levels = await sr_manager.calculate_sr_levels_with_method(market_data, "fractal", "both")
    volume_levels = await sr_manager.calculate_sr_levels_with_method(market_data, "volume", "both")

    print(f"Fractal method: {len(fractal_levels['support_levels'])} support, {len(fractal_levels['resistance_levels'])} resistance")
    print(f"Volume method: {len(volume_levels['support_levels'])} support, {len(volume_levels['resistance_levels'])} resistance")

asyncio.run(main())
```

### Live Trading Integration

```python
from src.trading.sr_trading_intelligence import create_sr_trading_intelligence

async def trading_example():
    # Initialize trading intelligence
    intelligence = await create_sr_trading_intelligence(config)

    # Get SR levels for current price
    current_price = 100.0
    trading_data = intelligence.get_sr_levels_for_trading(current_price)

    # Analyze trading intelligence
    ti = trading_data["trading_intelligence"]
    print(f"Market position: {ti['market_position']}")
    print(f"Trend direction: {ti['trend_direction']}")
    print(f"Risk level: {ti['risk_level']}")

    # Get position recommendations
    recommendations = trading_data["position_recommendations"]
    for rec in recommendations:
        print(f"Action: {rec['action']} @ {rec['entry_price']}")
        print(f"Confidence: {rec['confidence']}")
        print(f"Reason: {rec['reason']}")
```

### Price vs VWAP Comparison

```python
# Create sample levels for comparison
price_levels = [
    SRLevel(price=100.0, level_type="support", method="fractal",
            data_source="price", timestamp=datetime.now(), strength=0.8)
]

vwap_levels = [
    SRLevel(price=100.2, level_type="support", method="fractal",
            data_source="vwap", timestamp=datetime.now(), strength=0.9)
]

# Compare predictions
comparison = sr_manager.compare_price_vs_vwap_predictions(price_levels, vwap_levels)

print(f"Price quality: {comparison['quality_metrics']['price']['avg_quality']:.3f}")
print(f"VWAP quality: {comparison['quality_metrics']['vwap']['avg_quality']:.3f}")
print(f"Overlap: {comparison['overlap_analysis']['overlap_rate']:.1%}")

# Get recommendations
for rec in comparison['recommendations']:
    print(f"- {rec}")
```

## Configuration Options

### SR Levels Manager

| Parameter | Default | Description |
|-----------|---------|-------------|
| `storage_path` | "data/sr_levels" | Directory for storing SR levels |
| `max_levels` | 50 | Maximum number of levels to maintain |
| `min_strength` | 0.3 | Minimum strength for level retention |
| `proximity_threshold` | 0.005 | Proximity threshold for deduplication |

### Trading Intelligence

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_real_time_updates` | true | Enable real-time SR level updates |
| `update_interval_seconds` | 60 | Update frequency in seconds |
| `max_position_size` | 0.1 | Maximum position size |
| `risk_tolerance` | 0.02 | Risk tolerance threshold |

### SR Breakout Predictor

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sr_detection_method` | "fractal" | Primary detection method |
| `max_sr_levels` | 20 | Maximum levels to detect |
| `min_sr_strength` | 0.3 | Minimum detection strength |
| `enable_vwap_detection` | true | Enable VWAP-based detection |

## Testing

Run the comprehensive test suite:

```bash
python test_sr_levels_system.py
```

This will test:
- SR level calculation from backtesting data
- Individual detection methods (fractal, volume, pivot, ATR)
- Continuous updates with live data
- Trading intelligence functionality
- Price vs VWAP comparison
- Persistent storage
- Performance metrics

### Validation

Run the import validation script to ensure proper integration:

```bash
python validate_sr_imports.py
```

This validates:
- All required imports are working
- SRBreakoutPredictor methods are accessible
- SRLevelsManager can use SR calculation logic
- Integration between components is functional

## File Structure

```
src/
├── tactician/
│   ├── sr_levels_manager.py          # Core SR levels management
│   ├── sr_breakout_predictor.py      # SR detection and analysis
│   └── sr_detection_optimization.py  # SR optimization
├── trading/
│   └── sr_trading_intelligence.py    # Trading intelligence
└── training/steps/
    └── step2_5_sr_optimization.py   # Integration with training pipeline

config/
└── sr_levels_config.yaml             # Configuration file

data/
└── sr_levels/                        # SR levels storage
    ├── sr_levels.json               # Current levels
    └── sr_levels_history.json       # Historical data

test_sr_levels_system.py             # Comprehensive test suite
```

## Performance Considerations

### Memory Usage
- Each SR level uses approximately 1KB of memory
- With 50 levels: ~50KB total memory usage
- Enable garbage collection for long-running processes

### Storage
- JSON format for human readability
- Consider database storage for high-frequency updates
- Implement data compression for historical data

### Update Frequency
- Real-time updates: Every 60 seconds (configurable)
- Batch processing for multiple updates
- Asynchronous processing to avoid blocking

## Troubleshooting

### Common Issues

1. **SR Levels Not Detected**
   - Check market data quality
   - Verify detection parameters
   - Ensure sufficient historical data

2. **High Memory Usage**
   - Reduce `max_levels` parameter
   - Enable garbage collection
   - Implement level cleanup

3. **Slow Updates**
   - Increase `update_interval_seconds`
   - Use batch processing
   - Optimize data loading

### Debug Mode

Enable debug logging:
```yaml
reporting:
  log_level: "DEBUG"
  enable_structured_logging: true
```

### Performance Monitoring

Monitor key metrics:
- Level count and quality
- Update frequency and latency
- Memory usage
- Storage size

## Future Enhancements

### Planned Features
- Machine learning-based level validation
- Multi-timeframe analysis
- Advanced clustering algorithms
- Real-time alerts and notifications
- Web dashboard for monitoring

### Integration Opportunities
- Exchange APIs for real-time data
- Database systems for scalability
- Message queues for high-frequency updates
- Cloud storage for backup and sharing

## Support

For questions and support:
1. Check the test suite for usage examples
2. Review configuration options
3. Monitor logs for error messages
4. Validate data quality and format

## License

This SR Levels System is part of the Ares Trading System and follows the same licensing terms.