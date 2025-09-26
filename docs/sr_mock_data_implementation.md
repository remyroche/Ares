# SR Levels Mock Data Implementation

This document describes the comprehensive mock data implementation for the Support/Resistance (SR) levels system. The implementation provides realistic mock data generation for testing, development, and demonstration purposes.

## Overview

The mock data system consists of several components:

1. **SRMockDataGenerator** - Core mock data generation
2. **SRMockDataConfig** - Configuration management
3. **SRMockDataIntegration** - Integration with existing system
4. **SRMockDataManager** - Service management

## Features

### Mock Data Generation
- **Market Data**: Realistic OHLCV data with VWAP calculation
- **SR Levels**: Support and resistance levels with realistic properties
- **Trading Scenarios**: Breakout, bounce, and consolidation scenarios
- **Performance Metrics**: Comprehensive trading performance metrics

### Configuration Integration
- YAML-based configuration
- Configurable data points, seed, and output settings
- Validation and error handling
- Integration with existing SR levels configuration

### Data Export
- Multiple export formats (JSON, CSV, Parquet)
- Structured data organization
- Metadata and timestamps
- Configurable retention policies

## Quick Start

### Basic Usage

```python
from src.utils.sr_mock_data_generator import SRMockDataGenerator

# Create generator
generator = SRMockDataGenerator(seed=42)

# Generate market data
market_data = generator.generate_market_data(
    symbol="ETHUSDT",
    data_points=1000,
    start_price=3000.0
)

# Generate SR levels
sr_levels = generator.generate_sr_levels(market_data, num_levels=20)

# Generate trading scenarios
scenarios = generator.generate_trading_scenarios(
    market_data, sr_levels, num_scenarios=50
)

# Generate complete dataset
mock_data = generator.generate_complete_mock_dataset(
    data_points=1000,
    num_sr_levels=20,
    num_scenarios=50
)
```

### Configuration-Based Usage

```python
from src.config.sr_mock_data_config import create_mock_data_from_sr_config

# Load from configuration file
mock_data = create_mock_data_from_sr_config("config/sr_levels_config.yaml")
```

### Integration Usage

```python
from src.integration.sr_mock_data_integration import SRMockDataManager

# Create manager
manager = SRMockDataManager("config/sr_levels_config.yaml")

# Start service
manager.start_mock_data_service()

# Access data
market_data = manager.integration.get_market_data()
sr_levels = manager.integration.get_sr_levels()

# Export data
manager.export_all_mock_data("data/export")

# Stop service
manager.stop_mock_data_service()
```

## Configuration

The mock data system is configured through YAML files. Key configuration options:

```yaml
testing:
  # Mock data settings
  enable_mock_data: true
  mock_data_points: 1000
  mock_data_seed: 42
  mock_data_output_dir: "data/mock_sr_data"
  mock_data_validation: true
  mock_data_export_format: "json"
  mock_data_retention_days: 30

sr_levels_manager:
  max_levels: 20
  min_strength: 0.3
  proximity_threshold: 0.005

data_integration:
  symbol: "ETHUSDT"
  exchange: "BINANCE"
  timeframes: ["1m", "5m", "15m"]
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_mock_data` | bool | false | Enable/disable mock data generation |
| `mock_data_points` | int | 1000 | Number of market data points to generate |
| `mock_data_seed` | int | 42 | Random seed for reproducibility |
| `mock_data_output_dir` | str | "data/mock_sr_data" | Output directory for mock data |
| `mock_data_validation` | bool | true | Enable data validation |
| `mock_data_export_format` | str | "json" | Export format (json, csv, parquet) |
| `mock_data_retention_days` | int | 30 | Data retention period |

## Data Structures

### Market Data
Generated market data includes:
- **Timestamp**: ISO format timestamps
- **OHLC**: Open, High, Low, Close prices
- **Volume**: Trading volume
- **VWAP**: Volume Weighted Average Price

### SR Levels
Each SR level contains:
- **Price**: Level price
- **Type**: 'support' or 'resistance'
- **Strength**: Level strength (0.0-1.0)
- **Touch Count**: Number of touches
- **Timestamps**: First and last touch times
- **Bounce Rate**: Price bounce rate at level
- **Isolation Score**: Level isolation score
- **Volume**: Volume at level
- **Age**: Level age in days

### Trading Scenarios
Each scenario includes:
- **Scenario ID**: Unique identifier
- **Type**: 'breakout', 'bounce', 'false_breakout', 'consolidation'
- **Confidence**: Scenario confidence (0.0-1.0)
- **Risk/Reward**: Risk-reward ratio
- **Position Details**: Size, stop loss, take profit
- **Market Conditions**: Volatility, trend, volume

### Performance Metrics
Generated metrics include:
- **Success Rate**: Percentage of successful scenarios
- **PnL**: Total and average profit/loss
- **Risk Metrics**: Max drawdown, Sharpe ratio
- **Trading Stats**: Win rate, average win/loss
- **Performance Ratios**: Profit factor, recovery factor

## API Reference

### SRMockDataGenerator

#### `__init__(seed: int = 42)`
Initialize the mock data generator with a seed for reproducibility.

#### `generate_market_data(symbol: str, data_points: int, start_price: float, volatility: float, trend_strength: float) -> pd.DataFrame`
Generate realistic market data with OHLCV and VWAP.

#### `generate_sr_levels(market_data: pd.DataFrame, num_levels: int, min_strength: float, max_strength: float) -> List[SRLevel]`
Generate realistic Support/Resistance levels from market data.

#### `generate_trading_scenarios(market_data: pd.DataFrame, sr_levels: List[SRLevel], num_scenarios: int) -> List[Dict[str, Any]]`
Generate realistic trading scenarios based on market data and SR levels.

#### `generate_performance_metrics(scenarios: List[Dict[str, Any]], days: int) -> Dict[str, Any]`
Generate realistic performance metrics for the SR system.

#### `generate_complete_mock_dataset(data_points: int, num_sr_levels: int, num_scenarios: int, output_dir: str) -> Dict[str, Any]`
Generate a complete mock dataset for the SR levels system.

### SRMockDataConfig

#### `__init__(config_path: Optional[str] = None)`
Initialize the mock data configuration.

#### `is_mock_data_enabled() -> bool`
Check if mock data is enabled in configuration.

#### `get_mock_data_points() -> int`
Get the number of mock data points to generate.

#### `get_mock_data_seed() -> int`
Get the mock data seed for reproducibility.

#### `generate_mock_data() -> Dict[str, Any]`
Generate mock data based on current configuration.

#### `validate_mock_data_config() -> bool`
Validate the mock data configuration.

### SRMockDataIntegration

#### `__init__(config_path: Optional[str] = None)`
Initialize the mock data integration.

#### `initialize_mock_data() -> bool`
Initialize mock data if enabled in configuration.

#### `get_market_data() -> Optional[pd.DataFrame]`
Get mock market data.

#### `get_sr_levels() -> Optional[List[SRLevel]]`
Get mock SR levels.

#### `get_trading_scenarios() -> Optional[List[Dict[str, Any]]]`
Get mock trading scenarios.

#### `export_mock_data(output_dir: str) -> bool`
Export mock data to files.

### SRMockDataManager

#### `__init__(config_path: Optional[str] = None)`
Initialize the mock data manager.

#### `start_mock_data_service() -> bool`
Start the mock data service.

#### `stop_mock_data_service() -> bool`
Stop the mock data service.

#### `get_service_status() -> Dict[str, Any]`
Get the status of the mock data service.

#### `export_all_mock_data(output_dir: str) -> bool`
Export all mock data to files.

## Testing

The mock data system includes comprehensive tests:

```bash
# Run all tests
python -m pytest tests/test_sr_mock_data.py -v

# Run specific test class
python -m pytest tests/test_sr_mock_data.py::TestSRMockDataGenerator -v

# Run with coverage
python -m pytest tests/test_sr_mock_data.py --cov=src.utils.sr_mock_data_generator
```

### Test Coverage
- **Generator Tests**: Market data, SR levels, scenarios, metrics
- **Configuration Tests**: Loading, validation, generation
- **Integration Tests**: Data access, export, service lifecycle
- **Manager Tests**: Service management, data operations

## Examples

See `examples/sr_mock_data_example.py` for comprehensive usage examples:

- Basic mock data generation
- Configuration-based usage
- Integration patterns
- Advanced scenarios
- Service management

## Performance Considerations

### Memory Usage
- Market data: ~1MB per 1000 points
- SR levels: ~1KB per 100 levels
- Scenarios: ~10KB per 100 scenarios
- Total: ~1MB for typical 1000-point dataset

### Generation Speed
- Market data: ~100ms per 1000 points
- SR levels: ~50ms per 100 levels
- Scenarios: ~20ms per 100 scenarios
- Total: ~200ms for typical dataset

### Optimization Tips
- Use appropriate data point counts
- Enable caching for repeated operations
- Use efficient export formats
- Clean up temporary data regularly

## Troubleshooting

### Common Issues

1. **Configuration Not Found**
   ```
   FileNotFoundError: Could not find SR levels configuration file
   ```
   Solution: Ensure configuration file exists in expected location.

2. **Mock Data Disabled**
   ```
   ValueError: Mock data is disabled in configuration
   ```
   Solution: Set `enable_mock_data: true` in configuration.

3. **Invalid Configuration**
   ```
   Mock data configuration validation failed
   ```
   Solution: Check configuration values and format.

4. **Export Failures**
   ```
   Failed to export mock data
   ```
   Solution: Check output directory permissions and disk space.

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

When contributing to the mock data system:

1. Follow existing code patterns
2. Add comprehensive tests
3. Update documentation
4. Ensure backward compatibility
5. Validate configuration changes

## License

This mock data implementation is part of the SR levels system and follows the same licensing terms.