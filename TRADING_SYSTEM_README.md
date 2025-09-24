# Unified Trading System

A comprehensive, multi-exchange trading system that provides exchange-agnostic trading capabilities with robust error handling, configuration management, and live trading support.

## Architecture Overview

The trading system consists of several key components:

### 1. Live Trading Module (`live_trading/`)
- **Trading Orchestrator**: Main interface for submitting trading signals and managing trades
- **Unified Trading System**: High-level system that integrates all components
- **Order Manager**: Handles order lifecycle and execution
- **Data Streamer**: Manages real-time market data streaming
- **Configuration Management**: Centralized configuration for multiple exchanges
- **Error Handling**: Comprehensive error handling with retry logic

### 2. Exchange-Agnostic Receiver (`exchanges/`)
- **Trading Receiver**: Routes orders and data requests to appropriate exchanges
- **Order Router**: Manages order routing and tracking
- **Data Aggregator**: Aggregates data from multiple exchanges
- **Exchange Registry**: Manages exchange connections and health monitoring

### 3. Exchange APIs (`exchange/`)
- **Base Exchange**: Abstract interface for all exchange implementations
- **Exchange-Specific Implementations**: Binance, OKX, Gate.io, MEXC, Phemex
- **Factory Pattern**: Centralized exchange creation and configuration

## Key Features

### ✅ Multi-Exchange Support
- Support for 5 major exchanges: Binance, OKX, Gate.io, MEXC, Phemex
- Exchange-agnostic interface for seamless switching
- Failover and load balancing capabilities
- Unified API for all exchange operations

### ✅ Comprehensive Error Handling
- Categorized error types with appropriate recovery strategies
- Exponential backoff retry logic with jitter
- Error tracking and statistics
- Graceful degradation under failure conditions

### ✅ Advanced Configuration Management
- Multi-source configuration (environment, files, defaults)
- Runtime configuration updates
- Validation and type checking
- Configuration history and versioning

### ✅ Live Trading Capabilities
- Real-time order execution
- Market data streaming
- Position and risk management
- Performance tracking and metrics

### ✅ Robust Architecture
- Async/await throughout for high performance
- Type-safe interfaces with comprehensive validation
- Modular design for easy extension
- Comprehensive logging and monitoring

## Quick Start

### 1. Basic Setup

```python
import asyncio
from live_trading.unified_trading_system import create_trading_system
from live_trading.config import TradingConfig, TradingMode

async def main():
    # Create trading configuration
    config = TradingConfig(
        mode=TradingMode.PAPER,  # Use paper trading for testing
        symbols=["BTCUSDT", "ETHUSDT"],
        max_position_size=1000.0
    )

    # Create trading system
    system = await create_trading_system(
        trading_config=config,
        exchanges=["binance", "okx"],
        enable_paper_trading=True
    )

    # Start the system
    await system.start()

    # Submit a trading signal
    from live_trading.trading_orchestrator import TradingSignal

    signal = TradingSignal(
        symbol="BTCUSDT",
        action="buy",
        quantity=0.001,
        confidence=0.8,
        strategy="example_strategy"
    )

    success = await system.submit_trading_signal(signal)
    print(f"Signal submitted: {success}")

    # Get system status
    status = await system.get_system_status()
    print(f"System status: {status}")

    # Stop the system
    await system.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Configuration Management

```python
from live_trading.config_manager import create_config_manager, ConfigValidationLevel

async def config_example():
    # Create configuration manager
    config_manager = await create_config_manager(
        validation_level=ConfigValidationLevel.STRICT
    )

    # Update configuration
    updates = {
        "trading_config": {
            "max_position_size": 2000.0,
            "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        },
        "exchange_config": {
            "primary_exchange": "binance",
            "enable_failover": True
        }
    }

    await config_manager.update_configuration(updates)

    # Save configuration
    await config_manager.save_configuration("custom_config.json")

    # Get configuration status
    status = await config_manager.get_configuration_status()
    print(f"Configuration status: {status}")
```

### 3. Error Handling

```python
from live_trading.error_handler import create_default_error_handler, with_error_handling

# Create error handler
error_handler = create_default_error_handler()

# Use decorator for error handling
@with_error_handling(error_handler)
async def risky_operation():
    # Your trading logic here
    pass

# Handle errors manually
async def handle_with_retry():
    try:
        # Some operation that might fail
        result = await risky_operation()
    except Exception as e:
        # Handle error with retry logic
        result = await error_handler.handle_error(e, {"context": "example"})

    return result
```

## Advanced Usage

### Multi-Exchange Trading

```python
from live_trading.unified_trading_system import UnifiedTradingSystem, SystemConfig

async def multi_exchange_example():
    # Configure multiple exchanges
    system_config = SystemConfig(
        trading_config=TradingConfig(
            mode=TradingMode.LIVE,
            symbols=["BTCUSDT", "ETHUSDT"]
        ),
        exchanges=["binance", "okx", "gateio"],
        enable_websockets=True,
        enable_paper_trading=False
    )

    system = UnifiedTradingSystem(system_config)
    await system.initialize()
    await system.start()

    # Get account info from all exchanges
    account_summary = await system.get_account_summary()
    print(f"Account summary: {account_summary}")

    # Submit order to specific exchange
    from live_trading.trading_orchestrator import TradingSignal

    signal = TradingSignal(
        symbol="BTCUSDT",
        action="buy",
        quantity=0.001,
        confidence=0.85
    )

    # The system will automatically route to the best exchange
    await system.submit_trading_signal(signal)

    await system.stop()
```

### Custom Strategy Integration

```python
from live_trading.trading_orchestrator import TradingOrchestrator
from live_trading.unified_trading_system import UnifiedTradingSystem
from src.interfaces.base_interfaces import TradeDecision

class CustomStrategy:
    def __init__(self, trading_system: UnifiedTradingSystem):
        self.system = trading_system
        self.trading_orchestrator = None

    async def initialize(self):
        # Get the trading orchestrator from the system
        if hasattr(self.system, 'trading_orchestrator'):
            self.trading_orchestrator = self.system.trading_orchestrator

    async def generate_signals(self):
        # Your strategy logic here
        # Analyze market data, generate signals, etc.

        # Example: Create a trade decision
        decision = TradeDecision(
            symbol="BTCUSDT",
            action="buy",
            quantity=0.001,
            price=45000.0,  # Optional: current market price
            confidence=0.8,
            risk_score=0.3,
            leverage=1.0,
            stop_loss=44000.0,
            take_profit=47000.0
        )

        # Execute the decision
        if self.trading_orchestrator:
            success = await self.trading_orchestrator.execute_trade_decision(decision)
            return success

        return False
```

## Configuration

### Environment Variables

```bash
# Trading Configuration
TRADING_MODE=paper
EXCHANGE_NAME=binance
TRADING_SYMBOLS=BTCUSDT,ETHUSDT
MAX_POSITION_SIZE=1000.0
MAX_DAILY_LOSS=100.0

# Exchange Configuration
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_secret
OKX_API_KEY=your_okx_api_key
OKX_API_SECRET=your_okx_secret

# Primary exchange for routing
PRIMARY_EXCHANGE=binance
ENABLE_FAILOVER=true
FAILOVER_EXCHANGES=okx,gateio
```

### Configuration Files

Create `config/trading_config.json`:

```json
{
  "trading_config": {
    "mode": "paper",
    "exchange_name": "binance",
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "max_position_size": 1000.0,
    "max_daily_loss": 100.0,
    "max_leverage": 10.0,
    "stop_loss_percentage": 2.0,
    "take_profit_percentage": 4.0,
    "order_timeout": 30,
    "max_retries": 3,
    "retry_delay": 1.0,
    "data_update_interval": 1.0,
    "reconnect_attempts": 5,
    "reconnect_delay": 5.0
  },
  "exchange_config": {
    "primary_exchange": "binance",
    "exchanges": {
      "binance": {
        "name": "binance",
        "api_key": "your_binance_api_key",
        "api_secret": "your_binance_secret",
        "sandbox": false,
        "rate_limit": 1200,
        "timeout": 30,
        "enabled": true,
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "risk_limits": {},
        "custom_settings": {}
      },
      "okx": {
        "name": "okx",
        "api_key": "your_okx_api_key",
        "api_secret": "your_okx_secret",
        "sandbox": false,
        "rate_limit": 1200,
        "timeout": 30,
        "enabled": true,
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "risk_limits": {},
        "custom_settings": {}
      }
    },
    "enable_failover": true,
    "failover_exchanges": ["okx", "gateio"],
    "load_balancing": false,
    "load_balancing_strategy": "round_robin"
  }
}
```

## API Reference

### TradingOrchestrator

Main interface for trading operations:

```python
class TradingOrchestrator:
    async def submit_signal(self, signal: TradingSignal) -> bool
    async def execute_trade_decision(self, decision: TradeDecision) -> bool
    async def get_account_info(self) -> Dict[str, Any]
    async def get_positions(self) -> Dict[str, Position]
    async def get_market_data(self, symbol: str, data_type: str) -> Dict[str, Any]
    async def get_statistics(self) -> Dict[str, Any]
    async def pause_trading(self) -> None
    async def resume_trading(self) -> None
    async def emergency_stop(self) -> None
```

### UnifiedTradingSystem

High-level system interface:

```python
class UnifiedTradingSystem:
    async def initialize(self) -> None
    async def start(self) -> None
    async def stop(self) -> None
    async def submit_trading_signal(self, signal: TradingSignal) -> bool
    async def execute_trade_decision(self, decision: TradeDecision) -> bool
    async def get_account_summary(self) -> Dict[str, Any]
    async def get_system_status(self) -> Dict[str, Any]
    async def emergency_stop(self) -> None
    async def pause_trading(self) -> None
    async def resume_trading(self) -> None
```

### ConfigurationManager

Configuration management interface:

```python
class ConfigurationManager:
    async def load_configuration(self, sources: List[ConfigSource]) -> SystemConfig
    async def update_configuration(self, updates: Dict[str, Any]) -> SystemConfig
    async def save_configuration(self, file_path: str, format: str) -> None
    async def get_configuration_status(self) -> Dict[str, Any]
```

### ErrorHandler

Error handling interface:

```python
class ErrorHandler:
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> Optional[T]
    async def get_error_statistics(self) -> Dict[str, Any]
    async def clear_error_history(self) -> None
```

## Monitoring and Logging

The system provides comprehensive monitoring capabilities:

### System Metrics
- Order execution statistics
- Error rates and categories
- Performance metrics
- Exchange connectivity status
- Configuration change history

### Logging
- Structured logging with levels
- Error tracking with context
- Performance monitoring
- Audit trail for configuration changes

### Health Monitoring
- Exchange connectivity checks
- API rate limit monitoring
- System resource monitoring
- Automatic failover detection

## Security Considerations

### API Key Management
- Store API keys securely (environment variables, key management systems)
- Never log API keys or secrets
- Use separate keys for different environments

### Network Security
- SSL/TLS for all API communications
- IP whitelisting where supported
- Request signing and validation

### Risk Management
- Position size limits
- Daily loss limits
- Maximum leverage controls
- Order validation

## Testing

### Paper Trading
Always test strategies in paper trading mode first:

```python
config = TradingConfig(mode=TradingMode.PAPER)
```

### Unit Testing
```python
import unittest
from live_trading.unified_trading_system import create_trading_system

class TestTradingSystem(unittest.TestCase):
    async def test_paper_trading(self):
        system = await create_trading_system(
            trading_config=TradingConfig(mode=TradingMode.PAPER)
        )
        # Test your logic here
        await system.stop()
```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Verify API keys are correct and have proper permissions
2. **Rate Limiting**: Check rate limits and implement proper delays
3. **Network Issues**: Verify internet connection and exchange status
4. **Configuration Errors**: Validate configuration files and environment variables

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Error Analysis

```python
# Get detailed error statistics
error_stats = await error_handler.get_error_statistics()
print(f"Error statistics: {error_stats}")

# Get system status
system_status = await system.get_system_status()
print(f"System status: {system_status}")
```

## Contributing

### Adding New Exchanges

1. Implement the exchange interface in `exchange/`
2. Add factory method in `exchange/factory.py`
3. Update configuration schemas
4. Add tests and documentation

### Extending Functionality

1. Follow the existing architecture patterns
2. Use the error handling system
3. Add comprehensive tests
4. Update documentation

## License

This trading system is provided for educational and research purposes. Use at your own risk.

## Disclaimer

This software is for educational purposes only. Trading cryptocurrencies involves substantial risk of loss and is not suitable for every investor. Past performance does not guarantee future results. Only trade with money you can afford to lose.