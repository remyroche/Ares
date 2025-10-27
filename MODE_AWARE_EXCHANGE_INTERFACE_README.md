# Mode-Aware Exchange Interface

This implementation provides a mode-aware ExchangeInterface that can operate in either **TRADE** mode (live trading) or **PAPER** mode (simulation) based on configuration flags.

## Overview

The mode-aware system consists of several key components:

1. **ModeAwareExchangeInterface** - Main interface that routes orders based on mode
2. **OrderSimulator** - Handles order simulation for paper trading
3. **PortfolioSimulator** - Manages portfolio simulation and position tracking
4. **TradingModeConfig** - Configuration management for mode switching

## Features

### TRADE Mode
- Routes orders to real exchanges
- Uses existing OrderRouter and ExchangeRegistry
- Provides real-time market data and order execution
- Full integration with live trading infrastructure

### PAPER Mode
- Routes orders to simulator instead of real exchanges
- Fetches order book data for accurate simulation
- Tracks simulated positions and portfolio value
- Provides realistic order execution simulation

## Directory Structure

```
simulator/
├── __init__.py
├── simulator_interface.py      # Interface definitions
├── order_simulator.py          # Order simulation logic
└── portfolio_simulator.py      # Portfolio management

exchanges/
├── mode_aware_exchange_interface.py  # Main mode-aware interface
├── trading_mode_config.py            # Configuration management
└── mode_aware_integration.py         # Simple integration client

examples/
└── mode_aware_trading_example.py     # Usage examples
```

## Usage

### Basic Usage

```python
from exchanges.mode_aware_integration import create_trading_client

# Create client (automatically loads config from environment)
client = await create_trading_client()

# Place orders (automatically routes to simulator or real exchange)
buy_result = await client.buy("BTCUSDT", 0.1, 49000.0)  # Limit buy
sell_result = await client.sell("BTCUSDT", 0.05)         # Market sell

# Get positions and balance
positions = await client.get_positions("BTCUSDT")
balance = await client.get_balance()

await client.stop()
```

### Configuration

Set environment variables to control the mode:

```bash
# Paper trading mode (default)
export TRADING_MODE=PAPER
export INITIAL_BALANCE=100000.0

# Live trading mode
export TRADING_MODE=TRADE
```

### Advanced Usage

```python
from exchanges.mode_aware_exchange_interface import ModeAwareExchangeInterface, ModeAwareConfig, TradingMode

# Create custom configuration
config = ModeAwareConfig(
    mode=TradingMode.PAPER,
    initial_balance=50000.0,
    enable_order_book_simulation=True,
    log_trades=True
)

# Create interface
interface = ModeAwareExchangeInterface(config=config)
await interface.initialize()

# Update market data for simulation
await interface.update_market_data("BTCUSDT", {
    "price": 50000.0,
    "bid": 49999.0,
    "ask": 50001.0
})

# Process order book for accurate simulation
order_book = {
    "bids": [[49950.0, 0.5], [49900.0, 1.0]],
    "asks": [[50050.0, 0.5], [50100.0, 1.0]]
}
await interface.process_order_book("BTCUSDT", order_book)

# Create orders
order_result = await interface.create_order(
    exchange="simulator",  # Ignored in paper mode
    symbol="BTCUSDT",
    side="BUY",
    order_type="MARKET",
    quantity=0.1
)

await interface.close()
```

## Key Components

### OrderSimulator

Handles order simulation with features:
- Market and limit order processing
- Realistic fill simulation using order book data
- Position tracking and portfolio management
- Commission calculation
- Order status monitoring

### PortfolioSimulator

Manages simulated portfolio with features:
- Position tracking (long/short)
- PnL calculation (realized and unrealized)
- Trade history
- Performance metrics
- Risk management

### ModeAwareExchangeInterface

Main interface that provides:
- Mode-aware order routing
- Unified API for both modes
- Market data integration
- Order book processing
- Statistics and monitoring

## Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TRADING_MODE` | `PAPER` | Trading mode (PAPER/TRADE) |
| `INITIAL_BALANCE` | `100000.0` | Starting balance for simulation |
| `ENABLE_ORDER_BOOK_SIMULATION` | `true` | Use order book for accurate fills |
| `SIMULATION_COMMISSION_RATE` | `0.001` | Commission rate for simulation |
| `LOG_TRADES` | `true` | Enable trade logging |
| `ENABLE_RISK_MANAGEMENT` | `true` | Enable risk management |
| `MAX_POSITION_SIZE` | `10000.0` | Maximum position size |
| `MAX_DAILY_LOSS` | `5000.0` | Maximum daily loss limit |

## Integration with Existing System

The mode-aware interface integrates seamlessly with the existing exchange infrastructure:

1. **TRADE Mode**: Uses existing `OrderRouter` and `ExchangeRegistry`
2. **PAPER Mode**: Uses new simulator components
3. **Unified API**: Same interface for both modes
4. **Configuration**: Environment-based mode switching

## Benefits

1. **Risk-Free Testing**: Test strategies in paper mode before live trading
2. **Seamless Switching**: Switch between modes without code changes
3. **Realistic Simulation**: Order book data provides accurate fills
4. **Portfolio Tracking**: Complete position and PnL tracking
5. **Unified Interface**: Single API for both modes

## Example Scenarios

### Strategy Development
```python
# Develop and test strategy in paper mode
client = await create_trading_client()  # Uses PAPER mode by default

# Test strategy
for signal in strategy_signals:
    if signal.action == "BUY":
        await client.buy(signal.symbol, signal.quantity, signal.price)
    elif signal.action == "SELL":
        await client.sell(signal.symbol, signal.quantity, signal.price)

# Analyze results
stats = await client.get_statistics()
print(f"Strategy performance: {stats}")
```

### Live Trading
```python
# Switch to live trading
import os
os.environ["TRADING_MODE"] = "TRADE"

client = await create_trading_client()  # Now uses TRADE mode

# Execute real trades
await client.buy("BTCUSDT", 0.1, 49000.0)
```

### Backtesting with Realistic Fills
```python
# Use order book data for realistic simulation
client = await create_trading_client()

# Update with historical order book data
for timestamp, order_book in historical_data:
    await client.update_market_data("BTCUSDT", order_book["price"])
    await client.process_order_book("BTCUSDT", order_book)
    
    # Execute strategy
    if should_buy():
        await client.buy("BTCUSDT", quantity, price)
```

## Error Handling

The system includes comprehensive error handling:
- Mode validation
- Order validation
- Exchange connectivity checks
- Simulator state management
- Graceful fallbacks

## Monitoring and Logging

- Trade logging (configurable)
- Performance metrics
- Order statistics
- Portfolio tracking
- Error logging

## Future Enhancements

1. **Risk Management**: Advanced risk controls
2. **Slippage Modeling**: Realistic slippage simulation
3. **Latency Simulation**: Network latency modeling
4. **Market Impact**: Order size impact simulation
5. **Multi-Exchange**: Support for multiple exchanges in simulation

## Testing

Run the example to test the implementation:

```bash
python examples/mode_aware_trading_example.py
```

This will demonstrate both paper and live trading modes with various order types and scenarios.