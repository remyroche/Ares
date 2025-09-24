# Perpetual Futures Live Trading System

This document describes the complete live trading system architecture that has been implemented for perpetual futures trading.

## Architecture Overview

The live trading system follows a modular, exchange-agnostic architecture optimized for perpetual futures trading with the following key components:

### 1. Live Trading Module (`live_trading/`)

**Core Components:**
- `TradingManager`: Main coordinator for all trading operations
- `OrderManager`: Handles order placement, tracking, and validation
- `DataReceiver`: Manages market data streaming and processing
- `TradeExecutor`: Executes trades and manages positions
- `EventSystem`: Event-driven architecture for system communication

### 2. Exchange-Agnostic Receiver (`exchange/`)

**Components:**
- `ExchangeOrderReceiver`: Central hub for order routing and data requests
- `BaseExchange`: Abstract interface for all exchange implementations
- `BinanceExchange`, `OkxExchange`, `GateioExchange`, `MexcExchange`: Complete exchange implementations
- `ExchangeFactory`: Factory pattern for creating exchange instances

## Key Features

### ✅ Exchange-Agnostic Design
- Single interface for multiple exchanges (Binance, OKX, GateIO, MEXC)
- Easy to add new exchanges by implementing BaseExchange interface
- Automatic exchange selection and failover

### ✅ Comprehensive Order Management
- Order placement with validation and risk checks
- Real-time order status monitoring
- Automatic retry mechanisms
- Priority-based order queuing

### ✅ Real-Time Data Streaming
- Live market data from all supported exchanges
- Multiple timeframes and data aggregation
- Subscription-based data updates
- Historical data retrieval

### ✅ Risk Management
- Position size limits and daily trade limits
- Pre-trade validation and risk scoring
- Automatic stop-loss and take-profit handling
- Risk per trade controls

### ✅ Event-Driven Architecture
- Asynchronous event handling
- Priority-based event processing
- Extensible event system
- Real-time notifications

### ✅ Production-Ready Features
- Comprehensive error handling and logging
- Configuration management
- Connection pooling and rate limiting
- Backtesting compatibility

### ✅ Advanced Position Management
- Open positions with size & leverage, receive trade ID
- Close positions using trade ID
- Get detailed trade information by trade ID
- Get asset data formatted as klines
- Real-time position tracking and P&L calculation

## Directory Structure

```
/workspace/
├── live_trading/
│   ├── __init__.py
│   ├── trading_manager.py          # Main trading coordinator
│   ├── order_manager.py            # Order handling and routing
│   ├── data_receiver.py            # Market data streaming
│   ├── trade_executor.py           # Trade execution engine
│   ├── event_system.py             # Event-driven architecture
│   ├── config/
│   │   ├── __init__.py
│   │   └── trading_config.py       # Configuration management
│   └── execution/                  # Additional execution modules
├── exchange/
│   ├── __init__.py
│   ├── base_exchange.py            # Exchange interface
│   ├── binance.py                  # Complete Binance implementation
│   ├── okx.py                      # Complete OKX implementation
│   ├── gateio.py                   # Complete GateIO implementation
│   ├── mexc.py                     # Complete MEXC implementation
│   ├── factory.py                  # Exchange factory
│   └── order_receiver.py           # Exchange-agnostic receiver
├── live_trading_example.py         # Complete usage example
└── LIVE_TRADING_README.md          # This documentation
```

## Usage Example

### Basic Setup

```python
from live_trading.trading_manager import TradingManager, TradingConfig
from live_trading.event_system import EventBus, TradingEventPublisher

# Configuration
config = TradingConfig(
    exchange_name="binance",
    symbols=["BTCUSDT", "ETHUSDT"],
    max_position_size=10000.0,
    max_daily_trades=20,
    risk_per_trade=0.02,
    api_key="YOUR_API_KEY",
    api_secret="YOUR_API_SECRET"
)

# Initialize trading manager
trading_manager = TradingManager(config)
await trading_manager.initialize()
await trading_manager.start()
```

### Placing Orders

```python
from src.interfaces.base_interfaces import TradeDecision
from datetime import datetime

# Create trade decision
trade_decision = TradeDecision(
    timestamp=datetime.now(),
    symbol="BTCUSDT",
    action="BUY",
    quantity=0.001,
    price=50000.0,
    leverage=1.0,
    stop_loss=49000.0,
    take_profit=52000.0,
    confidence=0.8,
    risk_score=0.1
)

# Place order
result = await trading_manager.place_order(trade_decision)
```

### Position Management

#### Opening Perpetual Futures Positions with Leverage

```python
# Open a perpetual futures position with leverage
position_result = await trading_manager.open_position(
    symbol="BTCUSDT",             # Futures contract symbol
    side="BUY",                   # "BUY" or "SELL"
    quantity=0.001,              # Contract quantity
    leverage=5.0,                # Leverage multiplier (1x to 125x)
    order_type="MARKET",         # "MARKET" or "LIMIT"
    price=None                   # Price for limit orders
)

if position_result and position_result.get("success"):
    trade_id = position_result.get("trade_id")
    print(f"Futures position opened with trade ID: {trade_id}")
    print(f"Leverage: {position_result.get('leverage', 1)}x")
```

#### Closing Futures Positions

```python
# Close futures position using trade ID
close_result = await trading_manager.close_position("BTCUSDT", trade_id)

if close_result and close_result.get("success"):
    pnl = close_result.get("pnl", 0)
    print(f"Futures position closed. P&L: {pnl}")
    print(f"Close side: {close_result.get('close_side')}")
    print(f"Close quantity: {close_result.get('close_quantity')}")
```

#### Getting Trade Information

```python
# Get detailed trade information
trade_info = await trading_manager.get_trade_info("BTCUSDT", trade_id)

if trade_info:
    print(f"Trade details: {trade_info}")
```

#### Getting Asset Data (Klines)

```python
# Get recent market data
recent_data = await trading_manager.get_asset_data("BTCUSDT", "1m", 100)

# Get historical data with time range
from datetime import datetime, timedelta
end_time = datetime.now()
start_time = end_time - timedelta(hours=24)

historical_data = await trading_manager.get_asset_data(
    "BTCUSDT",
    "5m",
    100,
    start_time,
    end_time
)

# Access kline data
for data_point in recent_data[-5:]:  # Last 5 data points
    print(f"Time: {data_point.timestamp}, Price: {data_point.close}, Volume: {data_point.volume}")
```

### Data Streaming

```python
# Subscribe to market data
subscription_id = await data_receiver.subscribe("BTCUSDT", "1m")

# Get historical data
data = await data_receiver.get_historical_data("BTCUSDT", "1m", 100)

# Get latest price
price = await data_receiver.get_price("BTCUSDT")
```

### Event Handling

```python
# Set up event handlers
event_bus = EventBus()
await event_bus.start()

event_bus.subscribe("order_filled", your_order_handler)
event_bus.subscribe("market_data_update", your_data_handler)
event_bus.subscribe("risk_limit_exceeded", your_risk_handler)
```

## Exchange Support

### Supported Exchanges

1. **Binance** (`binance`)
   - Perpetual futures trading (USD-M and COIN-M contracts)
   - Complete futures API implementation
   - Leverage up to 125x
   - WebSocket support ready

2. **OKX** (`okx`)
   - Perpetual futures trading
   - Cross and isolated margin modes
   - Leverage up to 100x
   - Advanced order types

3. **GateIO** (`gateio`)
   - Perpetual futures trading
   - USD-M futures contracts
   - Leverage up to 100x
   - Complete implementation

4. **MEXC** (`mexc`)
   - Perpetual futures trading
   - USD-M futures contracts
   - Leverage up to 125x
   - High-frequency trading support

### Adding New Exchanges

To add a new exchange:

1. Create a new exchange class inheriting from `BaseExchange`
2. Implement all abstract methods
3. Add to `ExchangeFactory`
4. Update configuration defaults

## Configuration

### Trading Configuration

```python
config = TradingConfig(
    exchange_name="binance",        # Exchange to use
    symbols=["BTCUSDT"],           # Trading symbols
    max_position_size=10000.0,     # Maximum position value
    max_daily_trades=20,           # Daily trade limit
    risk_per_trade=0.02,           # Risk per trade (2%)
    enable_data_streaming=True,    # Enable live data
    enable_order_execution=True,   # Enable order execution
    api_key="your_key",            # API credentials
    api_secret="your_secret"
)
```

### Risk Management

- **Position Limits**: Maximum exposure per symbol and total portfolio
- **Daily Limits**: Maximum number of trades per day
- **Risk per Trade**: Maximum risk percentage per individual trade
- **Validation**: Pre-trade validation and risk scoring

## Event System

### Event Types

- `ORDER_CREATED`: New order created
- `ORDER_FILLED`: Order fully filled
- `ORDER_PARTIAL_FILLED`: Order partially filled
- `ORDER_CANCELLED`: Order cancelled
- `ORDER_REJECTED`: Order rejected
- `MARKET_DATA_UPDATE`: New market data
- `POSITION_UPDATE`: Position changed
- `RISK_LIMIT_EXCEEDED`: Risk limit exceeded
- `SYSTEM_STARTUP`: System started
- `SYSTEM_SHUTDOWN`: System stopped

### Custom Events

```python
# Publishing custom events
await event_bus.publish(
    EventType.CUSTOM_EVENT,
    {"custom_data": "value"},
    source="your_component",
    priority=2
)
```

## Error Handling

The system includes comprehensive error handling:

- **Connection Errors**: Automatic reconnection and failover
- **API Errors**: Rate limiting and retry mechanisms
- **Order Errors**: Validation and fallback strategies
- **Data Errors**: Graceful degradation and alternative data sources

## Logging

All components include structured logging:

- **INFO**: General operational messages
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors requiring attention
- **DEBUG**: Detailed debugging information

## Security

- API keys are encrypted in configuration
- Secure HTTP connections (HTTPS)
- Rate limiting to prevent API bans
- Sandbox/testnet support for all exchanges

## Performance

- **Asynchronous**: All operations are non-blocking
- **Connection Pooling**: Efficient HTTP connection management
- **Event-Driven**: Scalable event processing
- **Memory Efficient**: Configurable data retention

## Testing

The system is designed for easy testing:

- Mock exchanges for testing
- Event replay capabilities
- Configuration validation
- Comprehensive error simulation

## Production Deployment

For production use:

1. Set up proper API credentials
2. Configure risk parameters appropriately
3. Monitor system logs
4. Set up alerting for critical events
5. Regular backtesting and validation

## Support and Maintenance

The system is designed for maintainability:

- **Modular Design**: Easy to extend and modify
- **Documentation**: Comprehensive inline documentation
- **Error Tracking**: Detailed error reporting
- **Configuration**: Flexible configuration system

## Next Steps

1. **Configure your API keys** in the configuration files
2. **Set appropriate risk parameters** for your trading strategy
3. **Test with small amounts** before live trading
4. **Monitor system performance** and adjust as needed
5. **Extend functionality** based on your specific requirements

This live trading system provides a solid foundation for automated trading with multiple exchanges, comprehensive risk management, and real-time data processing.