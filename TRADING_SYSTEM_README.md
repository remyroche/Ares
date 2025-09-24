# Live Trading System

A comprehensive, production-ready trading system with exchange-agnostic architecture, real-time data streaming, risk management, and order execution capabilities.

## 🏗️ Architecture Overview

The trading system is built with a modular, exchange-agnostic architecture:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Live Trading  │    │  Exchange-Agnostic │    │    Exchanges    │
│     Module      │◄──►│     Receiver      │◄──►│   (CEX APIs)    │
│                 │    │                  │    │                 │
│ • Order Manager │    │ • TradingReceiver│    │ • Binance       │
│ • Data Streamer │    │ • OrderRouter    │    │ • OKX           │
│ • Risk Manager  │    │ • DataAggregator │    │ • GateIO        │
│ • Trading Engine│    │ • ExchangeRegistry│    │ • MEXC          │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 Directory Structure

```
live_trading/                 # Main trading module
├── __init__.py              # Module exports
├── config.py                # Trading configuration
├── order_manager.py         # Order lifecycle management
├── data_streamer.py         # Real-time data streaming
├── risk_manager.py          # Risk management system
└── trading_engine.py        # Main trading coordinator

exchanges/                   # Exchange-agnostic receiver
├── __init__.py              # Module exports
├── trading_receiver.py      # Main receiver interface
├── order_router.py          # Order routing logic
├── data_aggregator.py       # Multi-exchange data aggregation
└── exchange_registry.py     # Exchange instance management

exchange/                    # Exchange implementations
├── base_exchange.py         # Base exchange interface
├── factory.py              # Exchange factory
├── binance.py              # Binance implementation
├── okx.py                  # OKX implementation
├── gateio.py               # GateIO implementation
└── mexc.py                 # MEXC implementation
```

## 🚀 Quick Start

### 1. Basic Setup

```python
from live_trading import TradingEngine, TradingConfig, TradingMode
from exchange.factory import ExchangeFactory

# Configure trading parameters
config = TradingConfig(
    mode=TradingMode.PAPER,  # Start with paper trading
    exchange_name="binance",
    symbols=["BTCUSDT", "ETHUSDT"],
    max_position_size=1000.0,
    max_daily_loss=100.0,
    max_leverage=5.0
)

# Create exchange client
exchange_client = ExchangeFactory.get_exchange(config.exchange_name)

# Create and start trading engine
trading_engine = TradingEngine(config, exchange_client)
await trading_engine.start()
```

### 2. Execute Trades

```python
from src.interfaces.base_interfaces import TradeDecision
from datetime import datetime

# Create trade decision
decision = TradeDecision(
    timestamp=datetime.now(),
    symbol="BTCUSDT",
    action="buy",
    quantity=0.001,
    price=0.0,  # Market order
    leverage=1.0,
    stop_loss=45000.0,
    take_profit=55000.0,
    confidence=0.8,
    risk_score=0.3
)

# Execute trade
order = await trading_engine.execute_trade_decision(decision)
print(f"Order executed: {order.id}")
```

### 3. Use Exchange-Agnostic Receiver

```python
from exchanges import TradingReceiver

# Configure receiver with multiple exchanges
receiver_config = {
    "exchanges": {
        "binance": {
            "api_key": "your_binance_key",
            "api_secret": "your_binance_secret"
        },
        "okx": {
            "api_key": "your_okx_key",
            "api_secret": "your_okx_secret",
            "password": "your_okx_passphrase"
        }
    }
}

# Create receiver
receiver = TradingReceiver(receiver_config)
await receiver.start()

# Send orders to any exchange
response = await receiver.send_order(
    exchange="binance",
    symbol="BTCUSDT",
    side="buy",
    order_type="market",
    quantity=0.001
)
```

## 🔧 Configuration

### Trading Configuration

```python
from live_trading.config import TradingConfig, TradingMode

config = TradingConfig(
    # Trading Mode
    mode=TradingMode.PAPER,  # PAPER, LIVE, BACKTEST
    
    # Exchange Configuration
    exchange_name="binance",
    symbols=["BTCUSDT", "ETHUSDT"],
    
    # Risk Management
    max_position_size=1000.0,
    max_daily_loss=100.0,
    max_leverage=10.0,
    stop_loss_percentage=2.0,
    take_profit_percentage=4.0,
    
    # Order Management
    order_timeout=30,
    max_retries=3,
    retry_delay=1.0,
    
    # Data Streaming
    data_update_interval=1.0,
    reconnect_attempts=5,
    reconnect_delay=5.0,
    
    # Performance Monitoring
    performance_log_interval=60,
    trade_log_enabled=True,
    metrics_enabled=True
)
```

## 📊 Features

### 1. Order Management
- **Order Lifecycle**: Complete order tracking from creation to execution
- **Status Monitoring**: Real-time order status updates
- **Error Handling**: Comprehensive error handling and retry logic
- **Order History**: Complete order history and performance tracking

### 2. Real-Time Data Streaming
- **Multi-Symbol Support**: Stream data for multiple trading pairs
- **Data Types**: Ticker, trades, orderbook, and kline data
- **Automatic Reconnection**: Built-in reconnection logic for reliability
- **Data Normalization**: Standardized data format across exchanges

### 3. Risk Management
- **Position Limits**: Maximum position size enforcement
- **Daily Loss Limits**: Daily loss protection
- **Leverage Controls**: Maximum leverage enforcement
- **Risk Scoring**: Dynamic risk assessment for trades
- **Real-Time Monitoring**: Continuous risk monitoring and alerts

### 4. Exchange Support
- **Binance**: Full spot and futures support
- **OKX**: Complete API implementation
- **GateIO**: Spot trading support
- **MEXC**: Spot and futures support
- **Extensible**: Easy to add new exchanges

### 5. Exchange-Agnostic Interface
- **Unified API**: Single interface for all exchanges
- **Order Routing**: Automatic order routing to appropriate exchanges
- **Data Aggregation**: Multi-exchange data aggregation
- **Load Balancing**: Distribute orders across multiple exchanges

## 🛡️ Risk Management

The system includes comprehensive risk management:

```python
# Risk limits are enforced automatically
risk_limits = RiskLimits(
    max_position_size=1000.0,
    max_daily_loss=100.0,
    max_leverage=10.0,
    max_drawdown_percent=10.0,
    max_volatility=0.05,
    min_sharpe_ratio=1.0,
    max_orders_per_minute=60
)

# Validate trades before execution
is_valid, message = await risk_manager.validate_trade_decision(decision)
if not is_valid:
    print(f"Trade rejected: {message}")
```

## 📈 Performance Monitoring

Built-in performance tracking:

```python
# Get performance metrics
performance = await trading_engine.get_performance_metrics()
print(f"Win rate: {performance['win_rate']:.2%}")
print(f"Total PnL: ${performance['total_pnl']:.2f}")
print(f"Sharpe ratio: {performance['sharpe_ratio']:.2f}")

# Get position summary
positions = await trading_engine.get_position_summary()
for symbol, position in positions.items():
    print(f"{symbol}: {position['current_position']} @ ${position['current_price']}")
```

## 🔄 Data Streaming

Real-time data streaming with automatic reconnection:

```python
# Register data handlers
async def on_ticker_data(data):
    print(f"Ticker: {data['symbol']} @ ${data['data']['last_price']}")

async def on_trade_data(data):
    print(f"Trade: {data['symbol']} - {data['data']['quantity']} @ ${data['data']['price']}")

# Register handlers
trading_engine.data_streamer.register_handler("ticker", on_ticker_data)
trading_engine.data_streamer.register_handler("trade", on_trade_data)
```

## 🌐 Multi-Exchange Support

Trade across multiple exchanges:

```python
# Send orders to different exchanges
await receiver.send_order("binance", "BTCUSDT", "buy", "market", 0.001)
await receiver.send_order("okx", "BTCUSDT", "buy", "market", 0.001)

# Get aggregated data
aggregated_data = await receiver.get_aggregated_data(
    "BTCUSDT", 
    "ticker", 
    exchanges=["binance", "okx"]
)
```

## 📝 Logging and Monitoring

Comprehensive logging and monitoring:

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# System automatically logs:
# - Order executions
# - Risk violations
# - Data streaming events
# - Performance metrics
# - Error conditions
```

## 🚨 Error Handling

Robust error handling throughout:

```python
try:
    order = await trading_engine.execute_trade_decision(decision)
    if order:
        print(f"Order executed: {order.id}")
    else:
        print("Order rejected or failed")
except Exception as e:
    logger.error(f"Trading error: {e}")
    # System continues running with error logged
```

## 🔒 Security

Security best practices:

- **API Key Management**: Secure API key storage and handling
- **Signature Generation**: Proper HMAC signature generation for exchanges
- **SSL/TLS**: All connections use secure protocols
- **Rate Limiting**: Built-in rate limiting to prevent API abuse
- **Input Validation**: Comprehensive input validation and sanitization

## 📚 Examples

See `examples/live_trading_example.py` for comprehensive examples including:

- Basic trading setup
- Order execution
- Data streaming
- Risk management
- Multi-exchange trading
- Performance monitoring

## 🛠️ Development

### Adding New Exchanges

1. Create new exchange class inheriting from `BaseExchange`
2. Implement all abstract methods
3. Add to `ExchangeFactory`
4. Test with paper trading

### Custom Risk Rules

1. Extend `RiskManager` class
2. Override `validate_trade_decision` method
3. Add custom risk metrics
4. Register with trading engine

### Custom Data Handlers

1. Create async handler functions
2. Register with `DataStreamer`
3. Handle different data types (ticker, trades, etc.)
4. Implement error handling

## 📄 License

This trading system is part of the larger Ares trading platform and follows the same licensing terms.

## 🤝 Contributing

Contributions are welcome! Please ensure:

1. All tests pass
2. Code follows existing patterns
3. Documentation is updated
4. Risk management is considered
5. Paper trading is tested first

## ⚠️ Disclaimer

This is a trading system for educational and development purposes. Always:

- Test thoroughly with paper trading
- Understand the risks involved
- Start with small position sizes
- Monitor your trades closely
- Never trade with money you can't afford to lose

The developers are not responsible for any financial losses incurred through the use of this system.