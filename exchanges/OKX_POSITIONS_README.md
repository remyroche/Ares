# OKX Position Management Suite

A comprehensive position fetching and management suite for the OKX exchange, providing real-time position data, risk analysis, and portfolio management capabilities.

## 🚀 Features

### Core Position Methods
- **`get_all_positions(inst_type)`** - Fetch all positions across different instrument types
- **`get_position_by_symbol(symbol, inst_type)`** - Get specific position details for a symbol
- **`get_position_history(symbol, inst_type, after, before, limit)`** - Retrieve historical position changes
- **`get_position_margin(symbol)`** - Get position margin information
- **`get_position_funding(symbol)`** - Fetch funding fee information

### Risk Management
- **`get_position_risk_metrics(symbol)`** - Comprehensive risk analysis and metrics
- **`get_position_alerts(risk_threshold)`** - Automated risk alerts and notifications
- **`calculate_position_size(symbol, risk_amount, entry_price, stop_loss_price, leverage)`** - Position sizing calculator

### Portfolio Analytics
- **`get_position_summary(inst_type)`** - Portfolio overview and key metrics
- **`get_positions_stream(callback, inst_type)`** - Real-time position streaming
- **`get_position_risk_stream(symbol, callback)`** - Real-time risk monitoring

## 📋 Supported Instrument Types

- **SPOT** - Spot trading positions
- **MARGIN** - Margin trading positions  
- **SWAP** - Perpetual swap positions
- **FUTURES** - Futures contract positions
- **OPTION** - Options contract positions

## 🔧 Installation & Setup

```python
from exchanges.okx import OkxExchange

# Initialize exchange with your credentials
exchange = OkxExchange(
    api_key="your_api_key",
    api_secret="your_api_secret", 
    trade_symbol="BTCUSDT",
    password="your_passphrase"  # Optional
)

# Initialize the connection
await exchange._initialize_exchange()
```

## 📊 Usage Examples

### Basic Position Fetching

```python
# Get all spot positions
positions = await exchange.get_all_positions("SPOT")
print(f"Found {len(positions)} positions")

# Get specific symbol position
btc_position = await exchange.get_position_by_symbol("BTCUSDT", "SPOT")
if btc_position:
    print(f"BTC Position: {btc_position['size']} @ {btc_position['markPrice']}")
```

### Risk Analysis

```python
# Get comprehensive risk metrics
risk_metrics = await exchange.get_position_risk_metrics()
print(f"Portfolio Risk Score: {risk_metrics['portfolioRiskScore']:.2%}")
print(f"Total Unrealized PnL: ${risk_metrics['totalUnrealizedPnl']:.2f}")

# Check for alerts
alerts = await exchange.get_position_alerts(risk_threshold=0.05)
for alert in alerts:
    print(f"Alert: {alert['message']}")
```

### Position Sizing

```python
# Calculate optimal position size
position_calc = await exchange.calculate_position_size(
    symbol="BTCUSDT",
    risk_amount=1000,  # Risk $1000
    entry_price=50000,  # Entry at $50k
    stop_loss_price=48000,  # Stop loss at $48k
    leverage=2.0  # 2x leverage
)

print(f"Position Size: {position_calc['position_size']:.6f} BTC")
print(f"Margin Required: ${position_calc['margin_required']:.2f}")
```

### Real-time Monitoring

```python
# Position streaming
async def position_callback(positions):
    total_pnl = sum(float(p.get('unrealizedPnl', 0)) for p in positions)
    print(f"Total PnL: ${total_pnl:.2f}")

# Start streaming
await exchange.get_positions_stream(position_callback, "SPOT")

# Risk monitoring
async def risk_callback(risk_data):
    print(f"Portfolio Risk: {risk_data['portfolioRiskScore']:.2%}")

await exchange.get_position_risk_stream("BTCUSDT", risk_callback)
```

## 📈 Position Data Structure

Each position contains comprehensive information:

```python
{
    "symbol": "BTCUSDT",
    "instType": "SPOT",
    "size": "1.5",
    "side": "long",
    "markPrice": "50000",
    "avgPrice": "49000",
    "unrealizedPnl": "1500",
    "unrealizedPnlRatio": "0.03",
    "liquidationPrice": "45000",
    "margin": "1000",
    "notionalUsd": "75000",
    "marginRatio": "0.02",
    "marginMode": "isolated",
    "interest": "0",
    "lastUpdateTime": "1640995200000",
    "openTime": "1640995200000",
    "leverage": "1",
    "delta": "1",
    "gamma": "0",
    "theta": "0",
    "vega": "0"
}
```

## ⚠️ Risk Metrics

The risk analysis provides:

- **Portfolio Risk Score** - Overall portfolio risk percentage
- **Total Unrealized PnL** - Sum of all unrealized profits/losses
- **Total Notional Value** - Total position value in USD
- **Max Leverage** - Highest leverage across all positions
- **High Risk Positions** - Positions exceeding risk thresholds
- **Margin Utilization** - Percentage of margin used

## 🚨 Alert System

Automated alerts for:

- **HIGH_RISK** - Positions with high risk scores
- **HIGH_LEVERAGE** - Positions with excessive leverage (>10x)
- **PORTFOLIO_RISK** - Overall portfolio risk exceeding threshold

## 🔄 Streaming Capabilities

Real-time updates via polling (WebSocket implementation can be added):

- **Position Updates** - Live position changes
- **Risk Monitoring** - Continuous risk assessment
- **Margin Alerts** - Real-time margin warnings

## 🧪 Testing

Run the comprehensive test suite:

```bash
python examples/test_okx_positions.py
```

Or run the example demo:

```bash
python examples/okx_positions_example.py
```

## 📝 Error Handling

All methods include comprehensive error handling:

- Input validation
- API error handling
- Network timeout handling
- Data parsing error recovery
- Graceful degradation

## 🔐 Security

- API credentials are handled securely
- All requests are signed with HMAC-SHA256
- Rate limiting compliance
- SSL/TLS encryption

## 📊 Performance

- Efficient data parsing
- Minimal API calls
- Caching where appropriate
- Async/await for non-blocking operations

## 🛠️ Configuration

Environment variables for easy setup:

```bash
export OKX_API_KEY="your_api_key"
export OKX_API_SECRET="your_api_secret"
export OKX_PASSWORD="your_passphrase"
```

## 📚 API Reference

### Core Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `get_all_positions(inst_type)` | Get all positions | `inst_type: str` | `List[Dict]` |
| `get_position_by_symbol(symbol, inst_type)` | Get specific position | `symbol: str, inst_type: str` | `Dict` |
| `get_position_history(...)` | Get position history | `symbol, inst_type, after, before, limit` | `List[Dict]` |
| `get_position_margin(symbol)` | Get margin info | `symbol: str` | `Dict` |
| `get_position_funding(symbol)` | Get funding info | `symbol: str` | `List[Dict]` |

### Risk & Analytics

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `get_position_risk_metrics(symbol)` | Risk analysis | `symbol: str` | `Dict` |
| `get_position_alerts(risk_threshold)` | Get alerts | `risk_threshold: float` | `List[Dict]` |
| `get_position_summary(inst_type)` | Portfolio summary | `inst_type: str` | `Dict` |
| `calculate_position_size(...)` | Position sizing | `symbol, risk_amount, entry_price, stop_loss_price, leverage` | `Dict` |

### Streaming

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `get_positions_stream(callback, inst_type)` | Position streaming | `callback: Callable, inst_type: str` | `None` |
| `get_position_risk_stream(symbol, callback)` | Risk streaming | `symbol: str, callback: Callable` | `None` |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is part of the larger trading system and follows the same licensing terms.

## 🆘 Support

For issues and questions:

1. Check the test files for usage examples
2. Review the error logs for debugging
3. Ensure API credentials are correct
4. Verify network connectivity

## 🔄 Changelog

### v1.0.0
- Initial implementation of OKX position suite
- Core position fetching methods
- Risk analysis and metrics
- Real-time streaming capabilities
- Comprehensive error handling
- Full test coverage