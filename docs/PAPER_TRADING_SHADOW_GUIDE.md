# Paper Trading with Shadow Trading Guide

## Overview

The Ares trading bot now supports **paper trading with shadow trading**, which allows you to test your trading strategies with actual API calls to Binance's testnet while maintaining simulated trading logic. This provides a more realistic testing environment compared to pure simulation.

## How It Works

### Traditional Paper Trading vs Shadow Trading

**Traditional Paper Trading:**
- Uses simulated trading logic only
- No actual API calls to exchanges
- Completely isolated from real market conditions
- Limited testing of API interactions

**Shadow Trading (Enhanced Paper Trading):**
- Uses actual API calls to Binance's testnet
- Maintains simulated trading logic for safety
- Provides realistic market data and order execution
- Tests actual API interactions and rate limits
- No real money involved (testnet only)

### Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Ares Bot      │───▶│  Binance Testnet │───▶│  Real Market    │
│  (Paper Mode)   │    │     APIs         │    │   (Testnet)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Simulated      │    │  Real API Calls  │    │  Real Market    │
│  Trading Logic  │    │  (Orders, Data)  │    │  Data & Orders  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Setup Instructions

### 1. Environment Variables

Create or update your `.env` file with the following variables:

```bash
# Trading Environment
TRADING_ENVIRONMENT=PAPER

# Binance Testnet API Keys (Required for shadow trading)
BINANCE_TESTNET_API_KEY=your_testnet_api_key_here
BINANCE_TESTNET_API_SECRET=your_testnet_api_secret_here

# Optional: Other configuration
INITIAL_EQUITY=10000
TRADE_SYMBOL=ETHUSDT
EXCHANGE_NAME=BINANCE
TIMEFRAME=15m
```

### 2. Getting Binance Testnet API Keys

1. Visit [Binance Testnet](https://testnet.binance.vision/)
2. Create a testnet account
3. Generate API keys for the testnet
4. Add the keys to your `.env` file

### 3. Testing the Setup

Run the test script to verify your configuration:

```bash
python scripts/test_paper_trading_shadow.py
```

This will test:
- ✅ Testnet API key configuration
- ✅ Binance testnet connection
- ✅ Basic API calls
- ✅ Account information retrieval
- ✅ Market data access

## Usage

### Basic Paper Trading with Shadow Trading

```bash
python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE
```

### With GUI

```bash
python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE --gui
```

### Multiple Symbols

```bash
# Run for all supported symbols
python scripts/paper_trader_launcher.py

# Run for specific symbol
python scripts/paper_trader_launcher.py ETHUSDT BINANCE
```

## Features

### What Works with Shadow Trading

✅ **Real Market Data**
- Live price feeds from Binance testnet
- Real-time order book data
- Actual market conditions

✅ **API Interaction Testing**
- Real API calls to Binance testnet
- Rate limit testing
- Network latency simulation
- Error handling validation

✅ **Order Execution Simulation**
- Real order placement (testnet)
- Order status tracking
- Execution simulation

✅ **Account Management**
- Real account balance queries
- Position tracking
- Risk management testing

### What Remains Simulated

🔄 **Trading Logic**
- Entry/exit decisions
- Position sizing
- Risk calculations
- Performance tracking

🔄 **Portfolio Management**
- Simulated equity tracking
- Performance metrics
- Risk management rules

🔄 **Safety Features**
- No real money involved
- Simulated profit/loss
- Safe testing environment

## Configuration Options

### Exchange Configuration

The system automatically configures the Binance testnet client with optimal settings:

```python
exchange_config = {
    "binance_exchange": {
        "use_testnet": True,  # Use testnet APIs
        "api_key": settings.binance_testnet_api_key,
        "api_secret": settings.binance_testnet_api_secret,
        "timeout": 30,
        "max_retries": 3,
        "rate_limit_enabled": True,
        "rate_limit_requests": 1200,
        "rate_limit_window": 60,
    }
}
```

### Trading Environment

The system automatically detects the trading environment:

- `PAPER` with testnet API keys → Shadow trading enabled
- `PAPER` without testnet API keys → Fallback to simulation
- `LIVE` → Real trading (requires live API keys)
- `TESTNET` → Direct testnet trading

## Monitoring and Logs

### Log Files

Check the following log files for detailed information:

- `logs/ares_paper_trading.log` - Paper trading specific logs
- `logs/system.log` - General system logs
- `logs/error.log` - Error logs

### Key Log Messages

```
🚀 Initializing Ares Trading Bot in PAPER mode (shadow trading) for ETHUSDT on BINANCE...
🔗 Initializing Binance testnet connection for shadow trading...
✅ Binance testnet connection established for shadow trading
✅ Supervisor initialized with testnet exchange client
🔄 Starting supervisor for ETHUSDT with shadow trading...
```

## Troubleshooting

### Common Issues

1. **"Testnet API keys not found"**
   - Solution: Add `BINANCE_TESTNET_API_KEY` and `BINANCE_TESTNET_API_SECRET` to your `.env` file

2. **"Failed to initialize Binance testnet connection"**
   - Solution: Check your internet connection and API key validity
   - Verify your testnet API keys are correct

3. **"Rate limit exceeded"**
   - Solution: The system automatically handles rate limits, but you can adjust settings if needed

4. **"Account info not accessible"**
   - Solution: Ensure your testnet API keys have the necessary permissions

### Fallback Behavior

If testnet API keys are not available, the system automatically falls back to pure simulation:

```
⚠️  Warning: PAPER mode with shadow trading requires BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET. 
Falling back to simulated trading without API calls.
```

## Benefits

### For Development
- Test API interactions without risk
- Validate rate limiting and error handling
- Debug network issues
- Test order execution logic

### For Strategy Testing
- Real market conditions
- Actual price feeds
- Realistic execution delays
- Network latency simulation

### For Production Preparation
- Validate production-like environment
- Test monitoring and alerting
- Verify error handling
- Ensure system stability

## Security Notes

- **No Real Money**: All operations use Binance testnet
- **API Keys**: Testnet API keys have no real value
- **Isolation**: Paper trading logic is completely separate from live trading
- **Safety**: Multiple layers of protection prevent real trading

## Next Steps

1. **Test Your Setup**: Run the test script to verify configuration
2. **Start Paper Trading**: Use the launcher to begin shadow trading
3. **Monitor Performance**: Check logs and performance metrics
4. **Refine Strategy**: Adjust parameters based on testnet results
5. **Prepare for Live**: When ready, switch to live trading mode

## Support

For issues or questions:
- Check the logs for detailed error messages
- Run the test script to diagnose configuration issues
- Review this documentation for setup instructions
- Ensure your testnet API keys are valid and have proper permissions 