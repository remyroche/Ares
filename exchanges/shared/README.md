# Shared Exchange Utilities

This directory contains reusable utilities for exchange integrations, separating common functionality from exchange-specific implementations.

## Architecture

The shared utilities are organized into the following modules:

### 🔐 Authentication & Account Management (`auth/`)
- **APIKeyManager**: Manages API keys, permissions, IP allowlists, and key rotation
- **TimeSyncManager**: Handles clock skew detection and correction
- **SubaccountManager**: Manages subaccount operations and permissions
- **AuthenticationManager**: Unified authentication management

### 📊 Market Data & Metadata (`market/`)
- **MarketMetadataManager**: Manages market data, instrument specifications, and metadata caching
- **InstrumentManager**: Handles instrument specifications and contract details
- **PrecisionHelper**: Manages price and quantity precision, rounding, and validation
- **RiskTierManager**: Manages risk tiers, leverage limits, and position size restrictions

### 💰 Pricing & Market Data (`pricing/`)
- **PriceManager**: Handles price fetching, ticker data, and price validation
- **OHLCVManager**: Manages OHLCV data fetching, caching, and processing
- **MarketDataAggregator**: Aggregates market data from multiple sources

### 📋 Order Management (`orders/`)
- **OrderManager**: Manages order lifecycle, tracking, and execution
- **IdempotencyManager**: Handles idempotency keys for order operations
- **PositionManager**: Manages position tracking and calculations

### ⚠️ Risk Management (`risk/`)
- **RiskCalculator**: Calculates position risk, margin requirements, and risk metrics
- **LiquidationRiskManager**: Manages liquidation risk calculations and monitoring
- **MarginManager**: Handles margin calculations and requirements

### 📜 History & Monitoring (`history/`)
- **TradeHistoryManager**: Manages trade history fetching and pagination
- **PaginationManager**: Handles pagination for large datasets

### 💳 Wallet & Balances (`wallet/`)
- **BalanceManager**: Manages balance tracking, equity calculations, and validation
- **WalletManager**: Handles wallet operations and account type management

### 🔧 Reliability & Operations (`reliability/`)
- **RateLimitManager**: Manages rate limiting, backoff strategies, and request throttling
- **RetryManager**: Handles retry logic with exponential backoff
- **AuditLogger**: Provides comprehensive audit logging for all operations
- **SystemStatusManager**: Monitors system status and maintenance windows

## Usage Examples

### Basic Setup

```python
from exchanges.shared import (
    AuthenticationManager, MarketMetadataManager, PriceManager,
    OrderManager, RiskCalculator, BalanceManager
)

# Initialize managers
auth_manager = AuthenticationManager("okx")
market_metadata = MarketMetadataManager("okx")
price_manager = PriceManager("okx")
order_manager = OrderManager("okx")
risk_calculator = RiskCalculator("okx")
balance_manager = BalanceManager("okx")
```

### Authentication

```python
from exchanges.shared.auth import AuthConfig, APIKeyPermission

# Configure authentication
auth_config = AuthConfig(
    exchange_name="okx",
    api_key="your_api_key",
    api_secret="your_api_secret",
    passphrase="your_passphrase",
    permissions={APIKeyPermission.READ, APIKeyPermission.TRADE},
    auto_sync_time=True
)

# Authenticate
success = await auth_manager.authenticate(auth_config)
```

### Market Data

```python
# Get current price
price_data = await price_manager.get_price("BTCUSDT")
print(f"Current price: ${price_data.price}")

# Get OHLCV data
ohlcv_data = await ohlcv_manager.get_ohlcv("BTCUSDT", Timeframe.HOUR_1, 24)
print(f"Retrieved {len(ohlcv_data)} hourly candles")

# Get instrument information
instrument = market_metadata.get_instrument("BTCUSDT")
print(f"Tick size: {instrument.tick_size}")
print(f"Lot size: {instrument.lot_size}")
```

### Order Management

```python
from exchanges.shared.orders import OrderSide, OrderType

# Create order
order = order_manager.create_order(
    symbol="BTCUSDT",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=0.001
)

# Submit order
success = await order_manager.submit_order(order)
```

### Risk Management

```python
# Calculate position risk
position_risk = risk_calculator.calculate_position_risk(
    symbol="BTCUSDT",
    position_size=0.1,
    entry_price=50000.0,
    current_price=51000.0,
    leverage=2.0
)

print(f"Margin ratio: {position_risk.margin_ratio:.2%}")
print(f"Liquidation price: ${position_risk.liquidation_price:,.2f}")
print(f"Risk level: {position_risk.risk_level.value}")
```

### Precision Handling

```python
# Round prices and quantities
rounded_price = precision_helper.round_price(50000.123456789, "BTCUSDT")
rounded_qty = precision_helper.round_quantity(0.00123456789, "BTCUSDT")

# Validate order parameters
is_valid, errors = precision_helper.validate_order(
    symbol="BTCUSDT",
    side="buy",
    order_type="limit",
    price=50000.0,
    quantity=0.001
)
```

### Rate Limiting

```python
# Execute function with rate limiting
result = await rate_limit_manager.execute_with_rate_limit(
    "trading", 
    your_api_function, 
    "BTCUSDT", 
    "buy", 
    0.001
)
```

## Key Features

### 🔄 Idempotency
All order operations support idempotency keys to prevent duplicate submissions:

```python
# Generate idempotency key
key = idempotency_manager.create_order_key(
    symbol="BTCUSDT",
    side="buy",
    order_type="market",
    quantity=0.001
)

# Check for duplicate operations
is_duplicate = idempotency_manager.is_operation_duplicate(
    operation_type="create_order",
    parameters={"symbol": "BTCUSDT", "side": "buy", "quantity": 0.001}
)
```

### ⏱️ Time Synchronization
Automatic time synchronization with exchange servers:

```python
# Time sync is handled automatically by AuthenticationManager
# Manual sync if needed
await time_sync_manager.sync_time(get_server_time_function)

# Get adjusted timestamp for requests
timestamp = time_sync_manager.get_adjusted_timestamp()
```

### 📊 Risk Monitoring
Comprehensive risk monitoring and validation:

```python
# Validate position risk
is_safe, warnings = risk_calculator.validate_position_risk(position_risk)

# Calculate portfolio risk
portfolio_risk = risk_calculator.calculate_portfolio_risk(positions, total_equity)

# Get risk summary
summary = risk_calculator.get_risk_summary(portfolio_risk)
```

### 🎯 Precision Management
Automatic precision handling for prices and quantities:

```python
# Set precision configuration
config = PrecisionConfig(
    symbol="BTCUSDT",
    price_precision=2,
    quantity_precision=6,
    tick_size=0.01,
    lot_size=0.00001,
    min_notional=10.0
)
precision_helper.set_precision_config(config)

# Automatic rounding and validation
rounded_price = precision_helper.round_price(price, symbol)
is_valid = precision_helper.validate_price(price, symbol)
```


### 🔍 Audit Logging
Comprehensive audit logging for all operations:

```python
# Audit logging is handled automatically
# Manual logging if needed
audit_logger.log_operation(
    operation="create_order",
    parameters={"symbol": "BTCUSDT", "side": "buy"},
    result={"order_id": "12345"},
    success=True
)
```

## Configuration

### Rate Limiting
Configure rate limits for different endpoints:

```python
from exchanges.shared.reliability import RateLimit, RateLimitStrategy

# Set rate limits
general_limit = RateLimit(
    requests_per_second=20,
    requests_per_minute=1200,
    requests_per_hour=72000,
    burst_limit=50
)

rate_limit_manager.set_rate_limit("public", general_limit)
```

### Risk Thresholds
Configure risk monitoring thresholds:

```python
# Set risk thresholds
risk_calculator.set_risk_thresholds(
    warning_ratio=0.8,      # 80% margin ratio warning
    critical_ratio=0.9,     # 90% margin ratio critical
    liquidation_ratio=0.95  # 95% margin ratio liquidation
)
```

### Cache Settings
Configure caching for different data types:

```python
# Set cache TTL
price_manager.set_cache_ttl(30)  # 30 seconds
ohlcv_manager.set_cache_ttl(300)  # 5 minutes
balance_manager.set_cache_ttl(60)  # 1 minute
```

## Error Handling

All utilities include comprehensive error handling and logging:

```python
try:
    result = await price_manager.get_price("BTCUSDT")
    if result:
        print(f"Price: ${result.price}")
    else:
        print("Price not available")
except Exception as e:
    logger.error(f"Error getting price: {e}")
```

## Thread Safety

All utilities are designed to be thread-safe and can be used in concurrent environments:

```python
# Safe to use in multiple coroutines
async def fetch_prices(symbols):
    tasks = [price_manager.get_price(symbol) for symbol in symbols]
    return await asyncio.gather(*tasks)
```

## Performance Optimization

- **Caching**: Intelligent caching with TTL for frequently accessed data
- **Rate Limiting**: Automatic rate limiting to prevent API throttling
- **Batch Operations**: Support for batch operations where possible
- **Connection Pooling**: Efficient HTTP connection management
- **Background Tasks**: Automatic background synchronization and cleanup

## Monitoring & Statistics

Get comprehensive statistics for monitoring:

```python
# Get various statistics
price_stats = price_manager.get_price_statistics()
order_stats = order_manager.get_order_statistics()
balance_stats = balance_manager.get_balance_statistics()
rate_limit_stats = rate_limit_manager.get_rate_limit_statistics()
```

## Best Practices

1. **Always use shared utilities** instead of implementing exchange-specific logic
2. **Configure rate limits** appropriately for your use case
3. **Monitor risk levels** regularly and set appropriate thresholds
4. **Use idempotency keys** for all order operations
5. **Handle errors gracefully** and implement proper retry logic
6. **Monitor performance** using the built-in statistics
7. **Clean up resources** properly when closing connections

## Contributing

When adding new shared utilities:

1. Follow the existing module structure
2. Include comprehensive error handling
3. Add proper logging
4. Include unit tests
5. Update this README with usage examples
6. Ensure thread safety
7. Add performance monitoring capabilities