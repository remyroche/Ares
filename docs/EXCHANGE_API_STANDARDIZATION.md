# Exchange API Standardization

## Overview

This document describes the standardization of exchange APIs in the Ares trading bot. All exchanges now implement a common interface that ensures consistent method signatures, return types, and error handling across different exchange providers.

## Key Changes

### 1. Base Exchange Class

A new `BaseExchange` class has been created that implements the `IExchangeClient` interface and provides:

- **Standardized method signatures**: All exchanges now have the same parameter order and types
- **Common error handling**: Consistent error handling patterns across all exchanges
- **Data format standardization**: Raw exchange data is converted to standardized `MarketData` objects
- **Abstract method pattern**: Subclasses implement exchange-specific logic while maintaining consistent interfaces

### 2. Standardized Methods

All exchanges now implement these core methods with consistent signatures:

#### Core Interface Methods (from IExchangeClient)

```python
async def get_klines(symbol: str, interval: str, limit: int = 100) -> list[MarketData]
async def get_account_info() -> dict[str, Any]
async def create_order(symbol: str, side: str, quantity: float, price: Optional[float] = None, order_type: str = "MARKET") -> dict[str, Any]
async def get_position_risk(symbol: str) -> dict[str, Any]
```

#### Additional Standardized Methods

```python
async def get_historical_klines(symbol: str, interval: str, start_time_ms: int, end_time_ms: int, limit: int = 1000) -> list[MarketData]
async def get_historical_agg_trades(symbol: str, start_time_ms: int, end_time_ms: int, limit: int = 1000) -> list[dict[str, Any]]
async def get_open_orders(symbol: Optional[str] = None) -> list[dict[str, Any]]
async def cancel_order(symbol: str, order_id: Any) -> dict[str, Any]
async def get_order_status(symbol: str, order_id: Any) -> dict[str, Any]
async def close() -> None
```

### 3. Data Standardization

#### MarketData Structure

All kline data is now returned in a standardized `MarketData` format:

```python
@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str
```

#### Timestamp Conversion

The base class provides a `_convert_timestamp()` method that handles various timestamp formats:
- Unix timestamps (seconds or milliseconds)
- ISO format strings
- Other common datetime formats

## Exchange Implementations

### Binance Exchange

- **Inherits from**: `BaseExchange`
- **Special features**: Custom rate limiting, WebSocket support, direct API calls
- **Data format**: Uses Binance's native kline format `[open_time, open, high, low, close, volume, ...]`

### GateIO Exchange

- **Inherits from**: `BaseExchange`
- **Special features**: CCXT-based implementation
- **Data format**: Uses CCXT's standard format `[timestamp, open, high, low, close, volume, ...]`

### MEXC Exchange

- **Inherits from**: `BaseExchange`
- **Special features**: CCXT-based implementation
- **Data format**: Uses CCXT's standard format `[timestamp, open, high, low, close, volume, ...]`

### OKX Exchange

- **Inherits from**: `BaseExchange`
- **Special features**: CCXT-based implementation, requires password parameter
- **Data format**: Uses CCXT's standard format `[timestamp, open, high, low, close, volume, ...]`

## Usage Examples

### Basic Usage

```python
from exchange.factory import ExchangeFactory

# Create exchange instance
exchange = ExchangeFactory.get_exchange("binance")

# Get klines in standardized format
klines = await exchange.get_klines("BTCUSDT", "1h", 100)
for kline in klines:
    print(f"Time: {kline.timestamp}, Close: {kline.close}")

# Create order
order = await exchange.create_order(
    symbol="BTCUSDT",
    side="BUY",
    quantity=0.001,
    order_type="MARKET"
)

# Get account info
account = await exchange.get_account_info()
```

### Historical Data

```python
# Get historical klines
start_time = int(datetime(2024, 1, 1).timestamp() * 1000)
end_time = int(datetime(2024, 1, 31).timestamp() * 1000)

historical_klines = await exchange.get_historical_klines(
    symbol="BTCUSDT",
    interval="1h",
    start_time_ms=start_time,
    end_time_ms=end_time,
    limit=1000
)

# Get historical trades
historical_trades = await exchange.get_historical_agg_trades(
    symbol="BTCUSDT",
    start_time_ms=start_time,
    end_time_ms=end_time,
    limit=1000
)
```

### Order Management

```python
# Get open orders
open_orders = await exchange.get_open_orders("BTCUSDT")

# Cancel order
cancel_result = await exchange.cancel_order("BTCUSDT", order_id)

# Get order status
order_status = await exchange.get_order_status("BTCUSDT", order_id)
```

## Migration Guide

### For Existing Code

Existing code that uses exchange methods should continue to work without changes, as the standardized methods maintain backward compatibility for the most common use cases.

### For New Code

When writing new code, prefer the standardized methods:

```python
# ✅ Preferred - standardized interface
klines = await exchange.get_klines("BTCUSDT", "1h", 100)

# ❌ Avoid - exchange-specific methods
klines = await exchange.get_klines_raw("BTCUSDT", "1h", 100)
```

### Error Handling

All standardized methods include consistent error handling:

```python
try:
    klines = await exchange.get_klines("BTCUSDT", "1h", 100)
except Exception as e:
    logger.error(f"Failed to get klines: {e}")
    # Handle error appropriately
```

## Benefits

### 1. Consistency

- All exchanges have the same method signatures
- Consistent parameter names and types
- Standardized return data formats

### 2. Maintainability

- Single interface to learn and maintain
- Easier to add new exchanges
- Centralized error handling and logging

### 3. Type Safety

- Strong typing with proper type hints
- IDE support for autocomplete and error detection
- Better code documentation

### 4. Extensibility

- Easy to add new standardized methods
- Abstract base class pattern allows for exchange-specific optimizations
- Consistent interface makes testing easier

## Future Enhancements

### Planned Improvements

1. **WebSocket Standardization**: Standardize WebSocket interfaces across exchanges
2. **Rate Limiting**: Implement consistent rate limiting across all exchanges
3. **Error Recovery**: Enhanced error recovery and retry mechanisms
4. **Performance Monitoring**: Standardized performance metrics and monitoring

### Adding New Exchanges

To add a new exchange:

1. Create a new class that inherits from `BaseExchange`
2. Implement all required abstract methods
3. Add the exchange to the `ExchangeFactory`
4. Add appropriate tests

Example:

```python
class NewExchange(BaseExchange):
    def __init__(self, api_key: str, api_secret: str, trade_symbol: str):
        super().__init__(api_key, api_secret, trade_symbol)
        # Initialize exchange-specific components
        
    async def _initialize_exchange(self) -> None:
        # Initialize exchange connection
        
    async def _convert_to_market_data(self, raw_data, symbol, interval) -> list[MarketData]:
        # Convert exchange-specific data format to MarketData
        
    # Implement other abstract methods...
```

## Testing

### Unit Tests

Each exchange should have comprehensive unit tests covering:

- Standardized method implementations
- Data format conversions
- Error handling scenarios
- Rate limiting behavior

### Integration Tests

Integration tests should verify:

- Real API connectivity
- Data consistency across exchanges
- Performance characteristics
- Error recovery mechanisms

## Conclusion

The exchange API standardization provides a solid foundation for the Ares trading bot, ensuring consistent behavior across different exchange providers while maintaining the flexibility to handle exchange-specific features and optimizations. 