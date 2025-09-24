# Base Exchange Module

This directory contains the core base exchange functionality for the multi-exchange trading system.

## Overview

The `base_exchange/` module provides the foundational components for exchange-agnostic trading operations. It enables the system to:

1. **Send orders to all exchanges simultaneously**
2. **Route responses back from all exchanges**
3. **Handle multi-exchange message queuing and processing**
4. **Aggregate and manage responses from multiple exchanges**

## Architecture

### Core Components

#### 1. BaseExchange (`base_exchange.py`)
- Abstract base class for all exchange implementations
- Provides standardized method signatures
- Common functionality for all exchanges
- Multi-exchange broadcasting capabilities

#### 2. Exchange Interface (`exchange_interface.py`)
- Defines core interfaces for exchanges
- Type definitions and enums
- Exchange configuration structures
- Event and messaging interfaces

#### 3. Message Handler (`message_handler.py`)
- Handles messages between system and exchanges
- Priority-based message queuing
- Message routing strategies
- Batch order processing

#### 4. Response Handler (`response_handler.py`)
- Manages responses from exchanges
- Response aggregation strategies
- Callback mechanisms
- Response filtering and routing

## Key Features

### Multi-Exchange Order Broadcasting
```python
# Send order to all configured exchanges
responses = await receiver.send_order_to_all_exchanges(
    symbol="BTCUSDT",
    side="buy",
    order_type="market",
    quantity=0.001
)
```

### Intelligent Routing Strategies
```python
# Route with different strategies
response = await receiver.send_order_with_routing(
    symbol="BTCUSDT",
    side="buy",
    order_type="limit",
    quantity=0.001,
    routing_strategy="best_price"  # or "broadcast", "primary", "failover"
)
```

### Response Aggregation
```python
# Get aggregated responses from all exchanges
aggregated = await response_handler.get_aggregated_response(request_id)
```

### Message Queuing
```python
# Priority-based message queuing
await message_handler.send_message(message, target_exchanges)
```

## Usage Examples

### Basic Multi-Exchange Order
```python
from exchanges import TradingReceiver

# Initialize receiver
receiver = TradingReceiver(config)
await receiver.start()

# Send to all exchanges
responses = await receiver.send_order_to_all_exchanges(
    symbol="BTCUSDT",
    side="buy",
    order_type="market",
    quantity=0.001
)

# Check responses
for exchange, response in responses.items():
    print(f"{exchange}: {response.success}")
```

### Advanced Routing
```python
# Primary with failover
response = await receiver.send_order_with_routing(
    symbol="ETHUSDT",
    side="sell",
    order_type="limit",
    quantity=0.01,
    price=2000.0,
    routing_strategy="primary"
)

# Best price execution
response = await receiver.send_order_with_routing(
    symbol="BTCUSDT",
    side="buy",
    order_type="market",
    quantity=0.001,
    routing_strategy="best_price"
)
```

### Response Handling
```python
from exchanges.base_exchange import ExchangeResponseHandler, ResponseType

# Initialize response handler
response_handler = ExchangeResponseHandler()

# Register callback
callback_id = response_handler.register_response_callback(
    ResponseType.ORDER_EXECUTION,
    my_callback_function
)

# Handle responses
await response_handler.handle_response(response_data)
```

### Message Broadcasting
```python
from exchanges.base_exchange import ExchangeMessageHandler, MessageType

# Initialize message handler
message_handler = ExchangeMessageHandler()

# Create message
message = ExchangeMessage(
    id="order_123",
    message_type=MessageType.ORDER,
    payload={"symbol": "BTCUSDT", "quantity": 0.001}
)

# Send to multiple exchanges
responses = await message_handler.send_message(
    message,
    ["binance", "okx", "gateio"]
)
```

## Configuration

### Multi-Exchange Configuration
```python
config = {
    "exchanges": {
        "binance": {"api_key": "...", "api_secret": "..."},
        "okx": {"api_key": "...", "api_secret": "..."},
        "gateio": {"api_key": "...", "api_secret": "..."}
    },
    "primary_exchange": "binance",
    "failover_exchanges": ["okx", "gateio"],
    "broadcast_enabled": True,
    "load_balancing_enabled": False
}
```

### Routing Strategies
- **broadcast**: Send to all exchanges
- **primary**: Send to primary with failover
- **best_price**: Route to exchange with best price
- **round_robin**: Distribute across exchanges
- **least_loaded**: Route to least busy exchange

## Error Handling

The system provides comprehensive error handling:

### Automatic Retry
- Exponential backoff with jitter
- Configurable retry limits
- Error categorization

### Response Aggregation
- Collect responses from all exchanges
- Handle partial failures
- Aggregate successful responses

### Failover Support
- Automatic failover to backup exchanges
- Configurable failover chains
- Graceful degradation

## Monitoring and Statistics

### System Statistics
```python
stats = await receiver.get_statistics()
print(f"Multi-exchange orders: {stats['multi_exchange']['multi_exchange_orders']}")
print(f"Primary exchange: {stats['multi_exchange']['primary_exchange']}")
```

### Queue Monitoring
```python
queue_status = await message_handler.get_queue_status()
print(f"Pending messages: {queue_status['total_pending_messages']}")
```

### Response Statistics
```python
response_stats = await response_handler.get_response_statistics()
print(f"Active callbacks: {response_stats['registered_callbacks']}")
```

## Integration with Trading System

The base exchange module integrates seamlessly with the existing trading system:

### With TradingOrchestrator
```python
# Enhanced orchestrator can use multi-exchange features
signal = TradingSignal(
    symbol="BTCUSDT",
    action="buy",
    quantity=0.001,
    metadata={"routing_strategy": "best_price"}
)
```

### With UnifiedTradingSystem
```python
# System automatically uses multi-exchange capabilities
system = await create_trading_system(
    exchanges=["binance", "okx", "gateio"],
    enable_multi_exchange=True
)
```

## Performance Considerations

### Message Queuing
- Priority-based processing
- Configurable queue sizes
- Batch processing support

### Response Handling
- Asynchronous response processing
- Memory-efficient aggregation
- Timeout management

### Resource Management
- Connection pooling
- Rate limit management
- Resource cleanup

## Testing

### Unit Tests
```python
# Test multi-exchange functionality
async def test_multi_exchange():
    responses = await receiver.send_order_to_all_exchanges(...)
    assert len(responses) == 3  # binance, okx, gateio
```

### Integration Tests
```python
# Test with actual exchange connections
async def test_with_real_exchanges():
    receiver = TradingReceiver(real_config)
    await receiver.start()
    # ... test logic
    await receiver.stop()
```

## Troubleshooting

### Common Issues

1. **No Exchanges Registered**
   ```python
   # Check registered exchanges
   exchanges = await receiver.exchange_registry.get_registered_exchanges()
   print(f"Registered: {exchanges}")
   ```

2. **Message Queue Backlog**
   ```python
   # Check queue status
   status = await receiver.message_handler.get_queue_status()
   print(f"Queue size: {status['total_pending_messages']}")
   ```

3. **Response Timeouts**
   ```python
   # Check response handler
   stats = await receiver.response_handler.get_response_statistics()
   print(f"Pending: {stats['pending_responses']}")
   ```

### Debug Mode
```python
# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Get detailed statistics
stats = await receiver.get_statistics()
print(json.dumps(stats, indent=2))
```

## Future Enhancements

### Planned Features
- WebSocket streaming support
- Advanced order types
- Cross-exchange arbitrage detection
- Real-time performance analytics
- Machine learning-based routing

### Extension Points
- Custom routing strategies
- Additional exchange adapters
- Custom response processors
- Enhanced monitoring dashboards

## API Reference

### TradingReceiver Enhanced Methods
- `send_order_to_all_exchanges()` - Broadcast orders to all exchanges
- `send_order_with_routing()` - Intelligent routing strategies
- `get_multi_exchange_order_status()` - Track multi-exchange orders
- `cancel_multi_exchange_order()` - Cancel across all exchanges

### BaseExchange Components
- `ExchangeMessageHandler` - Message processing and routing
- `ExchangeResponseHandler` - Response management and aggregation
- `MultiExchangeBase` - Multi-exchange operation base class

This base exchange module provides a solid foundation for building sophisticated multi-exchange trading applications with robust error handling, intelligent routing, and comprehensive monitoring capabilities.