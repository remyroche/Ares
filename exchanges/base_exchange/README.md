# Base Exchange Module

This directory contains the core base exchange functionality for the ML model-based trading system.

## Overview

The `base_exchange/` module provides the foundational components for ML model-specific exchange operations. It enables the system to:

1. **Send orders to the exchange associated with each ML model**
2. **Route responses back from the appropriate exchange**
3. **Handle ML model-based message routing and processing**
4. **Manage ML model to exchange mappings**

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

### ML Model-Based Exchange Routing
```python
# Send order to the exchange associated with the ML model
response = await receiver.send_order_for_ml_model(
    symbol="BTCUSDT",
    side="buy",
    order_type="market",
    quantity=0.001,
    ml_model_id="binance_prophet_model"
)
```

### ML Model Management
```python
# Register ML model to exchange association
receiver.register_ml_model_exchange("my_model", "binance")

# Get exchange for ML model
exchange = receiver.get_ml_model_exchange("my_model")

# Set default exchange for unknown ML models
receiver.set_default_ml_exchange("binance")
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

### ML Model-Based Order Routing
```python
from exchanges import TradingReceiver

# Initialize receiver with ML model mappings
config = {
    "ml_model_exchanges": {
        "binance_prophet_model": "binance",
        "okx_random_forest": "okx",
        "gateio_neural_net": "gateio"
    },
    "default_ml_exchange": "binance"
}

receiver = TradingReceiver(config)
await receiver.start()

# Send order using ML model routing
response = await receiver.send_order_for_ml_model(
    symbol="BTCUSDT",
    side="buy",
    order_type="market",
    quantity=0.001,
    ml_model_id="binance_prophet_model"
)

print(f"Order sent to: {response.metadata['target_exchange']}")
print(f"Success: {response.success}")
```

### ML Model Management
```python
# Register ML model associations
receiver.register_ml_model_exchange("my_model", "binance")
receiver.register_ml_model_exchange("another_model", "okx")

# Get exchange for ML model
exchange = receiver.get_ml_model_exchange("my_model")  # Returns "binance"

# Set default exchange
receiver.set_default_ml_exchange("binance")

# Get all mappings
mappings = receiver.get_all_ml_model_exchanges()
print(f"ML Model Mappings: {mappings}")
```

### Intelligent Routing with ML Models
```python
# Route using ML model strategy
response = await receiver.send_order_with_routing(
    symbol="BTCUSDT",
    side="buy",
    order_type="market",
    quantity=0.001,
    routing_strategy="ml_model",
    ml_model_id="binance_prophet_model"
)

# The order will be sent to Binance (associated with the ML model)
print(f"Routed to: {response.metadata['target_exchange']}")
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
- **ml_model**: Send to exchange associated with ML model (DEFAULT)
- **broadcast**: Send to all exchanges (legacy compatibility)
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
print(f"ML models registered: {stats['ml_model']['registered_ml_models']}")
print(f"Default ML exchange: {stats['ml_model']['default_ml_exchange']}")
print(f"ML model mappings: {stats['ml_model']['ml_model_exchanges']}")
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

The base exchange module integrates seamlessly with ML model-based trading:

### ML Model-Based Trading
```python
# Each ML model is associated with a specific exchange
config = {
    "ml_model_exchanges": {
        "binance_prophet_model": "binance",
        "okx_random_forest": "okx",
        "gateio_neural_net": "gateio"
    }
}

receiver = TradingReceiver(config)

# Orders are sent to the exchange associated with each ML model
response = await receiver.send_order_for_ml_model(
    symbol="BTCUSDT",
    side="buy",
    order_type="market",
    quantity=0.001,
    ml_model_id="binance_prophet_model"
)
# This will automatically route to Binance
```

### ML Model Signal Processing
```python
# ML models send signals with their model ID
signal = TradingSignal(
    symbol="BTCUSDT",
    action="buy",
    quantity=0.001,
    confidence=0.85,
    strategy="binance_prophet_model",  # This determines target exchange
    metadata={"ml_model_id": "binance_prophet_model"}
)

# The system automatically routes to the correct exchange
response = await receiver.send_order_with_routing(
    symbol=signal.symbol,
    side=signal.action,
    order_type="market",
    quantity=signal.quantity,
    routing_strategy="ml_model",
    ml_model_id=signal.metadata["ml_model_id"]
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

### TradingReceiver ML Model Methods
- `send_order_for_ml_model()` - Send order to ML model-associated exchange
- `send_order_with_routing()` - Intelligent routing with ML model strategy
- `register_ml_model_exchange()` - Register ML model to exchange mapping
- `get_ml_model_exchange()` - Get exchange for specific ML model
- `set_default_ml_exchange()` - Set default exchange for unknown ML models

### ML Model Management Methods
- `unregister_ml_model_exchange()` - Remove ML model association
- `get_all_ml_model_exchanges()` - Get all ML model mappings
- `_get_ml_models_by_exchange()` - Get ML models grouped by exchange

### BaseExchange Components
- `ExchangeMessageHandler` - Message processing and routing
- `ExchangeResponseHandler` - Response management and aggregation
- `MultiExchangeBase` - ML model-based operation base class

This base exchange module provides a solid foundation for building sophisticated ML model-based trading applications where each ML model is associated with a specific exchange, ensuring orders are sent to the correct exchange for each model's data source.