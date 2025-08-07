# Enhanced Event Bus System Guide

## Overview

The Enhanced Event Bus is a comprehensive event-driven architecture implementation that provides event sourcing, versioning, and persistence capabilities for the Ares trading system. It enables decoupled communication between components while maintaining a complete audit trail of all system events.

## Key Features

### 1. Event Sourcing
- **Complete Audit Trail**: Every event is stored with full metadata including timestamps, correlation IDs, and source information
- **Event Replay**: Ability to replay events from any point in time for debugging or system recovery
- **State Reconstruction**: Rebuild system state from stored events

### 2. Event Versioning
- **Schema Versioning**: Events include schema version information for backward compatibility
- **Event Migration**: Automatic migration of events between schema versions
- **Version Validation**: Validate events against their schema versions

### 3. Event Persistence
- **Multiple Storage Options**: File-based storage with extensible interface for other backends
- **Efficient Storage**: Compressed JSON Lines format for optimal storage and retrieval
- **TTL Management**: Automatic cleanup of old events based on configurable time-to-live

### 4. Snapshots
- **Automatic Snapshots**: Create system snapshots at configurable intervals
- **Fast Recovery**: Quick system state restoration from snapshots plus incremental events
- **Multiple Aggregates**: Support for multiple aggregate snapshots

## Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Publishers    │    │  Enhanced       │    │   Subscribers   │
│                 │───▶│  Event Bus      │───▶│                 │
│ - TradingBot    │    │                 │    │ - RiskManager   │
│ - MarketData    │    │ - Queue         │    │ - Analyst       │
│ - Exchange      │    │ - Dispatcher    │    │ - Reporter      │
└─────────────────┘    │ - Persistence   │    └─────────────────┘
                       │ - Versioning    │
                       └─────────────────┘
                               │
                               ▼
                       ┌─────────────────┐
                       │  Event Store    │
                       │                 │
                       │ - Events        │
                       │ - Snapshots     │
                       │ - Metadata      │
                       └─────────────────┘
```

### Event Structure

```python
@dataclass
class Event:
    event_type: EventType
    data: Any
    metadata: EventMetadata

@dataclass
class EventMetadata:
    event_id: str
    version: str
    schema_version: str
    timestamp: datetime
    source: str
    correlation_id: Optional[str]
    causation_id: Optional[str]
    aggregate_id: Optional[str]
    sequence_number: int
    retry_count: int
    status: EventStatus
    tags: Dict[str, str]
```

## Configuration

### Event Bus Configuration

```python
"event_bus": {
    "processing_interval": 1,          # Seconds between processing cycles
    "max_history": 1000,              # Max events in memory
    "enable_persistence": True,        # Enable event storage
    "enable_snapshots": True,          # Enable snapshots
    "snapshot_frequency": 100,         # Snapshot every N events
    "storage_path": "event_store",     # Storage directory
    "max_retry_count": 3,             # Max retry attempts
    "batch_size": 100,                # Max events per batch
    "event_ttl_days": 30,             # Event time-to-live
    "compression_enabled": True,       # Enable compression
    "schema_validation": True,         # Enable validation
    "audit_trail": True,              # Enable audit logging
    "correlation_tracking": True,      # Enable correlation tracking
    "metrics_enabled": True,          # Enable metrics
    "health_check_interval": 60       # Health check interval
}
```

## Usage Examples

### Basic Usage

```python
from src.interfaces.enhanced_event_bus import setup_enhanced_event_bus, EventType

# Setup event bus
event_bus = await setup_enhanced_event_bus()

# Subscribe to events
def handle_market_data(event):
    print(f"Received: {event.data}")

event_bus.subscribe(EventType.MARKET_DATA_RECEIVED, handle_market_data)

# Publish events
await event_bus.publish(
    EventType.MARKET_DATA_RECEIVED,
    {"symbol": "BTCUSDT", "price": 50000},
    source="MarketDataProvider"
)

# Start processing
await event_bus.run()
```

### Event Replay

```python
# Replay events for specific aggregate
events = await event_bus.replay_events(
    aggregate_id="trader_1",
    from_sequence=0,
    to_sequence=100
)

# Rebuild state from events
state = await event_bus.rebuild_from_events("trader_1")
```

### Correlation Tracking

```python
# Publish related events with correlation ID
correlation_id = str(uuid.uuid4())

await event_bus.publish(
    EventType.TRADE_DECISION_MADE,
    trade_data,
    correlation_id=correlation_id,
    aggregate_id="trader_1"
)

await event_bus.publish(
    EventType.TRADE_EXECUTED,
    execution_data,
    correlation_id=correlation_id,
    aggregate_id="trader_1"
)
```

## Event Types

The system supports the following event types:

- `MARKET_DATA_RECEIVED`: Market data updates
- `ANALYSIS_COMPLETED`: Analysis results
- `STRATEGY_FORMULATED`: Strategy decisions
- `TRADE_DECISION_MADE`: Trading decisions
- `TRADE_EXECUTED`: Trade executions
- `RISK_ALERT`: Risk management alerts
- `PERFORMANCE_UPDATE`: Performance metrics
- `MODEL_UPDATED`: Model updates
- `SYSTEM_ERROR`: System errors
- `COMPONENT_STARTED`: Component lifecycle
- `COMPONENT_STOPPED`: Component lifecycle
- `SYSTEM_HEALTH_CHECK`: Health checks
- `CONFIGURATION_CHANGED`: Config changes
- `SNAPSHOT_CREATED`: Snapshot events

## Storage Format

### Event Storage
Events are stored in JSON Lines format with daily rotation:

```
event_store/
├── events/
│   ├── events_2024-01-15.jsonl
│   ├── events_2024-01-16.jsonl
│   └── ...
└── snapshots/
    ├── snapshot_system_100.json
    ├── snapshot_trader_1_150.json
    └── ...
```

### Event File Format

```json
{"event_type":"trade_executed","data":{"symbol":"BTCUSDT","quantity":1.0},"metadata":{"event_id":"uuid","version":"1.0.0","timestamp":"2024-01-15T10:30:00Z","source":"TradingBot"}}
```

## Error Handling

### Retry Mechanism
- Failed events are automatically retried up to `max_retry_count`
- Exponential backoff between retry attempts
- Failed events are logged for manual investigation

### Schema Validation
- Events are validated against their schema version
- Invalid events are rejected and logged
- Backward compatibility maintained through version migration

### Error Events
System errors are captured as events:

```python
await event_bus.publish(
    EventType.SYSTEM_ERROR,
    {
        "error_type": "ValidationError",
        "message": "Invalid event schema",
        "component": "EventBus",
        "stack_trace": traceback.format_exc()
    },
    source="EventBus"
)
```

## Monitoring and Metrics

### Built-in Metrics
- `events_processed`: Total events processed
- `events_failed`: Total events failed
- `snapshots_created`: Total snapshots created
- `replays_performed`: Total replay operations

### Health Checks
- Queue size monitoring
- Processing latency tracking
- Storage health verification
- Subscriber status monitoring

### Status Information

```python
status = event_bus.get_status()
# Returns:
# {
#     "timestamp": "2024-01-15T10:30:00Z",
#     "status": "running",
#     "metrics": {...},
#     "queue_size": 5,
#     "subscribers_count": {"market_data_received": 3},
#     "persistence_enabled": True,
#     "snapshots_enabled": True
# }
```

## Testing

### Unit Tests
Comprehensive test suite covering:
- Event creation and serialization
- Event bus functionality
- Storage operations
- Version management
- Error handling

### Integration Tests
End-to-end testing of:
- Complete event flow
- Persistence and replay
- Snapshot creation and restoration
- Error recovery

### Performance Tests
- Event throughput testing
- Memory usage monitoring
- Storage performance benchmarks

## Best Practices

### Event Design
1. **Immutable Events**: Events should be immutable once created
2. **Rich Metadata**: Include comprehensive metadata for debugging
3. **Correlation IDs**: Use correlation IDs to track related events
4. **Aggregate IDs**: Group related events by aggregate

### Performance
1. **Batch Processing**: Process events in batches for efficiency
2. **Async Operations**: Use async/await for non-blocking operations
3. **Memory Management**: Configure appropriate history limits
4. **Storage Optimization**: Use compression for large events

### Monitoring
1. **Health Checks**: Implement regular health checks
2. **Metrics Collection**: Monitor key performance indicators
3. **Error Tracking**: Comprehensive error logging and alerting
4. **Audit Trails**: Maintain complete audit trails for compliance

### Security
1. **Access Control**: Implement appropriate access controls
2. **Data Encryption**: Encrypt sensitive event data
3. **Audit Logging**: Log all access and modifications
4. **Compliance**: Ensure compliance with data retention policies

## Migration Guide

### From Basic Event Bus

1. **Update Imports**:
   ```python
   # Old
   from src.interfaces.event_bus import EventBus, setup_event_bus
   
   # New
   from src.interfaces.enhanced_event_bus import EnhancedEventBus, setup_enhanced_event_bus
   ```

2. **Update Configuration**:
   - Add enhanced event bus configuration to system config
   - Configure persistence and snapshot settings

3. **Update Event Publishing**:
   ```python
   # Old
   await event_bus.publish("market_data_received", data)
   
   # New
   await event_bus.publish(
       EventType.MARKET_DATA_RECEIVED,
       data,
       source="MarketDataProvider",
       correlation_id=correlation_id
   )
   ```

4. **Update Event Handling**:
   - Events now have rich metadata structure
   - Access data via `event.data` instead of direct dictionary

## Troubleshooting

### Common Issues

1. **Storage Permission Errors**:
   - Ensure write permissions to storage directory
   - Check disk space availability

2. **Performance Issues**:
   - Reduce batch size if memory constrained
   - Increase processing interval if CPU constrained
   - Optimize subscriber implementations

3. **Schema Validation Errors**:
   - Check event data against schema requirements
   - Update schema versions appropriately
   - Implement proper migration logic

4. **Correlation Tracking Issues**:
   - Ensure correlation IDs are propagated correctly
   - Use consistent correlation ID format
   - Implement correlation ID generation strategy

## Future Enhancements

### Planned Features
1. **Database Backend**: Support for SQL and NoSQL databases
2. **Event Streaming**: Integration with Kafka or similar systems
3. **Distributed Events**: Support for multi-node event distribution
4. **Event Projections**: Materialized views from event streams
5. **CQRS Support**: Command Query Responsibility Segregation
6. **Event Sourcing Framework**: Complete event sourcing implementation

### Performance Improvements
1. **Async Storage**: Fully asynchronous storage operations
2. **Bulk Operations**: Bulk insert/retrieve operations
3. **Caching Layer**: Redis-based event caching
4. **Compression**: Advanced compression algorithms
5. **Partitioning**: Event partitioning strategies

This enhanced event bus provides a solid foundation for building robust, scalable, and maintainable event-driven systems with complete audit trails and replay capabilities.