# Dependency Injection Guide for Ares Trading System

This guide explains how to use the enhanced dependency injection patterns implemented throughout the Ares trading system.

## Overview

The Ares trading system now implements proper dependency injection (DI) patterns that provide:

- **Loose coupling** between components
- **Easy testing** through dependency mocking
- **Flexible configuration** management
- **Scalable architecture** for adding new components
- **Proper lifecycle management** for services

## Core Components

### 1. DependencyContainer

The main DI container that manages service registration and resolution.

```python
from src.core.dependency_injection import DependencyContainer, ServiceLifetime

# Create container
container = DependencyContainer(config)

# Register services
container.register(
    IAnalyst,
    DIAnalyst,
    lifetime=ServiceLifetime.SINGLETON,
    config={"analysis_interval": 3600}
)

# Resolve services
analyst = container.resolve(IAnalyst)
```

### 2. Service Lifetimes

Three service lifetimes are supported:

- **SINGLETON**: One instance per container
- **TRANSIENT**: New instance every time
- **SCOPED**: One instance per scope

### 3. ServiceRegistry

Centralized service registration for all trading components.

```python
from src.core.service_registry import ServiceRegistry, create_configured_container

# Create configured container
container = create_configured_container(CONFIG)

# Or manually register
registry = ServiceRegistry(container)
registry.register_all_services(CONFIG)
```

## Usage Examples

### Basic Usage

```python
from src.core.di_launcher import DILauncher

# Create launcher
launcher = DILauncher(config)

# Launch paper trading
components = await launcher.launch_paper_trading("ETHUSDT", "BINANCE")

# Access components
analyst = components["analyst"]
strategist = components["strategist"]
```

### Advanced Usage with Custom Services

```python
from src.core.dependency_injection import AsyncServiceContainer
from src.core.enhanced_factories import ComprehensiveFactory

# Create container
container = AsyncServiceContainer(config)

# Register custom implementation
container.register(ICustomService, CustomServiceImpl)

# Create factory
factory = ComprehensiveFactory(container)

# Create complete system
system = await factory.create_full_system(config)
```

### Configuration Injection

Components automatically receive configuration through constructor injection:

```python
class MyTradingComponent(InjectableBase):
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        exchange_client: IExchangeClient | None = None,
        state_manager: IStateManager | None = None,
    ):
        super().__init__(config)
        self.exchange_client = exchange_client
        self.state_manager = state_manager
```

### Scope Management

```python
# Begin scope
container.begin_scope("trading_session")

# Create scoped services
scoped_service = container.resolve(IScopedService)

# End scope (automatically cleans up scoped instances)
container.end_scope("trading_session")
```

## Creating DI-Aware Components

### Base Classes

Use the provided base classes for automatic DI support:

```python
from src.core.injectable_base import TradingComponentBase

class MyAnalyst(TradingComponentBase, IAnalyst):
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        exchange_client: IExchangeClient | None = None,
        state_manager: IStateManager | None = None,
        event_bus: IEventBus | None = None,
    ):
        super().__init__(config, exchange_client, state_manager, event_bus)
    
    async def initialize(self) -> bool:
        """Initialize with dependency validation."""
        if not await super().initialize():
            return False
        
        # Custom initialization logic
        return True
```

### Factory Functions

Register factory functions for complex service creation:

```python
def create_custom_exchange(container: DependencyContainer) -> IExchangeClient:
    config = container.get_config("exchange", {})
    return CustomExchange(config)

container.register_factory(IExchangeClient, create_custom_exchange)
```

## Integration Patterns

### 1. Training Pipeline Integration

```python
from src.training.di_training_manager import DITrainingManager

# Training manager uses DI for all components
training_manager = container.resolve(DITrainingManager)
await training_manager.run_training_pipeline("ETHUSDT", "BINANCE")
```

### 2. Event-Driven Architecture

Components automatically participate in event-driven communication:

```python
# Components are wired automatically
analyst = container.resolve(IAnalyst)  # Publishes analysis events
strategist = container.resolve(IStrategist)  # Subscribes to analysis events
```

### 3. Configuration Management

All components receive configuration through DI:

```python
# Configuration is automatically injected
{
    "analyst": {
        "analysis_interval": 3600,
        "enable_technical_analysis": True
    },
    "strategist": {
        "risk_tolerance": 0.02
    }
}
```

## Best Practices

### 1. Use Interfaces

Always depend on interfaces, not concrete implementations:

```python
# Good
def __init__(self, exchange_client: IExchangeClient):
    pass

# Bad
def __init__(self, exchange_client: BinanceClient):
    pass
```

### 2. Constructor Injection

Use constructor injection for required dependencies:

```python
class MyService:
    def __init__(
        self,
        required_service: IRequiredService,
        optional_service: IOptionalService | None = None,
    ):
        self.required_service = required_service
        self.optional_service = optional_service
```

### 3. Lifecycle Management

Implement proper initialization and cleanup:

```python
class MyComponent(InjectableBase):
    async def initialize(self) -> bool:
        # Initialize component
        return True
    
    async def shutdown(self) -> None:
        # Clean up resources
        pass
```

### 4. Configuration Validation

Validate configuration in initialization:

```python
async def initialize(self) -> bool:
    if not await super().initialize():
        return False
    
    if not self._validate_config():
        return False
    
    return True
```

## Testing with DI

### Mock Dependencies

```python
from unittest.mock import Mock

# Create mock dependencies
mock_exchange = Mock(spec=IExchangeClient)
mock_state_manager = Mock(spec=IStateManager)

# Create container with mocks
container = DependencyContainer()
container.register_instance(IExchangeClient, mock_exchange)
container.register_instance(IStateManager, mock_state_manager)

# Test component
component = container.resolve(IAnalyst)
```

### Test Configuration

```python
test_config = {
    "analyst": {
        "analysis_interval": 60,  # Faster for testing
        "enable_technical_analysis": False
    }
}

container = DependencyContainer(test_config)
```

## Migration Guide

### From Old System

1. **Identify Dependencies**: List all dependencies for each component
2. **Add Type Hints**: Add proper type hints to constructors
3. **Inherit Base Classes**: Use `InjectableBase` or `TradingComponentBase`
4. **Register Services**: Register components in `ServiceRegistry`
5. **Update Creation**: Use DI container instead of manual instantiation

### Example Migration

Before:
```python
class OldAnalyst:
    def __init__(self, config):
        self.config = config
        self.exchange_client = BinanceClient(config["exchange"])
        self.state_manager = StateManager(config["state"])
```

After:
```python
class NewAnalyst(AnalystBase, IAnalyst):
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        exchange_client: IExchangeClient | None = None,
        state_manager: IStateManager | None = None,
    ):
        super().__init__(config, exchange_client, state_manager)
```

## Troubleshooting

### Common Issues

1. **Service Not Registered**: Ensure service is registered before resolution
2. **Circular Dependencies**: Redesign to avoid circular dependencies
3. **Missing Dependencies**: Check constructor parameters and type hints
4. **Configuration Issues**: Validate configuration structure

### Debugging

Enable debug logging to see DI container operations:

```python
import logging
logging.getLogger("DependencyContainer").setLevel(logging.DEBUG)
```

## Performance Considerations

- **Singleton services** are created once and reused
- **Transient services** have minimal overhead
- **Scoped services** are efficient within their scope
- **Container resolution** is fast for registered services

## Future Enhancements

Planned improvements include:

1. **Decorator-based registration**
2. **Automatic interface detection**
3. **Configuration hot-reloading**
4. **Health check integration**
5. **Metrics and monitoring**

This dependency injection system provides a solid foundation for scalable, testable, and maintainable trading system architecture.