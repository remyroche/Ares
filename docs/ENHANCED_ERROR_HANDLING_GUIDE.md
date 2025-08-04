# Enhanced Error Handling and Recovery Strategies Guide

## Overview

This guide documents the enhanced error handling system implemented in the Ares trading bot, which provides automatic recovery strategies, circuit breaker patterns, and comprehensive type safety with 100% type hint coverage.

## Table of Contents

1. [Core Components](#core-components)
2. [Circuit Breaker Pattern](#circuit-breaker-pattern)
3. [Recovery Strategies](#recovery-strategies)
4. [Error Recovery Manager](#error-recovery-manager)
5. [Enhanced Error Handler](#enhanced-error-handler)
6. [Type Safety](#type-safety)
7. [Usage Examples](#usage-examples)
8. [Best Practices](#best-practices)
9. [Testing](#testing)

## Core Components

### Circuit Breaker Pattern

The circuit breaker pattern prevents cascading failures by monitoring for failures and temporarily disabling operations when a threshold is reached.

```python
from src.utils.error_handler import CircuitBreaker, CircuitBreakerConfig

# Create circuit breaker configuration
config = CircuitBreakerConfig(
    failure_threshold=5,        # Number of failures before opening
    recovery_timeout=60.0,      # Time to wait before half-open
    expected_exception=ValueError,  # Exception type to monitor
    monitor_interval=10.0       # Monitoring interval
)

# Create circuit breaker
circuit_breaker = CircuitBreaker(config)

# Use circuit breaker
async def risky_operation():
    # Your operation here
    pass

result = await circuit_breaker.call(risky_operation)
```

### Recovery Strategies

Three types of recovery strategies are available:

#### 1. Retry Strategy

Implements exponential backoff with jitter for retrying failed operations.

```python
from src.utils.error_handler import RetryStrategy

strategy = RetryStrategy(
    max_retries=3,          # Maximum retry attempts
    base_delay=1.0,         # Base delay between retries
    max_delay=60.0,         # Maximum delay cap
    backoff_factor=2.0,     # Exponential backoff factor
    jitter=True             # Add random jitter to delays
)
```

#### 2. Fallback Strategy

Provides multiple fallback operations to try when the primary operation fails.

```python
from src.utils.error_handler import FallbackStrategy

def primary_operation():
    # Primary operation
    pass

def fallback_operation():
    # Fallback operation
    pass

strategy = FallbackStrategy(fallback_operations=[
    primary_operation,
    fallback_operation
])
```

#### 3. Graceful Degradation Strategy

Returns a default value when operations fail, allowing the system to continue operating.

```python
from src.utils.error_handler import GracefulDegradationStrategy

strategy = GracefulDegradationStrategy(
    default_return="degraded_value",
    error_types=[ValueError, TypeError]  # Optional: specific error types
)
```

## Error Recovery Manager

The `ErrorRecoveryManager` coordinates multiple recovery strategies and provides a unified interface for automatic error recovery.

```python
from src.utils.error_handler import ErrorRecoveryManager

manager = ErrorRecoveryManager()

# Add recovery strategies
manager.add_strategy(RetryStrategy(max_retries=2))
manager.add_strategy(GracefulDegradationStrategy(default_return=None))

# Execute with automatic recovery
result = await manager.execute_with_recovery(risky_operation)
```

## Enhanced Error Handler

The enhanced error handler provides decorators with built-in recovery strategies.

### Basic Usage

```python
from src.utils.error_handler import handle_errors, handle_specific_errors

# Generic error handling with recovery
@handle_errors(
    exceptions=(ValueError, TypeError),
    default_return=None,
    context="my_component",
    recovery_strategies=[
        RetryStrategy(max_retries=2),
        GracefulDegradationStrategy(default_return="fallback")
    ]
)
async def my_function():
    # Your function implementation
    pass

# Specific error handling
@handle_specific_errors(
    error_handlers={
        ValueError: (None, "Value error occurred"),
        TypeError: ("default", "Type error occurred")
    },
    default_return=None,
    context="my_component",
    recovery_strategies=[
        RetryStrategy(max_retries=1)
    ]
)
async def my_function():
    # Your function implementation
    pass
```

## Type Safety

All components have comprehensive type hints ensuring 100% type safety coverage.

### Type Variables

```python
from typing import TypeVar, Callable, Optional, Any

T = TypeVar('T')  # Generic type for return values
F = TypeVar('F', bound=Callable[..., Any])  # Function type
```

### Function Signatures

```python
def safe_operation(
    operation: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> Optional[T]:
    """Execute operation safely with type hints."""
    pass

async def safe_async_operation(
    operation: Callable[..., Awaitable[T]],
    *args: Any,
    **kwargs: Any,
) -> Optional[T]:
    """Execute async operation safely with type hints."""
    pass
```

## Usage Examples

### Example 1: Network Operations with Circuit Breaker

```python
from src.utils.error_handler import (
    create_circuit_breaker,
    create_retry_strategy,
    handle_errors
)

# Create circuit breaker for network operations
network_circuit_breaker = create_circuit_breaker(
    name="network_operations",
    failure_threshold=3,
    recovery_timeout=30.0,
    expected_exception=ConnectionError
)

@handle_errors(
    exceptions=(ConnectionError, TimeoutError),
    default_return=None,
    context="network_operations",
    recovery_strategies=[
        create_retry_strategy(max_retries=2, base_delay=1.0),
        create_graceful_degradation_strategy(default_return=None)
    ]
)
async def fetch_market_data(symbol: str) -> Optional[Dict[str, Any]]:
    # Network operation implementation
    pass
```

### Example 2: Database Operations with Fallback

```python
from src.utils.error_handler import (
    create_fallback_strategy,
    handle_specific_errors
)

def primary_database_query():
    # Primary database operation
    pass

def fallback_database_query():
    # Fallback database operation
    pass

@handle_specific_errors(
    error_handlers={
        DatabaseError: (None, "Database error occurred"),
        ConnectionError: (None, "Connection error occurred")
    },
    default_return=None,
    context="database_operations",
    recovery_strategies=[
        create_fallback_strategy([
            primary_database_query,
            fallback_database_query
        ])
    ]
)
async def execute_database_operation():
    # Database operation implementation
    pass
```

### Example 3: Position Division Strategy

```python
from src.utils.error_handler import (
    create_retry_strategy,
    create_graceful_degradation_strategy,
    handle_specific_errors
)

class PositionDivisionStrategy:
    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid position division configuration"),
            AttributeError: (False, "Missing required division parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="position division strategy initialization",
        recovery_strategies=[
            create_retry_strategy(max_retries=2, base_delay=1.0),
            create_graceful_degradation_strategy(default_return=False),
        ],
    )
    async def initialize(self) -> bool:
        # Initialization logic
        pass

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid input data for position division"),
            AttributeError: (None, "Strategy not properly initialized"),
        },
        default_return=None,
        context="position division analysis",
        recovery_strategies=[
            create_retry_strategy(max_retries=1, base_delay=0.5),
            create_graceful_degradation_strategy(default_return=None),
        ],
    )
    async def analyze_position_division(
        self,
        ml_predictions: Dict[str, Any],
        current_positions: List[Dict[str, Any]],
        current_price: float,
        short_term_analysis: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        # Analysis logic
        pass
```

## Best Practices

### 1. Strategy Selection

- **Retry Strategy**: Use for transient failures (network timeouts, temporary unavailability)
- **Fallback Strategy**: Use when you have alternative implementations
- **Graceful Degradation**: Use when you can provide a default response

### 2. Circuit Breaker Configuration

- Set appropriate failure thresholds based on your system's characteristics
- Monitor circuit breaker states in production
- Use different circuit breakers for different types of operations

### 3. Error Context

- Always provide meaningful context in error handlers
- Use specific exception types when possible
- Log recovery attempts and successes

### 4. Type Safety

- Always use type hints for function parameters and return values
- Use generic types for reusable components
- Validate type safety with mypy

### 5. Testing

- Test all recovery strategies individually
- Test circuit breaker state transitions
- Test error scenarios and recovery mechanisms

## Testing

The enhanced error handling system includes comprehensive tests:

```bash
# Run error handling tests
pytest tests/test_enhanced_error_handling.py -v

# Run with coverage
pytest tests/test_enhanced_error_handling.py --cov=src.utils.error_handler -v
```

### Test Categories

1. **Circuit Breaker Tests**: Test state transitions and failure handling
2. **Recovery Strategy Tests**: Test individual recovery strategies
3. **Error Recovery Manager Tests**: Test coordinated recovery
4. **Enhanced Error Handler Tests**: Test decorator functionality
5. **Type Safety Tests**: Verify type hint coverage

## Integration with Existing Code

The enhanced error handling system is designed to integrate seamlessly with existing code:

1. **Backward Compatibility**: Existing error handling decorators continue to work
2. **Gradual Migration**: Add recovery strategies incrementally
3. **Type Safety**: All existing type hints are preserved and enhanced

## Performance Considerations

- Circuit breakers add minimal overhead to successful operations
- Recovery strategies are only executed on failures
- Type hints have zero runtime cost
- Async operations are properly handled

## Monitoring and Observability

- All recovery attempts are logged
- Circuit breaker state changes are tracked
- Error contexts provide debugging information
- Performance metrics can be collected

## Conclusion

The enhanced error handling system provides:

- **Automatic Recovery**: Self-healing operations with multiple strategies
- **Circuit Breaker Protection**: Prevents cascading failures
- **Type Safety**: 100% type hint coverage for better development experience
- **Comprehensive Testing**: Thorough test coverage for all components
- **Production Ready**: Designed for real-world trading system requirements

This system significantly improves the reliability and maintainability of the Ares trading bot while providing clear error handling patterns for developers. 