# Migration Guide: Core Decorator System

This guide helps you migrate from the existing decorator system to the new centralized core decorator system.

## Overview

The new core decorator system provides:
- **Unified interface**: All decorators use the same `uniform_wrapper` pattern
- **Type preservation**: Full type hints are preserved through decoration
- **Async/sync support**: Single decorator works for both async and sync functions
- **Composability**: Easy composition with the `compose` function
- **No side effects**: No global state modifications
- **Idempotent**: Decorators can't be accidentally applied twice

## Import Changes

### Old Imports
```python
from src.utils.decorators import validate_data_quality, with_tracing_span
from src.utils.error_handler import handle_errors, handle_network_operations
from src.utils.validation_decorators import validate_dataframe
from src.utils.enhanced_decorators import cache_result
```

### New Imports
```python
# Single import location
from src.core.decorators import (
    validates,
    traced,
    handles_errors,
    retry,
    cached,
    compose,
)

# Or import everything
import src.core.decorators as decorators
```

## Common Migration Patterns

### 1. Error Handling

**Old:**
```python
from src.utils.error_handler import handle_errors

@handle_errors(
    exceptions=(ValueError, TypeError),
    default_return=None,
    context="data_processing"
)
def process_data(data: dict) -> dict:
    return transform(data)
```

**New:**
```python
from src.core.decorators import handles_errors
from src.core.errors import ValidationError

@handles_errors(
    ValueError, TypeError,
    fallback=None,
    map_to=ValidationError
)
def process_data(data: dict) -> dict:
    return transform(data)
```

### 2. Validation

**Old:**
```python
@validate_data_quality(
    required_columns=["id", "value"],
    check_nan=True
)
def analyze_dataframe(df: pd.DataFrame) -> dict:
    return {"mean": df["value"].mean()}
```

**New:**
```python
@validate_dataframe(
    columns=["id", "value"],
    dtypes={"id": int, "value": float},
    min_rows=1
)
def analyze_dataframe(df: pd.DataFrame) -> dict:
    return {"mean": df["value"].mean()}
```

### 3. Retry and Circuit Breaker

**Old:**
```python
@handle_network_operations(max_retries=3)
@circuit_breaker_protection(
    failure_threshold=5,
    recovery_timeout=60
)
def api_call():
    return requests.get(url)
```

**New:**
```python
@retry_with_circuit_breaker(
    max_attempts=3,
    failure_threshold=5,
    recovery_timeout=60
)
def api_call():
    return requests.get(url)

# Or compose them separately
@compose(
    retry(max_attempts=3, delay=1.0),
    circuit_breaker(failure_threshold=5)
)
def api_call():
    return requests.get(url)
```

### 4. Logging and Tracing

**Old:**
```python
@with_tracing_span(
    span_name="process_order",
    log_args=True
)
def process_order(order_id: str) -> dict:
    return {"status": "processed"}
```

**New:**
```python
@compose(
    traced(span_name="process_order", record_args=True),
    log_call(level="INFO", log_args=True)
)
def process_order(order_id: str) -> dict:
    return {"status": "processed"}
```

### 5. Caching

**Old:**
```python
@cache_result(ttl=300)
@intelligent_caching(
    cache_key="user_data",
    invalidate_on=["user_update"]
)
def get_user_data(user_id: str) -> dict:
    return fetch_from_db(user_id)
```

**New:**
```python
@cached(
    policy=CachePolicy.CROSS_REQUEST,
    ttl=300,
    key_func=lambda f, args, kw: f"user_data:{args[0]}"
)
def get_user_data(user_id: str) -> dict:
    return fetch_from_db(user_id)
```

### 6. Authentication/Authorization

**Old:**
```python
# Various custom implementations
@require_auth
@check_permission("admin")
def admin_operation():
    pass
```

**New:**
```python
@authenticated()
@requires_role("admin")
def admin_operation():
    pass

# Or with specific permissions
@requires_permission("users.delete", "admin.all", require_all=False)
def delete_user(user_id: str):
    pass
```

## Decorator Composition

The new system encourages explicit composition:

```python
# Define reusable decorator combinations
api_endpoint = compose(
    authenticated(),
    validates(),
    log_call(level="INFO"),
    handles_errors(fallback={"error": "Internal error"}),
    traced(kind=SpanKind.SERVER),
)

# Use the composition
@api_endpoint
def get_user_profile(user_id: str) -> dict:
    return fetch_profile(user_id)

@api_endpoint
def update_user_profile(user_id: str, data: dict) -> dict:
    return update_profile(user_id, data)
```

## Custom Decorators

Creating custom decorators that fit the system:

```python
from src.core.decorators import uniform_wrapper

def my_custom_decorator(*, option: str = "default"):
    def sync_handler(func, *args, **kwargs):
        # Pre-processing
        print(f"Starting {func.__name__} with {option}")
        result = func(*args, **kwargs)
        # Post-processing
        print(f"Completed {func.__name__}")
        return result
    
    async def async_handler(func, *args, **kwargs):
        # Pre-processing
        print(f"Starting {func.__name__} with {option}")
        result = await func(*args, **kwargs)
        # Post-processing
        print(f"Completed {func.__name__}")
        return result
    
    return uniform_wrapper(
        f"my_custom_decorator({option})",
        sync_handler,
        async_handler
    )
```

## Error Mapping

Register custom exception mappings:

```python
from src.core.errors import register_exception_mapping, ValidationError

# Map pandas errors
register_exception_mapping(
    pd.errors.EmptyDataError,
    lambda e: ValidationError("Empty DataFrame provided")
)

# Map custom errors
class MyCustomError(Exception):
    pass

register_exception_mapping(
    MyCustomError,
    lambda e: BusinessRuleError(str(e))
)
```

## Testing

The new decorators are easier to test:

```python
import pytest
from src.core.decorators import validates, handles_errors

def test_validation():
    @validates(strict=True)
    def add(a: int, b: int) -> int:
        return a + b
    
    # Should work
    assert add(1, 2) == 3
    
    # Should raise ValidationError
    with pytest.raises(ValidationError):
        add("1", "2")  # Wrong types

def test_error_handling():
    @handles_errors(ValueError, fallback=-1)
    def parse_int(s: str) -> int:
        return int(s)
    
    assert parse_int("123") == 123
    assert parse_int("abc") == -1  # Fallback value
```

## Performance Considerations

1. **Decorator Order**: Place caching decorators before expensive operations
2. **Selective Application**: Not every function needs every decorator
3. **Policy Selection**: Use `PER_REQUEST` caching for request-scoped data

```python
# Good: Cache before expensive validation
@cached(policy=CachePolicy.CROSS_REQUEST)
@validates()
@traced()
def expensive_operation(data: dict) -> dict:
    return process(data)

# Less optimal: Validation happens before cache check
@validates()
@cached(policy=CachePolicy.CROSS_REQUEST)
@traced()
def expensive_operation(data: dict) -> dict:
    return process(data)
```

## Gradual Migration Strategy

1. **Phase 1**: Update imports in new code
2. **Phase 2**: Migrate critical paths with high error rates
3. **Phase 3**: Update remaining decorators file by file
4. **Phase 4**: Remove old decorator imports

## Compatibility Layer

For gradual migration, create adapters:

```python
# src/utils/decorator_compat.py
from src.core.decorators import validates, handles_errors

# Adapter for old decorator signature
def validate_data_quality(**old_kwargs):
    # Map old arguments to new
    return validates(
        strict=old_kwargs.get("strict_mode", True),
        coerce=old_kwargs.get("coerce_types", False)
    )

# Adapter for old error handler
def handle_errors_compat(exceptions=None, default_return=None, **kwargs):
    return handles_errors(
        *(exceptions or (Exception,)),
        fallback=default_return
    )
```

## Common Issues and Solutions

### Issue: Decorator doesn't preserve async behavior
**Solution**: Ensure you're using the new decorators which handle async automatically

### Issue: Type hints are lost
**Solution**: The new system preserves types. Update your imports.

### Issue: Global state conflicts
**Solution**: New decorators use context variables instead of global state

### Issue: Performance regression
**Solution**: Check decorator order and caching policies

## Getting Help

- Check `src/core/examples/decorator_usage.py` for examples
- Run tests to verify migrations: `pytest tests/core/test_decorators.py`
- Use type checking: `mypy src/` to catch issues early