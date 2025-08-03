# Error Handling Guide

This document provides guidelines for implementing consistent error handling patterns across the Ares trading bot codebase using the `handle_errors` decorator and related utilities.

## Overview

The Ares trading bot uses a centralized error handling system with the `handle_errors` decorator to ensure consistent error handling patterns across all components.

## Core Error Handling Decorators

### 1. `handle_errors` - General Purpose Error Handling

**Usage**: General error handling for any function or method.

```python
from src.utils.error_handler import handle_errors

@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="function_name"
)
async def my_function():
    # Your code here
    return result
```

**Parameters**:
- `exceptions`: Tuple of exception types to catch (default: `(Exception,)`)
- `default_return`: Value to return on error (default: `None`)
- `log_errors`: Whether to log errors (default: `True`)
- `recovery_strategy`: Optional recovery function (default: `None`)
- `context`: Additional context for logging (default: `""`)

### 2. `handle_network_operations` - Network-Specific Error Handling

**Usage**: For network operations with retry logic.

```python
from src.utils.error_handler import handle_network_operations

@handle_network_operations(
    max_retries=3,
    default_return=None,
    context="api_call"
)
async def fetch_data():
    # Network operation here
    return data
```

**Parameters**:
- `max_retries`: Number of retry attempts (default: `3`)
- `default_return`: Value to return on error (default: `None`)
- `log_errors`: Whether to log errors (default: `True`)
- `context`: Additional context for logging (default: `""`)

### 3. `handle_data_processing_errors` - Data Processing Error Handling

**Usage**: For data processing operations that might encounter NaN/inf values.

```python
from src.utils.error_handler import handle_data_processing_errors

@handle_data_processing_errors(
    default_return=0.0,
    context="calculation"
)
def calculate_metrics(data):
    # Data processing here
    return result
```

### 4. `handle_file_operations` - File Operation Error Handling

**Usage**: For file operations with comprehensive error handling.

```python
from src.utils.error_handler import handle_file_operations

@handle_file_operations(
    default_return=None,
    context="file_operation"
)
def read_config_file():
    # File operation here
    return data
```

### 5. `handle_type_conversions` - Type Conversion Error Handling

**Usage**: For type conversion operations that might fail.

```python
from src.utils.error_handler import handle_type_conversions

@handle_type_conversions(
    default_return=0,
    context="type_conversion"
)
def convert_to_int(value):
    return int(value)
```

## Error Handling Patterns

### Pattern 1: Basic Error Handling

```python
@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="basic_operation"
)
async def basic_operation():
    # Your code here
    return True
```

### Pattern 2: Specific Exception Handling

```python
@handle_errors(
    exceptions=(ValueError, TypeError, KeyError),
    default_return={},
    context="data_processing"
)
def process_data(data):
    # Process data
    return result
```

### Pattern 3: Network Operations with Retries

```python
@handle_network_operations(
    max_retries=3,
    default_return=None,
    context="api_request"
)
async def fetch_market_data():
    # Network request here
    return data
```

### Pattern 4: Data Processing with NaN Handling

```python
@handle_data_processing_errors(
    default_return=0.0,
    context="calculation"
)
def calculate_ratio(numerator, denominator):
    return numerator / denominator
```

### Pattern 5: File Operations

```python
@handle_file_operations(
    default_return=None,
    context="config_loading"
)
def load_config_file(path):
    with open(path, 'r') as f:
        return json.load(f)
```

## Context Manager Usage

For code blocks that need error handling:

```python
from src.utils.error_handler import error_context

with error_context("database_operation", default_return={}):
    result = database_operation()
    return result
```

## Recovery Strategies

You can provide custom recovery strategies:

```python
def recovery_strategy(*args, **kwargs):
    # Custom recovery logic
    return fallback_value

@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    recovery_strategy=recovery_strategy,
    context="operation_with_recovery"
)
async def operation_with_recovery():
    # Your code here
    return result
```

## Error Recovery Utilities

### Safe Access Utilities

```python
from src.utils.error_handler import ErrorRecoveryStrategies

# Safe dictionary access
value = ErrorRecoveryStrategies.safe_dict_access(data, "key", default_value)

# Safe DataFrame access
column = ErrorRecoveryStrategies.safe_dataframe_access(df, "column_name", default_value)

# Safe numeric operations
result = ErrorRecoveryStrategies.safe_numeric_operation(
    operation_function, 
    *args, 
    default=0.0
)

# Safe type conversion
value = ErrorRecoveryStrategies.safe_type_conversion(input_value, int, default=0)
```

### Safe Operation Wrappers

```python
from src.utils.error_handler import (
    safe_file_operation,
    safe_network_operation,
    safe_database_operation,
    safe_dataframe_operation,
    safe_numeric_operation
)

# Safe file operations
result = safe_file_operation(open, "file.txt", "r")

# Safe network operations
result = await safe_network_operation(fetch_data, max_retries=3)

# Safe database operations
result = safe_database_operation(db_query, *args)

# Safe DataFrame operations
result = safe_dataframe_operation(df.calculate, *args)

# Safe numeric operations
result = safe_numeric_operation(division, a, b)
```

## Best Practices

### 1. Always Use Context

Provide meaningful context for better error tracking:

```python
# Good
@handle_errors(context="market_data_fetch")
async def fetch_market_data():
    pass

# Bad
@handle_errors()
async def fetch_market_data():
    pass
```

### 2. Choose Appropriate Default Returns

```python
# For functions that return data
@handle_errors(default_return={})

# For functions that return success status
@handle_errors(default_return=False)

# For functions that return numeric values
@handle_errors(default_return=0.0)

# For functions that return None
@handle_errors(default_return=None)
```

### 3. Handle Specific Exceptions

```python
# Good - handle specific exceptions
@handle_errors(
    exceptions=(ValueError, TypeError),
    default_return=0.0,
    context="numeric_calculation"
)
def calculate_ratio(a, b):
    return a / b

# Bad - catch all exceptions
@handle_errors()
def calculate_ratio(a, b):
    return a / b
```

### 4. Use Network Operations for API Calls

```python
# Good - use network operations decorator
@handle_network_operations(
    max_retries=3,
    default_return=None,
    context="binance_api_call"
)
async def fetch_klines():
    return await exchange.get_klines()

# Bad - use general error handling for network calls
@handle_errors()
async def fetch_klines():
    return await exchange.get_klines()
```

### 5. Use Data Processing Decorator for Calculations

```python
# Good - use data processing decorator
@handle_data_processing_errors(
    default_return=0.0,
    context="risk_calculation"
)
def calculate_risk_metrics(data):
    return data.mean() / data.std()

# Bad - use general error handling for data processing
@handle_errors()
def calculate_risk_metrics(data):
    return data.mean() / data.std()
```

## Migration Guidelines

### Step 1: Identify Current Error Handling

Look for patterns like:
```python
try:
    # code
except Exception as e:
    logger.error(f"Error: {e}")
    return default_value
```

### Step 2: Replace with Decorator

```python
@handle_errors(
    exceptions=(Exception,),
    default_return=default_value,
    context="function_name"
)
def function_name():
    # code
    return result
```

### Step 3: Update Imports

```python
from src.utils.error_handler import handle_errors
```

### Step 4: Test the Changes

Ensure the function behaves the same way with the decorator.

## Common Anti-Patterns to Avoid

### ❌ Don't: Nested Try-Catch Blocks

```python
# Bad
def bad_function():
    try:
        try:
            result = risky_operation()
        except ValueError:
            return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return None
```

### ✅ Do: Use Decorators

```python
# Good
@handle_errors(
    exceptions=(ValueError,),
    default_return=0,
    context="risky_operation"
)
@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="bad_function"
)
def good_function():
    result = risky_operation()
    return result
```

### ❌ Don't: Ignore Errors

```python
# Bad
def bad_function():
    try:
        return risky_operation()
    except Exception:
        pass  # Silent failure
```

### ✅ Do: Handle Errors Properly

```python
# Good
@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="risky_operation"
)
def good_function():
    return risky_operation()
```

### ❌ Don't: Inconsistent Error Handling

```python
# Bad - inconsistent patterns
def function1():
    try:
        return operation()
    except Exception as e:
        logger.error(f"Error: {e}")
        return None

def function2():
    try:
        return operation()
    except Exception:
        return False
```

### ✅ Do: Use Consistent Patterns

```python
# Good - consistent patterns
@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="operation"
)
def function1():
    return operation()

@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="operation"
)
def function2():
    return operation()
```

## Testing Error Handling

### Test Normal Operation

```python
def test_normal_operation():
    result = decorated_function()
    assert result == expected_value
```

### Test Error Conditions

```python
def test_error_condition():
    # Mock a failure condition
    with patch('some_module.risky_operation', side_effect=Exception("Test error")):
        result = decorated_function()
        assert result == default_return_value
```

### Test Recovery Strategies

```python
def test_recovery_strategy():
    # Mock a failure and recovery
    with patch('some_module.risky_operation', side_effect=Exception("Test error")):
        result = decorated_function_with_recovery()
        assert result == recovery_value
```

## Conclusion

Using the `handle_errors` decorator and related utilities ensures:

1. **Consistency**: All error handling follows the same patterns
2. **Maintainability**: Centralized error handling logic
3. **Reliability**: Proper error logging and recovery
4. **Debugging**: Better error tracking with context
5. **Performance**: Efficient error handling without code duplication

Follow these guidelines to implement consistent error handling across the Ares trading bot codebase. 