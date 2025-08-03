# Exception Handling Patterns Guide

This guide addresses the linting issues found in the codebase scan, specifically EM101/EM102 and TRY003 errors related to exception handling.

## Issues Identified

### EM101/EM102: Exception Message Formatting
- **EM101**: Exception must not use a string literal, assign to variable first
- **EM102**: Exception must not use an f-string literal, assign to variable first

### TRY003: Long Exception Messages
- **TRY003**: Avoid specifying long messages outside the exception class

### C901: Function Complexity
- **C901**: Functions too complex (cyclomatic complexity > 10)

### E501: Line Length
- **E501**: Lines too long (> 88 characters)

### BLE001: Blind Exception Catching
- **BLE001**: Do not catch blind exception: `Exception`

## Solutions

### 1. Use the New Exception Handling Utilities

We've created utilities in `src/utils/test_validation_utils.py` and enhanced `src/utils/error_handler.py`:

```python
from src.utils.test_validation_utils import TestValidator, create_validation_message
from src.utils.error_handler import safe_assertion, format_assertion_message

# Instead of:
raise AssertionError(f"Expected {expected}, got {actual}")

# Use:
error_message = format_assertion_message(expected, actual, context="validation")
safe_assertion(False, error_message, context="validation")
```

### 2. Use the TestValidator Class

```python
from src.utils.test_validation_utils import TestValidator

validator = TestValidator()

# Instead of:
if not is_valid:
    raise AssertionError(f"Should be valid: {errors}")

# Use:
validator.assert_validation_result(is_valid, errors, context="test")
```

### 3. Use Decorators for Exception Handling

```python
from src.utils.error_handler import handle_assertion_errors

@handle_assertion_errors(context="test validation")
def test_function():
    # Your test logic here
    pass
```

## Migration Examples

### Before (Problematic Code)
```python
def test_validation():
    is_valid, errors = validate_something()
    if not is_valid:
        raise AssertionError(f"Should be valid: {errors}")  # EM102
    if len(errors) != 0:
        raise AssertionError(f"Expected 0 errors, got {len(errors)}")  # EM102
```

### After (Fixed Code)
```python
from src.utils.test_validation_utils import TestValidator

def test_validation():
    validator = TestValidator()
    is_valid, errors = validate_something()
    
    # Use the utility method
    validator.assert_validation_result(is_valid, errors, context="validation")
    validator.assert_error_count(errors, 0, context="validation")
```

### Before (Problematic Code)
```python
def test_thresholds():
    if data_thresholds["min_data_rows"] != MIN_DATA_ROWS:
        raise AssertionError(
            f"Expected {MIN_DATA_ROWS}, got {data_thresholds['min_data_rows']}",  # EM102
        )
```

### After (Fixed Code)
```python
from src.utils.test_validation_utils import TestValidator

def test_thresholds():
    validator = TestValidator()
    validator.assert_threshold_value(
        data_thresholds["min_data_rows"],
        MIN_DATA_ROWS,
        "min_data_rows",
        context="data_collection"
    )
```

## Function Complexity Reduction

### Before (Complex Function)
```python
def test_error_thresholds():  # C901: too complex
    """Test error threshold validation."""
    print("🧪 Testing error thresholds...")
    
    # Data collection thresholds
    data_thresholds = CRITICAL_ERROR_THRESHOLDS["data_collection"]
    if data_thresholds["min_data_rows"] != MIN_DATA_ROWS:
        raise AssertionError(f"Expected {MIN_DATA_ROWS}, got {data_thresholds['min_data_rows']}")
    # ... many more checks
```

### After (Simplified Function)
```python
def test_error_thresholds():
    """Test error threshold validation."""
    print("🧪 Testing error thresholds...")
    
    test_data_collection_thresholds()
    test_preliminary_optimization_thresholds()
    test_coarse_optimization_thresholds()
    test_main_model_training_thresholds()

def test_data_collection_thresholds():
    """Test data collection thresholds."""
    validator = TestValidator()
    data_thresholds = CRITICAL_ERROR_THRESHOLDS["data_collection"]
    
    validator.assert_threshold_value(
        data_thresholds["min_data_rows"],
        MIN_DATA_ROWS,
        "min_data_rows",
        context="data_collection"
    )
    # ... other checks
```

## Line Length Reduction

### Before (Long Lines)
```python
raise AssertionError(f"Expected {MAX_MISSING_PERCENTAGE}, got {data_thresholds['max_missing_percentage']}")  # E501
```

### After (Short Lines)
```python
error_message = format_assertion_message(
    expected=MAX_MISSING_PERCENTAGE,
    actual=data_thresholds['max_missing_percentage'],
    context="data_collection.max_missing_percentage"
)
safe_assertion(False, error_message)
```

## Blind Exception Handling

### Before (Problematic)
```python
except Exception as e:  # BLE001
    print(f"❌ Test failed: {e}")
```

### After (Specific)
```python
except (ValueError, AttributeError, ImportError) as e:
    print(f"❌ Test failed: {e}")
```

## Implementation Checklist

1. **Import the utilities**:
   ```python
   from src.utils.test_validation_utils import TestValidator
   from src.utils.error_handler import safe_assertion, format_assertion_message
   ```

2. **Replace direct assertions** with utility methods

3. **Break down complex functions** into smaller, focused functions

4. **Use specific exception types** instead of generic `Exception`

5. **Format long lines** using the utility functions

6. **Test the changes** to ensure functionality is preserved

## Benefits

- **Consistent error handling** across the codebase
- **Better error messages** with proper context
- **Reduced complexity** in test functions
- **Compliance with linting rules**
- **Easier maintenance** and debugging

## Migration Strategy

1. Start with the most critical files (high error count)
2. Use the utilities for new code
3. Gradually migrate existing code
4. Run linting checks after each migration
5. Update tests to use the new patterns

This approach ensures consistent, maintainable, and linting-compliant exception handling throughout the codebase. 