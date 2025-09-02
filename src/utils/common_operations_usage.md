# Common Operations Module Usage Guide

This module provides commonly used operations that were identified as undefined in the codebase analysis.

## Installation

The module is already available at `src.utils.common_operations`. To use it in your code:

```python
from src.utils.common_operations import (
    get_current_datetime, safe_fillna, safe_mean,
    ensure_directory, safe_sleep, create_argument_parser
)
```

## DateTime Operations

```python
from src.utils.common_operations import get_current_datetime, format_datetime, parse_datetime

# Get current datetime
now = get_current_datetime()

# Format datetime
date_str = format_datetime(now, "%Y-%m-%d")

# Parse datetime
dt = parse_datetime("2024-01-01 12:00:00")
```

## DataFrame Operations

```python
from src.utils.common_operations import create_empty_dataframe, safe_fillna, safe_rolling

# Create empty DataFrame
df = create_empty_dataframe(['col1', 'col2', 'col3'])

# Fill NaN values safely
df_filled = safe_fillna(df, 0)

# Create rolling window
rolling = safe_rolling(df, window=5)
```

## Numeric Operations

```python
from src.utils.common_operations import safe_mean, safe_std

# Calculate mean safely
values = [1, 2, 3, None, 5]
mean_val = safe_mean(values)  # Handles None/NaN

# Calculate standard deviation
std_val = safe_std(values)
```

## File Operations

```python
from src.utils.common_operations import ensure_directory, safe_file_exists, safe_json_dump, safe_json_load

# Ensure directory exists
output_dir = ensure_directory("/workspace/output")

# Check file existence
if safe_file_exists("/path/to/file.txt"):
    print("File exists")

# Save/load JSON
data = {"key": "value"}
safe_json_dump(data, "output.json", indent=2)
loaded_data = safe_json_load("output.json")
```

## Async Operations

```python
from src.utils.common_operations import safe_sleep, safe_gather, create_async_task

async def main():
    # Async sleep
    await safe_sleep(1.0)
    
    # Gather multiple coroutines
    results = await safe_gather(
        fetch_data(),
        process_data(),
        save_results()
    )
    
    # Create async task
    task = create_async_task(background_job())
```

## Collection Operations

```python
from src.utils.common_operations import safe_append, safe_dict_get, safe_dict_items

# Safe list operations
my_list = []
my_list = safe_append(my_list, "item")

# Safe dictionary operations
my_dict = {"key": "value"}
value = safe_dict_get(my_dict, "key", default="default")
items = safe_dict_items(my_dict)
```

## String Operations

```python
from src.utils.common_operations import safe_lower, safe_upper, safe_join

# Safe string operations
text = "Hello World"
lower_text = safe_lower(text)
upper_text = safe_upper(text)

# Safe join
items = ["a", "b", None, "d"]
joined = safe_join(", ", items)  # "a, b, , d"
```

## Logging Operations

```python
from src.utils.common_operations import get_logger, setup_basic_logging

# Setup logging
setup_basic_logging()

# Get logger
logger = get_logger(__name__)
logger.info("This is a log message")
```

## Argument Parsing

```python
from src.utils.common_operations import create_argument_parser, add_common_arguments

# Create parser
parser = create_argument_parser("My Script Description")

# Add common arguments
add_common_arguments(parser)

# Add custom arguments
parser.add_argument('--custom', type=str, help='Custom argument')

# Parse arguments
args = parser.parse_args()
```

## Type Conversion

```python
from src.utils.common_operations import safe_float, safe_int

# Safe conversions
float_val = safe_float("3.14", default=0.0)
int_val = safe_int("42", default=0)

# Handles errors gracefully
bad_float = safe_float("not a number", default=-1.0)  # Returns -1.0
```

## Validation Utilities

```python
from src.utils.common_operations import validate_dataframe, validate_numeric_range

# Validate DataFrame
if validate_dataframe(df, required_columns=['price', 'volume']):
    print("DataFrame is valid")

# Validate numeric range
if validate_numeric_range(value, min_val=0, max_val=100):
    print("Value is in range")
```

## Memory Optimization

```python
from src.utils.common_operations import optimize_dataframe_dtypes

# Optimize DataFrame memory usage
df_optimized = optimize_dataframe_dtypes(df)
```

## Exception Handling

```python
from src.utils.common_operations import safe_exception_handler

@safe_exception_handler
def risky_operation():
    # This will log exceptions and return None on error
    return 1 / 0
```

## Migration Guide

To migrate existing code to use this module:

1. **Replace undefined function calls:**
   ```python
   # Before
   now = datetime.datetime.now()  # If datetime not imported
   
   # After
   from src.utils.common_operations import get_current_datetime
   now = get_current_datetime()
   ```

2. **Use safe operations:**
   ```python
   # Before
   mean_val = np.mean(values)  # May fail if values is empty
   
   # After
   from src.utils.common_operations import safe_mean
   mean_val = safe_mean(values)  # Returns NaN if empty
   ```

3. **Simplify common patterns:**
   ```python
   # Before
   parser = argparse.ArgumentParser(description='Script')
   parser.add_argument('--verbose', '-v', action='store_true')
   parser.add_argument('--config', type=str, default='config.json')
   
   # After
   from src.utils.common_operations import create_argument_parser, add_common_arguments
   parser = create_argument_parser('Script')
   add_common_arguments(parser)
   ```

This module helps maintain consistency across the codebase and reduces errors from undefined functions.