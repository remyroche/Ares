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

## Parquet Operations

```python
from src.utils.common_operations import safe_read_parquet, safe_to_parquet, list_parquet_files

# Read parquet file safely
df = safe_read_parquet("data.parquet", columns=["col1", "col2"])

# Write parquet file safely
success = safe_to_parquet(df, "output.parquet", compression="snappy")

# List all parquet files
parquet_files = list_parquet_files("data/", recursive=True)
```

## Hashing and Caching

```python
from src.utils.common_operations import generate_hash, generate_cache_key

# Generate hash for different data types
hash1 = generate_hash("my_string")  # MD5 by default
hash2 = generate_hash(df, algorithm="sha256")  # SHA256 for DataFrame

# Generate cache keys
cache_key = generate_cache_key("features", symbol, timeframe, df.shape[0])
```

## Enhanced DataFrame Operations

```python
from src.utils.common_operations import safe_copy, safe_resample, align_dataframes

# Safe copy
df_copy = safe_copy(df, deep=True)

# Resample time series data
resampled = safe_resample(df, "1H")  # Hourly resampling

# Align multiple DataFrames
df1, df2, df3 = align_dataframes(df1, df2, df3, method="inner")
```

## File System Operations

```python
from src.utils.common_operations import safe_glob, list_files, get_latest_file

# Glob for files
files = safe_glob("data/*.csv", recursive=True)

# List files with pattern
csv_files = list_files("data/", pattern="*.csv")
parquet_files = list_files("data/", suffix=".parquet")

# Get most recent file
latest = get_latest_file("logs/", pattern="*.log")
```

## Data Validation Extensions

```python
from src.utils.common_operations import validate_dataframe_schema, validate_data_quality

# Validate schema
is_valid, errors = validate_dataframe_schema(
    df, 
    required_columns=["open", "high", "low", "close"],
    column_types={"close": np.floating}
)

# Validate data quality
quality = validate_data_quality(df, max_nan_ratio=0.05, check_duplicates=True)
if not quality["is_valid"]:
    print(f"Quality issues: {quality['issues']}")
```

## Progress and Timing

```python
from src.utils.common_operations import timed_operation, format_bytes

@timed_operation("feature_engineering")
def process_features():
    # Your code here
    pass

# Format bytes
size = format_bytes(1024 * 1024 * 512)  # "512.00 MB"
```

## Batch Processing

```python
from src.utils.common_operations import chunked_iterable, parallel_map

# Process in chunks
items = list(range(1000))
for chunk in chunked_iterable(items, chunk_size=100):
    process_chunk(chunk)

# Parallel processing
results = parallel_map(expensive_function, items, max_workers=8)
```

## MLflow Integration

```python
from src.utils.common_operations import safe_log_metric, safe_log_params, safe_log_artifact

# Log metrics safely (won't fail if MLflow not available)
safe_log_metric("accuracy", 0.95)
safe_log_metric("loss", 0.05, step=100)

# Log parameters
safe_log_params({"learning_rate": 0.01, "batch_size": 32})

# Log artifacts
safe_log_artifact("model.pkl")
```