# Recommended Additions to common_operations.py

Based on the analysis of training steps in enhanced_training_manager, here are the operations that should be added to `common_operations.py` to improve compatibility and reduce code duplication:

## 1. Parquet Operations (High Priority)

```python
def safe_read_parquet(file_path: Union[str, Path], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Safely read parquet file with error handling."""
    try:
        return pd.read_parquet(file_path, columns=columns)
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to read parquet file {file_path}: {e}")
        return pd.DataFrame()

def safe_to_parquet(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> bool:
    """Safely write DataFrame to parquet with error handling."""
    try:
        df.to_parquet(file_path, **kwargs)
        return True
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to write parquet file {file_path}: {e}")
        return False

def list_parquet_files(directory: Union[str, Path], recursive: bool = True) -> List[Path]:
    """List all parquet files in a directory."""
    directory = Path(directory)
    if recursive:
        return list(directory.rglob("*.parquet"))
    return list(directory.glob("*.parquet"))
```

## 2. Hashing and Cache Key Generation

```python
def generate_hash(data: Union[str, bytes, pd.DataFrame], algorithm: str = "md5") -> str:
    """Generate hash for data with support for different types."""
    import hashlib
    
    if isinstance(data, pd.DataFrame):
        data = pd.util.hash_pandas_object(data).values.tobytes()
    elif isinstance(data, str):
        data = data.encode()
    
    if algorithm == "md5":
        return hashlib.md5(data).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(data).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

def generate_cache_key(prefix: str, *args, max_length: int = 16) -> str:
    """Generate a cache key from multiple inputs."""
    combined = f"{prefix}_" + "_".join(str(arg) for arg in args)
    hash_val = generate_hash(combined, "sha256")
    return hash_val[:max_length]
```

## 3. DataFrame Copy Operations

```python
def safe_copy(df: pd.DataFrame, deep: bool = True) -> pd.DataFrame:
    """Safely copy a DataFrame with error handling."""
    try:
        return df.copy(deep=deep)
    except Exception:
        return df

def safe_deepcopy(obj: Any) -> Any:
    """Safely deep copy an object."""
    from copy import deepcopy
    try:
        return deepcopy(obj)
    except Exception:
        return obj
```

## 4. File System Operations

```python
def safe_glob(pattern: str, recursive: bool = False) -> List[Path]:
    """Safely glob for files with error handling."""
    import glob
    try:
        files = glob.glob(pattern, recursive=recursive)
        return [Path(f) for f in files]
    except Exception:
        return []

def list_files(directory: Union[str, Path], pattern: str = "*", 
              suffix: Optional[str] = None) -> List[Path]:
    """List files in directory with optional pattern/suffix filter."""
    directory = Path(directory)
    if not directory.exists():
        return []
    
    if suffix:
        return [f for f in directory.iterdir() if f.is_file() and f.suffix == suffix]
    
    return [f for f in directory.glob(pattern) if f.is_file()]

def get_latest_file(directory: Union[str, Path], pattern: str = "*") -> Optional[Path]:
    """Get the most recently modified file matching pattern."""
    files = list_files(directory, pattern)
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)
```

## 5. Data Validation Extensions

```python
def validate_dataframe_schema(df: pd.DataFrame, 
                            required_columns: List[str],
                            column_types: Optional[Dict[str, type]] = None) -> tuple[bool, List[str]]:
    """Validate DataFrame schema including column types."""
    errors = []
    
    # Check required columns
    missing = set(required_columns) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")
    
    # Check column types if specified
    if column_types:
        for col, expected_type in column_types.items():
            if col in df.columns:
                actual_type = df[col].dtype
                if not np.issubdtype(actual_type, expected_type):
                    errors.append(f"Column {col} has type {actual_type}, expected {expected_type}")
    
    return len(errors) == 0, errors

def validate_data_quality(df: pd.DataFrame, 
                        max_nan_ratio: float = 0.1,
                        check_duplicates: bool = True) -> dict[str, Any]:
    """Comprehensive data quality validation."""
    quality_report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "issues": []
    }
    
    # Check NaN ratio
    nan_ratios = df.isna().sum() / len(df)
    high_nan_cols = nan_ratios[nan_ratios > max_nan_ratio]
    if not high_nan_cols.empty:
        quality_report["issues"].append({
            "type": "high_nan_ratio",
            "columns": high_nan_cols.to_dict()
        })
    
    # Check duplicates
    if check_duplicates:
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            quality_report["issues"].append({
                "type": "duplicates",
                "count": duplicates
            })
    
    quality_report["is_valid"] = len(quality_report["issues"]) == 0
    return quality_report
```

## 6. Time Series Operations

```python
def safe_resample(df: pd.DataFrame, rule: str, 
                 agg_dict: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Safely resample time series data."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    if agg_dict is None:
        # Default aggregations for common columns
        agg_dict = {
            "close": "last",
            "open": "first", 
            "high": "max",
            "low": "min",
            "volume": "sum"
        }
        # Only use columns that exist
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    return df.resample(rule).agg(agg_dict)

def align_dataframes(*dfs: pd.DataFrame, method: str = "inner") -> List[pd.DataFrame]:
    """Align multiple DataFrames by index."""
    if len(dfs) < 2:
        return list(dfs)
    
    # Find common index range
    if method == "inner":
        start = max(df.index.min() for df in dfs)
        end = min(df.index.max() for df in dfs)
        aligned = [df.loc[start:end] for df in dfs]
    else:  # outer
        aligned = list(dfs)
    
    return aligned
```

## 7. Collection Utilities

```python
def safe_defaultdict(default_factory: Callable) -> defaultdict:
    """Create a defaultdict safely."""
    from collections import defaultdict
    return defaultdict(default_factory)

def safe_counter(items: Optional[List[Any]] = None) -> Counter:
    """Create a Counter safely."""
    from collections import Counter
    return Counter(items or [])

def safe_deque(items: Optional[List[Any]] = None, maxlen: Optional[int] = None) -> deque:
    """Create a deque safely."""
    from collections import deque
    return deque(items or [], maxlen=maxlen)
```

## 8. Progress and Timing Utilities

```python
def timed_operation(operation_name: str):
    """Decorator to time operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            logger = get_logger(func.__module__)
            logger.info(f"Starting {operation_name}...")
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                logger.info(f"Completed {operation_name} in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"Failed {operation_name} after {elapsed:.2f}s: {e}")
                raise
        return wrapper
    return decorator

def format_bytes(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"
```

## 9. Batch Processing Utilities

```python
def chunked_iterable(iterable: List[Any], chunk_size: int) -> Generator[List[Any], None, None]:
    """Split an iterable into chunks."""
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]

def parallel_map(func: Callable, items: List[Any], 
                max_workers: Optional[int] = None) -> List[Any]:
    """Apply function to items in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, items))
```

## 10. MLflow Integration Helpers

```python
def safe_log_metric(key: str, value: float, step: Optional[int] = None) -> None:
    """Safely log metric to MLflow if available."""
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_metric(key, value, step)
    except Exception:
        pass

def safe_log_params(params: Dict[str, Any]) -> None:
    """Safely log parameters to MLflow if available."""
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_params(params)
    except Exception:
        pass

def safe_log_artifact(file_path: Union[str, Path]) -> None:
    """Safely log artifact to MLflow if available."""
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_artifact(str(file_path))
    except Exception:
        pass
```

## Integration Priority

### High Priority (Most Used):
1. Parquet operations
2. Hashing and cache key generation
3. File system operations
4. DataFrame copy operations

### Medium Priority (Frequently Used):
5. Data validation extensions
6. Time series operations
7. Progress and timing utilities

### Low Priority (Nice to Have):
8. Collection utilities
9. Batch processing utilities
10. MLflow integration helpers

## Usage Example

```python
from src.utils.common_operations import (
    safe_read_parquet, safe_to_parquet, generate_cache_key,
    validate_dataframe_schema, timed_operation, safe_copy
)

@timed_operation("feature_engineering")
def process_features(symbol: str, data_path: str):
    # Read data safely
    df = safe_read_parquet(data_path)
    
    # Validate schema
    is_valid, errors = validate_dataframe_schema(
        df, 
        required_columns=["open", "high", "low", "close", "volume"],
        column_types={"close": np.floating}
    )
    
    if not is_valid:
        raise ValueError(f"Invalid data: {errors}")
    
    # Generate cache key
    cache_key = generate_cache_key("features", symbol, df.shape[0])
    
    # Process features
    features = safe_copy(df)
    # ... feature engineering ...
    
    # Save results
    safe_to_parquet(features, f"cache/{cache_key}_features.parquet")
```

These additions would significantly improve code consistency, error handling, and compatibility across all training steps in the enhanced_training_manager.