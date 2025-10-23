# VectorBT Production Integration

This module provides a production-ready VectorBT integration for the Ares trading system. VectorBT is a critical dependency that provides high-performance vectorized backtesting and financial analysis capabilities.

## Features

- **Production-Ready**: Fast-fail behavior if VectorBT is not available
- **Performance Monitoring**: Built-in performance monitoring and optimization
- **Memory Management**: Optimized memory usage for large datasets
- **Error Handling**: Comprehensive error handling and validation
- **Configuration**: Centralized configuration management
- **Testing**: Comprehensive test suite

## Installation

### Automatic Installation

Run the installation script to automatically install and configure VectorBT:

```bash
python src/vectorbt/install_vectorbt.py
```

### Manual Installation

Install VectorBT and dependencies manually:

```bash
pip install vectorbt>=0.25.0
pip install numpy>=1.21.0 pandas>=1.3.0 scipy>=1.7.0 numba>=0.56.0
```

## Usage

### Basic Usage

```python
from src.vectorbt import (
    rolling_mean, rolling_std, rolling_apply,
    Portfolio, PortfolioFactory, Returns,
    RSI, MACD, BBANDS, ATR
)

# Rolling operations
data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
sma = rolling_mean(data, window=5)
std = rolling_std(data, window=5)

# Technical indicators
rsi = RSI.run(data)
macd = MACD.run(data)

# Portfolio creation
returns = data.pct_change().dropna()
portfolio = PortfolioFactory.from_returns(returns)
```

### Performance Monitoring

```python
from src.vectorbt import monitor_operation, get_performance_monitor

# Monitor operations
with monitor_operation("my_operation", data_size=len(data)):
    result = rolling_mean(data, window=10)

# Get performance statistics
monitor = get_performance_monitor()
stats = monitor.get_operation_stats("my_operation")
print(f"Average execution time: {stats['avg_execution_time']:.4f}s")
```

### Configuration

```python
from src.vectorbt import configure_vectorbt, PRODUCTION_CONFIG

# Configure for production
configure_vectorbt(PRODUCTION_CONFIG)

# Or use custom configuration
from src.vectorbt import VectorBTConfig
config = VectorBTConfig(
    memory_efficient=True,
    parallel=True,
    validate_data=True
)
configure_vectorbt(config)
```

## API Reference

### Core Functions

- `rolling_mean(data, window)`: Rolling mean calculation
- `rolling_std(data, window)`: Rolling standard deviation
- `rolling_apply(data, func, window)`: Rolling function application
- `rolling_corr(data1, data2, window)`: Rolling correlation
- `scale(data)`: Data scaling (z-score normalization)
- `rank(data)`: Data ranking
- `zscore(data)`: Z-score calculation
- `winsorize(data, limits)`: Data winsorization
- `clip(data, lower, upper)`: Data clipping

### Portfolio Operations

- `PortfolioFactory.from_returns(returns)`: Create portfolio from returns
- `PortfolioFactory.from_signals(close, entries, exits)`: Create portfolio from signals
- `ProductionPortfolioFactory`: Enhanced portfolio factory with validation

### Technical Indicators

- `RSI.run(data)`: Relative Strength Index
- `MACD.run(data)`: MACD indicator
- `BBANDS.run(data)`: Bollinger Bands
- `ATR.run(high, low, close)`: Average True Range
- `STOCH.run(high, low, close)`: Stochastic Oscillator

### Performance Monitoring

- `monitor_operation(name, data_size)`: Context manager for operation monitoring
- `profile_operation(func)`: Decorator for function profiling
- `get_performance_monitor()`: Get performance monitor instance
- `get_memory_usage()`: Get current memory usage

### Configuration

- `VectorBTConfig`: Configuration class
- `configure_vectorbt(config)`: Configure VectorBT
- `get_vectorbt_config()`: Get current configuration
- `PRODUCTION_CONFIG`: Production-ready configuration
- `DEVELOPMENT_CONFIG`: Development configuration

## Error Handling

The module provides comprehensive error handling with specific exception types:

- `VectorBTError`: Base exception for VectorBT errors
- `VectorBTConfigurationError`: Configuration-related errors
- `VectorBTDataError`: Data validation errors
- `VectorBTComputationError`: Computation errors

## Testing

Run the test suite:

```bash
python -m pytest src/vectorbt/test_vectorbt_integration.py -v
```

Or run the installation script which includes tests:

```bash
python src/vectorbt/install_vectorbt.py
```

## Performance Optimization

The module includes several performance optimizations:

1. **Memory Management**: Automatic memory optimization for DataFrames
2. **Chunked Processing**: Process large datasets in chunks
3. **Parallel Processing**: Utilize multiple CPU cores when available
4. **Caching**: Intelligent caching of expensive operations
5. **Monitoring**: Real-time performance monitoring

## Memory Usage

Monitor memory usage:

```python
from src.vectorbt import get_memory_usage, MemoryOptimizer

# Get current memory usage
memory_info = get_memory_usage()
print(f"Process memory: {memory_info['process_memory_mb']:.2f} MB")

# Optimize DataFrame memory
optimizer = MemoryOptimizer()
optimized_df = optimizer.optimize_dataframe_memory(df)
```

## Configuration Options

Environment variables for configuration:

- `VECTORBT_MEMORY_EFFICIENT`: Enable memory-efficient mode (default: true)
- `VECTORBT_PARALLEL`: Enable parallel processing (default: true)
- `VECTORBT_NUM_THREADS`: Number of threads for parallel processing
- `VECTORBT_MAX_MEMORY`: Maximum memory usage (0.0-1.0)
- `VECTORBT_CHUNK_SIZE`: Chunk size for large operations
- `VECTORBT_DEBUG`: Enable debug mode
- `VECTORBT_PRODUCTION`: Enable production mode

## Troubleshooting

### Common Issues

1. **ImportError**: VectorBT not installed
   - Solution: Run `python src/vectorbt/install_vectorbt.py`

2. **Memory Issues**: Out of memory errors
   - Solution: Enable memory-efficient mode and reduce chunk size

3. **Performance Issues**: Slow operations
   - Solution: Enable parallel processing and check system resources

4. **Validation Errors**: Data validation failures
   - Solution: Check data format and enable strict validation

### Debug Mode

Enable debug mode for detailed logging:

```python
import os
os.environ['VECTORBT_DEBUG'] = 'true'

from src.vectorbt import configure_vectorbt, DEVELOPMENT_CONFIG
configure_vectorbt(DEVELOPMENT_CONFIG)
```

## Dependencies

- Python 3.8+
- VectorBT >= 0.25.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- SciPy >= 1.7.0
- Numba >= 0.56.0
- psutil >= 5.8.0

## License

This module is part of the Ares trading system and follows the same license terms.