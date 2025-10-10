# Standardized Exchange Structure

This document describes the new standardized structure for exchange implementations that uses shared processing logic across all exchanges.

## Overview

The new structure follows a consistent pattern where:
- **Shared logic** is centralized in `exchanges/shared/`
- **Exchange-specific code** is minimal and focused only on API adapters
- **Data processing** is standardized across all exchanges
- **Quality validation** is consistent and exchange-agnostic

## Directory Structure

```
exchanges/
├── shared/                           # Shared processing logic
│   ├── klines_downloading_processing.py    # Main processing pipeline
│   ├── exchange_data_standardizer.py       # Data standardization
│   └── __init__.py
├── binance/                          # Binance-specific code
│   ├── klines_adapter.py            # Minimal API adapter only
│   └── __init__.py
├── bingx/                           # BingX-specific code
│   ├── klines_adapter.py            # Minimal API adapter only
│   └── __init__.py
├── mexc/                            # MEXC-specific code
│   ├── klines_adapter.py            # Minimal API adapter only
│   └── __init__.py
├── okx/                             # OKX-specific code
│   ├── klines_adapter.py            # Minimal API adapter only
│   └── __init__.py
├── gateio/                          # GateIO-specific code
│   ├── klines_adapter.py            # Minimal API adapter only
│   └── __init__.py
├── phemex/                          # Phemex-specific code
│   ├── klines_adapter.py            # Minimal API adapter only
│   └── __init__.py
└── examples/
    └── exchange_adapters_example.py # Usage examples
```

## Key Components

### 1. Shared Processing Pipeline (`exchanges/shared/klines_downloading_processing.py`)

The `KlinesDataProcessingPipeline` class provides:
- **Exchange-agnostic data processing**
- **Standardized data format conversion**
- **Gap detection and filling**
- **Duplicate handling**
- **Data quality validation**
- **Consolidated file creation**

### 2. Data Standardizer (`exchanges/shared/exchange_data_standardizer.py`)

The `ExchangeDataStandardizer` class provides:
- **Unified data format** across all exchanges
- **Exchange-specific column mapping**
- **Data type optimization**
- **Quality validation**

### 3. Exchange Adapters

Each exchange has a minimal adapter that:
- **Handles API-specific formatting**
- **Converts data to standard format**
- **Uses shared processing pipeline**
- **Provides consistent interface**

## Usage Examples

### Using Individual Exchange Adapters

```python
from exchanges.binance import create_binance_klines_adapter
from exchanges.okx import create_okx_klines_adapter

# Create adapters
binance_adapter = create_binance_klines_adapter()
okx_adapter = create_okx_klines_adapter()

# Get data
binance_data = await binance_adapter.get_klines_data("BTCUSDT", "1m")
okx_data = await okx_adapter.get_klines_data("BTCUSDT", "1m")
```

### Using Shared Pipeline

```python
from exchanges.shared import run_exchange_klines_pipeline

# Run complete pipeline for any exchange
results = await run_exchange_klines_pipeline(
    exchange="binance",
    symbol="BTCUSDT",
    interval="1m",
    years=2
)
```

### Data Quality Validation

```python
# Validate data quality
quality_result = adapter.validate_data_quality(data, "Validation context")
print(f"Quality passed: {quality_result['passed']}")
```

## Benefits

### 1. **Consistency**
- All exchanges use the same data processing logic
- Consistent API across all adapters
- Standardized data format

### 2. **Maintainability**
- Shared logic is centralized
- Changes apply to all exchanges
- Reduced code duplication

### 3. **Extensibility**
- Easy to add new exchanges
- Minimal exchange-specific code required
- Reusable components

### 4. **Quality**
- Centralized quality validation
- Consistent error handling
- Standardized data format

## Exchange-Specific Configurations

Each exchange adapter handles:
- **API endpoint differences**
- **Column name mapping**
- **Data format conversion**
- **Interval format conversion**

### Example: Column Mapping

```python
# Binance
column_mapping = {
    'openTime': 'open_time',
    'closeTime': 'close_time',
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'volume': 'volume'
}

# OKX
column_mapping = {
    'ts': 'timestamp',
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'vol': 'volume'
}
```

## Migration from Old Structure

The old monolithic exchange files (`binance.py`, `okx.py`, etc.) are still available for backward compatibility, but new development should use the standardized adapters.

### Backward Compatibility

- Old functions still work
- New functions are available
- Gradual migration possible

### New Development

- Use exchange adapters
- Use shared pipeline
- Follow standardized patterns

## Testing

Run the example script to test all exchanges:

```bash
python exchanges/examples/exchange_adapters_example.py
```

This will test:
- Individual adapter functionality
- Shared pipeline execution
- Data quality validation
- Error handling

## Future Enhancements

1. **Additional Exchanges**: Easy to add new exchanges following the same pattern
2. **Enhanced Validation**: More sophisticated quality checks
3. **Performance Optimization**: Improved data processing speed
4. **Monitoring**: Better logging and monitoring capabilities

## Conclusion

The new standardized structure provides a clean, maintainable, and extensible approach to exchange data processing. All exchanges now use the same shared logic while maintaining their specific API requirements through minimal adapters.