# Exchange OHLCV Data Standardization Guide

## Overview

This guide explains the comprehensive OHLCV data standardization system that ensures complete equivalency between all exchanges (binance, bingx, okx, mexc) and full compatibility with `src/utils/data/` utilities.

## Architecture

### Core Components

1. **UnifiedExchangeStandardizer** (`exchanges/shared/unified_exchange_standardizer.py`)
   - Single source of truth for OHLCV data standardization
   - Handles all exchange-specific data format conversions
   - Ensures complete equivalency across exchanges
   - Full integration with `src/utils/data/` utilities

2. **EnhancedUnifiedExchangeAdapter** (`exchanges/shared/enhanced_unified_exchange_interface.py`)
   - Wraps individual exchange implementations
   - Provides unified interface for all exchanges
   - Ensures standardized data output
   - Handles error cases and edge conditions

3. **Exchange-Specific Adapters** (`exchanges/{exchange}/klines_adapter.py`)
   - Binance, BingX, OKX, MEXC adapters
   - Use enhanced unified adapter internally
   - Provide exchange-specific optimizations
   - Maintain backward compatibility

## Data Standardization

### Standardized Data Format

All exchanges must convert their data to this exact format:

```python
@dataclass
class StandardizedOHLCVData:
    # Core OHLCV data (required)
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str
    
    # Exchange metadata (required)
    exchange: str
    source: ExchangeType
    
    # Additional standardized fields (optional)
    quote_volume: Optional[float] = None
    trades_count: Optional[int] = None
    taker_buy_base_volume: Optional[float] = None
    taker_buy_quote_volume: Optional[float] = None
    
    # Data quality metrics
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    quality_score: float = 100.0
    
    # Processing metadata
    processed_at: Optional[datetime] = None
    raw_data_hash: Optional[str] = None
```

### Exchange-Specific Mappings

Each exchange has specific field mappings and configurations:

#### Binance
```python
{
    'timestamp_field': 'open_time',
    'timestamp_unit': 'ms',
    'field_mapping': {
        'openTime': 'timestamp',
        'closeTime': 'close_time',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
        'quoteVolume': 'quote_volume',
        'trades': 'trades_count',
        'takerBuyBase': 'taker_buy_base_volume',
        'takerBuyQuote': 'taker_buy_quote_volume'
    }
}
```

#### BingX
```python
{
    'timestamp_field': 'open_time',
    'timestamp_unit': 'ms',
    'field_mapping': {
        'openTime': 'timestamp',
        'closeTime': 'close_time',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
        'quoteVolume': 'quote_volume',
        'trades': 'trades_count',
        'takerBuyBase': 'taker_buy_base_volume',
        'takerBuyQuote': 'taker_buy_quote_volume'
    }
}
```

#### OKX
```python
{
    'timestamp_field': 'timestamp',
    'timestamp_unit': 'ms',
    'field_mapping': {
        'ts': 'timestamp',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'vol': 'volume',
        'volCcy': 'quote_volume',
        'confirm': 'trades_count'
    }
}
```

#### MEXC
```python
{
    'timestamp_field': 'open_time',
    'timestamp_unit': 'ms',
    'field_mapping': {
        'openTime': 'timestamp',
        'closeTime': 'close_time',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
        'quoteVolume': 'quote_volume',
        'trades': 'trades_count',
        'takerBuyBase': 'taker_buy_base_volume',
        'takerBuyQuote': 'taker_buy_quote_volume'
    }
}
```

## Usage Examples

### Basic Usage

```python
from exchanges.shared.unified_exchange_standardizer import standardize_exchange_ohlcv

# Standardize data from any exchange
df = standardize_exchange_ohlcv(
    raw_data=raw_exchange_data,
    exchange="binance",  # or "bingx", "okx", "mexc"
    symbol="BTCUSDT",
    interval="1m",
    quality_level="standard"
)
```

### Using Exchange Adapters

```python
from exchanges.binance.klines_adapter import BinanceKlinesAdapter
from exchanges.bingx.klines_adapter import BingXKlinesAdapter

# Initialize adapters
binance_adapter = BinanceKlinesAdapter()
bingx_adapter = BingXKlinesAdapter()

# Get standardized data
binance_data = await binance_adapter.get_klines_data("BTCUSDT", "1m", limit=100)
bingx_data = await bingx_adapter.get_klines_data("BTCUSDT", "1m", limit=100)

# Data is now fully equivalent and compatible with src/utils/data/
```

### Using Enhanced Unified Manager

```python
from exchanges.shared.enhanced_unified_exchange_interface import EnhancedUnifiedExchangeManager

# Initialize manager
manager = EnhancedUnifiedExchangeManager()

# Register exchanges
manager.register_exchange(binance_exchange_instance, ExchangeType.BINANCE)
manager.register_exchange(bingx_exchange_instance, ExchangeType.BINGX)

# Get data from all exchanges
all_data = await manager.get_klines_from_all("BTCUSDT", "1m", limit=100)

# Compare data equivalency
comparison = manager.compare_exchange_data("BTCUSDT", "1m")
```

## Data Quality Levels

The system supports different data quality validation levels:

- **BASIC**: Minimal validation, fastest processing
- **STANDARD**: Balanced validation and performance (default)
- **STRICT**: Comprehensive validation, slower processing
- **CRITICAL**: Maximum validation, includes outlier detection

```python
from exchanges.shared.unified_exchange_standardizer import DataQualityLevel

# Use strict quality level
standardizer = UnifiedExchangeStandardizer(DataQualityLevel.STRICT)
```

## Integration with src/utils/data/

The standardized data is fully compatible with all `src/utils/data/` utilities:

```python
from src.utils.data import DataProcessor, DataQualityFramework, DataCleaner

# All these work seamlessly with standardized data
processor = DataProcessor()
quality_framework = DataQualityFramework()
cleaner = DataCleaner()

# Process standardized data
processed_df = processor.optimize_dataframe_dtypes(df)
quality_result = quality_framework.validate_dataframe_quality(df, "test")
cleaned_df = cleaner.handle_missing_values_intelligently(df)
```

## Validation and Testing

### Running Equivalency Tests

```bash
python test_exchange_equivalency.py
```

This will test:
1. **Data Standardization**: All exchanges produce standardized format
2. **Data Equivalency**: Data from different exchanges is equivalent
3. **src/utils/data/ Compatibility**: Data works with all utilities

### Manual Validation

```python
from exchanges.shared.unified_exchange_standardizer import validate_ohlcv_equivalency

# Compare two DataFrames
result = validate_ohlcv_equivalency(df1, df2, tolerance=1e-6)
print(f"Equivalent: {result['equivalent']}")
print(f"Issues: {result['errors']}")
```

## Key Features

### 1. Complete Equivalency
- All exchanges produce identical data format
- Same column names, data types, and structure
- Consistent timestamp handling across exchanges

### 2. Full src/utils/data/ Compatibility
- Works with all data processing utilities
- Optimized data types for memory efficiency
- Compatible with quality validation frameworks

### 3. Exchange-Agnostic Processing
- Single interface for all exchanges
- No need to handle exchange-specific logic
- Consistent error handling and validation

### 4. Comprehensive Validation
- Data quality scoring
- OHLCV relationship validation
- Timestamp consistency checks
- Outlier detection and handling

### 5. Memory Efficiency
- Optimized data types
- Feature-specific optimizations
- Streaming support for large datasets

## Error Handling

The system provides comprehensive error handling:

```python
try:
    df = await adapter.get_klines_data("BTCUSDT", "1m")
except Exception as e:
    # Handle exchange-specific errors
    logger.error(f"Failed to get data: {e}")
```

## Performance Considerations

1. **Data Type Optimization**: Automatic optimization of data types for memory efficiency
2. **Quality Level Selection**: Choose appropriate quality level for your use case
3. **Batch Processing**: Process multiple symbols/intervals efficiently
4. **Caching**: Reuse standardized data when possible

## Troubleshooting

### Common Issues

1. **Missing Data**: Check if exchange is available and API keys are valid
2. **Data Quality Issues**: Adjust quality level or check raw data
3. **Compatibility Issues**: Ensure using latest version of src/utils/data/

### Debug Mode

```python
import logging
logging.getLogger("UnifiedExchangeStandardizer").setLevel(logging.DEBUG)
```

## Migration Guide

### From Old System

If you're migrating from the old standardization system:

1. Update imports to use new modules
2. Replace `UnifiedExchangeAdapter` with `EnhancedUnifiedExchangeAdapter`
3. Update data access patterns to use new interface
4. Test with equivalency test suite

### Backward Compatibility

The new system maintains backward compatibility with existing code while providing enhanced features.

## Contributing

When adding new exchanges:

1. Create exchange-specific adapter in `exchanges/{exchange}/klines_adapter.py`
2. Add exchange configuration to `UnifiedExchangeStandardizer`
3. Update field mappings and interval mappings
4. Add tests to equivalency test suite
5. Update documentation

## Support

For issues or questions:
1. Check the equivalency test results
2. Review error logs
3. Validate data format manually
4. Check exchange-specific documentation