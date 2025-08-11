# Step 1.5: Unified Data Converter - Preserves Original Granularity

## Overview

The Step 1.5 Unified Data Converter transforms existing consolidated parquet files into a unified system that **preserves the original granularity** of each data type while providing efficient partitioned storage and querying capabilities.

## Purpose

This converter addresses the need to:
- **Maintain data integrity** by preserving original timestamps and granularity
- **Improve query performance** through partitioned storage
- **Enable efficient data access** for both detailed and summary analysis
- **Support future data collection** with consistent schema

## Architecture

### Input Data Sources
- **Klines**: 1-minute interval OHLCV data
- **Aggtrades**: Per-second/trade granularity data  
- **Futures**: Sparse funding rate intervals (every 8 hours)

### Output Structure
The converter creates **separate partitioned datasets** for each data type:

```
data_cache/unified/
├── klines/
│   └── {exchange}_{symbol}_{timeframe}/
│       ├── exchange=binance/
│       ├── symbol=BTCUSDT/
│       ├── timeframe=1m/
│       ├── year=2024/
│       ├── month=1/
│       └── day=1/
├── aggtrades/
│   └── {exchange}_{symbol}/
│       ├── exchange=binance/
│       ├── symbol=BTCUSDT/
│       ├── year=2024/
│       ├── month=1/
│       └── day=1/
├── futures/
│   └── {exchange}_{symbol}/
│       ├── exchange=binance/
│       ├── symbol=BTCUSDT/
│       ├── year=2024/
│       ├── month=1/
│       └── day=1/
├── metadata.json
└── _config.json
```

### Schema Design

#### Klines Dataset (1-minute intervals)
```python
{
    "timestamp": "int64",      # Milliseconds since epoch
    "exchange": "string",      # Exchange name
    "symbol": "string",        # Trading symbol
    "timeframe": "string",     # Timeframe (1m)
    "open": "float64",         # Opening price
    "high": "float64",         # High price
    "low": "float64",          # Low price
    "close": "float64",        # Closing price
    "volume": "float64",       # Volume
    "year": "int16",           # Partition column
    "month": "int8",           # Partition column
    "day": "int8"              # Partition column
}
```

#### Aggtrades Dataset (per-second/trade granularity)
```python
{
    "timestamp": "int64",              # Original timestamp (preserved)
    "exchange": "string",              # Exchange name
    "symbol": "string",                # Trading symbol
    "timeframe": "string",             # Base timeframe (1m)
    "trade_price": "float64",          # Trade price
    "trade_quantity": "float64",       # Trade quantity
    "trade_is_buyer_maker": "bool",    # Buyer maker flag
    "trade_id": "int64",               # Trade ID
    "year": "int16",                   # Partition column
    "month": "int8",                   # Partition column
    "day": "int8"                      # Partition column
}
```

#### Futures Dataset (sparse intervals)
```python
{
    "timestamp": "int64",      # Original timestamp (preserved)
    "exchange": "string",      # Exchange name
    "symbol": "string",        # Trading symbol
    "timeframe": "string",     # Base timeframe (1m)
    "funding_rate": "float64", # Funding rate
    "funding_time": "int64",   # Funding time
    "year": "int16",           # Partition column
    "month": "int8",           # Partition column
    "day": "int8"              # Partition column
}
```

## Usage

### Integration with Training Pipeline

The converter is automatically called at the beginning of Step 2 in the enhanced training manager:

```python
# Step 1.5: Unified Data Converter (NEW)
step_start = time.time()
self.logger.info("🔄 STEP 1.5: Unified Data Converter...")

from src.training.steps import step1_5_data_converter

step1_5_success = await step1_5_data_converter.run_step(
    symbol=symbol,
    exchange=exchange,
    timeframe=timeframe,
    data_dir=data_dir,
    force_rerun=training_input.get("force_rerun", False),
)
```

### Standalone Usage

```python
from src.training.steps.step1_5_data_converter import run_step

success = await run_step(
    symbol="BTCUSDT",
    exchange="binance",
    timeframe="1m",
    data_dir="data_cache",
    force_rerun=False
)
```

## Features

### ✅ Original Granularity Preservation
- **Klines**: Maintained at 1-minute intervals
- **Aggtrades**: Preserved at per-second/trade level
- **Futures**: Kept at sparse intervals (every 8 hours)

### ✅ Data Backup
- Automatically backs up existing consolidated files
- Creates timestamped backup directories
- Preserves original data integrity

### ✅ Intelligent Data Loading
- **Mac M1 Compatible**: Uses individual parquet files for aggtrades
- **Fallback Support**: CSV files as backup
- **Error Handling**: Graceful degradation for missing data

### ✅ Partitioned Storage
- **Hive-style partitioning** by date components
- **Efficient querying** with column pruning
- **Scalable storage** for large datasets

### ✅ Data Quality Assurance
- **Schema validation** for each dataset
- **Timestamp consistency** checks
- **Missing value handling** with appropriate defaults

### ✅ Future Infrastructure Setup
- **Configuration files** for future data collection
- **Metadata tracking** for dataset management
- **Consistent schema** across all data types

### ✅ Error Handling
- **Robust error recovery** with detailed logging
- **Graceful degradation** for missing data sources
- **Validation reporting** for data integrity

## Configuration

### Environment Variables
```bash
BLANK_TRAINING_MODE=1  # Limit data loading for testing
```

### Configuration Files
- `data_cache/unified/_config.json`: Future data collection settings
- `data_cache/unified/metadata.json`: Dataset metadata and schema

## Benefits

### 🚀 Performance Improvements
- **Faster queries** through partitioning
- **Reduced I/O** with column pruning
- **Efficient storage** with compression

### 🔧 Operational Benefits
- **Consistent schema** across all data types
- **Easy data management** with metadata
- **Scalable architecture** for future growth

### 📊 Analysis Benefits
- **Flexible querying** for different granularities
- **Rich data access** for detailed analysis
- **Summary statistics** when needed

## Migration Path

### From Consolidated Files
```
Before:
data_cache/
├── klines_binance_BTCUSDT_1m_consolidated.parquet
├── aggtrades_binance_BTCUSDT_consolidated.parquet
└── futures_binance_BTCUSDT_consolidated.parquet

After:
data_cache/unified/
├── klines/binance_BTCUSDT_1m/ (partitioned)
├── aggtrades/binance_BTCUSDT/ (partitioned)
├── futures/binance_BTCUSDT/ (partitioned)
├── metadata.json
└── _config.json
```

### Data Access Patterns
```python
# Query klines data
klines_df = await parquet_manager.read_partitioned_dataset(
    "data_cache/unified/klines/binance_BTCUSDT_1m",
    filters=[("year", "=", 2024), ("month", "=", 1)]
)

# Query aggtrades data (original granularity)
aggtrades_df = await parquet_manager.read_partitioned_dataset(
    "data_cache/unified/aggtrades/binance_BTCUSDT",
    filters=[("year", "=", 2024), ("month", "=", 1)]
)

# Query futures data (sparse intervals)
futures_df = await parquet_manager.read_partitioned_dataset(
    "data_cache/unified/futures/binance_BTCUSDT",
    filters=[("year", "=", 2024)]
)
```

## Edge Cases & Data Availability

### Missing Data Sources
The converter handles missing data gracefully:

1. **No Futures Data**: 
   - Futures dataset is skipped
   - Logs warning message
   - Continues with klines and aggtrades

2. **No Aggtrades Data**:
   - Aggtrades dataset is skipped  
   - Logs warning message
   - Continues with klines and futures

3. **No Klines Data**:
   - **Critical error** - conversion fails
   - Klines are required for the unified system

### Granularity Handling
Each data type maintains its original granularity:

1. **Klines (1-minute)**:
   - Preserved exactly as-is
   - No aggregation or modification

2. **Aggtrades (per-second/trade)**:
   - **Original timestamps preserved**
   - No forced aggregation to minute boundaries
   - Each trade record maintained individually

3. **Futures (sparse intervals)**:
   - **Original timestamps preserved**
   - No interpolation or filling
   - Sparse nature maintained

### Data Quality Assurance
- **Timestamp validation** for each dataset
- **Schema consistency** checks
- **Missing value handling** with appropriate defaults
- **Data type validation** for all columns

## Troubleshooting

### Common Issues

1. **Mac M1 Bus Errors**:
   - Use individual parquet files for aggtrades
   - Avoid large consolidated files
   - Check `test_individual_parquet.py` for compatibility

2. **Missing Data Sources**:
   - Check data availability in source files
   - Verify file paths and permissions
   - Review backup files if needed

3. **Partitioning Issues**:
   - Verify timestamp format (milliseconds)
   - Check date column generation
   - Validate partition column types

4. **Memory Issues**:
   - Reduce `max_files` parameter for testing
   - Use `BLANK_TRAINING_MODE=1`
   - Monitor system resources

### Validation
The converter includes comprehensive validation:
- **Dataset integrity** checks
- **Schema validation** for all data types
- **Sample data verification**
- **Partition structure validation**

### Logging
Detailed logs are written to:
- **Console output** for real-time monitoring
- **log/step1_5_converter.log** for persistent records
- **Structured logging** with emojis for easy scanning

## Future Enhancements

### Planned Features
- **Incremental updates** for new data
- **Data versioning** and rollback capabilities
- **Advanced compression** options
- **Query optimization** hints

### Integration Points
- **Real-time data feeds** integration
- **Multi-timeframe** support
- **Cross-exchange** data unification
- **Advanced analytics** pipeline integration 