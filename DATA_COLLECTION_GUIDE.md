# Data Collection Guide

This guide explains how to use the Enhanced Klines Processing Pipeline for collecting historical cryptocurrency data.

## Quick Start

The easiest way to collect data is to run the enhanced pipeline directly:

```bash
cd /Users/remyroche/Documents/Ares
python3 src/training/steps/data_collection/enhanced_klines_processing_pipeline.py
```

This will collect 4 years of ETHUSDT data from Binance with all features enabled.

## Command-Line Options

You can customize the data collection with the following options:

```bash
python3 src/training/steps/data_collection/enhanced_klines_processing_pipeline.py \
    --exchange binance \
    --symbol ETHUSDT \
    --interval 1m \
    --years 4 \
    --data-dir data_cache
```

### Available Options

- `--exchange`: Exchange name (default: `binance`)
  - Supported: `binance`, `bingx`, `okx`, `mexc`, `gateio`, `phemex`
- `--symbol`: Trading symbol (default: `ETHUSDT`)
- `--interval`: Data interval (default: `1m`)
  - Supported: `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`
- `--years`: Number of years of historical data (default: `4`)
- `--data-dir`: Output directory (default: `data_cache`)
- `--no-gap-filling`: Disable automatic gap filling
- `--no-resampling`: Disable automatic resampling to multiple timeframes
- `--no-quality-validation`: Disable quality validation

## Features

### ✅ Gap Detection & Filling
The pipeline automatically:
- Detects gaps in the data (missing candles)
- Fills gaps using intelligent interpolation
- Ensures continuous time series

### ✅ Data Resampling
Automatically generates multiple timeframes:
- Source: 1m data
- Generated: 5m, 15m, 30m, 1h (configurable)
- Only resamples data older than 1 day

### ✅ Quality Filtering
Comprehensive data quality checks:
- Duplicate detection and removal
- OHLCV validation
- Statistical distribution validation
- Quality scoring and assessment

### ✅ Efficient Storage
- Saves data in Parquet format
- Follows PipelineStandards directory structure
- Batch-compatible processing
- Memory-efficient streaming

## Output

Data is saved to:
```
data_cache/{exchange}/{symbol}/
```

For example, Binance ETHUSDT data goes to:
```
data_cache/binance/ethusdt/
```

## Examples

### Collect Bitcoin data from Binance
```bash
python3 src/training/steps/data_collection/enhanced_klines_processing_pipeline.py \
    --symbol BTCUSDT \
    --years 2
```

### Collect from BingX without resampling
```bash
python3 src/training/steps/data_collection/enhanced_klines_processing_pipeline.py \
    --exchange bingx \
    --symbol ETHUSDT \
    --no-resampling
```

### Collect 5-minute data directly
```bash
python3 src/training/steps/data_collection/enhanced_klines_processing_pipeline.py \
    --interval 5m \
    --years 1
```

## Troubleshooting

### Connection Issues
- The pipeline works with public market data (no API keys required)
- If connection fails, it will continue with public data access
- Rate limits are automatically respected

### Memory Issues
- For large data collections (>5 years), consider collecting in smaller chunks
- The pipeline uses streaming to minimize memory usage
- Gap filling and resampling are memory-intensive operations

### Data Quality
- Check the quality metrics in the output
- Failed quality checks don't stop the pipeline
- Review `stored_files` output to verify data was saved

## Alternative: Direct Adapter Approach

For simpler use cases, you can use the direct adapter script:

```bash
python3 run_binance_data_collection_direct.py
```

This bypasses the full pipeline and directly uses the Binance klines adapter.

## Fixed Issues

### ✅ ExchangeInterface
- Fixed duplicate `ExchangeType` enum definitions
- Now uses shared `exchanges.exchange_types.ExchangeType`
- Proper enum mapping for all supported exchanges
- No more "Unsupported exchange type" errors

### ✅ UnifiedTradingStandardizer
- Added missing import in `exchange_dispatcher.py`
- Now properly initialized during exchange connection

### ✅ Standalone Execution
- Pipeline can run directly without wrapper scripts
- Command-line argument support
- Proper error handling and logging

