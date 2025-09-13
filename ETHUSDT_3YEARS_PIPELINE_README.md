# ETHUSDT 3-Year 1m Klines Data Download Pipeline

This document explains how to use the enhanced `klines_downloading_processing.py` pipeline to download and process 3 years of ETHUSDT 1m klines data.

## Features

The pipeline includes:

- ✅ **Data Downloading**: Downloads 3 years of ETHUSDT 1m data using `HistoricalDataPipeline`
- 🧹 **Column Removal**: Automatically removes `taker_buy_base`, `taker_buy_quote`, and `year` columns
- 🔍 **Gap Detection**: Detects gaps > 1m and re-downloads missing data
- 🔍 **Duplicate Analysis**: Identifies false duplicates (warnings) and removes true duplicates
- ✅ **Quality Checks**: Runs comprehensive data quality validation
- 📊 **Progress Tracking**: Detailed logging and status reporting

## Quick Start

### Option 1: Run the standalone script

```bash
python download_ethusdt_3years_pipeline.py
```

### Option 2: Run directly from Python

```python
import asyncio
from src.training.steps.data_collection.klines_downloading_processing import run_ethusdt_3year_pipeline

async def main():
    results = await run_ethusdt_3year_pipeline()
    print(f"Pipeline success: {results['pipeline_success']}")

asyncio.run(main())
```

### Option 3: Run from command line

```bash
python -c "
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path('src')))
from src.training.steps.data_collection.klines_downloading_processing import run_ethusdt_3year_pipeline

asyncio.run(run_ethusdt_3year_pipeline())
"
```

## Pipeline Steps

The pipeline executes the following steps in order:

1. **📥 Download**: Download 3 years of ETHUSDT 1m data using `HistoricalDataPipeline`
2. **🧹 Column Removal**: Remove unwanted columns (`taker_buy_base`, `taker_buy_quote`, `year`)
3. **🔍 Gap Detection**: Detect gaps > 1m in the data
4. **🔄 Gap Filling**: Re-download missing data for detected gaps (with column removal)
5. **🔍 Duplicate Analysis**: Analyze and handle duplicate timestamps
6. **✅ Quality Check**: Run final data quality validation

## Configuration Options

### Default Settings
- **Symbol**: ETHUSDT
- **Years**: 3
- **Interval**: 1m
- **Max Gap**: 1 minute
- **Data Directory**: `historical_data`

### Custom Configuration

```python
results = await run_ethusdt_3year_pipeline(
    data_dir="custom_data_directory",
    interval="5m",  # Can change interval
    max_gap_minutes=5,  # Adjust gap threshold
    api_key="your_api_key",  # Optional
    api_secret="your_api_secret"  # Optional
)
```

## Output and Results

The pipeline returns a comprehensive results dictionary:

```python
{
    "symbol": "ETHUSDT",
    "years": 3,
    "interval": "1m",
    "pipeline_success": True,
    "steps_completed": ["download", "column_removal", "gap_handling", "duplicate_handling", "quality_check"],
    "errors": [],
    "warnings": [],
    "summary": {
        "download": {...},
        "column_removal": {...},
        "gap_handling": {...},
        "duplicate_handling": {...},
        "quality_check": {...}
    },
    "completion_time": "2025-09-13T..."
}
```

## Handling Warnings and Errors

### Warnings
- **False Duplicates**: Same timestamp, different values - requires manual review
- **Mixed Duplicates**: Combination of true and false duplicates - requires analysis

### Errors
- Network connectivity issues
- API rate limiting
- Disk space problems
- File permission issues

## Data Storage Structure

Downloaded data is stored in:
```
historical_data/
├── binance/
│   └── ethusdt/
│       ├── raw/
│       │   └── ethusdt_1m/
│       │       ├── ethusdt_1m_2022_01.parquet
│       │       ├── ethusdt_1m_2022_02.parquet
│       │       └── ...
│       └── processed/
│           └── ethusdt_1m/
│               ├── ethusdt_1m_2022_01.parquet
│               └── ...
```

## API Keys (Optional)

If you have Binance API keys, you can provide them for higher rate limits:

```python
results = await run_ethusdt_3year_pipeline(
    api_key="your_binance_api_key",
    api_secret="your_binance_api_secret"
)
```

Without API keys, the pipeline will use public endpoints with lower rate limits.

## Troubleshooting

### Common Issues

1. **Slow Download Speed**: This is normal for large datasets. The pipeline will show progress.
2. **Memory Usage**: Large datasets may require significant RAM during processing.
3. **Disk Space**: Ensure you have at least 10GB free space for 3 years of 1m data.

### Recovery from Interruptions

If the pipeline is interrupted, you can:
1. Check what steps completed in the results
2. Re-run the pipeline - it will detect existing data and continue
3. Use the quality checker to validate downloaded data

## Advanced Usage

### Custom Symbol Pipeline

```python
from src.training.steps.data_collection.klines_downloading_processing import run_custom_symbol_pipeline

results = await run_custom_symbol_pipeline(
    symbol="BTCUSDT",
    years=2,
    interval="5m"
)
```

### Individual Pipeline Components

You can also use individual components:

```python
from src.training.steps.data_collection.klines_downloading_processing import KlinesDataProcessingPipeline

pipeline = KlinesDataProcessingPipeline()

# Just remove columns
result = pipeline.remove_unwanted_columns("ETHUSDT", "1m")

# Just check duplicates
result = pipeline.handle_duplicates("ETHUSDT", "1m")

# Just run quality check
quality_checker = pipeline.quality_checker
result = quality_checker.check_processed_data_quality("ETHUSDT", ["1m"])
```

## Performance Notes

- **Download Time**: ~30-60 minutes for 3 years of 1m data (depending on network)
- **Processing Time**: ~5-15 minutes for quality checks and duplicate analysis
- **Memory Usage**: ~2-4GB RAM during processing
- **Disk Usage**: ~8-12GB for processed 3-year 1m data

## Integration with Ares CLI

The pipeline integrates seamlessly with the existing Ares CLI:

```bash
python ares_launcher.py step01 --symbol ETHUSDT --years 3
```

The enhanced pipeline ensures all downloaded data meets quality standards before proceeding to the next steps.
