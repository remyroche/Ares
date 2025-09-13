# Historical Data Pipeline for Binance Klines

A comprehensive toolkit for downloading, processing, and managing historical Binance klines data with gap detection, feature engineering, and optimized parquet storage.

## Features

- **Historical Data Download**: Download 3 years of Binance klines data with monthly parquet files
- **Gap Detection & Filling**: Automatically detect and fill data gaps over 1 minute
- **Feature Engineering**: Add price returns, volume returns, technical indicators, and more
- **Multi-timeframe Resampling**: Resample data to 5m, 15m, 30m, 1h with proper aggregation
- **Optimized Storage**: Use parquet partitioning for efficient storage and access
- **Unified API**: Single interface for all data operations
- **CLI Interface**: Easy-to-use command-line tools

## Quick Start

### 1. Download Historical Data

```python
from src.utils.data.historical_data_downloader import download_ethusdt_historical_data

# Download 3 years of ETHUSDT data
success = await download_ethusdt_historical_data(
    years=3,
    data_dir="historical_data",
    api_key="your_api_key",  # Optional
    api_secret="your_api_secret"  # Optional
)
```

### 2. Check and Fill Gaps

```python
from src.utils.data.gap_detector import detect_and_fill_gaps

# Detect and fill gaps
results = await detect_and_fill_gaps(
    symbol="ETHUSDT",
    interval="1m",
    max_gap_minutes=1,
    data_dir="historical_data"
)
```

### 3. Feature Engineering and Resampling

```python
from src.utils.data.feature_engineer import process_ethusdt_data

# Process data with feature engineering
results = process_ethusdt_data(
    data_dir="historical_data",
    target_intervals=["5m", "15m", "30m", "1h"]
)
```

### 4. Access Data

```python
from src.utils.data.klines_parquet import read_ethusdt_data

# Read raw data
raw_data = read_ethusdt_data(
    interval="1m",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    data_type="raw"
)

# Read processed data
processed_data = read_ethusdt_data(
    interval="5m",
    data_type="processed"
)
```

## Command Line Interface

The CLI provides easy access to all functionality:

```bash
# Download 3 years of ETHUSDT data
python src/utils/data/cli.py download --symbol ETHUSDT --years 3

# Check for gaps and fill them
python src/utils/data/cli.py gap-check --symbol ETHUSDT --fill

# Process data with feature engineering
python src/utils/data/cli.py process --symbol ETHUSDT --intervals 5m 15m 30m 1h

# Run complete pipeline
python src/utils/data/cli.py pipeline --symbol ETHUSDT --years 3 --intervals 5m 15m 30m 1h

# Check status
python src/utils/data/cli.py status --symbol ETHUSDT

# Get detailed info
python src/utils/data/cli.py info --symbol ETHUSDT --interval 1m --data-type raw

# List all available data
python src/utils/data/cli.py list
```

## Complete Pipeline

For a complete end-to-end solution:

```python
from src.utils.data.historical_data_pipeline import run_ethusdt_pipeline

# Run complete pipeline
results = await run_ethusdt_pipeline(
    years=3,
    data_dir="historical_data",
    api_key="your_api_key",  # Optional
    api_secret="your_api_secret",  # Optional
    target_intervals=["5m", "15m", "30m", "1h"]
)
```

## Data Structure

The pipeline creates the following directory structure:

```
historical_data/
└── binance/
    └── ethusdt/
        ├── raw/                          # Raw klines data
        │   ├── ethusdt_1m_2022_01.parquet
        │   ├── ethusdt_1m_2022_02.parquet
        │   └── ...
        └── processed/                    # Processed data with features
            ├── ethusdt_1m/              # Partitioned by year/month
            │   ├── year=2022/month=01/
            │   │   └── part-0.parquet
            │   └── ...
            ├── ethusdt_5m/
            ├── ethusdt_15m/
            ├── ethusdt_30m/
            └── ethusdt_1h/
```

## Features Added

### Price Features
- `close_return`: Close price percentage change
- `open_return`: Open price percentage change
- `high_return`: High price percentage change
- `low_return`: Low price percentage change
- `close_log_return`: Close price log return
- `open_log_return`: Open price log return

### Volume Features
- `volume_return`: Volume percentage change
- `volume_log_return`: Volume log return
- `volume_sma_20`: 20-period volume moving average
- `volume_ratio`: Volume relative to moving average

### Technical Indicators
- `rsi_14`: 14-period RSI
- `close_sma_5`: 5-period close price SMA
- `close_sma_20`: 20-period close price SMA
- `close_ema_12`: 12-period close price EMA
- `close_ema_26`: 26-period close price EMA
- `bb_upper`: Bollinger Bands upper band
- `bb_middle`: Bollinger Bands middle band
- `bb_lower`: Bollinger Bands lower band
- `bb_width`: Bollinger Bands width
- `bb_position`: Position within Bollinger Bands

### Volatility Features
- `volatility_20`: 20-period volatility
- `volatility_5`: 5-period volatility

### Time Features
- `hour`: Hour of day
- `day_of_week`: Day of week
- `is_weekend`: Weekend indicator

### Lagged Features
- `close_lag_1`, `close_lag_2`, etc.: Lagged close prices
- `volume_lag_1`, `volume_lag_2`, etc.: Lagged volumes

## API Reference

### HistoricalDataDownloader

Downloads historical klines data from Binance.

```python
downloader = HistoricalDataDownloader(data_dir="historical_data")

# Download data
success = await downloader.download_historical_klines(
    symbol="ETHUSDT",
    interval="1m",
    years=3,
    api_key="your_api_key",
    api_secret="your_api_secret"
)

# Get data summary
summary = downloader.get_data_summary("ETHUSDT")
```

### GapDetector

Detects and fills gaps in historical data.

```python
detector = GapDetector(data_dir="historical_data")

# Detect gaps
gaps = detector.detect_gaps("ETHUSDT", "1m", max_gap_minutes=1)

# Fill gaps
results = await detector.fill_gaps(gaps, api_key, api_secret)
```

### FeatureEngineer

Adds features and resamples data.

```python
engineer = FeatureEngineer(data_dir="historical_data")

# Process data
results = engineer.process_symbol_data(
    symbol="ETHUSDT",
    interval="1m",
    target_intervals=["5m", "15m", "30m", "1h"]
)
```

### KlinesParquetManager

Unified interface for data operations.

```python
manager = KlinesParquetManager(data_dir="historical_data")

# Read data
data = manager.read_data(
    symbol="ETHUSDT",
    interval="1m",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    data_type="raw"
)

# Write data
success = manager.write_data(data, "ETHUSDT", "1m", "raw")

# Get data info
info = manager.get_data_info("ETHUSDT", "1m", "raw")
```

## Requirements

- Python 3.8+
- pandas
- numpy
- pyarrow
- aiohttp (for downloading)
- exchange.binance (Binance API client)

## Installation

1. Install dependencies:
```bash
pip install pandas numpy pyarrow aiohttp
```

2. Ensure the exchange.binance module is available in your Python path.

## Usage Examples

### Example 1: Complete Pipeline

```python
import asyncio
from src.utils.data.historical_data_pipeline import run_ethusdt_pipeline

async def main():
    results = await run_ethusdt_pipeline(
        years=3,
        data_dir="historical_data",
        target_intervals=["5m", "15m", "30m", "1h"]
    )
    
    if results["pipeline_success"]:
        print("✅ Pipeline completed successfully!")
        print(f"Steps completed: {results['steps_completed']}")
    else:
        print(f"❌ Pipeline failed: {results['errors']}")

asyncio.run(main())
```

### Example 2: Custom Processing

```python
from src.utils.data.klines_parquet import get_klines_manager
from src.utils.data.feature_engineer import FeatureEngineer

# Get data manager
manager = get_klines_manager("historical_data")

# Read raw data
raw_data = manager.read_data("ETHUSDT", "1m", data_type="raw")

# Process with custom features
engineer = FeatureEngineer("historical_data")
featured_data = engineer._add_features(raw_data)

# Save processed data
manager.write_data(featured_data, "ETHUSDT", "1m", "processed")
```

### Example 3: Data Analysis

```python
from src.utils.data.klines_parquet import read_ethusdt_data
import pandas as pd

# Read processed data
data = read_ethusdt_data(
    interval="5m",
    data_type="processed",
    start_date=datetime(2023, 1, 1)
)

# Analyze data
print(f"Data shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print(f"Date range: {data.index.min()} to {data.index.max()}")

# Calculate some statistics
if 'close_return' in data.columns:
    print(f"Average return: {data['close_return'].mean():.4f}")
    print(f"Volatility: {data['close_return'].std():.4f}")
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and the src directory is in your Python path.

2. **API Rate Limits**: The Binance API has rate limits. The downloader includes delays to respect these limits.

3. **Memory Issues**: For large datasets, consider processing data in chunks or using the CLI with specific date ranges.

4. **Gap Detection**: If gaps are detected, the system will automatically attempt to fill them. Check the logs for details.

### Performance Tips

1. **Use Parquet**: The system uses parquet format for efficient storage and access.

2. **Partitioning**: Processed data is partitioned by year/month for better performance.

3. **Data Types**: The system optimizes data types to reduce memory usage.

4. **Chunked Processing**: For very large datasets, consider processing data in time-based chunks.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.