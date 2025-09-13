# Klines Data Collection Module

This directory contains scripts and utilities specifically for collecting, processing, and managing klines (candlestick) data from cryptocurrency exchanges.

## Files Overview

### Core Klines Utilities (from src/utils/data/)
- `klines_parquet.py` - Handles reading/writing klines data to/from parquet files with optimized storage
- `gap_detector.py` - Detects and fills gaps in klines data with per-month logging
- `historical_data_downloader.py` - Downloads historical klines data from Binance and other exchanges
- `historical_data_pipeline.py` - Orchestrates the complete klines data pipeline
- `basic_returns_engineer.py` - Processes basic returns and technical features from klines data

### Training Step Components (from src/training/steps/data_collection/)
- `unified_data_downloader.py` - Centralized download functionality for klines, aggtrades, and futures data
- `enhanced_append_data_downloader.py` - Data download functionality that ensures data is appended to existing files
- `unified_gap_filler.py` - Unified gap filling functionality for klines data
- `unified_resampler.py` - Unified resampling functionality for klines data to different timeframes

## Usage Examples

### Migration from parquet_utils (Backward Compatibility)
```python
# OLD WAY (still works)
from src.utils.parquet_utils import get_parquet_utils, safe_read_parquet

# NEW WAY (same API, enhanced functionality)
from src.steps.data_collection.klines_data import get_parquet_utils, safe_read_parquet

utils = get_parquet_utils()
df = safe_read_parquet("historical_data/binance/ethusdt/raw/ethusdt_1m_2024_09.parquet")
# Automatically detects klines data and uses enhanced reading methods
```

### Enhanced Klines Manager (New Features)
```python
from src.steps.data_collection.klines_data import get_klines_manager

manager = get_klines_manager()

# Automatic data discovery and reading
df = manager.read_data("ETHUSDT", "1m")

# Get comprehensive data information
info = manager.get_data_info("ETHUSDT", "1m")
print(f"Records: {info['total_records']:,}")
print(f"Date range: {info['date_range']}")

# Get data quality statistics
stats = manager.get_data_statistics("ETHUSDT", "1m")
```

### Basic Data Download
```python
from src.steps.data_collection.klines_data.unified_data_downloader import UnifiedDataDownloader

downloader = UnifiedDataDownloader()
await downloader.download_klines_data(
    symbol="ETHUSDT",
    interval="1m",
    start_date="2024-01-01",
    end_date="2024-12-31"
)
```

### Gap Detection and Filling
```python
from src.steps.data_collection.klines_data.gap_detector import GapDetector

detector = GapDetector()
gaps = detector.detect_gaps("ETHUSDT", "1m")
if gaps:
    await detector.fill_gaps(gaps, api_key, api_secret)
```

### Feature Engineering
```python
from src.steps.data_collection.klines_data.basic_returns_engineer import BasicReturnsEngineer

engineer = BasicReturnsEngineer()
results = engineer.process_symbol_data("ETHUSDT", "1m", ["5m", "15m", "1h"])
```

### Pipeline Orchestration
```python
from src.steps.data_collection.klines_data.historical_data_pipeline import HistoricalDataPipeline

pipeline = HistoricalDataPipeline()
results = await pipeline.run_complete_pipeline(
    symbol="ETHUSDT",
    years=3,
    target_intervals=["5m", "15m", "30m", "1h"]
)
```

## Data Flow

1. **Download** → `unified_data_downloader.py` or `enhanced_append_data_downloader.py`
2. **Gap Detection** → `gap_detector.py`
3. **Gap Filling** → `unified_gap_filler.py`
4. **Resampling** → `unified_resampler.py`
5. **Feature Engineering** → `basic_returns_engineer.py`
6. **Storage** → `klines_parquet.py`

## Configuration

All scripts use the centralized configuration system from `src.config`. Key settings include:

- Data directory paths
- Exchange API credentials
- Download batch sizes
- Gap detection thresholds
- Resampling intervals

## Dependencies

This module depends on:
- `src.utils.logger` - For comprehensive logging
- `src.utils.error_handler` - For error handling and recovery
- `src.utils.common_operations` - For common data operations
- `src.exchange.binance` - For Binance API integration

## Notes

- All scripts include comprehensive logging with per-month statistics where applicable
- Error handling is implemented using the centralized error handling system
- Memory optimization is built into all data processing operations
- Files are designed to work with both training and production environments
