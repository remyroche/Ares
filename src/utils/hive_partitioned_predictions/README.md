# Hive-Partitioned Prediction Storage

Production-grade Hive partitioning architecture for ML prediction storage with **NO metadata.json files**. The filesystem is the source of truth.

## Overview

This module provides a complete solution for storing and retrieving ML predictions using Apache Hive-style partitioning. It eliminates the race conditions and complexity of metadata.json files while providing excellent query performance through partition pruning.

### Key Features

- ✅ **No metadata.json** = No race conditions!
- ✅ **Thread-safe atomic writes** using temp files + rename
- ✅ **Smart monthly consolidation** for 95%+ file reduction
- ✅ **Efficient partition pruning** for fast queries
- ✅ **Polars support** for blazing-fast lazy evaluation
- ✅ **Race condition protection** with lock files
- ✅ **ZSTD compression** for efficient storage

## Directory Structure

```
artifacts/
├── specialists/
│   ├── predictions/
│   │   ├── model_version=v1.2.3/
│   │   │   ├── year=2025/
│   │   │   │   ├── month=11/
│   │   │   │   │   ├── day=01/
│   │   │   │   │   │   └── data.parquet  # 96 rows (15m data)
│   │   │   │   │   ├── day=02/
│   │   │   │   │   │   └── data.parquet
│   │   │   │   │   └── monthly_consolidated.parquet  # Created at month end
│   │   │   │   └── month=12/
│   │   │   │       └── ...
│   │   └── model_version=v1.2.4/
│   │       └── ...
│   └── models/
│       └── v1.2.3_monthly_2025-11.pkl
│
├── base_models/
│   ├── predictions/
│   │   └── model_version=v2.1.0/
│   │       └── year=2025/
│   │           └── month=11/
│   │               ├── day=01/
│   │               ├── monthly_consolidated.parquet
│   │               └── ...
│   └── models/
│       └── v2.1.0_monthly_2025-11.pkl
│
└── meta_layer/
    ├── predictions/
    │   └── model_version=v3.0.5/
    │       └── year=2025/
    │           └── month=11/
    │               ├── day=01/
    │               ├── monthly_consolidated.parquet
    │               └── ...
    └── models/
        └── v3.0.5_weekly_2025-W47.pkl
```

## Quick Start

### Writing Predictions

```python
from datetime import datetime
import pandas as pd
from src.utils.hive_partitioned_predictions import HivePartitionedWriter

# Initialize writer
writer = HivePartitionedWriter(
    layer_name="specialists",
    model_version="v1.2.3"
)

# Write predictions
df = pd.DataFrame({
    'prediction': [0.52, 0.48, 0.55],
    'confidence': [0.85, 0.90, 0.78]
}, index=pd.date_range('2025-11-01', periods=3, freq='15min'))

filepath = writer.write_predictions(
    df=df,
    prediction_date=datetime(2025, 11, 1),
    metadata={'symbol': 'ETHUSDT', 'exchange': 'binance'}
)

print(f"Written to: {filepath}")
# Output: artifacts/specialists/predictions/model_version=v1.2.3/year=2025/month=11/day=01/data.parquet
```

### Reading Predictions

```python
from src.utils.hive_partitioned_predictions import HivePartitionedReader

# Initialize reader
reader = HivePartitionedReader("specialists")

# Load recent predictions
df = reader.load_recent_predictions(
    days=56,  # Last 8 weeks
    model_version="v1.2.3"  # Optional, defaults to latest
)

print(f"Loaded {len(df)} predictions")
```

### Ultra-Fast Reading with Polars

```python
from src.utils.hive_partitioned_predictions import PolarsHiveReader

# Initialize Polars reader
reader = PolarsHiveReader("specialists")

# Load with lazy evaluation (blazing fast!)
df = reader.load_recent_predictions_lazy(
    days=56,
    model_version="v1.2.3",
    return_pandas=True  # Convert to pandas
)
```

### Monthly Compaction

```python
from src.utils.hive_partitioned_predictions import MonthlyCompactor

# Initialize compactor
compactor = MonthlyCompactor("specialists")

# Compact previous month
stats = compactor.compact_previous_month()

print(f"Compacted {stats['months_compacted']} months")
print(f"Files: {stats['files_before']} -> {stats['files_after']}")
print(f"Reduction: {100 * (1 - stats['files_after']/stats['files_before']):.1f}%")
```

## Performance Benefits

### Before (Traditional Storage)

```
30 days × 96 15-min candles = 2,880 rows
Storage: 30 files × 96 rows each = 2,880 rows across 30 files
Query: Must open and read 30 separate files
Inodes: 30 inodes used
```

### After (Monthly Consolidation)

```
30 days × 96 15-min candles = 2,880 rows
Storage: 1 file × 2,880 rows = 2,880 rows in 1 file
Query: Open and read 1 file
Inodes: 1 inode used (97% reduction!)
```

### Read Performance

| Operation | Daily Files | Monthly Consolidated | Speedup |
|-----------|-------------|---------------------|---------|
| Read 1 month | 30 file opens | 1 file open | **30x faster** |
| Read 3 months | 90 file opens | 3 file opens | **30x faster** |
| Read 1 year | 365 file opens | 12 file opens | **30x faster** |

## Scheduled Compaction

### Cron Job Setup

Add to your crontab to run compaction on the 1st of each month at 02:00 AM UTC:

```bash
# Run monthly compaction
0 2 1 * * cd /path/to/Ares && python -m src.utils.hive_partitioned_predictions compact
```

### Manual Compaction

```bash
# Compact all layers for previous month
python -m src.utils.hive_partitioned_predictions compact

# Compact specific layers
python -m src.utils.hive_partitioned_predictions compact specialists base_models

# Preserve daily files (for testing)
python -m src.utils.hive_partitioned_predictions compact --no-delete

# Enable debug logging
python -m src.utils.hive_partitioned_predictions compact --log-level DEBUG

# Write to log file
python -m src.utils.hive_partitioned_predictions compact --log-file compaction.log
```

### Programmatic Compaction

```python
from src.utils.hive_partitioned_predictions import monthly_compaction_job

# Run compaction for all layers
results = monthly_compaction_job()

for layer, stats in results.items():
    if stats:
        print(f"{layer}: {stats['months_compacted']} months compacted")
```

## Backfilling

If you need to compact historical data:

```python
from src.utils.hive_partitioned_predictions import backfill_compaction

# Compact Q4 2024
stats = backfill_compaction(
    layer="specialists",
    model_version="v1.2.3",
    start_year=2024,
    start_month=10,
    end_year=2024,
    end_month=12
)

print(f"Compacted {stats['months_compacted']} months")
```

## Thread Safety

### Writer Thread Safety

Each prediction write is thread-safe:

1. **Atomic writes**: Uses temp file + rename (POSIX atomic operation)
2. **Separate partitions**: Each day gets its own directory
3. **No shared state**: No metadata.json to coordinate

```python
# Safe to run in parallel!
from concurrent.futures import ThreadPoolExecutor

def write_predictions_for_day(day):
    writer = HivePartitionedWriter("specialists", "v1.2.3")
    writer.write_predictions(df, day)

with ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(write_predictions_for_day, days)
```

### Compactor Thread Safety

The compactor uses lock files to prevent race conditions:

```python
# Lock file prevents concurrent compaction of same month
# artifacts/specialists/predictions/model_version=v1.2.3/year=2025/month=11/.compaction.lock
```

## Integration with Existing Code

### Example: Integrate with Training Pipeline

```python
from src.utils.hive_partitioned_predictions import HivePartitionedWriter
from src.training.steps.base_step import BaseStep

class SpecialistStep(BaseStep):
    def __init__(self):
        super().__init__()
        self.hive_writer = HivePartitionedWriter(
            layer_name="specialists",
            model_version=self.model_version
        )

    def save_predictions(self, df: pd.DataFrame, prediction_date: datetime):
        # Old way: Save to versioned artifacts
        # self.versioned_store.add_data(df)

        # New way: Save to Hive partitions
        self.hive_writer.write_predictions(df, prediction_date)
```

### Example: Load in Live Trading

```python
from src.utils.hive_partitioned_predictions import HivePartitionedReader

class LiveTradingSystem:
    def __init__(self):
        self.reader = HivePartitionedReader("specialists")

    def load_recent_predictions(self):
        # Load last 8 weeks of predictions
        df = self.reader.load_recent_predictions(days=56)
        return df
```

## Migration Guide

### From Versioned Artifacts

If you're migrating from the old `VersionedArtifactStore`:

```python
# Old code
from src.utils.versioned_artifacts import VersionedArtifactStore

store = VersionedArtifactStore("ETHUSDT_binance_15m_long_analyst")
store.add_data(df)
df = store.query_by_index_range(start_date, end_date)

# New code
from src.utils.hive_partitioned_predictions import HivePartitionedWriter, HivePartitionedReader

# Writing
writer = HivePartitionedWriter("specialists", "v1.2.3")
writer.write_predictions(df, prediction_date)

# Reading
reader = HivePartitionedReader("specialists")
df = reader.load_recent_predictions(start_date=start_date, end_date=end_date)
```

### Migration Checklist

1. ✅ Update prediction writing code to use `HivePartitionedWriter`
2. ✅ Update prediction reading code to use `HivePartitionedReader`
3. ✅ Remove `metadata.json` generation code
4. ✅ Set up monthly compaction cron job
5. ✅ Backfill historical data (optional)
6. ✅ Test end-to-end pipeline

## Troubleshooting

### Issue: "No model versions found"

**Cause**: No predictions have been written yet, or the base path is incorrect.

**Solution**: Verify the base path and write some predictions first:

```python
writer = HivePartitionedWriter("specialists", "v1.2.3")
writer.write_predictions(df, datetime.now())
```

### Issue: "Lock timeout"

**Cause**: Another compaction process is running and holding the lock.

**Solution**: Wait for the other process to complete, or manually remove the lock file:

```bash
find artifacts -name ".compaction.lock" -delete
```

### Issue: Duplicate data after compaction

**Cause**: Daily files were not deleted after consolidation.

**Solution**: Ensure `delete_daily_files=True` (default) in compactor:

```python
compactor = MonthlyCompactor("specialists", delete_daily_files=True)
```

### Issue: Polars not available

**Cause**: Polars is not installed.

**Solution**: Install Polars:

```bash
pip install polars
```

## Advanced Usage

### Custom Metadata Columns

```python
writer = HivePartitionedWriter("specialists", "v1.2.3")

# Add custom metadata
writer.write_predictions(
    df=df,
    prediction_date=datetime.now(),
    metadata={
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'strategy': 'momentum'
    }
)
```

### Partition Discovery

```python
reader = HivePartitionedReader("specialists")

# Get available model versions
versions = reader.get_available_model_versions()
print(f"Available versions: {versions}")

# Get date range for a version
min_date, max_date = reader.get_date_range("v1.2.3")
print(f"Data available from {min_date} to {max_date}")
```

### Conditional Writing

```python
writer = HivePartitionedWriter("specialists", "v1.2.3")

# Check if partition already exists
if not writer.partition_exists(prediction_date):
    writer.write_predictions(df, prediction_date)
else:
    print("Partition already exists, skipping...")
```

## API Reference

See individual module documentation:

- [writer.py](./writer.py) - `HivePartitionedWriter` class
- [reader.py](./reader.py) - `HivePartitionedReader` and `PolarsHiveReader` classes
- [compactor.py](./compactor.py) - `MonthlyCompactor` class
- [jobs.py](./jobs.py) - Scheduled compaction jobs

## Testing

Run tests:

```bash
pytest tests/utils/hive_partitioned_predictions/
```

## License

Copyright (c) 2025 Ares Trading System
