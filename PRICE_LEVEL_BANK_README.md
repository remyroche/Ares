# Price Level Bank System

The **Price Level Bank** is a comprehensive system for storing, managing, and accessing historical price levels with their associated tags and features. This system optimizes ML model training by providing pre-computed historical analysis that can be reused across different feature generators.

## 🎯 Key Benefits

- **Eliminates redundant calculations** - Price level analysis is computationally expensive
- **Provides consistent tagging** - All features use the same historical data
- **Enables efficient ML training** - Features are pre-computed and cached
- **Supports multiple timeframes** - Works with any timeframe data
- **Persistent storage** - Data survives across sessions

## 🏗️ System Architecture

### Core Components

1. **PriceLevelBank** (`src/feature_generation/core/price_level_bank.py`)
   - Main storage and management system
   - Efficient indexing and querying
   - Persistent storage with auto-save

2. **Bank Builder** (`build_price_level_bank.py`)
   - Processes historical data
   - Calculates all price level tags
   - Populates the bank with data

3. **Query Interface** (`query_price_level_bank.py`)
   - Command-line tool for querying the bank
   - Export capabilities
   - Statistics and analysis

4. **Integrated Features** (updated in `support_resistance.py`)
   - Historical price level generators
   - Bank-first lookup for efficiency
   - Fallback to calculation if needed

## 📊 Data Structure

### PriceLevelData
Each price level contains:

```python
@dataclass
class PriceLevelData:
    price: float                    # Price level
    level_pct: float               # Level percentage (0.2 = 0.2%)
    symbol: str                    # Trading symbol
    timeframe: str                 # Timeframe
    timestamp: pd.Timestamp        # When this level was calculated

    # Historical tags (computed from past data)
    historical_crossings: int      # Number of times price crossed this level
    historical_bounces: int        # Number of bounces/reversals at this level
    historical_volume: float       # Volume traded at/near this level
    historical_touch_density: float # How concentrated touches are
    historical_time_decay: float   # Time-decayed importance
    historical_success_rate: float # Historical success rate as S/R

    # Additional metadata
    strength_score: float          # Overall strength (0-1)
    recency_score: float           # How recent the activity is
    clustering_score: float        # How clustered the activity is
    momentum_score: float          # Price momentum approaching level

    # Time-based features
    session_type: str              # 'asian', 'european', 'us'
    day_of_week: int               # 0-6
    hour_of_day: int               # 0-23

    # Statistical measures
    significance_level: float      # 0-1 scale
    confidence_interval: Tuple[float, float]
```

## 🚀 Quick Start

### 1. Build the Bank

```bash
# Single symbol build
python build_price_level_bank.py \
    --symbol BTCUSDT \
    --timeframe 1h \
    --start-date 2023-01-01 \
    --end-date 2024-01-01

# Batch build from config
python build_price_level_bank.py --config build_config.json

# Build from symbols file
python build_price_level_bank.py --symbols-file symbols.txt
```

### 2. Query the Bank

```bash
# Get most significant levels
python query_price_level_bank.py --symbol BTCUSDT --top 10

# Query by price range
python query_price_level_bank.py --symbol BTCUSDT --price-range 45000 55000

# Export all levels
python query_price_level_bank.py --export --output levels.csv

# Show bank statistics
python query_price_level_bank.py --stats
```

### 3. Use in Feature Generation

The bank integrates automatically with historical feature generators:

```python
from feature_generation.categories.support_resistance import (
    HistoricalPriceLevelCrossingGenerator,
    HistoricalPriceLevelBounceGenerator,
    HistoricalVolumeAtPriceLevelGenerator
)

# These will automatically check the bank first
crossing_gen = HistoricalPriceLevelCrossingGenerator(level_pct=0.2, window=100)
bounce_gen = HistoricalPriceLevelBounceGenerator(level_pct=0.2, window=100)
volume_gen = HistoricalVolumeAtPriceLevelGenerator(level_pct=0.2, window=100)
```

## 🔧 Configuration

### Bank Configuration

```python
from feature_generation.core.price_level_bank import PriceLevelBankConfig

config = PriceLevelBankConfig(
    storage_path="./data/price_level_bank",
    enable_persistence=True,
    auto_save_interval=100,  # Save every N operations
    max_levels_per_symbol=10000,
    default_lookback_window=200,
    cache_size=1000,
    enable_compression=True
)
```

### Build Configuration

```json
{
  "builds": [
    {
      "symbol": "BTCUSDT",
      "timeframe": "1h",
      "start_date": "2023-01-01",
      "end_date": "2024-01-01",
      "level_pcts": [0.1, 0.2, 0.5, 1.0, 2.0]
    }
  ]
}
```

## 📈 Usage Examples

### Example 1: ML Training Data Preparation

```bash
# 1. Build bank with extensive historical analysis
python build_price_level_bank.py \
    --symbol BTCUSDT \
    --timeframe 1h \
    --start-date 2020-01-01 \
    --end-date 2024-01-01

# 2. Export training features
python query_price_level_bank.py \
    --symbol BTCUSDT \
    --format csv \
    --output training_features.csv

# 3. Use in ML training
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('training_features.csv')
X = data[['historical_crossings', 'historical_bounces', 'historical_volume']]
y = data['historical_success_rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### Example 2: Feature Generation Pipeline

```python
from feature_generation.core.feature_bank import FeatureBank

# Initialize feature bank (automatically includes price level bank)
bank = FeatureBank()

# Generate features using bank data
features = bank.generate_features(
    data=df,
    categories=['support_resistance'],
    symbol='BTCUSDT',
    timeframe='1h'
)
```

### Example 3: Custom Analysis

```python
from query_price_level_bank import PriceLevelBankQuery

query = PriceLevelBankQuery()

# Get most significant levels by session
for session in ['asian', 'european', 'us']:
    levels = query.query_levels(
        symbol='BTCUSDT',
        min_significance=0.7
    )

    session_levels = [l for l in levels if l.get('session_type') == session]
    print(f"{session.upper()} session: {len(session_levels)} significant levels")
```

## 🔍 Advanced Features

### Query Filters

The bank supports advanced querying:

```python
levels = bank.query_levels(
    symbol='BTCUSDT',
    timeframe='1h',
    min_price=40000,
    max_price=60000,
    min_significance=0.7,
    limit=50
)
```

### Time Decay Analysis

Levels include time-decayed importance:

```python
# Get levels sorted by recency-weighted importance
recent_levels = sorted(levels, key=lambda x: x.historical_time_decay, reverse=True)
```

### Multi-Timeframe Consistency

Check consistency across timeframes:

```python
# Compare 1h and 4h levels
h1_levels = bank.query_levels(symbol='BTCUSDT', timeframe='1h', limit=100)
h4_levels = bank.query_levels(symbol='BTCUSDT', timeframe='4h', limit=100)

# Find overlapping levels
overlapping = []
for h1_level in h1_levels:
    for h4_level in h4_levels:
        if abs(h1_level.price - h4_level.price) / h1_level.price < 0.01:  # Within 1%
            overlapping.append({'h1': h1_level, 'h4': h4_level})
```

## 💾 Storage and Performance

### Storage Structure

```
./data/price_level_bank/
├── price_level_bank.pkl          # Main bank data
├── price_level_bank.pkl.bak      # Backup file
├── logs/                         # Operation logs
└── exports/                      # Exported data
```

### Performance Optimization

- **Indexing**: Fast lookup by symbol, price, timeframe
- **Caching**: Recent queries cached for speed
- **Compression**: Optional data compression
- **Incremental Updates**: Only changed data is saved

### Memory Management

- Configurable cache sizes
- Automatic cleanup of old data
- Chunked processing for large datasets
- Streaming data processing

## 🔧 Troubleshooting

### Common Issues

1. **Bank not found**: Ensure bank is built before querying
   ```bash
   python build_price_level_bank.py --symbol BTCUSDT --timeframe 1h --start-date 2023-01-01 --end-date 2024-01-01
   ```

2. **No data for symbol**: Check available symbols
   ```bash
   python query_price_level_bank.py --stats
   ```

3. **Slow queries**: Optimize query parameters
   ```python
   # Use specific filters instead of broad queries
   levels = bank.query_levels(symbol='BTCUSDT', min_significance=0.8, limit=20)
   ```

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 📊 Monitoring and Maintenance

### Bank Statistics

```bash
python query_price_level_bank.py --stats
```

### Cleanup Old Data

```python
# Remove old levels to save space
old_levels = bank.query_levels(min_significance=0.1)
for level in old_levels:
    # Remove if too old or insignificant
    pass
```

### Backup and Recovery

The bank automatically creates backups. Manual backup:

```python
bank.save_to_disk('./backup/price_level_bank_backup.pkl')
```

## 🚀 Best Practices

1. **Build with sufficient history**: Use at least 1 year of data
2. **Use appropriate level percentages**: Start with 0.1%, 0.2%, 0.5%, 1.0%
3. **Regular updates**: Rebuild periodically with fresh data
4. **Monitor significance**: Focus on levels with >0.7 significance
5. **Combine timeframes**: Use multiple timeframes for validation

## 🔮 Future Enhancements

- Real-time bank updates from live data
- Machine learning-based level significance scoring
- Integration with order flow and microstructure data
- Advanced pattern recognition for level relationships
- Multi-asset correlation analysis

---

This system provides a solid foundation for price level analysis in ML trading systems, eliminating redundant calculations and providing consistent, high-quality historical tags for model training.