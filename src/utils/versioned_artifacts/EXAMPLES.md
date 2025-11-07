# Versioned Artifact Store - Practical Examples

This document provides practical, real-world examples for using the Versioned Artifact Store.

## Example 1: Basic Artifact Storage and Retrieval

```python
import pandas as pd
import numpy as np
from src.utils.versioned_artifacts import VersionedArtifactStore

# Create sample market data
dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
market_data = pd.DataFrame({
    'close': np.random.randn(1000).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 1000),
    'high': np.random.randn(1000).cumsum() + 102,
    'low': np.random.randn(1000).cumsum() + 98
}, index=dates)

# Initialize store
store = VersionedArtifactStore(
    store_path="versioned_artifacts/ETHUSDT_binance_long_analyst",
    auto_version=True,
    enable_row_versioning=True
)

# Add initial data
initial_view = store.add_data(
    data=market_data,
    version_name="market_data_v1",
    metadata={
        'source': 'binance',
        'symbol': 'ETHUSDT',
        'timeframe': '15m',
        'description': 'Initial market data download'
    }
)

print(f"Stored {len(market_data)} rows")
print(f"View: {initial_view}")

# Retrieve data
retrieved_data = initial_view.materialize()
print(f"\nRetrieved shape: {retrieved_data.shape}")
```

## Example 2: Efficient Subsetting with Views

```python
# Create view with only specific columns (no full load)
price_view = initial_view.select_columns(['close', 'volume'])

# Add row filter (still no load)
high_volume_view = price_view.filter(
    lambda df: df['volume'] > df['volume'].quantile(0.75)
)

# Only now is data loaded
high_volume_data = high_volume_view.materialize()
print(f"High volume periods: {len(high_volume_data)} rows")

# Compare memory usage
full_size = market_data.memory_usage(deep=True).sum() / 1024**2
subset_size = high_volume_data.memory_usage(deep=True).sum() / 1024**2
print(f"Full data: {full_size:.2f} MB")
print(f"Subset: {subset_size:.2f} MB")
print(f"Savings: {(1 - subset_size/full_size)*100:.1f}%")
```

## Example 3: Row-Level Updates for Model Predictions

```python
# Simulate ML model predictions
predictions = np.random.rand(1000)

# Add predictions as new column
store.add_columns(
    columns={'analyst_prediction': predictions},
    version_name="market_data_v1"
)

# Later, update predictions for specific rows (e.g., corrections)
error_indices = [100, 150, 200, 250]
corrected_predictions = np.array([0.75, 0.82, 0.65, 0.88])

store.update_rows(
    row_indices=error_indices,
    columns=['analyst_prediction'],
    new_values={'analyst_prediction': corrected_predictions},
    version_name="market_data_v1"
)

print(f"Updated {len(error_indices)} rows")

# Track what changed
changes = store.get_changelog(version_name="market_data_v1")
print(f"\nTotal changes: {len(changes)}")
for change in changes[-3:]:  # Last 3 changes
    print(f"  {change.change_type.value}: {change.affected_rows} rows, {change.affected_columns}")
```

## Example 4: Artifact Chaining in Training Pipeline

```python
# Simulating the artifact chaining flow from ARTIFACT_CHAINING_GUIDE.md

# Step 1: Analyst Base Models
analyst_features = pd.DataFrame({
    'rsi': np.random.rand(1000),
    'macd': np.random.rand(1000),
    'bb_width': np.random.rand(1000)
}, index=dates)

analyst_predictions = pd.DataFrame({
    'lgbm_pred': np.random.rand(1000),
    'catboost_pred': np.random.rand(1000),
    'nn_pred': np.random.rand(1000)
}, index=dates)

# Store analyst features
analyst_view = store.add_data(
    data=analyst_features,
    version_name="analyst_base_features",
    metadata={'step': 'analyst_base', 'phase': 1}
)

# Store analyst predictions
analyst_pred_view = store.add_data(
    data=analyst_predictions,
    version_name="analyst_base_predictions",
    metadata={'step': 'analyst_base', 'phase': 1}
)

# Step 2: Analyst Ensemble
ensemble_prediction = analyst_predictions.mean(axis=1).values

store.add_columns(
    columns={'analyst_ensemble': ensemble_prediction},
    version_name="analyst_base_predictions"
)

# Step 3: Tactician Base Models (uses analyst ensemble)
# Only update during trading hours (example mask)
trading_hours = (dates.hour >= 9) & (dates.hour <= 17)
trading_indices = np.where(trading_hours)[0]

tactician_predictions = np.random.rand(len(trading_indices))

store.update_rows(
    row_indices=trading_indices.tolist(),
    columns=['tactician_signal'],
    new_values={'tactician_signal': tactician_predictions},
    version_name="analyst_base_predictions",
    create_new_version=True,
    new_version_name="full_pipeline_predictions"
)

# Step 4: Get final combined view
final_view = store.get_view("full_pipeline_predictions")
final_data = final_view.materialize()

print(f"\nFinal pipeline data shape: {final_data.shape}")
print(f"Columns: {list(final_data.columns)}")
print(f"Tactician signals: {final_data['tactician_signal'].notna().sum()} rows")
```

## Example 5: Combining Multiple Artifact Versions

```python
# Create separate versions for different model runs
model_runs = []

for i in range(3):
    run_predictions = pd.DataFrame({
        f'model_{i}_pred': np.random.rand(1000),
        f'model_{i}_confidence': np.random.rand(1000)
    }, index=dates)

    view = store.add_data(
        data=run_predictions,
        version_name=f"model_run_{i}",
        metadata={'run_id': i, 'timestamp': pd.Timestamp.now()}
    )
    model_runs.append(view)

# Combine all model runs
combined_view = store.combine_views(
    views=model_runs,
    strategy="merge",
    how="outer"
)

combined_data = combined_view.materialize()
print(f"\nCombined data shape: {combined_data.shape}")
print(f"Columns from all runs: {list(combined_data.columns)}")

# Calculate ensemble from combined data
ensemble_cols = [c for c in combined_data.columns if 'pred' in c]
ensemble = combined_data[ensemble_cols].mean(axis=1)
print(f"Ensemble mean: {ensemble.mean():.4f}")
```

## Example 6: Time-Travel Queries and Audit Trail

```python
from datetime import datetime, timedelta

# Simulate changes over time
for hour in range(5):
    # Update some rows
    update_indices = np.random.choice(1000, size=50, replace=False)
    update_values = np.random.rand(50)

    store.update_rows(
        row_indices=update_indices.tolist(),
        columns=['analyst_prediction'],
        new_values={'analyst_prediction': update_values},
        version_name="market_data_v1"
    )

    # Simulate time passing
    import time
    time.sleep(0.1)

# Query changes in the last hour
one_hour_ago = datetime.now() - timedelta(hours=1)
recent_changes = store.get_changelog(from_time=one_hour_ago)

print(f"\nChanges in last hour: {len(recent_changes)}")
for change in recent_changes[:5]:
    print(f"  {change.timestamp.strftime('%H:%M:%S')}: "
          f"{change.change_type.value} - "
          f"{len(change.affected_rows) if isinstance(change.affected_rows, list) else change.affected_rows} rows")

# Export audit trail
store.changelog.export_to_csv("audit_trail_example.csv")
print("\nAudit trail exported to: audit_trail_example.csv")
```

## Example 7: Row Version Tracking and Rollback

```python
# Track changes to specific rows
critical_rows = [100, 200, 300]

for row_idx in critical_rows:
    # Get version history
    history = store.row_tracker.get_version_history(row_idx)

    print(f"\nRow {row_idx} version history:")
    for version in history[-3:]:  # Last 3 versions
        print(f"  {version.timestamp.strftime('%H:%M:%S')}: {version.changes}")

# Rollback a specific row
if len(history) > 1:
    previous_version = history[-2]
    store.row_tracker.rollback_row(
        row_index=100,
        version_id=previous_version.version_id
    )
    print(f"\nRolled back row 100 to version {previous_version.version_id}")

# Reconstruct row at specific version
row_data = store.row_tracker.reconstruct_row(
    row_index=100,
    columns=['close', 'volume', 'analyst_prediction']
)
print(f"Reconstructed row 100: {row_data}")
```

## Example 8: Using with BaseStep (Adapter)

```python
from src.utils.versioned_artifacts import VersionedArtifactAdapter
from src.training.steps.base_step import BaseStep

class ExampleTrainingStep(BaseStep):
    def __init__(self, step_name: str):
        super().__init__(step_name)

        # Initialize versioned store adapter
        self.versioned_store = VersionedArtifactAdapter(
            store_dir="versioned_artifacts",
            symbol="ETHUSDT",
            exchange="binance",
            direction="long",
            model="analyst"
        )

    async def execute(self, config):
        # Load market data (traditional way)
        market_data = self._get_artifact("market_data")

        # Generate predictions
        predictions = self._generate_predictions(market_data)

        # Save using versioned store
        self.versioned_store.save(
            data=predictions,
            artifact_name="analyst_predictions",
            metadata={
                'step': self.step_name,
                'symbol': config['symbol'],
                'timeframe': config['timeframe']
            }
        )

        # Later, retrieve for ensemble training
        saved_predictions = self.versioned_store.get_artifact("analyst_predictions")

        # Use view for efficient subsetting
        high_confidence_view = self.versioned_store.get_view(
            artifact_name="analyst_predictions",
            columns=['prediction', 'confidence']
        ).filter(lambda df: df['confidence'] > 0.8)

        high_confidence_data = high_confidence_view.materialize()

        return {
            'success': True,
            'artifacts': ['analyst_predictions'],
            'metrics': {
                'total_predictions': len(predictions),
                'high_confidence': len(high_confidence_data)
            }
        }

    def _generate_predictions(self, data):
        # Placeholder for actual prediction logic
        return pd.DataFrame({
            'prediction': np.random.rand(len(data)),
            'confidence': np.random.rand(len(data))
        }, index=data.index)
```

## Example 9: ViewMask Boolean Operations

```python
from src.utils.versioned_artifacts import ViewMask

# Create masks for different conditions
dates_array = pd.date_range('2024-01-01', periods=1000, freq='15min')

# Trading hours mask
trading_hours = (dates_array.hour >= 9) & (dates_array.hour <= 17)
trading_mask = ViewMask(
    row_mask=trading_hours.values,
    name="trading_hours"
)

# High volume mask
view = store.get_view("market_data_v1")
data = view.materialize()
high_volume = data['volume'] > data['volume'].quantile(0.75)
volume_mask = ViewMask(
    row_mask=high_volume.values,
    name="high_volume"
)

# Combine masks
# Only trading hours with high volume
combined_mask = trading_mask & volume_mask

# Or either condition
either_mask = trading_mask | volume_mask

# Invert mask (non-trading hours)
non_trading_mask = ~trading_mask

print(f"Trading hours: {trading_mask.num_rows} rows")
print(f"High volume: {volume_mask.num_rows} rows")
print(f"Trading + High volume: {combined_mask.num_rows} rows")
print(f"Either condition: {either_mask.num_rows} rows")
print(f"Non-trading hours: {non_trading_mask.num_rows} rows")

# Use combined mask
filtered_view = store.get_view("market_data_v1", mask=combined_mask)
filtered_data = filtered_view.materialize()
print(f"\nFiltered data shape: {filtered_data.shape}")
```

## Example 10: Performance Comparison

```python
import time

# Traditional approach: Load full data, then filter
print("Traditional Approach:")
start = time.time()
full_data = store.get_view("market_data_v1").materialize()
subset1 = full_data[full_data['volume'] > full_data['volume'].quantile(0.75)]
subset1 = subset1[['close', 'volume']]
traditional_time = time.time() - start
print(f"  Time: {traditional_time:.4f}s")
print(f"  Memory: {subset1.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Versioned approach: Use views with lazy evaluation
print("\nVersioned Approach:")
start = time.time()
view = store.get_view("market_data_v1")
subset2 = view.select_columns(['close', 'volume']).filter(
    lambda df: df['volume'] > df['volume'].quantile(0.75)
).materialize()
versioned_time = time.time() - start
print(f"  Time: {versioned_time:.4f}s")
print(f"  Memory: {subset2.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\nSpeedup: {traditional_time/versioned_time:.2f}x")
print(f"Results match: {subset1.equals(subset2)}")
```

## Example 11: Incremental Feature Engineering

```python
# Start with base features
base_features = pd.DataFrame({
    'returns': np.random.randn(1000),
    'volume_change': np.random.randn(1000)
}, index=dates)

view = store.add_data(
    data=base_features,
    version_name="features_incremental",
    metadata={'feature_set': 'base'}
)

# Day 1: Add technical indicators
technical_features = {
    'rsi': np.random.rand(1000),
    'macd': np.random.randn(1000)
}
store.add_columns(technical_features, "features_incremental")
print("Day 1: Added technical indicators")

# Day 2: Add volatility features
volatility_features = {
    'realized_vol': np.random.rand(1000),
    'implied_vol': np.random.rand(1000)
}
store.add_columns(volatility_features, "features_incremental")
print("Day 2: Added volatility features")

# Day 3: Add ML-derived features
ml_features = {
    'embedding_1': np.random.randn(1000),
    'embedding_2': np.random.randn(1000),
    'embedding_3': np.random.randn(1000)
}
store.add_columns(ml_features, "features_incremental")
print("Day 3: Added ML embeddings")

# Get all features
all_features = store.get_view("features_incremental").materialize()
print(f"\nTotal features: {len(all_features.columns)}")
print(f"Features: {list(all_features.columns)}")

# Query changelog to see feature addition timeline
changes = store.get_changelog(version_name="features_incremental")
for change in changes:
    if change.change_type.value == "update_columns":
        print(f"  {change.timestamp.strftime('%Y-%m-%d')}: "
              f"Added {len(change.affected_columns)} features")
```

## Example 12: Statistics and Monitoring

```python
# Get comprehensive statistics
stats = store.get_statistics()

print("Store Statistics:")
print(f"  Path: {stats['store_path']}")
print(f"  Versions: {stats['num_versions']}")
print(f"  Current: {stats['current_version']}")
print(f"  HDF5 size: {stats['h5_file_size_mb']:.2f} MB")

print("\nChangelog Statistics:")
changelog_stats = stats['changelog']
print(f"  Total changes: {changelog_stats['total_changes']}")
print(f"  Changes by type: {changelog_stats['changes_by_type']}")

if 'row_versioning' in stats:
    print("\nRow Versioning Statistics:")
    row_stats = stats['row_versioning']
    print(f"  Tracked rows: {row_stats['total_rows_tracked']}")
    print(f"  Total versions: {row_stats['total_versions']}")
    print(f"  Avg versions/row: {row_stats['average_versions_per_row']:.2f}")

# Get detailed changelog statistics
detailed_stats = store.changelog.get_statistics()
print("\nDetailed Changelog:")
print(f"  Time range: {detailed_stats['time_range']['earliest']} to {detailed_stats['time_range']['latest']}")
print(f"  Top versions by changes: {detailed_stats['top_versions']}")
```

These examples demonstrate the full range of capabilities of the Versioned Artifact Store system!
