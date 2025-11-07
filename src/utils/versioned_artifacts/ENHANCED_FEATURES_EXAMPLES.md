# Enhanced Features Examples

This document demonstrates the enhanced features for operation-based column retrieval and multi-timeframe data handling.

## Feature 1: Column Tagging by Operation

### Problem
When adding 60 features from `final_feature_selection`, you want to later retrieve **only those 60 columns** without manually tracking column names.

### Solution
Use `add_columns_with_tags()` to tag columns by operation, then retrieve them with `get_columns_by_operation()`.

### Example: Adding Tagged Columns

```python
from src.utils.versioned_artifacts import VersionedArtifactStore
import pandas as pd
import numpy as np

# Initialize store
store = VersionedArtifactStore("versioned_artifacts/ETHUSDT_binance_long")

# Create base data
dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
base_data = pd.DataFrame({
    'close': np.random.randn(1000).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 1000)
}, index=dates)

# Add base data
base_view = store.add_data(base_data, version_name="market_data_v1")

# Later: Add 60 features from final_feature_selection
feature_dict = {
    f'feature_{i}': np.random.rand(1000)
    for i in range(60)
}

store.add_columns_with_tags(
    columns=feature_dict,
    version_name="market_data_v1",
    operation_name="final_feature_selection",
    tags={
        "feature_type": "selected",
        "source": "feature_selection_step",
        "num_features": 60
    }
)

print(f"Added {len(feature_dict)} features from final_feature_selection")
```

### Example: Retrieving Tagged Columns

```python
# Later in your code: Get ONLY the 60 features from final_feature_selection
selected_features = store.get_columns_by_operation("final_feature_selection")

print(f"Found {len(selected_features)} features from final_feature_selection")
# Output: Found 60 features from final_feature_selection

# Create a view with only these columns
features_view = store.get_view_by_operation("final_feature_selection")
features_df = features_view.materialize()

print(f"Features DataFrame shape: {features_df.shape}")
# Output: Features DataFrame shape: (1000, 60)

# Or manually select the columns
mask = ViewMask(column_mask=set(selected_features))
custom_view = store.get_view("market_data_v1", mask=mask)
```

## Example: Multiple Operations

```python
# Add technical indicators
technical_features = {
    'rsi': np.random.rand(1000),
    'macd': np.random.randn(1000),
    'bb_width': np.random.rand(1000)
}

store.add_columns_with_tags(
    columns=technical_features,
    operation_name="technical_indicators",
    tags={"feature_type": "technical", "category": "momentum"}
)

# Add regime features
regime_features = {
    'regime_trend': np.random.randint(0, 3, 1000),
    'regime_volatility': np.random.randint(0, 2, 1000)
}

store.add_columns_with_tags(
    columns=regime_features,
    operation_name="regime_detection",
    tags={"feature_type": "regime", "category": "market_state"}
)

# Later: Retrieve columns by operation
final_features = store.get_columns_by_operation("final_feature_selection")
tech_features = store.get_columns_by_operation("technical_indicators")
regime_cols = store.get_columns_by_operation("regime_detection")

print(f"Final features: {len(final_features)}")      # 60
print(f"Technical features: {len(tech_features)}")   # 3
print(f"Regime features: {len(regime_cols)}")        # 2

# Create view with specific feature sets
all_features = final_features + tech_features
features_view = store.get_view().select_columns(all_features)
combined_features = features_view.materialize()

print(f"Combined features shape: {combined_features.shape}")
# Output: Combined features shape: (1000, 63)
```

## Example: Querying by Tags

```python
# Get all columns tagged as "technical"
technical_cols = store.get_columns_by_tag("feature_type", "technical")

# Get all momentum features
momentum_cols = store.get_columns_by_tag("category", "momentum")

print(f"Technical columns: {technical_cols}")
print(f"Momentum columns: {momentum_cols}")
```

## Feature 2: Multi-Timeframe Data Handling

### Problem
You have data at different timeframes (e.g., 15m, 1h, 4h) and want to:
1. Use the lower timeframe (15m) as base
2. Forward-fill higher timeframe data so it's available at 15m resolution

### Solution
Use `add_multi_timeframe_data()` to automatically align and forward-fill different timeframes.

### Example: Combining Multiple Timeframes

```python
# Create base data at 15m timeframe
dates_15m = pd.date_range('2024-01-01', periods=4*24*30, freq='15min')  # 30 days
base_15m = pd.DataFrame({
    'close': np.random.randn(len(dates_15m)).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, len(dates_15m)),
    'high': np.random.randn(len(dates_15m)).cumsum() + 102,
    'low': np.random.randn(len(dates_15m)).cumsum() + 98
}, index=dates_15m)

print(f"Base 15m data shape: {base_15m.shape}")
# Output: Base 15m data shape: (2880, 4)

# Create 1h data (4x less points)
dates_1h = pd.date_range('2024-01-01', periods=24*30, freq='1h')
hourly_data = pd.DataFrame({
    'trend': np.random.choice(['up', 'down', 'sideways'], len(dates_1h)),
    'regime': np.random.choice(['bull', 'bear', 'neutral'], len(dates_1h)),
    'volatility_1h': np.random.rand(len(dates_1h))
}, index=dates_1h)

print(f"Hourly data shape: {hourly_data.shape}")
# Output: Hourly data shape: (720, 3)

# Create 4h data (16x less points)
dates_4h = pd.date_range('2024-01-01', periods=6*30, freq='4h')
four_hour_data = pd.DataFrame({
    'macro_trend': np.random.choice(['bullish', 'bearish', 'neutral'], len(dates_4h)),
    'market_phase': np.random.choice(['accumulation', 'distribution', 'markup'], len(dates_4h)),
    'strength_4h': np.random.rand(len(dates_4h))
}, index=dates_4h)

print(f"4h data shape: {four_hour_data.shape}")
# Output: 4h data shape: (180, 3)

# Combine all timeframes with forward-fill
multi_tf_view = store.add_multi_timeframe_data(
    base_data=base_15m,
    higher_tf_data={
        "1h": hourly_data,
        "4h": four_hour_data
    },
    version_name="multi_timeframe_features",
    forward_fill=True,
    metadata={
        "description": "Multi-timeframe features with forward-fill",
        "base_tf": "15m"
    }
)

# Materialize the combined data
combined_df = multi_tf_view.materialize()

print(f"\nCombined data shape: {combined_df.shape}")
# Output: Combined data shape: (2880, 10)
# Columns: close, volume, high, low (from 15m)
#          + trend_1h, regime_1h, volatility_1h (from 1h, forward-filled)
#          + macro_trend_4h, market_phase_4h, strength_4h (from 4h, forward-filled)

print(f"Columns: {list(combined_df.columns)}")

# Verify forward-fill worked
# Each 1h value should appear in 4 consecutive 15m rows
print("\nSample of forward-filled 1h data:")
print(combined_df[['close', 'trend_1h', 'macro_trend_4h']].head(20))
```

### Example: Retrieving Columns by Timeframe

```python
# Get only the 1h columns
hourly_cols = store.get_columns_by_timeframe("1h", version_name="multi_timeframe_features")
print(f"Hourly columns: {hourly_cols}")
# Output: Hourly columns: ['trend_1h', 'regime_1h', 'volatility_1h']

# Get only the 4h columns
four_hour_cols = store.get_columns_by_timeframe("4h", version_name="multi_timeframe_features")
print(f"4h columns: {four_hour_cols}")
# Output: 4h columns: ['macro_trend_4h', 'market_phase_4h', 'strength_4h']

# Get base timeframe columns
base_cols = store.get_columns_by_timeframe("base", version_name="multi_timeframe_features")
print(f"Base columns: {base_cols}")
# Output: Base columns: ['close', 'volume', 'high', 'low']

# Create view with only higher timeframe features
higher_tf_cols = hourly_cols + four_hour_cols
higher_tf_view = store.get_view("multi_timeframe_features").select_columns(higher_tf_cols)
higher_tf_df = higher_tf_view.materialize()

print(f"\nHigher timeframe features shape: {higher_tf_df.shape}")
# Output: Higher timeframe features shape: (2880, 6)
```

## Example: Real-World Feature Engineering Pipeline

```python
# Step 1: Start with raw market data (15m)
dates_15m = pd.date_range('2024-01-01', periods=4*24*30, freq='15min')
raw_data = pd.DataFrame({
    'open': np.random.randn(len(dates_15m)).cumsum() + 100,
    'high': np.random.randn(len(dates_15m)).cumsum() + 102,
    'low': np.random.randn(len(dates_15m)).cumsum() + 98,
    'close': np.random.randn(len(dates_15m)).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, len(dates_15m))
}, index=dates_15m)

view = store.add_data(raw_data, version_name="feature_pipeline", metadata={"step": 0})

# Step 2: Add basic technical indicators
basic_tech = {
    'returns': np.random.randn(len(dates_15m)),
    'log_volume': np.random.randn(len(dates_15m))
}
store.add_columns_with_tags(
    columns=basic_tech,
    version_name="feature_pipeline",
    operation_name="basic_features",
    tags={"step": 1, "type": "basic"}
)

# Step 3: Add advanced technical indicators
advanced_tech = {f'adv_feature_{i}': np.random.rand(len(dates_15m)) for i in range(20)}
store.add_columns_with_tags(
    columns=advanced_tech,
    operation_name="advanced_technical",
    tags={"step": 2, "type": "advanced"}
)

# Step 4: Add feature interactions
interactions = {f'interaction_{i}': np.random.rand(len(dates_15m)) for i in range(30)}
store.add_columns_with_tags(
    columns=interactions,
    operation_name="feature_interactions",
    tags={"step": 3, "type": "interactions"}
)

# Step 5: Final feature selection (60 features)
selected_features = {f'selected_feature_{i}': np.random.rand(len(dates_15m)) for i in range(60)}
store.add_columns_with_tags(
    columns=selected_features,
    operation_name="final_feature_selection",
    tags={"step": 4, "type": "selected", "final": True}
)

# Now retrieve specific feature sets for training
print("\n=== Retrieving Feature Sets ===")

# Get only final selected features for training
final_features = store.get_columns_by_operation("final_feature_selection")
print(f"Final selected features: {len(final_features)}")

# Create training dataset with only selected features + target
training_cols = final_features + ['close']  # close as target
training_view = store.get_view("feature_pipeline").select_columns(training_cols)
training_df = training_view.materialize()

print(f"Training dataset shape: {training_df.shape}")
# Output: Training dataset shape: (2880, 61)  # 60 features + 1 target

# Get feature interaction columns for analysis
interactions_cols = store.get_columns_by_operation("feature_interactions")
print(f"Interaction features: {len(interactions_cols)}")

# Query changelog to see feature engineering timeline
changes = store.get_changelog(version_name="feature_pipeline")
print(f"\n=== Feature Engineering Timeline ===")
for change in changes:
    if change.change_type.value == "update_columns":
        op = change.metadata.get('operation', 'unknown')
        num_cols = change.metadata.get('num_columns', 0)
        print(f"{change.timestamp.strftime('%Y-%m-%d %H:%M:%S')}: "
              f"Added {num_cols} columns from '{op}'")
```

## Example: Multi-Timeframe + Operations Combined

```python
# Combine multi-timeframe data with operation tagging

# Base 15m data
dates_15m = pd.date_range('2024-01-01', periods=2880, freq='15min')
base_15m = pd.DataFrame({
    'close': np.random.randn(2880).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 2880)
}, index=dates_15m)

# 1h regime data
dates_1h = pd.date_range('2024-01-01', periods=720, freq='1h')
regime_1h = pd.DataFrame({
    'regime': np.random.choice(['bull', 'bear', 'neutral'], 720)
}, index=dates_1h)

# 4h trend data
dates_4h = pd.date_range('2024-01-01', periods=180, freq='4h')
trend_4h = pd.DataFrame({
    'trend': np.random.choice(['up', 'down', 'sideways'], 180)
}, index=dates_4h)

# Add multi-timeframe data
store.add_multi_timeframe_data(
    base_data=base_15m,
    higher_tf_data={
        "1h": regime_1h,
        "4h": trend_4h
    },
    version_name="full_pipeline"
)

# Add 15m-specific features
features_15m = {f'feature_15m_{i}': np.random.rand(2880) for i in range(20)}
store.add_columns_with_tags(
    columns=features_15m,
    version_name="full_pipeline",
    operation_name="features_15m",
    tags={"timeframe": "15m", "type": "technical"}
)

# Retrieve different feature sets
base_cols = store.get_columns_by_timeframe("base", "full_pipeline")
hourly_cols = store.get_columns_by_timeframe("1h", "full_pipeline")
four_hour_cols = store.get_columns_by_timeframe("4h", "full_pipeline")
features_15m_cols = store.get_columns_by_operation("features_15m", "full_pipeline")

print(f"Base columns: {len(base_cols)}")
print(f"1h columns: {len(hourly_cols)}")
print(f"4h columns: {len(four_hour_cols)}")
print(f"15m features: {len(features_15m_cols)}")

# Create different views for different purposes

# 1. Training view: All features
all_features = base_cols + features_15m_cols
training_view = store.get_view("full_pipeline").select_columns(all_features)

# 2. Analysis view: Only higher timeframe features
analysis_view = store.get_view("full_pipeline").select_columns(hourly_cols + four_hour_cols)

# 3. Base view: Only raw market data
base_view = store.get_view("full_pipeline").select_columns(base_cols)

print(f"\nTraining view columns: {training_view.materialize().shape[1]}")
print(f"Analysis view columns: {analysis_view.materialize().shape[1]}")
print(f"Base view columns: {base_view.materialize().shape[1]}")
```

## Summary

### Key Benefits

1. **Operation-Based Retrieval**
   - Tag columns when adding them
   - Retrieve specific feature sets by operation name
   - No manual tracking of column names
   - Perfect for large feature sets (60+ features)

2. **Multi-Timeframe Support**
   - Automatically align different timeframes
   - Forward-fill higher timeframe data
   - Track which columns come from which timeframe
   - Retrieve columns by timeframe

3. **Combined Power**
   - Mix multi-timeframe data with operation tagging
   - Create complex feature engineering pipelines
   - Easily retrieve specific feature sets for different purposes
   - Full audit trail of all operations

### API Quick Reference

```python
# Add columns with tags
store.add_columns_with_tags(columns, operation_name="feature_selection", tags={...})

# Retrieve by operation
columns = store.get_columns_by_operation("feature_selection")

# Retrieve by tag
columns = store.get_columns_by_tag("type", "technical")

# Get view by operation
view = store.get_view_by_operation("feature_selection")

# Add multi-timeframe data
store.add_multi_timeframe_data(base_data, higher_tf_data={...})

# Retrieve by timeframe
columns = store.get_columns_by_timeframe("1h")
```
