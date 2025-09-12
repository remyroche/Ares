# 🏷️ Regime Data Tagging Approach Guide

## Overview

The market analysis pipeline uses a **TAGGING approach** (not splitting) for regime data processing. This preserves temporal continuity and maximizes data utilization.

## Key Concepts

### Tagging vs Splitting

| Approach | Description | Data Retention | Lookback Preservation |
|----------|-------------|---------------|---------------------|
| **Tagging** ✅ | Single dataset with regime labels | **100%** | **Full** |
| Splitting ❌ | Separate files per regime | ~60-80% | Broken at boundaries |

### How Tagging Works

1. **Single Unified Dataset**: All regime data is stored in one file
2. **Regime Labels**: Each row has a `composite_cluster_id` column indicating its regime
3. **Context Preservation**: Temporal continuity is maintained across regime transitions
4. **Lookback Preservation**: Full lookback periods are available for all features

## Usage in Downstream Steps

### Step 4: Regime Data Tagging
```python
# Creates unified dataset with regime tags
unified_data = await step.split_data_by_regimes(symbol, exchange, timeframe, data_dir)
# Result: Single parquet file with 'composite_cluster_id' column
```

### Step 5: Labeling Per Regime
```python
# Uses regime handler to process tagged data
async with RegimeProcessingContext(symbol, exchange, timeframe, data_dir) as ctx:
    regime_data = ctx.get_regime_data(regime_id, preserve_context=True)
    # regime_data includes context rows with 'is_regime_context' flag
```

### Step 6: Feature Engineering Per Regime
```python
# Processes tagged data with context preservation
regime_data = ctx.get_regime_data(regime_id, preserve_context=True)
# Full lookback periods available for feature engineering
```

## Key Benefits

### 1. Data Retention
- **100% data retention** (no rows lost to splitting boundaries)
- **No boundary artifacts** from regime transitions
- **Maximum utilization** of available data

### 2. Lookback Preservation
- **Full lookback periods** maintained for all features
- **Trading indicators** work correctly across regime transitions
- **Feature engineering** has complete historical context

### 3. Temporal Continuity
- **Market context** preserved around regime changes
- **Smooth transitions** between regimes
- **Realistic trading conditions** maintained

### 4. Management Efficiency
- **Single dataset** to manage (not multiple files per regime)
- **Simplified pipeline** with unified data access
- **Consistent processing** across all steps

## Implementation Details

### Data Structure
```python
# Unified dataset structure
{
    'timestamp': pd.Timestamp,
    'open': float,
    'high': float,
    'low': float,
    'close': float,
    'volume': float,
    'composite_cluster_id': int,  # Regime tag
    # ... other features
}
```

### Context Preservation
```python
# When filtering by regime with context preservation
regime_data = regime_handler.filter_data_by_regime(
    data, 
    regime_id, 
    preserve_context=True,  # Include context rows
    context_window=100      # Rows before/after regime
)

# Result includes:
# - Regime rows: data[data['composite_cluster_id'] == regime_id]
# - Context rows: data around regime transitions
# - 'is_regime_context' flag: indicates context vs regime rows
```

### Usage Pattern
```python
# Standard pattern for downstream steps
async with RegimeProcessingContext(symbol, exchange, timeframe, data_dir) as ctx:
    regime_ids = ctx.regime_ids
    
    for regime_id in regime_ids:
        # Get regime data with context preservation
        regime_data = ctx.get_regime_data(regime_id, preserve_context=True)
        
        # Process regime data (has full lookback periods)
        result = await process_regime(regime_data, regime_id)
        
        # Save results
        await regime_handler.save_regime_results(result, ...)
```

## Best Practices

### 1. Always Use Context Preservation
```python
# ✅ Good: Preserves lookback periods
regime_data = ctx.get_regime_data(regime_id, preserve_context=True)

# ❌ Bad: Loses context and lookback periods
regime_data = ctx.get_regime_data(regime_id, preserve_context=False)
```

### 2. Handle Context Rows Appropriately
```python
# Check for context rows and handle appropriately
if 'is_regime_context' in regime_data.columns:
    context_mask = regime_data['is_regime_context']
    # Use context rows for lookback, but don't train on them
    training_data = regime_data[~context_mask]
```

### 3. Use Regime Handler Consistently
```python
# ✅ Good: Use regime handler for consistent processing
from .regime_handler import regime_handler

# ❌ Bad: Direct file access bypasses tagging benefits
data = pd.read_parquet('regime_file.parquet')
```

## Migration from Splitting

If you have existing code that expects split files:

### Before (Splitting)
```python
# Old approach: separate files per regime
regime_file = f'regime_{regime_id}_data.parquet'
regime_data = pd.read_parquet(regime_file)
```

### After (Tagging)
```python
# New approach: unified dataset with tags
unified_data = await regime_handler.load_unified_regime_data(symbol, exchange, timeframe, data_dir)
regime_data = regime_handler.filter_data_by_regime(unified_data, regime_id, preserve_context=True)
```

## Performance Considerations

### Large Regime Counts
- **Chunked processing** for 100+ regimes
- **Memory optimization** for large datasets
- **Vectorized operations** for statistics calculation

### Context Window Optimization
- **Automatic optimization** based on regime characteristics
- **Rare regimes**: Larger context windows
- **Common regimes**: Smaller context windows

## Troubleshooting

### Common Issues

1. **Missing composite_cluster_id column**
   - Ensure you're using the unified dataset from Step 4
   - Check that regime tagging was completed successfully

2. **Broken lookback periods**
   - Use `preserve_context=True` when filtering regime data
   - Ensure context window is appropriate for your features

3. **Data loss**
   - Verify you're using tagging approach, not splitting
   - Check that context preservation is enabled

### Validation
```python
# Validate tagging approach is working
benefits = regime_handler.show_tagging_benefits(data, regime_id)
print(f"Data retention: {benefits['tagged_approach']['data_retention']}")
```

## Conclusion

The tagging approach provides superior data utilization and temporal continuity compared to traditional splitting. Always use the regime handler and context preservation to maximize the benefits of this approach.