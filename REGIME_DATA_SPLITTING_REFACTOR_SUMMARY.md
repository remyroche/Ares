# Regime Data Splitting Refactor Summary

## Overview
This document summarizes the refactoring of the regime data splitting logic in `src/steps/` to use labels instead of creating separate files per regime/cluster. This change ensures that trading indicators maintain the necessary lookback periods and temporal continuity.

## Problem Statement
The previous implementation created separate parquet files for each regime/cluster, which caused issues:
- **Trading indicators lost lookback periods** when data was split into regime-specific files
- **Temporal continuity was broken** between different regimes
- **Multiple file management** became complex and error-prone
- **Memory inefficiency** due to loading multiple small files

## Solution: Unified Dataset with Regime Labels
The new approach creates a **single unified dataset** with regime information stored as labels in the `composite_cluster_id` column, rather than splitting data into separate files.

### Key Benefits
1. **Maintains temporal continuity** - All data remains in chronological order
2. **Preserves lookback periods** - Trading indicators can access historical data across regime boundaries
3. **Simplified file management** - Single dataset instead of multiple regime files
4. **Better memory efficiency** - Single file load instead of multiple small files
5. **Easier regime-aware processing** - Filter by `composite_cluster_id` for regime-specific operations

## Files Modified

### 1. `src/training/steps/step4_regime_data_splitting.py`
**Changes:**
- Replaced regime-specific file creation with unified dataset approach
- Creates single file: `{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet`
- Adds regime labels mapping: `{exchange}_{symbol}_{timeframe}_regime_labels.json`
- Adds regime statistics: `{exchange}_{symbol}_{timeframe}_regime_statistics.json`
- Maintains temporal ordering for proper lookback periods

**New Output Structure:**
```
data/training/
├── {exchange}_{symbol}_{timeframe}_unified_regime_data.parquet  # Main dataset with regime labels
├── {exchange}_{symbol}_{timeframe}_regime_labels.json          # Regime mapping and usage instructions
├── {exchange}_{symbol}_{timeframe}_regime_statistics.json      # Statistical summary per regime
└── {exchange}_{symbol}_{timeframe}_regime_metadata.json        # Overall metadata and approach description
```

### 2. `src/training/steps/step8_regime_data_splitting.py`
**Changes:**
- Updated to match step4 approach for consistency
- Creates unified dataset instead of regime splits
- Maintains backward compatibility with legacy approach
- Enhanced MLflow logging for unified approach

### 3. `src/training/steps/step6_feature_engineering.py`
**Changes:**
- Updated `_load_regime_data()` to load unified regime dataset first
- Falls back to legacy approach for backward compatibility
- Enhanced regime data merging logic to handle both unified and legacy approaches
- Maintains temporal continuity for feature engineering

### 4. `src/training/steps/step9_hmm_based_training.py`
**Changes:**
- Updated `_load_hmm_composite_regime_data()` to use unified dataset
- Creates regime splits dynamically from unified dataset (80/10/10 train/val/test)
- Falls back to legacy approach for backward compatibility
- Maintains regime-specific training capabilities

### 5. `src/training/steps/step11_analyst_creation.py`
**Changes:**
- Updated `_load_regime_splits()` to use unified dataset
- Creates regime-specific data from unified dataset
- Falls back to legacy approach for backward compatibility
- Maintains analyst model creation per regime

### 6. `src/training/enhanced_training_manager.py`
**Changes:**
- Updated artifact patterns to reflect new file structure
- Changed from regime split files to unified dataset files
- Maintains dependency tracking for new approach

### 7. `src/utils/step_dependency_validator.py`
**Changes:**
- Updated required files for step5_regime_data_splitting
- Changed from regime split files to unified dataset files
- Maintains validation for new approach

## New Data Structure

### Unified Regime Dataset Format
```python
# Main dataset structure
{
    'timestamp': [...],           # Chronological timestamps
    'open': [...],               # OHLCV data
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...],
    'composite_cluster_id': [...], # Regime labels (0, 1, 2, ...)
    # ... other features and indicators
}
```

### Regime Labels Mapping
```json
{
    "regime_column": "composite_cluster_id",
    "regime_ids": [0, 1, 2, 3, 4],
    "total_regimes": 5,
    "data_shape": [100000, 50],
    "timestamp_range": {
        "start": "2023-01-01T00:00:00",
        "end": "2023-12-31T23:59:59"
    },
    "usage_instructions": {
        "description": "Load the unified dataset and filter by composite_cluster_id for regime-specific processing",
        "example": "regime_data = data[data['composite_cluster_id'] == regime_id]",
        "benefits": [
            "Maintains temporal continuity for trading indicators",
            "Preserves lookback periods",
            "Eliminates need for multiple file management",
            "Enables regime-aware processing with single dataset"
        ]
    }
}
```

## Usage Examples

### Loading Regime Data
```python
# Load unified dataset
unified_data = pd.read_parquet("data/training/BINANCE_ETHUSDT_1m_unified_regime_data.parquet")

# Filter for specific regime
regime_0_data = unified_data[unified_data['composite_cluster_id'] == 0]

# Filter for multiple regimes
regime_1_2_data = unified_data[unified_data['composite_cluster_id'].isin([1, 2])]

# Get all unique regimes
unique_regimes = unified_data['composite_cluster_id'].unique()
```

### Regime-Aware Processing
```python
# Process each regime while maintaining temporal context
for regime_id in unified_data['composite_cluster_id'].unique():
    regime_data = unified_data[unified_data['composite_cluster_id'] == regime_id]
    
    # Trading indicators now have full lookback period
    regime_data['sma_20'] = regime_data['close'].rolling(20).mean()
    regime_data['rsi_14'] = calculate_rsi(regime_data['close'], 14)
    
    # Process regime-specific features
    process_regime_features(regime_data)
```

## Backward Compatibility
The refactoring maintains backward compatibility:
- **Legacy regime files** are still supported as fallback
- **Existing pipelines** continue to work
- **Gradual migration** is possible
- **No breaking changes** to existing functionality

## Migration Path
1. **Run step4/step8** to create unified regime dataset
2. **Update dependent steps** to use new approach
3. **Test functionality** with unified dataset
4. **Remove legacy files** once migration is complete

## Testing
To test the new approach:
```bash
# Run regime data splitting with new approach
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --step step4_regime_data_splitting

# Verify unified dataset creation
ls -la data/training/*_unified_regime_data.parquet

# Test regime-aware processing
python -c "
import pandas as pd
data = pd.read_parquet('data/training/BINANCE_ETHUSDT_1m_unified_regime_data.parquet')
print(f'Dataset shape: {data.shape}')
print(f'Regimes: {data.composite_cluster_id.unique()}')
print(f'Date range: {data.timestamp.min()} to {data.timestamp.max()}')
"
```

## Performance Impact
- **Memory usage**: Reduced due to single file load
- **I/O operations**: Fewer file operations
- **Processing speed**: Faster due to vectorized operations on unified dataset
- **Storage efficiency**: Better compression on single large file

## Future Enhancements
1. **Regime transition features** - Analyze regime changes over time
2. **Cross-regime correlations** - Study relationships between regimes
3. **Regime-aware feature selection** - Optimize features per regime
4. **Dynamic regime detection** - Real-time regime identification

## Conclusion
This refactoring significantly improves the regime data handling by:
- **Eliminating the lookback period issue** that made trading indicators unusable
- **Maintaining temporal continuity** across regime boundaries
- **Simplifying data management** with a single unified dataset
- **Enabling more sophisticated regime-aware processing**

The new approach ensures that all subsequent steps can access regime information while maintaining the temporal context necessary for effective trading strategies and technical analysis.