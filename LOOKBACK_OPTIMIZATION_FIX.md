# Lookback Optimization Fix

## Problem

The lookback optimization step (step 3) was skipping optimization with the message:
```
⚠️ No individual feature results available; skipping lookback optimization
```

**Root Cause**: The step expected pre-computed `individual_feature_results` from a previous optimization run, but the feature generation pipeline doesn't produce this artifact.

## Solution

Modified the lookback optimization step to compute feature importance on-the-fly from the merged features and labels when `individual_feature_results` is not available.

### Changes Made

**File**: `/Users/remyroche/Documents/Ares/src/training/steps/pre_training/feature_generation_period_lookback_optimization_step.py`

#### 1. Added New Method: `_compute_feature_importance_from_data`

```python
def _compute_feature_importance_from_data(self, merged_data: pd.DataFrame) -> Dict[str, Dict]:
    """Compute feature importance from merged features and targets.
    
    Uses RandomForestRegressor to compute feature importances, then organizes
    them by category (momentum, volatility, volume, trend, other) to create
    a structure compatible with _optimize_lookback_periods_by_category.
    """
```

**What it does**:
1. Identifies target columns (`target_long`, `target_short`)
2. Separates feature columns from metadata columns
3. Trains a simple Random Forest (50 trees, depth 10) on non-zero targets
4. Extracts feature importances
5. Categorizes features by name patterns (momentum, volatility, volume, trend)
6. Creates result structure compatible with existing optimization logic

#### 2. Modified Optimization Logic

**Before**:
```python
if artifacts.get('individual_feature_results'):
    # Run optimization
else:
    tprint("⚠️ No individual feature results available; skipping lookback optimization")
```

**After**:
```python
if artifacts.get('individual_feature_results'):
    # Use pre-computed results (preferred)
    tprint("🎯 Optimizing lookback periods using pre-computed feature results...")
    optimization_output = self._optimize_lookback_periods_by_category(...)
elif features is not None and not features.empty:
    # Compute importance on-the-fly (fallback)
    tprint("🎯 Computing feature importance from merged data for lookback optimization...")
    individual_results = self._compute_feature_importance_from_data(features)
    if individual_results:
        optimization_output = self._optimize_lookback_periods_by_category(...)
else:
    tprint("⚠️ No data available for lookback optimization; skipping")
```

## How It Works Now

### Data Flow

1. **Step 1 (Labeling)**: Creates `labeled_data` with `target_long`, `target_short`
2. **Step 2 (Feature Generation)**: Creates `generated_features` 
3. **Step 3 (Lookback Optimization)**: 
   - Loads and merges features + labels (via `_load_generated_features`)
   - Computes feature importance using Random Forest
   - Categorizes features by type
   - Runs lookback optimization on each category
   - Saves optimized lookback periods

### Feature Categorization

Features are automatically categorized based on name patterns:
- **Momentum**: `momentum`, `rsi`, `macd`
- **Volatility**: `volatility`, `atr`, `std`
- **Volume**: `volume`, `obv`
- **Trend**: `trend`, `ema`, `sma`
- **Other**: Everything else

### Output Structure

```python
{
    'category_optimizations': {
        'momentum': {
            'feature_name_1': {
                'optimal_lookback': 20,
                'performance_score': 0.045,
                'stability_score': 0.036,
                'feature_name': 'feature_name_1',
                'category': 'momentum'
            },
            ...
        },
        'volatility': {...},
        ...
    }
}
```

## Benefits

1. **No Pre-computation Required**: Works directly with pipeline outputs
2. **Automatic Feature Categorization**: Organizes features by type
3. **Fast Computation**: Simple RF with 50 trees, depth 10
4. **Backward Compatible**: Still uses pre-computed results if available
5. **Robust**: Handles missing targets, insufficient data gracefully

## Testing

To test the fix, run:
```bash
python3 src/launcher/ares_launcher.py --feature_generation_period_lookback_optimization_step --symbol ETHUSDT --execution-mode blank
```

**Expected Output**:
```
🎯 Computing feature importance from merged data for lookback optimization...
📊 Using 341 features and target 'target_long'
📊 Training on 244 samples with non-zero targets
✅ Computed importance for 341 features across 5 categories
✅ OPTIMIZED LOOKBACKS BY CATEGORY:
   📊 MOMENTUM:
      rsi_14: [20]
      macd_12_26_9: [20]
   📊 VOLATILITY:
      atr_14: [20]
      ...
```

## Notes

- The default lookback is set to 20 periods (can be optimized further)
- Feature importance is computed only on samples with non-zero targets
- Requires at least 50 samples with non-zero targets for reliable results
- Uses sklearn's RandomForestRegressor (ensure sklearn is installed)
