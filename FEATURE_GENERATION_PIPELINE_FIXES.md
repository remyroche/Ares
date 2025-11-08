# Feature Generation Pipeline Fixes

## Date: 2025-11-08

## Issues Fixed

### 1. Missing Step Registration in Launcher
**Issue**: `feature_generation_feature_generation_step` was not registered in the launcher flags.

**Fix**: Added the step to `FEATURE_GENERATION_STEP_FLAGS` in `src/launcher/ares_launcher.py`:
- Added to flags list (line 34)
- Added to PRE_TRAINING stage (line 192)
- Added to BACKTESTING stage (line 207)

### 2. Version Name Collision
**Issue**: Version names only included timestamps down to seconds, causing collisions when steps ran quickly in succession.

**Fix**: Updated `src/utils/artifact_router.py` line 467-468 to include milliseconds:
```python
now = datetime.now()
version_name = f"{artifact_name}_{now.strftime('%Y%m%d_%H%M%S')}_{now.microsecond // 1000:03d}"
```

### 3. HDF5 Versioned Storage Type Mismatch
**Issue**: Artifact router tried to save dict data to HDF5 versioned storage, which only accepts DataFrames.

**Fix**: Updated `src/utils/artifact_router.py` lines 176-200 to check data type compatibility:
```python
if data_category.lower() in category_map:
    suggested_format = category_map[data_category.lower()]
    # HDF5 versioned only works with DataFrames, fallback to JSON/pickle for other types
    if suggested_format == 'hdf5_versioned' and not isinstance(data, pd.DataFrame):
        if isinstance(data, dict) and self._is_json_serializable(data):
            return 'json'
        else:
            return 'pickle'
    return suggested_format
```

### 4. Step 3 Data Loading Dependencies
**Issue**: `feature_generation_period_lookback_optimization_step` was only loading labels from step 1, not features from step 2.

**Fix**: Updated `_load_generated_features()` method in `src/training/steps/pre_training/feature_generation_period_lookback_optimization_step.py` (lines 3441-3553) to:
1. Load generated features from `feature_generation_feature_generation_step`
2. Load target labels from `feature_generation_labeling_integration_step`
3. Merge features and labels on index
4. Return merged DataFrame with both features and targets

## Pipeline Workflow (Corrected)

### Step 1: feature_generation_labeling_integration_step
- **Input**: Historical OHLCV data via Base Step
- **Output**: Labeled data with targets (HDF5 versioned)
- **Artifact**: `labeled_data_ETHUSDT_15m`

### Step 2: feature_generation_feature_generation_step
- **Input**: Historical OHLCV data via Base Step
- **Output**: Generated features (HDF5 versioned)
- **Artifact**: `generated_features_15m`

### Step 3: feature_generation_period_lookback_optimization_step
- **Input**: 
  - Features from step 2 (`generated_features_15m`)
  - Labels from step 1 (`labeled_data_ETHUSDT_15m`)
- **Output**: Lookback optimization results (HDF5 versioned)
- **Artifact**: `lookback_optimization`

### Step 4: feature_generation_interaction_generation_step
- **Input**:
  - Labels from step 1 (`labeled_data_ETHUSDT_15m`) - for target information
  - Features from step 2 (`generated_features_15m`) - base features for interactions
- **Output**: Interaction features (HDF5 versioned)
- **Artifact**: `analyst_interaction_features` or `tactician_interaction_features`
- **Command**: `python3 src/launcher/ares_launcher.py --feature_generation_interaction_generation_step --symbol ETHUSDT --execution-mode light`

### Step 5: feature_generation_final_feature_selection_step
- **Input**:
  - Labels from step 1 (`labeled_data_ETHUSDT_15m`) - for target-based selection
  - Lookback optimization from step 3 (`lookback_optimization`) - for optimal periods
  - Interaction features from step 4 (`analyst_interaction_features`) - interaction terms
- **Output**: Final selected features (HDF5 versioned)
- **Artifact**: `final_selected_features`
- **Command**: `python3 src/launcher/ares_launcher.py --feature_generation_final_feature_selection_step --symbol ETHUSDT --execution-mode light`

### 5. Artifact Manager Versioned Storage Priority
**Issue**: Artifact manager was falling back to old parquet files instead of loading from newly created versioned artifacts.

**Fix**: Updated `src/utils/artifact_manager.py` (lines 940-946, 1268-1346) to:
1. Add `_load_from_versioned_artifacts()` method that searches versioned_artifacts directory
2. Prioritize versioned artifacts as Step 0 in `get_artifact()` method (before exact path search)
3. Search stores in priority order: exact symbol match > UNKNOWN > any other
4. Use correct API methods: `store.get_view(version_name=...)` and `view.to_pandas()`

## Testing Status

- ✅ Step 1: `feature_generation_labeling_integration_step` - Completed successfully
  - Saves: `labeled_data_ETHUSDT_15m` (34,171 rows × 4 cols)
  
- ✅ Step 2: `feature_generation_feature_generation_step` - Completed successfully (after adding to launcher)
  - Saves: `generated_features_15m` (244 rows × 344 cols)
  
- ✅ Step 3: `feature_generation_period_lookback_optimization_step` - Completed successfully (after fixing data loading + versioned artifact priority)
  - Loads: features from step 2 + labels from step 1
  - Now correctly loads from versioned storage instead of old parquet files
  - Saves: `lookback_optimization`
  
- ✅ Step 4: `feature_generation_interaction_generation_step` - Ready to run (dependencies corrected)
  - Loads: labels from step 1 + features from step 2
  - Saves: `analyst_interaction_features` or `tactician_interaction_features`
  
- ✅ Step 5: `feature_generation_final_feature_selection_step` - Ready to run (dependencies corrected)
  - Loads: labels from step 1 + lookback optimization from step 3 + interactions from step 4
  - Saves: `final_selected_features`

## Summary

All fixes have been implemented to ensure the feature generation pipeline:
1. ✅ Properly registers all steps in the launcher
2. ✅ Avoids version name collisions with millisecond timestamps
3. ✅ Routes dict data to JSON/pickle instead of HDF5 when needed
4. ✅ Loads both features and labels for optimization steps
5. ✅ **Prioritizes versioned artifacts over old fallback files**

The pipeline now correctly uses the versioned artifact storage system, ensuring that steps always load the most recent data from previous steps rather than falling back to outdated parquet files.
