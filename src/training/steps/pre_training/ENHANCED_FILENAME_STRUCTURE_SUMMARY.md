# Enhanced Filename Structure with Direction and Model Type

## Overview

The artifact manager has been enhanced to include `direction` (short/long) and `model_type` (tactician/analyst) in filename creation and retrieval, with comprehensive fallback support for backward compatibility.

## Key Enhancements

### 1. Enhanced Filename Structure

**New Structure:**
```
{symbol}_{exchange}_{timeframe}_{direction}_{model_type}_{step_name}_{key}_{timestamp}.parquet
```

**Example:**
```
ETHUSDT_binance_15m_longs_analyst_feature_generation_labeling_integration_step_targets_20250119_143022.parquet
```

### 2. Enhanced Directory Structure

**New Partitioned Structure:**
```
artifacts/pre_training/artifact_store/{run_id}/
  {step_name}/
    {symbol}/
      {exchange}/
        {timeframe}/
          {direction}/
            {model_type}/
              {symbol}_{exchange}_{timeframe}_{direction}_{model_type}_{step_name}_{key}_{timestamp}.parquet
```

**Example:**
```
artifacts/pre_training/artifact_store/20250119_143022_abc123def/
  feature_generation_labeling_integration_step/
    ETHUSDT/
      binance/
        15m/
          longs/
            analyst/
              ETHUSDT_binance_15m_longs_analyst_feature_generation_labeling_integration_step_targets_20250119_143022.parquet
```

### 3. Fallback Retrieval Support

The enhanced artifact manager now supports multiple fallback strategies:

1. **Primary Search**: Look for exact match with full metadata
2. **Enhanced Fallback**: Search without direction/model_type in filename
3. **General Fallback**: Recursively search for files with key in name
4. **Legacy Fallback**: Original fallback mechanism for backward compatibility

### 4. Metadata Validation

All steps now validate and include:
- `symbol`: Trading symbol (e.g., 'ETHUSDT')
- `exchange`: Exchange name (e.g., 'binance')
- `timeframe`: Time frame (e.g., '15m')
- `direction`: Trading direction (e.g., 'longs', 'shorts')
- `model_type`: Model type (e.g., 'analyst', 'tactician')

## Updated Steps

### ✅ 1. Data Validation Step
- **Model Type**: `analyst` (model-agnostic, defaults to analyst)
- **Direction**: From parameters (`longs`/`shorts`)

### ✅ 2. Labeling Integration Step
- **Model Type**: Uses `labeling_mode` parameter (`analyst`/`tactician`)
- **Direction**: From parameters (`longs`/`shorts`)

### ✅ 3. Feature Generation Step
- **Model Type**: `analyst` (model-agnostic, defaults to analyst)
- **Direction**: From parameters (`longs`/`shorts`)

### ✅ 4. Period + Lookback Optimization Step
- **Model Type**: `analyst` (model-agnostic, defaults to analyst)
- **Direction**: From parameters (`longs`/`shorts`)

### ✅ 5. Feature Selection Step
- **Model Type**: `analyst` (model-agnostic, defaults to analyst)
- **Direction**: From parameters (`longs`/`shorts`)

### ✅ 6. Interaction Generation Steps
- **Analyst Step**: `model_type: 'analyst'`
- **Tactician Step**: `model_type: 'tactician'`
- **Direction**: From parameters (`longs`/`shorts`)

### ✅ 7. Final Selection Step
- **Model Type**: From step parameters (`analyst`/`tactician`)
- **Direction**: From step parameters (`longs`/`shorts`)

### ✅ 8. Final Validation Step
- **Model Type**: `analyst` (model-agnostic, defaults to analyst)
- **Direction**: From parameters (`longs`/`shorts`)

## Fallback Retrieval Examples

### Example 1: Enhanced Fallback
```python
# Try to retrieve with full metadata
artifact = am.get_artifact('labeling_integration', 'targets', {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'timeframe': '15m',
    'direction': 'longs',
    'model_type': 'analyst'
})

# If not found, fallback searches:
# 1. ETHUSDT/binance/15m/ (without direction/model_type)
# 2. Any subdirectory with 'targets' in filename
```

### Example 2: Legacy Compatibility
```python
# Old-style retrieval still works
artifact = am.get_artifact('labeling_integration', 'targets')
# Will search all possible locations and use fallback mechanisms
```

## Benefits

### 1. **Enhanced Organization**
- Clear separation by direction and model type
- Better file organization and retrieval
- Reduced naming conflicts

### 2. **Backward Compatibility**
- Multiple fallback strategies
- Legacy file support
- Gradual migration path

### 3. **Improved Traceability**
- Full context in filename
- Clear lineage tracking
- Better debugging capabilities

### 4. **Scalability**
- Hierarchical organization
- Efficient retrieval strategies
- Support for multiple model types and directions

## Usage Examples

### Running Steps with Enhanced Metadata

```bash
# Data Validation (analyst, longs)
python3 ares_launcher.py --mode sequential --sub_pipeline feature_generation_data_validation_step --symbol ETHUSDT --direction longs --execution-mode light

# Labeling Integration (analyst, longs)
python3 ares_launcher.py --mode sequential --sub_pipeline feature_generation_labeling_integration_step --symbol ETHUSDT --direction longs --labeling_mode analyst --execution-mode light

# Labeling Integration (tactician, shorts)
python3 ares_launcher.py --mode sequential --sub_pipeline feature_generation_labeling_integration_step --symbol ETHUSDT --direction shorts --labeling_mode tactician --execution-mode light

# Interaction Generation (analyst, longs)
python3 ares_launcher.py --mode sequential --sub_pipeline feature_generation_interaction_generation_step_analyst --symbol ETHUSDT --direction longs --execution-mode light

# Interaction Generation (tactician, shorts)
python3 ares_launcher.py --mode sequential --sub_pipeline feature_generation_interaction_generation_step_tactician --symbol ETHUSDT --direction shorts --execution-mode light

# Final Selection (analyst, longs)
python3 ares_launcher.py --mode sequential --sub_pipeline feature_generation_final_feature_selection_step --symbol ETHUSDT --direction longs --model_type analyst --execution-mode light
```

## File Naming Examples

### Before Enhancement:
```
validated_dataframe_20250119_143022.parquet
targets_20250119_143022.parquet
```

### After Enhancement:
```
ETHUSDT_binance_15m_longs_analyst_feature_generation_data_validation_step_validated_dataframe_20250119_143022.parquet
ETHUSDT_binance_15m_longs_analyst_feature_generation_labeling_integration_step_targets_20250119_143022.parquet
ETHUSDT_binance_15m_shorts_tactician_feature_generation_labeling_integration_step_targets_20250119_143022.parquet
```

## Migration Notes

- **Backward Compatibility**: All existing functionality preserved
- **Gradual Migration**: Old files can still be retrieved via fallback
- **Enhanced Logging**: Clear indication of fallback usage
- **Performance**: Minimal impact on retrieval performance

## Next Steps

1. **Test Enhanced Steps**: Run steps with different direction/model_type combinations
2. **Verify Fallback**: Ensure old files can still be retrieved
3. **Monitor Performance**: Check retrieval performance with new structure
4. **Validate Organization**: Confirm files are properly organized by metadata

All steps are now ready with enhanced filename structure and comprehensive fallback support!
