# Enhanced Artifact Manager Usage Guide

## Overview

The enhanced artifact manager now provides robust file management with proper metadata inclusion, data alignment validation, and support for both individual artifacts and shared datasets across pre-training steps.

## Key Improvements

### 1. Enhanced File Naming and Path Structure

**Before:**
```
artifacts/pre_training/artifact_store/20250119_143022_abc123def/
  feature_generation_data_validation_step/
    validated_dataframe_20250119_143022.parquet
```

**After:**
```
artifacts/pre_training/artifact_store/20250119_143022_abc123def/
  feature_generation_data_validation_step/
    ETHUSDT/
      binance/
        15m/
          ETHUSDT_binance_15m_feature_generation_data_validation_step_validated_dataframe_20250119_143022.parquet
```

### 2. Metadata Validation and Enhancement

The artifact manager now automatically validates and enhances metadata:

```python
# The artifact manager will automatically:
# 1. Validate critical metadata (symbol, exchange, timeframe)
# 2. Add step information and timestamps
# 3. Log warnings for missing critical metadata
# 4. Set defaults for missing information
```

### 3. Full Path Logging

All file operations now log the full path for transparency:

```
📁 Creating Parquet file: /Users/remyroche/Documents/Ares/artifacts/pre_training/artifact_store/20250119_143022_abc123def/feature_generation_data_validation_step/ETHUSDT/binance/15m/ETHUSDT_binance_15m_feature_generation_data_validation_step_validated_dataframe_20250119_143022.parquet
📁 Retrieved Parquet file: /Users/remyroche/Documents/Ares/artifacts/pre_training/artifact_store/20250119_143022_abc123def/feature_generation_labeling_integration_step/ETHUSDT/binance/15m/ETHUSDT_binance_15m_feature_generation_labeling_integration_step_targets_20250119_143022.parquet
```

## Usage Examples

### 1. Basic Usage with Proper Metadata

```python
from src.training.steps.pre_training.utils.artifact_manager import get_pretraining_artifact_manager

# Initialize artifact manager
am = get_pretraining_artifact_manager()

# Save artifacts with proper metadata
am.save(
    step_name='feature_generation_data_validation_step',
    artifacts={
        'validated_dataframe': validated_df,
        'validation_metrics': metrics_dict
    },
    metadata={
        'symbol': 'ETHUSDT',
        'exchange': 'binance', 
        'timeframe': '15m',
        'direction': 'longs',
        'intensity': 'blank'
    }
)
```

### 2. Shared Dataset Approach

For steps that build upon each other, use the shared dataset approach:

```python
# In labeling integration step
am.save_shared_dataset(
    step_name='feature_generation_labeling_integration_step',
    base_data=raw_data,  # OHLCV data
    additional_columns={
        'targets': targets_series,
        'labeling_metadata': labeling_metadata_series
    },
    metadata={
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m'
    }
)

# In feature generation step
am.save_shared_dataset(
    step_name='feature_generation_feature_generation_step', 
    base_data=labeled_data,  # Previous step's output
    additional_columns={
        'feature_1': feature_1_series,
        'feature_2': feature_2_series,
        # ... more features
    },
    metadata={
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m'
    }
)
```

### 3. Retrieving Artifacts

```python
# Retrieve specific artifacts
validated_df = am.get_artifact('feature_generation_data_validation_step', 'validated_dataframe')
targets = am.get_artifact('feature_generation_labeling_integration_step', 'targets')

# Retrieve all artifacts for a step
all_artifacts = am.get_step_artifacts('feature_generation_data_validation_step')
```

## Step Integration Requirements

### Required Metadata for Each Step

Each pre-training step should pass the following metadata:

```python
required_metadata = {
    'symbol': str,      # e.g., 'ETHUSDT'
    'exchange': str,    # e.g., 'binance' 
    'timeframe': str,   # e.g., '15m'
    'direction': str,   # e.g., 'longs'
    'intensity': str,   # e.g., 'blank'
    'created_at': str  # ISO timestamp (auto-added)
}
```

### Step-Specific Examples

#### Data Validation Step
```python
def _process_data(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    # Extract metadata from kwargs
    metadata = {
        'symbol': kwargs.get('symbol', 'UNKNOWN'),
        'exchange': kwargs.get('exchange', 'UNKNOWN'), 
        'timeframe': kwargs.get('timeframe', 'UNKNOWN'),
        'direction': kwargs.get('direction', 'longs'),
        'intensity': kwargs.get('intensity', 'blank')
    }
    
    # Process data...
    validated_df = self._validate_data(data)
    
    # Save with proper metadata
    am = get_pretraining_artifact_manager()
    am.save(
        step_name='feature_generation_data_validation_step',
        artifacts={'validated_dataframe': validated_df},
        metadata=metadata
    )
```

#### Labeling Integration Step
```python
def _process_data(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    metadata = {
        'symbol': kwargs.get('symbol', 'UNKNOWN'),
        'exchange': kwargs.get('exchange', 'UNKNOWN'),
        'timeframe': kwargs.get('timeframe', 'UNKNOWN'),
        'labeling_mode': kwargs.get('labeling_mode', 'analyst')
    }
    
    # Generate labels...
    labeled_df, targets = self._generate_labels(data)
    
    # Use shared dataset approach for better alignment
    am = get_pretraining_artifact_manager()
    am.save_shared_dataset(
        step_name='feature_generation_labeling_integration_step',
        base_data=data,  # Original OHLCV data
        additional_columns={
            'targets': targets,
            'labeling_metadata': labeling_metadata
        },
        metadata=metadata
    )
```

## Data Alignment Validation

The artifact manager now automatically validates:

1. **Timestamp Alignment**: Ensures all data has proper DatetimeIndex
2. **Row Alignment**: Validates that additional columns align with base data
3. **Duplicate Detection**: Warns about duplicate timestamps
4. **Missing Value Detection**: Identifies null values in critical columns

## Benefits

1. **Transparency**: Full path logging for all file operations
2. **Data Integrity**: Automatic validation of data alignment
3. **Traceability**: Enhanced metadata with step information
4. **Scalability**: Partitioned storage by symbol/exchange/timeframe
5. **Consistency**: Standardized file naming across all steps
6. **Debugging**: Clear warnings for missing metadata or alignment issues

## Migration Guide

### For Existing Steps

1. **Add metadata extraction** at the beginning of `_process_data`:
```python
metadata = {
    'symbol': kwargs.get('symbol', 'UNKNOWN'),
    'exchange': kwargs.get('exchange', 'UNKNOWN'),
    'timeframe': kwargs.get('timeframe', 'UNKNOWN')
}
```

2. **Update save calls** to include metadata:
```python
am.save(
    step_name=step_name,
    artifacts=artifacts_dict,
    metadata=metadata  # Add this line
)
```

3. **Consider shared dataset approach** for steps that build upon each other

### For New Steps

Always include the required metadata and consider using `save_shared_dataset` for steps that add columns to existing data.
