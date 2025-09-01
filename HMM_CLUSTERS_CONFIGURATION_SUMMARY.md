# HMM Clusters Configuration Summary

## Overview
Modified the ares_launcher CLI to configure the number of HMM clusters based on the training mode. This allows for faster training in light and blank modes by using fewer clusters.

## Changes Made

### 1. Modified `ares_launcher.py`

#### Updated `_run_unified_training` method
- Added `HMM_CLUSTERS` environment variable setting for each training mode:
  - **Light mode**: 2 clusters (`HMM_CLUSTERS=2`)
  - **Blank mode**: 4 clusters (`HMM_CLUSTERS=4`) 
  - **Full mode**: 20 clusters (`HMM_CLUSTERS=20`)

#### Updated `_run_step_pipeline` method
- Added `HMM_CLUSTERS` environment variable setting for step-based training
- Updated logging messages to show the number of HMM clusters being used

### 2. Modified `src/training/steps/step03_hmm_regime_discovery.py`

#### Updated cluster configuration
- Replaced hardcoded `n_clusters = 20` with environment variable-based configuration
- Added logging to show which number of clusters is being used
- Uses `int(os.environ.get("HMM_CLUSTERS", "20"))` to get cluster count

### 3. Modified `src/training/steps/step03_5_final_regime_clustering.py`

#### Updated default parameters
- Modified `optimized_params` to use environment variable for `n_clusters`
- Updated fallback value to use environment variable
- Added logging to show cluster count being used

## Configuration Details

| Training Mode | HMM Clusters | Use Case |
|---------------|--------------|----------|
| **Light** | 2 | Quick testing and development (30 days data) |
| **Blank** | 4 | Moderate testing (180 days data) |
| **Full** | 20 | Production training (730 days data) |

## Benefits

1. **Faster Training**: Light and blank modes now use significantly fewer clusters, reducing computational time
2. **Consistent Configuration**: All HMM-related steps now use the same cluster count based on training mode
3. **Backward Compatibility**: Full mode maintains the original 20 clusters for production use
4. **Clear Logging**: Users can see exactly how many clusters are being used for each training mode

## Usage Examples

```bash
# Light mode - uses 2 HMM clusters
python ares_launcher.py light --symbol ETHUSDT --exchange BINANCE

# Blank mode - uses 4 HMM clusters  
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE

# Full mode - uses 20 HMM clusters
python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE

# Step-based training with specific mode
python ares_launcher.py step3 --symbol ETHUSDT --exchange BINANCE --training-mode light
```

## Implementation Notes

- Environment variable `HMM_CLUSTERS` is set by the launcher based on training mode
- Default fallback is 20 clusters if environment variable is not set
- Changes affect both unified training and step-based training pipelines
- All HMM-related steps (step03 and step03_5) now respect the cluster configuration
- Logging has been added to make the configuration transparent to users

## Files Modified

1. `ares_launcher.py` - Main launcher with environment variable configuration
2. `src/training/steps/step03_hmm_regime_discovery.py` - HMM regime discovery step
3. `src/training/steps/step03_5_final_regime_clustering.py` - Final regime clustering step

## Testing

The implementation has been tested to ensure:
- Environment variables are set correctly for each training mode
- Default fallback works when environment variable is not set
- All training modes use the appropriate number of clusters
- Logging shows the correct cluster count being used