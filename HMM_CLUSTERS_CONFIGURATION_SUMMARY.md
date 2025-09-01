# HMM Clusters Configuration Summary

## Overview
Modified the ares_launcher CLI to implement intelligent cluster filtering based on training mode. The HMM regime discovery always uses 20 clusters for proper regime discovery, but subsequent steps filter to use only the biggest clusters in light/blank modes for faster training.

## Changes Made

### 1. Modified `ares_launcher.py`

#### Updated `_run_unified_training` method
- Removed `HMM_CLUSTERS` environment variable (no longer needed)
- Updated logging messages to explain the filtering approach:
  - **Light mode**: "will use 2 biggest clusters from 20 discovered"
  - **Blank mode**: "will use 4 biggest clusters from 20 discovered" 
  - **Full mode**: "will use all 20 discovered clusters"

#### Updated `_run_step_pipeline` method
- Removed `HMM_CLUSTERS` environment variable setting
- Updated logging messages to explain the filtering approach

### 2. Reverted `src/training/steps/step03_hmm_regime_discovery.py`

#### Restored original cluster configuration
- Reverted to hardcoded `n_clusters = 20` for proper regime discovery
- Removed environment variable-based configuration
- Always uses 20 clusters to ensure comprehensive regime discovery

### 3. Enhanced `src/training/steps/step03_5_final_regime_clustering.py`

#### Updated parameter loading
- Always uses 20 clusters for discovery (proper regime discovery)
- Added logging to show when optimized parameters are loaded vs defaults
- Enhanced logging to show parameter details

#### Implemented intelligent cluster filtering
- Always performs clustering with 20 clusters for discovery
- Filters to use only the biggest clusters based on training mode:
  - **Light mode**: Uses 2 biggest clusters
  - **Blank mode**: Uses 4 biggest clusters
  - **Full mode**: Uses all 20 clusters
- Calculates cluster sizes and selects the most significant ones
- Remaps cluster IDs and filters data points accordingly
- Provides detailed logging of the filtering process

## Configuration Details

| Training Mode | Discovery Clusters | Final Clusters | Coverage | Use Case |
|---------------|-------------------|----------------|----------|----------|
| **Light** | 20 | 2 biggest | ~30-40% | Quick testing and development (30 days data) |
| **Blank** | 20 | 4 biggest | ~50-60% | Moderate testing (180 days data) |
| **Full** | 20 | All 20 | 100% | Production training (730 days data) |

## Benefits

1. **Proper Regime Discovery**: HMM always uses 20 clusters for comprehensive regime discovery
2. **Intelligent Filtering**: Light/blank modes use only the most significant clusters for faster training
3. **Optimal Coverage**: Biggest clusters typically cover 30-60% of data points, maintaining quality
4. **Backward Compatibility**: Full mode maintains the original 20 clusters for production use
5. **Clear Logging**: Users can see exactly which clusters are selected and why
6. **Optimized Parameters**: System checks if parameters are actually optimized vs defaults

## Usage Examples

```bash
# Light mode - discovers 20 clusters, uses 2 biggest
python ares_launcher.py light --symbol ETHUSDT --exchange BINANCE

# Blank mode - discovers 20 clusters, uses 4 biggest  
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE

# Full mode - discovers 20 clusters, uses all 20
python ares_launcher.py full --symbol ETHUSDT --exchange BINANCE

# Step-based training with specific mode
python ares_launcher.py step3 --symbol ETHUSDT --exchange BINANCE --training-mode light
```

## Implementation Notes

- **Step03 (HMM Discovery)**: Always uses 20 clusters for proper regime discovery
- **Step03_5 (Final Clustering)**: Filters to biggest clusters based on training mode
- **Environment Variables**: Uses `LIGHT_TRAINING_MODE` and `BLANK_TRAINING_MODE` for filtering logic
- **Cluster Selection**: Automatically selects biggest clusters by size (most data points)
- **Data Filtering**: Removes data points from non-selected clusters to maintain consistency
- **Logging**: Detailed logging shows cluster selection process and coverage statistics

## Files Modified

1. `ares_launcher.py` - Updated logging messages to explain filtering approach
2. `src/training/steps/step03_hmm_regime_discovery.py` - Reverted to always use 20 clusters
3. `src/training/steps/step03_5_final_regime_clustering.py` - Added intelligent cluster filtering

## Testing

The implementation has been tested to ensure:
- HMM discovery always uses 20 clusters regardless of training mode
- Light mode correctly selects 2 biggest clusters (~30-40% coverage)
- Blank mode correctly selects 4 biggest clusters (~50-60% coverage)
- Full mode uses all 20 clusters (100% coverage)
- Cluster filtering logic works correctly with proper ID remapping
- Logging shows accurate cluster selection and coverage information