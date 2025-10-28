# SR Detection Feedback Loop Implementation

## Overview

The SR (Support/Resistance) detection pipeline now has an **automated feedback loop** that allows optimized parameters from `sr_parameter_optimization` to be automatically used by subsequent `sr_detection` runs.

## Pipeline Flow with Feedback Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INITIAL RUN (No Optimization)                │
└─────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │  1. SR Detection     │  ← Uses default parameters
    │  (Default params)    │
    └──────────┬───────────┘
               │ Produces: sr_detection_result
               ▼
    ┌──────────────────────┐
    │  2. SR Clustering    │
    │                      │
    └──────────┬───────────┘
               │ Produces: sr_clustering_result, sr_levels_dictionary
               ▼
    ┌──────────────────────┐
    │  3. SR Parameter     │
    │     Optimization     │
    └──────────┬───────────┘
               │ Produces: sr_parameter_optimization_result
               │           (optimized_parameters, quality_thresholds)
               ▼
         [Saved to artifact manager]


┌─────────────────────────────────────────────────────────────────────┐
│                    SUBSEQUENT RUNS (With Optimization)               │
└─────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │  1. SR Detection     │  ← Automatically loads optimized parameters!
    │  (Optimized params)  │  ← Applies quality thresholds!
    └──────────┬───────────┘
               │ Produces: sr_detection_result (IMPROVED)
               ▼
    ┌──────────────────────┐
    │  2. SR Clustering    │  ← Works with higher quality SR levels
    │                      │
    └──────────┬───────────┘
               │ Produces: sr_clustering_result (IMPROVED)
               ▼
    ┌──────────────────────┐
    │  3. SR Parameter     │  ← Further refines parameters
    │     Optimization     │
    └──────────┬───────────┘
               │ Produces: sr_parameter_optimization_result (UPDATED)
               ▼
         [Updated in artifact manager]
```

## Implementation Details

### 1. SR Detection Component (`sr_detection.py`)

#### Added Methods:

**`get_required_input_artifacts()`**
```python
def get_required_input_artifacts(self) -> List[str]:
    """Get list of optional input artifacts this component can use from previous steps."""
    return ['sr_parameter_optimization_result']
```

**`_load_optimized_parameters()`**
```python
async def _load_optimized_parameters(self) -> Optional[Dict[str, Any]]:
    """
    Load optimized parameters from previous sr_parameter_optimization run.
    
    Uses BaseStep's _get_artifact() to load:
    - optimized_parameters: Detection parameter values
    - quality_thresholds: Minimum quality standards
    - optimization_summary: Performance metrics
    """
```

**`_apply_quality_filters()`**
```python
def _apply_quality_filters(self, sr_levels: List[Dict[str, Any]], 
                          quality_thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Apply quality thresholds from optimization to filter SR levels.
    
    Filters based on:
    - min_strength: Minimum level strength
    - min_confidence: Minimum confidence score
    - min_touches: Minimum number of price touches
    """
```

#### Modified Methods:

- **`execute()`**: Now loads optimized parameters before detection
- **`_perform_enhanced_sr_detection()`**: Accepts `optimized_parameters` argument
- **`_detect_sr_levels_vectorbt()`**: Uses optimized parameters in detection
- **`_detect_sr_levels_traditional()`**: Uses optimized parameters in detection

### 2. SR Parameter Optimization Component (`sr_parameter_optimization.py`)

#### Already Implemented (No Changes Needed):

✅ **Properly saves artifacts** using BaseStep's `_save_artifact()`:
```python
async def _save_output_artifacts(self, artifacts: Dict[str, Any], config: Dict[str, Any]):
    """Save output artifacts using BaseStep artifact management."""
    for artifact_name, artifact_data in artifacts.items():
        artifact_path = self._save_artifact(
            data=artifact_data,
            artifact_name=artifact_name,
            artifact_type="data",
            compression="auto"
        )
```

✅ **Artifact Structure**:
```python
{
    'sr_parameter_optimization_result': {
        'optimized_parameters': {...},      # Parameter values
        'quality_thresholds': {...},        # Quality filters
        'parameter_optimization_metrics': {...},
        'optimization_summary': {...},
        'enhancement_details': {...},
        'metadata': {...}
    }
}
```

## How It Works

### First Run (No Optimization Available)

1. **SR Detection** runs with default parameters
   - Logs: `ℹ️ No optimized parameters found, using default detection parameters`
   - Metrics: `using_optimized_parameters: False`

2. **SR Clustering** clusters the detected levels

3. **SR Parameter Optimization** analyzes results and finds optimal parameters
   - Saves `sr_parameter_optimization_result` artifact
   - Contains optimized weights, thresholds, and quality standards

### Second Run (Optimization Available)

1. **SR Detection** automatically loads optimization results
   - Logs: `✅ Loaded X optimized parameters from previous optimization`
   - Logs: `Best score: 0.XX, Optimization time: XX.Xs`
   - Uses optimized parameters in detection algorithms
   - Applies quality filters to remove low-quality levels
   - Metrics: `using_optimized_parameters: True`

2. Detection results are higher quality with better filtering

3. The cycle continues, continuously improving

## Artifact Management (BaseStep)

Both components use BaseStep's artifact management:

### Saving Artifacts
```python
artifact_path = self._save_artifact(
    data=artifact_data,
    artifact_name='sr_parameter_optimization_result',
    artifact_type='data',
    compression='auto'
)
```

### Loading Artifacts
```python
optimization_result = self._get_artifact(
    artifact_name='sr_parameter_optimization_result',
    artifact_type='data'
)
```

### Artifact Context
```python
self.artifact_manager.set_context(
    symbol=symbol,
    exchange=exchange,
    direction=direction,
    model='Analyst'
)
```

## Metrics and Logging

### Detection Metrics Include:
```python
{
    'using_optimized_parameters': bool,
    'optimization_metadata': {...},
    'feedback_loop': {
        'used_optimized_parameters': bool,
        'optimization_timestamp': str,
        'optimization_score': float
    }
}
```

### Detection Result Metadata:
```python
'metadata': {
    'feedback_loop': {
        'used_optimized_parameters': True/False,
        'optimization_timestamp': '2024-...',
        'optimization_score': 0.XX
    }
}
```

## Benefits

1. **Automatic Improvement**: Each run benefits from previous optimization
2. **No Manual Intervention**: Parameters automatically flow between steps
3. **Quality Filtering**: Low-quality levels are automatically removed
4. **Traceable**: Metrics show whether optimized parameters were used
5. **Fallback Safe**: Works with defaults if optimization hasn't run yet
6. **Self-Improving**: Each optimization cycle refines parameters further

## Running the Feedback Loop

### Initial Run (Establish Baseline)
```bash
python src/launcher/ares_launcher.py stage MARKET_ANALYSIS --config config.yaml
```

This runs all three steps:
1. `sr_detection` (default params)
2. `sr_clustering`
3. `sr_parameter_optimization` (saves optimized params)

### Subsequent Runs (Use Optimized Parameters)
```bash
python src/launcher/ares_launcher.py stage MARKET_ANALYSIS --config config.yaml
```

Same command, but now:
1. `sr_detection` **automatically loads and uses** optimized parameters ✨
2. `sr_clustering` works with improved levels
3. `sr_parameter_optimization` further refines parameters

### Run Individual Steps
```bash
# Detection will automatically use optimized params if available
python src/launcher/ares_launcher.py step sr_detection --config config.yaml
```

## Verification

### Check if Optimized Parameters Were Used

Look for these log messages:

**Parameters Found:**
```
✅ Loaded 15 optimized parameters from previous optimization
   - Best score: 0.85
   - Optimization time: 123.4s
   - Total combinations tested: 500
```

**Parameters Not Found:**
```
ℹ️ No optimized parameters found, using default detection parameters
```

### Check Metrics Output

```python
{
    'success': True,
    'metrics': {
        'using_optimized_parameters': True,  # ← Feedback loop active
        'feedback_loop': {
            'used_optimized_parameters': True,
            'optimization_timestamp': '2024-10-28T...',
            'optimization_score': 0.85
        }
    }
}
```

## Troubleshooting

### "No optimized parameters found"

**Possible Causes:**
1. First run - optimization hasn't executed yet
2. Artifact was deleted or moved
3. Different symbol/exchange/direction context

**Solution:** Run the full MARKET_ANALYSIS stage once to generate optimized parameters

### Parameters not being applied

**Check:**
1. Artifact manager context is set correctly
2. Symbol, exchange, direction match between runs
3. Artifact format is correct (dict with 'optimized_parameters' key)

## Code Changes Summary

### Files Modified
- ✅ `/workspace/src/training/steps/market_analysis/components/sr_detection.py`
  - Added `get_required_input_artifacts()`
  - Added `_load_optimized_parameters()`
  - Added `_apply_quality_filters()`
  - Modified `execute()` to load and use optimized parameters
  - Modified `_perform_enhanced_sr_detection()` to accept optimized parameters
  - Modified detection methods to use optimized parameters
  - Enhanced metrics to track feedback loop usage

### Files Verified (No Changes Needed)
- ✅ `/workspace/src/training/steps/market_analysis/components/sr_parameter_optimization.py`
  - Already uses BaseStep's `_save_artifact()` correctly
  - Artifact structure is compatible with detection component

## Testing the Feedback Loop

```python
# Test script
import asyncio
from src.training.steps.market_analysis.components.sr_detection import SRDetectionComponent
from src.training.steps.market_analysis.components.sr_parameter_optimization import SRParameterOptimizationStep

async def test_feedback_loop():
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'direction': 'longs',
        'execution_mode': 'light'
    }
    
    # First run - no optimization available
    detection = SRDetectionComponent('sr_detection')
    result1 = await detection.execute(config)
    print(f"First run - Used optimized params: {result1['metrics']['using_optimized_parameters']}")
    
    # Run optimization (this would save parameters)
    # ... optimization step ...
    
    # Second run - should use optimized parameters
    detection2 = SRDetectionComponent('sr_detection')
    result2 = await detection2.execute(config)
    print(f"Second run - Used optimized params: {result2['metrics']['using_optimized_parameters']}")
    print(f"Feedback loop data: {result2['metrics']['feedback_loop']}")

asyncio.run(test_feedback_loop())
```

## Next Steps

The feedback loop is now **fully implemented and automated**. The system will:

1. ✅ Automatically load optimized parameters when available
2. ✅ Fall back to defaults when optimization hasn't run yet
3. ✅ Apply quality filters from optimization
4. ✅ Track usage in metrics for transparency
5. ✅ Self-improve with each optimization cycle

**The feedback loop is production-ready!** 🚀
