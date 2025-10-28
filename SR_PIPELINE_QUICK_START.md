# SR Pipeline - Quick Start Guide

## TL;DR - Correct Order

```
1. sr_parameter_optimization  → Finds optimal parameters
2. sr_detection              → Uses those parameters
3. sr_clustering             → Clusters the results
```

## Run It Now

```bash
# Test the implementation
python test_sr_pipeline_correct_order.py

# Run with configuration
python run_pipeline.py --config config/sr_pipeline_correct_order.yaml

# Run iterative refinement
python run_pipeline.py --config config/sr_pipeline_iterative.yaml
```

## Programmatic Usage

```python
import asyncio
from src.training.steps.market_analysis.components.sr_parameter_optimization import SRParameterOptimizationStep
from src.training.steps.market_analysis.components.sr_detection import SRDetectionComponent
from src.training.steps.market_analysis.components.sr_clustering import SRClusteringComponent

async def main():
    config = {'symbol': 'ETHUSDT', 'exchange': 'binance', 'timeframe': '15m', 'direction': 'longs'}
    
    # 1. Optimize parameters (runs first, no prior artifacts needed)
    opt_result = await SRParameterOptimizationStep().execute(config)
    
    # 2. Detect SR levels (loads and uses optimized parameters)
    det_result = await SRDetectionComponent().execute({**config, 'use_optimized_parameters': True})
    
    # 3. Cluster SR levels (clusters the optimized detections)
    clus_result = await SRClusteringComponent().execute(config)
    
    return opt_result, det_result, clus_result

asyncio.run(main())
```

## Key Changes Made

### ✅ sr_parameter_optimization.py
- Made input artifacts **optional** (can run without clustering data)
- Returns empty list from `get_required_input_artifacts()`
- Uses `_fetch_optional_input_artifacts()` instead of required fetch

### ✅ sr_detection.py
- Added `_load_optimized_parameters()` method
- Loads parameters from `sr_parameter_optimization_result` artifact
- Falls back to defaults if not available
- Uses parameters in detection logic

### ✅ sr_clustering.py
- No changes needed - already loads from sr_detection correctly

## Verify It Works

Check these metrics after running:

```python
# Step 1: Parameter Optimization
assert param_result['success'] == True
print(f"Parameters found: {len(param_result['artifacts']['optimized_parameters'])}")

# Step 2: SR Detection
assert detection_result['success'] == True
assert detection_result['metrics']['used_optimized_parameters'] == True  # ← Should be True!
print(f"SR levels detected: {detection_result['metrics']['total_levels']}")

# Step 3: SR Clustering
assert clustering_result['success'] == True
print(f"Clusters created: {clustering_result['metrics']['total_clusters']}")
```

## Why This Order Matters

### ❌ OLD (Wrong):
```
detection (default params) → clustering → optimization (too late!)
Result: Wasted computation, suboptimal detections
```

### ✅ NEW (Correct):
```
optimization → detection (uses optimized params) → clustering
Result: High-quality detections, meaningful clusters, efficient
```

## Files Created

- `CORRECT_SR_PIPELINE_ORDER.md` - Full documentation
- `SR_PIPELINE_ORDER_IMPLEMENTATION_SUMMARY.md` - Implementation details
- `SR_PIPELINE_QUICK_START.md` - This file
- `config/sr_pipeline_correct_order.yaml` - Production config
- `config/sr_pipeline_iterative.yaml` - Iterative refinement config
- `test_sr_pipeline_correct_order.py` - Test script

## Next Steps

1. ✅ Test: `python test_sr_pipeline_correct_order.py`
2. ✅ Review: Read `CORRECT_SR_PIPELINE_ORDER.md`
3. ✅ Configure: Edit `config/sr_pipeline_correct_order.yaml`
4. ✅ Run: Execute pipeline on your data
5. ✅ Iterate: Use `sr_pipeline_iterative.yaml` for refinement

## Troubleshooting

**Q: Detection not using optimized parameters?**
A: Check that sr_parameter_optimization completed successfully and saved artifacts.

**Q: Taking too long?**
A: Reduce `n_trials` in parameter optimization config (e.g., from 100 to 20).

**Q: No clusters created?**
A: Verify SR detection produced levels. Check clustering algorithm parameters.

## Benefits

✅ Better quality SR detections  
✅ More meaningful clusters  
✅ Efficient resource usage  
✅ Iterative refinement possible  
✅ No wasted computation  

---

**Ready to use!** 🚀
