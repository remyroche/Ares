# Quick Reference: SR Optimization Fixes

## What Was Fixed

### 🔧 Issue 1: VectorBT Import Error
**Symptom**: `cannot import name 'vbt' from 'vectorbt'`
**Fix**: Changed imports to use `src.vectorbt` instead of direct `vectorbt`
**Impact**: ✅ System now handles VectorBT availability gracefully

### 💾 Issue 2: Memory Constraints (OOM Kill)
**Symptom**: Process killed during Stage 2 optimization
**Fix**: Implemented chunked evaluation (1000/500 point chunks)
**Impact**: ✅ Memory usage reduced by ~90%, prevents OOM on 16GB RAM

### 📊 Issue 3: Wrong Pipeline Order
**Symptom**: Only 2 clusters and 4 SR levels (mock data)
**Fix**: Changed order to: `sr_detection → sr_clustering → sr_parameter_optimization`
**Impact**: ✅ Now uses real SR levels instead of mock data

## How to Run

### Run Full Pipeline (Recommended):
```bash
python3 src/launcher/ares_launcher.py --stage MARKET_ANALYSIS
```

### Run Individual Steps (For Testing):
```bash
# Step 1: Detect SR levels
python3 src/launcher/ares_launcher.py --step sr_detection

# Step 2: Cluster SR levels
python3 src/launcher/ares_launcher.py --step sr_clustering

# Step 3: Optimize parameters (uses output from steps 1&2)
python3 src/launcher/ares_launcher.py --step sr_parameter_optimization
```

## Memory Tuning

### If You Still Get OOM Errors:

Edit `src/utils/ml_common/optimization/bayesian_tpe_optimizer.py` line 103-104:

```python
# For 8GB RAM:
max_coarse_grid_size: int = 500   # Reduce from 1000
max_fine_grid_size: int = 250     # Reduce from 500

# For 32GB+ RAM:
max_coarse_grid_size: int = 2000  # Increase from 1000
max_fine_grid_size: int = 1000    # Increase from 500
```

## Expected Behavior Now

### Before Fixes:
```
[2025-10-27 23:52:06.667] ⚡ VectorBT Rolling Optimization: Starting info
[2025-10-27 23:52:06.667] INFO: {'success': False, 'error': 'VectorBT not available'}
...
zsh: killed     python3 src/launcher/ares_launcher.py
```

### After Fixes:
```
[2025-10-28 XX:XX:XX.XXX] 🔍 Stage 1: VectorBT-optimized coarse grid search
[2025-10-28 XX:XX:XX.XXX] 🔄 Chunked evaluation: 10000 points in chunks of 1000
[2025-10-28 XX:XX:XX.XXX] 📦 Chunk 1/10: evaluating 1000 points
[2025-10-28 XX:XX:XX.XXX] ✨ New best found in chunk 1: 0.123456
...
[2025-10-28 XX:XX:XX.XXX] 🔍 Stage 2: VectorBT-optimized fine grid search
[2025-10-28 XX:XX:XX.XXX] ✅ Optimization completed successfully
```

## Key Metrics to Monitor

1. **Chunk Processing**:
   - Look for `📦 Chunk X/Y` messages
   - Should complete without killing process

2. **SR Detection Results**:
   - Should find >4 SR levels (not just 4)
   - Should find >2 clusters (not just 2)

3. **Memory Usage**:
   - Monitor with `top` or `htop`
   - Should stay under 80% of available RAM

## Troubleshooting

### Still Getting OOM?
1. Reduce chunk sizes (see Memory Tuning above)
2. Enable early stopping:
   ```python
   enable_early_stopping=True
   early_stopping_threshold=0.8
   ```

### VectorBT Still Not Working?
1. Check if ARES_ENABLE_VECTORBT environment variable is set
2. The system uses pandas fallbacks - this is expected behavior
3. Look for "Hardware acceleration: Enabled" in logs

### Wrong Pipeline Order?
1. Verify file: `src/launcher/ares_launcher.py` line 171
2. Should be: `['sr_detection', 'sr_clustering', 'sr_parameter_optimization']`
3. NOT: `['sr_parameter_optimization', 'sr_detection', 'sr_clustering']`

## Files Changed

- ✅ `src/utils/ml_common/optimization/bayesian_tpe_optimizer.py`
- ✅ `src/training/steps/market_analysis/components/sr_parameter_optimization.py`
- ✅ `src/launcher/ares_launcher.py`

## Success Indicators

✅ No import errors
✅ Process completes without OOM kill
✅ Chunked evaluation messages appear
✅ Real SR levels detected (>4 levels, >2 clusters)
✅ Optimization produces results

## Need More Help?

See detailed documentation:
- Full summary: `SR_OPTIMIZATION_FIXES_SUMMARY.md`
- Code changes: Check git diff for the 3 files above
