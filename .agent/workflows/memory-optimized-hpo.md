---
description: How to run HPO with memory optimization to prevent OOM stalls
---

# Memory-Optimized HPO Run

## Quick Start (Memory-Safe)

// turbo-all

1. Kill any stalled HPO processes first:
```bash
pkill -f "meta_labeling_hpo_experiment"
```

2. Set memory limits via environment variables:
```bash
export PYTHONMALLOC=malloc
export MKL_NUM_THREADS=2
export OMP_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
```

3. Run HPO with memory constraints:
```bash
cd /Users/remyroche/Ares
PYTHONPATH=. python3 -u src/launcher/ares_launcher.py meta_labeling_hpo_experiment \
  --symbol ETHUSDT \
  --execution-mode blank \
  --force-hpo \
  --enable-labeling-hpo \
  2>&1 | tee outcomes/hpo_memopt_$(date +%Y%m%d_%H%M%S).log
```

## Memory Optimization Options

### Option 1: Limit Candidate Families (Fastest)
Edit `label_based_layer_2.py` and reduce max families probed:
```python
# In _select_best_geometry_via_race:
max_families_to_probe = 50  # Default is unlimited
```

### Option 2: Reduce Feature Count
Edit `meta_labeling_hpo_experiment_step.py`:
```python
max_features_per_candidate = 50  # Default is 100
```

### Option 3: Chunked Processing
Add `--chunk-mode` flag (if supported) to process families in batches with GC between.

### Option 4: Disable Memory-Heavy Features
In config, set:
```python
{
    "enable_lowvol_features": False,  # Hurst is memory-intensive
    "enable_lagged_residuals": False,  # Many lag features
}
```

## Monitoring Memory

Watch memory during run:
```bash
# In another terminal
watch -n 5 'ps aux | grep python3 | head -5'
```

Or use Activity Monitor on macOS.

## If OOM Occurs

1. Force kill: `pkill -9 -f meta_labeling_hpo`
2. Clear Python caches: `rm -rf __pycache__ **/__pycache__`
3. Restart with fewer features/families
