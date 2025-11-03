# HDP-HMM Tuning Quick Reference

## Overview
3-stage iterative grid refinement for HDP-HMM hyperparameter optimization.

## Usage Commands

### 1. Prepare Features (First Time Only)
```bash
cd /Users/remyroche/Documents/Ares
python3 hdp_hmm_prepare_data.py
```
Creates cached features:
- `hdp_hmm_features_cache.npy`
- `hdp_hmm_features_cache.pkl`

### 2. Run Tuning

#### Standard Run (Use Cached Features)
```bash
# Interactive mode
python3 hdp_hmm_isolated_tuning.py

# Background with logging
nohup python3 -u hdp_hmm_isolated_tuning.py > hdp_hmm_FINAL_RUN.log 2>&1 &
```

#### Fresh Run (Clear Cache First)
```bash
# Interactive mode
python3 hdp_hmm_isolated_tuning.py --clear-cache

# Background with logging
nohup python3 -u hdp_hmm_isolated_tuning.py --clear-cache > hdp_hmm_FINAL_RUN.log 2>&1 &
```

### 3. Monitor Progress

```bash
# Watch for success indicators
tail -f hdp_hmm_FINAL_RUN.log | grep "✅"

# Full output
tail -f hdp_hmm_FINAL_RUN.log

# Last 50 lines
tail -50 hdp_hmm_FINAL_RUN.log

# Check process status
ps aux | grep hdp_hmm_isolated_tuning

# Search for specific patterns
grep -i "stage\|complete\|error" hdp_hmm_FINAL_RUN.log
```

### 4. Manual Cache Management

```bash
# Delete cache files manually
rm hdp_hmm_features_cache.npy hdp_hmm_features_cache.pkl

# Check if cache exists
ls -lh hdp_hmm_features_cache.*
```

## When to Clear Cache

Use `--clear-cache` when:
- ✅ Data has been updated
- ✅ Feature engineering code has changed
- ✅ Previous cache is corrupted or incomplete
- ✅ You want to ensure fresh feature computation

Skip `--clear-cache` when:
- ✅ Data and features haven't changed (saves time)
- ✅ Running multiple tuning experiments
- ✅ Resuming interrupted tuning

## Output Files

All results saved to `outcomes/` directory:
- `hdp_hmm_stage1_{timestamp}.csv` - Stage 1 results
- `hdp_hmm_stage2_{timestamp}.csv` - Stage 2 results  
- `hdp_hmm_stage3_{timestamp}.csv` - Stage 3 results
- `hdp_hmm_iterative_all_results_{timestamp}.csv` - Combined results
- `stage{N}_checkpoint_{i}.csv` - Checkpoints every 50 tests

## Tuning Configuration

**Stage 1: Coarse Exploration**
- Grid: 4×6×4 = 96 tests
- Gibbs iterations: 50
- Purpose: Broad parameter space exploration

**Stage 2: Refinement**
- Grid: 4×6×4 = 96 tests  
- Gibbs iterations: 100
- Purpose: Zoom into best region from Stage 1

**Stage 3: Final Tuning**
- Grid: 4×6×4 = 96 tests
- Gibbs iterations: 200
- Purpose: Fine-tune optimal configuration

**Total: 288 tests** (vs 810 for full grid)

## Parameters Optimized

- **α (alpha)**: [1.0, 1.9] - Regime distribution balance
- **κ (kappa)**: [5.0, 45.0] - Regime persistence & temporal stability
- **γ (gamma)**: [3.0, 6.0] - Regime distinctness

## Composite Score Weighting

```python
composite_score = (
    silhouette_score * 0.3 +
    balance_score * 0.3 +
    temporal_smoothness * 0.2 +
    tanh(cv_ratio) * 0.2
)
```

## Example Workflow

```bash
# Complete workflow from scratch
cd /Users/remyroche/Documents/Ares

# Step 1: Prepare features (first time only)
python3 hdp_hmm_prepare_data.py

# Step 2: Run tuning with fresh cache
nohup python3 -u hdp_hmm_isolated_tuning.py --clear-cache > hdp_hmm_FINAL_RUN.log 2>&1 &

# Step 3: Monitor
tail -f hdp_hmm_FINAL_RUN.log | grep "✅"

# Step 4: Check results
ls -lh outcomes/hdp_hmm_*
```

## Troubleshooting

### Stalled or Hanging
```bash
# Kill the process
pkill -f hdp_hmm_isolated_tuning

# Clear cache and restart
python3 hdp_hmm_isolated_tuning.py --clear-cache
```

### Memory Issues
```bash
# Check memory usage
top -pid $(pgrep -f hdp_hmm_isolated_tuning)

# Clear cache to free memory
rm hdp_hmm_features_cache.*
```

### Corrupted Cache
```bash
# Delete and regenerate
python3 hdp_hmm_isolated_tuning.py --clear-cache
```

## Performance Tips

1. **Use cache** for repeated runs (saves ~1-2 minutes per run)
2. **Run in background** with `nohup` for long sessions
3. **Monitor checkpoints** to track progress every 50 tests
4. **Check logs** if process seems stuck
5. **Clear cache** if data/features have changed

## Related Files

- `hdp_hmm_prepare_data.py` - Feature preparation script
- `hdp_hmm_single_test.py` - Single parameter test runner
- `HDP_HMM_AUTO_TUNING_GUIDE.md` - Detailed tuning guide
- `HDP_HMM_USAGE_GUIDE.md` - General HDP-HMM usage

