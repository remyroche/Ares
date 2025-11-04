# Auto-Tuning Guide for Sticky Finite HMM

Complete guide on how to use the auto-tuner with the Sticky Finite HMM regime discovery step.

---

## 🚀 Quick Start

### Enable Auto-Tuning in Config

Add these parameters to your config to enable automatic hyperparameter optimization:

```python
config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h',
    
    # ✅ Enable auto-tuning
    'enable_auto_tuning': True,
    
    # Optional: Configure auto-tuning behavior
    'auto_tuning_config': {
        'use_hierarchical': True,  # Recommended (3-5x faster)
        'n_rounds': 2,  # Number of optimization rounds
        'tpe_trials': 100,  # TPE trials per round
        'timeout': 3600,  # 1 hour max (in seconds)
        'cache_dir': './cache/sticky_hmm_tuning'  # Optional: cache results
    }
}

# Run the step
from src.training.steps.market_analysis.sticky_finite_hmm_clustering import (
    StickyFiniteHMMRegimeDiscoveryStep
)

step = StickyFiniteHMMRegimeDiscoveryStep()
result = await step.execute(config)

# Check results
print(f"Success: {result['success']}")
print(f"Best Score: {result['composite_score']:.4f}")
print(f"Regimes Found: {result['n_regimes']}")

# Auto-tuning results (if enabled)
if 'auto_tuning_results' in result:
    tuning = result['auto_tuning_results']
    print(f"\nAuto-Tuning Summary:")
    print(f"  Best Score: {tuning['best_score']:.4f}")
    print(f"  Total Time: {tuning['total_time']:.1f}s")
    print(f"  Total Trials: {tuning['total_trials']}")
    print(f"  Best Params: {tuning['best_params']}")
```

---

## 📊 What Gets Optimized

### 5 Key Parameters

The auto-tuner optimizes these 5 parameters to maximize the composite quality score:

1. **K** (4-7): Number of regime states
2. **kappa** (5-50): Regime persistence/stickiness
3. **base_alpha** (0.1-1.0): Transition sparsity
4. **lr** (1e-4 to 1e-1): Learning rate
5. **pca_components** (10-20): PCA dimensionality

### 8 Fixed Parameters

These are set to sensible defaults (see `PARAMETER_GUIDE.md` for details):
- num_iters: 1000
- num_particles: 10
- prior_mean_scale: 10.0
- prior_cov_scale: 1.0
- patience: 50
- elbo_improvement_threshold: 1e-3
- min_features: 50
- max_features: 100

---

## ⚙️ Configuration Options

### `enable_auto_tuning` (bool)
- **Default**: `False`
- **Description**: Master switch to enable/disable auto-tuning
- **When True**: Runs optimization before clustering
- **When False**: Uses default or manually specified parameters

### `auto_tuning_config` (dict)

Optional dictionary to configure auto-tuning behavior:

#### `use_hierarchical` (bool)
- **Default**: `True`
- **Description**: Use hierarchical parameter optimization
- **Recommended**: `True` (3-5x faster than exhaustive search)
- **False**: Falls back to standard optimization (not fully implemented yet)

#### `n_rounds` (int)
- **Default**: `2`
- **Range**: 1-5
- **Description**: Number of optimization rounds
  - Round 1: Full exploration with coarse→fine→TPE
  - Round 2+: Refinement with narrowed search spaces
- **Trade-off**: More rounds = better results but slower
- **Recommended**: 2 for most cases, 3 for critical applications

#### `tpe_trials` (int)
- **Default**: `100`
- **Range**: 50-500
- **Description**: Number of TPE (Bayesian) optimization trials per round
- **Trade-off**: More trials = better exploration but slower
- **Recommended**: 
  - Quick test: 50
  - Standard: 100
  - Thorough: 200

#### `timeout` (int)
- **Default**: `3600` (1 hour)
- **Unit**: Seconds
- **Description**: Maximum time for entire auto-tuning process
- **Behavior**: Stops optimization if timeout is reached
- **Recommended**:
  - Quick test: 1800 (30 min)
  - Standard: 3600 (1 hour)
  - Thorough: 7200 (2 hours)

#### `cache_dir` (str, optional)
- **Default**: `None` (no caching)
- **Description**: Directory to cache optimization results
- **Benefit**: Can resume optimization if interrupted
- **Example**: `'./cache/sticky_hmm_tuning'`

---

## 📈 Expected Performance

### Timing Estimates

Based on ETHUSDT 1h data (~26,000 samples):

| Configuration | Expected Trials | Expected Time | Use Case |
|---------------|-----------------|---------------|----------|
| Quick Test | ~30-50 | 15-25 min | Initial exploration |
| Standard | ~50-150 | 25-90 min | Production use |
| Thorough | ~100-300 | 50-180 min | Critical applications |

**Per Trial Time**: ~30-60 seconds (depends on K and data size)

### Quality Improvement

Expected improvement over default parameters:

- **Composite Score**: +5% to +20%
- **Silhouette Score**: +10% to +30%
- **Temporal Smoothness**: +5% to +15%
- **Regime Balance**: +10% to +25%

*Actual improvements vary by dataset and market conditions*

---

## 🎯 When to Use Auto-Tuning

### ✅ Use Auto-Tuning When:

1. **New Asset**: First time running on an asset
2. **New Timeframe**: Switching from 1h to 4h or 1d
3. **Production Deployment**: Want best possible regime quality
4. **Research/Analysis**: Comparing different configurations
5. **Poor Default Performance**: Default params give composite_score < 0.65

### ❌ Skip Auto-Tuning When:

1. **Quick Iterations**: Testing code changes, not tuning
2. **Limited Time**: Need results in <30 minutes
3. **Known Good Params**: Already have optimized params for this asset
4. **Light Mode**: Running in `execution_mode='light'`

---

## 🔄 Integration with Execution Modes

Auto-tuning respects execution modes for efficient testing:

### Full Mode
```python
config = {
    'execution_mode': 'full',
    'enable_auto_tuning': True,
    'auto_tuning_config': {
        'n_rounds': 2,
        'tpe_trials': 100,
        'timeout': 3600
    }
}
```
- Runs complete optimization
- All trials evaluated thoroughly
- Best for production

### Light Mode
```python
config = {
    'execution_mode': 'light',
    'enable_auto_tuning': True,
    'auto_tuning_config': {
        'n_rounds': 1,
        'tpe_trials': 30,
        'timeout': 900  # 15 min
    }
}
```
- Reduced trials for faster results
- Quick exploration
- Good for development/testing

### Blank Mode
```python
config = {
    'execution_mode': 'blank',
    'enable_auto_tuning': False  # ⚠️ Skip auto-tuning
}
```
- Auto-tuning typically disabled
- Uses minimal resources
- Good for structure testing

---

## 📝 Example Workflows

### Workflow 1: First-Time Asset Analysis

```python
# Step 1: Run with auto-tuning to find best params
config = {
    'symbol': 'BTCUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h',
    'enable_auto_tuning': True,
    'auto_tuning_config': {
        'use_hierarchical': True,
        'n_rounds': 2,
        'tpe_trials': 100,
        'timeout': 3600,
        'cache_dir': './cache/btcusdt_1h_tuning'
    }
}

step = StickyFiniteHMMRegimeDiscoveryStep()
result = await step.execute(config)

# Step 2: Save best params for future use
best_params = result['auto_tuning_results']['best_params']
print(f"Found optimal params: {best_params}")

# Step 3: Use best params for future runs (skip auto-tuning)
config_future = {
    'symbol': 'BTCUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h',
    'enable_auto_tuning': False,  # Skip optimization
    'sticky_finite_hmm_params': best_params  # Use saved params
}
```

### Workflow 2: Periodic Re-tuning

```python
# Re-tune every 3 months to adapt to market changes
from datetime import datetime

last_tuned = datetime(2024, 10, 1)
now = datetime.now()
days_since_tuning = (now - last_tuned).days

if days_since_tuning > 90:  # 3 months
    print("Re-tuning parameters to adapt to market evolution...")
    config['enable_auto_tuning'] = True
else:
    print(f"Using cached params (tuned {days_since_tuning} days ago)")
    config['enable_auto_tuning'] = False
    config['sticky_finite_hmm_params'] = load_cached_params()
```

### Workflow 3: Multi-Asset Batch Processing

```python
assets = [
    ('BTCUSDT', 'binance'),
    ('ETHUSDT', 'binance'),
    ('SOLUSDT', 'binance')
]

optimal_params = {}

for symbol, exchange in assets:
    print(f"\n{'='*80}")
    print(f"Auto-tuning {symbol} on {exchange}")
    print(f"{'='*80}\n")
    
    config = {
        'symbol': symbol,
        'exchange': exchange,
        'regime_timeframe': '1h',
        'enable_auto_tuning': True,
        'auto_tuning_config': {
            'n_rounds': 2,
            'tpe_trials': 100,
            'timeout': 3600
        }
    }
    
    step = StickyFiniteHMMRegimeDiscoveryStep()
    result = await step.execute(config)
    
    # Save optimal params for this asset
    optimal_params[symbol] = result['auto_tuning_results']['best_params']
    print(f"✅ {symbol}: Score={result['composite_score']:.4f}, Params={optimal_params[symbol]}")

# Save all optimal params
import json
with open('optimal_params_by_asset.json', 'w') as f:
    json.dump(optimal_params, f, indent=2)
```

---

## 🎓 Tips & Best Practices

### 1. Start with Standard Configuration
Use the defaults for your first run:
```python
config = {
    'enable_auto_tuning': True,
    'auto_tuning_config': {}  # Use all defaults
}
```

### 2. Monitor Progress
The auto-tuner prints detailed progress:
- Current parameter group being optimized
- Stage (coarse grid → fine grid → TPE)
- Best score so far
- Time remaining

### 3. Cache Results
Use `cache_dir` to save results:
```python
'auto_tuning_config': {
    'cache_dir': f'./cache/{symbol}_{timeframe}_tuning'
}
```

### 4. Interpret Results
After auto-tuning, check:
- **Best score > 0.70**: Excellent regime discovery
- **K value**: Does it make sense? (4-7 regimes typical)
- **kappa**: Realistic regime durations?
- **Compare to defaults**: How much improvement?

### 5. Save Optimal Parameters
Store best params for each asset/timeframe:
```python
# Save to file
optimal_params = {
    'date': datetime.now().isoformat(),
    'symbol': symbol,
    'timeframe': timeframe,
    'params': result['auto_tuning_results']['best_params'],
    'score': result['composite_score']
}

with open(f'params_{symbol}_{timeframe}.json', 'w') as f:
    json.dump(optimal_params, f, indent=2)
```

### 6. Periodically Re-tune
Markets evolve. Re-run auto-tuning:
- Every 3-6 months for crypto
- After major market events (crashes, regime shifts)
- When quality metrics degrade

---

## ⚠️ Troubleshooting

### Auto-Tuning Fails to Import
**Error**: `Auto-tuner not available: No module named 'optuna'`

**Solution**: Install optimization dependencies
```bash
pip install optuna torch pyro-ppl
```

### Timeout Reached
**Issue**: Optimization stops before completion

**Solutions**:
- Increase `timeout` value
- Reduce `tpe_trials`
- Use `n_rounds=1` instead of 2

### Poor Scores After Tuning
**Issue**: Auto-tuned params don't improve over defaults

**Possible Causes**:
1. Data quality issues (missing data, outliers)
2. Insufficient features (check Feature Bank)
3. Timeout too short (stopped early)
4. Asset doesn't have clear regimes

**Solutions**:
- Check data quality
- Add more features
- Increase timeout and trials
- Try different clustering method (HDP-HMM)

### Memory Issues
**Issue**: System runs out of memory during optimization

**Solutions**:
- Reduce `tpe_trials` from 100 to 50
- Use `execution_mode='light'` for smaller data subset
- Close other applications
- Increase system swap space

---

## 📚 Related Documentation

- **PARAMETER_GUIDE.md**: Detailed explanation of all parameters
- **SUMMARY.md**: Overall Sticky Finite HMM status and features
- **IMPLEMENTATION_NOTES.md**: Technical implementation details
- **README.md**: Module overview and quickstart

---

## 🆘 Support

If you encounter issues:

1. Check logs for detailed error messages
2. Verify dependencies are installed
3. Try with reduced `tpe_trials` and `timeout`
4. Check data quality (no NaNs, sufficient samples)
5. Review `PARAMETER_GUIDE.md` for parameter meanings

For persistent issues, check the auto-tuner logs in the step execution results.

