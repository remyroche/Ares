# Auto-Tuning Integration Complete! 🎉

## Summary

The Sticky Finite HMM regime discovery step now automatically activates the auto-tuner when configured to do so. This provides end-to-end hyperparameter optimization integrated directly into the step execution.

---

## ✅ What Was Implemented

### 1. **Auto-Tuner Core** (`sticky_finite_hmm_auto_tuner.py`)
- Hierarchical parameter optimization (3 groups, 5 parameters)
- Grid search (coarse → fine) + TPE Bayesian optimization
- Expected ~50-150 trials vs 24,200 full grid combinations
- **Files**: 680 lines

### 2. **Step Integration** (`sticky_finite_hmm_regime_discovery_step.py`)
- Added `_run_auto_tuning()` method
- Automatic parameter discovery and application
- Async execution (non-blocking)
- Detailed progress logging
- **Changes**: +95 lines

### 3. **Documentation** (3 comprehensive guides)
- `PARAMETER_GUIDE.md` - All 13 parameters explained
- `AUTO_TUNING_GUIDE.md` - Complete usage guide with examples
- `AUTO_TUNING_INTEGRATION_SUMMARY.md` - This file

---

## 🎯 5 Optimized Parameters

| Parameter | Range | Default | What It Controls | Impact |
|-----------|-------|---------|------------------|---------|
| **K** | 4-7 | 5 | Number of regimes | 🔴 CRITICAL |
| **kappa** | 5-50 | 10.0 | Regime persistence (duration) | 🔴 CRITICAL |
| **base_alpha** | 0.1-1.0 | 0.5 | Transition sparsity | 🟡 IMPORTANT |
| **lr** | 1e-4 to 1e-1 | 1e-2 | Learning rate | 🟡 IMPORTANT |
| **pca_components** | 10-20 | 15 | Feature dimensionality | 🟡 IMPORTANT |

### Regime Duration Examples (kappa impact):
- kappa=10 → ~11 timesteps (~11 hours for 1h data)
- kappa=30 → ~28 timesteps
- kappa=50 → ~44 timesteps

---

## 🔒 8 Fixed Parameters

Not optimized (sensible defaults):
1. **num_iters**: 1000 - Sufficient with early stopping
2. **num_particles**: 10 - Good balance for gradient estimation
3. **prior_mean_scale**: 10.0 - Works well for standardized features
4. **prior_cov_scale**: 1.0 - Reasonable variance
5. **patience**: 50 - Robust early stopping
6. **elbo_improvement_threshold**: 1e-3 - Good convergence threshold
7. **min_features**: 50 - Adequate signal
8. **max_features**: 100 - Prevents overfitting

---

## 🚀 Usage

### Enable Auto-Tuning

```python
from src.training.steps.market_analysis.sticky_finite_hmm_clustering import (
    StickyFiniteHMMRegimeDiscoveryStep
)

config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h',
    
    # ✅ Enable auto-tuning
    'enable_auto_tuning': True,
    
    # Optional: Configure auto-tuning
    'auto_tuning_config': {
        'use_hierarchical': True,  # Recommended (3-5x faster)
        'n_rounds': 2,  # Optimization rounds
        'tpe_trials': 100,  # TPE trials per round
        'timeout': 3600,  # 1 hour max
        'cache_dir': './cache/tuning'  # Optional: cache results
    }
}

# Run with auto-tuning
step = StickyFiniteHMMRegimeDiscoveryStep()
result = await step.execute(config)

# Check results
print(f"Success: {result['success']}")
print(f"Composite Score: {result['composite_score']:.4f}")
print(f"Regimes: {result['n_regimes']}")

# Auto-tuning summary
if 'auto_tuning_results' in result:
    tuning = result['auto_tuning_results']
    print(f"\n🎯 Auto-Tuning Results:")
    print(f"  Best Score: {tuning['best_score']:.4f}")
    print(f"  Total Time: {tuning['total_time']:.1f}s")
    print(f"  Trials: {tuning['total_trials']}")
    print(f"  Best K: {tuning['best_params']['K']}")
    print(f"  Best kappa: {tuning['best_params']['kappa']:.2f}")
```

### Disable Auto-Tuning (Use Defaults or Manual Params)

```python
config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h',
    
    # ❌ Skip auto-tuning (default behavior)
    'enable_auto_tuning': False,
    
    # Optional: Manually specify params
    'sticky_finite_hmm_params': {
        'K': 5,
        'kappa': 20.0,
        'base_alpha': 0.3,
        'lr': 1e-2,
        'pca_components': 15
    }
}
```

---

## 📊 Expected Performance

### Timing (ETHUSDT 1h, ~26K samples)

| Configuration | Trials | Time | Use Case |
|---------------|--------|------|----------|
| Quick Test | ~30-50 | 15-25 min | Initial exploration |
| Standard | ~50-150 | 25-90 min | Production |
| Thorough | ~100-300 | 50-180 min | Critical applications |

**Per trial**: ~30-60 seconds

### Quality Improvement

Expected improvement over defaults:
- **Composite Score**: +5% to +20%
- **Silhouette Score**: +10% to +30%
- **Temporal Smoothness**: +5% to +15%

---

## 🔄 Execution Flow

### With Auto-Tuning Enabled:

```
1. Load market data
   ↓
2. 🎯 Run auto-tuning
   ├── Coarse grid search (structure, transitions, training)
   ├── Fine grid search (refine around best)
   └── TPE optimization (final refinement)
   ↓
3. Apply best parameters to config
   ↓
4. Run clustering with optimal params
   ↓
5. Save artifacts + auto-tuning results
```

### With Auto-Tuning Disabled:

```
1. Load market data
   ↓
2. Run clustering with default/manual params
   ↓
3. Save artifacts
```

---

## 🎓 Best Practices

### 1. When to Enable Auto-Tuning

✅ **Enable when**:
- First time running on an asset
- New timeframe (1h → 4h → 1d)
- Production deployment
- Research/comparison studies
- Poor default performance (score < 0.65)

❌ **Disable when**:
- Quick iterations/testing
- Limited time (<30 min available)
- Known good params available
- Light mode testing

### 2. Save Optimal Parameters

```python
# After auto-tuning
optimal_params = result['auto_tuning_results']['best_params']

# Save for future use
import json
with open(f'params_{symbol}_{timeframe}.json', 'w') as f:
    json.dump({
        'date': datetime.now().isoformat(),
        'symbol': symbol,
        'timeframe': timeframe,
        'params': optimal_params,
        'score': result['composite_score']
    }, f, indent=2)

# Future runs: load and use
with open(f'params_{symbol}_{timeframe}.json', 'r') as f:
    saved = json.load(f)
    config['sticky_finite_hmm_params'] = saved['params']
    config['enable_auto_tuning'] = False  # Skip re-tuning
```

### 3. Periodic Re-tuning

Markets evolve. Re-tune every 3-6 months:

```python
from datetime import datetime

last_tuned = datetime.fromisoformat(saved['date'])
days_since = (datetime.now() - last_tuned).days

if days_since > 90:  # 3 months
    config['enable_auto_tuning'] = True
else:
    config['enable_auto_tuning'] = False
    config['sticky_finite_hmm_params'] = saved['params']
```

### 4. Monitor Results

Check these after auto-tuning:
- **Best Score > 0.70**: Excellent
- **K value**: 4-7 expected (does it make sense?)
- **kappa**: Realistic regime durations?
- **Improvement**: How much better than defaults?

---

## 📁 File Structure

```
src/training/steps/market_analysis/sticky_finite_hmm_clustering/
├── sticky_finite_hmm_auto_tuner.py           # ✨ NEW: Auto-tuner core
├── sticky_finite_hmm_regime_discovery_step.py # ✨ UPDATED: Integrated auto-tuning
├── sticky_finite_hmm_clusterer.py            # Clustering implementation
├── standalone_runner.py                       # Standalone execution
├── __init__.py                               # ✨ UPDATED: Export auto-tuner
├── PARAMETER_GUIDE.md                        # ✨ NEW: All params explained
├── AUTO_TUNING_GUIDE.md                      # ✨ NEW: Usage guide
├── AUTO_TUNING_INTEGRATION_SUMMARY.md        # ✨ NEW: This file
├── SUMMARY.md                                # Module status
├── IMPLEMENTATION_NOTES.md                    # Technical details
└── README.md                                 # Module overview
```

---

## 🔧 Configuration Reference

### Full Config Example

```python
config = {
    # === REQUIRED ===
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h',
    
    # === AUTO-TUNING ===
    'enable_auto_tuning': True,  # Enable/disable
    
    'auto_tuning_config': {
        # Optimization strategy
        'use_hierarchical': True,  # Hierarchical (fast) vs standard (slow)
        'n_rounds': 2,  # Optimization rounds (1-5)
        'tpe_trials': 100,  # TPE trials per round (50-500)
        'timeout': 3600,  # Max time in seconds (1800-7200)
        'cache_dir': './cache/tuning',  # Optional: cache results
    },
    
    # === MANUAL PARAMS (used if auto-tuning disabled) ===
    'sticky_finite_hmm_params': {
        # Optimized params
        'K': 5,
        'kappa': 10.0,
        'base_alpha': 0.5,
        'lr': 1e-2,
        'pca_components': 15,
        
        # Fixed params (optional overrides)
        'num_iters': 1000,
        'num_particles': 10,
        'prior_mean_scale': 10.0,
        'prior_cov_scale': 1.0,
        'patience': 50,
        'elbo_improvement_threshold': 1e-3,
        'min_features': 50,
        'max_features': 100
    },
    
    # === EXECUTION MODE ===
    'execution_mode': 'full',  # 'full', 'light', or 'blank'
}
```

---

## ⚠️ Troubleshooting

### Auto-Tuner Not Available
**Error**: `Auto-tuner not available: No module named 'optuna'`

**Solution**:
```bash
pip install optuna torch pyro-ppl
```

### Timeout Reached
**Issue**: Optimization stops early

**Solutions**:
- Increase `timeout` (e.g., 7200 for 2 hours)
- Reduce `tpe_trials` (e.g., 50 instead of 100)
- Use `n_rounds=1` instead of 2

### Poor Results
**Issue**: Auto-tuned params don't improve over defaults

**Check**:
1. Data quality (no NaNs, sufficient samples)
2. Feature quality (check Feature Bank output)
3. Timeout sufficient (not stopped early)
4. Compare: default score vs auto-tuned score

### Memory Issues
**Solutions**:
- Reduce `tpe_trials` to 50
- Use `execution_mode='light'`
- Close other applications

---

## 📚 Additional Resources

- **PARAMETER_GUIDE.md**: Detailed parameter explanations
- **AUTO_TUNING_GUIDE.md**: Complete usage examples
- **SUMMARY.md**: Overall module status
- **IMPLEMENTATION_NOTES.md**: Technical implementation

---

## 🎉 Summary

**Auto-tuning is now fully integrated!**

- ✅ Enable with one config flag
- ✅ Optimizes 5 key parameters automatically
- ✅ 50-150 trials in 25-90 minutes
- ✅ Expected +5-20% improvement
- ✅ Detailed progress logging
- ✅ Results saved in step artifacts

Simply set `'enable_auto_tuning': True` in your config and the step handles the rest!

