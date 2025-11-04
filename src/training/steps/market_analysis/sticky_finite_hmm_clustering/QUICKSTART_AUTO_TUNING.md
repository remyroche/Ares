# Auto-Tuning Quickstart ⚡

## ✅ One-Line Activation - It's Automatic!

Auto-tuning is **enabled by default** in full mode. Just run:

```python
from src.training.steps.market_analysis.sticky_finite_hmm_clustering import (
    StickyFiniteHMMRegimeDiscoveryStep
)

# That's it! Auto-tuning runs automatically
config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h'
}

step = StickyFiniteHMMRegimeDiscoveryStep()
result = await step.execute(config)

# Auto-tuning happened automatically!
print(f"Score: {result['composite_score']:.4f}")
print(f"Best K: {result['auto_tuning_results']['best_params']['K']}")
print(f"Best kappa: {result['auto_tuning_results']['best_params']['kappa']:.2f}")
```

---

## 🎛️ Smart Behavior

Auto-tuning intelligently adapts to your execution mode:

### ✅ Auto-Tuning ON (Always, with Adaptive Speed)

```python
# Case 1: Full mode (100% trials: 2 rounds, 100 trials, 60 min)
config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h'
    # execution_mode defaults to 'full'
}
# → Auto-tuning: ✅ ON (complete optimization)

# Case 2: Light mode (50% trials: 1 round, 50 trials, 30 min)
config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h',
    'execution_mode': 'light'
}
# → Auto-tuning: ✅ ON (faster for testing)

# Case 3: Blank mode (25% trials: 1 round, 25 trials, 15 min)
config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h',
    'execution_mode': 'blank'
}
# → Auto-tuning: ✅ ON (minimal but still optimizes)

# Case 4: Force auto-tuning even with manual params
config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h',
    'enable_auto_tuning': True,  # Explicit enable
    'sticky_finite_hmm_params': {'K': 5}  # Will be ignored, auto-tuned params used
}
# → Auto-tuning: ✅ ON (explicit override)
```

### ⚡ Auto-Tuning with Adaptive Speed

```python
# Case 1: Light mode (faster auto-tuning)
config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h',
    'execution_mode': 'light'
}
# → Auto-tuning: ✅ ON (50% trials: 50 trials, 30 min max)
# → Message: "ℹ️  Auto-tuning enabled in 'light' mode (will use reduced trials for speed)"

# Case 2: Blank mode (minimal auto-tuning)
config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h',
    'execution_mode': 'blank'
}
# → Auto-tuning: ✅ ON (25% trials: 25 trials, 15 min max)
# → Message: "ℹ️  Auto-tuning enabled in 'blank' mode (will use reduced trials for speed)"
```

### ❌ Auto-Tuning OFF (Manual Override)

```python
# Case 1: Manual parameters provided
config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h',
    'sticky_finite_hmm_params': {
        'K': 5,
        'kappa': 20.0,
        'base_alpha': 0.3
    }
}
# → Auto-tuning: ❌ OFF (manual params = skip auto-tuning)
# → Message: "ℹ️  Manual params provided, skipping auto-tuning"

# Case 2: Explicit disable
config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h',
    'enable_auto_tuning': False
}
# → Auto-tuning: ❌ OFF (explicit disable)
```

---

## 🎯 Common Scenarios

### Scenario 1: First Time - Let It Auto-Tune

```python
# Just provide symbol and exchange
config = {
    'symbol': 'BTCUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h'
}

step = StickyFiniteHMMRegimeDiscoveryStep()
result = await step.execute(config)

# Auto-tuning runs automatically (~25-90 min)
# Best params found and applied
# Results saved with auto-tuning summary
```

### Scenario 2: Use Previously Found Params

```python
# Load saved optimal params
import json
with open('btcusdt_1h_params.json', 'r') as f:
    saved_params = json.load(f)

config = {
    'symbol': 'BTCUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h',
    'sticky_finite_hmm_params': saved_params  # Use saved params
}

step = StickyFiniteHMMRegimeDiscoveryStep()
result = await step.execute(config)

# Auto-tuning skipped (manual params provided)
# Uses saved optimal params directly
# Fast execution (~30-60 seconds)
```

### Scenario 3: Quick Test in Light Mode

```python
# Fast auto-tuning for testing
config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h',
    'execution_mode': 'light'  # Quick test
}

step = StickyFiniteHMMRegimeDiscoveryStep()
result = await step.execute(config)

# Auto-tuning runs with 50% trials (50 trials, ~15-30 min)
# Finds good params quickly
# Much faster than full mode
```

### Scenario 4: Force Auto-Tuning Despite Manual Params

```python
# Re-tune even though we have manual params
config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h',
    'enable_auto_tuning': True,  # Explicit override
    'sticky_finite_hmm_params': {'K': 5}  # Will be overridden
}

step = StickyFiniteHMMRegimeDiscoveryStep()
result = await step.execute(config)

# Auto-tuning runs despite manual params
# Finds better params than manual ones
# Manual params ignored
```

---

## ⚙️ Customize Auto-Tuning

```python
# Fine-tune the optimization process
config = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'regime_timeframe': '1h',
    
    # Auto-tuning happens automatically, but customize it:
    'auto_tuning_config': {
        'use_hierarchical': True,  # Fast hierarchical (recommended)
        'n_rounds': 2,  # Optimization rounds
        'tpe_trials': 100,  # Trials per round
        'timeout': 3600,  # 1 hour max
        'cache_dir': './cache/ethusdt_tuning'  # Cache results
    }
}

step = StickyFiniteHMMRegimeDiscoveryStep()
result = await step.execute(config)
```

---

## 📊 Quick Reference Table

| Config | Auto-Tuning | Trials | Time | Use Case |
|--------|-------------|--------|------|----------|
| Full mode (default) | ✅ ON | 100 (2 rounds) | ~25-90 min | Production |
| Light mode | ✅ ON | 50 (1 round) | ~15-30 min | Fast testing |
| Blank mode | ✅ ON | 25 (1 round) | ~8-15 min | Minimal test |
| With manual params | ❌ OFF | 0 | ~30-60 sec | Use saved params |
| `enable_auto_tuning=False` | ❌ OFF | 0 | ~30-60 sec | Skip optimization |

---

## 💡 Tips

### Save Optimal Params After First Run

```python
# First run: auto-tune
result = await step.execute(config)

# Save best params
best_params = result['auto_tuning_results']['best_params']
with open(f'params_{symbol}_{timeframe}.json', 'w') as f:
    json.dump(best_params, f, indent=2)

# Future runs: use saved params (much faster)
config['sticky_finite_hmm_params'] = best_params
```

### Re-tune Periodically

```python
from datetime import datetime

# Check last tuning date
try:
    with open('btcusdt_1h_params.json', 'r') as f:
        data = json.load(f)
        last_tuned = datetime.fromisoformat(data.get('date', '2020-01-01'))
        days_since = (datetime.now() - last_tuned).days
except:
    days_since = 999

# Re-tune every 3 months
if days_since > 90:
    # Auto-tune (default behavior)
    config = {'symbol': 'BTCUSDT', 'exchange': 'binance', 'regime_timeframe': '1h'}
else:
    # Use saved params
    config['sticky_finite_hmm_params'] = data['params']
```

---

## 🚀 That's It!

**Auto-tuning is now automatic and adaptive!**

- ✅ Full mode → Complete optimization (100 trials, ~25-90 min)
- ✅ Light mode → Fast optimization (50 trials, ~15-30 min)  
- ✅ Blank mode → Minimal optimization (25 trials, ~8-15 min)
- ✅ Manual params → Skip optimization (respects your choice)
- ✅ Explicit enable/disable → Always respected (override)

**No configuration needed for the default behavior!**

---

## 📚 More Info

- **AUTO_TUNING_GUIDE.md** - Complete usage guide with workflows
- **PARAMETER_GUIDE.md** - All parameters explained
- **AUTO_TUNING_INTEGRATION_SUMMARY.md** - Technical details

