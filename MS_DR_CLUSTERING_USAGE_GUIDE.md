# MS-DR Clustering Usage Guide

Quick reference for using MS-DR clustering with artifact management and default 60m/1h timeframes.

---

## 🚀 Quick Start

### Option 1: Via Launcher (Recommended)

**1. Create config file** (`ms_dr_config.yaml`):
```yaml
steps:
  - step: ms_dr_clustering
    config:
      symbol: ETHUSDT
      exchange: binance
      timeframe: 60m  # Default (1h)
      execution_mode: light
```

**2. Run launcher**:
```bash
python ares_launcher.py --config ms_dr_config.yaml
```

**3. Check artifacts**:
```
artifacts/market_analysis/ms_dr_clustering/YYYY-MM-DD_HHMMSS/
├── ms_dr_regime_labels.parquet.lz4
├── ms_dr_regime_probabilities.parquet.lz4
└── ms_dr_clustering_results.json
```

---

### Option 2: Direct Python Usage

**Simple clustering with artifact saving**:
```python
from src.training.steps.market_analysis.ms_dr_clustering import (
    perform_ms_dr_clustering_with_artifact_manager
)

result = perform_ms_dr_clustering_with_artifact_manager(
    symbol='ETHUSDT',
    timeframe='60m',  # Default
    save_artifacts=True
)

print(f"✅ Found {result['result'].n_clusters} regimes")
print(f"📁 Saved to: {result['artifacts']}")
```

---

### Option 3: With Hyperparameter Optimization

**Using hierarchical HPO (50-70% faster)**:
```python
from src.training.steps.market_analysis.ms_dr_clustering import (
    perform_enhanced_ms_dr_clustering,
    load_market_data_for_msdr
)

# Load data
df = load_market_data_for_msdr(
    symbol='ETHUSDT',
    timeframe='60m',
    execution_mode='light'
)

# Run with HPO
result = perform_enhanced_ms_dr_clustering(
    market_data=df,
    symbol='ETHUSDT',
    enable_optimization=True,
    use_hierarchical=True  # Fast!
)

print(f"Best params: {result['best_params']}")
print(f"Silhouette: {result['metrics']['silhouette_score']:.4f}")
```

---

## 📊 Timeframe Selection

### Recommended: 60m (1h) - Default

**Why 60m is optimal for MS-DR**:
- ✅ Captures meaningful regime changes
- ✅ Statistical significance for Markov-Switching
- ✅ Clear transition boundaries
- ✅ Better model convergence
- ✅ Balanced computation time

### Alternative Timeframes

```python
# Short-term (more regimes, faster switching)
timeframe='30m'

# Long-term (fewer, more stable regimes)
timeframe='4h'

# Note: '1h' is automatically normalized to '60m'
timeframe='1h'  # → becomes '60m'
```

---

## ⚙️ Configuration Quick Reference

### Execution Modes

```python
# Light mode (default) - 30 days, balanced
execution_mode='light'

# Full mode - all data, comprehensive
execution_mode='full'

# Blank mode - 90 days, minimal
execution_mode='blank'
```

### Common Configurations

**Default (recommended)**:
```python
config = {
    'symbol': 'ETHUSDT',
    'timeframe': '60m',
    'execution_mode': 'light',
    'enable_hyperparameter_optimization': False
}
```

**With HPO**:
```python
config = {
    'symbol': 'ETHUSDT',
    'timeframe': '60m',
    'execution_mode': 'full',
    'enable_hyperparameter_optimization': True,
    'use_hierarchical_optimization': True,
    'n_trials_per_group': 20
}
```

**Custom dates**:
```python
config = {
    'symbol': 'BTCUSDT',
    'timeframe': '60m',
    'start_date': '2024-01-01',
    'end_date': '2024-10-28'
}
```

---

## 📦 Accessing Results

### From Step Execution

```python
import asyncio
from src.training.steps.market_analysis.ms_dr_clustering_step import MSDRClusteringStep

async def run():
    step = MSDRClusteringStep()
    result = await step.execute({
        'symbol': 'ETHUSDT',
        'timeframe': '60m'
    })
    
    if result['success']:
        # Access results
        n_regimes = result['n_regimes']
        artifacts = result['artifacts']
        metrics = result['metrics']
        
        print(f"Regimes: {n_regimes}")
        print(f"Silhouette: {metrics['silhouette_score']:.4f}")
        print(f"Time: {result['execution_time']:.2f}s")

asyncio.run(run())
```

### From Convenience Function

```python
result_dict = perform_ms_dr_clustering_with_artifact_manager(
    symbol='ETHUSDT',
    timeframe='60m'
)

# Access MSDRResult
msdr_result = result_dict['result']

# Regime information
print(f"Regimes: {msdr_result.n_clusters}")
print(f"Labels: {msdr_result.cluster_labels}")
print(f"Transition persistence: {msdr_result.transition_persistence:.2%}")

# Quality metrics
print(f"Silhouette: {msdr_result.silhouette_score:.4f}")
print(f"AIC: {msdr_result.aic:.2f}")
print(f"BIC: {msdr_result.bic:.2f}")

# Artifacts
print(f"Saved artifacts: {result_dict['artifacts']}")
```

### Loading Saved Artifacts

```python
import pandas as pd
from src.utils.artifact_manager import ArtifactManager

# Initialize artifact manager
am = ArtifactManager(config={})
am.set_context(step_name='ms_dr_clustering')

# Load regime labels
regime_labels = am.load(
    artifact_name='ms_dr_regime_labels_ETHUSDT_60m',
    artifact_type='data'
)

# Load results
results = am.load(
    artifact_name='ms_dr_results_ETHUSDT_60m',
    artifact_type='metadata'
)

print(regime_labels.head())
print(f"N regimes: {results['n_regimes']}")
```

---

## 🎯 Common Use Cases

### 1. Quick Regime Discovery (No HPO)

```python
from src.training.steps.market_analysis.ms_dr_clustering import (
    perform_ms_dr_clustering_with_artifact_manager
)

# Fast execution, no optimization
result = perform_ms_dr_clustering_with_artifact_manager(
    symbol='ETHUSDT',
    timeframe='60m',
    data_dir='historical_data'
)

print(f"Found {result['result'].n_clusters} regimes in {result['result'].processing_time:.1f}s")
```

### 2. Production-Grade with HPO

```python
from src.training.steps.market_analysis.ms_dr_clustering import (
    perform_enhanced_ms_dr_clustering,
    load_market_data_for_msdr
)

# Load data
df = load_market_data_for_msdr(
    symbol='ETHUSDT',
    timeframe='60m',
    execution_mode='full'
)

# Optimize and cluster
result = perform_enhanced_ms_dr_clustering(
    market_data=df,
    symbol='ETHUSDT',
    enable_optimization=True,
    use_hierarchical=True,
    save_artifacts=True
)

# Best configuration found
print(f"Optimal n_regimes: {result['best_params']['n_regimes']}")
print(f"Optimal order: {result['best_params']['order']}")
```

### 3. Batch Processing Multiple Symbols

```python
import asyncio
from src.training.steps.market_analysis.ms_dr_clustering_step import MSDRClusteringStep

async def batch_process(symbols):
    step = MSDRClusteringStep()
    
    for symbol in symbols:
        print(f"\n🔄 Processing {symbol}...")
        
        result = await step.execute({
            'symbol': symbol,
            'timeframe': '60m',
            'execution_mode': 'light'
        })
        
        if result['success']:
            print(f"✅ {symbol}: {result['n_regimes']} regimes")
        else:
            print(f"❌ {symbol}: {result['error']}")

symbols = ['ETHUSDT', 'BTCUSDT', 'SOLUSDT']
asyncio.run(batch_process(symbols))
```

### 4. Custom Configuration with Enhancements

```python
from src.training.steps.market_analysis.ms_dr_clustering import (
    MSDRClusterer, MSDRConfig
)
from src.utils.data.klines_parquet import get_klines_manager

# Load data
km = get_klines_manager()
df = km.read_data(symbol='ETHUSDT', interval='60m', data_type='processed')

# Custom config with all enhancements
config = MSDRConfig(
    n_regimes=5,
    auto_select_regimes=True,
    min_regimes=2,
    max_regimes=10,
    ic_criterion='bic',
    enable_pca=True,
    pca_components=12,
    # Enhancements
    use_safe_math=True,
    use_memory_optimization=True,
    use_hardware_acceleration=True,
    use_vectorbt_operations=True,
    use_parallel_selection=True,
    max_workers=4
)

# Run clustering
clusterer = MSDRClusterer(config)
result = clusterer.fit_predict(df.values)

print(f"Regimes: {result.n_clusters}")
print(f"Quality: {result.silhouette_score:.4f}")
```

---

## 🔍 Troubleshooting

### No Data Loaded

```python
# Check available data
from src.utils.data.klines_parquet import get_klines_manager

km = get_klines_manager()
symbols = km.list_symbols()
print(f"Available symbols: {symbols}")

# Verify your symbol exists
if 'ETHUSDT' in symbols:
    df = km.read_data('ETHUSDT', '60m', 'processed')
    print(f"Loaded {len(df)} rows")
```

### HPO Timeout

```python
# Increase timeout
config = {
    'symbol': 'ETHUSDT',
    'enable_hyperparameter_optimization': True,
    'timeout_minutes': 60.0,  # Increase from default 30
    'n_trials_per_group': 15  # Or reduce trials
}
```

### Memory Issues

```python
# Use blank mode or shorter date range
config = {
    'symbol': 'ETHUSDT',
    'timeframe': '60m',
    'execution_mode': 'blank',  # Uses 90 days instead of all
    'start_date': '2024-09-01',
    'end_date': '2024-10-28'
}
```

---

## 📊 Understanding Results

### Regime Labels

```python
# Regime labels are integers starting from 0
result = perform_ms_dr_clustering_with_artifact_manager(...)
labels = result['result'].cluster_labels

print(f"Unique regimes: {np.unique(labels)}")
print(f"Regime 0 count: {np.sum(labels == 0)}")
print(f"Regime 1 count: {np.sum(labels == 1)}")
```

### Transition Matrix

```python
# Shows probability of switching from one regime to another
transition_matrix = result['result'].transition_matrix

# Diagonal elements = probability of staying in same regime
persistence = np.diag(transition_matrix)
print(f"Regime persistence: {persistence}")

# Off-diagonal = switching probabilities
print("Transition probabilities:")
print(transition_matrix)
```

### Quality Metrics

```python
metrics = result['metrics']

# Clustering quality (higher is better)
print(f"Silhouette: {metrics['silhouette_score']:.4f}")  # [-1, 1]
print(f"Calinski-Harabasz: {metrics['calinski_harabasz_score']:.2f}")  # Higher better

# Model selection (lower is better)
print(f"AIC: {metrics['aic']:.2f}")
print(f"BIC: {metrics['bic']:.2f}")

# Performance
print(f"Time: {metrics['processing_time_seconds']:.2f}s")
print(f"Memory: {metrics['memory_usage_mb']:.1f} MB")
```

---

## 🎓 Advanced Features

### Hierarchical HPO (Faster Optimization)

```python
# Standard HPO: ~15-30 min for 30 days of 60m data
result_standard = perform_enhanced_ms_dr_clustering(
    market_data=df,
    enable_optimization=True,
    use_hierarchical=False  # Standard
)

# Hierarchical HPO: ~6-12 min for same data (50-70% faster!)
result_hierarchical = perform_enhanced_ms_dr_clustering(
    market_data=df,
    enable_optimization=True,
    use_hierarchical=True  # Hierarchical
)
```

### Hardware Acceleration

```python
# Automatically detects and uses:
# - Available CPU cores
# - System memory
# - Optimal parallelization

config = MSDRConfig(
    use_hardware_acceleration=True,  # Auto-detect hardware
    max_workers=-1  # Use all available cores
)
```

### Safe Mathematical Operations

```python
# Prevents division by zero, NaN, and Inf errors
config = MSDRConfig(
    use_safe_math=True  # Enabled by default
)
# Automatically applies safe_divide and array validation
```

---

## 📚 Additional Resources

- **Full Documentation**: `MS_DR_CLUSTERING_INTEGRATION_COMPLETE.md`
- **Code Review**: `MS_DR_CLUSTERING_CODE_REVIEW.md`
- **Bug Fixes**: `MS_DR_CLUSTERING_FIXES_IMPLEMENTED.md`
- **Enhancements**: `MS_DR_CLUSTERING_IMPLEMENTATIONS_COMPLETE.md`
- **Quick Reference**: `MS_DR_CLUSTERING_QUICK_REFERENCE.md`

---

## 🎉 Summary

**Default Setup** (recommended for most use cases):
```python
from src.training.steps.market_analysis.ms_dr_clustering import (
    perform_ms_dr_clustering_with_artifact_manager
)

result = perform_ms_dr_clustering_with_artifact_manager(
    symbol='ETHUSDT',
    timeframe='60m',  # Default, optimal
    save_artifacts=True
)

print(f"✅ {result['result'].n_clusters} regimes discovered")
print(f"📁 Artifacts: {result['artifacts']}")
```

**Key Points**:
- ✅ Default timeframe is **60m (1h)** - optimal for MS-DR
- ✅ Artifacts are **automatically saved** with compression
- ✅ Hierarchical HPO provides **50-70% speedup**
- ✅ All **enhancements enabled** by default (safe math, memory opt, hardware accel)
- ✅ **Three execution modes** for different use cases (full/light/blank)

Happy regime discovering! 🚀
