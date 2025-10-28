# HDP-HMM Auto-Tuning Guide

Complete guide for automatic hyperparameter tuning of HDP-HMM clustering using multi-stage optimization.

## Overview

The HDP-HMM auto-tuner optimizes hyperparameters to maximize the **composite_score** from cluster quality assessment. It uses a three-stage approach:

1. **Coarse Grid Search** - Broad exploration with sparse grid
2. **Fine Grid Search** - Refinement around best results  
3. **TPE Optimization** - Bayesian optimization for final tuning

## Quick Start

### Basic Auto-Tuning

```python
import pandas as pd
from src.training.steps.market_analysis.hdp_hmm_clustering import run_hdp_hmm_auto_tuning

# Load market data
df = pd.read_csv("market_data.csv", index_col=0, parse_dates=True)

# Run auto-tuning (uses sensible defaults)
best_params, best_score, tuning_result = run_hdp_hmm_auto_tuning(
    market_data=df,
    symbol="ETHUSDT",
    exchange="binance",
    timeframe="1h"
)

print(f"Best composite score: {best_score:.4f}")
print(f"Best parameters: {best_params}")

# Use optimized parameters for final clustering
from src.training.steps.market_analysis.hdp_hmm_clustering import run_hdp_hmm_clustering

final_results = run_hdp_hmm_clustering(
    market_data=df,
    symbol="ETHUSDT",
    exchange="binance",
    timeframe="1h",
    **best_params
)
```

### With Custom Settings

```python
# More thorough search with custom timeout
best_params, best_score, tuning_result = run_hdp_hmm_auto_tuning(
    market_data=df,
    symbol="ETHUSDT",
    exchange="binance",
    timeframe="1h",
    coarse_grid_points=4,    # More exploration (4^7 = 16,384 combinations)
    fine_grid_points=4,      # More refinement
    tpe_trials=100,          # More TPE trials
    timeout=7200,            # 2 hours
    save_results=True
)
```

## Multi-Stage Optimization Explained

### Stage 1: Coarse Grid Search

**Purpose**: Broad exploration of parameter space

**How it works**:
- Creates a sparse grid across all parameters
- Default: 3 points per parameter → 3^7 = 2,187 combinations
- Evaluates each combination's composite score
- Identifies promising regions

**Example**:
```python
# For alpha ∈ [2.0, 5.0] with 3 points:
alpha_values = [2.0, 3.5, 5.0]

# Full grid:
# alpha: [2.0, 3.5, 5.0]
# kappa: [30.0, 50.0, 70.0]
# gamma: [2.0, 3.5, 5.0]
# ... etc
```

**Typical output**:
```
Stage 1: COARSE GRID SEARCH
============================
Evaluating 2,187 parameter combinations
Trial 1/2187: alpha=2.0, kappa=30.0, ... → score=0.42
Trial 2/2187: alpha=2.0, kappa=50.0, ... → score=0.51
...
✨ New best score: 0.65
Best params: {alpha: 3.5, kappa: 50.0, ...}
```

**Time**: ~20-40% of total tuning time

### Stage 2: Fine Grid Search

**Purpose**: Refinement around best coarse results

**How it works**:
- Narrows search space to ±20% around best parameters
- Creates denser grid in this region
- Further refines the optimal configuration

**Example**:
```python
# If best coarse: alpha=3.5, range [2.0, 5.0]
# Fine grid narrows to: alpha ∈ [2.9, 4.1] (±20% of original range)
# With 3 points: [2.9, 3.5, 4.1]
```

**Typical output**:
```
Stage 2: FINE GRID SEARCH
=========================
Evaluating 2,187 parameter combinations around best
Best params from coarse: {alpha: 3.5, kappa: 50.0, ...}
Best score: 0.65

Trial 1/2187: alpha=2.9, kappa=46.0, ... → score=0.63
Trial 2/2187: alpha=3.5, kappa=50.0, ... → score=0.65
...
✨ New best score: 0.68
```

**Time**: ~20-40% of total tuning time

### Stage 3: TPE Bayesian Optimization

**Purpose**: Final optimization using probabilistic models

**How it works**:
- Uses Tree-structured Parzen Estimator (TPE) from Optuna
- Learns from previous trials
- Intelligently samples promising regions
- Balances exploration vs exploitation

**Algorithm**:
```
1. Build probabilistic model of objective function
2. Sample promising parameters using acquisition function
3. Evaluate clustering with these parameters
4. Update model with results
5. Repeat until convergence or trial limit
```

**Typical output**:
```
Stage 3: TPE BAYESIAN OPTIMIZATION
===================================
Running 50 TPE trials

[I 2025-10-28 14:30:15,123] Trial 0 finished with value: 0.70
[I 2025-10-28 14:32:18,456] Trial 1 finished with value: 0.68
...
✨ New best score: 0.72
Best params: {alpha: 3.42, kappa: 48.5, ...}
```

**Time**: ~40-60% of total tuning time

## Parameters Being Tuned

### 1. alpha (Concentration Parameter)
- **Range**: 2.0 - 5.0 (default search space)
- **Effect**: Controls regime diversity
- **Higher values**: More regimes discovered
- **Lower values**: Fewer, broader regimes

### 2. kappa (Stickiness Parameter)
- **Range**: 30.0 - 70.0 (default)
- **Effect**: Controls regime persistence
- **Higher values**: Longer regime durations
- **Lower values**: More frequent regime switches

### 3. gamma (Base Distribution Hyperparameter)
- **Range**: 2.0 - 5.0 (default)
- **Effect**: Controls prior over states
- **Higher values**: More uniform state priors
- **Lower values**: Stronger prior concentration

### 4. n_iterations (Gibbs Sampling Iterations)
- **Range**: 100 - 200 (default)
- **Effect**: Convergence quality
- **Higher values**: Better convergence, slower
- **Lower values**: Faster, may not converge

### 5. min_features (Minimum Feature Count)
- **Range**: 40 - 60 (default)
- **Effect**: Ensures adequate signal
- **Higher values**: More comprehensive, riskier
- **Lower values**: Faster, may miss patterns

### 6. max_features (Maximum Feature Count)
- **Range**: 80 - 120 (default)
- **Effect**: Controls complexity
- **Higher values**: More detailed, slower
- **Lower values**: Faster, simpler

### 7. pca_components (PCA Dimensionality)
- **Range**: 8 - 15 (default)
- **Effect**: Feature space reduction
- **Higher values**: More variance retained
- **Lower values**: More aggressive reduction

## Objective Function: Composite Score

The auto-tuner maximizes the **composite_score** which combines:

### 1. Silhouette Score (20% weight)
- Measures cluster cohesion
- Range: -1 to 1 (higher is better)
- Target: ≥ 0.2

### 2. Davies-Bouldin Index (15% weight)
- Measures cluster separation
- Range: 0 to ∞ (lower is better)
- Target: ≤ 2.0

### 3. CV Ratio (15% weight)
- Between-cluster / within-cluster variance
- Range: 0 to ∞ (higher is better)
- Target: ≥ 1.0

### 4. Balance Score (15% weight)
- Cluster size distribution
- Range: 0 to 1 (higher is better)
- Target: ≥ 0.5

### 5. Temporal Smoothness (10% weight)
- Regime stability over time
- Range: 0 to 1 (higher is better)
- Target: ≥ 0.85

### 6. Noise Ratio (10% weight)
- Inverted (1 - ratio)
- Range: 0 to 1 (higher is better)
- Target: < 0.3 noise

**Formula**:
```python
composite_score = (
    0.20 * (silhouette + 1) / 2 +           # Normalized to [0, 1]
    0.15 * (1 / (1 + davies_bouldin)) +     # Inverted and normalized
    0.15 * tanh(cv_ratio) +                  # Sigmoid-like normalization
    0.15 * balance_score +
    0.10 * temporal_smoothness +
    0.10 * (1 - noise_ratio)
)
```

## Custom Search Spaces

You can define custom search spaces for your specific needs:

### Conservative Search
```python
from src.training.steps.market_analysis.hdp_hmm_clustering import (
    HDPHMMSearchSpace,
    run_hdp_hmm_auto_tuning
)

# Narrow search space for faster tuning
conservative_space = HDPHMMSearchSpace(
    alpha_min=2.5,
    alpha_max=4.0,         # Less extreme values
    kappa_min=40.0,
    kappa_max=60.0,        # Moderate persistence
    n_iterations_min=100,
    n_iterations_max=150,  # Faster convergence
    min_features_min=45,
    min_features_max=55,   # Focused feature range
    max_features_min=85,
    max_features_max=105
)

best_params, best_score, results = run_hdp_hmm_auto_tuning(
    market_data=df,
    search_space=conservative_space,
    tpe_trials=30  # Fewer trials needed
)
```

### Aggressive Search
```python
# Wide search space for thorough exploration
aggressive_space = HDPHMMSearchSpace(
    alpha_min=1.5,
    alpha_max=7.0,          # Very wide range
    kappa_min=20.0,
    kappa_max=90.0,         # Full persistence spectrum
    n_iterations_min=50,
    n_iterations_max=300,   # Wide convergence range
    min_features_min=30,
    min_features_max=70,    # Broad feature exploration
    max_features_min=60,
    max_features_max=140    # Maximum comprehensiveness
)

best_params, best_score, results = run_hdp_hmm_auto_tuning(
    market_data=df,
    search_space=aggressive_space,
    coarse_grid_points=5,   # More exploration
    tpe_trials=100,         # More refinement
    timeout=14400           # 4 hours
)
```

## Analyzing Results

### Accessing Tuning History

```python
# Run auto-tuning
best_params, best_score, tuning_result = run_hdp_hmm_auto_tuning(
    market_data=df,
    symbol="ETHUSDT"
)

# Access all trial results
print(f"Total trials: {tuning_result.n_trials}")
print(f"Total time: {tuning_result.total_time:.2f}s")

# Coarse grid results
print(f"\nCoarse grid: {len(tuning_result.coarse_grid_results)} trials")
for trial in tuning_result.coarse_grid_results[:5]:
    print(f"  Score: {trial['score']:.4f}, Params: {trial['params']}")

# Fine grid results  
print(f"\nFine grid: {len(tuning_result.fine_grid_results)} trials")

# TPE results
print(f"\nTPE: {len(tuning_result.tpe_results)} trials")

# Convergence info
print(f"\nConvergence: {tuning_result.convergence_info}")
```

### Visualizing Progress

```python
import matplotlib.pyplot as plt

# Extract scores from all stages
coarse_scores = [t['score'] for t in tuning_result.coarse_grid_results]
fine_scores = [t['score'] for t in tuning_result.fine_grid_results]
tpe_scores = [t['score'] for t in tuning_result.tpe_results]

# Plot progression
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(coarse_scores, bins=20)
plt.title("Coarse Grid Scores")
plt.xlabel("Composite Score")

plt.subplot(1, 3, 2)
plt.hist(fine_scores, bins=20)
plt.title("Fine Grid Scores")
plt.xlabel("Composite Score")

plt.subplot(1, 3, 3)
plt.plot(tpe_scores)
plt.title("TPE Optimization")
plt.xlabel("Trial")
plt.ylabel("Composite Score")

plt.tight_layout()
plt.savefig("tuning_progress.png")
```

## Time and Resource Estimates

### Quick Tuning (~30 minutes)
```python
run_hdp_hmm_auto_tuning(
    market_data=df,
    coarse_grid_points=2,   # 2^7 = 128 combinations
    fine_grid_points=2,     # 2^7 = 128 combinations  
    tpe_trials=20,
    timeout=1800  # 30 minutes
)
```
- Trials: ~276
- Best for: Initial exploration

### Standard Tuning (~1-2 hours)
```python
run_hdp_hmm_auto_tuning(
    market_data=df,
    coarse_grid_points=3,   # 3^7 = 2,187 combinations
    fine_grid_points=3,     # 3^7 = 2,187 combinations
    tpe_trials=50,
    timeout=7200  # 2 hours
)
```
- Trials: ~4,424
- Best for: Production optimization

### Thorough Tuning (~4-6 hours)
```python
run_hdp_hmm_auto_tuning(
    market_data=df,
    coarse_grid_points=4,   # 4^7 = 16,384 combinations
    fine_grid_points=4,     # 4^7 = 16,384 combinations
    tpe_trials=100,
    timeout=21600  # 6 hours
)
```
- Trials: ~32,868
- Best for: Research, optimal results

## Advanced Features

### Early Stopping

The TPE optimizer automatically stops if no improvement is seen:

```python
# TPE includes built-in early stopping
# If 20 consecutive trials show no improvement, it may stop early
best_params, best_score, results = run_hdp_hmm_auto_tuning(
    market_data=df,
    tpe_trials=100  # May stop earlier if converged
)
```

### Timeout Handling

The tuner respects timeout across all stages:

```python
# Timeout is distributed across stages
# If timeout during coarse/fine grid, TPE gets at least 1 minute
best_params, best_score, results = run_hdp_hmm_auto_tuning(
    market_data=df,
    timeout=3600  # 1 hour total
)
```

### Result Persistence

Results are automatically saved if `save_results=True`:

```python
best_params, best_score, results = run_hdp_hmm_auto_tuning(
    market_data=df,
    save_results=True  # Saves to artifacts/
)

# Saved artifacts:
# - best_hdp_hmm_params.json: Best parameters
# - tuning_history.json: All trial results  
# - quality_metrics.json: Quality assessment
```

## Best Practices

### 1. Start with Defaults
Always begin with default settings to get baseline results.

### 2. Use Timeout for Long Runs
Set appropriate timeout to prevent overnight runs:
```python
timeout=7200  # 2 hours max
```

### 3. Monitor Progress
Check intermediate results during tuning:
```python
# Tuner prints progress automatically
# Watch for:
# - "✨ New best score" messages
# - Score improvements between stages
# - Convergence indicators
```

### 4. Validate on Hold-Out Data
Always validate tuned parameters on unseen data:
```python
# Split data
train_data = df.iloc[:int(len(df)*0.8)]
test_data = df.iloc[int(len(df)*0.8):]

# Tune on training data
best_params, _, _ = run_hdp_hmm_auto_tuning(
    market_data=train_data,
    symbol="ETHUSDT"
)

# Validate on test data
test_results = run_hdp_hmm_clustering(
    market_data=test_data,
    symbol="ETHUSDT",
    **best_params
)

print(f"Train score: {best_score:.4f}")
print(f"Test score: {test_results['quality_metrics']['composite_score']:.4f}")
```

### 5. Iterate Based on Results
Use tuning insights to refine search space:
```python
# First pass
first_params, first_score, _ = run_hdp_hmm_auto_tuning(market_data=df)

# If alpha optimal at edge (e.g., 5.0), expand range
custom_space = HDPHMMSearchSpace(
    alpha_min=4.0,
    alpha_max=8.0  # Expand upper bound
)

# Second pass
final_params, final_score, _ = run_hdp_hmm_auto_tuning(
    market_data=df,
    search_space=custom_space
)
```

## Troubleshooting

### Issue: Tuning is too slow
**Solutions**:
1. Reduce grid points: `coarse_grid_points=2, fine_grid_points=2`
2. Reduce TPE trials: `tpe_trials=20`
3. Narrow search space
4. Set strict timeout: `timeout=1800`

### Issue: No improvement in later stages
**Cause**: Already found optimal parameters in early stage
**Action**: This is actually good! Early convergence means efficient tuning.

### Issue: Best parameters at search space boundaries
**Cause**: Optimal value outside search range
**Action**: Expand search space in that direction

### Issue: Highly variable scores
**Cause**: Insufficient data or noisy market
**Solutions**:
1. Increase `min_features` for more robust signal
2. Increase `n_iterations` for better convergence
3. Use more data if available

## Summary

- **Three stages**: Coarse grid → Fine grid → TPE optimization
- **Objective**: Maximize composite_score from cluster quality
- **Parameters tuned**: alpha, kappa, gamma, iterations, features, PCA
- **Default time**: 1-2 hours with standard settings
- **Customizable**: Search spaces, timeout, trial counts
- **Results**: Best parameters + complete tuning history
- **Best practice**: Start with defaults, iterate based on results

## Related Documentation

- `HDP_HMM_USAGE_GUIDE.md`: Basic usage and parameter reference
- `HDP_HMM_FEATURE_SELECTION_EXPLAINED.md`: Feature selection details
- `CLUSTER_QUALITY_ASSESSOR_GUIDE.md`: Quality metrics explanation
- `clustering_optimization_goals.py`: Optimization goals and constraints

## Example Workflow

Complete workflow from data to optimized clustering:

```python
import pandas as pd
from src.training.steps.market_analysis.hdp_hmm_clustering import (
    run_hdp_hmm_auto_tuning,
    run_hdp_hmm_clustering
)

# 1. Load data
df = pd.read_csv("ETHUSDT_1h.csv", index_col=0, parse_dates=True)

# 2. Run auto-tuning
print("Starting auto-tuning...")
best_params, best_score, tuning_result = run_hdp_hmm_auto_tuning(
    market_data=df,
    symbol="ETHUSDT",
    exchange="binance",
    timeframe="1h",
    timeout=7200  # 2 hours
)

# 3. Review results
print(f"\nBest composite score: {best_score:.4f}")
print(f"Best parameters:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

# 4. Run final clustering with best parameters
print("\nRunning final clustering with optimized parameters...")
final_results = run_hdp_hmm_clustering(
    market_data=df,
    symbol="ETHUSDT",
    exchange="binance",
    timeframe="1h",
    **best_params,
    save_results=True
)

# 5. Analyze regimes
print(f"\nDiscovered {final_results['n_clusters']} regimes")
print(f"Quality metrics:")
for metric, value in final_results['quality_metrics'].items():
    if isinstance(value, (int, float)):
        print(f"  {metric}: {value:.4f}")

print("\nDone! Results saved to artifacts/")
```
