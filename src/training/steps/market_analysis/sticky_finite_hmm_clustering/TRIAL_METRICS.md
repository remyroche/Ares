# Trial Metrics Enhancement for Sticky Finite HMM Auto-Tuning

## Overview

Enhanced the hyperparameter optimization objective function to log comprehensive metrics for each trial, providing full visibility into model performance and parameter effectiveness.

## Enhanced Trial Logging

### Each Trial Now Logs:

#### 1. **Trial Header**
```
================================================================================
🔍 TRIAL: K=5, kappa=15.00, base_alpha=0.350, lr=5.00e-03, pca_components=12
================================================================================
```

#### 2. **Performance Metrics**
```
✅ Trial complete: composite_score=0.756432
composite_score: 0.756432
n_clusters: 5
balance_score: 0.8234
cluster_balance: 12.3%-28.7%
cv_ratio: 2.4567
temporal_smoothness: 0.8901
transition_persistence: 0.8345
K: 5
kappa: 15.00
base_alpha: 0.350
lr: 5.00e-03
pca_components: 12
```

#### 3. **Trial Summary**
```
================================================================================
✅ TRIAL SUCCESSFUL: Score = 0.756432
================================================================================
```

## Metrics Explanation

### Core Quality Metrics

1. **composite_score** (0.0-1.0)
   - Overall quality score combining all metrics
   - Weighted sum of silhouette, balance, CV ratio, temporal smoothness
   - Higher is better
   - Primary optimization objective

2. **n_clusters** (integer)
   - Number of distinct regimes discovered
   - Should equal K for Sticky Finite HMM (K=5 typically)
   - Validates that all K states are being used

### Balance Metrics

3. **balance_score** (0.0-1.0)
   - Measures evenness of regime distribution
   - 1.0 = perfectly balanced (all regimes same size)
   - 0.0 = highly imbalanced (one regime dominates)
   - Prevents degenerate solutions

4. **cluster_balance** (percentage range)
   - Min%-Max% cluster sizes
   - Shows smallest to largest regime proportion
   - Example: "12.3%-28.7%" means smallest regime is 12.3% of data
   - Ideally ranges should not be too extreme (e.g., 15%-25% is good)

### Coefficient of Variation (CV) Metrics

5. **cv_ratio** (>= 1.0)
   - Ratio of between-regime CV to within-regime CV
   - Measures separation quality
   - Higher values = better separated regimes
   - Good values: > 1.5
   - Excellent values: > 2.0

### Temporal Metrics

6. **temporal_smoothness** (0.0-1.0)
   - Measures consistency of regime assignments over time
   - Penalizes flip-flop behavior (rapid regime changes)
   - Higher = more stable regime sequences
   - Good values: > 0.70
   - Excellent values: > 0.85

7. **transition_persistence** (0.0-1.0)
   - Average self-transition probability
   - Derived from learned transition matrix diagonal
   - Higher = regimes persist longer
   - Should align with kappa parameter
   - Expected value: (base_alpha + kappa) / (base_alpha * K + kappa)

### Hyperparameters

8. **K** (integer, fixed)
   - Number of states (regimes)
   - Fixed at 5 for Sticky Finite HMM
   - Not optimized (model architecture parameter)

9. **kappa** (5.0-50.0)
   - Stickiness parameter
   - Controls regime persistence/duration
   - Higher → longer regime durations
   - kappa=10 → ~11 timesteps
   - kappa=30 → ~28 timesteps
   - kappa=50 → ~44 timesteps

10. **base_alpha** (0.1-1.0)
    - Concentration for off-diagonal transitions
    - Controls transition sparsity
    - Lower (0.1) → sparse transitions, infrequent changes
    - Higher (1.0) → uniform transitions, frequent changes

11. **lr** (1e-4 to 1e-1, log scale)
    - Learning rate for SVI optimizer
    - Controls optimization speed/stability
    - Too high → unstable ELBO, poor convergence
    - Too low → very slow convergence
    - Typical good range: 1e-3 to 1e-2

12. **pca_components** (10-20)
    - Number of PCA components for dimensionality reduction
    - Balances information retention vs. noise
    - 10 → faster, may lose some patterns
    - 20 → more information, potentially more noise
    - Default: 15 (good balance)

## Example Trial Output

### Successful Trial
```
================================================================================
🔍 TRIAL: K=5, kappa=15.00, base_alpha=0.350, lr=5.00e-03, pca_components=12
================================================================================
🚀 Starting Sticky Finite HMM Clustering Pipeline
📊 Input validation passed: 1000 samples
...
🔄 Training Sticky Finite HMM with Pyro SVI
   Running SVI for 1000 iterations
      Iteration 0/1000: ELBO = -15234.52
      Iteration 50/1000: ELBO = -12456.78
   ...
✅ SVI training complete: Final ELBO = -8945.23

✅ Trial complete: composite_score=0.756432
composite_score: 0.756432
n_clusters: 5
balance_score: 0.8234
cluster_balance: 12.3%-28.7%
cv_ratio: 2.4567
temporal_smoothness: 0.8901
transition_persistence: 0.8345
K: 5
kappa: 15.00
base_alpha: 0.350
lr: 5.00e-03
pca_components: 12

================================================================================
✅ TRIAL SUCCESSFUL: Score = 0.756432
================================================================================
```

### Failed Trial
```
================================================================================
🔍 TRIAL: K=5, kappa=5.00, base_alpha=0.100, lr=1.00e-01, pca_components=20
================================================================================
🚀 Starting Sticky Finite HMM Clustering Pipeline
...
❌ Trial failed: ELBO diverged - numerical instability detected

status: FAILED
error: ELBO diverged - numerical instability detected
K: 5
kappa: 5.00
base_alpha: 0.100
lr: 1.00e-01
pca_components: 20

================================================================================
```

## Optimization Process

The hierarchical optimization uses these metrics to:

1. **Coarse Grid Search** - Broad parameter space exploration
   - Evaluates 3^5 = 243 combinations (or subset)
   - Identifies promising regions

2. **Fine Grid Search** - Refinement around best coarse results
   - Evaluates 5^5 = 3,125 combinations (or subset)
   - Narrows down to optimal neighborhood

3. **TPE (Bayesian Optimization)** - Final optimization
   - 100+ trials with intelligent sampling
   - Converges to optimal parameters
   - Uses all previous trial data

## Interpreting Results

### High-Quality Trial Characteristics
```
composite_score: > 0.70
balance_score: > 0.75
cv_ratio: > 1.5
temporal_smoothness: > 0.80
transition_persistence: > 0.70
cluster_balance: 15%-30% (not too extreme)
```

### Poor-Quality Trial Characteristics
```
composite_score: < 0.40
balance_score: < 0.50 (imbalanced)
cv_ratio: < 1.2 (poor separation)
temporal_smoothness: < 0.60 (unstable)
transition_persistence: < 0.50 (no persistence)
cluster_balance: 5%-60% (very imbalanced)
```

### Numerical Instability Indicators
```
- Very high lr (> 0.05) + low base_alpha (< 0.2)
- ELBO divergence (increasing instead of decreasing)
- NaN or Inf values in parameters
- All samples assigned to single regime (n_clusters = 1)
```

## Benefits

1. **Full Transparency** - See exactly what each trial produces
2. **Quick Diagnostics** - Identify problematic parameter combinations
3. **Progress Tracking** - Monitor optimization convergence
4. **Result Interpretation** - Understand why certain parameters work
5. **Debugging Support** - Troubleshoot failed trials easily
6. **Parameter Insights** - Learn relationships between hyperparameters and metrics

## Related Files

- `sticky_finite_hmm_auto_tuner.py` - Auto-tuning implementation
- `sticky_finite_hmm_clusterer.py` - Core clustering algorithm
- `cluster_quality_assessor.py` - Quality metrics computation
- `TPRINT_ENHANCEMENTS.md` - General logging enhancements

## Implementation Notes

### Location
- File: `sticky_finite_hmm_auto_tuner.py`
- Function: `sticky_finite_hmm_objective_function()`
- Lines: ~390-510

### Key Code Sections
```python
# Extract comprehensive metrics
quality_metrics = result.get('quality_metrics', {})
quality_assessment = quality_metrics.get('quality_assessment', {})

# Log detailed trial metrics
tprint_structured({
    "composite_score": f"{composite_score:.6f}",
    "n_clusters": n_clusters,
    "balance_score": f"{balance_score:.4f}",
    "cluster_balance": f"{min_cluster_size_pct:.1f}%-{max_cluster_size_pct:.1f}%",
    "cv_ratio": f"{cv_ratio:.4f}",
    "temporal_smoothness": f"{temporal_smoothness:.4f}",
    "transition_persistence": f"{transition_persistence:.4f}",
    "K": K,
    "kappa": f"{kappa:.2f}",
    "base_alpha": f"{base_alpha:.3f}",
    "lr": f"{lr:.2e}",
    "pca_components": pca_components
}, level="INFO")
```

## Date

Created: 2025-11-03
Last Updated: 2025-11-03

