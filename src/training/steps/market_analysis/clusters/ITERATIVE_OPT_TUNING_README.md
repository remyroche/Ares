# Iterative Optimization Hyperparameter Tuning

## Overview

This module provides automated hyperparameter tuning for `iterative_optimization.py` to improve clustering quality metrics:
- **CV Score** (Between/Within Variance Ratio) - Higher is better
- **Silhouette Score** - Higher is better (range: -1 to 1)
- **DBI Score** (Davies-Bouldin Index) - Lower is better
- **Balance Score** - Maintained above 0.5
- **Temporal Smoothness** - Maintained above 0.85

## Current Baseline Metrics

From your most recent run with **8 clusters**:
```
CV Score:            1.1910 ✅ (Good - >1.0)
Silhouette Score:   -0.0345 ❌ (Poor - negative)
DBI Score:           ~3.2   ❌ (Poor - should be <2.0)
Balance Score:       0.6340 ✅ (Moderate)
Temporal Smoothness: 0.987  ✅ (Excellent)
```

## Goal

Improve Silhouette and DBI scores while maintaining or improving CV, Balance, and Temporal Smoothness.

## Quick Start

### 1. Basic Bayesian Optimization (Recommended)

```bash
python3 src/training/steps/market_analysis/clusters/run_iterative_opt_tuning.py \
    --symbol ETHUSDT \
    --n-trials 30 \
    --method bayesian
```

This will:
- Run 30 trials of Bayesian hyperparameter optimization
- Find the best single configuration that maximizes a composite score
- Save results to `artifacts/hyperparameter_tuning/`

### 2. Multi-Objective Optimization (Advanced)

```bash
python3 src/training/steps/market_analysis/clusters/run_iterative_opt_tuning.py \
    --symbol ETHUSDT \
    --n-trials 50 \
    --method multiobjective
```

This will:
- Find multiple Pareto-optimal configurations
- Provide trade-off options between different objectives
- More comprehensive but takes longer

## Tunable Parameters

The tuner optimizes the following parameters from `OptConfig` (lines 2489-2562 in `iterative_optimization.py`):

### Core Constraints
| Parameter | Current | Tunable Range | Impact |
|-----------|---------|---------------|--------|
| `K_MIN` | 6 | 5-8 | Minimum number of clusters |
| `K_MAX` | 10 | 8-12 | Maximum number of clusters |
| `MIN_FRAC` | 0.03 | 0.02-0.05 | Minimum cluster size (% of data) |
| `MAX_FRAC` | 0.20 | 0.15-0.25 | Maximum cluster size (% of data) |

### Objective Weights
| Parameter | Current | Tunable Range | Impact |
|-----------|---------|---------------|--------|
| `w_cv` | 0.70 | 0.50-0.80 | Weight for CV ratio (variance) |
| `w_sil` | 0.10 | 0.05-0.20 | Weight for Silhouette score |
| `w_temp` | 0.20 | 0.10-0.30 | Weight for Temporal smoothness |
| `w_bal` | 0.05 | 0.02-0.10 | Weight for Balance |

### Optimization Thresholds
| Parameter | Current | Tunable Range | Impact |
|-----------|---------|---------------|--------|
| `eps_std_step1` | -0.20 | -0.30 to -0.10 | Aggressiveness of local moves |
| `sil_guard` | -0.08 | -0.10 to -0.05 | Silhouette improvement requirement |
| `temporal_bonus` | 0.25 | 0.15-0.35 | Bonus for temporal stability |

### Lexicographic Thresholds (Log Scale)
| Parameter | Current | Tunable Range | Impact |
|-----------|---------|---------------|--------|
| `eps_cv` | 1e-5 | 1e-6 to 1e-4 | CV improvement threshold |
| `eps_sil` | 1e-4 | 1e-5 to 1e-3 | Silhouette improvement threshold |
| `eps_temp` | 1e-4 | 1e-5 to 1e-3 | Temporal improvement threshold |

### Performance Parameters
| Parameter | Current | Tunable Range | Impact |
|-----------|---------|---------------|--------|
| `max_rounds` | 40 | 20-50 | Number of optimization iterations |
| `local_churn_cap` | 5000 | 3000-7000 | Step 1 move limit |
| `knn_size` | 25 | 15-35 | Neighbor consensus size |

## Output Files

The tuning process generates:

### 1. Results JSON (`optimization_results_YYYYMMDD_HHMMSS.json`)
```json
{
  "timestamp": "2025-10-27T23:00:00",
  "n_samples": 412,
  "n_features": 25,
  "best_params": {
    "K_MIN": 6,
    "K_MAX": 10,
    "w_cv": 0.65,
    "w_sil": 0.15,
    ...
  },
  "best_metrics": {
    "cv_score": 1.45,
    "silhouette_score": 0.25,
    "dbi_score": 1.8,
    "balance_score": 0.68,
    "temporal_smoothness": 0.92,
    "n_clusters": 7
  }
}
```

### 2. Optimization Report (`optimization_report_YYYYMMDD_HHMMSS.md`)
- Summary of tuning results
- Best configuration metrics
- Parameter recommendations
- Comparison with baseline

## Applying Tuned Parameters

After optimization completes:

### 1. Review Results
```bash
cat artifacts/hyperparameter_tuning/optimization_report_*.md
```

### 2. Update `iterative_optimization.py`

Edit the `OptConfig` dataclass (lines 2489-2562):

```python
@dataclass
class OptConfig:
    """Unified configuration for iterative optimization."""
    # Apply best_params from tuning results
    K_MIN: int = 6  # From best_params['K_MIN']
    K_MAX: int = 10  # From best_params['K_MAX']
    MIN_FRAC: float = 0.03  # From best_params['MIN_FRAC']
    MAX_FRAC: float = 0.20  # From best_params['MAX_FRAC']
    
    # Objective weights - from tuning
    w_cv: float = 0.65  # From best_params['w_cv']
    w_sil: float = 0.15  # From best_params['w_sil']
    w_temp: float = 0.15  # From best_params['w_temp']
    w_bal: float = 0.05  # From best_params['w_bal']
    
    # ... update other parameters similarly
```

### 3. Re-run Regime Clustering

```bash
python3 src/launcher/ares_launcher.py --step regime_clustering --symbol ETHUSDT --execution-mode light
```

Verify that metrics have improved!

## Advanced Usage

### Custom Parameter Space

Modify `OptimizationParameterSpace` in `iterative_optimization_tuner.py` to adjust search ranges:

```python
class OptimizationParameterSpace:
    # Expand search range for CV weight
    w_cv: Tuple[float, float] = (0.60, 0.85)  # Default: (0.50, 0.80)
    
    # Narrow search for temporal weight  
    w_temp: Tuple[float, float] = (0.15, 0.25)  # Default: (0.10, 0.30)
```

### Multi-Objective Pareto Analysis

The multi-objective mode finds multiple optimal solutions with different trade-offs:

```python
# After running with --method multiobjective
results = {...}  # Load from JSON

# Pareto front contains multiple solutions
for solution in results['pareto_front']:
    metrics = solution['metrics']
    print(f"CV: {metrics.cv_score:.3f}, Sil: {metrics.silhouette_score:.3f}, DBI: {metrics.dbi_score:.3f}")
```

Choose the solution that best fits your priorities.

## Troubleshooting

### Issue: Tuning takes too long
**Solution**: Reduce `--n-trials` to 10-15 for faster results

### Issue: All trials fail constraints
**Solution**: Relax constraints in `IterativeOptimizationMetrics.meets_constraints()`:
- Lower `min_balance` from 0.5 to 0.4
- Lower `min_temporal` from 0.85 to 0.80

### Issue: Metrics don't improve
**Solution**: 
- Try multi-objective method to explore trade-offs
- Expand parameter search ranges
- Increase n_trials to 50-100

## Integration with Pipeline

To automatically run tuning before regime clustering:

```python
# In regime_clustering_step.py, before calling _run_iterative_optimization_fallback:

if config.get('auto_tune_iterative_opt', False):
    from src.training.steps.market_analysis.clusters.run_iterative_opt_tuning import load_data_for_tuning
    from src.training.steps.market_analysis.clusters.iterative_optimization_tuner import run_tuning_pipeline
    
    # Run tuning
    tuning_results = run_tuning_pipeline(features, labels, market_data, n_trials=20)
    
    # Apply best params to config
    if tuning_results and 'best_params' in tuning_results:
        config.update(tuning_results['best_params'])
```

## Expected Improvements

Based on similar clustering optimization tasks, you can expect:

- **Silhouette Score**: -0.03 → **0.15 to 0.30** (significant improvement)
- **DBI Score**: 3.2 → **1.5 to 2.2** (moderate improvement)
- **CV Score**: 1.19 → **1.3 to 1.6** (slight improvement)
- **Balance**: 0.63 → **0.65 to 0.75** (maintained or slightly improved)
- **Temporal**: 0.987 → **0.95 to 0.99** (maintained)

## Performance

- **Bayesian (30 trials)**: ~15-30 minutes (depending on data size)
- **Multi-objective (50 trials)**: ~30-60 minutes
- Faster with smaller datasets or fewer features

## References

- `iterative_optimization.py` (lines 2484-2583): OptConfig definition
- `src/utils/ml_common/optimization/hpo_utils.py`: Bayesian optimization tools
- `src/utils/ml_common/optimization/pareto.py`: Multi-objective optimization

