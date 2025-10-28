# HDP-HMM Clustering Usage Guide

This guide explains how to use the HDP-HMM clustering module with integrated cluster quality assessment, optimization goals, and artifact manager.

## Overview

The HDP-HMM clustering module now includes:
- **Cluster Quality Assessment** via `cluster_quality_assessor.py`
- **Optimization Goals** via `clustering_optimization_goals.py`
- **Artifact Manager** via `artifact_manager.py` for data loading/saving

## Quick Start

### 1. Auto-Tuning (Recommended)

**Best way to get started** - Let the system find optimal parameters:

```python
import pandas as pd
from src.training.steps.market_analysis.hdp_hmm_clustering import run_hdp_hmm_auto_tuning

# Load market data
df = pd.read_csv("market_data.csv", index_col=0, parse_dates=True)

# Run auto-tuning to find best parameters
best_params, best_score, tuning_result = run_hdp_hmm_auto_tuning(
    market_data=df,
    symbol="ETHUSDT",
    exchange="binance",
    timeframe="1h",
    timeout=3600  # 1 hour
)

print(f"Best composite score: {best_score:.4f}")
print(f"Optimized parameters: {best_params}")

# Use optimized parameters for final clustering
from src.training.steps.market_analysis.hdp_hmm_clustering import run_hdp_hmm_clustering

final_results = run_hdp_hmm_clustering(
    market_data=df,
    symbol="ETHUSDT",
    **best_params
)
```

### 2. Basic Usage with DataFrame

```python
import pandas as pd
from src.training.steps.market_analysis.hdp_hmm_clustering import run_hdp_hmm_clustering

# Load your market data (OHLCV format)
df = pd.read_csv("market_data.csv", index_col=0, parse_dates=True)

# Run HDP-HMM clustering
results = run_hdp_hmm_clustering(
    market_data=df,
    symbol="BTCUSDT",
    exchange="binance",
    timeframe="30m",
    alpha=3.0,           # Higher = more regimes
    kappa=50.0,          # Higher = longer regime durations
    n_iterations=100,    # Gibbs sampling iterations
    save_results=True    # Save to artifacts
)

# Access results
print(f"Discovered {results['n_clusters']} regimes")
print(f"Quality score: {results['quality_metrics']['composite_score']:.3f}")
print(f"Meets constraints: {results['quality_metrics']['meets_constraints']}")
```

### 3. Loading Data from Artifacts

```python
from src.training.steps.market_analysis.hdp_hmm_clustering import (
    run_hdp_hmm_clustering_from_artifacts
)

# Load data from previously saved artifact and run clustering
results = run_hdp_hmm_clustering_from_artifacts(
    artifact_name="market_data",
    step_name="data_collection",
    symbol="BTCUSDT",
    exchange="binance",
    timeframe="30m",
    alpha=3.0,
    kappa=50.0,
    n_iterations=100
)
```

### 4. Just Loading Market Data

```python
from src.training.steps.market_analysis.hdp_hmm_clustering import (
    load_market_data_for_clustering
)

# Load market data from artifacts
df = load_market_data_for_clustering(
    symbol="BTCUSDT",
    exchange="binance",
    timeframe="30m",
    artifact_name="market_data",
    step_name="data_collection"
)

print(df.head())
```

## Function Reference

### `run_hdp_hmm_auto_tuning()`

**RECOMMENDED** - Automatically find optimal hyperparameters using multi-stage optimization.

**Multi-Stage Approach**:
1. **Coarse Grid Search** - Broad exploration (sparse grid)
2. **Fine Grid Search** - Refinement around best results
3. **TPE Optimization** - Bayesian optimization for final tuning

**Parameters**:
- `market_data` (pd.DataFrame): Market data with OHLCV columns
- `symbol` (str): Trading symbol (default: "ETHUSDT")
- `exchange` (str): Exchange name (default: "binance")
- `timeframe` (str): Timeframe (default: "1h")
- `search_space` (HDPHMMSearchSpace): Custom search space (optional)
- `coarse_grid_points` (int): Points per parameter in coarse grid (default: 3)
  - 3 points = 3^7 = 2,187 combinations
  - 4 points = 4^7 = 16,384 combinations
- `fine_grid_points` (int): Points per parameter in fine grid (default: 3)
- `tpe_trials` (int): Number of TPE trials (default: 50)
- `timeout` (float): Total timeout in seconds (optional)
- `save_results` (bool): Save results to artifacts (default: True)

**Returns**:
Tuple of `(best_params, best_score, tuning_result)`:
- `best_params`: Dictionary of optimized hyperparameters
- `best_score`: Best composite score achieved
- `tuning_result`: TuningResult object with complete history

**Example**:
```python
# Standard auto-tuning (1-2 hours)
best_params, best_score, results = run_hdp_hmm_auto_tuning(
    market_data=df,
    symbol="ETHUSDT",
    timeout=7200  # 2 hours
)

# Quick tuning (30 minutes)
best_params, best_score, results = run_hdp_hmm_auto_tuning(
    market_data=df,
    symbol="ETHUSDT",
    coarse_grid_points=2,  # Faster
    fine_grid_points=2,
    tpe_trials=20,
    timeout=1800
)

# Thorough tuning (4-6 hours)
best_params, best_score, results = run_hdp_hmm_auto_tuning(
    market_data=df,
    symbol="ETHUSDT",
    coarse_grid_points=4,  # More exploration
    tpe_trials=100,
    timeout=21600
)
```

**See Also**: `HDP_HMM_AUTO_TUNING_GUIDE.md` for detailed auto-tuning documentation

### `run_hdp_hmm_clustering()`

Main function to run HDP-HMM clustering with comprehensive quality assessment.

**Parameters:**
- `market_data` (pd.DataFrame): Market data with OHLCV columns
- `symbol` (str): Trading symbol (default: "BTCUSDT")
- `exchange` (str): Exchange name (default: "binance")
- `timeframe` (str): Timeframe (default: "30m")
- `min_features` (int): Minimum number of features (default: 50)
- `max_features` (int): Maximum number of features (default: 100)
- `alpha` (float): HDP concentration parameter (default: 3.0)
  - Higher values → more regimes discovered
  - Range: 1.0 - 10.0
- `kappa` (float): Stickiness parameter (default: 50.0)
  - Higher values → longer regime durations
  - Range: 10.0 - 100.0
- `gamma` (float): Base distribution hyperparameter (default: 3.0)
- `n_iterations` (int): Gibbs sampling iterations (default: 100)
- `max_states` (int): Maximum number of states (default: 20)
- `enable_pca` (bool): Enable PCA reduction (default: True)
- `pca_components` (int): Number of PCA components (default: 10)
- `save_results` (bool): Save results to artifacts (default: True)
- `output_dir` (str): Output directory (default: "artifacts")

**Returns:**
Dictionary containing:
- `cluster_labels`: Regime labels for each time step
- `cluster_probabilities`: Posterior probabilities
- `n_clusters`: Number of discovered regimes
- `transition_matrix`: State transition matrix
- `emission_params`: Emission distribution parameters
- `state_durations`: Average duration for each state
- `quality_metrics`: Comprehensive quality assessment including:
  - `silhouette_score`: Cluster cohesion score
  - `davies_bouldin_score`: Cluster separation score
  - `calinski_harabasz_score`: Variance ratio score
  - `balance_score`: Cluster balance score
  - `temporal_smoothness`: Temporal stability score
  - `composite_score`: Overall quality score
  - `meets_constraints`: Whether optimization constraints are met
- `feature_names`: Names of features used
- `feature_matrix`: Feature matrix used for clustering
- `metadata`: Additional metadata

### `run_hdp_hmm_clustering_from_artifacts()`

Run clustering on data loaded from artifacts.

**Parameters:**
- `artifact_name` (str): Name of the artifact to load
- `step_name` (str): Name of the step that saved the artifact
- `symbol` (str): Trading symbol
- `exchange` (str): Exchange name
- `timeframe` (str): Timeframe
- `artifact_dir` (str): Directory containing artifacts (default: "artifacts")
- `**clustering_kwargs`: Additional arguments passed to `run_hdp_hmm_clustering()`

**Returns:**
Same as `run_hdp_hmm_clustering()`

### `load_market_data_for_clustering()`

Helper function to load market data from artifacts.

**Parameters:**
- `symbol` (str): Trading symbol
- `exchange` (str): Exchange name
- `timeframe` (str): Timeframe
- `artifact_name` (str): Name of the artifact (default: "market_data")
- `step_name` (str): Step that saved the data (default: "data_collection")
- `artifact_dir` (str): Artifacts directory (default: "artifacts")

**Returns:**
DataFrame with market data

## Understanding the Results

### Quality Metrics

The clustering results include comprehensive quality metrics:

1. **Silhouette Score** (range: -1 to 1, higher is better)
   - Measures how similar each point is to its own cluster vs other clusters
   - Target: ≥ 0.2 (good), ≥ 0.5 (excellent)

2. **Davies-Bouldin Score** (range: 0 to ∞, lower is better)
   - Measures average similarity between clusters
   - Target: ≤ 2.0 (good), ≤ 1.0 (excellent)

3. **Calinski-Harabasz Score** (range: 0 to ∞, higher is better)
   - Ratio of between-cluster to within-cluster variance
   - Target: ≥ 50 (good), ≥ 100 (excellent)

4. **Balance Score** (range: 0 to 1, higher is better)
   - Measures how evenly samples are distributed across clusters
   - Target: ≥ 0.5 (good), ≥ 0.7 (excellent)

5. **Temporal Smoothness** (range: 0 to 1, higher is better)
   - Measures stability of regime assignments over time
   - Target: ≥ 0.85 (good), ≥ 0.95 (excellent)

6. **Composite Score** (range: 0 to 1, higher is better)
   - Weighted combination of all metrics
   - Automatically calculated using optimization goals

### Optimization Constraints

The module checks if results meet optimization constraints:

```python
if results['quality_metrics']['meets_constraints']:
    print("✅ Clustering meets all quality constraints!")
else:
    print("❌ Clustering does not meet some constraints")
    print(results['quality_metrics']['constraint_checks'])
```

## Advanced Usage

### Custom Optimization Goals

```python
from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
    ClusteringOptimizationGoals,
    OptimizationTargets,
    GoalConfig,
    OptimizationObjective
)

# Create custom optimization goals
custom_goals = ClusteringOptimizationGoals()
custom_goals.silhouette_score.weight = 0.4  # Increase silhouette importance
custom_goals.cv_score.weight = 0.2          # Decrease CV importance

# Create custom targets
custom_targets = OptimizationTargets()
custom_targets.min_silhouette_score = 0.3   # Stricter requirement
custom_targets.min_clusters = 4             # Fewer clusters
custom_targets.max_clusters = 8

# Use with clusterer
from src.training.steps.market_analysis.hdp_hmm_clustering.hdp_hmm_clusterer import (
    HDPHMMClusterer,
    HDPHMMConfig
)

config = HDPHMMConfig(
    alpha=3.0,
    kappa=50.0,
    n_iterations=100
)

clusterer = HDPHMMClusterer(
    config=config,
    artifact_manager=artifact_manager,
    optimization_goals=custom_goals,
    optimization_targets=custom_targets
)
```

### Using Artifact Manager Directly

```python
from src.utils.artifact_manager import ArtifactManager

# Initialize artifact manager
config = {
    "paths": {
        "data_dir": "artifacts",
        "cache_dir": "data_cache",
        "reports_dir": "reports"
    }
}
artifact_manager = ArtifactManager(config)
artifact_manager.set_context(
    step_name="hdp_hmm_clustering",
    symbol="BTCUSDT",
    exchange="binance",
    information="regime_discovery"
)

# Save clustering results
artifact_manager.save(
    data=results['cluster_labels'],
    artifact_name="cluster_labels",
    artifact_type="data"
)

# Load previous results
previous_labels = artifact_manager.get_artifact(
    artifact_name="cluster_labels",
    artifact_type="data"
)
```

## Parameter Tuning Guide

### Alpha (Concentration Parameter)
- **Low (1.0-2.0)**: Fewer regimes, more conservative
- **Medium (2.0-4.0)**: Balanced regime discovery (recommended)
- **High (4.0-10.0)**: More regimes, may overfit

### Kappa (Stickiness Parameter)
- **Low (10-30)**: Short regime durations, frequent switches
- **Medium (30-70)**: Balanced persistence (recommended)
- **High (70-100)**: Long regime durations, infrequent switches

### Iterations
- **Quick (50-100)**: Faster but may not converge
- **Standard (100-200)**: Good balance (recommended)
- **Thorough (200-500)**: Slower but better convergence

## File Locations

- **Main clusterer**: `src/training/steps/market_analysis/hdp_hmm_clustering/hdp_hmm_clusterer.py`
- **Standalone runner**: `src/training/steps/market_analysis/hdp_hmm_clustering/standalone_runner.py`
- **Quality assessor**: `src/training/steps/market_analysis/clusters/cluster_quality_assessor.py`
- **Optimization goals**: `src/training/steps/market_analysis/clusters/clustering_optimization_goals.py`
- **Artifact manager**: `src/utils/artifact_manager.py`

## Common Issues

### Issue: "HMM libraries not available"
**Solution**: Install required HMM library:
```bash
# Option 1: ssm (recommended, easier to install)
pip install ssm-jax

# Option 2: pyhsmm (more features, harder to install)
pip install git+https://github.com/mattjj/pyhsmm.git
```

### Issue: Poor clustering quality
**Solution**: 
1. Check data quality (NaN values, outliers)
2. Adjust alpha/kappa parameters
3. Increase number of iterations
4. Ensure sufficient data (minimum 500 samples recommended)

### Issue: Too many/few regimes
**Solution**:
- Too many: Decrease `alpha`, increase `kappa`
- Too few: Increase `alpha`, decrease `kappa`

## Examples

See `minimal_test_hdp_hmm.py` in the root directory for a complete working example.

## Auto-Tuning vs Manual Parameter Selection

### When to Use Auto-Tuning

✅ **Recommended for**:
- First-time users (let system find optimal params)
- Production systems (best performance)
- New markets/timeframes (unknown optimal params)
- Research (thorough optimization)
- When you have time (1-4 hours)

### When to Use Manual Parameters

✅ **Recommended for**:
- Quick exploration (< 5 minutes)
- Known good parameters (from previous tuning)
- Testing specific hypotheses
- Limited computational resources
- Real-time/streaming scenarios

### Hybrid Approach (Best Practice)

1. **Initial**: Run auto-tuning once to find optimal parameters
2. **Store**: Save best parameters for your market/timeframe
3. **Reuse**: Use saved parameters for subsequent runs
4. **Re-tune**: Periodically (monthly/quarterly) to adapt to regime changes

```python
# One-time: Find optimal parameters
best_params, best_score, _ = run_hdp_hmm_auto_tuning(
    market_data=historical_data,
    symbol="ETHUSDT",
    timeout=7200
)

# Save for reuse
import json
with open("ethusdt_1h_params.json", "w") as f:
    json.dump(best_params, f)

# Daily usage: Load and use saved parameters
with open("ethusdt_1h_params.json", "r") as f:
    saved_params = json.load(f)

results = run_hdp_hmm_clustering(
    market_data=new_data,
    symbol="ETHUSDT",
    **saved_params
)
```

## Support

For issues or questions, please refer to:
- **Auto-Tuning**: `HDP_HMM_AUTO_TUNING_GUIDE.md` - Complete auto-tuning guide
- **Feature Selection**: `HDP_HMM_FEATURE_SELECTION_EXPLAINED.md` - min/max_features explained
- **Module documentation**: Docstrings in each file
- **Quality assessment**: `CLUSTER_QUALITY_ASSESSOR_GUIDE.md`
- **Optimization goals**: `clustering_optimization_goals.py` docstrings
