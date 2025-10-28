# MS-DR Clustering Standalone Usage Guide

## Overview

This guide explains how to use the MS-DR (Markov-Switching Dynamic Regression) clustering as a standalone function with integrated:
- **Cluster Quality Assessor** (`src/training/steps/market_analysis/clusters/cluster_quality_assessor.py`)
- **Clustering Optimization Goals** (`src/training/steps/market_analysis/clusters/clustering_optimization_goals.py`)
- **Artifact Manager** (`src/training/steps/market_analysis/components/artifact_manager.py`)

## Integration Summary

### What Was Integrated

1. **Cluster Quality Assessor**: Provides comprehensive quality metrics including:
   - Silhouette scores (global and per-cluster)
   - Davies-Bouldin Index (DBI)
   - Calinski-Harabasz Index (CH)
   - Within/Between regime coefficient of variation
   - Temporal smoothness
   - Regime persistence
   - Balance metrics
   - Composite quality score

2. **Clustering Optimization Goals**: Defines optimization targets and constraints:
   - Primary goals: CV score, Silhouette, DBI
   - Constraint goals: Balance, Temporal smoothness
   - Composite score calculation
   - Constraint validation
   - Quality report formatting

3. **Artifact Manager**: Handles data loading:
   - Loads market data from artifacts
   - Supports multiple data formats (OHLCV, processed_data, klines)
   - Automatic discovery of latest session
   - Symbol/Exchange/Timeframe based organization

## Standalone Function Usage

### Method 1: With Artifact Manager (Recommended)

Use this method when you have market data stored in artifacts from a previous data collection step.

```python
from src.feature_generation.integration.enhanced_ms_dr_clustering_integration import (
    perform_ms_dr_clustering_with_artifact_manager
)

# Call with artifact manager to automatically load data
result = perform_ms_dr_clustering_with_artifact_manager(
    symbol="ETHUSDT",               # Trading symbol (default: ETHUSDT)
    exchange="binance",             # Exchange name (default: binance)
    timeframe="1h",                 # Timeframe (default: 1h)
    artifact_base_dir="artifacts",  # Base directory for artifacts
    min_features=50,                # Minimum number of features
    max_features=100,               # Maximum number of features
    auto_select_regimes=True,       # Auto-select optimal number of regimes
    min_regimes=2,                  # Minimum regimes to consider
    max_regimes=10                  # Maximum regimes to consider
)

# Access results
regime_labels = result['cluster_labels']
quality_score = result['quality_metrics']['quality_score']
n_regimes = result['n_clusters']
transition_matrix = result['transition_matrix']

print(f"Found {n_regimes} regimes with quality score: {quality_score:.3f}")
```

#### Function Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `symbol` | str | "ETHUSDT" | Trading symbol |
| `exchange` | str | "binance" | Exchange name |
| `timeframe` | str | "1h" | Timeframe (e.g., "30m", "1h") |
| `artifact_base_dir` | str | "artifacts" | Base directory for artifacts |
| `min_features` | int | 50 | Minimum number of features |
| `max_features` | int | 100 | Maximum number of features |
| `n_regimes` | int | 5 | Number of regimes (if not auto-selecting) |
| `model_type` | str | "autoregression" | Model type ('autoregression' or 'regression') |
| `auto_select_regimes` | bool | True | Auto-select optimal number of regimes |
| `min_regimes` | int | 2 | Minimum regimes to consider (if auto-selecting) |
| `max_regimes` | int | 10 | Maximum regimes to consider (if auto-selecting) |

### Method 2: With Manual Data Provision (Using BaseClass)

Use this method when you want to load data through a BaseStep class.

```python
from src.training.steps.base_step import BaseStep
from src.feature_generation.integration.enhanced_ms_dr_clustering_integration import (
    perform_enhanced_ms_dr_clustering
)

# Load data through BaseClass (inherits from BaseStep)
class DataLoader(BaseStep):
    def execute(self):
        # Your BaseStep implementation
        # Data is loaded via artifact_manager
        market_data = self.artifact_manager.get_artifact(
            artifact_name="market_data",
            artifact_type="data"
        )
        return market_data

# Initialize and load
loader = DataLoader(config={...})
market_data = loader.execute()

# Perform clustering
result = perform_enhanced_ms_dr_clustering(
    data=market_data,           # Your DataFrame with OHLCV data
    min_features=50,
    max_features=100,
    auto_select_regimes=True,
    min_regimes=2,
    max_regimes=10
)

# Access results
print(f"Found {result['n_clusters']} regimes")
```

### Method 3: Using the Module Import

You can also import from the module directly:

```python
from src.training.steps.market_analysis.ms_dr_clustering import (
    perform_ms_dr_clustering_with_artifact_manager,
    perform_enhanced_ms_dr_clustering
)

# Then use as shown in Method 1 or 2
```

## Return Value Structure

The function returns a dictionary with the following structure:

```python
{
    # Clustering results
    'cluster_labels': np.ndarray,           # Regime assignment for each time point
    'cluster_probabilities': np.ndarray,    # Probability distribution over regimes
    'n_clusters': int,                      # Number of discovered regimes
    'transition_matrix': np.ndarray,        # Regime transition probabilities
    
    # Model artifacts
    'regime_params': dict,                  # Parameters for each regime
    'regime_variances': np.ndarray,         # Variance in each regime
    'regime_durations': np.ndarray,         # Average duration of each regime
    
    # Quality metrics (from cluster_quality_assessor)
    'quality_metrics': {
        'silhouette_score': float,          # Global silhouette score (-1 to 1)
        'calinski_harabasz_score': float,   # CH index (higher is better)
        'davies_bouldin_score': float,      # DBI (lower is better)
        'within_regime_cv': float,          # Within-regime coefficient of variation
        'between_regime_cv': float,         # Between-regime coefficient of variation
        'balance_score': float,             # Cluster size balance (0 to 1)
        'temporal_smoothness': float,       # Temporal stability (0 to 1)
        'quality_score': float,             # Composite quality score (0 to 1)
        
        # Optimization metrics (from clustering_optimization_goals)
        'optimization_composite_score': float,      # Weighted composite score
        'meets_optimization_constraints': bool,     # Whether constraints are met
        'constraint_checks': dict,                  # Individual constraint results
    },
    
    # Feature information
    'feature_names': list,                  # Names of features used
    'feature_matrix': np.ndarray,           # Feature matrix
    
    # Metadata
    'metadata': dict,                       # Additional metadata
    'clusterer': MSDRClusterer,            # The clusterer object
    'ms_result': MSDRResult,               # Full MS-DR result object
    
    # Artifact manager info (only when using Method 1)
    'artifact_manager': {
        'symbol': str,
        'exchange': str,
        'timeframe': str,
        'artifact_base_dir': str,
        'data_loaded_from': str
    }
}
```

## Quality Metrics Interpretation

### Primary Metrics

1. **Silhouette Score** (0.2-1.0 is good, >0.3 is target)
   - Measures how similar each point is to its own cluster vs other clusters
   - Range: -1 (worst) to 1 (best)

2. **Calinski-Harabasz Score** (>50 is acceptable, >100 is good)
   - Higher values indicate better-defined clusters
   - Based on ratio of between-cluster to within-cluster variance

3. **Davies-Bouldin Score** (<2.0 is acceptable, <1.5 is good)
   - Lower values indicate better clustering
   - Measures average similarity between clusters

### Secondary Metrics

4. **Balance Score** (>0.5 is target, >0.7 is excellent)
   - Ensures clusters are reasonably balanced in size
   - Range: 0 (very imbalanced) to 1 (perfectly balanced)

5. **Temporal Smoothness** (>0.85 is target, >0.95 is excellent)
   - Measures stability of regime assignments over time
   - Higher values mean fewer regime switches

6. **Composite Quality Score** (0-1, higher is better)
   - Weighted combination of all metrics
   - Single number to assess overall quality

### Optimization Constraints

The system checks if the clustering meets predefined optimization goals:

```python
# Check if constraints are met
constraints_met = result['quality_metrics']['meets_optimization_constraints']

if constraints_met:
    print("✅ Clustering meets all optimization constraints")
else:
    print("❌ Some constraints not met")
    print(result['quality_metrics']['constraint_checks'])
```

## Auto-Tuner for Hyperparameter Optimization

The MS-DR clustering now includes an automatic hyperparameter tuner that optimizes the composite quality score using a staged optimization strategy:

**Stage 1**: Coarse Grid Search (broad exploration)  
**Stage 2**: Fine Grid Search around best results (local refinement)  
**Stage 3**: TPE (Tree-structured Parzen Estimator) optimization (final optimization)

### Using the Auto-Tuner

```python
from src.training.steps.market_analysis.ms_dr_clustering import (
    MSDRAutoTuner,
    auto_tune_ms_dr_clustering
)
from src.feature_generation.integration.enhanced_ms_dr_clustering_integration import (
    perform_ms_dr_clustering_with_artifact_manager
)

# Method 1: Using convenience function
result = auto_tune_ms_dr_clustering(
    data=market_data,
    n_trials=100,
    timeout_minutes=60.0,
    enable_staged_optimization=True
)

# Get best parameters
best_params = result['best_params']
best_score = result['best_score']

print(f"Best Score: {best_score:.4f}")
print(f"Best Parameters: {best_params}")

# Method 2: Using MSDRAutoTuner class
tuner = MSDRAutoTuner()

tuning_result = tuner.auto_tune(
    data=market_data,
    n_trials=100,
    timeout_minutes=60.0
)

# Access results
best_params = tuning_result['best_params']
best_score = tuning_result['best_score']
trial_history = tuning_result['trial_history']
optimization_summary = tuning_result['optimization_summary']

print(f"""
Auto-Tuning Results:
===================
Best Score: {best_score:.4f}
Best Parameters: {best_params}
Total Trials: {len(trial_history)}
Improvement: {optimization_summary['improvement']:.4f}
""")
```

### Auto-Tuner Search Space

The auto-tuner optimizes the following parameters:

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `n_regimes` | int | 3-12 | Number of market regimes |
| `order` | int | 1-5 | Autoregression order |
| `switching_variance` | categorical | [True, False] | Allow variance switching |
| `model_type` | categorical | ['autoregression', 'regression'] | Model type |
| `pca_components` | int | 5-20 | Number of PCA components |
| `pca_variance_threshold` | float | 0.85-0.99 | PCA variance threshold |

### Auto-Tuner Configuration

You can customize the tuning process:

```python
from src.training.steps.market_analysis.ms_dr_clustering import (
    MSDRAutoTuner,
    MSDRTuningConfig
)

# Create custom tuning configuration
tuning_config = MSDRTuningConfig(
    n_trials=150,
    coarse_grid_trials=40,
    fine_grid_trials=40,
    tpe_trials=70,
    coarse_grid_points=4,
    fine_grid_points=6,
    early_stopping_patience=15,
    timeout_minutes=90.0
)

# Initialize tuner with custom config
tuner = MSDRAutoTuner(tuning_config=tuning_config)

# Run auto-tuning
result = tuner.auto_tune(data=market_data)
```

## Complete Example

```python
from src.feature_generation.integration.enhanced_ms_dr_clustering_integration import (
    perform_ms_dr_clustering_with_artifact_manager
)
from src.training.steps.market_analysis.ms_dr_clustering import (
    auto_tune_ms_dr_clustering
)

# Step 1: Auto-tune hyperparameters (optional but recommended)
print("🎯 Step 1: Auto-tuning hyperparameters...")
tuning_result = auto_tune_ms_dr_clustering(
    data=market_data,
    n_trials=100,
    timeout_minutes=60.0
)

best_params = tuning_result['best_params']
print(f"Best parameters found: {best_params}")

# Step 2: Perform clustering with artifact manager
print("🚀 Step 2: Performing clustering with optimized parameters...")
result = perform_ms_dr_clustering_with_artifact_manager(
    symbol="ETHUSDT",
    exchange="binance",
    timeframe="1h",
    artifact_base_dir="artifacts",
    min_features=50,
    max_features=100,
    auto_select_regimes=True,
    min_regimes=2,
    max_regimes=10,
    model_type='autoregression',
    switching_variance=True
)

# Extract key results
regime_labels = result['cluster_labels']
n_regimes = result['n_clusters']
quality_score = result['quality_metrics']['quality_score']
silhouette = result['quality_metrics']['silhouette_score']
constraints_met = result['quality_metrics']['meets_optimization_constraints']

# Print summary
print(f"""
MS-DR Clustering Results:
========================
Regimes Found: {n_regimes}
Quality Score: {quality_score:.3f}
Silhouette Score: {silhouette:.3f}
Constraints Met: {constraints_met}

Transition Matrix:
{result['transition_matrix']}
""")

# Analyze regime characteristics
for i in range(n_regimes):
    regime_mask = regime_labels == i
    regime_size = regime_mask.sum()
    regime_duration = result['regime_durations'][i]
    
    print(f"""
Regime {i}:
  - Size: {regime_size} samples ({100*regime_size/len(regime_labels):.1f}%)
  - Avg Duration: {regime_duration:.1f} bars
  - Variance: {result['regime_variances'][i]:.4f}
""")

# Check if quality is acceptable
if quality_score > 0.5 and constraints_met:
    print("✅ High-quality clustering achieved!")
else:
    print("⚠️ Consider adjusting parameters or collecting more data")
```

## Artifact Manager Integration Details

### Data Loading Process

The artifact manager follows this process:

1. **Initialize**: Create artifact manager for symbol/exchange/timeframe
2. **Search**: Look for latest session in `artifacts/` directory
3. **Load**: Try to load market data from known artifact names:
   - `ohlcv` (OHLCV data)
   - `processed_data` (Processed market data)
   - `klines` (Raw klines data)
4. **Validate**: Ensure data is a valid DataFrame
5. **Return**: Pass data to clustering pipeline

### Expected Artifact Structure

```
artifacts/
└── BTCUSDT_binance_30m_20231015_143022/
    ├── market_data_ohlcv_20231015_143022.parquet
    ├── market_data_processed_data_20231015_143022.parquet
    └── market_data_klines_20231015_143022.parquet
```

### Custom Artifact Loading

If you need to customize data loading:

```python
from src.training.steps.market_analysis.components.artifact_manager import ArtifactManager

# Initialize artifact manager
artifact_manager = ArtifactManager(
    base_dir="artifacts",
    symbol="BTCUSDT",
    exchange="binance",
    timeframe="30m"
)

# Load specific artifacts
data = artifact_manager.load_artifacts_from_latest_session(
    component_name="market_data",
    artifact_names=["ohlcv"]
)

# Use with clustering
from src.feature_generation.integration.enhanced_ms_dr_clustering_integration import (
    perform_enhanced_ms_dr_clustering
)

result = perform_enhanced_ms_dr_clustering(
    data=data['ohlcv'],
    min_features=50,
    max_features=100,
    auto_select_regimes=True
)
```

## Troubleshooting

### Issue: "No market data found in artifacts"

**Solution**: Ensure you have run data collection first:

```python
# Run data collection step first
from src.training.steps.data_collection import collect_market_data

collect_market_data(
    symbol="BTCUSDT",
    exchange="binance",
    timeframe="30m"
)

# Then run clustering
result = perform_ms_dr_clustering_with_artifact_manager(
    symbol="BTCUSDT",
    exchange="binance",
    timeframe="30m"
)
```

**Alternative**: Provide data manually using Method 2

### Issue: Poor quality scores

**Solutions**:
1. Adjust feature range: `min_features=30, max_features=80`
2. Change regime range: `min_regimes=3, max_regimes=8`
3. Try different model type: `model_type='regression'`
4. Collect more historical data

### Issue: Too many/few regimes

**Solution**: Adjust regime selection parameters:

```python
result = perform_ms_dr_clustering_with_artifact_manager(
    symbol="BTCUSDT",
    exchange="binance",
    timeframe="30m",
    auto_select_regimes=True,
    min_regimes=4,      # Increase minimum
    max_regimes=8,      # Decrease maximum
)
```

## Related Files

- Main integration: `src/feature_generation/integration/enhanced_ms_dr_clustering_integration.py`
- MS-DR clusterer: `src/training/steps/market_analysis/ms_dr_clustering/ms_dr_clusterer.py`
- **Auto-tuner**: `src/training/steps/market_analysis/ms_dr_clustering/ms_dr_auto_tuner.py`
- Quality assessor: `src/training/steps/market_analysis/clusters/cluster_quality_assessor.py`
- Optimization goals: `src/training/steps/market_analysis/clusters/clustering_optimization_goals.py`
- Artifact manager: `src/training/steps/market_analysis/components/artifact_manager.py`
- **Regime categorization**: `src/feature_generation/categories/regime_feature_categorization.py`
- **Regime integration**: `src/feature_generation/categories/regime_feature_integration.py`
- Module init: `src/training/steps/market_analysis/ms_dr_clustering/__init__.py`

## Summary

The MS-DR clustering is now fully integrated with:

✅ **Cluster Quality Assessor**: Comprehensive quality metrics  
✅ **Clustering Optimization Goals**: Standardized targets and constraints  
✅ **Artifact Manager**: Automatic market data loading  
✅ **Auto-Tuner**: Staged hyperparameter optimization (Coarse Grid → Fine Grid → TPE)  
✅ **Regime Feature Categorization**: Priority-based feature selection for regime clustering  
✅ **Regime Feature Integration**: Dynamic regime detection and adaptive feature generation  

### Quick Start

```python
# Full pipeline with auto-tuning and regime features
from src.training.steps.market_analysis.ms_dr_clustering import (
    perform_ms_dr_clustering_with_artifact_manager,
    auto_tune_ms_dr_clustering
)

# Load data from artifacts (defaults: symbol=ETHUSDT, timeframe=1h)
# Automatically uses regime categorization and integration
result = perform_ms_dr_clustering_with_artifact_manager()

# Or with auto-tuning
tuning_result = auto_tune_ms_dr_clustering(
    data=market_data,
    n_trials=100
)
```

### Regime Feature Integration

The system automatically integrates two powerful regime feature modules:

**Regime Feature Categorization**:
- Categorizes features by use case (REGIME_CLUSTERING)
- Selects priority features optimized for MS-DR
- Validates feature sets to avoid data leakage
- Ensures lookahead-safe and stable features

**Regime Feature Integration**:
- Detects current market regime dynamically
- Generates regime-adaptive features
- Tracks regime transitions and stability
- Creates features for: trending, mean-reverting, volatile, and stable regimes

You can control these features:

```python
from src.feature_generation.integration.enhanced_ms_dr_clustering_integration import (
    EnhancedMSDRClusteringIntegration
)

# Customize regime feature usage
integrator = EnhancedMSDRClusteringIntegration(
    min_features=50,
    max_features=100,
    enable_regime_categorization=True,   # Use regime feature categorization
    enable_regime_integration=True,       # Use regime feature integration
    auto_select_regimes=True
)

result = integrator.cluster_with_ms_dr(market_data)
```

### Default Parameters

- **Symbol**: `ETHUSDT` (changed from BTCUSDT)
- **Exchange**: `binance`
- **Timeframe**: `1h` (changed from 30m)

Use `perform_ms_dr_clustering_with_artifact_manager()` for the complete standalone experience!
