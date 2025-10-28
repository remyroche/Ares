# SR (Support/Resistance) Detection Workflow

This document describes the complete SR detection workflow, which consists of three integrated steps that run in sequence to optimize, detect, and cluster support/resistance levels.

## Workflow Overview

The SR workflow consists of three main steps:

1. **SR Parameter Optimization** (`sr_parameter_optimization.py`)
   - Optimizes SR detection parameters using Bayesian optimization
   - Tests parameters using the actual EnhancedSRDetector
   - Outputs optimized parameters for SR detection

2. **SR Detection** (`sr_detection.py`)
   - Detects SR levels using optimized parameters from step 1
   - Uses EnhancedSRDetector with SHAP/LIME explanations
   - Saves SR levels as artifacts for downstream use

3. **SR Clustering** (`sr_clustering.py`)
   - Clusters detected SR levels for better organization
   - Reads SR detection results from artifacts automatically
   - Uses advanced clustering algorithms (HDBSCAN, DBSCAN, K-means, etc.)

## File Locations

```
/workspace/
├── scripts/
│   └── run_sr_workflow.py          # Main workflow runner script
├── src/training/steps/market_analysis/
│   ├── components/
│   │   ├── sr_parameter_optimization.py  # Step 1: Parameter optimization
│   │   ├── sr_detection.py              # Step 2: SR detection
│   │   └── sr_clustering.py             # Step 3: SR clustering
│   └── sr_detection.py                   # BaseStep version (legacy)
└── SR_WORKFLOW_README.md            # This file
```

## Running the Workflow

### Using the Workflow Runner Script

The easiest way to run the complete workflow is using the provided runner script:

```bash
python scripts/run_sr_workflow.py \
    --symbol ETHUSDT \
    --exchange binance \
    --timeframe 15m \
    --direction longs \
    --mode light
```

### Parameters

- `--symbol`: Trading symbol (default: ETHUSDT)
- `--exchange`: Exchange name (default: binance)
- `--timeframe`: Timeframe (default: 15m)
- `--direction`: Trading direction - 'longs' or 'shorts' (default: longs)
- `--mode`: Execution mode - 'light', 'full', or 'blank' (default: light)

### Execution Modes

- **light**: Fast execution with reduced data (recommended for testing)
- **full**: Complete execution with all data
- **blank**: Minimal execution for quick validation

## Running Individual Steps

You can also run each step individually if needed:

### Step 1: Parameter Optimization

```python
from src.training.steps.market_analysis.components.sr_parameter_optimization import SRParameterOptimizationStep

optimizer = SRParameterOptimizationStep()
result = await optimizer.execute({
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'timeframe': '15m',
    'execution_mode': 'light',
    'enable_bayesian_hpo': True,
    'enable_vectorbt': True,
    'enable_hardware_optimization': True
})

# Extract optimized parameters
optimized_params = result['metrics']['optimized_parameters']
```

### Step 2: SR Detection

```python
from src.training.steps.market_analysis.components.sr_detection import SRDetectionComponent

detector = SRDetectionComponent()
result = await detector.execute({
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'timeframe': '15m',
    'execution_mode': 'light',
    'sr_parameters': optimized_params,  # Pass optimized parameters from step 1
    'enable_shap_lime': True,
    'enable_vectorbt': True,
    'enable_hardware_optimization': True
})

# SR levels are automatically saved as artifacts
sr_levels = result['detection_result']['levels']
```

### Step 3: SR Clustering

```python
from src.training.steps.market_analysis.components.sr_clustering import SRClusteringComponent

clusterer = SRClusteringComponent()
result = await clusterer.execute({
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'timeframe': '15m',
    'execution_mode': 'light',
    'clustering_algorithm': 'ensemble',  # Use ensemble for best results
    'enable_hardware_optimization': True,
    'enable_vectorbt_optimization': True
})

# SR levels are loaded automatically from artifacts
clusters = result['clustering_result']['clusters']
```

## Workflow Details

### Step 1: SR Parameter Optimization

**Purpose**: Find optimal parameters for SR detection

**How it works**:
1. Creates a search space of possible SR detection parameters
2. Uses Bayesian optimization (TPE) to efficiently explore parameter space
3. For each parameter combination, runs actual SR detection using EnhancedSRDetector
4. Evaluates quality based on SR level strength, touch count, and clustering efficiency
5. Returns best parameter combination

**Key Features**:
- Bayesian optimization with staged search (coarse → fine → TPE)
- VectorBT optimization for fast parameter testing
- Hardware-aware optimization (M1 Mac support)
- Advanced validation (Purged CV, data leakage detection)
- Uses actual EnhancedSRDetector for realistic scoring

**Output Artifacts**:
- `sr_parameter_optimization_result`: Optimized parameters and metrics

### Step 2: SR Detection

**Purpose**: Detect SR levels using optimized parameters

**How it works**:
1. Loads optimized parameters from step 1 (if available)
2. Creates EnhancedSRDetector with these parameters
3. Detects SR levels from market data
4. Generates SHAP/LIME explanations for detected levels
5. Saves SR levels as artifacts for downstream use

**Key Features**:
- Uses optimized parameters from step 1
- SHAP/LIME explainability for SR levels
- VectorBT optimization for efficient detection
- Hardware optimization (M1 Mac support)
- Advanced validation and data leakage detection
- Fallback to default parameters if optimization not available

**Output Artifacts**:
- `sr_detection_result`: Detected SR levels with explanations
- `market_data`: Market data used for detection

### Step 3: SR Clustering

**Purpose**: Cluster SR levels for better organization

**How it works**:
1. Automatically loads SR levels from step 2 artifacts
2. Extracts features from SR levels (price, strength, touches, etc.)
3. Applies clustering algorithm (HDBSCAN, DBSCAN, K-means, or ensemble)
4. Organizes levels into meaningful clusters
5. Calculates clustering quality metrics

**Key Features**:
- Automatically reads SR detection results from artifacts
- Multiple clustering algorithms (HDBSCAN, DBSCAN, K-means, Spectral, GMM)
- Ensemble clustering for best results
- Hardware optimization (M1 Mac support, GPU acceleration)
- VectorBT optimization for feature extraction
- Advanced quality metrics (silhouette score, Calinski-Harabasz, Davies-Bouldin)
- Adaptive parameter tuning
- Data leakage detection
- SHAP/LIME explainability for clustering decisions

**Output Artifacts**:
- `sr_clustering_result`: Clustered SR levels with metrics
- `sr_levels_dictionary`: SR levels dictionary for feature bank access

## Artifact Flow

The workflow uses artifacts to pass data between steps:

```
┌─────────────────────────────────┐
│  Step 1: Parameter Optimization │
│  (sr_parameter_optimization.py) │
└────────────┬────────────────────┘
             │
             ├─> Artifact: sr_parameter_optimization_result
             │   └─> optimized_parameters
             │
             v
┌─────────────────────────────────┐
│  Step 2: SR Detection           │
│  (sr_detection.py)               │
│  - Reads: optimized_parameters  │
└────────────┬────────────────────┘
             │
             ├─> Artifact: sr_detection_result
             │   └─> levels (SR levels list)
             │
             v
┌─────────────────────────────────┐
│  Step 3: SR Clustering          │
│  (sr_clustering.py)              │
│  - Reads: sr_detection_result   │
└────────────┬────────────────────┘
             │
             └─> Artifact: sr_clustering_result
                 └─> clusters (clustered SR levels)
```

## Enhanced Features

All three steps include:

- **VectorBT Optimization**: Fast vectorized operations for better performance
- **Hardware Optimization**: M1 Mac GPU/CPU optimization for maximum speed
- **SHAP/LIME Explainability**: Understand why SR levels are detected/clustered
- **Advanced Validation**: Purged CV, temporal validation, data leakage detection
- **Memory Optimization**: Efficient memory usage for large datasets
- **Adaptive Tuning**: Automatic parameter adjustment based on data characteristics
- **Quality Metrics**: Comprehensive metrics for evaluating results

## Troubleshooting

### Issue: SR Detection not using optimized parameters

**Solution**: Check that step 1 completed successfully and produced `sr_parameter_optimization_result` artifact. If not available, step 2 will use default parameters.

### Issue: SR Clustering can't find SR levels

**Solution**: Verify that step 2 completed successfully and saved `sr_detection_result` artifact. Check logs for artifact loading messages.

### Issue: Out of memory errors

**Solution**: 
- Use `--mode light` for faster execution with less data
- Enable memory optimization in config
- Reduce batch size in clustering config

### Issue: Slow performance

**Solution**:
- Enable VectorBT optimization (`enable_vectorbt=True`)
- Enable hardware optimization (`enable_hardware_optimization=True`)
- Use M1 Mac for best performance
- Reduce parameter search space in step 1

## Examples

### Example 1: Quick Test Run

```bash
# Fast test run with minimal data
python scripts/run_sr_workflow.py \
    --symbol BTCUSDT \
    --exchange binance \
    --timeframe 5m \
    --mode light
```

### Example 2: Full Production Run

```bash
# Complete run with all features
python scripts/run_sr_workflow.py \
    --symbol ETHUSDT \
    --exchange binance \
    --timeframe 15m \
    --mode full
```

### Example 3: Custom Symbol and Direction

```bash
# Run for shorts on different symbol
python scripts/run_sr_workflow.py \
    --symbol SOLUSDT \
    --exchange binance \
    --timeframe 1h \
    --direction shorts \
    --mode full
```

## Integration with Pipeline

The SR workflow is designed to integrate seamlessly with the larger training pipeline:

1. **Pre-training**: Market data is loaded and preprocessed
2. **Market Analysis** (SR Workflow):
   - Parameter Optimization → SR Detection → SR Clustering
3. **Feature Generation**: Uses SR levels from clustering
4. **Model Training**: Trains models using SR features
5. **Backtesting**: Tests strategies using SR levels

## Performance Tips

1. **Use light mode** for development and testing
2. **Enable all optimizations** for production runs
3. **Run step 1 occasionally** to update parameters as market conditions change
4. **Cache results** to avoid re-running expensive operations
5. **Monitor memory usage** and adjust batch sizes accordingly
6. **Use ensemble clustering** for best results (but slower)
7. **Enable GPU acceleration** if available

## Configuration Options

### Parameter Optimization Config

```python
optimization_config = {
    'enable_bayesian_hpo': True,        # Use Bayesian optimization
    'enable_vectorbt': True,            # Enable VectorBT optimization
    'enable_hardware_optimization': True, # Enable hardware optimization
    'enable_sr_detection_testing': True, # Test with actual SR detection
    'n_trials': 100,                    # Number of optimization trials
    'coarse_grid_points': 20,           # Coarse grid search points
    'fine_grid_points': 50,             # Fine grid search points
    'tpe_trials': 100                   # TPE optimization trials
}
```

### SR Detection Config

```python
detection_config = {
    'sr_parameters': optimized_params,   # From step 1
    'enable_shap_lime': True,            # Enable explanations
    'enable_vectorbt': True,             # Enable VectorBT
    'enable_hardware_optimization': True, # Enable hardware optimization
    'shap_sample_size': 100,             # SHAP samples
    'lime_sample_size': 1000             # LIME samples
}
```

### SR Clustering Config

```python
clustering_config = {
    'clustering_algorithm': 'ensemble',  # Algorithm choice
    'enable_hardware_optimization': True, # Hardware optimization
    'enable_vectorbt_optimization': True, # VectorBT optimization
    'enable_memory_optimization': True,   # Memory optimization
    'enable_gpu_acceleration': True,      # GPU acceleration
    'min_cluster_size': 2,               # Minimum cluster size
    'enable_ensemble_clustering': True,   # Use ensemble
    'ensemble_algorithms': ['hdbscan', 'dbscan', 'kmeans', 'spectral']
}
```

## Version History

- **v2.0** (Current): 
  - Integrated EnhancedSRDetector in parameter optimization
  - Added optimized parameter passing between steps
  - Enhanced artifact management
  - Improved clustering with multiple source support
  
- **v1.0**: Initial separate implementations

## Support

For issues or questions:
1. Check this README first
2. Review logs for detailed error messages
3. Verify artifact paths and permissions
4. Ensure all dependencies are installed

## License

Part of the Ares trading system.
