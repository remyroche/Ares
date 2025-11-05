# Enhanced Statsmodel Clustering System

This module provides a comprehensive framework for enhanced statsmodel clustering with advanced feature engineering, hybrid clustering algorithms, hierarchical parameter optimization, and comprehensive quality assessment.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Enhanced Statsmodel Clustering                │
├─────────────────────────────────────────────────────────────────┤
│  Feature Engineering  │  Clustering Algorithms  │  Optimization  │
│                     │                        │               │
│ • Enhanced Features  │ • Hybrid Clustering     │ • Hierarchical  │
│ • Temporal Features │ • Static Clustering      │ • Multi-Objective  │
│ • Factor Exposures   │ • Ensemble Methods       │ • Economic Objectives │
│ • Rank Normalization │ • Churn Regularization   │ • Early Stopping   │
│ • Covariance Stabilization │                        │               │
└─────────────────────────────────────────────────────────────────┘
                              │
                    Quality Assessment
                              │
                    • Stability Metrics
                    • Calibration Tests
                    • Residual Analysis
                    • Sensitivity Analysis
                    • Change Point Detection
                    • Economic Validation
                    • CSV Export with Datetime
```

## 🚀 Key Features

### 1. Enhanced Feature Engineering
- **Multiple Feature Types**: Raw returns, log-returns, volatility, overnight returns
- **Rolling Features**: Mean, std, skewness, kurtosis, z-scores with proper `.shift(1)` to avoid look-ahead bias
- **Factor Exposures**: Market, size, value, momentum factors
- **Cross-Sectional Rank Normalization**: Reduces outliers and improves clustering
- **Covariance Stabilization**: Ledoit-Wolf shrinkage for robust estimation

### 2. Advanced Clustering Algorithms
- **Hybrid Clustering**: Combines static asset clustering with temporal modeling
- **Static Clustering**: Hierarchical, spectral, Louvain community detection
- **Temporal Modeling**: MarkovRegression for regime dynamics
- **Ensemble Methods**: Multiple algorithms with consensus clustering
- **Churn Regularization**: Stickiness priors for transitions

### 3. Hierarchical Parameter Optimizer
- **3-Stage Optimization**: 
  - Stage 1: Global search (random/Bayesian/forest)
  - Stage 2: Local refinement (BFGS/Nelder-Mead)
  - Stage 3: Validation on holdout/rolling windows
- **Economic Objectives**: Combines Sharpe, turnover, stability metrics
- **Multi-Objective Optimization**: Pareto front generation
- **Early Stopping**: Convergence monitoring and patience

### 4. Comprehensive Quality Assessment
- **Stability Metrics**: ARI/NMI across bootstrap samples
- **Calibration Tests**: Reliability diagrams for regime probabilities
- **Residual Analysis**: Serial correlation, heteroscedasticity tests
- **Sensitivity Analysis**: Perturb lookback window analysis
- **Change Point Detection**: Ruptures library integration
- **Economic Validation**: Regime-specific Sharpe, hit rates, drawdowns
- **CSV Export**: Datetime-based filenames with detailed metrics

## 📁 Directory Structure

```
src/training/steps/market_analysis/statsmodel_clustering/
├── __init__.py
├── feature_engineering/
│   ├── __init__.py
│   ├── enhanced_features.py
│   ├── temporal_features.py
│   ├── factor_exposures.py
│   ├── rank_normalization.py
│   └── covariance_stabilization.py
├── clustering/
│   ├── __init__.py
│   ├── hybrid_clustering.py
│   ├── temporal_clustering.py
│   ├── static_clustering.py
│   ├── ensemble_clustering.py
│   └── churn_regularization.py
├── optimization/
│   ├── __init__.py
│   ├── hierarchical_optimizer.py
│   ├── optuna_integration.py
│   ├── economic_objectives.py
│   └── multi_start_optimizer.py
├── assessment/
│   ├── __init__.py
│   ├── stability_metrics.py
│   ├── calibration_tests.py
│   ├── residual_tests.py
│   ├── sensitivity_analysis.py
│   ├── change_point_detection.py
│   ├── economic_validation.py
│   └── quality_integration.py
├── examples/
│   └── enhanced_clustering_example.py
└── utils/
    ├── result_converter.py
    └── diagnostics.py
```

## 🔧 Usage Examples

### Basic Usage

```python
from src.training.steps.market_analysis.statsmodel_clustering.feature_engineering import create_enhanced_feature_engineer
from src.training.steps.market_analysis.statsmodel_clustering.clustering import create_hybrid_clustering_engine
from src.training.steps.market_analysis.statsmodel_clustering.optimization import create_hierarchical_optimizer
from src.training.steps.market_analysis.statsmodel_clustering.assessment import create_quality_assessment_integrator

# 1. Enhanced Feature Engineering
feature_engineer = create_enhanced_feature_engineer(
    include_raw_returns=True,
    include_log_returns=True,
    include_realized_vol=True,
    include_rolling_features=True,
    rolling_windows=[5, 10, 20],
    shift_periods=1,
    enable_rank_normalization=True
)

features = feature_engineer.extract_features(
    price_data=price_data,
    volume_data=volume_data,
    market_data=market_data
)

# 2. Hybrid Clustering
clustering_engine = create_hybrid_clustering_engine(
    static_method='hierarchical',
    n_asset_clusters=5,
    n_regimes=3,
    aggregation_method='pca',
    covariance_method='ledoit_wolf'
)

clustering_results = clustering_engine.fit_predict(
    returns=returns,
    features=features
)

# 3. Hierarchical Parameter Optimization
def objective_function(params, data):
    # Your custom objective function
    clustering_engine = create_hybrid_clustering_engine(**params)
    results = clustering_engine.fit_predict(data[0], data[1])
    return results['quality_score']

parameter_space = {
    'n_asset_clusters': {'type': 'int', 'low': 3, 'high': 10},
    'n_regimes': {'type': 'int', 'low': 2, 'high': 5},
    'aggregation_method': {'type': 'categorical', 'choices': ['pca', 'mean', 'weighted_mean']},
    'covariance_method': {'type': 'categorical', 'choices': ['ledoit_wolf', 'exponential', 'shrunk']}
}

optimizer = create_hierarchical_optimizer(
    objective_function=objective_function,
    parameter_space=parameter_space,
    stage1_method='bayesian',
    stage1_n_trials=50,
    stage2_method='bfgs',
    enable_economic_objectives=True
)

optimization_results = optimizer.optimize(
    data=(returns, features)
)

# 4. Comprehensive Quality Assessment
quality_integrator = create_quality_assessment_integrator(
    output_dir="outcomes",
    include_datetime=True,
    integrate_with_cluster_assessor=True,
    enable_all_assessments=True
)

quality_results = quality_integrator.assess_quality(
    model=clustering_results['temporal_model'],
    data=features,
    regime_labels=clustering_results['regime_labels'],
    forward_returns=returns.shift(-1),
    timestamps=price_data.index,
    symbol="SYMBOL"
)
```

### Advanced Example

See `examples/enhanced_clustering_example.py` for a complete working example that demonstrates:

1. Sample data generation
2. Parameter space definition
3. Hierarchical optimization
4. Quality assessment integration
5. CSV export with datetime

## 📊 Quality Metrics

The system provides comprehensive quality metrics:

### Standard Quality Metrics
- **Silhouette Score**: Cluster separation and cohesion (-1 to 1)
- **Davies-Bouldin Index**: Cluster similarity (lower is better)
- **Calinski-Harabasz Index**: Between-cluster dispersion (higher is better)
- **Temporal Smoothness**: Regime persistence over time (0-1)
- **Regime Persistence**: Average regime duration

### Enhanced Quality Metrics
- **Stability ARI/NMI**: Consistency across bootstrap samples
- **Calibration Error**: Reliability of regime probabilities
- **Residual Tests**: Serial correlation, heteroscedasticity, normality
- **Sensitivity Analysis**: Robustness to parameter changes
- **Change Point Detection**: Alignment with regime boundaries
- **Economic Validation**: Regime-specific Sharpe, hit rates, drawdowns

## 🔄 Integration with Existing Systems

The enhanced statsmodel clustering system is designed to integrate seamlessly with existing components:

### Cluster Quality Assessor Integration
```python
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import ClusterQualityAssessor

# Automatic integration
quality_integrator = create_quality_assessment_integrator(
    integrate_with_cluster_assessor=True
)
```

### CSV Export with Datetime
```python
# Generates files like:
# - quality_assessment_detailed_SYMBOL_20251105_143022.csv
# - quality_assessment_summary_SYMBOL_20251105_143022.csv
# - cluster_quality_metrics_SYMBOL_20251105_143022.csv
```

## 🎯 Optimization Strategies

### Hierarchical Optimization
1. **Global Search**: Random/Bayesian/forest exploration
2. **Local Refinement**: BFGS/Nelder-Mead around best parameters
3. **Validation**: Rolling window/heldout validation

### Economic Objectives
- **Regime Sharpe Ratio**: Risk-adjusted returns by regime
- **Information Ratio**: Active return relative to benchmark
- **Turnover Penalty**: Penalize excessive regime switching
- **Regime Stability**: Reward persistent regimes

## 📈 Performance Considerations

### Computational Efficiency
- **Vectorized Operations**: NumPy/Pandas vectorization where possible
- **Parallel Processing**: Multi-core utilization for expensive operations
- **Memory Management**: Efficient data structures and garbage collection
- **Caching**: Reuse expensive calculations

### Anti-Leakage Safeguards
- **Temporal Shifting**: Proper `.shift(1)` for all rolling features
- **Forward Returns**: Use next-period returns for validation
- **Cross-Validation**: Time-aware validation splits
- **Bootstrap Sampling**: Preserve temporal structure

## 🛠️ Configuration Options

### Feature Engineering
```python
config = FeatureConfig(
    include_raw_returns=True,
    include_log_returns=True,
    include_realized_vol=True,
    vol_windows=[5, 10, 20],
    rolling_windows=[5, 10, 20],
    shift_periods=1,
    enable_rank_normalization=True
)
```

### Clustering
```python
config = HybridClusteringConfig(
    static_method='hierarchical',
    n_asset_clusters=5,
    n_regimes=3,
    aggregation_method='pca',
    covariance_method='ledoit_wolf'
)
```

### Optimization
```python
config = OptimizationConfig(
    stage1_method='bayesian',
    stage1_n_trials=50,
    stage2_method='bfgs',
    enable_economic_objectives=True,
    economic_weights={
        'sharpe_ratio': 0.4,
        'information_ratio': 0.3,
        'turnover_penalty': 0.2,
        'regime_stability': 0.1
    }
)
```

## 📚 Dependencies

### Required Dependencies
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `scipy`: Statistical functions
- `statsmodels`: Markov regression models

### Optional Dependencies
- `sklearn`: Machine learning algorithms (clustering, covariance)
- `optuna`: Bayesian optimization
- `skopt`: Forest-based optimization
- `networkx`: Graph algorithms (Louvain)
- `community`: Community detection

### Internal Dependencies
- `src.utils.tprint`: Logging utilities
- `src.training.steps.market_analysis.clusters.cluster_quality_assessor`: Quality assessment

## 🚨 Error Handling

The system includes comprehensive error handling:

### Graceful Degradation
- Falls back to simpler methods when advanced dependencies unavailable
- Provides informative warnings about missing capabilities
- Continues operation with reduced functionality

### Robust Error Recovery
- Try-catch blocks around all major operations
- Detailed error logging with context
- Fallback to safe default values

### Input Validation
- Parameter validation with clear error messages
- Data type checking and conversion
- Boundary condition handling

## 📝 Logging and Monitoring

### Progress Tracking
- Detailed progress reporting for long-running operations
- Stage-wise progress for multi-stage processes
- Performance metrics and timing information

### Debug Information
- Configurable debug levels
- Intermediate result inspection
- Parameter evolution tracking

## 🔮 Future Enhancements

### Planned Features
- **Deep Learning Integration**: Neural network-based regime detection
- **Real-Time Optimization**: Online parameter adaptation
- **Multi-Asset Support**: Cross-asset regime analysis
- **Advanced Visualization**: Interactive regime exploration tools

### Performance Improvements
- **GPU Acceleration**: CUDA support for intensive computations
- **Distributed Computing**: Multi-machine scaling
- **Incremental Learning**: Online model updates

## 📄 License and Credits

This enhanced statsmodel clustering system is part of the Ares trading framework.

### License
- Proprietary - All rights reserved

### Credits
- Enhanced by Ares Development Team
- Based on original statsmodel clustering framework
- Incorporates best practices from quantitative finance research

---

## 📞 Support

For questions, issues, or enhancement requests:

1. Check existing documentation and examples
2. Review error logs for diagnostic information
3. Consult the comprehensive example implementation
4. Contact the development team through proper channels

**Note**: This system requires careful parameter tuning and validation for production use. Always backtest thoroughly before deployment.
