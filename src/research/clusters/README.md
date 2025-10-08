# Market Regime Clustering Research Framework

A comprehensive research framework for discovering **implicit market dimensions** captured by existing features and assessing their **economic significance** for trading regime identification. This framework integrates with your existing feature engineering pipeline to discover which market dynamics are most relevant for regime-based trading.

## 🎯 Overview

This framework addresses the key research question: **"What market regimes (distinct behavioral patterns) can we discover, and do they have sufficient economic significance to justify training different ML models for each regime?"**

**Note**: This framework focuses on **regime discovery and validation**. ML model training should be implemented separately in your existing training pipeline based on the discovered regimes and generated trading rules.

## 🎯 Correct Research Flow

**Your Approach** (implemented in this framework):
```
Market Data → Discover Regimes → Validate Economic Significance → Train Different ML Models per Regime
```

The framework first discovers regimes (market behavioral patterns), then validates whether these regimes are economically meaningful enough to justify the complexity of training separate ML models for each regime.

## 🔬 Research Approach

### Integration with Existing Feature Engineering
- **Uses features from** `src/feature_engineering_roadmap/feature_generators.py`
- **Leverages** cross-timeframe analysis and microstructure proxies
- **Discovers** implicit dimensions from existing comprehensive feature set
- **Maintains** compatibility with current optimization systems

### Key Research Dimensions
- **Liquidity** (volume-based proxies, spread indicators from OHLCV)
- **Market Microstructure** (order flow proxies, trade intensity from existing features)
- **Volume** (volume patterns, volume-price relationships)
- **Momentum** (price momentum, trend strength, persistence)
- **Volatility** (realized volatility, volatility clustering, regime changes)
- **Correlation** (auto-correlation patterns, cross-timeframe correlations)

### Economic Significance Focus
- **Information Ratio** analysis for each dimension
- **Regime Separability** metrics (economic differences between regimes)
- **Sharpe Ratio** proxies for trading relevance
- **Feature Coherence** within dimensions

## 🏗️ Architecture

The framework consists of 7 main components:

```
src/research/clusters/
├── __init__.py                      # Main exports and framework entry point
├── dimension_analyzer.py            # Implicit dimension discovery from existing features
├── regime_clusterer.py             # Clustering with dimension analysis
├── feature_importance.py           # Economic significance analysis
├── validation_metrics.py           # Regime quality validation system
├── integration_layer.py            # HMM timing and integration strategies
├── visualization.py                # Visualization and reporting tools
├── refined_example.py              # Research workflow examples
├── correlation_analysis_explanation.md  # Correlation analysis guide
└── README.md                       # This documentation
```

## 🔄 Research Workflows

### 1. Enhanced Dimension-First Approach
```
Many Features → Statistical Dimensionality Analysis → Market Dimensions → Economic Relevance → Clustering → HMM Enhancement
```

**Detailed Pipeline Steps:**
1. **Comprehensive Feature Generation**: Use ALL available features from `src/feature_engineering_roadmap/` (100+ features)
2. **Statistical Dimensionality Analysis**: PCA, FA, ICA with statistical tests (KMO, Bartlett, etc.)
3. **Market Dimension Discovery**: Group features into market dimensions (liquidity, momentum, etc.)
4. **Economic Relevance Analysis**: Determine which dimensions influence price action (NEW STEP)
   - Price instability influence
   - Trend duration impact  
   - Reversal violence modulation
   - Momentum intensity effects
   - Trend acceleration impact
5. **Clustering with Statistical Validation**: AIC, BIC, Gap statistic, etc.
6. **Economic Validation**: 9 comprehensive economic metrics
7. **HMM Integration**: Optimal timing strategies

### 2. HMM-First Approach  
```
Features → HMM Regime Discovery → Dimension Analysis → Economic Validation
```

### 3. Comparative Approach
```
Both workflows → Compare Results → Identify Best Strategy
```

## 🚀 Quick Start

### New Data-Driven Approach (Recommended)

```python
import pandas as pd
import numpy as np
from src.research.clusters import (
    DataDrivenClusteringFramework,
    data_driven_regime_discovery,
    quick_regime_discovery
)

# Load your market data (OHLCV format)
market_data = pd.read_csv('your_market_data.csv')
features = pd.read_csv('your_features.csv')  # Your engineered features

# Quick discovery with optimal parameters
result = quick_regime_discovery(features, market_data)

print(f"Discovered {result.n_clusters} regimes")
print(f"Strategy: {result.recommendations['model_training_strategy']}")
print(f"Confidence: {result.recommendations['confidence_level']}")

# Or complete analysis with threshold discovery
result = data_driven_regime_discovery(features, market_data)

print(f"Optimal CV threshold: {result.optimal_cv_threshold:.3f}")
print(f"Optimal similarity threshold: {result.optimal_similarity_threshold:.3f}")

# Key research insights
if result.empirical_discovery_result:
    if result.empirical_discovery_result.cv_breaking_point:
        print(f"⚠️ CV breaks down at: {result.empirical_discovery_result.cv_breaking_point:.3f}")
    if result.empirical_discovery_result.similarity_breaking_point:
        print(f"⚠️ Similarity breaks down at: {result.empirical_discovery_result.similarity_breaking_point:.3f}")
```

### Advanced Data-Driven Usage

```python
from src.research.clusters import (
    DataDrivenClusteringConfig,
    EmpiricalDiscoveryConfig,
    SimilarityClusteringConfig,
    SimilarityMethod
)

# Custom configuration for empirical discovery
config = DataDrivenClusteringConfig(
    enable_threshold_discovery=True,
    discovery_config=EmpiricalDiscoveryConfig(
        cv_range=(0.1, 0.8, 20),  # Test CV from 0.1 to 0.8
        similarity_range=(0.3, 0.95, 15),  # Test similarity from 0.3 to 0.95
        min_economic_relevance=0.15,
        breaking_point_threshold=0.8  # 20% degradation threshold
    ),
    similarity_config=SimilarityClusteringConfig(
        similarity_method=SimilarityMethod.CORRELATION,
        enable_economic_validation=True
    )
)

framework = DataDrivenClusteringFramework(config)
result = framework.discover_optimal_regimes(features, market_data)

# Analyze breaking points
if result.empirical_discovery_result:
    print("📊 Empirical Findings:")
    print(f"   - CV breaking point: {result.empirical_discovery_result.cv_breaking_point}")
    print(f"   - Similarity breaking point: {result.empirical_discovery_result.similarity_breaking_point}")
    print(f"   - Answers: 'At what point do relaxed thresholds destroy economic relevance?'")
```

### Legacy Approach (Deprecated)

```python
# Old approach - no longer recommended
# Uses arbitrary cluster numbers and lacks CV validation
from src.research.clusters import RegimeClusterer, ClusteringMethod

clusterer = RegimeClusterer()
result = clusterer.run_single_method(features.values, ClusteringMethod.SIMILARITY_MATRIX)
```

### Integration with Existing HMM Systems

```python
from src.regime.clusters import HMMIntegrationLayer, IntegrationMethod

# Initialize integration layer
integration = HMMIntegrationLayer()

# Run comparative analysis between HMM and clustering
result = await integration.run_integration_analysis(
    market_data, method=IntegrationMethod.COMPARATIVE
)

# Generate integration report
report = integration.generate_integration_report(result)
```

## 📊 Components

### 1. Market Dimension Analyzer

Analyzes various market dimensions for their relevance to regime identification:

```python
from src.regime.clusters import MarketDimensionAnalyzer, DimensionAnalysisConfig

config = DimensionAnalysisConfig(
    lookback_periods=[5, 10, 20, 50],
    use_pca=True,
    use_mutual_information=True
)

analyzer = MarketDimensionAnalyzer(config)
results = analyzer.analyze_all_dimensions(market_data)

# Get top performing dimensions
top_dimensions = analyzer.get_top_dimensions(5)
print(f"Top dimension: {top_dimensions[0][0].value}")
```

**Supported Dimensions:**
- Liquidity (spread proxies, volume-based measures)
- Microstructure (order flow, trade intensity)
- Volume (patterns, correlations, momentum)
- Momentum (price momentum, trend strength)
- Volatility (realized vol, clustering, mean reversion)
- Correlation (auto-correlation, cross-correlations)
- Seasonality (time-based patterns)

### 2. Regime Clusterer

Advanced clustering algorithms optimized for market regime discovery:

```python
from src.regime.clusters import RegimeClusterer, ClusteringMethod

clusterer = RegimeClusterer()

# Run specific clustering method
result = clusterer.run_single_method(data, ClusteringMethod.KMEANS)

# Run all methods and compare
all_results = clusterer.run_all_methods(data)
comparison = clusterer.compare_methods()

# Optimize cluster number
optimization_results = clusterer.optimize_cluster_number(
    data, ClusteringMethod.GMM, k_range=(2, 15)
)
```

**Supported Methods:**
- Traditional: K-Means, Gaussian Mixture Models, Hierarchical
- Density-based: DBSCAN, HDBSCAN
- Advanced: Spectral Clustering, DTW Clustering
- Ensemble: Voting-based ensemble methods

### 3. Feature Importance Analyzer

Comprehensive feature importance analysis for regime-relevant features:

```python
from src.regime.clusters import RegimeFeatureImportance, ImportanceMethod

analyzer = RegimeFeatureImportance()

# Run single importance method
result = analyzer.analyze_single_method(
    features, regime_labels, ImportanceMethod.RANDOM_FOREST
)

# Run all methods
all_results = analyzer.analyze_all_methods(features, regime_labels)

# Get consensus features
consensus = analyzer.get_consensus_features(n=10, min_methods=2)
```

**Supported Methods:**
- Statistical: Mutual Information, Correlation, ANOVA F-test
- Model-based: Random Forest, XGBoost, SHAP values
- Regularization: LASSO/Ridge coefficients
- Time-series: Granger Causality
- Ensemble: Weighted combination of methods

### 4. Validation Metrics

Comprehensive validation system for regime quality assessment:

```python
from src.regime.clusters import RegimeValidationMetrics, ValidationMetric

validator = RegimeValidationMetrics()

# Validate specific metric
result = validator.validate_single_metric(
    market_data, regime_labels, ValidationMetric.SILHOUETTE_SCORE
)

# Run all validation metrics
all_results = validator.validate_all_metrics(market_data, regime_labels)

# Calculate composite score
composite_score = validator.calculate_composite_score()
```

**Validation Categories:**
- **Clustering Quality**: Silhouette, Calinski-Harabasz, Davies-Bouldin
- **Temporal Stability**: Consistency, persistence, transition frequency
- **Economic Significance**: Return/volatility separability, Sharpe ratios
- **Trading Relevance**: Predictability, signal quality, transition costs
- **Statistical Validation**: Homogeneity, separation, significance tests

### 5. ML Training Framework

Regime-aware machine learning model training:

```python
from src.regime.clusters import RegimeMLTrainer, TrainingStrategy

trainer = RegimeMLTrainer()

# Train regime-specific models
result = trainer.train_single_strategy(
    features, target, regime_labels, 
    TrainingStrategy.REGIME_SPECIFIC
)

# Train multi-regime aware model
result = trainer.train_single_strategy(
    features, target, regime_labels,
    TrainingStrategy.MULTI_REGIME
)

# Compare strategies
comparison = trainer.compare_strategies()
best_strategy = trainer.get_best_strategy()
```

**Training Strategies:**
- **Regime-Specific**: Separate models per regime
- **Multi-Regime**: Single model with regime features
- **Ensemble**: Combination of multiple approaches
- **Hierarchical**: Hierarchical regime-aware training

### 6. HMM Integration Layer

Seamless integration with existing HMM regime discovery systems:

```python
from src.regime.clusters import HMMIntegrationLayer, IntegrationMethod

integration = HMMIntegrationLayer()

# Run hybrid analysis
result = await integration.run_integration_analysis(
    market_data, method=IntegrationMethod.HYBRID
)

# Compare HMM vs Clustering
result = await integration.run_integration_analysis(
    market_data, method=IntegrationMethod.COMPARATIVE
)
```

**Integration Methods:**
- **HMM Enhanced**: HMM + clustering features
- **Clustering Enhanced**: Clustering + HMM features
- **Hybrid**: Combined approach
- **Ensemble**: Ensemble of both methods
- **Comparative**: Side-by-side comparison

### 7. Visualization System

Publication-quality visualizations and interactive dashboards:

```python
from src.regime.clusters import RegimeVisualization

visualizer = RegimeVisualization()

# Create regime timeseries plot
fig = visualizer.plot_regime_timeseries(market_data, regime_labels)

# Create feature importance heatmap
fig = visualizer.plot_feature_importance_heatmap(importance_results)

# Create interactive dashboard
dashboard_path = visualizer.create_interactive_dashboard(
    market_data, regime_labels, analysis_results
)

# Generate comprehensive report
report_path = visualizer.generate_comprehensive_report(
    market_data, regime_labels, analysis_results
)
```

**Visualization Types:**
- Regime timeseries with market data overlay
- Clustering quality metrics comparison
- Feature importance heatmaps and rankings
- Dimension analysis radar charts
- Validation metrics dashboards
- Interactive regime exploration tools

## 🔬 Research Applications

### Market Regime Discovery

```python
# Comprehensive regime discovery workflow
from src.regime.clusters import *

# 1. Analyze which dimensions matter most
dimension_analyzer = MarketDimensionAnalyzer()
dimension_results = dimension_analyzer.analyze_all_dimensions(market_data)
top_dimensions = dimension_analyzer.get_top_dimensions(3)

# 2. Use top dimensions for clustering
important_features = extract_dimension_features(market_data, top_dimensions)
clusterer = RegimeClusterer()
clustering_results = clusterer.run_all_methods(important_features)

# 3. Validate regime quality
validator = RegimeValidationMetrics()
validation_results = validator.validate_all_metrics(
    market_data, clustering_results.best_labels
)

# 4. Train trading models on discovered regimes
ml_trainer = RegimeMLTrainer()
trading_models = ml_trainer.train_all_strategies(
    features, trading_signals, clustering_results.best_labels
)
```

### Feature Importance Research

```python
# Discover which features are most important for regime identification
feature_analyzer = RegimeFeatureImportance()

# Test different feature sets
feature_sets = {
    'technical': technical_indicators,
    'microstructure': microstructure_features,
    'volume': volume_features,
    'volatility': volatility_features
}

importance_by_category = {}
for category, features in feature_sets.items():
    results = feature_analyzer.analyze_all_methods(features, regime_labels)
    importance_by_category[category] = results

# Find consensus across methods
consensus_features = feature_analyzer.get_consensus_features(20)
```

### Clustering Algorithm Comparison

```python
# Compare different clustering approaches for market regimes
clusterer = RegimeClusterer()

# Test different numbers of clusters
optimization_results = {}
for method in [ClusteringMethod.KMEANS, ClusteringMethod.GMM, ClusteringMethod.HIERARCHICAL]:
    optimization_results[method] = clusterer.optimize_cluster_number(
        market_data, method, k_range=(2, 20)
    )

# Compare ensemble methods
ensemble_config = ClusteringConfig(
    ensemble_methods=[ClusteringMethod.KMEANS, ClusteringMethod.GMM, ClusteringMethod.SPECTRAL],
    ensemble_voting='weighted'
)
ensemble_clusterer = RegimeClusterer(ensemble_config)
ensemble_results = ensemble_clusterer.run_single_method(
    market_data, ClusteringMethod.ENSEMBLE
)
```

## 📈 Integration with Existing Systems

### Enhancing HMM Regime Discovery

```python
# Use clustering research to enhance existing HMM systems
integration = HMMIntegrationLayer()

# Method 1: HMM Enhanced with clustering features
result = await integration.run_integration_analysis(
    market_data, method=IntegrationMethod.HMM_ENHANCED
)

# Method 2: Clustering enhanced with HMM features  
result = await integration.run_integration_analysis(
    market_data, method=IntegrationMethod.CLUSTERING_ENHANCED
)

# Method 3: Hybrid approach
result = await integration.run_integration_analysis(
    market_data, method=IntegrationMethod.HYBRID
)
```

### Migration from Existing Systems

```python
# Migrate existing HMM workflows to enhanced framework
from src.regime.clusters.integration_layer import HMMDataAdapter

adapter = HMMDataAdapter()

# Convert existing HMM results to clustering format
hmm_result = load_existing_hmm_result()
features, regime_labels = adapter.hmm_to_clustering_format(hmm_result)

# Apply new clustering research
clusterer = RegimeClusterer()
enhanced_results = clusterer.run_all_methods(features)

# Compare with original HMM
comparison = compare_regime_sets(hmm_result, enhanced_results)
```

## 🛠️ Configuration

### Dimension Analysis Configuration

```python
from src.regime.clusters import DimensionAnalysisConfig

config = DimensionAnalysisConfig(
    lookback_periods=[5, 10, 20, 50, 100],
    min_regime_samples=100,
    significance_threshold=0.05,
    volume_windows=[5, 10, 20],
    volatility_windows=[10, 20, 50],
    momentum_windows=[5, 10, 20, 50],
    use_pca=True,
    use_mutual_information=True,
    use_feature_importance=True
)
```

### Clustering Configuration

```python
from src.regime.clusters import ClusteringConfig

config = ClusteringConfig(
    n_clusters=5,
    random_state=42,
    kmeans_params={'n_init': 10, 'max_iter': 300},
    gmm_params={'covariance_type': 'full', 'max_iter': 100},
    dbscan_params={'eps': 0.5, 'min_samples': 5},
    ensemble_methods=['kmeans', 'gmm', 'hierarchical'],
    ensemble_voting='majority',
    min_cluster_size=50,
    max_clusters=20,
    silhouette_threshold=0.3
)
```

### Validation Configuration

```python
from src.regime.clusters import ValidationConfig

config = ValidationConfig(
    significance_level=0.05,
    bootstrap_samples=1000,
    confidence_level=0.95,
    min_regime_duration=10,
    risk_free_rate=0.02,
    transaction_cost=0.001,
    max_acceptable_drawdown=0.20
)
```

## 📊 Output and Results

### Analysis Reports

The framework generates comprehensive reports including:

1. **Dimension Analysis Report**
   - Top performing market dimensions
   - Feature importance by dimension
   - Composite scoring and rankings

2. **Clustering Quality Report**
   - Method comparison and rankings
   - Quality metrics (silhouette, etc.)
   - Optimal cluster number analysis

3. **Validation Report**
   - Regime quality assessment
   - Statistical significance tests
   - Economic relevance metrics

4. **ML Training Report**
   - Strategy comparison
   - Performance by regime
   - Feature importance analysis

5. **Integration Report**
   - HMM vs clustering comparison
   - Hybrid method results
   - Recommendations for implementation

### Visualizations

- **Static Plots**: High-quality matplotlib/seaborn charts
- **Interactive Dashboards**: Plotly-based interactive visualizations
- **Publication-Ready**: Configurable styling and export formats

### Data Exports

- JSON format for programmatic access
- CSV exports for spreadsheet analysis
- Pickle files for model persistence
- Markdown reports for documentation

## 🔧 Advanced Usage

### Custom Clustering Algorithms

```python
from src.research.clusters.regime_clusterer import BaseClusterer, ClusteringResult

class CustomClusterer(BaseClusterer):
    def fit_predict(self, data):
        # Implement your custom clustering algorithm
        labels = your_clustering_algorithm(data)
        metrics = self._calculate_metrics(data, labels)
        
        return ClusteringResult(
            method=ClusteringMethod.CUSTOM,
            labels=labels,
            n_clusters=len(np.unique(labels)),
            cluster_centers=None,
            metrics=metrics,
            metadata={'custom_param': 'value'}
        )

# Register and use custom clusterer
clusterer = RegimeClusterer()
clusterer.clusterers[ClusteringMethod.CUSTOM] = CustomClusterer(config)
```

### Custom Validation Metrics

```python
from src.research.clusters.validation_metrics import BaseValidator, ValidationResult

class CustomValidator(BaseValidator):
    def validate(self, data, regime_labels, **kwargs):
        # Implement your custom validation metric
        value = calculate_custom_metric(data, regime_labels)
        
        return ValidationResult(
            metric=ValidationMetric.CUSTOM,
            value=value,
            confidence_interval=None,
            p_value=None,
            interpretation="Custom metric interpretation",
            metadata={'calculation_details': 'info'}
        )

# Register and use custom validator
validator = RegimeValidationMetrics()
validator.validators[ValidationMetric.CUSTOM] = CustomValidator(config)
```

## 📚 Examples

See `example_usage.py` for a complete example that demonstrates:

1. Sample data generation with known regimes
2. Dimension analysis workflow
3. Clustering analysis and comparison
4. Feature importance analysis
5. Validation metrics calculation
6. ML model training on regimes
7. HMM integration analysis
8. Comprehensive visualization generation

Run the example:

```bash
cd src/research/clusters
python example_usage.py
```

## 🤝 Contributing

This research framework is designed to be extensible. You can contribute by:

1. **Adding New Dimensions**: Implement additional market dimension analyzers
2. **New Clustering Methods**: Add specialized clustering algorithms
3. **Validation Metrics**: Develop domain-specific validation measures
4. **ML Strategies**: Implement new regime-aware training approaches
5. **Visualizations**: Create new chart types and dashboards

## 📄 License

This framework is part of the larger trading system and follows the same licensing terms.

## 🔗 Integration Points

This framework integrates with:

- **Existing HMM Systems**: `src/training/steps/market_analysis/components/hmm_*`
- **Feature Engineering**: `src/feature_engineering_roadmap/`
- **Data Management**: `src/utils/data/`
- **Logging System**: `src/utils/logger`

**Note**: This framework focuses on regime discovery and validation. ML model training should be implemented separately based on the discovered regimes and generated trading rules.

## 📞 Support

For questions or issues with the regime clustering research framework:

1. Check the example usage in `example_usage.py`
2. Review the component documentation in each module
3. Examine the integration layer for HMM compatibility
4. Refer to the visualization examples for output formats

---

**Happy Regime Research! 🎯📊🤖**