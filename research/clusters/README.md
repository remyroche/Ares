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
├── integration_layer.py            # Clustering integration strategies
├── visualization.py                # Visualization and reporting tools
├── refined_example.py              # Research workflow examples
├── correlation_analysis_explanation.md  # Correlation analysis guide
└── README.md                       # This documentation
```

## 🔄 Research Workflows

### 1. Enhanced Dimension-First Approach
```
Many Features → Statistical Dimensionality Analysis → Market Dimensions → Economic Relevance → Clustering → Enhancement
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
7. **Integration Enhancement**: Optimal timing strategies

### 2. Clustering-First Approach  
```
Features → Clustering Regime Discovery → Dimension Analysis → Economic Validation
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
from research.clusters import (
    DataDrivenClusteringFramework,
    data_driven_regime_discovery,
    quick_regime_discovery
)

# Load your market data
market_data = pd.read_csv('your_market_data.csv', index_col=0, parse_dates=True)

# Quick regime discovery (automated)
regimes = quick_regime_discovery(market_data)
print(f"Discovered {len(regimes['cluster_characteristics'])} regimes")

# Advanced analysis with full framework
framework = DataDrivenClusteringFramework()
results = framework.run_complete_analysis(market_data)

# Access results
print(f"Best clustering method: {results.best_method}")
print(f"Economic significance: {results.economic_metrics}")
print(f"Regime characteristics: {results.cluster_characteristics}")
```

### Traditional Clustering Approach

```python
from research.clusters import RegimeClusterer, ClusteringMethod

# Initialize clusterer
clusterer = RegimeClusterer()

# Run all clustering methods
results = clusterer.run_all_methods(features.values, feature_names=features.columns)

# Get best method
best_method, best_result = clusterer.get_best_method()
print(f"Best method: {best_method}")
print(f"Number of clusters: {best_result.n_clusters}")
print(f"Silhouette score: {best_result.metrics['silhouette_score']}")
```

### Integration with Existing Systems

```python
from research.clusters import IntegrationLayer, IntegrationMethod

# Initialize integration layer
integration = IntegrationLayer()

# Run comparative analysis between different approaches
result = await integration.run_integration_analysis(
    market_data, method=IntegrationMethod.COMPARATIVE
)

# Access comprehensive results
print(f"Clustering results: {result.clustering_results}")
print(f"Dimension analysis: {result.dimension_analysis}")
print(f"Feature importance: {result.feature_importance}")
```

## 📊 Key Components

### 1. Dimension Analyzer

Discovers implicit market dimensions from your existing features:

```python
from research.clusters import MarketDimensionAnalyzer

analyzer = MarketDimensionAnalyzer()
dimensions = analyzer.analyze_dimensions(market_data)

print(f"Discovered dimensions: {dimensions['dimensions']}")
print(f"Economic relevance: {dimensions['economic_relevance']}")
```

### 2. Regime Clusterer

Comprehensive clustering with multiple methods and validation:

```python
from research.clusters import RegimeClusterer, ClusteringMethod

clusterer = RegimeClusterer()

# Run single method
result = clusterer.run_single_method(features.values, ClusteringMethod.SIMILARITY_MATRIX)

# Run all methods and compare
all_results = clusterer.run_all_methods(features.values, feature_names=features.columns)
```

### 3. Feature Importance Analysis

Economic significance analysis for discovered regimes:

```python
from research.clusters import RegimeFeatureImportance

importance_analyzer = RegimeFeatureImportance()
importance_results = importance_analyzer.analyze_importance(market_data, clustering_results)

print(f"Feature importance: {importance_results['feature_importance']}")
print(f"Economic metrics: {importance_results['economic_metrics']}")
```

### 4. Validation Metrics

Comprehensive regime quality validation:

```python
from research.clusters import RegimeValidationMetrics

validator = RegimeValidationMetrics()
validation_results = validator.validate_all_metrics(market_data, clustering_results)

print(f"Validation metrics: {validation_results}")
```

### 5. Integration Layer

Seamless integration with existing systems:

```python
from research.clusters import IntegrationLayer, IntegrationMethod

integration = IntegrationLayer()

# Run hybrid analysis
result = await integration.run_integration_analysis(
    market_data, method=IntegrationMethod.HYBRID
)

# Compare different approaches
result = await integration.run_integration_analysis(
    market_data, method=IntegrationMethod.COMPARATIVE
)
```

**Integration Methods:**
- **Clustering Enhanced**: Clustering + discovered features
- **Hybrid**: Combined approach
- **Ensemble**: Ensemble of multiple methods
- **Comparative**: Side-by-side comparison

## 📈 Integration with Existing Systems

### Enhancing Existing Regime Discovery

```python
# Use clustering research to enhance existing systems
integration = IntegrationLayer()

# Method 1: Enhanced clustering with discovered features
result = await integration.run_integration_analysis(
    market_data, method=IntegrationMethod.CLUSTERING_ENHANCED
)

# Method 2: Hybrid approach combining multiple methods
result = await integration.run_integration_analysis(
    market_data, method=IntegrationMethod.HYBRID
)
```

### Migration from Existing Systems

```python
# Migrate existing workflows to enhanced framework
from research.clusters.integration_layer import DataAdapter

adapter = DataAdapter()

# Convert existing results to clustering format
existing_result = load_existing_result()
features, regime_labels = adapter.convert_to_clustering_format(existing_result)

# Apply new clustering research
clusterer = RegimeClusterer()
enhanced_results = clusterer.run_all_methods(features)

# Compare with original
comparison = compare_regime_sets(existing_result, enhanced_results)
```

## 🛠️ Configuration

### Framework Configuration

```python
from research.clusters import DataDrivenClusteringConfig

config = DataDrivenClusteringConfig(
    n_clusters=5,
    enable_validation=True,
    enable_feature_importance=True,
    enable_empirical_thresholds=True,
    max_iterations=100
)

framework = DataDrivenClusteringFramework(config)
```

### Integration Configuration

```python
from research.clusters import IntegrationConfig, IntegrationMethod

config = IntegrationConfig(
    method=IntegrationMethod.HYBRID,
    clustering_n_clusters=5,
    analyze_dimensions=True,
    validate_results=True
)

integration = IntegrationLayer(config)
```

## 📊 Output and Results

### 1. Regime Discovery Results
- **Cluster assignments**: Regime labels for each time period
- **Cluster characteristics**: Statistical properties of each regime
- **Economic metrics**: Trading relevance of each regime

### 2. Dimension Analysis Results
- **Discovered dimensions**: Market dimensions from feature analysis
- **Feature importance**: Economic significance of each feature
- **Dimension coherence**: Consistency within dimensions

### 3. Validation Results
- **Clustering quality**: Silhouette score, ARI, etc.
- **Economic validation**: Sharpe ratios, information ratios
- **Stability analysis**: Regime persistence and transitions

### 4. Integration Report
- **Method comparison**: Performance comparison between approaches
- **Hybrid method results**: Combined approach results
- **Recommendations for implementation**

## 🔬 Research Examples

### Complete Research Workflow

```python
# 1. Load and prepare data
market_data = load_market_data()

# 2. Run comprehensive analysis
framework = DataDrivenClusteringFramework()
results = framework.run_complete_analysis(market_data)

# 3. Validate results
validator = RegimeValidationMetrics()
validation = validator.validate_all_metrics(market_data, results)

# 4. Analyze feature importance
importance_analyzer = RegimeFeatureImportance()
importance = importance_analyzer.analyze_importance(market_data, results)

# 5. Generate comprehensive report
report = generate_research_report(results, validation, importance)
```

### Step-by-Step Analysis

```python
# 1. Dimension discovery
dimension_analyzer = MarketDimensionAnalyzer()
dimensions = dimension_analyzer.analyze_dimensions(market_data)

# 2. Clustering analysis
clusterer = RegimeClusterer()
clustering_results = clusterer.run_all_methods(market_data.values)

# 3. Feature importance analysis
importance_analyzer = RegimeFeatureImportance()
importance_results = importance_analyzer.analyze_importance(market_data, clustering_results)

# 4. Validation metrics calculation
validator = RegimeValidationMetrics()
validation_results = validator.validate_all_metrics(market_data, clustering_results)

# 5. Integration analysis
integration = IntegrationLayer()
integration_results = await integration.run_integration_analysis(market_data)

# 6. Comprehensive visualization generation
visualizer = RegimeVisualizer()
visualizer.generate_comprehensive_report(all_results)
```

## 🚀 Getting Started

1. Check the example usage in `refined_example.py`
2. Review the component documentation in each module
3. Examine the integration layer for system compatibility
4. Refer to the visualization examples for output formats

## 📚 Dependencies

- **Core**: numpy, pandas, scikit-learn, scipy
- **Optional**: hdbscan (advanced clustering), talib (technical indicators)

## 🤝 Contributing

This framework integrates with:

- **Feature Engineering**: `src/feature_engineering_roadmap/`
- **Data Management**: `src/utils/data/`
- **Logging System**: `src/utils/logger`

## 📖 Documentation

1. Check the example usage in `refined_example.py`
2. Review the component documentation in each module
3. Examine the integration layer for system compatibility
4. Refer to the visualization examples for output formats

----

**Note**: This framework is designed for research and regime discovery. For production ML model training, integrate the discovered regimes with your existing training pipeline.