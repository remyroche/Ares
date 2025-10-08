# Market Factor Analysis

## 🎯 **Objective**

Transform engineered features into coherent, interpretable market dimensions through statistical factor analysis and feature clustering.

## 🔬 **Research Focus**

**Features → Implicit Market Dimensions**

- Statistical dimensionality reduction (PCA, FA, ICA)
- Feature clustering by similarity/correlation
- Dimension interpretation and naming
- Factor validation and coherence testing

## 📁 **Components**

### **`dimension_discovery.py`**
Core dimension discovery methods:
- Principal Component Analysis (PCA)
- Factor Analysis (FA) 
- Independent Component Analysis (ICA)
- Mutual Information-based clustering
- Statistical validation (KMO, Bartlett tests)

### **`factor_extraction.py`**
Advanced factor extraction:
- Rotated factor solutions (Varimax, Promax)
- Hierarchical factor models
- Sparse factor identification
- Cross-validation of factor stability

### **`feature_clustering.py`**
Feature grouping methods:
- Correlation-based clustering
- Mutual information clustering
- Graph-based community detection
- Ensemble clustering approaches

### **`statistical_analysis.py`**
Comprehensive statistical validation:
- Factor loading significance
- Communality analysis
- Factor score reliability
- Dimension interpretability metrics

## 🚀 **Usage**

```python
from src.research.cluster_analysis.market_factor_analysis import (
    MarketDimensionDiscoverer,
    FactorExtractor,
    FeatureClusterer,
    DimensionValidator
)

# 1. Discover implicit dimensions
discoverer = MarketDimensionDiscoverer()
dimensions = discoverer.discover_dimensions(
    features=engineered_features,
    methods=['pca', 'factor_analysis', 'ica']
)

# 2. Extract interpretable factors
extractor = FactorExtractor()
factors = extractor.extract_rotated_factors(
    features=engineered_features,
    n_factors=dimensions.optimal_n_factors
)

# 3. Cluster features by similarity
clusterer = FeatureClusterer()
feature_groups = clusterer.cluster_features(
    features=engineered_features,
    similarity_threshold=0.7
)

# 4. Validate dimension coherence
validator = DimensionValidator()
validation = validator.validate_dimensions(factors, feature_groups)
```

## 📊 **Expected Market Dimensions**

Based on feature engineering pipeline, expect to discover:

### **Volume Dimension**
- Volume patterns, ratios, momentum
- Volume-price relationships
- Trade intensity proxies

### **Volatility Dimension**  
- Realized volatility, clustering
- Volatility persistence, asymmetry
- GARCH-type effects

### **Momentum Dimension**
- Price momentum across timeframes
- Trend strength, persistence
- Momentum decay patterns

### **Mean Reversion Dimension**
- Price deviation from means
- Reversion speed indicators
- Oversold/overbought signals

### **Microstructure Dimension**
- Bid-ask spread proxies
- Order flow indicators
- Market depth estimates

### **Correlation Dimension**
- Auto-correlation patterns
- Cross-timeframe correlations
- Lead-lag relationships

## 📊 **Outputs**

### **Dimension Features**
```python
market_dimensions = {
    'volume': dimension_features_df,
    'volatility': dimension_features_df,
    'momentum': dimension_features_df,
    'mean_reversion': dimension_features_df,
    'microstructure': dimension_features_df,
    'correlation': dimension_features_df
}
```

### **Dimension Interpretations**
- Factor loadings and interpretations
- Feature contributions to each dimension
- Statistical significance of factors
- Economic meaning of dimensions

### **Validation Metrics**
- Factor reliability scores
- Dimension coherence measures
- Cross-validation stability
- Interpretability assessments

## 🔗 **Integration**

**Input Sources:**
- `src/feature_engineering_roadmap/`: Engineered features
- Feature correlation matrices
- Cross-timeframe feature relationships

**Downstream Usage:**
1. **Clustering**: Use dimensions to define market states
2. **Economic Relevance**: Test dimension-pattern relationships
3. **ML Training**: Use dimensions as feature groups

**Key Outputs for Next Steps:**
- `market_dimensions.pkl`: Dimension feature DataFrames
- `dimension_interpretations.json`: Factor meanings and loadings
- `dimension_validation.json`: Statistical validation results