# Cluster Analysis Research Framework

## 🎯 **Research Workflow Overview**

This framework implements a systematic 4-step approach to market analysis:

```
Features → Market Dimensions → Market States → Economic Relevance
   ↓              ↓               ↓              ↓
   1/         2/              3/             4/
Price       Market          Market         Economic
Patterns    Factor          State          Relevance
           Analysis        Clustering      Analysis
```

## 📁 **Directory Structure**

### **1. Price Patterns** (`price_patterns/`)
**Objective**: Discover and mathematically define price movement patterns

- **Mathematical Definitions**: Precise, reproducible pattern formulas
- **Pure Price Focus**: Only price movements, no external factors
- **ML Discovery**: LSTM, clustering, matrix profile pattern discovery
- **Binary + Intensity Targets**: Classification and regression targets

**Key Outputs**: Pattern labels, intensity gradients, mathematical definitions

### **2. Market Factor Analysis** (`market_factor_analysis/`)
**Objective**: Transform engineered features into coherent market dimensions

- **Dimension Discovery**: Statistical methods (PCA, FA, ICA) to find latent factors
- **Feature Clustering**: Group similar features into market dimensions
- **Factor Validation**: Statistical tests for dimension coherence
- **Dimension Naming**: Interpret and label discovered dimensions

**Key Outputs**: Market dimension features, dimension interpretations

### **3. Clustering** (`clustering/`)
**Objective**: Define coherent market states from implicit dimensions

- **Regime Discovery**: Cluster market periods by dimension characteristics
- **Optimal Selection**: Balance homogeneity vs sample count
- **Validation**: Statistical and economic validation of clusters
- **Stability Analysis**: Temporal consistency of market states

**Key Outputs**: Market state labels, cluster characteristics, validation metrics

### **4. Economic Relevance** (`economic_relevance/`)
**Objective**: Analyze dimension-pattern relationships and trading utility

- **Pattern-Dimension Analysis**: Which dimensions predict which patterns
- **Market State Relevance**: How patterns behave in different market states
- **Causal Analysis**: Establish causal relationships vs correlations
- **Trading Significance**: Economic value for trading strategies

**Key Outputs**: Relevance scores, causal relationships, trading recommendations

## 🚀 **Quick Start**

```python
from src.research.cluster_analysis import (
    PricePatternOrchestrator,
    MarketFactorAnalyzer, 
    MarketStateClusterer,
    EconomicRelevanceAnalyzer
)

# 1. Discover price patterns
pattern_analyzer = PricePatternOrchestrator()
patterns = pattern_analyzer.discover_all_patterns(price_data)

# 2. Extract market dimensions from features
factor_analyzer = MarketFactorAnalyzer()
dimensions = factor_analyzer.discover_market_dimensions(feature_data)

# 3. Cluster market states
clusterer = MarketStateClusterer()
market_states = clusterer.discover_market_states(dimensions)

# 4. Analyze economic relevance
relevance_analyzer = EconomicRelevanceAnalyzer()
relevance = relevance_analyzer.analyze_pattern_dimension_relevance(
    patterns, dimensions, market_states
)
```

## 🔬 **Research Questions Addressed**

1. **What price patterns exist?** → `price_patterns/`
2. **What market dimensions drive behavior?** → `market_factor_analysis/`
3. **What distinct market states exist?** → `clustering/`
4. **Which dimensions matter for which patterns?** → `economic_relevance/`

## 📊 **Integration Points**

- **Feature Engineering**: Uses features from `src/feature_engineering_roadmap/`
- **Data Management**: Integrates with `src/utils/data/`
- **ML Training**: Provides targets and features for model training
- **Trading Systems**: Generates regime-aware trading signals

## 🎯 **Key Benefits**

1. **Clear Separation of Concerns**: Each step has distinct objective
2. **Systematic Workflow**: Logical progression from features to trading
3. **Statistical Rigor**: Comprehensive validation at each step
4. **Economic Focus**: Trading utility drives analysis decisions
5. **Reproducible Research**: Mathematical precision in all definitions

---

**Research Framework Version**: 1.0  
**Last Updated**: December 2024