# Economic Relevance Analysis

## 🎯 **Objective**

Analyze the economic relevance of implicit market dimensions through their relationships with price patterns and market states. Determine which dimensions provide genuine trading value.

## 🔬 **Research Focus**

**Dimensions ↔ Patterns ↔ Market States → Trading Value**

- Pattern-dimension relationships
- Market state pattern behavior
- Causal analysis (not just correlation)
- Economic significance for trading

## 📁 **Components**

### **`pattern_dimension_analysis.py`**
Core pattern-dimension relationships:
- Which dimensions predict which patterns
- Predictive accuracy measurement
- Feature importance analysis
- Cross-validation testing

### **`market_state_relevance.py`**
Market state pattern analysis:
- How patterns behave in different market states
- State-dependent pattern strength
- Regime transition pattern analysis
- State-specific trading implications

### **`causal_analysis.py`**
Causal relationship identification:
- Granger Causality testing
- Instrumental Variables analysis
- Difference-in-Differences approaches
- Robustness testing across regimes

### **`trading_significance.py`**
Economic value measurement:
- Sharpe ratio improvements
- Information ratio analysis
- Trading signal quality
- Economic significance thresholds

## 🚀 **Usage**

```python
from research.cluster_analysis.economic_relevance import (
    PatternDimensionAnalyzer,
    MarketStateRelevanceAnalyzer,
    CausalAnalyzer,
    TradingSignificanceAnalyzer
)

# 1. Analyze pattern-dimension relationships
pattern_analyzer = PatternDimensionAnalyzer()
pattern_relevance = pattern_analyzer.analyze_all_relationships(
    patterns=price_patterns,
    dimensions=market_dimensions
)

# 2. Analyze market state effects
state_analyzer = MarketStateRelevanceAnalyzer()
state_effects = state_analyzer.analyze_pattern_state_relationships(
    patterns=price_patterns,
    market_states=cluster_labels
)

# 3. Establish causal relationships
causal_analyzer = CausalAnalyzer()
causal_results = causal_analyzer.test_causal_relationships(
    dimensions=market_dimensions,
    patterns=price_patterns,
    methods=['granger', 'instrumental_variables']
)

# 4. Measure economic significance
trading_analyzer = TradingSignificanceAnalyzer()
economic_value = trading_analyzer.measure_trading_significance(
    dimensions=market_dimensions,
    patterns=price_patterns,
    market_states=cluster_labels,
    price_data=market_data
)
```

## 🔬 **Key Research Questions**

### **1. Pattern-Dimension Relationships**
- Which dimensions predict momentum persistence?
- Which dimensions predict mean reversion speed?
- Which dimensions predict volatility expansion?
- Which dimensions predict breakout confirmation?

### **2. Market State Effects**
- Are momentum patterns stronger in trending states?
- Are reversion patterns stronger in mean-reverting states?
- Do volatility patterns cluster in high-vol states?
- How do patterns transition between states?

### **3. Causal Relationships**
- Do volume dimensions CAUSE momentum patterns?
- Do volatility dimensions CAUSE mean reversion?
- Do microstructure dimensions CAUSE breakout patterns?
- What are the causal lag structures?

### **4. Economic Significance**
- Which dimension-pattern combinations generate alpha?
- What are the Sharpe ratio improvements?
- Which relationships are robust across time?
- What are the transaction cost implications?

## 📊 **Analysis Framework**

### **Predictive Analysis**
```python
# For each dimension-pattern combination
for dimension in market_dimensions:
    for pattern in price_patterns:
        # Classification accuracy (binary patterns)
        if pattern.is_binary:
            accuracy = test_classification_accuracy(
                features=dimension_features,
                target=pattern.binary_labels
            )
        
        # Regression accuracy (intensity patterns)  
        if pattern.has_intensity:
            r2_score = test_regression_accuracy(
                features=dimension_features,
                target=pattern.intensity_values
            )
```

### **Market State Analysis**
```python
# For each pattern in each market state
for state in market_states:
    for pattern in price_patterns:
        state_mask = (market_state_labels == state)
        
        # Pattern frequency in this state
        pattern_frequency = pattern.labels[state_mask].mean()
        
        # Pattern intensity in this state
        pattern_intensity = pattern.intensity[state_mask].mean()
        
        # Statistical significance
        significance = test_state_pattern_difference(
            pattern.labels, market_state_labels, state
        )
```

### **Causal Testing**
```python
# Granger causality: Does X cause Y?
def granger_causality_test(dimension_X, pattern_Y, max_lags=10):
    # Restricted model: Y(t) = α + β₁Y(t-1) + ... + βₚY(t-p) + ε(t)
    # Unrestricted: Y(t) = α + β₁Y(t-1) + ... + βₚY(t-p) + γ₁X(t-1) + ... + γₚX(t-p) + ε(t)
    
    f_statistic, p_value = perform_f_test(restricted_rss, unrestricted_rss)
    return {'causality': p_value < 0.05, 'p_value': p_value}
```

## 📊 **Expected Findings**

### **Strong Relationships (Expected)**
- **Volume → Breakout Patterns**: High volume predicts breakout success
- **Volatility → Mean Reversion**: High volatility accelerates reversion
- **Momentum → Trend Persistence**: Strong momentum predicts continuation
- **Microstructure → Execution Impact**: Poor microstructure affects slippage

### **Market State Effects (Expected)**
- **Trending States**: Momentum patterns stronger, reversion patterns weaker
- **Mean-Reverting States**: Reversion patterns stronger, momentum patterns weaker  
- **High-Vol States**: All patterns more volatile, shorter duration
- **Low-Vol States**: Patterns more persistent, cleaner signals

### **Causal Structures (Hypotheses)**
- Volume changes → Momentum pattern emergence (1-2 day lag)
- Volatility spikes → Mean reversion acceleration (immediate)
- Microstructure deterioration → Pattern degradation (intraday)
- Correlation breakdown → Breakout pattern emergence (1-3 day lag)

## 📊 **Outputs**

### **Relevance Matrix**
```python
relevance_matrix = pd.DataFrame({
    'momentum_persistence': {
        'volume': 0.72,
        'volatility': 0.45, 
        'momentum': 0.89,
        'mean_reversion': 0.12,
        'microstructure': 0.34,
        'correlation': 0.56
    },
    'mean_reversion_speed': {
        'volume': 0.23,
        'volatility': 0.78,
        'momentum': 0.15,
        'mean_reversion': 0.91,
        'microstructure': 0.67,
        'correlation': 0.43
    }
})
```

### **Market State Effects**
```python
state_effects = {
    'trending_state': {
        'momentum_patterns': {'frequency': 0.45, 'intensity': 0.72},
        'reversion_patterns': {'frequency': 0.12, 'intensity': 0.23}
    },
    'mean_reverting_state': {
        'momentum_patterns': {'frequency': 0.18, 'intensity': 0.34},
        'reversion_patterns': {'frequency': 0.67, 'intensity': 0.89}
    }
}
```

### **Economic Significance**
```python
economic_significance = {
    'volume_momentum_combination': {
        'sharpe_improvement': 0.34,
        'information_ratio': 0.28,
        'economic_significance': True
    },
    'volatility_reversion_combination': {
        'sharpe_improvement': 0.41,
        'information_ratio': 0.35,
        'economic_significance': True
    }
}
```

## 🔗 **Integration**

**Input Sources:**
- `price_patterns/`: Pattern labels and intensities
- `market_factor_analysis/`: Market dimension features
- `clustering/`: Market state labels
- Price/volume data for validation

**Final Outputs:**
- **Trading Recommendations**: Which dimension-pattern combinations to use
- **Model Training Guidance**: How to structure regime-aware models
- **Risk Management Rules**: State-dependent position sizing
- **Research Conclusions**: What works, what doesn't, and why

**Key Deliverables:**
- `relevance_matrix.csv`: Dimension-pattern relationship strengths
- `market_state_effects.json`: How patterns behave in different states
- `causal_relationships.json`: Established causal structures
- `economic_significance.json`: Trading value assessments
- `trading_recommendations.md`: Final research conclusions and trading guidance