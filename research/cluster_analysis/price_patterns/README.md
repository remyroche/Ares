# Price Patterns Discovery & Definition

## 🎯 **Objective**

Discover and mathematically define price movement patterns using only price data. This provides clean, reproducible pattern definitions for downstream analysis.

## 🔬 **Research Focus**

**WHAT price does, not WHY it moves**

- Pure price action analysis
- No external factors (volume, news, fundamentals)
- Mathematical precision in pattern definitions
- Binary + intensity target generation

## 📁 **Components**

### **`mathematical_definitions.py`**
Core mathematical pattern definitions:
- Momentum Persistence
- Mean Reversion Speed  
- Volatility Expansion
- Confirmed Breakouts
- Trend Acceleration

Each pattern provides:
- Binary labels: `[0,1,0,1,...]` (classification)
- Intensity gradients: `[0.0,0.8,0.2,0.9,...]` (regression)
- Mathematical formulas
- Statistical validation

### **`pure_price_patterns.py`**
Pure price action implementations:
- No confounding variables
- Clean pattern isolation
- Reproducible results
- Economic significance testing

### **`ml_discovery/`**
Advanced pattern discovery:
- **`lstm_discovery.py`**: Neural pattern discovery
- **`matrix_profile_discovery.py`**: Exact motif identification  
- **`clustering_discovery.py`**: Clustering-based patterns
- **`anomaly_discovery.py`**: Anomaly-based patterns

### **`pattern_validation.py`**
Pattern quality assessment:
- Statistical significance testing
- Economic relevance validation
- Frequency analysis
- Predictability scoring

## 🚀 **Usage**

```python
from research.cluster_analysis.price_patterns import (
    MathematicalPatternDefinitions,
    PurePricePatternOrchestrator,
    PatternValidator
)

# 1. Define patterns mathematically
pattern_definer = MathematicalPatternDefinitions()
momentum_pattern = pattern_definer.momentum_persistence(prices)
reversion_pattern = pattern_definer.mean_reversion_speed(prices)

# 2. Discover patterns with ML
orchestrator = PurePricePatternOrchestrator()
all_patterns = orchestrator.discover_all_patterns(prices)

# 3. Validate pattern quality
validator = PatternValidator()
validation_results = validator.validate_patterns(all_patterns)

# 4. Export ML targets
binary_targets = orchestrator.export_binary_targets(all_patterns)
intensity_targets = orchestrator.export_intensity_targets(all_patterns)
```

## 📊 **Outputs**

### **Pattern Labels**
- **Binary**: Pattern exists (1) or not (0)
- **Intensity**: Pattern strength (0.0 to 1.0)
- **Combined**: Both binary and intensity

### **Pattern Definitions**
- Mathematical formulas
- Parameter specifications
- Validation criteria
- Economic interpretation

### **Statistical Analysis**
- Frequency distributions
- Duration statistics
- Magnitude analysis
- Significance testing

## 🔗 **Integration**

**Downstream Usage:**
1. **Market Factor Analysis**: Use patterns as validation targets
2. **Clustering**: Analyze pattern behavior across market states
3. **Economic Relevance**: Test which dimensions predict which patterns
4. **ML Training**: Use as supervised learning targets

**Key Outputs for Next Steps:**
- `pattern_labels.csv`: Binary and intensity targets
- `pattern_definitions.json`: Mathematical specifications
- `pattern_validation.json`: Quality metrics