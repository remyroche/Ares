# Price Patterns Research Framework

## 🎯 **Framework Overview**

This unified framework provides comprehensive mathematical discovery and definition of price patterns with both binary labels and gradient-based intensity measurements for ML applications.

**Core Focus**: Mathematical precision in pattern definition and ML-ready target generation.

## 📁 **Directory Structure**

```
src/research/price_patterns/
├── __init__.py                           # Main framework exports
├── core_patterns.py                     # 5 fundamental pure price patterns
├── gradient_targets.py                  # Binary + intensity gradient targets
├── lstm_discovery.py                    # LSTM autoencoder pattern discovery
├── matrix_profile_discovery.py          # Matrix profile motif discovery
├── pattern_discovery_framework.py       # Complete pattern framework (18+ patterns)
├── advanced_pattern_definitions.py      # Sophisticated pattern definitions
├── pure_price_action_patterns.py        # Pure price action implementations
├── ml_pure_price_pattern_discovery.py   # ML-based pure price discovery
├── run_pure_pattern_discovery.py        # Pure price action runner
├── run_complete_pattern_discovery.py    # Complete framework runner
├── pattern_discovery_example.py         # Basic usage examples
└── README.md                            # This documentation
```

## 🔬 **Core Innovation: Mathematical Pattern Precision**

### **Traditional Approach (Vague):**
- "Look for momentum patterns"
- "Identify breakouts"
- "Find mean reversion"

### **Mathematical Approach (Precise):**
```python
# Momentum Persistence Pattern
IF |momentum(t)| > 0.01 AND 
   same_direction ≥70% for 10 periods AND
   magnitude_decay ≥60% gradual
THEN binary_label = 1

# Intensity Gradient
intensity = momentum_magnitude * direction_persistence * decay_quality
# Result: [0.0, 0.8, 0.0, 0.9, 0.0, 0.6, ...]
```

## 📊 **Enhanced ML Targets**

### **Binary Labels (Classification):**
```python
binary_targets = [0, 1, 0, 1, 0, 1, ...]  # Pattern exists or not
```

### **Intensity Gradients (Regression):**
```python
intensity_targets = [0.0, 0.8, 0.0, 0.9, 0.0, 0.6, ...]  # Pattern strength
```

### **Benefits:**
- **Classification models**: Predict pattern occurrence
- **Regression models**: Predict pattern strength
- **Multi-task learning**: Train both simultaneously
- **Risk management**: Scale positions by pattern intensity

## 🎯 **Available Pattern Types**

### **Core Pure Price Patterns (5 Fundamental)**
1. **Momentum Persistence** - Price momentum continues with decay
2. **Price Reversion** - Price returns to reference levels
3. **Trend Acceleration** - Price movement speeds up
4. **Range Breakout** - Price breaks established ranges
5. **Extreme Reversal** - Large moves followed by reversal

### **Extended Pattern Framework (18+ Total)**
- False breakouts, gaps, consolidations
- Volume-price interactions (when using OHLCV data)
- Seasonal patterns, extreme movements
- Advanced momentum and reversion patterns

### **ML-Discovered Patterns**
- **LSTM Autoencoder**: Latent price sequence patterns
- **Matrix Profile**: Recurring price motifs and discords
- **Clustering**: Price shape families
- **Anomaly Detection**: Unusual price behaviors

## 🚀 **Quick Start**

### **Basic Pure Price Pattern Discovery:**
```python
from research.price_patterns import PurePricePatternOrchestrator

# Initialize orchestrator
orchestrator = PurePricePatternOrchestrator()

# Discover patterns in price series
results = orchestrator.discover_all_pure_patterns(price_series)

# Export binary labels
binary_targets = orchestrator.export_binary_labels(results)

# Export intensity gradients
intensity_targets = orchestrator.export_intensity_gradients(results)

# Combined targets (binary + intensity)
combined_targets = orchestrator.export_combined_targets(results)
```

### **Gradient Target Generation:**
```python
from research.price_patterns import GradientPatternTargetGenerator

# Generate enhanced targets with intensity
generator = GradientPatternTargetGenerator()
gradient_results = generator.generate_all_gradient_targets(price_series)

# Export ML-ready targets
ml_targets = generator.export_ml_ready_targets(gradient_results)

# Access different formats
binary_only = ml_targets['binary_only']
intensity_only = ml_targets['intensity_only'] 
combined = ml_targets['combined']
```

### **LSTM Pattern Discovery:**
```python
from research.price_patterns import LSTMPricePatternDiscovery

# Discover latent patterns (requires TensorFlow/PyTorch)
discoverer = LSTMPricePatternDiscovery(sequence_length=30)
lstm_patterns = discoverer.discover_lstm_patterns(price_series)
```

### **Matrix Profile Discovery:**
```python
from research.price_patterns import MatrixProfileOrchestrator

# Discover recurring motifs (requires stumpy)
orchestrator = MatrixProfileOrchestrator()
mp_results = orchestrator.run_complete_matrix_profile_analysis(price_series)
```

## 📈 **Command Line Usage**

### **Pure Price Action Discovery:**
```bash
cd /workspace/src/research/price_patterns
python run_pure_pattern_discovery.py --use_sample_data
```

### **Complete Pattern Framework:**
```bash
python run_complete_pattern_discovery.py --use_sample_data
```

### **With Your Data:**
```bash
python run_pure_pattern_discovery.py --data_path /path/to/your/prices.csv
```

## 🎯 **ML Training Applications**

### **Classification (Binary Labels):**
```python
# Predict pattern occurrence
X = market_dimension_features
y = binary_pattern_labels  # [0,1,0,1,...]

classifier = RandomForestClassifier()
classifier.fit(X, y)

# Will pattern occur?
pattern_probability = classifier.predict_proba(X_new)
```

### **Regression (Intensity Gradients):**
```python
# Predict pattern strength
X = market_dimension_features
y = intensity_gradients  # [0.0,0.8,0.2,0.9,...]

regressor = RandomForestRegressor()
regressor.fit(X, y)

# How strong will pattern be?
pattern_intensity = regressor.predict(X_new)
```

### **Multi-Task Learning:**
```python
# Combined approach
X = market_dimension_features
y_binary = binary_labels
y_intensity = intensity_gradients

# Enhanced trading decisions
will_occur = classifier.predict(X_new)
pattern_strength = regressor.predict(X_new)

if will_occur and pattern_strength > 0.6:
    position_size = base_size * pattern_strength
    execute_trade(position_size)
```

## 🔬 **Research Applications**

### **Market Dimension Relevance Testing:**
```python
# Test which dimensions predict which patterns
pattern_targets = discover_all_patterns(prices)

for pattern_name, pattern_target in pattern_targets.items():
    for dimension_name, dimension_features in market_dimensions.items():
        
        if '_intensity' in pattern_name:
            # Regression for intensity targets
            accuracy = test_regression_accuracy(dimension_features, pattern_target)
        else:
            # Classification for binary targets
            accuracy = test_classification_accuracy(dimension_features, pattern_target)
        
        print(f"{dimension_name} → {pattern_name}: {accuracy:.3f}")
```

### **Economic Significance Validation:**
```python
# Test if patterns generate profitable signals
for pattern_name, pattern_labels in binary_targets.items():
    backtest_results = backtest_pattern_strategy(
        prices, pattern_labels, intensity_gradients[f"{pattern_name}_intensity"]
    )
    
    print(f"{pattern_name} Sharpe ratio: {backtest_results.sharpe_ratio:.2f}")
```

## 🔧 **Installation Requirements**

### **Basic Framework:**
```bash
pip install numpy pandas scipy scikit-learn
```

### **Advanced ML Discovery:**
```bash
# For LSTM pattern discovery
pip install tensorflow>=2.8.0
# or
pip install torch>=1.12.0

# For matrix profile discovery
pip install stumpy
```

## 📊 **Output Formats**

### **CSV Exports:**
- `binary_pattern_targets.csv` - Binary labels for classification
- `intensity_pattern_targets.csv` - Gradient intensities for regression
- `combined_pattern_targets.csv` - Both binary and intensity

### **JSON Analysis:**
- `pure_pattern_analysis.json` - Statistical analysis and recommendations
- `pure_pattern_definitions.json` - Mathematical pattern definitions

### **Markdown Reports:**
- `gradient_targets_report.md` - Detailed gradient analysis
- Various discovery method reports

## 🎯 **Key Advantages**

### **1. Mathematical Precision**
- Exact formulas for each pattern
- Reproducible results across datasets
- No subjective interpretation

### **2. Enhanced ML Targets**
- Binary classification targets
- Continuous regression targets
- Multi-task learning capabilities

### **3. Pure Price Focus**
- Only price movements considered
- No confounding factors
- Clean dimension relevance testing

### **4. Advanced Discovery Methods**
- LSTM autoencoders for latent patterns
- Matrix profile for exact motifs
- Statistical validation for all patterns

## 🚀 **Integration with Market Analysis**

This framework integrates with your existing market analysis pipeline:

```python
# 1. Discover price patterns
from research.price_patterns import PurePricePatternOrchestrator

pattern_orchestrator = PurePricePatternOrchestrator()
pattern_results = pattern_orchestrator.discover_all_pure_patterns(prices)
ml_targets = pattern_orchestrator.export_combined_targets(pattern_results)

# 2. Test dimension relevance (from existing clusters framework)
from research.clusters import MarketDimensionAnalyzer

dimension_analyzer = MarketDimensionAnalyzer()
dimension_results = dimension_analyzer.analyze_all_dimensions(market_data)

# 3. Test which dimensions predict which patterns
for pattern_name in ml_targets.columns:
    for dim_name, dim_features in dimension_results.items():
        relevance_score = test_dimension_pattern_relevance(
            dim_features, ml_targets[pattern_name]
        )
        print(f"{dim_name} → {pattern_name}: {relevance_score:.3f}")
```

This framework provides the mathematical foundation needed to scientifically determine which market dimensions are relevant for predicting specific, well-defined price patterns.