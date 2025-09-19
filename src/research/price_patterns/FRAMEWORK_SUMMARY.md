# Price Patterns Research Framework - Complete Summary

## 🎯 **Unified Framework: `src/research/price_patterns/`**

All price pattern research is now consolidated in a single, focused directory with comprehensive mathematical pattern definitions and enhanced ML target generation.

## 📊 **Complete Framework Components**

### **🔬 Core Pattern Discovery**

#### **`core_patterns.py` - Pure Price Action Patterns**
**Focus**: WHAT price does (not WHY it moves)

**5 Fundamental Patterns:**
1. **Momentum Persistence** - Price momentum continues with gradual decay
2. **Price Reversion** - Price returns to reference levels  
3. **Trend Acceleration** - Price movement speeds up
4. **Range Breakout** - Price breaks established ranges
5. **Extreme Reversal** - Large moves followed by reversal

**Each pattern provides:**
- Binary labels: `[0,1,0,1,...]`
- Intensity gradients: `[0.0,0.8,0.2,0.9,...]`
- Mathematical definitions
- Statistical validation

#### **`pattern_discovery_framework.py` - Extended Pattern Library**
**18+ Mathematical Pattern Definitions:**
- All core patterns plus advanced variations
- False breakouts, gaps, consolidations
- Seasonal patterns, extreme movements
- Volume-price interactions (when using OHLCV)

### **📈 Enhanced ML Target Generation**

#### **`gradient_targets.py` - Binary + Intensity Targets**
**Innovation**: Pattern strength measurement

```python
# Traditional
binary_labels = [0, 1, 0, 1, 0, ...]

# Enhanced  
intensity_gradients = [0.0, 0.8, 0.0, 0.9, 0.0, ...]

# Benefits:
# - Regression targets (not just classification)
# - Pattern strength measurement
# - Risk management (scale by intensity)
# - Signal quality assessment
```

### **🤖 Advanced ML Discovery**

#### **`lstm_discovery.py` - Neural Pattern Discovery**
**Method**: LSTM autoencoders for latent price patterns

```python
# Implementation approach
autoencoder = LSTMAutoencoder(sequence_length=30, latent_dim=8)
autoencoder.fit(normalized_price_sequences)

# Discover patterns
latent_patterns = autoencoder.encode(sequences)
anomalies = find_reconstruction_anomalies(autoencoder, sequences)
clusters = cluster_latent_representations(latent_patterns)
```

**Expected Discoveries:**
- Non-linear price sequence relationships
- Complex multi-period patterns
- Latent momentum/reversion behaviors

#### **`matrix_profile_discovery.py` - Motif Discovery**
**Method**: Matrix profile for exact recurring subsequences

```python
# Implementation with stumpy
import stumpy

mp = stumpy.stump(price_returns, m=20)
motifs = stumpy.motifs(price_returns, mp, max_motifs=10)
discords = stumpy.discords(price_returns, mp, max_discords=5)
```

**Expected Discoveries:**
- Exact recurring price movement motifs
- Seasonal price patterns
- Rare/unusual price behaviors (discords)

## 🚀 **Usage Examples**

### **Quick Start - Pure Price Patterns:**
```bash
cd /workspace/src/research/price_patterns
python run_pure_pattern_discovery.py --use_sample_data
```

### **Complete Framework:**
```bash
python run_complete_pattern_discovery.py --use_sample_data
```

### **With Your Data:**
```bash
python run_pure_pattern_discovery.py --data_path /path/to/your/prices.csv
```

### **Programmatic Usage:**
```python
from src.research.price_patterns import (
    PurePricePatternOrchestrator,
    GradientPatternTargetGenerator,
    LSTMPricePatternDiscovery,
    MatrixProfileOrchestrator
)

# Core patterns with gradients
orchestrator = PurePricePatternOrchestrator()
results = orchestrator.discover_all_pure_patterns(prices)
ml_targets = orchestrator.export_combined_targets(results)

# Enhanced gradient targets
generator = GradientPatternTargetGenerator()
gradient_results = generator.generate_all_gradient_targets(prices)

# LSTM discovery (requires TensorFlow/PyTorch)
lstm_discoverer = LSTMPricePatternDiscovery()
lstm_patterns = lstm_discoverer.discover_lstm_patterns(prices)

# Matrix profile discovery (requires stumpy)
mp_orchestrator = MatrixProfileOrchestrator()
mp_results = mp_orchestrator.run_complete_matrix_profile_analysis(prices)
```

## 📈 **Output Formats**

### **CSV Files:**
- `binary_pattern_targets.csv` - Binary classification targets
- `intensity_pattern_targets.csv` - Regression targets
- `combined_pattern_targets.csv` - Both binary and intensity

### **JSON Analysis:**
- `pure_pattern_analysis.json` - Statistical analysis
- `pure_pattern_definitions.json` - Mathematical definitions

### **Example Output:**
```python
# combined_pattern_targets.csv
                momentum_persistence  momentum_persistence_intensity  price_reversion  price_reversion_intensity
2020-01-01                         0                            0.0                1                       0.7
2020-01-02                         1                            0.8                0                       0.0
2020-01-03                         0                            0.0                0                       0.0
2020-01-04                         1                            0.9                1                       0.6
```

## 🎯 **Research Question Answered**

> *"Which market dimensions predict which specific price patterns?"*

**Implementation:**
```python
# 1. Load pattern targets
targets = pd.read_csv('price_patterns/combined_pattern_targets.csv')

# 2. Test dimension predictive power
for pattern_col in targets.columns:
    for dimension_name, dimension_features in market_dimensions.items():
        
        if '_intensity' in pattern_col:
            # Regression for intensity
            accuracy = test_regression(dimension_features, targets[pattern_col])
        else:
            # Classification for binary
            accuracy = test_classification(dimension_features, targets[pattern_col])
        
        print(f"{dimension_name} → {pattern_col}: {accuracy:.3f}")
```

## 🔧 **Installation & Dependencies**

### **Basic Framework:**
```bash
pip install numpy pandas scipy scikit-learn
```

### **Advanced ML Discovery:**
```bash
# LSTM patterns
pip install tensorflow>=2.8.0

# Matrix profile patterns  
pip install stumpy

# Complete installation
pip install tensorflow stumpy numpy pandas scipy scikit-learn
```

## 🎯 **Key Framework Advantages**

### **1. Mathematical Precision**
- Exact formulas for reproducible results
- No subjective pattern interpretation
- Statistical validation for all patterns

### **2. Enhanced ML Targets**
- Binary classification targets
- Continuous regression targets
- Multi-task learning capabilities

### **3. Pure Price Focus**
- Only price movements matter
- No confounding factors
- Clean dimension relevance testing

### **4. Advanced Discovery**
- Neural network pattern discovery
- Exact motif identification
- Comprehensive pattern library

### **5. Research Foundation**
- Scientific testing of market dimension relevance
- Economic significance validation
- Trading strategy development

This unified framework provides everything needed to mathematically discover price patterns and scientifically determine which market dimensions are relevant for predicting them.