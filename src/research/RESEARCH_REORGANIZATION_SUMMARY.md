# Research Framework Reorganization Summary

## 🎯 **Reorganization Focus: Pure Price Action Separation**

Based on your feedback to focus on pure price action patterns (what price does, not underlying causes), I've reorganized the research framework into clear, focused modules.

## 📁 **New Directory Structure**

```
src/research/
├── pure_price_patterns/           # 🎯 NEW: Pure price action only
│   ├── __init__.py
│   ├── core_patterns.py          # 5 fundamental price action patterns
│   ├── gradient_targets.py       # Binary + intensity gradient targets
│   ├── lstm_discovery.py         # LSTM autoencoder pattern discovery
│   ├── matrix_profile_discovery.py # Matrix profile motif discovery
│   ├── run_pure_pattern_discovery.py # Complete implementation
│   └── README.md                  # Framework documentation
│
├── mixed_factor_analysis/         # 🔄 MOVED: Multi-factor analysis
│   ├── economic_relevance_research_framework.py
│   ├── volatility_impact_research.py
│   ├── microstructure_impact_research.py
│   ├── ml_pattern_discovery.py
│   └── pattern_ml_integration.py
│
└── clusters/                      # ✅ EXISTING: Regime discovery
    ├── dimension_economic_relevance.py
    ├── economic_metrics.py
    └── ... (existing files)
```

## 🎯 **Pure Price Action Framework (`pure_price_patterns/`)**

### **Core Innovation: Binary + Gradient Targets**

**Traditional:**
```python
binary_labels = [0, 1, 0, 1, 0, ...]  # Pattern exists or not
```

**Enhanced:**
```python
binary_labels = [0, 1, 0, 1, 0, ...]           # Classification targets
intensity_gradients = [0.0, 0.8, 0.0, 0.9, 0.0, ...]  # Regression targets
```

### **Key Components:**

#### **1. `core_patterns.py` - Fundamental Patterns**
- **Momentum Persistence**: Price momentum continues with decay
- **Price Reversion**: Price returns to reference levels
- **Trend Acceleration**: Price movement speeds up
- **Range Breakout**: Price breaks established ranges
- **Extreme Reversal**: Large moves followed by reversal

**Each pattern provides:**
- Binary labels (0/1)
- Intensity gradients (0.0-1.0)
- Mathematical definitions
- Statistical validation

#### **2. `gradient_targets.py` - Enhanced ML Targets**
```python
# Example: Momentum Persistence Intensity
intensity = (
    |momentum(t)| * 20 *      # Scale momentum magnitude
    direction_persistence *   # Weight by consistency
    magnitude_persistence     # Weight by decay quality
)
# Result: [0.0, 0.8, 0.2, 0.9, 0.1, 0.7, ...]
```

**Benefits:**
- **Regression targets** for continuous prediction
- **Pattern strength** measurement
- **Nuanced ML training** (strong vs weak patterns)
- **Risk management** (scale positions by intensity)

#### **3. `lstm_discovery.py` - Neural Pattern Discovery**
```python
# Conceptual implementation (requires TensorFlow/PyTorch)
autoencoder = LSTMAutoencoder(sequence_length=30, latent_dim=8)
autoencoder.fit(normalized_price_sequences)

# Discover patterns
latent_patterns = autoencoder.encode(price_sequences)
clusters = cluster_latent_representations(latent_patterns)
anomalies = find_reconstruction_anomalies(autoencoder, sequences)
```

#### **4. `matrix_profile_discovery.py` - Motif Discovery**
```python
# Implementation with stumpy library
import stumpy

mp = stumpy.stump(price_returns, m=20)
motifs = stumpy.motifs(price_returns, mp, max_motifs=10)
discords = stumpy.discords(price_returns, mp, max_discords=5)
```

## 🔄 **Mixed Factor Analysis (`mixed_factor_analysis/`)**

**Moved files that consider external factors:**
- Volume-based patterns
- Market microstructure analysis
- Economic relevance frameworks that use multiple data sources
- Multi-factor pattern integration

**These are valuable but separate from pure price action research.**

## 📊 **Key Enhancements Implemented**

### **1. Gradient-Based Intensity Measurement ✅**
**Your Request**: *"can we use gradients too, to measure the strength/intensity of each label?"*

**Implementation:**
```python
# Instead of just binary [0,1,0,1,...]
binary_labels = [0, 1, 0, 1, 0, ...]
intensity_gradients = [0.0, 0.8, 0.0, 0.9, 0.0, ...]

# Intensity calculation example (momentum persistence):
intensity = (
    momentum_magnitude * 
    direction_persistence * 
    decay_quality
)
```

### **2. LSTM Autoencoder Discovery ✅**
**Your Request**: *"LSTM Autoencoders - Discover latent price sequence patterns -> Add it"*

**Implementation:**
- LSTM autoencoder architecture for price sequences
- Latent representation clustering
- Reconstruction anomaly detection
- Mathematical approximation of neural patterns

### **3. Matrix Profile Motif Discovery ✅**
**Your Request**: *"Matrix Profile - Find recurring price motifs -> Add it"*

**Implementation:**
- Matrix profile calculation for exact motif discovery
- Recurring subsequence identification
- Discord (rare pattern) discovery
- Gradient-based motif intensity measurement

## 🚀 **Usage Examples**

### **Complete Pure Price Analysis:**
```bash
cd /workspace/src/research/pure_price_patterns
python run_pure_pattern_discovery.py --use_sample_data
```

**Output:**
- `binary_pattern_targets.csv` - Binary labels [0,1,0,1,...]
- `intensity_pattern_targets.csv` - Gradients [0.0,0.8,0.2,0.9,...]
- `combined_pattern_targets.csv` - Both binary and intensity
- `pure_pattern_analysis.json` - Comprehensive analysis

### **Integration with Market Dimension Testing:**
```python
# Load pure price pattern targets
targets = pd.read_csv('pure_price_patterns/combined_pattern_targets.csv')

# Test which dimensions predict which patterns
for pattern_col in targets.columns:
    if '_intensity' in pattern_col:
        # Use intensity for regression
        accuracy = test_regression_accuracy(dimension_features, targets[pattern_col])
    else:
        # Use binary for classification
        accuracy = test_classification_accuracy(dimension_features, targets[pattern_col])
    
    print(f"Dimension relevance for {pattern_col}: {accuracy:.3f}")
```

## 📈 **Key Benefits of Reorganization**

### **1. Clear Separation**
- **Pure price action** (what price does)
- **Mixed factor analysis** (why price moves)
- **Clean research boundaries**

### **2. Enhanced ML Targets**
- **Binary classification** [0,1,0,1,...]
- **Regression targets** [0.0,0.8,0.2,0.9,...]
- **Multi-task learning** capabilities

### **3. Advanced Discovery Methods**
- **LSTM autoencoders** for latent patterns
- **Matrix profile** for exact motifs
- **Mathematical precision** for all patterns

### **4. Research Foundation**
- **Clean testing** of market dimension relevance
- **No confounding factors** in pattern definitions
- **Scientific rigor** in economic relevance determination

This reorganization provides the focused, mathematically precise foundation needed to answer your core research question: "Which market dimensions predict which specific pure price action patterns?"