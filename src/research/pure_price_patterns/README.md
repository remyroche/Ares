# Pure Price Action Pattern Research Framework

## 🎯 **Framework Focus: Pure Price Action Only**

This framework focuses exclusively on **WHAT price does**, not **WHY it moves**. All patterns are defined mathematically using only price movements, without reference to volume, fundamentals, or market structure.

### **What We Include:**
✅ **Price movements** - Observable price behavior sequences
✅ **Mathematical precision** - Exact formulas for pattern identification  
✅ **Pattern shapes** - How price moves through time
✅ **Gradient intensities** - Strength/quality of each pattern occurrence

### **What We Exclude:**
❌ **Volume analysis** - Not part of pure price action
❌ **Fundamentals** - External factors causing moves
❌ **Market microstructure** - Underlying market mechanics
❌ **Sentiment indicators** - Why traders behave certain ways

## 📊 **Core Innovation: Binary + Gradient Targets**

### **Traditional Approach:**
```python
binary_labels = [0, 1, 0, 1, 0, 1, ...]  # Pattern exists or not
```

### **Enhanced Approach:**
```python
binary_labels = [0, 1, 0, 1, 0, 1, ...]     # Pattern exists or not
intensity_gradients = [0.0, 0.8, 0.0, 0.9, 0.0, 0.6, ...]  # Pattern strength
```

**Benefits:**
- **Regression targets**: Train models to predict pattern strength
- **Nuanced training**: Distinguish strong vs weak patterns
- **Risk management**: Scale positions by pattern intensity
- **Signal quality**: Measure confidence in pattern predictions

## 📈 **Complete Pattern Catalog**

### **Core Patterns (5 Fundamental Patterns)**

#### 1. **Momentum Persistence**
```
Mathematical Definition:
IF |momentum(t)| > 0.01 AND 
   same_direction ≥70% for 10 periods AND
   magnitude_decay ≥60% gradual
THEN binary_label = 1

Intensity = |momentum(t)| * direction_persistence * decay_quality
```

#### 2. **Price Reversion**
```
Mathematical Definition:
IF |price(t) - reference_level| / reference_level > 0.03 AND
   price moves ≥50% back toward reference_level within 15 periods
THEN binary_label = 1

Intensity = deviation_magnitude * reversion_speed * reversion_completeness
```

#### 3. **Trend Acceleration**
```
Mathematical Definition:
IF acceleration(t) and velocity(t) same sign AND
   |acceleration(t+k)| > |acceleration(t)| for ≥60% of next 8 periods
THEN binary_label = 1

Intensity = |acceleration(t)| * consistency * velocity_alignment
```

#### 4. **Range Breakout**
```
Mathematical Definition:
IF price breaks established range (range_size < 0.08) AND
   continues beyond range ≥60% for 8 periods
THEN binary_label = 1

Intensity = breakout_magnitude * continuation_strength * range_quality
```

#### 5. **Extreme Reversal**
```
Mathematical Definition:
IF |return(t)| > 2.5 * recent_volatility AND
   reversal ≥40% within 8 periods
THEN binary_label = 1

Intensity = extreme_magnitude * reversal_strength * reversal_speed
```

## 🤖 **ML-Based Pattern Discovery**

### **Implemented Methods:**

#### **1. LSTM Autoencoder Discovery**
- **Purpose**: Discover latent patterns in price sequences
- **Method**: Train LSTM autoencoder, analyze reconstruction errors and latent clusters
- **Expected Patterns**: Non-linear price relationships, complex sequences

#### **2. Matrix Profile Motif Discovery**  
- **Purpose**: Find exact recurring price movement subsequences
- **Method**: Calculate matrix profile, identify top motifs and discords
- **Expected Patterns**: Seasonal patterns, recurring shapes, rare behaviors

### **Implementation Requirements:**

#### **LSTM Discovery:**
```bash
pip install tensorflow>=2.8.0
# or
pip install torch>=1.12.0
```

#### **Matrix Profile Discovery:**
```bash
pip install stumpy
```

## 🚀 **Usage Examples**

### **Basic Pure Pattern Discovery:**
```python
from pure_price_patterns import PurePricePatternOrchestrator

# Discover core patterns
orchestrator = PurePricePatternOrchestrator()
results = orchestrator.discover_all_pure_patterns(price_series)

# Export binary labels
binary_targets = orchestrator.export_binary_labels(results)

# Export intensity gradients
intensity_targets = orchestrator.export_intensity_gradients(results)

# Combined targets
combined_targets = orchestrator.export_combined_targets(results)
```

### **Gradient Target Generation:**
```python
from pure_price_patterns import GradientPatternTargetGenerator

# Generate gradient-based targets
generator = GradientPatternTargetGenerator()
gradient_results = generator.generate_all_gradient_targets(price_series)

# Export ML-ready targets
ml_targets = generator.export_ml_ready_targets(gradient_results)

# Access different target formats
binary_only = ml_targets['binary_only']      # [0,1,0,1,...]
intensity_only = ml_targets['intensity_only']  # [0.0,0.8,0.2,0.9,...]
combined = ml_targets['combined']            # Both binary and intensity
```

### **LSTM Pattern Discovery:**
```python
from pure_price_patterns.lstm_discovery import LSTMPricePatternDiscovery

# Discover LSTM patterns (requires TensorFlow/PyTorch)
discoverer = LSTMPricePatternDiscovery(sequence_length=30)
lstm_patterns = discoverer.discover_lstm_patterns(price_series)

# Generate report
report = discoverer.generate_lstm_pattern_report(lstm_patterns)
```

### **Matrix Profile Discovery:**
```python
from pure_price_patterns.matrix_profile_discovery import MatrixProfileOrchestrator

# Discover matrix profile patterns (requires stumpy)
orchestrator = MatrixProfileOrchestrator()
mp_results = orchestrator.run_complete_matrix_profile_analysis(price_series)

# Export targets
mp_targets = orchestrator.export_matrix_profile_targets(mp_results)
```

### **Complete Analysis:**
```bash
# Run complete pure price pattern discovery
python run_pure_pattern_discovery.py --use_sample_data

# With your data
python run_pure_pattern_discovery.py --data_path /path/to/your/prices.csv

# Skip advanced methods for faster execution
python run_pure_pattern_discovery.py --use_sample_data --skip_advanced
```

## 📈 **ML Training Applications**

### **Classification (Binary Labels):**
```python
# Traditional binary classification
X = market_dimension_features
y = binary_pattern_labels  # [0,1,0,1,...]

classifier = RandomForestClassifier()
classifier.fit(X, y)

# Prediction: Will pattern occur?
pattern_probability = classifier.predict_proba(X_new)
```

### **Regression (Intensity Gradients):**
```python
# Enhanced regression approach
X = market_dimension_features  
y = intensity_gradients  # [0.0,0.8,0.2,0.9,...]

regressor = RandomForestRegressor()
regressor.fit(X, y)

# Prediction: How strong will pattern be?
pattern_intensity = regressor.predict(X_new)
```

### **Multi-Task Learning:**
```python
# Combined approach
X = market_dimension_features
y_binary = binary_labels
y_intensity = intensity_gradients

# Train both models
classifier = train_pattern_classifier(X, y_binary)
regressor = train_pattern_regressor(X, y_intensity)

# Enhanced predictions
will_occur = classifier.predict(X_new)
pattern_strength = regressor.predict(X_new)

# Trading decision
if will_occur and pattern_strength > 0.6:
    position_size = base_size * pattern_strength
    execute_trade(position_size)
```

## 🎯 **Integration with Market Dimension Analysis**

### **Research Question:**
> *"Which market dimensions (volatility, momentum, liquidity, etc.) predict which specific pure price action patterns?"*

### **Implementation:**
```python
# 1. Discover pure price patterns
pure_patterns = discover_all_pure_patterns(prices)

# 2. Generate ML targets
ml_targets = {
    'momentum_persistence': [0,1,0,1,0,...],
    'momentum_persistence_intensity': [0.0,0.8,0.0,0.9,0.0,...],
    'price_reversion': [1,0,0,1,1,...],
    'price_reversion_intensity': [0.7,0.0,0.0,0.6,0.8,...]
}

# 3. Test dimension predictive power
for pattern_name, pattern_target in ml_targets.items():
    for dimension_name, dimension_features in market_dimensions.items():
        
        # Test prediction accuracy
        accuracy = test_prediction_accuracy(dimension_features, pattern_target)
        
        print(f"{dimension_name} → {pattern_name}: {accuracy:.3f}")
```

## 🚀 **Key Advantages**

### **1. Pure Focus**
- Only price action matters
- No confounding factors
- Clear cause-effect relationships

### **2. Mathematical Precision**
- Exact pattern definitions
- Reproducible results
- No subjective interpretation

### **3. Enhanced ML Targets**
- Binary classification targets
- Continuous regression targets  
- Multi-task learning capabilities

### **4. Research Foundation**
- Clean testing of dimension relevance
- Economic significance validation
- Trading strategy development

This framework provides the mathematical precision and pure price action focus needed to scientifically determine which market dimensions are truly relevant for predicting specific price behaviors.