# Analyst-Labeler Alignment for Feature Relevance Analysis

## 🎯 **Overview**

This document explains how all feature relevance methods in our framework are aligned with analyst-labeler approaches for predicting price action and market movements. Every method evaluates the usefulness of features to predict specific price actions, making them directly applicable to analyst workflows.

## 📊 **Target Variable Alignment**

### **All Methods Predict Price Action**

Every feature relevance method in our framework evaluates features against **price action targets**:

1. **Mutual Information (MI)** → `MI(feature, price_action)`
2. **LASSO** → `price_action = β₀ + β₁×feature₁ + β₂×feature₂ + ...`
3. **LGBM/SHAP** → `price_action = f(feature₁, feature₂, ...)`
4. **Permutation Importance** → `Δ_performance_when_shuffling(feature)`

### **Price Action Target Types**

Our framework supports multiple analyst-style price action targets:

#### **1. Directional Prediction**
```python
# Binary classification: up/down/sideways
price_direction = {
    'up': returns > threshold,
    'down': returns < -threshold,
    'sideways': |returns| <= threshold
}
```

#### **2. Magnitude Prediction**
```python
# Multi-class classification: movement size
price_magnitude = {
    'large_up': returns > 1%,
    'small_up': 0.2% < returns <= 1%,
    'sideways': -0.2% <= returns <= 0.2%,
    'small_down': -1% <= returns < -0.2%,
    'large_down': returns < -1%
}
```

#### **3. Regime Prediction**
```python
# Market regime classification
volatility_regime = {
    'high_vol': rolling_volatility > threshold,
    'low_vol': rolling_volatility <= threshold
}
```

## 🔍 **Method-Specific Price Action Alignment**

### **1. Mutual Information (MI)**

**What it measures:** How much information each feature provides about price movements

**Analyst interpretation:**
- **High MI** = Feature is highly informative about price direction/magnitude
- **Low MI** = Feature provides little information about price movements
- **Non-linear relationships** = Captures complex price patterns that correlation misses

**Example:**
```python
# MI between VWAP basis and price direction
mi_score = MI(vwap_basis_w20, price_direction)
# High score = VWAP basis is very informative about price direction
```

### **2. LASSO Regression**

**What it measures:** Linear relationship between features and price movements

**Analyst interpretation:**
- **Non-zero coefficients** = Feature has linear predictive power for price movements
- **Coefficient magnitude** = Strength of linear relationship
- **Regularization path** = Order of feature importance for price prediction

**Example:**
```python
# LASSO model for price direction prediction
price_direction = β₀ + β₁×vwap_basis + β₂×volatility + β₃×momentum + ...
# β₁ > 0 means VWAP basis positively predicts price direction
```

### **3. LGBM/SHAP**

**What it measures:** Non-linear feature contributions to price movement predictions

**Analyst interpretation:**
- **Feature importance** = Overall contribution to price prediction accuracy
- **SHAP values** = Individual prediction contributions
- **Interaction effects** = How features work together to predict price movements

**Example:**
```python
# LGBM model for price magnitude prediction
model = LGBMClassifier()
model.fit(features, price_magnitude)
# feature_importance shows which features best predict price magnitude
```

### **4. Permutation Importance**

**What it measures:** Performance drop when feature information is removed

**Analyst interpretation:**
- **High importance** = Feature is crucial for price prediction
- **Low importance** = Feature can be removed without affecting price prediction
- **Stable importance** = Feature consistently helps predict price movements

**Example:**
```python
# Permutation importance for price direction prediction
perm_importance = permutation_importance(model, X, price_direction)
# High score = feature is essential for predicting price direction
```

## 📈 **Analyst Workflow Integration**

### **Step 1: Define Price Action Targets**
```python
from feature_comparison.analyst_labeler_integration import AnalystLabelerIntegration

# Initialize analyst-labeler integration
analyst = AnalystLabelerIntegration(
    price_threshold=0.002,  # 0.2% for significant moves
    volatility_threshold=0.02,  # 2% volatility threshold
    lookforward_periods=1  # 1-period prediction
)

# Create price action targets
targets = analyst.create_analyst_style_targets(data)
# Returns: {'price_direction', 'price_magnitude', 'volatility_regime', ...}
```

### **Step 2: Evaluate Feature Relevance**
```python
# Evaluate all methods against price action targets
results = analyst.evaluate_feature_relevance_for_targets(
    features, targets, 
    methods=['lgbm', 'lasso', 'mi', 'permutation']
)

# Each method now evaluates features for price prediction
```

### **Step 3: Analyst-Style Reporting**
```python
# Generate analyst-friendly report
report = analyst.create_analyst_style_report(results)

# Shows:
# - Which features best predict price direction
# - Which features best predict price magnitude  
# - Which features best predict market regimes
# - Method agreement on price prediction
```

## 🎯 **Price Action Prediction Scenarios**

### **Scenario 1: Intraday Trading**
**Target:** Next 1-hour price direction
**Features:** VWAP basis, momentum, volatility, volume
**Methods:** All methods evaluate how well features predict next-hour price direction

### **Scenario 2: Swing Trading**
**Target:** Next 1-day price magnitude
**Features:** Technical indicators, regime features, momentum
**Methods:** All methods evaluate how well features predict next-day price magnitude

### **Scenario 3: Risk Management**
**Target:** High volatility regime
**Features:** Volatility features, drawdown metrics, regime indicators
**Methods:** All methods evaluate how well features predict volatility regimes

## 📊 **Feature Categories for Price Prediction**

### **Price-Based Features**
- `ret_t1` - 1-period returns (target variable)
- `ret_ma_wW` - Moving averages of returns
- `ret_mom_kK` - Momentum features
- `ret_acc_k1` - Acceleration features

### **VWAP-Based Features**
- `vwap_basis_wW` - VWAP basis (price - VWAP)
- `rel_vwap_dev_wW` - Relative VWAP deviation
- `vwap_ret_wW` - VWAP returns

### **Volatility Features**
- `vol_wW` - Rolling volatility
- `vol_wW1_std_wW2` - Volatility of volatility
- `regime_highvol` - High volatility regime indicator

### **Volume Features**
- `vol_ret_t1` - Volume returns
- `vol_adv_wW` - Volume/ADV ratio
- `vw_ret_wW` - Volume-weighted returns

## 🔄 **Method Comparison for Price Prediction**

| Method | Price Action Focus | Strengths | Use Case |
|--------|-------------------|-----------|----------|
| **MI** | Information content about price movements | Non-linear relationships | Feature discovery |
| **LASSO** | Linear price prediction | Interpretable coefficients | Linear models |
| **LGBM** | Non-linear price prediction | High accuracy, interactions | Complex patterns |
| **SHAP** | Individual price prediction contributions | Explainable AI | Model interpretation |
| **Permutation** | Price prediction importance | Model-agnostic | Feature selection |

## 🎯 **Analyst Benefits**

### **1. Direct Price Prediction Focus**
- All methods evaluate features for price movement prediction
- No need to translate from abstract "relevance" to price prediction
- Direct alignment with trading and investment decisions

### **2. Multiple Price Action Perspectives**
- Directional prediction (up/down/sideways)
- Magnitude prediction (small/large moves)
- Regime prediction (high/low volatility)
- Custom analyst-defined targets

### **3. Method Validation**
- Multiple methods validate feature importance for price prediction
- Rank consistency ensures reliable price prediction features
- Bootstrap stability ensures robust price prediction features

### **4. Time-Series Safe**
- Purged CV prevents lookahead bias in price prediction
- Walk-forward validation mirrors trading deployment
- Out-of-sample testing ensures price prediction generalization

## 📋 **Usage Example**

```python
from feature_comparison.analyst_labeler_integration import AnalystLabelerIntegration

# Initialize for price prediction
analyst = AnalystLabelerIntegration(
    price_threshold=0.001,  # 0.1% significant move
    lookforward_periods=1   # Predict next period
)

# Create price action targets
targets = analyst.create_analyst_style_targets(market_data)

# Evaluate features for price prediction
results = analyst.evaluate_feature_relevance_for_targets(
    features, targets, 
    methods=['lgbm', 'lasso', 'mi', 'permutation']
)

# Get analyst-friendly report
report = analyst.create_analyst_style_report(results)
analyst.print_analyst_style_summary(report)

# Shows which features best predict:
# - Price direction (up/down/sideways)
# - Price magnitude (small/large moves)  
# - Market regimes (high/low volatility)
```

## ✅ **Key Alignment Points**

1. **All methods predict price action** - No abstract "relevance", only price prediction
2. **Multiple price action targets** - Direction, magnitude, regime prediction
3. **Analyst-friendly reporting** - Direct translation to trading decisions
4. **Time-series safe** - Prevents lookahead bias in price prediction
5. **Method validation** - Multiple methods validate price prediction features
6. **Production ready** - Walk-forward validation mirrors trading deployment

The framework is now fully aligned with analyst-labeler approaches, where every feature relevance method directly evaluates how well features predict price action and market movements.