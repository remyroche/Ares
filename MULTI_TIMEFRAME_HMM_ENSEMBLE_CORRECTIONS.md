# Multi-Timeframe HMM Ensemble System - Corrected Understanding

## ✅ **Corrected Understanding**

### 1. **Hazard Models are for Regime Transitions Only**
- **NOT for price predictions** - Hazard models predict whether a regime will transition to a different regime
- **Purpose**: Predict regime persistence vs. regime change
- **Target**: Binary classification (REGIME_CONTINUE vs. REGIME_CHANGE)
- **Location**: `src/training/steps/step1_8_regime_forecasting.py`

### 2. **Price Direction Predictions (BUY/SELL/HOLD) are Made Elsewhere**

#### **Primary Location**: `src/interfaces/base_interfaces.py`
```python
@dataclass
class AnalysisResult:
    signal: str  # 'BUY', 'SELL', 'HOLD'
```

#### **Global Meta-Learner**: `src/analyst/predictive_ensembles/ensemble_orchestrator.py`
```python
# Encode target labels (BUY, SELL, HOLD) to integers
self.global_meta_label_encoder = LabelEncoder()
y_encoded = self.global_meta_label_encoder.fit_transform(y_meta)
```

#### **Triple Barrier Labeling**: `src/training/steps/step4_analyst_labeling_feature_engineering_components/`
- Creates BUY/SELL/HOLD labels based on price barriers
- Used for training price direction prediction models

### 3. **Multi-Timeframe HMM Ensemble Focus**

#### **Primary Method**: Meta-Learner/Stacking (NOT Weighted Average)
- **Stacking Ensemble**: Advanced method with sophisticated feature engineering
- **Meta-Learner**: Primary method for combining timeframe predictions
- **Weighted Average**: Fallback method only

#### **Timeframes**: 1m, 5m, 15m, 30m (NOT 1h, 4h)
- **1m**: Lower weight (0.20) due to noise
- **5m**: Good balance (0.25)
- **15m**: Higher weight (0.30) for medium-term trends
- **30m**: Good for longer-term regime changes (0.25)

## 🔧 **Implementation Details**

### **Stacking Ensemble Features**
1. **Raw predictions** from each timeframe
2. **Cross-timeframe interactions** (multiplication, differences)
3. **Statistical features** across timeframes (mean, std, max, min, range)

### **Meta-Learner Configuration**
```python
"meta_learner": {
    "type": "lgbm",
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 6,
    "random_state": 42,
    "verbose": -1
}
```

### **Prediction Output**
- **REGIME_CHANGE**: Regime will transition in next period
- **REGIME_CONTINUE**: Regime will persist
- **Confidence**: Probability of regime change

## 📊 **Expected MAPE Reduction**

### **Current Single-Timeframe MAPE**: 63.2% (Regime 8)
### **Expected Multi-Timeframe MAPE**: 35-45%

**Improvement Factors**:
1. **Cross-timeframe signals**: Regime transitions have precursors across timeframes
2. **Feature diversity**: 4x more features (4 timeframes × ~86 features each)
3. **Meta-learning**: Sophisticated combination of predictions
4. **Dynamic weighting**: Adaptive weights based on performance

## 🎯 **Integration with Live Trading**

### **Current Live Trading System**
- ✅ **Combines LGBM models on multiple timeframes**
- ✅ **Uses HMM-derived cluster ensembles** (not old regime ensembles)
- ✅ **LGBM as the "general" model** on top of HMM clusters

### **New Multi-Timeframe HMM Ensemble**
- 🔄 **Enhances regime forecasting accuracy**
- 🔄 **Reduces MAPE through cross-timeframe learning**
- 🔄 **Provides better regime transition predictions**

## 📁 **Files Created/Modified**

1. **`src/training/steps/multi_timeframe_hmm_ensemble.py`** - Main ensemble system
2. **`src/config/multi_timeframe_hmm_ensemble_config.py`** - Configuration
3. **`scripts/train_multi_timeframe_hmm_ensemble.py`** - Training script
4. **`MULTI_TIMEFRAME_HMM_ENSEMBLE_CORRECTIONS.md`** - This summary

## 🚀 **Next Steps**

1. **Train the ensemble** using the provided script
2. **Validate MAPE reduction** on test data
3. **Integrate with live trading** pipeline
4. **Monitor performance** and adjust weights dynamically
