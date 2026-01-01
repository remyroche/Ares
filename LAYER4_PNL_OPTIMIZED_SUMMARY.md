# Layer4 PnL-Optimized ExtraTrees Implementation

## 🎯 **Layer4 Redesign Complete**

### **New Layer4 Architecture:**
Layer4 now uses an **ExtraTrees classifier trained on returns** with comprehensive features designed to **maximize PnL and Sortino while minimizing drawdown**.

---

## 📊 **Feature Set for Layer4 ExtraTrees**

### **1. OOF Predictions from Layer3**
- `meta_prob` - Primary Layer3 probability
- `meta_prob_*` - All Layer3 geometry probabilities
- **Purpose**: Base signal strength from Layer3 ensemble

### **2. Disagreement Features** (from `ensemble_disagreement.py`)
- `prediction_dispersion` - Variance of predictions across models
- `confidence_gap` - Margin between top predictions  
- `uncertainty` - Normalized entropy measure
- `prediction_range` - Range of predictions (max - min)
- `avg_divergence` - Average pairwise model divergence
- `max_confidence` - Highest confidence among models
- `disagreement_rate` - Proportion of models disagreeing on direction
- `snr_internal` - Mean Probability / Mean Internal Variance
- `snr_consensus` - Ensemble Mean Probability / StdDev of Model Predictions
- **Purpose**: Capture model uncertainty and ensemble disagreement

### **3. Average of Heads ProbA * ProbB**
- `avg_prob_product` - Average pairwise probability products
- **Purpose**: Capture consensus strength between Layer3 models

### **4. Past Precision**
- `past_precision` - Rolling accuracy of primary model in similar market conditions
- **Window**: 50 periods (configurable)
- **Purpose**: Historical performance in current market regime

### **5. Structural Break Scores**
- `sadf_score_norm` - SADF (Supremum Augmented Dickey-Fuller) for bubble detection
- `cusum_score_norm` - CUSUM filter for change point detection
- **Purpose**: Identify if signal occurred during bubble/crash regimes

### **6. Relative Strength**
- `vwap_distance` - Distance from Volume Weighted Average Price
- `vwap_ratio` - Price/VWAP ratio
- `relative_strength_ma` - Performance vs 20-period moving average
- `relative_strength_short` - Performance vs 10-period moving average
- **Purpose**: Sector-relative performance and benchmark distance

### **7. Drawdown State**
- `drawdown_from_peak` - Current drawdown from rolling peak
- `distance_from_trough` - Recovery potential from trough
- `is_near_peak` - Binary: within 10% of rolling peak
- `is_near_trough` - Binary: within 10% of rolling trough
- `drawdown_regime_severe` - >10% drawdown
- `drawdown_regime_moderate` - 5-10% drawdown  
- `drawdown_regime_mild` - 2-5% drawdown
- `drawdown_regime_none` - <2% drawdown
- **Purpose**: Current drawdown state and regime classification

### **8. Market Features**
- `volatility_zscore` - Volatility relative to recent history
- `volatility_regime` - High vs low volatility regime
- `hour_of_day`, `day_of_week` - Time-based features
- `is_session_start`, `is_session_end` - Session timing features

---

## 🎯 **Training Objective**

### **Target Variable:**
- **Binary classification**: Positive vs negative returns
- **Configurable**: Raw returns vs denoised returns
- **Sample weights**: Optional (absolute returns for emphasis on larger moves)

### **Optimization Goal:**
- **Primary**: Maximize Sortino ratio (downside deviation only)
- **Secondary**: Maximize PnL, minimize drawdown
- **Tertiary**: Standard AUC/Log Loss metrics

### **Hyperparameter Optimization:**
- **Framework**: Optuna with TPE sampler
- **Objective**: Sortino ratio from cross-validation
- **Strategy**: Long when prob > 0.6, Short when prob < 0.4
- **Time Series CV**: Prevents lookahead bias

---

## 📈 **Performance Metrics**

### **PnL-Focused Metrics:**
- **Total PnL** - Sum of all returns
- **Sharpe Ratio** - Risk-adjusted returns (all volatility)
- **Sortino Ratio** - Risk-adjusted returns (downside volatility only)
- **Maximum Drawdown** - Peak-to-trough decline
- **Win Rate** - Percentage of profitable trades

### **Classification Metrics:**
- **AUC** - Area under ROC curve
- **Log Loss** - Probabilistic calibration
- **Brier Score** - Probability accuracy

---

## 🔧 **Configuration Options**

```python
config = {
    'use_raw_returns': True,        # Use raw vs denoised returns
    'use_weights': True,             # Use sample weights in training
    'n_trials': 50,                 # Hyperparameter optimization trials
    'past_precision_window': 50,    # Rolling accuracy window
    'relative_strength_window': 20,  # VWAP/MA window
    'drawdown_window': 50           # Drawdown calculation window
}
```

---

## 🚀 **Integration Points**

### **Layer3 Integration:**
- Consumes all Layer3 OOF predictions
- Uses ensemble disagreement features
- Leverages Layer3 probability consensus

### **Layer5 Integration:**
- Generates `layer4_prob` for Layer5 compatibility
- Maintains existing probability proxy interface
- Seamless integration with position sizing pipeline

### **Pipeline Compatibility:**
- **Backward compatible** with existing `train_layer4_oof` interface
- **Legacy parameters** maintained for compatibility
- **Same output format** for downstream layers

---

## 🎯 **Key Advantages**

### **vs. Previous Triple Barrier:**
1. **Data-driven** vs rule-based approach
2. **Comprehensive features** vs simple volatility sizing
3. **PnL optimization** vs fixed risk management
4. **Adaptive thresholds** vs static parameters

### **vs. Standard ExtraTrees:**
1. **Sortino optimization** vs AUC optimization
2. **Financial features** vs generic ML features
3. **Time series CV** vs random CV
4. **PnL-based evaluation** vs classification metrics

---

## 📊 **Expected Outcomes**

### **Performance Goals:**
- **Higher Sortino ratio** through downside risk focus
- **Reduced drawdown** through drawdown state features
- **Improved PnL** through structural break detection
- **Better regime adaptation** through relative strength features

### **Robustness Goals:**
- **Market condition adaptation** through past precision
- **Bubble/crash awareness** through structural breaks
- **Ensemble uncertainty capture** through disagreement features
- **Drawdown-aware positioning** through drawdown state

---

## 🔄 **Usage Example**

```python
# Train Layer4 with PnL optimization
predictions_df, metadata = train_layer4_oof(
    oof_df=layer3_predictions,
    market_data=market_data,
    l3_prob_col='meta_prob',
    target_col='realized_return',
    config={
        'use_raw_returns': True,
        'use_weights': True,
        'n_trials': 50
    }
)

# Results include:
# - layer4_prob: Probability for Layer5
# - layer4_extratrees_prob: Raw ExtraTrees probability  
# - layer4_extratrees_confidence: Model confidence
# - Comprehensive PnL and Sortino metrics
```

---

## ✅ **Implementation Status**

- ✅ **Feature engineering** complete
- ✅ **Structural break detection** implemented
- ✅ **Disagreement features** integrated
- ✅ **PnL optimization** objective implemented
- ✅ **Hyperparameter optimization** configured
- ✅ **Backward compatibility** maintained
- ✅ **Layer5 integration** preserved

The new Layer4 is ready for testing and should provide significant improvements in PnL and Sortino while maintaining robust drawdown control.
