# Analyst Interaction Generation Report

## Executive Summary
- **Symbol**: ETHUSDT
- **Exchange**: binance
- **Timeframe**: 15m
- **Execution Mode**: light
- **Generated**: 2025-12-04 19:45:12
- **Total Features Generated**: 200
- **Processing Efficiency**: 0.0 features/second
- **Overall Status**: ✅ SUCCESS

## Feature Statistics
- **Total Features**: 200
- **Base Features**: 199
- **Interaction Features**: 1
- **Feature Density**: 18.2 features per category
- **Interaction Ratio**: 0.5%

## Category Coverage Analysis
- 🟢 **Trend**: 47 features (Excellent)
- 🟢 **Oscillator**: 64 features (Excellent)
- 🟢 **Momentum**: 21 features (Excellent)
- 🟢 **Returns**: 48 features (Excellent)
- 🟢 **Volatility**: 20 features (Excellent)
- 🔴 **Volume**: 0 features (Poor)
- 🔴 **Acceleration**: 0 features (Poor)
- 🔴 **Entropy**: 0 features (Poor)
- 🔴 **Spectral_Wavelet**: 0 features (Poor)
- 🔴 **Support_Resistance**: 0 features (Poor)
- 🔴 **Unknown**: 0 features (Poor)

### Category Health Summary
- **Total Categories**: 11
- **Balanced Categories**: 1 (9.1%)
- **Imbalanced Categories**: 10 (90.9%)
- **Category Diversity Score**: 0.09/1.0

## Performance Metrics & Efficiency
- **Phase 0 Time**: 1.31s
- **Phase 1 Time**: 2.16s
- **Phase 2 Time**: 20.69s
- **Phase 3.1 Time**: 0.00s
- **Phase 3.2 Time**: 0.00s
- **Phase 3.3 Time**: 0.00s
- **Phase 4 Time**: 0.00s
- **Total Time**: 0.00s
- **Processing Efficiency**: 0.0 features/second
- **Memory Usage**: 0.0 MB
- **CPU Utilization**: 0.0%

## Feature Engineering Pipeline Details

## Variant Generation Statistics

## Pruning Statistics & Quality Control

## LGBM Feature Selection (Gain + Permutation)
- **Method**: FeatureSelectionPipeline
- **Gain top-K**: 200
- **Permutation top-K**: 100
- **Top 10 features by gain:**
  - `volume_ema_10_volnorm`: 0.0000
  - `volume_ema_5_volnorm`: 0.0000
  - `vectorbt_volatility_comprehensive_50_volnorm`: 0.0000
  - `vectorbt_momentum_comprehensive_21_volnorm`: 0.0000
  - `vectorbt_trend_consistency_50_price_returns_volnorm`: 0.0000
  - `cumulative_returns_20_price_returns_volnorm`: 0.0000
  - `trend_score_14_volnorm`: 0.0000
  - `order_flow_imbalance_20_volnorm`: 0.0000
  - `mama_21_0.05_price_returns_volnorm`: 0.0000
  - `macd_delta_12_26_9_volnorm`: 0.0000
- **Top 10 features by permutation importance:**
  - `volume_ema_10_volnorm`: 0.0000
  - `volume_ema_5_volnorm`: 0.0000
  - `vectorbt_volatility_comprehensive_50_volnorm`: 0.0000
  - `vectorbt_momentum_comprehensive_21_volnorm`: 0.0000
  - `vectorbt_trend_consistency_50_price_returns_volnorm`: 0.0000
  - `cumulative_returns_20_price_returns_volnorm`: 0.0000
  - `trend_score_14_volnorm`: 0.0000
  - `order_flow_imbalance_20_volnorm`: 0.0000
  - `mama_21_0.05_price_returns_volnorm`: 0.0000
  - `macd_delta_12_26_9_volnorm`: 0.0000

## Interaction Discovery & Analysis
- **Feature Pairs Analyzed**: 0
- **Total Interactions Generated**: 0
- **Operations Per Pair**: 0 (x, div, minus, log, log_ratio)
- **Max Candidates Processed**: 0
- **Selection Method**: Early stopping
- **Valid Interactions Found**: 0
- **Interaction Success Rate**: 0.0%

## Model Performance & Validation
- **LGBM Training**: ✅ Successful
- **Tree Analysis**: ❌ Failed
- **Interaction Generation**: ✅ Successful
- **Model Accuracy**: 0.0000
- **Cross-Validation Score**: 0.0000
- **Feature Importance Consistency**: 0.00/1.0
- **Mean Feature-Target MI**: 0.000000

### Feature-Target Mutual Information by Target
- *No MI scores available*


## Technical Implementation Details
- **Adaptive Feature Selection**: 3-6 features per category based on signal strength
- **RobustScaler Bounding**: Prevents extreme values with percentile clipping
- **Category Protection**: Maintains ≥3 features per category during pruning
- **Corrected SHAP Analysis**: Standard SHAP values for interaction features
- **Causality Enforcement**: All features shifted to prevent lookahead bias
- **Memory Optimization**: VectorBT integration for efficient computation
- **Parallel Processing**: Multi-threaded variant generation
- **Error Handling**: Comprehensive exception management with fallbacks

## Actionable Insights & Recommendations

### Immediate Actions
1. **Category Balance**: ⚠️ Address 10 imbalanced categories
2. **Feature Quality**: ✅ High quality
3. **Processing Efficiency**: ⚠️ Consider optimization

### Strategic Recommendations
1. **Model Selection**: Prioritize features with highest SHAP importance scores
2. **Feature Engineering**: Focus on categories with <5 features for expansion
3. **Performance Monitoring**: Track feature stability across different market conditions
4. **Resource Optimization**: Consider parallel processing

### Next Steps
1. **Validation**: Test features on out-of-sample data
2. **Integration**: Incorporate into downstream model training pipeline
3. **Monitoring**: Set up feature drift detection
4. **Iteration**: Refine feature selection based on model performance

## Risk Assessment & Overfitting Prevention
- **Overfitting Risk**: High
- **Prevention Measures Applied**:
  - ✅ L1/L2 Regularization (reg_alpha=0.1, reg_lambda=0.1)
  - ✅ Time-series Cross-Validation with Gap (5-fold, gap=1)
  - ✅ Early Stopping (patience=20 rounds)
  - ✅ Feature Subsampling (subsample=0.8, colsample_bytree=0.8)
  - ✅ Increased min_child_samples (50-75)
  - ✅ Reduced Model Complexity (max_depth=4-5, num_leaves=15-20)
- **Computational Risk**: High
- **Data Quality Risk**: Low
- **Data Leakage Prevention**: Gap-based time-series splits implemented

---
*Comprehensive report generated by Analyst Interaction Generation Step v2.0*  
*Generated on 2025-12-04 19:45:12*
