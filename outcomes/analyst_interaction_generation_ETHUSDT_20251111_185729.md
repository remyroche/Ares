# Analyst Interaction Generation Report

## Executive Summary
- **Symbol**: ETHUSDT
- **Exchange**: binance
- **Timeframe**: 15m
- **Execution Mode**: blank
- **Generated**: 2025-11-11 18:57:29
- **Total Features Generated**: 160
- **Processing Efficiency**: 0.0 features/second
- **Overall Status**: ✅ SUCCESS

## Feature Statistics
- **Total Features**: 160
- **Base Features**: 80
- **Interaction Features**: 80
- **Feature Density**: 14.5 features per category
- **Interaction Ratio**: 50.0%

## Category Coverage Analysis
- 🟢 **Trend**: 14 features (Excellent)
- 🟢 **Oscillator**: 19 features (Excellent)
- 🔴 **Momentum**: 0 features (Poor)
- 🔴 **Returns**: 1 features (Poor)
- 🟢 **Volatility**: 50 features (Excellent)
- 🟢 **Volume**: 32 features (Excellent)
- 🔴 **Acceleration**: 0 features (Poor)
- 🔴 **Advanced_Statistical**: 0 features (Poor)
- 🟢 **Candlestick_Pattern**: 28 features (Excellent)
- 🟠 **Entropy**: 4 features (Adequate)
- 🟠 **Spectral_Wavelet**: 4 features (Adequate)

### Category Health Summary
- **Total Categories**: 11
- **Balanced Categories**: 4 (36.4%)
- **Imbalanced Categories**: 7 (63.6%)
- **Category Diversity Score**: 0.36/1.0

## Performance Metrics & Efficiency
- **Phase 0 Time**: 11.30s
- **Phase 1 Time**: 26.97s
- **Phase 2 Time**: 58.38s
- **Phase 3.1 Time**: 36.11s
- **Phase 3.2 Time**: 29.81s
- **Phase 3.3 Time**: 1222.52s
- **Phase 4 Time**: 0.00s
- **Total Time**: 0.00s
- **Processing Efficiency**: 0.0 features/second
- **Memory Usage**: 0.0 MB
- **CPU Utilization**: 0.0%

## Feature Engineering Pipeline Details

## Variant Generation Statistics

## Pruning Statistics & Quality Control
- **Correlation_Clustering**: Removed 2 features
- **Total Features Removed**: 2
- **Pruning Efficiency**: 1.2%

## Interaction Discovery & Analysis
- **Feature Pairs Analyzed**: 80
- **Total Interactions Generated**: 80
- **Operations Per Pair**: 5 (x, div, minus, log, log_ratio)
- **Max Candidates Processed**: 400
- **Selection Method**: MI-based selection
- **Valid Interactions Found**: 80
- **Interaction Success Rate**: 20.0%
- **Top Interaction Examples**:
  1. `candlestick_piercing_line_pattern_base_9x_ratio_log_ratio_candlestick_piercing_line_pattern_base_3x_ratio` (Score: 0.7380)
  2. `candlestick_harami_cross_pattern_base_27x_ratio_minus_candlestick_piercing_line_pattern_base_9x_ratio` (Score: 0.7249)
  3. `fibonacci_0.618_10_price_returns_base_x_vectorbt_smoothed_obv_10_base_3x_ratio` (Score: 0.6303)
  4. `candlestick_dark_cloud_cover_pattern_base_6x_ratio_minus_fibonacci_0.786_10_price_returns_base_9x_ratio` (Score: 0.5750)
  5. `fibonacci_0.618_10_price_returns_base_27x_ratio_log_volume_price_trend_base_3x_ratio` (Score: 0.5392)

## Model Performance & Validation
- **LGBM Training**: ✅ Successful
- **Tree Analysis**: ✅ Successful
- **Interaction Generation**: ✅ Successful
- **Model Accuracy**: 0.0881
- **Cross-Validation Score**: 0.0000
- **Feature Importance Consistency**: 0.34/1.0

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
1. **Category Balance**: ⚠️ Address 7 imbalanced categories
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
- **Overfitting Risk**: Medium
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
*Generated on 2025-11-11 18:57:29*
