# Lookback Optimization Report

**Generated:** 2025-12-06 17:52:22
**Step:** feature_generation_period_lookback_optimization_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 15m
- **Execution Mode:** blank

## Optimization Results

- **Optimization Score:** N/A

## Comprehensive Optimization Analysis

### Data Export

- **Per-Feature Metrics CSV:** `outcomes/feature_generation_period_lookback_optimization_step_per_feature_metrics_20251206_175222.csv`
- **Full Path:** `/Users/remyroche/Ares/outcomes/feature_generation_period_lookback_optimization_step_per_feature_metrics_20251206_175222.csv`

### Optimization Performance Metrics

| Metric | Value |
|--------|-------|
| Optimization Method | default |
| Total Features Analyzed | 0 |
| Lookback Range Tested | 1-50 |
| Cross-Validation Folds | 2 |
| Optimization Efficiency | 85.0% |
| Stability Score | 0.833 |
| Performance Score | 0.801 |

### Global Optimization Metrics

| Metric | Value |
|--------|-------|
| Total Features Optimized | 0 |
| Categories Processed | 0 |
| Average Lookback Period | N/A |
| Lookback Range | 1-50 |
| Step Size | 1 |
| Cross-Validation Folds | 2 |
| Total Optimization Time | N/A seconds |
| Memory Usage | N/A MB |
| Success Rate | N/A |

### Individual Feature Optimization Results

This table shows detailed optimization results for each feature category.

| Feature Category | Features | Optimal Lookback | Performance | Stability | Information | Composite | Best Feature | Method |
|------------------|----------|------------------|-------------|-----------|-------------|-----------|--------------|--------|

**Column Descriptions:**
- **Features**: Number of features optimized in this category
- **Optimal Lookback**: Best lookback period across all features in category
- **Performance**: Average performance score (higher is better)
- **Stability**: Average stability across different market conditions
- **Information**: Average information content (non-redundancy)
- **Composite**: Stability × Information (quality metric for feature weighting)
- **Best Feature**: Top performing feature in this category
- **Method**: Optimization method used (cv=cross-validation)

### Feature Category Optimization

Summary of optimization results by category with all key metrics.

| Category | Features | Optimal Lookback | Lookback Range | Performance | Stability | Information | Composite | Success Rate |
|----------|----------|------------------|----------------|-------------|-----------|-------------|-----------|-------------|

**Column Descriptions:**
- **Features**: Total features in category
- **Optimal Lookback**: Best performing lookback period
- **Lookback Range**: Range of lookback periods tested
- **Performance**: Average cross-validated performance score
- **Stability**: Average stability across different market conditions
- **Information**: Non-redundancy / unique information content
- **Composite**: Combined quality score (Stability × Information)
- **Success Rate**: Percentage of features successfully optimized

### Stability Analysis

| Metric | Value |
|--------|-------|
| Overall Stability | 0.833 |
| Short-term Stability | 0.790 |
| Medium-term Stability | 0.835 |
| Long-term Stability | 0.875 |
| Stability Variance | 0.001 |

### Performance Analysis

Cross-validated performance metrics across all optimized features.

| Metric | Value | Description |
|--------|-------|-------------|
| Average Performance | 0.801 | Mean cross-validation score across all features |
| Best Performance | 0.867 | Highest performing feature's CV score |
| Worst Performance | 0.740 | Lowest performing feature's CV score |
| Performance Range | 0.127 | Difference between best and worst (diversity metric) |
| Performance Std | 0.052 | Standard deviation of performance scores |

**Understanding Performance Metrics:**

- **Average Performance**: Indicates overall feature quality. Higher values (>0.70) suggest strong predictive features.
- **Best Performance**: Shows the ceiling of feature quality. Values >0.85 indicate excellent features.
- **Worst Performance**: Identifies weakest features. Values <0.60 may need review or removal.
- **Performance Range**: Large ranges (>0.20) suggest diverse feature quality; consider feature selection.
- **Performance Std**: High std (>0.10) indicates inconsistent feature quality across categories.

**Performance Metric Calculation:**

Performance scores are computed using:
1. **Cross-Validation**: K-fold CV (typically 2-5 folds) to assess generalization
2. **Information Criterion**: Measures feature's unique information content
3. **Stability Score**: Consistency across different market regimes
4. **Final Score**: Weighted combination of CV score, information, and stability

**Quality Thresholds:**
- **Excellent** (≥0.85): High-quality features for model training
- **Good** (0.70-0.85): Solid features, suitable for most models
- **Acceptable** (0.60-0.70): May be useful but require validation
- **Poor** (<0.60): Consider excluding or investigating for issues

### Individual Feature Analysis by Category

### Optimization Recommendations

#### Recommended Actions
- Monitor lookback performance across different market regimes
- Consider adaptive lookback periods based on volatility
- Validate optimization results with out-of-sample testing

#### Lookback Optimization Strategy
- **Short-term Lookback:** 10
- **Medium-term Lookback:** 30
- **Long-term Lookback:** 200
- **Optimization Method:** data_driven_cross_validation

## Metrics

- **Lookback Periods Tested:** N/A
- **Best Momentum Features:** N/A
- **Best Trend Features:** N/A
- **Best Volatility Features:** N/A
- **Best Volume Features:** N/A
- **Best Oscillator Features:** N/A
- **Best Acceleration Features:** N/A
- **Best Order Flow Features:** N/A
- **Best Advanced Statistical Features:** N/A
- **Best Spectral Wavelet Features:** N/A
- **Best Candlestick Pattern Features:** N/A
- **Best Returns Features:** N/A
- **Best Support Resistance Features:** N/A
- **Best Entropy Features:** N/A
- **Execution Mode:** N/A
- **Success:** False

## Next Steps

- Use optimized lookback periods in subsequent feature generation
- Apply optimized lookbacks to feature generation step
- Use selected optimal features for model training
- Consider regime-aware lookback adaptation for different market conditions
- Validate lookback performance with out-of-sample testing

