# Enhanced Data & Labels System - "Define what truth means"

This system implements a comprehensive data and labels management solution that addresses the core challenges in trading ML by defining what truth means, cleaning inputs, and ensuring stability over time.

## 🎯 Key Features

### 1. Trading-Aware Label Definitions
- **Analyst Labels**: "Should we trade?" (1 if expected PnL > fees + slippage)
- **Tactician Labels**: Direction/magnitude based on max favorable/adverse excursion
- **Regime Conditioning**: Volatility-scaled thresholds that adapt to market conditions
- **Risk Awareness**: Label 0 if trade would hit stop before target

### 2. Comprehensive Data Cleaning
- Remove bars with missing/outlier prices/volumes
- Align timestamps across timeframes to avoid mis-synchronization
- De-duplicate overlapping samples from sliding windows
- Check target shift: verify label distribution doesn't drift due to regime imbalance

### 3. Label Stability Monitoring
- Recompute labels after every data refresh (don't cache old ones)
- Track label leakage indicators (autocorrelation between label and near-past features)
- Check OOS label balance similarity to train (apply reweighting when needed)
- Detect and handle concept drift

### 4. Full Infrastructure Integration
- Seamless integration with existing volatility-aware labeler
- Native support for regime detection and feature engineering
- Compatible with all existing training pipelines
- Enhanced monitoring and validation capabilities

## 🚀 Quick Start

### Basic Usage

```python
from enhanced_data_labels_system import EnhancedDataLabelsSystem, create_trading_optimized_config

# Create enhanced system
config = create_trading_optimized_config()
enhanced_system = EnhancedDataLabelsSystem(config)

# Process market data
result = enhanced_system.process_market_data(market_data)

# Access results
labels = result['labels']
confidence_scores = result['confidence_scores']
data_quality = result['data_quality']
label_stability = result['label_stability']
```

### Integration Usage

```python
from infrastructure_integration import process_market_data_enhanced

# Process with full integration pipeline
result = process_market_data_enhanced(
    market_data=market_data,
    force_regime_detection=True,
    force_feature_engineering=True
)

# Access integrated results
processed_data = result['processed_data']
labels = result['labels']
regime_data = result['regime_data']
engineered_features = result['engineered_features']
```

### Validation

```python
from enhanced_labels_validation import run_enhanced_labels_validation

# Run comprehensive validation
validation_result = run_enhanced_labels_validation()

# Check validation status
overall_score = validation_result['overall_score']
overall_status = validation_result['overall_status']
```

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Enhanced Data & Labels System                │
├─────────────────────────────────────────────────────────────┤
│  Input: Market Data (OHLCV)                                │
│  ↓                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Data Quality    │  │ Regime Detection│  │ Feature     │ │
│  │ Assessment &    │  │ & Conditioning  │  │ Engineering │ │
│  │ Cleaning        │  │                 │  │             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│  ↓                                                         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           Trading-Aware Label Generation               │ │
│  │  • Analyst: "Should we trade?" (PnL > costs)          │ │
│  │  • Tactician: Direction/magnitude (excursion-based)   │ │
│  │  • Regime conditioning (volatility-scaled)            │ │
│  │  • Risk awareness (stop-loss protection)              │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ↓                                                         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           Label Stability Monitoring                   │ │
│  │  • Leakage detection (autocorrelation)                 │ │
│  │  • Drift detection (distribution changes)              │ │
│  │  • OOS balance checking                                │ │
│  │  • Recomputation on data refresh                       │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ↓                                                         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           Balancing & Weighting                        │ │
│  │  • Class balancing (under/over sampling)               │ │
│  │  • Sample weighting (volatility, confidence, time)     │ │
│  │  • Regime-aware rebalancing                            │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ↓                                                         │
│  Output: Enhanced Labels + Quality Metrics + Stability     │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Configuration

### Trading-Optimized Configuration

```python
from enhanced_data_labels_system import create_trading_optimized_config

config = create_trading_optimized_config()
# Optimized for production trading with strict quality requirements
```

### Research-Optimized Configuration

```python
from enhanced_data_labels_system import create_research_optimized_config

config = create_research_optimized_config()
# Optimized for research and experimentation with relaxed thresholds
```

### Custom Configuration

```python
from enhanced_data_labels_system import EnhancedDataLabelsConfig, TradingObjectiveConfig

config = EnhancedDataLabelsConfig(
    trading_objective=TradingObjectiveConfig(
        primary_objective="risk_adjusted_returns",
        max_drawdown_pct=0.05,
        target_sharpe_ratio=1.5,
        enable_regime_conditioning=True
    ),
    min_data_quality_score=0.8,
    min_label_stability_score=0.7
)
```

## 📈 Label Types

### Analyst Labels: "Should we trade?"
- **Purpose**: Binary decision on whether to enter a trade
- **Logic**: 1 if expected PnL > fees + slippage within horizon H, else 0
- **Features**:
  - Considers transaction costs (maker/taker fees, slippage)
  - Accounts for volatility and regime conditions
  - Includes risk management constraints
  - Provides confidence scores

### Tactician Labels: Direction/Magnitude
- **Purpose**: Direction and strength of trade signal
- **Logic**: 1 if max_favorable_excursion(H) ≥ θ_up and max_adverse_excursion(H) ≤ θ_down
- **Features**:
  - Volatility-scaled thresholds (θ_up = k × σ_t)
  - Regime-specific adjustments
  - Magnitude scoring for signal strength
  - Risk-aware filtering

## 🧹 Data Cleaning

### Automatic Cleaning Pipeline
1. **Outlier Detection**: Multiple methods (IQR, Z-score, Isolation Forest)
2. **Missing Value Handling**: Advanced imputation strategies
3. **Timestamp Alignment**: Ensures proper time series structure
4. **Deduplication**: Removes overlapping samples from sliding windows
5. **Trading-Specific Rules**: Removes invalid OHLCV combinations

### Quality Assessment
- **Completeness**: Measures data completeness
- **Accuracy**: Validates data consistency
- **Consistency**: Checks for data patterns
- **Timeliness**: Assesses data freshness

## 🔍 Stability Monitoring

### Leakage Detection
- **Autocorrelation Analysis**: Detects temporal dependencies
- **Mutual Information**: Identifies information leakage
- **Granger Causality**: Tests for predictive relationships

### Drift Detection
- **Kolmogorov-Smirnov Test**: Compares distributions
- **Wasserstein Distance**: Measures distribution shifts
- **Jensen-Shannon Divergence**: Detects concept drift

### OOS Balance Checking
- **Class Ratio Validation**: Ensures balanced representation
- **Regime Mix Validation**: Checks regime distribution
- **Temporal Drift Validation**: Monitors time-based changes

## ⚖️ Balancing & Weighting

### Class Balancing
- **Under-sampling**: Reduces majority class (no-trade samples)
- **Over-sampling**: Increases minority classes (SMOTE, ADASYN, Mixup)
- **Stratified Batching**: Ensures balanced batches for streaming

### Sample Weighting
- **Volatility Weighting**: w_t ∝ 1/σ_t (de-emphasize noisy periods)
- **Confidence Weighting**: w_t ∝ Δp (weight by label confidence)
- **Event Overlap Weighting**: López de Prado method
- **Time Decay Weighting**: Exponential decay for recency
- **Regime-Aware Weighting**: Inverse regime frequency

## 🔗 Integration Points

### Existing Infrastructure
- **Volatility-Aware Labeler**: Enhanced with trading-aware definitions
- **Regime Detection**: Native regime conditioning support
- **Feature Engineering**: Seamless feature integration
- **Training Pipelines**: Drop-in replacement for existing labelers

### Monitoring & Validation
- **Quality Metrics**: Comprehensive quality assessment
- **Performance Tracking**: Processing time and memory usage
- **Validation Suite**: Automated testing and validation
- **Recommendation Engine**: Actionable improvement suggestions

## 📊 Performance Metrics

### Data Quality Metrics
- **Overall Quality Score**: 0.0 - 1.0 (higher is better)
- **Quality Grade**: A, B, C, D, F
- **Component Scores**: Completeness, Accuracy, Consistency, Timeliness

### Label Quality Metrics
- **Predictability**: AUC/PR-AUC from baselines
- **Stability**: Variance of AUC across folds, PSI
- **Consistency**: Mutual information between labels
- **Balance**: Class distribution balance

### Stability Metrics
- **Leakage Score**: 0.0 - 1.0 (higher is better)
- **Drift Score**: 0.0 - 1.0 (higher is better)
- **Autocorrelation Score**: 0.0 - 1.0 (higher is better)

## 🧪 Validation & Testing

### Comprehensive Validation Suite
1. **Data Quality Validation**: Ensures clean, reliable data
2. **Label Generation Validation**: Verifies proper label structure
3. **Label Quality Validation**: Checks quality thresholds
4. **Stability Validation**: Tests for leakage and drift
5. **Trading Objective Validation**: Ensures alignment with goals
6. **Integration Validation**: Verifies infrastructure compatibility
7. **Performance Validation**: Tests speed and memory usage

### Automated Testing
```python
# Run comprehensive validation
validation_result = run_enhanced_labels_validation()

# Check validation status
if validation_result['overall_status'] == 'excellent':
    print("System is ready for production!")
else:
    print("System needs attention:", validation_result['recommendations'])
```

## 🚀 Production Usage

### Best Practices
1. **Regular Validation**: Run validation suite before each deployment
2. **Monitor Stability**: Track stability metrics over time
3. **Quality Thresholds**: Set appropriate quality thresholds for your use case
4. **Regime Awareness**: Enable regime conditioning for better performance
5. **Performance Monitoring**: Track processing time and memory usage

### Monitoring Dashboard
```python
# Get system status
status = enhanced_system.get_system_status()

# Get performance summary
performance = enhanced_system.get_performance_summary()

# Validate integration
integration_status = validate_system_integration()
```

## 🔧 Troubleshooting

### Common Issues

#### Low Data Quality
- **Symptom**: Quality score < 0.7
- **Solution**: Check data source, increase cleaning thresholds
- **Prevention**: Implement data quality monitoring

#### Label Instability
- **Symptom**: Stability score < 0.6
- **Solution**: Check for leakage, adjust stability thresholds
- **Prevention**: Regular stability monitoring

#### Poor Trading Alignment
- **Symptom**: Extreme label ratios (all 0 or all 1)
- **Solution**: Adjust trading objective parameters
- **Prevention**: Validate against trading requirements

### Debug Mode
```python
# Enable debug logging
import logging
logging.getLogger('EnhancedDataLabelsSystem').setLevel(logging.DEBUG)

# Run with detailed output
result = enhanced_system.process_market_data(market_data, force_recompute=True)
```

## 📚 Examples

### Complete Example
See `enhanced_labels_example.py` for a comprehensive demonstration of all features.

### Integration Example
```python
# Full integration pipeline
result = process_market_data_enhanced(
    market_data=market_data,
    force_regime_detection=True,
    force_feature_engineering=True,
    force_recompute=False
)

# Access all results
processed_data = result['processed_data']
labels = result['labels']
regime_data = result['regime_data']
engineered_features = result['engineered_features']
quality_metrics = result['data_quality']
stability_metrics = result['label_stability']
```

## 🤝 Contributing

The enhanced data and labels system is designed to be extensible and maintainable. Key areas for contribution:

1. **New Label Definitions**: Add custom label types
2. **Cleaning Strategies**: Implement new data cleaning methods
3. **Stability Metrics**: Add new stability detection methods
4. **Integration Points**: Connect with additional infrastructure
5. **Validation Tests**: Add new validation scenarios

## 📄 License

This enhanced data and labels system is part of the larger trading ML infrastructure and follows the same licensing terms.

---

**Note**: This system is designed to work seamlessly with existing infrastructure while providing significant improvements in data quality, label stability, and trading relevance. All existing components will natively benefit from these upgrades without requiring code changes.