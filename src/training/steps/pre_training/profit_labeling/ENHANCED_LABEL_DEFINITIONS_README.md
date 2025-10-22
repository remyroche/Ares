# Enhanced Label Definitions for Trading ML

## Overview

This module implements enhanced label definitions that align with actual trading objectives, addressing the common issue where "most ML accuracy problems are actually label-definition problems."

## Key Features

### 1. Analyst Labels - "Should we trade?"
**Definition**: Binary labels (0/1) indicating whether a trade should be taken based on expected profitability.

**Formula**:
```
Label = 1 if (Expected PnL > Fees + Slippage) else 0
```

**Key Components**:
- **Expected PnL**: Calculated over specified horizon (default: 60 minutes)
- **Trading Costs**: Includes maker/taker fees and slippage estimates
- **Risk Awareness**: Accounts for maximum position size and drawdown limits
- **Regime Conditioning**: Volatility-scaled thresholds

### 2. Tactician Labels - "Direction & Magnitude"
**Definition**: Labels indicating trade direction and signal strength based on price excursions.

**Formula**:
```
Label = 1 if (Max_Favorable_Excursion ≥ θ_up) AND (Max_Adverse_Excursion ≤ θ_down)
Magnitude = (Favorable_Excursion + |Adverse_Excursion|) / 2
```

**Key Components**:
- **Excursion Thresholds**: Configurable favorable (default: +1σ) and adverse (default: -2σ)
- **Magnitude Scaling**: Signal strength proportional to excursion size
- **Regime Sensitivity**: Volatility-adjusted thresholds
- **Horizon Flexibility**: Configurable look-ahead periods

### 3. Regime Conditioning
**Purpose**: Ensures labels work across different market conditions.

**Features**:
- **Volatility Scaling**: Thresholds adjust based on market volatility
- **Adaptive Thresholds**: Historical regime behavior informs current thresholds
- **Multi-Regime Support**: Different parameters for low/high volatility regimes

### 4. Risk Awareness
**Purpose**: Prevents labels that would result in unfavorable risk-reward profiles.

**Features**:
- **Stop-Loss Protection**: Label = 0 if stop-loss would be hit before target
- **Risk-Reward Ratios**: Minimum acceptable risk-reward ratios
- **Portfolio Risk Limits**: Position sizing constraints
- **Correlation Risk**: Avoid over-concentration

### 5. Data Quality & Cleaning
**Purpose**: Ensure high-quality, reliable training data.

**Features**:
- **Outlier Detection**: Multiple methods (IQR, Z-score, Isolation Forest)
- **Volume Filtering**: Minimum/maximum volume thresholds
- **Price Validation**: Remove extreme price changes and invalid data
- **Timestamp Alignment**: Handle gaps and misalignments
- **Deduplication**: Remove overlapping or duplicate samples

### 6. Stability & Quality Checks
**Purpose**: Monitor label quality and detect potential issues.

**Features**:
- **Autocorrelation Detection**: Identify potential data leakage
- **Balance Monitoring**: Ensure reasonable class distribution
- **Drift Detection**: Track label distribution changes over time
- **OOS Balance Checks**: Verify out-of-sample label balance
- **Recomputations**: Automatic label refresh on data updates

## Usage Examples

### Basic Analyst Labeler

```python
from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import create_enhanced_analyst_labeler

# Create analyst labeler
labeler = create_enhanced_analyst_labeler()

# Generate labels
result = labeler.generate_labels(market_data)
analyst_labels = result.labels['analyst_target']
confidence_scores = result.labels['analyst_confidence']
```

### Basic Tactician Labeler

```python
from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import create_enhanced_tactician_labeler

# Create tactician labeler
labeler = create_enhanced_tactician_labeler()

# Generate labels
result = labeler.generate_labels(market_data)
tactician_labels = result.labels['tactician_target']
magnitude_scores = result.labels['tactician_magnitude']
```

### Custom Configuration

```python
from src.training.steps.pre_training.profit_labeling.enhanced_label_definitions import (
    EnhancedLabelDefinitions,
    AnalystLabelConfig,
    TacticianLabelConfig,
    TradingCosts
)

# Create custom configuration
analyst_config = AnalystLabelConfig(
    horizon_minutes=120,  # 2-hour horizon
    min_profit_threshold_usd=10.0,
    trading_costs=TradingCosts(
        maker_fee=0.0005,  # VIP fees
        taker_fee=0.001,
        slippage_pct=0.0005
    )
)

# Initialize enhanced labeler
labeler = EnhancedLabelDefinitions(
    analyst_config=analyst_config
)

# Generate labels
labels, confidence = labeler.generate_analyst_labels(
    market_data, volatility_series, regime_data
)
```

### Integration with Multi-Horizon Labeler

```python
from src.training.steps.pre_training.multi_horizon_profit_labeler import (
    MultiHorizonProfitLabeler,
    MultiHorizonConfig
)

# Configure for enhanced labels
config = MultiHorizonConfig(
    enable_enhanced_labels=True,
    label_definition_type="analyst",  # or "tactician"
    enable_regime_aware_labeling=True
)

# Initialize and use
labeler = MultiHorizonProfitLabeler(config)
results = await labeler.execute_labeling(
    symbol="ETHUSDT",
    exchange="binance",
    timeframe="15m"
)
```

## Configuration Options

### Analyst Label Configuration

```python
@dataclass
class AnalystLabelConfig:
    horizon_minutes: int = 60                    # Trading horizon
    min_profit_threshold_usd: float = 5.0        # Minimum profit threshold
    trading_costs: TradingCosts = TradingCosts() # Fee and slippage settings
    min_confidence_threshold: float = 0.6        # Minimum confidence for labels
    min_volume_threshold: float = 1000.0         # Minimum volume filter
    max_spread_pct: float = 0.01                 # Maximum spread filter
    max_position_size_pct: float = 0.05          # Max position size
    max_drawdown_pct: float = 0.02               # Max drawdown limit
    enable_regime_conditioning: bool = True      # Enable volatility scaling
    volatility_scaling_factor: float = 1.0       # Volatility adjustment factor
```

### Tactician Label Configuration

```python
@dataclass
class TacticianLabelConfig:
    favorable_excursion_threshold: float = 1.0   # +1σ favorable threshold
    adverse_excursion_threshold: float = -2.0    # -2σ adverse threshold
    horizon_minutes: int = 30                    # Look-ahead horizon
    min_direction_confidence: float = 0.7        # Minimum confidence
    magnitude_scaling: bool = True               # Enable magnitude scaling
    max_magnitude: float = 5.0                  # Maximum magnitude value
    enable_regime_conditioning: bool = True      # Enable volatility scaling
    volatility_sensitivity: float = 1.0          # Sensitivity to volatility
```

### Trading Costs Configuration

```python
@dataclass
class TradingCosts:
    maker_fee: float = 0.001      # 0.1% maker fee
    taker_fee: float = 0.002      # 0.2% taker fee
    slippage_pct: float = 0.001   # 0.1% slippage estimate
    min_trade_size: float = 10.0  # Minimum trade size in USD

    def total_costs(self, trade_size_usd: float, is_maker: bool = True) -> float:
        """Calculate total trading costs."""
        return trade_size_usd * (self.maker_fee if is_maker else self.taker_fee) + \
               trade_size_usd * self.slippage_pct
```

## Advanced Features

### Regime Classification

The system automatically classifies market regimes based on volatility:

```python
# Automatic regime classification
regime_data = labeler._classify_regimes_from_volatility(volatility_series)

# Custom regime classification
regime_data = pd.Series(['low_vol', 'normal', 'high_vol'], index=market_data.index)
```

### Stability Monitoring

```python
# Check label stability
stability_results = labeler.check_label_stability(
    current_labels=analyst_labels,
    historical_labels=historical_labels,  # Optional
    market_data=market_data               # Optional
)

# Access stability metrics
print(f"Is stable: {stability_results['is_stable']}")
print(f"Issues: {stability_results['issues']}")
print(f"Metrics: {stability_results['metrics']}")
```

### Data Cleaning Pipeline

```python
# Apply comprehensive data cleaning
cleaned_data = labeler._apply_data_cleaning(market_data)

# Custom cleaning configuration
cleaning_config = DataCleaningConfig(
    outlier_method="iqr",
    outlier_threshold=3.0,
    min_volume_threshold=1000.0,
    enforce_timestamp_alignment=True,
    enable_deduplication=True
)
```

## Integration with Existing Pipeline

The enhanced label definitions are fully integrated with the existing volatility-aware labeling pipeline:

1. **Backward Compatible**: Existing code continues to work unchanged
2. **Configurable**: Enable enhanced labels via configuration flags
3. **Performance Optimized**: Caching and parallel processing support
4. **Quality Assured**: Built-in quality scoring and validation

### Migration Guide

To migrate from standard to enhanced labels:

```python
# Before (standard labels)
config = VolatilityAwareConfig()

# After (enhanced labels)
config = VolatilityAwareConfig(
    enable_enhanced_labels=True,
    label_definition_type=LabelDefinitionType.ANALYST
)
```

## Performance Considerations

### Speed vs Accuracy Trade-offs

```python
# Fast configuration (speed optimized)
fast_config = VolatilityAwareConfig(
    min_data_points=500,
    enable_caching=True,
    parallel_processing=True
)

# Accurate configuration (quality optimized)
accurate_config = VolatilityAwareConfig(
    min_data_points=2000,
    enable_caching=True,
    parallel_processing=True,
    # More sophisticated components enabled
)
```

### Memory Management

- **Caching**: Configurable cache duration and size limits
- **Batch Processing**: Process large datasets in chunks
- **Intermediate Results**: Optional saving of intermediate computations

## Monitoring & Validation

### Label Quality Metrics

```python
# Access quality scores from results
quality_scores = result.quality_scores

for target, quality in quality_scores.items():
    print(f"{target}:")
    print(f"  Predictability (AUC): {quality.predictability:.3f}")
    print(f"  Stability: {quality.stability:.3f}")
    print(f"  Balance: {quality.balance:.3f}")
    print(f"  Overall Quality: {quality.overall_quality:.3f}")
```

### Stability Monitoring

```python
# Monitor for label drift over time
drift_score = labeler._calculate_drift_score(current_labels, historical_labels)

# Monitor for autocorrelation (potential leakage)
autocorrelation = labeler._check_autocorrelation(labels)

# Monitor class balance
balance_deviation = abs(labels.mean() - 0.5)
```

## Troubleshooting

### Common Issues

1. **Low Label Quality**
   - Check data quality and cleaning settings
   - Verify volatility estimates are reasonable
   - Consider adjusting thresholds for your specific market

2. **Imbalanced Labels**
   - Review profitability thresholds
   - Check for regime-specific imbalances
   - Consider reweighting strategies

3. **High Autocorrelation**
   - Review for potential data leakage
   - Check feature engineering pipeline
   - Verify no future information in features

4. **Poor OOS Performance**
   - Enable stability checks and monitoring
   - Verify OOS label balance matches training
   - Consider regime-specific model validation

### Debug Mode

```python
import logging

# Enable debug logging
logging.getLogger('EnhancedLabelDefinitions').setLevel(logging.DEBUG)

# Run with detailed output
result = labeler.generate_labels(market_data)
```

## Contributing

To extend the label definitions:

1. **Add New Label Types**: Extend `LabelDefinitionType` enum
2. **Custom Logic**: Implement in `EnhancedLabelDefinitions` class
3. **Configuration**: Add new configuration dataclasses
4. **Integration**: Update `volatility_aware_labeler.py` integration

## References

- "Advances in Financial Machine Learning" by Marcos López de Prado
- "Machine Learning for Asset Managers" by Marcos López de Prado
- "Quantitative Trading" by Ernie Chan

## License

This enhanced labeling system is part of the broader trading ML framework and follows the same licensing terms.