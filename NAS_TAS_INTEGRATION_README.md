# NAS & TAS Integration with Analyst & Tactician

## Overview

This document describes the integration of **NAS (Neural Architecture Search)** and **TAS (Tree Architecture Search)** into the **Analyst** and **Tactician** systems for live trading. The integration ensures that:

1. **NAS** is trained on the correct **5m timeframe** for regime-based analysis (Analyst - IF we trade)
2. **TAS** is trained on the correct **1m timeframe** for signal-based analysis (Tactician - WHEN we trade)
3. Both systems become part of the base ensemble models for their respective components

## Architecture Overview

### Analyst Integration (5m Timeframe)
- **Purpose**: Determines **IF** we should enter a trade
- **Timeframe**: 5-minute candles
- **Training**: Per-regime (HMM-detected market conditions)
- **Model Type**: NAS (Neural Architecture Search)
- **Integration**: `AnalystNASIntegration` class

### Tactician Integration (1m Timeframe)
- **Purpose**: Determines **WHEN** we should enter/exit trades
- **Timeframe**: 1-minute candles
- **Training**: Per-signal-type (pattern-based signals)
- **Model Type**: TAS (Tree Architecture Search)
- **Integration**: `TacticianTASIntegration` class

## Key Components

### AnalystNASIntegration

```python
from src.training.steps.market_analysis.nas_regime.core.enhanced_nas_engine import (
    AnalystNASIntegration, AnalystNASIntegrationConfig
)

# Configuration for Analyst NAS integration
config = AnalystNASIntegrationConfig(
    analyst_name="analyst_nas_ensemble",
    timeframe="5m",  # 5-minute timeframe
    regime_types=[
        "bull_trending", "bear_trending", "sideways",
        "volatile", "breakout"
    ],
    regime_confidence_threshold=0.7,
    enable_live_training=True,
    live_update_interval=300,  # 5 minutes
    max_base_models=5
)

# Initialize integration
analyst_nas = AnalystNASIntegration(config)
await analyst_nas.initialize()
```

### TacticianTASIntegration

```python
from src.training.steps.market_analysis.tas_regime.core.enhanced_tas_engine import (
    TacticianTASIntegration, TacticianTASIntegrationConfig
)

# Configuration for Tactician TAS integration
config = TacticianTASIntegrationConfig(
    tactician_name="tactician_tas_ensemble",
    timeframe="1m",  # 1-minute timeframe
    signal_types=[
        "bullish_continuation", "bearish_continuation",
        "bullish_reversal", "bearish_reversal", "neutral",
        "breakout_up", "breakout_down"
    ],
    signal_confidence_threshold=0.7,
    analyst_signal_required=True,  # Coordinate with Analyst
    analyst_confidence_threshold=0.6,
    enable_live_training=True,
    live_update_interval=60,  # 1 minute
    max_base_models=7
)

# Initialize integration
tactician_tas = TacticianTASIntegration(config)
await tactician_tas.initialize()
```

## Training Process

### Analyst NAS Training (5m, Regime-Based)

1. **Data Preparation**: Load 5m market data with regime labels
2. **Regime Detection**: Use HMM to detect market regimes
3. **Per-Regime Training**:
   - Filter data by regime type
   - Generate regime-specific features
   - Perform NAS search for optimal neural architecture
   - Store regime-specific models
4. **Ensemble Creation**: Combine regime models into stacking ensemble
5. **Performance Validation**: Cross-validate ensemble performance

```python
# Training example
market_data_5m = load_market_data("5m", lookback_hours=24)
target_data_5m = generate_targets(market_data_5m)

# Train regime models
training_results = await analyst_nas.train_regime_models(
    market_data=market_data_5m,
    target_data=target_data_5m,
    validation_data=(X_val, y_val)
)

print(f"Trained {len(training_results['trained_models'])} regime models")
print(f"Ensemble performance: {training_results['ensemble_performance']}")
```

### Tactician TAS Training (1m, Signal-Based)

1. **Data Preparation**: Load 1m market data with signal labels
2. **Signal Detection**: Detect trading signals using pattern analysis
3. **Analyst Coordination**: Cache and validate Analyst signals
4. **Per-Signal Training**:
   - Filter data by signal type
   - Generate signal-specific features (including micro-patterns)
   - Perform TAS search for optimal tree architecture
   - Store signal-specific models
5. **Ensemble Creation**: Combine signal models into stacking ensemble
6. **Performance Validation**: Cross-validate ensemble performance

```python
# Training example
market_data_1m = load_market_data("1m", lookback_minutes=60)
target_data_1m = generate_targets(market_data_1m)

# Get analyst signals for coordination
analyst_signals = await get_analyst_signals(market_data_1m)

# Train signal models
training_results = await tactician_tas.train_signal_models(
    market_data=market_data_1m,
    target_data=target_data_1m,
    analyst_signals=analyst_signals,
    validation_data=(X_val, y_val)
)

print(f"Trained {len(training_results['trained_models'])} signal models")
print(f"Analyst signals used: {training_results['analyst_signals_used']}")
```

## Prediction Process

### Analyst NAS Prediction

```python
# Get regime-aware predictions
market_data_5m = get_latest_market_data("5m", lookback_candles=20)

predictions = await analyst_nas.predict_with_regime_ensemble(market_data_5m)

# Results include:
# - ensemble_prediction: Final ensemble prediction
# - current_regime: Detected market regime
# - regime_confidence: Confidence in regime detection
# - regime_prediction: Regime-specific model prediction (if available)
```

### Tactician TAS Prediction

```python
# Get signal-aware predictions with Analyst coordination
market_data_1m = get_latest_market_data("1m", lookback_candles=10)
analyst_decision = get_analyst_decision()  # From Analyst system

predictions = await tactician_tas.predict_with_signal_ensemble(
    market_data=market_data_1m,
    analyst_decision=analyst_decision
)

# Results include:
# - ensemble_prediction: Final ensemble prediction
# - current_signal: Detected trading signal
# - analyst_compatible: Whether signal matches Analyst direction
# - timing_confidence: Combined confidence score
```

## Live Trading Integration

### Analyst NAS Live Updates

```python
# Update models with live data
new_data_5m = get_live_market_data("5m", minutes=5)
target_data_5m = generate_live_targets(new_data_5m)

success = await analyst_nas.update_live_models(new_data_5m, target_data_5m)

if success:
    print("✅ Analyst models updated with live data")
```

### Tactician TAS Live Updates

```python
# Update models with live data and analyst coordination
new_data_1m = get_live_market_data("1m", minutes=1)
target_data_1m = generate_live_targets(new_data_1m)
analyst_signals = get_latest_analyst_signals()

success = await tactician_tas.update_live_models(
    new_data=new_data_1m,
    target_data=target_data_1m,
    analyst_signals=analyst_signals
)

if success:
    print("✅ Tactician models updated with live data")
```

## Model Persistence

### Saving Models

```python
# Save Analyst NAS models
await analyst_nas.save_models("models/analyst_nas_models")

# Save Tactician TAS models
await tactician_tas.save_models("models/tactician_tas_models")
```

### Loading Models

```python
# Load Analyst NAS models (would need load functionality)
# analyst_nas.load_models("models/analyst_nas_models")

# Load Tactician TAS models (would need load functionality)
# tactician_tas.load_models("models/tactician_tas_models")
```

## Configuration Examples

### Analyst NAS Configuration

```python
analyst_config = AnalystNASIntegrationConfig(
    # Basic settings
    analyst_name="production_analyst_nas",
    output_dir="models/production/analyst_nas",
    timeframe="5m",

    # Regime settings
    regime_types=[
        "strong_bull", "weak_bull", "sideways",
        "weak_bear", "strong_bear", "volatile"
    ],
    regime_confidence_threshold=0.75,

    # Training settings
    enable_live_training=True,
    live_update_interval=300,  # 5 minutes
    model_retraining_threshold=0.03,  # 3% performance drop

    # NAS engine settings
    nas_config=NASSearchConfig(
        search_strategy=SearchStrategy.ENHANCED_BAYESIAN,
        population_size=50,
        max_generations=100,
        max_evaluations=800,
        enable_multi_objective=True,
        objective_weights={
            'performance': 1.0,
            'complexity': 0.2,
            'efficiency': 0.3,
            'trading_viability': 0.9  # High weight for trading
        }
    ),

    # Ensemble settings
    max_base_models=6,
    enable_model_diversity=True,
    diversity_threshold=0.35
)
```

### Tactician TAS Configuration

```python
tactician_config = TacticianTASIntegrationConfig(
    # Basic settings
    tactician_name="production_tactician_tas",
    output_dir="models/production/tactician_tas",
    timeframe="1m",

    # Signal settings
    signal_types=[
        "bullish_continuation", "bearish_continuation",
        "bullish_reversal", "bearish_reversal",
        "neutral", "breakout_up", "breakout_down",
        "momentum_up", "momentum_down"
    ],
    signal_confidence_threshold=0.75,

    # Analyst coordination
    analyst_signal_required=True,
    analyst_confidence_threshold=0.65,
    max_signal_delay_seconds=45,  # Stricter timing

    # Training settings
    enable_live_training=True,
    live_update_interval=60,  # 1 minute
    model_retraining_threshold=0.02,  # 2% performance drop

    # TAS engine settings
    tas_config=TASConfig(
        search_strategy=TreeSearchStrategy.ENHANCED_BAYESIAN,
        population_size=50,
        max_generations=100,
        max_evaluations=800,
        enable_multi_objective=True,
        objective_weights={
            'performance': 1.0,
            'complexity': 0.2,
            'efficiency': 0.3,
            'timing_precision': 0.9  # High weight for timing
        },
        max_trees=40,  # More trees for complex patterns
        max_tree_depth=25  # Deeper trees for precision
    ),

    # Ensemble settings
    max_base_models=8,  # More models for timing diversity
    enable_model_diversity=True,
    diversity_threshold=0.4  # Higher diversity requirement
)
```

## Performance Monitoring

### Monitoring Analyst NAS Performance

```python
# Get performance metrics
analyst_metrics = {
    'regime_performance': analyst_nas.regime_performance,
    'ensemble_diversity': analyst_nas.ensemble_manager.diversity_score,
    'model_count': len(analyst_nas.ensemble_manager.models),
    'last_training': analyst_nas.last_training_time
}

print(f"Analyst NAS Performance: {analyst_metrics}")
```

### Monitoring Tactician TAS Performance

```python
# Get performance metrics
tactician_metrics = {
    'signal_performance': tactician_tas.signal_performance,
    'analyst_compatibility_rate': calculate_compatibility_rate(tactician_tas.analyst_signals_cache),
    'ensemble_diversity': tactician_tas.ensemble_manager.diversity_score,
    'model_count': len(tactician_tas.ensemble_manager.models),
    'last_training': tactician_tas.last_training_time
}

print(f"Tactician TAS Performance: {tactician_metrics}")
```

## Integration with Existing Systems

### Integration with Current Analyst

```python
# Replace existing analyst model training with NAS
class EnhancedAnalyst:
    def __init__(self, config):
        self.config = config
        self.nas_integration = None
        self.traditional_models = {}  # Keep for fallback

    async def initialize(self):
        # Initialize NAS integration
        nas_config = AnalystNASIntegrationConfig(**self.config.get('nas_config', {}))
        self.nas_integration = AnalystNASIntegration(nas_config)
        await self.nas_integration.initialize()

        # Keep traditional models as fallback
        self._initialize_traditional_models()

    async def analyze_market(self, market_data):
        # Try NAS prediction first
        nas_results = await self.nas_integration.predict_with_regime_ensemble(market_data)

        # Fallback to traditional models if needed
        if nas_results.get('error'):
            traditional_results = await self._traditional_analysis(market_data)
            return traditional_results

        return nas_results
```

### Integration with Current Tactician

```python
# Replace existing tactician model training with TAS
class EnhancedTactician:
    def __init__(self, config):
        self.config = config
        self.tas_integration = None
        self.analyst_integration = None
        self.traditional_models = {}  # Keep for fallback

    async def initialize(self):
        # Initialize TAS integration
        tas_config = TacticianTASIntegrationConfig(**self.config.get('tas_config', {}))
        self.tas_integration = TacticianTASIntegration(tas_config)
        await self.tas_integration.initialize()

        # Initialize analyst integration for coordination
        self.analyst_integration = get_analyst_integration()

        # Keep traditional models as fallback
        self._initialize_traditional_models()

    async def generate_signals(self, market_data, analyst_decision):
        # Try TAS prediction first
        tas_results = await self.tas_integration.predict_with_signal_ensemble(
            market_data, analyst_decision
        )

        # Fallback to traditional models if needed
        if tas_results.get('error'):
            traditional_results = await self._traditional_signals(market_data)
            return traditional_results

        return tas_results
```

## Best Practices

### 1. Data Quality
- Ensure high-quality 5m data for Analyst training
- Ensure high-quality 1m data for Tactician training
- Validate regime and signal labels before training

### 2. Training Frequency
- Analyst: Retrain every 5 minutes during active hours
- Tactician: Retrain every 1 minute during active hours
- Use performance thresholds to avoid unnecessary retraining

### 3. Ensemble Management
- Monitor ensemble diversity regularly
- Remove underperforming regime/signal models
- Balance model count with performance requirements

### 4. Live Trading
- Start with paper trading to validate integration
- Monitor prediction latency (should be < 100ms)
- Implement circuit breakers for model failures

### 5. Performance Monitoring
- Track regime detection accuracy
- Monitor signal timing precision
- Validate Analyst-Tactician coordination
- Log all predictions for offline analysis

## Troubleshooting

### Common Issues

1. **Low Regime Confidence**
   - Increase historical data for HMM training
   - Adjust regime confidence threshold
   - Validate market data quality

2. **Poor Signal Detection**
   - Increase pattern recognition training data
   - Adjust signal confidence threshold
   - Validate 1m data granularity

3. **Analyst-Tactician Mismatch**
   - Check signal delay settings
   - Validate direction mapping
   - Monitor analyst signal quality

4. **Training Performance Issues**
   - Reduce search space complexity
   - Increase computational resources
   - Use performance estimators

### Debugging Commands

```python
# Check NAS engine status
print(f"NAS Engine: {analyst_nas.nas_engine.evaluation_count} evaluations")
print(f"Current regime: {analyst_nas.current_regime}")

# Check TAS engine status
print(f"TAS Engine: {tactician_tas.tas_engine.evaluation_count} evaluations")
print(f"Current signal: {tactician_tas.current_signal}")

# Validate ensemble performance
analyst_ensemble = analyst_nas.ensemble_manager
tactician_ensemble = tactician_tas.ensemble_manager

print(f"Analyst ensemble diversity: {analyst_ensemble.diversity_score}")
print(f"Tactician ensemble diversity: {tactician_ensemble.diversity_score}")
```

## Conclusion

The NAS and TAS integration provides a sophisticated, adaptive approach to trading model development:

- **Analyst NAS**: Learns optimal neural architectures for each market regime on 5m data
- **Tactician TAS**: Learns optimal tree architectures for each signal type on 1m data
- **Coordination**: Analyst and Tactician work together with proper signal validation
- **Adaptability**: Models continuously adapt to changing market conditions
- **Performance**: Ensemble approach provides robust, diverse predictions

This integration transforms the Analyst and Tactician from static models into adaptive, learning systems that evolve with market conditions while maintaining the critical distinction between "IF we trade" (Analyst) and "WHEN we trade" (Tactician).