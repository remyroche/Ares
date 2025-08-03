# Multi-Timeframe Regime Integration System

## Overview

The Multi-Timeframe Regime Integration system provides a unified approach to market regime classification and regime-specific optimization across multiple timeframes. The system follows the principle that there should be **only ONE regime classification** based on the 1-hour timeframe, which represents the strategic (macro trend) level.

## Key Principles

1. **Single Regime Classification**: Only the 1-hour timeframe is used for regime classification
2. **Regime Propagation**: Regime information is propagated to all other timeframes
3. **Consistent Regime**: All timeframes use the same regime classification
4. **Regime-Specific Optimization**: Each timeframe can access regime-specific TP/SL optimization

## Architecture

### Components

1. **HMM Regime Classifier** (`src/analyst/hmm_regime_classifier.py`)
   - Uses Hidden Markov Model for regime classification
   - Works ONLY with 1-hour timeframe data
   - Identifies market regimes: BULL_TREND, BEAR_TREND, SIDEWAYS_RANGE, SR_ZONE_ACTION, HIGH_IMPACT_CANDLE

2. **Regime-Specific TP/SL Optimizer** (`src/training/regime_specific_tpsl_optimizer.py`)
   - Optimizes TP/SL parameters based on current market regime
   - Uses timeframe analysis results for base parameters
   - Separate from HMM classifier (as requested)

3. **Multi-Timeframe Regime Integration** (`src/analyst/multi_timeframe_regime_integration.py`)
   - Integrates HMM classifier with multi-timeframe system
   - Manages regime propagation across timeframes
   - Provides unified interface for regime access

### Timeframe Hierarchy

```
Strategic (1h) ← Regime Classification
    ↓
Tactical (15m) ← Regime Propagation
    ↓
Primary (5m) ← Regime Propagation
    ↓
Execution (1m) ← Regime Propagation
```

## Implementation Details

### 1. HMM Regime Classifier

The HMM classifier is specifically designed for 1-hour timeframe data:

```python
# Key features
- Validates 1h timeframe data
- Uses 20-period volatility for 1h data
- Annualized volatility calculation: std * sqrt(252 * 24)
- Minimum 1000 data points required
- Caches regime for 15 minutes
```

**Regime Classification Process:**
1. Calculate log_returns and volatility_20 from 1h data
2. Train HMM with 4 hidden states
3. Interpret states and map to market regimes
4. Train LightGBM classifier on HMM-derived labels
5. Provide regime predictions with confidence scores

### 2. Regime-Specific TP/SL Optimizer

The optimizer uses the timeframe analysis results to determine optimal parameters:

```python
# Base parameters from timeframe analysis
regime_parameters = {
    "BULL_TREND": {
        "target_pct": 0.4,
        "stop_pct": 0.2,
        "risk_reward_ratio": 2.0,
        "avg_duration_minutes": 37.2,
        "success_rate": 6.99
    },
    "SIDEWAYS_RANGE": {
        "target_pct": 0.5,
        "stop_pct": 0.3,
        "risk_reward_ratio": 1.67,
        "avg_duration_minutes": 67.4,
        "success_rate": 7.81
    }
    # ... other regimes
}
```

**Optimization Process:**
1. Identify current regime using HMM classifier
2. Use regime-specific base parameters
3. Optimize using Optuna with backtesting simulation
4. Cache results for reuse
5. Provide optimized TP/SL parameters

### 3. Multi-Timeframe Integration

The integration ensures consistent regime information across timeframes:

```python
# Key features
- Regime classification only on 1h data
- 15-minute regime cache
- Timeframe-specific adjustments
- Unified interface for all timeframes
```

**Regime Propagation Process:**
1. Classify regime using 1h data
2. Cache regime information
3. Propagate to other timeframes with adjustments
4. Provide timeframe-specific regime information

## Configuration

### HMM Regime Classifier Configuration

```python
"hmm_regime_classifier": {
    "n_states": 4,  # Number of hidden states
    "n_iter": 100,  # Maximum iterations for HMM training
    "random_state": 42,
    "volatility_period": 20,  # 20 periods for 1h timeframe
    "min_data_points": 1000,  # Minimum data points for training
}
```

### Regime-Specific TP/SL Optimizer Configuration

```python
"regime_specific_tpsl_optimizer": {
    "n_trials": 100,  # Number of optimization trials
    "min_trades": 20,  # Minimum trades for optimization
    "optimization_metric": "sharpe_ratio",  # Optimization target
    "cache_duration_minutes": 60,  # Cache optimization results
}
```

### Multi-Timeframe Integration Configuration

```python
"multi_timeframe_regime_integration": {
    "enable_propagation": True,  # Enable regime propagation
    "smoothing_window": 5,  # Smoothing window for regime changes
    "regime_cache_duration_minutes": 15,  # Cache regime for 15 minutes
    "strategic_timeframe": "1h",  # Strategic timeframe for regime classification
}
```

## Usage Examples

### Basic Usage

```python
from src.analyst.multi_timeframe_regime_integration import MultiTimeframeRegimeIntegration

# Initialize
regime_integration = MultiTimeframeRegimeIntegration(config)
await regime_integration.initialize()

# Train HMM classifier (if needed)
await regime_integration.train_hmm_classifier(historical_data_1h)

# Get regime for specific timeframe
regime_info = await regime_integration.get_regime_for_timeframe(
    timeframe="5m",
    current_data=data_5m,
    data_1h=data_1h
)

# Get regime-specific optimization
optimization_params = await regime_integration.get_regime_specific_optimization(
    timeframe="5m",
    current_data=data_5m,
    data_1h=data_1h,
    historical_data=historical_data_1h
)
```

### Integration with Existing Systems

```python
# In your trading system
async def get_trading_parameters(timeframe: str, current_data: pd.DataFrame, data_1h: pd.DataFrame):
    # Get regime-specific optimization
    params = await regime_integration.get_regime_specific_optimization(
        timeframe=timeframe,
        current_data=current_data,
        data_1h=data_1h,
        historical_data=historical_data
    )
    
    # Use optimized parameters
    target_pct = params['target_pct']
    stop_pct = params['stop_pct']
    regime = params['regime']
    
    return {
        'target_pct': target_pct,
        'stop_pct': stop_pct,
        'regime': regime,
        'confidence': params['confidence']
    }
```

## Testing

### Test Script

Run the test script to verify the integration:

```bash
# Basic test
python scripts/test_multi_timeframe_regime_integration.py --symbol ETHUSDT

# Test with HMM training
python scripts/test_multi_timeframe_regime_integration.py --symbol ETHUSDT --train-hmm
```

### Test Output Example

```
[14:30:15] INFO: 🚀 Starting full multi-timeframe regime integration test...
[14:30:16] INFO: ✅ Loaded 1000 candles for 1h
[14:30:16] INFO: ✅ Loaded 1000 candles for 5m
[14:30:17] INFO: 🎓 Training HMM regime classifier...
[14:30:20] INFO: ✅ HMM regime classifier trained successfully
[14:30:21] INFO: 📊 Current regime: BULL_TREND
[14:30:21] INFO: 📊 Confidence: 0.85
[14:30:22] INFO: 📊 Testing regime for 5m...
[14:30:22] INFO:    Regime: BULL_TREND
[14:30:22] INFO:    Confidence: 0.85
[14:30:22] INFO:    Volatility multiplier: 1.3
[14:30:23] INFO: 🎯 Testing optimization for 5m...
[14:30:23] INFO:    Target %: 0.400
[14:30:23] INFO:    Stop %: 0.200
[14:30:23] INFO:    Risk/Reward: 2.00
[14:30:24] INFO: ✅ Full test completed successfully!
```

## Integration with Step2PrelimOpt

The regime-specific TP/SL optimizer can be integrated with the existing Step2PrelimOpt system:

```python
# In Step2PrelimOpt
async def optimize_sl_tp_with_regime(self, data: pd.DataFrame, timeframe: str):
    # Get regime-specific optimization
    regime_params = await self.regime_integration.get_regime_specific_optimization(
        timeframe=timeframe,
        current_data=data,
        data_1h=data_1h,
        historical_data=historical_data
    )
    
    # Use regime-specific parameters as starting point
    target_pct = regime_params['target_pct']
    stop_pct = regime_params['stop_pct']
    
    # Run additional optimization if needed
    # ... existing optimization logic
    
    return optimized_params
```

## Benefits

1. **Consistent Regime Classification**: Single source of truth for market regime
2. **Regime-Specific Optimization**: Tailored TP/SL parameters for each regime
3. **Multi-Timeframe Support**: Regime information available across all timeframes
4. **Performance Optimization**: Cached results and efficient propagation
5. **Integration Ready**: Easy to integrate with existing systems

## Future Enhancements

1. **Dynamic Regime Adaptation**: Real-time regime change detection
2. **Regime-Specific Models**: Different model architectures per regime
3. **Advanced Propagation**: More sophisticated regime propagation algorithms
4. **Performance Monitoring**: Track regime classification accuracy
5. **Automated Training**: Automatic retraining based on regime changes

## Troubleshooting

### Common Issues

1. **HMM Training Fails**
   - Ensure sufficient 1h data (minimum 1000 points)
   - Check data quality and timeframe validation
   - Verify configuration parameters

2. **Regime Classification Errors**
   - Validate 1h timeframe data
   - Check HMM model training status
   - Verify feature calculation

3. **Optimization Issues**
   - Ensure sufficient historical data
   - Check regime parameter configuration
   - Verify optimization metric settings

### Debug Information

```python
# Get integration statistics
stats = regime_integration.get_integration_statistics()
print(f"Current regime: {stats['current_regime']}")
print(f"HMM trained: {stats['hmm_trained']}")
print(f"Last update: {stats['last_regime_update']}")
```

## Conclusion

The Multi-Timeframe Regime Integration system provides a robust, scalable solution for regime classification and optimization across multiple timeframes. By centralizing regime classification on the 1-hour timeframe and propagating this information to other timeframes, the system ensures consistency while providing regime-specific optimizations for improved trading performance. 