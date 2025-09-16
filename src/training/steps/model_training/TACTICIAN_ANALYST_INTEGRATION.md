# Tactician-Analyst Integration Implementation

This document describes the implementation of the requested adjustments to the Tactician model training, integrating it with the Analyst model for optimal entry point finding.

## Overview

The Tactician model has been enhanced to work in conjunction with the Analyst model, implementing the following key adjustments:

1. **Green Light Filtering**: Only trains on periods where the Analyst gives a green light to start a trade
2. **Analyst Feature Integration**: Includes the Analyst's model outputs as input features for the Tactician's ML models
3. **Single Model Training**: Uses a single model for all regimes (not per-regime like the Analyst)
4. **Entry Point Optimization**: Implements specialized barrier settings for finding the best entry points

## Architecture

### Model Hierarchy
```
Analyst (5m timeframe) → Green Light Signals + Model Outputs
    ↓
Tactician (1m timeframe) → Entry Point Optimization
```

### Key Components

1. **TacticianSingleModelTrainingStep**: Main training class for single-model training
2. **TacticianBarrierConfig**: Specialized barrier configuration for entry point optimization
3. **TacticianAnalystIntegration**: Integration class that orchestrates the workflow
4. **TacticianBarrierLabeler**: Specialized labeler for entry point optimization

## Implementation Details

### 1. Green Light Filtering

The Tactician only trains on periods where the Analyst provides a green light signal (binary value = 1).

```python
# Filter data to only include green light periods
green_light_mask = analyst_signals == 1
X_filtered = X[green_light_mask]
y_filtered = y[green_light_mask]
```

**Benefits:**
- Focuses training on high-probability trade opportunities
- Reduces noise from low-probability periods
- Improves model efficiency and performance

### 2. Analyst Feature Integration

The Tactician includes all Analyst model outputs as additional input features:

```python
# Combine base features with analyst outputs
additional_features = []
if all_analyst_models_outputs is not None:
    for model_name, model_outputs in all_analyst_models_outputs.items():
        additional_features.append(model_outputs)
        additional_feature_names.extend([f"analyst_{model_name}_{i}" for i in range(model_outputs.shape[1])])

X_combined = np.column_stack([X] + additional_features)
```

**Benefits:**
- Leverages Analyst's market analysis capabilities
- Provides rich context for entry timing decisions
- Enables the Tactician to understand market conditions

### 3. Single Model Training

Unlike the Analyst which uses per-regime training, the Tactician uses a single model for all regimes:

```python
# Train single model for all regimes
trained_model = self.training_utils.train_single_model(
    model_type=primary_model_type,
    X=X,  # All regimes combined
    y=y,  # All regimes combined
    model_name=self.config.single_model_name
)
```

**Benefits:**
- Simpler model architecture
- Better generalization across regimes
- Faster training and inference
- Easier deployment and maintenance

### 4. Entry Point Optimization

Specialized barrier configuration optimized for entry point finding:

```python
@dataclass
class TacticianBarrierConfig(TripleBarrierConfig):
    # Optimized barriers for entry point finding
    profit_take_multiplier: float = 0.0015  # 0.15% - tighter for entry optimization
    stop_loss_multiplier: float = 0.0010    # 0.10% - tighter for entry optimization
    time_barrier_minutes: int = 15          # 15 minutes - shorter for entry timing
    entry_window_minutes: int = 5           # 5-minute window to find best entry
    min_entry_confidence: float = 0.6       # Minimum confidence for entry
```

**Benefits:**
- Shorter time horizons for precise entry timing
- Tighter profit/loss barriers for entry optimization
- Confidence-based filtering for high-quality entries

## Usage Examples

### Basic Usage

```python
from src.training.steps.model_training.tactician_analyst_integration_example import (
    create_tactician_analyst_integration
)
from src.utils.ml_common.config import TacticianTrainingConfig

# Create configuration
config = TacticianTrainingConfig(
    model_name="tactician_single_model",
    timeframe="1m",
    model_types=["NeuralObliviousDecisionEnsembles", "CatBoostRegressor"],
    use_single_model=True,
    single_model_name="tactician_unified_model"
)

# Create integration instance
integration = create_tactician_analyst_integration(config)

# Train with analyst integration
result = integration.train_tactician_with_analyst_integration(
    X=X,  # Base features (1m timeframe)
    y=y,  # Target values
    regime_labels=regime_labels,
    analyst_signals=analyst_signals,  # Binary green light signals
    all_analyst_models_outputs=analyst_outputs,  # Analyst model predictions
    hmm_regime_features=hmm_features
)
```

### Advanced Usage with Custom Barriers

```python
from src.training.steps.market_analysis.triple_barrier_labeling.tactician_barrier_config import (
    create_tactician_barrier_labeler
)

# Create custom barrier labeler
barrier_labeler = create_tactician_barrier_labeler(
    profit_take_multiplier=0.002,  # 0.2% profit take
    stop_loss_multiplier=0.0015,   # 0.15% stop loss
    time_barrier_minutes=20,       # 20-minute time barrier
    entry_window_minutes=8,        # 8-minute entry window
    min_entry_confidence=0.7       # 70% minimum confidence
)

# Generate tactician labels
label_result = barrier_labeler.apply_tactician_labeling(ohlc_data, analyst_signals)
```

## Configuration Options

### TacticianTrainingConfig

```python
@dataclass
class TacticianTrainingConfig(BaseTrainingConfig):
    # Model types to train
    model_types: List[str] = ["NODE", "CatBoostRegressor", "LGBMRegressor", "Ridge"]
    
    # Analyst integration
    analyst_model_path: str = "./models/analyst_ensemble"
    analyst_output_names: List[str] = ["signal_strength", "confidence", "risk_score", "regime_label"]
    analyst_threshold: float = 0.6
    
    # Single model training
    use_single_model: bool = True
    single_model_name: str = "tactician_unified_model"
```

### TacticianBarrierConfig

```python
@dataclass
class TacticianBarrierConfig(TripleBarrierConfig):
    # Optimized barriers for entry point finding
    profit_take_multiplier: float = 0.0015  # 0.15%
    stop_loss_multiplier: float = 0.0010    # 0.10%
    time_barrier_minutes: int = 15          # 15 minutes
    
    # Entry-specific parameters
    entry_window_minutes: int = 5           # 5-minute entry window
    min_entry_confidence: float = 0.6       # 60% minimum confidence
    entry_signal_decay: float = 0.1         # Signal decay rate
```

## Performance Considerations

### Memory Usage
- Single model training reduces memory requirements compared to per-regime training
- Green light filtering reduces training data size
- Optimized for 1m timeframe data

### Training Speed
- Single model training is faster than per-regime training
- Vectorized operations where possible
- Efficient feature combination

### Model Performance
- Focused training on high-probability periods improves model quality
- Analyst features provide rich context for better predictions
- Entry-specific optimization improves timing accuracy

## Integration with Existing Pipeline

The new Tactician implementation integrates seamlessly with the existing training pipeline:

1. **Analyst Training**: Train Analyst models first (per-regime)
2. **Analyst Inference**: Generate green light signals and model outputs
3. **Tactician Training**: Train Tactician using Analyst outputs
4. **Tactician Inference**: Generate entry point predictions

### Pipeline Integration Example

```python
# Step 1: Train Analyst (existing)
analyst_results = train_analyst_models(X_5m, y_5m, regime_labels)

# Step 2: Generate Analyst outputs
analyst_signals, analyst_outputs = generate_analyst_outputs(X_1m, analyst_models)

# Step 3: Train Tactician with integration
tactician_results = integration.train_tactician_with_analyst_integration(
    X=X_1m, y=y_1m, regime_labels=regime_labels,
    analyst_signals=analyst_signals,
    all_analyst_models_outputs=analyst_outputs
)

# Step 4: Generate Tactician predictions
entry_predictions = tactician_model.predict(X_1m_new, analyst_outputs_new)
```

## Testing and Validation

### Unit Tests
- Test green light filtering logic
- Test feature combination
- Test single model training
- Test barrier configuration

### Integration Tests
- Test full Analyst-Tactician workflow
- Test with different market conditions
- Test performance metrics

### Example Test

```python
def test_tactician_analyst_integration():
    # Create test data
    X = np.random.randn(1000, 50)
    y = np.random.randn(1000)
    analyst_signals = np.random.choice([0, 1], 1000, p=[0.8, 0.2])
    
    # Test integration
    integration = create_tactician_analyst_integration()
    result = integration.train_tactician_with_analyst_integration(
        X, y, np.zeros(1000), analyst_signals
    )
    
    assert 'error' not in result
    assert result['integration_metadata']['green_light_filtering'] == True
    assert result['integration_metadata']['single_model_training'] == True
```

## Best Practices

### Data Preparation
1. Ensure Analyst signals are properly aligned with 1m data
2. Validate feature dimensions across all inputs
3. Handle missing data appropriately

### Model Selection
1. Start with simpler models (Ridge, ElasticNet) for baseline
2. Use ensemble methods (NODE, CatBoost) for better performance
3. Consider model complexity vs. training time trade-offs

### Hyperparameter Tuning
1. Use HPO for optimal model parameters
2. Tune barrier parameters based on market conditions
3. Validate on out-of-sample data

### Monitoring
1. Track green light rate from Analyst
2. Monitor Tactician model performance
3. Validate entry point accuracy

## Troubleshooting

### Common Issues

1. **No Green Light Signals**
   - Check Analyst model performance
   - Adjust Analyst threshold
   - Verify signal generation logic

2. **Feature Dimension Mismatches**
   - Ensure all inputs have same sample count
   - Check feature alignment after filtering
   - Validate feature names

3. **Poor Model Performance**
   - Increase training data
   - Adjust barrier parameters
   - Try different model types

4. **Memory Issues**
   - Reduce batch size
   - Use data streaming
   - Optimize feature selection

### Debug Mode

Enable debug logging for detailed information:

```python
import logging
logging.getLogger('TacticianSingleModelTraining').setLevel(logging.DEBUG)
logging.getLogger('TacticianAnalystIntegration').setLevel(logging.DEBUG)
```

## Future Enhancements

### Potential Improvements
1. **Dynamic Barrier Adjustment**: Adjust barriers based on market volatility
2. **Multi-Timeframe Integration**: Incorporate multiple timeframes
3. **Reinforcement Learning**: Use RL for entry point optimization
4. **Real-time Adaptation**: Adapt to changing market conditions

### Research Directions
1. **Optimal Entry Timing**: Research optimal entry point strategies
2. **Risk-Adjusted Returns**: Incorporate risk metrics in training
3. **Market Regime Adaptation**: Dynamic adaptation to market regimes
4. **Ensemble Methods**: Advanced ensemble techniques for entry prediction

## Conclusion

The enhanced Tactician implementation successfully addresses all requested adjustments:

✅ **Green Light Filtering**: Only trains on Analyst green light periods  
✅ **Analyst Feature Integration**: Includes Analyst outputs as features  
✅ **Single Model Training**: Uses one model for all regimes  
✅ **Entry Point Optimization**: Specialized barriers for entry timing  

This implementation provides a robust foundation for entry point optimization while maintaining compatibility with the existing Analyst model architecture.