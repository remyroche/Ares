# Scenario-Based Tactician Implementation

## Overview

This document describes the implementation of the probabilistic scenario analysis plan for the Tactician, which extends the existing multi-output prediction system with sophisticated scenario-based decision making.

## Implementation Summary

The scenario-based Tactician implementation provides:

1. **Probabilistic Scenario Analysis**: 6 distinct scenarios with specific profit/loss targets
2. **Enhanced Decision Making**: Combines multi-output and scenario analysis for better entry/exit decisions
3. **Step17 Optimization Ready**: All parameters configurable for optimization
4. **Backward Compatibility**: Works alongside existing Tactician systems
5. **Robust Error Handling**: Fallback mechanisms and graceful degradation

## Core Components

### 1. ScenarioBasedPredictor

**Location**: `src/tactician/scenario_based_predictor.py`

**Purpose**: Implements the probabilistic scenario analysis with configurable parameters.

**Key Features**:
- 6 scenario definitions (3 profit zones, 2 risk zones, 1 neutral)
- Configurable profit targets and stop losses
- Time-based scenario evaluation
- LightGBM-based multi-class classification
- Feature engineering with 15 technical indicators
- Confidence scoring and scenario analysis

**Scenarios**:
- **Label 0**: Profit Zone 1 (Small Profit): +0.5% before -0.5%
- **Label 1**: Profit Zone 2 (Medium Profit): +1% before -0.5%
- **Label 2**: Profit Zone 3 (Large Profit): +1.5% before -0.5%
- **Label 3**: Risk Zone 1 (Small Loss): -0.5% before +0.5%
- **Label 4**: Risk Zone 2 (Medium Loss): -1% before +0.5%
- **Label 5**: Neutral: No scenario triggered within time limit

### 2. Enhanced MLTacticsManager

**Location**: `src/tactician/ml_tactics_manager.py`

**Purpose**: Integrates scenario analysis with existing multi-output predictions.

**Key Features**:
- Combines multi-output and scenario predictions
- Enhanced entry/exit decision logic
- Confidence boosting with scenario analysis
- Detailed reasoning for decisions
- Backward compatibility with existing systems

### 3. Configuration System

**Location**: `src/config/tactician_triple_barrier_config.yaml`

**Purpose**: Provides step17 optimization parameters for all scenario analysis components.

## Configuration Parameters

All parameters are configurable for step17 optimization:

### Scenario Definitions
```yaml
scenario_analysis:
  # Profit targets and stop losses for each scenario
  profit_zone_1_target: 0.005      # 0.5% profit target
  profit_zone_1_stop_loss: -0.005  # 0.5% stop loss
  profit_zone_2_target: 0.01       # 1.0% profit target
  profit_zone_2_stop_loss: -0.005  # 0.5% stop loss
  profit_zone_3_target: 0.015      # 1.5% profit target
  profit_zone_3_stop_loss: -0.005  # 0.5% stop loss
  risk_zone_1_target: 0.005        # 0.5% profit target (for risk scenario)
  risk_zone_1_stop_loss: -0.005    # 0.5% stop loss
  risk_zone_2_target: 0.01         # 1.0% profit target (for risk scenario)
  risk_zone_2_stop_loss: -0.005    # 0.5% stop loss
  neutral_target: 0.0              # Neutral scenario target
  neutral_stop_loss: 0.0           # Neutral scenario stop loss
```

### Time and Model Configuration
```yaml
  # Time limit for scenario evaluation
  time_limit_minutes: 30
  
  # Model configuration
  n_estimators: 100
  learning_rate: 0.1
  max_depth: 6
  num_leaves: 31
  subsample: 0.8
  colsample_bytree: 0.8
  random_state: 42
```

### Decision Thresholds
```yaml
  # Decision thresholds
  profit_zone_combined_threshold: 0.6    # Minimum combined profit zone probability
  risk_zone_combined_threshold: 0.2      # Maximum combined risk zone probability
  exit_risk_threshold: 0.5               # Risk threshold for exit signals
  neutral_threshold: 0.3                 # Neutral scenario threshold
  confidence_threshold: 0.7              # Minimum confidence for decisions
```

### Feature Engineering
```yaml
  # Feature engineering parameters
  lookback_periods: 20
  volatility_window: 20
  rsi_period: 14
  ma_short_period: 5
  ma_long_period: 20
  volume_ma_period: 10
```

## Usage Examples

### Basic Usage

```python
from src.tactician.ml_tactics_manager import MLTacticsManager

# Initialize with configuration
config = load_configuration()
manager = MLTacticsManager(config)
await manager.initialize()

# Generate enhanced predictions
enhanced_predictions = await manager.generate_enhanced_predictions(
    market_data=market_data,
    analyst_barriers=analyst_barriers,
    symbol="BTCUSDT",
    timeframe="1m",
    analyst_confidence=0.8
)

# Access results
entry_signal = enhanced_predictions["enhanced_decisions"]["entry_signal"]
confidence = enhanced_predictions["enhanced_decisions"]["confidence"]
reasoning = enhanced_predictions["enhanced_decisions"]["reasoning"]
scenario_analysis = enhanced_predictions["enhanced_decisions"]["scenario_analysis"]
```

### Direct Scenario Predictor Usage

```python
from src.tactician.scenario_based_predictor import ScenarioBasedPredictor

# Initialize predictor
predictor = ScenarioBasedPredictor(config)
await predictor.initialize()

# Extract features and predict
features = predictor.extract_features(market_data)
features = features.reshape(1, -1)
predictions = await predictor.predict_scenarios(features, market_data)

# Access scenario probabilities
probabilities = predictions["probabilities"]
scenario_analysis = predictions["scenario_analysis"]
confidence = predictions["confidence"]
```

## Decision Logic

### Entry Decision Logic

The enhanced entry decision combines both systems:

1. **Multi-Output System Requirements**:
   - Green light signal from existing multi-output system
   - Base confidence above threshold

2. **Scenario Analysis Enhancement**:
   - Combined profit zone probability > threshold (default: 0.6)
   - Combined risk zone probability < threshold (default: 0.2)
   - Scenario confidence > threshold (default: 0.7)

3. **Final Decision**:
   - Entry signal = Multi-output GREEN AND Scenario analysis FAVORABLE
   - Enhanced confidence = Base confidence * 1.2 (capped at 1.0)

### Exit Decision Logic

For open positions, monitor scenario analysis:

1. **Risk Zone Monitoring**:
   - If risk zone probability > exit threshold (default: 0.5)
   - Generate exit signal

2. **Confidence Monitoring**:
   - If scenario confidence drops significantly
   - Consider reducing position size

## Feature Engineering

The scenario predictor extracts 15 features:

1. **Price Momentum** (3 features):
   - 5-period momentum
   - 10-period momentum
   - 20-period momentum

2. **Volatility** (3 features):
   - 5-period volatility
   - 10-period volatility
   - 20-period volatility

3. **Volume** (2 features):
   - Volume trend
   - Volume moving average ratio

4. **Technical Indicators** (4 features):
   - RSI (normalized to 0-1)
   - Moving average ratio
   - Price range
   - Upper/lower shadows

5. **Additional Features** (3 features):
   - Latest return
   - Price range ratio
   - Shadow ratios

## Training Process

### Scenario Labeling

1. **Look-Ahead Analysis**: For each data point, look ahead up to `time_limit_minutes`
2. **Scenario Detection**: Determine which scenario occurs first
3. **Label Assignment**: Assign the first-occurring scenario as the label

### Model Training

1. **Feature Extraction**: Extract 15 features from market data
2. **Target Preparation**: Generate scenario labels using look-ahead analysis
3. **Model Training**: Train LightGBM classifier with multi-class configuration
4. **Performance Metrics**: Calculate accuracy, log loss, and feature importance

## Integration with Existing Systems

### Backward Compatibility

The implementation maintains full backward compatibility:

1. **Existing Multi-Output System**: Continues to work unchanged
2. **Enhanced Predictions**: New method that combines both systems
3. **Gradual Migration**: Can be enabled/disabled via configuration
4. **Fallback Mechanisms**: Graceful degradation if scenario analysis fails

### Step17 Optimization Integration

All parameters are exposed for step17 optimization:

1. **Scenario Parameters**: Profit targets, stop losses, time limits
2. **Model Parameters**: LightGBM hyperparameters
3. **Decision Thresholds**: Entry/exit criteria
4. **Feature Parameters**: Technical indicator periods

## Error Handling and Fallbacks

### Robust Error Handling

1. **Configuration Validation**: Validates all parameters on initialization
2. **Feature Extraction**: Handles insufficient data gracefully
3. **Model Training**: Validates training data and model performance
4. **Prediction Fallbacks**: Returns fallback predictions if model fails

### Fallback Mechanisms

1. **Untrained Model**: Returns heuristic-based predictions
2. **Invalid Configuration**: Uses default parameters
3. **Insufficient Data**: Returns neutral scenario predictions
4. **Model Errors**: Returns fallback with error metadata

## Testing

### Comprehensive Test Suite

**Location**: `test_scenario_based_tactician_implementation.py`

**Test Coverage**:
1. **ScenarioBasedPredictor**: Core functionality and configuration
2. **Enhanced MLTacticsManager**: Integration and decision logic
3. **Optimization Parameters**: Step17 parameter configurability
4. **Integration**: Backward compatibility with existing systems
5. **Error Handling**: Fallback mechanisms and error recovery

### Running Tests

```bash
python test_scenario_based_tactician_implementation.py
```

## Performance Considerations

### Computational Efficiency

1. **Feature Extraction**: Optimized for real-time processing
2. **Model Inference**: LightGBM provides fast predictions
3. **Memory Usage**: Minimal memory footprint
4. **Training Time**: Configurable model complexity

### Scalability

1. **Parallel Processing**: Supports concurrent predictions
2. **Model Persistence**: Can save/load trained models
3. **Incremental Training**: Supports model updates
4. **Resource Management**: Efficient memory and CPU usage

## Future Enhancements

### Potential Improvements

1. **Dynamic Scenarios**: Adaptive scenario definitions based on market conditions
2. **Multi-Timeframe**: Scenario analysis across multiple timeframes
3. **Advanced Features**: Additional technical indicators and market microstructure features
4. **Ensemble Methods**: Combine multiple scenario models
5. **Real-Time Adaptation**: Online learning for scenario parameters

### Optimization Opportunities

1. **Hyperparameter Tuning**: Automated optimization of all parameters
2. **Feature Selection**: Dynamic feature importance and selection
3. **Model Architecture**: Experiment with different ML algorithms
4. **Scenario Refinement**: Data-driven scenario definition optimization

## Conclusion

The scenario-based Tactician implementation provides a sophisticated enhancement to the existing prediction system while maintaining full backward compatibility. All parameters are configurable for step17 optimization, making it ready for production deployment and continuous improvement.

The implementation successfully addresses the original plan requirements:
- ✅ Probabilistic scenario analysis with 6 distinct scenarios
- ✅ Enhanced entry/exit decision logic
- ✅ All parameters configurable for step17 optimization
- ✅ Backward compatibility with existing systems
- ✅ Robust error handling and fallback mechanisms
- ✅ Comprehensive testing and validation