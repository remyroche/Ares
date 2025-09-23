# Tactician Pre-ML Training Orchestrator

## Overview

The Tactician Pre-ML Training Orchestrator implements a comprehensive dual-directional training pipeline that separates long and short signals from the Analyst, applies full feature optimization, and trains separate Tactician models for each direction. This approach provides superior performance by optimizing features and training models specifically for each trading direction.

## Key Features

### 🎯 Dual-Directional Training
- **Signal Separation**: Automatically separates long and short signals from Analyst outputs
- **Confidence Filtering**: Only uses signals with confidence >= 0.5
- **Subsequent Data Inclusion**: Includes 45 minutes of data after each signal
- **Directional Optimization**: Different feature optimization for longs vs shorts

### 🔧 Feature Optimization Pipeline
- **Feature Lookback Optimization**: Optimizes lookback periods for each direction
- **PID-based Feature Generation**: Generates directional features using PID controllers
- **Multi-Horizon Profit Labeling**: Creates direction-specific profit targets and horizons
- **Final Feature Selection**: Selects optimal features for each direction

### 🎯 Model Training
- **Base Models**: Trains XGBoost, RandomForest, and CatBoost for each direction
- **Ensemble Models**: Creates stacking and voting ensembles for each direction
- **Directional Performance**: Separate performance metrics for long and short models

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Market Data   │    │  Analyst Outputs │    │   Confidence    │
│                 │    │                  │    │   Threshold     │
└─────────┬───────┘    └─────────┬────────┘    └─────────────────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │   Signal Separation       │
                    │  - Long signals (≥0.5)    │
                    │  - Short signals (≥0.5)   │
                    │  - 45min subsequent data  │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │  Feature Optimization     │
                    │  - Lookback optimization  │
                    │  - PID feature generation │
                    │  - Horizon labeling       │
                    │  - Feature selection      │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │    Model Training         │
                    │  - Base models (long)     │
                    │  - Base models (short)    │
                    │  - Ensemble models        │
                    └───────────────────────────┘
```

## Components

### 1. TacticianPreMLOrchestrator
Main orchestrator class that coordinates the entire pipeline:
- Signal separation with confidence filtering
- Feature optimization for each direction
- Model training for both directions
- Result aggregation and reporting

### 2. TacticianPreMLIntegrationStep
Integration component that bridges the orchestrator with the existing sub_pipeline:
- Sub-pipeline compatibility
- Data extraction and validation
- Result processing for backward compatibility

### 3. Signal Separation
Separates Analyst signals into long and short categories:
- Confidence threshold filtering (≥0.5)
- Directional signal identification
- Subsequent data inclusion (45 minutes)
- Sample balancing and validation

### 4. Feature Optimization
Applies the full feature optimization pipeline:
- **Lookback Optimization**: Optimizes feature lookback periods
- **PID Feature Generation**: Creates directional features
- **Horizon Labeling**: Generates profit targets and labels
- **Feature Selection**: Selects optimal feature sets

### 5. Dual Training
Trains separate models for each direction:
- **Base Models**: XGBoost, RandomForest, CatBoost
- **Ensemble Models**: Stacking and voting ensembles
- **Performance Tracking**: Separate metrics for each direction

## Configuration

### TacticianPreMLConfig
```python
@dataclass
class TacticianPreMLConfig:
    # Signal filtering
    confidence_threshold: float = 0.5
    subsequent_minutes: int = 45
    
    # Feature optimization settings
    enable_lookback_optimization: bool = True
    enable_pid_feature_generation: bool = True
    enable_horizon_labeling: bool = True
    enable_feature_selection: bool = True
    
    # Training settings
    enable_base_training: bool = True
    enable_ensemble_training: bool = True
    
    # Data processing
    max_samples_per_direction: Optional[int] = None
    enable_data_validation: bool = True
    enable_progress_logging: bool = True
    
    # Output settings
    save_intermediate_results: bool = True
    output_directory: str = "generated/tactician_pre_ml_training"
```

## Usage

### Basic Usage
```python
from src.training.steps.model_training.tactician_pre_ml_orchestrator import (
    TacticianPreMLOrchestrator, TacticianPreMLConfig
)

# Create configuration
config = TacticianPreMLConfig(
    confidence_threshold=0.5,
    subsequent_minutes=45,
    enable_lookback_optimization=True,
    enable_pid_feature_generation=True,
    enable_horizon_labeling=True,
    enable_feature_selection=True,
    enable_base_training=True,
    enable_ensemble_training=True
)

# Create orchestrator
orchestrator = TacticianPreMLOrchestrator(config)

# Execute training
result = await orchestrator.execute_full_orchestration(market_data, analyst_outputs)
```

### Integration with Sub-Pipeline
```python
from src.training.steps.model_training.tactician_pre_ml_integration import (
    execute_tactician_pre_ml_training_integration
)

# Execute through sub-pipeline
results = await execute_tactician_pre_ml_training_integration(
    sub_pipeline=sub_pipeline,
    symbol="BTCUSDT",
    exchange="binance",
    timeframe="1m",
    data=market_data,
    analyst_outputs=analyst_outputs,
    config=config
)
```

## Results Structure

### TacticianPreMLResult
```python
@dataclass
class TacticianPreMLResult:
    long_training_result: TacticianTrainingResult
    short_training_result: TacticianTrainingResult
    signal_separation_result: SignalSeparationResult
    total_processing_time: float
    configuration: TacticianPreMLConfig
```

### Signal Separation Results
```python
@dataclass
class SignalSeparationResult:
    long_signals: pd.DataFrame
    short_signals: pd.DataFrame
    long_confidence_scores: np.ndarray
    short_confidence_scores: np.ndarray
    long_indices: np.ndarray
    short_indices: np.ndarray
    total_samples: int
    long_samples: int
    short_samples: int
    confidence_threshold: float
    separation_time: float
```

### Training Results
```python
@dataclass
class TacticianTrainingResult:
    direction: str  # 'long' or 'short'
    base_models: Dict[str, Any]
    ensemble_models: Dict[str, Any]
    training_metrics: Dict[str, Any]
    model_performance: Dict[str, Any]
    training_time: float
```

## Integration Points

### 1. Models Training Sub-Pipeline
The orchestrator is integrated as step 3 in the models_training sub_pipeline:
1. `analyst_model_training`
2. `analyst_ensemble_training`
3. **`tactician_pre_ml_training`** ← New step
4. `tactician_lookback_optimization`
5. `tactician_models_training`
6. `tactician_ensemble_training`

### 2. Existing Components
The orchestrator leverages existing components:
- **Feature Lookback Optimization**: From `market_analysis/feature_lookback_optimization/`
- **PID Feature Generation**: From `market_analysis/pid_based_feature_generation/`
- **Multi-Horizon Labeling**: From `market_analysis/multi_horizon_profit_labeler.py`
- **Feature Selection**: From `market_analysis/final_feature_selection_step.py`
- **Tactician Training**: From existing `tactician_models_training_refactored.py`

## Performance Benefits

### 1. Directional Specialization
- **Optimized Features**: Different feature sets for long vs short signals
- **Directional Labels**: Separate profit targets and horizons
- **Specialized Models**: Models trained specifically for each direction

### 2. Confidence-Based Filtering
- **Quality Signals**: Only uses high-confidence Analyst signals (≥0.5)
- **Reduced Noise**: Filters out low-confidence predictions
- **Better Training**: Higher quality training data

### 3. Comprehensive Feature Optimization
- **Lookback Optimization**: Optimal feature lookback periods
- **PID Features**: Advanced directional feature generation
- **Horizon Labeling**: Multi-horizon profit probability labeling
- **Feature Selection**: Optimal feature subset selection

## Output Files

The orchestrator generates comprehensive output files:

### Models
- `tactician_long_base/` - Long signal base models
- `tactician_short_base/` - Short signal base models
- `tactician_long_ensemble/` - Long signal ensemble models
- `tactician_short_ensemble/` - Short signal ensemble models

### Reports
- `orchestration_summary.json` - Complete orchestration summary
- `signal_separation_report.json` - Signal separation results
- `feature_optimization_report.json` - Feature optimization results
- `training_performance_report.json` - Training performance metrics

### Metrics
- Signal separation statistics
- Feature optimization metrics
- Model training performance
- Directional performance comparison

## Error Handling

The orchestrator includes comprehensive error handling:
- **Signal Validation**: Validates Analyst outputs and market data
- **Feature Validation**: Ensures feature optimization succeeds
- **Training Validation**: Validates model training results
- **Graceful Degradation**: Continues with available data if some components fail

## Monitoring and Logging

Comprehensive logging and monitoring:
- **Progress Tracking**: Detailed progress logs for each step
- **Performance Metrics**: Real-time performance monitoring
- **Error Reporting**: Detailed error reporting with context
- **Summary Reports**: Comprehensive summary reports

## Future Enhancements

### Planned Improvements
1. **Dynamic Confidence Thresholds**: Adaptive confidence thresholds based on market conditions
2. **Advanced Feature Engineering**: Additional feature engineering techniques
3. **Model Architecture Optimization**: Advanced model architectures for each direction
4. **Real-time Adaptation**: Real-time model adaptation based on performance

### Integration Opportunities
1. **Advanced Analytics**: Integration with advanced analytics tools
2. **Performance Monitoring**: Real-time performance monitoring
3. **A/B Testing**: A/B testing framework for different configurations
4. **Automated Optimization**: Automated hyperparameter optimization

## Conclusion

The Tactician Pre-ML Training Orchestrator provides a comprehensive solution for dual-directional Tactician training with advanced feature optimization. By separating long and short signals, applying direction-specific feature optimization, and training specialized models for each direction, it delivers superior performance compared to traditional single-direction training approaches.

The orchestrator is fully integrated with the existing models_training sub_pipeline and provides backward compatibility while adding powerful new capabilities for directional trading strategy development.