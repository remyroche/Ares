# Ray-Based Model Trainer

A distributed model training system using Ray for CPU-bound model training and data processing tasks.

## Overview

The `RayModelTrainer` is designed to handle both analyst and tactician models with parallel processing capabilities. It replaces the previous asyncio-based implementation with Ray for better performance on CPU-bound tasks.

## Key Features

- **Distributed Training**: Uses Ray for parallel model training across multiple CPUs
- **Multiple Model Types**: Supports both analyst (multi-timeframe) and tactician (1m) models
- **Feature Engineering**: Comprehensive feature generation from OHLCV data
- **Model Persistence**: Automatic model and scaler storage with metadata
- **Error Handling**: Robust error handling with detailed logging
- **Configuration Driven**: Flexible configuration system

## Architecture

### Core Components

1. **RayModelTrainer**: Main trainer class that orchestrates the training process
2. **ModelConfig**: Configuration dataclass for model training parameters
3. **TrainingData**: Container for training data with features and labels
4. **Ray Remote Functions**: Distributed training functions for parallel execution

### Training Flow

1. **Initialization**: Setup Ray cluster and validate configuration
2. **Data Preparation**: Generate synthetic data and features (replace with real data collection)
3. **Model Training**: Parallel training using Ray remote functions
4. **Model Storage**: Save models, scalers, and metadata
5. **Cleanup**: Shutdown Ray cluster

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Ray (if not already installed)
pip install ray
```

## Configuration

### Basic Configuration

```python
config = {
    "ray": {
        "num_cpus": 4,        # Number of CPUs to use
        "num_gpus": 0,        # Number of GPUs to use
        "logging_level": "info"
    },
    "model_trainer": {
        "enable_analyst_models": True,
        "enable_tactician_models": True,
        "model_directory": "models",
        "analyst_models": {
            "timeframes": ["1h", "15m", "5m", "1m"]
        },
        "tactician_models": {
            "timeframes": ["1m"]
        }
    }
}
```

### Model Configuration

```python
@dataclass
class ModelConfig:
    model_type: str          # "analyst" or "tactician"
    timeframe: str           # Timeframe (e.g., "1m", "1h")
    features: List[str]      # Feature column names
    target_column: str       # Target column name
    test_size: float = 0.2   # Test split ratio
    random_state: int = 42   # Random seed
    n_estimators: int = 100  # Number of estimators
    max_depth: int = 10      # Max tree depth
```

## Usage

### Basic Usage

```python
from src.training.model_trainer import setup_model_trainer

# Setup trainer
trainer = setup_model_trainer(config)

if trainer:
    # Training input
    training_input = {
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "timeframe": "1m",
        "lookback_days": 30
    }
    
    # Train models
    results = trainer.train_models(training_input)
    
    if results:
        print("Training completed successfully!")
        print(f"Analyst models: {len(results.get('analyst_models', {}))}")
        print(f"Tactician models: {len(results.get('tactician_models', {}))}")
    
    # Cleanup
    trainer.stop()
```

### Advanced Usage

```python
# Check training status
status = trainer.get_training_status()
print(f"Training status: {status}")

# Load trained models
model, scaler = trainer.load_model("analyst", "1m")
if model and scaler:
    # Use model for predictions
    predictions = model.predict(scaler.transform(features))
```

## Features

### Feature Engineering

The trainer generates comprehensive features from OHLCV data:

- **Price-based features**: Price changes, high-low ratios, open-close ratios
- **Moving averages**: 5, 10, 20 period moving averages
- **Volatility features**: Rolling standard deviations
- **Volume features**: Volume moving averages and ratios
- **Technical indicators**: RSI, MACD

### Model Types

#### Analyst Models
- **Timeframes**: 1h, 15m, 5m, 1m
- **Algorithm**: RandomForestClassifier
- **Purpose**: Multi-timeframe analysis and trend prediction

#### Tactician Models
- **Timeframes**: 1m only
- **Algorithm**: GradientBoostingClassifier
- **Purpose**: High-frequency trading decisions

### Model Metrics

Each trained model includes comprehensive metrics:

- **Accuracy**: Overall prediction accuracy
- **Precision**: Precision score for positive predictions
- **Recall**: Recall score for positive predictions
- **F1 Score**: Harmonic mean of precision and recall
- **Cross-validation**: Mean and standard deviation of CV scores
- **Feature Importance**: Importance scores for all features

## File Structure

```
models/
├── model_metadata.json          # Model metadata
├── analyst_1h_20231201_120000.pkl
├── analyst_1h_20231201_120000_scaler.pkl
├── analyst_15m_20231201_120000.pkl
├── analyst_15m_20231201_120000_scaler.pkl
├── tactician_1m_20231201_120000.pkl
└── tactician_1m_20231201_120000_scaler.pkl
```

## Testing

Run the test script to verify the implementation:

```bash
python test_model_trainer.py
```

Expected output:
```
🚀 Testing Ray-based Model Trainer...
✅ Model trainer setup successful!
📊 Training status: {...}
🧠 Starting model training...
✅ Training completed successfully!
📈 Analyst models trained: 4
  - 1h: Accuracy=0.850, Precision=0.820, Recall=0.780
  - 15m: Accuracy=0.845, Precision=0.815, Recall=0.775
  - 5m: Accuracy=0.840, Precision=0.810, Recall=0.770
  - 1m: Accuracy=0.835, Precision=0.805, Recall=0.765
🎯 Tactician models trained: 1
  - 1m: Accuracy=0.880, Precision=0.850, Recall=0.820
🔍 Testing model loading...
✅ Successfully loaded analyst_1m model
✅ Successfully loaded tactician_1m model
🛑 Cleaning up...
✅ Cleanup completed!
```

## Performance Benefits

### Ray vs Asyncio

1. **CPU-bound Tasks**: Ray is optimized for CPU-intensive tasks like model training
2. **Parallel Processing**: True parallel execution across multiple cores
3. **Resource Management**: Better resource allocation and management
4. **Scalability**: Easy scaling across multiple machines
5. **Fault Tolerance**: Built-in fault tolerance and recovery

### Performance Comparison

| Metric | Asyncio | Ray |
|--------|---------|-----|
| CPU Utilization | Limited | Full |
| Parallel Execution | No | Yes |
| Memory Management | Basic | Advanced |
| Scalability | Single machine | Multi-machine |
| Fault Tolerance | Manual | Built-in |

## Error Handling

The trainer includes comprehensive error handling:

- **Configuration Validation**: Validates all configuration parameters
- **Ray Initialization**: Handles Ray cluster setup errors
- **Training Errors**: Captures and logs training failures
- **Storage Errors**: Handles model storage failures
- **Cleanup Errors**: Ensures proper resource cleanup

## Logging

The trainer provides detailed logging:

- **Initialization**: Ray setup and configuration validation
- **Training Progress**: Model training status and metrics
- **Error Logging**: Detailed error messages with context
- **Performance**: Training time and resource usage

## Future Enhancements

1. **Real Data Integration**: Replace synthetic data with real market data
2. **Advanced Models**: Support for deep learning models (PyTorch, TensorFlow)
3. **Hyperparameter Tuning**: Integration with Optuna for automated tuning
4. **Model Versioning**: Version control for trained models
5. **A/B Testing**: Framework for model comparison and testing
6. **Real-time Training**: Continuous model updates with new data

## Troubleshooting

### Common Issues

1. **Ray Initialization Failed**
   - Check available CPUs/GPUs
   - Verify Ray installation
   - Check system resources

2. **Training Failed**
   - Verify data quality
   - Check feature generation
   - Review model configuration

3. **Model Loading Failed**
   - Check model file paths
   - Verify model metadata
   - Ensure proper cleanup

### Debug Mode

Enable debug logging:

```python
config["ray"]["logging_level"] = "debug"
```

## Contributing

1. Follow the existing code structure
2. Add comprehensive error handling
3. Include unit tests for new features
4. Update documentation
5. Ensure Ray compatibility

## License

This implementation is part of the trading system project.