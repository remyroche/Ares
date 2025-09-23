# Advanced ML Architecture Integration Guide

## Overview

This guide provides step-by-step instructions for integrating the advanced ML architecture into your existing trading system. The architecture combines CLVSA, MultiScaleNBEATS, RegimeNAS, Meta-labels, and Regime-specific HPO for superior performance.

## Quick Start

### 1. Basic Integration

```python
from src.training.architectures import (
    create_integrated_ml_architecture,
    create_integrated_ml_trainer,
    IntegratedMLConfig
)

# Create configuration
config = IntegratedMLConfig(
    input_features=50,
    sequence_length=60,
    forecast_horizon=12,
    use_clvsa=True,
    use_multiscale_nbeats=True,
    use_regime_nas=True,
    use_meta_labels=True,
    use_hpo=True
)

# Create integrated architecture
model = create_integrated_ml_architecture(config)
trainer = create_integrated_ml_trainer(model, config)
```

### 2. Training Pipeline

```python
# Prepare data
X_train, y_train, regime_labels_train = prepare_training_data()
X_val, y_val, regime_labels_val = prepare_validation_data()

# Create data loaders
train_loader = create_data_loader(X_train, y_train, regime_labels_train)
val_loader = create_data_loader(X_val, y_val, regime_labels_val)

# Train the model
history = trainer.train(train_loader, val_loader, epochs=100)

# Optimize hyperparameters (optional)
if config.use_hpo:
    hpo_results = trainer.optimize_hyperparameters(X_train, y_train, regime_labels_train)
```

### 3. Inference

```python
# Make predictions
with torch.no_grad():
    predictions = model(X_test, regime_ids_test)

# Extract results
forecast = predictions['prediction']
uncertainty = predictions['uncertainty']
regime_prediction = predictions['regime_prediction']
```

## Component-Specific Integration

### CLVSA Architecture

```python
from src.training.architectures.clvsa_architecture import CLVSAArchitecture, CLVSAConfig

# Create CLVSA model
clvsa_config = CLVSAConfig(
    input_features=50,
    sequence_length=60,
    num_regimes=3,
    conv_filters=[32, 64, 128],
    lstm_hidden_size=128,
    attention_heads=8
)

clvsa_model = CLVSAArchitecture(clvsa_config)
```

### MultiScaleNBEATS

```python
from src.training.architectures.multiscale_nbeats import MultiScaleNBEATS, MultiScaleNBEATSConfig

# Create MultiScaleNBEATS model
nbeats_config = MultiScaleNBEATSConfig(
    input_features=50,
    sequence_length=60,
    forecast_horizon=12,
    scales=[1, 3, 6, 12],
    use_attention=True
)

nbeats_model = MultiScaleNBEATS(nbeats_config)
```

### RegimeNAS Framework

```python
from src.training.architectures.regime_nas_framework import RegimeNASFramework, RegimeNASConfig

# Create RegimeNAS model
regime_nas_config = RegimeNASConfig(
    input_features=50,
    sequence_length=60,
    regime_levels=[RegimeLevel.MICRO, RegimeLevel.SHORT, RegimeLevel.MEDIUM]
)

regime_nas_model = RegimeNASFramework(regime_nas_config)
```

### Meta-Labels and Patterns

```python
from src.training.architectures.meta_labels_patterns import MetaLabelsPatternsSystem, MetaLabelsConfig

# Create Meta-labels model
meta_labels_config = MetaLabelsConfig(
    input_features=50,
    sequence_length=60,
    pattern_types=[PatternType.TREND, PatternType.REVERSAL, PatternType.CONSOLIDATION],
    num_pattern_clusters=10
)

meta_labels_model = MetaLabelsPatternsSystem(meta_labels_config)
```

### Regime-Specific HPO

```python
from src.training.architectures.regime_specific_hpo import RegimeSpecificHPO, RegimeHPOConfig

# Create HPO system
hpo_config = RegimeHPOConfig(
    num_regimes=3,
    optimization_trials=100,
    model_types=['xgboost', 'lightgbm', 'catboost']
)

hpo_system = RegimeSpecificHPO(hpo_config)
```

## Configuration Options

### Integrated ML Configuration

```python
@dataclass
class IntegratedMLConfig:
    # Input configuration
    input_features: int = 50
    sequence_length: int = 60
    forecast_horizon: int = 12
    
    # Component toggles
    use_clvsa: bool = True
    use_multiscale_nbeats: bool = True
    use_regime_nas: bool = True
    use_meta_labels: bool = True
    use_hpo: bool = True
    
    # Ensemble configuration
    ensemble_method: str = 'weighted_average'
    ensemble_weights: Dict[str, float] = {
        'clvsa': 0.3,
        'multiscale_nbeats': 0.3,
        'regime_nas': 0.2,
        'meta_labels': 0.2
    }
```

### CLVSA Configuration

```python
@dataclass
class CLVSAConfig:
    input_features: int = 50
    sequence_length: int = 60
    num_regimes: int = 3
    
    # Convolutional layers
    conv_filters: List[int] = [32, 64, 128]
    conv_kernel_sizes: List[int] = [3, 5, 7]
    
    # LSTM configuration
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = True
    
    # Attention mechanism
    attention_heads: int = 8
    attention_dim: int = 64
    
    # Variational components
    latent_dim: int = 32
    kl_weight: float = 0.1
```

### MultiScaleNBEATS Configuration

```python
@dataclass
class MultiScaleNBEATSConfig:
    input_features: int = 50
    sequence_length: int = 60
    forecast_horizon: int = 12
    
    # Multi-scale configuration
    scales: List[int] = [1, 3, 6, 12]
    scale_weights: List[float] = [0.4, 0.3, 0.2, 0.1]
    
    # NBEATS blocks
    num_blocks: int = 3
    block_layers: List[int] = [256, 128, 64]
    
    # Basis functions
    basis_functions: List[str] = ['trend', 'seasonality', 'residual']
    trend_degree: int = 3
    seasonality_periods: List[int] = [12, 24, 48]
```

## Training Strategies

### 1. End-to-End Training

Train all components together with the integrated architecture:

```python
# Create integrated model
model = create_integrated_ml_architecture(config)
trainer = create_integrated_ml_trainer(model, config)

# Train end-to-end
history = trainer.train(train_loader, val_loader, epochs=100)
```

### 2. Component-Wise Training

Train each component separately:

```python
# Train CLVSA
clvsa_trainer = CLVSATrainer(clvsa_model, clvsa_config)
clvsa_history = clvsa_trainer.train(train_loader, val_loader, epochs=50)

# Train MultiScaleNBEATS
nbeats_trainer = MultiScaleNBEATSTrainer(nbeats_model, nbeats_config)
nbeats_history = nbeats_trainer.train(train_loader, val_loader, epochs=50)

# Train RegimeNAS
regime_nas_trainer = RegimeNASTrainer(regime_nas_model, regime_nas_config)
regime_nas_history = regime_nas_trainer.train(train_loader, val_loader, epochs=50)

# Train Meta-labels
meta_labels_trainer = MetaLabelsPatternsTrainer(meta_labels_model, meta_labels_config)
meta_labels_history = meta_labels_trainer.train(train_loader, val_loader, epochs=50)
```

### 3. Hyperparameter Optimization

```python
# Optimize hyperparameters for each regime
hpo_results = trainer.optimize_hyperparameters(X_train, y_train, regime_labels_train)

# Get optimized models
optimized_models = trainer.model.hpo_system.regime_models

# Use optimized models for prediction
predictions = trainer.model.hpo_system.predict_with_regime_models(X_test, regime_labels_test)
```

## Data Preparation

### 1. Time Series Data

```python
def prepare_time_series_data(data, sequence_length, forecast_horizon):
    """Prepare time series data for training."""
    sequences = []
    targets = []
    
    for i in range(sequence_length, len(data) - forecast_horizon + 1):
        sequence = data[i-sequence_length:i]
        target = data[i:i+forecast_horizon]
        
        sequences.append(sequence)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)
```

### 2. Regime Labels

```python
def create_regime_labels(data, num_regimes=3):
    """Create regime labels based on volatility."""
    volatility = data.rolling(20).std()
    
    # Classify into regimes based on volatility percentiles
    regime_labels = pd.cut(volatility, bins=num_regimes, labels=False)
    
    return regime_labels.values
```

### 3. Meta-Labels

```python
def create_meta_labels(data, pattern_types):
    """Create meta-labels from market patterns."""
    meta_labels = {}
    
    for pattern_type in pattern_types:
        if pattern_type == 'trend':
            meta_labels['trend'] = detect_trend_patterns(data)
        elif pattern_type == 'reversal':
            meta_labels['reversal'] = detect_reversal_patterns(data)
        elif pattern_type == 'consolidation':
            meta_labels['consolidation'] = detect_consolidation_patterns(data)
    
    return meta_labels
```

## Performance Monitoring

### 1. Training Metrics

```python
def monitor_training_metrics(history):
    """Monitor training metrics."""
    metrics = {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'learning_rate': history['lr']
    }
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Training Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(metrics['learning_rate'])
    plt.title('Learning Rate')
    
    plt.subplot(1, 3, 3)
    plt.plot(metrics['val_loss'])
    plt.title('Validation Loss')
    
    plt.tight_layout()
    plt.show()
```

### 2. Model Performance

```python
def evaluate_model_performance(model, test_loader):
    """Evaluate model performance."""
    model.eval()
    
    predictions = []
    targets = []
    uncertainties = []
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            
            predictions.append(outputs['prediction'])
            targets.append(target['prediction'])
            uncertainties.append(outputs['uncertainty'])
    
    # Calculate metrics
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    uncertainties = torch.cat(uncertainties, dim=0)
    
    mse = F.mse_loss(predictions, targets)
    mae = F.l1_loss(predictions, targets)
    
    return {
        'mse': mse.item(),
        'mae': mae.item(),
        'uncertainty_mean': uncertainties.mean().item(),
        'uncertainty_std': uncertainties.std().item()
    }
```

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Training Instability**
   - Reduce learning rate
   - Increase gradient clipping
   - Use learning rate scheduling

3. **Poor Performance**
   - Check data preprocessing
   - Verify regime labels
   - Tune hyperparameters

4. **Slow Training**
   - Use GPU acceleration
   - Optimize data loading
   - Use mixed precision

### Debugging Tips

1. **Check Data Quality**
   ```python
   # Verify data shapes and types
   print(f"X shape: {X.shape}")
   print(f"y shape: {y.shape}")
   print(f"Regime labels shape: {regime_labels.shape}")
   ```

2. **Monitor Gradients**
   ```python
   # Check for gradient issues
   for name, param in model.named_parameters():
       if param.grad is not None:
           print(f"{name}: {param.grad.norm()}")
   ```

3. **Validate Outputs**
   ```python
   # Check output validity
   with torch.no_grad():
       outputs = model(X_test)
       for key, value in outputs.items():
           print(f"{key}: {value.shape}, {torch.isfinite(value).all()}")
   ```

## Best Practices

### 1. Data Preparation
- Ensure proper time series alignment
- Handle missing values appropriately
- Normalize features consistently
- Create meaningful regime labels

### 2. Model Training
- Start with smaller models for testing
- Use validation sets for early stopping
- Monitor training metrics closely
- Save model checkpoints regularly

### 3. Hyperparameter Tuning
- Use regime-specific optimization
- Start with default parameters
- Use Bayesian optimization
- Validate on out-of-sample data

### 4. Production Deployment
- Test thoroughly before deployment
- Monitor performance in production
- Implement fallback mechanisms
- Update models regularly

## Conclusion

This integration guide provides comprehensive instructions for implementing the advanced ML architecture. The system combines state-of-the-art techniques for superior performance in financial ML applications.

For additional support, refer to the individual component documentation and test suites provided in the architecture modules.