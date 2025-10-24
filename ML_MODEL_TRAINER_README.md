# ML Model Trainer - Unified Pipeline

A single, unified pipeline for training all ML models with comprehensive configuration management, feature engineering, validation, and analysis capabilities.

## Overview

The ML Model Trainer provides a single interface for training four types of ML models:
- **Analyst Base Models** - Market analysis and regime detection
- **Analyst Ensemble Models** - Combined analyst models for enhanced performance
- **Tactician Base Models** - Entry/exit timing and position sizing
- **Tactician Ensemble Models** - Combined tactician models for enhanced performance

## Key Features

### Unified Pipeline
- Single entry point for all model training
- Consistent interface across all model types
- Shared infrastructure for common tasks

### Configuration-Driven
- 4 separate config files for each model type
- YAML-based configuration for easy modification
- Comprehensive parameter specification

### Comprehensive Management
- **Feature Engineering**: Automated feature creation and selection
- **Data Preprocessing**: Data cleaning, validation, and preparation
- **Cross-Validation**: Time-series aware validation strategies
- **Hyperparameter Optimization**: Automated HPO with Optuna
- **Data Leakage Detection**: Comprehensive leakage prevention
- **Metrics Analysis**: Multiple evaluation metrics and analysis
- **SHAP Analysis**: Model interpretability and feature importance
- **Model Evaluation**: Comprehensive model assessment

### Performance Optimized
- Parallel training support
- Hardware optimization
- Memory management
- GPU acceleration support

## Configuration Files

### 1. Analyst Base Config (`analyst_base_config.yaml`)
```yaml
model_type: "analyst_base"
timeframe: "15m"

# ML Models to Train
models:
  - name: "lightgbm"
    type: "LIGHTGBM"
    enabled: true
    parameters:
      objective: "binary"
      metric: "binary_logloss"
      # ... more parameters

# Target Configuration
targets:
  primary: "analyst_target"  # Binary classification
  secondary: "analyst_confidence"  # Confidence score

# Input Configuration
inputs:
  base_features:
    - "price_features"
    - "volume_features"
    - "technical_indicators"
  analyst_features:
    enable_patchtst_features: true
    enable_regime_features: true
    enable_multi_timeframe: true
```

### 2. Analyst Ensemble Config (`analyst_ensemble_config.yaml`)
```yaml
model_type: "analyst_ensemble"

# Ensemble Models
models:
  - name: "voting_ensemble"
    type: "VOTING"
    enabled: true
  - name: "stacking_ensemble"
    type: "STACKING"
    enabled: true

# Base Models for Ensemble
base_models:
  - name: "lightgbm"
    type: "LIGHTGBM"
  - name: "catboost"
    type: "CATBOOST"
  - name: "xgboost"
    type: "XGBOOST"
```

### 3. Tactician Base Config (`tactician_base_config.yaml`)
```yaml
model_type: "tactician_base"

# ML Models to Train
models:
  - name: "lightgbm"
    type: "LIGHTGBM"
    enabled: true
  - name: "neural_network"
    type: "NEURAL_NETWORK"
    enabled: true

# Target Configuration
targets:
  primary: "entry_timing"  # Entry timing signal
  secondary: "exit_timing"  # Exit timing signal
  tertiary: "position_sizing"  # Position sizing signal

# Input Configuration
inputs:
  tactician_features:
    enable_entry_timing: true
    enable_exit_timing: true
    enable_position_sizing: true
```

### 4. Tactician Ensemble Config (`tactician_ensemble_config.yaml`)
```yaml
model_type: "tactician_ensemble"

# Ensemble Models
models:
  - name: "stacking_ensemble"
    type: "STACKING"
    enabled: true

# Base Models for Ensemble
base_models:
  - name: "lightgbm"
    type: "LIGHTGBM"
  - name: "catboost"
    type: "CATBOOST"
  - name: "neural_network"
    type: "NEURAL_NETWORK"
```

## Usage

### Command Line Interface

```bash
# Train all models with default configs
python src/training/cli_ml_model_trainer.py --timeframe 15m

# Train specific model types
python src/training/cli_ml_model_trainer.py --model-types analyst_base tactician_base

# Use custom config directory
python src/training/cli_ml_model_trainer.py --config-dir custom_configs/

# Enable parallel training
python src/training/cli_ml_model_trainer.py --parallel --max-workers 8

# Verbose output
python src/training/cli_ml_model_trainer.py --verbose
```

### Python API

```python
import asyncio
from src.training.ml_model_trainer import MLModelTrainer, MLModelTrainerConfig, ModelType

async def main():
    # Create configuration
    config = MLModelTrainerConfig(
        model_types=[
            ModelType.ANALYST_BASE,
            ModelType.TACTICIAN_BASE
        ],
        timeframe="15m",
        enable_parallel_training=True,
        max_workers=4
    )
    
    # Create trainer
    trainer = MLModelTrainer(config)
    
    # Define config paths
    config_paths = {
        ModelType.ANALYST_BASE: "config/ml_model_trainer/analyst_base_config.yaml",
        ModelType.TACTICIAN_BASE: "config/ml_model_trainer/tactician_base_config.yaml"
    }
    
    # Prepare data
    data = {
        'features': your_features,
        'targets': your_targets,
        'metadata': your_metadata
    }
    
    # Train models
    results = await trainer.train_models(data, config_paths)
    
    # Process results
    for model_type, model_results in results.items():
        for result in model_results:
            if result.success:
                print(f"{result.model_name}: {result.metrics}")

# Run
asyncio.run(main())
```

### Example Usage

```bash
# Run the example script
python examples/ml_model_trainer_example.py
```

## Configuration Options

### Model Configuration
- **Model Types**: Specify which models to train
- **Model Parameters**: Detailed parameter configuration for each model
- **Ensemble Methods**: Voting, Stacking, Averaging, Blending
- **Base Models**: Configuration for ensemble base models

### Target Configuration
- **Primary Targets**: Main prediction targets
- **Secondary Targets**: Additional targets (confidence, etc.)
- **Target Weights**: Weighting for multi-target scenarios
- **Target Types**: Classification, regression, multi-output

### Input Configuration
- **Base Features**: Common features across all models
- **Model-Specific Features**: Features specific to each model type
- **Regime Outputs**: Regime model outputs from previous steps
- **Previous Model Outputs**: Outputs from previous model training

### Training Configuration
- **Validation Split**: Ratio for validation data
- **Cross-Validation**: CV strategy and parameters
- **Early Stopping**: Early stopping configuration
- **Hyperparameter Optimization**: HPO method and parameters

### Feature Engineering
- **PatchTST Features**: Time series transformer features
- **Regime Features**: Market regime detection features
- **Multi-Timeframe Features**: Features across multiple timeframes
- **Timing Features**: Entry/exit timing features
- **Position Sizing Features**: Position sizing features

### Data Quality & Leakage Prevention
- **Leakage Detection**: Multiple leakage detection methods
- **Data Validation**: Comprehensive data quality checks
- **Outlier Detection**: Outlier detection and handling
- **Temporal Consistency**: Time-series specific validation

### Metrics & Evaluation
- **Primary Metrics**: Main evaluation metrics
- **Secondary Metrics**: Additional evaluation metrics
- **SHAP Analysis**: Model interpretability analysis
- **Feature Importance**: Feature importance calculation
- **Trading Metrics**: Trading-specific performance metrics

### Performance Monitoring
- **Training Monitoring**: Real-time training monitoring
- **Model Performance**: Model performance tracking
- **Resource Usage**: Memory and GPU usage tracking
- **Ensemble Monitoring**: Ensemble-specific monitoring

### Output Configuration
- **Model Artifacts**: Model saving configuration
- **Results**: Prediction and analysis results
- **Reports**: Comprehensive HTML reports
- **Checkpoints**: Model checkpointing

## Architecture

```
ML Model Trainer Pipeline
├── Configuration Management
│   ├── Analyst Base Config
│   ├── Analyst Ensemble Config
│   ├── Tactician Base Config
│   └── Tactician Ensemble Config
├── Data Processing
│   ├── Data Loading
│   ├── Data Preprocessing
│   ├── Feature Engineering
│   └── Target Generation
├── Model Training
│   ├── Analyst Base Training
│   ├── Analyst Ensemble Training
│   ├── Tactician Base Training
│   └── Tactician Ensemble Training
├── Validation & Analysis
│   ├── Cross-Validation
│   ├── Hyperparameter Optimization
│   ├── Data Leakage Detection
│   ├── Metrics Analysis
│   └── SHAP Analysis
└── Output & Reporting
    ├── Model Artifacts
    ├── Predictions
    ├── Feature Importance
    └── Reports
```

## Benefits

### Single Pipeline
- **Unified Interface**: One pipeline for all model types
- **Consistent Behavior**: Same validation, monitoring, and reporting
- **Easy Maintenance**: Single codebase to maintain
- **Resource Efficiency**: Shared infrastructure and parallel execution

### Configuration-Driven
- **Flexible Configuration**: Easy to modify without code changes
- **Model-Specific Settings**: Tailored configuration for each model type
- **Parameter Management**: Centralized parameter management
- **Easy Experimentation**: Quick configuration changes for experiments

### Comprehensive Management
- **Automated Pipeline**: Minimal manual intervention required
- **Quality Assurance**: Built-in data quality and leakage detection
- **Performance Monitoring**: Real-time monitoring and optimization
- **Comprehensive Analysis**: Detailed metrics and interpretability analysis

### Production Ready
- **Error Handling**: Robust error handling and recovery
- **Logging**: Comprehensive logging and monitoring
- **Scalability**: Parallel execution and resource optimization
- **Reproducibility**: Consistent results with proper seeding

## Dependencies

- Python 3.8+
- scikit-learn
- pandas
- numpy
- optuna (for hyperparameter optimization)
- shap (for model interpretability)
- yaml
- asyncio

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install optional dependencies
pip install optuna shap
```

## Examples

See the `examples/` directory for comprehensive usage examples:
- `ml_model_trainer_example.py` - Basic usage examples
- `cli_ml_model_trainer.py` - Command line interface

## Configuration Reference

For detailed configuration options, see the individual config files:
- `config/ml_model_trainer/analyst_base_config.yaml`
- `config/ml_model_trainer/analyst_ensemble_config.yaml`
- `config/ml_model_trainer/tactician_base_config.yaml`
- `config/ml_model_trainer/tactician_ensemble_config.yaml`

## Support

For questions, issues, or contributions, please refer to the project documentation or contact the development team.