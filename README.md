# Ares Trading System

## Overview

Ares is an advanced algorithmic trading system that uses machine learning and quantitative analysis to identify and execute profitable trading opportunities. The system has been completely refactored to use an autonomous step-based architecture for improved modularity, maintainability, and performance.

## Key Features

### 🤖 **Autonomous Step Architecture**
- **Modular Design**: Each processing step is self-contained and can be executed independently
- **Step Registry**: Global registry for step discovery and execution
- **Automatic Artifact Management**: Centralized storage with automatic format conversion
- **Simplified Launcher**: Easy-to-use command-line interface for step execution

### 📊 **Comprehensive Trading Pipeline**
- **Data Collection**: Automated data downloading and processing from multiple exchanges
- **Market Analysis**: Support/Resistance detection, regime analysis, and market structure analysis
- **Feature Engineering**: Advanced feature generation and selection for machine learning models
- **Model Training**: Analyst and Tactician models with ensemble methods
- **Backtesting**: Comprehensive parameter optimization and strategy validation

### 🚀 **Advanced ML Capabilities**
- **Multi-Model Architecture**: Separate Analyst (WHAT to trade) and Tactician (WHEN to trade) models
- **Regime-Aware Training**: Models trained specifically for different market conditions
- **Ensemble Methods**: Advanced ensemble techniques for improved prediction accuracy
- **Hyperparameter Optimization**: Automated optimization using Bayesian methods

### ⚡ **Performance Optimized**
- **M1/M2/M3 Mac Optimization**: Hardware-accelerated processing for Apple Silicon
- **VectorBT Integration**: High-performance vectorized backtesting
- **Automatic CSV Export**: Smart format conversion for small datasets
- **Memory Management**: Intelligent memory optimization and caching

## Architecture

### Step-Based System

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DATA_COLLECTION   │    │   MARKET_ANALYSIS   │    │   PRE_TRAINING   │
│                     │    │                     │    │                     │
│ • data_download     │    │ • sr_detection      │    │ • data_validation  │
│ • data_conversion   │    │ • sr_clustering     │    │ • feature_gen      │
│ • data_validation   │    │ • sr_optimization   │    │ • feature_selection│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────┐    ┌─────────────────┐
│  MODEL_TRAINING  │    │   BACKTESTING   │
│                     │    │                     │
│ • analyst_training  │    │ • final_params      │
│ • tactician_training│    │ • real_params       │
│ • ensemble_training │    │ • strategy_validation│
└─────────────────┘    └─────────────────┘
```

### Core Components

1. **BaseStep**: Abstract base class for all processing steps
2. **Artifact Manager**: Centralized data storage and retrieval
3. **Step Registry**: Global step discovery and execution
4. **Ares Launcher**: Command-line interface for system operation

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Ares

# Install dependencies
pip install -r requirements.txt

# Set up configuration
cp config/environments/development.json.example config/environments/development.json
```

### Basic Usage

```bash
# Run a single step
python ares_launcher.py step data_download --symbol ETHUSDT --timeframe 15m --direction longs

# Run multiple steps
python ares_launcher.py steps sr_detection,sr_clustering --symbol ETHUSDT --timeframe 15m --direction longs

# Run an entire stage
python ares_launcher.py stage DATA_COLLECTION --symbol ETHUSDT --timeframe 15m

# Run complete pipeline
python ares_launcher.py pipeline --symbol ETHUSDT --timeframe 15m --direction longs
```

### Configuration

The system uses configuration files to manage parameters:

```yaml
# config/training_config.json
{
  "symbol": "ETHUSDT",
  "exchange": "binance",
  "timeframe": "15m",
  "direction": "longs",
  "execution_mode": "full"
}
```

## Available Steps

### DATA_COLLECTION
- **data_download**: Download raw market data from exchanges
- **data_conversion**: Convert and standardize data formats
- **data_validation**: Validate data quality and integrity

### MARKET_ANALYSIS
- **sr_detection**: Detect Support/Resistance levels
- **sr_clustering**: Group S/R levels into clusters
- **sr_parameter_optimization**: Optimize S/R detection parameters

### PRE_TRAINING
- **feature_generation_data_validation_step**: Enhanced data validation
- **feature_generation_period_lookback_optimization_step**: Optimize time periods
- **feature_generation_feature_generation_step**: Generate trading features
- **feature_generation_feature_selection_step**: Select optimal features

### MODEL_TRAINING
- **analyst_models_training**: Train Analyst models
- **tactician_models_training**: Train Tactician models
- **analyst_ensemble_training**: Train Analyst ensemble models
- **tactician_ensemble_training**: Train Tactician ensemble models

### BACKTESTING
- **final_parameters_optimization**: Optimize final system parameters
- **real_parameters_optimization**: Optimize real trading parameters

## Artifact Management

### Automatic Format Generation
- **Parquet Files**: Primary format for all data (always generated)
- **CSV Files**: Automatically generated for DataFrames with < 2000 rows
- **Compression**: Automatic compression for large datasets
- **Metadata**: Automatic metadata tracking and versioning

### Example Usage
```python
# Save artifact (auto-generates Parquet + CSV if applicable)
artifact_path = self._save_artifact(data, 'my_artifact', 'data')

# Retrieve artifact
data = self._get_artifact('my_artifact', 'data')
```

## Development

### Creating New Steps

1. **Inherit from BaseStep**:
```python
from src.training.steps.base_step import BaseStep

class MyStep(BaseStep):
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Your step logic here
        pass
```

2. **Register the step**:
```python
# In __init__.py
from src.training.steps.base_step import step_registry
step_registry.register("my_step", MyStep)
```

3. **Run the step**:
```bash
python ares_launcher.py step my_step --symbol ETHUSDT --timeframe 15m
```

### Documentation
- [Step Development Guide](docs/step_development_guide.md)
- [Artifact Management Guide](docs/artifact_management.md)
- [Step Reference Guide](docs/step_reference.md)

## Performance

### Execution Modes

#### Light Mode
- Reduced dataset sizes
- Simplified algorithms
- Faster execution
- Lower resource usage

#### Full Mode
- Complete dataset processing
- Full algorithm complexity
- Maximum accuracy
- Production-ready results

### Resource Requirements
- **Memory**: 2-8GB RAM depending on dataset size
- **Storage**: 1-10GB per symbol/timeframe combination
- **CPU**: Multi-core processing for optimization steps
- **GPU**: Optional acceleration for ML training steps

## Monitoring and Logging

### Logging
The system provides comprehensive logging at multiple levels:

```bash
# Enable debug logging
python ares_launcher.py step <step_name> --symbol ETHUSDT --timeframe 15m --debug

# Verbose output
python ares_launcher.py step <step_name> --symbol ETHUSDT --timeframe 15m --verbose
```

### Artifact Monitoring
```python
# Check artifact statistics
stats = self.artifact_manager.get_artifact_stats()

# Clean up old artifacts
self.artifact_manager.cleanup_old_artifacts(days=30)
```

## Troubleshooting

### Common Issues

#### Step Not Found
```bash
# List available steps
python ares_launcher.py list-steps

# Check step information
python ares_launcher.py step-info <step_name>
```

#### Insufficient Data
```bash
# Check data availability
python ares_launcher.py check-data --symbol ETHUSDT --timeframe 15m

# Run data collection first
python ares_launcher.py step data_download --symbol ETHUSDT --timeframe 15m
```

#### Memory Issues
```bash
# Use light mode for large datasets
python ares_launcher.py step <step_name> --symbol ETHUSDT --timeframe 15m --execution-mode light
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings
- Write tests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Check the documentation in the `docs/` directory
- Review the step reference guide
- Check existing issues in the repository
- Create a new issue if needed

## Changelog

### Version 2.0.0 (Current)
- **BREAKING**: Complete refactoring to autonomous step architecture
- **NEW**: Step registry and simplified launcher
- **NEW**: Automatic artifact management with CSV export
- **NEW**: Comprehensive documentation
- **IMPROVED**: Performance optimization for M1/M2/M3 Macs
- **IMPROVED**: Enhanced error handling and logging

### Version 1.x (Legacy)
- Original pipeline-based architecture
- Complex sub-pipeline system
- Manual artifact management
- Limited documentation

---

**Note**: This system is for educational and research purposes. Always test thoroughly before using with real money.
