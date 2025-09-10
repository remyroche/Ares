# Production Configuration Templates

This directory contains production-ready configuration templates for the comprehensive training pipeline.

## Available Configurations

### Production (`production_config.json`)
- **Purpose**: Production environment with full features and optimized performance
- **Resources**: High (16 workers, 90% memory, GPU enabled)
- **Features**: All features enabled, comprehensive monitoring, security
- **Use Case**: Live trading, production deployment

### Development (`development_config.json`)
- **Purpose**: Development environment with debugging enabled
- **Resources**: Medium (4 workers, 60% memory, no GPU)
- **Features**: Most features enabled, debugging tools, reduced resource usage
- **Use Case**: Development, testing, experimentation

### Testing (`testing_config.json`)
- **Purpose**: Testing environment with minimal resource usage
- **Resources**: Low (1 worker, 30% memory, no GPU)
- **Features**: Basic features only, fast execution
- **Use Case**: Unit tests, integration tests, CI/CD

## Usage

```python
from config_loader import load_config, validate_config

# Load configuration
config = load_config('production')

# Validate configuration
validate_config(config)

# Use with pipeline
from src.training.steps.comprehensive_training_pipeline import ComprehensiveTrainingPipeline
pipeline = ComprehensiveTrainingPipeline(config)
```

## Configuration Structure

Each configuration includes:

- **Basic Settings**: Symbol, exchange, timeframes, directories
- **Performance Settings**: Workers, memory, GPU, timeouts
- **Model Training**: Training parameters, validation, optimization
- **Evaluation**: Metrics, backtesting, validation methods
- **Feature Engineering**: Feature types, selection methods, thresholds
- **Data Quality**: Validation, cleaning, drift detection
- **Performance**: Optimization, caching, parallel processing
- **Monitoring**: Performance monitoring, alerting, notifications
- **Security**: Encryption, audit logging, access control

## Customization

To create a custom configuration:

1. Copy an existing configuration file
2. Modify the parameters as needed
3. Validate the configuration
4. Use with the pipeline

## Best Practices

- **Production**: Use production config for live trading
- **Development**: Use development config for experimentation
- **Testing**: Use testing config for automated tests
- **Validation**: Always validate configurations before use
- **Monitoring**: Enable monitoring in production environments
- **Security**: Enable security features in production
