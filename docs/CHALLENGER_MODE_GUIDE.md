# Challenger Mode Guide

## Overview

Challenger mode allows you to run paper trading using a challenger model instead of the production model. This is useful for testing new models in a safe environment before promoting them to production.

## Features

- **Safe Testing**: Paper trading with challenger models
- **Model Comparison**: Compare challenger vs production model performance
- **Easy Setup**: Simple configuration through state management
- **GUI Support**: Full GUI integration for challenger mode

## Usage

### Basic Challenger Mode

```bash
# Run challenger paper trading
python ares_launcher.py challenger --symbol ETHUSDT --exchange BINANCE

# With GUI
python ares_launcher.py challenger --symbol ETHUSDT --exchange BINANCE --gui
```

### GUI Mode

```bash
# Launch GUI with challenger mode
python ares_launcher.py gui --mode challenger --symbol ETHUSDT --exchange BINANCE
```

## Setup

### 1. Set Up Challenger Model

First, you need to set up a challenger model run ID:

```bash
# List available models
python scripts/setup_challenger_model.py --list-models

# Set up a challenger model
python scripts/setup_challenger_model.py --run-id <mlflow_run_id>

# Clear challenger model (if needed)
python scripts/setup_challenger_model.py --clear
```

### 2. Verify Model Status

The challenger model should have the following MLflow tags:
- `model_status`: Should be "candidate" or "challenger"
- `mlflow.runName`: Descriptive name for the model

## How It Works

### Model Selection

Challenger mode uses a different model selection process:

1. **Challenger Model**: Uses `challenger_model_run_id` from state
2. **MLflow Artifact**: Loads `model_challenger` artifact
3. **Fallback**: Returns HOLD signal if no challenger model is available

### State Management

The system tracks challenger models through:
- `challenger_model_run_id`: Stored in state manager
- `model_type`: Set to "challenger" in analyst
- Separate logging and monitoring

### Paper Trading

Challenger mode runs paper trading with:
- **No Real Money**: All trades are simulated
- **Robust Monitoring**: Full trade information and performance metrics
- **Model Tracking**: Clear identification of challenger model usage

## Architecture

### Components

1. **ModularAnalyst**: Supports `model_type` parameter
2. **AresPipeline**: Configurable for challenger mode
3. **StateManager**: Tracks `challenger_model_run_id`
4. **ModelManager**: Loads challenger models from MLflow

### Data Flow

```
Challenger Mode Flow:
1. User sets challenger_model_run_id
2. AresPipeline initializes with model_type="challenger"
3. ModularAnalyst uses challenger model for predictions
4. Paper trading executes with challenger signals
5. Performance metrics tracked separately
```

## Configuration

### State Variables

- `challenger_model_run_id`: MLflow run ID for challenger model
- `model_type`: Set to "challenger" in analyst

### MLflow Artifacts

- `model_challenger`: Challenger model artifact
- `model_live`: Production model artifact

## Monitoring

### Logs

Challenger mode provides detailed logging:
- Model loading status
- Prediction confidence
- Trade execution details
- Performance metrics

### Performance Tracking

Track challenger model performance:
- PnL comparison with production
- Sharpe ratio analysis
- Win rate statistics
- Risk metrics

## Best Practices

### 1. Model Selection

- Choose models with good backtesting performance
- Verify model artifacts are properly saved
- Test with small position sizes first

### 2. Monitoring

- Monitor challenger vs production performance
- Set up alerts for significant performance differences
- Track model drift and degradation

### 3. Promotion

- Only promote models that consistently outperform
- Use A/B testing for validation
- Maintain model version history

## Troubleshooting

### Common Issues

1. **No Challenger Model Found**
   - Verify `challenger_model_run_id` is set
   - Check MLflow run exists and is accessible
   - Ensure model artifacts are properly saved

2. **Model Loading Errors**
   - Verify MLflow tracking server is running
   - Check artifact paths are correct
   - Ensure model compatibility

3. **Performance Issues**
   - Monitor system resources
   - Check for model drift
   - Verify data quality

### Debug Commands

```bash
# Check challenger model status
python scripts/setup_challenger_model.py --list-models

# Verify state configuration
python -c "from src.utils.state_manager import StateManager; sm = StateManager(); print(sm.get_state('challenger_model_run_id'))"

# Test model loading
python -c "import mlflow; client = mlflow.tracking.MlflowClient(); print(client.get_run('<run_id>'))"
```

## Integration

### With Existing Workflows

Challenger mode integrates with:
- **Model Training**: Use trained models as challengers
- **Backtesting**: Validate challenger performance
- **Production**: Compare with live trading results
- **GUI**: Full dashboard support

### With A/B Testing

Challenger mode complements A/B testing:
- **Parallel Testing**: Run challenger alongside production
- **Performance Comparison**: Track relative performance
- **Safe Promotion**: Validate before production deployment

## Examples

### Complete Workflow

```bash
# 1. Train a new model
python ares_launcher.py model_trainer --symbol ETHUSDT --exchange BINANCE

# 2. Set up challenger model
python scripts/setup_challenger_model.py --run-id <new_model_run_id>

# 3. Test challenger mode
python ares_launcher.py challenger --symbol ETHUSDT --exchange BINANCE --gui

# 4. Monitor performance and promote if successful
```

### Batch Testing

```bash
# Test multiple symbols with challenger
for symbol in ETHUSDT BTCUSDT ADAUSDT; do
    python ares_launcher.py challenger --symbol $symbol --exchange BINANCE &
done
```

## Security

### Paper Trading Only

- Challenger mode only supports paper trading
- No real money is at risk
- All trades are simulated

### Model Isolation

- Challenger models are isolated from production
- No interference with live trading
- Separate state management

## Future Enhancements

### Planned Features

1. **Automated Model Selection**: Auto-select best challenger models
2. **Performance Alerts**: Real-time performance monitoring
3. **Model Versioning**: Enhanced model version management
4. **A/B Testing Integration**: Seamless integration with A/B testing

### Advanced Features

1. **Ensemble Challengers**: Multiple challenger models
2. **Dynamic Switching**: Automatic model switching based on performance
3. **Risk Management**: Advanced risk controls for challenger models
4. **Reporting**: Enhanced performance reporting and analytics

## Support

For issues with challenger mode:
1. Check the logs for error messages
2. Verify model configuration
3. Test with known good models
4. Contact the development team

---

*This guide covers the challenger mode functionality. For more information, see the main Ares documentation.* 