# A/B/C Testing Framework for Paper Trading

A comprehensive framework for paper-trading multiple models simultaneously with advanced statistical analysis, risk management, and performance monitoring.

## 🚀 Features

### Core Components

- **A/B/C Testing Framework**: Compare multiple trading models with statistical rigor
- **Multi-Model Orchestrator**: Coordinate and manage multiple models simultaneously
- **Paper Trading Engine**: Realistic market simulation with slippage, fees, and latency
- **Risk Management System**: Advanced position sizing and risk controls
- **Statistical Analysis**: Comprehensive statistical testing and validation
- **Performance Monitoring**: Real-time monitoring with alerts and notifications
- **Results Visualization**: Interactive dashboards and comprehensive reports
- **Configuration Management**: Flexible configuration system with validation

### Advanced Capabilities

- **Realistic Market Simulation**: Order book simulation, market impact, partial fills
- **Multiple Position Sizing Methods**: Kelly, Fixed Fractional, Volatility Adjusted, Risk Parity
- **Statistical Rigor**: Multiple testing correction, effect size analysis, power analysis
- **Risk Controls**: Circuit breakers, correlation limits, drawdown protection
- **M1 Hardware Optimizations**: GPU acceleration, memory optimization, parallel processing
- **Comprehensive Reporting**: HTML dashboards, statistical reports, executive summaries

## 📁 Framework Structure

```
src/training/steps/backtesting/
├── abc_testing_framework.py          # Core A/B/C testing framework
├── multi_model_orchestrator.py       # Multi-model coordination
├── paper_trading_engine.py           # Realistic paper trading simulation
├── risk_management.py                # Risk management and position sizing
├── statistical_analysis.py           # Statistical testing and validation
├── performance_monitoring.py         # Real-time monitoring and alerts
├── results_visualization.py          # Visualization and reporting
├── configuration_management.py       # Configuration system
├── abc_testing_integration_example.py # Complete integration example
└── README.md                         # This file
```

## 🛠️ Installation and Setup

### Prerequisites

```bash
# Required Python packages
pip install numpy pandas scipy scikit-learn
pip install lightgbm xgboost catboost
pip install plotly dash bokeh
pip install pyyaml jsonschema
pip install asyncio-mqtt  # For real-time monitoring
```

### M1 Hardware Optimizations

The framework includes optimizations for M1/M2 Macs:

```python
# GPU acceleration (if available)
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Memory optimization
import psutil
memory_limit = psutil.virtual_memory().total * 0.8  # Use 80% of available memory
```

## 🚀 Quick Start

### Basic A/B/C Test

```python
import asyncio
from src.training.steps.backtesting.abc_testing_integration_example import ABCTestingIntegrationExample

async def run_basic_test():
    # Initialize the framework
    integration = ABCTestingIntegrationExample("config/my_test")
    
    # Define test configuration
    test_config = {
        "test_name": "My_ABC_Test",
        "symbol": "BTCUSDT",
        "exchange": "BINANCE",
        "timeframe": "1h",
        "start_date": "2024-01-01T00:00:00",
        "end_date": "2024-03-31T23:59:59",
        "models": [
            {
                "model_id": "model_a",
                "model_name": "RandomForest",
                "model_type": "random_forest",
                "initial_capital": 100000.0,
                "model_params": {"n_estimators": 100, "max_depth": 10}
            },
            {
                "model_id": "model_b",
                "model_name": "LightGBM",
                "model_type": "lightgbm",
                "initial_capital": 100000.0,
                "model_params": {"n_estimators": 200, "learning_rate": 0.1}
            }
        ]
    }
    
    # Run the test
    results = await integration.run_complete_abc_test(test_config)
    return results

# Execute the test
results = asyncio.run(run_basic_test())
```

### Advanced Configuration

```python
# Advanced test configuration with all options
advanced_config = {
    "test_name": "Advanced_ABC_Test",
    "test_description": "Comprehensive model comparison",
    "symbol": "ETHUSDT",
    "exchange": "BINANCE",
    "timeframe": "4h",
    "start_date": "2024-01-01T00:00:00",
    "end_date": "2024-06-30T23:59:59",
    "models": [
        {
            "model_id": "rf_model",
            "model_name": "RandomForest_Advanced",
            "model_type": "random_forest",
            "initial_capital": 100000.0,
            "max_position_size": 0.10,
            "risk_per_trade": 0.025,
            "model_params": {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 3,
                "min_samples_leaf": 2,
                "random_state": 42
            }
        },
        {
            "model_id": "lgb_model",
            "model_name": "LightGBM_Advanced",
            "model_type": "lightgbm",
            "initial_capital": 100000.0,
            "max_position_size": 0.10,
            "risk_per_trade": 0.025,
            "model_params": {
                "n_estimators": 300,
                "max_depth": 12,
                "learning_rate": 0.08,
                "num_leaves": 50,
                "random_state": 42
            }
        },
        {
            "model_id": "xgb_model",
            "model_name": "XGBoost_Advanced",
            "model_type": "xgboost",
            "initial_capital": 100000.0,
            "max_position_size": 0.10,
            "risk_per_trade": 0.025,
            "model_params": {
                "n_estimators": 250,
                "max_depth": 10,
                "learning_rate": 0.09,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42
            }
        }
    ],
    "statistical_testing": {
        "enable_statistical_testing": True,
        "confidence_level": 0.95,
        "alpha": 0.05,
        "min_sample_size": 200,
        "enable_multiple_testing_correction": True,
        "correction_method": "bonferroni",
        "effect_size_threshold": 0.3,
        "power_analysis": True,
        "power_threshold": 0.85
    },
    "risk_management": {
        "global_risk_limit": 0.20,
        "max_concurrent_positions": 10,
        "correlation_threshold": 0.70,
        "enable_circuit_breakers": True,
        "circuit_breaker_threshold": 0.10
    }
}
```

## 📊 Model Types Supported

The framework supports a wide range of machine learning models:

### Tree-Based Models
- **Random Forest**: Ensemble of decision trees
- **LightGBM**: Gradient boosting with leaf-wise growth
- **XGBoost**: Extreme gradient boosting
- **CatBoost**: Categorical boosting
- **Extra Trees**: Extremely randomized trees

### Neural Networks
- **TabNet**: Attention-based tabular learning
- **Time Series Transformer**: Transformer for time series
- **TCN**: Temporal Convolutional Network
- **LSTM**: Long Short-Term Memory
- **WaveNet**: Dilated causal convolution
- **Temporal Fusion Transformer**: Advanced time series model

### Linear Models
- **Ridge Regression**: L2 regularized linear regression
- **Logistic Regression**: For classification tasks
- **Elastic Net**: L1 + L2 regularization
- **Huber Regression**: Robust to outliers
- **Histogram Gradient Boosting**: Fast gradient boosting

## 🎯 Position Sizing Methods

### Available Methods

1. **Fixed**: Fixed position size
2. **Fixed Fractional**: Fixed percentage of capital
3. **Kelly Criterion**: Optimal position sizing based on win rate and payoff
4. **Volatility Adjusted**: Adjusts size based on asset volatility
5. **Risk Parity**: Equal risk contribution from each position
6. **Optimal F**: Ralph Vince's optimal f
7. **ATR Based**: Based on Average True Range
8. **Correlation Adjusted**: Adjusts for correlation with existing positions

### Example Usage

```python
from src.training.steps.backtesting.risk_management import PositionSizingMethod

# Configure position sizing
position_sizing_config = PositionSizingConfig(
    method=PositionSizingMethod.KELLY,
    base_risk_per_trade=0.02,
    max_position_size=0.10,
    min_position_size=0.005,
    volatility_lookback=20,
    kelly_fraction=0.25,  # Use 25% of Kelly optimal
    enable_dynamic_sizing=True,
    enable_correlation_adjustment=True
)
```

## 📈 Statistical Analysis

### Statistical Tests

- **T-Test**: Compare means between models
- **Mann-Whitney U**: Non-parametric comparison
- **Chi-Square**: Test independence
- **Kolmogorov-Smirnov**: Compare distributions
- **Fisher's Exact**: Exact test for small samples
- **Wilcoxon Signed-Rank**: Paired comparison

### Multiple Testing Correction

- **Bonferroni**: Conservative correction
- **Holm**: Step-down procedure
- **Benjamini-Hochberg**: False discovery rate control
- **Benjamini-Yekutieli**: Conservative FDR control

### Effect Size Analysis

- **Cohen's d**: Standardized mean difference
- **Hedges' g**: Bias-corrected effect size
- **Glass's Δ**: Effect size using control group SD
- **Common Language Effect Size**: Probability of superiority

## 🚨 Risk Management

### Risk Limits

```python
from src.training.steps.backtesting.risk_management import RiskLimits

risk_limits = RiskLimits(
    max_portfolio_risk=0.15,        # 15% max portfolio risk
    max_position_risk=0.05,         # 5% max position risk
    max_correlation=0.70,           # 70% max correlation
    max_drawdown=0.10,              # 10% max drawdown
    max_leverage=1.0,               # No leverage
    max_concurrent_positions=8,     # Max 8 positions
    max_daily_loss=0.05,            # 5% max daily loss
    var_confidence_level=0.95,      # 95% VaR
    enable_circuit_breakers=True,   # Enable circuit breakers
    circuit_breaker_threshold=0.08  # 8% loss triggers circuit breaker
)
```

### Circuit Breakers

The framework includes automatic circuit breakers that halt trading when:
- Drawdown exceeds threshold
- Daily loss exceeds limit
- Portfolio risk exceeds limit
- Correlation between positions too high

## 📊 Performance Monitoring

### Real-time Metrics

- Portfolio value and returns
- Risk metrics (VaR, CVaR, volatility)
- Drawdown monitoring
- Position-level metrics
- Trade execution quality

### Alerting System

```python
from src.training.steps.backtesting.performance_monitoring import AlertConfig

alert_config = AlertConfig(
    enable_email_alerts=True,
    enable_slack_alerts=False,
    alert_thresholds={
        "max_drawdown": 0.12,
        "min_sharpe_ratio": 0.8,
        "max_volatility": 0.25,
        "min_win_rate": 0.45
    },
    email_settings={
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "your_email@gmail.com",
        "recipients": ["alerts@yourcompany.com"]
    }
)
```

## 📈 Results Visualization

### Generated Reports

1. **Performance Comparison Report**: Side-by-side model comparison
2. **Statistical Analysis Report**: Statistical test results
3. **Risk Analysis Report**: Risk metrics and limits
4. **Correlation Analysis Report**: Model correlation analysis
5. **Executive Summary**: High-level overview
6. **Comprehensive Dashboard**: Interactive HTML dashboard

### Example Dashboard Features

- Interactive performance charts
- Statistical test results
- Risk metrics visualization
- Correlation heatmaps
- Trade analysis
- Performance attribution

## ⚙️ Configuration Management

### Configuration Schema

The framework uses JSON Schema for configuration validation:

```python
from src.training.steps.backtesting.configuration_management import ConfigurationManager

# Initialize configuration manager
config_manager = ConfigurationManager("config/abc_testing")

# Create configuration from template
config = config_manager.create_configuration_from_template(
    template_id="abc_testing",
    parameters={
        "test_name": "My_Test",
        "symbol": "BTCUSDT",
        "models": [...]
    },
    name="my_test_config",
    environment="production"
)
```

### Configuration Validation

All configurations are validated against schemas to ensure:
- Required fields are present
- Data types are correct
- Values are within acceptable ranges
- Dependencies are satisfied

## 🔧 Advanced Usage

### Custom Model Integration

```python
from src.training.steps.backtesting.multi_model_orchestrator import ModelConfig

# Create custom model configuration
custom_model = ModelConfig(
    model_id="custom_model",
    model_name="My_Custom_Model",
    model_type="custom",
    model_class=MyCustomModel,
    model_params={"param1": "value1"},
    initial_capital=100000.0,
    max_position_size=0.08,
    risk_per_trade=0.02
)
```

### Custom Risk Metrics

```python
from src.training.steps.backtesting.risk_management import RiskCalculator

class CustomRiskCalculator(RiskCalculator):
    def calculate_custom_metric(self, data):
        # Implement custom risk metric
        return custom_metric_value
```

### Custom Visualization

```python
from src.training.steps.backtesting.results_visualization import ResultsVisualizer

class CustomVisualizer(ResultsVisualizer):
    def generate_custom_report(self, results):
        # Implement custom visualization
        pass
```

## 🧪 Testing and Validation

### Unit Tests

```bash
# Run unit tests
python -m pytest tests/unit/backtesting/ -v

# Run with coverage
python -m pytest tests/unit/backtesting/ --cov=src.training.steps.backtesting --cov-report=html
```

### Integration Tests

```bash
# Run integration tests
python -m pytest tests/integration/backtesting/ -v
```

### Performance Tests

```bash
# Run performance benchmarks
python tests/performance/backtesting_benchmarks.py
```

## 📚 Examples

### Example 1: Basic A/B Test

```python
# See abc_testing_integration_example.py for complete example
```

### Example 2: Multi-Asset Testing

```python
# Test multiple assets simultaneously
multi_asset_config = {
    "test_name": "Multi_Asset_Test",
    "assets": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
    "models": [...],
    # ... other configuration
}
```

### Example 3: Walk-Forward Validation

```python
# Use walk-forward validation for robust testing
walk_forward_config = {
    "validation_method": "walk_forward",
    "train_window": 252,  # 1 year
    "test_window": 63,    # 3 months
    "step_size": 21       # 1 month
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `max_concurrent_models` or increase system memory
2. **Slow Performance**: Enable GPU acceleration or reduce data size
3. **Configuration Errors**: Check schema validation errors
4. **Model Loading Issues**: Verify model files and dependencies

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug logging for specific components
logger = logging.getLogger('src.training.steps.backtesting')
logger.setLevel(logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of existing ML infrastructure
- Inspired by quantitative finance best practices
- Incorporates M1 hardware optimizations
- Uses industry-standard statistical methods

## 📞 Support

For questions and support:
- Create an issue in the repository
- Check the documentation
- Review the examples
- Contact the development team

---

**Happy Testing! 🚀**