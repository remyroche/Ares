# 🎯 Unified Configuration Examples

## Overview

This document provides comprehensive examples of how to use the unified configuration system to eliminate duplication and provide a clean, builder-pattern interface for all backtesting components.

## 🚀 Basic Usage

### **Simple Configuration**
```python
from src.training.steps.backtesting.unified_config import create_config

# Create a basic configuration
config = create_config().build()

# Use the configuration
print(f"Symbol: {config.data.symbol}")
print(f"Exchange: {config.data.exchange}")
print(f"Mode: {config.mode}")
```

### **Custom Configuration**
```python
# Create a custom configuration
config = (create_config()
          .set_symbol("ETHUSDT")
          .set_exchange("binance")
          .set_timeframe("4h")
          .set_initial_capital(50000.0)
          .enable_gpu_acceleration(True)
          .build())
```

## 🎯 Configuration Presets

### **Testing Configuration**
```python
from src.training.steps.backtesting.unified_config import create_testing_config

# Quick testing configuration
config = create_testing_config()
# Automatically sets:
# - BLANK mode
# - GPU acceleration disabled
# - Parallel processing disabled
# - Small initial capital (10,000)
# - Minimal simulations (100)
```

### **Development Configuration**
```python
from src.training.steps.backtesting.unified_config import create_development_config

# Development configuration
config = create_development_config()
# Automatically sets:
# - LIGHT mode
# - GPU acceleration enabled
# - Parallel processing enabled (2 workers)
# - Moderate initial capital (50,000)
# - Moderate simulations (500)
```

### **Production Configuration**
```python
from src.training.steps.backtesting.unified_config import create_production_config

# Production configuration
config = create_production_config()
# Automatically sets:
# - FULL mode
# - All optimizations enabled
# - Maximum workers (4)
# - Full initial capital (100,000)
# - Full simulations (1000)
```

## 🔧 Advanced Configuration

### **Hardware Optimization**
```python
config = (create_config()
          .set_hardware_config(
              enable_gpu_acceleration=True,
              enable_memory_optimization=True,
              enable_parallel_processing=True,
              max_workers=8,
              gpu_memory_limit=0.8
          )
          .build())
```

### **Data Configuration**
```python
config = (create_config()
          .set_data_config(
              symbol="BTCUSDT",
              exchange="binance",
              timeframe="1h",
              data_dir="/workspace/data",
              start_date="2024-01-01",
              end_date="2024-01-31",
              data_type="processed",
              cache_enabled=True,
              compression="snappy"
          )
          .build())
```

### **Validation Configuration**
```python
config = (create_config()
          .set_validation_config(
              validation_enabled=True,
              monitoring_enabled=True,
              enable_cv_validation=True,
              enable_hpo=True,
              cv_folds=5,
              cv_method="purged",
              lookahead_bias_protection=True,
              overfitting_detection=True
          )
          .build())
```

### **Backtesting Configuration**
```python
config = (create_config()
          .set_backtesting_config(
              initial_capital=100000.0,
              commission_rate=0.001,
              slippage_rate=0.0005,
              max_position_size=0.1,
              min_position_size=0.01,
              rebalance_frequency="daily",
              risk_free_rate=0.02,
              max_drawdown=0.2,
              stop_loss=0.05,
              take_profit=0.1
          )
          .build())
```

### **Monte Carlo Configuration**
```python
config = (create_config()
          .set_monte_carlo_config(
              n_simulations=1000,
              confidence_level=0.95,
              simulation_horizon=252,
              mode=MonteCarloMode.HYBRID,
              bootstrap_sample_size=0.8,
              parametric_distribution="normal",
              var_confidence=0.05,
              expected_shortfall_confidence=0.01
          )
          .build())
```

### **A/B Testing Configuration**
```python
config = (create_config()
          .set_ab_testing_config(
              test_type=ABTestType.COMPREHENSIVE,
              significance_level=0.05,
              power=0.8,
              min_sample_size=30,
              test_duration_days=252,
              warmup_period_days=30,
              cooldown_period_days=7,
              multiple_comparison_correction="bonferroni",
              effect_size_threshold=0.1,
              confidence_interval=0.95
          )
          .build())
```

### **Optimization Configuration**
```python
config = (create_config()
          .set_optimization_config(
              optimization_method=OptimizationMethod.BAYESIAN,
              n_trials=100,
              timeout_seconds=3600,
              early_stopping_patience=10,
              convergence_threshold=1e-6,
              objective_metric="sharpe_ratio",
              minimize_objective=False,
              hpo_method="bayesian"
          )
          .build())
```

### **Reporting Configuration**
```python
config = (create_config()
          .set_reporting_config(
              report_type=ReportType.COMPREHENSIVE,
              output_dir="reports",
              output_format="html",
              enable_plots=True,
              plot_style="seaborn",
              figure_size=(12, 8),
              dpi=300,
              include_performance_metrics=True,
              include_risk_analysis=True,
              include_trade_analysis=True,
              include_portfolio_analysis=True,
              include_visualizations=True
          )
          .build())
```

### **Logging Configuration**
```python
config = (create_config()
          .set_logging_config(
              level="INFO",
              enable_console=True,
              enable_file=True,
              log_file="backtesting.log",
              enable_debug=False,
              log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
          )
          .build())
```

## 🎯 Predefined Presets

### **Crypto Day Trading**
```python
from src.training.steps.backtesting.unified_config import ConfigurationPresets

config = ConfigurationPresets.crypto_day_trading()
# Automatically configured for:
# - BTCUSDT on Binance
# - 1h timeframe
# - 100,000 initial capital
# - 0.001 commission rate
# - Production optimizations
```

### **Crypto Swing Trading**
```python
config = ConfigurationPresets.crypto_swing_trading()
# Automatically configured for:
# - ETHUSDT on Binance
# - 4h timeframe
# - 50,000 initial capital
# - 0.001 commission rate
# - Production optimizations
```

### **Forex Scalping**
```python
config = ConfigurationPresets.forex_scalping()
# Automatically configured for:
# - EURUSD on OANDA
# - 1m timeframe
# - 10,000 initial capital
# - 0.0001 commission rate
# - Production optimizations
```

### **Stock Swing Trading**
```python
config = ConfigurationPresets.stock_swing_trading()
# Automatically configured for:
# - AAPL on Yahoo
# - 1d timeframe
# - 100,000 initial capital
# - 0.005 commission rate
# - Production optimizations
```

## 🔄 Migration from Old Configuration

### **Before (Duplicated Configuration)**
```python
# Old way - lots of duplication
class OldBacktestingConfig:
    def __init__(self):
        self.symbol = "BTCUSDT"
        self.exchange = "binance"
        self.timeframe = "1h"
        self.data_dir = "/workspace/data"
        self.initial_capital = 100000.0
        self.commission_rate = 0.001
        self.slippage_rate = 0.0005
        self.enable_gpu_acceleration = True
        self.enable_parallel_processing = True
        self.max_workers = 4
        self.validation_enabled = True
        self.monitoring_enabled = True
        # ... many more duplicated parameters
```

### **After (Unified Configuration)**
```python
# New way - unified and clean
config = (create_config()
          .set_symbol("BTCUSDT")
          .set_exchange("binance")
          .set_timeframe("1h")
          .set_data_dir("/workspace/data")
          .set_initial_capital(100000.0)
          .set_commission_rate(0.001)
          .set_slippage_rate(0.0005)
          .enable_gpu_acceleration(True)
          .enable_parallel_processing(True, max_workers=4)
          .enable_validation(True)
          .enable_monitoring(True)
          .build())
```

## 🎯 Component-Specific Usage

### **Real Backtesting Engine**
```python
from src.utils.nas_tas.backtesting_engine import RealBacktestingEngine

# Create configuration
config = (create_config()
          .set_symbol("BTCUSDT")
          .set_exchange("binance")
          .set_timeframe("1h")
          .for_production()
          .build())

# Use with engine
engine = RealBacktestingEngine(config)
data = await engine.load_market_data()
data = engine.calculate_technical_indicators(data)
signals = engine.generate_trading_signals(data)
results = await engine.execute_backtest(data, signals)
```

### **Monte Carlo Engine**
```python
from src.training.steps.backtesting.real_monte_carlo_engine import RealMonteCarloEngine

# Create configuration
config = (create_config()
          .set_n_simulations(1000)
          .set_confidence_level(0.95)
          .for_production()
          .build())

# Use with engine
engine = RealMonteCarloEngine(config)
results = await engine.run_simulation(returns_data)
```

### **A/B Testing Engine**
```python
from src.training.steps.backtesting.real_ab_testing_engine import RealABTestingEngine

# Create configuration
config = (create_config()
          .set_significance_level(0.05)
          .set_power(0.8)
          .for_production()
          .build())

# Use with engine
engine = RealABTestingEngine(config)
results = await engine.run_ab_test(strategy_a_results, strategy_b_results)
```

### **Parameters Optimization**
```python
from src.training.steps.backtesting.real_parameters_optimization import RealParametersOptimizer

# Create configuration
config = (create_config()
          .set_n_trials(100)
          .set_optimization_method(OptimizationMethod.BAYESIAN)
          .for_production()
          .build())

# Use with engine
optimizer = RealParametersOptimizer(config)
results = await optimizer.optimize_parameters(objective_function)
```

### **Reporting Engine**
```python
from src.training.steps.backtesting.real_reporting_engine import RealReportingEngine

# Create configuration
config = (create_config()
          .set_output_dir("reports")
          .set_output_format("html")
          .for_production()
          .build())

# Use with engine
engine = RealReportingEngine(config)
report = await engine.generate_report(backtest_results)
```

## 🎯 Sub-Pipeline Usage

### **Updated Sub-Pipeline**
```python
from src.training.steps.backtesting.sub_pipeline import BacktestingSubPipeline
from src.training.steps.backtesting.unified_config import create_config

# Create unified configuration
config = (create_config()
          .set_symbol("BTCUSDT")
          .set_exchange("binance")
          .set_timeframe("1h")
          .for_production()
          .build())

# Create sub-pipeline configuration
sub_pipeline_config = SubPipelineConfig(unified_config=config)

# Execute sub-pipeline
pipeline = BacktestingSubPipeline()
results = await pipeline.execute_sub_pipeline("basic_backtesting_pre", sub_pipeline_config)
```

## 🎯 Benefits of Unified Configuration

### **1. Eliminates Duplication**
- **Before**: 20+ duplicated parameters across components
- **After**: Single source of truth for all configuration

### **2. Type Safety**
- **Before**: String-based configuration with no validation
- **After**: Strongly typed configuration with validation

### **3. Builder Pattern**
- **Before**: Complex constructor with many parameters
- **After**: Fluent interface with method chaining

### **4. Presets**
- **Before**: Manual configuration for each use case
- **After**: Predefined presets for common scenarios

### **5. Validation**
- **Before**: No configuration validation
- **After**: Comprehensive validation with helpful error messages

### **6. Extensibility**
- **Before**: Hard to add new configuration options
- **After**: Easy to extend with new components and parameters

## 🎯 Best Practices

### **1. Use Presets When Possible**
```python
# Good - use presets for common scenarios
config = create_production_config()

# Better - customize presets
config = (create_config()
          .for_production()
          .set_symbol("ETHUSDT")
          .set_timeframe("4h")
          .build())
```

### **2. Validate Configuration**
```python
# Configuration is automatically validated
try:
    config = (create_config()
              .set_initial_capital(-1000)  # Invalid - negative capital
              .build())
except ValueError as e:
    print(f"Configuration error: {e}")
```

### **3. Use Component-Specific Configuration**
```python
# Configure only what you need
config = (create_config()
          .set_backtesting_config(initial_capital=50000.0)
          .set_monte_carlo_config(n_simulations=500)
          .build())
```

### **4. Chain Configuration Methods**
```python
# Good - fluent interface
config = (create_config()
          .set_symbol("BTCUSDT")
          .set_exchange("binance")
          .set_timeframe("1h")
          .enable_gpu_acceleration(True)
          .enable_parallel_processing(True)
          .build())
```

### **5. Use Custom Parameters**
```python
# For component-specific parameters
config = (create_config()
          .set_custom_params(
              custom_indicator_param=0.5,
              custom_risk_param=0.1
          )
          .build())
```

## 🎯 Conclusion

The unified configuration system provides:

1. **Elimination of Duplication**: Single source of truth for all configuration
2. **Type Safety**: Strongly typed configuration with validation
3. **Builder Pattern**: Fluent interface for easy configuration
4. **Presets**: Predefined configurations for common scenarios
5. **Extensibility**: Easy to add new components and parameters
6. **Validation**: Comprehensive validation with helpful error messages

This system makes configuration management much cleaner, more maintainable, and less error-prone.