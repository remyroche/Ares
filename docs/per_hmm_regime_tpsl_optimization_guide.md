# Per-HMM Regime Triple Barrier Thresholds and TPSL Parameters Optimization Guide

## Overview

The Per-HMM Regime TPSL Optimization system provides comprehensive optimization of triple barrier thresholds and Take Profit/Stop Loss (TPSL) parameters for each HMM regime identified by the HMM regime discovery system. This system enables regime-specific parameter optimization that adapts to different market conditions.

## Key Features

### 1. **Per-Regime Optimization**
- Optimizes parameters specifically for each HMM regime (hmm_cluster_0 through hmm_cluster_7)
- Regime-specific triple barrier thresholds
- Regime-specific TPSL parameters
- Dynamic parameter adjustment based on regime characteristics

### 2. **Triple Barrier Optimization**
- **Profit Take Multiplier**: Optimizes the profit target threshold (0.1% to 1%)
- **Stop Loss Multiplier**: Optimizes the stop loss threshold (0.05% to 0.5%)
- **Time Barrier**: Optimizes the time limit for trades (15 minutes to 2 hours)
- **Max Lookahead**: Optimizes the maximum lookahead period (50 to 200 bars)

### 3. **TPSL Parameter Optimization**
- **Target Percentage**: Optimizes take profit levels (0.2% to 2%)
- **Stop Percentage**: Optimizes stop loss levels (0.1% to 1%)
- **Risk-Reward Ratio**: Optimizes risk-reward ratios (1.5:1 to 4:1)
- **Position Sizing**: Optimizes position sizing percentages (1% to 5% of capital)

### 4. **Regime-Specific Adjustments**
- **Volatility Multiplier**: Adjusts parameters based on market volatility
- **Momentum Multiplier**: Adjusts parameters based on price momentum
- **Confidence Threshold**: Minimum confidence required for regime identification

## System Architecture

### Core Components

1. **PerHMMRegimeTPSLOptimizer**: Main optimization class
2. **HMMCompositeManager**: Manages HMM regime data
3. **OptimizedTripleBarrierLabeling**: Generates triple barrier labels
4. **Configuration System**: Manages optimization parameters and bounds

### Data Flow

```
Market Data → HMM Regime Identification → Regime-Specific Optimization → Parameter Validation → Backtesting → Results
```

## Installation and Setup

### Prerequisites

```bash
pip install optuna pandas numpy scikit-learn
```

### Configuration

The system uses a comprehensive configuration file located at:
```
src/config/per_hmm_regime_tpsl_config.py
```

### Basic Usage

```python
from src.training.steps.analyst_training_components.per_hmm_regime_tpsl_optimizer import (
    PerHMMRegimeTPSLOptimizer
)

# Initialize optimizer
config = {
    "per_hmm_regime_tpsl_optimizer": {
        "n_trials": 200,
        "min_trades_per_regime": 30,
        "cv_folds": 5,
        "optimization_metric": "sharpe_ratio"
    }
}

optimizer = PerHMMRegimeTPSLOptimizer(config)
await optimizer.initialize()

# Get optimized parameters
optimized_params = await optimizer.get_optimized_parameters(
    current_data, historical_data, exchange, symbol, timeframe
)
```

## Regime-Specific Configurations

### HMM Cluster 0: Low Volatility Sideways
- **Characteristics**: Low volatility, sideways movement, high frequency
- **Default Parameters**:
  - Triple Barrier: 0.3% profit take, 0.2% stop loss, 45 minutes
  - TPSL: 0.5% target, 0.3% stop, 1.67:1 risk-reward
- **Optimization Focus**: High frequency, low risk trades

### HMM Cluster 1: Moderate Volatility Trending
- **Characteristics**: Moderate volatility, trending movement, medium frequency
- **Default Parameters**:
  - Triple Barrier: 0.5% profit take, 0.3% stop loss, 60 minutes
  - TPSL: 0.8% target, 0.4% stop, 2.0:1 risk-reward
- **Optimization Focus**: Trend following with moderate risk

### HMM Cluster 2: High Volatility Breakout
- **Characteristics**: High volatility, breakout movement, low frequency
- **Default Parameters**:
  - Triple Barrier: 0.8% profit take, 0.4% stop loss, 30 minutes
  - TPSL: 1.2% target, 0.6% stop, 2.0:1 risk-reward
- **Optimization Focus**: High reward, high risk breakout trades

### HMM Cluster 3: Extreme Volatility Crisis
- **Characteristics**: Extreme volatility, crisis movement, very low frequency
- **Default Parameters**:
  - Triple Barrier: 1.5% profit take, 0.8% stop loss, 20 minutes
  - TPSL: 2.0% target, 1.0% stop, 2.0:1 risk-reward
- **Optimization Focus**: Crisis trading with extreme caution

### HMM Cluster 4: Low Volatility Trending
- **Characteristics**: Low volatility, trending movement, medium frequency
- **Default Parameters**:
  - Triple Barrier: 0.4% profit take, 0.2% stop loss, 90 minutes
  - TPSL: 0.6% target, 0.3% stop, 2.0:1 risk-reward
- **Optimization Focus**: Steady trend following with low risk

### HMM Cluster 5: Moderate Volatility Sideways
- **Characteristics**: Moderate volatility, sideways movement, high frequency
- **Default Parameters**:
  - Triple Barrier: 0.4% profit take, 0.3% stop loss, 60 minutes
  - TPSL: 0.7% target, 0.4% stop, 1.75:1 risk-reward
- **Optimization Focus**: Range trading with moderate frequency

### HMM Cluster 6: High Volatility Sideways
- **Characteristics**: High volatility, sideways movement, medium frequency
- **Default Parameters**:
  - Triple Barrier: 0.6% profit take, 0.4% stop loss, 45 minutes
  - TPSL: 1.0% target, 0.5% stop, 2.0:1 risk-reward
- **Optimization Focus**: Wide range trading with high volatility

### HMM Cluster 7: Moderate Volatility Breakout
- **Characteristics**: Moderate volatility, breakout movement, low frequency
- **Default Parameters**:
  - Triple Barrier: 0.6% profit take, 0.4% stop loss, 40 minutes
  - TPSL: 0.9% target, 0.5% stop, 1.8:1 risk-reward
- **Optimization Focus**: Controlled breakout trading

## Optimization Process

### 1. **Regime Identification**
```python
regime, confidence, regime_info = await optimizer.identify_current_hmm_regime(
    current_data, exchange, symbol, timeframe
)
```

### 2. **Parameter Optimization**
```python
optimized_params = await optimizer.optimize_regime_parameters(
    regime, historical_data, current_data, force_optimization=True
)
```

### 3. **Cross-Validation**
The system uses TimeSeriesSplit cross-validation to ensure robust parameter selection:
- 5-fold cross-validation by default
- Prevents overfitting to specific time periods
- Validates parameter stability across different market conditions

### 4. **Performance Metrics**
The optimization targets multiple performance metrics:
- **Primary**: Sharpe Ratio
- **Secondary**: Total Return, Win Rate, Calmar Ratio, Max Drawdown

## Advanced Features

### 1. **Regime Transition Handling**
- Smooth parameter transitions between regimes
- Confidence-based regime switching
- Cooldown periods to prevent rapid regime changes

### 2. **Dynamic Parameter Adjustment**
- Volatility-based parameter scaling
- Momentum-based time barrier adjustment
- Real-time parameter updates

### 3. **Risk Management**
- Position sizing optimization
- Maximum drawdown constraints
- Risk-reward ratio validation

### 4. **Performance Tracking**
- Comprehensive regime statistics
- Optimization history tracking
- Performance comparison across regimes

## Testing and Validation

### Running Tests
```bash
python test_per_hmm_regime_tpsl_optimization.py --symbol ETHUSDT --exchange BINANCE --timeframe 30m
```

### Test Coverage
1. **Regime Identification**: Tests HMM regime detection
2. **Single Regime Optimization**: Tests optimization for individual regimes
3. **Full Pipeline**: Tests complete optimization workflow
4. **All Regimes**: Tests optimization for all regimes
5. **Statistics**: Tests reporting and statistics functionality

### Validation Metrics
- **Parameter Consistency**: Ensures parameters are within valid bounds
- **Performance Validation**: Validates optimization results through backtesting
- **Regime Stability**: Checks regime identification consistency
- **Cross-Validation**: Ensures robust parameter selection

## Configuration Options

### Optimization Settings
```python
"optimization": {
    "n_trials": 200,                    # Number of optimization trials
    "min_trades_per_regime": 30,        # Minimum trades for validation
    "cv_folds": 5,                      # Cross-validation folds
    "optimization_metric": "sharpe_ratio", # Target metric
    "optimization_timeout": 3600,       # Timeout in seconds
    "parallel_trials": 4                # Parallel trials
}
```

### Parameter Bounds
```python
"triple_barrier_bounds": {
    "profit_take_multiplier": (0.001, 0.01),  # 0.1% to 1%
    "stop_loss_multiplier": (0.0005, 0.005),  # 0.05% to 0.5%
    "time_barrier_minutes": (15, 120),         # 15 min to 2 hours
    "max_lookahead": (50, 200)                 # 50 to 200 bars
}
```

### Performance Constraints
```python
"constraints": {
    "min_risk_reward_ratio": 1.2,       # Minimum risk-reward ratio
    "max_position_size": 0.1,           # Maximum position size (10%)
    "min_win_rate": 0.4,                # Minimum win rate (40%)
    "max_drawdown_threshold": 0.15      # Maximum drawdown (15%)
}
```

## Best Practices

### 1. **Data Quality**
- Ensure sufficient historical data for each regime
- Validate data quality before optimization
- Use appropriate timeframes for regime identification

### 2. **Optimization Strategy**
- Start with conservative parameter bounds
- Use cross-validation to prevent overfitting
- Monitor optimization convergence

### 3. **Risk Management**
- Set appropriate position sizing limits
- Monitor regime transition frequency
- Implement stop-loss mechanisms

### 4. **Performance Monitoring**
- Track regime-specific performance
- Monitor parameter stability
- Validate optimization results regularly

## Troubleshooting

### Common Issues

1. **Insufficient Data**
   - Error: "Not enough trades for regime optimization"
   - Solution: Increase historical data or reduce minimum trade requirements

2. **Regime Identification Failures**
   - Error: "No HMM regime data available"
   - Solution: Ensure HMM regime discovery has been completed

3. **Optimization Convergence**
   - Error: "Optimization failed to converge"
   - Solution: Increase number of trials or adjust parameter bounds

4. **Performance Degradation**
   - Issue: Poor optimization results
   - Solution: Check data quality and regime identification accuracy

### Debug Mode
Enable debug logging for detailed troubleshooting:
```python
config["logging"]["log_level"] = "DEBUG"
```

## Performance Benchmarks

### Expected Performance by Regime

| Regime | Expected Sharpe | Expected Win Rate | Expected Max DD |
|--------|----------------|-------------------|-----------------|
| hmm_cluster_0 | 0.8-1.2 | 60-70% | 5-8% |
| hmm_cluster_1 | 1.0-1.5 | 55-65% | 8-12% |
| hmm_cluster_2 | 1.2-1.8 | 50-60% | 12-18% |
| hmm_cluster_3 | 1.5-2.2 | 45-55% | 15-25% |
| hmm_cluster_4 | 1.0-1.4 | 60-70% | 6-10% |
| hmm_cluster_5 | 0.9-1.3 | 55-65% | 8-12% |
| hmm_cluster_6 | 1.1-1.6 | 50-60% | 10-15% |
| hmm_cluster_7 | 1.2-1.7 | 50-60% | 10-16% |

## Future Enhancements

### Planned Features
1. **Machine Learning Integration**: ML-based regime prediction
2. **Real-time Optimization**: Continuous parameter updates
3. **Multi-Asset Support**: Cross-asset regime correlation
4. **Advanced Risk Models**: VaR and CVaR integration
5. **Portfolio Optimization**: Multi-regime portfolio management

### Research Areas
1. **Regime Transition Prediction**: Predictive regime switching
2. **Dynamic Parameter Bounds**: Adaptive parameter ranges
3. **Market Microstructure**: Order book integration
4. **Sentiment Analysis**: News and social media integration

## Conclusion

The Per-HMM Regime TPSL Optimization system provides a comprehensive solution for regime-specific parameter optimization. By leveraging HMM regime identification and advanced optimization techniques, it enables adaptive trading strategies that respond to changing market conditions.

The system's modular design, comprehensive configuration options, and robust validation mechanisms make it suitable for both research and production trading environments. Regular monitoring and validation ensure optimal performance across different market regimes.

For questions and support, please refer to the system documentation or contact the development team.