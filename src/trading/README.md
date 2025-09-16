# Trading Module Organization

This document describes the new organization of the trading components in `src/trading/`.

## Overview

The trading module has been reorganized to provide clear separation of concerns and better maintainability. The structure follows your requirements with dedicated modules for regime detection, execution, monitoring, data, sizing, signal generation, and backtesting.

## Directory Structure

```
src/trading/
├── __init__.py                 # Main module exports
├── README.md                   # This documentation
├── config/                     # Trading configuration
│   ├── __init__.py
│   ├── trading_config.py       # Main trading configuration
│   ├── execution_config.py     # Execution and exchange settings
│   └── regime_config.py        # Regime detection configuration
├── regime/                     # ML-based regime detection
│   ├── __init__.py
│   ├── regime_detector.py      # Main regime detection engine
│   ├── regime_classifier.py    # Regime classification logic
│   ├── regime_analyzer.py      # Regime analysis and metrics
│   └── regime_weights.py       # Regime weight management
├── execution/                  # Order execution and exchange interfaces
│   ├── __init__.py
│   ├── order_manager.py        # Order management system
│   ├── exchange_interface.py   # Exchange abstraction layer
│   ├── paper_trader.py         # Paper trading implementation
│   ├── live_trader.py          # Live trading implementation
│   └── paper_trading_integration.py  # Integration layer
├── monitoring/                 # Real-time trading monitoring
│   ├── __init__.py
│   ├── trade_monitor.py        # Trade execution monitoring
│   ├── performance_tracker.py  # Performance tracking
│   ├── regime_monitor.py       # Regime change monitoring
│   └── alert_manager.py        # Alert and notification system
├── data/                       # Live data collection and providers
│   ├── __init__.py
│   ├── live_data_collector.py  # Real-time data collection
│   ├── market_data_provider.py # Unified data provider interface
│   └── data_validator.py       # Data validation and quality checks
├── utils/                      # Trading utilities
│   ├── __init__.py
│   ├── trading_utils.py        # General trading utilities
│   ├── regime_utils.py         # Regime-specific utilities
│   └── validation_utils.py     # Validation helpers
├── sizing/                     # Leverage and position management
│   ├── __init__.py
│   ├── position_sizer.py       # Position sizing engine
│   ├── leverage_manager.py     # Leverage management
│   ├── risk_calculator.py      # Risk calculations
│   └── portfolio_allocator.py  # Portfolio allocation
├── signal_generation/          # Analyst and tactician signal integration
│   ├── __init__.py
│   ├── analyst_signals.py      # Analyst signal generation
│   ├── tactician_signals.py    # Tactician signal generation
│   ├── signal_combiner.py      # Signal combination logic
│   └── signal_validator.py     # Signal validation
└── backtesting/                # Backtesting engine and analysis
    ├── __init__.py
    ├── backtest_engine.py      # Main backtesting engine
    ├── regime_backtester.py    # Regime-aware backtesting
    ├── performance_analyzer.py # Performance analysis
    └── report_generator.py     # Report generation
```

## Key Features

### 1. Regime Detection (`regime/`)
- **ML-based detection** of 15-25 market regimes with percentage weights
- **Integration** with existing HMM and market analysis components
- **Confidence scoring** and regime transition detection
- **Performance tracking** and regime weight optimization

### 2. Execution (`execution/`)
- **Unified order management** for both paper and live trading
- **Exchange abstraction** supporting multiple exchanges
- **Risk management** and position validation
- **Integration** with existing paper trading components

### 3. Sizing (`sizing/`)
- **Position sizing** based on regime probabilities and risk parameters
- **Leverage management** with configurable limits
- **Multiple sizing methods**: fixed, volatility-adjusted, regime-based, Kelly, risk parity
- **Portfolio allocation** and risk-based sizing

### 4. Signal Generation (`signal_generation/`)
- **Analyst and tactician integration** with existing components
- **Signal combination** using multiple methods (weighted average, regime-based, ensemble)
- **Regime-aware weighting** for signal combination
- **Confidence scoring** and signal validation

### 5. Data (`data/`)
- **Live data collection** with real-time market data feeds
- **Unified data provider** interface for multiple sources
- **Data validation** and quality checks
- **Caching** and performance optimization

### 6. Monitoring (`monitoring/`)
- **Real-time trade monitoring** and execution tracking
- **Performance metrics** and regime change detection
- **Alert system** for important events
- **Integration** with existing monitoring components

### 7. Backtesting (`backtesting/`)
- **Regime-aware backtesting** with historical regime data
- **Performance analysis** and metrics calculation
- **Report generation** in multiple formats
- **Integration** with existing backtesting components

## Configuration

The trading system uses a hierarchical configuration structure:

- **`TradingConfig`**: Main trading parameters, risk limits, and system settings
- **`ExecutionConfig`**: Exchange settings, order execution parameters, and error handling
- **`RegimeConfig`**: Regime detection parameters, ML model settings, and regime weights

## Integration with Existing Components

The new trading organization integrates with your existing components:

- **HMM Models**: Regime detection uses existing HMM training and models
- **Market Analysis**: Integrates with enhanced market analysis components
- **Analyst/Tactician**: Signal generation connects to existing analyst and tactician systems
- **Paper Trading**: Existing paper trading components are preserved and enhanced
- **Monitoring**: Integrates with existing monitoring and reporting systems

## Usage Example

```python
from src.trading import (
    TradingConfig, RegimeDetector, PositionSizer, 
    SignalCombiner, PaperTrader
)

# Initialize configuration
config = TradingConfig(
    mode=TradingMode.PAPER,
    risk_level=RiskLevel.MODERATE,
    symbols=["ETHUSDT", "BTCUSDT"]
)

# Initialize components
regime_detector = RegimeDetector(config.regime_config)
position_sizer = PositionSizer(config)
signal_combiner = SignalCombiner(config)
paper_trader = PaperTrader(config)

# Initialize all components
await regime_detector.initialize()
await position_sizer.initialize()
await signal_combiner.initialize()
await paper_trader.initialize()

# Detect regime
regime_detection = await regime_detector.detect_regime(market_data)

# Generate signals
analyst_signal = await analyst.generate_signal(market_data)
tactician_signal = await tactician.generate_signal(market_data)

# Combine signals
combined_result = await signal_combiner.combine_signals(
    symbol="ETHUSDT",
    analyst_signal=analyst_signal,
    tactician_signal=tactician_signal,
    regime_probabilities=regime_detection.regime_probabilities
)

# Calculate position size
position_result = await position_sizer.calculate_position_size(
    symbol="ETHUSDT",
    current_price=3000.0,
    regime_probabilities=regime_detection.regime_probabilities,
    regime_confidence=regime_detection.confidence,
    portfolio_value=10000.0,
    available_balance=5000.0
)

# Execute trade
if combined_result.combined_signal.signal_type == 'buy':
    await paper_trader.execute_buy_order(
        symbol="ETHUSDT",
        quantity=position_result.recommended_size,
        price=3000.0,
        timestamp=datetime.now()
    )
```

## Migration Status

- ✅ **Directory structure created**
- ✅ **Configuration modules implemented**
- ✅ **Regime detection framework created**
- ✅ **Position sizing engine implemented**
- ✅ **Signal combination logic created**
- ✅ **Data provider interface created**
- ✅ **Existing components moved and organized**

## Next Steps

1. **Complete remaining modules**: Implement missing components in each directory
2. **Update imports**: Update all import statements throughout the codebase
3. **Integration testing**: Test integration with existing components
4. **Documentation**: Complete documentation for all modules
5. **Performance optimization**: Optimize for production use

## Benefits

- **Clear separation of concerns**: Each module has a specific responsibility
- **Better maintainability**: Related components are grouped together
- **Easier testing**: Each component can be tested independently
- **Scalability**: Easy to add new strategies, risk models, or execution engines
- **Integration**: Seamless integration with existing ML pipeline components
- **Regime awareness**: All components are designed to work with regime detection
- **Risk management**: Built-in risk management at every level