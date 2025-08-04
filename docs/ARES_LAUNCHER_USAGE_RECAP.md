# Ares Launcher Usage Recap

## 🚀 **Overview**

`ares_launcher.py` is the comprehensive launcher for the Ares trading bot that provides a unified interface for all operations. This document provides a complete recap of all available modes and their usage.

## 📋 **Complete Usage Guide**

### **1. Data Loading** 📊
**Purpose**: Load historical market data (klines, aggtrades, futures) without backtesting.

**Usage**:
```bash
# Load data for ETHUSDT on Binance
python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE

# Load data for BTCUSDT on MEXC
python ares_launcher.py load --symbol BTCUSDT --exchange MEXC

# Load data for ETHUSDT on Gate.io
python ares_launcher.py load --symbol ETHUSDT --exchange GATEIO
```

**Features**:
- Downloads historical klines data
- Downloads aggregated trades data
- Downloads futures data (if available)
- Stores data in organized structure
- Supports multiple exchanges (BINANCE, MEXC, GATEIO)

**Data Structure**:
```
data/
├── ETHUSDT_1h.csv
├── ETHUSDT_aggtrades.csv
├── ETHUSDT_futures.csv
└── ...
```

---

### **2. Backtesting** 🔄
**Purpose**: Run comprehensive backtesting on historical data with performance analysis.

**Usage**:
```bash
# Run backtesting for ETHUSDT on Binance
python ares_launcher.py backtest --symbol ETHUSDT --exchange BINANCE

# Run backtesting with GUI
python ares_launcher.py backtest --symbol ETHUSDT --exchange BINANCE --gui
```

**Features**:
- Uses existing historical data
- Comprehensive performance metrics
- Trade analysis and statistics
- Risk management evaluation
- Performance visualization
- Optional GUI integration

**Output**:
- Backtest results in `backtests/` directory
- Performance reports
- Trade logs
- Risk metrics

---

### **3. Blank Training** 🎯
**Purpose**: Run enhanced model training with efficiency optimizations for large datasets.

**Usage**:
```bash
# Run blank training for ETHUSDT on Binance
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE

# Run blank training with GUI
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --gui
```

**Features**:
- Uses existing data (no new downloads)
- Enhanced training pipeline
- Efficiency optimizations
- Large dataset handling
- Model validation
- Performance optimization

**Training Pipeline**:
1. Data preparation and cleaning
2. Feature engineering
3. Model training (multiple algorithms)
4. Hyperparameter optimization
5. Model validation and testing
6. Model persistence

---

### **4. Model Training** 🤖
**Purpose**: Complete model training with backtesting and paper trading integration.

**Usage**:
```bash
# Run complete model training
python ares_launcher.py model_trainer --symbol ETHUSDT --exchange BINANCE

# Run with GUI
python ares_launcher.py model_trainer --symbol ETHUSDT --exchange BINANCE --gui
```

**Features**:
- Complete training pipeline
- Backtesting integration
- Paper trading simulation
- Performance validation
- Model comparison
- Automated optimization

**Pipeline Steps**:
1. Data collection and preparation
2. Model training
3. Backtesting validation
4. Paper trading simulation
5. Performance analysis
6. Model deployment

---

### **5. Paper Trading** 📈
**Purpose**: Simulated trading environment for testing strategies without real money.

**Usage**:
```bash
# Run paper trading
python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE

# Run with GUI
python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE --gui
```

**Features**:
- Real-time market data simulation
- Virtual portfolio management
- Trade execution simulation
- Performance tracking
- Risk management
- Real-time monitoring

**Benefits**:
- Test strategies safely
- Validate model performance
- Debug trading logic
- Optimize parameters
- Real-time feedback

---

### **6. Shadow Trading** 🌐
**Purpose**: Use exchange testnet for real API testing without real money.

**Usage**:
```bash
# Run shadow trading (testnet)
python ares_launcher.py live --symbol ETHUSDT --exchange BINANCE
```

**Features**:
- Real exchange API integration
- Testnet environment
- Real order execution (testnet)
- Real-time market data
- Actual trading logic testing
- Exchange-specific features

**Supported Exchanges**:
- Binance Testnet
- MEXC Testnet
- Gate.io Testnet

---

### **7. Unified Regime Classifier** 🧠
**Purpose**: Train and use the advanced unified regime classification system.

**Usage**:
```bash
# Train regime classifier on 2 years data
python ares_launcher.py regime --regime-subcommand load --symbol ETHUSDT --exchange BINANCE

# Train regime classifier on 2 years data (full training)
python ares_launcher.py regime --regime-subcommand train --symbol ETHUSDT --exchange BINANCE

# Quick training on 30 days data
python ares_launcher.py regime --regime-subcommand train_blank --symbol ETHUSDT --exchange BINANCE
```

**Features**:
- **HMM-based labeling**: Hidden Markov Model for basic regime detection
- **Ensemble prediction**: Random Forest, LightGBM, SVM with majority voting
- **Advanced classification**: HUGE_CANDLE, SR_ZONE_ACTION detection
- **SR Analyzer integration**: Support/Resistance level analysis
- **Candle Analyzer integration**: Advanced candle pattern recognition
- **Adaptive thresholds**: Volatility-based size classification

**Regime Types**:
- **BULL**: Bullish market conditions
- **BEAR**: Bearish market conditions
- **SIDEWAYS**: Sideways/consolidation market
- **HUGE_CANDLE**: Large candle detection
- **SR_ZONE_ACTION**: Support/Resistance zone activity

**Training Modes**:
- **load**: Train on 2 years data (730 days)
- **train**: Full training on 2 years data
- **train_blank**: Quick training on 30 days data

---

### **8. Portfolio Trading** 💼
**Purpose**: Multi-token trading with portfolio management.

**Usage**:
```bash
# Run portfolio trading
python ares_launcher.py portfolio

# Run with GUI
python ares_launcher.py portfolio --gui
```

**Features**:
- Multi-token trading
- Portfolio diversification
- Risk management
- Performance tracking
- Automated rebalancing
- Multi-exchange support

---

### **9. GUI Interface** 🖥️
**Purpose**: Web-based graphical user interface for monitoring and control.

**Usage**:
```bash
# Launch GUI only
python ares_launcher.py gui

# Launch GUI with specific mode
python ares_launcher.py gui --mode paper --symbol ETHUSDT --exchange BINANCE
```

**Features**:
- Real-time monitoring
- Performance dashboards
- Trade visualization
- Configuration management
- System status monitoring
- Interactive controls

**GUI Modes**:
- **paper**: Paper trading interface
- **backtest**: Backtesting interface
- **training**: Training interface
- **portfolio**: Portfolio management interface

---

### **10. Challenger Mode** ⚡
**Purpose**: Advanced trading with challenger model integration.

**Usage**:
```bash
# Run challenger trading
python ares_launcher.py challenger --symbol ETHUSDT --exchange BINANCE

# Run with GUI
python ares_launcher.py challenger --symbol ETHUSDT --exchange BINANCE --gui
```

**Features**:
- Advanced model integration
- Enhanced performance
- Sophisticated strategies
- Real-time adaptation
- Advanced risk management

---

## 🔧 **Common Parameters**

### **Basic Parameters**
```bash
--symbol SYMBOL          # Trading symbol (e.g., ETHUSDT, BTCUSDT)
--exchange EXCHANGE      # Exchange name (BINANCE, MEXC, GATEIO)
--gui                    # Enable GUI integration
--lookback-days DAYS     # Historical data lookback period
```

### **Advanced Parameters**
```bash
--regime-subcommand CMD  # Regime classifier subcommand
--mode MODE             # GUI mode specification
--quick-test           # Quick testing mode
--verbose              # Verbose logging
```

---

## 📊 **Data Requirements**

### **For Training**
- Historical klines data (OHLCV)
- Aggregated trades data
- Futures data (optional)
- Minimum 30 days for quick training
- Recommended 2 years for full training

### **For Backtesting**
- Historical data in proper format
- Sufficient data for lookback period
- Clean, validated data

### **For Paper Trading**
- Real-time market data access
- Historical data for initialization
- Exchange API credentials (for shadow trading)

---

## 🚀 **Quick Start Examples**

### **1. Load Data and Train Model**
```bash
# Step 1: Load historical data
python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE

# Step 2: Train regime classifier
python ares_launcher.py regime --regime-subcommand train --symbol ETHUSDT --exchange BINANCE

# Step 3: Run paper trading
python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE
```

### **2. Quick Testing**
```bash
# Quick training on 30 days data
python ares_launcher.py regime --regime-subcommand train_blank --symbol ETHUSDT --exchange BINANCE

# Quick backtesting
python ares_launcher.py backtest --symbol ETHUSDT --exchange BINANCE
```

### **3. Full Pipeline**
```bash
# Complete pipeline with GUI
python ares_launcher.py model_trainer --symbol ETHUSDT --exchange BINANCE --gui
```

### **4. Portfolio Management**
```bash
# Multi-token portfolio trading
python ares_launcher.py portfolio --gui
```

---

## 📈 **Performance Monitoring**

### **Real-time Monitoring**
- GUI interface for live monitoring
- Performance dashboards
- Trade visualization
- Risk metrics

### **Historical Analysis**
- Backtest results analysis
- Performance reports
- Trade logs
- Risk assessment

### **System Health**
- Model performance tracking
- System status monitoring
- Error logging and debugging
- Performance optimization

---

## 🔒 **Security and Safety**

### **Paper Trading**
- No real money involved
- Safe strategy testing
- Risk-free experimentation

### **Shadow Trading**
- Testnet environment
- Real API testing
- No financial risk

### **Data Protection**
- Secure data storage
- Encrypted configurations
- Safe API key management

---

## 🛠️ **Troubleshooting**

### **Common Issues**
1. **Data not found**: Run data loading first
2. **API errors**: Check exchange credentials
3. **Training failures**: Verify data quality
4. **GUI not loading**: Check port availability

### **Debug Mode**
```bash
# Enable verbose logging
python ares_launcher.py [mode] --symbol ETHUSDT --exchange BINANCE --verbose
```

### **Log Files**
- Check `log/` directory for detailed logs
- Error logs in `log/ares_errors_*.log`
- Performance logs in `log/ares_*.log`

---

## 📚 **Advanced Usage**

### **Custom Configurations**
- Modify `src/config.py` for custom settings
- Environment variables for sensitive data
- Configuration files for different environments

### **API Integration**
- Exchange-specific API configurations
- WebSocket connections for real-time data
- REST API for historical data

### **Model Customization**
- Custom feature engineering
- Algorithm selection
- Hyperparameter tuning
- Model ensemble configuration

---

## 🎯 **Best Practices**

### **Data Management**
- Regular data updates
- Data quality validation
- Backup and recovery
- Version control for configurations

### **Training Strategy**
- Start with quick training
- Validate with backtesting
- Test with paper trading
- Deploy with shadow trading

### **Risk Management**
- Position sizing
- Stop-loss configuration
- Portfolio diversification
- Performance monitoring

---

## 📞 **Support and Documentation**

### **Documentation**
- `docs/` directory for detailed documentation
- Code comments for implementation details
- Configuration examples

### **Logging**
- Comprehensive logging system
- Error tracking and debugging
- Performance monitoring
- System health checks

### **Community**
- GitHub repository for issues
- Documentation updates
- Feature requests
- Bug reports

---

## 🎉 **Conclusion**

The Ares Launcher provides a comprehensive, unified interface for all trading bot operations. From data loading to live trading, from model training to portfolio management, it offers a complete solution for algorithmic trading.

**Key Benefits**:
- ✅ **Unified Interface**: Single launcher for all operations
- ✅ **Comprehensive Coverage**: All trading bot features
- ✅ **Easy to Use**: Simple command-line interface
- ✅ **GUI Integration**: Web-based monitoring interface
- ✅ **Flexible Configuration**: Customizable for different needs
- ✅ **Safe Testing**: Paper trading and shadow trading modes
- ✅ **Advanced Features**: Regime classification, portfolio management
- ✅ **Performance Monitoring**: Real-time tracking and analysis

The launcher is designed to be user-friendly while providing access to all advanced features of the Ares trading system.

---

## 📋 **Complete Command Reference**

### **Data Operations**
```bash
# Load data
python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE

# Load data with custom lookback
python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE --lookback-days 365
```

### **Training Operations**
```bash
# Blank training
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE

# Model training
python ares_launcher.py model_trainer --symbol ETHUSDT --exchange BINANCE

# Regime classifier training
python ares_launcher.py regime --regime-subcommand train --symbol ETHUSDT --exchange BINANCE

# Quick regime training
python ares_launcher.py regime --regime-subcommand train_blank --symbol ETHUSDT --exchange BINANCE
```

### **Trading Operations**
```bash
# Paper trading
python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE

# Shadow trading
python ares_launcher.py live --symbol ETHUSDT --exchange BINANCE

# Challenger trading
python ares_launcher.py challenger --symbol ETHUSDT --exchange BINANCE

# Portfolio trading
python ares_launcher.py portfolio
```

### **Analysis Operations**
```bash
# Backtesting
python ares_launcher.py backtest --symbol ETHUSDT --exchange BINANCE

# Backtesting with GUI
python ares_launcher.py backtest --symbol ETHUSDT --exchange BINANCE --gui
```

### **GUI Operations**
```bash
# GUI only
python ares_launcher.py gui

# GUI with specific mode
python ares_launcher.py gui --mode paper --symbol ETHUSDT --exchange BINANCE
```

### **Advanced Operations**
```bash
# Verbose logging
python ares_launcher.py [mode] --symbol ETHUSDT --exchange BINANCE --verbose

# Quick testing
python ares_launcher.py [mode] --symbol ETHUSDT --exchange BINANCE --quick-test
```

This comprehensive guide covers all aspects of the Ares Launcher, providing users with a complete reference for all available operations and their usage. 