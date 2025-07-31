# Ares Launcher System

## Overview

The Ares Launcher System provides a unified interface for launching the Ares trading bot with various operational modes. It consolidates multiple launcher scripts into a single, comprehensive system that handles process management, GUI integration, and portfolio management.

## Architecture

### Core Components

#### **AresLauncher Class**
- **Process Management**: Tracks and manages all subprocesses
- **GUI Integration**: Launches and manages GUI server
- **Portfolio Management**: Handles multi-token trading
- **Cleanup System**: Graceful termination of all processes

#### **Supported Modes**
1. **Blank Training**: Fast testing with minimal trials
2. **Backtesting**: Comprehensive validation
3. **Paper Trading**: Safe shadow trading
4. **Live Trading**: Production trading
5. **Portfolio Trading**: Multi-token with portfolio manager
6. **GUI Integration**: Web interface with optional trading modes

---

## Usage Guide

### Basic Commands

#### **1. Blank Training Run**
```bash
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE
```
**Purpose**: Fast testing and development
- Uses `scripts/blank_training_run.py`
- Minimal hyperparameter trials (4 total)
- Reduced data lookback period
- Perfect for quick validation

**Expected Duration**: 5-15 minutes
**Use Case**: Development, testing, quick validation

#### **2. Backtesting**
```bash
python ares_launcher.py backtest --symbol ETHUSDT --exchange BINANCE
```
**Purpose**: Comprehensive model validation
- Uses `scripts/training_cli.py full-test-run`
- Full training pipeline
- Complete backtesting analysis
- Model preparation for trading

**Expected Duration**: 30-60 minutes
**Use Case**: Model validation, performance analysis

#### **3. Paper Trading**
```bash
python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE
```
**Purpose**: Safe trading with real market data
- Runs backtesting first
- Launches paper trading bot
- No real money involved
- Real market conditions

**Expected Duration**: Backtesting + continuous trading
**Use Case**: Strategy validation, risk-free testing

#### **4. Live Trading**
```bash
python ares_launcher.py live --symbol ETHUSDT --exchange BINANCE
```
**Purpose**: Production trading with real money
- Runs backtesting first
- Launches live trading bot
- Real money trading
- Production environment

**Expected Duration**: Backtesting + continuous trading
**Use Case**: Production trading, real profit/loss

#### **5. Portfolio Trading**
```bash
python ares_launcher.py portfolio
```
**Purpose**: Multi-token trading with portfolio management
- Launches portfolio manager
- Runs backtesting for all tokens in config
- Launches live trading bots for all tokens
- Comprehensive portfolio management

**Expected Duration**: Variable (depends on number of tokens)
**Use Case**: Multi-token portfolio, diversified trading

#### **6. GUI Only**
```bash
python ares_launcher.py gui
```
**Purpose**: Web interface access
- Launches GUI server only
- Access via web browser
- No trading functionality
- Monitoring and configuration

**Expected Duration**: Continuous
**Use Case**: Monitoring, configuration, analysis

#### **7. GUI + Trading Mode**
```bash
python ares_launcher.py gui --mode paper --symbol ETHUSDT --exchange BINANCE
```
**Purpose**: GUI with trading functionality
- Launches GUI server
- Runs specified trading mode
- Web interface + trading
- Real-time monitoring

**Expected Duration**: Continuous
**Use Case**: Trading with web interface, real-time monitoring

#### **8. All Modes with GUI Integration**
```bash
# Blank training with GUI
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --gui

# Backtesting with GUI
python ares_launcher.py backtest --symbol ETHUSDT --exchange BINANCE --gui

# Paper trading with GUI
python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE --gui

# Live trading with GUI
python ares_launcher.py live --symbol ETHUSDT --exchange BINANCE --gui

# Portfolio trading with GUI
python ares_launcher.py portfolio --gui
```
**Purpose**: All trading modes with web interface
- Launches GUI for all trading modes
- Real-time monitoring via web interface
- Enhanced user experience
- Comprehensive dashboard access

**Expected Duration**: Variable based on mode
**Use Case**: All trading scenarios with web monitoring

---

## Technical Features

### **Process Management**

#### **Automatic Cleanup**
```python
def cleanup(self):
    """Cleanup processes on exit."""
    # Terminate GUI process
    # Terminate portfolio process
    # Terminate all trading bots
    # Graceful shutdown with timeout
```

#### **Signal Handling**
```python
# Responds to Ctrl+C and SIGTERM
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
```

#### **Process Tracking**
- **GUI Process**: Web interface server
- **Portfolio Process**: Global portfolio manager
- **Trading Processes**: Individual trading bots
- **Training Processes**: Model training and validation

### **Safety Features**

#### **Backtesting First**
```python
def run_paper_trading(self, symbol: str, exchange: str):
    # First run backtesting
    if not self.run_backtesting(symbol, exchange):
        return False
    # Then launch trading bot
```

#### **Validation Pipeline**
1. **Model Training**: Enhanced 4-stage optimization
2. **Backtesting**: Comprehensive validation
3. **Model Preparation**: Ready for trading
4. **Trading Launch**: Safe deployment

#### **Error Handling**
- **Graceful failures**: Continue with other tokens
- **Process isolation**: One failure doesn't affect others
- **Automatic cleanup**: Clean termination on errors
- **Detailed logging**: Comprehensive error reporting

### **GUI Integration**

#### **GUI Server Launch**
```python
def launch_gui(self):
    # Uses GUI/start.sh to launch both API server and frontend
    gui_cmd = ["bash", "GUI/start.sh"]
    self.gui_process = subprocess.Popen(gui_cmd, cwd="GUI")
```

#### **GUI Components**
- **API Server**: FastAPI server on port 8000
- **Frontend**: React/Vite app on port 3000
- **WebSocket**: Real-time updates
- **Dashboard**: Comprehensive trading dashboard

#### **GUI + Trading Modes**
- **GUI + Blank**: Web interface with fast testing
- **GUI + Backtest**: Web interface with validation
- **GUI + Paper**: Web interface with paper trading
- **GUI + Live**: Web interface with live trading
- **GUI + Portfolio**: Web interface with multi-token trading

#### **GUI Integration Options**
- **--gui flag**: Add GUI to any trading mode
- **GUI-only mode**: Launch GUI without trading
- **GUI + specific mode**: Launch GUI with specific trading mode
- **Automatic health checks**: Verify GUI components are running

### **Portfolio Management**

#### **Multi-Token Support**
```python
def run_portfolio_trading(self):
    # Get all supported tokens from config
    supported_tokens = CONFIG.get("SUPPORTED_TOKENS", {})
    # Launch trading bots for each token
    # Manage portfolio across all tokens
```

#### **Portfolio Manager Integration**
- **Global Portfolio Manager**: Manages all tokens
- **Risk Management**: Portfolio-level risk control
- **Performance Tracking**: Multi-token performance
- **Capital Allocation**: Dynamic capital distribution

---

## Configuration

### **Supported Tokens Configuration**
```python
# In config.py
"SUPPORTED_TOKENS": {
    "BINANCE": [
        "ETHUSDT",
        "BTCUSDT",
        "ADAUSDT",
        # ... more tokens
    ]
}
```

### **Environment Variables**
```bash
# Trading environment
TRADING_ENVIRONMENT=PAPER  # PAPER, LIVE, TESTNET

# API credentials
BINANCE_LIVE_API_KEY=your_api_key
BINANCE_LIVE_API_SECRET=your_api_secret

# Logging
LOG_LEVEL=INFO
```

### **Process Configuration**
```python
# Timeout settings
GUI_STARTUP_TIMEOUT = 5  # seconds
PORTFOLIO_STARTUP_TIMEOUT = 10  # seconds
PROCESS_TERMINATION_TIMEOUT = 5  # seconds
```

---

## Workflow Examples

### **Development Workflow**
```bash
# 1. Quick testing
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE

# 2. Validation
python ares_launcher.py backtest --symbol ETHUSDT --exchange BINANCE

# 3. Safe testing
python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE

# 4. Production
python ares_launcher.py live --symbol ETHUSDT --exchange BINANCE
```

### **Development Workflow with GUI**
```bash
# 1. Quick testing with GUI
python ares_launcher.py blank --symbol ETHUSDT --exchange BINANCE --gui

# 2. Validation with GUI
python ares_launcher.py backtest --symbol ETHUSDT --exchange BINANCE --gui

# 3. Safe testing with GUI
python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE --gui

# 4. Production with GUI
python ares_launcher.py live --symbol ETHUSDT --exchange BINANCE --gui
```

### **Portfolio Management Workflow**
```bash
# 1. Launch portfolio trading
python ares_launcher.py portfolio

# 2. Monitor via GUI
python ares_launcher.py gui --mode paper --symbol ETHUSDT --exchange BINANCE
```

### **Portfolio Management Workflow with GUI**
```bash
# 1. Launch portfolio trading with GUI
python ares_launcher.py portfolio --gui

# 2. Monitor via web interface
# Open browser to http://localhost:3000
```

### **GUI-Centric Workflow**
```bash
# 1. Launch GUI with paper trading
python ares_launcher.py gui --mode paper --symbol ETHUSDT --exchange BINANCE

# 2. Access web interface
# Open browser to http://localhost:3000

# 3. Monitor and configure via web interface
```

### **GUI-Only Workflow**
```bash
# 1. Launch GUI only
python ares_launcher.py gui

# 2. Access web interface
# Open browser to http://localhost:3000

# 3. Use web interface for all operations
```

---

## Monitoring and Logging

### **Log Levels**
- **INFO**: General operations and status
- **WARNING**: Non-critical issues
- **ERROR**: Process failures and errors
- **CRITICAL**: System failures

### **Log Categories**
- **Process Management**: Process start/stop/cleanup
- **Trading Operations**: Trading bot activities
- **GUI Operations**: Web interface activities
- **Portfolio Management**: Multi-token operations
- **Error Handling**: Failures and recovery

### **Monitoring Features**
- **Real-time Status**: Process health monitoring
- **Performance Metrics**: Trading performance tracking
- **Error Alerts**: Automatic error notification
- **Resource Usage**: CPU, memory, network monitoring

---

## Error Handling

### **Common Error Scenarios**

#### **1. Process Startup Failures**
```python
# GUI server fails to start
if self.gui_process.poll() is not None:
    self.logger.error("❌ Failed to start GUI server")
    return False
```

#### **2. Backtesting Failures**
```python
# Backtesting fails for a token
if not self.run_backtesting(symbol, exchange):
    self.logger.warning(f"⚠️  Skipping {symbol} due to backtesting failure")
    continue
```

#### **3. Trading Bot Failures**
```python
# Individual trading bot fails
if process.poll() is not None:
    self.logger.error(f"❌ Trading bot for {symbol} failed")
    # Continue with other bots
```

### **Recovery Strategies**
- **Automatic Retry**: Retry failed operations
- **Graceful Degradation**: Continue with available services
- **Process Isolation**: One failure doesn't affect others
- **Cleanup on Exit**: Proper resource cleanup

---

## Performance Considerations

### **Resource Usage**
- **Memory**: ~100-500MB per trading bot
- **CPU**: Variable based on trading frequency
- **Network**: API calls to exchanges
- **Disk**: Log files and data storage

### **Optimization Tips**
- **Use blank training** for development
- **Run backtesting** during off-peak hours
- **Monitor resource usage** during live trading
- **Use GUI** for monitoring instead of console

### **Scaling Considerations**
- **Single Token**: Light resource usage
- **Multi-Token**: Higher resource requirements
- **Portfolio Trading**: Maximum resource usage
- **GUI Integration**: Additional web server overhead

---

## Security Features

### **API Key Management**
- **Environment Variables**: Secure credential storage
- **No Hardcoding**: Credentials never in code
- **Access Control**: Limited API permissions
- **Monitoring**: API usage tracking

### **Trading Safety**
- **Paper Trading**: Safe testing environment
- **Validation**: Always backtest before live
- **Risk Management**: Portfolio-level risk control
- **Monitoring**: Real-time performance tracking

### **Process Security**
- **Isolation**: Process-level isolation
- **Cleanup**: Proper resource cleanup
- **Error Handling**: Secure error handling
- **Logging**: Audit trail maintenance

---

## Troubleshooting

### **Common Issues**

#### **1. GUI Not Starting**
```bash
# Check if port is in use
lsof -i :5000

# Check GUI server logs
tail -f logs/gui_server.log
```

#### **2. Trading Bot Not Starting**
```bash
# Check API credentials
echo $BINANCE_LIVE_API_KEY

# Check trading environment
echo $TRADING_ENVIRONMENT

# Check logs
tail -f logs/trading_bot.log
```

#### **3. Portfolio Manager Issues**
```bash
# Check portfolio manager logs
tail -f logs/portfolio_manager.log

# Check database connection
python -c "from src.database.sqlite_manager import SQLiteManager; print('DB OK')"
```

### **Debug Commands**
```bash
# Verbose logging
LOG_LEVEL=DEBUG python ares_launcher.py paper --symbol ETHUSDT --exchange BINANCE

# Check process status
ps aux | grep python

# Check port usage
netstat -tulpn | grep :5000
```

---

## Future Enhancements

### **Planned Features**
1. **Web Dashboard**: Enhanced web interface
2. **Mobile App**: Mobile monitoring app
3. **Alert System**: Email/SMS notifications
4. **Advanced Analytics**: Performance analytics
5. **Cloud Deployment**: Cloud-based deployment
6. **API Integration**: REST API for external tools

### **Performance Improvements**
1. **Parallel Processing**: Multi-threaded operations
2. **Caching**: Intelligent data caching
3. **Optimization**: Process optimization
4. **Monitoring**: Enhanced monitoring capabilities

---

## Conclusion

The Ares Launcher System provides a comprehensive, safe, and flexible way to run the Ares trading bot. It consolidates multiple launcher scripts into a unified interface while maintaining all the functionality and adding enhanced process management, GUI integration, and portfolio management capabilities.

### **Key Benefits**
- **Unified Interface**: Single script for all operations
- **Process Management**: Automatic cleanup and monitoring
- **Safety Features**: Always validates before live trading
- **GUI Integration**: Optional web interface
- **Portfolio Support**: Multi-token trading
- **Error Handling**: Robust error handling and recovery

### **Next Steps**
1. **Test all modes** with different tokens
2. **Monitor performance** in production
3. **Gather feedback** for improvements
4. **Implement enhancements** based on usage

---

*This document should be updated as the launcher system evolves and new features are added.* 