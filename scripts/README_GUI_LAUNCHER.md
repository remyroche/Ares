# GUI Launcher System for Ares Trading Bot

This system automatically launches the GUI when starting the Ares trading bot in any mode. The GUI provides real-time monitoring, kill switch control, backtesting, model management, and trade analysis.

## 🚀 Quick Start

### 1. Install GUI Dependencies
```bash
# Check if dependencies are available
python scripts/gui_launcher.py --check-deps

# Install dependencies if needed
python scripts/gui_launcher.py --install-deps
```

### 2. Launch Bot with GUI

#### Paper Trading
```bash
python scripts/launch_paper_trading.py BTCUSDT BINANCE
```

#### Live Trading (with safety confirmation)
```bash
python scripts/launch_live_trading.py BTCUSDT BINANCE
```

#### Portfolio Manager
```bash
python scripts/launch_portfolio_manager.py
```

#### Full Test Run
```bash
python scripts/launch_full_test_run.py BTCUSDT
```

#### Custom Command
```bash
python scripts/launch_with_gui.py main_launcher.py trade ETHUSDT BINANCE paper
```

## 📋 Available Scripts

### Core Launcher
- **`gui_launcher.py`** - Main GUI launcher with dependency management
- **`launch_with_gui.py`** - Generic wrapper for any bot command

### Specific Mode Launchers
- **`launch_paper_trading.py`** - Paper trading with GUI
- **`launch_live_trading.py`** - Live trading with GUI (with safety checks)
- **`launch_portfolio_manager.py`** - Portfolio manager with GUI
- **`launch_full_test_run.py`** - Full test run with GUI

## 🔧 Features

### Automatic GUI Launch
- Automatically detects bot mode from command line arguments
- Starts both API server (port 8000) and frontend (port 3000)
- Monitors processes and restarts if they crash
- Graceful shutdown on Ctrl+C

### Dependency Management
- Checks for Node.js, npm, and Python dependencies
- Automatic installation of npm packages
- Validation of GUI directory structure

### Safety Features
- Live trading confirmation prompt
- Process monitoring and auto-restart
- Graceful shutdown handling
- Error logging and reporting

## 🎯 GUI Features

### Real-time Monitoring
- Live performance metrics
- Position tracking
- Trade history
- System status

### Kill Switch Control
- Emergency stop functionality
- Reason logging
- Real-time status updates
- WebSocket notifications

### Backtesting
- Strategy testing interface
- Parameter optimization
- Results comparison
- Export capabilities

### Model Management
- Model performance visualization
- Deployment controls
- Version tracking
- Performance comparison

### Trade Analysis
- Detailed trade breakdown
- Performance metrics
- Technical indicators
- Risk analysis

### System Management
- CPU/Memory monitoring
- Uptime tracking
- System controls
- Status indicators

## 📊 Usage Examples

### Paper Trading BTC/USDT
```bash
python scripts/launch_paper_trading.py BTCUSDT BINANCE
```
- Launches paper trading for Bitcoin
- Opens GUI at http://localhost:3000
- API docs at http://localhost:8000/docs

### Live Trading with Safety
```bash
python scripts/launch_live_trading.py ETHUSDT BINANCE
```
- Prompts for confirmation before live trading
- Uses real money - trade carefully!
- Full GUI monitoring and control

### Portfolio Management
```bash
python scripts/launch_portfolio_manager.py
```
- Launches global portfolio manager
- Multi-asset portfolio tracking
- Risk management interface

### Full Test Run
```bash
python scripts/launch_full_test_run.py BTCUSDT
```
- Complete training and testing pipeline
- Model validation and backtesting
- Performance analysis

### Custom Commands
```bash
# Any bot command with GUI
python scripts/launch_with_gui.py main_launcher.py backtest soft
python scripts/launch_with_gui.py scripts/training_cli.py train BTCUSDT
```

## ⚙️ Configuration

### Disable GUI
```bash
# Use --no-gui flag
python scripts/launch_paper_trading.py BTCUSDT BINANCE --no-gui
```

### Check Dependencies
```bash
python scripts/gui_launcher.py --check-deps
```

### Install Dependencies
```bash
python scripts/gui_launcher.py --install-deps
```

### Custom Bot Command
```bash
python scripts/gui_launcher.py --bot-command main_launcher.py trade BTCUSDT BINANCE paper
```

## 🔍 Troubleshooting

### GUI Won't Start
1. Check dependencies: `python scripts/gui_launcher.py --check-deps`
2. Install dependencies: `python scripts/gui_launcher.py --install-deps`
3. Check ports: Ensure ports 3000 and 8000 are available
4. Check logs: Look for error messages in the terminal

### Port Conflicts
- Frontend: Change port in `GUI/vite.config.js`
- API: Change port in `GUI/api_server.py`
- Update both files to match

### Process Issues
- GUI launcher monitors and auto-restarts crashed processes
- Check terminal output for error messages
- Use Ctrl+C to gracefully shutdown

### Dependencies Missing
```bash
# Install Node.js and npm
# Then run:
python scripts/gui_launcher.py --install-deps
```

## 📁 File Structure

```
scripts/
├── gui_launcher.py              # Main GUI launcher
├── launch_with_gui.py           # Generic wrapper
├── launch_paper_trading.py      # Paper trading launcher
├── launch_live_trading.py       # Live trading launcher
├── launch_portfolio_manager.py  # Portfolio manager launcher
├── launch_full_test_run.py      # Full test run launcher
└── README_GUI_LAUNCHER.md      # This file

GUI/
├── api_server.py                # FastAPI backend
├── package.json                 # Frontend dependencies
├── vite.config.js              # Vite configuration
├── src/                        # React frontend
└── start_gui.sh               # Manual GUI startup script
```

## 🚨 Safety Notes

### Live Trading
- Always use paper trading first
- Double-check settings before live trading
- Monitor positions closely
- Use kill switch if needed

### System Resources
- GUI requires additional system resources
- Monitor CPU and memory usage
- Close unnecessary applications
- Consider running on dedicated machine

### Network Security
- GUI runs on localhost by default
- Don't expose to public network
- Use firewall if needed
- Secure API keys and credentials

## 🔄 Integration

The GUI launcher integrates with:
- **main_launcher.py** - Core bot launcher
- **training_cli.py** - Training pipeline
- **StateManager** - Kill switch control
- **PerformanceReporter** - Trade analysis
- **SQLiteManager** - Data persistence

## 📈 Performance

### Startup Time
- API server: ~3 seconds
- Frontend: ~5 seconds
- Total: ~8 seconds

### Resource Usage
- API server: ~50MB RAM
- Frontend: ~100MB RAM
- Total: ~150MB RAM

### Monitoring
- Process health checks every 5 seconds
- Auto-restart on crashes
- Graceful shutdown handling

## 🎯 Next Steps

1. **Test all modes** with different symbols
2. **Configure alerts** for important events
3. **Customize dashboard** for your needs
4. **Set up monitoring** for production use
5. **Backup configurations** regularly

---

For more information, see the main GUI documentation in `GUI/README.md`. 