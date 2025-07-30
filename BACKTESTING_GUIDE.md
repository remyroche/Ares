# 🚀 Ares Trading Bot - Backtesting Guide

## 📋 **Overview**

The Ares trading bot supports two types of backtesting:

1. **Initial Backtesting** - Quick backtesting using existing models
2. **In-Depth Backtesting** - Full training pipeline with model optimization

## 🎯 **1. Initial Backtesting (Quick Backtesting)**

### **Purpose**
- Test existing models with historical data
- Quick performance evaluation
- No model training involved

### **Command Structure**
```bash
python backtesting/main_orchestrator.py --mode backtest --symbol <SYMBOL> --exchange <EXCHANGE>
```

### **Examples**

#### **Basic Backtesting (BTCUSDT)**
```bash
python backtesting/main_orchestrator.py --mode backtest --symbol BTCUSDT --exchange binance
```

#### **Backtesting Different Assets**
```bash
# Ethereum
python backtesting/main_orchestrator.py --mode backtest --symbol ETHUSDT --exchange binance

# Cardano
python backtesting/main_orchestrator.py --mode backtest --symbol ADAUSDT --exchange binance

# Solana
python backtesting/main_orchestrator.py --mode backtest --symbol SOLUSDT --exchange binance
```

#### **Using Default Settings**
```bash
# Uses default symbol from config (usually BTCUSDT)
python backtesting/main_orchestrator.py --mode backtest
```

### **What Happens During Initial Backtesting**
1. **Data Loading**: Loads historical klines, aggregated trades, and futures data
2. **Feature Engineering**: Calculates technical indicators and features
3. **Model Prediction**: Uses existing trained models to generate predictions
4. **Backtesting**: Simulates trading based on predictions
5. **Performance Analysis**: Generates performance metrics and reports

### **Output Files**
- **Backtest Results**: `data/backtest_results_<SYMBOL>_<TIMESTAMP>.json`
- **Trade Logs**: `data/trade_logs_<SYMBOL>_<TIMESTAMP>.csv`
- **Performance Reports**: `reports/backtest_report_<SYMBOL>_<TIMESTAMP>.html`

## 🔬 **2. In-Depth Backtesting (Full Training Pipeline)**

### **Purpose**
- Complete model training from scratch
- Hyperparameter optimization
- Walk-forward validation
- Monte Carlo simulations
- A/B testing setup

### **Command Structure**
```bash
python backtesting/main_orchestrator.py --mode full_backtest --symbol <SYMBOL> --exchange <EXCHANGE>
```

### **Examples**

#### **Full Training + Backtesting (BTCUSDT)**
```bash
python backtesting/main_orchestrator.py --mode full_backtest --symbol BTCUSDT --exchange binance
```

#### **Full Training + Backtesting (Different Assets)**
```bash
# Ethereum with full training
python backtesting/main_orchestrator.py --mode full_backtest --symbol ETHUSDT --exchange binance

# Cardano with full training
python backtesting/main_orchestrator.py --mode full_backtest --symbol ADAUSDT --exchange binance
```

### **What Happens During In-Depth Backtesting**

#### **Phase 1: Training Pipeline**
1. **Data Collection**: Downloads historical data if not available
2. **Data Preparation**: Cleans and prepares data for training
3. **Feature Engineering**: Calculates all technical indicators and features
4. **Model Training**: Trains multiple ML models (LightGBM, XGBoost, Neural Networks)
5. **Ensemble Creation**: Creates meta-models and ensemble predictions
6. **Hyperparameter Optimization**: Uses Bayesian optimization to find best parameters
7. **Walk-Forward Validation**: Tests models on out-of-sample data
8. **Monte Carlo Validation**: Runs multiple simulations for robustness
9. **A/B Testing Setup**: Prepares models for live A/B testing

#### **Phase 2: Backtesting**
1. **Model Loading**: Loads newly trained models
2. **Prediction Generation**: Uses optimized models for predictions
3. **Trading Simulation**: Simulates trading with new models
4. **Performance Analysis**: Comprehensive performance evaluation
5. **Report Generation**: Detailed reports and visualizations

### **Output Files**
- **Trained Models**: `models/<SYMBOL>_<MODEL_TYPE>_<TIMESTAMP>.joblib`
- **Training Data**: `data/training/<SYMBOL>_training_data_<TIMESTAMP>.csv`
- **Optimization Results**: `data/optimization_results_<SYMBOL>_<TIMESTAMP>.json`
- **Validation Results**: `data/validation_results_<SYMBOL>_<TIMESTAMP>.json`
- **Backtest Results**: `data/backtest_results_<SYMBOL>_<TIMESTAMP>.json`
- **Comprehensive Reports**: `reports/full_training_report_<SYMBOL>_<TIMESTAMP>.html`

## 🛠️ **3. Alternative Training Methods**

### **Using Training CLI (Recommended for Advanced Users)**

#### **Full Training via CLI**
```bash
python scripts/training_cli.py train BTCUSDT BINANCE
```

#### **Model Retraining**
```bash
python scripts/training_cli.py retrain BTCUSDT BINANCE
```

#### **Check Training Status**
```bash
python scripts/training_cli.py status BTCUSDT
```

#### **List Available Models**
```bash
python scripts/training_cli.py list-models
```

### **Direct Training Manager Usage**

#### **Python Script for Custom Training**
```python
import asyncio
from src.database.sqlite_manager import SQLiteManager
from src.training.training_manager import TrainingManager

async def custom_training():
    # Initialize
    db_manager = SQLiteManager()
    await db_manager.initialize()
    
    training_manager = TrainingManager(db_manager)
    
    # Run full training
    success = await training_manager.run_full_training("BTCUSDT", "BINANCE")
    
    if success:
        print("✅ Training completed successfully")
    else:
        print("❌ Training failed")

# Run the training
asyncio.run(custom_training())
```

## 📊 **4. Configuration Options**

### **Modifying Backtesting Parameters**

Edit `src/config.py` to customize:

```python
# Backtesting Configuration
"backtesting": {
    "initial_equity": 10000,  # Starting capital
    "commission_rate": 0.001,  # Trading commission
    "slippage": 0.0001,       # Price slippage
    "max_position_size": 0.1,  # Maximum position size (% of equity)
    "risk_per_trade": 0.02,   # Risk per trade (% of equity)
    "stop_loss_pct": 0.05,    # Stop loss percentage
    "take_profit_pct": 0.10,  # Take profit percentage
}

# Training Configuration
"training": {
    "lookback_years": 2,      # Historical data years
    "train_test_split": 0.8,  # Training/test split
    "validation_split": 0.2,  # Validation split
    "hyperparameter_tuning": True,
    "ensemble_methods": ["stacking", "blending"],
    "monte_carlo_simulations": 100,
    "walk_forward_windows": 10
}
```

### **Environment Variables**

Set in `.env` file:
```bash
# Trading Environment
TRADING_ENVIRONMENT=PAPER  # or LIVE for live trading

# API Keys (for data download)
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret

# Database Configuration
SQLITE_DB_PATH=data/ares_local_db.sqlite
```

## 📈 **5. Expected Results**

### **Initial Backtesting Results**
- **Execution Time**: 5-15 minutes
- **Memory Usage**: 2-4 GB RAM
- **Output**: Performance metrics, trade logs, basic reports

### **In-Depth Backtesting Results**
- **Execution Time**: 30 minutes - 2 hours
- **Memory Usage**: 4-8 GB RAM
- **Output**: Trained models, comprehensive reports, validation results

### **Performance Metrics**
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Total Return**: Overall percentage gain/loss

## 🔧 **6. Troubleshooting**

### **Common Issues**

#### **Data Download Failures**
```bash
# Check API keys in .env file
cat .env | grep BINANCE

# Verify internet connection
ping api.binance.com
```

#### **Memory Issues**
```bash
# Increase Python memory limit
export PYTHONMALLOC=malloc
export PYTHONUNBUFFERED=1
```

#### **Model Training Failures**
```bash
# Check available disk space
df -h

# Verify Python dependencies
pip list | grep -E "(scikit-learn|lightgbm|xgboost)"
```

### **Debug Mode**

Enable debug logging:
```bash
# Set debug environment variable
export ARES_DEBUG=1

# Run with verbose output
python backtesting/main_orchestrator.py --mode backtest --symbol BTCUSDT -v
```

## 📋 **7. Quick Start Checklist**

### **Before Running Backtesting**
- [ ] **Environment Setup**: Python 3.8+, all dependencies installed
- [ ] **Configuration**: `.env` file with API keys (for data download)
- [ ] **Directories**: `data/`, `models/`, `reports/` directories exist
- [ ] **Database**: SQLite database initialized
- [ ] **Internet**: Stable connection for data download

### **Running Initial Backtesting**
```bash
# 1. Quick backtest with existing models
python backtesting/main_orchestrator.py --mode backtest --symbol BTCUSDT

# 2. Check results
ls -la data/backtest_results_*
ls -la reports/
```

### **Running In-Depth Backtesting**
```bash
# 1. Full training + backtesting
python backtesting/main_orchestrator.py --mode full_backtest --symbol BTCUSDT

# 2. Check all outputs
ls -la models/
ls -la data/training/
ls -la reports/
```

## 🎯 **8. Best Practices**

### **For Initial Backtesting**
- Start with default settings
- Test multiple timeframes
- Compare different symbols
- Use paper trading environment

### **For In-Depth Backtesting**
- Run during off-peak hours
- Monitor system resources
- Save results for comparison
- Document parameter changes

### **Performance Optimization**
- Use SSD storage for faster I/O
- Increase RAM for large datasets
- Use multiple CPU cores for training
- Monitor GPU usage if available

This comprehensive guide should help you successfully run both types of backtesting with the Ares trading bot. 