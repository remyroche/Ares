# 🚀 Ares Trading Bot - Live Deployment Checklist

## ✅ **CRITICAL PRE-DEPLOYMENT CHECKS**

### 1. **Environment Configuration**
- [ ] **Create `.env` file** from `.env.example`
- [ ] **Set `TRADING_ENVIRONMENT="LIVE"`** in `.env`
- [ ] **Add Binance Live API credentials**:
  - `BINANCE_LIVE_API_KEY="your_live_api_key"`
  - `BINANCE_LIVE_API_SECRET="your_live_api_secret"`
- [ ] **Add Email credentials** for alerts:
  - `EMAIL_SENDER_ADDRESS="your_email@gmail.com"`
  - `EMAIL_SENDER_PASSWORD="your_app_password"`
  - `EMAIL_RECIPIENT_ADDRESS="recipient_email@gmail.com"`

### 2. **Directory Structure**
- [ ] **Create required directories**:
  ```bash
  mkdir -p data logs reports models checkpoints backtests
  ```
- [ ] **Verify SQLite database path**: `data/ares_local_db.sqlite`

### 3. **Dependencies Installation**
- [ ] **Install all requirements**:
  ```bash
  pip install -r requirements.txt
  ```
- [ ] **Verify key packages**: `aiohttp`, `websockets`, `pandas`, `numpy`, `pydantic-settings`

### 4. **Database Initialization**
- [ ] **Initialize SQLite database**:
  ```bash
  python -c "from src.database.sqlite_manager import sqlite_manager; import asyncio; asyncio.run(sqlite_manager.initialize())"
  ```

### 5. **Configuration Validation**
- [ ] **Test configuration loading**:
  ```bash
  python -c "from src.config import settings; print(f'Trading Environment: {settings.trading_environment}')"
  ```
- [ ] **Verify API connectivity** (testnet first):
  ```bash
  python -c "from exchange.binance import BinanceExchange; import asyncio; print('API test successful')"
  ```

## 🔧 **SYSTEM COMPONENTS VERIFICATION**

### 6. **Core Modules**
- [ ] **Error Handler**: All decorators working
- [ ] **Logger**: Structured logging functional
- [ ] **State Manager**: File persistence working
- [ ] **Database Manager**: SQLite operations working
- [ ] **Exchange Client**: API calls functional

### 7. **Trading Components**
- [ ] **Analyst**: Feature engineering and analysis
- [ ] **Strategist**: Strategy formulation
- [ ] **Tactician**: Trade execution
- [ ] **Supervisor**: Risk management and monitoring
- [ ] **Sentinel**: System health monitoring

### 8. **New Features (Recently Added)**
- [ ] **Profit Sweep System**: Daily balance checking and transfer
- [ ] **Withdrawal Alerts**: Email notifications for manual withdrawal
- [ ] **Daily Balance Logging**: CSV file tracking
- [ ] **Micro-movement Trading**: Enhanced for small price variations

## 🧪 **TESTING PHASES**

### 9. **Paper Trading Test**
- [ ] **Run in PAPER mode** first:
  ```bash
  python -m src.main
  ```
- [ ] **Verify all components start** without errors
- [ ] **Check logs** for any warnings or errors
- [ ] **Test profit sweep** with paper trading

### 10. **Testnet Validation**
- [ ] **Switch to TESTNET**:
  ```bash
  # Set in .env: TRADING_ENVIRONMENT="TESTNET"
  ```
- [ ] **Add testnet API credentials**
- [ ] **Run full trading cycle** on testnet
- [ ] **Verify profit sweep** works with real API calls

### 11. **Live Trading Preparation**
- [ ] **Switch to LIVE mode**:
  ```bash
  # Set in .env: TRADING_ENVIRONMENT="LIVE"
  ```
- [ ] **Add live API credentials**
- [ ] **Start with small position sizes**
- [ ] **Monitor closely for first 24 hours**

## 📊 **MONITORING SETUP**

### 12. **Logging Configuration**
- [ ] **Verify log files** are being created in `logs/`
- [ ] **Check log rotation** is working
- [ ] **Test error reporting** via email

### 13. **Database Monitoring**
- [ ] **Verify SQLite database** is being updated
- [ ] **Check backup scheduling** is working
- [ ] **Test database migration** tools

### 14. **Performance Monitoring**
- [ ] **Verify performance metrics** are being calculated
- [ ] **Check drawdown monitoring** is active
- [ ] **Test risk management** triggers

## 🚨 **SAFETY MEASURES**

### 15. **Risk Management**
- [ ] **Set conservative position sizes** initially
- [ ] **Verify stop-loss mechanisms** are working
- [ ] **Test pause trading** functionality
- [ ] **Check drawdown limits** are enforced

### 16. **Emergency Procedures**
- [ ] **Test kill switch** functionality
- [ ] **Verify email alerts** are working
- [ ] **Test manual intervention** procedures
- [ ] **Document emergency contacts**

### 17. **Backup & Recovery**
- [ ] **Test database backup** functionality
- [ ] **Verify state persistence** across restarts
- [ ] **Test crash recovery** procedures
- [ ] **Document recovery procedures**

## 📈 **LIVE DEPLOYMENT**

### 18. **Go-Live Checklist**
- [ ] **All tests passed** in testnet
- [ ] **Risk parameters** set conservatively
- [ ] **Monitoring dashboards** ready
- [ ] **Emergency procedures** documented
- [ ] **Team notifications** sent

### 19. **Post-Deployment Monitoring**
- [ ] **Monitor first 24 hours** closely
- [ ] **Check all alerts** are working
- [ ] **Verify profit sweep** is functioning
- [ ] **Monitor performance** against expectations
- [ ] **Document any issues** encountered

### 20. **Ongoing Maintenance**
- [ ] **Daily balance checks** via profit sweep
- [ ] **Weekly performance reviews**
- [ ] **Monthly model retraining** (automated)
- [ ] **Regular backup verification**
- [ ] **System health monitoring**

## 🔧 **TROUBLESHOOTING**

### Common Issues:
1. **API Connection Errors**: Check credentials and network
2. **Database Errors**: Verify SQLite permissions and disk space
3. **Email Alerts Not Working**: Check SMTP credentials
4. **Profit Sweep Failing**: Verify transfer permissions on Binance
5. **Performance Degradation**: Check system resources and logs

### Emergency Contacts:
- **Technical Issues**: Check logs in `logs/` directory
- **Trading Issues**: Check `data/daily_balances.csv` for balance history
- **System Health**: Monitor `src/sentinel/sentinel.py` outputs

---

## 📝 **DEPLOYMENT NOTES**

### Environment Variables Required:
```bash
TRADING_ENVIRONMENT=LIVE
BINANCE_LIVE_API_KEY=your_key
BINANCE_LIVE_API_SECRET=your_secret
EMAIL_SENDER_ADDRESS=your_email
EMAIL_SENDER_PASSWORD=your_password
EMAIL_RECIPIENT_ADDRESS=recipient_email
```

### Key Files to Monitor:
- `data/ares_local_db.sqlite` - Main database
- `data/daily_balances.csv` - Balance tracking
- `logs/` - System logs
- `reports/` - Performance reports

### Commands for Monitoring:
```bash
# Check system status
python -c "from src.utils.state_manager import StateManager; sm = StateManager(); print(sm.get_state('global_trading_status'))"

# Check recent balances
tail -f data/daily_balances.csv

# Monitor logs
tail -f logs/ares_*.log
```

---

**⚠️ IMPORTANT**: Always start with paper trading, then testnet, before going live. Monitor closely during the first deployment and have emergency procedures ready. 