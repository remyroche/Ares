# Cross-Asset Trading System

This document describes the enhanced trading system that supports **simultaneous trading across multiple cryptocurrencies** while ensuring **only one trade executes at a time** and provides **consolidated cross-asset reporting**.

## 🚀 Overview

The cross-asset trading system extends the existing single-symbol trading architecture to support:

1. **Multi-Symbol Trading**: Launch trading on several cryptocurrencies simultaneously
2. **Trade Semaphore**: Ensure only one trade executes at a time across all symbols
3. **Consolidated Reporting**: Dedicated module with unified performance reporting
4. **Cross-Asset Risk Management**: Portfolio-level risk controls and correlation monitoring

## 📁 New Components

### Core Components

- **`CrossAssetTradingManager`** (`execution/cross_asset_trading_manager.py`)
  - Main coordinator for multi-symbol trading
  - Manages trade execution queue and semaphore
  - Provides consolidated reporting
  - Handles cross-asset risk management

- **`CrossAssetConfig`** (`config/cross_asset_config.py`)
  - Configuration system for multiple symbols
  - Symbol-specific parameters and risk limits
  - Cross-asset strategy settings

### Enhanced Components

- **Enhanced `TradingOrchestrator`**
  - Added trade decision callbacks for cross-asset coordination
  - Improved integration with cross-asset manager

## 🎯 Key Features

### 1. Multi-Symbol Trading
```python
# Trade multiple symbols simultaneously
symbols = ['ETHUSDT', 'BTCUSDT', 'ADAUSDT', 'SOLUSDT']
manager = await start_cross_asset_trading(
    symbols=symbols,
    exchange='binance',
    trading_mode='paper'
)
```

### 2. Trade Semaphore System
- **Single Trade Execution**: Only one trade can execute at a time across all symbols
- **Queue Management**: Trades are queued and executed sequentially
- **Cross-Asset Risk Checks**: Each trade is validated against portfolio-level constraints

### 3. Consolidated Reporting
```python
# Generate comprehensive cross-asset reports
report = await manager.generate_consolidated_report("session")
await manager.export_consolidated_report(report, "cross_asset_report.json")
```

### 4. Cross-Asset Risk Management
- **Portfolio Exposure Limits**: Maximum exposure across all symbols
- **Symbol Concentration Limits**: Maximum exposure per symbol
- **Correlation Monitoring**: Track cross-asset correlations
- **Dynamic Allocation**: Adjust position sizes based on market conditions

## ⚙️ Configuration

### Basic Setup
```python
config = {
    'symbols': ['ETHUSDT', 'BTCUSDT', 'ADAUSDT'],
    'exchange': 'binance',
    'trading_mode': 'paper',
    'total_account_balance': 10000.0,
    'max_concurrent_symbols': 3,
    'risk_per_trade': 0.02
}
```

### Symbol-Specific Configuration
```python
symbol_configs = {
    'ETHUSDT': {
        'account_balance': 3000.0,
        'volatility_adjustment': 1.2,
        'confidence_threshold': 0.65,
        'max_position_size': 0.3
    },
    'BTCUSDT': {
        'account_balance': 4000.0,
        'volatility_adjustment': 1.0,
        'confidence_threshold': 0.7,
        'max_position_size': 0.35
    }
}
```

## 📊 Usage Examples

### 1. Quick Start
```python
from src.trading.execution.cross_asset_trading_manager import start_cross_asset_trading

async def main():
    # Start trading on multiple symbols
    manager = await start_cross_asset_trading(
        symbols=['ETHUSDT', 'BTCUSDT', 'ADAUSDT'],
        trading_mode='paper'
    )

    # Monitor performance
    stats = manager.get_manager_stats()
    print(f"Total PnL: ${stats['current_session']['total_pnl']:.2f}")

    # Generate report
    report = await manager.generate_consolidated_report()
    print(f"Cross-asset correlation: {report['cross_asset_metrics']['cross_correlation']}")

    # Cleanup
    await manager.stop_trading_session()
```

### 2. Advanced Configuration
```python
from src.trading.config.cross_asset_config import CrossAssetConfig, SymbolConfiguration

async def advanced_setup():
    # Create comprehensive configuration
    config = CrossAssetConfig(
        symbols=['ETHUSDT', 'BTCUSDT', 'SOLUSDT', 'ADAUSDT', 'DOTUSDT'],
        total_account_balance=10000.0,
        max_concurrent_symbols=3,
        strategy='equal_weight',
        enable_correlation_monitoring=True
    )

    # Customize symbol configurations
    eth_config = SymbolConfiguration(
        symbol='ETHUSDT',
        account_balance=3000.0,
        volatility_adjustment=1.2,
        confidence_threshold=0.65
    )
    config.update_symbol_config('ETHUSDT', eth_config)

    # Start trading
    manager = CrossAssetTradingManager(config)
    await manager.initialize()
    await manager.start_trading_session()
```

### 3. Real-Time Monitoring
```python
async def monitor_trading():
    while manager.is_running:
        stats = manager.get_manager_stats()

        # Check trade queue
        trades_in_queue = stats['current_session']['trades_in_queue']
        executed_trades = stats['current_session']['executed_trades']

        # Monitor individual orchestrators
        for symbol, orch_stats in stats['orchestrator_stats'].items():
            status = orch_stats['status']
            total_trades = orch_stats['performance_metrics']['total_trades']

        await asyncio.sleep(30)  # Check every 30 seconds
```

## 📈 Monitoring & Reporting

### Real-Time Statistics
```python
stats = manager.get_manager_stats()
print(f"""
Cross-Asset Trading Statistics:
- Symbols Active: {len(stats['current_session']['symbols'])}
- Total PnL: ${stats['current_session']['total_pnl']:.2f}
- Trades in Queue: {stats['current_session']['trades_in_queue']}
- Executed Trades: {stats['current_session']['executed_trades']}
- Success Rate: {stats['performance_metrics']['successful_trades'] / stats['performance_metrics']['total_trades']:.1%}
""")
```

### Consolidated Reports
```python
# Generate comprehensive report
report = await manager.generate_consolidated_report("session")

# Key sections in the report:
print("Session Info:", report['session_info'])
print("Cross-Asset Metrics:", report['cross_asset_metrics'])
print("Symbol Performance:", report['symbol_performance'])
print("Executed Trades:", len(report['executed_trades']))
```

### Export Options
```python
# Export to JSON
await manager.export_consolidated_report(report, "cross_asset_report.json")

# Export to CSV
report_csv = manager.generate_consolidated_report("session", "csv")
```

## 🔧 Architecture Changes

### Before (Single Symbol)
```
TradingOrchestrator (ETH only)
    ↓
Single trade execution
    ↓
Single symbol reporting
```

### After (Multi-Symbol with Cross-Asset Control)
```
CrossAssetTradingManager
    ↓
Multiple TradingOrchestrators (ETH, BTC, ADA, etc.)
    ↓
Trade Queue + Semaphore (one trade at a time)
    ↓
Consolidated Reporting & Risk Management
```

### Key Integration Points

1. **Trade Decision Flow**:
   - Each `TradingOrchestrator` generates trade decisions
   - Decisions are queued in `CrossAssetTradingManager`
   - Semaphore ensures sequential execution

2. **Risk Management**:
   - Pre-trade validation at portfolio level
   - Symbol concentration limits
   - Cross-correlation monitoring

3. **Reporting Integration**:
   - Individual orchestrator statistics
   - Cross-asset performance aggregation
   - Consolidated risk metrics

## 🚨 Safety Features

### Trade Execution Control
- **Semaphore**: Only one trade executes at a time
- **Queue Management**: FIFO order processing
- **Timeout Handling**: Automatic cleanup of stuck trades

### Risk Management
- **Portfolio Limits**: Maximum total exposure
- **Symbol Limits**: Maximum per-symbol exposure
- **Correlation Checks**: Prevent over-concentration

### Error Handling
- **Graceful Degradation**: Individual symbol failures don't stop others
- **Automatic Recovery**: Failed trades are marked and skipped
- **Comprehensive Logging**: Full audit trail

## 🏃 Running the Demo

### Quick Demo
```bash
cd /workspace
python -m src.trading.examples.cross_asset_trading_demo
```

### Custom Demo Duration
```bash
# Run for 2 hours
DEMO_DURATION_HOURS=2 python -m src.trading.examples.cross_asset_trading_demo
```

### Demo Output
The demo generates:
- Real-time statistics every 5 minutes
- Consolidated performance reports
- Individual symbol performance
- Final comprehensive summary
- Exported JSON reports in `demo_cross_asset_reports/`

## 🔍 Monitoring Files

### Generated Reports
- `demo_cross_asset_reports/final_demo_report.json` - Complete summary
- `demo_cross_asset_reports/cross_asset_report_[timestamp].json` - Periodic reports
- `trading_reports/` - Individual symbol reports

### Log Files
- `logs/` - System logs with cross-asset events
- Real-time console output with colored indicators

## 🎛️ Customization

### Adding New Symbols
```python
# Add new symbol
config.add_symbol('LINKUSDT', SymbolConfiguration(
    symbol='LINKUSDT',
    account_balance=800.0,
    volatility_adjustment=1.6,
    confidence_threshold=0.55
))
```

### Custom Strategies
```python
# Market cap weighted allocation
config.strategy = CrossAssetStrategy.MARKET_CAP_WEIGHT

# Correlation minimized allocation
config.strategy = CrossAssetStrategy.CORRELATION_MINIMIZED
```

### Risk Parameter Tuning
```python
# Adjust risk parameters
config.max_portfolio_risk = 0.03  # 3% max portfolio risk
config.max_symbol_concentration = 0.25  # 25% max per symbol
config.risk_per_trade = 0.015  # 1.5% per trade
```

## 🔗 Integration with Existing System

The cross-asset trading system is designed to integrate seamlessly with existing components:

- **Existing `TradingOrchestrator`** instances work unchanged
- **Monitoring systems** receive enhanced cross-asset data
- **Reporting systems** get consolidated views
- **Risk management** is enhanced with portfolio-level controls

### Backward Compatibility
- Single-symbol trading continues to work as before
- Existing configurations are preserved
- New features are opt-in via configuration

## 🚨 Important Notes

### System Requirements
- Python 3.8+
- asyncio support for concurrent execution
- Sufficient memory for multiple symbol data streams

### Performance Considerations
- Each symbol runs its own `TradingOrchestrator`
- Trade semaphore ensures sequential execution
- Memory usage scales with number of symbols

### Risk Management
- Always test with paper trading first
- Monitor cross-asset correlations
- Set appropriate risk limits for your capital

## 📞 Support

For issues or questions about the cross-asset trading system:

1. Check the demo output for error messages
2. Review the generated reports for performance insights
3. Examine the system logs for detailed debugging information
4. Ensure all symbols have proper market data access

---

**Next Steps**: Run the demo to see cross-asset trading in action, then customize the configuration for your specific trading strategy and risk tolerance.