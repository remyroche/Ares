# Daily Trading Recording Template

## Overview

This template provides a **single-line-per-day** summary of all trading activities, capturing comprehensive metrics in a structured CSV format for long-term analysis and tracking.

## 📊 **CSV Structure: One Line Per Day**

### **File Format**
- **Filename**: `daily_trading_log.csv`
- **Format**: CSV with headers
- **Frequency**: One record per trading day
- **Backup**: Monthly backup files (`daily_records_YYYY_MM.csv`)

### **Record Structure** (50+ fields per day)

## 📅 **Field Definitions**

### **1. Date & Basic Trading Info**
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `date` | ISO Date | Trading date | `2024-12-17` |
| `total_trades` | Integer | Total trades executed | `15` |
| `winning_trades` | Integer | Profitable trades | `9` |
| `losing_trades` | Integer | Loss-making trades | `5` |
| `break_even_trades` | Integer | Zero PnL trades | `1` |
| `win_rate` | Float | Win rate percentage | `0.6000` (60%) |

### **2. Performance Metrics**
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `total_pnl` | Float | Total profit/loss ($) | `287.50` |
| `gross_profit` | Float | Total profits ($) | `425.75` |
| `gross_loss` | Float | Total losses ($) | `138.25` |
| `profit_factor` | Float | Gross profit / gross loss | `3.08` |
| `best_trade` | Float | Best single trade ($) | `156.80` |
| `worst_trade` | Float | Worst single trade ($) | `-45.20` |
| `avg_trade_pnl` | Float | Average PnL per trade | `19.17` |
| `sharpe_ratio` | Float | Risk-adjusted return | `1.25` |

### **3. Risk Metrics**
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `max_drawdown` | Float | Maximum drawdown % | `0.0850` (8.5%) |
| `avg_portfolio_risk` | Float | Average portfolio risk % | `0.0235` (2.35%) |
| `max_portfolio_risk` | Float | Maximum portfolio risk % | `0.0450` (4.5%) |
| `avg_leverage` | Float | Average leverage used | `1.8` |
| `max_leverage` | Float | Maximum leverage used | `3.2` |

### **4. ML Model Performance**
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `models_used_count` | Integer | Number of unique models | `4` |
| `models_used_list` | String | Pipe-separated model list | `analyst_ensemble_v1.2\|tactician_timing_v1.1\|hmm_regime_v2.0` |
| `avg_model_confidence` | Float | Average model confidence | `0.7650` (76.5%) |
| `best_model_accuracy` | Float | Best model accuracy | `0.8200` (82%) |
| `worst_model_accuracy` | Float | Worst model accuracy | `0.6800` (68%) |
| `model_agreement_score` | Float | Inter-model agreement | `0.1500` (high agreement) |

### **5. Signal Analysis**
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `avg_signal_confidence` | Float | Average signal confidence | `0.7420` (74.2%) |
| `avg_signal_strength` | Float | Average signal strength | `0.8100` (81%) |
| `signal_accuracy` | Float | Signal prediction accuracy | `0.7333` (73.3%) |

### **6. Regime Analysis**
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `primary_regime` | String | Most common regime | `trending_up` |
| `regime_changes` | Integer | Number of regime transitions | `3` |
| `avg_regime_confidence` | Float | Average regime confidence | `0.8150` (81.5%) |
| `regime_stability` | Float | Regime stability score | `0.8000` (80%) |

### **7. Execution Quality**
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `avg_execution_quality` | Float | Average execution quality | `0.9200` (92%) |
| `avg_slippage` | Float | Average slippage | `0.0012` (0.12%) |
| `avg_commission` | Float | Average commission ($) | `12.50` |
| `execution_success_rate` | Float | Successful executions % | `0.9333` (93.3%) |
| `avg_execution_time_ms` | Float | Average execution time | `245.60` ms |

### **8. Market Context**
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `market_volatility` | Float | Average market volatility | `0.0285` (2.85%) |
| `market_trend` | String | Overall market trend | `bullish` |
| `avg_price` | Float | Average trading price | `3025.40` |
| `price_range_pct` | Float | Daily price range % | `0.0320` (3.2%) |
| `volume_profile` | String | Volume characterization | `high` |

### **9. Feature Importance**
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `top_features` | JSON String | Top 5 most important features | `{"close": 0.25, "sma_20": 0.18, "rsi": 0.12, "volatility_20": 0.10, "volume": 0.08}` |

### **10. Notable Events**
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `notable_events_count` | Integer | Number of notable events | `2` |
| `notable_events` | String | Pipe-separated events | `HIGH_WIN_RATE:60.0%\|LARGE_WIN:156.8` |

### **11. Session Information**
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `sessions_count` | Integer | Number of trading sessions | `2` |
| `total_session_duration_hours` | Float | Total session time | `6.50` |
| `avg_session_duration_hours` | Float | Average session duration | `3.25` |

### **12. System Health**
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `system_uptime_pct` | Float | System uptime percentage | `98.50` |
| `error_count` | Integer | Number of errors | `0` |
| `warning_count` | Integer | Number of warnings | `2` |

## 🎯 **Notable Events Codes**

### **Performance Events**
- `LARGE_WIN:XXX` - Large winning trade (>$500)
- `LARGE_LOSS:XXX` - Large losing trade (<-$200)
- `HIGH_WIN_RATE:XX%` - Win rate >80%
- `HIGH_PROFIT_FACTOR:X.X` - Profit factor >3.0
- `EXCELLENT_DAY` - Multiple positive metrics

### **Risk Events**
- `HIGH_DRAWDOWN:XX%` - Drawdown >10%
- `HIGH_LEVERAGE:X.X` - Leverage >5x
- `HIGH_RISK_EXPOSURE` - Portfolio risk >5%
- `RISK_LIMIT_HIT` - Risk limits reached

### **Model Events**
- `HIGH_MODEL_CONFIDENCE:XX%` - Model confidence >90%
- `LOW_MODEL_AGREEMENT` - Models disagreeing
- `MODEL_PERFORMANCE_DEGRADATION` - Accuracy dropping
- `NEW_MODEL_DEPLOYED` - New model version

### **Market Events**
- `HIGH_VOLATILITY` - Market volatility >5%
- `REGIME_CHANGE:FROM->TO` - Major regime transition
- `BREAKOUT_EVENT` - Significant price breakout
- `MARKET_STRESS` - Unusual market conditions

### **System Events**
- `HIGH_ACTIVITY:XX` - >50 trades in a day
- `NO_TRADING` - Zero trades executed
- `SYSTEM_RESTART` - System was restarted
- `POOR_EXECUTION_QUALITY` - Execution issues

## 📈 **Example Daily Records**

### **Successful Trading Day**
```csv
2024-12-17,15,9,5,1,0.6000,287.50,425.75,138.25,3.08,156.80,-45.20,19.17,1.25,0.0850,0.0235,0.0450,1.8,3.2,4,analyst_ensemble_v1.2|hmm_regime_v2.0|tactician_timing_v1.1|volatility_model,0.7650,0.8200,0.6800,0.1500,0.7420,0.8100,0.7333,trending_up,3,0.8150,0.8000,0.9200,0.0012,12.50,0.9333,245.60,0.0285,bullish,3025.40,0.0320,high,"{""close"": 0.25, ""sma_20"": 0.18, ""rsi"": 0.12, ""volatility_20"": 0.10, ""volume"": 0.08}",2,HIGH_WIN_RATE:60.0%|LARGE_WIN:156.8,2,6.50,3.25,98.50,0,2
```

### **Challenging Trading Day**
```csv
2024-12-16,8,4,4,0,0.5000,-45.20,180.30,225.50,0.80,95.40,-85.60,-5.65,0.15,0.1250,0.0320,0.0520,2.1,4.5,3,analyst_ensemble_v1.2|tactician_timing_v1.1|hmm_regime_v2.0,0.6890,0.7500,0.6200,0.1300,0.6750,0.7200,0.5000,high_volatility,5,0.7200,0.3750,0.8500,0.0018,8.75,0.8750,320.80,0.0420,bearish,2985.75,0.0580,very_high,"{""volatility_20"": 0.28, ""close"": 0.22, ""rsi"": 0.15, ""macd"": 0.12, ""volume"": 0.10}",3,HIGH_REGIME_VOLATILITY:5|HIGH_SLIPPAGE:0.2%,1,4.25,4.25,96.20,1,5
```

### **No Trading Day**
```csv
2024-12-14,0,0,0,0,0.0000,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.0000,0.0000,0.0000,0.0,0.0,0,,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,unknown,0,0.0000,0.0000,0.0000,0.0000,0.00,0.0000,0.00,0.0000,neutral,0.00,0.0000,normal,"{}",1,NO_TRADING,0,0.00,0.00,100.00,0,0
```

## 🔧 **Usage Examples**

### **Automatic Daily Recording**
```python
from src.trading.reporting.daily_recorder import record_daily_trading_summary

# Automatically record today's trading
success = await record_daily_trading_summary(
    trades=completed_trades,
    sessions=trading_sessions,
    target_date=date.today()
)
```

### **Reading Historical Data**
```python
from src.trading.reporting.daily_recorder import get_trading_history

# Get last 30 days of trading history
history_df = await get_trading_history(days=30)

# Analyze trends
print(f"Average daily PnL: ${history_df['total_pnl'].mean():.2f}")
print(f"Best day: ${history_df['total_pnl'].max():.2f}")
print(f"Worst day: ${history_df['total_pnl'].min():.2f}")
print(f"Average win rate: {history_df['win_rate'].mean():.1%}")
```

### **Trend Analysis**
```python
# Load daily records
df = pd.read_csv('daily_trading_records/daily_trading_log.csv')
df['date'] = pd.to_datetime(df['date'])

# Performance trends
df['cumulative_pnl'] = df['total_pnl'].cumsum()
df['rolling_win_rate'] = df['win_rate'].rolling(7).mean()  # 7-day average

# Model performance trends
df['model_confidence_trend'] = df['avg_model_confidence'].rolling(7).mean()

# Risk trends
df['risk_trend'] = df['avg_portfolio_risk'].rolling(7).mean()
```

## 📊 **Analysis Capabilities**

### **Performance Analysis**
```python
# Monthly performance
monthly_performance = df.groupby(df['date'].dt.to_period('M')).agg({
    'total_pnl': 'sum',
    'total_trades': 'sum',
    'win_rate': 'mean',
    'sharpe_ratio': 'mean'
})

# Best/worst days
best_days = df.nlargest(5, 'total_pnl')[['date', 'total_pnl', 'win_rate', 'notable_events']]
worst_days = df.nsmallest(5, 'total_pnl')[['date', 'total_pnl', 'win_rate', 'notable_events']]
```

### **Model Performance Tracking**
```python
# Model confidence trends
model_confidence_trend = df[['date', 'avg_model_confidence', 'best_model_accuracy', 'model_agreement_score']]

# Model usage analysis
df['primary_model'] = df['models_used_list'].str.split('|').str[0]
model_usage = df['primary_model'].value_counts()
```

### **Risk Analysis**
```python
# Risk metrics over time
risk_analysis = df[['date', 'max_drawdown', 'avg_portfolio_risk', 'avg_leverage']]

# High-risk days
high_risk_days = df[df['max_portfolio_risk'] > 0.05]  # >5% portfolio risk
```

### **Market Regime Analysis**
```python
# Regime distribution
regime_distribution = df['primary_regime'].value_counts()

# Regime performance
regime_performance = df.groupby('primary_regime').agg({
    'total_pnl': 'mean',
    'win_rate': 'mean',
    'avg_regime_confidence': 'mean'
})
```

## 🚀 **Key Benefits**

### **Long-Term Tracking**
- **Historical performance** trends over months/years
- **Model evolution** tracking across versions
- **Risk exposure** patterns over time
- **Market regime** performance analysis

### **Quick Daily Review**
- **Single line** contains all key metrics
- **Notable events** highlight important occurrences
- **Performance summary** at a glance
- **Risk assessment** for the day

### **Trend Analysis**
- **Rolling averages** for performance trends
- **Seasonal patterns** in trading performance
- **Model degradation** detection over time
- **Risk accumulation** monitoring

### **Reporting & Compliance**
- **Daily trading log** for regulatory requirements
- **Performance attribution** by model and strategy
- **Risk monitoring** for compliance limits
- **Audit trail** for all trading decisions

## 📋 **Template Files Created**

1. **`daily_trading_log_template.csv`** - Empty template with headers
2. **`daily_trading_log_example.csv`** - Example with sample data
3. **`daily_recorder.py`** - Python implementation
4. **`DAILY_RECORDING_TEMPLATE.md`** - This documentation

## 🔄 **Automated Recording Process**

The system automatically:
1. **Collects all trades** from the day
2. **Aggregates performance metrics** across all trades
3. **Calculates model performance** statistics
4. **Identifies notable events** based on thresholds
5. **Writes single CSV line** with all metrics
6. **Creates monthly backups** for data safety
7. **Validates data integrity** before writing

## 💡 **Usage Integration**

### **In Trading Orchestrator**
```python
# At end of each trading day
await orchestrator.record_daily_summary()
```

### **Scheduled Recording**
```python
# Run daily at market close
@scheduled_task(time="16:00")  # 4 PM market close
async def daily_recording_task():
    await record_daily_trading_summary(
        trades=get_todays_trades(),
        sessions=get_todays_sessions()
    )
```

This template provides **complete daily visibility** into trading operations while maintaining a **compact, analyzable format** perfect for long-term performance tracking and regulatory compliance.