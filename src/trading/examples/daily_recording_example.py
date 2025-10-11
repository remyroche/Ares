"""
Daily Recording Example

Demonstrates the daily trading recording system that creates
one-line-per-day summaries with comprehensive metrics.
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_structured, LogLevel
from ..reporting.daily_recorder import (
    daily_recorder, record_daily_trading_summary,
    get_daily_trading_summary, get_trading_history
)
from ..monitoring.comprehensive_trade_monitor import comprehensive_trade_monitor

async def demonstrate_daily_recording():
    """
    Demonstrate the daily recording system with examples.
    """
    
    tprint_info("📅 Daily Trading Recording System Demonstration")
    print("=" * 80)
    
    try:
        # Step 1: Show daily recording template structure
        tprint_info("📋 Step 1: Daily Recording Template Structure")
        
        show_template_structure()
        
        # Step 2: Simulate daily recording
        tprint_info("📝 Step 2: Simulating Daily Recording")
        
        await simulate_daily_recordings()
        
        # Step 3: Read and analyze historical data
        tprint_info("📊 Step 3: Analyzing Historical Daily Records")
        
        await analyze_historical_records()
        
        # Step 4: Show trend analysis
        tprint_info("📈 Step 4: Performance Trend Analysis")
        
        await show_trend_analysis()
        
        tprint_success("🎉 Daily recording demonstration completed!")
        
    except Exception as e:
        tprint_warning(f"❌ Demonstration failed: {e}")

def show_template_structure():
    """Show the structure of the daily recording template."""
    
    template_info = {
        "File Format": "CSV with headers",
        "Frequency": "One line per trading day",
        "Total Fields": "50+ comprehensive metrics",
        "Field Categories": [
            "Basic Trading (6 fields): trades, wins, losses, win rate",
            "Performance (8 fields): PnL, profit factor, Sharpe ratio",
            "Risk (5 fields): drawdown, leverage, portfolio risk",
            "ML Models (6 fields): confidence, accuracy, agreement",
            "Signals (3 fields): confidence, strength, accuracy",
            "Regime (4 fields): type, changes, confidence, stability",
            "Execution (5 fields): quality, slippage, timing",
            "Market (5 fields): volatility, trend, price range",
            "Features (1 field): top 5 important features (JSON)",
            "Events (2 fields): notable events and count",
            "Sessions (3 fields): count, duration, average",
            "System (3 fields): uptime, errors, warnings"
        ]
    }
    
    tprint_structured(template_info, LogLevel.INFO)
    
    # Show example field explanations
    tprint_info("📊 Key Field Examples:")
    
    example_fields = {
        "total_pnl": "287.50 (Total profit/loss for the day in $)",
        "win_rate": "0.6000 (60% of trades were profitable)",
        "models_used_list": "analyst_ensemble_v1.2|tactician_timing_v1.1|hmm_regime_v2.0",
        "avg_model_confidence": "0.7650 (76.5% average model confidence)",
        "primary_regime": "trending_up (Most common market regime)",
        "top_features": '{"close": 0.25, "sma_20": 0.18, "rsi": 0.12}',
        "notable_events": "HIGH_WIN_RATE:60.0%|LARGE_WIN:156.8",
        "sharpe_ratio": "1.25 (Risk-adjusted return measure)"
    }
    
    for field, explanation in example_fields.items():
        tprint_info(f"  {field}: {explanation}")

async def simulate_daily_recordings():
    """Simulate creating daily records for multiple days."""
    try:
        # Simulate 5 days of trading data
        simulation_days = [
            {
                'date': date.today() - timedelta(days=4),
                'trades': 15,
                'performance': 'excellent',
                'notable_events': ['HIGH_WIN_RATE:73.3%', 'LARGE_WIN:225.4']
            },
            {
                'date': date.today() - timedelta(days=3),
                'trades': 8,
                'performance': 'poor',
                'notable_events': ['HIGH_VOLATILITY', 'LARGE_LOSS:85.6']
            },
            {
                'date': date.today() - timedelta(days=2),
                'trades': 0,
                'performance': 'no_trading',
                'notable_events': ['NO_TRADING']
            },
            {
                'date': date.today() - timedelta(days=1),
                'trades': 12,
                'performance': 'good',
                'notable_events': ['HIGH_MODEL_CONFIDENCE:89.5%']
            },
            {
                'date': date.today(),
                'trades': 18,
                'performance': 'excellent',
                'notable_events': ['HIGH_ACTIVITY:18', 'EXCELLENT_DAY']
            }
        ]
        
        for day_data in simulation_days:
            # Create mock trades for the day
            mock_trades = create_mock_daily_trades(day_data)
            mock_sessions = create_mock_daily_sessions(day_data)
            
            # Record daily summary
            success = await record_daily_trading_summary(
                trades=mock_trades,
                sessions=mock_sessions,
                target_date=day_data['date']
            )
            
            if success:
                tprint_success(f"✅ Recorded daily summary for {day_data['date']}")
            else:
                tprint_warning(f"⚠️ Failed to record for {day_data['date']}")
        
        tprint_success("✅ Simulated 5 days of daily recordings")
        
    except Exception as e:
        tprint_warning(f"❌ Daily recording simulation failed: {e}")

def create_mock_daily_trades(day_data: Dict[str, Any]) -> List:
    """Create mock trades for a simulation day."""
    from ..monitoring.comprehensive_trade_monitor import DetailedTradeMetrics
    
    trades = []
    trade_count = day_data['trades']
    target_date = day_data['date']
    performance = day_data['performance']
    
    if trade_count == 0:
        return trades
    
    # Generate mock trades based on performance
    for i in range(trade_count):
        # Create timestamp for the day
        timestamp = datetime.combine(target_date, datetime.min.time()) + timedelta(
            hours=np.random.uniform(9, 16),  # Trading hours
            minutes=np.random.uniform(0, 60)
        )
        
        # Generate PnL based on performance
        if performance == 'excellent':
            pnl = np.random.uniform(-20, 150) if np.random.random() < 0.8 else np.random.uniform(-50, -10)
        elif performance == 'good':
            pnl = np.random.uniform(-30, 80) if np.random.random() < 0.65 else np.random.uniform(-60, -15)
        elif performance == 'poor':
            pnl = np.random.uniform(-80, 40) if np.random.random() < 0.4 else np.random.uniform(-100, -20)
        else:
            pnl = np.random.uniform(-50, 50)
        
        # Create mock trade
        trade = DetailedTradeMetrics(
            trade_id=f"mock_trade_{target_date}_{i:03d}",
            timestamp=timestamp,
            symbol='ETHUSDT',
            action=np.random.choice(['buy', 'sell']),
            quantity=np.random.uniform(0.1, 1.0),
            price=3000 + np.random.uniform(-100, 100),
            pnl_absolute=pnl,
            pnl_percentage=pnl / 3000,
            signal_confidence=np.random.uniform(0.5, 0.9),
            signal_strength=np.random.uniform(0.6, 0.9),
            regime_type=np.random.choice(['trending_up', 'trending_down', 'sideways', 'high_volatility']),
            regime_confidence=np.random.uniform(0.6, 0.9),
            portfolio_risk=np.random.uniform(0.01, 0.05),
            leverage=np.random.uniform(1.0, 3.0),
            execution_quality=np.random.uniform(0.8, 0.98),
            slippage=np.random.uniform(0.0005, 0.002),
            commission=np.random.uniform(5, 25),
            execution_time_ms=np.random.uniform(100, 500),
            volatility_estimate=np.random.uniform(0.02, 0.06)
        )
        
        # Add mock model information
        trade.models_used = {
            'analyst_ensemble': {'model_type': 'analyst'},
            'tactician_timing': {'model_type': 'tactician'},
            'hmm_regime': {'model_type': 'hmm'}
        }
        
        trade.model_confidences = {
            'analyst_ensemble': np.random.uniform(0.6, 0.9),
            'tactician_timing': np.random.uniform(0.6, 0.9),
            'hmm_regime': np.random.uniform(0.7, 0.95)
        }
        
        trade.model_weights = {
            'analyst_ensemble': 0.6,
            'tactician_timing': 0.4,
            'hmm_regime': 1.0
        }
        
        trade.feature_importance = {
            'close': np.random.uniform(0.2, 0.3),
            'sma_20': np.random.uniform(0.15, 0.25),
            'rsi': np.random.uniform(0.1, 0.2),
            'volatility_20': np.random.uniform(0.08, 0.15),
            'volume': np.random.uniform(0.05, 0.12)
        }
        
        trades.append(trade)
    
    return trades

def create_mock_daily_sessions(day_data: Dict[str, Any]) -> List:
    """Create mock sessions for a simulation day."""
    from ..monitoring.comprehensive_trade_monitor import TradingSessionMetrics

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    
    if day_data['trades'] == 0:
        return []
    
    target_date = day_data['date']
    
    # Create 1-3 sessions for the day
    session_count = np.random.randint(1, 4)
    sessions = []
    
    for i in range(session_count):
        start_time = datetime.combine(target_date, datetime.min.time()) + timedelta(
            hours=9 + i * 3,  # Spread sessions throughout day
            minutes=np.random.uniform(0, 30)
        )
        
        end_time = start_time + timedelta(hours=np.random.uniform(1, 4))
        
        session = TradingSessionMetrics(
            session_id=f"session_{target_date}_{i}",
            start_time=start_time,
            end_time=end_time,
            total_trades=day_data['trades'] // session_count,
            total_pnl=np.random.uniform(-100, 300)
        )
        
        sessions.append(session)
    
    return sessions

async def analyze_historical_records():
    """Analyze historical daily records."""
    try:
        # Get historical data
        history_df = await get_trading_history(days=30)
        
        if history_df.empty:
            tprint_warning("⚠️ No historical data available")
            return
        
        tprint_info(f"📊 Analyzing {len(history_df)} days of trading history")
        
        # Calculate summary statistics
        summary_stats = {
            "Trading Days": len(history_df[history_df['total_trades'] > 0]),
            "Total Trades": int(history_df['total_trades'].sum()),
            "Average Daily PnL": f"${history_df['total_pnl'].mean():.2f}",
            "Best Day": f"${history_df['total_pnl'].max():.2f}",
            "Worst Day": f"${history_df['total_pnl'].min():.2f}",
            "Average Win Rate": f"{history_df['win_rate'].mean():.1%}",
            "Average Sharpe Ratio": f"{history_df['sharpe_ratio'].mean():.3f}",
            "Max Drawdown": f"{history_df['max_drawdown'].max():.1%}"
        }
        
        tprint_structured(summary_stats, LogLevel.INFO)
        
        # Show most common regimes
        regime_counts = {}
        for regimes in history_df['primary_regime'].dropna():
            regime_counts[regimes] = regime_counts.get(regimes, 0) + 1
        
        if regime_counts:
            tprint_info("🎯 Most Common Regimes:")
            for regime, count in sorted(regime_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                tprint_info(f"  {regime}: {count} days")
        
        # Show notable events summary
        all_events = []
        for events_str in history_df['notable_events'].dropna():
            if events_str:
                events = events_str.split('|')
                all_events.extend(events)
        
        if all_events:
            event_counts = {}
            for event in all_events:
                event_type = event.split(':')[0]
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            tprint_info("🚨 Notable Events Summary:")
            for event_type, count in sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                tprint_info(f"  {event_type}: {count} occurrences")
        
    except Exception as e:
        tprint_warning(f"❌ Historical analysis failed: {e}")

async def show_trend_analysis():
    """Show trend analysis from daily records."""
    try:
        # Get historical data
        history_df = await get_trading_history(days=30)
        
        if len(history_df) < 7:
            tprint_warning("⚠️ Insufficient data for trend analysis")
            return
        
        # Calculate trends
        history_df['date'] = pd.to_datetime(history_df['date'])
        history_df = history_df.sort_values('date')
        
        # Performance trends
        history_df['cumulative_pnl'] = history_df['total_pnl'].cumsum()
        history_df['rolling_win_rate_7d'] = history_df['win_rate'].rolling(7).mean()
        history_df['rolling_sharpe_7d'] = history_df['sharpe_ratio'].rolling(7).mean()
        
        # Model performance trends
        history_df['model_confidence_trend'] = history_df['avg_model_confidence'].rolling(7).mean()
        
        # Risk trends
        history_df['risk_trend'] = history_df['avg_portfolio_risk'].rolling(7).mean()
        
        # Show recent trends
        recent_data = history_df.tail(7)
        
        trend_analysis = {
            "Performance Trend": {
                "7-day PnL": f"${recent_data['total_pnl'].sum():.2f}",
                "PnL Trend": "📈 Improving" if recent_data['total_pnl'].iloc[-1] > recent_data['total_pnl'].iloc[0] else "📉 Declining",
                "Win Rate Trend": f"{recent_data['rolling_win_rate_7d'].iloc[-1]:.1%}",
                "Sharpe Trend": f"{recent_data['rolling_sharpe_7d'].iloc[-1]:.3f}"
            },
            "Model Performance Trend": {
                "Confidence Trend": f"{recent_data['model_confidence_trend'].iloc[-1]:.1%}",
                "Model Stability": "Stable" if recent_data['model_agreement_score'].std() < 0.1 else "Variable"
            },
            "Risk Trend": {
                "Risk Level": f"{recent_data['risk_trend'].iloc[-1]:.2%}",
                "Risk Trend": "📈 Increasing" if recent_data['avg_portfolio_risk'].iloc[-1] > recent_data['avg_portfolio_risk'].iloc[0] else "📉 Decreasing"
            }
        }
        
        tprint_structured(trend_analysis, LogLevel.INFO)
        
        # Show best and worst days
        best_day = history_df.loc[history_df['total_pnl'].idxmax()]
        worst_day = history_df.loc[history_df['total_pnl'].idxmin()]
        
        tprint_info("🏆 Best Trading Day:")
        tprint_info(f"  Date: {best_day['date'].strftime('%Y-%m-%d')}")
        tprint_info(f"  PnL: ${best_day['total_pnl']:.2f}")
        tprint_info(f"  Win Rate: {best_day['win_rate']:.1%}")
        tprint_info(f"  Trades: {int(best_day['total_trades'])}")
        tprint_info(f"  Events: {best_day['notable_events']}")
        
        tprint_info("📉 Worst Trading Day:")
        tprint_info(f"  Date: {worst_day['date'].strftime('%Y-%m-%d')}")
        tprint_info(f"  PnL: ${worst_day['total_pnl']:.2f}")
        tprint_info(f"  Win Rate: {worst_day['win_rate']:.1%}")
        tprint_info(f"  Trades: {int(worst_day['total_trades'])}")
        tprint_info(f"  Events: {worst_day['notable_events']}")
        
    except Exception as e:
        tprint_warning(f"❌ Trend analysis failed: {e}")

async def demonstrate_daily_summary_access():
    """Demonstrate accessing daily summaries."""
    try:
        tprint_info("🔍 Accessing Daily Summaries")
        
        # Get today's summary
        today_summary = await get_daily_trading_summary(date.today())
        
        if today_summary:
            tprint_success("✅ Found today's trading summary")
            
            key_metrics = {
                "Date": today_summary['date'],
                "Total Trades": today_summary['total_trades'],
                "PnL": f"${float(today_summary['total_pnl']):.2f}",
                "Win Rate": f"{float(today_summary['win_rate']):.1%}",
                "Primary Regime": today_summary['primary_regime'],
                "Notable Events": today_summary['notable_events']
            }
            
            tprint_structured(key_metrics, LogLevel.INFO)
        else:
            tprint_info("📝 No summary found for today (no trading activity)")
        
        # Show how to query specific metrics
        tprint_info("📊 Example Queries:")
        tprint_info("  # Get last 7 days performance")
        tprint_info("  history = await get_trading_history(days=7)")
        tprint_info("  weekly_pnl = history['total_pnl'].sum()")
        tprint_info("  ")
        tprint_info("  # Find best performing regime")
        tprint_info("  regime_performance = history.groupby('primary_regime')['total_pnl'].mean()")
        tprint_info("  ")
        tprint_info("  # Track model confidence trends")
        tprint_info("  confidence_trend = history['avg_model_confidence'].rolling(7).mean()")
        
    except Exception as e:
        tprint_warning(f"❌ Daily summary access demonstration failed: {e}")

# Main execution
if __name__ == "__main__":
    async def main():
        """Main demonstration."""
        try:
            await demonstrate_daily_recording()
            await demonstrate_daily_summary_access()
            
        except KeyboardInterrupt:
            tprint_info("⏹️ Demonstration stopped by user")
        except Exception as e:
            tprint_warning(f"❌ Demonstration failed: {e}")
    
    # Run the demonstration
    asyncio.run(main())

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
