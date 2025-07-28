# src/supervisor/performance_reporter.py
import pandas as pd
import os
import datetime
from src.config import CONFIG
from src.utils.logger import system_logger

class PerformanceReporter:
    def __init__(self, config=CONFIG, firestore_manager=None):
        self.config = config.get("supervisor", {})
        self.global_config = config
        self.firestore_manager = firestore_manager
        self.logger = system_logger.getChild('PerformanceReporter')

        self.daily_summary_log_filename = self.config.get("daily_summary_log_filename", "reports/daily_summary_log.csv")
        self.strategy_performance_log_filename = self.config.get("strategy_performance_log_filename", "reports/strategy_performance_log.csv")

        self._initialize_daily_summary_csv()
        self._initialize_strategy_performance_csv()

    def _initialize_daily_summary_csv(self):
        if not os.path.exists(self.daily_summary_log_filename):
            with open(self.daily_summary_log_filename, 'w') as f:
                f.write("Date,TotalTrades,WinRate,NetPnL,MaxDrawdown,EndingCapital,AllocatedCapitalMultiplier\n")

    def _initialize_strategy_performance_csv(self):
        if not os.path.exists(self.strategy_performance_log_filename):
            with open(self.strategy_performance_log_filename, 'w') as f:
                f.write("Date,Regime,TotalTrades,WinRate,NetPnL,AvgPnLPerTrade,TradeDuration\n")

    async def generate_performance_report(self, trade_logs: list, current_date: datetime.date, allocated_capital: float):
        self.logger.info(f"Generating Performance Report for {current_date}...")
        
        if not trade_logs:
            daily_summary = {
                "Date": current_date.strftime('%Y-%m-%d'), "TotalTrades": 0, "WinRate": 0.0, "NetPnL": 0.0,
                "MaxDrawdown": 0.0, "EndingCapital": allocated_capital,
                "AllocatedCapitalMultiplier": allocated_capital / self.global_config['INITIAL_EQUITY']
            }
            return {"daily_summary": daily_summary, "strategy_breakdown": {}}

        df_trades = pd.DataFrame(trade_logs)
        df_trades['realized_pnl_usd'] = pd.to_numeric(df_trades['realized_pnl_usd'], errors='coerce').fillna(0)

        total_trades = len(df_trades)
        win_rate = (len(df_trades[df_trades['realized_pnl_usd'] > 0]) / total_trades * 100) if total_trades > 0 else 0.0
        net_pnl = df_trades['realized_pnl_usd'].sum()

        equity_curve = pd.Series([allocated_capital] + (allocated_capital + df_trades['realized_pnl_usd'].cumsum()).tolist())
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = -drawdown.min() * 100

        daily_summary = {
            "Date": current_date.strftime('%Y-%m-%d'), "TotalTrades": total_trades, "WinRate": round(win_rate, 2),
            "NetPnL": round(net_pnl, 2), "MaxDrawdown": round(max_drawdown, 2),
            "EndingCapital": round(allocated_capital + net_pnl, 2),
            "AllocatedCapitalMultiplier": round((allocated_capital + net_pnl) / self.global_config['INITIAL_EQUITY'], 2)
        }

        strategy_breakdown = {}
        if 'market_state_at_entry' in df_trades.columns:
            for regime, regime_trades in df_trades.groupby('market_state_at_entry'):
                # ... logic to calculate per-regime metrics ...
                strategy_breakdown[regime] = {
                    "TotalTrades": len(regime_trades),
                    # ... other metrics
                }
        
        return {"daily_summary": daily_summary, "strategy_breakdown": strategy_breakdown}

    async def update_daily_summary_csv_and_firestore(self, daily_summary: dict):
        # ... (implementation remains the same) ...
        pass

    async def update_strategy_performance_log_and_firestore(self, current_date: datetime.date, strategy_breakdown: dict):
        # ... (implementation remains the same) ...
        pass
