# src/supervisor/performance_reporter.py
import pandas as pd
import numpy as np
import os
import datetime
import json
from src.config import CONFIG
from src.utils.logger import system_logger

class PerformanceReporter:
    """
    Handles all performance logging and reporting for the Supervisor.
    It calculates daily and per-regime metrics, and saves them to CSV and Firestore.
    """
    def __init__(self, config=CONFIG, firestore_manager=None):
        self.config = config.get("supervisor", {})
        self.global_config = config
        self.firestore_manager = firestore_manager
        self.logger = system_logger.getChild('PerformanceReporter')

        self.daily_summary_log_filename = self.config.get("daily_summary_log_filename", "reports/daily_summary_log.csv")
        self.strategy_performance_log_filename = self.config.get("strategy_performance_log_filename", "reports/strategy_performance_log.csv")

        # Ensure the reports directory exists
        os.makedirs(os.path.dirname(self.daily_summary_log_filename), exist_ok=True)

        self._initialize_daily_summary_csv()
        self._initialize_strategy_performance_csv()

    def _initialize_daily_summary_csv(self):
        """Ensures the daily summary CSV file exists with correct headers."""
        if not os.path.exists(self.daily_summary_log_filename):
            with open(self.daily_summary_log_filename, 'w') as f:
                f.write("Date,TotalTrades,WinRate,NetPnL,MaxDrawdown,EndingCapital,AllocatedCapitalMultiplier\n")
            self.logger.info(f"Created daily summary log: {self.daily_summary_log_filename}")

    def _initialize_strategy_performance_csv(self):
        """Ensures the strategy performance CSV file exists with correct headers."""
        if not os.path.exists(self.strategy_performance_log_filename):
            with open(self.strategy_performance_log_filename, 'w') as f:
                f.write("Date,Regime,TotalTrades,WinRate,NetPnL,AvgPnLPerTrade,TradeDuration\n")
            self.logger.info(f"Created strategy performance log: {self.strategy_performance_log_filename}")

    async def generate_performance_report(self, trade_logs: list, current_date: datetime.date, allocated_capital: float):
        """
        Generates a detailed performance report for a given period.
        """
        self.logger.info(f"Generating Performance Report for {current_date}...")
        
        initial_equity_for_period = allocated_capital
        
        if not trade_logs:
            self.logger.info("No trades recorded for this period. Generating empty report.")
            daily_summary = {
                "Date": current_date.strftime('%Y-%m-%d'),
                "TotalTrades": 0, "WinRate": 0.0, "NetPnL": 0.0,
                "MaxDrawdown": 0.0, "EndingCapital": initial_equity_for_period,
                "AllocatedCapitalMultiplier": initial_equity_for_period / self.global_config['INITIAL_EQUITY'] if self.global_config['INITIAL_EQUITY'] > 0 else 0
            }
            return {"daily_summary": daily_summary, "strategy_breakdown": {}}

        df_trades = pd.DataFrame(trade_logs)
        df_trades['realized_pnl_usd'] = pd.to_numeric(df_trades['realized_pnl_usd'], errors='coerce').fillna(0)

        # --- Daily Summary Metrics ---
        total_trades = len(df_trades)
        wins = df_trades[df_trades['realized_pnl_usd'] > 0]
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
        net_pnl = df_trades['realized_pnl_usd'].sum()

        # Calculate Max Drawdown for the period
        equity_curve = pd.Series([initial_equity_for_period] + (initial_equity_for_period + df_trades['realized_pnl_usd'].cumsum()).tolist())
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = -drawdown.min() * 100 if not drawdown.empty else 0.0

        ending_capital = initial_equity_for_period + net_pnl
        allocated_capital_multiplier = ending_capital / self.global_config['INITIAL_EQUITY'] if self.global_config['INITIAL_EQUITY'] > 0 else 0

        daily_summary = {
            "Date": current_date.strftime('%Y-%m-%d'),
            "TotalTrades": total_trades,
            "WinRate": round(win_rate, 2),
            "NetPnL": round(net_pnl, 2),
            "MaxDrawdown": round(max_drawdown, 2),
            "EndingCapital": round(ending_capital, 2),
            "AllocatedCapitalMultiplier": round(allocated_capital_multiplier, 2)
        }

        # --- Strategy Performance Breakdown ---
        strategy_breakdown = {}
        if 'market_state_at_entry' in df_trades.columns:
            for regime, regime_trades in df_trades.groupby('market_state_at_entry'):
                regime_total_trades = len(regime_trades)
                regime_wins = regime_trades[regime_trades['realized_pnl_usd'] > 0]
                regime_win_rate = (len(regime_wins) / regime_total_trades * 100) if regime_total_trades > 0 else 0.0
                regime_net_pnl = regime_trades['realized_pnl_usd'].sum()
                
                strategy_breakdown[regime] = {
                    "TotalTrades": regime_total_trades,
                    "WinRate": round(regime_win_rate, 2),
                    "NetPnL": round(regime_net_pnl, 2),
                    "AvgPnLPerTrade": round(regime_net_pnl / regime_total_trades, 2) if regime_total_trades > 0 else 0.0,
                    "TradeDuration": 0.0  # Placeholder - would require entry/exit timestamps in logs
                }
        
        self.logger.info("Performance Report Generated.")
        return {"daily_summary": daily_summary, "strategy_breakdown": strategy_breakdown}

    async def update_daily_summary_csv_and_firestore(self, daily_summary: dict):
        """Appends the daily summary to the CSV log and saves to Firestore."""
        try:
            # CSV
            row = [
                daily_summary["Date"], daily_summary["TotalTrades"], daily_summary["WinRate"],
                daily_summary["NetPnL"], daily_summary["MaxDrawdown"], daily_summary["EndingCapital"],
                daily_summary["AllocatedCapitalMultiplier"]
            ]
            with open(self.daily_summary_log_filename, 'a') as f:
                f.write(",".join(map(str, row)) + "\n")
            self.logger.info(f"Appended daily summary for {daily_summary['Date']} to CSV.")

            # Firestore
            if self.firestore_manager and self.firestore_manager.firestore_enabled:
                collection_name = self.daily_summary_log_filename.split('/')[-1].replace('.csv', '')
                await self.firestore_manager.set_document(
                    collection_name, doc_id=daily_summary["Date"], data=daily_summary, is_public=False
                )
                self.logger.info(f"Saved daily summary for {daily_summary['Date']} to Firestore.")
        except Exception as e:
            self.logger.error(f"Error updating daily summary (CSV/Firestore): {e}", exc_info=True)

    async def update_strategy_performance_log_and_firestore(self, current_date: datetime.date, strategy_breakdown: dict):
        """Appends strategy performance breakdown to its CSV log and saves to Firestore."""
        try:
            # CSV
            with open(self.strategy_performance_log_filename, 'a') as f:
                for regime, metrics in strategy_breakdown.items():
                    row = [
                        current_date.strftime('%Y-%m-%d'), regime, metrics["TotalTrades"], metrics["WinRate"],
                        metrics["NetPnL"], metrics["AvgPnLPerTrade"], metrics["TradeDuration"]
                    ]
                    f.write(",".join(map(str, row)) + "\n")
            self.logger.info(f"Appended strategy performance for {current_date} to CSV.")

            # Firestore
            if self.firestore_manager and self.firestore_manager.firestore_enabled:
                collection_name = self.strategy_performance_log_filename.split('/')[-1].replace('.csv', '')
                for regime, metrics in strategy_breakdown.items():
                    doc_id = f"{current_date.isoformat()}_{regime}"
                    doc_data = {"date": current_date.isoformat(), "regime": regime, **metrics}
                    await self.firestore_manager.set_document(
                        collection_name, doc_id=doc_id, data=doc_data, is_public=False
                    )
                self.logger.info(f"Saved strategy performance for {current_date} to Firestore.")
        except Exception as e:
            self.logger.error(f"Error updating strategy performance log (CSV/Firestore): {e}", exc_info=True)
