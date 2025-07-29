# src/supervisor/performance_reporter.py
import pandas as pd
import numpy as np
import os
import datetime
import json
import uuid # Import uuid for unique trade IDs
from typing import Any, Dict, Union # Added imports for type hinting

from src.config import CONFIG
from src.utils.logger import system_logger
# Import both managers for type hinting, but use the one passed in __init__
from src.database.firestore_manager import FirestoreManager
from src.database.sqlite_manager import SQLiteManager


class PerformanceReporter:
    """
    Handles all performance logging and reporting for the Supervisor.
    It calculates daily and per-regime metrics, and saves them to CSV and Firestore.
    Now also records detailed individual trade logs.
    """
    def __init__(self, config=CONFIG, db_manager: Union[FirestoreManager, SQLiteManager, None] = None): # Fixed: Accept generic db_manager
        self.config = config.get("supervisor", {})
        self.global_config = config
        self.db_manager = db_manager # Use the passed db_manager
        self.logger = system_logger.getChild('PerformanceReporter')

        # Use new monthly filename formats
        self.daily_summary_log_filename_format = CONFIG.get("DAILY_SUMMARY_LOG_FILENAME_FORMAT", "reports/daily_summary_log_%Y-%m.csv")
        self.strategy_performance_log_filename_format = CONFIG.get("STRATEGY_PERFORMANCE_LOG_FILENAME_FORMAT", "reports/strategy_performance_log_%Y-%m.csv")
        self.detailed_trade_log_filename = CONFIG.get("DETAILED_TRADE_LOG_FILE", "reports/detailed_trade_log.csv") # This remains a single file

        # Ensure the reports directory exists
        os.makedirs(os.path.dirname(self.detailed_trade_log_filename), exist_ok=True)
        os.makedirs(os.path.dirname(self.daily_summary_log_filename_format.replace('%Y-%m', '2000-01')), exist_ok=True) # Create example dir for format
        
        self._initialize_detailed_trade_log_csv() # Initialize the new log file

    def _get_current_monthly_filename(self, base_format: str) -> str:
        """Generates a monthly filename based on the current date."""
        return datetime.datetime.now().strftime(base_format)

    def _initialize_daily_summary_csv(self):
        """Ensures the daily summary CSV file exists with correct headers."""
        filename = self._get_current_monthly_filename(self.daily_summary_log_filename_format)
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write("Date,TotalTrades,WinRate,NetPnL,MaxDrawdown,EndingCapital,AllocatedCapitalMultiplier\n")
            self.logger.info(f"Created daily summary log: {filename}")

    def _initialize_strategy_performance_csv(self):
        """Ensures the strategy performance CSV file exists with correct headers."""
        filename = self._get_current_monthly_filename(self.strategy_performance_log_filename_format)
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write("Date,Regime,TotalTrades,WinRate,NetPnL,AvgPnLPerTrade,TradeDuration\n")
            self.logger.info(f"Created strategy performance log: {filename}")

    def _initialize_detailed_trade_log_csv(self):
        """Ensures the detailed trade log CSV file exists with correct headers."""
        if not os.path.exists(self.detailed_trade_log_filename):
            headers = [
                "TradeID", "Token", "Exchange", "Side", "EntryTimestampUTC", "ExitTimestampUTC",
                "TradeDurationSeconds", "NetPnLUSD", "PnLPercentage", "ExitReason",
                "EntryPrice", "ExitPrice", "QuantityBaseAsset", "NotionalSizeUSD", "LeverageUsed",
                "IntendedStopLossPrice", "IntendedTakeProfitPrice", "ActualStopLossPrice", "ActualTakeProfitPrice",
                "OrderTypeEntry", "OrderTypeExit", "EntryFeesUSD", "ExitFeesUSD", "SlippageEntryPct", "SlippageExitPct",
                "MarketRegimeAtEntry", "TacticianSignal", "EnsemblePredictionAtEntry", "EnsembleConfidenceAtEntry",
                "DirectionalConfidenceAtEntry", "MarketHealthScoreAtEntry", "LiquidationSafetyScoreAtEntry",
                "TrendStrengthAtEntry", "ADXValueAtEntry", "RSIValueAtEntry", "MACDHistogramValueAtEntry",
                "PriceVsVWAPRatioAtEntry", "VolumeDeltaAtEntry", "GlobalRiskMultiplierAtEntry",
                "AvailableAccountEquityAtEntry", "TradingEnvironment", "IsTradingPausedAtEntry",
                "KillSwitchActiveAtEntry", "ModelVersionID",
                "BaseModelPredictionsAtEntry", # New header
                "EnsembleWeightsAtEntry" # New header
            ]
            with open(self.detailed_trade_log_filename, 'w') as f:
                f.write(",".join(headers) + "\n")
            self.logger.info(f"Created detailed trade log: {self.detailed_trade_log_filename}")

    async def generate_performance_report(self, trade_logs: list, current_date: datetime.date, allocated_capital: float):
        """
        Generates a detailed performance report for a given period.
        This method now expects 'trade_logs' to be the list of detailed trade dictionaries.
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

        # Use 'NetPnLUSD' for calculations now
        df_trades = pd.DataFrame(trade_logs)
        df_trades['NetPnLUSD'] = pd.to_numeric(df_trades['NetPnLUSD'], errors='coerce').fillna(0)

        # --- Daily Summary Metrics ---
        total_trades = len(df_trades)
        wins = df_trades[df_trades['NetPnLUSD'] > 0]
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
        net_pnl = df_trades['NetPnLUSD'].sum()

        # Calculate Max Drawdown for the period
        # This needs to be based on the equity curve from the start of the period
        # For simplicity, if actual equity curve is not passed, use cumulative PnL
        equity_curve = pd.Series([initial_equity_for_period] + (initial_equity_for_period + df_trades['NetPnLUSD'].cumsum()).tolist())
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
        # Use 'MarketRegimeAtEntry' for grouping
        if 'MarketRegimeAtEntry' in df_trades.columns:
            for regime, regime_trades in df_trades.groupby('MarketRegimeAtEntry'):
                regime_total_trades = len(regime_trades)
                regime_wins = regime_trades[regime_trades['NetPnLUSD'] > 0]
                regime_win_rate = (len(regime_wins) / regime_total_trades * 100) if regime_total_trades > 0 else 0.0
                regime_net_pnl = regime_trades['NetPnLUSD'].sum()
                
                strategy_breakdown[regime] = {
                    "TotalTrades": regime_total_trades,
                    "WinRate": round(regime_win_rate, 2),
                    "NetPnL": round(regime_net_pnl, 2),
                    "AvgPnLPerTrade": round(regime_net_pnl / regime_total_trades, 2) if regime_total_trades > 0 else 0.0,
                    "TradeDuration": round(regime_trades['TradeDurationSeconds'].mean(), 2) if regime_total_trades > 0 else 0.0
                }
        
        self.logger.info("Performance Report Generated.")
        return {"daily_summary": daily_summary, "strategy_breakdown": strategy_breakdown}

    async def update_daily_summary_csv_and_firestore(self, daily_summary: dict):
        """Appends the daily summary to the CSV log and saves to DB."""
        try:
            # CSV
            filename = self._get_current_monthly_filename(self.daily_summary_log_filename_format)
            row = [
                daily_summary["Date"], daily_summary["TotalTrades"], daily_summary["WinRate"],
                daily_summary["NetPnL"], daily_summary["MaxDrawdown"], daily_summary["EndingCapital"],
                daily_summary["AllocatedCapitalMultiplier"]
            ]
            with open(filename, 'a') as f:
                f.write(",".join(map(str, row)) + "\n")
            self.logger.info(f"Appended daily summary for {daily_summary['Date']} to CSV: {filename}.")

            # DB
            if self.db_manager: # Fixed: Use db_manager
                collection_name = "daily_summaries" # Use a fixed collection name
                await self.db_manager.set_document( # Fixed: Use set_document for upsert
                    collection_name, doc_id=daily_summary["Date"], data=daily_summary, is_public=False
                )
                self.logger.info(f"Saved daily summary for {daily_summary['Date']} to DB.")
        except Exception as e:
            self.logger.error(f"Error updating daily summary (CSV/DB): {e}", exc_info=True)

    async def update_strategy_performance_log_and_firestore(self, current_date: datetime.date, strategy_breakdown: dict):
        """Appends strategy performance breakdown to its CSV log and saves to DB."""
        try:
            # CSV
            filename = self._get_current_monthly_filename(self.strategy_performance_log_filename_format)
            with open(filename, 'a') as f:
                for regime, metrics in strategy_breakdown.items():
                    row = [
                        current_date.strftime('%Y-%m-%d'), regime, metrics["TotalTrades"], metrics["WinRate"],
                        metrics["NetPnL"], metrics["AvgPnLPerTrade"], metrics["TradeDuration"]
                    ]
                    f.write(",".join(map(str, row)) + "\n")
            self.logger.info(f"Appended strategy performance for {current_date} to CSV: {filename}.")

            # DB
            if self.db_manager: # Fixed: Use db_manager
                collection_name = "strategy_performance_logs" # Use a fixed collection name
                for regime, metrics in strategy_breakdown.items():
                    doc_id = f"{current_date.isoformat()}_{regime}"
                    doc_data = {"date": current_date.isoformat(), "regime": regime, **metrics}
                    # Use add_document as it's a log, not a single document that gets updated by ID
                    await self.db_manager.add_document( 
                        collection_name, data=doc_data, is_public=False
                    )
                self.logger.info(f"Saved strategy performance for {current_date} to DB.")
        except Exception as e:
            self.logger.error(f"Error updating strategy performance log (CSV/DB): {e}", exc_info=True)

    async def record_detailed_trade_log(self, trade_data: Dict[str, Any]):
        """
        Records a single detailed trade log entry to the CSV file and DB.
        """
        self.logger.info(f"Recording detailed trade log for Trade ID: {trade_data.get('TradeID')}")
        
        # Define the order of headers for the CSV
        headers = [
            "TradeID", "Token", "Exchange", "Side", "EntryTimestampUTC", "ExitTimestampUTC",
            "TradeDurationSeconds", "NetPnLUSD", "PnLPercentage", "ExitReason",
            "EntryPrice", "ExitPrice", "QuantityBaseAsset", "NotionalSizeUSD", "LeverageUsed",
            "IntendedStopLossPrice", "IntendedTakeProfitPrice", "ActualStopLossPrice", "ActualTakeProfitPrice",
            "OrderTypeEntry", "OrderTypeExit", "EntryFeesUSD", "ExitFeesUSD", "SlippageEntryPct", "SlippageExitPct",
            "MarketRegimeAtEntry", "TacticianSignal", "EnsemblePredictionAtEntry", "EnsembleConfidenceAtEntry",
            "DirectionalConfidenceAtEntry", "MarketHealthScoreAtEntry", "LiquidationSafetyScoreAtEntry",
            "TrendStrengthAtEntry", "ADXValueAtEntry", "RSIValueAtEntry", "MACDHistogramValueAtEntry",
            "PriceVsVWAPRatioAtEntry", "VolumeDeltaAtEntry", "GlobalRiskMultiplierAtEntry",
            "AvailableAccountEquityAtEntry", "TradingEnvironment", "IsTradingPausedAtEntry",
            "KillSwitchActiveAtEntry", "ModelVersionID",
            "BaseModelPredictionsAtEntry", # New header
            "EnsembleWeightsAtEntry" # New header
        ]

        try:
            # Prepare the row data in the correct order, handling missing keys
            row_values = []
            for header in headers:
                value = trade_data.get(header, "") # Get value or empty string if not present
                # Handle nested dictionary values (like base_model_predictions, ensemble_weights)
                if isinstance(value, dict):
                    row_values.append(json.dumps(value)) # Convert dicts to JSON string
                else:
                    row_values.append(str(value))
            
            with open(self.detailed_trade_log_filename, 'a') as f:
                f.write(",".join(row_values) + "\n")
            self.logger.info(f"Detailed trade log for {trade_data.get('TradeID')} appended to CSV.")

            # DB
            if self.db_manager: # Fixed: Use db_manager
                collection_name = "detailed_trade_logs" # Or a more specific name
                # Use trade_data.get('TradeID') as doc_id for unique identification
                # Use set_document for detailed_trade_logs as TradeID is a PRIMARY KEY
                await self.db_manager.set_document(
                    collection_name, doc_id=trade_data.get('TradeID'), data=trade_data, is_public=False
                )
                self.logger.info(f"Detailed trade log for {trade_data.get('TradeID')} saved to DB.")

        except Exception as e:
            self.logger.error(f"Error recording detailed trade log: {e}", exc_info=True)


