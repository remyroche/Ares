import pandas as pd
import os
import datetime
import json
from typing import Any, Dict, Union, Optional, List

from src.config import CONFIG
from src.utils.logger import system_logger

# Import SQLite and InfluxDB managers directly, make Firebase optional
from src.database.sqlite_manager import SQLiteManager
try:
    from src.database.influxdb_manager import InfluxDBManager
except ImportError:
    InfluxDBManager = None

# Make Firebase optional - only import if available
try:
    from src.database.firestore_manager import FirestoreManager
    FIREBASE_AVAILABLE = True
except ImportError:
    FirestoreManager = None
    FIREBASE_AVAILABLE = False

from emails.ares_mailer import AresMailer


class PerformanceReporter:
    """
    Handles all performance logging and reporting for the Supervisor.
    It calculates and logs daily metrics, records detailed trades to monthly files,
    and sends a daily summary email report.
    """

    def __init__(
        self,
        config=CONFIG,
        db_manager: Union[SQLiteManager, InfluxDBManager, None] = None,
        mailer: Optional[AresMailer] = None,
    ):
        self.config = config.get("supervisor", {})
        self.global_config = config
        self.db_manager = db_manager
        self.mailer = mailer
        self.logger = system_logger.getChild("PerformanceReporter")

        # Use new monthly filename formats for all logs
        self.daily_summary_log_filename_format = CONFIG.get(
            "DAILY_SUMMARY_LOG_FILENAME_FORMAT", "reports/daily_summary_log_%Y-%m.csv"
        )
        self.strategy_performance_log_filename_format = CONFIG.get(
            "STRATEGY_PERFORMANCE_LOG_FILENAME_FORMAT",
            "reports/strategy_performance_log_%Y-%m.csv",
        )
        self.detailed_trade_log_filename_format = CONFIG.get(
            "DETAILED_TRADE_LOG_FILENAME_FORMAT", "reports/detailed_trade_log_%Y-%m.csv"
        )

        # Ensure the reports directory exists
        os.makedirs("reports", exist_ok=True)

        # Initialize CSV files for the current month
        self._initialize_daily_summary_csv()
        self._initialize_strategy_performance_csv()
        self._initialize_detailed_trade_log_csv()

    def _get_current_monthly_filename(self, base_format: str) -> str:
        """Generates a monthly filename based on the current date."""
        return datetime.datetime.now().strftime(base_format)

    def _initialize_daily_summary_csv(self):
        """Ensures the daily summary CSV file for the current month exists with correct headers."""
        filename = self._get_current_monthly_filename(
            self.daily_summary_log_filename_format
        )
        if not os.path.exists(filename):
            with open(filename, "w", newline="") as f:
                f.write(
                    "Date,TotalTrades,WinRate,NetPnL,MaxDrawdown,EndingCapital,AllocatedCapitalMultiplier\n"
                )
            self.logger.info(f"Created daily summary log: {filename}")

    def _initialize_strategy_performance_csv(self):
        """Ensures the strategy performance CSV file for the current month exists with correct headers."""
        filename = self._get_current_monthly_filename(
            self.strategy_performance_log_filename_format
        )
        if not os.path.exists(filename):
            with open(filename, "w", newline="") as f:
                f.write(
                    "Date,Regime,TotalTrades,WinRate,NetPnL,AvgPnLPerTrade,TradeDuration\n"
                )
            self.logger.info(f"Created strategy performance log: {filename}")

    def _initialize_detailed_trade_log_csv(self):
        """Ensures the detailed trade log CSV file for the current month exists with correct headers."""
        filename = self._get_current_monthly_filename(
            self.detailed_trade_log_filename_format
        )
        if not os.path.exists(filename):
            headers = [
                "TradeID",
                "Token",
                "Exchange",
                "Side",
                "EntryTimestampUTC",
                "ExitTimestampUTC",
                "TradeDurationSeconds",
                "NetPnLUSD",
                "PnLPercentage",
                "ExitReason",
                "EntryPrice",
                "ExitPrice",
                "QuantityBaseAsset",
                "NotionalSizeUSD",
                "LeverageUsed",
                "IntendedStopLossPrice",
                "IntendedTakeProfitPrice",
                "ActualStopLossPrice",
                "ActualTakeProfitPrice",
                "OrderTypeEntry",
                "OrderTypeExit",
                "EntryFeesUSD",
                "ExitFeesUSD",
                "SlippageEntryPct",
                "SlippageExitPct",
                "MarketRegimeAtEntry",
                "TacticianSignal",
                "EnsemblePredictionAtEntry",
                "EnsembleConfidenceAtEntry",
                "DirectionalConfidenceAtEntry",
                "MarketHealthScoreAtEntry",
                "LiquidationSafetyScoreAtEntry",
                "TrendStrengthAtEntry",
                "ADXValueAtEntry",
                "RSIValueAtEntry",
                "MACDHistogramValueAtEntry",
                "PriceVsVWAPRatioAtEntry",
                "VolumeDeltaAtEntry",
                "GlobalRiskMultiplierAtEntry",
                "AvailableAccountEquityAtEntry",
                "TradingEnvironment",
                "IsTradingPausedAtEntry",
                "KillSwitchActiveAtEntry",
                "ModelVersionID",
                "BaseModelPredictionsAtEntry",
                "EnsembleWeightsAtEntry",
                "VolatilityAtEntryATR",
            ]
            with open(filename, "w", newline="") as f:
                f.write(",".join(headers) + "\n")
            self.logger.info(f"Created detailed trade log: {filename}")

    async def generate_performance_report(
        self, trade_logs: list, current_date: datetime.date, allocated_capital: float
    ):
        """
        Generates a detailed performance report for a given period.
        """
        self.logger.info(f"Generating Performance Report for {current_date}...")

        initial_equity_for_period = allocated_capital

        if not trade_logs:
            self.logger.info(
                "No trades recorded for this period. Generating empty report."
            )
            daily_summary = {
                "Date": current_date.strftime("%Y-%m-%d"),
                "TotalTrades": 0,
                "WinRate": 0.0,
                "NetPnL": 0.0,
                "MaxDrawdown": 0.0,
                "EndingCapital": initial_equity_for_period,
                "AllocatedCapitalMultiplier": initial_equity_for_period
                / self.global_config.get("INITIAL_EQUITY", 1)
                if self.global_config.get("INITIAL_EQUITY")
                else 0,
            }
            return {"daily_summary": daily_summary, "strategy_breakdown": {}}

        df_trades = pd.DataFrame(trade_logs)
        df_trades["NetPnLUSD"] = pd.to_numeric(
            df_trades["NetPnLUSD"], errors="coerce"
        ).fillna(0)

        total_trades = len(df_trades)
        wins = df_trades[df_trades["NetPnLUSD"] > 0]
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
        net_pnl = df_trades["NetPnLUSD"].sum()

        equity_curve = pd.Series(
            [initial_equity_for_period]
            + (initial_equity_for_period + df_trades["NetPnLUSD"].cumsum()).tolist()
        )
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = -drawdown.min() * 100 if not drawdown.empty else 0.0

        ending_capital = initial_equity_for_period + net_pnl
        allocated_capital_multiplier = (
            ending_capital / self.global_config.get("INITIAL_EQUITY", 1)
            if self.global_config.get("INITIAL_EQUITY")
            else 0
        )

        daily_summary = {
            "Date": current_date.strftime("%Y-%m-%d"),
            "TotalTrades": total_trades,
            "WinRate": round(win_rate, 2),
            "NetPnL": round(net_pnl, 2),
            "MaxDrawdown": round(max_drawdown, 2),
            "EndingCapital": round(ending_capital, 2),
            "AllocatedCapitalMultiplier": round(allocated_capital_multiplier, 2),
        }

        strategy_breakdown = {}
        if "MarketRegimeAtEntry" in df_trades.columns:
            for regime, regime_trades in df_trades.groupby("MarketRegimeAtEntry"):
                regime_total_trades = len(regime_trades)
                regime_wins = regime_trades[regime_trades["NetPnLUSD"] > 0]
                regime_win_rate = (
                    (len(regime_wins) / regime_total_trades * 100)
                    if regime_total_trades > 0
                    else 0.0
                )
                regime_net_pnl = regime_trades["NetPnLUSD"].sum()

                strategy_breakdown[str(regime)] = {
                    "TotalTrades": regime_total_trades,
                    "WinRate": round(regime_win_rate, 2),
                    "NetPnL": round(regime_net_pnl, 2),
                    "AvgPnLPerTrade": round(regime_net_pnl / regime_total_trades, 2)
                    if regime_total_trades > 0
                    else 0.0,
                    "TradeDuration": round(
                        pd.to_numeric(
                            regime_trades["TradeDurationSeconds"], errors="coerce"
                        ).mean(),
                        2,
                    )
                    if regime_total_trades > 0
                    else 0.0,
                }

        self.logger.info("Performance Report Generated.")
        return {
            "daily_summary": daily_summary,
            "strategy_breakdown": strategy_breakdown,
        }

    async def record_detailed_trade_log(self, trade_data: Dict[str, Any]):
        """
        Records a single detailed trade log entry to the monthly CSV file and DB.
        """
        self.logger.info(
            f"Recording detailed trade log for Trade ID: {trade_data.get('TradeID')}"
        )

        filename = self._get_current_monthly_filename(
            self.detailed_trade_log_filename_format
        )
        self._initialize_detailed_trade_log_csv()  # Ensures file/headers exist for the current month

        headers = [
            "TradeID",
            "Token",
            "Exchange",
            "Side",
            "EntryTimestampUTC",
            "ExitTimestampUTC",
            "TradeDurationSeconds",
            "NetPnLUSD",
            "PnLPercentage",
            "ExitReason",
            "EntryPrice",
            "ExitPrice",
            "QuantityBaseAsset",
            "NotionalSizeUSD",
            "LeverageUsed",
            "IntendedStopLossPrice",
            "IntendedTakeProfitPrice",
            "ActualStopLossPrice",
            "ActualTakeProfitPrice",
            "OrderTypeEntry",
            "OrderTypeExit",
            "EntryFeesUSD",
            "ExitFeesUSD",
            "SlippageEntryPct",
            "SlippageExitPct",
            "MarketRegimeAtEntry",
            "TacticianSignal",
            "EnsemblePredictionAtEntry",
            "EnsembleConfidenceAtEntry",
            "DirectionalConfidenceAtEntry",
            "MarketHealthScoreAtEntry",
            "LiquidationSafetyScoreAtEntry",
            "TrendStrengthAtEntry",
            "ADXValueAtEntry",
            "RSIValueAtEntry",
            "MACDHistogramValueAtEntry",
            "PriceVsVWAPRatioAtEntry",
            "VolumeDeltaAtEntry",
            "GlobalRiskMultiplierAtEntry",
            "AvailableAccountEquityAtEntry",
            "TradingEnvironment",
            "IsTradingPausedAtEntry",
            "KillSwitchActiveAtEntry",
            "ModelVersionID",
            "BaseModelPredictionsAtEntry",
            "EnsembleWeightsAtEntry",
            "VolatilityAtEntryATR",
        ]

        try:
            # CSV Logging
            row_values = []
            for header in headers:
                value = trade_data.get(header, "")
                if isinstance(value, dict):
                    row_values.append(json.dumps(value))
                else:
                    row_values.append(str(value))

            with open(filename, "a", newline="") as f:
                f.write(",".join(row_values) + "\n")
            self.logger.info(
                f"Detailed trade log for {trade_data.get('TradeID')} appended to {filename}."
            )

            # Database Logging
            if self.db_manager:
                collection_name = "detailed_trade_logs"
                await self.db_manager.set_document(
                    collection_name,
                    doc_id=trade_data.get("TradeID"),
                    data=trade_data,
                    is_public=False,
                )
                self.logger.info(
                    f"Detailed trade log for {trade_data.get('TradeID')} saved to DB."
                )

        except Exception as e:
            self.logger.error(f"Error recording detailed trade log: {e}", exc_info=True)

    async def generate_daily_trade_summary_report(
        self, trade_logs: List[Dict[str, Any]], current_date: datetime.date
    ):
        """
        Generates and sends a daily summary email report of all trades for the day.
        """
        if not self.mailer:
            self.logger.info("Mailer not configured, skipping daily email report.")
            return

        self.logger.info(
            f"Generating daily trade summary email for {current_date.strftime('%Y-%m-%d')}"
        )

        try:
            if not trade_logs:
                subject = f"Ares Daily Report for {current_date.strftime('%Y-%m-%d')}: No Trades"
                body = "There were no new trades executed today."
                await self.mailer.send_email(subject=subject, body=body)
                self.logger.info("Sent daily report email (no trades).")
                return

            df_trades = pd.DataFrame(trade_logs)
            df_trades["NetPnLUSD"] = pd.to_numeric(
                df_trades["NetPnLUSD"], errors="coerce"
            ).fillna(0)

            total_trades = len(df_trades)
            net_pnl = df_trades["NetPnLUSD"].sum()
            wins = len(df_trades[df_trades["NetPnLUSD"] > 0])
            losses = total_trades - wins
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

            subject = f"Ares Daily Report for {current_date.strftime('%Y-%m-%d')}: ${net_pnl:,.2f} PnL"

            # Build HTML email body
            body = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: sans-serif; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #dddddd; text-align: left; padding: 8px; }}
                    th {{ background-color: #f2f2f2; }}
                    .summary-card {{ padding: 15px; background-color: #f9f9f9; border-left: 5px solid #4CAF50; margin-bottom: 20px; }}
                </style>
            </head>
            <body>
                <h2>Ares Daily Trading Summary: {current_date.strftime("%B %d, %Y")}</h2>
                <div class="summary-card">
                    <p><strong>Total Net PnL:</strong> ${net_pnl:,.2f}</p>
                    <p><strong>Total Trades:</strong> {total_trades}</p>
                    <p><strong>Win/Loss:</strong> {wins} / {losses}</p>
                    <p><strong>Win Rate:</strong> {win_rate:.2f}%</p>
                </div>
                <h3>Today's Trades:</h3>
                <table>
                    <tr>
                        <th>Trade ID</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Net PnL (USD)</th>
                        <th>Entry Vol (ATR)</th>
                        <th>Exit Reason</th>
                    </tr>
            """

            for trade in trade_logs:
                pnl = float(trade.get("NetPnLUSD", 0))
                pnl_color = "green" if pnl >= 0 else "red"
                body += f"""
                    <tr>
                        <td>{trade.get("TradeID", "N/A")}</td>
                        <td>{trade.get("Token", "N/A")}</td>
                        <td>{trade.get("Side", "N/A")}</td>
                        <td style="color:{pnl_color};">${pnl:,.2f}</td>
                        <td>{float(trade.get("VolatilityAtEntryATR", 0)):.4f}</td>
                        <td>{trade.get("ExitReason", "N/A")}</td>
                    </tr>
                """

            body += """
                </table>
            </body>
            </html>
            """

            await self.mailer.send_email(subject=subject, body=body)
            self.logger.info(
                f"Successfully sent daily trade summary email for {current_date.strftime('%Y-%m-%d')}."
            )

        except Exception as e:
            self.logger.error(
                f"Failed to generate or send daily summary email: {e}", exc_info=True
            )
