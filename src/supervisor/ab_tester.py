# src/supervisor/ab_tester.py
import datetime
import copy
import os
import json
import logging
from src.config import CONFIG
from src.utils.logger import system_logger
from src.paper_trader import PaperTrader

# from src.tactician.tactician import Tactician  # Circular import - removed
from emails.ares_mailer import send_email


class ABTester:
    def __init__(self, config=CONFIG, reporter=None):
        self.config = config.get("supervisor", {})
        self.global_config = config
        self.reporter = reporter
        self.logger: logging.Logger = system_logger.getChild("ABTester")

        self.ab_test_active = False
        self.ab_test_start_date = None
        self.challenger_tactician = None
        self.challenger_paper_trader = None
        self.challenger_params = None
        self.champion_params_snapshot = None
        self.shadow_log_file = "data/shadow_trades.jsonl"

    def start_ab_test(self, challenger_analyst, challenger_params, allocated_capital):
        self.logger.info(
            "--- INITIATING 1-WEEK A/B TEST (Challenger in Shadow Mode) ---"
        )
        self.ab_test_active = True
        self.ab_test_start_date = datetime.datetime.now()

        # Store both champion and challenger params for the final report
        self.champion_params_snapshot = copy.deepcopy(self.global_config["best_params"])
        self.challenger_params = challenger_params

        challenger_config = copy.deepcopy(self.global_config)
        challenger_config["best_params"] = self.challenger_params
        # Import here to avoid circular import
        from src.tactician.tactician import Tactician

        self.challenger_tactician = Tactician(config=challenger_config)

        # Instantiate the challenger's PaperTrader in shadow_mode
        self.challenger_paper_trader = PaperTrader(
            initial_equity=allocated_capital, shadow_mode=True
        )

        self.logger.info(
            "Challenger Tactician and Paper Trader are now active in SHADOW MODE."
        )

        # Clear the shadow log at the start of a new test to ensure a clean slate
        if os.path.exists(self.shadow_log_file):
            try:
                open(self.shadow_log_file, "w").close()
                self.logger.info("Cleared previous shadow trade log.")
            except Exception as e:
                self.logger.error(f"Could not clear shadow log file: {e}")

    async def evaluate_and_conclude_ab_test(self, champion_trade_logs):
        if not self.ab_test_active:
            return

        self.logger.info(
            "--- CONCLUDING A/B TEST & GENERATING REPORT (Reading from Shadow Log) ---"
        )
        self.ab_test_active = False

        # --- Performance Calculation ---
        champion_report = await self.reporter.generate_performance_report(
            champion_trade_logs,
            datetime.date.today(),
            self.reporter.risk_allocator.get_current_allocated_capital(),
        )
        champion_pnl = champion_report["daily_summary"]["NetPnL"]
        champion_sharpe = champion_report["daily_summary"].get("Sharpe Ratio", 0.0)

        # Read challenger trades from the shadow log file
        challenger_trades = []
        try:
            if os.path.exists(self.shadow_log_file):
                with open(self.shadow_log_file, "r") as f:
                    for line in f:
                        challenger_trades.append(json.loads(line))
        except Exception as e:
            self.logger.error(
                f"Could not read shadow trade log file: {e}", exc_info=True
            )
            challenger_trades = []

        # The performance report for the challenger is now based on the logged shadow trades
        challenger_report = await self.reporter.generate_performance_report(
            challenger_trades,
            datetime.date.today(),
            self.challenger_paper_trader.equity,
        )
        challenger_pnl = challenger_report["daily_summary"]["NetPnL"]
        challenger_sharpe = challenger_report["daily_summary"].get("Sharpe Ratio", 0.0)

        # --- Parameter Difference Highlighting ---
        param_diffs = {
            key: {
                "champion": self.champion_params_snapshot.get(key),
                "challenger": value,
            }
            for key, value in self.challenger_params.items()
            if value != self.champion_params_snapshot.get(key)
        }

        # --- Decision Logic ---
        performance_threshold = 1.05  # 5% improvement needed
        challenger_won = challenger_pnl > (champion_pnl * performance_threshold)
        outcome_summary = "Challenger WON" if challenger_won else "Champion Retained"

        # --- Construct Detailed Email Report ---
        email_subject = f"Ares A/B Test Complete: {outcome_summary}"
        email_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #e8f4fd; padding: 15px; border-left: 5px solid #007bff; }}
                .action {{ background-color: #fffbe6; padding: 15px; border-left: 5px solid #ffc107; font-size: 1.1em; }}
            </style>
        </head>
        <body>
            <h2>Ares A/B Test Conclusion Report</h2>
            <p><b>Test Period:</b> {self.ab_test_start_date.date()} to {datetime.date.today()}</p>
            
            <div class="summary">
                <h3>Overall Outcome: {outcome_summary}</h3>
                <p>The challenger model's performance was <b>{challenger_pnl / champion_pnl if champion_pnl != 0 else float("inf"):.2f}x</b> that of the champion based on Net PnL.</p>
                <p>A performance improvement of at least 5% was required for promotion.</p>
            </div>

            <h3>Performance Comparison</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Champion (Live/Paper)</th>
                    <th>Challenger (Shadow)</th>
                </tr>
                <tr>
                    <td><b>Net PnL</b></td>
                    <td>${champion_pnl:,.2f}</td>
                    <td><b>${challenger_pnl:,.2f}</b></td>
                </tr>
                <tr>
                    <td>Sharpe Ratio</td>
                    <td>{champion_sharpe:.2f}</td>
                    <td>{challenger_sharpe:.2f}</td>
                </tr>
                <tr>
                    <td>Total Trades</td>
                    <td>{champion_report["daily_summary"]["TotalTrades"]}</td>
                    <td>{challenger_report["daily_summary"]["TotalTrades"]}</td>
                </tr>
                <tr>
                    <td>Win Rate</td>
                    <td>{champion_report["daily_summary"]["WinRate"]}%</td>
                    <td>{challenger_report["daily_summary"]["WinRate"]}%</td>
                </tr>
                 <tr>
                    <td>Max Drawdown</td>
                    <td>{champion_report["daily_summary"]["MaxDrawdown"]}%</td>
                    <td>{challenger_report["daily_summary"]["MaxDrawdown"]}%</td>
                </tr>
            </table>

            <h3>Key Parameter Changes (Challenger vs. Champion)</h3>
            <pre>{json.dumps(param_diffs, indent=2)}</pre>

            <div class="action">
                <h3>Action Required</h3>
                <p>To promote the new challenger model to live trading, please reply to this email with the exact subject line:</p>
                <p><b>PROMOTE CHALLENGER</b></p>
            </div>
        </body>
        </html>
        """

        # Send the email
        send_email(email_subject, email_body)
        self.logger.info("A/B test conclusion report sent via email.")

        if challenger_won:
            self.logger.critical(
                "Challenger model outperformed. Creating flag to promote to champion."
            )
            with open(self.global_config.get("PROMOTE_CHALLENGER_FLAG_FILE"), "w") as f:
                f.write("PROMOTE")
        else:
            self.logger.info("Champion model performed better. Keeping the champion.")

        # Reset A/B test state
        self.challenger_tactician = None
        self.challenger_paper_trader = None
        self.champion_params_snapshot = None
        self.challenger_params = None
