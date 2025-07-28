# src/supervisor/ab_tester.py
import datetime
import copy
import os
from src.config import CONFIG
from src.utils.logger import system_logger
from src.paper_trader import PaperTrader
from src.tactician.tactician import Tactician

class ABTester:
    def __init__(self, config=CONFIG, reporter=None):
        self.config = config.get("supervisor", {})
        self.global_config = config
        self.reporter = reporter
        self.logger = system_logger.getChild('ABTester')
        
        self.ab_test_active = False
        self.ab_test_start_date = None
        self.challenger_tactician = None
        self.challenger_paper_trader = None

    def start_ab_test(self, challenger_analyst, challenger_params, allocated_capital):
        self.logger.info("--- INITIATING 1-WEEK A/B TEST ---")
        self.ab_test_active = True
        self.ab_test_start_date = datetime.datetime.now()
        
        challenger_config = copy.deepcopy(self.global_config)
        challenger_config['BEST_PARAMS'] = challenger_params
        self.challenger_tactician = Tactician(config=challenger_config)
        self.challenger_paper_trader = PaperTrader(initial_equity=allocated_capital)

        self.logger.info("Challenger Tactician and Paper Trader are now active for paper trading.")

    async def evaluate_and_conclude_ab_test(self, champion_trade_logs):
        if not self.ab_test_active:
            return

        self.logger.info("--- CONCLUDING A/B TEST ---")
        self.ab_test_active = False

        # Generate a detailed report comparing the two
        champion_report = await self.reporter.generate_performance_report(champion_trade_logs, datetime.date.today(), self.reporter.risk_allocator.get_current_allocated_capital())
        champion_pnl = champion_report['daily_summary']['NetPnL']
        
        challenger_trades = self.challenger_paper_trader.trades
        challenger_report = await self.reporter.generate_performance_report(challenger_trades, datetime.date.today(), self.challenger_paper_trader.equity)
        challenger_pnl = challenger_report['daily_summary']['NetPnL']

        report_str = f"""
        ========================================
        A/B Test Conclusion Report
        ========================================
        Test Period: {self.ab_test_start_date.date()} to {datetime.date.today()}

        --- Champion Performance ---
        Net PnL: ${champion_pnl:,.2f}
        Total Trades: {champion_report['daily_summary']['TotalTrades']}
        Win Rate: {champion_report['daily_summary']['WinRate']}%

        --- Challenger Performance (Paper Traded) ---
        Net PnL: ${challenger_pnl:,.2f}
        Total Trades: {challenger_report['daily_summary']['TotalTrades']}
        Win Rate: {challenger_report['daily_summary']['WinRate']}%
        ========================================
        """
        self.logger.info(report_str)

        if challenger_pnl > champion_pnl:
            self.logger.critical("Challenger model outperformed. Creating flag to promote to champion.")
            # Create a flag file. The main pipeline will detect this and trigger the hot-swap.
            with open(self.global_config.get("PROMOTE_CHALLENGER_FLAG_FILE"), 'w') as f:
                f.write("PROMOTE")
        else:
            self.logger.info("Champion model performed better. Keeping the champion and deleting challenger files.")
            # Optional: Clean up challenger models
            # import shutil
            # shutil.rmtree('models/challenger', ignore_errors=True)

        self.challenger_tactician = None
        self.challenger_paper_trader = None
