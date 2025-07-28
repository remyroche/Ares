# src/supervisor/ab_tester.py
import datetime
import copy
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

        self.logger.info("Challenger Tactician and Paper Trader are now active.")

    def evaluate_and_conclude_ab_test(self, champion_trade_logs):
        if not self.ab_test_active:
            return

        self.logger.info("--- CONCLUDING A/B TEST ---")
        self.ab_test_active = False

        champion_report = self.reporter.generate_performance_report(champion_trade_logs, datetime.date.today(), self.reporter.risk_allocator.get_current_allocated_capital())
        champion_pnl = champion_report['daily_summary']['NetPnL']
        
        challenger_trades = self.challenger_paper_trader.trades
        challenger_report = self.reporter.generate_performance_report(challenger_trades, datetime.date.today(), self.reporter.risk_allocator.get_current_allocated_capital())
        challenger_pnl = challenger_report['daily_summary']['NetPnL']

        self.logger.info(f"A/B Test Results: Champion PnL: ${champion_pnl:.2f}, Challenger PnL: ${challenger_pnl:.2f}")

        if challenger_pnl > champion_pnl:
            self.logger.info("Challenger model outperformed. Promoting to champion.")
            # In a real system, this would trigger a model promotion and pipeline restart.
        else:
            self.logger.info("Champion model performed better. Keeping the champion.")

        self.challenger_tactician = None
        self.challenger_paper_trader = None
