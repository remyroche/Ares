import pandas as pd
import datetime
import os
import asyncio
from loguru import logger

# Assuming a global config dictionary and logger are available
# from src.config import settings # Using settings from our established config
from src.config import settings
from src.database.firestore_manager import db_manager
from .performance_reporter import PerformanceReporter
from .risk_allocator import RiskAllocator
from .optimizer import Optimizer
from .ab_tester import ABTester

class Supervisor:
    """
    The Supervisor module (Meta-Learning Governor) optimizes the entire trading strategy
    and manages capital allocation over long time horizons. It also handles enhanced
    performance reporting and A/B testing by orchestrating its sub-modules.
    """
    def __init__(self):
        # Using the global settings object for configuration
        self.config = settings.supervisor_config if hasattr(settings, 'supervisor_config') else {}
        self.logger = logger
        
        # Initialize sub-modules
        # Pass the db_manager instance to components that need it
        self.reporter = PerformanceReporter(settings, db_manager)
        self.risk_allocator = RiskAllocator(settings)
        self.optimizer = Optimizer(settings, db_manager)
        self.ab_tester = ABTester(settings, self.reporter)
        
        # State for configurable retraining schedule
        self.retraining_schedule_config = self.config.get("retraining_schedule", {})
        self.next_retraining_date = None
        if self.retraining_schedule_config.get("enabled", False):
            try:
                self.next_retraining_date = datetime.datetime.strptime(
                    self.retraining_schedule_config.get("first_retraining_date"), "%Y-%m-%d"
                ).date()
            except (ValueError, TypeError):
                self.logger.error("Invalid 'first_retraining_date' format in config. Please use YYYY-MM-DD. Disabling scheduled retraining.")
                self.retraining_schedule_config["enabled"] = False


    async def orchestrate_supervision(self, current_date: datetime.date, total_equity: float, 
                                      daily_trade_logs: list, daily_pnl_per_regime: dict,
                                      historical_daily_pnl_data: pd.DataFrame):
        """
        Main orchestration method for the Supervisor, called periodically (e.g., daily/weekly).
        """
        self.logger.info(f"\n--- Supervisor: Starting Orchestration for {current_date} ---")

        # 1. A/B Testing Management
        if self.ab_tester.ab_test_active and (datetime.datetime.now().date() - self.ab_tester.ab_test_start_date.date()).days >= 7:
            await self.ab_tester.evaluate_and_conclude_ab_test(daily_trade_logs)

        # 2. Dynamic Risk Allocation
        today_net_pnl = sum(t.get('realized_pnl_usd', 0) for t in daily_trade_logs)
        new_daily_pnl_row = pd.DataFrame([{'Date': current_date, 'NetPnL': today_net_pnl}])
        if 'Date' in historical_daily_pnl_data.columns:
            historical_daily_pnl_data['Date'] = pd.to_datetime(historical_daily_pnl_data['Date'])
        
        updated_historical_pnl = pd.concat([historical_daily_pnl_data, new_daily_pnl_row]).drop_duplicates(subset=['Date']).sort_values('Date')
        
        self.risk_allocator.calculate_dynamic_capital_allocation(updated_historical_pnl)

        # 3. Performance Reporting
        report = await self.reporter.generate_performance_report(daily_trade_logs, current_date, self.risk_allocator.get_current_allocated_capital())
        await self.reporter.update_daily_summary_csv_and_firestore(report["daily_summary"])
        await self.reporter.update_strategy_performance_log_and_firestore(current_date, report["strategy_breakdown"])

        # 4. Configurable Retraining and Optimization Trigger
        if self.retraining_schedule_config.get("enabled", False) and self.next_retraining_date and current_date >= self.next_retraining_date:
            self.logger.info(f"Scheduled retraining date reached. Kicking off training pipeline for {current_date.strftime('%Y-%m')}...")
            # In a real system, this would trigger an external process, e.g., a Celery task or a script run.
            # For demonstration, we log the action. A real implementation would use:
            # os.system('python backtesting/training_pipeline.py &')
            
            # Schedule the next run
            period_days = self.retraining_schedule_config.get("retraining_period_days", 30)
            self.next_retraining_date = current_date + datetime.timedelta(days=period_days)
            self.logger.info(f"Next retraining scheduled for: {self.next_retraining_date}")

        # 5. Global System Optimization (Meta-Learning) - Run periodically
        if current_date.day % self.config.get("meta_learning_frequency_days", 7) == 0:
            await self.optimizer.implement_global_system_optimization(updated_historical_pnl, report["strategy_breakdown"])
        
        self.logger.info(f"--- Supervisor: Orchestration Complete for {current_date} ---")
        return {
            "allocated_capital": self.risk_allocator.get_current_allocated_capital(),
            "daily_summary": report["daily_summary"],
            "strategy_breakdown": report["strategy_breakdown"]
        }

    def start_ab_test(self, challenger_analyst, challenger_params):
        """Public method to initiate an A/B test."""
        self.ab_tester.start_ab_test(challenger_analyst, challenger_params, self.risk_allocator.get_current_allocated_capital())

    def get_current_allocated_capital(self):
        """Public method to get current capital from the risk allocator."""
        return self.risk_allocator.get_current_allocated_capital()
