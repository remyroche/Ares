# src/supervisor/supervisor.py

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any

from src.exchange.binance import BinanceExchange
from src.utils.logger import logger
from src.config import settings, CONFIG # Import CONFIG for retrain_interval_days
from src.utils.state_manager import StateManager
from src.database.firestore_manager import FirestoreManager
from src.supervisor.performance_monitor import PerformanceMonitor

class Supervisor:
    """
    The Supervisor acts as the high-level risk and performance manager for the system.
    It runs independently, monitoring overall portfolio health and enforcing risk policies
    by adjusting parameters or pausing trading when necessary.
    """

    def __init__(self, exchange_client: BinanceExchange, state_manager: StateManager, firestore_manager: FirestoreManager):
        """
        Initializes the Supervisor.

        Args:
            exchange_client (BinanceExchange): The exchange client for interacting with the exchange.
            state_manager (StateManager): The state manager for persisting and retrieving system state.
            firestore_manager (FirestoreManager): The Firestore manager for database operations.
        """
        self.exchange = exchange_client
        self.state_manager = state_manager
        self.firestore_manager = firestore_manager
        self.logger = logger.getChild('Supervisor')
        self.config = settings.get("supervisor", {}) # Specific supervisor config
        self.global_config = CONFIG # Access to global CONFIG for retrain_interval_days

        # Initialize performance and risk states
        self.state_manager.set_state_if_not_exists("peak_equity", settings.get("initial_equity", 10000))
        self.state_manager.set_state_if_not_exists("is_trading_paused", False)
        self.state_manager.set_state_if_not_exists("global_risk_multiplier", 1.0)
        # Initialize last_retrain_timestamp. Use current time if not set.
        self.state_manager.set_state_if_not_exists("last_retrain_timestamp", datetime.now().isoformat())

        # Initialize the PerformanceMonitor
        self.performance_monitor = PerformanceMonitor(config=settings, firestore_manager=self.firestore_manager)

    async def start(self):
        """
        Starts the main supervisor monitoring loop.
        """
        self.logger.info("Supervisor started. Monitoring overall system performance and risk.")
        
        # Initial update of account state
        await self._update_account_state()

        check_interval = self.config.get("check_interval_seconds", 300) # Check every 5 minutes
        retrain_interval_days = self.global_config['supervisor'].get("retrain_interval_days", 30) # Default to 30 days
        
        while True:
            try:
                await asyncio.sleep(check_interval)
                self.logger.info("--- Running Supervisor Health Check ---")
                
                # 1. Get the latest account status from the exchange
                await self._update_account_state()
                
                # 2. Check performance against risk thresholds
                await self._check_performance_and_risk()

                # 3. Prepare live metrics and run performance monitoring
                current_equity = self.state_manager.get_state("account_equity")
                peak_equity = self.state_manager.get_state("peak_equity")

                # Placeholder for live metrics. In a real system, these would be
                # calculated by a dedicated performance tracking module.
                live_metrics = {
                    'Final Equity': current_equity,
                    'Max Drawdown (%)': ((peak_equity - current_equity) / peak_equity * 100) if peak_equity > 0 else 0,
                    'Total Trades': self.state_manager.get_state("total_trades", 0), # Assuming total_trades is tracked elsewhere
                    'Profit Factor': self.state_manager.get_state("live_profit_factor", 0), # Placeholder
                    'Sharpe Ratio': self.state_manager.get_state("live_sharpe_ratio", 0), # Placeholder
                    'Win Rate (%)': self.state_manager.get_state("live_win_rate", 0) # Placeholder
                }
                await self.performance_monitor.monitor_performance(live_metrics)

                # 4. Check for scheduled model retraining
                await self._check_for_retraining(retrain_interval_days)

            except asyncio.CancelledError:
                self.logger.info("Supervisor task cancelled.")
                break
            except Exception as e:
                self.logger.error(f"An error occurred in the Supervisor loop: {e}", exc_info=True)

    async def _update_account_state(self):
        """
        Fetches the current account equity from the exchange and updates the state.
        """
        try:
            account_info = await self.exchange.get_account_info()
            # For futures, total wallet balance is often in 'totalWalletBalance'
            current_equity = float(account_info.get('totalWalletBalance', 0))
            
            if current_equity > 0:
                self.state_manager.set_state("account_equity", current_equity)
                self.logger.info(f"Updated account equity: ${current_equity:,.2f}")

                # Update peak equity
                peak_equity = self.state_manager.get_state("peak_equity")
                if current_equity > peak_equity:
                    self.state_manager.set_state("peak_equity", current_equity)
                    self.logger.info(f"New peak equity reached: ${current_equity:,.2f}")
            else:
                self.logger.warning("Could not retrieve a valid account balance.")

        except Exception as e:
            self.logger.error(f"Failed to update account state: {e}")

    async def _check_performance_and_risk(self):
        """
        Calculates drawdown and adjusts risk parameters or pauses trading if necessary.
        """
        peak_equity = self.state_manager.get_state("peak_equity")
        current_equity = self.state_manager.get_state("account_equity")

        if not peak_equity or not current_equity or peak_equity == 0:
            self.logger.warning("Cannot check performance; equity data is missing.")
            return

        # Calculate Drawdown
        drawdown = (peak_equity - current_equity) / peak_equity
        self.logger.info(f"Current Drawdown: {drawdown:.2%}")

        # Get risk thresholds from config
        pause_threshold = self.config.get("pause_trading_drawdown_pct", 0.20) # 20%
        risk_reduction_threshold = self.config.get("risk_reduction_drawdown_pct", 0.10) # 10%

        # Check for severe drawdown to pause all trading
        if drawdown >= pause_threshold:
            if not self.state_manager.get_state("is_trading_paused"):
                await self._pause_trading(f"Drawdown of {drawdown:.2%} exceeded pause threshold of {pause_threshold:.2%}")
            return # No further action if trading is paused

        # Check for moderate drawdown to reduce risk
        if drawdown >= risk_reduction_threshold:
            new_risk_multiplier = 0.5 # Reduce risk by 50%
            if self.state_manager.get_state("global_risk_multiplier") != new_risk_multiplier:
                self.logger.warning(f"Drawdown of {drawdown:.2%} exceeded risk reduction threshold. Reducing risk multiplier to {new_risk_multiplier}.")
                self.state_manager.set_state("global_risk_multiplier", new_risk_multiplier)
        else:
            # If performance recovers, restore risk
            if self.state_manager.get_state("global_risk_multiplier") != 1.0:
                self.logger.info("Performance has recovered. Restoring global risk multiplier to 1.0.")
                self.state_manager.set_state("global_risk_multiplier", 1.0)
        
        # If trading was paused but has now recovered above the pause threshold, resume it
        if self.state_manager.get_state("is_trading_paused") and drawdown < pause_threshold:
            await self._resume_trading()

    async def _pause_trading(self, reason: str):
        """Pauses all new trading activity."""
        self.logger.critical(f"PAUSING ALL TRADING. Reason: {reason}")
        self.state_manager.set_state("is_trading_paused", True)
        # Here you could add logic to send an alert (email, etc.)

    async def _resume_trading(self):
        """Resumes trading activity."""
        self.logger.info("Resuming trading activity. Drawdown has recovered to an acceptable level.")
        self.state_manager.set_state("is_trading_paused", False)

    async def _check_for_retraining(self, retrain_interval_days: int):
        """
        Checks if it's time to trigger a system retraining based on the configured interval.
        """
        last_retrain_timestamp_str = self.state_manager.get_state("last_retrain_timestamp")
        
        try:
            last_retrain_datetime = datetime.fromisoformat(last_retrain_timestamp_str)
        except (TypeError, ValueError):
            self.logger.warning("Invalid last_retrain_timestamp found in state. Resetting to now.")
            last_retrain_datetime = datetime.now()
            self.state_manager.set_state("last_retrain_timestamp", last_retrain_datetime.isoformat())

        next_retrain_due = last_retrain_datetime + timedelta(days=retrain_interval_days)
        
        if datetime.now() >= next_retrain_due:
            self.logger.info(f"Retraining interval of {retrain_interval_days} days has passed. Triggering system retraining.")
            await self._trigger_retraining()
            # Update the timestamp after triggering retraining
            self.state_manager.set_state("last_retrain_timestamp", datetime.now().isoformat())
        else:
            time_until_next_retrain = next_retrain_due - datetime.now()
            self.logger.info(f"Next system retraining due in: {time_until_next_retrain.days} days, {time_until_next_retrain.seconds // 3600} hours.")

    async def _trigger_retraining(self):
        """
        Placeholder method to trigger the full system retraining pipeline.
        In a real scenario, this would call the relevant training orchestration module.
        """
        self.logger.info("Initiating full system retraining pipeline...")
        # TODO: Implement the actual call to the training pipeline here.
        # Example: await training_pipeline.run_training_pipeline()
        self.logger.info("System retraining initiated (placeholder).")

