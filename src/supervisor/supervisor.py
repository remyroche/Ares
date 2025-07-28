import asyncio
import time
from typing import Dict, Any

from src.exchange.binance import BinanceExchange
from src.utils.logger import logger
from src.config import settings
from src.utils.state_manager import StateManager

class Supervisor:
    """
    The Supervisor acts as the high-level risk and performance manager for the system.
    It runs independently, monitoring overall portfolio health and enforcing risk policies
    by adjusting parameters or pausing trading when necessary.
    """

    def __init__(self, exchange_client: BinanceExchange, state_manager: StateManager):
        self.exchange = exchange_client
        self.state_manager = state_manager
        self.logger = logger.getChild('Supervisor')
        self.config = settings.get("supervisor", {})
        
        # Initialize performance and risk states
        self.state_manager.set_state_if_not_exists("peak_equity", settings.get("initial_equity", 10000))
        self.state_manager.set_state_if_not_exists("is_trading_paused", False)
        self.state_manager.set_state_if_not_exists("global_risk_multiplier", 1.0)

    async def start(self):
        """
        Starts the main supervisor monitoring loop.
        """
        self.logger.info("Supervisor started. Monitoring overall system performance and risk.")
        
        # Initial update of account state
        await self._update_account_state()

        check_interval = self.config.get("check_interval_seconds", 300) # Check every 5 minutes
        while True:
            try:
                await asyncio.sleep(check_interval)
                self.logger.info("--- Running Supervisor Health Check ---")
                
                # 1. Get the latest account status from the exchange
                await self._update_account_state()
                
                # 2. Check performance against risk thresholds
                await self._check_performance_and_risk()

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
