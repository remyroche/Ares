# src/supervisor/global_portfolio_manager.py

import asyncio
from datetime import datetime
from src.utils.logger import logger
from src.config import settings
from src.utils.state_manager import StateManager
from src.database.firestore_manager import FirestoreManager
from src.exchange.binance import BinanceExchange # Assuming Binance for now, can be made generic

class GlobalPortfolioManager:
    """
    A centralized manager that monitors the overall health of the entire trading
    portfolio across all configured exchanges and accounts. It enforces global
    risk policies, including capital allocation.
    """

    def __init__(self, state_manager: StateManager, firestore_manager: FirestoreManager):
        self.state_manager = state_manager
        self.firestore_manager = firestore_manager
        self.logger = logger.getChild('GlobalPortfolioManager')
        self.config = settings.get("global_portfolio_manager", {})
        self.risk_config = settings.get("risk_management", {}) # Load risk management config
        self.exchange_configs = settings.get("exchanges", [])
        self.exchange_clients = []

        # Initialize state for the global portfolio
        self.state_manager.set_state_if_not_exists("global_peak_equity", settings.get("initial_equity", 10000))
        self.state_manager.set_state_if_not_exists("global_trading_status", "RUNNING") # RUNNING, PAUSED, REDUCE_EXPOSURE

    def _initialize_exchange_clients(self):
        """Initializes clients for all exchanges defined in the config."""
        for ex_config in self.exchange_configs:
            if ex_config.get("name").lower() == "binance":
                self.exchange_clients.append(
                    BinanceExchange(
                        api_key=ex_config.get("api_key"),
                        api_secret=ex_config.get("api_secret"),
                        paper_trade=ex_config.get("paper_trade", True)
                    )
                )
            # Add other exchanges here as needed
        self.logger.info(f"Initialized {len(self.exchange_clients)} exchange client(s).")

    async def start(self):
        """Starts the main monitoring loop for the global portfolio."""
        self.logger.info("Global Portfolio Manager started.")
        self._initialize_exchange_clients()
        
        # Publish risk parameters on startup for all bots to consume
        await self._publish_risk_parameters()
        
        check_interval = self.config.get("check_interval_seconds", 300)

        while True:
            try:
                await asyncio.sleep(check_interval)
                self.logger.info("--- Running Global Portfolio Health Check ---")
                
                total_equity = await self._calculate_total_portfolio_equity()
                
                if total_equity is not None:
                    self._update_peak_equity(total_equity)
                    await self._check_global_risk(total_equity)
                    await self._check_total_exposure()

            except asyncio.CancelledError:
                self.logger.info("Global Portfolio Manager task cancelled.")
                break
            except Exception as e:
                self.logger.error(f"An error occurred in the GlobalPortfolioManager loop: {e}", exc_info=True)

    async def _publish_risk_parameters(self):
        """Publishes risk parameters from config to Firestore for local bots to use."""
        max_allocation_per_pair = self.risk_config.get("max_allocation_per_pair_usd", 5000)
        self.logger.info(f"Publishing risk parameters: Max allocation per pair = ${max_allocation_per_pair:,.2f}")
        
        await self.firestore_manager.set_document(
            "global_state",
            "risk_params",
            {"max_allocation_per_pair_usd": max_allocation_per_pair, "timestamp": datetime.now().isoformat()}
        )

    async def _calculate_total_portfolio_equity(self) -> float | None:
        """Fetches and aggregates equity from all configured exchange accounts."""
        total_equity = 0.0
        for client in self.exchange_clients:
            try:
                account_info = await client.get_account_info()
                equity = float(account_info.get('totalWalletBalance', 0))
                total_equity += equity
                self.logger.debug(f"Fetched equity ${equity:,.2f} from exchange.")
            except Exception as e:
                self.logger.error(f"Failed to fetch account info from an exchange: {e}")
                return None
        
        self.logger.info(f"Total portfolio equity across all exchanges: ${total_equity:,.2f}")
        self.state_manager.set_state("global_current_equity", total_equity)
        return total_equity

    def _update_peak_equity(self, current_equity: float):
        """Updates the global peak equity."""
        peak_equity = self.state_manager.get_state("global_peak_equity")
        if current_equity > peak_equity:
            self.state_manager.set_state("global_peak_equity", current_equity)
            self.logger.info(f"New global peak equity reached: ${current_equity:,.2f}")

    async def _check_total_exposure(self):
        """Checks if the total deployed capital exceeds the configured global limit."""
        max_allowed_exposure = self.risk_config.get("global_max_allocated_capital_usd", 50000)
        total_exposure = 0.0

        for client in self.exchange_clients:
            try:
                positions = await client.get_open_positions()
                for pos in positions:
                    # notional value is a good measure of exposure
                    total_exposure += abs(float(pos.get('notional', 0)))
            except Exception as e:
                self.logger.error(f"Could not get position exposure from an exchange: {e}")
        
        self.logger.info(f"Current total exposure: ${total_exposure:,.2f}. Global limit: ${max_allowed_exposure:,.2f}")

        if total_exposure > max_allowed_exposure:
            await self._update_global_status("REDUCE_EXPOSURE", f"Total exposure of ${total_exposure:,.2f} exceeds limit of ${max_allowed_exposure:,.2f}.")
        else:
            # If exposure is back within limits, we can return to a normal running state
            # (unless paused for another reason like drawdown)
            current_status = self.state_manager.get_state("global_trading_status")
            if current_status == "REDUCE_EXPOSURE":
                await self._update_global_status("RUNNING", "Total exposure is back within acceptable limits.")


    async def _check_global_risk(self, current_equity: float):
        """Checks global drawdown and updates the global trading status in Firestore."""
        peak_equity = self.state_manager.get_state("global_peak_equity")
        if peak_equity == 0:
            return

        drawdown = (peak_equity - current_equity) / peak_equity
        self.logger.info(f"Global portfolio drawdown: {drawdown:.2%}")

        pause_threshold = self.risk_config.get("pause_trading_drawdown_pct", 0.20)
        
        if drawdown >= pause_threshold:
            await self._update_global_status("PAUSED", f"Drawdown of {drawdown:.2%} exceeded the global threshold of {pause_threshold:.2%}.")
        else:
            # If we are not in a drawdown state, ensure we are not paused for this reason.
            current_status = self.state_manager.get_state("global_trading_status")
            if current_status == "PAUSED":
                await self._update_global_status("RUNNING", "Global portfolio has recovered from drawdown.")

    async def _update_global_status(self, new_status: str, reason: str):
        """
        Helper method to update the global trading status in the state manager and Firestore.
        """
        current_status = self.state_manager.get_state("global_trading_status")
        if new_status == current_status:
            return # No change needed

        self.logger.warning(f"Updating global status from '{current_status}' to '{new_status}'. Reason: {reason}")
        self.state_manager.set_state("global_trading_status", new_status)
        
        try:
            await self.firestore_manager.set_document(
                "global_state", 
                "status", 
                {"trading_status": new_status, "reason": reason, "timestamp": datetime.now().isoformat()}
            )
            self.logger.info(f"Successfully published new global status '{new_status}' to Firestore.")
        except Exception as e:
            self.logger.error(f"Failed to publish global status to Firestore: {e}", exc_info=True)

