# src/supervisor/main.py

import asyncio
import logging
import time
from typing import Any # Import Any for type hinting

# Import necessary modules
from src.config import settings, CONFIG # Import settings for trading_mode, and CONFIG for other params
from src.database.firestore_manager import FirestoreManager, db_manager
from src.supervisor.ab_tester import ABTester
from src.supervisor.performance_reporter import PerformanceReporter
from src.supervisor.risk_allocator import RiskAllocator
from src.supervisor.monitoring import Monitoring
from src.sentinel.sentinel import Sentinel
from src.analyst.analyst import Analyst
from src.strategist.strategist import Strategist
from src.tactician.tactician import Tactician
from src.paper_trader import PaperTrader # Import PaperTrader for paper trading mode
from src.utils.state_manager import StateManager
from src.utils.model_manager import ModelManager


logger = logging.getLogger(__name__)


class Supervisor:
    """
    The central, real-time orchestrator of the Ares Trading Bot.
    It initializes, manages, and connects all the core components of the
    trading pipeline, ensuring they run concurrently and communicate efficiently.
    """
    def __init__(self, exchange_client: Any): # Accept the exchange client from main.py
        self.logger = logger
        self.state_manager = StateManager(state_file=CONFIG.get("state_file", "ares_state.json")) # Pass state_file from CONFIG
        self.state = self.state_manager.get_state("global_trading_status") # Use get_state() to load current state
        self.config = CONFIG # Use the global CONFIG dictionary for general settings

        # Initialize FirestoreManager (it uses settings internally)
        self.firestore_manager = FirestoreManager()
        
        # Initialize Supervisor sub-components, passing necessary dependencies
        self.risk_allocator = RiskAllocator(self.config)
        self.performance_reporter = PerformanceReporter(self.config, self.firestore_manager) # Pass firestore_manager
        self.ab_tester = ABTester(self.config, self.performance_reporter)
        self.monitoring = Monitoring(self.firestore_manager)
        
        # Determine the actual trading client (PaperTrader or live exchange_client)
        if settings.trading_environment == "PAPER":
            self.trader = PaperTrader(initial_equity=settings.initial_equity)
            self.logger.info("Paper Trader initialized for simulation.")
        elif settings.trading_environment == "LIVE":
            self.trader = exchange_client # Use the live exchange client passed from main
            self.logger.info("Live Trader (BinanceExchange) initialized for live operations.")
        else:
            self.trader = None
            self.logger.error(f"Unknown trading environment: '{settings.trading_environment}'. Trading will be disabled.")
            raise ValueError(f"Invalid TRADING_ENVIRONMENT: {settings.trading_environment}") # Halt if invalid

        # Initialize ModelManager first, which will load the champion models
        # Pass performance_reporter to ModelManager so it can pass it to Tactician
        self.model_manager = ModelManager(firestore_manager=self.firestore_manager, performance_reporter=self.performance_reporter)

        # Initialize the core real-time components, getting instances from ModelManager
        if self.trader:
            self.sentinel = Sentinel(self.trader, self.state_manager) # Sentinel needs the real trader
            self.analyst = self.model_manager.get_analyst() # Get Analyst instance from ModelManager
            self.strategist = self.model_manager.get_strategist() # Get Strategist instance from ModelManager
            # Tactician instance is already created by ModelManager with performance_reporter
            self.tactician = self.model_manager.get_tactician() 

            # Ensure the Analyst, Strategist, Tactician instances from ModelManager
            # have their exchange_client and state_manager set if they need it for live ops.
            # This is a critical point for dependency injection.
            # For the training pipeline, these are mostly placeholders.
            if hasattr(self.analyst, 'exchange') and self.analyst.exchange is None:
                self.analyst.exchange = self.trader
            if hasattr(self.analyst, 'state_manager') and self.analyst.state_manager is None:
                self.analyst.state_manager = self.state_manager

            if hasattr(self.strategist, 'exchange') and self.strategist.exchange is None:
                self.strategist.exchange = self.trader
            if hasattr(self.strategist, 'state_manager') and self.strategist.state_manager is None:
                self.strategist.state_manager = self.state_manager

            if hasattr(self.tactician, 'exchange') and self.tactician.exchange is None:
                self.tactician.exchange = self.trader
            if hasattr(self.tactician, 'state_manager') and self.tactician.state_manager is None:
                self.tactician.state_manager = self.state_manager


        else:
            self.sentinel = None
            self.analyst = None
            self.strategist = None
            self.tactician = None
            self.logger.critical("Core trading components not initialized due to invalid trading environment.")
            
        self.running = False
        
        self.market_data_queue = asyncio.Queue(maxsize=100)
        self.analysis_queue = asyncio.Queue(maxsize=100)
        self.signal_queue = asyncio.Queue(maxsize=50)

    async def start(self):
        """
        Starts all bot components and the main processing loop.
        """
        self.logger.info("Supervisor starting all components...")
        self.running = True

        if hasattr(db_manager, 'initialize') and asyncio.iscoroutinefunction(db_manager.initialize):
            await db_manager.initialize()

        tasks = []
        if self.trader and self.sentinel and self.analyst and self.strategist and self.tactician:
            tasks.extend([
                asyncio.create_task(self.sentinel.start(), name="Sentinel_Task"),
                asyncio.create_task(self.analyst.start(), name="Analyst_Task"),
                asyncio.create_task(self.strategist.start(), name="Strategist_Task"),
                asyncio.create_task(self.tactician.start(), name="Tactician_Task")
            ])
            if isinstance(self.trader, PaperTrader):
                tasks.append(asyncio.create_task(self.trader.run_simulation(), name="PaperTrader_Simulation_Task"))
        else:
            self.logger.error("Cannot start supervisor: Core trading components are not initialized.")
            self.running = False
            return

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("Supervisor tasks cancelled. Beginning graceful shutdown...")
        finally:
            self.running = False
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            
            if self.trader and hasattr(self.trader, 'close'):
                await self.trader.close()
            self.state_manager._save_state_to_file() # Call internal save method
            self.logger.info("All components have been shut down and state has been saved.")

    async def _synchronize_exchange_state(self):
        """
        Fetches the current account equity and open positions from the exchange
        and updates the persistent state. This is key for crash recovery.
        """
        try:
            # 1. Update account equity and peak equity
            account_info = await self.exchange.get_account_info()
            current_equity = float(account_info.get('totalWalletBalance', 0))
            
            if current_equity > 0:
                self.state_manager.set_state("account_equity", current_equity)
                self.logger.debug(f"Updated account equity: ${current_equity:,.2f}")

                peak_equity = self.state_manager.get_state("global_peak_equity") # Use global_peak_equity from state
                if current_equity > peak_equity:
                    self.state_manager.set_state("global_peak_equity", current_equity)
                    self.logger.info(f"New peak equity reached: ${current_equity:,.2f}")
            else:
                self.logger.warning("Could not retrieve a valid account balance.")

            # 2. Update open positions state for crash recovery
            open_positions = await self.exchange.get_open_positions()
            symbol = settings.trade_symbol # Use settings.trade_symbol
            active_position_on_exchange = None
            
            for position in open_positions:
                if position.get('symbol') == symbol and float(position.get('positionAmt', 0)) != 0:
                    # Capture more details for active_position
                    active_position_on_exchange = {
                        "symbol": position['symbol'],
                        "amount": float(position['positionAmt']),
                        "entry_price": float(position['entryPrice']),
                        "leverage": int(position.get('leverage', 1)),
                        "direction": "LONG" if float(position['positionAmt']) > 0 else "SHORT",
                        "trade_id": self.state_manager.get_state("current_position", {}).get("trade_id"), # Attempt to recover trade_id
                        "entry_timestamp": self.state_manager.get_state("current_position", {}).get("entry_timestamp"), # Attempt to recover timestamp
                        "stop_loss": self.state_manager.get_state("current_position", {}).get("stop_loss"),
                        "take_profit": self.state_manager.get_state("current_position", {}).get("take_profit"),
                        "entry_fees_usd": self.state_manager.get_state("current_position", {}).get("entry_fees_usd", 0.0),
                        "entry_context": self.state_manager.get_state("current_position", {}).get("entry_context", {})
                    }
                    self.logger.debug(f"Found active position on exchange for {symbol}.")
                    break 

            # Synchronize the state file with what's on the exchange
            current_state_position = self.state_manager.get_state('current_position') # Use 'current_position'
            
            # Only update if there's a meaningful change or new position found
            if active_position_on_exchange != current_state_position:
                self.logger.info(f"State mismatch or update: Synchronizing position state with exchange. New state: {active_position_on_exchange}")
                self.state_manager.set_state('current_position', active_position_on_exchange) # Update 'current_position'

        except Exception as e:
            self.logger.error(f"Failed to synchronize state with exchange: {e}", exc_info=True)
