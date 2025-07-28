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
# No need to import src.exchange.binance.exchange here directly, it's passed in


logger = logging.getLogger(__name__)


class Supervisor:
    """
    The central, real-time orchestrator of the Ares Trading Bot.
    It initializes, manages, and connects all the core components of the
    trading pipeline, ensuring they run concurrently and communicate efficiently.
    """
    def __init__(self, exchange_client: Any): # Accept the exchange client from main.py
        self.logger = logger
        self.state_manager = StateManager('ares_state.json') # Initialize StateManager with its file path
        self.state = self.state_manager.get_state() # Use get_state() to load current state
        self.config = CONFIG # Use the global CONFIG dictionary for general settings

        # Initialize FirestoreManager (it uses settings internally)
        self.firestore_manager = FirestoreManager()
        
        # Initialize Supervisor sub-components, passing necessary dependencies
        self.risk_allocator = RiskAllocator(self.config)
        self.performance_reporter = PerformanceReporter(self.config, self.firestore_manager)
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

        # Initialize the core real-time components, passing the selected trader and state manager
        if self.trader:
            self.sentinel = Sentinel(self.trader, self.state_manager)
            self.analyst = Analyst(self.trader, self.state_manager)
            self.strategist = Strategist(self.state_manager) # Strategist doesn't need exchange client directly
            self.tactician = Tactician(self.trader, self.state_manager) # Tactician needs the trading client
        else:
            # If trader is None due to invalid environment, set all components to None
            self.sentinel = None
            self.analyst = None
            self.strategist = None
            self.tactician = None
            self.logger.critical("Core trading components not initialized due to invalid trading environment.")
            
        self.running = False
        
        # Asynchronous queues for communication between components.
        # Note: In the current architecture (e.g., src/ares_pipeline.py),
        # inter-module communication primarily happens via the StateManager.
        # These queues might be remnants of a different design or for future expansion.
        self.market_data_queue = asyncio.Queue(maxsize=100)
        self.analysis_queue = asyncio.Queue(maxsize=100)
        self.signal_queue = asyncio.Queue(maxsize=50)

    async def start(self):
        """
        Starts all bot components and the main processing loop.
        This method orchestrates the concurrent execution of the main modules.
        """
        self.logger.info("Supervisor starting all components...")
        self.running = True

        # Initialize the Firestore database manager asynchronously
        if hasattr(db_manager, 'initialize') and asyncio.iscoroutinefunction(db_manager.initialize):
            await db_manager.initialize()

        # Create a list of all concurrent tasks for the main modules
        tasks = []
        if self.trader and self.sentinel and self.analyst and self.strategist and self.tactician:
            tasks.extend([
                asyncio.create_task(self.sentinel.start(), name="Sentinel_Task"),
                asyncio.create_task(self.analyst.start(), name="Analyst_Task"),
                asyncio.create_task(self.strategist.start(), name="Strategist_Task"),
                asyncio.create_task(self.tactician.start(), name="Tactician_Task")
            ])
            # If in paper trading mode, also run the simulation engine
            if isinstance(self.trader, PaperTrader):
                tasks.append(asyncio.create_task(self.trader.run_simulation(), name="PaperTrader_Simulation_Task"))
        else:
            self.logger.error("Cannot start supervisor: Core trading components are not initialized.")
            self.running = False
            return # Exit early if components are not ready

        try:
            # Run all component tasks concurrently
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self.logger.info("Supervisor tasks cancelled. Beginning graceful shutdown...")
        finally:
            self.running = False
            # Ensure all tasks are properly cancelled on exit
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for all tasks to acknowledge cancellation
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Perform final cleanup actions
            if self.trader and hasattr(self.trader, 'close'):
                await self.trader.close() # Close the exchange client (real or paper)
            self.state_manager.save_state() # Call save_state without arguments
            self.logger.info("All components have been shut down and state has been saved.")
