# src/supervisor/main.py

import asyncio
import logging
import time

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
from src.exchange.binance import exchange # Import the global BinanceExchange instance for live trading

logger = logging.getLogger(__name__)


class Supervisor:
    """
    The central, real-time orchestrator of the Ares Trading Bot.
    It initializes, manages, and connects all the core components of the
    trading pipeline, ensuring they run concurrently and communicate efficiently.
    """
    def __init__(self):
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
        
        # Initialize the core real-time components, passing the global exchange instance and state manager
        self.sentinel = Sentinel(exchange, self.state_manager)
        self.analyst = Analyst(exchange, self.state_manager)
        self.strategist = Strategist(self.state_manager)
        
        # Initialize the trader based on the configured trading environment
        if settings.trading_environment == "PAPER":
            # In paper trading mode, `self.trader` is an instance of PaperTrader
            self.trader = PaperTrader(initial_equity=settings.initial_equity)
            self.logger.info("Paper Trader initialized for simulation.")
            # The tactician interacts with the PaperTrader as its exchange client
            self.tactician = Tactician(self.trader, self.state_manager)
        elif settings.trading_environment == "LIVE":
            # In live trading mode, `self.trader` is the global BinanceExchange instance
            self.trader = exchange
            self.logger.info("Live Trader (BinanceExchange) initialized for live operations.")
            # The tactician interacts with the real BinanceExchange client
            self.tactician = Tactician(self.trader, self.state_manager)
        else:
            self.trader = None
            self.tactician = None
            self.logger.error(f"Unknown trading environment: '{settings.trading_environment}'. Trading will be disabled.")
            
        self.running = False
        
        # Asynchronous queues for communication between components.
        # Note: In the current architecture (e.g., src/main.py and src/ares_pipeline.py),
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
        tasks = [
            asyncio.create_task(self.sentinel.start(), name="Sentinel_Task"),
            asyncio.create_task(self.analyst.start(), name="Analyst_Task"),
            asyncio.create_task(self.strategist.start(), name="Strategist_Task"),
        ]
        
        # Only add tactician task if it was successfully initialized
        if self.tactician:
             tasks.append(asyncio.create_task(self.tactician.start(), name="Tactician_Task"))

        # If in paper trading mode, also run the simulation engine
        if isinstance(self.trader, PaperTrader):
            tasks.append(asyncio.create_task(self.trader.run_simulation(), name="PaperTrader_Simulation_Task"))

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
            
            # Save the final state of the bot
            self.state_manager.save_state() # Call save_state without arguments
            self.logger.info("All components have been shut down and state has been saved.")
