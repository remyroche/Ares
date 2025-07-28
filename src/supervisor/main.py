# src/supervisor/main.py

import asyncio
import logging
import time

# The user-provided code had several imports for components not defined in the current context
# (Sentinel, Analyst, Strategist, Tactician, PaperTrader, StateManager, db_manager).
# We are assuming these exist in the user's project structure.
from src.config import Config
from src.database.firestore_manager import FirestoreManager, db_manager
from src.supervisor.ab_tester import ABTester
from src.supervisor.performance_reporter import PerformanceReporter
from src.supervisor.risk_allocator import RiskAllocator
from src.supervisor.monitoring import Monitoring
from src.sentinel.sentinel import Sentinel
from src.analyst.analyst import Analyst
from src.strategist.strategist import Strategist
from src.tactician.tactician import Tactician
from src.paper_trader import PaperTrader
from src.utils.state_manager import StateManager


logger = logging.getLogger(__name__)


class Supervisor:
    """
    The central, real-time orchestrator of the Ares Trading Bot.
    It initializes, manages, and connects all the core components of the
    trading pipeline, ensuring they run concurrently and communicate efficiently.
    """
    def __init__(self):
        self.logger = logger
        self.state_manager = StateManager('ares_state.json')
        self.state = self.state_manager.load_state()
        self.config = Config()
        self.firestore_manager = FirestoreManager(self.config)
        self.risk_allocator = RiskAllocator(self.config, self.firestore_manager)
        self.performance_reporter = PerformanceReporter(self.firestore_manager)
        self.ab_tester = ABTester(self.firestore_manager)
        self.monitoring = Monitoring(self.firestore_manager)
        
        # Initialize the core real-time components
        self.sentinel = Sentinel()
        self.analyst = Analyst()
        self.strategist = Strategist(self.state)
        
        # Initialize the trader based on the configured mode
        if self.config.PAPER_TRADING:
            self.trader = PaperTrader(self.state)
            self.logger.info("Paper Trader initialized.")
        else:
            self.trader = None # Live trading not implemented
            self.logger.error("Live trading mode is not fully implemented yet.")
            # In a real scenario, you might want to raise an exception or handle this differently
            # raise NotImplementedError("Live trading not implemented.")

        if self.trader:
            self.tactician = Tactician(self.trader)
        else:
            self.tactician = None
        
        self.running = False
        
        # Asynchronous queues for communication between components
        self.market_data_queue = asyncio.Queue(maxsize=100)
        self.analysis_queue = asyncio.Queue(maxsize=100)
        self.signal_queue = asyncio.Queue(maxsize=50)

    async def start(self):
        """Starts all bot components and the main processing loop."""
        self.logger.info("Supervisor starting all components...")
        self.running = True

        # Initialize the database manager asynchronously if it has an init method
        if hasattr(db_manager, 'initialize') and asyncio.iscoroutinefunction(db_manager.initialize):
            await db_manager.initialize()

        # Create a list of all concurrent tasks to run
        tasks = [
            asyncio.create_task(self.sentinel.run(self.market_data_queue)),
            asyncio.create_task(self.analyst.run(self.market_data_queue, self.analysis_queue)),
            asyncio.create_task(self.strategist.run(self.analysis_queue, self.signal_queue)),
        ]
        
        if self.tactician:
             tasks.append(asyncio.create_task(self.tactician.run(self.signal_queue)))

        # If in paper trading mode, also run the simulation engine
        if isinstance(self.trader, PaperTrader):
            tasks.append(asyncio.create_task(self.trader.run_simulation()))

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
            self.state_manager.save_state(self.state)
            self.logger.info("All components have been shut down and state has been saved.")
