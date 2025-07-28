import asyncio
from loguru import logger

from src.config import settings
from src.sentinel.sentinel import Sentinel
from src.analyst.analyst import Analyst
from src.strategist.strategist import Strategist
from src.tactician.tactician import Tactician
from src.paper_trader import PaperTrader
from src.utils.state_manager import StateManager
from src.database.firestore_manager import db_manager

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
        
        # Initialize the core real-time components
        self.sentinel = Sentinel()
        self.analyst = Analyst()
        self.strategist = Strategist(self.state)
        
        # Initialize the trader based on the configured mode
        if settings.trading_mode == "PAPER":
            self.trader = PaperTrader(self.state)
            self.logger.info("Paper Trader initialized.")
        else:
            # Placeholder for a future live trading implementation
            self.logger.error("Live trading mode is not fully implemented yet.")
            raise NotImplementedError("Live trading not implemented.")

        self.tactician = Tactician(self.trader)
        self.running = False
        
        # Asynchronous queues for communication between components
        self.market_data_queue = asyncio.Queue(maxsize=100)
        self.analysis_queue = asyncio.Queue(maxsize=100)
        self.signal_queue = asyncio.Queue(maxsize=50)

    async def start(self):
        """Starts all bot components and the main processing loop."""
        self.logger.info("Supervisor starting all components...")
        self.running = True

        # Initialize the database manager asynchronously
        await db_manager.initialize()

        # Create a list of all concurrent tasks to run
        tasks = [
            asyncio.create_task(self.sentinel.run(self.market_data_queue)),
            asyncio.create_task(self.analyst.run(self.market_data_queue, self.analysis_queue)),
            asyncio.create_task(self.strategist.run(self.analysis_queue, self.signal_queue)),
            asyncio.create_task(self.tactician.run(self.signal_queue)),
        ]

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
