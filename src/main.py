import asyncio
from src.utils.logger import logger
from src.config import settings
from src.utils.state_manager import StateManager
from src.exchange.binance import exchange # Using the global exchange instance
from src.sentinel.sentinel import Sentinel
from src.analyst.analyst import Analyst
from src.strategist.strategist import Strategist
from src.tactician.tactician import Tactician
from src.supervisor.supervisor import Supervisor

async def main():
    """
    The main entry point and orchestrator for the Ares Trading Bot.
    Initializes and runs all modules concurrently.
    """
    logger.info("--- Starting Ares Trading Bot ---")

    # 1. Initialize shared components
    if not exchange:
        logger.critical("Exchange client failed to initialize. Cannot start bot.")
        return
        
    state_manager = StateManager('ares_state.json')
    logger.info("State Manager initialized.")

    # 2. Initialize all core modules
    sentinel = Sentinel(exchange, state_manager)
    analyst = Analyst(exchange, state_manager)
    strategist = Strategist(state_manager)
    tactician = Tactician(exchange, state_manager)
    supervisor = Supervisor(exchange, state_manager)
    logger.info("All core modules initialized.")

    # 3. Create concurrent tasks for each module
    tasks = [
        asyncio.create_task(sentinel.start(), name="Sentinel"),
        asyncio.create_task(analyst.start(), name="Analyst"),
        asyncio.create_task(strategist.start(), name="Strategist"),
        asyncio.create_task(tactician.start(), name="Tactician"),
        asyncio.create_task(supervisor.start(), name="Supervisor")
    ]
    
    logger.info("All module tasks created. Bot is now live.")

    # 4. Run all tasks concurrently and manage shutdown
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Main task cancelled. Shutting down all modules...")
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await exchange.close()
        state_manager.save_state()
        logger.info("--- Ares Trading Bot has been shut down gracefully ---")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (KeyboardInterrupt).")
