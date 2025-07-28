import asyncio
from src.utils.logger import logger
from src.config import settings
from src.supervisor.main import Supervisor # Import Supervisor only
from src.exchange.binance import exchange # Import the global exchange instance

async def main():
    """
    The main entry point and orchestrator for the Ares Trading Bot.
    Initializes the Supervisor and runs it.
    """
    logger.info("--- Starting Ares Trading Bot ---")

    # The Supervisor is now responsible for initializing all core modules
    # and managing the trading environment (PaperTrader vs Live Exchange).
    supervisor = Supervisor(exchange_client=exchange) 
    logger.info("Supervisor initialized.")

    # 3. Run the Supervisor's main loop
    try:
        await supervisor.start()
    except asyncio.CancelledError:
        logger.info("Main task cancelled. Shutting down all modules...")
    finally:
        # Supervisor's finally block should handle its own cleanup.
        # Ensure exchange is closed, and state is saved.
        # The supervisor.start() method's finally block should handle this.
        logger.info("--- Ares Trading Bot has been shut down gracefully ---")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (KeyboardInterrupt).")

