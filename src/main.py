import asyncio
from src.utils.logger import logger, setup_logging # Import setup_logging
from src.config import settings
from src.supervisor.main import Supervisor # Import Supervisor only
from src.exchange.binance import exchange # Import the global exchange instance
from src.utils.error_handler import register_global_exception_handler, unregister_global_exception_handler # Import handler functions

async def main():
    """
    The main entry point and orchestrator for the Ares Trading Bot.
    Initializes the Supervisor and runs it.
    """
    logger.info("--- Starting Ares Trading Bot ---")

    # Register the global exception handler as early as possible
    register_global_exception_handler()

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
        # Ensure the global exception handler is unregistered on graceful shutdown
        unregister_global_exception_handler()
        logger.info("--- Ares Trading Bot has been shut down gracefully ---")


if __name__ == "__main__":
    # Setup logging first to ensure all subsequent logs are formatted correctly
    setup_logging() 
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (KeyboardInterrupt).")
    except Exception as e:
        # Unhandled exceptions will be caught by sys.excepthook, but this ensures a final log
        logger.critical(f"A critical unhandled error occurred in main execution: {e}", exc_info=True)
