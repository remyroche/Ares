import asyncio
from src.utils.logger import logger, setup_logging # Import setup_logging
from src.config import settings, CONFIG
from src.supervisor.main import Supervisor # Import Supervisor only
from src.exchange.binance import exchange # Import the global exchange instance
from src.utils.error_handler import register_global_exception_handler, unregister_global_exception_handler # Import handler functions

# Import both database managers
from src.database.firestore_manager import db_manager as firestore_db_manager
from src.database.sqlite_manager import sqlite_manager as local_db_manager


async def main():
    """
    The main entry point and orchestrator for the Ares Trading Bot.
    Initializes the Supervisor and runs it.
    """
    logger.info("--- Starting Ares Trading Bot ---")

    # Register the global exception handler as early as possible
    register_global_exception_handler()

    # --- Select and Initialize the Database Manager ---
    if CONFIG["DATABASE_TYPE"] == "firestore":
        db_manager_instance = firestore_db_manager
        logger.info("Using Firestore as the database manager.")
    elif CONFIG["DATABASE_TYPE"] == "sqlite":
        db_manager_instance = local_db_manager
        logger.info(f"Using SQLite as the database manager: {CONFIG['SQLITE_DB_PATH']}.")
    else:
        logger.critical(f"Invalid DATABASE_TYPE configured: {CONFIG['DATABASE_TYPE']}. Exiting.")
        return # Exit if database type is invalid

    await db_manager_instance.initialize()
    # --- End Database Manager Selection ---


    # The Supervisor is now responsible for initializing all core modules
    # and managing the trading environment (PaperTrader vs Live Exchange).
    # Pass the selected db_manager_instance
    supervisor = Supervisor(exchange_client=exchange, firestore_manager=db_manager_instance) 
    logger.info("Supervisor initialized.")

    # 3. Run the Supervisor's main loop
    try:
        await supervisor.start()
    except asyncio.CancelledError:
        logger.info("Main task cancelled. Shutting down all modules...")
    finally:
        # Ensure the global exception handler is unregistered on graceful shutdown
        unregister_global_exception_handler()
        # Ensure the database connection is closed
        if db_manager_instance:
            await db_manager_instance.close()
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
