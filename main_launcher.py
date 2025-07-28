import asyncio
from loguru import logger
import signal

from src.config import settings
from src.supervisor.main import Supervisor
from exchange.binance import exchange

class MainLauncher:
    """
    Main entry point for the Ares Trading Bot.
    Initializes and runs the Supervisor in an asyncio event loop.
    """
    def __init__(self):
        self.supervisor = Supervisor()
        self.main_task = None

    async def run(self):
        """Initializes and starts the supervisor's main loop."""
        logger.info("Initializing Ares Trading Bot...")
        logger.info(f"Trading Mode: {settings.trading_mode}")
        
        # The main task that runs the bot's logic
        self.main_task = asyncio.create_task(self.supervisor.start())
        
        await self.main_task

    def shutdown(self, signame):
        """Gracefully shuts down the application."""
        logger.warning(f"Received shutdown signal: {signame}. Shutting down...")
        if self.main_task:
            self.main_task.cancel()
        
        # Clean up other resources like the exchange session
        asyncio.create_task(exchange.close())
        
        # Gather all remaining tasks and cancel them
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        
        logger.info("Shutdown complete.")

async def main():
    launcher = MainLauncher()
    
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for signame in ('SIGINT', 'SIGTERM'):
        loop.add_signal_handler(
            getattr(signal, signame),
            lambda signame=signame: launcher.shutdown(signame)
        )

    try:
        await launcher.run()
    except asyncio.CancelledError:
        logger.info("Main launcher task cancelled.")
    except Exception as e:
        logger.critical(f"A critical error occurred in the main launcher: {e}")

if __name__ == "__main__":
    # Configure logger
    logger.add(
        "logs/ares_bot_{time}.log",
        rotation="1 day",
        retention="7 days",
        level=settings.log_level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application stopped by user.")
