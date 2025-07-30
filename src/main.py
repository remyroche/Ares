import asyncio
import signal
import sys
from typing import Optional

from src.utils.logger import system_logger
from src.config import settings, CONFIG
from src.utils.state_manager import StateManager
from src.database.sqlite_manager import SQLiteManager
from src.supervisor.supervisor import Supervisor
from src.utils.error_handler import (
    handle_errors,
    handle_data_processing_errors,
    handle_file_operations,
    handle_network_operations,
    handle_type_conversions,
    error_context,
    ErrorRecoveryStrategies
)

# Global variables for graceful shutdown
shutdown_event = asyncio.Event()
supervisor_instance: Optional[Supervisor] = None

@handle_errors(
    exceptions=(KeyboardInterrupt, SystemExit),
    default_return=None,
    context="signal_handler"
)
def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    system_logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    shutdown_event.set()

@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="initialize_system"
)
async def initialize_system() -> bool:
    """Initialize the trading system components."""
    try:
        system_logger.info("🚀 Initializing Ares Trading Bot...")
        
        # Initialize database manager
        db_manager = SQLiteManager()
        await db_manager.initialize()
        system_logger.info("✅ Database initialized")
        
        # Initialize state manager
        state_manager = StateManager()
        system_logger.info("✅ State manager initialized")
        
        # Initialize supervisor
        global supervisor_instance
        supervisor_instance = Supervisor(
            exchange_client=None,  # Will be initialized by supervisor
            state_manager=state_manager,
            db_manager=db_manager
        )
        system_logger.info("✅ Supervisor initialized")
        
        return True
        
    except Exception as e:
        system_logger.error(f"❌ Failed to initialize system: {e}")
        return False

@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="start_supervisor"
)
async def start_supervisor() -> bool:
    """Start the supervisor and begin trading operations."""
    try:
        if supervisor_instance is None:
            system_logger.error("Supervisor not initialized")
            return False
        
        system_logger.info("🔄 Starting supervisor...")
        await supervisor_instance.start()
        system_logger.info("✅ Supervisor started successfully")
        return True
        
    except Exception as e:
        system_logger.error(f"❌ Failed to start supervisor: {e}")
        return False

@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="shutdown_system"
)
async def shutdown_system():
    """Gracefully shutdown the trading system."""
    try:
        system_logger.info("🛑 Shutting down Ares Trading Bot...")
        
        if supervisor_instance:
            await supervisor_instance.stop()
            system_logger.info("✅ Supervisor stopped")
        
        system_logger.info("✅ Shutdown complete")
        
    except Exception as e:
        system_logger.error(f"❌ Error during shutdown: {e}")

@handle_errors(
    exceptions=(Exception,),
    default_return=1,
    context="main"
)
async def main():
    """Main entry point for the Ares trading bot."""
    try:
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        system_logger.info("🎯 Ares Trading Bot Starting...")
        system_logger.info(f"Environment: {settings.trading_environment}")
        system_logger.info(f"Symbol: {settings.trade_symbol}")
        system_logger.info(f"Timeframe: {settings.timeframe}")
        
        # Initialize system components
        if not await initialize_system():
            system_logger.error("❌ System initialization failed")
            return 1
        
        # Start supervisor
        if not await start_supervisor():
            system_logger.error("❌ Supervisor startup failed")
            return 1
        
        system_logger.info("🎉 Ares Trading Bot is now running!")
        system_logger.info("Press Ctrl+C to stop the bot gracefully")
        
        # Wait for shutdown signal
        await shutdown_event.wait()
        
        # Graceful shutdown
        await shutdown_system()
        
        return 0
        
    except Exception as e:
        system_logger.error(f"❌ Critical error in main: {e}")
        await shutdown_system()
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        system_logger.info("🛑 Bot stopped by user")
        sys.exit(0)
    except Exception as e:
        system_logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
