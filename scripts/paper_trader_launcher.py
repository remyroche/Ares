#!/usr/bin/env python3
"""
Paper Trading Launcher for Ares Trading Bot

This script launches the Ares trading bot in paper trading mode.
It initializes all necessary components and starts the supervisor.
For paper trading, it uses Binance's testnet API for actual API calls
while maintaining a simulated trading environment.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import CONFIG, settings
from src.database.sqlite_manager import SQLiteManager
from src.exchange.factory import ExchangeFactory
from src.supervisor.supervisor import Supervisor
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.signal_handler import GracefulShutdown
from src.utils.state_manager import StateManager

# Global variables for graceful shutdown
supervisor_tasks = []


@handle_errors(
    exceptions=(KeyboardInterrupt, SystemExit),
    default_return=None,
    context="signal_handler",
)
def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    system_logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    # The shutdown will be handled by the GracefulShutdown context manager


@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="initialize_system",
)
async def initialize_and_start_supervisor(symbol: str, exchange_name: str):
    """Initialize the trading system components for shadow trading with testnet APIs."""
    try:
        system_logger.info(
            f"🚀 Initializing Ares Trading Bot in PAPER mode (shadow trading) for {symbol} on {exchange_name}...",
        )

        # Force trading environment to PAPER for this execution path
        settings.trading_environment = "PAPER"
        system_logger.info(
            f"Trading environment forced to: {settings.trading_environment}",
        )

        # Initialize database manager
        db_manager = SQLiteManager()
        await db_manager.initialize()
        system_logger.info("✅ Database initialized")

        # Initialize state manager (use a specific state file for paper trading if desired)
        state_manager = StateManager(state_file=f"data/ares_paper_state_{symbol}.json")
        system_logger.info("✅ State manager initialized")

        # Initialize BinanceExchange with testnet configuration for shadow trading
        system_logger.info(
            "🔗 Initializing Binance testnet connection for shadow trading...",
        )

        # Create configuration for testnet
        exchange_config = {
            "binance_exchange": {
                "use_testnet": True,  # Use testnet APIs
                "api_key": settings.binance_testnet_api_key,
                "api_secret": settings.binance_testnet_api_secret,
                "timeout": 30,
                "max_retries": 3,
                "rate_limit_enabled": True,
                "rate_limit_requests": 1200,
                "rate_limit_window": 60,
            },
        }

        # Initialize the BinanceExchange client
        exchange_client = await setup_binance_exchange(exchange_config)
        if not exchange_client:
            system_logger.error("❌ Failed to initialize Binance testnet client")
            return False

        # Initialize the exchange client
        if not await exchange_client.initialize():
            system_logger.error("❌ Failed to initialize Binance testnet connection")
            return False

        system_logger.info(
            "✅ Binance testnet connection established for shadow trading",
        )

        # Initialize supervisor with the testnet exchange client
        supervisor = Supervisor(
            exchange_client=exchange_client,
            state_manager=state_manager,
            db_manager=db_manager,
        )
        system_logger.info("✅ Supervisor initialized with testnet exchange client")

        system_logger.info(
            f"🔄 Starting supervisor for {symbol} with shadow trading...",
        )
        await supervisor.start()
        system_logger.info(f"✅ Supervisor for {symbol} has finished its run.")

    except Exception as e:
        system_logger.error(
            f"❌ Failed to initialize or start supervisor for {symbol}: {e}",
        )


@handle_errors(exceptions=(Exception,), default_return=None, context="shutdown_system")
async def shutdown_system():
    """Gracefully shutdown the trading system."""
    global supervisor_tasks
    try:
        system_logger.info("🛑 Shutting down Ares Trading Bot...")

        for task in supervisor_tasks:
            task.cancel()
        await asyncio.gather(*supervisor_tasks, return_exceptions=True)

        system_logger.info("✅ Shutdown complete")

    except Exception as e:
        system_logger.error(f"❌ Error during shutdown: {e}")


@handle_errors(exceptions=(Exception,), default_return=1, context="main")
async def main():
    """Main entry point for the Ares trading bot (PAPER mode with shadow trading)."""
    # Ensure logging is set up using the centralized system
    ensure_logging_setup()

    parser = argparse.ArgumentParser(
        description="Launch Ares Trading Bot in Paper Trading mode with shadow trading via Binance testnet.",
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default=None,
        type=str,
        help="The trading symbol (e.g., BTCUSDT). If omitted, runs for all supported symbols.",
    )
    parser.add_argument(
        "exchange",
        nargs="?",
        default="BINANCE",
        type=str,
        help="The exchange name (e.g., BINANCE).",
    )
    args = parser.parse_args()

    # Use centralized signal handling
    with GracefulShutdown("PaperTraderLauncher") as signal_handler:
        # Add shutdown callback
        signal_handler.add_async_shutdown_callback(shutdown_system)

        try:
            # Determine which symbols to run
            if args.symbol:
                symbols_to_run = [args.symbol]
            else:
                # Get all supported tokens from config
                supported_tokens = CONFIG.get("SUPPORTED_TOKENS", {}).get(
                    args.exchange,
                    ["ETHUSDT"],
                )
                symbols_to_run = supported_tokens

            system_logger.info(
                f"📋 Running paper trading with shadow trading for {len(symbols_to_run)} symbols: {symbols_to_run}",
            )

            # Create tasks for each symbol
            global supervisor_tasks
            supervisor_tasks = []

            for symbol in symbols_to_run:
                task = asyncio.create_task(
                    initialize_and_start_supervisor(symbol, args.exchange),
                )
                supervisor_tasks.append(task)

            # Wait for all tasks to complete or for shutdown signal
            try:
                await asyncio.gather(*supervisor_tasks)
            except asyncio.CancelledError:
                system_logger.info("Tasks cancelled due to shutdown request")

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
