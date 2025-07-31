#!/usr/bin/env python3
"""
Script to launch the Ares Trading Bot in Paper Trading mode.

This script allows you to specify the trading symbol and exchange,
and forces the TRADING_ENVIRONMENT to "PAPER" for this execution.

Usage:
    python scripts/paper_trader_launcher.py <SYMBOL> <EXCHANGE>
    Example: python scripts/paper_trader_launcher.py BTCUSDT BINANCE
"""

import asyncio
import signal
import sys
import argparse
from typing import Optional
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger, setup_logging
from src.config import settings, CONFIG
from src.utils.state_manager import StateManager
from src.database.sqlite_manager import SQLiteManager
from src.supervisor.main import Supervisor
from src.utils.error_handler import (
    handle_errors,
)

# Global variables for graceful shutdown
shutdown_event = asyncio.Event()
supervisor_tasks: list[asyncio.Task] = []


@handle_errors(
    exceptions=(KeyboardInterrupt, SystemExit),
    default_return=None,
    context="signal_handler",
)
def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    system_logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    shutdown_event.set()


@handle_errors(
    exceptions=(Exception,), default_return=False, context="initialize_system"
)
async def initialize_and_start_supervisor(symbol: str, exchange_name: str):
    """Initialize the trading system components, forcing PAPER mode."""
    try:
        system_logger.info(
            f"🚀 Initializing Ares Trading Bot in PAPER mode for {symbol} on {exchange_name}..."
        )

        # Force trading environment to PAPER for this execution path
        settings.trading_environment = "PAPER"
        system_logger.info(
            f"Trading environment forced to: {settings.trading_environment}"
        )

        # Initialize database manager
        db_manager = SQLiteManager()
        await db_manager.initialize()
        system_logger.info("✅ Database initialized")

        # Initialize state manager (use a specific state file for paper trading if desired)
        state_manager = StateManager(state_file=f"data/ares_paper_state_{symbol}.json")
        system_logger.info("✅ State manager initialized")

        # Initialize supervisor
        supervisor = Supervisor(
            symbol=symbol,
            exchange_name=exchange_name,
            exchange_client=None,  # Will be initialized by supervisor (PaperTrader)
            state_manager=state_manager,
            db_manager=db_manager,
        )
        system_logger.info("✅ Supervisor initialized")

        system_logger.info(f"🔄 Starting supervisor for {symbol}...")
        await supervisor.start()
        system_logger.info(f"✅ Supervisor for {symbol} has finished its run.")

    except Exception as e:
        system_logger.error(f"❌ Failed to initialize or start supervisor for {symbol}: {e}")


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
    """Main entry point for the Ares trading bot (PAPER mode)."""
    setup_logging()  # Ensure logging is set up for this script

    parser = argparse.ArgumentParser(
        description="Launch Ares Trading Bot in Paper Trading mode."
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

    try:
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        symbols_to_run = []
        if args.symbol:
            symbols_to_run.append((args.symbol.upper(), args.exchange.upper()))
        else:
            system_logger.info("No symbol specified, running for all supported tokens.")
            supported_tokens = CONFIG.get("SUPPORTED_TOKENS", {})
            for exchange, tokens in supported_tokens.items():
                for token in tokens:
                    symbols_to_run.append((token, exchange))

        if not symbols_to_run:
            system_logger.error("No symbols to run. Check your config or arguments.")
            return 1

        global supervisor_tasks
        for symbol, exchange_name in symbols_to_run:
            task = asyncio.create_task(initialize_and_start_supervisor(symbol, exchange_name))
            supervisor_tasks.append(task)

        system_logger.info(f"🎉 {len(supervisor_tasks)} Ares Paper Trading Bot instance(s) are now running!")
        system_logger.info("Press Ctrl+C to stop the bot gracefully")

        # Wait for shutdown signal
        await shutdown_event.wait()

        # Graceful shutdown
        system_logger.info("Shutdown event received, cancelling tasks...")
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
