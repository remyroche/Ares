import argparse
import asyncio
import sys
import os

# Ensure the src directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.logger import setup_logging, logger
from src.config import settings, CONFIG
from src.exchange.binance import BinanceExchange
from src.utils.state_manager import StateManager
from src.database.firestore_manager import FirestoreManager
from src.supervisor.supervisor import Supervisor
from src.ares_pipeline import AresPipeline
from backtesting.main_orchestrator import main as run_soft_backtest

def main():
    """
    The main entry point for the Ares trading bot.
    Handles command-line arguments to run in different operational modes.
    """
    # 1. Setup logging as the very first action.
    setup_logging()

    # 2. Setup the command-line argument parser.
    parser = argparse.ArgumentParser(
        description="Ares Trading Bot Main Launcher.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='The trading symbol to use (e.g., BTCUSDT).')
    parser.add_argument('--exchange', type=str, default='binance', help='The exchange to connect to (default: binance).')

    subparsers = parser.add_subparsers(dest='mode', required=True, help='Operating mode')

    # --- Backtest Mode ---
    backtest_parser = subparsers.add_parser('backtest', help='Run the bot in backtesting mode.')
    backtest_parser.add_argument(
        'type', 
        choices=['soft', 'deep'], 
        help="soft: Runs a standard backtest using the backtesting suite.\n"
             "deep: Runs an in-depth analysis using the Supervisor's walk-forward and Monte Carlo validation."
    )

    # --- Trade Mode ---
    trade_parser = subparsers.add_parser('trade', help='Run the bot in trading mode.')
    trade_parser.add_argument(
        'type', 
        choices=['paper', 'live'], 
        help="paper: Run in paper trading mode with no real funds.\n"
             "live: Run in live trading mode with real funds."
    )

    args = parser.parse_args()

    # Override config with any provided command-line arguments
    CONFIG['trading']['symbol'] = args.symbol
    settings['exchange']['name'] = args.exchange
    
    logger.info(f"Starting Ares in mode: '{args.mode}', type: '{args.type}'")
    logger.info(f"Configuration: Exchange='{args.exchange}', Symbol='{args.symbol}'")

    # 3. Execute the selected mode.
    if args.mode == 'backtest':
        if args.type == 'soft':
            logger.info("--- Running Soft Backtest ---")
            run_soft_backtest()
            logger.info("--- Soft Backtest Complete ---")
        elif args.type == 'deep':
            logger.info("--- Running Deep Backtest (Walk-Forward & Monte Carlo) ---")
            asyncio.run(run_deep_backtest())
            logger.info("--- Deep Backtest Complete ---")

    elif args.mode == 'trade':
        is_paper = args.type == 'paper'
        settings['paper_trade'] = is_paper
        logger.info(f"Paper Trading Mode: {'Enabled' if is_paper else 'Disabled'}")
        asyncio.run(run_trading_bot(is_paper))


async def run_deep_backtest():
    """Initializes components and runs the Supervisor's validation suite."""
    logger.info("Initializing components for deep backtest...")
    state_manager = StateManager()
    firestore_manager = FirestoreManager()
    # Exchange client is initialized in paper mode for deep backtesting.
    exchange_client = BinanceExchange(
        api_key=settings.get('api_key'),
        api_secret=settings.get('api_secret'),
        paper_trade=True
    )
    
    supervisor = Supervisor(exchange_client, state_manager, firestore_manager)
    
    logger.info("Supervisor initialized. Starting deep analysis...")
    await supervisor.run_walk_forward_analysis()
    await supervisor.run_monte_carlo_simulation()


async def run_trading_bot(is_paper_trade: bool):
    """Initializes and runs the full trading system (Supervisor + AresPipeline)."""
    logger.info("Initializing system components for trading...")
    
    # Initialize core components
    firestore_manager = FirestoreManager()
    state_manager = StateManager()
    exchange_client = BinanceExchange(
        api_key=settings.get('api_key'),
        api_secret=settings.get('api_secret'),
        paper_trade=is_paper_trade
    )
    
    # Initialize main operational components
    supervisor = Supervisor(exchange_client, state_manager, firestore_manager)
    ares_pipeline = AresPipeline(exchange_client, state_manager, firestore_manager, supervisor)

    # Create and run asyncio tasks for the supervisor and the trading pipeline
    supervisor_task = asyncio.create_task(supervisor.start())
    pipeline_task = asyncio.create_task(ares_pipeline.start())

    logger.info("Ares is now running. Supervisor and Trading Pipeline are active.")
    
    # Wait for tasks to complete (they will run indefinitely until the bot is stopped)
    await asyncio.gather(supervisor_task, pipeline_task)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Ares shutdown requested by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"A critical error occurred in the main launcher: {e}", exc_info=True)
        sys.exit(1)
