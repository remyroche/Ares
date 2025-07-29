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
from src.supervisor.supervisor import Supervisor as LocalSupervisor
from src.supervisor.global_portfolio_manager import GlobalPortfolioManager
from src.ares_pipeline import AresPipeline
from backtesting.main_orchestrator import main as run_soft_backtest

def main():
    """
    The main entry point for the Ares trading bot.
    Handles command-line arguments to run in different operational modes.
    """
    setup_logging()
    parser = argparse.ArgumentParser(description="Ares Trading Bot Main Launcher.")
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Operating mode')

    # --- Global Portfolio Manager Mode ---
    subparsers.add_parser('manager', help='Run the Global Portfolio Manager.')

    # --- Backtest Mode ---
    backtest_parser = subparsers.add_parser('backtest', help='Run the bot in backtesting mode.')
    backtest_parser.add_argument('type', choices=['soft', 'deep'], help="soft: Standard backtest. deep: In-depth validation.")

    # --- Trade Mode ---
    trade_parser = subparsers.add_parser('trade', help='Run an individual trading bot for a specific market.')
    trade_parser.add_argument('--symbol', type=str, required=True, help='The trading symbol (e.g., BTCUSDT).')
    trade_parser.add_argument('--exchange', type=str, required=True, help='The exchange name from config.')
    trade_parser.add_argument('type', choices=['paper', 'live'], help="paper or live trading.")
    
    args = parser.parse_args()
    logger.info(f"Starting Ares in mode: '{args.mode}'")

    if args.mode == 'manager':
        asyncio.run(run_global_manager())
    elif args.mode == 'backtest':
        # Backtesting logic remains the same
        pass
    elif args.mode == 'trade':
        is_paper = args.type == 'paper'
        asyncio.run(run_trading_bot(args.exchange, args.symbol, is_paper))

async def run_global_manager():
    """Initializes and runs the GlobalPortfolioManager."""
    logger.info("Initializing Global Portfolio Manager...")
    # The manager uses its own state file for global equity tracking
    manager_state = StateManager(state_file='global_portfolio_state.json')
    firestore_manager = FirestoreManager()
    manager = GlobalPortfolioManager(manager_state, firestore_manager)
    await manager.start()

async def run_trading_bot(exchange_name: str, symbol: str, is_paper_trade: bool):
    """Initializes and runs a single trading bot instance."""
    logger.info(f"Initializing trading bot for {symbol} on {exchange_name} (Paper: {is_paper_trade}).")
    
    # Find the specific exchange config
    exchange_config = next((item for item in settings.get("exchanges", []) if item["name"] == exchange_name), None)
    if not exchange_config:
        logger.critical(f"Exchange '{exchange_name}' not found in config. Exiting.")
        return

    # Each bot instance gets its own state file
    state_manager = StateManager(state_file=f"state_{exchange_name}_{symbol}.json")
    firestore_manager = FirestoreManager()
    exchange_client = BinanceExchange(
        api_key=exchange_config.get('api_key'),
        api_secret=exchange_config.get('api_secret'),
        paper_trade=is_paper_trade
    )
    
    # Initialize components for this specific bot
    local_supervisor = LocalSupervisor(exchange_client, state_manager, firestore_manager)
    ares_pipeline = AresPipeline(exchange_client, state_manager, firestore_manager, local_supervisor)

    # Run the bot's tasks concurrently
    supervisor_task = asyncio.create_task(local_supervisor.start())
    pipeline_task = asyncio.create_task(ares_pipeline.start())
    logger.info(f"Bot for {symbol} is now running.")
    await asyncio.gather(supervisor_task, pipeline_task)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Ares shutdown requested by user.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"A critical error occurred in the main launcher: {e}", exc_info=True)
        sys.exit(1)

