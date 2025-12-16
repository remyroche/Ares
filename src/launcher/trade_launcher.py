"""
Trade Launcher

Command-line interface for launching trading in PAPER or TRADE mode.
"""

import asyncio
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from exchanges.exchange_dispatcher import ExchangeDispatcher, ExchangeConfig, ExchangeType, TradingMode as ExchangeTradingMode
from src.simulator import SimulatorConfig, PaperTradingSimulator
from live_trading.trading_engine import TradingEngine
from live_trading.config import TradingConfig, TradingMode as LiveTradingMode
from src.utils.logger import system_logger
from src.launcher.trading_launcher import get_parameter_manager
from src.utils.api_key_loader import get_api_keys
from src.utils.tprint import tprint
from src.utils.trading_verification.delta_checker import run_delta_check_cli


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = system_logger.getChild("TradeLauncher")


async def main():
    """Main entry point for trade launcher."""
    tprint("🚀 trade_launcher.main() - Starting trade launcher", "INFO")

    parser = argparse.ArgumentParser(
        description="Launch trading system in PAPER or TRADE mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paper trading on Binance with defaults (ETHUSDT, long, 1000 USDT)
  python src/launcher/trade_launcher.py --mode paper --exchange binance
  
  # Paper trading on Binance with BTC, both directions
  python src/launcher/trade_launcher.py --mode paper --direction both --exchange binance --asset BTCUSDT
  
  # Live trading on OKX, long only (default)
  python src/launcher/trade_launcher.py --mode trade --direction long --exchange okx --asset ETHUSDT
  
  # Custom initial balance for paper trading
  python src/launcher/trade_launcher.py --mode paper --initial-balance 50000 --exchange binance --asset BTCUSDT
        """
    )
    
    # Core arguments
    parser.add_argument('--mode', required=True, choices=['paper', 'trade'],
                       help='Trading mode: paper (simulated) or trade (live)')
    parser.add_argument('--direction', required=False, default='long',
                       choices=['long', 'short', 'both'],
                       help='Trading direction: long, short, or both (default: long)')
    parser.add_argument('--exchange', required=True,
                       choices=['binance', 'okx', 'gateio', 'mexc', 'phemex'],
                       help='Exchange to use')
    parser.add_argument('--asset', required=False, default='ETHUSDT',
                       help='Trading symbol (e.g., BTCUSDT, ETHUSDT) (default: ETHUSDT)')
    
    # Paper trading specific
    parser.add_argument('--initial-balance', type=float, default=1000.0,
                       help='Initial balance for paper trading (default: 1000 USDT)')
    parser.add_argument('--state-file', default='simulator_state.db',
                       help='SQLite database file for simulator state (default: simulator_state.db)')
    parser.add_argument('--reset-state', action='store_true',
                       help='Reset/clear previous simulator state')
    
    # Logging and config
    parser.add_argument('--log-level', default=None,
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: DEBUG for paper mode, INFO for trade mode)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate configuration without starting trading')
    parser.add_argument('--delta-check', action='store_true',
                       help='Run Delta Check (Backtest vs Live parity) before starting')
    
    args = parser.parse_args()

    tprint(f"📋 trade_launcher.main() - Parsed arguments: mode={args.mode}, direction={args.direction}, exchange={args.exchange}, asset={args.asset}", "INFO")
    tprint(f"📋 trade_launcher.main() - Additional args: initial_balance={args.initial_balance}, state_file={args.state_file}, reset_state={args.reset_state}, dry_run={args.dry_run}", "INFO")

    # Set default log level based on mode if not explicitly provided
    if args.log_level is None:
        if args.mode == 'paper':
            args.log_level = 'DEBUG'
        else:
            args.log_level = 'INFO'
        tprint(f"📊 trade_launcher.main() - Set default log level: {args.log_level} (based on mode: {args.mode})", "INFO")

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    tprint(f"📊 trade_launcher.main() - Logging level set to: {args.log_level}", "INFO")
    
    logger.info("=" * 80)
    logger.info("Trade Launcher Starting")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Direction: {args.direction}")
    logger.info(f"Exchange: {args.exchange}")
    logger.info(f"Asset: {args.asset}")
    
    try:
        tprint("🔧 trade_launcher.main() - Entering main try block", "INFO")

        # Determine trading mode
        exchange_mode = ExchangeTradingMode.PAPER if args.mode == "paper" else ExchangeTradingMode.TRADE
        live_trading_mode = LiveTradingMode.PAPER if args.mode == "paper" else LiveTradingMode.LIVE
        tprint(f"🎯 trade_launcher.main() - Trading mode determined: exchange_mode={exchange_mode}, live_trading_mode={live_trading_mode}", "INFO")

        # Set TRADE/LIVE flags as environment variables for other components
        use_live = (args.mode == "trade")
        os.environ["TRADE"] = "1" if use_live else "0"
        os.environ["LIVE"] = "1" if use_live else "0"
        tprint(f"🔑 trade_launcher.main() - Environment variables set: TRADE={os.environ['TRADE']}, LIVE={os.environ['LIVE']}", "INFO")
        logger.info(f"Using {'LIVE' if use_live else 'TESTNET'} API keys")
        
        # Load API keys from secret/api_keys.json
        tprint(f"🔑 trade_launcher.main() - Loading API keys for exchange={args.exchange}, use_live={use_live}", "INFO")
        api_keys = get_api_keys(args.exchange, use_live=use_live)
        api_key = api_keys.get("api_key") or ""
        api_secret = api_keys.get("api_secret") or ""
        api_password = api_keys.get("password")
        tprint(f"🔑 trade_launcher.main() - API keys loaded: api_key={'present' if api_key else 'missing'}, api_secret={'present' if api_secret else 'missing'}, api_password={'present' if api_password else 'not set'}", "INFO")

        # Validate credentials for trade mode
        if exchange_mode == ExchangeTradingMode.TRADE:
            tprint("🔍 trade_launcher.main() - Validating credentials for TRADE mode", "INFO")
            if not api_key or not api_secret:
                tprint(f"❌ trade_launcher.main() - API credentials missing for TRADE mode", "ERROR")
                logger.error(
                    f"API key and secret are required for TRADE mode. "
                    f"Please ensure secret/api_keys.json contains {args.exchange}.live.api_key and api_secret"
                )
                tprint("🔚 trade_launcher.main() - Exiting with return code: 1", "ERROR")
                return 1
            tprint("✅ trade_launcher.main() - API credentials validated for TRADE mode", "SUCCESS")
        
        # For paper mode, use testnet keys if available, otherwise use dummy keys
        if exchange_mode == ExchangeTradingMode.PAPER:
            tprint("📄 trade_launcher.main() - Handling PAPER mode credentials", "INFO")
            if not api_key or not api_secret:
                tprint(f"⚠️ trade_launcher.main() - Testnet keys not found, using dummy keys for paper trading", "WARNING")
                logger.warning(
                    f"Testnet API keys not found for {args.exchange}. "
                    f"Using dummy keys for paper trading simulation. "
                    f"Consider adding testnet keys to secret/api_keys.json for better data access."
                )
                api_key = "paper_key"
                api_secret = "paper_secret"
                tprint(f"📄 trade_launcher.main() - Dummy keys set: api_key={api_key}, api_secret={api_secret}", "INFO")
            else:
                tprint(f"✅ trade_launcher.main() - Using testnet keys for paper trading", "SUCCESS")
        
        # Initialize exchange config
        tprint(f"⚙️ trade_launcher.main() - Creating exchange config: exchange_type={args.exchange.upper()}, use_testnet={exchange_mode == ExchangeTradingMode.PAPER}, trade_symbol={args.asset}, mode={exchange_mode}", "INFO")
        exchange_config = ExchangeConfig(
            exchange_type=ExchangeType[args.exchange.upper()],
            api_key=api_key,
            api_secret=api_secret,
            password=api_password,
            use_testnet=(exchange_mode == ExchangeTradingMode.PAPER),  # Use testnet for paper trading
            trade_symbol=args.asset,
            mode=exchange_mode
        )
        tprint("✅ trade_launcher.main() - Exchange config created successfully", "SUCCESS")

        # Create exchange dispatcher
        tprint("🔌 trade_launcher.main() - Creating exchange dispatcher", "INFO")
        dispatcher = ExchangeDispatcher(exchange_config)
        tprint("✅ trade_launcher.main() - Exchange dispatcher created", "SUCCESS")
        
        # Initialize simulator if in paper mode
        simulator = None
        if exchange_mode == ExchangeTradingMode.PAPER:
            tprint("🎮 trade_launcher.main() - Initializing paper trading simulator", "INFO")
            logger.info("Initializing paper trading simulator...")

            # Reset state if requested
            if args.reset_state and Path(args.state_file).exists():
                tprint(f"🗑️ trade_launcher.main() - Resetting simulator state: removing {args.state_file}", "INFO")
                logger.info(f"Removing existing state file: {args.state_file}")
                Path(args.state_file).unlink()
                tprint(f"✅ trade_launcher.main() - State file removed successfully", "SUCCESS")

            # Create simulator config
            tprint("⚙️ trade_launcher.main() - Creating simulator config", "INFO")
            simulator_config = SimulatorConfig()
            tprint("✅ trade_launcher.main() - Simulator config created", "SUCCESS")

            # Create simulator
            tprint(f"🎮 trade_launcher.main() - Creating simulator: exchange={args.exchange}, initial_balance={args.initial_balance}, direction_constraint={args.direction}, db_path={args.state_file}", "INFO")
            simulator = PaperTradingSimulator(
                config=simulator_config,
                exchange=args.exchange,
                initial_balance=args.initial_balance,
                direction_constraint=args.direction,
                db_path=args.state_file
            )
            tprint("✅ trade_launcher.main() - Simulator created successfully", "SUCCESS")

            # Set simulator callback in dispatcher
            tprint("🔗 trade_launcher.main() - Setting simulator callback in dispatcher", "INFO")
            dispatcher.set_simulator_callback(simulator.simulate_order)
            tprint("✅ trade_launcher.main() - Simulator callback set", "SUCCESS")

            logger.info(f"✅ Simulator initialized with balance: {args.initial_balance} USDT")
            logger.info(f"✅ Direction constraint: {args.direction}")
            tprint(f"💰 trade_launcher.main() - Simulator initialized: balance={args.initial_balance} USDT, direction={args.direction}", "SUCCESS")
        else:
            tprint("⏭️ trade_launcher.main() - Skipping simulator initialization (TRADE mode)", "INFO")
        
        # Initialize exchange dispatcher
        tprint("🔌 trade_launcher.main() - Initializing exchange dispatcher", "INFO")
        logger.info("Initializing exchange dispatcher...")
        success = await dispatcher.initialize()
        tprint(f"🔌 trade_launcher.main() - Dispatcher initialization result: success={success}", "INFO")

        if not success:
            tprint("❌ trade_launcher.main() - Failed to initialize exchange dispatcher", "ERROR")
            logger.error("Failed to initialize exchange dispatcher")
            tprint("🔚 trade_launcher.main() - Exiting with return code: 1", "ERROR")
            return 1

        tprint("✅ trade_launcher.main() - Exchange dispatcher initialized successfully", "SUCCESS")
        logger.info("✅ Exchange dispatcher initialized")
        
        # Dry run check
        if args.dry_run:
            tprint("🧪 trade_launcher.main() - Dry run mode: configuration validated successfully", "SUCCESS")
            logger.info("Dry run completed successfully - configuration validated")
            logger.info("Use without --dry-run to start actual trading")
            tprint("🔚 trade_launcher.main() - Exiting dry run with return code: 0", "SUCCESS")
            return 0

        # Delta Check (Pre-flight)
        if args.delta_check:
            tprint("🕵️ trade_launcher.main() - Running Delta Check...", "INFO")
            delta_exit_code = await run_delta_check_cli(args.asset, "15m", args.exchange)
            if delta_exit_code != 0:
                tprint("❌ Delta Check FAILED. Aborting trading start.", "ERROR")
                return delta_exit_code
            tprint("✅ Delta Check PASSED. Proceeding.", "SUCCESS")
        
        # Create trading config
        tprint(f"⚙️ trade_launcher.main() - Creating trading config: mode={live_trading_mode}, exchange={args.exchange}, symbols={[args.asset]}, direction={args.direction}", "INFO")
        trading_config = TradingConfig(
            mode=live_trading_mode,
            exchange_name=args.exchange,
            symbols=[args.asset],
            direction=args.direction
        )
        tprint("✅ trade_launcher.main() - Trading config created successfully", "SUCCESS")

        # Create and start trading engine
        tprint("🚀 trade_launcher.main() - Creating trading engine", "INFO")
        logger.info("Starting trading engine...")

        trading_engine = TradingEngine(trading_config, dispatcher)
        tprint("✅ trade_launcher.main() - Trading engine created", "SUCCESS")
        
        # Inject simulator into order manager if in paper mode
        if simulator:
            tprint("🔗 trade_launcher.main() - Injecting simulator into order manager", "INFO")
            trading_engine.order_manager.simulator = simulator
            tprint("✅ trade_launcher.main() - Simulator injected into order manager", "SUCCESS")
            logger.info("✅ Simulator injected into order manager")
        else:
            tprint("⏭️ trade_launcher.main() - Skipping simulator injection (TRADE mode)", "INFO")

        # Initialize parameter manager for hot swapping
        tprint("🔧 trade_launcher.main() - Initializing parameter manager for hot swapping", "INFO")
        parameter_manager = get_parameter_manager(trading_engine)
        parameter_manager.set_trading_engine(trading_engine)
        tprint("✅ trade_launcher.main() - Parameter manager initialized - hot swap enabled", "SUCCESS")
        logger.info("✅ Trading parameter manager initialized - hot swap enabled")
        
        # Start the trading engine
        tprint("🚀 trade_launcher.main() - Starting trading engine", "INFO")
        await trading_engine.start()
        tprint("✅ trade_launcher.main() - Trading engine started successfully", "SUCCESS")

        # Start monitoring after engine is running
        tprint("👀 trade_launcher.main() - Starting parameter manager monitoring", "INFO")
        parameter_manager._start_monitoring()
        tprint("✅ trade_launcher.main() - Parameter manager monitoring started", "SUCCESS")

        logger.info("=" * 80)
        logger.info("Trading system is running!")
        logger.info("=" * 80)
        tprint("=" * 80, "SUCCESS")
        tprint("🎉 trade_launcher.main() - Trading system is running!", "SUCCESS")
        tprint("=" * 80, "SUCCESS")

        # Display status
        tprint("📊 trade_launcher.main() - Retrieving trading status", "INFO")
        status = await trading_engine.get_trading_status()
        tprint(f"📊 trade_launcher.main() - Trading status: mode={status['mode']}, symbols={status['symbols']}, active={status['trading_active']}", "INFO")
        logger.info(f"Mode: {status['mode']}")
        logger.info(f"Symbols: {status['symbols']}")
        logger.info(f"Status: {'ACTIVE' if status['trading_active'] else 'INACTIVE'}")
        
        if simulator:
            tprint("💰 trade_launcher.main() - Retrieving simulator performance metrics", "INFO")
            metrics = simulator.get_performance_metrics()
            tprint(f"💰 trade_launcher.main() - Simulator metrics: initial_balance={metrics['initial_balance']}, current_balance={metrics['current_balance']}, net_pnl={metrics['net_pnl']:.2f}, net_pnl_pct={metrics['net_pnl_pct']:.2%}", "INFO")
            logger.info(f"Initial Balance: {metrics['initial_balance']} USDT")
            logger.info(f"Current Balance: {metrics['current_balance']} USDT")
            logger.info(f"Net PnL: {metrics['net_pnl']:.2f} USDT ({metrics['net_pnl_pct']:.2%})")

        tprint("⏳ trade_launcher.main() - Entering main monitoring loop (Press Ctrl+C to stop)", "INFO")
        logger.info("Press Ctrl+C to stop...")
        
        # Run until interrupted
        try:
            tprint("🔄 trade_launcher.main() - Starting monitoring loop inner try block", "INFO")
            last_report_date = None

            while True:
                await asyncio.sleep(60)  # Update every minute
                tprint("🔄 trade_launcher.main() - Monitoring loop iteration", "DEBUG")

                # Display periodic status
                if simulator:
                    metrics = simulator.get_performance_metrics()
                    tprint(f"📊 trade_launcher.main() - Periodic status update: balance={metrics['current_balance']:.2f}, pnl={metrics['net_pnl']:.2f} ({metrics['net_pnl_pct']:.2%}), trades={metrics['total_trades']}", "INFO")
                    logger.info(
                        f"Balance: {metrics['current_balance']:.2f} USDT | "
                        f"PnL: {metrics['net_pnl']:.2f} ({metrics['net_pnl_pct']:.2%}) | "
                        f"Trades: {metrics['total_trades']}"
                    )

                # Generate daily report at end of day (or once per day)
                from datetime import date
                current_date = date.today()
                if last_report_date != current_date:
                    tprint(f"📅 trade_launcher.main() - New day detected: {current_date}, generating daily report", "INFO")
                    logger.info("Generating daily report...")

                    # Generate report for paper trading
                    if simulator:
                        tprint(f"📄 trade_launcher.main() - Generating daily report for paper trading: asset={args.asset}, date={current_date}", "INFO")
                        await simulator.generate_daily_report(args.asset, current_date)
                    else:
                        # Generate report for live trading
                        tprint(f"📄 trade_launcher.main() - Generating daily report for live trading: asset={args.asset}, date={current_date}", "INFO")
                        await trading_engine.order_manager.generate_daily_report(args.asset, current_date)

                    last_report_date = current_date
                    tprint("✅ trade_launcher.main() - Daily report generated successfully", "SUCCESS")
                    logger.info("✅ Daily report generated")

        except KeyboardInterrupt:
            tprint("⏹️ trade_launcher.main() - KeyboardInterrupt received, shutting down", "WARNING")
            logger.info("\nReceived stop signal, shutting down...")
        
        finally:
            tprint("🧹 trade_launcher.main() - Entering finally block for cleanup", "INFO")
            # Stop parameter manager monitoring
            try:
                tprint("🛑 trade_launcher.main() - Stopping parameter manager monitoring", "INFO")
                parameter_manager = get_parameter_manager()
                parameter_manager.stop_monitoring()
                tprint("✅ trade_launcher.main() - Parameter manager stopped successfully", "SUCCESS")
            except Exception as e:
                tprint(f"⚠️ trade_launcher.main() - Error stopping parameter manager: {e}", "WARNING")
                logger.warning(f"Error stopping parameter manager: {e}")

            # Stop the trading engine
            tprint("🛑 trade_launcher.main() - Stopping trading engine", "INFO")
            logger.info("Stopping trading engine...")
            await trading_engine.stop()
            tprint("✅ trade_launcher.main() - Trading engine stopped successfully", "SUCCESS")

            logger.info("✅ Trading system stopped successfully")
            tprint("✅ trade_launcher.main() - Trading system stopped successfully", "SUCCESS")

        tprint("🔚 trade_launcher.main() - Exiting with return code: 0", "SUCCESS")
        return 0
        
    except Exception as e:
        tprint(f"❌ trade_launcher.main() - Exception caught in main try block: {type(e).__name__}: {e}", "ERROR")
        logger.exception(f"Error during trading: {e}")
        tprint("🔚 trade_launcher.main() - Exiting with return code: 1", "ERROR")
        return 1


if __name__ == "__main__":
    tprint("🎬 trade_launcher - Starting script execution", "INFO")
    exit_code = asyncio.run(main())
    tprint(f"🎬 trade_launcher - Script execution completed with exit code: {exit_code}", "INFO")
    sys.exit(exit_code)
