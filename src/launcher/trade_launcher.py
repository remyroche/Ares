"""
Trade Launcher

Command-line interface for launching trading in PAPER or TRADE mode.
"""

import asyncio
import argparse
import logging
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


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = system_logger.getChild("TradeLauncher")


async def main():
    """Main entry point for trade launcher."""
    
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
    
    # API credentials (optional for paper mode, required for trade mode)
    parser.add_argument('--api-key', help='Exchange API key')
    parser.add_argument('--api-secret', help='Exchange API secret')
    parser.add_argument('--api-password', help='Exchange API password (if required)')
    
    # Paper trading specific
    parser.add_argument('--initial-balance', type=float, default=1000.0,
                       help='Initial balance for paper trading (default: 1000 USDT)')
    parser.add_argument('--state-file', default='simulator_state.db',
                       help='SQLite database file for simulator state (default: simulator_state.db)')
    parser.add_argument('--reset-state', action='store_true',
                       help='Reset/clear previous simulator state')
    
    # Logging and config
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate configuration without starting trading')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info("=" * 80)
    logger.info("Trade Launcher Starting")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Direction: {args.direction}")
    logger.info(f"Exchange: {args.exchange}")
    logger.info(f"Asset: {args.asset}")
    
    try:
        # Determine trading mode
        exchange_mode = ExchangeTradingMode.PAPER if args.mode == "paper" else ExchangeTradingMode.TRADE
        live_trading_mode = LiveTradingMode.PAPER if args.mode == "paper" else LiveTradingMode.LIVE
        
        # Validate credentials for trade mode
        if exchange_mode == ExchangeTradingMode.TRADE:
            if not args.api_key or not args.api_secret:
                logger.error("API key and secret are required for TRADE mode")
                return 1
        
        # Initialize exchange config
        exchange_config = ExchangeConfig(
            exchange_type=ExchangeType[args.exchange.upper()],
            api_key=args.api_key or "paper_key",
            api_secret=args.api_secret or "paper_secret",
            password=args.api_password,
            use_testnet=(exchange_mode == ExchangeTradingMode.PAPER),  # Use testnet for paper trading
            trade_symbol=args.asset,
            mode=exchange_mode
        )
        
        # Create exchange dispatcher
        dispatcher = ExchangeDispatcher(exchange_config)
        
        # Initialize simulator if in paper mode
        simulator = None
        if exchange_mode == ExchangeTradingMode.PAPER:
            logger.info("Initializing paper trading simulator...")
            
            # Reset state if requested
            if args.reset_state and Path(args.state_file).exists():
                logger.info(f"Removing existing state file: {args.state_file}")
                Path(args.state_file).unlink()
            
            # Create simulator config
            simulator_config = SimulatorConfig()
            
            # Create simulator
            simulator = PaperTradingSimulator(
                config=simulator_config,
                exchange=args.exchange,
                initial_balance=args.initial_balance,
                direction_constraint=args.direction,
                db_path=args.state_file
            )
            
            # Set simulator callback in dispatcher
            dispatcher.set_simulator_callback(simulator.simulate_order)
            
            logger.info(f"✅ Simulator initialized with balance: {args.initial_balance} USDT")
            logger.info(f"✅ Direction constraint: {args.direction}")
        
        # Initialize exchange dispatcher
        logger.info("Initializing exchange dispatcher...")
        success = await dispatcher.initialize()
        
        if not success:
            logger.error("Failed to initialize exchange dispatcher")
            return 1
        
        logger.info("✅ Exchange dispatcher initialized")
        
        # Dry run check
        if args.dry_run:
            logger.info("Dry run completed successfully - configuration validated")
            logger.info("Use without --dry-run to start actual trading")
            return 0
        
        # Create trading config
        trading_config = TradingConfig(
            mode=live_trading_mode,
            exchange_name=args.exchange,
            symbols=[args.asset]
        )
        
        # Create and start trading engine
        logger.info("Starting trading engine...")
        
        trading_engine = TradingEngine(trading_config, dispatcher)
        
        # Inject simulator into order manager if in paper mode
        if simulator:
            trading_engine.order_manager.simulator = simulator
            logger.info("✅ Simulator injected into order manager")
        
        # Initialize parameter manager for hot swapping
        parameter_manager = get_parameter_manager(trading_engine)
        parameter_manager.set_trading_engine(trading_engine)
        
        logger.info("✅ Trading parameter manager initialized - hot swap enabled")
        
        # Start the trading engine
        await trading_engine.start()
        
        # Start monitoring after engine is running
        parameter_manager._start_monitoring()
        
        logger.info("=" * 80)
        logger.info("Trading system is running!")
        logger.info("=" * 80)
        
        # Display status
        status = await trading_engine.get_trading_status()
        logger.info(f"Mode: {status['mode']}")
        logger.info(f"Symbols: {status['symbols']}")
        logger.info(f"Status: {'ACTIVE' if status['trading_active'] else 'INACTIVE'}")
        
        if simulator:
            metrics = simulator.get_performance_metrics()
            logger.info(f"Initial Balance: {metrics['initial_balance']} USDT")
            logger.info(f"Current Balance: {metrics['current_balance']} USDT")
            logger.info(f"Net PnL: {metrics['net_pnl']:.2f} USDT ({metrics['net_pnl_pct']:.2%})")
        
        logger.info("Press Ctrl+C to stop...")
        
        # Run until interrupted
        try:
            last_report_date = None
            
            while True:
                await asyncio.sleep(60)  # Update every minute
                
                # Display periodic status
                if simulator:
                    metrics = simulator.get_performance_metrics()
                    logger.info(
                        f"Balance: {metrics['current_balance']:.2f} USDT | "
                        f"PnL: {metrics['net_pnl']:.2f} ({metrics['net_pnl_pct']:.2%}) | "
                        f"Trades: {metrics['total_trades']}"
                    )
                
                # Generate daily report at end of day (or once per day)
                from datetime import date
                current_date = date.today()
                if last_report_date != current_date:
                    logger.info("Generating daily report...")
                    
                    # Generate report for paper trading
                    if simulator:
                        await simulator.generate_daily_report(args.asset, current_date)
                    else:
                        # Generate report for live trading
                        await trading_engine.order_manager.generate_daily_report(args.asset, current_date)
                    
                    last_report_date = current_date
                    logger.info("✅ Daily report generated")
        
        except KeyboardInterrupt:
            logger.info("\nReceived stop signal, shutting down...")
        
        finally:
            # Stop parameter manager monitoring
            try:
                parameter_manager = get_parameter_manager()
                parameter_manager.stop_monitoring()
            except Exception as e:
                logger.warning(f"Error stopping parameter manager: {e}")
            
            # Stop the trading engine
            logger.info("Stopping trading engine...")
            await trading_engine.stop()
            
            logger.info("✅ Trading system stopped successfully")
        
        return 0
        
    except Exception as e:
        logger.exception(f"Error during trading: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
