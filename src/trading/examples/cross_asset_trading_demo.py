"""
Cross-Asset Trading Demo

Comprehensive demonstration of multi-cryptocurrency trading with:
1. Simultaneous trading across multiple symbols
2. Single trade execution at a time (trade semaphore)
3. Consolidated cross-asset performance reporting
4. Real-time monitoring of all assets

This example shows how to:
- Configure multiple symbols for trading
- Set up the CrossAssetTradingManager
- Monitor cross-asset performance in real-time
- Generate consolidated reports
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import signal
import sys
import os

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_structured, LogLevel
)

# Import cross-asset trading components
from ..execution.cross_asset_trading_manager import (
    CrossAssetTradingManager, create_cross_asset_trading_manager,
    start_cross_asset_trading, CrossAssetTrade, TradeStatus
)
from ..config.cross_asset_config import CrossAssetConfig, SymbolConfiguration, TradingMode

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossAssetTradingDemo:
    """
    Demonstration of cross-asset trading capabilities.
    """

    def __init__(self):
        self.manager: CrossAssetTradingManager = None
        self.is_running = False
        self.demo_duration_hours = 1  # Demo runs for 1 hour
        self.reporting_interval = 300  # Generate reports every 5 minutes

    async def setup_demo(self) -> bool:
        """
        Set up the cross-asset trading demonstration.

        Returns:
            bool: True if setup successful
        """
        try:
            tprint_info("🚀 Setting up Cross-Asset Trading Demonstration")
            print("=" * 80)

            # Create comprehensive configuration
            config = await self._create_demo_configuration()

            # Create and initialize trading manager
            self.manager = create_cross_asset_trading_manager(config)

            success = await self.manager.initialize()
            if not success:
                tprint_error("❌ Failed to initialize cross-asset trading manager")
                return False

            tprint_success("✅ Cross-asset trading manager initialized successfully")
            return True

        except Exception as e:
            tprint_error(f"❌ Demo setup failed: {e}")
            return False

    async def _create_demo_configuration(self) -> Dict[str, Any]:
        """Create comprehensive demo configuration."""
        try:
            tprint_info("🔄 Creating demo configuration...")

            # Define symbols to trade
            symbols = [
                'ETHUSDT',  # Ethereum
                'BTCUSDT',  # Bitcoin
                'ADAUSDT',  # Cardano
                'SOLUSDT',  # Solana
                'DOTUSDT',  # Polkadot
            ]

            # Create base configuration
            config = {
                'symbols': symbols,
                'primary_symbol': 'ETHUSDT',
                'exchange': 'binance',
                'trading_mode': 'paper',
                'total_account_balance': 10000.0,

                # Cross-asset strategy
                'strategy': 'equal_weight',
                'rebalancing_frequency': 'daily',
                'rebalancing_threshold': 0.1,

                # Risk management
                'max_concurrent_symbols': 3,
                'max_portfolio_risk': 0.05,
                'max_symbol_concentration': 0.3,
                'risk_per_trade': 0.02,

                # Trade execution control
                'max_trades_per_minute': 5,
                'max_trades_per_hour': 50,
                'trade_timeout_seconds': 30,

                # Symbol-specific configurations
                'symbol_configs': await self._create_symbol_configs(symbols),

                # Reporting
                'consolidated_reporting': True,
                'real_time_monitoring': True,
                'export_directory': 'demo_cross_asset_reports',

                # Advanced settings
                'enable_dynamic_allocation': True,
                'enable_correlation_monitoring': True,
                'enable_liquidity_monitoring': True,

                # Demo-specific settings
                'demo_mode': True,
                'enhanced_logging': True,
                'performance_targets': {
                    'target_win_rate': 0.6,
                    'target_sharpe_ratio': 1.5,
                    'max_drawdown_limit': 0.15
                }
            }

            tprint_success(f"✅ Created configuration for {len(symbols)} symbols")
            return config

        except Exception as e:
            tprint_error(f"❌ Failed to create demo configuration: {e}")
            raise

    async def _create_symbol_configs(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Create symbol-specific configurations."""
        try:
            tprint_info("🔄 Creating symbol-specific configurations...")

            symbol_configs = {}

            # Base configuration for all symbols
            base_config = {
                'exchange': 'binance',
                'trading_mode': 'paper',
                'base_position_size': 0.1,
                'max_position_size': 0.25,
                'min_position_size': 0.01,
                'risk_level': 'moderate',
                'max_portfolio_risk': 0.02,
                'max_symbol_risk': 0.05,
                'confidence_threshold': 0.6,
                'max_daily_trades': 10,
                'min_trade_interval': 60
            }

            # Symbol-specific adjustments
            symbol_adjustments = {
                'ETHUSDT': {
                    'account_balance': 3000.0,
                    'volatility_adjustment': 1.2,
                    'liquidity_factor': 0.9,
                    'correlation_factor': 0.8,
                    'confidence_threshold': 0.65,
                    'max_position_size': 0.3
                },
                'BTCUSDT': {
                    'account_balance': 4000.0,
                    'volatility_adjustment': 1.0,
                    'liquidity_factor': 1.0,
                    'correlation_factor': 1.0,
                    'confidence_threshold': 0.7,
                    'max_position_size': 0.35
                },
                'ADAUSDT': {
                    'account_balance': 1000.0,
                    'volatility_adjustment': 1.5,
                    'liquidity_factor': 0.5,
                    'correlation_factor': 0.4,
                    'confidence_threshold': 0.55,
                    'max_position_size': 0.2
                },
                'SOLUSDT': {
                    'account_balance': 1000.0,
                    'volatility_adjustment': 1.3,
                    'liquidity_factor': 0.6,
                    'correlation_factor': 0.7,
                    'confidence_threshold': 0.58,
                    'max_position_size': 0.22
                },
                'DOTUSDT': {
                    'account_balance': 1000.0,
                    'volatility_adjustment': 1.4,
                    'liquidity_factor': 0.4,
                    'correlation_factor': 0.5,
                    'confidence_threshold': 0.56,
                    'max_position_size': 0.21
                }
            }

            for symbol in symbols:
                config = base_config.copy()
                if symbol in symbol_adjustments:
                    config.update(symbol_adjustments[symbol])

                symbol_configs[symbol] = config
                tprint_info(f"  📊 {symbol}: Balance ${config['account_balance']".0f"}, "
                           f"Volatility {config['volatility_adjustment']".1f"}x")

            tprint_success(f"✅ Created symbol configurations")
            return symbol_configs

        except Exception as e:
            tprint_error(f"❌ Failed to create symbol configurations: {e}")
            raise

    async def run_demo(self) -> bool:
        """
        Run the cross-asset trading demonstration.

        Returns:
            bool: True if demo completed successfully
        """
        try:
            tprint_info("🚀 Starting Cross-Asset Trading Demonstration")
            print("=" * 80)

            # Start trading session
            success = await self.manager.start_trading_session()
            if not success:
                tprint_error("❌ Failed to start trading session")
                return False

            self.is_running = True

            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            # Run demo for specified duration
            start_time = datetime.now()
            end_time = start_time + timedelta(hours=self.demo_duration_hours)

            tprint_info(f"📊 Demo will run for {self.demo_duration_hours} hour(s)")
            tprint_info(f"📊 Will generate reports every {self.reporting_interval} seconds")

            # Main demo loop
            while self.is_running and datetime.now() < end_time:
                try:
                    # Generate periodic reports
                    await self._generate_periodic_reports()

                    # Display live statistics
                    await self._display_live_stats()

                    # Wait for next iteration
                    await asyncio.sleep(self.reporting_interval)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    tprint_error(f"❌ Demo loop error: {e}")
                    await asyncio.sleep(30)  # Wait before retrying

            # Final summary and cleanup
            await self._generate_final_summary()

            tprint_success("🎉 Cross-Asset Trading Demonstration Completed!")
            return True

        except Exception as e:
            tprint_error(f"❌ Demo execution failed: {e}")
            return False

        finally:
            # Cleanup
            if self.manager:
                await self.manager.stop_trading_session()

    async def _generate_periodic_reports(self):
        """Generate periodic performance reports."""
        try:
            # Get manager statistics
            stats = self.manager.get_manager_stats()

            # Generate consolidated report
            report = await self.manager.generate_consolidated_report("session")

            # Display key metrics
            if 'cross_asset_metrics' in report:
                metrics = report['cross_asset_metrics']
                tprint_info("📊 Cross-Asset Performance Update:")
                tprint_structured({
                    'Total Trades': metrics.get('total_trades', 0),
                    'Total PnL': f"${metrics.get('total_pnl', 0):.2".2f"
                    'Success Rate': f"{metrics.get('success_rate', 0):.1%".1%"
                    'Max Drawdown': f"{metrics.get('max_drawdown', 0):.1%".1%"
                    'Cross-Correlation': f"{metrics.get('cross_correlation', 0):.3f".3f"
                }, LogLevel.INFO)

            # Show symbol performance
            if 'symbol_performance' in report:
                tprint_info("📈 Symbol Performance:")
                for symbol, perf in report['symbol_performance'].items():
                    pnl = perf.get('total_pnl', 0)
                    status = perf.get('status', 'unknown')
                    tprint_info(f"  {symbol}: ${pnl:+.2".2f" {status}")

        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate periodic reports: {e}")

    async def _display_live_stats(self):
        """Display live trading statistics."""
        try:
            stats = self.manager.get_manager_stats()

            if 'current_session' in stats:
                session = stats['current_session']
                tprint_info("🔴 LIVE TRADING STATS:")
                tprint_structured({
                    'Symbols Active': len(session.get('symbols', [])),
                    'Total PnL': f"${session.get('total_pnl', 0):.2".2f"
                    'Trades in Queue': session.get('trades_in_queue', 0),
                    'Executed Trades': session.get('executed_trades', 0),
                    'Session Duration': f"{(datetime.now() - session.get('start_time_parsed', datetime.now())).seconds // 60}m"
                }, LogLevel.INFO)

        except Exception as e:
            tprint_warning(f"⚠️ Failed to display live stats: {e}")

    async def _generate_final_summary(self):
        """Generate final comprehensive summary."""
        try:
            tprint_info("📊 Generating Final Summary...")

            # Generate comprehensive final report
            final_report = await self.manager.generate_consolidated_report("session")

            # Export report
            success = await self.manager.export_consolidated_report(final_report, "final_demo_report.json")
            if success:
                tprint_success("✅ Final report exported")

            # Display final statistics
            if 'cross_asset_metrics' in final_report:
                metrics = final_report['cross_asset_metrics']

                tprint_info("🏆 FINAL CROSS-ASSET RESULTS:")
                print("=" * 60)
                tprint_structured({
                    '🎯 Total Trades Executed': final_report.get('cross_asset_metrics', {}).get('total_trades', 0),
                    '💰 Total Profit/Loss': f"${final_report.get('cross_asset_metrics', {}).get('total_pnl', 0):.2".2f"
                    '📈 Success Rate': f"{final_report.get('cross_asset_metrics', {}).get('success_rate', 0):.1%".1%"
                    '📉 Maximum Drawdown': f"{final_report.get('cross_asset_metrics', {}).get('max_drawdown', 0):.1%".1%"
                    '🔗 Cross-Asset Correlation': f"{final_report.get('cross_asset_metrics', {}).get('cross_correlation', 0):.3f".3f"
                    '⏱️  Demo Duration': f"{self.demo_duration_hours}h"
                }, LogLevel.INFO)

            # Show per-symbol breakdown
            if 'symbol_performance' in final_report:
                tprint_info("📊 PER-SYMBOL BREAKDOWN:")
                for symbol, perf in final_report['symbol_performance'].items():
                    pnl = perf.get('total_pnl', 0)
                    status = perf.get('status', 'unknown')
                    emoji = "📈" if pnl > 0 else "📉" if pnl < 0 else "➡️"
                    tprint_info(f"  {emoji} {symbol}: ${pnl:+.2".2f" {status}")

            print("=" * 60)

        except Exception as e:
            tprint_error(f"❌ Failed to generate final summary: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        tprint_info(f"⏹️ Received signal {signum}, shutting down gracefully...")
        self.is_running = False

    async def cleanup(self):
        """Clean up demo resources."""
        try:
            tprint_info("🧹 Cleaning up demo resources...")

            if self.manager:
                await self.manager.stop_trading_session()

            tprint_success("✅ Demo cleanup completed")

        except Exception as e:
            tprint_error(f"❌ Cleanup error: {e}")

async def main():
    """
    Main function to run the cross-asset trading demonstration.
    """
    demo = CrossAssetTradingDemo()

    try:
        # Setup demo
        success = await demo.setup_demo()
        if not success:
            return False

        # Run demo
        success = await demo.run_demo()
        if not success:
            return False

        return True

    except KeyboardInterrupt:
        tprint_info("⏹️ Demo interrupted by user")
        return False

    except Exception as e:
        tprint_error(f"❌ Demo failed: {e}")
        return False

    finally:
        # Cleanup
        await demo.cleanup()

if __name__ == "__main__":
    try:
        # Configure demo duration via environment variable (optional)
        demo_duration = int(os.getenv('DEMO_DURATION_HOURS', '1'))
        print(f"🚀 Starting Cross-Asset Trading Demo (Duration: {demo_duration}h)")

        success = asyncio.run(main())

        if success:
            tprint_success("✅ Demo completed successfully!")
            print("\n📁 Check the 'demo_cross_asset_reports/' directory for detailed reports")
            print("📊 Key files:")
            print("   - final_demo_report.json: Comprehensive performance summary")
            print("   - cross_asset_report_[timestamp].json: Periodic reports")
            sys.exit(0)
        else:
            tprint_error("❌ Demo failed or was interrupted")
            sys.exit(1)

    except KeyboardInterrupt:
        tprint_info("⏹️ Demo stopped by user")
        sys.exit(130)

    except Exception as e:
        tprint_error(f"❌ Demo crashed: {e}")
        sys.exit(1)