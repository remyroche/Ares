#!/usr/bin/env python3
"""
Enhanced Trading Launcher

This module provides a comprehensive launcher for paper trading, live trading,
and backtesting with integrated detailed reporting capabilities.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.integration.paper_trading_integration import PaperTradingIntegration, setup_paper_trading_integration
from src.backtesting.enhanced_backtester import EnhancedBacktester, setup_enhanced_backtester


class EnhancedTradingLauncher:
    """
    Enhanced trading launcher with comprehensive reporting integration.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize enhanced trading launcher.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("EnhancedTradingLauncher")
        
        # Trading components
        self.paper_trading_integration: Optional[PaperTradingIntegration] = None
        self.enhanced_backtester: Optional[EnhancedBacktester] = None
        
        # Launcher state
        self.is_initialized: bool = False
        self.current_mode: str = "none"  # "paper", "live", "backtest"
        
        # Configuration
        self.launcher_config = config.get("enhanced_trading_launcher", {})
        self.enable_paper_trading = self.launcher_config.get("enable_paper_trading", True)
        self.enable_live_trading = self.launcher_config.get("enable_live_trading", False)
        self.enable_backtesting = self.launcher_config.get("enable_backtesting", True)
        self.enable_detailed_reporting = self.launcher_config.get("enable_detailed_reporting", True)

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid launcher configuration"),
            AttributeError: (False, "Missing required launcher parameters"),
        },
        default_return=False,
        context="launcher initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize enhanced trading launcher.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Enhanced Trading Launcher...")

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for enhanced trading launcher")
                return False

            # Initialize components based on configuration
            await self._initialize_components()

            self.is_initialized = True
            self.logger.info("✅ Enhanced Trading Launcher initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Enhanced Trading Launcher initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate launcher configuration."""
        try:
            # Check if at least one trading mode is enabled
            if not any([
                self.enable_paper_trading,
                self.enable_live_trading,
                self.enable_backtesting,
            ]):
                self.logger.error("At least one trading mode must be enabled")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="components initialization",
    )
    async def _initialize_components(self) -> None:
        """Initialize trading components."""
        try:
            # Initialize paper trading integration
            if self.enable_paper_trading:
                self.paper_trading_integration = await setup_paper_trading_integration(self.config)
                if self.paper_trading_integration:
                    self.logger.info("✅ Paper trading integration initialized")
                else:
                    self.logger.warning("⚠️ Failed to initialize paper trading integration")

            # Initialize enhanced backtester
            if self.enable_backtesting:
                self.enhanced_backtester = await setup_enhanced_backtester(self.config)
                if self.enhanced_backtester:
                    self.logger.info("✅ Enhanced backtester initialized")
                else:
                    self.logger.warning("⚠️ Failed to initialize enhanced backtester")

        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid paper trading parameters"),
            AttributeError: (False, "Missing paper trading components"),
        },
        default_return=False,
        context="paper trading launch",
    )
    async def launch_paper_trading(
        self,
        trading_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Launch paper trading with enhanced reporting.

        Args:
            trading_config: Additional trading configuration

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_initialized:
                self.logger.error("Launcher not initialized")
                return False

            if not self.paper_trading_integration:
                self.logger.error("Paper trading integration not available")
                return False

            self.logger.info("🚀 Launching paper trading with enhanced reporting...")
            self.current_mode = "paper"

            # Update configuration if provided
            if trading_config:
                self.config.update(trading_config)

            # Generate initial report
            await self.paper_trading_integration.generate_comprehensive_report("initial")

            self.logger.info("✅ Paper trading launched successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error launching paper trading: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid live trading parameters"),
            AttributeError: (False, "Missing live trading components"),
        },
        default_return=False,
        context="live trading launch",
    )
    async def launch_live_trading(
        self,
        trading_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Launch live trading with enhanced reporting.

        Args:
            trading_config: Additional trading configuration

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_initialized:
                self.logger.error("Launcher not initialized")
                return False

            if not self.enable_live_trading:
                self.logger.error("Live trading not enabled")
                return False

            self.logger.info("🚀 Launching live trading with enhanced reporting...")
            self.current_mode = "live"

            # Update configuration if provided
            if trading_config:
                self.config.update(trading_config)

            # TODO: Initialize live trading components
            # This would integrate with the existing live trading system
            self.logger.warning("⚠️ Live trading not yet implemented")

            return True

        except Exception as e:
            self.logger.error(f"Error launching live trading: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid backtest parameters"),
            AttributeError: (False, "Missing backtest components"),
        },
        default_return=False,
        context="backtest launch",
    )
    async def launch_backtest(
        self,
        historical_data: pd.DataFrame,
        strategy_signals: pd.DataFrame,
        backtest_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Launch enhanced backtest with comprehensive reporting.

        Args:
            historical_data: Historical market data
            strategy_signals: Strategy signals DataFrame
            backtest_config: Additional backtest configuration

        Returns:
            Dict[str, Any]: Backtest results with detailed metrics
        """
        try:
            if not self.is_initialized:
                self.logger.error("Launcher not initialized")
                return {}

            if not self.enhanced_backtester:
                self.logger.error("Enhanced backtester not available")
                return {}

            self.logger.info("🚀 Launching enhanced backtest with comprehensive reporting...")
            self.current_mode = "backtest"

            # Update configuration if provided
            if backtest_config:
                self.config.update(backtest_config)

            # Run backtest
            results = await self.enhanced_backtester.run_backtest(
                historical_data=historical_data,
                strategy_signals=strategy_signals,
                trade_metadata=backtest_config,
            )

            # Generate comprehensive report
            await self.enhanced_backtester.generate_backtest_report("comprehensive")

            self.logger.info("✅ Enhanced backtest completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Error launching backtest: {e}")
            return {}

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid trade parameters"),
            AttributeError: (False, "Missing trade components"),
        },
        default_return=False,
        context="trade execution",
    )
    async def execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        trade_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Execute trade with integrated reporting.

        Args:
            symbol: Trading symbol
            side: Trade side ("buy" or "sell")
            quantity: Trade quantity
            price: Trade price
            timestamp: Trade timestamp
            trade_metadata: Additional trade metadata

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_initialized:
                self.logger.error("Launcher not initialized")
                return False

            if self.current_mode == "paper" and self.paper_trading_integration:
                return await self.paper_trading_integration.execute_trade(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    timestamp=timestamp,
                    trade_metadata=trade_metadata,
                )
            elif self.current_mode == "live":
                # TODO: Implement live trading execution
                self.logger.warning("⚠️ Live trading execution not yet implemented")
                return False
            else:
                self.logger.error(f"Trade execution not available for mode: {self.current_mode}")
                return False

        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return False

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for current mode."""
        try:
            if self.current_mode == "paper" and self.paper_trading_integration:
                return self.paper_trading_integration.get_performance_metrics()
            elif self.current_mode == "backtest" and self.enhanced_backtester:
                return self.enhanced_backtester.get_backtest_results()
            elif self.current_mode == "live":
                # TODO: Implement live trading metrics
                return {"mode": "live", "status": "not_implemented"}
            else:
                return {"mode": self.current_mode, "status": "no_metrics_available"}

        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {}

    def get_trade_history(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get trade history for current mode."""
        try:
            if self.current_mode == "paper" and self.paper_trading_integration:
                return self.paper_trading_integration.get_trade_history(symbol)
            elif self.current_mode == "backtest" and self.enhanced_backtester:
                results = self.enhanced_backtester.get_backtest_results()
                return results.get("trade_history", [])
            else:
                return []

        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            return []

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary for current mode."""
        try:
            if self.current_mode == "paper" and self.paper_trading_integration:
                return self.paper_trading_integration.get_portfolio_summary()
            elif self.current_mode == "backtest" and self.enhanced_backtester:
                results = self.enhanced_backtester.get_backtest_results()
                return {
                    "final_portfolio_value": results.get("final_portfolio_value", 0.0),
                    "current_positions": results.get("current_positions", {}),
                    "performance_metrics": results.get("performance_metrics", {}),
                }
            else:
                return {}

        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {}

    async def generate_comprehensive_report(
        self,
        report_type: str = "comprehensive",
        export_formats: List[str] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive report for current mode."""
        try:
            if export_formats is None:
                export_formats = ["json", "csv", "html"]

            if self.current_mode == "paper" and self.paper_trading_integration:
                return await self.paper_trading_integration.generate_comprehensive_report(
                    report_type, export_formats
                )
            elif self.current_mode == "backtest" and self.enhanced_backtester:
                return await self.enhanced_backtester.generate_backtest_report(
                    report_type, export_formats
                )
            else:
                return await self._generate_basic_report(report_type, export_formats)

        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="basic report generation",
    )
    async def _generate_basic_report(
        self,
        report_type: str,
        export_formats: List[str],
    ) -> Dict[str, Any]:
        """Generate basic report when detailed reporting is not available."""
        try:
            import json
            import os
            from datetime import datetime

            # Get basic data
            performance_metrics = self.get_performance_metrics()
            trade_history = self.get_trade_history()
            portfolio_summary = self.get_portfolio_summary()

            report_data = {
                "report_type": f"basic_{report_type}",
                "generated_at": datetime.now().isoformat(),
                "current_mode": self.current_mode,
                "performance_metrics": performance_metrics,
                "portfolio_summary": portfolio_summary,
                "trade_history": trade_history,
                "launcher_status": {
                    "is_initialized": self.is_initialized,
                    "current_mode": self.current_mode,
                    "enable_detailed_reporting": self.enable_detailed_reporting,
                }
            }

            # Export reports
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = "reports/launcher"
            os.makedirs(report_dir, exist_ok=True)

            for format_type in export_formats:
                if format_type == "json":
                    filename = f"launcher_report_{timestamp}.json"
                    filepath = os.path.join(report_dir, filename)
                    with open(filepath, "w") as f:
                        json.dump(report_data, f, indent=2, default=str)
                    self.logger.info(f"✅ Exported launcher JSON report: {filepath}")

            return report_data

        except Exception as e:
            self.logger.error(f"Error generating basic report: {e}")
            return {}

    def get_launcher_status(self) -> Dict[str, Any]:
        """Get launcher status."""
        return {
            "is_initialized": self.is_initialized,
            "current_mode": self.current_mode,
            "enable_paper_trading": self.enable_paper_trading,
            "enable_live_trading": self.enable_live_trading,
            "enable_backtesting": self.enable_backtesting,
            "enable_detailed_reporting": self.enable_detailed_reporting,
            "paper_trading_available": self.paper_trading_integration is not None,
            "enhanced_backtester_available": self.enhanced_backtester is not None,
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="launcher cleanup",
    )
    async def stop(self) -> None:
        """Stop enhanced trading launcher."""
        try:
            # Stop current mode
            if self.current_mode == "paper" and self.paper_trading_integration:
                await self.paper_trading_integration.stop()
            elif self.current_mode == "backtest" and self.enhanced_backtester:
                self.enhanced_backtester.stop()

            # Generate final report
            await self.generate_comprehensive_report("final")

            self.current_mode = "none"
            self.logger.info("✅ Enhanced Trading Launcher stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping launcher: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="enhanced trading launcher setup",
)
async def setup_enhanced_trading_launcher(
    config: Dict[str, Any] | None = None,
) -> EnhancedTradingLauncher | None:
    """
    Setup enhanced trading launcher.

    Args:
        config: Configuration dictionary

    Returns:
        EnhancedTradingLauncher: Configured launcher instance
    """
    try:
        if config is None:
            config = {}

        launcher = EnhancedTradingLauncher(config)
        success = await launcher.initialize()

        if success:
            return launcher
        else:
            return None

    except Exception as e:
        system_logger.error(f"Error setting up enhanced trading launcher: {e}")
        return None