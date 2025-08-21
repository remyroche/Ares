#!/usr/bin/env python3
"""
Enhanced Trading Launcher

This module provides a comprehensive launcher for paper trading, live trading,
and backtesting with integrated detailed reporting capabilities.
"""

from datetime import datetime
from typing import Any

import pandas as pd

from src.backtesting.enhanced_backtester import (
    EnhancedBacktester,
    setup_enhanced_backtester,
)
from src.integration.paper_trading_integration import (
    PaperTradingIntegration,
    setup_paper_trading_integration,
)
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    execution_error,
    failed,
    initialization_error,
    invalid,
    warning,
)


class EnhancedTradingLauncher:
    """
    Enhanced trading launcher with comprehensive reporting integration.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize enhanced trading launcher.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("EnhancedTradingLauncher")

        # Trading components
        self.paper_trading_integration: PaperTradingIntegration | None = None
        self.enhanced_backtester: EnhancedBacktester | None = None

        # Launcher state
        self.is_initialized: bool = False
        self.current_mode: str = "none"  # "paper", "live", "backtest"

        # Configuration
        self.launcher_config = config.get("enhanced_trading_launcher", {})
        self.enable_paper_trading = self.launcher_config.get(
            "enable_paper_trading",
            True,
        )
        self.enable_live_trading = self.launcher_config.get(
            "enable_live_trading",
            False,
        )
        self.enable_backtesting = self.launcher_config.get("enable_backtesting", True)
        self.enable_detailed_reporting = self.launcher_config.get(
            "enable_detailed_reporting",
            True,
        )

    def print(self, message: str) -> None:
        """Lightweight print wrapper to ensure class uses logger and stdout consistently."""
        try:
            self.logger.info(message)
        finally:
            try:
                builtins_print = __builtins__["print"] if isinstance(__builtins__, dict) else __builtins__.print  # type: ignore
                builtins_print(message)
            except Exception:
                pass

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
                self.print(
                    invalid("Invalid configuration for enhanced trading launcher"),
                )
                return False

            # Initialize components based on configuration
            await self._initialize_components()

            self.is_initialized = True
            self.logger.info("✅ Enhanced Trading Launcher initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Enhanced Trading Launcher initialization failed: {e}",
            )
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
            if not any(
                [
                    self.enable_paper_trading,
                    self.enable_live_trading,
                    self.enable_backtesting,
                ],
            ):
                self.print(error("At least one trading mode must be enabled"))
                return False

            return True

        except Exception:
            self.print(error("Error validating configuration: {e}"))
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
                self.paper_trading_integration = await setup_paper_trading_integration(
                    self.config,
                )
                if self.paper_trading_integration:
                    self.logger.info("✅ Paper trading integration initialized")
                else:
                    self.logger.warning(
                        "⚠️ Failed to initialize paper trading integration",
                    )

            # Initialize enhanced backtester
            if self.enable_backtesting:
                self.enhanced_backtester = await setup_enhanced_backtester(self.config)
                if self.enhanced_backtester:
                    self.logger.info("✅ Enhanced backtester initialized")
                else:
                    self.print(failed("⚠️ Failed to initialize enhanced backtester"))

        except Exception:
            self.print(initialization_error("Error initializing components: {e}"))

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
        trading_config: dict[str, Any] | None = None,
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
                self.print(initialization_error("Launcher not initialized"))
                return False

            if not self.paper_trading_integration:
                self.print(error("Paper trading integration not available"))
                return False

            self.logger.info("🚀 Launching paper trading with enhanced reporting...")
            self.current_mode = "paper"

            # Update configuration if provided
            if trading_config:
                self.config.update(trading_config)

            # Generate initial report
            await self.paper_trading_integration.generate_comprehensive_report(
                "initial",
            )

            self.logger.info("✅ Paper trading launched successfully")
            return True

        except Exception:
            self.print(error("Error launching paper trading: {e}"))
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid live trading parameters"),
            AttributeError: (False, "Missing live trading components"),
        },
        default_return=False,
        context="live trading launch",
    )
    async def launch_live_trading(self) -> bool:
        """Launch live trading (placeholder)."""
        try:
            if not self.is_initialized:
                self.print(initialization_error("Launcher not initialized"))
                return False

            if not self.enable_live_trading:
                self.print(error("Live trading not enabled"))
                return False

            # Placeholder for future implementation
            self.logger.info("⚠️ Live trading not yet implemented")
            return False

        except Exception:
            self.print(error("Error launching live trading: {e}"))
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid backtest parameters"),
            AttributeError: (False, "Missing backtest components"),
        },
        default_return=False,
        context="backtest launch",
    )
    async def launch_backtest(self) -> bool:
        """Launch backtest using enhanced backtester."""
        try:
            if not self.is_initialized:
                self.print(initialization_error("Launcher not initialized"))
                return False

            if not self.enhanced_backtester:
                self.print(error("Enhanced backtester not available"))
                return False

            # Placeholder backtest launch
            self.logger.info("Starting backtest...")
            return await self.enhanced_backtester.run_backtest()

        except Exception:
            self.print(error("Error launching backtest: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="trade execution",
    )
    async def execute_trade(self, payload: dict[str, Any]) -> bool:
        """Execute a trade via the paper trading integration."""
        try:
            if not self.paper_trading_integration:
                self.print(initialization_error("Launcher not initialized"))
                return False

            symbol = payload.get("symbol")
            side = payload.get("side")
            quantity = float(payload.get("quantity", 0))
            price = payload.get("price")

            return await self.paper_trading_integration.execute_trade(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
            )
        except Exception:
            self.print(error("Error executing trade: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="performance metrics retrieval",
    )
    async def get_performance_metrics(self) -> dict[str, Any] | None:
        try:
            if not self.paper_trading_integration:
                return None
            return await self.paper_trading_integration.get_performance_metrics()
        except Exception:
            self.print(error("Error getting performance metrics: {e}"))
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="trade history retrieval",
    )
    async def get_trade_history(self) -> list[dict[str, Any]] | None:
        try:
            if not self.paper_trading_integration:
                return None
            return await self.paper_trading_integration.get_trade_history()
        except Exception:
            self.print(error("Error getting trade history: {e}"))
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="portfolio summary retrieval",
    )
    async def get_portfolio_summary(self) -> dict[str, Any] | None:
        try:
            if not self.paper_trading_integration:
                return None
            return await self.paper_trading_integration.get_portfolio_summary()
        except Exception:
            self.print(error("Error getting portfolio summary: {e}"))
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="comprehensive report generation",
    )
    async def generate_comprehensive_report(self) -> bool:
        try:
            if not self.paper_trading_integration:
                return False
            return await self.paper_trading_integration.generate_comprehensive_report(
                "from_launcher",
            )
        except Exception:
            self.print(error("Error generating comprehensive report: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="basic report generation",
    )
    async def generate_basic_report(self) -> bool:
        try:
            if not self.paper_trading_integration:
                return False
            return await self.paper_trading_integration.generate_basic_report()
        except Exception:
            self.print(error("Error generating basic report: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="launcher cleanup",
    )
    async def stop(self) -> None:
        try:
            self.logger.info("🛑 Stopping Enhanced Trading Launcher...")
            if self.paper_trading_integration:
                await self.paper_trading_integration.stop()
            self.logger.info("✅ Enhanced Trading Launcher stopped successfully")
        except Exception:
            self.print(error("Error stopping launcher: {e}"))


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="enhanced trading launcher setup",
)
async def setup_enhanced_trading_launcher(
    config: dict[str, Any] | None = None,
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
        return None

    except Exception as e:
        system_logger.exception(f"Error setting up enhanced trading launcher: {e}")
        return None
