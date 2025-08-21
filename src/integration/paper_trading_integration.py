#!/usr/bin/env python3
"""
Paper Trading Integration Module

This module ensures the enhanced reporting system is natively integrated
when launching paper/live trading and provides consistent metrics for
backtesting and walk-forward analysis.
"""

from datetime import datetime
from typing import Any

from src.paper_trader import PaperTrader, setup_paper_trader
from src.reports.paper_trading_reporter import (
    PaperTradingReporter,
    setup_paper_trading_reporter,
)
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
    initialization_error,
    invalid,
)


class PaperTradingIntegration:
    """
    Integration module for paper trading with enhanced reporting.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize paper trading integration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("PaperTradingIntegration")

        # Core components
        self.paper_trader: PaperTrader | None = None
        self.reporter: PaperTradingReporter | None = None

        # Integration state
        self.is_initialized: bool = False
        self.is_running: bool = False

        # Configuration
        self.integration_config = config.get("paper_trading_integration", {})
        self.enable_detailed_reporting = self.integration_config.get(
            "enable_detailed_reporting",
            True,
        )
        self.enable_real_time_reporting = self.integration_config.get(
            "enable_real_time_reporting",
            True,
        )
        self.report_interval = self.integration_config.get(
            "report_interval",
            3600,
        )  # 1 hour

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
            ValueError: (False, "Invalid integration configuration"),
            AttributeError: (False, "Missing required integration parameters"),
        },
        default_return=False,
        context="integration initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize paper trading integration with enhanced reporting.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Paper Trading Integration...")

            # Initialize enhanced paper trader
            self.paper_trader = await setup_paper_trader(self.config)
            if not self.paper_trader:
                self.print(failed("Failed to initialize enhanced paper trader"))
                return False

            # Initialize detailed reporter
            if self.enable_detailed_reporting:
                self.reporter = await setup_paper_trading_reporter(self.config)
                if not self.reporter:
                    self.logger.warning(
                        "Failed to initialize detailed reporter, continuing without detailed reporting",
                    )

            # Validate integration
            if not self._validate_integration():
                self.print(failed("Integration validation failed"))
                return False

            self.is_initialized = True
            self.logger.info("✅ Paper Trading Integration initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Paper Trading Integration initialization failed: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="integration validation",
    )
    def _validate_integration(self) -> bool:
        """Validate integration components."""
        try:
            if not self.paper_trader:
                self.print(initialization_error("Paper trader not initialized"))
                return False

            if self.enable_detailed_reporting and not self.reporter:
                self.print(
                    initialization_error(
                        "Detailed reporter not initialized but required",
                    ),
                )
                return False

            return True

        except Exception:
            self.print(error("Error validating integration: {e}"))
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid trade parameters"),
            AttributeError: (False, "Missing components for trade execution"),
        },
        default_return=False,
        context="trade execution",
    )
    async def execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None = None,
    ) -> bool:
        """Execute a trade via paper trader and record via reporter if available."""
        try:
            if not self.is_initialized:
                self.print(initialization_error("Integration not initialized"))
                return False

            if side not in {"BUY", "SELL"}:
                self.print(invalid("Invalid trade side: {side}"))
                return False

            if not self.paper_trader:
                self.print(initialization_error("Paper trader not available"))
                return False

            # Execute via paper trader
            order_result = await self.paper_trader.execute_trade(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
            )

            # Record via reporter if present
            if self.reporter and order_result:
                await self.reporter.record_trade_execution(order_result)

            return bool(order_result)

        except Exception:
            self.print(error("Error executing integrated trade: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="real-time report generation",
    )
    async def generate_real_time_report(self) -> str | None:
        """Generate a real-time report via reporter if available."""
        try:
            if not self.reporter:
                return None
            return await self.reporter.generate_html_report()
        except Exception:
            self.print(error("Error generating real-time report: {e}"))
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="performance metrics retrieval",
    )
    async def get_performance_metrics(self) -> dict[str, Any] | None:
        try:
            if not self.paper_trader:
                return None
            return await self.paper_trader.get_performance_metrics()
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
            if not self.paper_trader:
                return None
            return await self.paper_trader.get_trade_history()
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
            if not self.paper_trader:
                return None
            return await self.paper_trader.get_portfolio_summary()
        except Exception:
            self.print(error("Error getting portfolio summary: {e}"))
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="comprehensive report generation",
    )
    async def generate_comprehensive_report(self, context: str = "initial") -> bool:
        try:
            if not self.reporter or not self.paper_trader:
                return False

            metrics = await self.paper_trader.get_performance_metrics()
            trades = await self.paper_trader.get_trade_history()
            html = await self.reporter.generate_html_report(metrics=metrics, trades=trades, context=context)
            return html is not None
        except Exception:
            self.print(error("Error generating comprehensive report: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="basic report generation",
    )
    async def generate_basic_report(self) -> bool:
        try:
            if not self.reporter:
                return False
            html = await self.reporter.generate_html_report()
            return html is not None
        except Exception:
            self.print(error("Error generating basic report: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="integration cleanup",
    )
    async def stop(self) -> None:
        try:
            self.logger.info("🛑 Stopping Paper Trading Integration...")
            self.is_running = False
            if self.paper_trader:
                await self.paper_trader.stop()
            self.logger.info("✅ Paper Trading Integration stopped successfully")
        except Exception:
            self.print(error("Error stopping integration: {e}"))


# Global integration instance
paper_trading_integration: PaperTradingIntegration | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="paper trading integration setup",
)
async def setup_paper_trading_integration(
    config: dict[str, Any] | None = None,
) -> PaperTradingIntegration | None:
    try:
        global paper_trading_integration
        if config is None:
            config = {}
        paper_trading_integration = PaperTradingIntegration(config)
        success = await paper_trading_integration.initialize()
        if success:
            return paper_trading_integration
        return None
    except Exception as e:
        print(f"Error setting up paper trading integration: {e}")
        return None
