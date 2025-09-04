#!/usr/bin/env python3
"""
Paper Trading Integration Module

Integrates the `PaperTrader` with the `PaperTradingReporter` and provides
helper methods to execute trades and generate reports in real time.
"""
from src.core.decorators import handles_errors, validates as secure_data_processing

from src.core.domain import (
    PerformanceLevel,
    comprehensive_validation,
    performance_monitor
)

from datetime import datetime
from typing import Any, TYPE_CHECKING
import json
import os

from src.utils.comprehensive_logger import get_comprehensive_logger
from src.utils.logger import system_logger
import asyncio
from src.utils.warning_symbols import (
    error,
    failed,
    initialization_error,
    invalid,
    warning,
)
from src.paper_trader import PaperTrader, setup_paper_trader

if TYPE_CHECKING:  # Only for type hints to avoid runtime import of corrupted modules
    from src.reports.paper_trading_reporter import PaperTradingReporter

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
        self.report_interval = self.integration_config.get("report_interval", 3600)

    @performance_monitor(level=PerformanceLevel.DETAILED)
    @handles_errors(
        exceptions=(ValueError, AttributeError),
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

            # Initialize paper trader
            self.paper_trader = await setup_paper_trader(self.config)
            if not self.paper_trader:
                self.logger.error(failed("Failed to initialize paper trader"))
                return False

            # Initialize detailed reporter
            if self.enable_detailed_reporting:
                try:
                    from src.reports.paper_trading_reporter import (
from src.core.decorators.errors import handles_errors
                        setup_paper_trading_reporter as _setup_reporter,
                    )

                    self.reporter = await _setup_reporter(self.config)
                    if not self.reporter:
                        self.logger.warning(
                            "Failed to initialize detailed reporter, continuing without detailed reporting",
                        )
                        self.enable_detailed_reporting = False
                except Exception as e:
                    self.logger.warning(
                        warning(
                            f"Detailed reporter unavailable, continuing without it: {e}",
                        ),
                    )
                    self.enable_detailed_reporting = False

            # Validate integration
            if not self._validate_integration():
                self.logger.error(failed("Integration validation failed"))
                return False

            self.is_initialized = True
            self.logger.info("✅ Paper Trading Integration initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Paper Trading Integration initialization failed: {e}",
            )
            return False

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="integration validation",
    )
    def _validate_integration(self) -> bool:
        """Validate integration components."""
        try:
            if not self.paper_trader:
                self.logger.error(initialization_error("Paper trader not initialized"))
                return False

            # If reporter failed to initialize, degrade gracefully (don't block integration)
            if self.enable_detailed_reporting and not self.reporter:
                self.logger.warning(
                    warning(
                        "Detailed reporter not initialized; proceeding without detailed reporting",
                    ),
                )
                self.enable_detailed_reporting = False

            return True

        except Exception as e:
            self.logger.error(error(f"Error validating integration: {e}"))
            return False

    @performance_monitor(level=PerformanceLevel.DETAILED)
    @secure_data_processing
    @comprehensive_validation()
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="integrated trade execution",
    )
    async def execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        trade_metadata: dict[str, Any] | None = None,
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
            if not self.is_initialized or not self.paper_trader:
                self.logger.error(initialization_error("Integration not initialized"))
                return False

            # Prepare trade metadata
            if trade_metadata is None:
                trade_metadata = {}

            # Add default metadata
            trade_metadata.update(
                {
                    "exchange": "paper",
                    "leverage": trade_metadata.get("leverage", 1.0),
                    "duration": trade_metadata.get("duration", "unknown"),
                    "strategy": trade_metadata.get("strategy", "unknown"),
                    "order_type": trade_metadata.get("order_type", "market"),
                    "portfolio_percentage": trade_metadata.get(
                        "portfolio_percentage",
                        0.0,
                    ),
                    "risk_percentage": trade_metadata.get("risk_percentage", 0.0),
                    "max_position_size": trade_metadata.get("max_position_size", 0.0),
                    "position_ranking": trade_metadata.get("position_ranking", 0),
                    "execution_quality": trade_metadata.get("execution_quality", 0.0),
                    "risk_metrics": trade_metadata.get("risk_metrics", {}),
                    "notes": trade_metadata.get("notes"),
                }
            )

            # Execute trade
            side_lower = side.lower()
            if side_lower == "buy":
                success = await self.paper_trader.execute_buy_order(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    timestamp=timestamp,
                )
            elif side_lower == "sell":
                success = await self.paper_trader.execute_sell_order(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    timestamp=timestamp,
                )
            else:
                self.logger.error(invalid(f"Invalid trade side: {side}"))
                return False

            if success:
                self.logger.info(
                    f"✅ Integrated trade executed: {side} {quantity} {symbol} @ ${price:.4f}",
            )

                # Also write to dedicated trades log (optional)
                try:
                    cl = get_comprehensive_logger()
                    if cl:
                        cl.log_trade(
                            f"{side.upper()} {quantity} {symbol} @ ${price:.4f} ts={timestamp.isoformat()}",
                        )
                except Exception:
                    # Trade logging should not affect execution
                    pass

                # Generate real-time report if enabled
                if self.enable_real_time_reporting and self.reporter:
                    await self._generate_real_time_report()

            return success

        except Exception as e:
            self.logger.error(error(f"Error executing integrated trade: {e}"))
            return False

    @performance_monitor(level=PerformanceLevel.DETAILED)
    @handles_errors(
        exceptions=(Exception,),
        default_return=None,
        context="real-time report generation",
    )
    async def _generate_real_time_report(self) -> None:
        """Generate real-time performance report."""
        try:
            if self.reporter:
                await self.reporter.generate_detailed_report("real_time", ["json"])

        except Exception as e:
            self.logger.error(error(f"Error generating real-time report: {e}"))

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics."""
        try:
            # Get basic performance metrics
            basic_metrics = (
                self.paper_trader.calculate_performance() if self.paper_trader else {}
            )

            # Get detailed metrics if reporter is available
            detailed_metrics: dict[str, Any] = {}
            if self.reporter:
                detailed_metrics = self.reporter.get_performance_metrics()
                portfolio_summary = self.reporter.get_portfolio_summary()
                detailed_metrics["portfolio_summary"] = portfolio_summary

            # Combine metrics
            combined_metrics = {**basic_metrics, **detailed_metrics}

            # Add integration status
            combined_metrics.update(
                {
                    "integration_status": {
                        "is_initialized": self.is_initialized,
                        "is_running": self.is_running,
                        "enable_detailed_reporting": self.enable_detailed_reporting,
                        "enable_real_time_reporting": self.enable_real_time_reporting,
                    },
                }
            )

            return combined_metrics

        except Exception as e:
            self.logger.error(error(f"Error getting performance metrics: {e}"))
            return {}

    def get_trade_history(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Get trade history with optional filtering."""
        try:
            if self.paper_trader:
                return self.paper_trader.get_trade_history(symbol)
            return []

        except Exception as e:
            self.logger.error(error(f"Error getting trade history: {e}"))
            return []

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get portfolio summary."""
        try:
            if self.reporter:
                return self.reporter.get_portfolio_summary()
            if self.paper_trader:
                positions = self.paper_trader.get_all_positions()
                balance = self.paper_trader.get_balance()
                return {
                    "total_value": sum(
                        pos.get("total_cost", 0.0) for pos in positions.values()
                    ),
                    "balance": balance,
                    "positions_count": len(positions),
                    "symbol_positions": positions,
                }
            return {}

        except Exception as e:
            self.logger.error(error(f"Error getting portfolio summary: {e}"))
            return {}

    @performance_monitor(level=PerformanceLevel.BASIC)
    async def generate_comprehensive_report(
        self,
        report_type: str = "comprehensive",
        export_formats: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate comprehensive trading report."""
        try:
            if export_formats is None:
                export_formats = ["json", "csv", "html"]

            if self.reporter:
                return await self.reporter.generate_detailed_report(
                    report_type,
                    export_formats,
                )
            # Fallback to basic report
            return await self._generate_basic_report(report_type, export_formats)

        except Exception as e:
            self.logger.error(error(f"Error generating comprehensive report: {e}"))
            return {}

    @performance_monitor(level=PerformanceLevel.BASIC)
    @handles_errors(
        exceptions=(Exception,),
        default_return={},
        context="basic report generation",
    )
    async def _generate_basic_report(
        self,
        report_type: str,
        export_formats: list[str],
    ) -> dict[str, Any]:
        """Generate basic report when detailed reporter is not available."""
        try:
            # Get basic data
            performance_metrics = self.get_performance_metrics()
            trade_history = self.get_trade_history()
            portfolio_summary = self.get_portfolio_summary()

            report_data = {
                "report_type": report_type,
                "generated_at": datetime.now().isoformat(),
                "performance_metrics": performance_metrics,
                "portfolio_summary": portfolio_summary,
                "trade_history": trade_history,
                "integration_status": {
                    "is_initialized": self.is_initialized,
                    "is_running": self.is_running,
                },
            }

            # Export reports
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = "reports/paper_trading"
            os.makedirs(report_dir, exist_ok=True)

            for format_type in export_formats:
                if format_type == "json":
                    filename = f"basic_paper_trading_report_{timestamp}.json"
                    filepath = os.path.join(report_dir, filename)
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(report_data, f, indent=2, default=str)
                    self.logger.info(f"✅ Exported basic JSON report: {filepath}")

            return report_data

        except Exception as e:
            self.logger.error(error(f"Error generating basic report: {e}"))
            return {}

    def get_integration_status(self) -> dict[str, Any]:
        """Get integration status."""
        return {
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "enable_detailed_reporting": self.enable_detailed_reporting,
            "enable_real_time_reporting": self.enable_real_time_reporting,
            "paper_trader_available": self.paper_trader is not None,
            "reporter_available": self.reporter is not None,
        }

    @performance_monitor(level=PerformanceLevel.BASIC)
    @handles_errors(
        exceptions=(Exception,),
        default_return=None,
        context="integration cleanup",
    )
    async def stop(self) -> None:
        """Stop paper trading integration."""
        try:
            self.is_running = False

            # Stop paper trader
            if self.paper_trader:
                await self.paper_trader.stop()

            # Generate final report
            await self.generate_comprehensive_report("final")

            self.logger.info("✅ Paper Trading Integration stopped successfully")

        except Exception as e:
            self.logger.error(error(f"Error stopping integration: {e}"))

@handles_errors(
    exceptions=(Exception,),
    default_return=None,
    context="paper trading integration setup",
)
async def setup_paper_trading_integration(
    config: dict[str, Any] | None = None,
) -> PaperTradingIntegration | None:
    """
    Setup paper trading integration.

    Args:
        config: Configuration dictionary

    Returns:
        PaperTradingIntegration: Configured integration instance
    """
    try:
        if config is None:
            config = {}

        integration = PaperTradingIntegration(config)
        success = await integration.initialize()

        if success:
            return integration
        return None

    except Exception as e:
        system_logger.exception(
            error(f"Error setting up paper trading integration: {e}"),
        )
        return None
