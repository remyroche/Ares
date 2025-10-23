#!/usr/bin/env python3
import pandas as pd
import pandas as pd
import pandas as pd
from src.core.error_classes import execution_error, initialization_error
from ..utils.logger import system_logger
from ..core.decorators import handles_errors
import pandas as pd

"""
Enhanced Trading Launcher

Provides a comprehensive launcher for paper trading, live trading, and
backtesting with integrated detailed reporting capabilities.
"""

from src.core.domain import (
    PerformanceLevel,
    performance_monitor
)

from datetime import datetime
from typing import Any, TYPE_CHECKING
import json
import os

try:
    import pandas as pd
except Exception:  # Fallback for environments without pandas
    class _PD:
        pass
        DataFrame = Any  # type: ignore
    pd = _PD()  # type: ignore

from ..utils.logger import system_logger
from src.utils.warning_symbols import (
       error,
   execution_error,
   failed,
   initialization_error,
   invalid,
   warning,
)

import logging
import time
from typing import TYPE_CHECKING, Any

try:
    from src.integration.paper_trading_integration import (
        PaperTradingIntegration,
        setup_paper_trading_integration,
    )
except ImportError:
    # Fallback classes if module is not available
    class PaperTradingIntegration:
        pass
    def setup_paper_trading_integration():
        pass
if TYPE_CHECKING:
    pass

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
        self.enhanced_backtester: Any = None
        self.live_trading_system: Any = None

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

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid launcher configuration"),
            AttributeError: (False, "Missing required launcher parameters"),
        },
        default_return=False,
        context="launcher initialization",
    )
    @performance_monitor(level=PerformanceLevel.DETAILED)
    async def initialize(self) -> bool:
        """
        Initialize enhanced trading launcher.

        Returns:
            bool: True if initialization successful = False otherwise
        """
        try:
            self.logger.info("Initializing Enhanced Trading Launcher...")

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(
                    invalid("Invalid configuration for enhanced trading launcher")
                )
                return False

            # Initialize components based on configuration
            await self._initialize_components()

            self.is_initialized = True
            self.logger.info("✅ Enhanced Trading Launcher initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Enhanced Trading Launcher initialization failed: {e}"
            )
            return False

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate launcher configuration."""
        try:
            # Check if at least one trading mode is enabled
            if not (self.enable_paper_trading or self.enable_live_trading or self.enable_backtesting):
                self.logger.error(error("At least one trading mode must be enabled"))
                return False

            return True

        except Exception as e:
            self.logger.error(error(f"Error validating configuration: {e}"))
            return False

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="components initialization",
    )
    @performance_monitor(level=PerformanceLevel.BASIC)
    async def _initialize_components(self) -> None:
        """Initialize trading components."""
        try:
            # Initialize paper trading integration
            if self.enable_paper_trading:
                self.paper_trading_integration = await setup_paper_trading_integration(
                    self.config
                )
                if self.paper_trading_integration:
                    self.logger.info("✅ Paper trading integration initialized")
                else:
                    self.logger.warning(
                        "⚠️ Failed to initialize paper trading integration"
                    )

            # Initialize enhanced backtester
            if self.enable_backtesting:
                try:
                    from src.training.enhanced_backtester import (  # type: ignore
                        setup_enhanced_backtester as _setup_backtester,
                    )
                    self.enhanced_backtester = await _setup_backtester(self.config)
                except Exception as e:
                    self.logger.error(failed(f"Backtester import/setup failed: {e}"))
                    self.enhanced_backtester = None
                if self.enhanced_backtester:
                    self.logger.info("✅ Enhanced backtester initialized")
                else:
                    self.logger.error(failed("⚠️ Failed to initialize enhanced backtester"))

            # Initialize live trading system
            if self.enable_live_trading:
                try:
                    from live_trading.unified_trading_system import create_trading_system
                    from live_trading.config import TradingConfig, TradingMode
                    
                    # Create trading configuration for live trading
                    live_trading_config = TradingConfig(
                        mode=TradingMode.LIVE,
                        symbols=self.launcher_config.get("live_trading_symbols", ["BTCUSDT", "ETHUSDT"]),
                        max_position_size=self.launcher_config.get("max_position_size", 1000.0),
                        max_daily_loss=self.launcher_config.get("max_daily_loss", 100.0),
                        enable_risk_management=self.launcher_config.get("enable_risk_management", True)
                    )
                    
                    # Create live trading system
                    self.live_trading_system = await create_trading_system(
                        trading_config=live_trading_config,
                        exchanges=self.launcher_config.get("live_trading_exchanges", ["binance"]),
                        enable_websockets=True,
                        enable_paper_trading=False
                    )
                    
                    if self.live_trading_system:
                        self.logger.info("✅ Live trading system initialized")
                    else:
                        self.logger.error(failed("⚠️ Failed to initialize live trading system"))
                        
                except Exception as e:
                    self.logger.error(failed(f"Live trading system import/setup failed: {e}"))
                    self.live_trading_system = None

        except Exception as e:
            self.logger.error(initialization_error(f"Error initializing components: {e}"))

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid paper trading parameters"),
            AttributeError: (False, "Missing paper trading components"),
        },
        default_return=False,
        context="paper trading launch",
    )
    @performance_monitor(level=PerformanceLevel.DETAILED)
    async def launch_paper_trading(
        self,
        trading_config: dict[str, Any] | None = None,
    ) -> bool:
        """
        Launch paper trading with enhanced reporting.

        Args:
            trading_config: Additional trading configuration

        Returns:
            bool: True if successful = False otherwise
        """
        try:
            if not self.is_initialized:
                self.logger.error(initialization_error("Launcher not initialized"))
                return False

            if not self.paper_trading_integration:
                self.logger.error(error("Paper trading integration not available"))
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

        except Exception as e:
            self.logger.error(error(f"Error launching paper trading: {e}"))
            return False

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid live trading parameters"),
            AttributeError: (False, "Missing live trading components"),
        },
        default_return=False,
        context="live trading launch",
    )
    @performance_monitor(level=PerformanceLevel.DETAILED)
    async def launch_live_trading(
        self,
        trading_config: dict[str, Any] | None = None,
    ) -> bool:
        """
        Launch live trading with enhanced reporting.

        Args:
            trading_config: Additional trading configuration

        Returns:
            bool: True if successful = False otherwise
        """
        try:
            if not self.is_initialized:
                self.logger.error(initialization_error("Launcher not initialized"))
                return False

            if not self.enable_live_trading:
                self.logger.error(error("Live trading not enabled"))
                return False

            self.logger.info("🚀 Launching live trading with enhanced reporting...")
            self.current_mode = "live"

            # Update configuration if provided
            if trading_config:
                self.config.update(trading_config)

            # Initialize live trading components
            await self._initialize_live_trading_components()

            return True

        except Exception as e:
            self.logger.error(error(f"Error launching live trading: {e}"))
            return False

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid backtest parameters"),
            AttributeError: (False, "Missing backtest components"),
        },
        default_return=False,
        context="backtest launch",
    )
    @performance_monitor(level=PerformanceLevel.DETAILED)
    async def launch_backtest(
        self,
        historical_data: pd.DataFrame,
        strategy_signals: pd.DataFrame,
        backtest_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
                self.logger.error(initialization_error("Launcher not initialized"))
                return {}

            if not self.enhanced_backtester:
                self.logger.error(error("Enhanced backtester not available"))
                return {}

            self.logger.info(
                "🚀 Launching enhanced backtest with comprehensive reporting...",
            )
            self.current_mode = "backtest"

            # Update configuration if provided
            if backtest_config:
                self.config.update(backtest_config)

            # Run backtest
            results = await self.enhanced_backtester.run_backtest(
                historical_data=historical_data,
                strategy_signals=strategy_signals,
                backtest_config=backtest_config or {},
            )

            # Generate comprehensive report
            await self.enhanced_backtester.generate_backtest_report("comprehensive")

            self.logger.info("✅ Enhanced backtest completed successfully")
            return results

        except Exception as e:
            self.logger.error(error(f"Error launching backtest: {e}"))
            return {}

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid trade parameters"),
            AttributeError: (False, "Missing trade components"),
        },
        default_return=False,
        context="trade execution",
    )
    @performance_monitor(level=PerformanceLevel.DETAILED)
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
            bool: True if successful = False otherwise
        """
        try:
            if not self.is_initialized:
                self.logger.error(initialization_error("Launcher not initialized"))
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
            if self.current_mode == "live":
                return await self._execute_live_trade(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    timestamp=timestamp,
                    trade_metadata=trade_metadata,
                )
            self.logger.error(
                f"Trade execution not available for mode: {self.current_mode}",
            )
            return False

        except Exception as e:
            self.logger.error(error(f"Error executing trade: {e}"))
            return False

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics for current mode."""
        try:
            if self.current_mode == "paper" and self.paper_trading_integration:
                return self.paper_trading_integration.get_performance_metrics()
            if self.current_mode == "backtest" and self.enhanced_backtester:
                return self.enhanced_backtester.get_backtest_results()
            if self.current_mode == "live":
                return await self._get_live_trading_metrics()
            return {"mode": self.current_mode, "status": "no_metrics_available"}

        except Exception as e:
            self.logger.error(error(f"Error getting performance metrics: {e}"))
            return {}

    def get_trade_history(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Get trade history for current mode."""
        try:
            if self.current_mode == "paper" and self.paper_trading_integration:
                return self.paper_trading_integration.get_trade_history(symbol)
            if self.current_mode == "backtest" and self.enhanced_backtester:
                results = self.enhanced_backtester.get_backtest_results()
                return results.get("trade_history", [])
            return []

        except Exception as e:
            self.logger.error(error(f"Error getting trade history: {e}"))
            return []

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get portfolio summary for current mode."""
        try:
            if self.current_mode == "paper" and self.paper_trading_integration:
                return self.paper_trading_integration.get_portfolio_summary()
            if self.current_mode == "backtest" and self.enhanced_backtester:
                results = self.enhanced_backtester.get_backtest_results()
                return {
                    "final_portfolio_value": results.get("final_portfolio_value", 0.0),
                    "current_positions": results.get("current_positions", {}),
                    "performance_metrics": results.get("performance_metrics", {}),
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
        """Generate comprehensive report for current mode."""
        try:
            if export_formats is None:
                export_formats = ["json", "csv", "html"]

            if self.current_mode == "paper" and self.paper_trading_integration:
                return await self.paper_trading_integration.generate_comprehensive_report(
                    report_type,
                    export_formats,
                )
            if self.current_mode == "backtest" and self.enhanced_backtester:
                return await self.enhanced_backtester.generate_backtest_report(
                    report_type,
                    export_formats,
                )
            return await self._generate_basic_report(report_type, export_formats)

        except Exception as e:
            self.logger.error(error(f"Error generating comprehensive report: {e}"))
            return {}

    @handles_errors(
        exceptions=(Exception,),
        default_return=None,
        context="basic report generation",
    )
    @performance_monitor(level=PerformanceLevel.BASIC)
    async def _generate_basic_report(
        self,
        report_type: str,
        export_formats: list[str],
    ) -> dict[str, Any]:
        """Generate basic report when detailed reporting is not available."""
        try:
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
                },
            }

            # Export reports
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = "reports/launcher"
            os.makedirs(report_dir, exist_ok=True)

            for format_type in export_formats:
                if format_type == "json":
                    filename = f"launcher_report_{timestamp}.json"
                    filepath = os.path.join(report_dir, filename)
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(report_data, f, indent=2, default=str)
                    self.logger.info(f"✅ Exported launcher JSON report: {filepath}")

            return report_data

        except Exception as e:
            self.logger.error(error(f"Error generating basic report: {e}"))
            return {}

    def get_launcher_status(self) -> dict[str, Any]:
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
            "live_trading_available": self.live_trading_system is not None,
        }

    @handles_errors(
        exceptions=(Exception,),
        default_return=None,
        context="launcher cleanup",
    )
    @performance_monitor(level=PerformanceLevel.BASIC)
    async def stop(self) -> None:
        """Stop enhanced trading launcher."""
        try:
            # Stop current mode
            if self.current_mode == "paper" and self.paper_trading_integration:
                await self.paper_trading_integration.stop()
            elif self.current_mode == "backtest" and self.enhanced_backtester:
                self.enhanced_backtester.stop()
            elif self.current_mode == "live" and self.live_trading_system:
                await self.live_trading_system.stop()

            # Generate final report
            await self.generate_comprehensive_report("final")

            self.current_mode = "none"
            self.logger.info("✅ Enhanced Trading Launcher stopped successfully")

        except Exception as e:
            self.logger.error(error(f"Error stopping launcher: {e}"))

    async def _initialize_live_trading_components(self) -> None:
        """Initialize live trading components."""
        try:
            if not self.live_trading_system:
                self.logger.error(error("Live trading system not available"))
                return

            # Start the live trading system
            await self.live_trading_system.start()
            self.logger.info("✅ Live trading components initialized and started")

        except Exception as e:
            self.logger.error(error(f"Error initializing live trading components: {e}"))
            raise

    async def _execute_live_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        trade_metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Execute live trade through the live trading system."""
        try:
            if not self.live_trading_system:
                self.logger.error(error("Live trading system not available"))
                return False

            # Create trade decision
            from src.interfaces.base_interfaces import TradeDecision
            
            trade_decision = TradeDecision(
                symbol=symbol,
                action=side,
                quantity=quantity,
                price=price,
                confidence=trade_metadata.get("confidence", 0.8) if trade_metadata else 0.8,
                risk_score=trade_metadata.get("risk_score", 0.5) if trade_metadata else 0.5,
                leverage=trade_metadata.get("leverage", 1.0) if trade_metadata else 1.0,
                stop_loss=trade_metadata.get("stop_loss") if trade_metadata else None,
                take_profit=trade_metadata.get("take_profit") if trade_metadata else None,
                timestamp=timestamp
            )

            # Execute trade decision
            success = await self.live_trading_system.execute_trade_decision(trade_decision)
            
            if success:
                self.logger.info(f"✅ Live trade executed: {symbol} {side} {quantity}")
            else:
                self.logger.warning(f"⚠️ Live trade execution failed: {symbol} {side} {quantity}")

            return success

        except Exception as e:
            self.logger.error(error(f"Error executing live trade: {e}"))
            return False

    async def _get_live_trading_metrics(self) -> dict[str, Any]:
        """Get live trading performance metrics."""
        try:
            if not self.live_trading_system:
                return {"mode": "live", "status": "system_not_available"}

            # Get system status
            system_status = await self.live_trading_system.get_system_status()
            
            # Get account summary
            account_summary = await self.live_trading_system.get_account_summary()
            
            return {
                "mode": "live",
                "status": "active" if system_status.get("system_running", False) else "inactive",
                "system_status": system_status,
                "account_summary": account_summary,
                "uptime_seconds": system_status.get("uptime_seconds", 0),
                "total_trades_executed": system_status.get("total_trades_executed", 0),
                "total_signals_processed": system_status.get("total_signals_processed", 0),
                "errors": system_status.get("errors", 0),
                "warnings": system_status.get("warnings", 0)
            }

        except Exception as e:
            self.logger.error(error(f"Error getting live trading metrics: {e}"))
            return {"mode": "live", "status": "error", "error": str(e)}

@handles_errors(
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
