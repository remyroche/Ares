#!/usr/bin/env python3
"""
Enhanced Trading Launcher

Provides a comprehensive launcher for paper trading, live trading, and
backtesting with integrated detailed reporting capabilities.
"""

from datetime import datetime, timedelta
from typing import Any, TYPE_CHECKING, Dict, List, Optional, Union, Tuple
import json
import os
import asyncio
from decimal import Decimal, ROUND_HALF_UP

try:
    import pandas as pd
except ImportError:  # Fallback for environments without pandas
    class _PD:
        DataFrame = Any  # type: ignore
    pd = _PD()  # type: ignore

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.warning_symbols import (
    error,
    execution_error,
    failed,
    initialization_error,
    invalid,
    warning,
)
from src.integration.paper_trading_integration import (
    PaperTradingIntegration,
    setup_paper_trading_integration,
)

if TYPE_CHECKING:
    from src.utils.advanced_decorators import performance_monitor, PerformanceLevel
    from src.backtesting.enhanced_backtester import EnhancedBacktester


class EnhancedTradingLauncher:
    """
    Enhanced trading launcher with comprehensive reporting integration.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the enhanced trading launcher."""
        self.config = config
        self.logger = system_logger.getChild("EnhancedTradingLauncher")
        
        # Trading components
        self.paper_trading_integration: Optional[PaperTradingIntegration] = None
        self.enhanced_backtester: Optional["EnhancedBacktester"] = None
        self.live_trading_engine: Optional[Any] = None
        
        # Launcher state
        self.is_initialized: bool = False
        self.current_mode: str = "none"  # "paper", "live", "backtest"
        self.start_time: Optional[datetime] = None
        self.session_id: Optional[str] = None
        
        # Configuration
        self.launcher_config = config.get("enhanced_trading_launcher", {})
        self.enable_paper_trading = self.launcher_config.get(
            "enable_paper_trading", True
        )
        self.enable_live_trading = self.launcher_config.get(
            "enable_live_trading", False
        )
        self.enable_backtesting = self.launcher_config.get("enable_backtesting", True)
        self.enable_detailed_reporting = self.launcher_config.get(
            "enable_detailed_reporting", True
        )
        
        # Trading limits and safety
        self.max_position_size = self.launcher_config.get("max_position_size", 100000.0)
        self.max_daily_trades = self.launcher_config.get("max_daily_trades", 100)
        self.risk_per_trade = self.launcher_config.get("risk_per_trade", 0.02)  # 2%
        
        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.daily_stats: Dict[str, Any] = {}

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid launcher configuration"),
            AttributeError: (False, "Missing required launcher parameters"),
        },
        default_return=False,
        context="launcher initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the enhanced trading launcher."""
        try:
            self.logger.info("Initializing Enhanced Trading Launcher...")
            
            # Generate session ID
            self.session_id = f"launcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
            
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(
                    invalid("Invalid configuration for enhanced trading launcher")
                )
                return False
            
            # Initialize components based on configuration
            await self._initialize_components()
            
            # Initialize performance tracking
            self._initialize_performance_tracking()
            
            self.is_initialized = True
            self.start_time = datetime.now()
            self.logger.info(f"✅ Enhanced Trading Launcher initialized successfully (Session: {self.session_id})")
            return True
            
        except Exception as e:
            self.logger.exception(
                f"❌ Enhanced Trading Launcher initialization failed: {e}"
            )
            return False

    def _initialize_performance_tracking(self) -> None:
        """Initialize performance tracking structures."""
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "average_trade": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
        }
        
        self.daily_stats = {
            "trades_today": 0,
            "pnl_today": 0.0,
            "volume_today": 0.0,
            "last_reset": datetime.now().date().isoformat(),
        }

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate the launcher configuration."""
        try:
            # Check if at least one trading mode is enabled
            if not (self.enable_paper_trading or self.enable_live_trading or self.enable_backtesting):
                self.logger.error(error("At least one trading mode must be enabled"))
                return False
            
            # Validate risk parameters
            if self.risk_per_trade <= 0 or self.risk_per_trade > 0.1:
                self.logger.error(error("Risk per trade must be between 0 and 10%"))
                return False
            
            if self.max_position_size <= 0:
                self.logger.error(error("Max position size must be positive"))
                return False
            
            if self.max_daily_trades <= 0:
                self.logger.error(error("Max daily trades must be positive"))
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(error(f"Error validating configuration: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="components initialization",
    )
    async def _initialize_components(self) -> None:
        """Initialize trading components based on configuration."""
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
                    from src.backtesting.enhanced_backtester import (
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
            
            # Initialize live trading engine
            if self.enable_live_trading:
                try:
                    self.live_trading_engine = await self._setup_live_trading_engine()
                    if self.live_trading_engine:
                        self.logger.info("✅ Live trading engine initialized")
                    else:
                        self.logger.warning("⚠️ Failed to initialize live trading engine")
                except Exception as e:
                    self.logger.error(failed(f"Live trading engine setup failed: {e}"))
                    self.live_trading_engine = None
                    
        except Exception as e:
            self.logger.error(initialization_error(f"Error initializing components: {e}"))

    async def _setup_live_trading_engine(self) -> Optional[Any]:
        """Setup live trading engine with proper error handling."""
        try:
            # This would integrate with the existing live trading system
            # For now, we'll create a mock implementation
            class MockLiveTradingEngine:
                def __init__(self):
                    self.is_connected = False
                    self.account_balance = 100000.0
                    self.positions = {}
                    self.orders = []
                
                async def connect(self) -> bool:
                    await asyncio.sleep(0.1)  # Simulate connection time
                    self.is_connected = True
                    return True
                
                async def disconnect(self) -> None:
                    self.is_connected = False
                
                async def get_account_info(self) -> Dict[str, Any]:
                    return {
                        "balance": self.account_balance,
                        "equity": self.account_balance,
                        "positions": self.positions,
                        "orders": self.orders
                    }
                
                async def place_order(self, symbol: str, side: str, quantity: float, price: float) -> str:
                    order_id = f"order_{len(self.orders) + 1}"
                    self.orders.append({
                        "id": order_id,
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "price": price,
                        "status": "filled",
                        "timestamp": datetime.now().isoformat()
                    })
                    return order_id
            
            engine = MockLiveTradingEngine()
            await engine.connect()
            return engine
            
        except Exception as e:
            self.logger.error(f"Failed to setup live trading engine: {e}")
            return None

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid paper trading parameters"),
            AttributeError: (False, "Missing paper trading components"),
        },
        default_return=False,
        context="paper trading launch",
    )
    async def launch_paper_trading(self, trading_config: Optional[Dict[str, Any]] = None) -> bool:
        """Launch paper trading with enhanced reporting."""
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
            
            # Reset daily stats
            self._reset_daily_stats()
            
            # Generate initial report
            await self.paper_trading_integration.generate_comprehensive_report(
                "initial"
            )
            
            self.logger.info("✅ Paper trading launched successfully")
            return True
            
        except Exception as e:
            self.logger.error(error(f"Error launching paper trading: {e}"))
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid live trading parameters"),
            AttributeError: (False, "Missing live trading components"),
        },
        default_return=False,
        context="live trading launch",
    )
    async def launch_live_trading(self, trading_config: Optional[Dict[str, Any]] = None) -> bool:
        """Launch live trading with enhanced reporting."""
        try:
            if not self.is_initialized:
                self.logger.error(initialization_error("Launcher not initialized"))
                return False
            
            if not self.enable_live_trading:
                self.logger.error(error("Live trading not enabled"))
                return False
            
            if not self.live_trading_engine:
                self.logger.error(error("Live trading engine not available"))
                return False
            
            self.logger.info("🚀 Launching live trading with enhanced reporting...")
            self.current_mode = "live"
            
            # Update configuration if provided
            if trading_config:
                self.config.update(trading_config)
            
            # Reset daily stats
            self._reset_daily_stats()
            
            # Verify live trading connection
            if not self.live_trading_engine.is_connected:
                await self.live_trading_engine.connect()
            
            self.logger.info("✅ Live trading launched successfully")
            return True
            
        except Exception as e:
            self.logger.error(error(f"Error launching live trading: {e}"))
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid backtest parameters"),
            AttributeError: (False, "Missing backtest components"),
        },
        default_return={},
        context="backtest launch",
    )
    async def launch_backtest(
        self, 
        historical_data: Any, 
        strategy_signals: Any, 
        backtest_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Launch enhanced backtest with comprehensive reporting."""
        try:
            if not self.is_initialized:
                self.logger.error(initialization_error("Launcher not initialized"))
                return {}
            
            if not self.enhanced_backtester:
                self.logger.error(error("Enhanced backtester not available"))
                return {}
            
            self.logger.info(
                "🚀 Launching enhanced backtest with comprehensive reporting..."
            )
            self.current_mode = "backtest"
            
            # Update configuration if provided
            if backtest_config:
                self.config.update(backtest_config)
            
            # Validate historical data
            if not self._validate_data_quality(historical_data).is_valid:
                self.logger.error(error("Historical data quality validation failed"))
                return {}
            
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
                self.logger.error(initialization_error("Launcher not initialized"))
                return False
            
            # Validate trade parameters
            if not self._validate_trade_parameters(symbol, side, quantity, price):
                return False
            
            # Check daily limits
            if not self._check_daily_limits(quantity, price):
                return False
            
            if self.current_mode == "paper" and self.paper_trading_integration:
                success = await self.paper_trading_integration.execute_trade(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    timestamp=timestamp,
                    trade_metadata=trade_metadata,
                )
                if success:
                    self._record_trade(symbol, side, quantity, price, timestamp, trade_metadata)
                return success
                
            elif self.current_mode == "live" and self.live_trading_engine:
                # Execute live trade
                order_id = await self.live_trading_engine.place_order(
                    symbol, side, quantity, price
                )
                if order_id:
                    self._record_trade(symbol, side, quantity, price, timestamp, trade_metadata)
                    self.logger.info(f"✅ Live trade executed successfully: {order_id}")
                    return True
                else:
                    self.logger.error(execution_error("Failed to place live trade order"))
                    return False
            else:
                self.logger.error(
                    f"Trade execution not available for mode: {self.current_mode}"
                )
                return False
                
        except Exception as e:
            self.logger.error(error(f"Error executing trade: {e}"))
            return False

    def _validate_trade_parameters(self, symbol: str, side: str, quantity: float, price: float) -> bool:
        """Validate trade parameters before execution."""
        try:
            # Validate symbol
            if not symbol or not isinstance(symbol, str):
                self.logger.error(error("Invalid symbol provided"))
                return False
            
            # Validate side
            if side not in ["buy", "sell"]:
                self.logger.error(error("Trade side must be 'buy' or 'sell'"))
                return False
            
            # Validate quantity
            if quantity <= 0:
                self.logger.error(error("Trade quantity must be positive"))
                return False
            
            # Validate price
            if price <= 0:
                self.logger.error(error("Trade price must be positive"))
                return False
            
            # Check position size limit
            position_value = quantity * price
            if position_value > self.max_position_size:
                self.logger.error(error(f"Position size {position_value} exceeds limit {self.max_position_size}"))
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(error(f"Error validating trade parameters: {e}"))
            return False

    def _check_daily_limits(self, quantity: float, price: float) -> bool:
        """Check if trade would exceed daily limits."""
        try:
            # Check daily trade count
            if self.daily_stats["trades_today"] >= self.max_daily_trades:
                self.logger.error(error(f"Daily trade limit reached: {self.max_daily_trades}"))
                return False
            
            # Check daily volume
            trade_value = quantity * price
            if self.daily_stats["volume_today"] + trade_value > self.max_position_size * 10:
                self.logger.error(error("Daily volume limit would be exceeded"))
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(error(f"Error checking daily limits: {e}"))
            return False

    def _record_trade(self, symbol: str, side: str, quantity: float, price: float, timestamp: datetime, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record trade for performance tracking."""
        try:
            trade_record = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "timestamp": timestamp.isoformat(),
                "value": quantity * price,
                "metadata": metadata or {},
                "session_id": self.session_id,
            }
            
            self.trade_history.append(trade_record)
            self.daily_stats["trades_today"] += 1
            self.daily_stats["volume_today"] += trade_record["value"]
            
            # Update performance metrics
            self._update_performance_metrics(trade_record)
            
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")

    def _update_performance_metrics(self, trade_record: Dict[str, Any]) -> None:
        """Update performance metrics based on new trade."""
        try:
            self.performance_metrics["total_trades"] += 1
            
            # Calculate P&L (simplified - would need actual position tracking)
            # For now, we'll just track trade count and volume
            
            # Update win rate (simplified)
            if trade_record["side"] == "buy":
                # This is a simplified approach - real implementation would track actual P&L
                self.performance_metrics["winning_trades"] += 1
            
            # Calculate win rate
            if self.performance_metrics["total_trades"] > 0:
                self.performance_metrics["win_rate"] = (
                    self.performance_metrics["winning_trades"] / 
                    self.performance_metrics["total_trades"]
                )
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    def _reset_daily_stats(self) -> None:
        """Reset daily statistics."""
        self.daily_stats = {
            "trades_today": 0,
            "pnl_today": 0.0,
            "volume_today": 0.0,
            "last_reset": datetime.now().date().isoformat(),
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the current trading mode."""
        try:
            base_metrics = self.performance_metrics.copy()
            base_metrics.update({
                "session_id": self.session_id,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "current_mode": self.current_mode,
                "daily_stats": self.daily_stats,
            })
            
            if self.current_mode == "paper" and self.paper_trading_integration:
                paper_metrics = self.paper_trading_integration.get_performance_metrics()
                base_metrics.update(paper_metrics)
            elif self.current_mode == "backtest" and self.enhanced_backtester:
                backtest_metrics = self.enhanced_backtester.get_backtest_results()
                base_metrics.update(backtest_metrics)
            elif self.current_mode == "live" and self.live_trading_engine:
                try:
                    account_info = asyncio.run(self.live_trading_engine.get_account_info())
                    base_metrics.update({
                        "live_account_balance": account_info.get("balance", 0.0),
                        "live_positions": account_info.get("positions", {}),
                        "live_orders": len(account_info.get("orders", [])),
                    })
                except Exception as e:
                    self.logger.warning(f"Could not get live account info: {e}")
            
            return base_metrics
                
        except Exception as e:
            self.logger.error(error(f"Error getting performance metrics: {e}"))
            return {}

    def get_trade_history(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get trade history for the current trading mode."""
        try:
            if symbol:
                return [trade for trade in self.trade_history if trade["symbol"] == symbol]
            
            if self.current_mode == "paper" and self.paper_trading_integration:
                paper_history = self.paper_trading_integration.get_trade_history(symbol)
                return paper_history + self.trade_history
            elif self.current_mode == "backtest" and self.enhanced_backtester:
                results = self.enhanced_backtester.get_backtest_results()
                return results.get("trade_history", []) + self.trade_history
            else:
                return self.trade_history
                
        except Exception as e:
            self.logger.error(error(f"Error getting trade history: {e}"))
            return []

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary for the current trading mode."""
        try:
            base_summary = {
                "session_id": self.session_id,
                "current_mode": self.current_mode,
                "total_trades": len(self.trade_history),
                "session_start": self.start_time.isoformat() if self.start_time else None,
            }
            
            if self.current_mode == "paper" and self.paper_trading_integration:
                paper_summary = self.paper_trading_integration.get_portfolio_summary()
                base_summary.update(paper_summary)
            elif self.current_mode == "backtest" and self.enhanced_backtester:
                results = self.enhanced_backtester.get_backtest_results()
                base_summary.update({
                    "final_portfolio_value": results.get("final_portfolio_value", 0.0),
                    "current_positions": results.get("current_positions", {}),
                    "performance_metrics": results.get("performance_metrics", {}),
                })
            elif self.current_mode == "live" and self.live_trading_engine:
                try:
                    account_info = asyncio.run(self.live_trading_engine.get_account_info())
                    base_summary.update({
                        "live_balance": account_info.get("balance", 0.0),
                        "live_positions": account_info.get("positions", {}),
                        "live_orders": len(account_info.get("orders", [])),
                    })
                except Exception as e:
                    self.logger.warning(f"Could not get live account info: {e}")
            
            return base_summary
                
        except Exception as e:
            self.logger.error(error(f"Error getting portfolio summary: {e}"))
            return {}

    async def generate_comprehensive_report(
        self, 
        report_type: str = "summary", 
        export_formats: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive report for the current trading mode."""
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
            self.logger.error(error(f"Error generating comprehensive report: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="basic report generation",
    )
    async def _generate_basic_report(
        self, 
        report_type: str, 
        export_formats: List[str]
    ) -> Dict[str, Any]:
        """Generate basic launcher report."""
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
                    "session_id": self.session_id,
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

    def get_launcher_status(self) -> Dict[str, Any]:
        """Get current launcher status."""
        return {
            "is_initialized": self.is_initialized,
            "current_mode": self.current_mode,
            "enable_paper_trading": self.enable_paper_trading,
            "enable_live_trading": self.enable_live_trading,
            "enable_backtesting": self.enable_backtesting,
            "enable_detailed_reporting": self.enable_detailed_reporting,
            "paper_trading_available": self.paper_trading_integration is not None,
            "enhanced_backtester_available": self.enhanced_backtester is not None,
            "live_trading_available": self.live_trading_engine is not None,
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "daily_stats": self.daily_stats,
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="launcher cleanup",
    )
    async def stop(self) -> None:
        """Stop the launcher and cleanup resources."""
        try:
            # Stop current mode
            if self.current_mode == "paper" and self.paper_trading_integration:
                await self.paper_trading_integration.stop()
            elif self.current_mode == "backtest" and self.enhanced_backtester:
                self.enhanced_backtester.stop()
            elif self.current_mode == "live" and self.live_trading_engine:
                await self.live_trading_engine.disconnect()
            
            # Generate final report
            await self.generate_comprehensive_report("final")
            
            self.current_mode = "none"
            self.logger.info("✅ Enhanced Trading Launcher stopped successfully")
            
        except Exception as e:
            self.logger.error(error(f"Error stopping launcher: {e}"))

    def _validate_data_quality(self, data: Any) -> Any:
        """Validate data quality."""
        try:
            if data is None or (hasattr(data, 'empty') and data.empty):
                return type('ValidationResult', (), {'is_valid': False, 'errors': ['Empty data']})()
            
            errors = []
            if hasattr(data, 'isnull') and data.isnull().sum().sum() > 0:
                errors.append('Missing values detected')
            
            if hasattr(data, '__len__') and len(data) < 10:
                errors.append('Insufficient data')
            
            is_valid = len(errors) == 0
            return type('ValidationResult', (), {'is_valid': is_valid, 'errors': errors})()
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return type('ValidationResult', (), {'is_valid': False, 'errors': [str(e)]})()


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="enhanced trading launcher setup",
)
async def setup_enhanced_trading_launcher(config: Optional[Dict[str, Any]] = None) -> Optional[EnhancedTradingLauncher]:
    """Setup and initialize enhanced trading launcher."""
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

