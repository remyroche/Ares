#!/usr/bin/env python3
"""
Enhanced Backtester with Comprehensive Reporting

This module provides enhanced backtesting capabilities with detailed reporting
that matches the paper trading metrics for consistency across all trading modes.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.reports.paper_trading_reporter import PaperTradingReporter, setup_paper_trading_reporter


class EnhancedBacktester:
    """
    Enhanced backtester with comprehensive reporting capabilities.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize enhanced backtester.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("EnhancedBacktester")
        
        # Backtesting state
        self.is_running: bool = False
        self.current_position: Dict[str, Any] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.portfolio_value: float = 10000.0
        self.initial_balance: float = 10000.0
        
        # Configuration
        self.backtest_config = config.get("enhanced_backtester", {})
        self.initial_balance = self.backtest_config.get("initial_balance", 10000.0)
        self.commission_rate = self.backtest_config.get("commission_rate", 0.001)
        self.slippage_rate = self.backtest_config.get("slippage_rate", 0.0005)
        self.max_position_size = self.backtest_config.get("max_position_size", 0.1)
        
        # Enhanced reporting
        self.reporter: Optional[PaperTradingReporter] = None
        self.enable_detailed_reporting = self.backtest_config.get("enable_detailed_reporting", True)
        
        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {}
        self.equity_curve: List[float] = []
        self.drawdown_curve: List[float] = []

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid backtester configuration"),
            AttributeError: (False, "Missing required backtester parameters"),
        },
        default_return=False,
        context="backtester initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize enhanced backtester with reporting capabilities.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Enhanced Backtester...")

            # Load backtester configuration
            await self._load_backtester_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for enhanced backtester")
                return False

            # Initialize backtesting state
            await self._initialize_backtesting_state()

            # Initialize detailed reporting
            if self.enable_detailed_reporting:
                await self._initialize_detailed_reporting()

            self.logger.info("✅ Enhanced Backtester initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Enhanced Backtester initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="backtester configuration loading",
    )
    async def _load_backtester_configuration(self) -> None:
        """Load backtester configuration."""
        try:
            # Set default parameters
            self.backtest_config.setdefault("initial_balance", 10000.0)
            self.backtest_config.setdefault("commission_rate", 0.001)
            self.backtest_config.setdefault("slippage_rate", 0.0005)
            self.backtest_config.setdefault("max_position_size", 0.1)
            self.backtest_config.setdefault("enable_detailed_reporting", True)

            # Update configuration
            self.initial_balance = self.backtest_config["initial_balance"]
            self.commission_rate = self.backtest_config["commission_rate"]
            self.slippage_rate = self.backtest_config["slippage_rate"]
            self.max_position_size = self.backtest_config["max_position_size"]
            self.enable_detailed_reporting = self.backtest_config["enable_detailed_reporting"]

        except Exception as e:
            self.logger.error(f"Error loading backtester configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate backtester configuration."""
        try:
            if self.initial_balance <= 0:
                self.logger.error("Initial balance must be positive")
                return False

            if self.commission_rate < 0 or self.commission_rate > 0.1:
                self.logger.error("Commission rate must be between 0 and 0.1")
                return False

            if self.slippage_rate < 0 or self.slippage_rate > 0.1:
                self.logger.error("Slippage rate must be between 0 and 0.1")
                return False

            if self.max_position_size <= 0 or self.max_position_size > 1.0:
                self.logger.error("Max position size must be between 0 and 1")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="backtesting state initialization",
    )
    async def _initialize_backtesting_state(self) -> None:
        """Initialize backtesting state."""
        try:
            self.portfolio_value = self.initial_balance
            self.current_position = {}
            self.trade_history = []
            self.equity_curve = [self.initial_balance]
            self.drawdown_curve = [0.0]

            self.logger.info(f"✅ Backtesting state initialized with balance: ${self.portfolio_value:.2f}")

        except Exception as e:
            self.logger.error(f"Error initializing backtesting state: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="detailed reporting initialization",
    )
    async def _initialize_detailed_reporting(self) -> None:
        """Initialize detailed reporting system."""
        try:
            if self.enable_detailed_reporting:
                self.reporter = await setup_paper_trading_reporter(self.config)
                if self.reporter:
                    self.logger.info("✅ Detailed reporting initialized successfully")
                else:
                    self.logger.warning("⚠️ Failed to initialize detailed reporting")

        except Exception as e:
            self.logger.error(f"Error initializing detailed reporting: {e}")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid backtest parameters"),
            AttributeError: (False, "Missing backtest components"),
        },
        default_return=False,
        context="backtest execution",
    )
    async def run_backtest(
        self,
        historical_data: pd.DataFrame,
        strategy_signals: pd.DataFrame,
        trade_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run enhanced backtest with comprehensive reporting.

        Args:
            historical_data: Historical market data
            strategy_signals: Strategy signals DataFrame
            trade_metadata: Additional trade metadata

        Returns:
            Dict[str, Any]: Backtest results with detailed metrics
        """
        try:
            self.logger.info("Starting enhanced backtest...")
            self.is_running = True

            # Initialize results
            results = {
                "trades": [],
                "performance_metrics": {},
                "equity_curve": [],
                "drawdown_curve": [],
                "detailed_analysis": {},
            }

            # Process each signal
            for index, row in strategy_signals.iterrows():
                if not self.is_running:
                    break

                timestamp = row.name if hasattr(row.name, 'isoformat') else pd.Timestamp(index)
                signal = row.get('signal', 0)  # 1 for buy, -1 for sell, 0 for hold
                price = row.get('close', 0)
                symbol = row.get('symbol', 'UNKNOWN')

                if signal != 0 and price > 0:
                    # Execute trade
                    trade_result = await self._execute_backtest_trade(
                        symbol=symbol,
                        signal=signal,
                        price=price,
                        timestamp=timestamp,
                        trade_metadata=trade_metadata,
                    )
                    
                    if trade_result:
                        results["trades"].append(trade_result)

                # Update equity curve
                self._update_equity_curve()

            # Calculate final performance metrics
            results["performance_metrics"] = self._calculate_performance_metrics()
            results["equity_curve"] = self.equity_curve.copy()
            results["drawdown_curve"] = self.drawdown_curve.copy()

            # Generate detailed analysis
            if self.reporter:
                results["detailed_analysis"] = await self._generate_detailed_analysis()

            self.logger.info("✅ Enhanced backtest completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="backtest trade execution",
    )
    async def _execute_backtest_trade(
        self,
        symbol: str,
        signal: int,
        price: float,
        timestamp: datetime,
        trade_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute a trade during backtesting."""
        try:
            if trade_metadata is None:
                trade_metadata = {}

            # Calculate position size
            position_size = self.portfolio_value * self.max_position_size
            quantity = position_size / price

            # Execute trade based on signal
            if signal == 1:  # Buy signal
                return await self._execute_buy_trade(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    timestamp=timestamp,
                    trade_metadata=trade_metadata,
                )
            elif signal == -1:  # Sell signal
                return await self._execute_sell_trade(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    timestamp=timestamp,
                    trade_metadata=trade_metadata,
                )

            return None

        except Exception as e:
            self.logger.error(f"Error executing backtest trade: {e}")
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="buy trade execution",
    )
    async def _execute_buy_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        trade_metadata: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Execute a buy trade during backtesting."""
        try:
            # Calculate costs
            total_cost = quantity * price
            commission = total_cost * self.commission_rate
            slippage = total_cost * self.slippage_rate
            total_with_fees = total_cost + commission + slippage

            # Check if we have enough balance
            if total_with_fees > self.portfolio_value:
                self.logger.warning(f"Insufficient balance for buy trade: ${total_with_fees:.2f} > ${self.portfolio_value:.2f}")
                return None

            # Execute the trade
            self.portfolio_value -= total_with_fees

            # Update position
            if symbol not in self.current_position:
                self.current_position[symbol] = {
                    "quantity": 0,
                    "avg_price": 0,
                    "total_cost": 0,
                }

            position = self.current_position[symbol]
            old_quantity = position["quantity"]
            old_total_cost = position["total_cost"]

            # Update position
            new_quantity = old_quantity + quantity
            new_total_cost = old_total_cost + total_cost
            new_avg_price = new_total_cost / new_quantity if new_quantity > 0 else 0

            position["quantity"] = new_quantity
            position["avg_price"] = new_avg_price
            position["total_cost"] = new_total_cost

            # Record trade
            trade_record = {
                "timestamp": timestamp,
                "symbol": symbol,
                "side": "BUY",
                "quantity": quantity,
                "price": price,
                "total_cost": total_cost,
                "commission": commission,
                "slippage": slippage,
                "portfolio_value_after": self.portfolio_value,
            }
            self.trade_history.append(trade_record)

            # Record detailed trade if reporting is enabled
            if self.enable_detailed_reporting and self.reporter:
                await self._record_detailed_backtest_trade(
                    symbol=symbol,
                    side="long",
                    quantity=quantity,
                    price=price,
                    timestamp=timestamp,
                    total_cost=total_cost,
                    commission=commission,
                    slippage=slippage,
                    trade_metadata=trade_metadata,
                )

            return trade_record

        except Exception as e:
            self.logger.error(f"Error executing buy trade: {e}")
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="sell trade execution",
    )
    async def _execute_sell_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        trade_metadata: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Execute a sell trade during backtesting."""
        try:
            # Check if we have enough position
            if symbol not in self.current_position or self.current_position[symbol]["quantity"] < quantity:
                self.logger.warning(f"Insufficient position for sell trade: {quantity} > {self.current_position.get(symbol, {}).get('quantity', 0)}")
                return None

            # Calculate proceeds
            total_proceeds = quantity * price
            commission = total_proceeds * self.commission_rate
            slippage = total_proceeds * self.slippage_rate
            net_proceeds = total_proceeds - commission - slippage

            # Execute the trade
            self.portfolio_value += net_proceeds

            # Update position
            position = self.current_position[symbol]
            old_quantity = position["quantity"]
            old_total_cost = position["total_cost"]

            # Update position
            new_quantity = old_quantity - quantity
            if new_quantity > 0:
                # Calculate remaining cost proportionally
                remaining_ratio = new_quantity / old_quantity
                new_total_cost = old_total_cost * remaining_ratio
                new_avg_price = new_total_cost / new_quantity
            else:
                # Position closed
                new_total_cost = 0
                new_avg_price = 0

            position["quantity"] = new_quantity
            position["avg_price"] = new_avg_price
            position["total_cost"] = new_total_cost

            # Remove position if quantity is zero
            if new_quantity <= 0:
                del self.current_position[symbol]

            # Calculate PnL
            pnl = net_proceeds - (quantity * position.get("avg_price", 0))
            pnl_percentage = (pnl / (quantity * position.get("avg_price", 1))) * 100 if position.get("avg_price", 0) > 0 else 0

            # Record trade
            trade_record = {
                "timestamp": timestamp,
                "symbol": symbol,
                "side": "SELL",
                "quantity": quantity,
                "price": price,
                "total_proceeds": total_proceeds,
                "commission": commission,
                "slippage": slippage,
                "net_proceeds": net_proceeds,
                "pnl": pnl,
                "pnl_percentage": pnl_percentage,
                "portfolio_value_after": self.portfolio_value,
            }
            self.trade_history.append(trade_record)

            # Record detailed trade if reporting is enabled
            if self.enable_detailed_reporting and self.reporter:
                await self._record_detailed_backtest_trade(
                    symbol=symbol,
                    side="short",
                    quantity=quantity,
                    price=price,
                    timestamp=timestamp,
                    total_proceeds=total_proceeds,
                    net_proceeds=net_proceeds,
                    pnl=pnl,
                    pnl_percentage=pnl_percentage,
                    commission=commission,
                    slippage=slippage,
                    trade_metadata=trade_metadata,
                )

            return trade_record

        except Exception as e:
            self.logger.error(f"Error executing sell trade: {e}")
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="detailed backtest trade recording",
    )
    async def _record_detailed_backtest_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        trade_metadata: Dict[str, Any],
        **kwargs,
    ) -> None:
        """Record detailed backtest trade information."""
        try:
            # Prepare trade data
            trade_data = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "timestamp": timestamp.isoformat(),
                "exchange": "backtest",
                "leverage": trade_metadata.get("leverage", 1.0),
                "duration": trade_metadata.get("duration", "backtest"),
                "strategy": trade_metadata.get("strategy", "backtest_strategy"),
                "order_type": trade_metadata.get("order_type", "market"),
                "portfolio_percentage": trade_metadata.get("portfolio_percentage", 0.0),
                "risk_percentage": trade_metadata.get("risk_percentage", 0.0),
                "max_position_size": trade_metadata.get("max_position_size", 0.0),
                "position_ranking": trade_metadata.get("position_ranking", 0),
                "status": "closed" if side == "short" else "open",
                "execution_quality": trade_metadata.get("execution_quality", 0.0),
                "risk_metrics": trade_metadata.get("risk_metrics", {}),
                "notes": trade_metadata.get("notes"),
            }

            # Add PnL information
            if side == "long":
                trade_data.update({
                    "total_cost": kwargs.get("total_cost", 0.0),
                    "absolute_pnl": 0.0,
                    "percentage_pnl": 0.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.0,
                    "net_pnl": 0.0,
                })
            else:  # short/sell
                trade_data.update({
                    "total_proceeds": kwargs.get("total_proceeds", 0.0),
                    "net_proceeds": kwargs.get("net_proceeds", 0.0),
                    "absolute_pnl": kwargs.get("pnl", 0.0),
                    "percentage_pnl": kwargs.get("pnl_percentage", 0.0),
                    "realized_pnl": kwargs.get("pnl", 0.0),
                    "net_pnl": kwargs.get("pnl", 0.0),
                })

            # Add commission and slippage
            trade_data.update({
                "commission": kwargs.get("commission", 0.0),
                "slippage": kwargs.get("slippage", 0.0),
            })

            # Get market indicators (simulated for backtesting)
            market_indicators = trade_metadata.get("market_indicators", {})
            market_health = trade_metadata.get("market_health", {})
            ml_confidence = trade_metadata.get("ml_confidence", {})

            # Record the trade
            await self.reporter.record_trade(
                trade_data=trade_data,
                market_indicators=market_indicators,
                market_health=market_health,
                ml_confidence=ml_confidence,
            )

        except Exception as e:
            self.logger.error(f"Error recording detailed backtest trade: {e}")

    def _update_equity_curve(self) -> None:
        """Update equity curve and drawdown."""
        try:
            # Calculate current portfolio value
            current_value = self.portfolio_value
            
            # Add unrealized PnL from open positions
            for symbol, position in self.current_position.items():
                if position["quantity"] > 0:
                    # This is simplified - in real implementation you'd need current price
                    current_value += position["quantity"] * position["avg_price"]

            self.equity_curve.append(current_value)

            # Calculate drawdown
            peak = max(self.equity_curve)
            current_drawdown = (peak - current_value) / peak if peak > 0 else 0.0
            self.drawdown_curve.append(current_drawdown)

        except Exception as e:
            self.logger.error(f"Error updating equity curve: {e}")

    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        try:
            if not self.trade_history:
                return {
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                    "total_return": 0.0,
                }

            # Calculate basic metrics
            total_trades = len(self.trade_history)
            sell_trades = [t for t in self.trade_history if t["side"] == "SELL"]

            # Calculate P&L
            total_pnl = sum(t.get("pnl", 0) for t in sell_trades)
            total_cost = sum(t.get("total_cost", 0) for t in self.trade_history if t["side"] == "BUY")
            total_proceeds = sum(t.get("net_proceeds", 0) for t in sell_trades)

            # Calculate win rate
            profitable_trades = len([t for t in sell_trades if t.get("pnl", 0) > 0])
            win_rate = profitable_trades / len(sell_trades) if sell_trades else 0.0

            # Calculate max drawdown
            max_drawdown = max(self.drawdown_curve) if self.drawdown_curve else 0.0

            # Calculate Sharpe ratio
            if len(self.equity_curve) > 1:
                returns = []
                for i in range(1, len(self.equity_curve)):
                    ret = (self.equity_curve[i] - self.equity_curve[i - 1]) / self.equity_curve[i - 1]
                    returns.append(ret)

                if returns:
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    sharpe_ratio = avg_return / std_return if std_return > 0 else 0.0
                else:
                    sharpe_ratio = 0.0
            else:
                sharpe_ratio = 0.0

            # Calculate total return
            total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance

            return {
                "total_trades": total_trades,
                "sell_trades": len(sell_trades),
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "total_cost": total_cost,
                "total_proceeds": total_proceeds,
                "current_portfolio_value": self.portfolio_value,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "total_return": total_return,
                "final_equity": self.equity_curve[-1] if self.equity_curve else self.initial_balance,
            }

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="detailed analysis generation",
    )
    async def _generate_detailed_analysis(self) -> Dict[str, Any]:
        """Generate detailed analysis of backtest results."""
        try:
            if self.reporter:
                return await self.reporter.generate_detailed_report("backtest_analysis")
            return {}

        except Exception as e:
            self.logger.error(f"Error generating detailed analysis: {e}")
            return {}

    def get_backtest_results(self) -> Dict[str, Any]:
        """Get comprehensive backtest results."""
        return {
            "performance_metrics": self._calculate_performance_metrics(),
            "equity_curve": self.equity_curve,
            "drawdown_curve": self.drawdown_curve,
            "trade_history": self.trade_history,
            "current_positions": self.current_position,
            "final_portfolio_value": self.portfolio_value,
        }

    async def generate_backtest_report(
        self,
        report_type: str = "comprehensive",
        export_formats: List[str] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive backtest report."""
        try:
            if export_formats is None:
                export_formats = ["json", "csv", "html"]

            if self.reporter:
                return await self.reporter.generate_detailed_report(report_type, export_formats)
            else:
                # Fallback to basic report
                return await self._generate_basic_backtest_report(report_type, export_formats)

        except Exception as e:
            self.logger.error(f"Error generating backtest report: {e}")
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="basic backtest report generation",
    )
    async def _generate_basic_backtest_report(
        self,
        report_type: str,
        export_formats: List[str],
    ) -> Dict[str, Any]:
        """Generate basic backtest report when detailed reporter is not available."""
        try:
            # Get backtest results
            results = self.get_backtest_results()
            performance_metrics = results["performance_metrics"]

            report_data = {
                "report_type": f"backtest_{report_type}",
                "generated_at": datetime.now().isoformat(),
                "performance_metrics": performance_metrics,
                "equity_curve": results["equity_curve"],
                "drawdown_curve": results["drawdown_curve"],
                "trade_history": results["trade_history"],
                "current_positions": results["current_positions"],
                "final_portfolio_value": results["final_portfolio_value"],
            }

            # Export reports
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = "reports/backtesting"
            os.makedirs(report_dir, exist_ok=True)

            for format_type in export_formats:
                if format_type == "json":
                    filename = f"backtest_report_{timestamp}.json"
                    filepath = os.path.join(report_dir, filename)
                    with open(filepath, "w") as f:
                        json.dump(report_data, f, indent=2, default=str)
                    self.logger.info(f"✅ Exported backtest JSON report: {filepath}")

            return report_data

        except Exception as e:
            self.logger.error(f"Error generating basic backtest report: {e}")
            return {}

    def stop(self) -> None:
        """Stop backtesting."""
        self.is_running = False
        self.logger.info("✅ Enhanced Backtester stopped")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="enhanced backtester setup",
)
async def setup_enhanced_backtester(
    config: Dict[str, Any] | None = None,
) -> EnhancedBacktester | None:
    """
    Setup enhanced backtester.

    Args:
        config: Configuration dictionary

    Returns:
        EnhancedBacktester: Configured backtester instance
    """
    try:
        if config is None:
            config = {}

        backtester = EnhancedBacktester(config)
        success = await backtester.initialize()

        if success:
            return backtester
        else:
            return None

    except Exception as e:
        system_logger.error(f"Error setting up enhanced backtester: {e}")
        return None