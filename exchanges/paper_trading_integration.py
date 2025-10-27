"""
Paper Trading Integration

This module provides integration between the trading launcher and paper trading
functionality, allowing seamless switching between live and paper trading modes.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from .paper_trading_wrapper import PaperTradingWrapper, create_paper_trading_wrapper
from .paper_trading_simulator import SimulatorConfig
from .base_exchange.exchange_interface import IExchange, OrderSide, OrderType


class PaperTradingIntegration:
    """
    Integration class that manages paper trading functionality
    and provides a unified interface for the trading launcher.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize paper trading integration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Paper trading components
        self.wrappers: Dict[str, PaperTradingWrapper] = {}
        self.trading_mode: str = "TRADE"
        self.is_active: bool = False
        
        # Configuration
        self.paper_config = config.get("paper_trading", {})
        self.simulator_config = self._create_simulator_config()
        
        self.logger.info("Paper Trading Integration initialized")
    
    def _create_simulator_config(self) -> SimulatorConfig:
        """Create simulator configuration from config"""
        return SimulatorConfig(
            maker_fee_rate=self.paper_config.get("maker_fee_rate", 0.001),
            taker_fee_rate=self.paper_config.get("taker_fee_rate", 0.001),
            base_slippage_rate=self.paper_config.get("base_slippage_rate", 0.0001),
            slippage_volatility_factor=self.paper_config.get("slippage_volatility_factor", 0.5),
            max_slippage_rate=self.paper_config.get("max_slippage_rate", 0.01),
            market_impact_factor=self.paper_config.get("market_impact_factor", 0.0001),
            min_volume_threshold=self.paper_config.get("min_volume_threshold", 1000.0),
            execution_delay_ms=self.paper_config.get("execution_delay_ms", (10, 100)),
            partial_fill_probability=self.paper_config.get("partial_fill_probability", 0.1),
            rejection_probability=self.paper_config.get("rejection_probability", 0.001),
            max_position_size=self.paper_config.get("max_position_size", 1000000.0),
            max_daily_trades=self.paper_config.get("max_daily_trades", 1000)
        )
    
    def register_exchange(self, exchange_name: str, exchange: IExchange) -> None:
        """
        Register an exchange for paper trading.
        
        Args:
            exchange_name: Name of the exchange
            exchange: Exchange interface instance
        """
        wrapper = create_paper_trading_wrapper(
            exchange=exchange,
            trading_mode=self.trading_mode,
            simulator_config=self.simulator_config
        )
        
        self.wrappers[exchange_name] = wrapper
        self.logger.info(f"Registered exchange '{exchange_name}' for paper trading")
    
    def set_trading_mode(self, mode: str) -> bool:
        """
        Set trading mode for all registered exchanges.
        
        Args:
            mode: "TRADE" or "PAPER"
            
        Returns:
            True if successful, False otherwise
        """
        if mode.upper() not in ["TRADE", "PAPER"]:
            self.logger.error(f"Invalid trading mode: {mode}")
            return False
        
        self.trading_mode = mode.upper()
        
        # Update all wrappers
        for wrapper in self.wrappers.values():
            wrapper.set_trading_mode(self.trading_mode)
        
        self.logger.info(f"Trading mode set to: {self.trading_mode}")
        return True
    
    def get_trading_mode(self) -> str:
        """Get current trading mode"""
        return self.trading_mode
    
    def is_paper_mode(self) -> bool:
        """Check if currently in paper trading mode"""
        return self.trading_mode == "PAPER"
    
    def is_trade_mode(self) -> bool:
        """Check if currently in live trading mode"""
        return self.trading_mode == "TRADE"
    
    async def initialize(self) -> bool:
        """
        Initialize paper trading integration.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Initializing Paper Trading Integration...")
            
            # Initialize all registered exchanges
            for exchange_name, wrapper in self.wrappers.items():
                try:
                    await wrapper.initialize()
                    self.logger.info(f"✅ Initialized exchange: {exchange_name}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to initialize exchange {exchange_name}: {e}")
                    return False
            
            self.is_active = True
            self.logger.info("✅ Paper Trading Integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Paper Trading Integration: {e}")
            return False
    
    async def close(self) -> None:
        """Close paper trading integration"""
        try:
            self.logger.info("Closing Paper Trading Integration...")
            
            # Close all registered exchanges
            for exchange_name, wrapper in self.wrappers.items():
                try:
                    await wrapper.close()
                    self.logger.info(f"✅ Closed exchange: {exchange_name}")
                except Exception as e:
                    self.logger.error(f"❌ Error closing exchange {exchange_name}: {e}")
            
            self.is_active = False
            self.logger.info("✅ Paper Trading Integration closed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error closing Paper Trading Integration: {e}")
    
    async def execute_trade(
        self,
        exchange_name: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "market",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a trade through the specified exchange.
        
        Args:
            exchange_name: Name of the exchange
            symbol: Trading symbol
            side: Trade side ("buy" or "sell")
            quantity: Trade quantity
            price: Trade price
            order_type: Order type ("market" or "limit")
            **kwargs: Additional order parameters
            
        Returns:
            Trade execution result
        """
        try:
            if exchange_name not in self.wrappers:
                raise ValueError(f"Exchange '{exchange_name}' not registered")
            
            wrapper = self.wrappers[exchange_name]
            
            # Convert string parameters to enums
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            order_type_enum = OrderType.MARKET if order_type.lower() == "market" else OrderType.LIMIT
            
            # Execute trade
            result = await wrapper.create_order(
                symbol=symbol,
                side=order_side,
                order_type=order_type_enum,
                quantity=quantity,
                price=price if order_type.lower() == "limit" else None,
                **kwargs
            )
            
            self.logger.info(f"Trade executed: {side} {quantity} {symbol} @ {price}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            raise
    
    async def cancel_trade(self, exchange_name: str, order_id: str) -> Dict[str, Any]:
        """
        Cancel a trade through the specified exchange.
        
        Args:
            exchange_name: Name of the exchange
            order_id: Order ID to cancel
            
        Returns:
            Cancellation result
        """
        try:
            if exchange_name not in self.wrappers:
                raise ValueError(f"Exchange '{exchange_name}' not registered")
            
            wrapper = self.wrappers[exchange_name]
            result = await wrapper.cancel_order(order_id)
            
            self.logger.info(f"Trade cancelled: {order_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error cancelling trade: {e}")
            raise
    
    async def get_trade_status(self, exchange_name: str, order_id: str) -> Dict[str, Any]:
        """
        Get trade status from the specified exchange.
        
        Args:
            exchange_name: Name of the exchange
            order_id: Order ID
            
        Returns:
            Trade status information
        """
        try:
            if exchange_name not in self.wrappers:
                raise ValueError(f"Exchange '{exchange_name}' not registered")
            
            wrapper = self.wrappers[exchange_name]
            return await wrapper.get_order_status(order_id)
            
        except Exception as e:
            self.logger.error(f"Error getting trade status: {e}")
            raise
    
    async def get_market_data(
        self,
        exchange_name: str,
        symbol: str,
        data_type: str = "ticker",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get market data from the specified exchange.
        
        Args:
            exchange_name: Name of the exchange
            symbol: Trading symbol
            data_type: Type of data ("ticker", "klines", "orderbook")
            **kwargs: Additional parameters
            
        Returns:
            Market data
        """
        try:
            if exchange_name not in self.wrappers:
                raise ValueError(f"Exchange '{exchange_name}' not registered")
            
            wrapper = self.wrappers[exchange_name]
            
            if data_type == "ticker":
                return await wrapper.get_ticker(symbol)
            elif data_type == "klines":
                return await wrapper.get_klines(symbol, **kwargs)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
                
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            raise
    
    def get_positions(self, exchange_name: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get positions from exchanges.
        
        Args:
            exchange_name: Optional exchange name filter
            
        Returns:
            Dictionary of positions by exchange
        """
        positions = {}
        
        for name, wrapper in self.wrappers.items():
            if exchange_name is None or name == exchange_name:
                if wrapper.is_paper_mode():
                    positions[name] = wrapper.get_positions()
                else:
                    positions[name] = []  # Live trading positions not tracked here
        
        return positions
    
    def get_trade_history(self, exchange_name: Optional[str] = None, 
                         symbol: Optional[str] = None, limit: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get trade history from exchanges.
        
        Args:
            exchange_name: Optional exchange name filter
            symbol: Optional symbol filter
            limit: Maximum number of trades
            
        Returns:
            Dictionary of trade history by exchange
        """
        history = {}
        
        for name, wrapper in self.wrappers.items():
            if exchange_name is None or name == exchange_name:
                if wrapper.is_paper_mode():
                    history[name] = wrapper.get_trade_history(symbol, limit)
                else:
                    history[name] = []  # Live trading history not tracked here
        
        return history
    
    def get_performance_metrics(self, exchange_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics from exchanges.
        
        Args:
            exchange_name: Optional exchange name filter
            
        Returns:
            Dictionary of performance metrics by exchange
        """
        metrics = {}
        
        for name, wrapper in self.wrappers.items():
            if exchange_name is None or name == exchange_name:
                if wrapper.is_paper_mode():
                    metrics[name] = wrapper.get_performance_metrics()
                else:
                    metrics[name] = {"mode": "live_trading", "metrics_available": False}
        
        return metrics
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get overall portfolio summary"""
        try:
            all_positions = self.get_positions()
            all_metrics = self.get_performance_metrics()
            
            # Aggregate data across exchanges
            total_portfolio_value = 0.0
            total_realized_pnl = 0.0
            total_commission = 0.0
            total_trades = 0
            active_positions = 0
            
            for exchange_name, metrics in all_metrics.items():
                if metrics.get("mode") != "live_trading":
                    total_portfolio_value += metrics.get("current_portfolio_value", 0.0)
                    total_realized_pnl += metrics.get("total_realized_pnl", 0.0)
                    total_commission += metrics.get("total_commission", 0.0)
                    total_trades += metrics.get("total_trades", 0)
                    active_positions += len(all_positions.get(exchange_name, []))
            
            return {
                "trading_mode": self.trading_mode,
                "total_portfolio_value": total_portfolio_value,
                "total_realized_pnl": total_realized_pnl,
                "total_commission": total_commission,
                "total_trades": total_trades,
                "active_positions": active_positions,
                "exchanges": list(self.wrappers.keys()),
                "is_active": self.is_active,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    async def generate_comprehensive_report(
        self,
        report_type: str = "comprehensive",
        export_formats: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive trading report.
        
        Args:
            report_type: Type of report to generate
            export_formats: List of export formats
            
        Returns:
            Report data
        """
        try:
            if export_formats is None:
                export_formats = ["json"]
            
            # Gather all data
            portfolio_summary = self.get_portfolio_summary()
            positions = self.get_positions()
            trade_history = self.get_trade_history()
            performance_metrics = self.get_performance_metrics()
            
            report_data = {
                "report_type": report_type,
                "generated_at": datetime.now().isoformat(),
                "trading_mode": self.trading_mode,
                "portfolio_summary": portfolio_summary,
                "positions": positions,
                "trade_history": trade_history,
                "performance_metrics": performance_metrics,
                "integration_status": {
                    "is_active": self.is_active,
                    "registered_exchanges": list(self.wrappers.keys()),
                    "trading_mode": self.trading_mode
                }
            }
            
            # Export reports if requested
            if export_formats:
                await self._export_report(report_data, export_formats)
            
            return report_data
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            return {}
    
    async def _export_report(self, report_data: Dict[str, Any], formats: List[str]) -> None:
        """Export report in specified formats"""
        try:
            import json
            import os
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = "reports/paper_trading"
            os.makedirs(report_dir, exist_ok=True)
            
            for format_type in formats:
                if format_type == "json":
                    filename = f"paper_trading_report_{timestamp}.json"
                    filepath = os.path.join(report_dir, filename)
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(report_data, f, indent=2, default=str)
                    self.logger.info(f"✅ Exported report: {filepath}")
                    
        except Exception as e:
            self.logger.error(f"Error exporting report: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            "is_active": self.is_active,
            "trading_mode": self.trading_mode,
            "registered_exchanges": list(self.wrappers.keys()),
            "simulator_config": {
                "maker_fee_rate": self.simulator_config.maker_fee_rate,
                "taker_fee_rate": self.simulator_config.taker_fee_rate,
                "base_slippage_rate": self.simulator_config.base_slippage_rate,
                "max_slippage_rate": self.simulator_config.max_slippage_rate,
                "max_position_size": self.simulator_config.max_position_size,
                "max_daily_trades": self.simulator_config.max_daily_trades
            }
        }


async def setup_paper_trading_integration(config: Dict[str, Any]) -> Optional[PaperTradingIntegration]:
    """
    Setup paper trading integration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PaperTradingIntegration instance or None if setup fails
    """
    try:
        integration = PaperTradingIntegration(config)
        success = await integration.initialize()
        
        if success:
            return integration
        else:
            return None
            
    except Exception as e:
        logging.getLogger(__name__).error(f"Error setting up paper trading integration: {e}")
        return None