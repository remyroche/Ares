"""
Trading Engine

Main engine that coordinates all trading operations.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Awaitable
import logging

from .config import TradingConfig, TradingMode
from .order_manager import OrderManager, Order
from .data_streamer import DataStreamer, StreamData
from .risk_manager import RiskManager
from ..src.interfaces.base_interfaces import TradeDecision, AnalysisResult, StrategyResult, MarketData


class TradingEngine:
    """Main trading engine that coordinates all trading operations"""
    
    def __init__(self, config: TradingConfig, exchange_client: Any):
        self.config = config
        self.exchange_client = exchange_client
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.order_manager = OrderManager(config, exchange_client)
        self.data_streamer = DataStreamer(config, exchange_client)
        self.risk_manager = RiskManager(config, exchange_client)
        
        # Trading state
        self._running = False
        self._trading_active = False
        self._last_analysis: Dict[str, AnalysisResult] = {}
        self._last_strategy: Dict[str, StrategyResult] = {}
        
        # Event handlers
        self.trading_handlers: Dict[str, List[Callable[[Any], Awaitable[None]]]] = {
            "on_trade_executed": [],
            "on_risk_violation": [],
            "on_data_received": [],
            "on_error": []
        }
        
        # Performance tracking
        self.trade_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        
    async def start(self) -> None:
        """Start the trading engine"""
        if self._running:
            return
            
        self.logger.info("Starting trading engine...")
        
        try:
            # Initialize exchange client
            if hasattr(self.exchange_client, '_initialize_exchange'):
                await self.exchange_client._initialize_exchange()
            
            # Start components
            await self.order_manager.start()
            await self.data_streamer.start()
            await self.risk_manager.start()
            
            # Register event handlers
            self._register_internal_handlers()
            
            # Start trading
            self._running = True
            self._trading_active = True
            
            self.logger.info("Trading engine started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start trading engine: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the trading engine"""
        if not self._running:
            return
            
        self.logger.info("Stopping trading engine...")
        
        self._running = False
        self._trading_active = False
        
        # Stop components
        await self.order_manager.stop()
        await self.data_streamer.stop()
        await self.risk_manager.stop()
        
        # Close exchange connection
        if hasattr(self.exchange_client, 'close'):
            await self.exchange_client.close()
        
        self.logger.info("Trading engine stopped")
    
    def register_handler(self, event_type: str, handler: Callable[[Any], Awaitable[None]]) -> None:
        """Register event handler"""
        if event_type in self.trading_handlers:
            self.trading_handlers[event_type].append(handler)
    
    async def execute_trade_decision(self, decision: TradeDecision) -> Optional[Order]:
        """Execute a trade decision"""
        if not self._trading_active:
            self.logger.warning("Trading is not active, ignoring trade decision")
            return None
        
        try:
            # Validate trade decision with risk manager
            try:
                is_valid, message = await self.risk_manager.validate_trade_decision(decision)

                if not is_valid:
                    self.logger.error(f"❌ Trade decision rejected by risk manager: {message}")
                    await self._notify_handlers("on_risk_violation", {
                        "type": "trade_rejected",
                        "message": message,
                        "decision": decision,
                        "symbol": decision.symbol,
                        "action": decision.action,
                        "quantity": decision.quantity
                    })
                    return None
            except Exception as e:
                self.logger.error(f"❌ Risk validation failed: {e}")
                self.logger.warning("⚠️ Proceeding with trade despite risk validation failure - RISK MANAGEMENT DISABLED")
                # Continue with trade execution
            
            # Create order from decision
            order = await self.order_manager.create_order_from_decision(decision)
            
            # Track trade
            trade_record = {
                "timestamp": datetime.now(),
                "decision": decision,
                "order": order,
                "symbol": decision.symbol,
                "action": decision.action,
                "quantity": decision.quantity,
                "price": decision.price,
                "confidence": decision.confidence,
                "risk_score": decision.risk_score
            }
            self.trade_history.append(trade_record)
            
            # Notify handlers
            await self._notify_handlers("on_trade_executed", trade_record)
            
            self.logger.info(f"Trade executed: {decision.symbol} {decision.action} {decision.quantity}")
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error executing trade decision: {e}")
            await self._notify_handlers("on_error", {
                "type": "trade_execution_error",
                "error": str(e),
                "decision": decision
            })
            return None
    
    async def update_analysis(self, symbol: str, analysis: AnalysisResult) -> None:
        """Update analysis result for a symbol"""
        self._last_analysis[symbol] = analysis
        self.logger.debug(f"Analysis updated for {symbol}: {analysis.signal}")
    
    async def update_strategy(self, symbol: str, strategy: StrategyResult) -> None:
        """Update strategy result for a symbol"""
        self._last_strategy[symbol] = strategy
        self.logger.debug(f"Strategy updated for {symbol}: {strategy.position_bias}")
    
    async def get_trading_status(self) -> Dict[str, Any]:
        """Get current trading status"""
        order_status = await self.order_manager.get_performance_metrics()
        streaming_status = await self.data_streamer.get_streaming_status()
        risk_summary = await self.risk_manager.get_risk_summary()
        
        return {
            "running": self._running,
            "trading_active": self._trading_active,
            "mode": self.config.mode.value,
            "symbols": self.config.symbols,
            "order_manager": order_status,
            "data_streamer": streaming_status,
            "risk_manager": risk_summary,
            "total_trades": len(self.trade_history),
            "last_update": datetime.now().isoformat()
        }
    
    async def get_position_summary(self) -> Dict[str, Any]:
        """Get position summary"""
        positions = {}
        
        for symbol in self.config.symbols:
            try:
                # Get current position from exchange
                account_info = await self.exchange_client.get_account_info()
                
                # Get risk metrics
                risk_metrics = await self.risk_manager.calculate_risk_metrics(symbol)
                
                # Get latest market data
                ticker = await self.exchange_client.get_ticker(symbol)
                current_price = float(ticker.get("last", 0)) if ticker else 0.0
                
                positions[symbol] = {
                    "current_position": risk_metrics.current_position,
                    "position_value": risk_metrics.position_value,
                    "current_price": current_price,
                    "daily_pnl": risk_metrics.daily_pnl,
                    "unrealized_pnl": risk_metrics.unrealized_pnl,
                    "leverage": risk_metrics.leverage,
                    "risk_score": risk_metrics.risk_score,
                    "last_analysis": self._last_analysis.get(symbol),
                    "last_strategy": self._last_strategy.get(symbol)
                }
                
            except Exception as e:
                self.logger.error(f"❌ Error getting position for {symbol}: {e}")
                self.logger.warning(f"⚠️ Using default values for {symbol} - position data may be inaccurate")
                positions[symbol] = {
                    "error": str(e),
                    "current_position": 0.0,
                    "position_value": 0.0,
                    "current_price": 0.0,
                    "daily_pnl": 0.0,
                    "unrealized_pnl": 0.0,
                    "leverage": 0.0,
                    "risk_score": 0.0
                }
        
        return positions
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self.trade_history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "average_trade_size": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0
            }
        
        # Calculate basic metrics
        total_trades = len(self.trade_history)
        winning_trades = 0
        losing_trades = 0
        total_pnl = 0.0
        trade_sizes = []
        
        for trade in self.trade_history:
            # Simplified PnL calculation (would need more sophisticated logic)
            pnl = trade.get("pnl", 0.0)
            total_pnl += pnl
            
            if pnl > 0:
                winning_trades += 1
            elif pnl < 0:
                losing_trades += 1
            
            trade_sizes.append(abs(trade["quantity"]))
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        average_trade_size = sum(trade_sizes) / len(trade_sizes) if trade_sizes else 0.0
        
        # Get risk metrics
        risk_summary = await self.risk_manager.get_risk_summary()
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "average_trade_size": average_trade_size,
            "max_drawdown": risk_summary.get("max_drawdown", 0.0),
            "sharpe_ratio": risk_summary.get("sharpe_ratio", 0.0),
            "daily_pnl": risk_summary.get("total_daily_pnl", 0.0),
            "risk_violations": risk_summary.get("risk_violations_count", 0),
            "timestamp": datetime.now().isoformat()
        }
    
    async def pause_trading(self) -> None:
        """Pause trading operations"""
        self._trading_active = False
        self.logger.info("Trading paused")
    
    async def resume_trading(self) -> None:
        """Resume trading operations"""
        if self._running:
            self._trading_active = True
            self.logger.info("Trading resumed")
        else:
            self.logger.warning("Cannot resume trading - engine is not running")
    
    async def emergency_stop(self) -> None:
        """Emergency stop - cancel all orders and pause trading"""
        self.logger.warning("Emergency stop triggered!")
        
        try:
            # Cancel all active orders
            active_orders = await self.order_manager.get_active_orders()
            for order in active_orders:
                await self.order_manager.cancel_order(order.id)
            
            # Pause trading
            await self.pause_trading()
            
            self.logger.info("Emergency stop completed - all orders cancelled and trading paused")
            
        except Exception as e:
            self.logger.error(f"❌ Error during emergency stop: {e}")
            self.logger.warning("⚠️ Emergency stop completed with errors - some orders may not have been cancelled")
    
    def _register_internal_handlers(self) -> None:
        """Register internal event handlers"""
        # Order manager handlers
        self.order_manager.register_handler("on_order_filled", self._on_order_filled)
        self.order_manager.register_handler("on_order_failed", self._on_order_failed)
        
        # Data streamer handlers
        self.data_streamer.register_handler("ticker", self._on_ticker_data)
        self.data_streamer.register_handler("trade", self._on_trade_data)
        self.data_streamer.register_handler("kline", self._on_kline_data)
        
        # Risk manager handlers (would be implemented if risk manager had events)
    
    async def _on_order_filled(self, order: Order) -> None:
        """Handle order filled event"""
        self.logger.info(f"Order filled: {order.id}")
        
        # Update position in risk manager
        await self.risk_manager.update_position(
            order.symbol,
            order.filled_quantity if order.side.value == "buy" else -order.filled_quantity,
            order.average_price or order.price or 0.0
        )
        
        # Notify handlers
        await self._notify_handlers("on_trade_executed", {
            "type": "order_filled",
            "order": order,
            "timestamp": datetime.now()
        })
    
    async def _on_order_failed(self, order: Order) -> None:
        """Handle order failed event"""
        self.logger.warning(f"Order failed: {order.id} - {order.error_message}")
        
        # Notify handlers
        await self._notify_handlers("on_error", {
            "type": "order_failed",
            "order": order,
            "error": order.error_message,
            "timestamp": datetime.now()
        })
    
    async def _on_ticker_data(self, stream_data: StreamData) -> None:
        """Handle ticker data"""
        self.logger.debug(f"Ticker data received: {stream_data.symbol}")
        
        # Notify handlers
        await self._notify_handlers("on_data_received", {
            "type": "ticker",
            "symbol": stream_data.symbol,
            "data": stream_data.data,
            "timestamp": stream_data.timestamp
        })
    
    async def _on_trade_data(self, stream_data: StreamData) -> None:
        """Handle trade data"""
        self.logger.debug(f"Trade data received: {stream_data.symbol}")
        
        # Notify handlers
        await self._notify_handlers("on_data_received", {
            "type": "trade",
            "symbol": stream_data.symbol,
            "data": stream_data.data,
            "timestamp": stream_data.timestamp
        })
    
    async def _on_kline_data(self, stream_data: StreamData) -> None:
        """Handle kline data"""
        self.logger.debug(f"Kline data received: {stream_data.symbol}")
        
        # Convert to MarketData
        market_data = MarketData(
            symbol=stream_data.symbol,
            timestamp=stream_data.timestamp,
            open=stream_data.data["open"],
            high=stream_data.data["high"],
            low=stream_data.data["low"],
            close=stream_data.data["close"],
            volume=stream_data.data["volume"],
            interval=stream_data.data["interval"]
        )
        
        # Notify handlers
        await self._notify_handlers("on_data_received", {
            "type": "kline",
            "symbol": stream_data.symbol,
            "market_data": market_data,
            "timestamp": stream_data.timestamp
        })
    
    async def _notify_handlers(self, event_type: str, data: Any) -> None:
        """Notify registered handlers"""
        if event_type in self.trading_handlers:
            for handler in self.trading_handlers[event_type]:
                try:
                    await handler(data)
                except Exception as e:
                    self.logger.error(f"❌ Error in trading handler: {e}")
                    self.logger.warning("⚠️ Handler failed - continuing with other handlers")
    
    async def get_trade_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history"""
        trades = self.trade_history
        
        if symbol:
            trades = [trade for trade in trades if trade["symbol"] == symbol]
        
        return trades[-limit:] if limit > 0 else trades