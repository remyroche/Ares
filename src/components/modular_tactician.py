# src/components/modular_tactician.py

import asyncio
import uuid
from typing import Dict, Any, Optional
from datetime import datetime

from src.interfaces import ITactician, StrategyResult, AnalysisResult, TradeDecision, MarketData, EventType
from src.interfaces.base_interfaces import IExchangeClient, IStateManager, IPerformanceReporter, IEventBus
from src.utils.logger import system_logger
from src.config import settings, CONFIG
from src.utils.error_handler import (
    handle_errors,
    handle_data_processing_errors,
    handle_network_operations,
    handle_type_conversions,
    error_context,
    ErrorRecoveryStrategies,
    safe_numeric_operation
)

class ModularTactician(ITactician):
    """
    Modular implementation of the Tactician that implements the ITactician interface.
    Uses dependency injection and event-driven communication.
    """

    def __init__(self, exchange_client: IExchangeClient, state_manager: IStateManager,
                 performance_reporter: IPerformanceReporter, event_bus: Optional[IEventBus] = None):
        """
        Initialize the modular tactician.

        Args:
            exchange_client: Exchange client for trading
            state_manager: State manager for persistence
            performance_reporter: Performance reporter for logging
            event_bus: Optional event bus for communication
        """
        self.exchange = exchange_client
        self.state_manager = state_manager
        self.performance_reporter = performance_reporter
        self.event_bus = event_bus
        self.logger = system_logger.getChild('ModularTactician')
        self.config = settings.get("tactician", {})
        self.running = False
        self.current_position = None
        self.trade_history = []

        self.logger.info("ModularTactician initialized")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="modular_tactician_start"
    )
    async def start(self) -> None:
        """Start the modular tactician"""
        self.logger.info("Starting ModularTactician")
        self.running = True
        
        # Subscribe to strategy events if event bus is available
        if self.event_bus:
            await self.event_bus.subscribe(
                EventType.STRATEGY_FORMULATED,
                self._handle_strategy_result
            )
            
        self.logger.info("ModularTactician started")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="modular_tactician_stop"
    )
    async def stop(self) -> None:
        """Stop the modular tactician"""
        self.logger.info("Stopping ModularTactician")
        self.running = False
        
        # Unsubscribe from events if event bus is available
        if self.event_bus:
            await self.event_bus.unsubscribe(
                EventType.STRATEGY_FORMULATED,
                self._handle_strategy_result
            )
            
        self.logger.info("ModularTactician stopped")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="execute_trade_decision"
    )
    async def execute_trade_decision(self, strategy_result: StrategyResult, 
                                   analysis_result: AnalysisResult) -> Optional[TradeDecision]:
        """
        Execute trade decision based on strategy and analysis.
        
        Args:
            strategy_result: Strategy result from strategist
            analysis_result: Analysis result from analyst
            
        Returns:
            Trade decision if executed, None otherwise
        """
        if not self.running:
            self.logger.warning("Tactician not running, skipping trade execution")
            return None
            
        self.logger.debug(f"Executing trade decision for {strategy_result.symbol}")
        
        try:
            # Check exit conditions first
            if await self._check_exit_conditions(strategy_result, analysis_result):
                return await self._execute_close_position(strategy_result, analysis_result)
                
            # Check entry conditions
            if await self._check_entry_conditions(strategy_result, analysis_result):
                return await self._execute_open_position(strategy_result, analysis_result)
                
            return None
            
        except Exception as e:
            self.logger.error(f"Trade execution failed: {e}", exc_info=True)
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=0.0,
        context="calculate_position_size"
    )
    async def calculate_position_size(self, strategy_result: StrategyResult, 
                                   account_balance: float) -> float:
        """
        Calculate position size based on strategy and account balance.
        
        Args:
            strategy_result: Strategy result
            account_balance: Current account balance
            
        Returns:
            Position size
        """
        try:
            # Base position size from strategy
            base_size = strategy_result.max_notional_size
            
            # Adjust based on account balance
            risk_per_trade = strategy_result.risk_parameters.get("risk_per_trade", 0.02)
            max_position_value = account_balance * risk_per_trade
            
            # Calculate position size
            current_price = await self._get_current_price(strategy_result.symbol)
            if current_price > 0:
                position_size = min(base_size, max_position_value / current_price)
            else:
                position_size = base_size
                
            # Apply minimum position size
            min_position_size = self.config.get("min_position_size", 0.001)
            position_size = max(position_size, min_position_size)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="calculate_risk_parameters"
    )
    async def calculate_risk_parameters(self, strategy_result: StrategyResult, 
                                     market_data: MarketData) -> Dict[str, float]:
        """
        Calculate risk parameters based on strategy and market data.
        
        Args:
            strategy_result: Strategy result
            market_data: Current market data
            
        Returns:
            Risk parameters
        """
        try:
            risk_params = strategy_result.risk_parameters.copy()
            
            # Calculate ATR-based stop loss
            atr_stop_loss = await self._calculate_atr_stop_loss(market_data)
            if atr_stop_loss > 0:
                risk_params["stop_loss_pct"] = atr_stop_loss
                
            # Calculate ATR-based take profit
            atr_take_profit = await self._calculate_atr_take_profit(market_data)
            if atr_take_profit > 0:
                risk_params["take_profit_pct"] = atr_take_profit
                
            return risk_params
            
        except Exception as e:
            self.logger.error(f"Error calculating risk parameters: {e}")
            return strategy_result.risk_parameters

    async def _handle_strategy_result(self, event) -> None:
        """Handle strategy result events"""
        strategy_result = event.data
        # Get latest analysis result from state
        analysis_result = self.state_manager.get_state("latest_analysis_result")
        if analysis_result:
            await self.execute_trade_decision(strategy_result, analysis_result)

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="check_entry_conditions"
    )
    async def _check_entry_conditions(self, strategy_result: StrategyResult, 
                                   analysis_result: AnalysisResult) -> bool:
        """Check if entry conditions are met"""
        try:
            # Check if we have a position bias
            if strategy_result.position_bias == "NEUTRAL":
                return False
                
            # Check confidence threshold
            if analysis_result.confidence < self.config.get("min_confidence", 0.6):
                return False
                
            # Check if we already have a position
            if self.current_position:
                return False
                
            # Check market conditions
            market_conditions = strategy_result.market_conditions
            if market_conditions.get("volatility") == "HIGH" and not self.config.get("allow_high_volatility", False):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking entry conditions: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="check_exit_conditions"
    )
    async def _check_exit_conditions(self, strategy_result: StrategyResult, 
                                   analysis_result: AnalysisResult) -> bool:
        """Check if exit conditions are met"""
        try:
            if not self.current_position:
                return False
                
            # Check for stop loss or take profit
            current_price = await self._get_current_price(strategy_result.symbol)
            if not current_price:
                return False
                
            position = self.current_position
            entry_price = position.get("entry_price", 0)
            
            if entry_price <= 0:
                return False
                
            # Calculate P&L
            if position.get("side") == "LONG":
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
                
            # Check stop loss
            stop_loss_pct = strategy_result.risk_parameters.get("stop_loss_pct", 0.05)
            if pnl_pct <= -stop_loss_pct:
                self.logger.info(f"Stop loss triggered: {pnl_pct:.2%}")
                return True
                
            # Check take profit
            take_profit_pct = strategy_result.risk_parameters.get("take_profit_pct", 0.10)
            if pnl_pct >= take_profit_pct:
                self.logger.info(f"Take profit triggered: {pnl_pct:.2%}")
                return True
                
            # Check for signal reversal
            if strategy_result.position_bias == "NEUTRAL":
                self.logger.info("Signal reversal detected")
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="execute_open_position"
    )
    async def _execute_open_position(self, strategy_result: StrategyResult, 
                                   analysis_result: AnalysisResult) -> Optional[TradeDecision]:
        """Execute opening a new position"""
        try:
            symbol = strategy_result.symbol
            side = "BUY" if strategy_result.position_bias == "LONG" else "SELL"
            
            # Get account balance
            account_info = await self.exchange.get_account_info()
            balance = account_info.get("totalWalletBalance", 0)
            
            # Calculate position size
            position_size = await self.calculate_position_size(strategy_result, balance)
            
            # Get current price
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return None
                
            # Calculate leverage
            leverage = await self._determine_leverage(strategy_result, analysis_result)
            
            # Calculate risk parameters
            risk_params = await self.calculate_risk_parameters(strategy_result, 
                                                            MarketData(symbol=symbol, timestamp=datetime.now(),
                                                                      open=current_price, high=current_price,
                                                                      low=current_price, close=current_price,
                                                                      volume=0, interval="1m"))
            
            # Create trade decision
            trade_decision = TradeDecision(
                timestamp=datetime.now(),
                symbol=symbol,
                action=f"OPEN_{side}",
                quantity=position_size,
                price=current_price,
                leverage=leverage,
                stop_loss=current_price * (1 - risk_params["stop_loss_pct"]) if side == "BUY" else current_price * (1 + risk_params["stop_loss_pct"]),
                take_profit=current_price * (1 + risk_params["take_profit_pct"]) if side == "BUY" else current_price * (1 - risk_params["take_profit_pct"]),
                confidence=analysis_result.confidence,
                risk_score=analysis_result.risk_metrics.get("liquidation_risk", 0.5)
            )
            
            # Execute the trade
            if await self._execute_trade(trade_decision):
                self.current_position = {
                    "symbol": symbol,
                    "side": side,
                    "entry_price": current_price,
                    "quantity": position_size,
                    "leverage": leverage,
                    "entry_time": datetime.now(),
                    "trade_id": str(uuid.uuid4())
                }
                
                # Publish trade executed event
                if self.event_bus:
                    await self.event_bus.publish(
                        EventType.TRADE_EXECUTED,
                        trade_decision,
                        "ModularTactician"
                    )
                    
                return trade_decision
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error executing open position: {e}", exc_info=True)
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="execute_close_position"
    )
    async def _execute_close_position(self, strategy_result: StrategyResult, 
                                   analysis_result: AnalysisResult) -> Optional[TradeDecision]:
        """Execute closing an existing position"""
        try:
            if not self.current_position:
                return None
                
            symbol = self.current_position["symbol"]
            side = "SELL" if self.current_position["side"] == "LONG" else "BUY"
            
            # Get current price
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return None
                
            # Create trade decision
            trade_decision = TradeDecision(
                timestamp=datetime.now(),
                symbol=symbol,
                action=f"CLOSE_{self.current_position['side']}",
                quantity=self.current_position["quantity"],
                price=current_price,
                leverage=self.current_position["leverage"],
                stop_loss=0.0,
                take_profit=0.0,
                confidence=analysis_result.confidence,
                risk_score=analysis_result.risk_metrics.get("liquidation_risk", 0.5)
            )
            
            # Execute the trade
            if await self._execute_trade(trade_decision):
                # Log the trade
                await self._log_trade(trade_decision, self.current_position)
                
                # Clear current position
                self.current_position = None
                
                # Publish trade executed event
                if self.event_bus:
                    await self.event_bus.publish(
                        EventType.TRADE_EXECUTED,
                        trade_decision,
                        "ModularTactician"
                    )
                    
                return trade_decision
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error executing close position: {e}", exc_info=True)
            return None

    @handle_network_operations(
        exceptions=(Exception,),
        default_return=False,
        context="execute_trade"
    )
    async def _execute_trade(self, trade_decision: TradeDecision) -> bool:
        """Execute a trade on the exchange"""
        try:
            # Create order
            order_result = await self.exchange.create_order(
                symbol=trade_decision.symbol,
                side=trade_decision.action.split("_")[1],  # BUY or SELL
                quantity=trade_decision.quantity,
                price=trade_decision.price,
                order_type="MARKET"
            )
            
            if order_result and order_result.get("status") == "FILLED":
                self.logger.info(f"Trade executed: {trade_decision.action} {trade_decision.quantity} {trade_decision.symbol} at {trade_decision.price}")
                return True
            else:
                self.logger.warning(f"Trade execution failed: {order_result}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return False

    @handle_network_operations(
        exceptions=(Exception,),
        default_return=0.0,
        context="get_current_price"
    )
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            # Get latest kline data
            klines = await self.exchange.get_klines(symbol, "1m", 1)
            if klines and len(klines) > 0:
                return klines[0].close
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return 0.0

    @handle_data_processing_errors(
        exceptions=(ValueError, TypeError, ZeroDivisionError),
        default_return=1.0,
        context="determine_leverage"
    )
    async def _determine_leverage(self, strategy_result: StrategyResult, 
                                analysis_result: AnalysisResult) -> float:
        """Determine leverage based on strategy and analysis"""
        try:
            base_leverage = strategy_result.leverage_cap
            
            # Adjust based on confidence
            confidence = analysis_result.confidence
            if confidence > 0.8:
                leverage_multiplier = 1.2
            elif confidence > 0.6:
                leverage_multiplier = 1.0
            else:
                leverage_multiplier = 0.8
                
            # Adjust based on risk score
            risk_score = analysis_result.risk_metrics.get("liquidation_risk", 0.5)
            if risk_score > 0.7:
                risk_multiplier = 0.5
            elif risk_score < 0.3:
                risk_multiplier = 1.2
            else:
                risk_multiplier = 1.0
                
            leverage = base_leverage * leverage_multiplier * risk_multiplier
            
            # Apply limits
            max_leverage = self.config.get("max_leverage", 10.0)
            min_leverage = self.config.get("min_leverage", 1.0)
            
            return max(min_leverage, min(leverage, max_leverage))
            
        except Exception as e:
            self.logger.error(f"Error determining leverage: {e}")
            return 1.0

    @handle_data_processing_errors(
        exceptions=(ValueError, TypeError, ZeroDivisionError),
        default_return=0.05,
        context="calculate_atr_stop_loss"
    )
    async def _calculate_atr_stop_loss(self, market_data: MarketData) -> float:
        """Calculate ATR-based stop loss"""
        try:
            # This would typically calculate ATR from historical data
            # For now, use a simple percentage
            return 0.05  # 5% default stop loss
        except Exception as e:
            self.logger.error(f"Error calculating ATR stop loss: {e}")
            return 0.05

    @handle_data_processing_errors(
        exceptions=(ValueError, TypeError, ZeroDivisionError),
        default_return=0.10,
        context="calculate_atr_take_profit"
    )
    async def _calculate_atr_take_profit(self, market_data: MarketData) -> float:
        """Calculate ATR-based take profit"""
        try:
            # This would typically calculate ATR from historical data
            # For now, use a simple percentage
            return 0.10  # 10% default take profit
        except Exception as e:
            self.logger.error(f"Error calculating ATR take profit: {e}")
            return 0.10

    async def _log_trade(self, trade_decision: TradeDecision, position: Dict[str, Any]) -> None:
        """Log trade details"""
        try:
            # Calculate P&L
            entry_price = position.get("entry_price", 0)
            exit_price = trade_decision.price
            quantity = trade_decision.quantity
            
            if position.get("side") == "LONG":
                pnl = (exit_price - entry_price) * quantity
            else:
                pnl = (entry_price - exit_price) * quantity
                
            # Log to performance reporter
            trade_data = {
                "trade_id": position.get("trade_id"),
                "symbol": trade_decision.symbol,
                "side": position.get("side"),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "pnl": pnl,
                "entry_time": position.get("entry_time"),
                "exit_time": trade_decision.timestamp,
                "leverage": position.get("leverage")
            }
            
            await self.performance_reporter.log_trade(trade_data)
            
            # Add to trade history
            self.trade_history.append(trade_data)
            
        except Exception as e:
            self.logger.error(f"Error logging trade: {e}") 