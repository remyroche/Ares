"""
Paper Trading Simulator

Main simulator that coordinates order simulation with realistic fills,
fees, slippage, and position management.
"""

import asyncio
import random
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, date
import uuid
import logging

from .config import SimulatorConfig
from .fee_calculator import FeeCalculator
from .slippage_calculator import SlippageCalculator
from .order_validator import OrderValidator
from .position_manager import PositionManager, Position
from .persistence import SimulatorPersistence
from src.trading.reporting.trade_reporting_manager import (
    TradeRecord, trade_reporting_manager, generate_daily_recap
)
from src.utils.tprint import tprint_info, tprint_success, tprint_error, tprint_debug


class PaperTradingSimulator:
    """
    Main paper trading simulator.
    
    Coordinates order simulation with realistic market fills, fees, slippage,
    and position management. Uses order book data for accurate price discovery.
    """
    
    def __init__(
        self,
        config: SimulatorConfig,
        exchange: str,
        initial_balance: float,
        direction_constraint: str = "both",
        simulator_id: Optional[str] = None,
        db_path: str = "simulator_state.db"
    ):
        """
        Initialize paper trading simulator.
        
        Args:
            config: Simulator configuration
            exchange: Exchange name (e.g., "binance")
            initial_balance: Starting balance in USDT
            direction_constraint: Trading direction ("long", "short", "both")
            simulator_id: Unique simulator identifier (generated if None)
            db_path: Path to SQLite database
        """
        self.config = config
        self.config.validate()
        
        self.exchange = exchange
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.direction_constraint = direction_constraint
        self.simulator_id = simulator_id or str(uuid.uuid4())
        self.db_path = db_path
        
        # Initialize components
        self.fee_calculator = FeeCalculator(config)
        self.slippage_calculator = SlippageCalculator(config)
        self.order_validator = OrderValidator(config)
        self.position_manager = PositionManager(config)
        self.persistence = SimulatorPersistence(db_path)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize state
        self._save_state()
        
        self.logger.info(
            f"Paper trading simulator initialized: {self.simulator_id} "
            f"({exchange}, balance={initial_balance:.2f}, direction={direction_constraint})"
        )
    
    async def simulate_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float],
        order_book: Dict[str, Any],
        trading_signal_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Simulate an order execution with realistic fills.
        
        Args:
            symbol: Trading symbol
            side: Order side ("buy", "sell", "long", "short")
            order_type: Order type ("market", "limit")
            quantity: Order quantity
            price: Limit price (None for market orders)
            order_book: Current order book data
            trading_signal_metadata: Metadata about trading signal that triggered this order
            
        Returns:
            Simulated order response matching exchange format
        """
        start_time = datetime.now()
        
        # Simulate latency
        if self.config.enable_latency_simulation:
            latency_ms = random.randint(
                self.config.latency_range_ms[0],
                self.config.latency_range_ms[1]
            )
            await asyncio.sleep(latency_ms / 1000.0)
        else:
            latency_ms = 0
        
        try:
            # Validate order
            current_positions = {
                pos.symbol: {
                    "quantity": pos.quantity,
                    "direction": pos.direction,
                    "avg_entry_price": pos.avg_entry_price
                }
                for pos_list in self.position_manager.positions.values()
                for pos in pos_list
            }
            
            # Get current price for validation
            reference_price = self.slippage_calculator._get_best_price(
                order_book, "asks" if side.lower() in ["buy", "long"] else "bids"
            )
            
            validation = self.order_validator.validate_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price or reference_price,
                current_balance=self.current_balance,
                current_positions=current_positions,
                direction_constraint=self.direction_constraint,
                current_price=reference_price
            )
            
            if not validation.is_valid:
                self.logger.warning(f"Order validation failed: {validation.message}")
                return self._create_rejected_response(
                    symbol, side, order_type, quantity, validation.message
                )
            
            # Calculate fill price with slippage
            fill_result = self.slippage_calculator.calculate_fill_price(
                order_book=order_book,
                side=side,
                quantity=quantity,
                order_type=order_type,
                limit_price=price
            )
            
            if fill_result.filled_quantity == 0:
                return self._create_rejected_response(
                    symbol, side, order_type, quantity, "Order could not be filled"
                )
            
            fill_price = fill_result.avg_fill_price
            
            # Calculate fees
            fee_result = self.fee_calculator.calculate_fee(
                exchange=self.exchange,
                quantity=fill_result.filled_quantity,
                price=fill_price,
                order_type=order_type,
                is_maker=order_type.lower() == "limit"
            )
            
            # Update balance
            is_buy = side.lower() in ["buy", "long"]
            notional = fill_result.filled_quantity * fill_price
            
            if is_buy:
                cost = notional + fee_result.fee_amount
                if cost > self.current_balance:
                    return self._create_rejected_response(
                        symbol, side, order_type, quantity,
                        f"Insufficient balance: need {cost:.2f}, have {self.current_balance:.2f}"
                    )
                self.current_balance -= cost
            else:
                # Selling - add proceeds to balance
                proceeds = notional - fee_result.fee_amount
                self.current_balance += proceeds
            
            # Update positions
            direction = "long" if is_buy else "short"
            
            # Check if this is closing a position
            existing_positions = self.position_manager.get_positions(symbol=symbol)
            is_closing = False
            
            for pos in existing_positions:
                if pos.direction != direction and pos.quantity != 0:
                    is_closing = True
                    # Close the opposite position
                    self.position_manager.reduce_position(
                        symbol=symbol,
                        quantity=fill_result.filled_quantity,
                        price=fill_price
                    )
                    break
            
            if not is_closing:
                # Opening or adding to position
                self.position_manager.add_position(
                    symbol=symbol,
                    direction=direction,
                    quantity=fill_result.filled_quantity,
                    price=fill_price,
                    metadata=trading_signal_metadata or {}
                )
            
            # Calculate PnL for closing trades
            pnl = 0.0
            if is_closing and existing_positions:
                # Simplified PnL calculation
                for pos in existing_positions:
                    if pos.direction != direction:
                        closed_qty = min(abs(pos.quantity), fill_result.filled_quantity)
                        if pos.direction == "long":
                            pnl += (fill_price - pos.avg_entry_price) * closed_qty
                        else:
                            pnl += (pos.avg_entry_price - fill_price) * closed_qty
                        pnl -= fee_result.fee_amount
            
            # Save trade to database
            trade_data = {
                "symbol": symbol,
                "side": side,
                "direction": direction,
                "quantity": fill_result.filled_quantity,
                "price": fill_price,
                "fee": fee_result.fee_amount,
                "slippage": fill_result.slippage_pct,
                "pnl": pnl,
                "is_maker": fee_result.is_maker,
                "fill_details": [
                    {"price": p, "quantity": q}
                    for p, q in fill_result.price_levels_used
                ],
                "latency_ms": latency_ms,
                "order_type": order_type,
                "trading_signal": trading_signal_metadata,
                "timestamp": datetime.now().isoformat()
            }
            
            self.persistence.save_trade(self.simulator_id, trade_data)
            
            # Update state
            self._save_state()
            
            # Save positions to database
            for pos_list in self.position_manager.positions.values():
                for pos in pos_list:
                    self.persistence.save_position(self.simulator_id, pos)
            
            self.logger.info(
                f"Order executed: {symbol} {side} {fill_result.filled_quantity} @ {fill_price:.6f} "
                f"(fee={fee_result.fee_amount:.4f}, slippage={fill_result.slippage_pct:.2%})"
            )
            
            # Record trade for reporting
            await self._record_trade_for_reporting(
                symbol=symbol,
                side=side,
                direction=direction,
                quantity=fill_result.filled_quantity,
                price=fill_price,
                fee=fee_result.fee_amount,
                slippage=fill_result.slippage_pct,
                pnl=pnl,
                is_closing=is_closing,
                trading_signal_metadata=trading_signal_metadata,
                latency_ms=latency_ms
            )
            
            # Return exchange-compatible response
            return {
                "orderId": str(uuid.uuid4()),
                "symbol": symbol,
                "status": "FILLED",
                "side": side.upper(),
                "type": order_type.upper(),
                "quantity": fill_result.filled_quantity,
                "price": fill_price,
                "avgPrice": fill_price,
                "filledQuantity": fill_result.filled_quantity,
                "remainingQuantity": fill_result.remaining_quantity,
                "fee": fee_result.fee_amount,
                "feeAsset": "USDT",
                "fillPrice": fill_price,
                "slippagePct": fill_result.slippage_pct,
                "latencyMs": latency_ms,
                "createdAt": start_time.isoformat(),
                "updatedAt": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error simulating order: {e}")
            return self._create_rejected_response(
                symbol, side, order_type, quantity, str(e)
            )
    
    def _create_rejected_response(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        reason: str
    ) -> Dict[str, Any]:
        """Create a rejected order response."""
        return {
            "orderId": str(uuid.uuid4()),
            "symbol": symbol,
            "status": "REJECTED",
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity,
            "rejectedReason": reason,
            "createdAt": datetime.now().isoformat()
        }
    
    def _save_state(self) -> None:
        """Save current simulator state to database."""
        config_json = self.config.to_dict()
        import json
        config_json_str = json.dumps(config_json)
        
        self.persistence.save_simulator_state(
            simulator_id=self.simulator_id,
            mode="paper",
            exchange=self.exchange,
            asset="",
            initial_balance=self.initial_balance,
            current_balance=self.current_balance,
            direction_constraint=self.direction_constraint,
            config_json=config_json_str
        )
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        return self.persistence.get_positions(self.simulator_id, status="open")
    
    def get_trade_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history."""
        return self.persistence.get_trades(self.simulator_id, symbol=symbol, limit=limit)
    
    async def _record_trade_for_reporting(
        self,
        symbol: str,
        side: str,
        direction: str,
        quantity: float,
        price: float,
        fee: float,
        slippage: float,
        pnl: float,
        is_closing: bool,
        trading_signal_metadata: Optional[Dict[str, Any]],
        latency_ms: float
    ) -> None:
        """Record trade for reporting system"""
        try:
            # Extract metadata
            metadata = trading_signal_metadata or {}
            
            # Extract confidence scores
            analyst_confidence = metadata.get('analyst_confidence', 0.0)
            tactician_confidence = metadata.get('tactician_confidence', 0.0)
            strategist_confidence = metadata.get('strategist_confidence', 0.0)
            ensemble_confidence = metadata.get('confidence', 0.0)
            signal_strength = metadata.get('signal_strength', 0.0)
            
            # Extract SHAP/feature importance
            shap_values = metadata.get('shap_values', {})
            top_features = sorted(
                shap_values.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3] if shap_values else []
            
            # Extract regime information
            regime_probs = metadata.get('regime_probabilities', {})
            top_regimes = sorted(
                regime_probs.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3] if regime_probs else []
            
            # Extract context
            volume = metadata.get('volume', 0.0)
            volatility = metadata.get('volatility', 0.0)
            trend = metadata.get('trend', 'neutral')
            
            # Determine entry/exit
            entry_datetime = datetime.now()
            exit_datetime = entry_datetime if is_closing else None
            entry_price = price
            exit_price = price if is_closing else None
            
            # Calculate PnL percentages
            net_gain_loss_pct = None
            net_gain_loss_absolute = None
            
            if is_closing and pnl != 0:
                net_gain_loss_absolute = pnl
                # Estimate percentage (simplified)
                net_gain_loss_pct = (pnl / (quantity * price)) if (quantity * price) > 0 else 0.0
            
            # Create trade record
            trade_record = TradeRecord(
                trade_id=str(uuid.uuid4()),
                timestamp=entry_datetime,
                exchange=self.exchange,
                asset=symbol,
                mode="paper",
                entry_datetime=entry_datetime,
                exit_datetime=exit_datetime,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                side=side,
                direction=direction,
                net_gain_loss_pct=net_gain_loss_pct,
                net_gain_loss_absolute=net_gain_loss_absolute,
                realized_pnl=pnl if is_closing else None,
                fees=fee,
                slippage_pct=slippage,
                analyst_confidence=analyst_confidence,
                tactician_confidence=tactician_confidence,
                strategist_confidence=strategist_confidence,
                ensemble_confidence=ensemble_confidence,
                signal_strength=signal_strength,
                top_feature_1=top_features[0][0] if len(top_features) > 0 else "",
                top_feature_1_importance=top_features[0][1] if len(top_features) > 0 else 0.0,
                top_feature_2=top_features[1][0] if len(top_features) > 1 else "",
                top_feature_2_importance=top_features[1][1] if len(top_features) > 1 else 0.0,
                top_feature_3=top_features[2][0] if len(top_features) > 2 else "",
                top_feature_3_importance=top_features[2][1] if len(top_features) > 2 else 0.0,
                regime_1=top_regimes[0][0] if len(top_regimes) > 0 else "",
                regime_1_probability=top_regimes[0][1] if len(top_regimes) > 0 else 0.0,
                regime_2=top_regimes[1][0] if len(top_regimes) > 1 else "",
                regime_2_probability=top_regimes[1][1] if len(top_regimes) > 1 else 0.0,
                regime_3=top_regimes[2][0] if len(top_regimes) > 2 else "",
                regime_3_probability=top_regimes[2][1] if len(top_regimes) > 2 else 0.0,
                volume=volume,
                volatility=volatility,
                trend=trend,
                execution_time_ms=latency_ms,
                execution_quality=1.0 - slippage  # Simplified quality metric
            )
            
            # Record trade
            await trade_reporting_manager.record_trade(trade_record)
            
            tprint_debug(f"📊 Trade recorded for reporting: {trade_record.trade_id}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to record trade for reporting: {e}")
    
    async def generate_daily_report(self, symbol: str, target_date: Optional[date] = None) -> bool:
        """
        Generate daily report for a specific symbol.
        
        Args:
            symbol: Trading symbol
            target_date: Date to generate report for (defaults to today)
            
        Returns:
            True if successful
        """
        try:
            tprint_info(f"📊 Generating daily report for {symbol} ({target_date or date.today()})")
            
            result = await generate_daily_recap(
                mode="paper",
                exchange=self.exchange,
                asset=symbol,
                target_date=target_date
            )
            
            if result:
                tprint_success(f"✅ Daily report generated for {symbol}")
            else:
                tprint_error(f"❌ Failed to generate daily report for {symbol}")
            
            return result
        except Exception as e:
            tprint_error(f"❌ Failed to generate daily report: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        trades = self.get_trade_history()
        
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_fees": 0.0,
                "avg_slippage": 0.0,
                "profit_factor": 0.0
            }
        
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.get("pnl", 0) > 0)
        losing_trades = sum(1 for t in trades if t.get("pnl", 0) < 0)
        
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        total_fees = sum(t.get("fee", 0) for t in trades)
        avg_slippage = sum(t.get("slippage", 0) for t in trades) / total_trades if total_trades > 0 else 0.0
        
        total_profits = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
        total_losses = abs(sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0))
        profit_factor = total_profits / total_losses if total_losses > 0 else 0.0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0.0,
            "total_pnl": total_pnl,
            "total_fees": total_fees,
            "avg_slippage": avg_slippage,
            "profit_factor": profit_factor,
            "current_balance": self.current_balance,
            "initial_balance": self.initial_balance,
            "net_pnl": self.current_balance - self.initial_balance,
            "net_pnl_pct": (self.current_balance - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0.0
        }
