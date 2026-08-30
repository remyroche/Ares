"""
Portfolio Simulator

Handles portfolio simulation and position management for paper trading.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import logging

from .simulator_interface import SimulatedPosition


class PortfolioSimulator:
    """Simulates portfolio management for paper trading"""

    def __init__(self, initial_balance: float = 100000.0):
        """
        Initialize the portfolio simulator
        
        Args:
            initial_balance: Starting balance for simulation
        """
        self.logger = logging.getLogger(__name__)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: Dict[str, SimulatedPosition] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}

    def update_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        commission: float = 0.0
    ) -> None:
        """Update position after a trade"""
        try:
            side = side.upper()
            
            if symbol not in self.positions:
                self.positions[symbol] = SimulatedPosition(
                    symbol=symbol,
                    side=side,
                    quantity=0.0,
                    average_price=0.0,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
            
            position = self.positions[symbol]
            
            # Record trade
            trade = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "commission": commission,
                "timestamp": datetime.now(timezone.utc),
                "balance_before": self.balance
            }
            
            if position.side == side:
                # Same side - add to position
                total_quantity = position.quantity + quantity
                total_value = (position.quantity * position.average_price) + (quantity * price)
                position.average_price = total_value / total_quantity if total_quantity > 0 else 0
                position.quantity = total_quantity
                
                # Update balance
                if side == "BUY":
                    self.balance -= (quantity * price + commission)
                else:  # SELL
                    self.balance += (quantity * price - commission)
                    
            else:
                # Opposite side - reduce or close position
                if quantity >= position.quantity:
                    # Close position and potentially reverse
                    remaining_quantity = quantity - position.quantity
                    
                    # Calculate realized PnL for closed portion
                    if position.quantity > 0:
                        if position.side == "BUY":
                            realized_pnl = (price - position.average_price) * position.quantity
                        else:  # position.side == "SELL"
                            realized_pnl = (position.average_price - price) * position.quantity
                        
                        position.realized_pnl += realized_pnl
                        trade["realized_pnl"] = realized_pnl
                    
                    # Update position
                    position.quantity = remaining_quantity
                    position.side = side
                    position.average_price = price if remaining_quantity > 0 else 0
                    
                    # Update balance
                    if side == "BUY":
                        self.balance -= (quantity * price + commission)
                    else:  # SELL
                        self.balance += (quantity * price - commission)
                else:
                    # Reduce position
                    if position.side == "BUY":
                        realized_pnl = (price - position.average_price) * quantity
                        self.balance += (quantity * price - commission)
                    else:  # position.side == "SELL"
                        realized_pnl = (position.average_price - price) * quantity
                        self.balance -= (quantity * price + commission)
                    
                    position.quantity -= quantity
                    position.realized_pnl += realized_pnl
                    trade["realized_pnl"] = realized_pnl
            
            position.updated_at = datetime.now(timezone.utc)
            trade["balance_after"] = self.balance
            
            # Record trade
            self.trade_history.append(trade)
            
            # Remove position if quantity is zero
            if position.quantity == 0:
                del self.positions[symbol]
            
            self.logger.info(f"Updated position for {symbol}: {side} {quantity} @ {price}")
            
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")

    def update_unrealized_pnl(self, symbol: str, current_price: float) -> None:
        """Update unrealized PnL for a position"""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            
            if position.side == "BUY":
                position.unrealized_pnl = (current_price - position.average_price) * position.quantity
            else:  # SELL
                position.unrealized_pnl = (position.average_price - current_price) * position.quantity
            
            position.updated_at = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.error(f"Error updating unrealized PnL: {e}")

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        try:
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            total_pnl = total_unrealized_pnl + total_realized_pnl
            
            # Calculate portfolio value
            portfolio_value = self.balance
            for position in self.positions.values():
                # This would need current market prices to be accurate
                portfolio_value += position.quantity * position.average_price
            
            return {
                "balance": self.balance,
                "portfolio_value": portfolio_value,
                "unrealized_pnl": total_unrealized_pnl,
                "realized_pnl": total_realized_pnl,
                "total_pnl": total_pnl,
                "position_count": len(self.positions),
                "trade_count": len(self.trade_history),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {
                "balance": self.balance,
                "portfolio_value": self.balance,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "total_pnl": 0.0,
                "position_count": 0,
                "trade_count": 0,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def get_position_details(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get detailed position information"""
        try:
            positions = []
            
            for position in self.positions.values():
                if symbol is None or position.symbol == symbol:
                    positions.append({
                        "symbol": position.symbol,
                        "side": position.side,
                        "quantity": position.quantity,
                        "average_price": position.average_price,
                        "unrealized_pnl": position.unrealized_pnl,
                        "realized_pnl": position.realized_pnl,
                        "created_at": position.created_at.isoformat(),
                        "updated_at": position.updated_at.isoformat()
                    })
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting position details: {e}")
            return []

    def get_trade_history(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history"""
        try:
            trades = []
            
            for trade in self.trade_history:
                if symbol is None or trade["symbol"] == symbol:
                    trades.append(trade)
            
            # Sort by timestamp descending and limit
            trades.sort(key=lambda x: x["timestamp"], reverse=True)
            return trades[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            return []

    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        try:
            if not self.trade_history:
                return {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate": 0.0,
                    "total_return": 0.0,
                    "max_drawdown": 0.0
                }
            
            # Calculate basic metrics
            total_trades = len(self.trade_history)
            winning_trades = sum(1 for trade in self.trade_history if trade.get("realized_pnl", 0) > 0)
            losing_trades = sum(1 for trade in self.trade_history if trade.get("realized_pnl", 0) < 0)
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Calculate total return
            total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
            
            # Calculate max drawdown (simplified)
            max_drawdown = 0.0
            peak_balance = self.initial_balance
            for trade in self.trade_history:
                if trade["balance_after"] > peak_balance:
                    peak_balance = trade["balance_after"]
                else:
                    drawdown = ((peak_balance - trade["balance_after"]) / peak_balance) * 100
                    max_drawdown = max(max_drawdown, drawdown)
            
            self.performance_metrics = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "initial_balance": self.initial_balance,
                "current_balance": self.balance
            }
            
            return self.performance_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}

    def reset_portfolio(self, new_balance: Optional[float] = None) -> None:
        """Reset portfolio to initial state"""
        try:
            self.balance = new_balance if new_balance is not None else self.initial_balance
            self.positions.clear()
            self.trade_history.clear()
            self.performance_metrics.clear()
            
            self.logger.info(f"Portfolio reset with balance: ${self.balance:,.2f}")
            
        except Exception as e:
            self.logger.error(f"Error resetting portfolio: {e}")