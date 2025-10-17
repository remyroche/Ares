"""
Order Management Utilities

Provides utilities for order management, idempotency, and position handling.
"""

from .order_manager import OrderManager
from .idempotency_manager import IdempotencyManager

# Position management class
class PositionManager:
    """
    Comprehensive position management for trading accounts.
    
    Provides functionality for tracking, managing, and analyzing trading positions
    with support for multiple symbols, leverage, and risk management.
    """
    
    def __init__(self, account_id: str = None):
        """
        Initialize the PositionManager.
        
        Args:
            account_id: Unique account identifier
        """
        self.account_id = account_id or self._generate_account_id()
        self.positions = {}  # {symbol: position_data}
        self.position_history = []
        self.leverage_settings = {}  # {symbol: leverage}
        self.risk_limits = {}  # {symbol: risk_limits}
        self.position_callbacks = []
    
    def _generate_account_id(self) -> str:
        """Generate a unique account ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def open_position(self, symbol: str, side: str, size: float, entry_price: float,
                     leverage: float = 1.0, stop_loss: float = None, 
                     take_profit: float = None, **kwargs) -> str:
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            side: Position side ('long' or 'short')
            size: Position size
            entry_price: Entry price
            leverage: Leverage multiplier
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            **kwargs: Additional position data
            
        Returns:
            Position ID
        """
        position_id = self._generate_position_id()
        
        # Calculate position metrics
        notional_value = size * entry_price
        margin_required = notional_value / leverage
        unrealized_pnl = 0.0
        
        position = {
            'position_id': position_id,
            'symbol': symbol,
            'side': side,
            'size': size,
            'entry_price': entry_price,
            'current_price': entry_price,
            'leverage': leverage,
            'notional_value': notional_value,
            'margin_required': margin_required,
            'unrealized_pnl': unrealized_pnl,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'status': 'open',
            'opened_at': self._get_timestamp(),
            'last_updated': self._get_timestamp(),
            **kwargs
        }
        
        # Store position
        self.positions[symbol] = position
        
        # Record in history
        self.position_history.append(position.copy())
        
        # Set leverage for symbol
        self.leverage_settings[symbol] = leverage
        
        # Notify callbacks
        self._notify_position_callbacks('opened', position)
        
        return position_id
    
    def update_position_price(self, symbol: str, current_price: float):
        """
        Update position with current market price.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        old_pnl = position['unrealized_pnl']
        
        # Update current price
        position['current_price'] = current_price
        position['last_updated'] = self._get_timestamp()
        
        # Calculate unrealized PnL
        entry_price = position['entry_price']
        size = position['size']
        side = position['side']
        
        if side == 'long':
            position['unrealized_pnl'] = size * (current_price - entry_price)
        else:  # short
            position['unrealized_pnl'] = size * (entry_price - current_price)
        
        # Update notional value
        position['notional_value'] = size * current_price
        
        # Check stop loss and take profit
        self._check_stop_loss_take_profit(symbol)
        
        # Notify callbacks if PnL changed significantly
        if abs(position['unrealized_pnl'] - old_pnl) > 0.01:
            self._notify_position_callbacks('updated', position)
    
    def close_position(self, symbol: str, exit_price: float = None, 
                      reason: str = 'manual') -> dict:
        """
        Close a position.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price (if None, uses current price)
            reason: Reason for closing
            
        Returns:
            Closed position data
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        if exit_price is None:
            exit_price = position['current_price']
        
        # Calculate final PnL
        entry_price = position['entry_price']
        size = position['size']
        side = position['side']
        
        if side == 'long':
            realized_pnl = size * (exit_price - entry_price)
        else:  # short
            realized_pnl = size * (entry_price - exit_price)
        
        # Update position
        position['exit_price'] = exit_price
        position['realized_pnl'] = realized_pnl
        position['status'] = 'closed'
        position['closed_at'] = self._get_timestamp()
        position['close_reason'] = reason
        
        # Calculate return percentage
        margin_used = position['margin_required']
        return_percentage = (realized_pnl / margin_used) * 100 if margin_used > 0 else 0
        
        position['return_percentage'] = return_percentage
        
        # Record in history
        self.position_history.append(position.copy())
        
        # Remove from active positions
        closed_position = self.positions.pop(symbol)
        
        # Notify callbacks
        self._notify_position_callbacks('closed', closed_position)
        
        return closed_position
    
    def get_position(self, symbol: str) -> dict:
        """Get position data for a symbol."""
        return self.positions.get(symbol, {}).copy()
    
    def get_all_positions(self) -> dict:
        """Get all active positions."""
        return {symbol: pos.copy() for symbol, pos in self.positions.items()}
    
    def get_position_history(self, symbol: str = None, limit: int = None) -> list:
        """
        Get position history.
        
        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of positions to return
            
        Returns:
            List of position records
        """
        history = self.position_history.copy()
        
        if symbol:
            history = [pos for pos in history if pos['symbol'] == symbol]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def calculate_portfolio_metrics(self) -> dict:
        """Calculate portfolio-level metrics."""
        if not self.positions:
            return {
                'total_positions': 0,
                'total_unrealized_pnl': 0.0,
                'total_margin_used': 0.0,
                'portfolio_value': 0.0,
                'winning_positions': 0,
                'losing_positions': 0
            }
        
        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions.values())
        total_margin_used = sum(pos['margin_required'] for pos in self.positions.values())
        
        winning_positions = len([pos for pos in self.positions.values() if pos['unrealized_pnl'] > 0])
        losing_positions = len([pos for pos in self.positions.values() if pos['unrealized_pnl'] < 0])
        
        portfolio_value = total_margin_used + total_unrealized_pnl
        
        return {
            'total_positions': len(self.positions),
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_margin_used': total_margin_used,
            'portfolio_value': portfolio_value,
            'winning_positions': winning_positions,
            'losing_positions': losing_positions,
            'win_rate': winning_positions / len(self.positions) if self.positions else 0
        }
    
    def set_leverage(self, symbol: str, leverage: float):
        """
        Set leverage for a symbol.
        
        Args:
            symbol: Trading symbol
            leverage: Leverage multiplier
        """
        self.leverage_settings[symbol] = leverage
        
        # Update existing position if it exists
        if symbol in self.positions:
            self.positions[symbol]['leverage'] = leverage
            # Recalculate margin required
            notional_value = self.positions[symbol]['notional_value']
            self.positions[symbol]['margin_required'] = notional_value / leverage
    
    def set_risk_limits(self, symbol: str, max_position_size: float = None,
                       max_loss_percent: float = None, max_drawdown: float = None):
        """
        Set risk limits for a symbol.
        
        Args:
            symbol: Trading symbol
            max_position_size: Maximum position size
            max_loss_percent: Maximum loss percentage
            max_drawdown: Maximum drawdown
        """
        self.risk_limits[symbol] = {
            'max_position_size': max_position_size,
            'max_loss_percent': max_loss_percent,
            'max_drawdown': max_drawdown
        }
    
    def check_risk_limits(self, symbol: str) -> list:
        """
        Check if position violates risk limits.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of risk limit violations
        """
        violations = []
        
        if symbol not in self.positions or symbol not in self.risk_limits:
            return violations
        
        position = self.positions[symbol]
        limits = self.risk_limits[symbol]
        
        # Check position size limit
        if limits.get('max_position_size') and position['size'] > limits['max_position_size']:
            violations.append(f"Position size {position['size']} exceeds limit {limits['max_position_size']}")
        
        # Check loss percentage limit
        if limits.get('max_loss_percent'):
            loss_percent = abs(position['unrealized_pnl']) / position['margin_required'] * 100
            if loss_percent > limits['max_loss_percent']:
                violations.append(f"Loss percentage {loss_percent:.2f}% exceeds limit {limits['max_loss_percent']}%")
        
        return violations
    
    def add_position_callback(self, callback):
        """Add a callback function for position events."""
        self.position_callbacks.append(callback)
    
    def _check_stop_loss_take_profit(self, symbol: str):
        """Check if stop loss or take profit should be triggered."""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        current_price = position['current_price']
        side = position['side']
        
        # Check stop loss
        if position.get('stop_loss'):
            should_trigger = False
            if side == 'long' and current_price <= position['stop_loss']:
                should_trigger = True
            elif side == 'short' and current_price >= position['stop_loss']:
                should_trigger = True
            
            if should_trigger:
                self.close_position(symbol, current_price, 'stop_loss')
                return
        
        # Check take profit
        if position.get('take_profit'):
            should_trigger = False
            if side == 'long' and current_price >= position['take_profit']:
                should_trigger = True
            elif side == 'short' and current_price <= position['take_profit']:
                should_trigger = True
            
            if should_trigger:
                self.close_position(symbol, current_price, 'take_profit')
    
    def _notify_position_callbacks(self, event_type: str, position: dict):
        """Notify position callbacks of events."""
        for callback in self.position_callbacks:
            try:
                callback(event_type, position)
            except Exception as e:
                print(f"Error in position callback: {e}")
    
    def _generate_position_id(self) -> str:
        """Generate a unique position ID."""
        import uuid
        return f"pos_{str(uuid.uuid4())[:8]}"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

__all__ = [
    "OrderManager",
    "IdempotencyManager",
    "PositionManager"
]