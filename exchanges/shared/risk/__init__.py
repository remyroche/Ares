"""
Risk Management Utilities

Provides utilities for risk calculation, liquidation risk management,
and margin management.
"""

from .risk_calculator import RiskCalculator

# Risk management classes
class LiquidationRiskManager:
    """
    Manages liquidation risk for trading positions.
    
    Provides functionality to calculate liquidation prices, monitor risk levels,
    and trigger risk management actions.
    """
    
    def __init__(self, initial_margin_ratio: float = 0.1, maintenance_margin_ratio: float = 0.05):
        """
        Initialize the LiquidationRiskManager.
        
        Args:
            initial_margin_ratio: Initial margin requirement (default: 10%)
            maintenance_margin_ratio: Maintenance margin requirement (default: 5%)
        """
        self.initial_margin_ratio = initial_margin_ratio
        self.maintenance_margin_ratio = maintenance_margin_ratio
        self.positions = {}
        self.risk_alerts = []
    
    def add_position(self, symbol: str, side: str, size: float, entry_price: float, 
                    leverage: float = 1.0, margin: float = None):
        """
        Add a position to monitor for liquidation risk.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            side: Position side ('long' or 'short')
            size: Position size
            entry_price: Entry price of the position
            leverage: Leverage used (default: 1.0)
            margin: Initial margin used (if None, calculated from size * entry_price / leverage)
        """
        if margin is None:
            margin = (size * entry_price) / leverage
        
        self.positions[symbol] = {
            'side': side,
            'size': size,
            'entry_price': entry_price,
            'leverage': leverage,
            'margin': margin,
            'unrealized_pnl': 0.0,
            'liquidation_price': self._calculate_liquidation_price(side, size, entry_price, leverage, margin)
        }
    
    def _calculate_liquidation_price(self, side: str, size: float, entry_price: float, 
                                   leverage: float, margin: float) -> float:
        """Calculate liquidation price for a position."""
        if side == 'long':
            # For long positions: liquidation_price = entry_price * (1 - initial_margin_ratio + maintenance_margin_ratio)
            return entry_price * (1 - self.initial_margin_ratio + self.maintenance_margin_ratio)
        else:  # short
            # For short positions: liquidation_price = entry_price * (1 + initial_margin_ratio - maintenance_margin_ratio)
            return entry_price * (1 + self.initial_margin_ratio - self.maintenance_margin_ratio)
    
    def update_position_price(self, symbol: str, current_price: float):
        """
        Update current price for a position and recalculate risk metrics.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        entry_price = position['entry_price']
        size = position['size']
        side = position['side']
        
        # Calculate unrealized PnL
        if side == 'long':
            position['unrealized_pnl'] = size * (current_price - entry_price)
        else:  # short
            position['unrealized_pnl'] = size * (entry_price - current_price)
        
        # Check if position is at risk
        self._check_liquidation_risk(symbol, current_price)
    
    def _check_liquidation_risk(self, symbol: str, current_price: float):
        """Check if position is approaching liquidation."""
        position = self.positions[symbol]
        liquidation_price = position['liquidation_price']
        
        # Calculate distance to liquidation
        if position['side'] == 'long':
            distance_to_liquidation = (current_price - liquidation_price) / liquidation_price
        else:  # short
            distance_to_liquidation = (liquidation_price - current_price) / liquidation_price
        
        # Risk levels
        if distance_to_liquidation <= 0.05:  # Within 5% of liquidation
            self._add_risk_alert(symbol, 'CRITICAL', f"Position within 5% of liquidation price")
        elif distance_to_liquidation <= 0.10:  # Within 10% of liquidation
            self._add_risk_alert(symbol, 'HIGH', f"Position within 10% of liquidation price")
        elif distance_to_liquidation <= 0.20:  # Within 20% of liquidation
            self._add_risk_alert(symbol, 'MEDIUM', f"Position within 20% of liquidation price")
    
    def _add_risk_alert(self, symbol: str, level: str, message: str):
        """Add a risk alert."""
        alert = {
            'timestamp': self._get_timestamp(),
            'symbol': symbol,
            'level': level,
            'message': message
        }
        self.risk_alerts.append(alert)
    
    def get_risk_summary(self) -> dict:
        """Get a summary of all position risks."""
        summary = {
            'total_positions': len(self.positions),
            'total_unrealized_pnl': sum(pos['unrealized_pnl'] for pos in self.positions.values()),
            'positions_at_risk': 0,
            'recent_alerts': self.risk_alerts[-10:] if self.risk_alerts else []
        }
        
        for position in self.positions.values():
            if position['unrealized_pnl'] < 0:  # Negative PnL indicates risk
                summary['positions_at_risk'] += 1
        
        return summary
    
    def close_position(self, symbol: str):
        """Close a position and remove from monitoring."""
        if symbol in self.positions:
            del self.positions[symbol]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


class MarginManager:
    """
    Manages margin requirements and calculations for trading positions.
    
    Provides functionality to calculate initial margin, maintenance margin,
    and margin calls.
    """
    
    def __init__(self, base_margin_ratio: float = 0.1):
        """
        Initialize the MarginManager.
        
        Args:
            base_margin_ratio: Base margin requirement ratio
        """
        self.base_margin_ratio = base_margin_ratio
        self.positions = {}
        self.margin_call_threshold = 0.8  # 80% of initial margin
    
    def calculate_initial_margin(self, symbol: str, size: float, price: float, 
                               leverage: float = 1.0) -> float:
        """
        Calculate initial margin required for a position.
        
        Args:
            symbol: Trading symbol
            size: Position size
            price: Entry price
            leverage: Leverage multiplier
            
        Returns:
            Required initial margin
        """
        notional_value = size * price
        margin_ratio = self.base_margin_ratio / leverage
        return notional_value * margin_ratio
    
    def calculate_maintenance_margin(self, symbol: str, size: float, price: float) -> float:
        """
        Calculate maintenance margin for a position.
        
        Args:
            symbol: Trading symbol
            size: Position size
            price: Current price
            
        Returns:
            Required maintenance margin
        """
        notional_value = size * price
        maintenance_ratio = self.base_margin_ratio * 0.5  # 50% of initial margin
        return notional_value * maintenance_ratio
    
    def add_position(self, symbol: str, side: str, size: float, entry_price: float, 
                    leverage: float = 1.0):
        """
        Add a position for margin monitoring.
        
        Args:
            symbol: Trading symbol
            side: Position side ('long' or 'short')
            size: Position size
            entry_price: Entry price
            leverage: Leverage used
        """
        initial_margin = self.calculate_initial_margin(symbol, size, entry_price, leverage)
        
        self.positions[symbol] = {
            'side': side,
            'size': size,
            'entry_price': entry_price,
            'leverage': leverage,
            'initial_margin': initial_margin,
            'current_margin': initial_margin,
            'unrealized_pnl': 0.0
        }
    
    def update_position(self, symbol: str, current_price: float):
        """
        Update position with current price and recalculate margin.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        entry_price = position['entry_price']
        size = position['size']
        side = position['side']
        
        # Calculate unrealized PnL
        if side == 'long':
            position['unrealized_pnl'] = size * (current_price - entry_price)
        else:  # short
            position['unrealized_pnl'] = size * (entry_price - current_price)
        
        # Update current margin
        position['current_margin'] = position['initial_margin'] + position['unrealized_pnl']
    
    def check_margin_call(self, symbol: str) -> bool:
        """
        Check if a position requires a margin call.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if margin call is required
        """
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        maintenance_margin = self.calculate_maintenance_margin(
            symbol, position['size'], position['entry_price']
        )
        
        return position['current_margin'] < maintenance_margin
    
    def get_margin_ratio(self, symbol: str) -> float:
        """
        Get current margin ratio for a position.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current margin ratio
        """
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        return position['current_margin'] / position['initial_margin']
    
    def get_portfolio_margin_summary(self) -> dict:
        """Get summary of all positions' margin status."""
        total_initial_margin = sum(pos['initial_margin'] for pos in self.positions.values())
        total_current_margin = sum(pos['current_margin'] for pos in self.positions.values())
        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions.values())
        
        margin_calls = [symbol for symbol in self.positions.keys() if self.check_margin_call(symbol)]
        
        return {
            'total_positions': len(self.positions),
            'total_initial_margin': total_initial_margin,
            'total_current_margin': total_current_margin,
            'total_unrealized_pnl': total_unrealized_pnl,
            'margin_calls_required': len(margin_calls),
            'margin_call_symbols': margin_calls,
            'portfolio_margin_ratio': total_current_margin / total_initial_margin if total_initial_margin > 0 else 0
        }

__all__ = [
    "RiskCalculator",
    "LiquidationRiskManager",
    "MarginManager"
]