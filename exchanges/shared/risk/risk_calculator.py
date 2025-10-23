"""
Risk Calculation Utilities

Handles position risk calculations, margin requirements, and risk metrics.
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import system_logger


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PositionRisk:
    """Position risk data structure"""
    symbol: str
    position_size: float
    entry_price: float
    current_price: float
    leverage: float
    margin_used: float
    unrealized_pnl: float
    margin_ratio: float
    liquidation_price: float
    risk_level: RiskLevel
    maintenance_margin: float
    initial_margin: float
    notional_value: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PortfolioRisk:
    """Portfolio risk data structure"""
    total_equity: float
    total_margin_used: float
    total_unrealized_pnl: float
    portfolio_margin_ratio: float
    risk_level: RiskLevel
    positions: List[PositionRisk]
    max_leverage_used: float
    total_notional: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class RiskCalculator:
    """
    Calculates various risk metrics for positions and portfolios.
    """
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.logger = system_logger.getChild(f"RiskCalculator.{exchange_name}")
        
        # Risk thresholds
        self.margin_ratio_warning = 0.8  # 80% margin ratio warning
        self.margin_ratio_critical = 0.9  # 90% margin ratio critical
        self.margin_ratio_liquidation = 0.95  # 95% margin ratio liquidation
        
        # Default margin requirements
        self.default_initial_margin = 0.1  # 10% initial margin
        self.default_maintenance_margin = 0.05  # 5% maintenance margin
    
    def calculate_position_risk(
        self,
        symbol: str,
        position_size: float,
        entry_price: float,
        current_price: float,
        leverage: float,
        initial_margin: Optional[float] = None,
        maintenance_margin: Optional[float] = None
    ) -> PositionRisk:
        """
        Calculate risk metrics for a single position.
        
        Args:
            symbol: Trading symbol
            position_size: Position size (positive for long, negative for short)
            entry_price: Entry price
            current_price: Current market price
            leverage: Leverage used
            initial_margin: Initial margin requirement (optional)
            maintenance_margin: Maintenance margin requirement (optional)
            
        Returns:
            PositionRisk object
        """
        # Use default margins if not provided
        initial_margin = initial_margin or self.default_initial_margin
        maintenance_margin = maintenance_margin or self.default_maintenance_margin
        
        # Calculate notional value
        notional_value = abs(position_size) * current_price
        
        # Calculate margin used
        margin_used = notional_value * initial_margin / leverage
        
        # Calculate unrealized PnL
        if position_size > 0:  # Long position
            unrealized_pnl = position_size * (current_price - entry_price)
        else:  # Short position
            unrealized_pnl = abs(position_size) * (entry_price - current_price)
        
        # Calculate margin ratio
        margin_ratio = margin_used / (margin_used + unrealized_pnl) if (margin_used + unrealized_pnl) > 0 else 1.0
        
        # Calculate liquidation price
        liquidation_price = self._calculate_liquidation_price(
            entry_price, position_size, leverage, initial_margin, maintenance_margin
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(margin_ratio)
        
        return PositionRisk(
            symbol=symbol,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            leverage=leverage,
            margin_used=margin_used,
            unrealized_pnl=unrealized_pnl,
            margin_ratio=margin_ratio,
            liquidation_price=liquidation_price,
            risk_level=risk_level,
            maintenance_margin=maintenance_margin,
            initial_margin=initial_margin,
            notional_value=notional_value
        )
    
    def _calculate_liquidation_price(
        self,
        entry_price: float,
        position_size: float,
        leverage: float,
        initial_margin: float,
        maintenance_margin: float
    ) -> float:
        """Calculate liquidation price for a position."""
        if position_size == 0:
            return 0.0
        
        # Calculate the price at which maintenance margin is reached
        if position_size > 0:  # Long position
            # For long: liquidation_price = entry_price * (1 - initial_margin + maintenance_margin) / leverage
            liquidation_price = entry_price * (1 - initial_margin + maintenance_margin) / leverage
        else:  # Short position
            # For short: liquidation_price = entry_price * (1 + initial_margin - maintenance_margin) / leverage
            liquidation_price = entry_price * (1 + initial_margin - maintenance_margin) / leverage
        
        return max(0.0, liquidation_price)
    
    def _determine_risk_level(self, margin_ratio: float) -> RiskLevel:
        """Determine risk level based on margin ratio."""
        if margin_ratio >= self.margin_ratio_liquidation:
            return RiskLevel.CRITICAL
        elif margin_ratio >= self.margin_ratio_critical:
            return RiskLevel.HIGH
        elif margin_ratio >= self.margin_ratio_warning:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def calculate_portfolio_risk(
        self,
        positions: List[PositionRisk],
        total_equity: float
    ) -> PortfolioRisk:
        """
        Calculate portfolio-level risk metrics.
        
        Args:
            positions: List of position risks
            total_equity: Total account equity
            
        Returns:
            PortfolioRisk object
        """
        # Calculate totals
        total_margin_used = sum(pos.margin_used for pos in positions)
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
        total_notional = sum(pos.notional_value for pos in positions)
        
        # Calculate portfolio margin ratio
        portfolio_margin_ratio = total_margin_used / (total_equity + total_unrealized_pnl) if (total_equity + total_unrealized_pnl) > 0 else 1.0
        
        # Find maximum leverage used
        max_leverage_used = max((pos.leverage for pos in positions), default=1.0)
        
        # Determine portfolio risk level
        portfolio_risk_level = self._determine_risk_level(portfolio_margin_ratio)
        
        return PortfolioRisk(
            total_equity=total_equity,
            total_margin_used=total_margin_used,
            total_unrealized_pnl=total_unrealized_pnl,
            portfolio_margin_ratio=portfolio_margin_ratio,
            risk_level=portfolio_risk_level,
            positions=positions,
            max_leverage_used=max_leverage_used,
            total_notional=total_notional
        )
    
    def calculate_var(
        self,
        positions: List[PositionRisk],
        confidence_level: float = 0.95,
        time_horizon: int = 1
    ) -> float:
        """
        Calculate Value at Risk (VaR) for the portfolio.
        
        Args:
            positions: List of position risks
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days
            
        Returns:
            VaR value
        """
        if not positions:
            return 0.0
        
        # Calculate portfolio value
        portfolio_value = sum(pos.notional_value for pos in positions)
        
        # Simple VaR calculation using normal distribution assumption
        # This is a simplified version - in practice, you'd use historical data
        z_score = self._get_z_score(confidence_level)
        
        # Assume 1% daily volatility (this should be calculated from historical data)
        daily_volatility = 0.01
        
        # Calculate VaR
        var = portfolio_value * z_score * daily_volatility * math.sqrt(time_horizon)
        
        return var
    
    def _get_z_score(self, confidence_level: float) -> float:
        """Get Z-score for given confidence level."""
        # Common Z-scores for normal distribution
        z_scores = {
            0.90: 1.28,
            0.95: 1.65,
            0.99: 2.33,
            0.999: 3.09
        }
        
        return z_scores.get(confidence_level, 1.65)
    
    def calculate_max_position_size(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        leverage: float,
        available_margin: float,
        risk_tolerance: float = 0.8
    ) -> float:
        """
        Calculate maximum position size based on available margin and risk tolerance.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            current_price: Current market price
            leverage: Leverage to use
            available_margin: Available margin
            risk_tolerance: Risk tolerance (0.0 to 1.0)
            
        Returns:
            Maximum position size
        """
        # Calculate maximum notional value
        max_notional = available_margin * leverage / self.default_initial_margin
        
        # Apply risk tolerance
        max_notional *= risk_tolerance
        
        # Calculate maximum position size
        max_position_size = max_notional / current_price
        
        return max_position_size
    
    def calculate_margin_requirement(
        self,
        symbol: str,
        position_size: float,
        current_price: float,
        leverage: float,
        initial_margin: Optional[float] = None
    ) -> float:
        """
        Calculate margin requirement for a position.
        
        Args:
            symbol: Trading symbol
            position_size: Position size
            current_price: Current market price
            leverage: Leverage to use
            initial_margin: Initial margin requirement (optional)
            
        Returns:
            Required margin
        """
        initial_margin = initial_margin or self.default_initial_margin
        notional_value = abs(position_size) * current_price
        required_margin = notional_value * initial_margin / leverage
        
        return required_margin
    
    def calculate_liquidation_distance(
        self,
        position_risk: PositionRisk
    ) -> float:
        """
        Calculate the distance to liquidation as a percentage.
        
        Args:
            position_risk: Position risk data
            
        Returns:
            Distance to liquidation as percentage
        """
        if position_risk.position_size == 0:
            return 100.0
        
        current_price = position_risk.current_price
        liquidation_price = position_risk.liquidation_price
        
        if position_risk.position_size > 0:  # Long position
            distance = (current_price - liquidation_price) / current_price
        else:  # Short position
            distance = (liquidation_price - current_price) / current_price
        
        return max(0.0, distance * 100)
    
    def get_risk_summary(self, portfolio_risk: PortfolioRisk) -> Dict[str, Any]:
        """Get a summary of portfolio risk."""
        high_risk_positions = [pos for pos in portfolio_risk.positions if pos.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        
        return {
            "total_equity": portfolio_risk.total_equity,
            "total_margin_used": portfolio_risk.total_margin_used,
            "total_unrealized_pnl": portfolio_risk.total_unrealized_pnl,
            "portfolio_margin_ratio": portfolio_risk.portfolio_margin_ratio,
            "risk_level": portfolio_risk.risk_level.value,
            "max_leverage_used": portfolio_risk.max_leverage_used,
            "total_notional": portfolio_risk.total_notional,
            "high_risk_positions": len(high_risk_positions),
            "total_positions": len(portfolio_risk.positions),
            "margin_utilization": portfolio_risk.total_margin_used / portfolio_risk.total_equity if portfolio_risk.total_equity > 0 else 0
        }
    
    def set_risk_thresholds(
        self,
        warning_ratio: float = 0.8,
        critical_ratio: float = 0.9,
        liquidation_ratio: float = 0.95
    ) -> None:
        """Set risk thresholds."""
        self.margin_ratio_warning = warning_ratio
        self.margin_ratio_critical = critical_ratio
        self.margin_ratio_liquidation = liquidation_ratio
        
        self.logger.info(f"Set risk thresholds: warning={warning_ratio}, critical={critical_ratio}, liquidation={liquidation_ratio}")
    
    def set_default_margins(
        self,
        initial_margin: float = 0.1,
        maintenance_margin: float = 0.05
    ) -> None:
        """Set default margin requirements."""
        self.default_initial_margin = initial_margin
        self.default_maintenance_margin = maintenance_margin
        
        self.logger.info(f"Set default margins: initial={initial_margin}, maintenance={maintenance_margin}")
    
    def validate_position_risk(self, position_risk: PositionRisk) -> Tuple[bool, List[str]]:
        """
        Validate position risk and return warnings.
        
        Args:
            position_risk: Position risk to validate
            
        Returns:
            (is_safe, list_of_warnings)
        """
        warnings = []
        
        # Check margin ratio
        if position_risk.margin_ratio >= self.margin_ratio_liquidation:
            warnings.append(f"CRITICAL: Margin ratio {position_risk.margin_ratio:.2%} is at liquidation level")
        elif position_risk.margin_ratio >= self.margin_ratio_critical:
            warnings.append(f"HIGH: Margin ratio {position_risk.margin_ratio:.2%} is at critical level")
        elif position_risk.margin_ratio >= self.margin_ratio_warning:
            warnings.append(f"WARNING: Margin ratio {position_risk.margin_ratio:.2%} is high")
        
        # Check liquidation distance
        liquidation_distance = self.calculate_liquidation_distance(position_risk)
        if liquidation_distance < 5.0:  # Less than 5% distance to liquidation
            warnings.append(f"CRITICAL: Only {liquidation_distance:.1f}% distance to liquidation")
        elif liquidation_distance < 10.0:  # Less than 10% distance to liquidation
            warnings.append(f"WARNING: Only {liquidation_distance:.1f}% distance to liquidation")
        
        # Check leverage
        if position_risk.leverage > 10.0:
            warnings.append(f"WARNING: High leverage {position_risk.leverage}x")
        
        is_safe = len(warnings) == 0 or not any("CRITICAL" in warning for warning in warnings)
        
        return is_safe, warnings