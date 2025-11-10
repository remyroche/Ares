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
from src.utils.tprint import tprint


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
        tprint(f"🔧 RiskCalculator.__init__ called with exchange_name={exchange_name}", "INFO")
        self.exchange_name = exchange_name
        self.logger = system_logger.getChild(f"RiskCalculator.{exchange_name}")

        # Risk thresholds
        self.margin_ratio_warning = 0.8  # 80% margin ratio warning
        self.margin_ratio_critical = 0.9  # 90% margin ratio critical
        self.margin_ratio_liquidation = 0.95  # 95% margin ratio liquidation

        # Default margin requirements
        self.default_initial_margin = 0.1  # 10% initial margin
        self.default_maintenance_margin = 0.05  # 5% maintenance margin
        tprint(f"✅ RiskCalculator initialized for {exchange_name} with thresholds: warning={self.margin_ratio_warning}, critical={self.margin_ratio_critical}", "SUCCESS")
    
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
        tprint(f"🔧 calculate_position_risk called with symbol={symbol}, position_size={position_size}, entry_price={entry_price}, current_price={current_price}, leverage={leverage}", "INFO")

        # Use default margins if not provided
        initial_margin = initial_margin or self.default_initial_margin
        maintenance_margin = maintenance_margin or self.default_maintenance_margin

        # Calculate notional value
        notional_value = abs(position_size) * current_price
        tprint(f"📊 Calculated notional_value={notional_value} for {symbol}", "INFO")

        # Calculate margin used
        margin_used = notional_value * initial_margin / leverage

        # Calculate unrealized PnL
        if position_size > 0:  # Long position
            unrealized_pnl = position_size * (current_price - entry_price)
        else:  # Short position
            unrealized_pnl = abs(position_size) * (entry_price - current_price)

        tprint(f"📊 Calculated margin_used={margin_used}, unrealized_pnl={unrealized_pnl} for {symbol}", "INFO")

        # Calculate margin ratio
        margin_ratio = margin_used / (margin_used + unrealized_pnl) if (margin_used + unrealized_pnl) > 0 else 1.0

        # Calculate liquidation price
        liquidation_price = self._calculate_liquidation_price(
            entry_price, position_size, leverage, initial_margin, maintenance_margin
        )

        # Determine risk level
        risk_level = self._determine_risk_level(margin_ratio)

        tprint(f"✅ Position risk calculated for {symbol}: margin_ratio={margin_ratio:.4f}, liquidation_price={liquidation_price}, risk_level={risk_level.value}", "SUCCESS")

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
        tprint(f"🔧 _calculate_liquidation_price called with entry_price={entry_price}, position_size={position_size}, leverage={leverage}", "INFO")

        if position_size == 0:
            tprint(f"⚠️ Position size is 0, returning liquidation price 0.0", "WARNING")
            return 0.0

        # Calculate the price at which maintenance margin is reached
        if position_size > 0:  # Long position
            # For long: liquidation_price = entry_price * (1 - initial_margin + maintenance_margin) / leverage
            liquidation_price = entry_price * (1 - initial_margin + maintenance_margin) / leverage
        else:  # Short position
            # For short: liquidation_price = entry_price * (1 + initial_margin - maintenance_margin) / leverage
            liquidation_price = entry_price * (1 + initial_margin - maintenance_margin) / leverage

        liquidation_price = max(0.0, liquidation_price)
        tprint(f"✅ Calculated liquidation_price={liquidation_price} for position_size={position_size}", "SUCCESS")
        return liquidation_price
    
    def _determine_risk_level(self, margin_ratio: float) -> RiskLevel:
        """Determine risk level based on margin ratio."""
        tprint(f"🔧 _determine_risk_level called with margin_ratio={margin_ratio:.4f}", "INFO")

        if margin_ratio >= self.margin_ratio_liquidation:
            tprint(f"❌ CRITICAL risk level detected: margin_ratio={margin_ratio:.4f} >= {self.margin_ratio_liquidation}", "ERROR")
            return RiskLevel.CRITICAL
        elif margin_ratio >= self.margin_ratio_critical:
            tprint(f"⚠️ HIGH risk level detected: margin_ratio={margin_ratio:.4f} >= {self.margin_ratio_critical}", "WARNING")
            return RiskLevel.HIGH
        elif margin_ratio >= self.margin_ratio_warning:
            tprint(f"⚠️ MEDIUM risk level detected: margin_ratio={margin_ratio:.4f} >= {self.margin_ratio_warning}", "WARNING")
            return RiskLevel.MEDIUM
        else:
            tprint(f"✅ LOW risk level: margin_ratio={margin_ratio:.4f}", "SUCCESS")
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
        tprint(f"🔧 calculate_portfolio_risk called with {len(positions)} positions, total_equity={total_equity}", "INFO")

        # Calculate totals
        total_margin_used = sum(pos.margin_used for pos in positions)
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
        total_notional = sum(pos.notional_value for pos in positions)

        tprint(f"📊 Portfolio totals: margin_used={total_margin_used}, unrealized_pnl={total_unrealized_pnl}, notional={total_notional}", "INFO")

        # Calculate portfolio margin ratio
        portfolio_margin_ratio = total_margin_used / (total_equity + total_unrealized_pnl) if (total_equity + total_unrealized_pnl) > 0 else 1.0

        # Find maximum leverage used
        max_leverage_used = max((pos.leverage for pos in positions), default=1.0)

        # Determine portfolio risk level
        portfolio_risk_level = self._determine_risk_level(portfolio_margin_ratio)

        tprint(f"✅ Portfolio risk calculated: margin_ratio={portfolio_margin_ratio:.4f}, risk_level={portfolio_risk_level.value}, max_leverage={max_leverage_used}", "SUCCESS")

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
        tprint(f"🔧 calculate_var called with {len(positions)} positions, confidence_level={confidence_level}, time_horizon={time_horizon}", "INFO")

        if not positions:
            tprint(f"⚠️ No positions provided, returning VaR=0.0", "WARNING")
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

        tprint(f"✅ VaR calculated: portfolio_value={portfolio_value}, var={var}", "SUCCESS")
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

        z_score = z_scores.get(confidence_level, 1.65)
        tprint(f"🔧 _get_z_score: confidence_level={confidence_level}, z_score={z_score}", "INFO")
        return z_score
    
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
        tprint(f"🔧 calculate_max_position_size called for {symbol} with available_margin={available_margin}, leverage={leverage}, risk_tolerance={risk_tolerance}", "INFO")

        # Calculate maximum notional value
        max_notional = available_margin * leverage / self.default_initial_margin

        # Apply risk tolerance
        max_notional *= risk_tolerance

        # Calculate maximum position size
        max_position_size = max_notional / current_price

        tprint(f"✅ Maximum position size calculated for {symbol}: {max_position_size} (max_notional={max_notional})", "SUCCESS")
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
        tprint(f"🔧 calculate_margin_requirement called for {symbol} with position_size={position_size}, current_price={current_price}, leverage={leverage}", "INFO")

        initial_margin = initial_margin or self.default_initial_margin
        notional_value = abs(position_size) * current_price
        required_margin = notional_value * initial_margin / leverage

        tprint(f"✅ Margin requirement calculated for {symbol}: {required_margin} (notional={notional_value})", "SUCCESS")
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
        tprint(f"🔧 calculate_liquidation_distance called for {position_risk.symbol}", "INFO")

        if position_risk.position_size == 0:
            tprint(f"⚠️ Position size is 0 for {position_risk.symbol}, returning 100% distance", "WARNING")
            return 100.0

        current_price = position_risk.current_price
        liquidation_price = position_risk.liquidation_price

        if position_risk.position_size > 0:  # Long position
            distance = (current_price - liquidation_price) / current_price
        else:  # Short position
            distance = (liquidation_price - current_price) / current_price

        distance_pct = max(0.0, distance * 100)

        if distance_pct < 5.0:
            tprint(f"❌ CRITICAL: {position_risk.symbol} liquidation distance is only {distance_pct:.2f}%", "ERROR")
        elif distance_pct < 10.0:
            tprint(f"⚠️ WARNING: {position_risk.symbol} liquidation distance is {distance_pct:.2f}%", "WARNING")
        else:
            tprint(f"✅ {position_risk.symbol} liquidation distance: {distance_pct:.2f}%", "SUCCESS")

        return distance_pct
    
    def get_risk_summary(self, portfolio_risk: PortfolioRisk) -> Dict[str, Any]:
        """Get a summary of portfolio risk."""
        tprint(f"🔧 get_risk_summary called for portfolio with {len(portfolio_risk.positions)} positions", "INFO")

        high_risk_positions = [pos for pos in portfolio_risk.positions if pos.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]

        summary = {
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

        if len(high_risk_positions) > 0:
            tprint(f"⚠️ Portfolio has {len(high_risk_positions)} high risk positions", "WARNING")

        tprint(f"✅ Risk summary generated: risk_level={summary['risk_level']}, margin_ratio={summary['portfolio_margin_ratio']:.4f}", "SUCCESS")
        return summary
    
    def set_risk_thresholds(
        self,
        warning_ratio: float = 0.8,
        critical_ratio: float = 0.9,
        liquidation_ratio: float = 0.95
    ) -> None:
        """Set risk thresholds."""
        tprint(f"🔧 set_risk_thresholds called with warning={warning_ratio}, critical={critical_ratio}, liquidation={liquidation_ratio}", "INFO")

        self.margin_ratio_warning = warning_ratio
        self.margin_ratio_critical = critical_ratio
        self.margin_ratio_liquidation = liquidation_ratio

        self.logger.info(f"Set risk thresholds: warning={warning_ratio}, critical={critical_ratio}, liquidation={liquidation_ratio}")
        tprint(f"✅ Risk thresholds updated successfully", "SUCCESS")
    
    def set_default_margins(
        self,
        initial_margin: float = 0.1,
        maintenance_margin: float = 0.05
    ) -> None:
        """Set default margin requirements."""
        tprint(f"🔧 set_default_margins called with initial={initial_margin}, maintenance={maintenance_margin}", "INFO")

        self.default_initial_margin = initial_margin
        self.default_maintenance_margin = maintenance_margin

        self.logger.info(f"Set default margins: initial={initial_margin}, maintenance={maintenance_margin}")
        tprint(f"✅ Default margins updated successfully", "SUCCESS")
    
    def validate_position_risk(self, position_risk: PositionRisk) -> Tuple[bool, List[str]]:
        """
        Validate position risk and return warnings.

        Args:
            position_risk: Position risk to validate

        Returns:
            (is_safe, list_of_warnings)
        """
        tprint(f"🔧 validate_position_risk called for {position_risk.symbol}", "INFO")

        warnings = []

        # Check margin ratio
        if position_risk.margin_ratio >= self.margin_ratio_liquidation:
            warnings.append(f"CRITICAL: Margin ratio {position_risk.margin_ratio:.2%} is at liquidation level")
            tprint(f"❌ CRITICAL: {position_risk.symbol} margin ratio {position_risk.margin_ratio:.2%} at liquidation level", "ERROR")
        elif position_risk.margin_ratio >= self.margin_ratio_critical:
            warnings.append(f"HIGH: Margin ratio {position_risk.margin_ratio:.2%} is at critical level")
            tprint(f"⚠️ HIGH: {position_risk.symbol} margin ratio {position_risk.margin_ratio:.2%} at critical level", "WARNING")
        elif position_risk.margin_ratio >= self.margin_ratio_warning:
            warnings.append(f"WARNING: Margin ratio {position_risk.margin_ratio:.2%} is high")
            tprint(f"⚠️ WARNING: {position_risk.symbol} margin ratio {position_risk.margin_ratio:.2%} is high", "WARNING")

        # Check liquidation distance
        liquidation_distance = self.calculate_liquidation_distance(position_risk)
        if liquidation_distance < 5.0:  # Less than 5% distance to liquidation
            warnings.append(f"CRITICAL: Only {liquidation_distance:.1f}% distance to liquidation")
        elif liquidation_distance < 10.0:  # Less than 10% distance to liquidation
            warnings.append(f"WARNING: Only {liquidation_distance:.1f}% distance to liquidation")

        # Check leverage
        if position_risk.leverage > 10.0:
            warnings.append(f"WARNING: High leverage {position_risk.leverage}x")
            tprint(f"⚠️ WARNING: {position_risk.symbol} using high leverage {position_risk.leverage}x", "WARNING")

        is_safe = len(warnings) == 0 or not any("CRITICAL" in warning for warning in warnings)

        if is_safe:
            tprint(f"✅ Position {position_risk.symbol} validated successfully, no critical issues", "SUCCESS")
        else:
            tprint(f"❌ Position {position_risk.symbol} validation failed with {len(warnings)} warnings", "ERROR")

        return is_safe, warnings