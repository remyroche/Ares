"""
Liquidation Risk Manager for trading operations.

This module provides standardized liquidation risk management functionality
across different exchange implementations.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    LIQUIDATION = "liquidation"


class AlertType(Enum):
    """Alert type enumeration."""
    WARNING = "warning"
    MARGIN_CALL = "margin_call"
    LIQUIDATION_WARNING = "liquidation_warning"
    LIQUIDATION_IMMINENT = "liquidation_imminent"


@dataclass
class LiquidationRisk:
    """Liquidation risk assessment."""
    symbol: str
    current_price: float
    liquidation_price: float
    margin_ratio: float
    leverage: float
    risk_level: RiskLevel
    distance_to_liquidation: float  # Percentage
    estimated_liquidation_time: Optional[datetime] = None
    alerts: Optional[List[AlertType]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.alerts is None:
            self.alerts = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RiskThresholds:
    """Risk threshold configuration."""
    warning_threshold: float = 0.2  # 20% distance to liquidation
    margin_call_threshold: float = 0.1  # 10% distance to liquidation
    liquidation_warning_threshold: float = 0.05  # 5% distance to liquidation
    critical_threshold: float = 0.02  # 2% distance to liquidation


class LiquidationRiskManager:
    """
    Standardized liquidation risk manager for trading operations.
    
    Provides unified interface for liquidation risk management across different exchanges.
    """
    
    def __init__(self, exchange_name: str, thresholds: Optional[RiskThresholds] = None):
        """
        Initialize liquidation risk manager.
        
        Args:
            exchange_name: Name of the exchange
            thresholds: Risk threshold configuration
        """
        self.exchange_name = exchange_name
        self.thresholds = thresholds or RiskThresholds()
        self.logger = logging.getLogger(f"{__name__}.{exchange_name}")
        self.risk_assessments: Dict[str, LiquidationRisk] = {}
    
    def assess_liquidation_risk(self, 
                              symbol: str,
                              position_size: float,
                              entry_price: float,
                              current_price: float,
                              leverage: float,
                              margin_balance: float,
                              maintenance_margin_rate: float = 0.005) -> LiquidationRisk:
        """
        Assess liquidation risk for a position.
        
        Args:
            symbol: Trading symbol
            position_size: Position size
            entry_price: Entry price
            current_price: Current price
            leverage: Leverage multiplier
            margin_balance: Available margin balance
            maintenance_margin_rate: Maintenance margin rate (default 0.5%)
            
        Returns:
            LiquidationRisk assessment
        """
        try:
            # Calculate liquidation price
            liquidation_price = self._calculate_liquidation_price(
                position_size, entry_price, leverage, maintenance_margin_rate
            )
            
            # Calculate margin ratio
            margin_ratio = self._calculate_margin_ratio(
                position_size, current_price, liquidation_price, margin_balance
            )
            
            # Calculate distance to liquidation
            distance_to_liquidation = self._calculate_distance_to_liquidation(
                current_price, liquidation_price
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(distance_to_liquidation)
            
            # Generate alerts
            alerts = self._generate_alerts(distance_to_liquidation, risk_level)
            
            # Estimate liquidation time (simplified)
            estimated_liquidation_time = self._estimate_liquidation_time(
                distance_to_liquidation, risk_level
            )
            
            # Create risk assessment
            risk = LiquidationRisk(
                symbol=symbol,
                current_price=current_price,
                liquidation_price=liquidation_price,
                margin_ratio=margin_ratio,
                leverage=leverage,
                risk_level=risk_level,
                distance_to_liquidation=distance_to_liquidation,
                estimated_liquidation_time=estimated_liquidation_time,
                alerts=alerts,
                metadata={
                    "position_size": position_size,
                    "entry_price": entry_price,
                    "margin_balance": margin_balance,
                    "maintenance_margin_rate": maintenance_margin_rate,
                    "assessment_time": datetime.now().isoformat()
                }
            )
            
            # Store assessment
            self.risk_assessments[symbol] = risk
            
            self.logger.info(f"Assessed liquidation risk for {symbol}: {risk_level.value} ({distance_to_liquidation:.2f}%)")
            return risk
            
        except Exception as e:
            self.logger.error(f"Failed to assess liquidation risk for {symbol}: {e}")
            raise
    
    def get_risk_assessment(self, symbol: str) -> Optional[LiquidationRisk]:
        """
        Get risk assessment for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Risk assessment if exists, None otherwise
        """
        return self.risk_assessments.get(symbol)
    
    def get_all_risk_assessments(self) -> List[LiquidationRisk]:
        """
        Get all risk assessments.
        
        Returns:
            List of all risk assessments
        """
        return list(self.risk_assessments.values())
    
    def get_high_risk_positions(self) -> List[LiquidationRisk]:
        """
        Get high risk positions.
        
        Returns:
            List of high risk positions
        """
        return [
            risk for risk in self.risk_assessments.values()
            if risk.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL, RiskLevel.LIQUIDATION]
        ]
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get risk summary.
        
        Returns:
            Dictionary with risk summary
        """
        assessments = list(self.risk_assessments.values())
        
        risk_counts = {}
        for risk_level in RiskLevel:
            risk_counts[risk_level.value] = sum(1 for r in assessments if r.risk_level == risk_level)
        
        high_risk_positions = self.get_high_risk_positions()
        
        return {
            "total_positions": len(assessments),
            "risk_distribution": risk_counts,
            "high_risk_count": len(high_risk_positions),
            "critical_positions": len([r for r in high_risk_positions if r.risk_level == RiskLevel.CRITICAL]),
            "exchange": self.exchange_name,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_liquidation_price(self, 
                                  position_size: float,
                                  entry_price: float,
                                  leverage: float,
                                  maintenance_margin_rate: float) -> float:
        """
        Calculate liquidation price.
        
        Args:
            position_size: Position size
            entry_price: Entry price
            leverage: Leverage multiplier
            maintenance_margin_rate: Maintenance margin rate
            
        Returns:
            Liquidation price
        """
        # Simplified liquidation price calculation
        # For long positions: liquidation_price = entry_price * (1 - (1/leverage) + maintenance_margin_rate)
        # For short positions: liquidation_price = entry_price * (1 + (1/leverage) - maintenance_margin_rate)
        
        if position_size > 0:  # Long position
            liquidation_price = entry_price * (1 - (1/leverage) + maintenance_margin_rate)
        else:  # Short position
            liquidation_price = entry_price * (1 + (1/leverage) - maintenance_margin_rate)
        
        return max(liquidation_price, 0.0)
    
    def _calculate_margin_ratio(self,
                             position_size: float,
                             current_price: float,
                             liquidation_price: float,
                             margin_balance: float) -> float:
        """
        Calculate margin ratio.
        
        Args:
            position_size: Position size
            current_price: Current price
            liquidation_price: Liquidation price
            margin_balance: Available margin balance
            
        Returns:
            Margin ratio
        """
        # Simplified margin ratio calculation
        unrealized_pnl = position_size * (current_price - liquidation_price)
        total_margin = margin_balance + unrealized_pnl
        
        if total_margin <= 0:
            return 0.0
        
        return margin_balance / total_margin
    
    def _calculate_distance_to_liquidation(self,
                                         current_price: float,
                                         liquidation_price: float) -> float:
        """
        Calculate distance to liquidation as percentage.
        
        Args:
            current_price: Current price
            liquidation_price: Liquidation price
            
        Returns:
            Distance to liquidation in percentage
        """
        if current_price <= 0 or liquidation_price <= 0:
            return 0.0
        
        distance = abs(current_price - liquidation_price) / current_price * 100
        return distance
    
    def _determine_risk_level(self, distance_to_liquidation: float) -> RiskLevel:
        """
        Determine risk level based on distance to liquidation.
        
        Args:
            distance_to_liquidation: Distance to liquidation in percentage
            
        Returns:
            Risk level
        """
        if distance_to_liquidation <= self.thresholds.critical_threshold:
            return RiskLevel.LIQUIDATION
        elif distance_to_liquidation <= self.thresholds.liquidation_warning_threshold:
            return RiskLevel.CRITICAL
        elif distance_to_liquidation <= self.thresholds.margin_call_threshold:
            return RiskLevel.HIGH
        elif distance_to_liquidation <= self.thresholds.warning_threshold:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_alerts(self, 
                        distance_to_liquidation: float,
                        risk_level: RiskLevel) -> List[AlertType]:
        """
        Generate alerts based on risk assessment.
        
        Args:
            distance_to_liquidation: Distance to liquidation in percentage
            risk_level: Risk level
            
        Returns:
            List of alerts
        """
        alerts = []
        
        if risk_level == RiskLevel.LIQUIDATION:
            alerts.append(AlertType.LIQUIDATION_IMMINENT)
        elif risk_level == RiskLevel.CRITICAL:
            alerts.append(AlertType.LIQUIDATION_WARNING)
        elif risk_level == RiskLevel.HIGH:
            alerts.append(AlertType.MARGIN_CALL)
        elif risk_level == RiskLevel.MEDIUM:
            alerts.append(AlertType.WARNING)
        
        return alerts
    
    def _estimate_liquidation_time(self,
                                  distance_to_liquidation: float,
                                  risk_level: RiskLevel) -> Optional[datetime]:
        """
        Estimate time until liquidation (simplified).
        
        Args:
            distance_to_liquidation: Distance to liquidation in percentage
            risk_level: Risk level
            
        Returns:
            Estimated liquidation time or None
        """
        # This is a very simplified estimation
        # In reality, this would depend on volatility, market conditions, etc.
        
        if risk_level == RiskLevel.LOW:
            return None
        
        # Estimate based on risk level (very rough approximation)
        time_minutes = {
            RiskLevel.MEDIUM: 60,  # 1 hour
            RiskLevel.HIGH: 30,    # 30 minutes
            RiskLevel.CRITICAL: 10, # 10 minutes
            RiskLevel.LIQUIDATION: 5  # 5 minutes
        }
        
        minutes = time_minutes.get(risk_level, 0)
        if minutes > 0:
            return datetime.now() + timedelta(minutes=minutes)
        
        return None