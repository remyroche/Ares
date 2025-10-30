"""
Uncertainty-Based Position Sizer for Tactician

This module provides sophisticated position sizing and leverage calculation
based on model uncertainty, ML confidence, and market volatility.

Key Features:
- Position size scaling inversely with uncertainty
- Confidence-based position sizing with configurable power scaling
- Volatility-adjusted position sizing
- Leverage calculation with uncertainty constraints
- Regime-aware position sizing
- Kelly Criterion integration for optimal sizing
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.utils.logger import system_logger
from src.core.decorators import handles_errors

logger = system_logger.getChild('UncertaintyPositionSizer')


class UncertaintyPositionSizer:
    """
    Calculate position sizes and leverage based on uncertainty metrics.
    
    Uses multiple factors to determine optimal position size:
    - ML confidence: Higher confidence = larger positions
    - Uncertainty: Higher uncertainty = smaller positions
    - Volatility: Higher volatility = smaller positions
    - Risk parameters: User-defined risk limits
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the uncertainty-based position sizer.
        
        Args:
            config: Configuration dictionary with sizing parameters
        """
        self.config = config or {}
        self.logger = logger.getChild('UncertaintyPositionSizer')
        
        # Base sizing parameters
        self.base_position_size = self.config.get('base_position_size', 0.02)  # 2% of capital
        self.max_position_size = self.config.get('max_position_size', 0.10)  # 10% max
        self.min_position_size = self.config.get('min_position_size', 0.005)  # 0.5% min
        
        # Leverage parameters
        self.base_leverage = self.config.get('base_leverage', 1.0)
        self.max_leverage = self.config.get('max_leverage', 5.0)
        self.min_leverage = self.config.get('min_leverage', 1.0)
        
        # Scaling parameters
        self.confidence_power = self.config.get('confidence_scaling_power', 2.0)  # Square by default
        self.uncertainty_sensitivity = self.config.get('uncertainty_sensitivity', 1.0)
        self.volatility_sensitivity = self.config.get('volatility_sensitivity', 1.0)
        
        # Risk management
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.02)  # 2% max risk
        self.use_kelly_criterion = self.config.get('use_kelly_criterion', False)
        self.kelly_fraction = self.config.get('kelly_fraction', 0.25)  # Quarter Kelly
        
        # Regime adjustments
        self.regime_multipliers = self.config.get('regime_multipliers', {
            'high_volatility': 0.5,
            'low_volatility': 1.2,
            'trending': 1.1,
            'ranging': 0.9,
            'normal': 1.0
        })
        
        self.logger.info(f"✅ UncertaintyPositionSizer initialized: base={self.base_position_size}, "
                        f"max={self.max_position_size}, confidence_power={self.confidence_power}")
    
    @handles_errors(fallback=0.01, context="position size calculation")
    def calculate_position_size(
        self,
        confidence: float,
        uncertainty: float,
        volatility: float,
        account_balance: float,
        regime: Optional[str] = None,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> float:
        """
        Calculate optimal position size based on uncertainty and confidence.
        
        Args:
            confidence: ML confidence score (0.0 to 1.0)
            uncertainty: Combined uncertainty metric (0.0 to 1.0)
            volatility: Market volatility measure (e.g., normalized ATR)
            account_balance: Current account balance
            regime: Market regime ('high_volatility', 'low_volatility', etc.)
            win_rate: Historical win rate for Kelly Criterion (optional)
            avg_win: Average win size for Kelly Criterion (optional)
            avg_loss: Average loss size for Kelly Criterion (optional)
        
        Returns:
            float: Position size in account currency
        
        Examples:
            >>> sizer = UncertaintyPositionSizer()
            >>> # High confidence, low uncertainty scenario
            >>> size = sizer.calculate_position_size(0.8, 0.1, 0.02, 10000.0)
            >>> size > 100.0  # Should be reasonably large
            True
            >>> # Low confidence, high uncertainty scenario
            >>> size = sizer.calculate_position_size(0.3, 0.8, 0.05, 10000.0)
            >>> size < 100.0  # Should be small
            True
        """
        try:
            # Validate inputs
            confidence = np.clip(confidence, 0.0, 1.0)
            uncertainty = np.clip(uncertainty, 0.0, 1.0)
            volatility = max(0.0, volatility)
            
            if account_balance <= 0:
                self.logger.warning("Invalid account balance, using fallback size")
                return self.min_position_size * 1000  # Assume $1000 fallback
            
            # Calculate confidence multiplier with power scaling
            # Higher power = more aggressive scaling with confidence
            confidence_multiplier = confidence ** self.confidence_power
            
            # Calculate uncertainty multiplier (inverse relationship)
            # Higher uncertainty = smaller positions
            uncertainty_multiplier = 1.0 / (1.0 + uncertainty * self.uncertainty_sensitivity)
            
            # Calculate volatility multiplier (inverse relationship)
            # Higher volatility = smaller positions
            # Normalize volatility to reasonable range (assume 0.01-0.10 typical range)
            normalized_volatility = min(volatility, 0.10) / 0.10  # Cap at 10%
            volatility_multiplier = 1.0 / (1.0 + normalized_volatility * self.volatility_sensitivity)
            
            # Apply regime multiplier if provided
            regime_multiplier = self.regime_multipliers.get(regime, 1.0) if regime else 1.0
            
            # Calculate base position size as percentage of account
            base_size_pct = self.base_position_size
            
            # Apply all multipliers
            adjusted_size_pct = (base_size_pct * 
                               confidence_multiplier * 
                               uncertainty_multiplier * 
                               volatility_multiplier * 
                               regime_multiplier)
            
            # Apply Kelly Criterion if enabled and stats available
            if self.use_kelly_criterion and all([win_rate, avg_win, avg_loss]):
                kelly_size_pct = self._calculate_kelly_size(win_rate, avg_win, avg_loss)
                # Blend with uncertainty-based size (50/50)
                adjusted_size_pct = (adjusted_size_pct + kelly_size_pct) / 2.0
            
            # Apply min/max constraints
            final_size_pct = np.clip(adjusted_size_pct, self.min_position_size, self.max_position_size)
            
            # Convert to account currency
            position_size = final_size_pct * account_balance
            
            self.logger.debug(
                f"Position size calculated: {position_size:.2f} "
                f"(pct={final_size_pct:.4f}, conf={confidence:.2f}, unc={uncertainty:.2f}, "
                f"vol={volatility:.4f}, regime={regime})"
            )
            
            return float(position_size)
            
        except Exception as e:
            self.logger.error(f"❌ Position size calculation failed: {e}")
            # Return minimum position size as fallback
            return self.min_position_size * account_balance
    
    @handles_errors(fallback=1.0, context="leverage calculation")
    def calculate_leverage(
        self,
        confidence: float,
        uncertainty: float,
        volatility: float,
        regime: Optional[str] = None,
        max_drawdown_tolerance: Optional[float] = None
    ) -> float:
        """
        Calculate optimal leverage based on uncertainty and confidence.
        
        Args:
            confidence: ML confidence score (0.0 to 1.0)
            uncertainty: Combined uncertainty metric (0.0 to 1.0)
            volatility: Market volatility measure
            regime: Market regime
            max_drawdown_tolerance: Maximum acceptable drawdown (0.0 to 1.0)
        
        Returns:
            float: Leverage multiplier (1.0 = no leverage)
        
        Examples:
            >>> sizer = UncertaintyPositionSizer()
            >>> # High confidence, low uncertainty - higher leverage
            >>> lev = sizer.calculate_leverage(0.9, 0.1, 0.01)
            >>> lev > 1.0
            True
            >>> # Low confidence, high uncertainty - minimum leverage
            >>> lev = sizer.calculate_leverage(0.3, 0.9, 0.08)
            >>> lev == 1.0
            True
        """
        try:
            # Validate inputs
            confidence = np.clip(confidence, 0.0, 1.0)
            uncertainty = np.clip(uncertainty, 0.0, 1.0)
            volatility = max(0.0, volatility)
            
            # Leverage factor based on confidence
            # Only use leverage when confidence is high
            if confidence < 0.6:
                confidence_factor = 0.0  # No leverage below 60% confidence
            else:
                # Scale from 0 to 1 as confidence goes from 0.6 to 1.0
                confidence_factor = (confidence - 0.6) / 0.4
            
            # Uncertainty factor (inverse)
            # High uncertainty = low leverage
            uncertainty_factor = 1.0 - uncertainty
            
            # Volatility factor (inverse)
            # High volatility = low leverage
            normalized_volatility = min(volatility, 0.10) / 0.10
            volatility_factor = 1.0 - normalized_volatility
            
            # Combine factors (geometric mean for conservative estimate)
            combined_factor = (confidence_factor * uncertainty_factor * volatility_factor) ** (1/3)
            
            # Apply regime multiplier
            regime_mult = self.regime_multipliers.get(regime, 1.0) if regime else 1.0
            
            # Calculate leverage
            leverage = self.base_leverage + (self.max_leverage - self.base_leverage) * combined_factor * regime_mult
            
            # Apply drawdown constraint if provided
            if max_drawdown_tolerance is not None:
                # Reduce leverage if we're close to max drawdown
                dd_tolerance = max(0.0, min(1.0, max_drawdown_tolerance))
                dd_multiplier = 1.0 - (1.0 - dd_tolerance)
                leverage *= dd_multiplier
            
            # Apply min/max constraints
            final_leverage = np.clip(leverage, self.min_leverage, self.max_leverage)
            
            self.logger.debug(
                f"Leverage calculated: {final_leverage:.2f}x "
                f"(conf={confidence:.2f}, unc={uncertainty:.2f}, vol={volatility:.4f})"
            )
            
            return float(final_leverage)
            
        except Exception as e:
            self.logger.error(f"❌ Leverage calculation failed: {e}")
            return self.min_leverage
    
    def _calculate_kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Kelly % = (Win Rate * Avg Win - (1 - Win Rate) * Avg Loss) / Avg Win
        
        Args:
            win_rate: Historical win rate (0.0 to 1.0)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)
        
        Returns:
            float: Kelly-optimal position size as percentage
        """
        try:
            if avg_win <= 0:
                return self.base_position_size
            
            # Kelly formula
            kelly_pct = ((win_rate * avg_win) - ((1 - win_rate) * avg_loss)) / avg_win
            
            # Apply Kelly fraction for risk management (typically 1/4 or 1/2 Kelly)
            fractional_kelly = kelly_pct * self.kelly_fraction
            
            # Ensure non-negative and within bounds
            kelly_size = max(0.0, min(fractional_kelly, self.max_position_size))
            
            return kelly_size
            
        except Exception as e:
            self.logger.error(f"❌ Kelly calculation failed: {e}")
            return self.base_position_size
    
    @handles_errors(fallback=(0.01, 1.0), context="combined sizing calculation")
    def calculate_position_and_leverage(
        self,
        confidence: float,
        uncertainty: float,
        volatility: float,
        account_balance: float,
        regime: Optional[str] = None,
        **kwargs
    ) -> Tuple[float, float]:
        """
        Calculate both position size and leverage simultaneously.
        
        Args:
            confidence: ML confidence score
            uncertainty: Combined uncertainty metric
            volatility: Market volatility
            account_balance: Current account balance
            regime: Market regime
            **kwargs: Additional parameters for Kelly Criterion, etc.
        
        Returns:
            Tuple of (position_size, leverage)
        """
        position_size = self.calculate_position_size(
            confidence=confidence,
            uncertainty=uncertainty,
            volatility=volatility,
            account_balance=account_balance,
            regime=regime,
            **kwargs
        )
        
        leverage = self.calculate_leverage(
            confidence=confidence,
            uncertainty=uncertainty,
            volatility=volatility,
            regime=regime
        )
        
        return position_size, leverage
    
    def get_sizing_explanation(
        self,
        confidence: float,
        uncertainty: float,
        volatility: float,
        position_size: float,
        leverage: float
    ) -> Dict[str, Any]:
        """
        Get human-readable explanation of sizing decision.
        
        Args:
            confidence: ML confidence used
            uncertainty: Uncertainty metric used
            volatility: Volatility used
            position_size: Calculated position size
            leverage: Calculated leverage
        
        Returns:
            Dict with explanation of sizing factors
        """
        return {
            'position_size': position_size,
            'leverage': leverage,
            'factors': {
                'confidence': {
                    'value': confidence,
                    'impact': 'positive' if confidence > 0.6 else 'negative',
                    'strength': confidence ** self.confidence_power
                },
                'uncertainty': {
                    'value': uncertainty,
                    'impact': 'negative',
                    'strength': 1.0 / (1.0 + uncertainty * self.uncertainty_sensitivity)
                },
                'volatility': {
                    'value': volatility,
                    'impact': 'negative',
                    'strength': 1.0 / (1.0 + (min(volatility, 0.10) / 0.10) * self.volatility_sensitivity)
                }
            },
            'reasoning': self._generate_reasoning(confidence, uncertainty, volatility, position_size, leverage)
        }
    
    def _generate_reasoning(
        self,
        confidence: float,
        uncertainty: float,
        volatility: float,
        position_size: float,
        leverage: float
    ) -> str:
        """Generate human-readable reasoning for sizing decision."""
        reasons = []
        
        if confidence > 0.75:
            reasons.append("High confidence supports larger position")
        elif confidence < 0.50:
            reasons.append("Low confidence limits position size")
        
        if uncertainty > 0.6:
            reasons.append("High uncertainty reduces position size")
        elif uncertainty < 0.3:
            reasons.append("Low uncertainty allows larger position")
        
        if volatility > 0.05:
            reasons.append("High volatility constrains position size")
        elif volatility < 0.02:
            reasons.append("Low volatility permits larger position")
        
        if leverage > 1.5:
            reasons.append(f"Leverage {leverage:.1f}x applied due to favorable conditions")
        elif leverage <= 1.0:
            reasons.append("No leverage due to unfavorable risk/uncertainty profile")
        
        return "; ".join(reasons) if reasons else "Standard position sizing applied"
    
    def update_configuration(self, new_config: Dict[str, Any]) -> None:
        """
        Update sizer configuration parameters.
        
        Args:
            new_config: New configuration parameters to merge
        """
        self.config.update(new_config)
        
        # Re-load parameters
        self.base_position_size = self.config.get('base_position_size', self.base_position_size)
        self.max_position_size = self.config.get('max_position_size', self.max_position_size)
        self.min_position_size = self.config.get('min_position_size', self.min_position_size)
        self.base_leverage = self.config.get('base_leverage', self.base_leverage)
        self.max_leverage = self.config.get('max_leverage', self.max_leverage)
        self.confidence_power = self.config.get('confidence_scaling_power', self.confidence_power)
        
        self.logger.info("Updated position sizer configuration")


# Factory function
def create_uncertainty_position_sizer(config: Optional[Dict[str, Any]] = None) -> UncertaintyPositionSizer:
    """
    Factory function to create an UncertaintyPositionSizer.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        UncertaintyPositionSizer: Initialized sizer instance
    """
    return UncertaintyPositionSizer(config)


__all__ = [
    'UncertaintyPositionSizer',
    'create_uncertainty_position_sizer'
]


