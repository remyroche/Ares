"""
Position Sizer

Calculates optimal position sizes based on regime probabilities, risk parameters,
and portfolio constraints.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from ..config.trading_config import TradingConfig, RiskLevel
from ..config.regime_config import RegimeType
from .risk_calculator import RiskCalculator

logger = system_logger.getChild('PositionSizer')

@dataclass
class PositionSizeResult:
    """Position sizing result."""
    symbol: str
    recommended_size: float
    max_size: float
    min_size: float
    leverage: float
    risk_amount: float
    confidence: float
    regime_weights: Dict[RegimeType, float]
    sizing_method: str
    metadata: Dict[str, Any]

class PositionSizer:
    """
    Position sizing engine that calculates optimal position sizes based on:
    - Regime probabilities and confidence
    - Risk parameters and portfolio constraints
    - Market volatility and conditions
    - Available capital and leverage limits
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logger.getChild('PositionSizer')
        
        # Risk calculator
        self.risk_calculator = RiskCalculator(config)
        
        # Position sizing methods
        self.sizing_methods = {
            'fixed': self._calculate_fixed_size,
            'volatility_adjusted': self._calculate_volatility_adjusted_size,
            'regime_based': self._calculate_regime_based_size,
            'kelly': self._calculate_kelly_size,
            'risk_parity': self._calculate_risk_parity_size
        }
        
        # Default sizing method based on risk level
        self.default_method = self._get_default_method()
        
    def _get_default_method(self) -> str:
        """Get default sizing method based on risk level."""
        method_map = {
            RiskLevel.CONSERVATIVE: 'fixed',
            RiskLevel.MODERATE: 'volatility_adjusted',
            RiskLevel.AGGRESSIVE: 'regime_based'
        }
        return method_map.get(self.config.risk_level, 'volatility_adjusted')
    
    @handles_errors
    @log_execution_time()
    @traced(span_name="calculate_position_size")
    async def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        regime_probabilities: Dict[RegimeType, float],
        regime_confidence: float,
        portfolio_value: float,
        available_balance: float,
        volatility: Optional[float] = None,
        method: Optional[str] = None
    ) -> PositionSizeResult:
        """
        Calculate optimal position size for a symbol.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            regime_probabilities: Regime probabilities from detector
            regime_confidence: Overall regime confidence
            portfolio_value: Total portfolio value
            available_balance: Available balance for trading
            volatility: Current volatility (optional)
            method: Sizing method to use (optional)
            
        Returns:
            PositionSizeResult: Position sizing recommendation
        """
        try:
            if method is None:
                method = self.default_method
            
            if method not in self.sizing_methods:
                raise ValueError(f"Unknown sizing method: {method}")
            
            # Calculate base position size
            sizing_func = self.sizing_methods[method]
            base_size = await sizing_func(
                symbol, current_price, regime_probabilities, regime_confidence,
                portfolio_value, available_balance, volatility
            )
            
            # Apply risk constraints
            max_size = self._calculate_max_size(portfolio_value, available_balance)
            min_size = self._calculate_min_size(portfolio_value)
            
            # Clamp to limits
            recommended_size = max(min(base_size, max_size), min_size)
            
            # Calculate leverage
            leverage = self._calculate_leverage(recommended_size, available_balance)
            
            # Calculate risk amount
            risk_amount = self._calculate_risk_amount(recommended_size, current_price)
            
            # Create result
            result = PositionSizeResult(
                symbol=symbol,
                recommended_size=recommended_size,
                max_size=max_size,
                min_size=min_size,
                leverage=leverage,
                risk_amount=risk_amount,
                confidence=regime_confidence,
                regime_weights=regime_probabilities,
                sizing_method=method,
                metadata={
                    'portfolio_value': portfolio_value,
                    'available_balance': available_balance,
                    'current_price': current_price,
                    'volatility': volatility,
                    'risk_level': self.config.risk_level.value
                }
            )
            
            self.logger.debug(f"Position size calculated for {symbol}: {recommended_size:.4f} (method: {method})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Position sizing failed for {symbol}: {e}")
            raise
    
    async def _calculate_fixed_size(
        self,
        symbol: str,
        current_price: float,
        regime_probabilities: Dict[RegimeType, float],
        regime_confidence: float,
        portfolio_value: float,
        available_balance: float,
        volatility: Optional[float] = None
    ) -> float:
        """Calculate fixed position size based on base position size."""
        try:
            # Use base position size from config
            base_size_ratio = self.config.base_position_size
            
            # Adjust based on regime confidence
            confidence_multiplier = 0.5 + (regime_confidence * 0.5)  # 0.5 to 1.0
            
            # Calculate position value
            position_value = portfolio_value * base_size_ratio * confidence_multiplier
            
            # Convert to quantity
            position_size = position_value / current_price
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ Fixed sizing calculation failed: {e}")
            return 0.0
    
    async def _calculate_volatility_adjusted_size(
        self,
        symbol: str,
        current_price: float,
        regime_probabilities: Dict[RegimeType, float],
        regime_confidence: float,
        portfolio_value: float,
        available_balance: float,
        volatility: Optional[float] = None
    ) -> float:
        """Calculate volatility-adjusted position size."""
        try:
            if volatility is None:
                volatility = 0.02  # Default 2% volatility
            
            # Base position size
            base_size_ratio = self.config.base_position_size
            
            # Volatility adjustment (inverse relationship)
            volatility_multiplier = 0.02 / max(volatility, 0.001)  # Normalize to 2% volatility
            volatility_multiplier = min(volatility_multiplier, 2.0)  # Cap at 2x
            
            # Confidence adjustment
            confidence_multiplier = 0.5 + (regime_confidence * 0.5)
            
            # Calculate position value
            position_value = portfolio_value * base_size_ratio * volatility_multiplier * confidence_multiplier
            
            # Convert to quantity
            position_size = position_value / current_price
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ Volatility-adjusted sizing calculation failed: {e}")
            return 0.0
    
    async def _calculate_regime_based_size(
        self,
        symbol: str,
        current_price: float,
        regime_probabilities: Dict[RegimeType, float],
        regime_confidence: float,
        portfolio_value: float,
        available_balance: float,
        volatility: Optional[float] = None
    ) -> float:
        """Calculate regime-based position size."""
        try:
            # Base position size
            base_size_ratio = self.config.base_position_size
            
            # Regime-based multipliers
            regime_multipliers = {
                RegimeType.TRENDING_UP: 1.5,
                RegimeType.TRENDING_DOWN: 0.5,
                RegimeType.SIDEWAYS: 0.8,
                RegimeType.HIGH_VOLATILITY: 0.6,
                RegimeType.LOW_VOLATILITY: 1.2,
                RegimeType.BREAKOUT: 1.8,
                RegimeType.REVERSAL: 0.7,
                RegimeType.MOMENTUM: 1.4,
                RegimeType.MEAN_REVERSION: 0.9,
            }
            
            # Calculate weighted regime multiplier
            regime_multiplier = 1.0
            for regime, probability in regime_probabilities.items():
                multiplier = regime_multipliers.get(regime, 1.0)
                regime_multiplier += (multiplier - 1.0) * probability
            
            # Confidence adjustment
            confidence_multiplier = 0.5 + (regime_confidence * 0.5)
            
            # Calculate position value
            position_value = portfolio_value * base_size_ratio * regime_multiplier * confidence_multiplier
            
            # Convert to quantity
            position_size = position_value / current_price
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ Regime-based sizing calculation failed: {e}")
            return 0.0
    
    async def _calculate_kelly_size(
        self,
        symbol: str,
        current_price: float,
        regime_probabilities: Dict[RegimeType, float],
        regime_confidence: float,
        portfolio_value: float,
        available_balance: float,
        volatility: Optional[float] = None
    ) -> float:
        """Calculate Kelly criterion position size."""
        try:
            # Estimate win probability and average win/loss from regime probabilities
            bullish_regimes = [
                RegimeType.TRENDING_UP, RegimeType.BREAKOUT, RegimeType.MOMENTUM
            ]
            bearish_regimes = [
                RegimeType.TRENDING_DOWN, RegimeType.REVERSAL, RegimeType.BREAKDOWN
            ]
            
            win_probability = sum(regime_probabilities.get(regime, 0) for regime in bullish_regimes)
            loss_probability = sum(regime_probabilities.get(regime, 0) for regime in bearish_regimes)
            
            # Normalize probabilities
            total_prob = win_probability + loss_probability
            if total_prob > 0:
                win_probability /= total_prob
                loss_probability /= total_prob
            
            # Estimate average win/loss ratio (simplified)
            avg_win_loss_ratio = 1.5  # Assume 1.5:1 win/loss ratio
            
            # Kelly formula: f = (bp - q) / b
            # where b = odds, p = win probability, q = loss probability
            kelly_fraction = (avg_win_loss_ratio * win_probability - loss_probability) / avg_win_loss_ratio
            
            # Apply Kelly fraction with safety factor
            kelly_fraction = max(0, min(kelly_fraction * 0.25, 0.1))  # Cap at 10% with 25% safety factor
            
            # Calculate position value
            position_value = portfolio_value * kelly_fraction
            
            # Convert to quantity
            position_size = position_value / current_price
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ Kelly sizing calculation failed: {e}")
            return 0.0
    
    async def _calculate_risk_parity_size(
        self,
        symbol: str,
        current_price: float,
        regime_probabilities: Dict[RegimeType, float],
        regime_confidence: float,
        portfolio_value: float,
        available_balance: float,
        volatility: Optional[float] = None
    ) -> float:
        """Calculate risk parity position size."""
        try:
            if volatility is None:
                volatility = 0.02  # Default 2% volatility
            
            # Target risk per position (equal risk allocation)
            target_risk_per_position = self.config.max_portfolio_risk / 5  # Assume 5 positions max
            
            # Calculate position size to achieve target risk
            position_value = (portfolio_value * target_risk_per_position) / volatility
            
            # Convert to quantity
            position_size = position_value / current_price
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ Risk parity sizing calculation failed: {e}")
            return 0.0
    
    def _calculate_max_size(self, portfolio_value: float, available_balance: float) -> float:
        """Calculate maximum position size based on constraints."""
        try:
            # Maximum position size as percentage of portfolio
            max_portfolio_ratio = self.config.max_position_size
            
            # Maximum position size as percentage of available balance
            max_balance_ratio = 0.8  # Use 80% of available balance
            
            # Calculate maximum position value
            max_portfolio_value = portfolio_value * max_portfolio_ratio
            max_balance_value = available_balance * max_balance_ratio
            
            # Use the smaller of the two
            max_position_value = min(max_portfolio_value, max_balance_value)
            
            return max_position_value
            
        except Exception as e:
            self.logger.error(f"❌ Max size calculation failed: {e}")
            return 0.0
    
    def _calculate_min_size(self, portfolio_value: float) -> float:
        """Calculate minimum position size."""
        try:
            min_position_value = portfolio_value * self.config.min_position_size
            return min_position_value
            
        except Exception as e:
            self.logger.error(f"❌ Min size calculation failed: {e}")
            return 0.0
    
    def _calculate_leverage(self, position_size: float, available_balance: float) -> float:
        """Calculate leverage for position."""
        try:
            if available_balance <= 0:
                return 1.0
            
            leverage = position_size / available_balance
            return min(leverage, self.config.max_leverage)
            
        except Exception as e:
            self.logger.error(f"❌ Leverage calculation failed: {e}")
            return 1.0
    
    def _calculate_risk_amount(self, position_size: float, current_price: float) -> float:
        """Calculate risk amount for position."""
        try:
            position_value = position_size * current_price
            risk_amount = position_value * self.config.max_portfolio_risk
            return risk_amount
            
        except Exception as e:
            self.logger.error(f"❌ Risk amount calculation failed: {e}")
            return 0.0
    
    def get_available_methods(self) -> list[str]:
        """Get list of available sizing methods."""
        return list(self.sizing_methods.keys())
    
    def set_default_method(self, method: str):
        """Set default sizing method."""
        if method in self.sizing_methods:
            self.default_method = method
            self.logger.info(f"Default sizing method set to: {method}")
        else:
            self.logger.warning(f"Unknown sizing method: {method}")
    
    async def stop(self):
        """Stop position sizer."""
        try:
            self.logger.info("🛑 Stopping Position Sizer...")
            await self.risk_calculator.stop()
            self.logger.info("✅ Position Sizer stopped successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping Position Sizer: {e}")