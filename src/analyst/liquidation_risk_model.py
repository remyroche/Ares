# src/analyst/liquidation_risk_model.py
import pandas as pd
import numpy as np
from scipy.stats import norm # For normal distribution CDF
from typing import Tuple, List, Dict, Any, Optional
from src.utils.logger import system_logger
from src.utils.error_handler import (
    handle_errors,
    handle_data_processing_errors,
    handle_type_conversions,
    error_context,
    ErrorRecoveryStrategies
)

class ProbabilisticLiquidationRiskModel:
    """
    Calculates a probabilistic Liquidation Safety Score (LSS) based on market conditions,
    position size, leverage, and historical volatility.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = system_logger.getChild('LiquidationRiskModel')

    @handle_errors(
        exceptions=(ValueError, TypeError, KeyError),
        default_return=50.0,
        context="calculate_lss"
    )
    def calculate_lss(self, 
                     current_price: float,
                     position_size: float,
                     leverage: int,
                     side: str,
                     atr: float,
                     market_volatility: float = None,
                     account_balance: float = None) -> float:
        """
        Calculate Liquidation Safety Score (LSS) - higher is safer.
        
        Args:
            current_price: Current asset price
            position_size: Position size in base currency
            leverage: Current leverage
            side: 'long' or 'short'
            atr: Average True Range
            market_volatility: Optional market volatility metric
            account_balance: Optional account balance for additional risk assessment
            
        Returns:
            float: LSS score between 0-100 (higher = safer)
        """
        try:
            if current_price <= 0 or position_size <= 0 or leverage <= 0:
                self.logger.warning("Invalid parameters for LSS calculation")
                return 50.0
            
            # Calculate base liquidation distance
            liquidation_distance = self._calculate_liquidation_distance(
                current_price, position_size, leverage, side
            )
            
            # Calculate volatility risk
            volatility_risk = self._calculate_volatility_risk(atr, market_volatility)
            
            # Calculate position size risk
            position_risk = self._calculate_position_risk(position_size, account_balance)
            
            # Calculate leverage risk
            leverage_risk = self._calculate_leverage_risk(leverage)
            
            # Combine risks into final LSS
            lss = self._combine_risk_factors(
                liquidation_distance, volatility_risk, position_risk, leverage_risk
            )
            
            self.logger.info(f"LSS calculated: {lss:.2f} "
                           f"(Distance: {liquidation_distance:.4f}, "
                           f"Vol Risk: {volatility_risk:.2f}, "
                           f"Pos Risk: {position_risk:.2f}, "
                           f"Lev Risk: {leverage_risk:.2f})")
            
            return max(0.0, min(100.0, lss))
            
        except Exception as e:
            self.logger.error(f"Error calculating LSS: {e}")
            return 50.0

    @handle_data_processing_errors(
        default_return=0.0,
        context="calculate_liquidation_distance"
    )
    def _calculate_liquidation_distance(self, current_price: float, position_size: float, 
                                      leverage: int, side: str) -> float:
        """Calculate the distance to liquidation price."""
        try:
            # Calculate liquidation price
            if side.lower() == 'long':
                liquidation_price = current_price * (1 - 1/leverage)
                distance = (current_price - liquidation_price) / current_price
            elif side.lower() == 'short':
                liquidation_price = current_price * (1 + 1/leverage)
                distance = (liquidation_price - current_price) / current_price
            else:
                self.logger.warning(f"Invalid side: {side}")
                return 0.0
            
            return distance
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidation distance: {e}")
            return 0.0

    @handle_data_processing_errors(
        default_return=0.0,
        context="calculate_volatility_risk"
    )
    def _calculate_volatility_risk(self, atr: float, market_volatility: float = None) -> float:
        """Calculate risk based on volatility."""
        try:
            # Use ATR as base volatility measure
            volatility_measure = atr if atr > 0 else 0.01
            
            # Normalize volatility (higher volatility = higher risk)
            # Assuming 5% daily volatility is "normal"
            normalized_volatility = volatility_measure / 0.05
            
            # Convert to risk score (0-100, higher = more risky)
            volatility_risk = min(100.0, normalized_volatility * 50)
            
            return volatility_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility risk: {e}")
            return 50.0

    @handle_data_processing_errors(
        default_return=0.0,
        context="calculate_position_risk"
    )
    def _calculate_position_risk(self, position_size: float, account_balance: float = None) -> float:
        """Calculate risk based on position size relative to account."""
        try:
            if account_balance is None or account_balance <= 0:
                # Default risk if no account balance provided
                return 25.0
            
            # Calculate position size as percentage of account
            position_pct = (position_size / account_balance) * 100
            
            # Higher position percentage = higher risk
            if position_pct <= 1.0:
                risk = 10.0  # Very small position
            elif position_pct <= 5.0:
                risk = 25.0  # Small position
            elif position_pct <= 10.0:
                risk = 50.0  # Medium position
            elif position_pct <= 20.0:
                risk = 75.0  # Large position
            else:
                risk = 100.0  # Very large position
            
            return risk
            
        except Exception as e:
            self.logger.error(f"Error calculating position risk: {e}")
            return 50.0

    @handle_data_processing_errors(
        default_return=0.0,
        context="calculate_leverage_risk"
    )
    def _calculate_leverage_risk(self, leverage: int) -> float:
        """Calculate risk based on leverage."""
        try:
            # Higher leverage = higher risk
            if leverage <= 5:
                risk = 10.0  # Very low leverage
            elif leverage <= 10:
                risk = 25.0  # Low leverage
            elif leverage <= 25:
                risk = 50.0  # Medium leverage
            elif leverage <= 50:
                risk = 75.0  # High leverage
            else:
                risk = 100.0  # Very high leverage
            
            return risk
            
        except Exception as e:
            self.logger.error(f"Error calculating leverage risk: {e}")
            return 50.0

    @handle_data_processing_errors(
        default_return=50.0,
        context="combine_risk_factors"
    )
    def _combine_risk_factors(self, liquidation_distance: float, volatility_risk: float,
                             position_risk: float, leverage_risk: float) -> float:
        """Combine all risk factors into final LSS score."""
        try:
            # Weights for different risk factors
            weights = self.config.get("lss_weights", {
                "liquidation_distance": 0.4,
                "volatility_risk": 0.25,
                "position_risk": 0.2,
                "leverage_risk": 0.15
            })
            
            # Convert liquidation distance to safety score (higher distance = safer)
            distance_safety = min(100.0, liquidation_distance * 1000)  # Scale factor
            
            # Calculate weighted average of safety scores
            lss = (
                distance_safety * weights["liquidation_distance"] +
                (100.0 - volatility_risk) * weights["volatility_risk"] +
                (100.0 - position_risk) * weights["position_risk"] +
                (100.0 - leverage_risk) * weights["leverage_risk"]
            )
            
            return lss
            
        except Exception as e:
            self.logger.error(f"Error combining risk factors: {e}")
            return 50.0

    @handle_errors(
        exceptions=(ValueError, TypeError, KeyError),
        default_return={},
        context="get_risk_metrics"
    )
    def get_risk_metrics(self, current_price: float, position_size: float, 
                        leverage: int, side: str, atr: float) -> Dict[str, Any]:
        """
        Get comprehensive risk metrics for a position.
        
        Returns:
            Dict containing various risk metrics
        """
        try:
            lss = self.calculate_lss(current_price, position_size, leverage, side, atr)
            
            # Calculate liquidation price
            if side.lower() == 'long':
                liquidation_price = current_price * (1 - 1/leverage)
            else:
                liquidation_price = current_price * (1 + 1/leverage)
            
            # Calculate distance to liquidation
            distance_to_liquidation = abs(current_price - liquidation_price) / current_price
            
            return {
                "lss": lss,
                "liquidation_price": liquidation_price,
                "distance_to_liquidation": distance_to_liquidation,
                "leverage": leverage,
                "position_size": position_size,
                "side": side,
                "current_price": current_price,
                "atr": atr
            }
            
        except Exception as e:
            self.logger.error(f"Error getting risk metrics: {e}")
            return {}
