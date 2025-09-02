# src/tactician/leverage_sizer.py

"""
Simplified Leverage Sizer for high leverage trading.
Uses ML confidence scores, liquidation risk model, and market health analysis.
"""

from datetime import datetime
import logging
import contextlib
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


def handle_errors(func):
    """Simple error handling decorator."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            return None
    return wrapper


def handle_specific_errors(error_handlers, default_return=None, context=""):
    """Decorator for handling specific error types with custom return values."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_type = type(e)
                if error_type in error_handlers:
                    return_value, error_message = error_handlers[error_type]
                    logger.error(f"{error_message} in {context}: {e}")
                    return return_value
                else:
                    logger.error(f"Unexpected error in {context}: {e}")
                    return default_return
        return wrapper
    return decorator


class LeverageSizer:
    """
    Simplified leverage sizer that uses ML confidence scores and liquidation risk model
    to set leverage between 10x and 100x.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the leverage sizer."""
        self.config: Dict[str, Any] = config
        self.logger = logger
        
        # Backward-compatibility shim for legacy self.print calls
        if not hasattr(self, "print"):
            def _shim_print(message: str) -> None:
                with contextlib.suppress(Exception):
                    self.logger.error(str(message))
            self.print = _shim_print  # type: ignore[attr-defined]

        # Load configuration from step17 optimization results
        self.leverage_config: Dict[str, Any] = self.config.get("leverage_sizing", {})

        # Load step17 optimized parameters
        step17_config = self.config.get("step17_optimization", {})
        leverage_optimization = step17_config.get("leverage", {})

        # Load optimized leverage parameters
        self.min_leverage: float = leverage_optimization.get("min_leverage", 10.0)
        self.max_leverage: float = leverage_optimization.get("max_leverage", 100.0)
        self.confidence_threshold: float = leverage_optimization.get("confidence_threshold", 0.6)
        self.liquidation_buffer: float = leverage_optimization.get("liquidation_buffer", 0.05)

        # NEW: Combined confidence threshold for leverage sizing (optimizable in step17)
        self.leverage_combined_threshold: float = leverage_optimization.get("leverage_combined_threshold", 0.75)

        # Load optimized component weights
        self.ml_weight: float = leverage_optimization.get("ml_weight", 0.6)
        self.liquidation_weight: float = leverage_optimization.get("liquidation_weight", 0.4)

        # Load additional optimized parameters
        self.leverage_multiplier: float = leverage_optimization.get("leverage_multiplier", 1.0)
        self.risk_adjustment_factor: float = leverage_optimization.get("risk_adjustment_factor", 1.0)
        self.confidence_boost_threshold: float = leverage_optimization.get("confidence_boost_threshold", 0.8)
        self.max_risk_leverage: float = leverage_optimization.get("max_risk_leverage", 50.0)

        self.is_initialized: bool = False
        self.leverage_sizing_history: List[Dict[str, Any]] = []

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid leverage sizer configuration"),
            AttributeError: (False, "Missing required leverage parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="leverage sizer initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the leverage sizer asynchronously."""
        self.logger.info("Initializing leverage sizer...")

        # Validate configuration
        if not self._validate_configuration():
            return False

        self.is_initialized = True
        self.logger.info("✅ Leverage sizer initialized successfully")
        return True

    def _validate_configuration(self) -> bool:
        """Validate the configuration parameters."""
        try:
            required_keys = [
                "min_leverage",
                "max_leverage",
                "confidence_threshold",
                "liquidation_buffer",
            ]
            for key in required_keys:
                if not hasattr(self, key):
                    self.logger.error(f"Missing required configuration key: {key}")
                    return False

            # Validate parameter ranges
            if self.min_leverage <= 0 or self.max_leverage <= 0:
                self.logger.error("Leverage values must be positive")
                return False

            if self.min_leverage >= self.max_leverage:
                self.logger.error("min_leverage must be less than max_leverage")
                return False

            if not (0 <= self.confidence_threshold <= 1):
                self.logger.error("confidence_threshold must be between 0 and 1")
                return False

            if not (0 <= self.liquidation_buffer <= 1):
                self.logger.error("liquidation_buffer must be between 0 and 1")
                return False

            if not (0 <= self.leverage_combined_threshold <= 1):
                self.logger.error("leverage_combined_threshold must be between 0 and 1")
                return False

            # Validate weights
            if not (0 <= self.ml_weight <= 1) or not (0 <= self.liquidation_weight <= 1):
                self.logger.error("Weights must be between 0 and 1")
                return False

            if abs(self.ml_weight + self.liquidation_weight - 1.0) > 0.01:
                self.logger.error("Weights must sum to approximately 1.0")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def refresh_step17_configuration(self) -> bool:
        """Refresh configuration from step17 optimization results."""
        try:
            step17_config = self.config.get("step17_optimization", {})
            leverage_optimization = step17_config.get("leverage", {})

            # Update parameters
            self.min_leverage = leverage_optimization.get("min_leverage", self.min_leverage)
            self.max_leverage = leverage_optimization.get("max_leverage", self.max_leverage)
            self.confidence_threshold = leverage_optimization.get("confidence_threshold", self.confidence_threshold)
            self.liquidation_buffer = leverage_optimization.get("liquidation_buffer", self.liquidation_buffer)
            self.leverage_combined_threshold = leverage_optimization.get("leverage_combined_threshold", self.leverage_combined_threshold)
            self.ml_weight = leverage_optimization.get("ml_weight", self.ml_weight)
            self.liquidation_weight = leverage_optimization.get("liquidation_weight", self.liquidation_weight)
            self.leverage_multiplier = leverage_optimization.get("leverage_multiplier", self.leverage_multiplier)
            self.risk_adjustment_factor = leverage_optimization.get("risk_adjustment_factor", self.risk_adjustment_factor)
            self.confidence_boost_threshold = leverage_optimization.get("confidence_boost_threshold", self.confidence_boost_threshold)
            self.max_risk_leverage = leverage_optimization.get("max_risk_leverage", self.max_risk_leverage)

            self.logger.info("✅ Step17 configuration refreshed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error refreshing step17 configuration: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: ({}, "Invalid leverage calculation parameters"),
            AttributeError: ({}, "Missing required leverage data"),
            KeyError: ({}, "Missing required data keys"),
        },
        default_return={},
        context="leverage sizing calculation",
    )
    async def calculate_leverage(self, 
                               ml_confidence: float,
                               market_data: Dict[str, Any],
                               position_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate optimal leverage based on ML confidence and market conditions."""
        if not self.is_initialized:
            self.logger.error("Leverage sizer not initialized")
            return {}

        try:
            self.logger.info(f"🔄 Calculating leverage for ML confidence: {ml_confidence:.3f}")

            # Calculate ML-based leverage
            ml_leverage = self._calculate_ml_leverage(ml_confidence)
            if ml_leverage is None:
                return {}

            # Calculate liquidation-safe leverage
            liquidation_leverage = self._calculate_liquidation_safe_leverage(market_data, position_data)
            if liquidation_leverage is None:
                return {}

            # Calculate weighted leverage
            weighted_leverage = self._calculate_weighted_leverage(ml_leverage, liquidation_leverage)
            if weighted_leverage is None:
                return {}

            # Apply leverage modifiers
            final_leverage = self._apply_leverage_modifiers(weighted_leverage, market_data, position_data)
            if final_leverage is None:
                return {}

            # Generate leverage reason
            reason = self._generate_leverage_reason(ml_leverage, liquidation_leverage, weighted_leverage, final_leverage)

            # Store in history
            leverage_record = {
                "timestamp": datetime.now().isoformat(),
                "ml_confidence": ml_confidence,
                "ml_leverage": ml_leverage,
                "liquidation_leverage": liquidation_leverage,
                "weighted_leverage": weighted_leverage,
                "final_leverage": final_leverage,
                "reason": reason,
                "market_data": market_data,
                "position_data": position_data
            }
            self.leverage_sizing_history.append(leverage_record)

            result = {
                "leverage": final_leverage,
                "reason": reason,
                "components": {
                    "ml_leverage": ml_leverage,
                    "liquidation_leverage": liquidation_leverage,
                    "weighted_leverage": weighted_leverage
                },
                "confidence": ml_confidence,
                "timestamp": leverage_record["timestamp"]
            }

            self.logger.info(f"✅ Leverage calculated: {final_leverage:.1f}x - {reason}")
            return result

        except Exception as e:
            self.logger.error(f"Error calculating leverage: {e}")
            return {}

    def _calculate_ml_leverage(self, ml_confidence: float) -> Optional[float]:
        """Calculate leverage based on ML confidence score."""
        try:
            if not (0 <= ml_confidence <= 1):
                self.logger.error(f"Invalid ML confidence: {ml_confidence}")
                return None

            # Base leverage calculation
            if ml_confidence < self.confidence_threshold:
                # Low confidence: use minimum leverage
                base_leverage = self.min_leverage
            elif ml_confidence >= self.confidence_boost_threshold:
                # High confidence: boost leverage
                confidence_boost = (ml_confidence - self.confidence_boost_threshold) / (1.0 - self.confidence_boost_threshold)
                boost_factor = 1.0 + (confidence_boost * 0.5)  # Up to 50% boost
                base_leverage = self.max_leverage * boost_factor
            else:
                # Medium confidence: linear interpolation
                confidence_range = self.confidence_boost_threshold - self.confidence_threshold
                confidence_position = (ml_confidence - self.confidence_threshold) / confidence_range
                base_leverage = self.min_leverage + (confidence_position * (self.max_leverage - self.min_leverage))

            # Apply ML weight
            ml_leverage = base_leverage * self.ml_weight

            return min(ml_leverage, self.max_leverage)

        except Exception as e:
            self.logger.error(f"Error calculating ML leverage: {e}")
            return None

    def _calculate_liquidation_safe_leverage(self, market_data: Dict[str, Any], position_data: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """Calculate leverage that is safe from liquidation."""
        try:
            # Extract market volatility
            volatility = market_data.get("volatility", 0.02)  # Default 2% volatility
            price = market_data.get("price", 100.0)
            
            # Calculate safe leverage based on volatility
            # Higher volatility = lower safe leverage
            volatility_factor = max(0.1, 1.0 - (volatility * 10))  # Reduce leverage as volatility increases
            
            # Base safe leverage
            safe_leverage = self.max_leverage * volatility_factor
            
            # Apply liquidation buffer
            safe_leverage *= (1.0 - self.liquidation_buffer)
            
            # Consider position-specific factors
            if position_data:
                position_size = position_data.get("size", 0)
                current_leverage = position_data.get("leverage", 1.0)
                
                # Reduce leverage for large positions
                if position_size > 1000000:  # $1M+
                    size_factor = 0.8
                elif position_size > 100000:  # $100K+
                    size_factor = 0.9
                else:
                    size_factor = 1.0
                
                safe_leverage *= size_factor
                
                # Consider current leverage to avoid rapid changes
                if current_leverage > 0:
                    leverage_change_factor = min(1.5, max(0.5, safe_leverage / current_leverage))
                    safe_leverage *= leverage_change_factor

            # Apply liquidation weight
            liquidation_leverage = safe_leverage * self.liquidation_weight
            
            return max(liquidation_leverage, self.min_leverage)

        except Exception as e:
            self.logger.error(f"Error calculating liquidation safe leverage: {e}")
            return self.min_leverage

    def _calculate_weighted_leverage(self, ml_leverage: float, liquidation_leverage: float) -> Optional[float]:
        """Calculate weighted leverage combining ML and liquidation factors."""
        try:
            if ml_leverage is None or liquidation_leverage is None:
                return None

            # Weighted combination
            weighted_leverage = (ml_leverage * self.ml_weight) + (liquidation_leverage * self.liquidation_weight)
            
            # Apply leverage multiplier
            weighted_leverage *= self.leverage_multiplier
            
            # Apply risk adjustment factor
            weighted_leverage *= self.risk_adjustment_factor
            
            # Ensure within bounds
            weighted_leverage = max(self.min_leverage, min(weighted_leverage, self.max_leverage))
            
            return weighted_leverage

        except Exception as e:
            self.logger.error(f"Error calculating weighted leverage: {e}")
            return self.min_leverage

    def _apply_leverage_modifiers(self, base_leverage: float, market_data: Dict[str, Any], position_data: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """Apply additional leverage modifiers based on market and position conditions."""
        try:
            modified_leverage = base_leverage
            
            # Market health modifier
            market_health = market_data.get("market_health", 1.0)
            if market_health < 0.5:  # Poor market health
                modified_leverage *= 0.7  # Reduce leverage by 30%
            elif market_health > 0.8:  # Good market health
                modified_leverage *= 1.1  # Increase leverage by 10%
            
            # Trend strength modifier
            trend_strength = market_data.get("trend_strength", 0.5)
            if trend_strength > 0.7:  # Strong trend
                modified_leverage *= 1.2  # Increase leverage by 20%
            elif trend_strength < 0.3:  # Weak trend
                modified_leverage *= 0.8  # Reduce leverage by 20%
            
            # Position correlation modifier
            if position_data:
                correlation = position_data.get("correlation", 0.0)
                if abs(correlation) > 0.8:  # High correlation
                    modified_leverage *= 0.9  # Reduce leverage by 10%
            
            # Ensure within bounds
            modified_leverage = max(self.min_leverage, min(modified_leverage, self.max_leverage))
            
            return modified_leverage

        except Exception as e:
            self.logger.error(f"Error applying leverage modifiers: {e}")
            return base_leverage

    def _generate_leverage_reason(self, ml_leverage: float, liquidation_leverage: float, weighted_leverage: float, final_leverage: float) -> str:
        """Generate human-readable reason for the leverage decision."""
        try:
            reasons = []
            
            # ML confidence reason
            if ml_leverage > weighted_leverage * 0.8:
                reasons.append("High ML confidence")
            elif ml_leverage < weighted_leverage * 0.4:
                reasons.append("Low ML confidence")
            
            # Liquidation safety reason
            if liquidation_leverage < weighted_leverage * 0.6:
                reasons.append("Liquidation risk mitigation")
            elif liquidation_leverage > weighted_leverage * 1.2:
                reasons.append("Conservative liquidation buffer")
            
            # Market conditions reason
            if final_leverage > weighted_leverage * 1.1:
                reasons.append("Favorable market conditions")
            elif final_leverage < weighted_leverage * 0.9:
                reasons.append("Risk-averse market conditions")
            
            if not reasons:
                reasons.append("Balanced risk-return optimization")
            
            return f"Leverage: {final_leverage:.1f}x - {'; '.join(reasons)}"

        except Exception as e:
            self.logger.error(f"Error generating leverage reason: {e}")
            return f"Leverage: {final_leverage:.1f}x (Error generating reason)"

    def get_leverage_sizing_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get leverage sizing history."""
        try:
            if limit:
                return self.leverage_sizing_history[-limit:]
            return self.leverage_sizing_history.copy()
        except Exception as e:
            self.logger.error(f"Error retrieving leverage history: {e}")
            return []

    @handle_specific_errors(
        error_handlers={
            Exception: (None, "Error during leverage sizer cleanup"),
        },
        default_return=None,
        context="leverage sizer cleanup",
    )
    async def stop(self) -> None:
        """Stop the leverage sizer."""
        try:
            self.logger.info("Stopping leverage sizer...")
            self.is_initialized = False
            self.logger.info("✅ Leverage sizer stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping leverage sizer: {e}")

    @handle_specific_errors(
        error_handlers={
            Exception: (None, "Error during leverage sizer cleanup"),
        },
        default_return=None,
        context="leverage sizer cleanup",
    )
    async def cleanup(self) -> None:
        """Clean up leverage sizer resources."""
        try:
            self.logger.info("Cleaning up leverage sizer...")
            self.leverage_sizing_history.clear()
            self.is_initialized = False
            self.logger.info("✅ Leverage sizer cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during leverage sizer cleanup: {e}")

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid leverage sizer configuration"),
            Exception: (None, "Error setting up leverage sizer"),
        },
        default_return=None,
        context="leverage sizer setup",
    )
    async def setup_leverage_sizer(config: Dict[str, Any]) -> Optional['LeverageSizer']:
        """Factory function to create and initialize a leverage sizer."""
        try:
            if config is None:
                raise ValueError("Configuration cannot be None")
            
            leverage_sizer = LeverageSizer(config)
            success = await leverage_sizer.initialize()
            
            if success:
                logger.info("✅ Leverage sizer setup completed successfully")
                return leverage_sizer
            else:
                logger.error("❌ Leverage sizer setup failed")
                return None
                
        except Exception as e:
            logger.error(f"Error setting up leverage sizer: {e}")
            return None
