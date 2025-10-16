"""
Regime transition handling with position protection mechanisms.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from enum import Enum
import time

logger = logging.getLogger(__name__)

class RegimeTransitionType(Enum):
    """Types of regime transitions."""
    SMOOTH = "smooth"
    ABRUPT = "abrupt"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

class PositionProtectionLevel(Enum):
    """Levels of position protection during regime transitions."""
    NONE = "none"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    EMERGENCY = "emergency"

class RegimeTransitionHandler:
    """
    Handles regime transitions with position protection mechanisms.

    This class monitors regime changes and implements protective measures
    to prevent significant losses during volatile market transitions.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the regime transition handler.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.current_regime: Optional[str] = None
        self.previous_regime: Optional[str] = None
        self.regime_history: List[Dict[str, Any]] = []
        self.transition_detection_window = config.get('transition_detection_window', 10)
        self.volatility_threshold = config.get('volatility_threshold', 0.05)
        self.position_protection_enabled = config.get('position_protection_enabled', True)
        self.emergency_exit_threshold = config.get('emergency_exit_threshold', 0.1)

    def detect_regime_transition(self, current_regime: str,
                               regime_confidence: float,
                               market_volatility: float) -> Optional[RegimeTransitionType]:
        """
        Detect if a regime transition is occurring.

        Args:
            current_regime: Current market regime
            regime_confidence: Confidence in regime classification
            market_volatility: Current market volatility

        Returns:
            RegimeTransitionType if transition detected, None otherwise
        """
        if self.current_regime is None:
            self.current_regime = current_regime
            return None

        if current_regime != self.current_regime:
            # Regime change detected
            transition_info = {
                'timestamp': datetime.now(),
                'from_regime': self.current_regime,
                'to_regime': current_regime,
                'confidence': regime_confidence,
                'volatility': market_volatility
            }

            self.regime_history.append(transition_info)
            self.previous_regime = self.current_regime
            self.current_regime = current_regime

            # Determine transition type
            transition_type = self._classify_transition(transition_info)

            logger.info(f"Regime transition detected: {self.previous_regime} -> {current_regime} "
                       f"(Type: {transition_type.value}, Confidence: {regime_confidence:.3f})")

            return transition_type

        return None

    def _classify_transition(self, transition_info: Dict[str, Any]) -> RegimeTransitionType:
        """Classify the type of regime transition."""
        confidence = transition_info['confidence']
        volatility = transition_info['volatility']

        if confidence < 0.3:
            return RegimeTransitionType.UNKNOWN
        elif volatility > self.volatility_threshold * 2:
            return RegimeTransitionType.VOLATILE
        elif confidence < 0.6:
            return RegimeTransitionType.ABRUPT
        else:
            return RegimeTransitionType.SMOOTH

    def get_position_protection_level(self, transition_type: RegimeTransitionType,
                                    current_volatility: float,
                                    position_size: float) -> PositionProtectionLevel:
        """
        Determine the appropriate position protection level.

        Args:
            transition_type: Type of regime transition
            current_volatility: Current market volatility
            position_size: Current position size

        Returns:
            PositionProtectionLevel to apply
        """
        if not self.position_protection_enabled:
            return PositionProtectionLevel.NONE

        # Emergency protection for extreme volatility
        if current_volatility > self.emergency_exit_threshold:
            return PositionProtectionLevel.EMERGENCY

        # High volatility with large position
        if current_volatility > self.volatility_threshold and position_size > 0.1:
            return PositionProtectionLevel.AGGRESSIVE

        # Transition-specific protection
        if transition_type == RegimeTransitionType.VOLATILE:
            return PositionProtectionLevel.AGGRESSIVE
        elif transition_type == RegimeTransitionType.ABRUPT:
            return PositionProtectionLevel.CONSERVATIVE
        elif transition_type == RegimeTransitionType.UNKNOWN:
            return PositionProtectionLevel.CONSERVATIVE

        return PositionProtectionLevel.NONE

    def calculate_position_adjustment(self, protection_level: PositionProtectionLevel,
                                    current_position_size: float,
                                    current_leverage: float) -> Dict[str, float]:
        """
        Calculate position adjustments based on protection level.

        Args:
            protection_level: Level of protection to apply
            current_position_size: Current position size
            current_leverage: Current leverage

        Returns:
            Dictionary with adjusted position parameters
        """
        adjustments = {
            'position_size': current_position_size,
            'leverage': current_leverage,
            'stop_loss_tightening': 1.0,
            'take_profit_reduction': 1.0
        }

        if protection_level == PositionProtectionLevel.NONE:
            return adjustments

        elif protection_level == PositionProtectionLevel.CONSERVATIVE:
            adjustments['position_size'] *= 0.8
            adjustments['leverage'] *= 0.9
            adjustments['stop_loss_tightening'] = 0.8
            adjustments['take_profit_reduction'] = 0.9

        elif protection_level == PositionProtectionLevel.AGGRESSIVE:
            adjustments['position_size'] *= 0.5
            adjustments['leverage'] *= 0.7
            adjustments['stop_loss_tightening'] = 0.6
            adjustments['take_profit_reduction'] = 0.7

        elif protection_level == PositionProtectionLevel.EMERGENCY:
            adjustments['position_size'] = 0.0  # Close position
            adjustments['leverage'] = 1.0
            adjustments['stop_loss_tightening'] = 0.5
            adjustments['take_profit_reduction'] = 0.5

        return adjustments

    def should_exit_position(self, transition_type: RegimeTransitionType,
                           current_volatility: float,
                           position_pnl: float) -> bool:
        """
        Determine if position should be exited due to regime transition.

        Args:
            transition_type: Type of regime transition
            current_volatility: Current market volatility
            position_pnl: Current position P&L

        Returns:
            True if position should be exited
        """
        # Emergency exit for extreme volatility
        if current_volatility > self.emergency_exit_threshold:
            return True

        # Exit for volatile transitions with losses
        if (transition_type == RegimeTransitionType.VOLATILE and
            position_pnl < -0.02):  # 2% loss
            return True

        # Exit for unknown transitions with significant losses
        if (transition_type == RegimeTransitionType.UNKNOWN and
            position_pnl < -0.05):  # 5% loss
            return True

        return False

    def get_transition_summary(self) -> Dict[str, Any]:
        """Get summary of recent regime transitions."""
        if not self.regime_history:
            return {'total_transitions': 0, 'recent_transitions': []}

        recent_transitions = self.regime_history[-5:]  # Last 5 transitions

        transition_types = [t['transition_type'] for t in recent_transitions
                          if 'transition_type' in t]

        return {
            'total_transitions': len(self.regime_history),
            'recent_transitions': recent_transitions,
            'current_regime': self.current_regime,
            'previous_regime': self.previous_regime,
            'transition_types': transition_types
        }

    def reset_transition_history(self) -> None:
        """Reset the transition history."""
        self.regime_history.clear()
        self.current_regime = None
        self.previous_regime = None
        logger.info("Regime transition history reset")

# Global handler instance
_global_handler: Optional[RegimeTransitionHandler] = None

def get_global_handler() -> Optional[RegimeTransitionHandler]:
    """Get the global regime transition handler."""
    return _global_handler

def set_global_handler(handler: RegimeTransitionHandler) -> None:
    """Set the global regime transition handler."""
    global _global_handler
    _global_handler = handler

def handle_regime_transition(current_regime: str,
                           regime_confidence: float,
                           market_volatility: float,
                           current_position_size: float = 0.0,
                           current_leverage: float = 1.0) -> Dict[str, Any]:
    """
    Handle regime transition with position protection.

    Args:
        current_regime: Current market regime
        regime_confidence: Confidence in regime classification
        market_volatility: Current market volatility
        current_position_size: Current position size
        current_leverage: Current leverage

    Returns:
        Dictionary with transition handling results
    """
    handler = get_global_handler()

    if handler is None:
        logger.warning("No regime transition handler available")
        return {
            'transition_detected': False,
            'protection_level': PositionProtectionLevel.NONE,
            'adjustments': {
                'position_size': current_position_size,
                'leverage': current_leverage,
                'stop_loss_tightening': 1.0,
                'take_profit_reduction': 1.0
            }
        }

    # Detect transition
    transition_type = handler.detect_regime_transition(
        current_regime, regime_confidence, market_volatility
    )

    if transition_type is None:
        return {
            'transition_detected': False,
            'protection_level': PositionProtectionLevel.NONE,
            'adjustments': {
                'position_size': current_position_size,
                'leverage': current_leverage,
                'stop_loss_tightening': 1.0,
                'take_profit_reduction': 1.0
            }
        }

    # Get protection level
    protection_level = handler.get_position_protection_level(
        transition_type, market_volatility, current_position_size
    )

    # Calculate adjustments
    adjustments = handler.calculate_position_adjustment(
        protection_level, current_position_size, current_leverage
    )

    return {
        'transition_detected': True,
        'transition_type': transition_type,
        'protection_level': protection_level,
        'adjustments': adjustments,
        'should_exit': handler.should_exit_position(
            transition_type, market_volatility, 0.0  # PnL not available here
        )
    }
