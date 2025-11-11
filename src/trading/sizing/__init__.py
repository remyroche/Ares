"""
Sizing Module

Simplified leverage and position management for trading.
Uses ML confidence scores and Kelly criterion for position sizing.
Based on existing tactician approach.
"""

from typing import Optional, Dict, Any
from src.utils.tprint import tprint
from .position_sizer import PositionSizer  # Temporarily removed setup_position_sizer due to import issues
from .leverage_manager import LeverageManager, setup_leverage_manager
from .risk_calculator import RiskCalculator, setup_risk_calculator
from ..config.trading_config import TradingConfig

__all__ = [
    "PositionSizer",
    # "setup_position_sizer",  # Temporarily disabled
    "LeverageManager",
    "setup_leverage_manager",
    "RiskCalculator",
    "setup_risk_calculator",
    "setup_sizing_components"
]

async def setup_sizing_components(config: TradingConfig) -> Dict[str, Optional[Any]]:
    """
    Setup and initialize all sizing components with proper integration.

    Args:
        config: Trading configuration

    Returns:
        Dictionary with 'position_sizer', 'leverage_manager', and 'risk_calculator'
    """
    tprint(f"setup_sizing_components called with config: {type(config).__name__}")

    # Import here to avoid circular import issues
    from .position_sizer import setup_position_sizer

    tprint("Initializing sizing components: risk_calculator, leverage_manager, position_sizer")

    # Initialize components
    risk_calculator: Optional[RiskCalculator] = await setup_risk_calculator(config)
    tprint(f"Risk calculator initialized: {risk_calculator is not None}")

    leverage_manager: Optional[LeverageManager] = await setup_leverage_manager(config)
    tprint(f"Leverage manager initialized: {leverage_manager is not None}")

    position_sizer: Optional[PositionSizer] = await setup_position_sizer(config, leverage_manager, risk_calculator)
    tprint(f"Position sizer initialized: {position_sizer is not None}")

    result = {
        'position_sizer': position_sizer,
        'leverage_manager': leverage_manager,
        'risk_calculator': risk_calculator
    }

    tprint(f"setup_sizing_components returning with {sum(v is not None for v in result.values())}/3 components initialized")
    return result
