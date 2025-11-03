"""
Sizing Module

Simplified leverage and position management for trading.
Uses ML confidence scores and Kelly criterion for position sizing.
Based on existing tactician approach.
"""

from typing import Optional, Dict, Any
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
    # Import here to avoid circular import issues
    from .position_sizer import setup_position_sizer
    
    # Initialize components
    risk_calculator: Optional[RiskCalculator] = await setup_risk_calculator(config)
    leverage_manager: Optional[LeverageManager] = await setup_leverage_manager(config)
    position_sizer: Optional[PositionSizer] = await setup_position_sizer(config, leverage_manager, risk_calculator)
    
    return {
        'position_sizer': position_sizer,
        'leverage_manager': leverage_manager,
        'risk_calculator': risk_calculator
    }
