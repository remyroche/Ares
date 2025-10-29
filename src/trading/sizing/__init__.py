"""
Sizing Module

Simplified leverage and position management for trading.
Uses ML confidence scores and Kelly criterion for position sizing.
Based on existing tactician approach.
"""

from typing import Optional, Dict
from .position_sizer import PositionSizer, setup_position_sizer
from .leverage_manager import LeverageManager, setup_leverage_manager
from .risk_calculator import RiskCalculator, setup_risk_calculator
from ..config.trading_config import TradingConfig

__all__ = [
    "PositionSizer",
    "setup_position_sizer",
    "LeverageManager",
    "setup_leverage_manager",
    "RiskCalculator",
    "setup_risk_calculator",
    "setup_sizing_components"
]

async def setup_sizing_components(config: TradingConfig) -> Dict[str, Optional]:
    """
    Setup and initialize all sizing components with proper integration.
    
    Args:
        config: Trading configuration
        
    Returns:
        Dictionary with 'position_sizer', 'leverage_manager', and 'risk_calculator'
    """
    # Initialize components
    risk_calculator = await setup_risk_calculator(config)
    leverage_manager = await setup_leverage_manager(config)
    position_sizer = await setup_position_sizer(config, leverage_manager, risk_calculator)
    
    return {
        'position_sizer': position_sizer,
        'leverage_manager': leverage_manager,
        'risk_calculator': risk_calculator
    }
