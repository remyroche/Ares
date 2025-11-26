"""
Sizing Module

Simplified leverage and position management for trading.
Uses ML confidence scores and Kelly criterion for position sizing.
Based on existing tactician approach.
"""

from typing import Optional, Dict, Any
from src.utils.tprint import tprint, tprint_warning, tprint_info
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

    # Lazy import to avoid circular dependencies
    from .position_sizer import setup_position_sizer
    from ..integration import get_optimized_params_integration

    # Step 1: Load optimized parameters and apply them to the TradingConfig
    optimized_params: Dict[str, Any] = {}
    try:
        params_integration = get_optimized_params_integration()

        # Best-effort extraction of symbol/exchange/timeframe/direction from config
        symbol = getattr(config, 'symbol', getattr(config, 'primary_symbol', 'ETHUSDT'))
        exchange = getattr(config, 'exchange', 'binance')
        timeframe = getattr(config, 'timeframe', '15m')
        direction = getattr(config, 'direction', 'long')

        tprint_info(
            f"🔄 Loading optimized parameters for sizing components: symbol={symbol}, "
            f"exchange={exchange}, timeframe={timeframe}, direction={direction}"
        )

        optimized_params = await params_integration.get_optimized_parameters(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
        )

        if optimized_params:
            # Apply high-level thresholds/weights to the config itself
            try:
                params_integration.apply_to_config(config, optimized_params)
            except Exception as exc:
                tprint_warning(f"⚠️ Failed to apply optimized parameters to TradingConfig: {exc}")
        else:
            tprint_warning("⚠️ No optimized parameters available for sizing components; using defaults")

    except Exception as exc:
        tprint_warning(f"⚠️ Optimized parameters integration for sizing components failed: {exc}")

    tprint("Initializing sizing components: risk_calculator, leverage_manager, position_sizer")

    # Initialize components
    risk_calculator: Optional[RiskCalculator] = await setup_risk_calculator(config)
    tprint(f"Risk calculator initialized: {risk_calculator is not None}")

    leverage_manager: Optional[LeverageManager] = await setup_leverage_manager(config)
    tprint(f"Leverage manager initialized: {leverage_manager is not None}")

    position_sizer: Optional[PositionSizer] = await setup_position_sizer(config, leverage_manager, risk_calculator)
    tprint(f"Position sizer initialized: {position_sizer is not None}")

    # Step 2: Apply optimized parameters to individual components (if available)
    if optimized_params:
        try:
            params_integration = get_optimized_params_integration()

            if position_sizer is not None:
                try:
                    params_integration.apply_to_position_sizer(position_sizer, optimized_params)
                except Exception as exc:
                    tprint_warning(f"⚠️ Failed to apply optimized params to PositionSizer: {exc}")

            if risk_calculator is not None:
                try:
                    params_integration.apply_to_risk_calculator(risk_calculator, optimized_params)
                except Exception as exc:
                    tprint_warning(f"⚠️ Failed to apply optimized params to RiskCalculator: {exc}")

            if leverage_manager is not None:
                try:
                    params_integration.apply_to_leverage_manager(leverage_manager, optimized_params)
                except Exception as exc:
                    tprint_warning(f"⚠️ Failed to apply optimized params to LeverageManager: {exc}")

            tprint_info("✅ Optimized parameters applied to sizing components (where available)")

        except Exception as exc:
            tprint_warning(f"⚠️ Optimized parameters post-application failed: {exc}")

    result = {
        'position_sizer': position_sizer,
        'leverage_manager': leverage_manager,
        'risk_calculator': risk_calculator,
    }

    tprint(
        f"setup_sizing_components returning with "
        f"{sum(v is not None for v in result.values())}/3 components initialized"
    )
    return result
