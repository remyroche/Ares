"""
Cross-Asset Trading package.

Exports the cross-asset trading manager, trade gate, consolidated reporting,
and configuration helpers for multi-symbol trading.
"""

from .trade_gate import GlobalTradeGate
from .cross_asset_trading_manager import CrossAssetTradingManager, start_cross_asset_trading
from .consolidated_reporting import (
    generate_consolidated_report,
    generate_live_portfolio_dashboard,
)
from .cross_asset_config import CrossAssetConfig

__all__ = [
    "GlobalTradeGate",
    "CrossAssetTradingManager",
    "start_cross_asset_trading",
    "generate_consolidated_report",
    "generate_live_portfolio_dashboard",
    "CrossAssetConfig",
]

