"""
Centralized Trading Costs Configuration

This module provides a single source of truth for all transaction cost assumptions
used throughout the HPO, labeling, and backtesting pipeline.

Rationale:
- Fees: ~0.1% per side on most exchanges (0.1% buy + 0.1% sell = 0.2% round-trip)
- Slippage: ~0.05% per side in normal conditions
- Spread: ~0.01-0.02% per side for liquid pairs

Total round-trip estimate: 0.25-0.35% → we use 0.30% as default
"""

from typing import Any, Dict, Optional

# ==============================================================================
# CENTRALIZED TRANSACTION COST - SINGLE SOURCE OF TRUTH
# ==============================================================================

# Default round-trip transaction cost (buy + sell + slippage + spread)
# 0.003 = 0.30% per complete trade
DEFAULT_TRANSACTION_COST: float = 0.003

# Config key for overriding in YAML/JSON configs
TRANSACTION_COST_CONFIG_KEY: str = "transaction_cost"

# Minimum floor: never assume costs below this (protects against overfitting)
MIN_TRANSACTION_COST: float = 0.001  # 0.1%

# Maximum cap: for stress testing / conservative scenarios
MAX_TRANSACTION_COST: float = 0.01  # 1.0%


def get_transaction_cost(config: Optional[Dict[str, Any]] = None) -> float:
    """
    Get transaction cost from config or use default.
    
    Args:
        config: Optional config dict that may contain 'transaction_cost' key
        
    Returns:
        Transaction cost as a float (e.g., 0.003 for 0.3%)
    """
    if config is None:
        return DEFAULT_TRANSACTION_COST
    
    try:
        cost = config.get(TRANSACTION_COST_CONFIG_KEY)
        if cost is None:
            return DEFAULT_TRANSACTION_COST
        cost = float(cost)
        if cost < MIN_TRANSACTION_COST:
            cost = MIN_TRANSACTION_COST
        if cost > MAX_TRANSACTION_COST:
            cost = MAX_TRANSACTION_COST
        return cost
    except (TypeError, ValueError):
        return DEFAULT_TRANSACTION_COST


def get_transaction_cost_bps(config: Optional[Dict[str, Any]] = None) -> float:
    """Get transaction cost in basis points (1 bp = 0.01%)."""
    return get_transaction_cost(config) * 10000


def log_transaction_cost_info(cost: float, context: str = "") -> str:
    """Generate a log message about transaction cost being used."""
    bps = cost * 10000
    pct = cost * 100
    msg = f"[TRANSACTION_COST] {context}: {pct:.2f}% ({bps:.0f} bps) per round-trip trade"
    return msg
