"""Minimal WalletManager implementation.

This module provides a lightweight WalletManager so that imports from
`exchanges.shared.wallet` succeed in environments where only market
or historical data is used (e.g. klines processing), without requiring
full wallet/trading infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import logging


logger = logging.getLogger(__name__)


@dataclass
class WalletInfo:
    """Simple wallet information container."""
    balances: Dict[str, float]


class WalletManager:
    """Lightweight wallet manager stub.

    Stores an in-memory map of currency -> balance. This is sufficient
    for non-trading contexts where components only need a WalletManager
    type to exist and basic status reporting.
    """

    def __init__(self, exchange_name: str) -> None:
        self.exchange_name = exchange_name
        self._balances: Dict[str, float] = {}
        self.logger = logging.getLogger(f"{__name__}.{exchange_name}")

    def get_wallet_info(self) -> WalletInfo:
        """Return a snapshot of the current wallet balances."""
        return WalletInfo(balances=dict(self._balances))

    def set_balance(self, currency: str, amount: float) -> None:
        """Set balance for a given currency (used mainly for tests or stubs)."""
        self._balances[currency.upper()] = float(amount)
        self.logger.debug(
            "Set balance for %s on %s: %s",
            currency,
            self.exchange_name,
            amount,
        )

    def get_status(self) -> Dict[str, Any]:
        """Return a minimal status summary for diagnostics."""
        return {
            "exchange": self.exchange_name,
            "currencies": list(self._balances.keys()),
        }
