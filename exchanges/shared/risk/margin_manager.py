"""Minimal MarginManager implementation.

This module provides a lightweight MarginManager class so that components
which import `exchanges.shared.risk.MarginManager` can be used in
market-data / backtesting contexts without requiring full margin
infrastructure.

The current implementation focuses on keeping track of simple margin
snapshots per account and exposing a small status API. It is safe to use
in data-only pipelines such as klines downloading/processing, and can be
extended later if live trading requires richer behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import logging


logger = logging.getLogger(__name__)


@dataclass
class MarginInfo:
    """Simple container for margin information.

    Attributes:
        total_margin: Total margin allocated to the account.
        used_margin: Margin currently in use.
        free_margin: Remaining free margin (derived).
    """

    total_margin: float = 0.0
    used_margin: float = 0.0
    free_margin: float = 0.0


class MarginManager:
    """Lightweight margin manager stub.

    This class is intentionally minimal and is primarily intended to
    satisfy imports from shared exchange utilities in environments where
    only market data (e.g. klines) is required. It can be safely used in
    backtesting and data-processing pipelines without impacting live
    trading logic.
    """

    def __init__(self, exchange_name: str) -> None:
        self.exchange_name = exchange_name
        self._margins: Dict[str, MarginInfo] = {}
        self.logger = logging.getLogger(f"{__name__}.{exchange_name}")

    # ------------------------------------------------------------------
    # Basic API
    # ------------------------------------------------------------------
    def get_margin(self, account_id: str = "default") -> MarginInfo:
        """Return current margin snapshot for the given account.

        If no snapshot exists yet, a zeroed MarginInfo is returned.
        """
        return self._margins.get(account_id, MarginInfo())

    def update_margin(
        self,
        account_id: str,
        total_margin: float,
        used_margin: float,
    ) -> MarginInfo:
        """Update margin snapshot for an account.

        Args:
            account_id: Logical account identifier (e.g. "cross", "isolated").
            total_margin: Total margin allocated to the account.
            used_margin: Margin currently in use.

        Returns:
            Updated MarginInfo instance.
        """
        free_margin = max(total_margin - used_margin, 0.0)
        info = MarginInfo(total_margin=total_margin, used_margin=used_margin, free_margin=free_margin)
        self._margins[account_id] = info
        self.logger.debug(
            "Updated margin for %s on %s: total=%s used=%s free=%s",
            account_id,
            self.exchange_name,
            total_margin,
            used_margin,
            free_margin,
        )
        return info

    def get_status(self) -> Dict[str, Any]:
        """Return a lightweight status summary for monitoring/diagnostics."""
        return {
            "exchange": self.exchange_name,
            "accounts": list(self._margins.keys()),
        }
