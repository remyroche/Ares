"""
OKX exchange package with lazy exports to avoid circular imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .klines_adapter import OkxKlinesAdapter, create_okx_klines_adapter

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from ..okx import OkxExchange

__all__ = [
    'OkxExchange',
    'OkxKlinesAdapter',
    'create_okx_klines_adapter',
    'create_okx_exchange'
]


def __getattr__(name: str) -> Any:
    """Lazily expose attributes from the module level to avoid circular imports."""
    if name == 'OkxExchange':
        from ..okx import OkxExchange as _OkxExchange
        return _OkxExchange
    raise AttributeError(f"module 'exchanges.okx' has no attribute {name!r}")


def create_okx_exchange(*args: Any, **kwargs: Any):
    """Proxy to the actual OKX exchange factory with lazy import."""
    from ..okx import create_okx_exchange as _create_okx_exchange
    return _create_okx_exchange(*args, **kwargs)