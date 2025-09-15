"""
Backwards Compatibility Layer

This module provides backwards compatibility with existing feature generation code,
allowing seamless migration to the unified feature generation system.
"""

from .legacy_adapter import (
    LegacyFeatureAdapter,
    migrate_legacy_features,
    get_legacy_adapter,
    enable_legacy_compatibility
)

__all__ = [
    "LegacyFeatureAdapter",
    "migrate_legacy_features",
    "get_legacy_adapter",
    "enable_legacy_compatibility"
]