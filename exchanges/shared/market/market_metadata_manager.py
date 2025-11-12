"""
Market Metadata Manager - Alias for market_metadata.py

This file provides backward compatibility by re-exporting MarketMetadataManager
from the market_metadata module.
"""

from .market_metadata import MarketMetadataManager

__all__ = ['MarketMetadataManager']