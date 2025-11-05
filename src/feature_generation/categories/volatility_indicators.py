"""
Volatility indicators feature generators.
This module provides volatility-based indicators like Bollinger Bands and ATR.
"""

from .volatility import VectorBTBollingerBandsGenerator, VectorBTAverageTrueRangeGenerator

# Create aliases for compatibility
BollingerBandsGenerator = VectorBTBollingerBandsGenerator
ATRGenerator = VectorBTAverageTrueRangeGenerator

__all__ = ['BollingerBandsGenerator', 'ATRGenerator']
