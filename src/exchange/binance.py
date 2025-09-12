# Import the enhanced version for backward compatibility
from .binance_enhanced import BinanceExchangeEnhanced as BinanceExchange

# Re-export for backward compatibility
__all__ = ['BinanceExchange']