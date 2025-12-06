"""
Lightweight stubs for exchanges.shared to satisfy ExchangeInterface imports during
data processing runs that do not rely on full trading-stack implementations.
"""

# Auth
from .auth import AuthenticationManager, APIKeyManager, TimeSyncManager, SubaccountManager

# Market
from .market import MarketMetadataManager, InstrumentManager, PrecisionHelper, RiskTierManager

# Pricing
from .pricing import PriceManager, OHLCVManager, MarketDataAggregator

# Orders
from .orders import OrderManager, IdempotencyManager, PositionManager

# Risk
from .risk import RiskCalculator, LiquidationRiskManager, MarginManager

# Wallet
from .wallet import BalanceManager, WalletManager

# Reliability
from .reliability import RateLimitManager, RetryManager, AuditLogger, SystemStatusManager

# History
from .history import TradeHistoryManager, PaginationManager

# High-level managers
from .high_level import (
    HighLevelAuthManager, HighLevelMarketManager, HighLevelOrderManager,
    HighLevelRiskManager, HighLevelBalanceManager, HighLevelRateLimitManager
)

# Trading standardizer
from .UnifiedTradingStandardizer import UnifiedTradingStandardizer
try:
    # Use the real pipeline when available
    from src.training.steps.data_collection.klines_downloading_processing import (
        KlinesDataProcessingPipeline,
    )
except Exception:
    class KlinesDataProcessingPipeline:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("KlinesDataProcessingPipeline not available")

# Interfaces (minimal stubs)
from .interfaces_typed import (
    tprint, handle_errors, handle_async_errors, DataSource, ValidationResult,
    IHighLevelAuthManager, IHighLevelMarketManager, IHighLevelOrderManager,
    IHighLevelRiskManager, IHighLevelBalanceManager, IHighLevelRateLimitManager,
)

__all__ = [
    "AuthenticationManager", "APIKeyManager", "TimeSyncManager", "SubaccountManager",
    "MarketMetadataManager", "InstrumentManager", "PrecisionHelper", "RiskTierManager",
    "PriceManager", "OHLCVManager", "MarketDataAggregator",
    "OrderManager", "IdempotencyManager", "PositionManager",
    "RiskCalculator", "LiquidationRiskManager", "MarginManager",
    "BalanceManager", "WalletManager",
    "RateLimitManager", "RetryManager", "AuditLogger", "SystemStatusManager",
    "TradeHistoryManager", "PaginationManager",
    "HighLevelAuthManager", "HighLevelMarketManager", "HighLevelOrderManager",
    "HighLevelRiskManager", "HighLevelBalanceManager", "HighLevelRateLimitManager",
    "UnifiedTradingStandardizer",
    "tprint", "handle_errors", "handle_async_errors", "DataSource", "ValidationResult",
    "IHighLevelAuthManager", "IHighLevelMarketManager", "IHighLevelOrderManager",
    "IHighLevelRiskManager", "IHighLevelBalanceManager", "IHighLevelRateLimitManager",
]
