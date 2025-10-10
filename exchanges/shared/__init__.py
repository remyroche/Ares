"""
Shared Exchange Utilities

This module provides reusable utilities for exchange integrations,
separating common functionality from exchange-specific implementations.
"""

# High-level interfaces (recommended for most use cases)
from .high_level_wrappers_typed import (
    HighLevelAuthManager,
    HighLevelMarketManager,
    HighLevelOrderManager,
)
from .high_level_wrappers_typed_part2 import (
    HighLevelRiskManager,
    HighLevelBalanceManager,
    HighLevelRateLimitManager
)

# Low-level utilities (for advanced use cases)
from .auth import (
    APIKeyManager,
    TimeSyncManager,
    SubaccountManager,
    AuthenticationManager
)
from .market import (
    MarketMetadataManager,
    InstrumentManager,
    PrecisionHelper,
    RiskTierManager
)
from .pricing import (
    PriceManager,
    OHLCVManager,
    MarketDataAggregator
)
from .orders import (
    OrderManager,
    IdempotencyManager,
    PositionManager
)
from .risk import (
    RiskCalculator,
    LiquidationRiskManager,
    MarginManager
)
from .history import (
    TradeHistoryManager,
    PaginationManager
)
from .wallet import (
    BalanceManager,
    WalletManager
)
from .reliability import (
    RateLimitManager,
    RetryManager,
    AuditLogger,
    SystemStatusManager
)

# Interfaces and enums
from .interfaces_typed import (
    DataSource,
    ValidationResult,
    IDataManager,
    IValidationManager,
    IConfigurationManager,
    IErrorHandler,
    IPerformanceMonitor,
    IExchangeUtility,
    IHighLevelAuthManager,
    IHighLevelMarketManager,
    IHighLevelOrderManager,
    IHighLevelRiskManager,
    IHighLevelBalanceManager,
    IHighLevelRateLimitManager,
    tprint,
    handle_errors,
    handle_async_errors
)

__all__ = [
    # High-level interfaces (recommended)
    "HighLevelAuthManager",
    "HighLevelMarketManager", 
    "HighLevelOrderManager",
    "HighLevelRiskManager",
    "HighLevelBalanceManager",
    "HighLevelRateLimitManager",
    
    # Interfaces and enums
    "DataSource",
    "ValidationResult",
    "IDataManager",
    "IValidationManager", 
    "IConfigurationManager",
    "IErrorHandler",
    "IPerformanceMonitor",
    "IExchangeUtility",
    "IHighLevelAuthManager",
    "IHighLevelMarketManager",
    "IHighLevelOrderManager", 
    "IHighLevelRiskManager",
    "IHighLevelBalanceManager",
    "IHighLevelRateLimitManager",
    "tprint",
    "handle_errors",
    "handle_async_errors",
    
    # Low-level utilities (advanced use cases)
    # Auth
    "APIKeyManager",
    "TimeSyncManager", 
    "SubaccountManager",
    "AuthenticationManager",
    
    # Market
    "MarketMetadataManager",
    "InstrumentManager",
    "PrecisionHelper",
    "RiskTierManager",
    
    # Pricing
    "PriceManager",
    "OHLCVManager", 
    "MarketDataAggregator",
    
    # Orders
    "OrderManager",
    "IdempotencyManager",
    "PositionManager",
    
    # Risk
    "RiskCalculator",
    "LiquidationRiskManager",
    "MarginManager",
    
    # History
    "TradeHistoryManager",
    "PaginationManager",
    
    # Wallet
    "BalanceManager",
    "WalletManager",
    
    # Reliability
    "RateLimitManager",
    "RetryManager",
    "AuditLogger",
    "SystemStatusManager"
]