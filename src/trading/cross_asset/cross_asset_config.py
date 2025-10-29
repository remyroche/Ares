from dataclasses import dataclass, field
from src.config.leverage_constants import MAX_LEVERAGE, validate_leverage
from typing import Any, Dict, List, Optional
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

@dataclass
class SymbolConfig:
    symbol: str
    volatility_factor: float = 1.0
    liquidity_factor: float = 1.0
    max_position_usd: Optional[float] = None
    max_leverage: float = MAX_LEVERAGE  # Using centralized max leverage
    enabled: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate symbol configuration after initialization."""
        if not self.symbol or not isinstance(self.symbol, str):
            tprint_error(f"Invalid symbol configuration: symbol must be a non-empty string")
            raise ValueError("Symbol must be a non-empty string")
        
        if self.volatility_factor <= 0:
            tprint_warning(f"Symbol {self.symbol}: volatility_factor must be positive, using default 1.0")
            self.volatility_factor = 1.0
            
        if self.liquidity_factor <= 0:
            tprint_warning(f"Symbol {self.symbol}: liquidity_factor must be positive, using default 1.0")
            self.liquidity_factor = 1.0
            
        if self.max_leverage > MAX_LEVERAGE:
            tprint_warning(f"Symbol {self.symbol}: max_leverage {self.max_leverage} exceeds MAX_LEVERAGE {MAX_LEVERAGE}, capping")
            self.max_leverage = MAX_LEVERAGE
            
        tprint_info(f"SymbolConfig initialized: {self.symbol} (enabled={self.enabled}, leverage={self.max_leverage})")

@dataclass
class AllocationConfig:
    strategy: str = "equal_weight"  # equal_weight | market_cap_weighted | custom
    custom_weights: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate allocation configuration."""
        valid_strategies = ["equal_weight", "market_cap_weighted", "custom"]
        if self.strategy not in valid_strategies:
            tprint_warning(f"Invalid allocation strategy '{self.strategy}', defaulting to 'equal_weight'")
            self.strategy = "equal_weight"
        
        if self.strategy == "custom" and not self.custom_weights:
            tprint_warning("Custom allocation strategy selected but no custom_weights provided")
        
        if self.custom_weights:
            total_weight = sum(self.custom_weights.values())
            if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
                tprint_warning(f"Custom weights sum to {total_weight}, expected 1.0")
        
        tprint_info(f"AllocationConfig initialized: strategy={self.strategy}")

@dataclass
class CrossAssetConfig:
    symbols: List[SymbolConfig]
    trading_mode: str = "paper"
    exchange: str = "binance"
    account_balance: float = 10_000.0
    single_active_trade: bool = True
    report_interval_s: int = 60
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    orchestrator_base_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate cross-asset configuration."""
        if not self.symbols:
            tprint_error("CrossAssetConfig requires at least one symbol")
            raise ValueError("At least one symbol configuration is required")
        
        enabled_symbols = [s for s in self.symbols if s.enabled]
        if not enabled_symbols:
            tprint_warning("No enabled symbols in CrossAssetConfig")
        
        if self.account_balance <= 0:
            tprint_error(f"Invalid account_balance: {self.account_balance}, must be positive")
            raise ValueError("Account balance must be positive")
        
        if self.report_interval_s <= 0:
            tprint_warning(f"Invalid report_interval_s: {self.report_interval_s}, using default 60")
            self.report_interval_s = 60
        
        valid_modes = ["paper", "live"]
        if self.trading_mode not in valid_modes:
            tprint_warning(f"Invalid trading_mode '{self.trading_mode}', defaulting to 'paper'")
            self.trading_mode = "paper"
        
        tprint_info(f"CrossAssetConfig initialized: {len(enabled_symbols)}/{len(self.symbols)} symbols enabled, mode={self.trading_mode}, exchange={self.exchange}")

    def to_orchestrator_config(self, symbol: str) -> Dict[str, Any]:
        """Convert cross-asset config to orchestrator config for a specific symbol."""
        tprint_info(f"Creating orchestrator config for symbol: {symbol}")
        cfg: Dict[str, Any] = dict(self.orchestrator_base_config)
        cfg.update({
            "symbol": symbol,
            "exchange": self.exchange,
            "trading_mode": self.trading_mode,
            "account_balance": self.account_balance,
        })
        return cfg
