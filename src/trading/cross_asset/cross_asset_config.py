from dataclasses import dataclass, field
from src.config.leverage_constants import MAX_LEVERAGE, validate_leverage
from typing import Any, Dict, List, Optional

@dataclass
class SymbolConfig:
    symbol: str
    volatility_factor: float = 1.0
    liquidity_factor: float = 1.0
    max_position_usd: Optional[float] = None
    max_leverage: float = MAX_LEVERAGE  # Using centralized max leverage
    enabled: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AllocationConfig:
    strategy: str = "equal_weight"  # equal_weight | market_cap_weighted | custom
    custom_weights: Dict[str, float] = field(default_factory=dict)

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

    def to_orchestrator_config(self, symbol: str) -> Dict[str, Any]:
        cfg = dict(self.orchestrator_base_config)
        cfg.update({
            "symbol": symbol,
            "exchange": self.exchange,
            "trading_mode": self.trading_mode,
            "account_balance": self.account_balance,
        })
        return cfg
