from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.leverage_constants import MIN_LEVERAGE, MAX_LEVERAGE


@dataclass
class SymbolConfig:
    symbol: str
    volatility_factor: float = 1.0
    liquidity_factor: float = 1.0
    max_position_usd: Optional[float] = None
    MAX_LEVERAGE
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

