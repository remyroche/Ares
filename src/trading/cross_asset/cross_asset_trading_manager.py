from typing import Any, Dict, List, Optional
import asyncio

from .trade_gate import GlobalTradeGate
from .cross_asset_config import CrossAssetConfig
from ..execution.trading_orchestrator import create_trading_orchestrator, TradingOrchestrator
from .consolidated_reporting import generate_consolidated_report, generate_live_portfolio_dashboard

class CrossAssetTradingManager:
    """
    Manages multiple TradingOrchestrator instances (one per symbol) with a shared
    GlobalTradeGate to ensure only one trade executes at a time across assets.
    Provides consolidated reporting and lifecycle management.
    """

    def __init__(self, config: CrossAssetConfig) -> None:
        self.config = config
        self.trade_gate = GlobalTradeGate(enable_queue=True) if config.single_active_trade else None
        self._orchestrators: Dict[str, TradingOrchestrator] = {}
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        symbols = [s.symbol for s in self.config.symbols if s.enabled]
        orchestrators: List[TradingOrchestrator] = []
        for symbol in symbols:
            orch_cfg = self.config.to_orchestrator_config(symbol)
            if self.trade_gate:
                orch_cfg["trade_gate"] = self.trade_gate
            orchestrator = create_trading_orchestrator(orch_cfg)
            ok = await orchestrator.initialize()
            if not ok:
                continue
            ok = await orchestrator.start_trading_session()
            if ok:
                self._orchestrators[symbol] = orchestrator
                orchestrators.append(orchestrator)

        # Keep references for potential future coordination tasks
        self._tasks = []

    async def stop(self) -> None:
        for orch in self._orchestrators.values():
            try:
                await orch.stop_trading_session()
            except Exception:
                pass
        self._orchestrators.clear()
        self._tasks.clear()

    def get_manager_stats(self) -> Dict[str, Any]:
        stats = {
            "symbols": list(self._orchestrators.keys()),
            "gate": self.trade_gate.state() if self.trade_gate else {"enabled": False},
            "orchestrators": {},
        }
        for sym, orch in self._orchestrators.items():
            try:
                stats["orchestrators"][sym] = orch.get_orchestrator_stats()
            except Exception:
                stats["orchestrators"][sym] = {"error": "unavailable"}
        return stats

    async def generate_consolidated_report(self) -> Dict[str, Any]:
        return await generate_consolidated_report()

    async def generate_live_portfolio_dashboard(self) -> Dict[str, Any]:
        return await generate_live_portfolio_dashboard()

async def start_cross_asset_trading(
    symbols: List[str],
    trading_mode: str = "paper",
    exchange: str = "binance",
    account_balance: float = 10_000.0,
    orchestrator_base_config: Optional[Dict[str, Any]] = None,
) -> CrossAssetTradingManager:
    from .cross_asset_config import SymbolConfig

    base = orchestrator_base_config or {}
    cfg = CrossAssetConfig(
        symbols=[SymbolConfig(symbol=s) for s in symbols],
        trading_mode=trading_mode,
        exchange=exchange,
        account_balance=account_balance,
        single_active_trade=True,
        orchestrator_base_config=base,
    )
    manager = CrossAssetTradingManager(cfg)
    await manager.start()
    return manager
