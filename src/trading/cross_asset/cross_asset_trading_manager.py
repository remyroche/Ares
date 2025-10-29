from typing import Any, Dict, List, Optional
import asyncio

from .trade_gate import GlobalTradeGate
from .cross_asset_config import CrossAssetConfig
from ..execution.trading_orchestrator import create_trading_orchestrator, TradingOrchestrator
from .consolidated_reporting import generate_consolidated_report, generate_live_portfolio_dashboard
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success, tprint_structured, LogLevel

class CrossAssetTradingManager:
    """
    Manages multiple TradingOrchestrator instances (one per symbol) with a shared
    GlobalTradeGate to ensure only one trade executes at a time across assets.
    Provides consolidated reporting and lifecycle management.
    """

    def __init__(self, config: CrossAssetConfig) -> None:
        """Initialize the cross-asset trading manager with configuration."""
        tprint_info(f"Initializing CrossAssetTradingManager with {len(config.symbols)} symbols")
        self.config: CrossAssetConfig = config
        self.trade_gate: Optional[GlobalTradeGate] = GlobalTradeGate(enable_queue=True) if config.single_active_trade else None
        self._orchestrators: Dict[str, TradingOrchestrator] = {}
        self._tasks: List[asyncio.Task[Any]] = []
        
        if self.trade_gate:
            tprint_info("GlobalTradeGate enabled for single active trade enforcement")
        else:
            tprint_info("GlobalTradeGate disabled - multiple concurrent trades allowed")

    async def start(self) -> None:
        """Start trading orchestrators for all enabled symbols."""
        tprint_info("Starting cross-asset trading manager")
        symbols: List[str] = [s.symbol for s in self.config.symbols if s.enabled]
        tprint_info(f"Found {len(symbols)} enabled symbols: {symbols}")
        
        if not symbols:
            tprint_error("No enabled symbols found - cannot start trading")
            return
        
        orchestrators: List[TradingOrchestrator] = []
        for symbol in symbols:
            tprint_info(f"Initializing trading orchestrator for symbol: {symbol}")
            orch_cfg: Dict[str, Any] = self.config.to_orchestrator_config(symbol)
            if self.trade_gate:
                orch_cfg["trade_gate"] = self.trade_gate
            orchestrator: TradingOrchestrator = create_trading_orchestrator(orch_cfg)
            ok: bool = await orchestrator.initialize()
            if not ok:
                tprint_error(f"Failed to initialize orchestrator for {symbol}")
                continue
            
            ok = await orchestrator.start_trading_session()
            if ok:
                self._orchestrators[symbol] = orchestrator
                orchestrators.append(orchestrator)
                tprint_success(f"Started trading orchestrator for {symbol}")
            else:
                tprint_error(f"Failed to start trading session for {symbol}")

        # Keep references for potential future coordination tasks
        self._tasks = []
        
        if orchestrators:
            tprint_success(f"Cross-asset trading manager started: {len(orchestrators)}/{len(symbols)} orchestrators active")
        else:
            tprint_error("No orchestrators successfully started - trading manager inactive")

    async def stop(self) -> None:
        """Stop all trading orchestrators and clean up resources."""
        tprint_info(f"Stopping cross-asset trading manager ({len(self._orchestrators)} orchestrators)")
        stopped_count: int = 0
        for symbol, orch in self._orchestrators.items():
            try:
                tprint_info(f"Stopping orchestrator for {symbol}")
                await orch.stop_trading_session()
                stopped_count += 1
            except Exception as e:
                tprint_error(f"Error stopping orchestrator for {symbol}: {e}")
        self._orchestrators.clear()
        self._tasks.clear()
        tprint_success(f"Cross-asset trading manager stopped ({stopped_count} orchestrators stopped)")

    def get_manager_stats(self) -> Dict[str, Any]:
        """Get statistics about the manager and all orchestrators."""
        tprint_info("Collecting cross-asset trading manager statistics")
        stats: Dict[str, Any] = {
            "symbols": list(self._orchestrators.keys()),
            "gate": self.trade_gate.state() if self.trade_gate else {"enabled": False},
            "orchestrators": {},
        }
        for sym, orch in self._orchestrators.items():
            try:
                stats["orchestrators"][sym] = orch.get_orchestrator_stats()
            except Exception as e:
                tprint_error(f"Error getting stats for orchestrator {sym}: {e}")
                stats["orchestrators"][sym] = {"error": "unavailable"}
        return stats

    async def generate_consolidated_report(self) -> Dict[str, Any]:
        """Generate a consolidated report across all symbols."""
        tprint_info("Generating consolidated report via manager")
        return await generate_consolidated_report()

    async def generate_live_portfolio_dashboard(self) -> Dict[str, Any]:
        """Generate a live portfolio dashboard."""
        tprint_info("Generating live portfolio dashboard via manager")
        return await generate_live_portfolio_dashboard()

async def start_cross_asset_trading(
    symbols: List[str],
    trading_mode: str = "paper",
    exchange: str = "binance",
    account_balance: float = 10_000.0,
    orchestrator_base_config: Optional[Dict[str, Any]] = None,
) -> CrossAssetTradingManager:
    """
    Convenience function to start cross-asset trading with a simplified configuration.
    
    Args:
        symbols: List of symbol strings to trade
        trading_mode: Trading mode ("paper" or "live")
        exchange: Exchange name
        account_balance: Account balance in USD
        orchestrator_base_config: Base configuration for orchestrators
        
    Returns:
        Initialized and started CrossAssetTradingManager instance
    """
    from .cross_asset_config import SymbolConfig

    tprint_info(f"Starting cross-asset trading: {len(symbols)} symbols, mode={trading_mode}, exchange={exchange}")
    base: Dict[str, Any] = orchestrator_base_config or {}
    cfg: CrossAssetConfig = CrossAssetConfig(
        symbols=[SymbolConfig(symbol=s) for s in symbols],
        trading_mode=trading_mode,
        exchange=exchange,
        account_balance=account_balance,
        single_active_trade=True,
        orchestrator_base_config=base,
    )
    manager: CrossAssetTradingManager = CrossAssetTradingManager(cfg)
    await manager.start()
    tprint_success("Cross-asset trading started successfully")
    return manager
