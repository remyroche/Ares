from typing import Any, Dict, List
import numpy as np
import pandas as pd

from ..monitoring.comprehensive_trade_monitor import comprehensive_trade_monitor
from ..reporting.performance_reporter import generate_trading_report


async def generate_consolidated_report(report_name: str = "cross_asset_portfolio") -> Dict[str, Any]:
    """
    Build a consolidated cross-asset report from the global trade monitor.
    Returns portfolio metrics and per-symbol breakdown, plus a comprehensive report export.
    """
    trades = comprehensive_trade_monitor.completed_trades
    session_metrics = comprehensive_trade_monitor.current_session

    # High-level comprehensive report export (JSON/CSV/HTML handled inside)
    full_report = await generate_trading_report(
        trades=trades,
        session_metrics=session_metrics,
        report_name=report_name,
    )

    # Per-symbol aggregation
    df = pd.DataFrame([t.to_dict() for t in trades]) if trades else pd.DataFrame()
    per_symbol = {}
    if not df.empty and "symbol" in df.columns:
        grouped = df.groupby("symbol")
        for sym, g in grouped:
            pnl = g["pnl_absolute"].dropna() if "pnl_absolute" in g.columns else pd.Series(dtype=float)
            per_symbol[sym] = {
                "trades": int(len(g)),
                "total_pnl": float(pnl.sum()) if not pnl.empty else 0.0,
                "avg_pnl": float(pnl.mean()) if not pnl.empty else 0.0,
                "win_rate": float((pnl > 0).mean()) if not pnl.empty else 0.0,
            }

    # Simple correlation between symbol PnL series (if timestamps align)
    correlations = {}
    if not df.empty and "timestamp" in df.columns and "symbol" in df.columns:
        # build a pivot of pnl by timestamp x symbol
        if "pnl_absolute" in df.columns:
            pivot = (
                df[["timestamp", "symbol", "pnl_absolute"]]
                .dropna()
                .assign(timestamp=lambda x: pd.to_datetime(x["timestamp"]))
                .pivot_table(index="timestamp", columns="symbol", values="pnl_absolute", aggfunc="sum")
                .fillna(0.0)
            )
            if pivot.shape[1] >= 2:
                corr = pivot.corr().fillna(0.0)
                correlations = corr.to_dict()

    portfolio = {
        "total_trades": int(len(trades)),
        "completed_trades": int(len(trades)),
        "per_symbol": per_symbol,
        "correlations": correlations,
    }

    return {"portfolio": portfolio, "comprehensive_report": full_report}


async def generate_live_portfolio_dashboard() -> Dict[str, Any]:
    active = comprehensive_trade_monitor.active_trades
    completed = comprehensive_trade_monitor.completed_trades
    session = comprehensive_trade_monitor.current_session
    return {
        "active_trades": len(active),
        "completed_trades": len(completed),
        "session_id": session.session_id if session else None,
        "per_symbol_active": _count_by_symbol(active.values()),
        "per_symbol_completed": _count_by_symbol(completed),
    }


def _count_by_symbol(trades: List[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for t in trades:
        sym = getattr(t, "symbol", None) or (t.get("symbol") if isinstance(t, dict) else None)
        if sym:
            counts[sym] = counts.get(sym, 0) + 1
    return counts

