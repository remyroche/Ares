from typing import Any, Dict, List, Union, Optional
import numpy as np
import pandas as pd

from ..monitoring.comprehensive_trade_monitor import comprehensive_trade_monitor, DetailedTradeMetrics
from ..reporting.performance_reporter import generate_trading_report
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success, tprint_structured, LogLevel

async def generate_consolidated_report(report_name: str = "cross_asset_portfolio") -> Dict[str, Any]:
    """
    Build a consolidated cross-asset report from the global trade monitor.
    Returns portfolio metrics and per-symbol breakdown, plus a comprehensive report export.
    """
    tprint_info(f"Generating consolidated report: {report_name}")
    trades: List[Any] = comprehensive_trade_monitor.completed_trades
    session_metrics: Any = comprehensive_trade_monitor.current_session
    tprint_info(f"Found {len(trades)} completed trades for report generation")

    # High-level comprehensive report export (JSON/CSV/HTML handled inside)
    tprint_info(f"Generating comprehensive trading report for {len(trades)} trades")
    full_report: Dict[str, Any] = await generate_trading_report(
        trades=trades,
        session_metrics=session_metrics,
        report_name=report_name,
    )
    tprint_success("Comprehensive trading report generated successfully")

    # Per-symbol aggregation
    tprint_info("Aggregating per-symbol metrics")
    df: pd.DataFrame = pd.DataFrame([t.to_dict() for t in trades]) if trades else pd.DataFrame()
    per_symbol: Dict[str, Dict[str, float]] = {}
    if not df.empty and "symbol" in df.columns:
        grouped = df.groupby("symbol")
        for sym, g in grouped:
            pnl: pd.Series = g["pnl_absolute"].dropna() if "pnl_absolute" in g.columns else pd.Series(dtype=float)
            per_symbol[sym] = {
                "trades": int(len(g)),
                "total_pnl": float(pnl.sum()) if not pnl.empty else 0.0,
                "avg_pnl": float(pnl.mean()) if not pnl.empty else 0.0,
                "win_rate": float((pnl > 0).mean()) if not pnl.empty else 0.0,
            }
        tprint_info(f"Aggregated metrics for {len(per_symbol)} symbols")
    else:
        tprint_warning("No trades data available for per-symbol aggregation")

    # Simple correlation between symbol PnL series (if timestamps align)
    tprint_info("Calculating cross-symbol correlations")
    correlations: Dict[str, Dict[str, float]] = {}
    if not df.empty and "timestamp" in df.columns and "symbol" in df.columns:
        # build a pivot of pnl by timestamp x symbol
        if "pnl_absolute" in df.columns:
            pivot: pd.DataFrame = (
                df[["timestamp", "symbol", "pnl_absolute"]]
                .dropna()
                .assign(timestamp=lambda x: pd.to_datetime(x["timestamp"]))
                .pivot_table(index="timestamp", columns="symbol", values="pnl_absolute", aggfunc="sum")
                .fillna(0.0)
            )
            if pivot.shape[1] >= 2:
                corr: pd.DataFrame = pivot.corr().fillna(0.0)
                correlations = corr.to_dict()
                tprint_info(f"Calculated correlations for {len(correlations)} symbols")
            else:
                tprint_warning("Insufficient symbols for correlation calculation")
        else:
            tprint_warning("Missing 'pnl_absolute' column for correlation calculation")
    else:
        tprint_warning("Missing required columns for correlation calculation")

    portfolio: Dict[str, Any] = {
        "total_trades": int(len(trades)),
        "completed_trades": int(len(trades)),
        "per_symbol": per_symbol,
        "correlations": correlations,
    }

    tprint_success(f"Consolidated report generated: {len(trades)} trades, {len(per_symbol)} symbols")
    return {"portfolio": portfolio, "comprehensive_report": full_report}

async def generate_live_portfolio_dashboard() -> Dict[str, Any]:
    tprint_info("Generating live portfolio dashboard")
    active: Dict[str, Any] = comprehensive_trade_monitor.active_trades
    completed: List[Any] = comprehensive_trade_monitor.completed_trades
    session: Any = comprehensive_trade_monitor.current_session
    
    active_count: int = len(active)
    completed_count: int = len(completed)
    tprint_info(f"Portfolio status: {active_count} active, {completed_count} completed trades")
    
    dashboard_data: Dict[str, Any] = {
        "active_trades": active_count,
        "completed_trades": completed_count,
        "session_id": session.session_id if session else None,
        "per_symbol_active": _count_by_symbol(list(active.values())),
        "per_symbol_completed": _count_by_symbol(completed),
    }
    
    tprint_success("Live portfolio dashboard generated")
    return dashboard_data

def _count_by_symbol(trades: List[Union[DetailedTradeMetrics, Dict[str, Any], Any]]) -> Dict[str, int]:
    """
    Count trades by symbol from a list of trade objects or dictionaries.
    
    Args:
        trades: List of trade objects (with .symbol attribute) or dictionaries (with 'symbol' key)
        
    Returns:
        Dictionary mapping symbol names to trade counts
    """
    counts: Dict[str, int] = {}
    for t in trades:
        sym: Optional[str] = getattr(t, "symbol", None) or (t.get("symbol") if isinstance(t, dict) else None)
        if sym:
            counts[sym] = counts.get(sym, 0) + 1
    return counts
