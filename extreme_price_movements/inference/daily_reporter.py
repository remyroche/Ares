"""Daily deployment reporting and profit skim utilities."""

import json
import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from extreme_price_movements.inference.data_fetcher import classify_api_error
from extreme_price_movements.inference.trade_logger import TradeLogger
from extreme_price_movements.portfolio_manager import PortfolioManager
from extreme_price_movements.utils import tprint

DEFAULT_REPORT_TO = "cryptoalias.rp@proton.me"
DEFAULT_STATE_PATH = "extreme_price_movements/logs/daily_report_state.json"
REPORT_TRADE_COLUMNS = [
    "timestamp",
    "entry_time",
    "exit_time",
    "holding_time_hours",
    "symbol",
    "side",
    "strategy_id",
    "meta_pred",
    "calibrated_score",
    "ridge_position_size",
    "quote_size",
    "entry_notional_quote",
    "exit_notional_quote",
    "entry_price",
    "actual_entry_price",
    "actual_exit_price",
    "signal_price",
    "decision_mid",
    "signal_gap_bps",
    "ticker_spread_bps",
    "expected_fill_price",
    "expected_fill_slippage_bps",
    "expected_total_entry_friction_bps",
    "liquidity_capacity_weight",
    "orderbook_capacity_quote_within_slippage",
    "gross_pnl_pct",
    "net_pnl_pct",
    "gross_pnl_amount",
    "net_pnl_amount",
    "net_pnl",
    "entry_fee_quote",
    "exit_fee_quote",
    "gross_to_net_cost_quote",
    "gross_to_net_cost_pct",
    "mfe",
    "mae",
    "exit_reason",
    "exit_reason_detail",
    "stop_policy_params_source",
    "stop_policy_params_hash",
    "stop_policy_schema",
    "decision_module",
    "status",
    "order_error_category",
    "error",
]


def _load_dotenv_if_present(path: str = ".env") -> None:
    """Load missing environment variables from a simple .env file."""
    env_path = Path(path)
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as exc:
        tprint(f"[DailyReporter] Failed to load .env: {exc}")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        tprint(f"[DailyReporter] Failed to read state {path}: {exc}")
        return {}


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _coerce_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _format_trade_report(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "No trades logged since the previous daily message."
    cols = [col for col in REPORT_TRADE_COLUMNS if col in trades.columns]
    view = trades.loc[:, cols].tail(200).copy()
    for col in view.columns:
        view[col] = view[col].astype(str).str.slice(0, 120)
    return view.to_csv(index=False)


def _trade_closed_mask(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=bool)
    mask = pd.Series(False, index=trades.index)
    if "lifecycle_event" in trades.columns:
        mask |= (
            trades["lifecycle_event"]
            .fillna("")
            .astype(str)
            .str.lower()
            .isin({"exit_filled", "closed", "exit"})
        )
    if "status" in trades.columns:
        mask |= (
            trades["status"]
            .fillna("")
            .astype(str)
            .str.lower()
            .isin({"closed", "completed"})
        )
    for col in (
        "actual_exit_price",
        "realized_exit_price",
        "net_pnl_amount",
        "net_pnl",
    ):
        if col in trades.columns:
            mask |= pd.to_numeric(trades[col], errors="coerce").notna()
    return mask


def _format_money(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:,.4f}"


def _format_pct(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.4%}"


def _elapsed_hours(start: Any, end: Any) -> float:
    start_ts = pd.to_datetime(start, utc=True, errors="coerce")
    end_ts = pd.to_datetime(end, utc=True, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        return np.nan
    return float(
        (pd.Timestamp(end_ts) - pd.Timestamp(start_ts)).total_seconds() / 3600.0
    )


def _strategy_trade_recap(trades: pd.DataFrame, *, total_balance: float) -> str:
    """Readable net-PnL recap by strategy for the daily deployment email."""
    if trades.empty:
        return "No trade events logged in this reporting window."

    work = trades.copy()
    strategy = work.get("strategy_id", pd.Series(index=work.index, dtype=object))
    work["strategy_id"] = strategy.fillna("unknown").astype(str).replace("", "unknown")
    closed = work.loc[_trade_closed_mask(work)].copy()
    if closed.empty:
        entries = (
            work.groupby("strategy_id", dropna=False)
            .size()
            .rename("events")
            .reset_index()
        )
        lines = [
            "No closed trades with realised net PnL yet.",
            "",
            "Trade events by strategy:",
            "strategy_id | events",
        ]
        lines.extend(
            f"{row.strategy_id} | {int(row.events)}" for row in entries.itertuples()
        )
        return "\n".join(lines)

    net_amount = pd.to_numeric(
        closed.get("net_pnl_amount", pd.Series(index=closed.index, dtype=float)),
        errors="coerce",
    )
    fallback_net = pd.to_numeric(
        closed.get("net_pnl", pd.Series(index=closed.index, dtype=float)),
        errors="coerce",
    )
    closed["net_amount"] = net_amount.fillna(fallback_net).fillna(0.0)
    closed["net_pct"] = pd.to_numeric(
        closed.get("net_pnl_pct", pd.Series(index=closed.index, dtype=float)),
        errors="coerce",
    )
    closed["notional"] = pd.Series(np.nan, index=closed.index, dtype=float)
    for notional_col in (
        "entry_notional_quote",
        "quote_size",
        "position_size_after_liquidity",
        "ridge_position_size",
    ):
        closed["notional"] = closed["notional"].fillna(
            pd.to_numeric(
                closed.get(notional_col, pd.Series(index=closed.index, dtype=float)),
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan)
        )
    closed["notional"] = closed["notional"].fillna(0.0)
    closed["holding_time_hours"] = pd.to_numeric(
        closed.get("holding_time_hours", pd.Series(index=closed.index, dtype=float)),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    if "time_in_trade_hours" in closed.columns:
        closed["holding_time_hours"] = closed["holding_time_hours"].fillna(
            pd.to_numeric(closed["time_in_trade_hours"], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
        )
    if "entry_time" in closed.columns:
        exit_source = (
            closed["exit_time"]
            if "exit_time" in closed.columns
            else closed.get("timestamp", pd.Series(index=closed.index, dtype=object))
        )
        computed_holding = [
            _elapsed_hours(entry, exit_)
            for entry, exit_ in zip(closed["entry_time"], exit_source)
        ]
        closed["holding_time_hours"] = closed["holding_time_hours"].fillna(
            pd.Series(computed_holding, index=closed.index, dtype=float)
        )
    if (closed["notional"] <= 0.0).all():
        entry = pd.to_numeric(
            closed.get(
                "actual_entry_price", pd.Series(index=closed.index, dtype=float)
            ),
            errors="coerce",
        )
        entry = entry.fillna(
            pd.to_numeric(
                closed.get("entry_price", pd.Series(index=closed.index, dtype=float)),
                errors="coerce",
            )
        )
        size = pd.to_numeric(
            closed.get("size", pd.Series(index=closed.index, dtype=float)),
            errors="coerce",
        )
        closed["notional"] = (entry.abs() * size.abs()).fillna(0.0)

    rows: List[str] = []
    header = (
        "strategy_id | closed | won | lost | avg_hold_hours | avg_net/trade | "
        "total_net | bankroll_net | total_notional"
    )
    total_closed = int(len(closed))
    total_won = int((closed["net_amount"] > 0.0).sum())
    total_lost = int((closed["net_amount"] < 0.0).sum())
    total_net = float(closed["net_amount"].sum())
    total_notional = float(closed["notional"].sum())
    bankroll_pct = (
        total_net / total_balance
        if np.isfinite(total_balance) and total_balance > 0.0
        else np.nan
    )
    rows.append("Overall net recap:")
    rows.append(
        f"closed={total_closed} won={total_won} lost={total_lost} "
        "avg_hold_hours="
        f"{_format_money(float(closed['holding_time_hours'].mean()))} "
        f"avg_net/trade={_format_money(total_net / max(total_closed, 1))} "
        f"total_net={_format_money(total_net)} "
        f"bankroll_net={_format_pct(bankroll_pct)} "
        f"total_notional={_format_money(total_notional)}"
    )
    rows.append("")
    rows.append("Per-strategy net recap:")
    rows.append(header)
    for strategy_id, grp in closed.groupby("strategy_id", dropna=False, sort=True):
        n = int(len(grp))
        net = float(grp["net_amount"].sum())
        won = int((grp["net_amount"] > 0.0).sum())
        lost = int((grp["net_amount"] < 0.0).sum())
        notional = float(grp["notional"].sum())
        bankroll = (
            net / total_balance
            if np.isfinite(total_balance) and total_balance > 0.0
            else np.nan
        )
        avg_holding_hours = float(grp["holding_time_hours"].mean())
        rows.append(
            f"{strategy_id} | {n} | {won} | {lost} | "
            f"{_format_money(avg_holding_hours)} | "
            f"{_format_money(net / max(n, 1))} | {_format_money(net)} | "
            f"{_format_pct(bankroll)} | {_format_money(notional)}"
        )
    execution_cols = {
        "ticker_spread_bps": "avg_spread_bps",
        "expected_fill_slippage_bps": "avg_book_slip_bps",
        "expected_total_entry_friction_bps": "avg_total_friction_bps",
        "liquidity_capacity_weight": "avg_liq_weight",
        "signal_gap_bps": "avg_signal_gap_bps",
    }
    available = [col for col in execution_cols if col in closed.columns]
    if available:
        rows.append("")
        rows.append("Execution-quality recap:")
        rows.append(
            "strategy_id | " + " | ".join(execution_cols[col] for col in available)
        )
        for strategy_id, grp in closed.groupby("strategy_id", dropna=False, sort=True):
            vals = []
            for col in available:
                series = pd.to_numeric(grp[col], errors="coerce")
                mean_val = float(series.mean()) if series.notna().any() else np.nan
                vals.append(_format_money(mean_val))
            rows.append(f"{strategy_id} | " + " | ".join(vals))
    mfe_loss_lines = _mfe_losing_trade_recap(closed)
    if mfe_loss_lines:
        rows.append("")
        rows.extend(mfe_loss_lines)
    return "\n".join(rows)


def _mfe_losing_trade_recap(closed: pd.DataFrame) -> List[str]:
    """Count trades that had meaningful MFE but still closed net-negative."""
    if closed.empty or "mfe" not in closed.columns:
        return []
    mfe = pd.to_numeric(closed["mfe"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    net = pd.to_numeric(
        closed.get("net_amount", pd.Series(index=closed.index, dtype=float)),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    losing = net < 0.0
    if not losing.any() or not mfe.notna().any():
        return []
    thresholds = (0.01, 0.015, 0.02, 0.025, 0.03)
    parts = [
        f"MFE>={thr:.1%}: {int(((mfe >= thr) & losing).sum())}"
        for thr in thresholds
    ]
    return ["MFE-but-losing recap:", " | ".join(parts)]


def _trades_since(logger: TradeLogger, since_ts: Optional[str]) -> pd.DataFrame:
    df = logger.read_logs()
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=logger.columns)
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    valid = ts.notna()
    if since_ts:
        since = pd.Timestamp(since_ts)
        if since.tzinfo is None:
            since = since.tz_localize("UTC")
        else:
            since = since.tz_convert("UTC")
        valid &= ts > since
    return df.loc[valid].copy()


CONFIDENCE_RECAP_COLUMNS = [
    "rank_percentile",
    "sizer_rank_percentile",
    "meta_train_rank_pct",
    "base_train_rank_pct",
    "base_rank_pct",
    "calibrated_score",
    "meta_pred",
]


def _confidence_calibration_recap(trades: pd.DataFrame) -> str:
    """Summarise whether higher-confidence closed trades performed better."""
    if trades.empty:
        return "insufficient closed trades for confidence/outcome recap"

    closed = trades.loc[_trade_closed_mask(trades)].copy()
    if closed.empty:
        return "insufficient closed trades for confidence/outcome recap"

    confidence_source = ""
    confidence = pd.Series(index=closed.index, dtype=float)
    for col in CONFIDENCE_RECAP_COLUMNS:
        if col not in closed.columns:
            continue
        candidate = pd.to_numeric(closed[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if candidate.notna().any():
            confidence_source = col
            confidence = candidate.clip(0.0, 1.0)
            break
    if not confidence_source:
        return "insufficient closed trades for confidence/outcome recap"

    pnl_pct = pd.to_numeric(
        closed.get("net_pnl_pct", pd.Series(index=closed.index, dtype=float)),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    gross_pnl_pct = pd.to_numeric(
        closed.get("gross_pnl_pct", pd.Series(index=closed.index, dtype=float)),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    pnl_pct = pnl_pct.fillna(gross_pnl_pct)

    net_amount = pd.to_numeric(
        closed.get("net_pnl_amount", pd.Series(index=closed.index, dtype=float)),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    fallback_net = pd.to_numeric(
        closed.get("net_pnl", pd.Series(index=closed.index, dtype=float)),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    net_amount = net_amount.fillna(fallback_net)

    notional = pd.to_numeric(
        closed.get("ridge_position_size", pd.Series(index=closed.index, dtype=float)),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    if notional.isna().all() or (notional.fillna(0.0).abs() <= 0.0).all():
        entry = pd.to_numeric(
            closed.get(
                "actual_entry_price", pd.Series(index=closed.index, dtype=float)
            ),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        entry = entry.fillna(
            pd.to_numeric(
                closed.get("entry_price", pd.Series(index=closed.index, dtype=float)),
                errors="coerce",
            ).replace([np.inf, -np.inf], np.nan)
        )
        size = pd.to_numeric(
            closed.get("size", pd.Series(index=closed.index, dtype=float)),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        notional = entry.abs() * size.abs()
    fallback_pct = net_amount / notional.where(notional.abs() > 0.0)
    pnl_pct = pnl_pct.fillna(fallback_pct.replace([np.inf, -np.inf], np.nan))

    win = pd.Series(np.nan, index=closed.index, dtype=float)
    win.loc[pnl_pct.notna()] = (pnl_pct.loc[pnl_pct.notna()] > 0.0).astype(float)
    amount_only = win.isna() & net_amount.notna()
    win.loc[amount_only] = (net_amount.loc[amount_only] > 0.0).astype(float)

    usable = confidence.notna() & win.notna()
    if not usable.any():
        return "insufficient closed trades for confidence/outcome recap"

    recap = pd.DataFrame(
        {
            "confidence": confidence.loc[usable],
            "pnl_pct": pnl_pct.loc[usable],
            "net_amount": net_amount.loc[usable],
            "win": win.loc[usable],
            "mfe": pd.to_numeric(
                closed.get("mfe", pd.Series(index=closed.index, dtype=float)),
                errors="coerce",
            )
            .replace([np.inf, -np.inf], np.nan)
            .loc[usable],
            "mae": pd.to_numeric(
                closed.get("mae", pd.Series(index=closed.index, dtype=float)),
                errors="coerce",
            )
            .replace([np.inf, -np.inf], np.nan)
            .loc[usable],
        }
    )

    overall_avg_pnl = (
        float(recap["pnl_pct"].mean()) if recap["pnl_pct"].notna().any() else np.nan
    )
    total_net = (
        float(recap["net_amount"].sum())
        if recap["net_amount"].notna().any()
        else np.nan
    )
    lines = [
        "closed_trades="
        f"{len(recap)} confidence_source={confidence_source} "
        f"avg_confidence={float(recap['confidence'].mean()):.4f} "
        f"hit_rate={_format_pct(float(recap['win'].mean()))} "
        f"avg_net_pnl_pct={_format_pct(overall_avg_pnl)} "
        f"total_net={_format_money(total_net)}"
    ]
    extras = []
    if recap["mfe"].notna().any():
        extras.append(f"avg_mfe={_format_pct(float(recap['mfe'].mean()))}")
    if recap["mae"].notna().any():
        extras.append(f"avg_mae={_format_pct(float(recap['mae'].mean()))}")
    if extras:
        lines.append(" ".join(extras))

    lines.extend(
        [
            "Buckets:",
            "bucket | trades | avg_conf | hit_rate | avg_net_pnl_pct | total_net | edge_vs_all",
        ]
    )
    bucket_defs = [
        ("<0.50", recap["confidence"] < 0.50),
        ("0.50-0.80", (recap["confidence"] >= 0.50) & (recap["confidence"] < 0.80)),
        ("0.80-0.95", (recap["confidence"] >= 0.80) & (recap["confidence"] < 0.95)),
        (">=0.95", recap["confidence"] >= 0.95),
    ]
    bucket_stats: Dict[str, Dict[str, float]] = {}
    for label, mask in bucket_defs:
        bucket = recap.loc[mask]
        avg_pnl = (
            float(bucket["pnl_pct"].mean())
            if bucket["pnl_pct"].notna().any()
            else np.nan
        )
        bucket_net = (
            float(bucket["net_amount"].sum())
            if bucket["net_amount"].notna().any()
            else np.nan
        )
        edge = (
            avg_pnl - overall_avg_pnl
            if np.isfinite(avg_pnl) and np.isfinite(overall_avg_pnl)
            else np.nan
        )
        bucket_stats[label] = {"avg_pnl": avg_pnl, "edge": edge}
        avg_conf = float(bucket["confidence"].mean()) if not bucket.empty else np.nan
        hit_rate = float(bucket["win"].mean()) if not bucket.empty else np.nan
        lines.append(
            f"{label} | {len(bucket)} | {_format_pct(avg_conf)} | "
            f"{_format_pct(hit_rate)} | {_format_pct(avg_pnl)} | "
            f"{_format_money(bucket_net)} | {_format_pct(edge)}"
        )

    paired = recap[["confidence", "pnl_pct"]].dropna()
    if len(paired) >= 3:
        spearman = float(
            paired["confidence"].corr(paired["pnl_pct"], method="spearman")
        )
        slope = float(np.polyfit(paired["confidence"], paired["pnl_pct"], 1)[0])
        lines.append(
            "Monotonicity: "
            f"spearman={spearman:.4f} "
            f"slope_pp_per_10pct_conf={slope * 0.10 * 100.0:.4f}"
        )

    if confidence_source == "calibrated_score":
        expected_hit_rate = float(recap["confidence"].mean())
        realised_hit_rate = float(recap["win"].mean())
        brier = float(np.mean((recap["confidence"] - recap["win"]) ** 2))
        lines.append(
            "Probability calibration: "
            f"expected_hit_rate={expected_hit_rate:.3f} "
            f"realised_hit_rate={realised_hit_rate:.3f} "
            f"hit_rate_gap={realised_hit_rate - expected_hit_rate:.3f} "
            f"brier={brier:.4f}"
        )

    top_avg = bucket_stats[">=0.95"]["avg_pnl"]
    if np.isfinite(top_avg) and np.isfinite(overall_avg_pnl):
        diff = top_avg - overall_avg_pnl
        if diff >= 0.0:
            verdict = f"top-confidence trades outperformed by {_format_pct(abs(diff))} per trade"
        else:
            verdict = f"top-confidence trades underperformed by {_format_pct(abs(diff))} per trade"
    else:
        verdict = "insufficient closed trades for confidence/outcome recap"
    lines.append(f"Verdict: {verdict}")
    return "\n".join(lines)


def _model_drift_summary(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "Model drift summary: no trades in reporting window."

    pred_col = (
        "calibrated_score" if "calibrated_score" in trades.columns else "meta_pred"
    )
    pred = pd.to_numeric(trades.get(pred_col, pd.Series(dtype=float)), errors="coerce")
    pnl = pd.to_numeric(
        trades.get("net_pnl_pct", pd.Series(dtype=float)), errors="coerce"
    )
    if pnl.isna().all():
        pnl = pd.to_numeric(
            trades.get("gross_pnl_pct", pd.Series(dtype=float)), errors="coerce"
        )
    if pnl.isna().all():
        pnl = pd.to_numeric(
            trades.get("net_pnl", pd.Series(dtype=float)), errors="coerce"
        )

    closed_mask = pnl.notna()
    closed_n = int(closed_mask.sum())
    executed_mask = (
        trades.get("status", pd.Series(index=trades.index, dtype=str))
        .astype(str)
        .str.lower()
        .isin({"pending", "completed", "closed", "recorded"})
    )
    error_mask = (
        trades.get("status", pd.Series(index=trades.index, dtype=str))
        .astype(str)
        .str.lower()
        .isin({"failed", "rejected", "error"})
    )
    order_error_categories = (
        trades.get("order_error_category", pd.Series(dtype=str)).fillna("").astype(str)
    )
    unexplained_errors = int(
        (
            (order_error_categories == "")
            & error_mask.reindex(trades.index, fill_value=False)
        ).sum()
    )

    lines = [
        "Model drift and execution diagnostics:",
        f"trades_logged={len(trades)} executed_or_pending={int(executed_mask.sum())} "
        f"order_errors={int(error_mask.sum())} unexplained_order_errors={unexplained_errors}",
    ]

    finite_pred = pred.replace([np.inf, -np.inf], np.nan).dropna()
    if not finite_pred.empty:
        lines.append(
            f"{pred_col}: mean={float(finite_pred.mean()):.4f} "
            f"std={float(finite_pred.std(ddof=0)):.4f} "
            f"range=[{float(finite_pred.min()):.4f},{float(finite_pred.max()):.4f}] "
            f"top10_count={int((finite_pred.rank(pct=True) >= 0.90).sum())}"
        )

    if closed_n <= 0:
        lines.append(
            "realised_hit_rate=unavailable; no closed trades with realised PnL yet."
        )
        return "\n".join(lines)

    outcome = (pnl.loc[closed_mask] > 0).astype(float)
    pred_closed = pred.loc[closed_mask].clip(0.0, 1.0)
    valid = pred_closed.notna()
    realised_hit_rate = float(outcome.mean())
    if valid.any():
        expected_hit_rate = float(pred_closed.loc[valid].mean())
        calibration_error = abs(realised_hit_rate - expected_hit_rate)
        brier = float(np.mean((pred_closed.loc[valid] - outcome.loc[valid]) ** 2))
        lines.append(
            f"closed_trades={closed_n} expected_hit_rate={expected_hit_rate:.3f} "
            f"realised_hit_rate={realised_hit_rate:.3f} "
            f"abs_calibration_error={calibration_error:.3f} brier={brier:.4f}"
        )
    else:
        lines.append(
            f"closed_trades={closed_n} realised_hit_rate={realised_hit_rate:.3f}; "
            "expected hit-rate unavailable because calibrated scores are missing."
        )

    pnl_closed = pnl.loc[closed_mask].replace([np.inf, -np.inf], np.nan).dropna()
    if not pnl_closed.empty:
        lines.append(
            f"realised_pnl_pct: mean={float(pnl_closed.mean()):.4%} "
            f"sum={float(pnl_closed.sum()):.4%} "
            f"range=[{float(pnl_closed.min()):.4%},{float(pnl_closed.max()):.4%}]"
        )
    mae = pd.to_numeric(trades.get("mae", pd.Series(dtype=float)), errors="coerce")
    mae_closed = mae.loc[closed_mask].replace([np.inf, -np.inf], np.nan).dropna()
    if not mae_closed.empty:
        lines.append(
            f"mae: mean={float(mae_closed.mean()):.4%} "
            f"p90={float(mae_closed.quantile(0.90)):.4%} "
            f"max={float(mae_closed.max()):.4%}"
        )
    return "\n".join(lines)


def transfer_profit_to_spot(
    exchange: Any,
    *,
    amount: float,
    asset: str = "USDT",
    transfer_type: str = "MARGIN_MAIN",
) -> Dict[str, Any]:
    """Transfer saved profit to spot through ccxt's Binance asset-transfer API."""
    amount_f = _coerce_float(amount, default=0.0)
    if amount_f <= 0.0:
        return {"success": True, "skipped": True, "reason": "zero_amount"}
    transfer = getattr(exchange, "sapiPostAssetTransfer", None)
    if not callable(transfer):
        transfer = getattr(exchange, "sapi_post_asset_transfer", None)
    if not callable(transfer):
        return {
            "success": False,
            "skipped": True,
            "error_category": "transfer_method_unavailable",
            "error": "exchange does not expose sapiPostAssetTransfer",
        }
    payload = {
        "type": str(transfer_type),
        "asset": str(asset),
        "amount": f"{amount_f:.8f}",
    }
    try:
        response = transfer(payload)
        return {
            "success": True,
            "skipped": False,
            "request": payload,
            "response": response,
        }
    except Exception as exc:
        return {
            "success": False,
            "skipped": False,
            "request": payload,
            "error_category": classify_api_error(exc),
            "error": str(exc),
        }


@dataclass
class DailyDeploymentReporter:
    """Run the daily balance checkpoint, profit skim, and Gmail report."""

    state_path: str = DEFAULT_STATE_PATH
    smtp_factory: Callable[..., Any] = smtplib.SMTP
    env_file: str = ".env"

    def _state(self) -> Dict[str, Any]:
        return _read_json(Path(self.state_path))

    def _save_state(self, state: Dict[str, Any]) -> None:
        _write_json_atomic(Path(self.state_path), state)

    def _send_email(
        self,
        *,
        subject: str,
        body: str,
        recipient: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        _load_dotenv_if_present(self.env_file)
        gmail_user = os.environ.get("GMAIL_USER", "").strip()
        gmail_password = os.environ.get("GMAIL_APP_PASSWORD", "").strip()
        smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com").strip()
        smtp_port = int(os.environ.get("SMTP_PORT", "587") or 587)
        if not gmail_user or not gmail_password:
            return {
                "success": False,
                "error_category": "missing_smtp_credentials",
                "error": "GMAIL_USER or GMAIL_APP_PASSWORD is missing",
            }

        message = EmailMessage()
        message["From"] = gmail_user
        message["To"] = recipient
        message["Subject"] = subject
        message.set_content(body)

        timeout = float(config.get("daily_report_smtp_timeout_seconds", 30.0))
        try:
            with self.smtp_factory(smtp_host, smtp_port, timeout=timeout) as smtp:
                smtp.starttls()
                smtp.login(gmail_user, gmail_password)
                smtp.send_message(message)
            return {"success": True, "recipient": recipient}
        except Exception as exc:
            return {
                "success": False,
                "error_category": classify_api_error(exc),
                "error": str(exc),
            }

    def _build_body(
        self,
        *,
        now: pd.Timestamp,
        total_balance: float,
        available_balance: float,
        previous_best_balance: float,
        amount_to_save: float,
        transfer_result: Dict[str, Any],
        trades: pd.DataFrame,
    ) -> str:
        return "\n".join(
            [
                "Extreme price movement deployment daily report",
                "",
                "Account",
                f"  datetime: {now.isoformat()}",
                f"  total_balance_usdt: {total_balance:.8f}",
                f"  available_balance_usdt_for_profit_skim: {available_balance:.8f}",
                "  previous_best_available_balance_usdt: "
                f"{previous_best_balance:.8f}",
                f"  amount_saved_to_spot_usdt: {amount_to_save:.8f}",
                "  transfer_result: "
                f"{json.dumps(transfer_result, default=str, sort_keys=True)}",
                "",
                "Model Drift And Execution",
                _model_drift_summary(trades),
                "",
                "Confidence Calibration Recap",
                _confidence_calibration_recap(trades),
                "",
                "Net Strategy Recap",
                _strategy_trade_recap(trades, total_balance=total_balance),
                "",
                "Trades Since Previous Message",
                _format_trade_report(trades),
            ]
        )

    def maybe_run(
        self,
        *,
        exchange: Any,
        portfolio_mgr: PortfolioManager,
        trade_logger: TradeLogger,
        config: Optional[Dict[str, Any]] = None,
        now: Optional[pd.Timestamp] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Run the daily report if due; returns an execution summary."""
        cfg = dict(config or {})
        interval_hours = float(cfg.get("daily_report_interval_hours", 24.0))
        now_ts = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
        if now_ts.tzinfo is None:
            now_ts = now_ts.tz_localize("UTC")
        else:
            now_ts = now_ts.tz_convert("UTC")

        state = self._state()
        last_report = state.get("last_report_ts")
        if last_report and not force:
            last_ts = pd.Timestamp(last_report)
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("UTC")
            else:
                last_ts = last_ts.tz_convert("UTC")
            elapsed_hours = (now_ts - last_ts).total_seconds() / 3600.0
            if elapsed_hours < interval_hours:
                return {
                    "sent": False,
                    "reason": "not_due",
                    "elapsed_hours": elapsed_hours,
                }

        snapshot = portfolio_mgr.fetch_exchange_snapshot(exchange)
        total_balance = _coerce_float(snapshot.get("total_balance"))
        if not np.isfinite(total_balance):
            tprint("[DailyReporter] Skipping daily report: total balance unavailable")
            return {
                "sent": False,
                "reason": "balance_unavailable",
                "snapshot_errors": snapshot.get("errors", []),
            }

        available_balance = _coerce_float(
            snapshot.get("free_balance"),
            default=total_balance,
        )
        previous_best = _coerce_float(
            state.get("previous_best_available_balance_usdt"),
            default=_coerce_float(
                state.get("previous_best_balance_usdt"),
                default=available_balance,
            ),
        )
        amount_to_save = max(0.0, (available_balance - previous_best) / 20.0)
        transfer_enabled = bool(
            cfg.get("daily_report_transfer_enabled", cfg.get("mode") == "live")
        )
        transfer_result: Dict[str, Any]
        if transfer_enabled:
            transfer_result = transfer_profit_to_spot(
                exchange,
                amount=amount_to_save,
                transfer_type=str(cfg.get("daily_report_transfer_type", "MARGIN_MAIN")),
            )
        else:
            transfer_result = {
                "success": True,
                "skipped": True,
                "reason": "transfer_disabled",
            }

        if (
            transfer_enabled
            and amount_to_save > 0.0
            and bool(transfer_result.get("success"))
        ):
            # Persist the new high-water mark immediately after a successful
            # transfer so an SMTP failure cannot repeat the same transfer.
            state = dict(state)
            state["previous_best_available_balance_usdt"] = max(
                previous_best,
                available_balance,
            )
            state["previous_best_balance_usdt"] = max(
                _coerce_float(
                    state.get("previous_best_balance_usdt"),
                    default=total_balance,
                ),
                total_balance,
            )
            state["last_total_balance_usdt"] = total_balance
            state["last_available_balance_usdt"] = available_balance
            state["last_amount_saved_to_spot_usdt"] = amount_to_save
            state["last_transfer_result"] = transfer_result
            state["last_transfer_ts"] = now_ts.isoformat()
            self._save_state(state)

        trades = _trades_since(trade_logger, state.get("last_trade_report_ts"))
        recipient = str(
            cfg.get("daily_report_email_to")
            or os.environ.get("EPM_REPORT_EMAIL_TO")
            or DEFAULT_REPORT_TO
        )
        subject = cfg.get("daily_report_subject", "EPM daily deployment report")
        body = self._build_body(
            now=now_ts,
            total_balance=total_balance,
            available_balance=available_balance,
            previous_best_balance=previous_best,
            amount_to_save=amount_to_save,
            transfer_result=transfer_result,
            trades=trades,
        )
        email_result = self._send_email(
            subject=str(subject), body=body, recipient=recipient, config=cfg
        )
        if not email_result.get("success"):
            tprint(
                "[DailyReporter] Daily report email failed: "
                f"{email_result.get('error_category')}: {email_result.get('error')}"
            )
            return {
                "sent": False,
                "reason": "email_failed",
                "email_result": email_result,
                "transfer_result": transfer_result,
                "amount_to_save": amount_to_save,
            }

        new_state = dict(state)
        new_state["previous_best_available_balance_usdt"] = max(
            previous_best,
            available_balance,
        )
        new_state["previous_best_balance_usdt"] = max(
            _coerce_float(state.get("previous_best_balance_usdt"), default=total_balance),
            total_balance,
        )
        new_state["last_report_ts"] = now_ts.isoformat()
        new_state["last_trade_report_ts"] = now_ts.isoformat()
        new_state["last_total_balance_usdt"] = total_balance
        new_state["last_available_balance_usdt"] = available_balance
        new_state["last_amount_saved_to_spot_usdt"] = amount_to_save
        new_state["last_transfer_result"] = transfer_result
        self._save_state(new_state)
        tprint(
            "[DailyReporter] Daily report sent: "
            f"total={total_balance:.2f} previous_best={previous_best:.2f} "
            f"available={available_balance:.2f} "
            f"saved={amount_to_save:.2f} trades={len(trades)}"
        )
        return {
            "sent": True,
            "email_result": email_result,
            "transfer_result": transfer_result,
            "amount_to_save": amount_to_save,
            "trade_count": int(len(trades)),
            "total_balance": total_balance,
            "available_balance": available_balance,
            "previous_best_balance": previous_best,
        }
