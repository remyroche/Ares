"""Daily deployment reporting and profit skim utilities."""

import html
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
    "policy_archetype",
    "local_side_archetype",
    "market_mode",
    "meta_pred",
    "calibrated_score",
    "policy_rank_pct",
    "rank_percentile",
    "effective_threshold",
    "deployment_rank_threshold",
    "rank_threshold",
    "archetype_hit_surprise_threshold",
    "archetype_hit_surprise_threshold_delta",
    "archetype_hit_surprise_applied",
    "archetype_hit_surprise_reason",
    "archetype_hit_surprise_matched_key",
    "archetype_hit_surprise_actual_hit_rate",
    "archetype_hit_surprise_expected_hit_rate",
    "archetype_hit_surprise_hit_rate_delta",
    "archetype_hit_surprise_hit_rate_surprise_z",
    "archetype_hit_surprise_support_confidence",
    "strategy_ev_hit_rate",
    "strategy_ev_avg_net_return",
    "strategy_ev_gate_allowed",
    "strategy_ev_gate_reason",
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
    "perp_effective_leverage",
    "perp_rank_leverage",
    "perp_risk_cap_leverage",
    "perp_rank_number",
    "perp_rank_x",
    "perp_stop_loss_pct",
    "perp_full_wallet",
    "perp_available_wallet",
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


def _html_escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _html_metric(label: str, value: str, *, accent: str = "neutral") -> str:
    colors = {
        "good": "#087f5b",
        "bad": "#c92a2a",
        "warn": "#b7791f",
        "neutral": "#24435c",
    }
    color = colors.get(accent, colors["neutral"])
    return (
        '<td class="metric">'
        f'<div class="metric-label">{_html_escape(label)}</div>'
        f'<div class="metric-value" style="color:{color}">{_html_escape(value)}</div>'
        "</td>"
    )


def _html_pre(text: str) -> str:
    return f'<pre class="preblock">{_html_escape(text)}</pre>'


def _html_section(title: str, body: str) -> str:
    return (
        '<section class="section">'
        f"<h2>{_html_escape(title)}</h2>"
        f"{body}"
        "</section>"
    )


def _html_trade_table(trades: pd.DataFrame, *, max_rows: int = 40) -> str:
    if trades.empty:
        return '<p class="muted">No trades logged since the previous daily message.</p>'
    preferred = [
        "timestamp",
        "symbol",
        "side",
        "strategy_id",
        "policy_archetype",
        "status",
        "calibrated_score",
        "policy_rank_pct",
        "rank_percentile",
        "effective_threshold",
        "archetype_hit_surprise_threshold",
        "archetype_hit_surprise_threshold_delta",
        "archetype_hit_surprise_actual_hit_rate",
        "strategy_ev_hit_rate",
        "entry_notional_quote",
        "net_pnl_amount",
        "net_pnl_pct",
        "exit_reason",
    ]
    cols = [col for col in preferred if col in trades.columns]
    if not cols:
        cols = [col for col in REPORT_TRADE_COLUMNS if col in trades.columns][:10]
    if not cols:
        return '<p class="muted">Trade rows were present, but no display columns were available.</p>'
    view = trades.loc[:, cols].tail(max_rows).copy()
    headers = "".join(f"<th>{_html_escape(col)}</th>" for col in view.columns)
    body_rows = []
    for _, row in view.iterrows():
        cells = []
        for col in view.columns:
            value = row.get(col)
            if isinstance(value, float):
                value = f"{value:.6g}" if np.isfinite(value) else ""
            cells.append(f"<td>{_html_escape(str(value)[:96])}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-wrap"><table>'
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


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


def _first_existing_column(frame: pd.DataFrame, candidates: List[str]) -> str:
    for col in candidates:
        if col in frame.columns:
            return col
    return ""


def _numeric_column(frame: pd.DataFrame, col: str) -> pd.Series:
    if not col:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame.get(col, pd.Series(index=frame.index, dtype=float)), errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def _mode_text(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    values = values.loc[~values.str.lower().isin({"", "nan", "none", "na", "n/a"})]
    if values.empty:
        return "n/a"
    return str(values.value_counts().idxmax())[:64]


def _archetype_policy_recap(trades: pd.DataFrame) -> str:
    """Summarise live thresholds and hit-rate-surprise context by policy archetype."""
    if trades.empty:
        return "No trade events logged in this reporting window."

    arch_col = _first_existing_column(
        trades,
        ["policy_archetype", "local_side_archetype", "archetype_policy_key", "source_tag"],
    )
    if not arch_col:
        return "No archetype column is available in the trade log yet."

    work = trades.copy()
    work["_policy_archetype"] = (
        work[arch_col].fillna("missing").astype(str).replace("", "missing")
    )
    work["_closed"] = _trade_closed_mask(work)

    rank_col = _first_existing_column(work, ["policy_rank_pct", "rank_percentile", "calibrated_score"])
    threshold_col = _first_existing_column(
        work,
        [
            "effective_threshold",
            "deployment_rank_threshold",
            "rank_threshold",
            "archetype_hit_surprise_threshold",
        ],
    )
    hit_threshold_col = _first_existing_column(
        work, ["archetype_hit_surprise_threshold", threshold_col]
    )
    net_col = _first_existing_column(work, ["net_pnl_pct", "gross_pnl_pct"])
    net_amount_col = _first_existing_column(work, ["net_pnl_amount", "net_pnl"])

    work["_rank"] = _numeric_column(work, rank_col)
    work["_threshold"] = _numeric_column(work, threshold_col)
    work["_hit_threshold"] = _numeric_column(work, hit_threshold_col)
    work["_threshold_delta"] = _numeric_column(
        work, "archetype_hit_surprise_threshold_delta"
    )
    work["_quality_adjustment"] = _numeric_column(
        work, "archetype_hit_surprise_quality_adjustment"
    )
    work["_priority_multiplier"] = _numeric_column(
        work, "archetype_hit_surprise_priority_multiplier"
    )
    work["_rank_adjustment"] = _numeric_column(
        work, "archetype_hit_surprise_rank_adjustment"
    )
    work["_actual_hr"] = _numeric_column(
        work, "archetype_hit_surprise_actual_hit_rate"
    )
    work["_expected_hr"] = _numeric_column(
        work, "archetype_hit_surprise_expected_hit_rate"
    )
    work["_hr_delta"] = _numeric_column(work, "archetype_hit_surprise_hit_rate_delta")
    work["_hr_z"] = _numeric_column(
        work, "archetype_hit_surprise_hit_rate_surprise_z"
    )
    work["_support_conf"] = _numeric_column(
        work, "archetype_hit_surprise_support_confidence"
    )
    work["_ev_hr"] = _numeric_column(work, "strategy_ev_hit_rate")
    work["_ev_net"] = _numeric_column(work, "strategy_ev_avg_net_return")
    work["_net_pct"] = _numeric_column(work, net_col)
    work["_net_amount"] = _numeric_column(work, net_amount_col)

    lines = [
        f"archetype_source={arch_col}",
        (
            "columns: archetype | events | closed | hit_rate | avg_net_pct | "
            "avg_rank | avg_threshold | archetype_hr_threshold | threshold_delta | "
            "quality_adj | priority_mult | rank_adj | "
            "recent_hr | expected_hr | hr_delta | surprise_z | support | "
            "strategy_ev_hr | strategy_ev_net | applied | reason"
        ),
    ]
    for archetype, group in work.groupby("_policy_archetype", dropna=False, sort=True):
        closed = group.loc[group["_closed"]]
        closed_n = int(len(closed))
        hit_rate = (
            float((closed["_net_amount"].fillna(closed["_net_pct"]) > 0.0).mean())
            if closed_n
            else np.nan
        )
        avg_net_pct = float(closed["_net_pct"].mean()) if closed_n else np.nan
        applied = int(
            group.get(
                "archetype_hit_surprise_applied",
                pd.Series(False, index=group.index),
            )
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin({"1", "true", "yes", "y"})
            .sum()
        )
        lines.append(
            " | ".join(
                [
                    str(archetype)[:72],
                    str(int(len(group))),
                    str(closed_n),
                    _format_pct(hit_rate),
                    _format_pct(avg_net_pct),
                    _format_pct(float(group["_rank"].mean())),
                    _format_pct(float(group["_threshold"].mean())),
                    _format_pct(float(group["_hit_threshold"].mean())),
                    _format_pct(float(group["_threshold_delta"].mean())),
                    _format_pct(float(group["_quality_adjustment"].mean())),
                    _format_money(float(group["_priority_multiplier"].mean())),
                    _format_pct(float(group["_rank_adjustment"].mean())),
                    _format_pct(float(group["_actual_hr"].mean())),
                    _format_pct(float(group["_expected_hr"].mean())),
                    _format_pct(float(group["_hr_delta"].mean())),
                    _format_money(float(group["_hr_z"].mean())),
                    _format_money(float(group["_support_conf"].mean())),
                    _format_pct(float(group["_ev_hr"].mean())),
                    _format_pct(float(group["_ev_net"].mean())),
                    str(applied),
                    _mode_text(group.get("archetype_hit_surprise_reason", pd.Series(index=group.index, dtype=object))),
                ]
            )
        )
    return "\n".join(lines)


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


def _dynamic_strategy_performance_recap(config: Dict[str, Any]) -> str:
    root = Path(
        str(
            config.get("live_data_root")
            or config.get("data_root")
            or "data"
        )
    )
    path = root / "live_state" / "dynamic_strategy_performance.json"
    if not path.exists():
        return "  unavailable: dynamic performance report has not been written yet"
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return f"  unavailable: could not read {path}: {exc}"
    strategies = payload.get("strategies", {})
    if not isinstance(strategies, dict) or not strategies:
        reason = payload.get("reason", "no_strategy_rows")
        return f"  unavailable: {reason}"
    rows = list(strategies.values())
    rows.sort(
        key=lambda row: (
            _coerce_float(row.get("inference_drift_score_21d"), default=-1.0),
            _coerce_float(row.get("inference_drift_score"), default=-1.0),
        ),
        reverse=True,
    )
    lines = [
        f"  updated_at: {payload.get('updated_at', '')}",
        "  top30 strategy x meta-head diagnostics:",
    ]
    for row in rows[:30]:
        sid = row.get("strategy_id", "")
        mh = str(row.get("meta_head_hash", ""))[:8]
        mult = _coerce_float(row.get("threshold_multiplier"), default=1.0)
        recent_hit = _coerce_float(row.get("recent_weighted_hit_rate_21d"))
        expected_hit = _coerce_float(row.get("expected_hit_rate_oos_top40"))
        drift7 = _coerce_float(row.get("inference_drift_score_7d"))
        drift21 = _coerce_float(row.get("inference_drift_score_21d"))
        uncertainty7 = _coerce_float(row.get("uncertainty_score_ratio_7d"))
        uncertainty21 = _coerce_float(row.get("uncertainty_score_ratio_21d"))
        perf7 = _coerce_float(row.get("dynamic_performance_hit_ratio_7d"))
        perf21 = _coerce_float(row.get("dynamic_performance_hit_ratio_21d"))
        reason = row.get("reason", "")
        lines.append(
            "  "
            f"{sid}#{mh}: mult={mult:.3f} "
            f"hit21={recent_hit:.3f} expected={expected_hit:.3f} "
            f"drift7={drift7:.3f} drift21={drift21:.3f} "
            f"uncert_ratio7={uncertainty7:.3f} "
            f"uncert_ratio21={uncertainty21:.3f} "
            f"perf_ratio7={perf7:.3f} perf_ratio21={perf21:.3f} "
            f"reason={reason}"
        )
    if payload.get("history_backfill_required"):
        lines.append(
            "  history_backfill_required: true "
            "(multiplier remains neutral until enough resolved recent outcomes exist)"
        )
    parity = payload.get("parity_loading_checker", {})
    if parity:
        lines.append(
            "  parity_loading_checker: "
            f"status={parity.get('status')} sample_rate={parity.get('sample_rate')}"
        )
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


def _live_drift_recap_summary(config: Dict[str, Any]) -> str:
    """Human-readable summary of the latest live drift recap for email reports."""
    if not bool(config.get("daily_report_include_live_drift_recap", True)):
        return "Live drift recap: disabled by daily_report_include_live_drift_recap."

    explicit_path = str(config.get("daily_report_live_drift_recap_path") or "").strip()
    if explicit_path:
        recap_path = Path(explicit_path)
    else:
        live_root = Path(
            config.get("live_data_root")
            or config.get("data_root")
            or "data"
        )
        recap_path = (
            live_root
            / "live_state"
            / "drift_monitoring"
            / "latest"
            / "drift_recap.json"
        )

    recap = _read_json(recap_path)
    if not recap:
        return f"Live drift recap: unavailable at {recap_path}"

    reason = str(recap.get("reason") or "").strip()
    if reason:
        return f"Live drift recap: skipped reason={reason} path={recap_path}"

    lines = [
        "Live drift recap:",
        f"  asof_ts: {recap.get('asof_ts', '')}",
        f"  label_maturity_cutoff_ts: {recap.get('label_maturity_cutoff_ts', '')}",
        f"  ledger_rows: {recap.get('ledger_rows', 0)} "
        f"scored_metric_rows: {recap.get('scored_metric_rows', 0)} "
        f"regime_feature_rows: {recap.get('regime_feature_rows', 0)}",
        f"  source: {recap_path}",
    ]

    family_scores = recap.get("family_scores") or recap.get("all_family_scores") or {}
    if not isinstance(family_scores, dict) or not family_scores:
        lines.append("  family_scores: unavailable")
        return "\n".join(lines)

    def _window_sort_key(item: tuple[str, Any]) -> float:
        label = str(item[0])
        digits = "".join(ch for ch in label if ch.isdigit() or ch == ".")
        return _coerce_float(digits, default=9999.0)

    for window, families in sorted(family_scores.items(), key=_window_sort_key):
        if not isinstance(families, dict) or not families:
            continue
        lines.append(f"  {window}:")
        ordered = sorted(
            families.items(),
            key=lambda item: _coerce_float(
                (item[1] or {}).get("family_score")
                if isinstance(item[1], dict)
                else np.nan,
                default=-1.0,
            ),
            reverse=True,
        )
        for family, values in ordered[:8]:
            if not isinstance(values, dict):
                continue
            score = _coerce_float(values.get("family_score"))
            coverage = _coerce_float(values.get("family_metric_coverage_ratio"))
            reliable = _coerce_float(values.get("family_reliable_baseline_ratio"))
            matured = _coerce_float(values.get("family_matured_label_coverage_ratio"))
            lines.append(
                f"    {family}: score={score:.3f} "
                f"coverage={coverage:.2f} reliable={reliable:.2f} "
                f"matured={matured:.2f}"
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
        html_body: Optional[str] = None,
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
        if html_body:
            message.add_alternative(html_body, subtype="html")

        timeout = float(config.get("daily_report_smtp_timeout_seconds", 30.0))
        try:
            with self.smtp_factory(smtp_host, smtp_port, timeout=timeout) as smtp:
                smtp.starttls()
                smtp.login(gmail_user, gmail_password)
                refused = smtp.send_message(message) or {}
            if refused:
                refused_recipients = sorted(str(address) for address in refused)
                return {
                    "success": False,
                    "error_category": "smtp_recipient_refused",
                    "error": "SMTP server refused one or more recipients",
                    "recipient": recipient,
                    "refused_recipients": refused_recipients,
                }
            return {
                "success": True,
                "recipient": recipient,
                "accepted_recipients": [recipient],
            }
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
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        cfg = dict(config or {})
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
                "Live Drift Recap",
                _live_drift_recap_summary(cfg),
                "",
                "Dynamic Strategy Performance",
                _dynamic_strategy_performance_recap(cfg),
                "",
                "Archetype Threshold Recap",
                _archetype_policy_recap(trades),
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

    def _build_html_body(
        self,
        *,
        now: pd.Timestamp,
        total_balance: float,
        available_balance: float,
        previous_best_balance: float,
        amount_to_save: float,
        transfer_result: Dict[str, Any],
        trades: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        cfg = dict(config or {})
        delta_balance = available_balance - previous_best_balance
        transfer_summary = json.dumps(transfer_result, default=str, sort_keys=True)
        account_metrics = (
            '<table class="metrics"><tr>'
            + _html_metric("Total Balance", f"{total_balance:,.4f} USDT")
            + _html_metric("Available", f"{available_balance:,.4f} USDT")
            + _html_metric(
                "Vs Previous Best",
                f"{delta_balance:,.4f} USDT",
                accent="good" if delta_balance >= 0 else "bad",
            )
            + _html_metric(
                "Saved To Spot",
                f"{amount_to_save:,.4f} USDT",
                accent="good" if amount_to_save > 0 else "neutral",
            )
            + "</tr></table>"
            f'<p class="kv"><span>Datetime</span>{_html_escape(now.isoformat())}</p>'
            f'<p class="kv"><span>Transfer</span>{_html_escape(transfer_summary[:500])}</p>'
        )
        sections = [
            _html_section("Account", account_metrics),
            _html_section("Model Drift And Execution", _html_pre(_model_drift_summary(trades))),
            _html_section("Live Drift Recap", _html_pre(_live_drift_recap_summary(cfg))),
            _html_section(
                "Dynamic Strategy Performance",
                _html_pre(_dynamic_strategy_performance_recap(cfg)),
            ),
            _html_section(
                "Archetype Threshold Recap",
                _html_pre(_archetype_policy_recap(trades)),
            ),
            _html_section(
                "Confidence Calibration",
                _html_pre(_confidence_calibration_recap(trades)),
            ),
            _html_section(
                "Net Strategy Recap",
                _html_pre(_strategy_trade_recap(trades, total_balance=total_balance)),
            ),
            _html_section("Trades Since Previous Message", _html_trade_table(trades)),
        ]
        css = """
        body { margin:0; padding:0; background:#f4f6f8; color:#152536; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; }
        .page { max-width:980px; margin:0 auto; padding:24px 18px 36px; }
        .header { padding:6px 0 18px; border-bottom:3px solid #24435c; margin-bottom:18px; }
        h1 { margin:0; font-size:24px; line-height:1.25; color:#102a43; letter-spacing:0; }
        .subtitle { margin:7px 0 0; color:#52677a; font-size:13px; }
        .section { background:#ffffff; border:1px solid #d8e0e8; border-radius:8px; padding:16px; margin:14px 0; }
        h2 { margin:0 0 12px; font-size:16px; line-height:1.3; color:#102a43; letter-spacing:0; }
        .metrics { width:100%; border-collapse:separate; border-spacing:8px; margin:0 0 8px; table-layout:fixed; }
        .metric { background:#f8fafc; border:1px solid #dde5ed; border-radius:8px; padding:12px; vertical-align:top; }
        .metric-label { color:#627286; font-size:12px; text-transform:uppercase; letter-spacing:0; margin-bottom:6px; }
        .metric-value { font-size:20px; line-height:1.2; font-weight:700; word-break:break-word; }
        .kv { margin:8px 0 0; color:#20364a; font-size:13px; line-height:1.45; }
        .kv span { display:inline-block; min-width:92px; color:#627286; font-weight:600; }
        .preblock { margin:0; padding:12px; overflow-x:auto; white-space:pre-wrap; word-break:break-word; background:#f8fafc; border:1px solid #dde5ed; border-radius:6px; font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; color:#1f2d3d; }
        .table-wrap { overflow-x:auto; border:1px solid #dde5ed; border-radius:6px; }
        table { width:100%; border-collapse:collapse; font-size:12px; }
        th { background:#edf2f7; color:#334e68; text-align:left; padding:8px; border-bottom:1px solid #d8e0e8; white-space:nowrap; }
        td { padding:8px; border-bottom:1px solid #edf2f7; vertical-align:top; color:#1f2d3d; }
        tr:last-child td { border-bottom:0; }
        .muted { color:#627286; margin:0; }
        @media (max-width:700px) {
          .page { padding:14px 10px 24px; }
          .section { padding:12px; }
          .metrics, .metrics tbody, .metrics tr, .metric { display:block; width:auto; }
          .metric { margin:0 0 8px; }
          .metric-value { font-size:18px; }
        }
        """
        return (
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<style>{css}</style></head><body><div class=\"page\">"
            "<div class=\"header\"><h1>Extreme Price Movement Deployment Report</h1>"
            f"<p class=\"subtitle\">Daily deployment recap for {_html_escape(now.isoformat())}</p></div>"
            + "".join(sections)
            + "</div></body></html>"
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

        market_mode = str(cfg.get("market_mode") or cfg.get("mode") or "").lower()
        execution_account = str(cfg.get("execution_account") or "").lower()
        is_perps = (
            market_mode == "perps"
            or execution_account in {"perp", "perps", "future", "futures", "swap"}
        )
        snapshot = portfolio_mgr.fetch_exchange_snapshot(
            exchange,
            quote_currency=str(
                cfg.get("live_quote_currency")
                or cfg.get("quote_currency")
                or ("USD" if is_perps else "USDC")
            ).upper(),
            execution_account=execution_account or ("perps" if is_perps else "margin"),
            margin_mode=str(cfg.get("margin_mode") or "cross").lower(),
        )
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
            config=cfg,
        )
        html_body = self._build_html_body(
            now=now_ts,
            total_balance=total_balance,
            available_balance=available_balance,
            previous_best_balance=previous_best,
            amount_to_save=amount_to_save,
            transfer_result=transfer_result,
            trades=trades,
            config=cfg,
        )
        email_result = self._send_email(
            subject=str(subject),
            body=body,
            html_body=html_body,
            recipient=recipient,
            config=cfg,
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
