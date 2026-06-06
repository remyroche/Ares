"""Replay live closed-trade exits with the deployed simple-policy stop logic.

This is an audit tool. It does not change live trading behavior. Given a live
closed-trade extract, it replays the same stop/trailing policy over cached live
minute bars when available, and over the cached 5m/price observations embedded
in live monitor logs otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from extreme_price_movements.inference.simple_policy_stop import (
    compute_initial_simple_policy_stop_decision,
    compute_simple_policy_stop_decision,
    load_simple_policy_stop_params_by_strategy,
)
from extreme_price_movements.inference.execution_fill_model import stop_exit_fill_price


DEFAULT_CLOSED_TRADES = (
    Path("extreme_price_movements")
    / "reports"
    / "inference_mismatch_investigation"
    / "latest_closed_live_trades_20260605.csv"
)
DEFAULT_OUT_DIR = (
    Path("extreme_price_movements")
    / "reports"
    / "inference_mismatch_investigation"
    / "live_closed_trade_exit_replay_20260605"
)

_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_SL_MULT_RE = re.compile(rf"\bsl_mult=({_FLOAT_RE})")
_BARRIER_RE = re.compile(rf"\bbarrier_frac=({_FLOAT_RE})")
_TRIGGER_RE = re.compile(rf"\btrigger=({_FLOAT_RE})")
_PRICE_BAR_5M_RE = re.compile(
    rf"(?P<ts>\d{{4}}-\d{{2}}-\d{{2}}T[^\s]+)\s+price_bar_5m\s+"
    rf"open=(?P<open>{_FLOAT_RE})\s+high=(?P<high>{_FLOAT_RE})\s+"
    rf"low=(?P<low>{_FLOAT_RE})\s+close=(?P<close>{_FLOAT_RE})"
)
_CLOSEABLE_SAMPLE_RE = re.compile(
    rf"(?P<ts>\d{{4}}-\d{{2}}-\d{{2}}T[^\s]+)\s+live_closeable_price_sample\s+"
    rf".*?\bprice=(?P<price>{_FLOAT_RE})\b"
)
_STOP_FILLED_RE = re.compile(
    rf"(?P<ts>\d{{4}}-\d{{2}}-\d{{2}}T[^\s]+)\s+stop_order_filled\s+"
    rf".*?\bfill_price=(?P<fill_price>{_FLOAT_RE})\b"
    rf".*?\bstop_reason=(?P<stop_reason>[^\s]+)"
)

POLICY_STOP_EXIT_BASE_GAP_BPS = float(
    os.environ.get("EPM_SIMPLE_POLICY_STOP_EXIT_BASE_GAP_BPS", "15.0")
)
POLICY_STOP_EXIT_ALPHA_THROUGH = float(
    os.environ.get("EPM_SIMPLE_POLICY_STOP_EXIT_ALPHA_THROUGH", "0.05")
)
POLICY_STOP_EXIT_MAX_GAP_BPS = float(
    os.environ.get("EPM_SIMPLE_POLICY_STOP_EXIT_MAX_GAP_BPS", "75.0")
)


@dataclass(frozen=True)
class ParsedRecap:
    bars: pd.DataFrame
    stop_fill_ts: Optional[pd.Timestamp]
    stop_fill_price: float
    stop_reason: str
    source: str


def _to_ts(value: Any) -> Optional[pd.Timestamp]:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _parse_float_from_detail(pattern: re.Pattern[str], detail: Any) -> float:
    match = pattern.search(str(detail or ""))
    if not match:
        return np.nan
    return _safe_float(match.group(1))


def _safe_symbol_path(symbol: str) -> str:
    return str(symbol).replace("/", "_")


def _candidate_execution_1m_dirs(data_root: Path, symbol: str) -> List[Path]:
    safe = _safe_symbol_path(symbol)
    exchange_root = data_root / "exchanges" / "krakenfutures"
    return [
        exchange_root / "execution_1m" / "ohlcv" / f"symbol={safe}",
        exchange_root
        / "exchanges"
        / "krakenfutures"
        / "execution_1m"
        / "ohlcv"
        / f"symbol={safe}",
        data_root / "execution_1m" / "ohlcv" / f"symbol={safe}",
    ]


def _read_cached_execution_1m(
    *,
    data_root: Path,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    paths: List[Path] = []
    for root in _candidate_execution_1m_dirs(data_root, symbol):
        if root.exists():
            paths.extend(sorted(root.glob("year=*/**/*.parquet")))
            paths.extend(sorted(root.glob("*.parquet")))
    if not paths:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for path in paths:
        try:
            df = pd.read_parquet(path, columns=["ts", "open", "high", "low", "close", "volume"])
        except Exception:
            continue
        if df.empty or "ts" not in df.columns:
            continue
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df[(df["ts"] >= start) & (df["ts"] <= end)]
        if not df.empty:
            df["observation_source"] = "execution_1m_cache"
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["ts"]).drop_duplicates(subset=["ts"], keep="last")
    return out.sort_values("ts").reset_index(drop=True)


def _iter_json_monitor_lines(log_path: Path, around_line: Optional[int]) -> Iterable[Mapping[str, Any]]:
    if not log_path.exists():
        return
    line_min = None
    line_max = None
    if around_line is not None and around_line > 0:
        line_min = max(1, int(around_line) - 80)
        line_max = int(around_line) + 80
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            if line_min is not None and lineno < line_min:
                continue
            if line_max is not None and lineno > line_max:
                break
            marker = "INFERENCE_MONITOR_HEARTBEAT "
            if marker not in line:
                continue
            try:
                payload = json.loads(line.split(marker, 1)[1])
            except Exception:
                continue
            if isinstance(payload, Mapping):
                yield payload


def _extract_trade_recap_from_log(row: Mapping[str, Any], workspace: Path) -> str:
    log_value = str(row.get("log") or "").strip()
    if not log_value:
        return ""
    log_path = Path(log_value)
    if not log_path.is_absolute():
        log_path = workspace / log_path
    around_line = None
    try:
        around_line = int(float(row.get("line")))
    except Exception:
        around_line = None

    symbol = str(row.get("symbol") or "")
    close_order_id = str(row.get("close_order_id") or "").strip()
    dedupe_key = str(row.get("dedupe_key") or "").strip()
    for payload in _iter_json_monitor_lines(log_path, around_line):
        statuses = payload.get("statuses")
        if not isinstance(statuses, Mapping):
            continue
        candidates: Sequence[Any]
        if symbol in statuses:
            candidates = [statuses[symbol]]
        else:
            candidates = list(statuses.values())
        for status in candidates:
            if not isinstance(status, Mapping):
                continue
            closed = status.get("closed_trade")
            if not isinstance(closed, Mapping):
                continue
            recap = str(closed.get("trade_recap") or "")
            if not recap:
                continue
            if close_order_id and close_order_id not in json.dumps(closed, default=str):
                continue
            if dedupe_key and dedupe_key not in json.dumps(closed, default=str):
                continue
            return recap
    return ""


def _parse_recap_observations(recap: str) -> ParsedRecap:
    rows: List[Dict[str, Any]] = []
    stop_fill_ts: Optional[pd.Timestamp] = None
    stop_fill_price = np.nan
    stop_reason = ""
    for line in str(recap or "").splitlines():
        m = _PRICE_BAR_5M_RE.search(line)
        if m:
            ts = _to_ts(m.group("ts"))
            if ts is not None:
                rows.append(
                    {
                        "ts": ts,
                        "open": _safe_float(m.group("open")),
                        "high": _safe_float(m.group("high")),
                        "low": _safe_float(m.group("low")),
                        "close": _safe_float(m.group("close")),
                        "volume": np.nan,
                        "observation_source": "live_trade_recap_price_bar_5m",
                    }
                )
            continue
        m = _CLOSEABLE_SAMPLE_RE.search(line)
        if m:
            ts = _to_ts(m.group("ts"))
            px = _safe_float(m.group("price"))
            if ts is not None and np.isfinite(px):
                rows.append(
                    {
                        "ts": ts,
                        "open": px,
                        "high": px,
                        "low": px,
                        "close": px,
                        "volume": np.nan,
                        "observation_source": "live_trade_recap_closeable_sample",
                    }
                )
            continue
        m = _STOP_FILLED_RE.search(line)
        if m:
            stop_fill_ts = _to_ts(m.group("ts"))
            stop_fill_price = _safe_float(m.group("fill_price"))
            stop_reason = str(m.group("stop_reason") or "")

    if rows:
        bars = pd.DataFrame(rows)
        bars["ts"] = pd.to_datetime(bars["ts"], utc=True, errors="coerce")
        bars = bars.dropna(subset=["ts"])
        bars = bars.drop_duplicates(subset=["ts", "observation_source"], keep="last")
        bars = bars.sort_values("ts").reset_index(drop=True)
    else:
        bars = pd.DataFrame()
    source = "live_trade_recap_5m_and_closeable_samples" if not bars.empty else "none"
    return ParsedRecap(bars, stop_fill_ts, stop_fill_price, stop_reason, source)


def _combined_cached_bars(
    *,
    data_root: Path,
    row: Mapping[str, Any],
    workspace: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Tuple[pd.DataFrame, str, ParsedRecap]:
    symbol = str(row.get("symbol") or "")
    minute = _read_cached_execution_1m(data_root=data_root, symbol=symbol, start=start, end=end)
    recap = _parse_recap_observations(_extract_trade_recap_from_log(row, workspace))
    frames = []
    sources = []
    if not minute.empty:
        frames.append(minute)
        sources.append("execution_1m_cache")
    if not recap.bars.empty:
        rb = recap.bars[(recap.bars["ts"] >= start) & (recap.bars["ts"] <= end)].copy()
        if not rb.empty:
            frames.append(rb)
            sources.append(recap.source)
    if not frames:
        return pd.DataFrame(), "none", recap
    bars = pd.concat(frames, ignore_index=True, sort=False)
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True, errors="coerce")
    bars = bars.dropna(subset=["ts"])
    bars = bars.sort_values(["ts", "observation_source"]).reset_index(drop=True)
    return bars, "+".join(dict.fromkeys(sources)), recap


def _state_after_initial(
    *,
    entry_price: float,
    stop_price: float,
    strategy_id: str,
    side: str,
    rank_percentile: float,
) -> Dict[str, Any]:
    return {
        "entry_price": float(entry_price),
        "stop_price": float(stop_price),
        "peak_price": float(entry_price),
        "mfe": 0.0,
        "mae": 0.0,
        "bars_in_trade": 0,
        "strategy_id": strategy_id,
        "side": side,
        "rank_percentile": rank_percentile,
    }


def _market_row(bar: Mapping[str, Any]) -> pd.DataFrame:
    ts = _to_ts(bar.get("ts"))
    data = {
        "open": _safe_float(bar.get("open")),
        "high": _safe_float(bar.get("high")),
        "low": _safe_float(bar.get("low")),
        "close": _safe_float(bar.get("close")),
    }
    if ts is None:
        return pd.DataFrame([data])
    return pd.DataFrame([data], index=pd.DatetimeIndex([ts]))


def _stop_hit(side: str, stop_price: float, bar: Mapping[str, Any]) -> Tuple[bool, float]:
    return stop_exit_fill_price(
        side=side,
        stop_px=stop_price,
        candle_high=_safe_float(bar.get("high")),
        candle_low=_safe_float(bar.get("low")),
        base_gap_bps=POLICY_STOP_EXIT_BASE_GAP_BPS,
        alpha_through=POLICY_STOP_EXIT_ALPHA_THROUGH,
        max_gap_bps=POLICY_STOP_EXIT_MAX_GAP_BPS,
    )


def _recover_barrier_frac(row: Mapping[str, Any], policy_params: Mapping[str, Any]) -> float:
    barrier_frac = _parse_float_from_detail(_BARRIER_RE, row.get("exit_reason_detail"))
    if np.isfinite(barrier_frac) and barrier_frac > 0.0:
        return float(barrier_frac)
    trigger = _parse_float_from_detail(_TRIGGER_RE, row.get("exit_reason_detail"))
    cap_mult = _safe_float(policy_params.get("capital_protect_mfe_mult"))
    if np.isfinite(trigger) and trigger > 0.0 and np.isfinite(cap_mult) and cap_mult > 0.0:
        return float(trigger / cap_mult)
    return np.nan


def _basis_points(new_value: float, ref_value: float, side: str = "long") -> float:
    if not np.isfinite(new_value) or not np.isfinite(ref_value) or ref_value == 0.0:
        return np.nan
    raw = (float(new_value) / float(ref_value) - 1.0) * 10000.0
    return raw if str(side).lower() == "long" else -raw


def replay_one_anchor(
    *,
    row: Mapping[str, Any],
    policy_params: Mapping[str, Any],
    entry_price: float,
    entry_anchor: str,
    bars: pd.DataFrame,
    recap: ParsedRecap,
) -> Dict[str, Any]:
    strategy_id = str(row.get("strategy_id") or "")
    side = str(row.get("side") or "long").lower()
    sl_mult = _parse_float_from_detail(_SL_MULT_RE, row.get("exit_reason_detail"))
    barrier_frac = _recover_barrier_frac(row, policy_params)
    params = dict(policy_params)
    if np.isfinite(sl_mult) and sl_mult > 0:
        params["sl_mult"] = float(sl_mult)
    if np.isfinite(barrier_frac) and barrier_frac > 0:
        params["barrier_frac"] = float(barrier_frac)

    rank_percentile = _safe_float(
        row.get("policy_rank_pct"),
        _safe_float(row.get("rank_percentile"), _safe_float(row.get("rank_pct"), 0.5)),
    )
    live_stop_price = _safe_float(row.get("stop_price"))
    live_exit_price = _safe_float(row.get("exit_price"))
    live_entry_price = _safe_float(row.get("entry_price"))

    out: Dict[str, Any] = {
        "symbol": row.get("symbol"),
        "side": side,
        "strategy_id": strategy_id,
        "entry_anchor": entry_anchor,
        "entry_price_used": entry_price,
        "live_entry_price": live_entry_price,
        "policy_entry_price": _safe_float(row.get("policy_entry_price")),
        "live_stop_price": live_stop_price,
        "live_exit_price": live_exit_price,
        "live_entry_to_exit_bps": _basis_points(live_exit_price, live_entry_price, side),
        "parsed_sl_mult": sl_mult,
        "parsed_barrier_frac": barrier_frac,
        "rank_percentile": rank_percentile,
        "coverage_status": "ok",
    }
    try:
        initial = compute_initial_simple_policy_stop_decision(
            entry_price=float(entry_price),
            policy_params=params,
            side=side,
            strategy_id=strategy_id,
            barrier_frac=float(barrier_frac) if np.isfinite(barrier_frac) else None,
            state={"strategy_id": strategy_id, "barrier_frac": barrier_frac},
            require_metadata=True,
        )
    except Exception as exc:
        out.update(
            {
                "coverage_status": "initial_stop_error",
                "error": f"{type(exc).__name__}: {exc}",
                "replay_hit": False,
            }
        )
        return out

    current_stop = float(initial.stop_price)
    current_stop_reason = str(initial.reason)
    state = _state_after_initial(
        entry_price=float(entry_price),
        stop_price=current_stop,
        strategy_id=strategy_id,
        side=side,
        rank_percentile=rank_percentile,
    )
    out.update(
        {
            "initial_stop": current_stop,
            "initial_stop_reason": current_stop_reason,
            "initial_stop_vs_live_stop_bps": _basis_points(current_stop, live_stop_price, side),
        }
    )

    if bars.empty:
        out.update(
            {
                "coverage_status": "no_cached_bars",
                "replay_hit": False,
                "replay_exit_reason": "",
                "replay_exit_ts": "",
                "replay_exit_price": np.nan,
            }
        )
        return out

    events: List[Dict[str, Any]] = []
    for _, bar in bars.iterrows():
        bar_ts = _to_ts(bar.get("ts"))
        hit, fill_price = _stop_hit(side, current_stop, bar)
        if hit:
            out.update(
                {
                    "replay_hit": True,
                    "replay_exit_reason": current_stop_reason,
                    "replay_exit_ts": bar_ts.isoformat() if bar_ts is not None else "",
                    "replay_exit_price": fill_price,
                    "replay_exit_price_vs_live_bps": _basis_points(fill_price, live_exit_price, side),
                    "replay_vs_live_exit_status": "replayed_stop_cross",
                    "replay_exit_vs_live_fill_event_bps": _basis_points(
                        fill_price, recap.stop_fill_price, side
                    ),
                    "replay_exit_from_observation_source": bar.get("observation_source"),
                    "events_json": json.dumps(events, default=str),
                    "live_stop_fill_ts": recap.stop_fill_ts.isoformat() if recap.stop_fill_ts is not None else "",
                    "live_stop_fill_price_from_recap": recap.stop_fill_price,
                    "live_stop_reason_from_recap": recap.stop_reason,
                }
            )
            return out

        try:
            decision = compute_simple_policy_stop_decision(
                state=state,
                latest_market_state=_market_row(bar),
                policy_params=params,
                side=side,
                require_metadata=True,
            )
        except Exception as exc:
            out.update(
                {
                    "coverage_status": "policy_update_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "replay_hit": False,
                    "events_json": json.dumps(events, default=str),
                }
            )
            return out

        state["peak_price"] = decision.peak_price
        state["mfe"] = decision.mfe
        state["mae"] = decision.mae
        state["bars_in_trade"] = int(state.get("bars_in_trade", 0)) + 1
        if decision.should_exit:
            px = _safe_float(bar.get("close"))
            out.update(
                {
                    "replay_hit": True,
                    "replay_exit_reason": decision.exit_reason or decision.reason,
                    "replay_exit_ts": bar_ts.isoformat() if bar_ts is not None else "",
                    "replay_exit_price": px,
                    "replay_exit_price_vs_live_bps": _basis_points(px, live_exit_price, side),
                    "replay_vs_live_exit_status": "replayed_policy_exit",
                    "replay_exit_vs_live_fill_event_bps": _basis_points(
                        px, recap.stop_fill_price, side
                    ),
                    "replay_exit_from_observation_source": bar.get("observation_source"),
                    "events_json": json.dumps(events, default=str),
                    "live_stop_fill_ts": recap.stop_fill_ts.isoformat() if recap.stop_fill_ts is not None else "",
                    "live_stop_fill_price_from_recap": recap.stop_fill_price,
                    "live_stop_reason_from_recap": recap.stop_reason,
                }
            )
            return out
        if decision.should_replace and decision.stop_price is not None:
            current_stop = float(decision.stop_price)
            current_stop_reason = str(decision.reason)
            state["stop_price"] = current_stop
            events.append(
                {
                    "ts": bar_ts.isoformat() if bar_ts is not None else "",
                    "event": "replace_stop",
                    "stop_price": current_stop,
                    "reason": current_stop_reason,
                    "detail": decision.reason_detail,
                }
            )

    out.update(
        {
            "replay_hit": False,
            "replay_exit_reason": "not_hit_in_cached_bars",
            "replay_exit_ts": "",
            "replay_exit_price": np.nan,
            "replay_exit_price_vs_live_bps": np.nan,
            "replay_vs_live_exit_status": (
                "live_fill_not_reproducible_from_cached_bars"
                if recap.stop_fill_ts is not None
                else "not_hit_no_live_fill_event"
            ),
            "replay_exit_vs_live_fill_event_bps": np.nan,
            "events_json": json.dumps(events, default=str),
            "live_stop_fill_ts": recap.stop_fill_ts.isoformat() if recap.stop_fill_ts is not None else "",
            "live_stop_fill_price_from_recap": recap.stop_fill_price,
            "live_stop_reason_from_recap": recap.stop_reason,
        }
    )
    return out


def _anchor_prices(row: Mapping[str, Any]) -> List[Tuple[str, float]]:
    anchors: List[Tuple[str, float]] = []
    for key, label in (
        ("policy_entry_price", "policy_entry"),
        ("entry_price", "realized_entry"),
        ("theoretical_entry_price", "theoretical_entry"),
    ):
        px = _safe_float(row.get(key))
        if np.isfinite(px) and px > 0.0 and all(abs(px - prev) > abs(px) * 1e-9 for _, prev in anchors):
            anchors.append((label, px))
    return anchors


def _summarise(results: pd.DataFrame) -> Dict[str, Any]:
    if results.empty:
        return {"rows": 0}
    grouped = (
        results.groupby(["entry_anchor", "coverage_status"], dropna=False)
        .size()
        .reset_index(name="rows")
        .to_dict(orient="records")
    )
    return {
        "rows": int(len(results)),
        "unique_trades": int(results[["symbol", "entry_time", "exit_time"]].drop_duplicates().shape[0])
        if {"symbol", "entry_time", "exit_time"}.issubset(results.columns)
        else None,
        "coverage_by_anchor": grouped,
        "replay_hits": int(pd.Series(results.get("replay_hit", False)).fillna(False).astype(bool).sum()),
        "mean_replay_exit_vs_live_bps": float(
            pd.to_numeric(results.get("replay_exit_price_vs_live_bps"), errors="coerce").mean()
        )
        if "replay_exit_price_vs_live_bps" in results
        else None,
    }


def run(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    workspace = Path.cwd()
    data_root = Path(args.data_root)
    closed_path = Path(args.closed_trades)
    if not closed_path.is_absolute():
        closed_path = workspace / closed_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = workspace / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    closed = pd.read_csv(closed_path)
    if args.symbols:
        wanted = {s.strip() for s in str(args.symbols).split(",") if s.strip()}
        closed = closed[closed["symbol"].isin(wanted)].copy()
    if args.limit:
        closed = closed.head(int(args.limit)).copy()

    params_by_strategy = load_simple_policy_stop_params_by_strategy(
        str(data_root), str(args.run_id)
    )
    results: List[Dict[str, Any]] = []
    for idx, row in closed.iterrows():
        row_d = row.to_dict()
        entry_ts = _to_ts(row_d.get("entry_time"))
        exit_ts = _to_ts(row_d.get("exit_time"))
        base = {
            "source_row": int(idx),
            "symbol": row_d.get("symbol"),
            "side": row_d.get("side"),
            "strategy_id": row_d.get("strategy_id"),
            "entry_time": entry_ts.isoformat() if entry_ts is not None else "",
            "exit_time": exit_ts.isoformat() if exit_ts is not None else "",
            "live_exit_reason_detail": row_d.get("exit_reason_detail"),
            "live_log": row_d.get("log"),
            "live_log_line": row_d.get("line"),
        }
        if entry_ts is None or exit_ts is None or exit_ts <= entry_ts:
            bad = dict(base)
            bad.update({"coverage_status": "invalid_time_window", "replay_hit": False})
            results.append(bad)
            continue
        strategy_id = str(row_d.get("strategy_id") or "")
        policy_params = params_by_strategy.get(strategy_id)
        if not policy_params:
            bad = dict(base)
            bad.update({"coverage_status": "missing_policy_params", "replay_hit": False})
            results.append(bad)
            continue
        bars, source, recap = _combined_cached_bars(
            data_root=data_root,
            row=row_d,
            workspace=workspace,
            start=entry_ts,
            end=exit_ts + pd.Timedelta(minutes=5),
        )
        bar_meta = {
            "bar_source": source,
            "bar_count": int(len(bars)),
            "first_bar_ts": bars["ts"].min().isoformat() if not bars.empty else "",
            "last_bar_ts": bars["ts"].max().isoformat() if not bars.empty else "",
        }
        anchors = _anchor_prices(row_d)
        if not anchors:
            bad = dict(base)
            bad.update(bar_meta)
            bad.update({"coverage_status": "missing_entry_anchor", "replay_hit": False})
            results.append(bad)
            continue
        for anchor, entry_price in anchors:
            replay = replay_one_anchor(
                row=row_d,
                policy_params=policy_params,
                entry_price=entry_price,
                entry_anchor=anchor,
                bars=bars,
                recap=recap,
            )
            out = dict(base)
            out.update(bar_meta)
            out.update(replay)
            results.append(out)

    result_df = pd.DataFrame(results)
    result_path = out_dir / "live_closed_trade_exit_replay.csv"
    summary_path = out_dir / "live_closed_trade_exit_replay_summary.json"
    result_df.to_csv(result_path, index=False)
    summary = _summarise(result_df)
    summary.update(
        {
            "closed_trades_path": str(closed_path),
            "run_id": str(args.run_id),
            "data_root": str(data_root),
            "result_path": str(result_path),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return result_df, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--run-id", default="20260525_010004_nopenalty")
    parser.add_argument("--closed-trades", default=str(DEFAULT_CLOSED_TRADES))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--symbols", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result_df, summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not result_df.empty:
        cols = [
            "symbol",
            "entry_anchor",
            "coverage_status",
            "bar_source",
            "bar_count",
            "initial_stop",
            "live_stop_price",
            "replay_hit",
            "replay_exit_reason",
            "replay_exit_ts",
            "replay_exit_price",
            "live_exit_price",
            "replay_exit_price_vs_live_bps",
            "replay_vs_live_exit_status",
        ]
        existing = [c for c in cols if c in result_df.columns]
        print(result_df[existing].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
