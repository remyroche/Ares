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
from extreme_price_movements.data_store import canonical_kraken_execution_1m_root


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
_PRICE_BAR_RE = re.compile(
    rf"(?P<ts>\d{{4}}-\d{{2}}-\d{{2}}T[^\s]+)\s+price_bar_(?P<interval>5m|15m)\s+"
    rf"open=(?P<open>{_FLOAT_RE})\s+high=(?P<high>{_FLOAT_RE})\s+"
    rf"low=(?P<low>{_FLOAT_RE})\s+close=(?P<close>{_FLOAT_RE})"
)
_CLOSEABLE_SAMPLE_RE = re.compile(
    rf"(?P<ts>\d{{4}}-\d{{2}}-\d{{2}}T[^\s]+)\s+live_closeable_price_sample\s+"
    rf".*?\bprice=(?P<price>{_FLOAT_RE})\b"
)
_SENTINEL_SAMPLE_RE = re.compile(
    rf"(?P<ts>\d{{4}}-\d{{2}}-\d{{2}}T[^\s]+)\s+lightweight_stop_sentinel_sample\s+"
    rf".*?\bprice=(?P<price>{_FLOAT_RE})\b"
)
_SENTINEL_PRETRIGGER_RE = re.compile(
    rf"(?P<ts>\d{{4}}-\d{{2}}-\d{{2}}T[^\s]+)\s+lightweight_stop_sentinel_pretrigger\s+"
    rf".*?\bprice=(?P<price>{_FLOAT_RE})\b"
    rf".*?\bstop_price=(?P<stop_price>{_FLOAT_RE})\b"
    rf".*?\bstop_reason=(?P<stop_reason>[^\s]+)\s+"
    rf".*?\bexit_reason=(?P<exit_reason>[^\s]+)"
)
_SOFTWARE_PRE_REPLACE_BREACH_RE = re.compile(
    rf"(?P<ts>\d{{4}}-\d{{2}}-\d{{2}}T[^\s]+)\s+software_policy_stop_breached_before_exchange_replace\s+"
    rf".*?\bpolicy_stop=(?P<stop_price>{_FLOAT_RE})\b"
    rf".*?\bcurrent_price=(?P<price>{_FLOAT_RE})\b"
    rf".*?\bstop_reason=(?P<stop_reason>[^\s]+)\s+"
    rf".*?\bexit_reason=(?P<exit_reason>[^\s]+)"
)
_STOP_FILLED_RE = re.compile(
    rf"(?P<ts>\d{{4}}-\d{{2}}-\d{{2}}T[^\s]+)\s+stop_order_filled\s+"
    rf".*?\bfill_price=(?P<fill_price>{_FLOAT_RE})\b"
    rf".*?\bstop_reason=(?P<stop_reason>[^\s]+)"
)
_SOFTWARE_POLICY_STOP_CLOSE_RE = re.compile(
    rf"(?P<ts>\d{{4}}-\d{{2}}-\d{{2}}T[^\s]+)\s+software_policy_stop_close\s+"
    rf".*?\bcurrent_price=(?P<price>{_FLOAT_RE})\b"
)
_STOP_REPLACED_RE = re.compile(
    rf"(?P<ts>\d{{4}}-\d{{2}}-\d{{2}}T[^\s]+)\s+stop_replaced\s+"
    rf"stop_reason=(?P<stop_reason>[^\s]+)\s+"
    rf".*?\bnew_stop=(?P<stop_price>{_FLOAT_RE})\b"
)
_EXCHANGE_STOP_ADOPTED_RE = re.compile(
    rf"(?P<ts>\d{{4}}-\d{{2}}-\d{{2}}T[^\s]+)\s+"
    rf"existing_stop_adopted_on_reattach\s+"
    rf".*?\bstop_price=(?P<stop_price>{_FLOAT_RE})\b"
    rf"(?:.*?\bstop_reason=(?P<stop_reason>[^\s]+))?"
)
_SPREAD_BPS_RE = re.compile(rf"\bspread_bps=(?P<spread_bps>{_FLOAT_RE})\b")

POLICY_STOP_EXIT_BASE_GAP_BPS = float(
    os.environ.get("EPM_SIMPLE_POLICY_STOP_EXIT_BASE_GAP_BPS", "15.0")
)
POLICY_STOP_EXIT_ALPHA_THROUGH = float(
    os.environ.get("EPM_SIMPLE_POLICY_STOP_EXIT_ALPHA_THROUGH", "0.05")
)
POLICY_STOP_EXIT_MAX_GAP_BPS = float(
    os.environ.get("EPM_SIMPLE_POLICY_STOP_EXIT_MAX_GAP_BPS", "75.0")
)
DEFAULT_PERP_EXIT_QUOTE_HALF_SPREAD_BPS = float(
    os.environ.get("EPM_SIMPLE_POLICY_PERP_EXIT_QUOTE_HALF_SPREAD_BPS", "40.0")
)
DEFAULT_SPOT_EXIT_QUOTE_HALF_SPREAD_BPS = float(
    os.environ.get("EPM_SIMPLE_POLICY_SPOT_EXIT_QUOTE_HALF_SPREAD_BPS", "0.0")
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


def _row_event_ts(row: Mapping[str, Any]) -> pd.Timestamp:
    for key in ("exit_time", "entry_time", "timestamp", "created_at"):
        ts = _to_ts(row.get(key))
        if ts is not None:
            return ts
    return pd.Timestamp.min.tz_localize("UTC")


def _select_closed_trade_rows(
    closed: pd.DataFrame,
    *,
    symbols: str = "",
    limit: Optional[int] = None,
    since: Any = None,
) -> pd.DataFrame:
    """Filter closed trades and keep the latest rows for focused live audits."""
    out = closed.copy()
    if symbols:
        wanted = {s.strip() for s in str(symbols).split(",") if s.strip()}
        if wanted and "symbol" in out.columns:
            out = out[out["symbol"].isin(wanted)].copy()
    if out.empty:
        return out

    # The lifecycle ledger can contain entry and exit rows for the same trade.
    # Prefer rows with a valid UTC entry/exit interval before sorting: the
    # display timestamp may be naive CEST and must not make an entry row look
    # newer than its corresponding closed-trade row.
    entry_ts = out.get("entry_time", pd.Series(index=out.index, dtype=object)).map(_to_ts)
    exit_ts = out.get("exit_time", pd.Series(index=out.index, dtype=object)).map(_to_ts)
    valid_closed = entry_ts.notna() & exit_ts.notna() & (exit_ts > entry_ts)
    if bool(valid_closed.any()):
        out = out.loc[valid_closed].copy()
        out["_event_ts_for_replay_sort"] = exit_ts.loc[valid_closed]
    else:
        out["_event_ts_for_replay_sort"] = out.apply(_row_event_ts, axis=1)
    since_ts = _to_ts(since)
    if since is not None and since_ts is None:
        raise ValueError(f"Invalid --since timestamp: {since!r}")
    if since_ts is not None:
        out = out.loc[out["_event_ts_for_replay_sort"] >= since_ts].copy()
    out = out.sort_values("_event_ts_for_replay_sort")
    if limit:
        out = out.tail(int(limit)).copy()
    return out.drop(columns=["_event_ts_for_replay_sort"], errors="ignore")


def _parse_float_from_detail(pattern: re.Pattern[str], detail: Any) -> float:
    match = pattern.search(str(detail or ""))
    if not match:
        return np.nan
    return _safe_float(match.group(1))


def _safe_symbol_path(symbol: str) -> str:
    return str(symbol).replace("/", "_")


def _candidate_execution_1m_dirs(data_root: Path, symbol: str) -> List[Path]:
    safe = _safe_symbol_path(symbol)
    return [canonical_kraken_execution_1m_root(data_root) / "ohlcv" / f"symbol={safe}"]


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
    embedded = str(row.get("trade_recap") or "").strip()
    if embedded and embedded.lower() != "nan":
        return embedded

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
        m = _PRICE_BAR_RE.search(line)
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
                        "observation_source": (
                            f"live_trade_recap_price_bar_{m.group('interval')}"
                        ),
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
        m = _SENTINEL_SAMPLE_RE.search(line)
        if m:
            ts = _to_ts(m.group("ts"))
            px = _safe_float(m.group("price"))
            spread_match = _SPREAD_BPS_RE.search(line)
            spread_bps = (
                _safe_float(spread_match.group("spread_bps"))
                if spread_match is not None
                else np.nan
            )
            if ts is not None and np.isfinite(px):
                rows.append(
                    {
                        "ts": ts,
                        "open": px,
                        "high": px,
                        "low": px,
                        "close": px,
                        "volume": np.nan,
                        "spread_bps": spread_bps,
                        "observation_source": "live_trade_recap_stop_sentinel_sample",
                    }
                )
            continue
        m = _SOFTWARE_POLICY_STOP_CLOSE_RE.search(line)
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
                        "observation_source": (
                            "live_trade_recap_software_policy_stop_sample"
                        ),
                    }
                )
            continue
        m = _STOP_REPLACED_RE.search(line)
        if m:
            ts = _to_ts(m.group("ts"))
            stop_price = _safe_float(m.group("stop_price"))
            if ts is not None and np.isfinite(stop_price):
                rows.append(
                    {
                        "ts": ts,
                        "open": np.nan,
                        "high": np.nan,
                        "low": np.nan,
                        "close": np.nan,
                        "volume": np.nan,
                        "observation_source": "live_trade_recap_stop_replaced",
                        "logged_stop_replace": True,
                        "logged_stop_price": float(stop_price),
                        "logged_stop_reason": str(m.group("stop_reason") or ""),
                    }
                )
            continue
        m = _EXCHANGE_STOP_ADOPTED_RE.search(line)
        if m:
            ts = _to_ts(m.group("ts"))
            stop_price = _safe_float(m.group("stop_price"))
            if ts is not None and np.isfinite(stop_price):
                rows.append(
                    {
                        "ts": ts,
                        "open": np.nan,
                        "high": np.nan,
                        "low": np.nan,
                        "close": np.nan,
                        "volume": np.nan,
                        "observation_source": (
                            "live_trade_recap_exchange_stop_adopted"
                        ),
                        "logged_stop_replace": True,
                        "logged_stop_price": float(stop_price),
                        # Reattachment changes the hosted executable price, not
                        # the policy's semantic exit reason. Preserve the
                        # existing reason when the exchange event omits one.
                        "logged_stop_reason": str(m.group("stop_reason") or ""),
                    }
                )
            continue
        m = _SENTINEL_PRETRIGGER_RE.search(line)
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
                        "observation_source": "live_trade_recap_logged_exit_trigger",
                        "logged_exit_trigger": True,
                        "logged_stop_price": _safe_float(m.group("stop_price")),
                        "logged_stop_reason": str(m.group("stop_reason") or ""),
                        "logged_exit_reason": str(m.group("exit_reason") or ""),
                    }
                )
            continue
        m = _SOFTWARE_PRE_REPLACE_BREACH_RE.search(line)
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
                        "observation_source": "live_trade_recap_logged_exit_trigger",
                        "logged_exit_trigger": True,
                        "logged_stop_price": _safe_float(m.group("stop_price")),
                        "logged_stop_reason": str(m.group("stop_reason") or ""),
                        "logged_exit_reason": str(m.group("exit_reason") or ""),
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


def _logged_live_software_handoff_exit(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a replay-compatible logged exit for live-only software handoffs."""
    reason = str(row.get("reason") or row.get("exit_reason") or "").strip()
    detail = str(row.get("exit_reason_detail") or "").strip()
    trigger_type = str(row.get("close_trigger_type") or "").strip().lower()
    execution_method = str(row.get("close_execution_method") or "").strip().lower()
    is_handoff = (
        "exchange_valid_giveback_fallback_handoff" in reason
        or "exchange_valid_giveback_fallback_handoff" in detail
        or "software_executable_stop_breach_pre_replace" in reason
        or "software_executable_stop_breach_pre_replace" in detail
    )
    if not is_handoff:
        return None
    if trigger_type and trigger_type not in {
        "software_bid_ask_sentinel",
        "software_policy_stop",
        "software_policy_stop_pre_replace",
    }:
        return None
    if (
        execution_method
        and "software" not in execution_method
        and "ask_bid" not in execution_method
    ):
        return None
    ts = _to_ts(row.get("exit_time"))
    px = _safe_float(
        row.get("exit_price"),
        _safe_float(
            row.get("actual_exit_price"),
            _safe_float(row.get("realized_exit_price")),
        ),
    )
    if ts is None or not (np.isfinite(px) and px > 0.0):
        return None
    return {
        "reason": reason
        or (
            "software_executable_stop_breach_pretrigger:"
            "exchange_valid_giveback_fallback_handoff"
        ),
        "ts": ts,
        "price": float(px),
        "status": "logged_live_software_handoff",
    }


def _logged_live_exchange_stop_fill(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a replay-compatible logged exit for exchange stop fills."""
    trigger_type = str(row.get("close_trigger_type") or "").strip().lower()
    price_source = str(row.get("close_price_source") or "").strip().lower()
    execution_method = str(row.get("close_execution_method") or "").strip().lower()
    is_exchange_stop = (
        trigger_type == "exchange_stop_order"
        or price_source == "exchange_stop_order_fill"
        or "exchange_stop_order" in execution_method
    )
    if not is_exchange_stop:
        return None
    ts = _to_ts(row.get("exit_time"))
    px = _safe_float(
        row.get("exit_price"),
        _safe_float(
            row.get("actual_exit_price"),
            _safe_float(row.get("realized_exit_price")),
        ),
    )
    if ts is None or not (np.isfinite(px) and px > 0.0):
        return None
    reason = str(row.get("reason") or row.get("exit_reason") or "").strip()
    stop_reason = str(row.get("stop_reason") or "").strip()
    if not reason:
        reason = f"stop_order_filled:{stop_reason or 'exchange_stop_order'}"
    return {
        "reason": reason,
        "ts": ts,
        "price": float(px),
        "status": "logged_live_exchange_stop_fill_from_closed_trade",
    }


def _combined_cached_bars(
    *,
    data_root: Path,
    row: Mapping[str, Any],
    workspace: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    strict_execution_1m_only: bool = False,
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
        if strict_execution_1m_only and not rb.empty:
            # Strict replay sources every market price from the canonical 1m
            # cache, but the executable stop is exchange state rather than a
            # derived price observation. Retain only timestamped stop-order
            # transitions so replay can test whether those hosted stops would
            # have triggered on the independent 1m path.
            rb = rb[
                rb["observation_source"].isin(
                    {
                        "live_trade_recap_stop_replaced",
                        "live_trade_recap_exchange_stop_adopted",
                    }
                )
            ].copy()
        if not rb.empty:
            frames.append(rb)
            sources.append(
                "live_trade_recap_exchange_stop_state"
                if strict_execution_1m_only
                else recap.source
            )
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
    barrier_frac: float,
    strategy_id: str,
    side: str,
    rank_percentile: float,
) -> Dict[str, Any]:
    return {
        "entry_price": float(entry_price),
        "stop_price": float(stop_price),
        # The caller receives this from the initial decision, where the raw
        # row barrier has already had the policy ATR transform applied.
        "barrier_frac": float(barrier_frac),
        "barrier_pct": float(barrier_frac),
        "barrier_frac_is_effective": True,
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


def _exit_quote_half_spread_bps(row: Mapping[str, Any]) -> float:
    env_value = os.environ.get("EPM_SIMPLE_POLICY_EXIT_QUOTE_HALF_SPREAD_BPS")
    if env_value not in {None, ""}:
        parsed = _safe_float(env_value)
        if np.isfinite(parsed):
            return max(0.0, float(parsed))
    for key in ("exit_quote_half_spread_bps", "exit_spread_cost_bps"):
        parsed = _safe_float(row.get(key))
        if np.isfinite(parsed):
            return max(0.0, float(parsed))
    symbol = str(row.get("symbol") or "")
    return (
        max(0.0, float(DEFAULT_PERP_EXIT_QUOTE_HALF_SPREAD_BPS))
        if ":USD" in symbol
        else max(0.0, float(DEFAULT_SPOT_EXIT_QUOTE_HALF_SPREAD_BPS))
    )


def _stop_hit(
    side: str,
    stop_price: float,
    bar: Mapping[str, Any],
    *,
    quote_half_spread_bps: float,
) -> Tuple[bool, float]:
    return stop_exit_fill_price(
        side=side,
        stop_px=stop_price,
        candle_high=_safe_float(bar.get("high")),
        candle_low=_safe_float(bar.get("low")),
        base_gap_bps=POLICY_STOP_EXIT_BASE_GAP_BPS,
        alpha_through=POLICY_STOP_EXIT_ALPHA_THROUGH,
        max_gap_bps=POLICY_STOP_EXIT_MAX_GAP_BPS,
        quote_half_spread_bps=quote_half_spread_bps,
    )


def _recover_barrier_frac(
    row: Mapping[str, Any],
    policy_params: Mapping[str, Any],
    *,
    entry_price: float = np.nan,
    side: str = "long",
) -> float:
    barrier_frac = _parse_float_from_detail(_BARRIER_RE, row.get("exit_reason_detail"))
    if np.isfinite(barrier_frac) and barrier_frac > 0.0:
        return float(barrier_frac)
    for key in (
        "barrier_frac",
        "barrier_pct",
        "policy_effective_barrier_pct",
        "policy_barrier_frac",
    ):
        barrier_frac = _safe_float(row.get(key))
        if np.isfinite(barrier_frac) and barrier_frac > 0.0:
            return float(barrier_frac)
    trigger = _parse_float_from_detail(_TRIGGER_RE, row.get("exit_reason_detail"))
    cap_mult = _safe_float(policy_params.get("capital_protect_mfe_mult"))
    if np.isfinite(trigger) and trigger > 0.0 and np.isfinite(cap_mult) and cap_mult > 0.0:
        return float(trigger / cap_mult)
    # If the lifecycle row retained its entry stop, invert the exact initial
    # stop geometry. This is more faithful than the train median and remains
    # causal because both the stop and sl_mult were fixed at entry.
    entry = _safe_float(entry_price)
    initial_stop = _safe_float(
        row.get("shadow_initial_stop_price"),
        _safe_float(row.get("initial_stop_price"), _safe_float(row.get("entry_stop_price"))),
    )
    sl_mult = _safe_float(policy_params.get("sl_mult"))
    if (
        np.isfinite(entry)
        and entry > 0.0
        and np.isfinite(initial_stop)
        and initial_stop > 0.0
        and np.isfinite(sl_mult)
        and sl_mult > 0.0
    ):
        distance = (
            (entry - initial_stop) / entry
            if str(side).lower() == "long"
            else (initial_stop - entry) / entry
        )
        if np.isfinite(distance) and distance > 0.0:
            return float(distance / sl_mult)
    # The promoted 1-minute trailing pathway persists the train-fitted ATR
    # anchor under these names. Closed lifecycle rows do not always retain the
    # entry-time effective barrier, so use the frozen policy anchor rather than
    # failing the audit or inventing a new value.
    for key in (
        "median_barrier_frac",
        "policy_median_barrier_frac",
        "median_atr_frac",
        "policy_median_atr_frac",
    ):
        barrier_frac = _safe_float(policy_params.get(key))
        if np.isfinite(barrier_frac) and barrier_frac > 0.0:
            return float(barrier_frac)
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
    ignore_logged_exit_events: bool = False,
) -> Dict[str, Any]:
    strategy_id = str(row.get("strategy_id") or "")
    side = str(row.get("side") or "long").lower()
    sl_mult = _parse_float_from_detail(_SL_MULT_RE, row.get("exit_reason_detail"))
    barrier_frac = _recover_barrier_frac(
        row,
        policy_params,
        entry_price=float(entry_price),
        side=side,
    )
    params = dict(policy_params)
    if np.isfinite(sl_mult) and sl_mult > 0:
        params["sl_mult"] = float(sl_mult)
    if np.isfinite(barrier_frac) and barrier_frac > 0:
        params["barrier_frac"] = float(barrier_frac)

    rank_percentile = _safe_float(
        row.get("policy_rank_pct"),
        _safe_float(row.get("rank_percentile"), _safe_float(row.get("rank_pct"), 0.5)),
    )
    # Closed trade rows often carry the final/tightened stop in ``stop_price``.
    # For an entry-to-exit replay, seed the path from the entry stop when the
    # logger preserved it; otherwise compute the initial stop from policy params.
    live_initial_stop_price = _safe_float(
        row.get("shadow_initial_stop_price"),
        _safe_float(row.get("initial_stop_price"), _safe_float(row.get("entry_stop_price"))),
    )
    live_final_stop_price = _safe_float(
        row.get("shadow_live_stop_price"),
        _safe_float(row.get("stop_price"), _safe_float(row.get("final_placed_stop"))),
    )
    live_exit_price = _safe_float(
        row.get("exit_price"),
        _safe_float(row.get("actual_exit_price"), _safe_float(row.get("realized_exit_price"))),
    )
    live_entry_price = _safe_float(row.get("entry_price"))
    exit_quote_half_spread_bps = _exit_quote_half_spread_bps(row)

    out: Dict[str, Any] = {
        "symbol": row.get("symbol"),
        "side": side,
        "strategy_id": strategy_id,
        "entry_anchor": entry_anchor,
        "entry_price_used": entry_price,
        "live_entry_price": live_entry_price,
        "policy_entry_price": _safe_float(row.get("policy_entry_price")),
        "live_initial_stop_price": live_initial_stop_price,
        "live_final_stop_price": live_final_stop_price,
        "live_stop_price": live_initial_stop_price,
        "live_exit_price": live_exit_price,
        "live_entry_to_exit_bps": _basis_points(live_exit_price, live_entry_price, side),
        "parsed_sl_mult": sl_mult,
        "parsed_barrier_frac": barrier_frac,
        "rank_percentile": rank_percentile,
        "exit_quote_half_spread_bps": exit_quote_half_spread_bps,
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

    computed_initial_stop = float(initial.stop_price)
    use_logged_stop = np.isfinite(live_initial_stop_price) and live_initial_stop_price > 0.0
    current_stop = float(live_initial_stop_price if use_logged_stop else computed_initial_stop)
    current_stop_reason = str(initial.reason)
    state = _state_after_initial(
        entry_price=float(entry_price),
        stop_price=current_stop,
        barrier_frac=float(initial.barrier_frac),
        strategy_id=strategy_id,
        side=side,
        rank_percentile=rank_percentile,
    )
    state["semantic_stop_reason"] = current_stop_reason
    out.update(
        {
            "initial_stop": current_stop,
            "computed_initial_stop": computed_initial_stop,
            "initial_stop_source": "logged_live_stop" if use_logged_stop else "computed_policy_stop",
            "initial_stop_reason": current_stop_reason,
            "initial_stop_vs_live_stop_bps": _basis_points(
                current_stop,
                live_initial_stop_price,
                side,
            ),
        }
    )

    events: List[Dict[str, Any]] = []
    logged_software_exit = (
        None if ignore_logged_exit_events else _logged_live_software_handoff_exit(row)
    )
    if logged_software_exit is not None:
        px = float(logged_software_exit["price"])
        out.update(
            {
                "replay_hit": True,
                "replay_exit_reason": str(logged_software_exit["reason"]),
                "replay_exit_ts": logged_software_exit["ts"].isoformat(),
                "replay_exit_price": px,
                "replay_exit_price_vs_live_bps": _basis_points(
                    px,
                    live_exit_price,
                    side,
                ),
                "replay_vs_live_exit_status": logged_software_exit["status"],
                "replay_exit_vs_live_fill_event_bps": np.nan,
                "replay_exit_from_observation_source": "closed_trade_logged_software_handoff",
                "events_json": json.dumps(events, default=str),
                "live_stop_fill_ts": recap.stop_fill_ts.isoformat()
                if recap.stop_fill_ts is not None
                else "",
                "live_stop_fill_price_from_recap": recap.stop_fill_price,
                "live_stop_reason_from_recap": recap.stop_reason,
            }
        )
        return out

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

    close_trigger_type = str(row.get("close_trigger_type") or "").lower()
    close_execution_method = str(row.get("close_execution_method") or "").lower()
    if (
        not ignore_logged_exit_events
        and
        recap.stop_fill_ts is not None
        and np.isfinite(recap.stop_fill_price)
        and (
            close_trigger_type == "exchange_stop_order"
            or "exchange" in close_execution_method
            or str(row.get("close_price_source") or "").lower() == "exchange_stop_order_fill"
        )
    ):
        out.update(
            {
                "replay_hit": True,
                "replay_exit_reason": f"stop_order_filled:{recap.stop_reason or current_stop_reason}",
                "replay_exit_ts": recap.stop_fill_ts.isoformat(),
                "replay_exit_price": recap.stop_fill_price,
                "replay_exit_price_vs_live_bps": _basis_points(
                    recap.stop_fill_price,
                    live_exit_price,
                    side,
                ),
                "replay_vs_live_exit_status": "logged_live_exchange_stop_fill",
                "replay_exit_vs_live_fill_event_bps": 0.0,
                "replay_exit_from_observation_source": "live_trade_recap_stop_order_filled",
                "events_json": json.dumps(events, default=str),
                "live_stop_fill_ts": recap.stop_fill_ts.isoformat(),
                "live_stop_fill_price_from_recap": recap.stop_fill_price,
                "live_stop_reason_from_recap": recap.stop_reason,
            }
        )
        return out

    logged_exchange_stop = (
        None if ignore_logged_exit_events else _logged_live_exchange_stop_fill(row)
    )
    if logged_exchange_stop is not None:
        px = float(logged_exchange_stop["price"])
        out.update(
            {
                "replay_hit": True,
                "replay_exit_reason": str(logged_exchange_stop["reason"]),
                "replay_exit_ts": logged_exchange_stop["ts"].isoformat(),
                "replay_exit_price": px,
                "replay_exit_price_vs_live_bps": _basis_points(
                    px,
                    live_exit_price,
                    side,
                ),
                "replay_vs_live_exit_status": logged_exchange_stop["status"],
                "replay_exit_vs_live_fill_event_bps": 0.0,
                "replay_exit_from_observation_source": "closed_trade_exchange_stop_fill",
                "events_json": json.dumps(events, default=str),
                "live_stop_fill_ts": recap.stop_fill_ts.isoformat()
                if recap.stop_fill_ts is not None
                else "",
                "live_stop_fill_price_from_recap": recap.stop_fill_price,
                "live_stop_reason_from_recap": recap.stop_reason,
            }
        )
        return out

    logged_trigger_series = (
        bars["logged_exit_trigger"]
        if "logged_exit_trigger" in bars.columns
        else pd.Series(False, index=bars.index)
    )
    use_logged_exit_trigger_path = (
        not ignore_logged_exit_events
        and logged_trigger_series.map(
            lambda value: value is True
            or str(value).strip().lower() in {"1", "true", "yes"}
        ).any()
    )
    has_live_recap_observations = (
        bars.get("observation_source", pd.Series("", index=bars.index))
        .fillna("")
        .astype(str)
        .str.startswith("live_trade_recap_")
        .any()
    )
    for _, bar in bars.iterrows():
        bar_ts = _to_ts(bar.get("ts"))
        observation_source = str(bar.get("observation_source") or "")
        policy_bar_observation = observation_source in {
            "live_trade_recap_price_bar_5m",
            "live_trade_recap_price_bar_15m",
        }
        completed_policy_bar = policy_bar_observation or (
            str(params.get("replay_timeframe") or "").lower() == "1m"
            and observation_source == "execution_1m_cache"
        )
        logged_stop_replace_value = bar.get("logged_stop_replace", False)
        logged_stop_replace = (
            logged_stop_replace_value is True
            or str(logged_stop_replace_value).strip().lower()
            in {"1", "true", "yes"}
        )
        if logged_stop_replace:
            next_stop = _safe_float(bar.get("logged_stop_price"))
            next_reason = str(bar.get("logged_stop_reason") or current_stop_reason)
            if np.isfinite(next_stop) and next_stop > 0.0:
                current_stop = float(next_stop)
                current_stop_reason = next_reason
                state["stop_price"] = current_stop
                if next_reason not in {
                    "exchange_valid_giveback_fallback",
                    "exchange_valid_stop_fallback",
                }:
                    state["semantic_stop_reason"] = next_reason
                events.append(
                    {
                        "ts": bar_ts.isoformat() if bar_ts is not None else "",
                        "event": "persisted_live_stop_replace",
                        "stop_price": current_stop,
                        "reason": current_stop_reason,
                    }
                )
            continue
        logged_trigger_value = bar.get("logged_exit_trigger", False)
        logged_trigger = (
            logged_trigger_value is True
            or str(logged_trigger_value).strip().lower() in {"1", "true", "yes"}
        )
        if logged_trigger and not ignore_logged_exit_events:
            px = _safe_float(bar.get("close"))
            out.update(
                {
                    "replay_hit": True,
                    "replay_exit_reason": str(
                        bar.get("logged_exit_reason")
                        or bar.get("logged_stop_reason")
                        or current_stop_reason
                    ),
                    "replay_exit_ts": bar_ts.isoformat() if bar_ts is not None else "",
                    "replay_exit_price": px,
                    "replay_exit_price_vs_live_bps": _basis_points(px, live_exit_price, side),
                    "replay_vs_live_exit_status": "logged_live_exit_trigger",
                    "replay_exit_vs_live_fill_event_bps": _basis_points(
                        px, recap.stop_fill_price, side
                    ),
                    "replay_exit_from_observation_source": bar.get("observation_source"),
                    "events_json": json.dumps(events, default=str),
                    "live_stop_fill_ts": recap.stop_fill_ts.isoformat() if recap.stop_fill_ts is not None else "",
                    "live_stop_fill_price_from_recap": recap.stop_fill_price,
                    "live_stop_reason_from_recap": recap.stop_reason,
                    "logged_stop_price": bar.get("logged_stop_price"),
                    "logged_stop_reason": bar.get("logged_stop_reason"),
                }
            )
            return out
        if use_logged_exit_trigger_path:
            continue
        # Production records completed policy bars for state updates but leaves
        # exchange-stop fills to the exchange order and executable ticker
        # sentinel. Re-triggering those stops from OHLC ranges double counts the
        # spread and can create exits that never occurred live.
        if policy_bar_observation:
            hit, fill_price = False, np.nan
        else:
            observation_is_executable_price = observation_source in {
                "live_trade_recap_closeable_sample",
                "live_trade_recap_stop_sentinel_sample",
                "live_trade_recap_software_policy_stop_sample",
                "live_trade_recap_logged_exit_trigger",
            }
            hit, fill_price = _stop_hit(
                side,
                current_stop,
                bar,
                # Recap samples are already the closeable bid/ask. Applying a
                # second half-spread creates premature replay exits.
                quote_half_spread_bps=(
                    0.0
                    if observation_is_executable_price
                    else exit_quote_half_spread_bps
                ),
            )
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

        observation_price = _safe_float(bar.get("close"))
        if policy_bar_observation:
            state["capital_protection_disabled_for_observation"] = True
            state.pop("capital_protect_observation_ts", None)
            state.pop("capital_protect_current_price", None)
        else:
            state.pop("capital_protection_disabled_for_observation", None)
            if bar_ts is not None:
                state["capital_protect_observation_ts"] = bar_ts
            else:
                state.pop("capital_protect_observation_ts", None)
            if np.isfinite(observation_price):
                state["capital_protect_current_price"] = float(observation_price)
            else:
                state.pop("capital_protect_current_price", None)
        observed_spread_bps = _safe_float(bar.get("spread_bps"))
        if np.isfinite(observed_spread_bps):
            state["spread_bps"] = float(observed_spread_bps)

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
        state["capital_protect_armed"] = bool(
            getattr(decision, "capital_protect_armed", False)
        )
        state["capital_protect_armed_now"] = bool(
            getattr(decision, "capital_protect_armed_now", False)
        )
        crossed_ts = getattr(decision, "capital_protect_crossed_ts", None)
        if crossed_ts is None:
            state.pop("capital_protect_crossed_ts", None)
        else:
            state["capital_protect_crossed_ts"] = str(crossed_ts)
        state["capital_protect_pending"] = bool(
            getattr(decision, "capital_protect_pending", False)
        )
        activation_return = getattr(
            decision, "capital_protect_activation_return", None
        )
        if activation_return is not None:
            state["capital_protect_activation_return"] = float(
                activation_return
            )
        if completed_policy_bar:
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
        ticker_driven_observation = observation_source in {
            "live_trade_recap_closeable_sample",
            "live_trade_recap_stop_sentinel_sample",
            "live_trade_recap_software_policy_stop_sample",
        }
        ticker_driven_reason = str(decision.reason) in {
            "trailing_profit",
            "capital_preservation",
            "capital_preservation_armed",
        }
        # Production's lightweight sentinel observes bid/ask every 30 seconds,
        # but it may only tighten ticker-driven trailing/capital protection.
        # Applying ordinary policy-stop pressure to each sentinel sample makes
        # the audit exit earlier than live and is not the deployed cadence.
        replacement_allowed = (
            not ticker_driven_observation or ticker_driven_reason
        )
        if (
            decision.should_replace
            and decision.stop_price is not None
            and replacement_allowed
        ):
            requested_stop = float(decision.stop_price)
            events.append(
                {
                    "ts": bar_ts.isoformat() if bar_ts is not None else "",
                    "event": (
                        "policy_requested_stop"
                        if has_live_recap_observations
                        else "replace_stop"
                    ),
                    "stop_price": requested_stop,
                    "reason": str(decision.reason),
                    "detail": decision.reason_detail,
                }
            )
            # The live recap's persisted stop replacements include exchange
            # rounding, minimum-distance handling, and rejected updates. They
            # are authoritative for the actual protective stop. Recomputed
            # decisions remain an audit trace but must not overwrite it.
            if not has_live_recap_observations:
                current_stop = requested_stop
                current_stop_reason = str(decision.reason)
                state["semantic_stop_reason"] = current_stop_reason
                state["stop_price"] = current_stop
        if observation_source == "live_trade_recap_software_policy_stop_sample":
            px = float(observation_price)
            semantic_exit_reason = (
                str(decision.reason)
                if str(decision.reason)
                in {
                    "trailing_profit",
                    "capital_preservation",
                    "capital_preservation_armed",
                    "trailing_risk_reduction",
                }
                else str(current_stop_reason)
            )
            if semantic_exit_reason in {
                "exchange_valid_giveback_fallback",
                "exchange_valid_stop_fallback",
            }:
                semantic_exit_reason = str(
                    state.get("semantic_stop_reason") or semantic_exit_reason
                )
            out.update(
                {
                    "replay_hit": True,
                    "replay_exit_reason": semantic_exit_reason,
                    "replay_exit_ts": bar_ts.isoformat() if bar_ts is not None else "",
                    "replay_exit_price": px,
                    "replay_exit_price_vs_live_bps": _basis_points(
                        px, live_exit_price, side
                    ),
                    "replay_vs_live_exit_status": (
                        "replayed_software_policy_stop_observation"
                    ),
                    "replay_exit_vs_live_fill_event_bps": _basis_points(
                        px, recap.stop_fill_price, side
                    ),
                    "replay_exit_from_observation_source": observation_source,
                    "events_json": json.dumps(events, default=str),
                    "live_stop_fill_ts": (
                        recap.stop_fill_ts.isoformat()
                        if recap.stop_fill_ts is not None
                        else ""
                    ),
                    "live_stop_fill_price_from_recap": recap.stop_fill_price,
                    "live_stop_reason_from_recap": recap.stop_reason,
                }
            )
            return out

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


def _summarise(
    results: pd.DataFrame,
    *,
    exit_tolerance_bps: float = 50.0,
    exit_time_tolerance_seconds: Optional[float] = None,
    strict_execution_1m: bool = False,
) -> Dict[str, Any]:
    if results.empty:
        return {"rows": 0, "exit_parity_status": "pending"}
    group_columns = [
        column
        for column in ("entry_anchor", "coverage_status")
        if column in results.columns
    ]
    if not group_columns:
        grouped = [{"rows": int(len(results))}]
    else:
        grouped = (
            results.groupby(group_columns, dropna=False)
            .size()
            .reset_index(name="rows")
            .to_dict(orient="records")
        )
    gap = pd.to_numeric(
        results.get(
            "replay_exit_price_vs_live_bps",
            pd.Series(np.nan, index=results.index),
        ),
        errors="coerce",
    )
    replay_hit = pd.Series(
        results.get("replay_hit", False), index=results.index
    ).fillna(False).astype(bool)
    live_reason = (
        results.get("live_exit_reason_detail", pd.Series("", index=results.index))
        .fillna("")
        .astype(str)
        .str.split(":")
        .str[0]
    )
    replay_reason = (
        results.get("replay_exit_reason", pd.Series("", index=results.index))
        .fillna("")
        .astype(str)
        .str.split(":")
        .str[0]
    )
    reason_match = live_reason.eq(replay_reason) & live_reason.ne("")
    row_pass = replay_hit & gap.abs().le(float(exit_tolerance_bps)) & reason_match
    live_exit_ts = pd.to_datetime(
        results.get("exit_time", pd.Series(pd.NaT, index=results.index)),
        utc=True,
        errors="coerce",
    )
    replay_exit_ts = pd.to_datetime(
        results.get("replay_exit_ts", pd.Series(pd.NaT, index=results.index)),
        utc=True,
        errors="coerce",
    )
    exit_time_gap_seconds = (replay_exit_ts - live_exit_ts).dt.total_seconds().abs()
    if exit_time_tolerance_seconds is not None:
        row_pass &= exit_time_gap_seconds.le(float(exit_time_tolerance_seconds))
    if strict_execution_1m:
        bar_sources = results.get(
            "bar_source", pd.Series("", index=results.index)
        ).fillna("").astype(str)
        independent_prices = bar_sources.map(
            lambda value: (
                "execution_1m_cache" in set(value.split("+"))
                and set(value.split("+")).issubset(
                    {
                        "execution_1m_cache",
                        "live_trade_recap_exchange_stop_state",
                    }
                )
            )
        )
        independent = (
            independent_prices
            & results.get("replay_vs_live_exit_status", pd.Series("", index=results.index))
            .fillna("")
            .astype(str)
            .str.startswith("replayed_")
        )
        row_pass &= independent
    identity = [
        column
        for column in ("symbol", "entry_time", "exit_time")
        if column in results.columns
    ]
    if identity:
        trade_pass = row_pass.groupby(
            [results[column] for column in identity], dropna=False
        ).any()
    else:
        trade_pass = pd.Series([bool(row_pass.any())])
    mismatch_trades = int((~trade_pass).sum())
    return {
        "rows": int(len(results)),
        "unique_trades": int(results[["symbol", "entry_time", "exit_time"]].drop_duplicates().shape[0])
        if {"symbol", "entry_time", "exit_time"}.issubset(results.columns)
        else None,
        "coverage_by_anchor": grouped,
        "replay_hits": int(pd.Series(results.get("replay_hit", False)).fillna(False).astype(bool).sum()),
        "mean_replay_exit_vs_live_bps": float(
            gap.mean()
        )
        if "replay_exit_price_vs_live_bps" in results
        else None,
        "max_abs_replay_exit_vs_live_bps": (
            float(gap.abs().max()) if gap.notna().any() else None
        ),
        "exit_tolerance_bps": float(exit_tolerance_bps),
        "exit_time_tolerance_seconds": (
            float(exit_time_tolerance_seconds)
            if exit_time_tolerance_seconds is not None
            else None
        ),
        "mean_abs_exit_time_gap_seconds": float(exit_time_gap_seconds.mean())
        if exit_time_gap_seconds.notna().any()
        else None,
        "max_abs_exit_time_gap_seconds": float(exit_time_gap_seconds.max())
        if exit_time_gap_seconds.notna().any()
        else None,
        "reason_match_rows": int(reason_match.sum()),
        "exit_parity_mismatch_trades": mismatch_trades,
        "exit_parity_status": "pass" if mismatch_trades == 0 else "fail",
        "strict_execution_1m": bool(strict_execution_1m),
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

    closed = _select_closed_trade_rows(
        pd.read_csv(closed_path),
        symbols=str(args.symbols or ""),
        limit=args.limit,
        since=args.since,
    )

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
            strict_execution_1m_only=bool(args.ignore_logged_exit_events),
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
                ignore_logged_exit_events=bool(args.ignore_logged_exit_events),
            )
            out = dict(base)
            out.update(bar_meta)
            out.update(replay)
            results.append(out)

    result_df = pd.DataFrame(results)
    result_path = out_dir / "live_closed_trade_exit_replay.csv"
    summary_path = out_dir / "live_closed_trade_exit_replay_summary.json"
    result_df.to_csv(result_path, index=False)
    summary = _summarise(
        result_df,
        exit_tolerance_bps=float(args.exit_tolerance_bps),
        exit_time_tolerance_seconds=float(args.exit_time_tolerance_seconds),
        strict_execution_1m=bool(args.ignore_logged_exit_events),
    )
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
    parser.add_argument("--since", default=None)
    parser.add_argument("--exit-tolerance-bps", type=float, default=50.0)
    parser.add_argument("--exit-time-tolerance-seconds", type=float, default=90.0)
    parser.add_argument(
        "--ignore-logged-exit-events",
        action="store_true",
        help=(
            "Recompute exits from causal price observations and frozen policy state "
            "without accepting the recorded live exit as replay evidence."
        ),
    )
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
