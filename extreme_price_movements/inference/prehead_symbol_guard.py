"""Live application of pre-head symbol guard artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from extreme_price_movements.inference.dynamic_hr_surprise_threshold import infer_head
from extreme_price_movements.inference.symbol_mapping import normalise_symbol, symbol_base


@dataclass(frozen=True)
class PreheadSymbolGuardResult:
    blocked: bool
    reason: str = ""
    state_age_days: float = np.nan
    guard_policy: str = ""
    head: str = ""
    side: str = ""


def _parse_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if value in (None, ""):
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _state_age_days(as_of: Any, now: Any | None) -> float:
    ts = _parse_timestamp(as_of)
    if ts is None:
        return float("inf")
    now_ts = _parse_timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if now_ts is None:
        now_ts = pd.Timestamp.now(tz="UTC")
    return float(max((now_ts - ts).total_seconds(), 0.0) / 86400.0)


def _strategy_side(strategy_id: Any, side: Any = None) -> str:
    if side:
        side_s = str(side).strip().lower()
        if side_s in {"long", "short"}:
            return side_s
    sid = str(strategy_id or "").strip().lower()
    if sid.startswith("long_"):
        return "long"
    if sid.startswith("short_"):
        return "short"
    return ""


def load_prehead_symbol_guard_state(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _blocked_symbols_for(
    state: Mapping[str, Any],
    *,
    head: str,
    side: str,
) -> set[str]:
    blocked = state.get("blocked")
    if not isinstance(blocked, Mapping):
        return set()
    head_payload = blocked.get(head)
    if not isinstance(head_payload, Mapping):
        return set()
    raw_symbols = head_payload.get(side) or head_payload.get("*") or []
    if not isinstance(raw_symbols, list):
        return set()
    out: set[str] = set()
    for raw in raw_symbols:
        norm = normalise_symbol(str(raw))
        if not norm:
            continue
        out.add(norm)
        base = symbol_base(norm)
        if base:
            out.add(base)
    return out


def prehead_symbol_guard_result(
    *,
    symbol: str,
    strategy_id: str,
    side: str,
    state: Mapping[str, Any] | None,
    enabled: bool,
    now: Any | None,
    max_state_age_days: float | None = None,
) -> PreheadSymbolGuardResult:
    head = infer_head(strategy_id)
    resolved_side = _strategy_side(strategy_id, side)
    if not enabled:
        return PreheadSymbolGuardResult(False, "disabled", head=head, side=resolved_side)
    if not state:
        return PreheadSymbolGuardResult(False, "missing_state", head=head, side=resolved_side)
    age_days = _state_age_days(state.get("as_of"), now)
    if max_state_age_days is not None and age_days > float(max_state_age_days):
        return PreheadSymbolGuardResult(
            False,
            "stale_state",
            state_age_days=age_days,
            guard_policy=str(state.get("policy_name") or ""),
            head=head,
            side=resolved_side,
        )
    if head == "unknown" or not resolved_side:
        return PreheadSymbolGuardResult(
            False,
            "unknown_head_or_side",
            state_age_days=age_days,
            guard_policy=str(state.get("policy_name") or ""),
            head=head,
            side=resolved_side,
        )
    blocked = _blocked_symbols_for(state, head=head, side=resolved_side)
    if not blocked:
        return PreheadSymbolGuardResult(
            False,
            "no_blocked_symbols_for_head_side",
            state_age_days=age_days,
            guard_policy=str(state.get("policy_name") or ""),
            head=head,
            side=resolved_side,
        )
    norm = normalise_symbol(str(symbol))
    base = symbol_base(norm)
    is_blocked = norm in blocked or bool(base and base in blocked)
    return PreheadSymbolGuardResult(
        bool(is_blocked),
        "prehead_symbol_guard_block" if is_blocked else "not_blocked",
        state_age_days=age_days,
        guard_policy=str(state.get("policy_name") or ""),
        head=head,
        side=resolved_side,
    )
