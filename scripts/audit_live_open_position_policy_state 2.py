#!/usr/bin/env python3
"""Audit current live position state against the promoted 1-minute policy.

The completed-bar monitor and the executable-price sentinel are independent
runtime paths.  Their latest snapshots must agree on the state that drives the
protective order, while every active strategy must resolve to a hash-validated
simple_policy_optimiser artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from extreme_price_movements.inference.simple_policy_stop import (
    WINNER_POLICY_PATHWAY_ID,
    WINNER_REPLAY_TIMEFRAME,
    load_simple_policy_stop_params_by_strategy,
    validate_simple_policy_stop_params,
)


def _latest_payload(path: Path, marker: str) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    if not path.is_file():
        return latest
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            token = f"{marker} "
            if token not in line:
                continue
            try:
                candidate = json.loads(line.split(token, 1)[1])
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(candidate, dict):
                latest = candidate
    return latest


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _close(left: Any, right: Any, *, atol: float = 1e-10) -> bool:
    a, b = _finite(left), _finite(right)
    if not np.isfinite(a) or not np.isfinite(b):
        return False
    return bool(np.isclose(a, b, rtol=1e-9, atol=atol))


def _latest_strategy_by_symbol(trades_path: Path) -> dict[str, str]:
    if not trades_path.is_file():
        return {}
    frame = pd.read_csv(
        trades_path,
        usecols=lambda name: name in {
            "timestamp", "decision_ts", "symbol", "strategy_id", "lifecycle_event"
        },
        low_memory=False,
    )
    if frame.empty:
        return {}
    event = frame.get("lifecycle_event", pd.Series("", index=frame.index)).astype(str)
    frame = frame.loc[event.str.startswith("entry", na=False)].copy()
    if frame.empty:
        return {}
    decision_ts = pd.to_datetime(frame.get("decision_ts"), utc=True, errors="coerce")
    display_ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce")
    frame["_sort_ts"] = decision_ts.fillna(display_ts)
    frame = frame.sort_values("_sort_ts").drop_duplicates("symbol", keep="last")
    return dict(zip(frame["symbol"].astype(str), frame["strategy_id"].astype(str)))


def audit(
    *,
    inference_log: Path,
    trade_log: Path,
    data_root: str,
    run_id: str,
    max_snapshot_age_seconds: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    monitor = _latest_payload(inference_log, "INFERENCE_MONITOR_HEARTBEAT")
    sentinel = _latest_payload(inference_log, "EXECUTABLE_STOP_SENTINEL_HEARTBEAT")
    monitor_statuses = monitor.get("statuses") if isinstance(monitor, Mapping) else {}
    sentinel_statuses = sentinel.get("statuses") if isinstance(sentinel, Mapping) else {}
    monitor_statuses = monitor_statuses if isinstance(monitor_statuses, Mapping) else {}
    sentinel_statuses = sentinel_statuses if isinstance(sentinel_statuses, Mapping) else {}
    active_symbols = set(map(str, monitor.get("symbols") or []))
    sentinel_symbols = set(map(str, sentinel.get("symbols") or []))
    strategies = _latest_strategy_by_symbol(trade_log)
    params_by_strategy = load_simple_policy_stop_params_by_strategy(data_root, run_id)

    monitor_ts = pd.to_datetime(monitor.get("timestamp"), utc=True, errors="coerce")
    sentinel_ts = pd.to_datetime(sentinel.get("timestamp"), utc=True, errors="coerce")
    age_seconds = (
        abs((monitor_ts - sentinel_ts).total_seconds())
        if pd.notna(monitor_ts) and pd.notna(sentinel_ts)
        else np.inf
    )

    rows: list[dict[str, Any]] = []
    for symbol in sorted(active_symbols | sentinel_symbols):
        monitor_row = monitor_statuses.get(symbol, {})
        monitor_row = monitor_row if isinstance(monitor_row, Mapping) else {}
        price_action = monitor_row.get("price_action", {})
        price_action = price_action if isinstance(price_action, Mapping) else {}
        sentinel_row = sentinel_statuses.get(symbol, {})
        sentinel_row = sentinel_row if isinstance(sentinel_row, Mapping) else {}
        source_strategy_id = strategies.get(symbol, "")
        side = str(sentinel_row.get("side") or "").lower()
        strategy_candidates = [source_strategy_id]
        if source_strategy_id and side in {"long", "short"}:
            strategy_candidates.append(f"{side}_{source_strategy_id}")
        if side in {"long", "short"}:
            strategy_candidates.append(f"{side}__parent")
        strategy_id = next(
            (candidate for candidate in strategy_candidates if candidate in params_by_strategy),
            source_strategy_id,
        )
        params = params_by_strategy.get(strategy_id)
        artifact_valid = False
        artifact_error = ""
        if not strategy_id:
            artifact_error = "missing_active_strategy_id"
        elif params is None:
            artifact_error = "strategy_missing_from_policy_artifact"
        else:
            try:
                validated = validate_simple_policy_stop_params(
                    params,
                    state={"strategy_id": strategy_id},
                    require_barrier=False,
                )
                artifact_valid = bool(
                    validated.policy_pathway_id == WINNER_POLICY_PATHWAY_ID
                    and validated.replay_timeframe == WINNER_REPLAY_TIMEFRAME
                )
                if not artifact_valid:
                    artifact_error = "policy_pathway_or_timeframe_mismatch"
            except Exception as exc:  # fail closed and preserve the exact cause
                artifact_error = f"{type(exc).__name__}: {exc}"

        policy_stop = price_action.get("stop_price_after")
        sentinel_policy_stop = sentinel_row.get("policy_executable_stop_price")
        monitor_mfe, sentinel_mfe = price_action.get("mfe"), sentinel_row.get("mfe")
        monitor_peak, sentinel_peak = price_action.get("peak_price"), sentinel_row.get("peak_price")
        coverage_match = symbol in active_symbols and symbol in sentinel_symbols
        state_match = bool(
            coverage_match
            and _close(policy_stop, sentinel_policy_stop)
            and _close(monitor_mfe, sentinel_mfe)
            and _close(monitor_peak, sentinel_peak)
        )
        exchange_protected = bool(sentinel_row.get("exchange_trigger_is_more_protective", False))
        timeframe_match = str(price_action.get("policy_timeframe") or "").lower() == "1m"
        row_pass = bool(
            artifact_valid
            and state_match
            and exchange_protected
            and timeframe_match
            and age_seconds <= float(max_snapshot_age_seconds)
        )
        rows.append(
            {
                "symbol": symbol,
                "side": side,
                "source_strategy_id": source_strategy_id,
                "strategy_id": strategy_id,
                "artifact_valid": artifact_valid,
                "artifact_error": artifact_error,
                "policy_timeframe": price_action.get("policy_timeframe"),
                "policy_stop": _finite(policy_stop),
                "sentinel_policy_stop": _finite(sentinel_policy_stop),
                "exchange_stop": _finite(sentinel_row.get("exchange_trigger_stop_price")),
                "monitor_mfe": _finite(monitor_mfe),
                "sentinel_mfe": _finite(sentinel_mfe),
                "monitor_peak": _finite(monitor_peak),
                "sentinel_peak": _finite(sentinel_peak),
                "coverage_match": coverage_match,
                "state_match": state_match,
                "exchange_protected": exchange_protected,
                "timeframe_match": timeframe_match,
                "snapshot_age_seconds": float(age_seconds),
                "row_pass": row_pass,
            }
        )

    result = pd.DataFrame(rows)
    if not active_symbols and not sentinel_symbols:
        status, reason = "pending", "no_open_positions"
    elif result.empty:
        status, reason = "fail", "active_position_snapshot_missing"
    elif bool(result["row_pass"].all()):
        status, reason = "pass", "open_position_policy_state_exact"
    else:
        status, reason = "fail", "open_position_policy_state_mismatch"
    summary = {
        "status": status,
        "reason": reason,
        "run_id": run_id,
        "monitor_timestamp": str(monitor.get("timestamp") or ""),
        "sentinel_timestamp": str(sentinel.get("timestamp") or ""),
        "snapshot_age_seconds": float(age_seconds),
        "active_positions": int(len(active_symbols)),
        "sentinel_positions": int(len(sentinel_symbols)),
        "rows": int(len(result)),
        "passing_rows": int(result.get("row_pass", pd.Series(dtype=bool)).sum()),
        "mismatch_symbols": result.loc[~result["row_pass"], "symbol"].tolist()
        if not result.empty
        else [],
        "required_policy_pathway_id": WINNER_POLICY_PATHWAY_ID,
        "required_replay_timeframe": WINNER_REPLAY_TIMEFRAME,
    }
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-log", required=True)
    parser.add_argument("--trade-log", default="inference_trades.csv")
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-snapshot-age-seconds", type=float, default=180.0)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result, summary = audit(
        inference_log=Path(args.inference_log),
        trade_log=Path(args.trade_log),
        data_root=args.data_root,
        run_id=args.run_id,
        max_snapshot_age_seconds=args.max_snapshot_age_seconds,
    )
    result.to_csv(output_dir / "open_position_policy_state.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    raise SystemExit(1 if summary["status"] == "fail" else 0)


if __name__ == "__main__":
    main()
