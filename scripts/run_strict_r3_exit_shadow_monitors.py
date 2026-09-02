#!/usr/bin/env python3
"""Run the read-only Strict-R3 one-minute exit shadows and VWAP ladder.

The process has no order-creation, amendment, cancellation or private
exchange-write path.  It observes the active live-state ledger, reads fresh
public one-minute bars and one public full-depth book per open position, then
records hypothetical exits in independent arm state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.data_fetcher import make_exchange
from extreme_price_movements.inference.run_inference import _load_live_policy_bars
from extreme_price_movements.inference.strict_r3_exit_shadow import (
    ARM_NAMES,
    SCHEMA,
    _shadow_vwap_pnl,
    _update_close_state,
    bootstrap_shadow_position,
    directional_exit_vwap_size_ladder,
    evaluate_shadow_bar,
)
from extreme_price_movements.inference.strict_r3_live_execution import (
    StrictR3ExecutionContract,
    _is_rich_policy,
    _policy_payload,
    _rich_policy_params,
    atomic_json,
    utc,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON mapping")
    return dict(value)


def _latest_completed_boundary(now: pd.Timestamp) -> pd.Timestamp:
    return utc(now).floor("min")


def _normalise_bars(
    bars: object, *, start: pd.Timestamp, end: pd.Timestamp,
) -> pd.DataFrame:
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        return pd.DataFrame()
    required = {"close"}
    if not required.issubset(bars.columns) or not isinstance(bars.index, pd.DatetimeIndex):
        return pd.DataFrame()
    result = bars.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index, utc=True))
    return result.loc[(result.index >= start) & (result.index < end)].sort_index()


def _new_shadow_state(
    *, execution_bundle: Path, live_state: Path, started_at: pd.Timestamp,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": "read_only_shadow",
        "execution_bundle": str(execution_bundle),
        "execution_bundle_sha256": _sha256(execution_bundle),
        "live_state": str(live_state),
        "started_at": utc(started_at).isoformat(),
        "arms": {name: {} for name in ARM_NAMES},
        "vwap_ladder_history": {},
    }


def _seed_if_needed(
    shadow: dict[str, Any], *, position: Mapping[str, Any], boundary: pd.Timestamp,
    bootstraps: list[dict[str, Any]],
) -> None:
    candidate_id = str(position["candidate_id"])
    for arm in ARM_NAMES:
        arm_positions = shadow["arms"].setdefault(arm, {})
        if candidate_id in arm_positions:
            continue
        arm_positions[candidate_id] = bootstrap_shadow_position(
            position, start_boundary=boundary,
        )
        bootstraps.append({
            "arm": arm,
            "candidate_id": candidate_id,
            "symbol": str(position["symbol"]),
            "effective_from": boundary.isoformat(),
            "reason": "new_shadow_position_starts_without_historical_book_backfill",
        })


def _shadow_pnl_summary(shadow: Mapping[str, Any]) -> dict[str, dict[str, float | int]]:
    """Aggregate only completed shadow exits; open positions remain marked."""
    summary: dict[str, dict[str, float | int]] = {}
    for arm in ARM_NAMES:
        values: list[Mapping[str, Any]] = []
        for position in shadow.get("arms", {}).get(arm, {}).values():
            if not isinstance(position, Mapping):
                continue
            exit_row = position.get("shadow_exit")
            pnl = exit_row.get("shadow_pnl") if isinstance(exit_row, Mapping) else None
            if isinstance(pnl, Mapping):
                values.append(pnl)
        gross_quote = sum(float(value["gross_pnl_quote"]) for value in values)
        net_quote = sum(float(value["policy_net_pnl_quote"]) for value in values)
        summary[arm] = {
            "closed_shadow_positions": len(values),
            "gross_pnl_quote": gross_quote,
            "policy_net_pnl_quote": net_quote,
            "mean_gross_pnl_bps": (
                sum(float(value["gross_pnl_bps"]) for value in values) / len(values)
                if values else 0.0
            ),
            "mean_policy_net_pnl_bps": (
                sum(float(value["policy_net_pnl_bps"]) for value in values) / len(values)
                if values else 0.0
            ),
        }
    return summary


def run_once(
    *, exchange: Any, contract: StrictR3ExecutionContract, live_state_path: Path,
    shadow_state_path: Path, execution_bundle: Path, out_root: Path, now: object,
) -> dict[str, Any]:
    """Run one no-order observation pass and persist an immutable receipt."""
    timestamp = utc(now)
    boundary = _latest_completed_boundary(timestamp)
    live_state = _load_json(live_state_path)
    if shadow_state_path.exists():
        shadow = _load_json(shadow_state_path)
        if shadow.get("schema") != SCHEMA:
            raise ValueError("shadow state schema mismatch")
        if str(shadow.get("execution_bundle_sha256")) != _sha256(execution_bundle):
            raise ValueError("shadow state execution bundle identity mismatch")
    else:
        shadow = _new_shadow_state(
            execution_bundle=execution_bundle,
            live_state=live_state_path,
            started_at=timestamp,
        )
    policy_payload = _policy_payload(contract.exit_policy)
    if not _is_rich_policy(policy_payload):
        raise ValueError("VWAP shadow arms currently require the frozen rich exit policy")
    params, median_atr_fraction = _rich_policy_params(policy_payload)
    source_positions = {
        str(row["candidate_id"]): dict(row)
        for row in live_state.get("positions", [])
        if isinstance(row, Mapping)
    }
    pnl_backfills: list[dict[str, str]] = []
    for arm in ARM_NAMES:
        for candidate_id, shadow_position in shadow["arms"].get(arm, {}).items():
            exit_row = shadow_position.get("shadow_exit")
            if (
                not isinstance(exit_row, Mapping)
                or isinstance(exit_row.get("shadow_pnl"), Mapping)
            ):
                continue
            exit_vwap = pd.to_numeric(exit_row.get("directional_exit_vwap"), errors="coerce")
            if not pd.notna(exit_vwap) or float(exit_vwap) <= 0.0:
                continue
            mutable_exit = dict(exit_row)
            mutable_exit["shadow_pnl"] = _shadow_vwap_pnl(
                shadow_position,
                exit_vwap=float(exit_vwap),
                policy_cost_bps=contract.baseline_policy_cost_bps,
            )
            shadow_position["shadow_exit"] = mutable_exit
            pnl_backfills.append({"arm": arm, "candidate_id": str(candidate_id)})
    bootstraps: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    vwap_ladders: list[dict[str, Any]] = []
    for candidate_id, source_position in source_positions.items():
        _seed_if_needed(
            shadow, position=source_position, boundary=boundary, bootstraps=bootstraps,
        )
        symbol = str(source_position["exchange_symbol"])
        # A single public book is deliberately shared across all arms and the
        # 1×/5×/10×/50×/100× telemetry ladder.  It gives every alternative
        # identical contemporaneous executable evidence and cannot affect the
        # actual position monitor.
        needs_bar_processing = any(
            not bool(shadow["arms"][arm][candidate_id].get("closed", False))
            and utc(shadow["arms"][arm][candidate_id]["next_bar_ts"]) < boundary
            for arm in ARM_NAMES
        )
        order_book: Mapping[str, Any] | None = None
        book_error: str | None = None
        # The ladder is deliberately refreshed for every open *live* position,
        # even if a shadow arm has already hypothetically exited.  It is a
        # liquidity observation, not an arm-specific exit decision.
        try:
            value = exchange.fetch_order_book(symbol)
            if not isinstance(value, Mapping):
                raise ValueError("exchange did not return an order-book mapping")
            order_book = value
        except Exception as exc:
            book_error = str(exc)
        if order_book is not None:
            ladder = directional_exit_vwap_size_ladder(
                order_book,
                side=str(source_position.get("side") or "long"),
                amount=float(source_position["amount"]),
                contract_size=float(source_position.get("contract_size", 1.0)),
                effective_leverage=source_position.get(
                    "effective_leverage", source_position.get("leverage"),
                ),
            )
            ladder_record = {
                "candidate_id": candidate_id,
                "symbol": str(source_position["symbol"]),
                "bar_end": boundary.isoformat(),
                "book_observed_at": timestamp.isoformat(),
                "ladder": ladder,
            }
            histories = shadow.setdefault("vwap_ladder_history", {})
            history = histories.setdefault(candidate_id, [])
            if not history or str(history[-1].get("bar_end")) != boundary.isoformat():
                history.append(ladder_record)
            vwap_ladders.append(ladder_record)
        else:
            vwap_ladders.append({
                "candidate_id": candidate_id,
                "symbol": str(source_position["symbol"]),
                "bar_end": boundary.isoformat(),
                "book_observed_at": timestamp.isoformat(),
                "status": "fresh_book_unavailable",
                "error": book_error,
            })
        bars_by_start: dict[str, pd.DataFrame] = {}
        for arm in ARM_NAMES:
            shadow_position = shadow["arms"][arm][candidate_id]
            if bool(shadow_position.get("closed", False)):
                continue
            start = utc(shadow_position["next_bar_ts"])
            if start >= boundary:
                actions.append({
                    "arm": arm, "action": "no_new_completed_bar",
                    "candidate_id": candidate_id, "next_bar_ts": start.isoformat(),
                })
                continue
            start_key = start.isoformat()
            if start_key not in bars_by_start:
                bars_by_start[start_key] = _normalise_bars(
                    _load_live_policy_bars(
                        cfg={"execution_account": "perps", "exchange": "krakenfutures"},
                        exchange=exchange, symbol=symbol, timeframe="1m", start=start, end=boundary,
                    ),
                    start=start, end=boundary,
                )
            # Copy because each arm has independent state, not independent
            # source bars.  This guarantees an apples-to-apples comparison.
            bars = bars_by_start[start_key].copy()
            if bars.empty:
                actions.append({
                    "arm": arm, "action": "missing_completed_bar",
                    "candidate_id": candidate_id, "from": start.isoformat(),
                    "to": boundary.isoformat(),
                })
                continue
            # A lagged process has no historical book snapshots.  Advance its
            # close-only threshold state through older completed bars but
            # explicitly do not fabricate a VWAP proposal for them.
            for stamp, row in bars.iloc[:-1].iterrows():
                _update_close_state(
                    shadow_position,
                    close=float(row["close"]),
                    bar_end=utc(stamp) + pd.Timedelta(minutes=1),
                    params=params, median_atr_fraction=median_atr_fraction,
                )
                actions.append({
                    "arm": arm, "action": "historical_bar_state_only",
                    "candidate_id": candidate_id,
                    "bar_end": (utc(stamp) + pd.Timedelta(minutes=1)).isoformat(),
                    "reason": "no_point_in_time_book_snapshot_available",
                })
            stamp, row = next(reversed(list(bars.iloc[-1:].iterrows())))
            if order_book is None:
                actions.append({
                    "arm": arm, "action": "fresh_book_unavailable",
                    "candidate_id": candidate_id,
                    "bar_end": (utc(stamp) + pd.Timedelta(minutes=1)).isoformat(),
                    "error": book_error,
                })
                continue
            next_position, proposals = evaluate_shadow_bar(
                shadow_position,
                close=float(row["close"]),
                bar_end=utc(stamp) + pd.Timedelta(minutes=1),
                order_book=order_book, params=params,
                median_atr_fraction=median_atr_fraction,
                policy_cost_bps=contract.baseline_policy_cost_bps,
                book_observed_at=timestamp,
            )
            proposal = dict(proposals[arm])
            proposal.update({"arm": arm, "candidate_id": candidate_id, "symbol": symbol})
            if bool(proposal["would_exit"]):
                next_position["closed"] = True
                next_position["shadow_exit"] = dict(proposal)
                proposal["action"] = "would_exit"
            else:
                proposal["action"] = "continue"
            shadow["arms"][arm][candidate_id] = next_position
            actions.append(proposal)
    # Do not erase closed positions: they remain explicit shadow outcomes.
    shadow["last_observed_at"] = timestamp.isoformat()
    shadow["last_live_state_sha256"] = _sha256(live_state_path)
    shadow["last_live_state_as_of_ts"] = live_state.get("as_of_ts")
    shadow["pnl_summary"] = _shadow_pnl_summary(shadow)
    atomic_json(shadow_state_path, shadow)
    result = {
        "schema": SCHEMA,
        "mode": "read_only_shadow",
        "status": "success",
        "observed_at": timestamp.isoformat(),
        "completed_boundary": boundary.isoformat(),
        "live_state": str(live_state_path),
        "live_state_sha256": _sha256(live_state_path),
        "execution_bundle": str(execution_bundle),
        "execution_bundle_sha256": _sha256(execution_bundle),
        "exit_policy_sha256": contract.exit_policy_sha256,
        "source_positions": len(source_positions),
        "bootstraps": bootstraps,
        "shadow_exit_pnl_backfills": pnl_backfills,
        "pnl_summary": shadow["pnl_summary"],
        "directional_vwap_size_ladders": vwap_ladders,
        "actions": actions,
        "exchange_write_calls": 0,
    }
    run_dir = out_root / f"shadow_{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}"
    if run_dir.exists():
        raise FileExistsError(f"immutable shadow receipt exists: {run_dir}")
    run_dir.mkdir(parents=True)
    atomic_json(run_dir / "run_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-bundle", type=Path, required=True)
    parser.add_argument("--live-state", type=Path, required=True)
    parser.add_argument("--shadow-state", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--now", default=None)
    parser.add_argument("--interval-seconds", type=float, default=None)
    args = parser.parse_args()
    contract = StrictR3ExecutionContract.load(args.execution_bundle, root=ROOT)
    exchange = make_exchange("perps")
    if str(getattr(exchange, "id", "")) != "krakenfutures":
        raise ValueError("exit shadows require Kraken Futures")
    interval = None if args.interval_seconds is None else max(1.0, float(args.interval_seconds))
    next_run = time.monotonic()
    while True:
        now = pd.Timestamp(args.now) if args.now else pd.Timestamp.now(tz="UTC")
        now = utc(now)
        try:
            result = run_once(
                exchange=exchange, contract=contract,
                live_state_path=args.live_state,
                shadow_state_path=args.shadow_state,
                execution_bundle=args.execution_bundle,
                out_root=args.out_root,
                now=now,
            )
        except Exception as exc:
            result = {
                "schema": SCHEMA,
                "mode": "read_only_shadow",
                "status": "failed_closed",
                "observed_at": now.isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "exchange_write_calls": 0,
            }
            run_dir = args.out_root / f"shadow_{now.strftime('%Y%m%dT%H%M%S%fZ')}"
            run_dir.mkdir(parents=True, exist_ok=False)
            atomic_json(run_dir / "run_manifest.json", result)
        print(json.dumps(result, default=str), flush=True)
        if interval is None or args.now:
            if result.get("status") != "success":
                raise RuntimeError(str(result.get("error")))
            return
        next_run += interval
        time.sleep(max(0.0, next_run - time.monotonic()))


if __name__ == "__main__":
    main()
