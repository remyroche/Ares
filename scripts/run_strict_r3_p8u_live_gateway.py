#!/usr/bin/env python3
"""Submit *fresh* P8U portfolio proposals through the proven Kraken executor.

This is intentionally a narrow bridge.  The P8U scorer and its portfolio
adapter remain target-free / no-order components.  This process accepts only
one immutable staged-score commit, verifies the frozen Router50 → F72 → Under
→ dual-MC1 contract, reconstructs a portfolio state from confirmed live
positions, then (and only with ``--submit-orders``) invokes the existing
Kraken Futures entry primitive which immediately installs a reduce-only stop.

The script has no feature or model-training code, cannot score a stale
decision, and never turns an unfilled proposal into a live position.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.data_fetcher import make_exchange
from extreme_price_movements.inference.p8u_execution_portfolio_adapter import (
    P8UExecutionContract,
    P8UPortfolioState,
    prepare_execution_intent,
)
from extreme_price_movements.inference.strict_r3_live_execution import (
    DUAL_BCF_CURRENT_PORTFOLIO_SCHEMA,
    StrictR3ExecutionContract,
    _entry_blueprint_from_decision,
    _fetch_exchange_positions_for_monitor,
    _kraken_flex_margin_snapshot,
    _open_live_position,
    atomic_json,
    initial_state,
    live_state_lock,
    load_state,
    sha256_file,
    utc,
)
from scripts.materialize_strict_r3_target_free_hourly_grid_v2 import _signal_atr_panel
from scripts.run_tp6_sl4_exact170_canonical_consensus import _read_downloaded_15m_decision_open


SCHEMA = "strict_r3_p8u_kraken_live_gateway_v1"
STATE_SCHEMA = "strict_r3_kraken_live_state_v1"


def _sha256(path: Path) -> str:
    return sha256_file(Path(path))


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _root_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _validate_hash_bound_file(descriptor: Mapping[str, Any], *, role: str) -> Path:
    path_value, expected = descriptor.get("path"), descriptor.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise ValueError(f"{role} lacks hash-bound path")
    path = _root_path(path_value).resolve()
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"{role} hash mismatch")
    return path


def _load_gateway_contract(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unknown P8U live gateway schema")
    if payload.get("side") != "long" or payload.get("exchange_id") != "krakenfutures":
        raise ValueError("gateway is restricted to Kraken Futures long-only")
    if payload.get("order_submission") is not True:
        raise ValueError("live gateway requires an explicit order_submission=true contract")
    if int(payload.get("maximum_decision_age_seconds", -1)) != 900:
        raise ValueError("gateway must retain the frozen 15-minute signal-age guard")
    adapter_path = _validate_hash_bound_file(dict(payload.get("no_order_adapter") or {}), role="no-order adapter")
    bundle_path = _validate_hash_bound_file(dict(payload.get("bundle") or {}), role="P8U bundle")
    policy_path = _validate_hash_bound_file(dict(payload.get("policy") or {}), role="rich policy")
    # The activation receipt binds the SHA-256 of *this* gateway config.  A
    # reciprocal config→activation hash would be circular, so the config
    # carries only its repository-relative activation path; any edit to that
    # path changes this config's hash and invalidates the activation receipt.
    activation_value = payload.get("activation_path")
    if not isinstance(activation_value, str) or not activation_value:
        raise ValueError("gateway lacks activation_path")
    activation_path = _root_path(activation_value).resolve()
    if not activation_path.is_file():
        raise FileNotFoundError("P8U activation authorization is absent")
    runtime = dict(payload.get("gateway_runtime") or {})
    current = Path(__file__).resolve()
    if runtime.get("path") != str(current.relative_to(ROOT)) or runtime.get("sha256") != _sha256(current):
        raise ValueError("gateway runtime is not sealed by its activation contract")
    activation = json.loads(activation_path.read_text(encoding="utf-8"))
    if (
        activation.get("schema") != "strict_r3_p8u_kraken_live_activation_v1"
        or activation.get("authorized") is not True
        or activation.get("manual_user_authorization") is not True
        or activation.get("stale_signal_execution_prohibited") is not True
        or activation.get("bundle_sha256") != payload["bundle"]["sha256"]
        or activation.get("gateway_contract_sha256") != _sha256(path)
        or activation.get("no_order_adapter_sha256") != _sha256(adapter_path)
        or activation.get("policy_sha256") != _sha256(policy_path)
    ):
        raise ValueError("P8U activation authorization does not bind this exact gateway")
    expiry = utc(activation["expires_at"])
    if pd.Timestamp.now(tz="UTC") >= expiry:
        raise ValueError("P8U activation has expired; a fresh causal MC1 vintage is required")
    adapter = P8UExecutionContract.load(adapter_path, workspace_root=ROOT)
    if str(adapter.payload["bundle"]["sha256"]) != str(payload["bundle"]["sha256"]):
        raise ValueError("gateway and no-order adapter disagree on P8U bundle")
    if str(adapter.payload["policy"]["sha256"]) != str(payload["policy"]["sha256"]):
        raise ValueError("gateway and no-order adapter disagree on rich policy")
    # The P8U preproduction bundle itself must remain incapable of order I/O.
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("runtime", {}).get("order_submission") is not False:
        raise ValueError("P8U model bundle must remain a no-order artifact")
    payload["__gateway_path__"] = str(path)
    payload["__gateway_sha256__"] = _sha256(path)
    payload["__adapter_path__"] = str(adapter_path)
    payload["__policy_path__"] = str(policy_path)
    payload["__activation_path__"] = str(activation_path)
    payload["__activation_sha256__"] = _sha256(activation_path)
    payload["__authorized_after__"] = utc(activation["authorized_after"]).isoformat()
    return payload, _sha256(path)


def _generic_execution_contract(contract: Mapping[str, Any]) -> StrictR3ExecutionContract:
    return StrictR3ExecutionContract(
        inference_bundle=_root_path(str(contract["bundle"]["path"])).resolve(),
        inference_bundle_sha256=str(contract["bundle"]["sha256"]),
        exit_policy=Path(str(contract["__policy_path__"])).resolve(),
        exit_policy_sha256=str(contract["policy"]["sha256"]),
        exchange_id="krakenfutures",
        side="long",
        leverage=7.0,
        maximum_decision_age_seconds=int(contract["maximum_decision_age_seconds"]),
        order_submission_authorized=True,
        activation_authorization=Path(str(contract["__activation_path__"])).resolve(),
        activation_authorization_sha256=str(contract["__activation_sha256__"]),
        authorized_after=utc(contract["__authorized_after__"]),
        maximum_entry_slippage_bps=float(contract.get("maximum_entry_slippage_bps", 100.0)),
        maximum_exit_slippage_bps=float(contract.get("maximum_exit_slippage_bps", 100.0)),
        baseline_policy_cost_bps=100.0,
        minimum_execution_adjusted_ev_bps=50.0,
        maximum_live_spread_bps=100.0,
        execution_microstructure_buffer_bps=10.0,
        execution_book_telemetry_only=False,
        execution_adjusted_ev_veto_enabled=True,
        protective_stop_exit_vwap_adjustment=True,
        protective_stop_book_levels=10,
        close_based_hard_stop_monitor_enabled=True,
        openpositions_503_fills_fallback_enabled=True,
    )


def _ensure_live_state(
    *, state_path: Path, decision_ts: pd.Timestamp, contract: Mapping[str, Any]
) -> dict[str, Any]:
    state = load_state(state_path, decision_ts=decision_ts)
    if state.get("schema") != STATE_SCHEMA:
        raise ValueError("P8U live state schema mismatch")
    identity = {
        "inference_bundle_sha256": str(contract["bundle"]["sha256"]),
        "exit_policy_sha256": str(contract["policy"]["sha256"]),
        "activation_authorization_sha256": str(contract["__activation_sha256__"]),
        "p8u_gateway_contract_sha256": str(contract["__gateway_sha256__"]),
        "p8u_no_order_adapter_sha256": _sha256(Path(str(contract["__adapter_path__"]))),
    }
    if not state_path.exists():
        state.update(identity)
        state["p8u_processed_score_receipts"] = []
        state["p8u_live_gateway_schema"] = SCHEMA
    else:
        for key, expected in identity.items():
            if state.get(key) != expected:
                raise ValueError(f"P8U live state identity mismatch: {key}")
        if state.get("p8u_live_gateway_schema") != SCHEMA:
            raise ValueError("existing live state does not belong to P8U gateway")
        if not isinstance(state.get("p8u_processed_score_receipts"), list):
            raise ValueError("P8U live state lacks processed score receipt ledger")
    return state


def _p8u_portfolio_state(
    *, state: Mapping[str, Any], adapter: P8UExecutionContract, wallet: float
) -> P8UPortfolioState:
    position_rows = []
    for row in state.get("positions", []):
        if not isinstance(row, Mapping):
            raise ValueError("P8U live state contains invalid position row")
        position_rows.append({
            "candidate_id": str(row["candidate_id"]),
            "symbol": str(row["symbol"]),
            "initial_margin_fraction": float(adapter.payload["portfolio"]["margin_slot_fraction"]),
        })
    payload = {
        "schema": "strict_r3_p8u_execution_portfolio_state_v1",
        "execution_contract_sha256": adapter.sha256,
        "bundle_sha256": str(adapter.payload["bundle"]["sha256"]),
        "wallet_equity_quote": float(wallet),
        "open_positions": position_rows,
        "pending_intents": [],
        "processed_score_commit_sha256": list(state.get("p8u_processed_score_receipts", [])),
    }
    result = P8UPortfolioState(payload)
    result.validate(adapter)
    return result


def _execution_lineage(
    *, symbol: str, source_ts: pd.Timestamp, decision_ts: pd.Timestamp
) -> tuple[float, float]:
    decision_index = pd.DatetimeIndex([decision_ts])
    opening = _read_downloaded_15m_decision_open(symbol, decision_index)
    decision_open = pd.to_numeric(opening.reindex(decision_index), errors="coerce").iloc[0]
    atr_panel = _signal_atr_panel(
        [symbol], pd.DatetimeIndex([source_ts]),
        policy_bar_root=ROOT / "15m_ohlcv_perp", bar_phase_minutes=0,
    )
    atr = pd.to_numeric(atr_panel.loc[source_ts, symbol], errors="coerce")
    if not np.isfinite(decision_open) or float(decision_open) <= 0.0:
        raise ValueError("no finite exact 15-minute decision open")
    if not np.isfinite(atr) or float(atr) <= 0.0:
        raise ValueError("no finite decision-time Wilder14 ATR")
    return float(decision_open), float(atr)


def _decision_for_proposal(
    *, proposal: Mapping[str, Any], source_ts: pd.Timestamp, policy: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_id = str(proposal["candidate_id"])
    symbol = str(proposal["symbol"])
    decision_ts = utc(proposal["__decision_ts__"])
    decision_open, atr = _execution_lineage(
        symbol=symbol, source_ts=source_ts, decision_ts=decision_ts,
    )
    params = dict(policy["params"])
    decision = dict(proposal)
    decision.update({
        "candidate_id": candidate_id,
        "__decision_ts__": decision_ts,
        "__symbol__": symbol,
        "side_name": "long",
        "decision_open": decision_open,
        "signal_atr": atr,
        "portfolio_policy_schema": DUAL_BCF_CURRENT_PORTFOLIO_SCHEMA,
        "bcf_mc1_expected_net_bps": float(proposal["bcf_mc1_expected_bps"]),
        "policy_parent_name": "SimplePolicyOptimiser rich smooth-capital-protection",
        "policy_timeout_hours": 12,
        "policy_cost_bps_once": 100.0,
        "policy_sl_atr": float(params["sl_mult"]),
        "policy_trailing_activation_atr": float(params["trailing_activation_mult"]),
        "policy_trailing_giveback_atr": float(params["fixed_trailing_gap_mult"]),
    })
    return decision


def _reconcile_account(
    *, exchange: Any, state: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    tracked = {
        str(row["candidate_id"]): dict(row)
        for row in state.get("positions", []) if isinstance(row, Mapping)
    }
    exchange_positions, audit = _fetch_exchange_positions_for_monitor(
        exchange, tracked_positions=tracked, expected_side="long", allow_503_fills_fallback=True,
    )
    tracked_symbols = {str(row["exchange_symbol"]) for row in tracked.values()}
    untracked = sorted(set(exchange_positions).difference(tracked_symbols))
    if untracked:
        raise ValueError(f"untracked Kraken positions block P8U entry: {untracked}")
    return exchange_positions, audit


def execute(
    *, gateway_path: Path, staged_commit: Path, state_path: Path, out_dir: Path,
    submit_orders: bool, now: object | None,
) -> dict[str, Any]:
    contract, gateway_sha = _load_gateway_contract(gateway_path)
    adapter = P8UExecutionContract.load(Path(str(contract["__adapter_path__"])), workspace_root=ROOT)
    policy = json.loads(Path(str(contract["__policy_path__"])).read_text(encoding="utf-8"))
    if str(policy.get("schema")) != "strict_r3_rich_simple_policy_challenger_v1":
        raise ValueError("live gateway requires the sealed rich SimplePolicyOptimiser policy")
    stamp = utc(now or pd.Timestamp.now(tz="UTC"))
    staged_commit = Path(staged_commit).resolve()
    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError("P8U live gateway receipt path must be immutable")
    receipt = json.loads((staged_commit / "receipt.json").read_text(encoding="utf-8"))
    decision_ts = utc(receipt["decision_timestamp"])
    source_ts = utc(receipt["source_timestamp"])
    # The score commit is already point-in-time and the entry blueprint is
    # rechecked against a live executable book below.  A fixed post-close wait
    # makes that decision staler without improving causality.  Freshness is
    # instead enforced by the sealed maximum-decision-age guard in the
    # execution contract and by the per-proposal live price/impact checks.
    execution_contract = _generic_execution_contract(contract)
    outcomes: list[dict[str, Any]] = []
    with live_state_lock(state_path):
        state = _ensure_live_state(state_path=state_path, decision_ts=decision_ts, contract=contract)
        if submit_orders:
            exchange = make_exchange("perps")
            _, account_audit = _reconcile_account(exchange=exchange, state=state)
            wallet = float(_kraken_flex_margin_snapshot(exchange)["margin_equity"])
        else:
            exchange = None
            account_audit = {"status": "not_called_dry_run"}
            wallet = float(contract.get("dry_run_wallet_equity_quote", 1000.0))
        portfolio_state = _p8u_portfolio_state(state=state, adapter=adapter, wallet=wallet)
        auction, audit, _ = prepare_execution_intent(
            contract=adapter, state=portfolio_state, staged_commit=staged_commit, now=stamp,
        )
        proposed = auction.loc[auction["execution_action"].eq("propose")].copy()
        out_dir.mkdir(parents=True, exist_ok=False)
        auction.to_parquet(out_dir / "auction.parquet", index=False, compression="zstd")
        for rank, (_, proposal) in enumerate(proposed.iterrows(), start=1):
            row = proposal.to_dict()
            row["portfolio_priority_rank"] = rank
            try:
                decision = _decision_for_proposal(
                    proposal=row, source_ts=source_ts, policy=policy,
                )
                shadow = _entry_blueprint_from_decision(decision)
                if submit_orders:
                    assert exchange is not None
                    position, execution = _open_live_position(
                        exchange,
                        decision=decision,
                        shadow_position=shadow,
                        contract=execution_contract,
                        slot_margin=float(adapter.payload["portfolio"]["margin_slot_fraction"]),
                        wallet_equity=wallet,
                        observed_at=stamp,
                    )
                    state["positions"].append(position)
                    outcomes.append({
                        "candidate_id": decision["candidate_id"], "symbol": decision["__symbol__"],
                        "status": "opened_live", "entry_order_id": position.get("entry_order_id"),
                        "stop_order_id": position.get("stop_order_id"),
                        "entry_fill_ts": position.get("entry_fill_ts"),
                        "execution": execution,
                    })
                else:
                    outcomes.append({
                        "candidate_id": decision["candidate_id"], "symbol": decision["__symbol__"],
                        "status": "prepared_dry_run", "decision_open": decision["decision_open"],
                        "signal_atr": decision["signal_atr"],
                    })
            except Exception as exc:
                outcomes.append({
                    "candidate_id": str(row.get("candidate_id")), "symbol": str(row.get("symbol")),
                    "status": "rejected_before_or_during_execution", "reason": f"{type(exc).__name__}:{exc}",
                })
        commit_receipt_hash = _sha256(staged_commit / "receipt.json")
        # A commit is considered consumed only after every proposal has a terminal
        # result.  This prevents duplicate orders while retaining failures as a
        # durable, reviewable terminal outcome.
        state["p8u_processed_score_receipts"].append(commit_receipt_hash)
        state["processed_decision_ids"] = sorted(set([
            *state.get("processed_decision_ids", []), str(decision_ts.isoformat()),
        ]))
        state["as_of_ts"] = stamp.isoformat()
        state["p8u_last_gateway_receipt"] = str(out_dir / "gateway_receipt.json")
        if submit_orders:
            atomic_json(state_path, state)
    result = {
        "schema": SCHEMA,
        "status": "completed_live" if submit_orders else "completed_dry_run",
        "order_submission": bool(submit_orders),
        "gateway_contract_sha256": gateway_sha,
        "p8u_bundle_sha256": contract["bundle"]["sha256"],
        "no_order_adapter_sha256": adapter.sha256,
        "policy_sha256": contract["policy"]["sha256"],
        "staged_commit": str(staged_commit),
        "staged_commit_receipt_sha256": _sha256(staged_commit / "receipt.json"),
        "decision_timestamp": decision_ts.isoformat(),
        "source_timestamp": source_ts.isoformat(),
        "evaluated_at": stamp.isoformat(),
        "account_reconciliation": account_audit,
        "portfolio": audit,
        "proposed_entries": int(len(proposed)),
        "outcomes": outcomes,
        "state_path": str(state_path.resolve()),
    }
    _write_once(out_dir / "gateway_receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-contract", type=Path, required=True)
    parser.add_argument("--staged-commit", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--now", default=None)
    parser.add_argument("--submit-orders", action="store_true")
    args = parser.parse_args()
    print(json.dumps(execute(
        gateway_path=args.gateway_contract,
        staged_commit=args.staged_commit,
        state_path=args.state,
        out_dir=args.out_dir,
        submit_orders=bool(args.submit_orders),
        now=args.now,
    ), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
