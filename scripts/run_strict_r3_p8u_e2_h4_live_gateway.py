#!/usr/bin/env python3
"""Exchange-writing gateway for the separately sealed P8U E2/H4 successor.

This process owns entry execution only.  It consumes a fresh, target-free
upstream score commit, materialises E2's causal 15-minute fields, makes the
bounded E2 replacement, and applies the ordinary constrained auction.  It
never builds features from outcomes, never retries a consumed decision, and
uses the established Kraken entry primitive solely after all successor
identities, account state, and execution checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.data_fetcher import make_exchange
from extreme_price_movements.inference.p8u_e2_h4_auction import (
    E2_AUCTION_SCHEMA,
    E2AuctionSpec,
    apply_e2_before_auction,
)
from extreme_price_movements.inference.p8u_e2_h4_live_features import (
    _hourly_signal_atr,
    materialize_e2_15m_features,
)
from extreme_price_movements.inference.p8u_e2_h4_live_parity import (
    P8UE2H4LiveParityBundle,
    apply_e2_replacement,
)
from extreme_price_movements.inference.strict_r3_live_execution import (
    DUAL_BCF_CURRENT_PORTFOLIO_SCHEMA,
    StrictR3ExecutionContract,
    _entry_blueprint_from_decision,
    _fetch_exchange_positions_for_monitor,
    _kraken_flex_margin_snapshot,
    _open_live_position,
    atomic_json,
    live_state_lock,
    load_state,
    sha256_file,
    utc,
)
SCHEMA = "strict_r3_p8u_e2_h4_kraken_live_gateway_v1"
STATE_SCHEMA = "strict_r3_kraken_live_state_v1"
RAW_15M_ROOT = ROOT / "data_perp/exchanges/krakenfutures/raw/ohlcv_15m"
SHARED_15M_ROOT = ROOT / "15m_ohlcv_perp"
OFFICIAL_H1_ROOT = ROOT / "data_perp/exchanges/krakenfutures/frozen_contract_backfill_hourly"


def _sha256(path: Path) -> str:
    return sha256_file(Path(path))


def _root_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _bound_file(descriptor: Mapping[str, Any], *, role: str) -> Path:
    path_value, expected = descriptor.get("path"), descriptor.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise ValueError(f"{role} lacks a hash-bound file")
    path = _root_path(path_value).resolve()
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"{role} hash mismatch")
    return path


def _bound_directory(descriptor: Mapping[str, Any], *, role: str) -> Path:
    path_value, expected = descriptor.get("path"), descriptor.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        raise ValueError(f"{role} lacks a hash-bound directory")
    root = _root_path(path_value).resolve()
    manifest = root / "bundle_manifest.json"
    if not manifest.is_file() or _sha256(manifest) != expected:
        raise ValueError(f"{role} bundle manifest hash mismatch")
    return root


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unknown E2/H4 live gateway schema")
    if payload.get("side") != "long" or payload.get("exchange_id") != "krakenfutures":
        raise ValueError("E2/H4 gateway is restricted to Kraken Futures long-only")
    if payload.get("order_submission") is not True:
        raise ValueError("successor live gateway requires order_submission=true")
    if int(payload.get("maximum_decision_age_seconds", -1)) != 900:
        raise ValueError("successor must retain the 15-minute stale-decision guard")
    upstream = _bound_file(dict(payload.get("bundle") or {}), role="upstream P8U")
    e2_root = _bound_directory(dict(payload.get("e2_h4_bundle") or {}), role="E2/H4")
    policy = _bound_file(dict(payload.get("policy") or {}), role="rich policy")
    release = _bound_file(dict(payload.get("release_candidate") or {}), role="no-order release")
    dependencies = payload.get("runtime_dependencies")
    if not isinstance(dependencies, Mapping):
        raise ValueError("successor gateway lacks hash-bound runtime dependencies")
    required_dependencies = {
        "strict_execution_engine", "e2_authority",
        "e2_features", "e2_auction", "h4_state", "h4_policy_wrapper",
        "continuation_features", "fifteen_minute_features", "live_bar_loader",
    }
    if set(dependencies) != required_dependencies:
        raise ValueError("successor runtime dependency contract is incomplete")
    for role, descriptor in dependencies.items():
        _bound_file(dict(descriptor), role=f"runtime dependency {role}")
    activation_value = payload.get("activation_path")
    if not isinstance(activation_value, str) or not activation_value:
        raise ValueError("successor gateway lacks activation path")
    activation_path = _root_path(activation_value).resolve()
    if not activation_path.is_file():
        raise FileNotFoundError("successor activation receipt is absent")
    runtime = dict(payload.get("gateway_runtime") or {})
    current = Path(__file__).resolve()
    if runtime.get("path") != str(current.relative_to(ROOT)) or runtime.get("sha256") != _sha256(current):
        raise ValueError("successor gateway runtime is not sealed")
    activation = json.loads(activation_path.read_text(encoding="utf-8"))
    if (
        activation.get("schema") != "strict_r3_p8u_e2_h4_kraken_live_activation_v1"
        or activation.get("authorized") is not True
        or activation.get("manual_user_authorization") is not True
        or activation.get("stale_signal_execution_prohibited") is not True
        or activation.get("gateway_contract_sha256") != _sha256(path)
        or activation.get("upstream_bundle_sha256") != _sha256(upstream)
        or activation.get("e2_h4_bundle_manifest_sha256") != _sha256(e2_root / "bundle_manifest.json")
        or activation.get("policy_sha256") != _sha256(policy)
        or activation.get("release_candidate_sha256") != _sha256(release)
        or activation.get("gateway_runtime_sha256") != _sha256(current)
    ):
        raise ValueError("successor activation does not bind this exact gateway")
    if pd.Timestamp.now(tz="UTC") >= utc(activation["expires_at"]):
        raise ValueError("successor activation has expired")
    upstream_manifest = json.loads(upstream.read_text())
    if upstream_manifest.get("runtime", {}).get("order_submission") is not False:
        raise ValueError("upstream model bundle must remain no-order")
    bundle = P8UE2H4LiveParityBundle.load(e2_root)
    if bundle.manifest_sha256 != _sha256(e2_root / "bundle_manifest.json"):
        raise AssertionError("E2/H4 manifest identity drifted")
    payload.update({
        "__gateway_path__": str(path),
        "__gateway_sha256__": _sha256(path),
        "__upstream_root__": str(upstream),
        "__policy_path__": str(policy),
        "__activation_path__": str(activation_path),
        "__activation_sha256__": _sha256(activation_path),
        "__authorized_after__": utc(activation["authorized_after"]).isoformat(),
        "__e2_h4_root__": str(e2_root),
        "__release_path__": str(release),
        "__release_sha256__": _sha256(release),
    })
    return payload, _sha256(path)


def _generic_contract(contract: Mapping[str, Any]) -> StrictR3ExecutionContract:
    """Build the established execution contract without importing its gateway."""
    return StrictR3ExecutionContract(
        inference_bundle=Path(str(contract["__upstream_root__"])),
        inference_bundle_sha256=str(contract["bundle"]["sha256"]),
        exit_policy=Path(str(contract["__policy_path__"])),
        exit_policy_sha256=str(contract["policy"]["sha256"]),
        exchange_id="krakenfutures",
        side="long",
        leverage=7.0,
        maximum_decision_age_seconds=int(contract["maximum_decision_age_seconds"]),
        order_submission_authorized=True,
        activation_authorization=Path(str(contract["__activation_path__"])),
        activation_authorization_sha256=str(contract["__activation_sha256__"]),
        authorized_after=utc(contract["__authorized_after__"]),
        maximum_entry_slippage_bps=float(contract.get("maximum_entry_slippage_bps", 100.0)),
        maximum_exit_slippage_bps=float(contract.get("maximum_exit_slippage_bps", 100.0)),
        baseline_policy_cost_bps=100.0,
        minimum_execution_adjusted_ev_bps=float(contract["execution"]["minimum_execution_adjusted_ev_bps"]),
        maximum_live_spread_bps=float(contract["execution"]["price_gap_guard_bps"]),
        execution_microstructure_buffer_bps=10.0,
        execution_book_telemetry_only=False,
        execution_adjusted_ev_veto_enabled=True,
        protective_stop_exit_vwap_adjustment=True,
        protective_stop_book_levels=10,
        close_based_hard_stop_monitor_enabled=True,
        openpositions_503_fills_fallback_enabled=True,
    )


def _ensure_state(*, path: Path, decision_ts: pd.Timestamp, contract: Mapping[str, Any], bundle: P8UE2H4LiveParityBundle) -> dict[str, Any]:
    state = load_state(path, decision_ts=decision_ts)
    if state.get("schema") != STATE_SCHEMA:
        raise ValueError("successor live-state schema mismatch")
    identity = {
        "inference_bundle_sha256": str(contract["bundle"]["sha256"]),
        "exit_policy_sha256": str(contract["policy"]["sha256"]),
        "activation_authorization_sha256": str(contract["__activation_sha256__"]),
        "p8u_e2_h4_gateway_contract_sha256": str(contract["__gateway_sha256__"]),
        "p8u_e2_h4_bundle_manifest_sha256": str(bundle.manifest_sha256),
        "p8u_e2_h4_release_candidate_sha256": str(contract["__release_sha256__"]),
    }
    if not path.exists():
        state.update(identity)
        state["p8u_e2_h4_processed_score_receipts"] = []
        state["p8u_e2_h4_live_gateway_schema"] = SCHEMA
    else:
        for key, expected in identity.items():
            if state.get(key) != expected:
                raise ValueError(f"successor live-state identity mismatch: {key}")
        if state.get("p8u_e2_h4_live_gateway_schema") != SCHEMA:
            raise ValueError("existing state belongs to a different gateway")
        if not isinstance(state.get("p8u_e2_h4_processed_score_receipts"), list):
            raise ValueError("successor state lacks processed-score ledger")
    return state


def _auction_state(state: Mapping[str, Any]) -> dict[str, Any]:
    positions: list[dict[str, Any]] = []
    for row in state.get("positions", []):
        if not isinstance(row, Mapping):
            raise ValueError("successor state contains malformed position")
        positions.append({
            "symbol": str(row["symbol"]),
            "initial_margin_fraction": 0.10,
        })
    return {"open_positions": positions, "pending_intents": []}


def _reconcile(exchange: Any, state: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    tracked = {str(row["candidate_id"]): dict(row) for row in state.get("positions", []) if isinstance(row, Mapping)}
    exchange_positions, audit = _fetch_exchange_positions_for_monitor(
        exchange, tracked_positions=tracked, expected_side="long", allow_503_fills_fallback=True,
    )
    tracked_symbols = {str(row["exchange_symbol"]) for row in tracked.values()}
    untracked = sorted(set(exchange_positions).difference(tracked_symbols))
    if untracked:
        raise ValueError(f"untracked Kraken positions block successor entry: {untracked}")
    return exchange_positions, audit


def _symbol_filename(symbol: str) -> str:
    return f"{symbol.lower().replace('/', '')}_15m.parquet"


def _load_15m_open(path: Path, decision_ts: pd.Timestamp) -> float | None:
    """Return one contemporaneous open from an immutable local 15-minute cache."""
    if not path.is_file():
        return None
    try:
        frame = pd.read_parquet(path, columns=["open"])
    except Exception:
        return None
    if not isinstance(frame.index, pd.DatetimeIndex):
        return None
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    value = pd.to_numeric(frame.loc[frame.index == decision_ts, "open"], errors="coerce")
    if len(value) != 1 or not np.isfinite(value.iloc[0]) or float(value.iloc[0]) <= 0.0:
        return None
    return float(value.iloc[0])


def _decision_open(symbol: str, decision_ts: pd.Timestamp) -> float:
    """Canonical local-cache decision-open adapter; no later bar can fill it."""
    filename = _symbol_filename(symbol)
    for root in (RAW_15M_ROOT, SHARED_15M_ROOT):
        value = _load_15m_open(root / filename, decision_ts)
        if value is not None:
            return value
    hourly_path = OFFICIAL_H1_ROOT / f"{symbol.split('/')[0]}_USD_USD.parquet"
    if hourly_path.is_file():
        try:
            hourly = pd.read_parquet(hourly_path, columns=["open", "volume"])
            hourly.index = pd.to_datetime(hourly.index, utc=True, errors="coerce")
            row = hourly.loc[hourly.index == decision_ts]
            if len(row) == 1:
                opening = float(pd.to_numeric(row["open"], errors="coerce").iloc[0])
                volume = float(pd.to_numeric(row["volume"], errors="coerce").iloc[0])
                if np.isfinite(opening) and opening > 0.0 and np.isfinite(volume) and volume > 0.0:
                    return opening
        except Exception:
            pass
    raise ValueError("no finite exact 15-minute decision open")


def _signal_atr(symbol: str, source_ts: pd.Timestamp) -> float:
    path = SHARED_15M_ROOT / _symbol_filename(symbol)
    if not path.is_file():
        raise ValueError("no 15-minute source for decision-time ATR")
    bars = pd.read_parquet(path, columns=["high", "low", "close"])
    bars.index = pd.to_datetime(bars.index, utc=True, errors="raise")
    atr = pd.to_numeric(_hourly_signal_atr(bars).reindex([source_ts]), errors="coerce").iloc[0]
    if not np.isfinite(atr) or float(atr) <= 0.0:
        raise ValueError("no finite decision-time Wilder14 ATR")
    return float(atr)


def _decision_for_proposal(*, proposal: Mapping[str, Any], source_ts: pd.Timestamp, policy: Mapping[str, Any]) -> dict[str, Any]:
    symbol = str(proposal["symbol"])
    decision_ts = utc(proposal["__decision_ts__"])
    params = dict(policy["params"])
    decision = dict(proposal)
    decision.update({
        "candidate_id": str(proposal["candidate_id"]),
        "__decision_ts__": decision_ts,
        "__symbol__": symbol,
        "side_name": "long",
        "decision_open": _decision_open(symbol, decision_ts),
        "signal_atr": _signal_atr(symbol, source_ts),
        "portfolio_policy_schema": E2_AUCTION_SCHEMA,
        "bcf_mc1_expected_net_bps": float(proposal["bcf_mc1_expected_bps"]),
        "policy_parent_name": "SimplePolicyOptimiser rich smooth-capital-protection",
        "policy_timeout_hours": 12,
        "policy_cost_bps_once": 100.0,
        "policy_sl_atr": float(params["sl_mult"]),
        "policy_trailing_activation_atr": float(params["trailing_activation_mult"]),
        "policy_trailing_giveback_atr": float(params["fixed_trailing_gap_mult"]),
        "e2_action": str(proposal.get("e2_action")),
        "h4_mc1_expected_bps": float(proposal["bcf_mc1_expected_bps"]),
    })
    return decision


def execute(*, gateway_path: Path, staged_commit: Path, state_path: Path, out_dir: Path, submit_orders: bool, now: object | None) -> dict[str, Any]:
    contract, gateway_sha = _load_contract(gateway_path)
    bundle = P8UE2H4LiveParityBundle.load(Path(str(contract["__e2_h4_root__"])))
    policy = json.loads(Path(str(contract["__policy_path__"])).read_text())
    if str(policy.get("schema")) != "strict_r3_rich_simple_policy_challenger_v1":
        raise ValueError("successor requires sealed rich parent policy")
    staged_commit, out_dir = Path(staged_commit).resolve(), Path(out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError("successor gateway receipt path must be immutable")
    receipt_path = staged_commit / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("outcome_columns_consumed") not in (None, []):
        raise ValueError("successor refuses non-target-free staged score commit")
    decision_ts, source_ts = utc(receipt["decision_timestamp"]), utc(receipt["source_timestamp"])
    timestamp = utc(now or pd.Timestamp.now(tz="UTC"))
    scores = pd.read_parquet(staged_commit / "routed_scores.parquet")
    required = {"candidate_id", "__decision_ts__", "__symbol__", "bcf_mc1_expected_bps", "current_mc1_expected_bps", "dual_mc1_min_bps", "bcf_final_score"}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"staged score commit lacks successor inputs: {missing}")
    features = materialize_e2_15m_features(scores.loc[:, ["candidate_id", "__decision_ts__", "__symbol__"]], bars_root=ROOT / "15m_ohlcv_perp")
    for frame in (scores, features):
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        frame["__symbol__"] = frame["__symbol__"].astype(str)
        if frame["candidate_id"].duplicated().any():
            raise ValueError("successor target-free inputs duplicate identities")
    collision = set(scores.columns).intersection(set(features.columns) - {"candidate_id", "__decision_ts__", "__symbol__"})
    if collision:
        scores = scores.drop(columns=sorted(collision))
    candidates = scores.merge(features, on=["candidate_id", "__decision_ts__", "__symbol__"], how="left", validate="one_to_one")
    selected, pairs = apply_e2_replacement(candidates, bundle=bundle)
    generic = _generic_contract(contract)
    outcomes: list[dict[str, Any]] = []
    with live_state_lock(state_path):
        state = _ensure_state(path=state_path, decision_ts=decision_ts, contract=contract, bundle=bundle)
        consumed = _sha256(receipt_path)
        if consumed in set(map(str, state["p8u_e2_h4_processed_score_receipts"])):
            raise ValueError("successor refuses a previously consumed score commit")
        if submit_orders:
            exchange = make_exchange("perps")
            _, account_audit = _reconcile(exchange, state)
            wallet = float(_kraken_flex_margin_snapshot(exchange)["margin_equity"])
        else:
            exchange, account_audit, wallet = None, {"status": "not_called_dry_run"}, 1000.0
        auction = apply_e2_before_auction(
            selected, state=_auction_state(state), wallet_equity_quote=wallet, spec=E2AuctionSpec(),
        )
        proposed = auction.loc[auction["execution_action"].eq("propose")].copy()
        out_dir.mkdir(parents=True, exist_ok=False)
        features.to_parquet(out_dir / "e2_features_target_free.parquet", index=False, compression="zstd")
        selected.to_parquet(out_dir / "e2_candidate_selection_target_free.parquet", index=False, compression="zstd")
        pairs.to_parquet(out_dir / "e2_pair_predictions_target_free.parquet", index=False, compression="zstd")
        auction.to_parquet(out_dir / "auction.parquet", index=False, compression="zstd")
        for rank, (_, proposal) in enumerate(proposed.iterrows(), start=1):
            row = proposal.to_dict()
            row["portfolio_priority_rank"] = rank
            try:
                decision = _decision_for_proposal(proposal=row, source_ts=source_ts, policy=policy)
                shadow = _entry_blueprint_from_decision(decision)
                if submit_orders:
                    assert exchange is not None
                    position, execution = _open_live_position(
                        exchange, decision=decision, shadow_position=shadow, contract=generic,
                        slot_margin=0.10, wallet_equity=wallet, observed_at=timestamp,
                    )
                    position.update({
                        "h4_mc1_expected_bps": float(row["bcf_mc1_expected_bps"]),
                        "p8u_e2_action": str(row.get("e2_action")),
                        "p8u_e2_h4_bundle_manifest_sha256": bundle.manifest_sha256,
                    })
                    state["positions"].append(position)
                    outcomes.append({"candidate_id": decision["candidate_id"], "symbol": decision["__symbol__"], "status": "opened_live", "entry_order_id": position.get("entry_order_id"), "stop_order_id": position.get("stop_order_id"), "entry_fill_ts": position.get("entry_fill_ts"), "execution": execution})
                else:
                    outcomes.append({"candidate_id": decision["candidate_id"], "symbol": decision["__symbol__"], "status": "prepared_dry_run"})
            except Exception as exc:
                outcomes.append({"candidate_id": str(row.get("candidate_id")), "symbol": str(row.get("symbol")), "status": "rejected_before_or_during_execution", "reason": f"{type(exc).__name__}:{exc}"})
        state["p8u_e2_h4_processed_score_receipts"].append(consumed)
        state["processed_decision_ids"] = sorted(set([*state.get("processed_decision_ids", []), decision_ts.isoformat()]))
        state["as_of_ts"] = timestamp.isoformat()
        state["p8u_e2_h4_last_gateway_receipt"] = str(out_dir / "gateway_receipt.json")
        if submit_orders:
            atomic_json(state_path, state)
    result = {
        "schema": SCHEMA, "status": "completed_live" if submit_orders else "completed_dry_run", "order_submission": bool(submit_orders),
        "gateway_contract_sha256": gateway_sha, "upstream_bundle_sha256": contract["bundle"]["sha256"], "e2_h4_bundle_manifest_sha256": bundle.manifest_sha256,
        "release_candidate_sha256": contract["__release_sha256__"], "policy_sha256": contract["policy"]["sha256"],
        "staged_commit": str(staged_commit), "staged_commit_receipt_sha256": _sha256(receipt_path), "decision_timestamp": decision_ts.isoformat(), "source_timestamp": source_ts.isoformat(), "evaluated_at": timestamp.isoformat(),
        "target_free_candidate_rows": int(len(candidates)), "e2_feature_complete_rows": int(features["e2_feature_source_status"].isin(["ok", "complete"]).sum()), "e2_selected_rows": int(selected["e2_entry_selected"].sum()), "e2_replacements": int(selected["e2_action"].eq("e2_q50_agreement_replacement").sum()),
        "account_reconciliation": account_audit, "proposed_entries": int(len(proposed)), "outcomes": outcomes, "state_path": str(state_path.resolve()),
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
    print(json.dumps(execute(gateway_path=args.gateway_contract, staged_commit=args.staged_commit, state_path=args.state, out_dir=args.out_dir, submit_orders=bool(args.submit_orders), now=args.now), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
