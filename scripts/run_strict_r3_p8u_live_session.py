#!/usr/bin/env python3
"""Persistent, fail-closed hourly session for the frozen P8U live gateway.

For each new UTC decision hour it performs exactly one causal chain:

completed source hour → append-only public-source successor → stateful P8U
score → deterministic no-order auction → authorised fresh gateway.

Every stage writes an immutable run directory.  A partial public source, a
score failure, a stale decision, an artifact mismatch, or an account mismatch
is terminal for that decision; the loop simply waits for the next hour.  It
does not retry a gateway receipt and never submits an order from an earlier
decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.strict_r3_live_execution import atomic_json, utc


SCHEMA = "strict_r3_p8u_live_session_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _root_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _run(command: list[str], *, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        raise RuntimeError(f"subprocess failed ({completed.returncode}): {log}")


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("order_submission") is not True:
        raise ValueError("invalid P8U live session contract")
    if int(payload.get("post_score_execution_wait_seconds", -1)) != 0:
        raise ValueError("P8U live session must retain zero artificial post-score wait")
    runtime = dict(payload.get("runtime") or {})
    current = Path(__file__).resolve()
    if runtime.get("path") != str(current.relative_to(ROOT)) or runtime.get("sha256") != _sha256(current):
        raise ValueError("P8U live session runtime is not hash-bound")
    for key in (
        "gateway_contract", "bundle", "source_manifest", "initial_source_state",
        "regular_bootstrap_root", "direct_bootstrap_root", "score_root",
        "source_refresh_runtime", "score_runner_runtime",
    ):
        descriptor = payload.get(key)
        if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("path"), str):
            raise ValueError(f"session lacks {key}")
        candidate = _root_path(str(descriptor["path"])).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"session artifact missing: {key}")
        expected = descriptor.get("sha256")
        if isinstance(expected, str) and candidate.is_file() and _sha256(candidate) != expected:
            raise ValueError(f"session artifact hash mismatch: {key}")
    if tuple(payload.get("regular_state_components") or ()) != (
        "raw", "causal_transform", "derived", "nested", "oi_iqr", "fixed_ffd",
        "spectral", "grouped", "ewma", "regime_transition",
    ):
        raise ValueError("session would alter the sealed regular-state component contract")
    contract_sha = _sha256(path)
    gateway_payload = json.loads(
        _root_path(str(payload["gateway_contract"]["path"])).read_text(encoding="utf-8")
    )
    activation_path = gateway_payload.get("activation_path")
    if not isinstance(activation_path, str):
        raise ValueError("gateway contract lacks an activation path")
    activation = json.loads(_root_path(activation_path).read_text(encoding="utf-8"))
    if (
        activation.get("session_contract_sha256") != contract_sha
        or activation.get("session_runtime_sha256") != runtime.get("sha256")
    ):
        raise ValueError("P8U activation does not bind this persistent session")
    return payload, contract_sha


def _load_state(path: Path, *, contract_sha: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema": SCHEMA,
            "session_contract_sha256": contract_sha,
            "latest_source_state": str(_root_path(str(contract["initial_source_state"]["path"])).resolve()),
            "processed_decisions": [],
            "updated_at": None,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("session_contract_sha256") != contract_sha:
        raise ValueError("P8U live session state belongs to another contract")
    if not isinstance(payload.get("processed_decisions"), list):
        raise ValueError("P8U session processed-decision ledger is invalid")
    if not Path(str(payload.get("latest_source_state") or "")).is_file():
        raise ValueError("P8U session lacks a readable source-state predecessor")
    return payload


def _run_one(*, contract: Mapping[str, Any], contract_sha: str, state_path: Path, decision_ts: pd.Timestamp) -> dict[str, Any]:
    decision_ts = utc(decision_ts).floor("h")
    source_ts = decision_ts - pd.Timedelta(hours=1)
    now = pd.Timestamp.now(tz="UTC")
    if now < decision_ts:
        raise ValueError("refusing to run an incomplete decision hour")
    if now > decision_ts + pd.Timedelta(minutes=15):
        raise ValueError("fresh P8U session will not execute an expired decision")
    run_name = decision_ts.strftime("%Y%m%dT%H%M%SZ")
    run_root = _root_path(str(contract["run_root"])) / f"decision_{run_name}"
    if run_root.exists():
        raise FileExistsError("immutable P8U session decision root already exists")
    run_root.mkdir(parents=True)
    state = _load_state(state_path, contract_sha=contract_sha, contract=contract)
    if decision_ts.isoformat() in set(map(str, state["processed_decisions"])):
        raise ValueError("P8U decision is already terminal in session state")
    predecessor = Path(str(state["latest_source_state"])).resolve()
    source_out = run_root / "source_successor"
    refresh_out = run_root / "source_refresh"
    _run([
        sys.executable, str(_root_path(str(contract["source_refresh_runtime"]["path"])).resolve()),
        "--source-state", str(predecessor),
        "--canonical-manifest", str(_root_path(str(contract["source_manifest"]["path"])).resolve()),
        "--end-exclusive", decision_ts.isoformat(),
        "--out-dir", str(source_out), "--refresh-root", str(refresh_out),
    ], log=run_root / "source_refresh_runner.log")
    successor = source_out / "source_panel_state.joblib"
    if not successor.is_file():
        raise RuntimeError("source refresh reported success without a successor panel")
    # Source publication is independently complete, target-free, and
    # append-only.  Persist its successor *before* scoring or account I/O so
    # a downstream failure cannot strand the next candle behind an already
    # valid predecessor.  This is not a decision completion: the same hour
    # can never be re-executed after a score/gateway failure, but the public
    # source chain remains continuous for the following hour.
    state["latest_source_state"] = str(successor.resolve())
    state["updated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    state["last_source_refresh"] = str(refresh_out.resolve())
    atomic_json(state_path, state)
    # Advance the exact sequential score ledger.  The stateful executor rejects
    # non-adjacent source timestamps, so this cannot silently skip an hour.
    _run([
        sys.executable, "scripts/run_strict_r3_p8u_stateful_single_timestamp_executor.py",
        "--bundle", str(_root_path(str(contract["bundle"]["path"])).resolve()),
        "--source-state", str(successor),
        "--regular-bootstrap-root", str(_root_path(str(contract["regular_bootstrap_root"]["path"])).resolve()),
        "--direct-bootstrap-root", str(_root_path(str(contract["direct_bootstrap_root"]["path"])).resolve()),
        "--regular-state-scope", str(contract["regular_state_scope"]),
        "--regular-state-components", *map(str, contract["regular_state_components"]),
        "--timestamp", source_ts.isoformat(),
        "--out-root", str(_root_path(str(contract["score_root"]["path"])).resolve()),
    ], log=run_root / "score_runner.log")
    commit = _root_path(str(contract["score_root"]["path"])) / "scoring" / "commits" / source_ts.strftime("%Y%m%dT%H%M%SZ")
    if not (commit / "receipt.json").is_file():
        raise RuntimeError("stateful scorer did not publish its immutable score commit")
    # The feature vector is point-in-time and the gateway performs a fresh
    # executable-book/price-gap/impact check immediately before any order.
    # Do not add an artificial post-close delay here: once the immutable score
    # commit is available, the decision is fresher—not less causal.  The
    # gateway's sealed 15-minute maximum-age guard remains authoritative.
    _run([
        sys.executable, "scripts/run_strict_r3_p8u_live_gateway.py",
        "--gateway-contract", str(_root_path(str(contract["gateway_contract"]["path"])).resolve()),
        "--staged-commit", str(commit),
        "--state", str(_root_path(str(contract["live_state_path"])).resolve()),
        "--out-dir", str(run_root / "gateway"),
        "--now", pd.Timestamp.now(tz="UTC").isoformat(),
        "--submit-orders",
    ], log=run_root / "gateway_runner.log")
    gateway_receipt = json.loads((run_root / "gateway" / "gateway_receipt.json").read_text())
    state["processed_decisions"].append(decision_ts.isoformat())
    state["updated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    state["last_run"] = str(run_root.resolve())
    atomic_json(state_path, state)
    result = {
        "schema": SCHEMA,
        "status": "terminal",
        "decision_timestamp": decision_ts.isoformat(),
        "source_timestamp": source_ts.isoformat(),
        "source_successor": str(successor.resolve()),
        "score_commit": str(commit.resolve()),
        "gateway_receipt": str((run_root / "gateway" / "gateway_receipt.json").resolve()),
        "gateway_status": gateway_receipt.get("status"),
        "proposed_entries": gateway_receipt.get("proposed_entries"),
        "outcomes": gateway_receipt.get("outcomes"),
    }
    _write_once(run_root / "session_receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-contract", type=Path, required=True)
    parser.add_argument("--session-state", type=Path, required=True)
    parser.add_argument("--once", action="store_true", help="Run the current UTC decision hour once.")
    parser.add_argument("--loop", action="store_true", help="Stay persistent and process each new fresh UTC decision hour.")
    args = parser.parse_args()
    if bool(args.once) == bool(args.loop):
        parser.error("select exactly one of --once or --loop")
    contract, contract_sha = _load_contract(args.session_contract)
    state_path = args.session_state.resolve()
    if args.once:
        print(json.dumps(_run_one(
            contract=contract, contract_sha=contract_sha, state_path=state_path,
            decision_ts=pd.Timestamp.now(tz="UTC").floor("h"),
        ), indent=2, sort_keys=True, default=str))
        return
    # A daemon is launched between candles.  Its first executable decision is
    # the *next* fresh UTC hour, never the partially elapsed current hour.
    last_seen: pd.Timestamp | None = pd.Timestamp.now(tz="UTC").floor("h")
    while True:
        current = pd.Timestamp.now(tz="UTC").floor("h")
        if last_seen is None or current > last_seen:
            last_seen = current
            try:
                print(json.dumps(_run_one(
                    contract=contract, contract_sha=contract_sha, state_path=state_path,
                    decision_ts=current,
                ), sort_keys=True, default=str), flush=True)
            except Exception as exc:
                # The failure is durable but intentionally non-retryable for
                # this decision: a second gateway pass could duplicate orders.
                failure_root = _root_path(str(contract["run_root"])) / f"decision_{current.strftime('%Y%m%dT%H%M%SZ')}_failed"
                failure_root.mkdir(parents=True, exist_ok=True)
                _write_once(failure_root / "failure.json", {
                    "schema": SCHEMA, "status": "failed_closed", "decision_timestamp": current.isoformat(),
                    "error": f"{type(exc).__name__}:{exc}",
                })
                print(json.dumps({"status": "failed_closed", "decision_timestamp": current.isoformat(), "error": str(exc)}, sort_keys=True), flush=True)
        time.sleep(2.0)


if __name__ == "__main__":
    main()
