#!/usr/bin/env python3
"""Persistent fresh-hour session for the separately sealed P8U E2/H4 live path."""

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


SCHEMA = "strict_r3_p8u_e2_h4_live_session_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _root_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _run(command: list[str], *, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"subprocess failed ({result.returncode}): {log}")


def _file(descriptor: Mapping[str, Any], *, role: str) -> Path:
    value = descriptor.get("path")
    if not isinstance(value, str):
        raise ValueError(f"session lacks {role}")
    path = _root_path(value).resolve()
    if not path.exists():
        raise FileNotFoundError(f"session artifact missing: {role}")
    expected = descriptor.get("sha256")
    if isinstance(expected, str) and path.is_file() and _sha256(path) != expected:
        raise ValueError(f"session hash mismatch: {role}")
    return path


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    if payload.get("schema") != SCHEMA or payload.get("order_submission") is not True:
        raise ValueError("invalid E2/H4 persistent session contract")
    if int(payload.get("post_score_execution_wait_seconds", -1)) != 0:
        raise ValueError("successor session may not add a post-score execution wait")
    runtime = dict(payload.get("runtime") or {})
    current = Path(__file__).resolve()
    if runtime.get("path") != str(current.relative_to(ROOT)) or runtime.get("sha256") != _sha256(current):
        raise ValueError("successor session runtime is not sealed")
    for key in ("gateway_contract", "bundle", "e2_h4_bundle", "source_manifest", "initial_source_state", "regular_bootstrap_root", "direct_bootstrap_root", "score_root", "source_refresh_runtime", "score_runner_runtime", "gateway_runtime"):
        _file(dict(payload.get(key) or {}), role=key)
    if tuple(payload.get("regular_state_components") or ()) != (
        "raw", "causal_transform", "derived", "nested", "oi_iqr", "fixed_ffd", "spectral", "grouped", "ewma", "regime_transition",
    ):
        raise ValueError("session would alter the sealed upstream feature-state contract")
    contract_sha = _sha256(path)
    gateway = json.loads(_file(dict(payload["gateway_contract"]), role="gateway").read_text())
    activation_path = gateway.get("activation_path")
    if not isinstance(activation_path, str):
        raise ValueError("successor gateway lacks activation")
    activation = json.loads(_root_path(activation_path).read_text())
    if activation.get("session_contract_sha256") != contract_sha or activation.get("session_runtime_sha256") != runtime.get("sha256"):
        raise ValueError("successor activation does not bind this persistent session")
    return payload, contract_sha


def _load_state(path: Path, *, contract: Mapping[str, Any], contract_sha: str) -> dict[str, Any]:
    if not path.exists():
        return {"schema": SCHEMA, "session_contract_sha256": contract_sha, "latest_source_state": str(_file(dict(contract["initial_source_state"]), role="initial_source_state")), "processed_decisions": [], "updated_at": None}
    payload = json.loads(path.read_text())
    if payload.get("schema") != SCHEMA or payload.get("session_contract_sha256") != contract_sha:
        raise ValueError("successor session state belongs to another contract")
    if not isinstance(payload.get("processed_decisions"), list) or not Path(str(payload.get("latest_source_state") or "")).is_file():
        raise ValueError("successor session state is incomplete")
    return payload


def _run_one(*, contract: Mapping[str, Any], contract_sha: str, state_path: Path, decision_ts: pd.Timestamp) -> dict[str, Any]:
    decision_ts = utc(decision_ts).floor("h")
    source_ts = decision_ts - pd.Timedelta(hours=1)
    now = pd.Timestamp.now(tz="UTC")
    if now < decision_ts or now > decision_ts + pd.Timedelta(minutes=15):
        raise ValueError("successor session refuses incomplete or stale decision")
    name = decision_ts.strftime("%Y%m%dT%H%M%SZ")
    run_root = _root_path(str(contract["run_root"])) / f"decision_{name}"
    if run_root.exists():
        raise FileExistsError("successor session decision root is immutable")
    run_root.mkdir(parents=True)
    state = _load_state(state_path, contract=contract, contract_sha=contract_sha)
    if decision_ts.isoformat() in set(map(str, state["processed_decisions"])):
        raise ValueError("successor session decision already terminal")
    predecessor = Path(str(state["latest_source_state"])).resolve()
    source_out, refresh_out = run_root / "source_successor", run_root / "source_refresh"
    _run([sys.executable, str(_file(dict(contract["source_refresh_runtime"]), role="source_refresh_runtime")), "--source-state", str(predecessor), "--canonical-manifest", str(_file(dict(contract["source_manifest"]), role="source_manifest")), "--end-exclusive", decision_ts.isoformat(), "--out-dir", str(source_out), "--refresh-root", str(refresh_out)], log=run_root / "source_refresh_runner.log")
    successor = source_out / "source_panel_state.joblib"
    if not successor.is_file():
        raise RuntimeError("source refresh did not produce a successor state")
    state.update({"latest_source_state": str(successor.resolve()), "updated_at": pd.Timestamp.now(tz="UTC").isoformat(), "last_source_refresh": str(refresh_out.resolve())})
    atomic_json(state_path, state)
    _run([sys.executable, str(_file(dict(contract["score_runner_runtime"]), role="score_runner_runtime")), "--bundle", str(_file(dict(contract["bundle"]), role="bundle")), "--source-state", str(successor), "--regular-bootstrap-root", str(_file(dict(contract["regular_bootstrap_root"]), role="regular_bootstrap_root")), "--direct-bootstrap-root", str(_file(dict(contract["direct_bootstrap_root"]), role="direct_bootstrap_root")), "--regular-state-scope", str(contract["regular_state_scope"]), "--regular-state-components", *map(str, contract["regular_state_components"]), "--timestamp", source_ts.isoformat(), "--out-root", str(_file(dict(contract["score_root"]), role="score_root"))], log=run_root / "score_runner.log")
    commit = _file(dict(contract["score_root"]), role="score_root") / "scoring" / "commits" / source_ts.strftime("%Y%m%dT%H%M%SZ")
    if not (commit / "receipt.json").is_file():
        raise RuntimeError("stateful scorer did not publish the expected score commit")
    _run([sys.executable, str(_file(dict(contract["gateway_runtime"]), role="gateway_runtime")), "--gateway-contract", str(_file(dict(contract["gateway_contract"]), role="gateway_contract")), "--staged-commit", str(commit), "--state", str(_root_path(str(contract["live_state_path"])).resolve()), "--out-dir", str(run_root / "gateway"), "--now", pd.Timestamp.now(tz="UTC").isoformat(), "--submit-orders"], log=run_root / "gateway_runner.log")
    gateway_receipt = json.loads((run_root / "gateway" / "gateway_receipt.json").read_text())
    state.update({"processed_decisions": [*state["processed_decisions"], decision_ts.isoformat()], "updated_at": pd.Timestamp.now(tz="UTC").isoformat(), "last_run": str(run_root.resolve())})
    atomic_json(state_path, state)
    result = {"schema": SCHEMA, "status": "terminal", "decision_timestamp": decision_ts.isoformat(), "source_timestamp": source_ts.isoformat(), "source_successor": str(successor.resolve()), "score_commit": str(commit.resolve()), "gateway_receipt": str((run_root / "gateway" / "gateway_receipt.json").resolve()), "gateway_status": gateway_receipt.get("status"), "proposed_entries": gateway_receipt.get("proposed_entries"), "outcomes": gateway_receipt.get("outcomes")}
    _write_once(run_root / "session_receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-contract", type=Path, required=True)
    parser.add_argument("--session-state", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    if bool(args.once) == bool(args.loop):
        parser.error("select exactly one of --once or --loop")
    contract, contract_sha = _load_contract(args.session_contract)
    state_path = args.session_state.resolve()
    if args.once:
        print(json.dumps(_run_one(contract=contract, contract_sha=contract_sha, state_path=state_path, decision_ts=pd.Timestamp.now(tz="UTC").floor("h")), indent=2, sort_keys=True, default=str))
        return
    last_seen = pd.Timestamp.now(tz="UTC").floor("h")
    while True:
        current = pd.Timestamp.now(tz="UTC").floor("h")
        if current > last_seen:
            last_seen = current
            try:
                print(json.dumps(_run_one(contract=contract, contract_sha=contract_sha, state_path=state_path, decision_ts=current), sort_keys=True, default=str), flush=True)
            except Exception as exc:
                failure_root = _root_path(str(contract["run_root"])) / f"decision_{current.strftime('%Y%m%dT%H%M%SZ')}_failed"
                failure_root.mkdir(parents=True, exist_ok=True)
                _write_once(failure_root / "failure.json", {"schema": SCHEMA, "status": "failed_closed", "decision_timestamp": current.isoformat(), "error": f"{type(exc).__name__}:{exc}"})
                print(json.dumps({"status": "failed_closed", "decision_timestamp": current.isoformat(), "error": str(exc)}, sort_keys=True), flush=True)
        time.sleep(2.0)


if __name__ == "__main__":
    main()
