#!/usr/bin/env python3
"""Advance a sealed P8U feature state one immutable hour at a time.

This is an *offline feature worker*, not an inference or exchange process.  It
uses ``materialize_strict_r3_forward_features_incremental_v13.py`` as the
single canonical constructor and only publishes a new state pointer after:

1. the candidate universe is target-free and exactly one signal hour;
2. the materialised output contains every sealed P8U feature; and
3. when configured, an all-feature full-causal parity audit passes.

The active cache is never modified in place.  A private timestamped cache is
advanced first, then the small JSON pointer is atomically replaced.  Thus a
crash leaves the old resumable state selected.  Feature state remains a named
disk-backed contract; a long-lived supervisor can invoke this worker for each
append-only request without rebuilding feature history.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import resource
import time
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_warm_feature_state import (  # noqa: E402
    P8UWarmFeatureConfig,
    P8UWarmFeatureRequest,
    assert_feature_output_contract,
    assert_next_hour,
    atomic_json,
    audit_feature_parity,
    read_warm_ledger,
    sha256_file,
)


def _copy_cache(source: Path, target: Path) -> None:
    """Create a private mutable state copy without touching active cache."""
    if target.exists():
        raise FileExistsError(f"warm worker staged cache already exists: {target}")
    shutil.copytree(source, target, copy_function=shutil.copy2)


def _request_transaction_root(config: P8UWarmFeatureConfig, request: P8UWarmFeatureRequest) -> Path:
    return config.state_root / "transactions" / request.signal_ts.strftime("%Y%m%dT%H%M%SZ")


def _bundle_latest_timestamp(bundle: Path) -> object | None:
    payload = json.loads((bundle / "state_bundle_manifest.json").read_text())
    return payload.get("latest_state_timestamp")


def _worker_command(
    *,
    config: P8UWarmFeatureConfig,
    request: P8UWarmFeatureRequest,
    cache_dir: Path,
    output_dir: Path,
    initial_bundle: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        "-X",
        "faulthandler",
        str(ROOT / "scripts/materialize_strict_r3_forward_features_incremental_v13.py"),
        "--candidates", str(request.candidates),
        "--panel-state", str(request.panel_state),
        "--cache-dir", str(cache_dir),
        "--cache-is-already-private",
        "--requested-features-json", str(config.payload["feature_plan_path"]),
        "--feature-cache-namespace", config.state_contract_id,
        "--stateful-tail-hours", str(int(config.payload["stateful_tail_hours"])),
        "--side", "long",
        "--out-dir", str(output_dir),
    ]
    # This remains an explicit, hash-bound contract choice.  Only the listed
    # non-associative worksets replay complete history at bootstrap; ordinary
    # hourly appends keep the bounded stateful path.
    for selector in config.payload.get("raw_rolling_exact_seed_selectors", []):
        command.extend(["--raw-rolling-exact-seed-selector", str(selector)])
    for family in config.payload.get("stateful_exact_families", []):
        command.extend(["--stateful-exact-family", str(family)])
    if "final14" in set(config.payload.get("stateful_exact_families", [])):
        command.extend([
            "--expected-final14-contract-hash",
            str(config.payload["final14_contract_hash"]),
        ])
    if "orderbook_precomposite" in set(config.payload.get("stateful_exact_families", [])):
        command.extend([
            "--expected-orderbook-precomposite-contract-hash",
            str(config.payload["orderbook_precomposite_contract_hash"]),
        ])
    if initial_bundle is not None:
        command.extend([
            "--restore-state-bundle", str(initial_bundle),
            "--expected-state-contract-hash", config.feature_union_sha256,
        ])
    return command


def _read_feature_manifest(path: Path) -> dict[str, Any]:
    manifest = path / "feature_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("incremental materializer produced no feature manifest")
    payload = json.loads(manifest.read_text())
    if payload.get("outcome_columns_consumed") not in ([], None):
        raise ValueError("warm materializer consumed outcomes")
    return payload


def _bytes_under(path: Path) -> int:
    """Return the retained on-disk state footprint without traversing links."""
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _child_max_rss_bytes() -> int | None:
    """Best-effort child peak RSS for the profile; unavailable is explicit."""
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if peak <= 0:
        return None
    # macOS returns bytes; Linux reports KiB. This repository's live host is
    # macOS, but keep the receipt portable for offline Linux replays.
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _commit_pointer(
    *,
    config: P8UWarmFeatureConfig,
    request: P8UWarmFeatureRequest,
    transaction: Path,
    candidate_rows: int,
    parity: dict[str, Any] | None,
    materializer_manifest: dict[str, Any],
    runtime_profile: dict[str, Any],
) -> dict[str, Any]:
    """Atomically publish the only authoritative current-state pointer."""
    cache = transaction / "cache"
    if not cache.is_dir():
        raise FileNotFoundError("transaction cache disappeared before pointer commit")
    output = transaction / "features" / "canonical120_features.parquet"
    receipt = {
        "schema": "strict_r3_p8u_warm_feature_commit_v1",
        "state_name": config.state_name,
        "state_contract_id": config.state_contract_id,
        "signal_ts": request.signal_ts.isoformat(),
        "candidates": str(request.candidates),
        "candidates_sha256": sha256_file(request.candidates),
        "panel_state": str(request.panel_state),
        "panel_state_sha256": sha256_file(request.panel_state),
        "cache_dir": str(cache),
        "features": str(output),
        "features_sha256": sha256_file(output),
        "candidate_rows": int(candidate_rows),
        "required_features": int(len(config.feature_plan)),
        "materializer_manifest": materializer_manifest,
        "runtime_profile": runtime_profile,
        "parity": parity,
        "outcome_columns_consumed": [],
    }
    atomic_json(transaction / "commit_receipt.json", receipt)
    ledger = {
        "schema": "strict_r3_p8u_warm_feature_ledger_v1",
        "state_name": config.state_name,
        "state_contract_id": config.state_contract_id,
        "config": str(config.path),
        "config_sha256": sha256_file(config.path),
        "feature_union_sha256": config.feature_union_sha256,
        "last_signal_ts": request.signal_ts.isoformat(),
        "active_commit": str(transaction / "commit_receipt.json"),
        "active_cache": str(cache),
        "active_features": str(output),
        "active_features_sha256": receipt["features_sha256"],
        "parity_status": (parity or {}).get("status", "not_run"),
        "last_runtime_profile": runtime_profile,
        "updated_at_unix": time.time(),
    }
    atomic_json(config.ledger_path, ledger)
    return receipt


def advance_one(
    *,
    config: P8UWarmFeatureConfig,
    request: P8UWarmFeatureRequest,
) -> dict[str, Any]:
    """Build and validate one stateful timestamp, then publish its pointer."""
    config.state_root.mkdir(parents=True, exist_ok=True)
    ledger = read_warm_ledger(config.ledger_path)
    candidate_rows = request.validate_candidate_timestamp()
    initial_bundle: Path | None = None
    active_cache: Path | None = None
    if ledger is None:
        initial_bundle = config.require_state_bundle()
        assert_next_hour(
            request.signal_ts,
            ledger=None,
            bundle_latest_timestamp=_bundle_latest_timestamp(initial_bundle),
        )
    else:
        if str(ledger.get("config_sha256")) != sha256_file(config.path):
            raise ValueError("warm state ledger belongs to a different config revision")
        if str(ledger.get("feature_union_sha256")) != config.feature_union_sha256:
            raise ValueError("warm state ledger belongs to a different P8U feature plan")
        if str(ledger.get("state_contract_id")) != config.state_contract_id:
            raise ValueError("warm state ledger belongs to a different state contract ID")
        assert_next_hour(request.signal_ts, ledger=ledger, bundle_latest_timestamp=None)
        active_cache = Path(str(ledger["active_cache"])).resolve()
        if not active_cache.is_dir():
            raise FileNotFoundError("warm state ledger points to no active cache")

    transaction = _request_transaction_root(config, request)
    if transaction.exists():
        # An unfinished transaction is intentionally never reused.  It may
        # contain a half-written SQLite cache; preserving it makes diagnosis
        # possible while avoiding an ambiguous state advance.
        raise FileExistsError(f"warm feature transaction already exists: {transaction}")
    transaction.mkdir(parents=True)
    staged_cache = transaction / "cache"
    if active_cache is not None:
        _copy_cache(active_cache, staged_cache)
    output_dir = transaction / "features"
    command = _worker_command(
        config=config,
        request=request,
        cache_dir=staged_cache,
        output_dir=output_dir,
        initial_bundle=initial_bundle,
    )
    log = transaction / "materializer.log"
    child_started = time.perf_counter()
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    child_elapsed = time.perf_counter() - child_started
    # Preserve the child termination mode even when no Python traceback reaches
    # the materializer log (for example a native extension signal or host kill).
    # The staged cache is intentionally left uncommitted, but the receipt makes
    # such failures diagnosable without ever reusing a partial state advance.
    atomic_json(
        transaction / "materializer_exit.json",
        {
            "returncode": int(completed.returncode),
            "elapsed_seconds": float(child_elapsed),
            "command": command,
            "status": "success" if completed.returncode == 0 else "failed",
        },
    )
    if completed.returncode:
        raise RuntimeError(f"P8U warm materializer failed; inspect {log}")
    features = output_dir / "canonical120_features.parquet"
    assert_feature_output_contract(features, config.feature_plan)
    materializer_manifest = _read_feature_manifest(output_dir)
    if int(materializer_manifest.get("new_rows", -1)) != candidate_rows:
        raise AssertionError("P8U warm materializer changed candidate count")
    parity: dict[str, Any] | None = None
    parity_required = bool(config.payload.get("parity_required", True))
    if request.reference_features is not None:
        parity = audit_feature_parity(
            incremental_features=features,
            reference_features=request.reference_features,
            required_features=config.feature_plan,
            out_dir=transaction / "parity",
            atol=float(config.payload.get("parity_atol", 1e-6)),
            rtol=float(config.payload.get("parity_rtol", 1e-6)),
        )
        if parity["status"] != "pass":
            raise AssertionError("P8U warm feature parity audit failed")
    elif parity_required:
        raise ValueError("P8U warm feature advance requires a full causal parity reference")
    runtime_profile = {
        "schema": "strict_r3_p8u_warm_feature_runtime_profile_v1",
        "materializer_elapsed_seconds": float(child_elapsed),
        "materializer_phase_runtime_seconds": materializer_manifest.get(
            "phase_runtime_seconds", {}
        ),
        "materializer_runtime_before_state_commit_seconds": materializer_manifest.get(
            "runtime_seconds_before_state_commit"
        ),
        "child_max_rss_bytes": _child_max_rss_bytes(),
        "candidate_rows": int(candidate_rows),
        "feature_output_bytes": int(features.stat().st_size),
        "transaction_cache_bytes": _bytes_under(staged_cache),
        "active_cache_clone_mode": "private_full_copy_once_no_inner_materializer_clone",
        "measurement_note": (
            "peak RSS is a best-effort subprocess aggregate; phase timing is "
            "the materializer's own measured breakdown"
        ),
    }
    return _commit_pointer(
        config=config,
        request=request,
        transaction=transaction,
        candidate_rows=candidate_rows,
        parity=parity,
        materializer_manifest=materializer_manifest,
        runtime_profile=runtime_profile,
    )


def _requests(args: argparse.Namespace, config: P8UWarmFeatureConfig) -> list[Path]:
    raw = [Path(path).resolve() for path in args.request]
    if args.request_dir is not None:
        directory = Path(args.request_dir).resolve()
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        raw.extend(sorted(directory.glob("*.json")))
    if not raw:
        raise ValueError("provide at least one --request or --request-dir")
    # Read request timestamps before running so a supervisor cannot accidentally
    # hand a long-lived worker an out-of-order queue.
    parsed = [P8UWarmFeatureRequest.load(path, root=config.root) for path in raw]
    if len({request.signal_ts for request in parsed}) != len(parsed):
        raise ValueError("warm feature request queue contains duplicate timestamps")
    return [request.path for request in sorted(parsed, key=lambda item: item.signal_ts)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--request", action="append", default=[])
    parser.add_argument("--request-dir", type=Path)
    args = parser.parse_args()
    config = P8UWarmFeatureConfig.load(args.config, root=ROOT)
    lock_path = config.state_root / ".worker.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another P8U warm feature worker is active") from exc
        try:
            for path in _requests(args, config):
                request = P8UWarmFeatureRequest.load(path, root=config.root)
                print(json.dumps(advance_one(config=config, request=request), sort_keys=True), flush=True)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
