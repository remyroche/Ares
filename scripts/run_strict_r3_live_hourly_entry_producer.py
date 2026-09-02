#!/usr/bin/env python3
"""Run the strict-R3 live entry producer once per fresh UTC hour.

This is deliberately a thin operational wrapper around the sealed scorer,
live-hour audit, runtime checkpoint and exchange executor.  It never computes
models itself.  A failed source refresh, scorer, audit or checkpoint produces
an immutable fail-closed receipt and no entry attempt.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data_perp/artifacts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_inference_bundle import StrictR3InferenceBundle


_WARM_BUNDLES: dict[Path, tuple[str, StrictR3InferenceBundle]] = {}
# The producer is deliberately persistent.  Keep the exchange client and the
# narrow execution boundary in this process as well: spawning a fresh Python
# interpreter after scoring re-imported the very large generic trade executor
# and could consume several minutes before preflight.  This cache is keyed by
# the sealed execution-bundle content hash, so a reseal always creates a new
# client/contract and can never silently inherit a changed runtime contract.
_WARM_LIVE_EXECUTORS: dict[Path, tuple[str, Any, Any, Any, Any, Any]] = {}
# A source retry can be useful while an hour is settling, but a live entry may
# never be created from an increasingly stale decision.  This hard safety
# deadline is deliberately shorter than the historical scoring window.
LIVE_EXECUTION_DEADLINE_SECONDS = 900.0


class ExpiredLiveDecision(RuntimeError):
    """The current decision can no longer cross the order-execution boundary."""


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _retry_schedule(value: str | None, *, first_retry_seconds: float | None) -> tuple[float, ...]:
    """Parse bounded retry checkpoints as elapsed seconds from first refresh."""
    if value:
        parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
        if parsed:
            return parsed
    first = 30.0 if first_retry_seconds is None else float(first_retry_seconds)
    return (first, 60.0, 120.0, 180.0)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_historical_static_contract(path: Path) -> dict[str, object]:
    """Resolve a sealed historical overlay without importing its old sources.

    Runtime-only successors need to verify a previously sealed predecessor
    after a reviewed implementation update.  This routine reads only the
    predecessor JSON and its immutable base JSON.  The caller binds the
    predecessor file hash, compares every static contract field with the
    current fully validated bundle, and permits only the declared code delta.
    It must never be used to execute the historical contract.
    """
    root = ROOT.resolve()
    source = path.resolve()
    if root not in source.parents:
        raise ValueError("historical contract escapes repository root")
    payload = json.loads(source.read_text())
    if payload.get("schema") != "strict_r3_inference_bundle_overlay_v1":
        return dict(payload)
    base_relative = payload.get("base_bundle")
    if not isinstance(base_relative, str) or not base_relative:
        raise ValueError("historical overlay lacks base_bundle")
    base_path = (root / base_relative).resolve()
    if root not in base_path.parents or not base_path.is_file():
        raise ValueError("historical overlay base escapes repository root")
    base = json.loads(base_path.read_text())
    allowed = {
        "admission_contract", "runtime", "paths", "sha256",
        "runtime_code_sha256", "dual_bcf_current",
    }
    overrides = payload.get("overrides") or {}
    if not isinstance(overrides, dict) or set(overrides).difference(allowed):
        raise ValueError("historical overlay contains unsupported overrides")
    merged = dict(base)
    for key, value in overrides.items():
        if key in {"runtime", "paths", "sha256", "runtime_code_sha256"}:
            resolved = dict(base.get(key) or {})
            resolved.update(dict(value or {}))
            merged[key] = resolved
        else:
            merged[key] = value
    merged["overlay"] = {
        "schema": "strict_r3_inference_bundle_overlay_v1",
        "path": str(source),
        "base_bundle": base_relative,
        "base_bundle_sha256": _sha(base_path),
        "overlay_sha256": _sha(source),
    }
    return merged


def _static_contract_for_runtime_bridge(payload: dict[str, object]) -> dict[str, object]:
    """Remove only code/provenance fields outside the economic contract."""
    static = json.loads(json.dumps(payload))
    static.pop("runtime_code_sha256", None)
    static.pop("version_note", None)
    static.pop("runtime_reseal", None)
    # A successor's human-readable purpose is lineage documentation rather
    # than executable authority.  Runtime bridges already bind the exact
    # source delta and must not reject an otherwise identical contract merely
    # because its documentation names the new repair.
    static.pop("purpose", None)
    static.pop("overlay", None)
    runtime = static.get("runtime")
    if isinstance(runtime, dict):
        feature_state = runtime.get("feature_state")
        if isinstance(feature_state, dict):
            # This separately verified receipt binds identical persisted
            # operator payloads across the source-only transition.
            feature_state.pop("one_time_state_reseal", None)
    return static


def _approved_calibration_static_match(
    *,
    current_payload: dict[str, object],
    prior_payload: dict[str, object],
    bridge: dict[str, object],
) -> bool:
    """Allow one explicitly hash-bound BCF replay-ledger replacement.

    A normal runtime reseal may not alter model, feature, policy, or
    calibration inputs.  The same-bundle BCF repair is the narrowly declared
    exception: it replaces only the BCF MC1 resolved replay ledger while
    preserving all model weights and decision semantics.  Validate the exact
    current path and content hash, then compare the remaining static contract
    byte-for-byte in its resolved JSON representation.
    """
    relative = str(bridge.get("approved_calibration_artifact") or "")
    expected_hash = str(bridge.get("approved_calibration_artifact_sha256") or "")
    if not relative or not expected_hash:
        return False
    artifact = (ROOT / relative).resolve()
    if ROOT not in artifact.parents or not artifact.is_file() or _sha(artifact) != expected_hash:
        raise ValueError("approved calibration artifact is absent or hash-mismatched")
    paths = dict(current_payload.get("paths") or {})
    hashes = dict(current_payload.get("sha256") or {})
    if (
        str(paths.get("bcf_mc1_ledger") or "") != relative
        or str(hashes.get("dual_bcf_mc1_ledger") or "") != expected_hash
    ):
        raise ValueError("approved calibration artifact is not the active BCF MC1 ledger")
    current_static = _static_contract_for_runtime_bridge(current_payload)
    prior_static = _static_contract_for_runtime_bridge(prior_payload)
    for payload in (current_static, prior_static):
        payload_paths = payload.get("paths")
        if isinstance(payload_paths, dict):
            payload_paths.pop("bcf_mc1_ledger", None)
        payload_hashes = payload.get("sha256")
        if isinstance(payload_hashes, dict):
            payload_hashes.pop("dual_bcf_mc1_ledger", None)
    return current_static == prior_static


def _load_warm_bundle(path: Path) -> tuple[str, StrictR3InferenceBundle, bool]:
    """Return a hash-bound parsed bundle, reloading only after a reseal.

    The cache is process-local and keyed by the file content hash, never by
    filename alone.  It improves boundary latency but cannot make a changed
    contract executable without a fresh full validation.
    """
    digest = _sha(path)
    cached = _WARM_BUNDLES.get(path)
    if cached is not None and cached[0] == digest:
        return digest, cached[1], True
    bundle = StrictR3InferenceBundle.load(path, root=ROOT)
    _WARM_BUNDLES[path] = (digest, bundle)
    return digest, bundle, False


def _load_warm_live_executor(
    execution_bundle: Path,
) -> tuple[Any, Any, Any, Any, Any]:
    """Return a hash-bound, long-lived Kraken execution boundary.

    This is a latency optimisation only.  It loads the exact sealed contract
    and Kraken market metadata before a decision boundary, then reuses those
    immutable definitions for fresh ticker/book/private-account calls.  Every
    entry still performs its existing live preflight and all state mutations
    remain under ``live_state_lock``.
    """
    path = execution_bundle.resolve()
    digest = _sha(path)
    cached = _WARM_LIVE_EXECUTORS.get(path)
    if cached is not None and cached[0] == digest:
        return cached[1:]

    # Keep this import out of the scoring critical path.  The generic research
    # executor has a large dependency graph; importing it on every accepted
    # hour was the source of the multi-minute post-score delay.
    from extreme_price_movements.inference.data_fetcher import make_exchange
    from extreme_price_movements.inference.strict_r3_live_execution import (
        StrictR3ExecutionContract,
        atomic_json,
        execute_verified_hour,
        live_state_lock,
    )

    contract = StrictR3ExecutionContract.load(path, root=ROOT)
    exchange = make_exchange("perps")
    if str(getattr(exchange, "id", "")) != "krakenfutures":
        raise ValueError("canonical live execution requires Kraken Futures")
    loaded = (contract, exchange, execute_verified_hour, atomic_json, live_state_lock)
    _WARM_LIVE_EXECUTORS[path] = (digest, *loaded)
    return loaded


def _execute_verified_hour_warm(
    *,
    execution_bundle: Path,
    hourly_run: Path,
    state_path: Path,
    out: Path,
    live_hour_audit: Path,
    runtime_checkpoint: Path,
) -> dict[str, Any]:
    """Execute one immutable scored hour through the warmed boundary."""
    contract, exchange, execute_verified_hour, atomic_json, live_state_lock = (
        _load_warm_live_executor(execution_bundle)
    )
    now = pd.Timestamp.now(tz="UTC")
    with live_state_lock(state_path):
        result = execute_verified_hour(
            exchange=exchange,
            contract=contract,
            hourly_run=hourly_run,
            state_path=state_path,
            now=now,
            submit_orders=True,
            live_hour_audit=live_hour_audit,
            runtime_checkpoint=runtime_checkpoint,
        )
    atomic_json(out, result)
    return result


def _runtime_tag(inference_bundle: str) -> str:
    """Return a stable receipt namespace from the sealed bundle filename.

    A runtime-only successor must never overwrite or suppress an earlier
    version's immutable receipts at the same decision timestamp.  The legacy
    v45 name is retained for the v45 bundle; successors get their own lineage.
    """
    match = re.search(r"(?:^|_)(v\d+)(?:_|$)", Path(inference_bundle).stem)
    if not match:
        raise ValueError("inference bundle filename lacks a version namespace")
    return match.group(1)


class SuccessfulProducerReceiptExists(FileExistsError):
    """A completed immutable hour that an idempotent service must skip."""

    def __init__(self, receipt: Path):
        self.receipt = receipt
        super().__init__(f"successful immutable producer receipt exists: {receipt}")


class TerminalExecutionReceiptExists(FileExistsError):
    """A separately sealed live execution review owns this decision hour.

    Operational recovery can complete the audited executor directly after a
    producer stopped at its pre-execution intent boundary.  That review is a
    terminal decision whether it submits an order or rejects every candidate;
    the scheduler must wait for the next hour rather than repeat it.
    """

    def __init__(self, receipt: Path):
        self.receipt = receipt
        super().__init__(f"terminal immutable execution receipt exists: {receipt}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write a small receipt atomically so a crash never yields partial JSON."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _pid_is_live(pid: int) -> bool:
    """Return true only when the recorded producer process is still present.

    A false positive from PID reuse remains fail-closed; it can delay a retry
    but can never cause one while an older receipt may still own execution.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminalize_dead_pre_execution_lease(candidate: Path) -> bool:
    """Terminalize only a proven dead pre-execution receipt.

    New receipts write a lease immediately, then an execution-intent marker
    before the executor subprocess can be created.  A missing terminal
    manifest can therefore be advanced automatically *only* if its owner is
    dead and the intent marker is absent.  Anything else stays fail-closed for
    explicit investigation, avoiding duplicate exchange writes.
    """
    intent = candidate / "execution_attempt_started.json"
    lease_path = candidate / "producer_lease.json"
    if intent.is_file() or not lease_path.is_file():
        return False
    try:
        lease = json.loads(lease_path.read_text())
        pid = int(lease["pid"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    if _pid_is_live(pid):
        return False
    _write_json_atomic(candidate / "run_manifest.json", {
        "schema": "strict_r3_live_hourly_entry_producer_v1",
        "status": "failed_closed",
        "mode": "live",
        "exchange_order_submission": False,
        "execution_attempt_started": False,
        "terminalization": {
            "kind": "dead_pre_execution_producer_lease",
            "owner_pid": pid,
            "lease": lease,
            "execution_intent_marker_present": False,
            "reason": "owner process was absent before execution intent",
        },
    })
    return True


def _next_receipt(prefix: str) -> tuple[Path, int]:
    """Reserve the next immutable attempt after a terminal fail-closed run.

    A successful receipt is never retried: doing so could duplicate an order.
    A failed receipt has no exchange submission and can be safely followed by
    a later attempt for the same fresh decision.
    """
    attempt = 1
    while True:
        candidate = ARTIFACTS / f"{prefix}_v{attempt}"
        if not candidate.exists():
            return candidate, attempt
        manifest = candidate / "run_manifest.json"
        if not manifest.is_file():
            if _terminalize_dead_pre_execution_lease(candidate):
                attempt += 1
                continue
            raise FileExistsError(
                f"producer receipt is in progress or incomplete: {candidate}"
            )
        try:
            payload = json.loads(manifest.read_text())
            status = str(payload.get("status") or "")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise FileExistsError(
                f"producer receipt cannot be classified safely: {candidate}"
            ) from exc
        if status == "pass":
            raise SuccessfulProducerReceiptExists(candidate)
        if bool(payload.get("execution_attempt_started")):
            raise FileExistsError(
                "producer receipt reached the exchange-execution boundary and "
                f"requires explicit investigation: {candidate}"
            )
        attempt += 1


def _completed_live_decision_receipt(*, decision: pd.Timestamp) -> Path | None:
    """Return a successful live receipt from any sealed runtime namespace.

    Runtime-only reseals intentionally produce a fresh receipt namespace.  A
    restarted service must nevertheless never recompute or submit a decision
    that a predecessor runtime has already completed.  Failed pre-execution
    attempts remain retryable only when no successful live receipt exists.
    """
    tag = _utc(decision).strftime("%Y%m%dT%H%M%SZ")
    completed: list[Path] = []
    for manifest in sorted(
        ARTIFACTS.glob(f"strict_r3_live_hourly_producer_*_{tag}_v*/run_manifest.json")
    ):
        try:
            payload = json.loads(manifest.read_text())
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise FileExistsError(
                f"cannot classify prior producer receipt safely: {manifest}"
            ) from exc
        if (
            str(payload.get("status") or "") == "pass"
            and str(payload.get("mode") or "") == "live"
            and bool(payload.get("exchange_order_submission"))
            and _utc(payload.get("decision_ts")) == _utc(decision)
        ):
            completed.append(manifest.parent)
    return completed[-1] if completed else None


def _terminal_direct_execution_receipt(
    *, decision: pd.Timestamp, inference_bundle_hash: str
) -> Path | None:
    """Find a completed direct executor decision for the exact score bundle.

    This covers only an explicitly sealed JSON receipt, never an in-progress
    directory.  Binding it to the current inference hash prevents a stale or
    differently-scored review from suppressing a new producer decision.
    """
    tag = _utc(decision).strftime("%Y%m%dT%H%M%SZ")
    terminal: list[Path] = []
    for receipt in sorted(ARTIFACTS.glob(f"strict_r3_live_execution_*_{tag}_v*")):
        if not receipt.is_file():
            continue
        try:
            payload = json.loads(receipt.read_text())
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise FileExistsError(
                f"cannot classify direct execution receipt safely: {receipt}"
            ) from exc
        if (
            str(payload.get("schema") or "") == "strict_r3_kraken_live_execution_v1"
            and str(payload.get("mode") or "") == "live"
            and _utc(payload.get("decision_ts")) == _utc(decision)
            and str(payload.get("inference_bundle_sha256") or "") == inference_bundle_hash
            and isinstance(payload.get("actions"), list)
            and "state_sha256" in payload
        ):
            terminal.append(receipt)
    return terminal[-1] if terminal else None


def _run(command: list[str], *, log: Path) -> None:
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle,
                                   stderr=subprocess.STDOUT, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"stage failed ({completed.returncode}); see {log}")


def _successful_predecessor(
    *,
    decision: pd.Timestamp,
    bundle_hash: str,
    compatible_bundle_hashes: set[str],
    bootstrap: Path,
) -> Path:
    """Find the latest verified lock-step state, never a stale score.

    A live-window completion is preferred.  A separately labelled ``backfill``
    run or a completed ``stateful_recovery`` hour is admissible only as a
    causal *state bridge* after an operational interruption: it has zero
    exchange calls, the same sealed bundle, and a complete
    feature/Geometry/K9/portfolio successor state.  A recovery hour is
    eligible only after its enclosing recovery chain has completed; this
    prevents a live producer from ever consuming a partially rebuilt state.
    Neither form is executable evidence and neither is treated as a live
    decision.
    """
    def _validated_candidate(path: Path) -> tuple[pd.Timestamp, int, Path] | None:
        """Return one verified causal state predecessor, or ``None``.

        The explicitly configured bootstrap predecessor is normally the
        direct, hash-bound predecessor of the fresh hour.  Validate that path
        first rather than scanning the historical artifact estate at the live
        decision boundary.  The broad scan below remains a fail-closed
        recovery fallback when that exact predecessor is unavailable.
        """
        try:
            path = path.resolve()
            if ARTIFACTS.resolve() not in path.parents or path.name != "run_manifest.json":
                return None
            payload = json.loads(path.read_text())
            timestamp = _utc(payload["decision_ts"])
            is_live = "_live_" in path.parent.name
            is_backfill = "_backfill_" in path.parent.name
            is_recovery = path.parent.parent.name.startswith("hour_") and (
                path.parent.parent.parent.name.startswith("strict_r3_stateful_recovery_")
            )
            recovery_complete = False
            if is_recovery:
                hour_receipt = path.parent.parent
                recovery_root = hour_receipt.parent
                hour_manifest = json.loads(
                    (hour_receipt / "recovery_hour_manifest.json").read_text()
                )
                recovery_manifest = json.loads(
                    (recovery_root / "run_manifest.json").read_text()
                )
                recovery_complete = (
                    str(hour_manifest.get("schema")) == "strict_r3_stateful_recovery_v1"
                    and str(hour_manifest.get("status")) == "complete"
                    and _utc(hour_manifest.get("decision_ts")) == timestamp
                    and str(hour_manifest.get("run")) == str(path.parent.relative_to(ROOT))
                    and int(hour_manifest.get("exchange_calls", -1)) == 0
                    and bool(hour_manifest.get("order_submission_enabled")) is False
                    and str(recovery_manifest.get("schema")) == "strict_r3_stateful_recovery_v1"
                    and str(recovery_manifest.get("status")) == "complete"
                    and str(recovery_manifest.get("final_run")) == str(path.parent.relative_to(ROOT))
                    and int(recovery_manifest.get("exchange_calls", -1)) == 0
                    and bool(recovery_manifest.get("order_submission_enabled")) is False
                )
            valid_completion = bool(payload.get("completed_within_live_decision_window")) if is_live else (
                is_backfill
                and str(payload.get("mode")) == "shadow-only"
                and int(payload.get("exchange_calls", -1)) == 0
            ) or recovery_complete
            geometry_state_manifest = (
                path.parent / "cycle/score/geometry_k9_state/run_manifest.json"
            )
            # Geometry/K9 is explicitly lock-step state, not a stateless score
            # feature.  A state emitted for hour t is usable only to score the
            # immediately following hour.  Treating any older completed run as
            # a generic predecessor can otherwise let a restart skip an hour
            # and fail much later inside scoring (after expensive source and
            # feature work).  Validate this boundary before accepting either a
            # direct bootstrap or the archival recovery fallback.
            geometry_next_decision: pd.Timestamp | None = None
            if geometry_state_manifest.is_file():
                geometry_payload = json.loads(geometry_state_manifest.read_text())
                geometry_next_decision = _utc(geometry_payload["next_decision_ts"])
            if (
                timestamp < decision
                and str(payload.get("hashes", {}).get("inference_bundle"))
                in compatible_bundle_hashes
                and valid_completion
                and (path.parent / "feature_state/bundle/state_bundle_manifest.json").is_file()
                and (path.parent / "cycle/next_portfolio_state.json").is_file()
                and (path.parent / "cycle/score/geometry_k9_state/causal_geometry_k9_history.parquet").is_file()
                and geometry_next_decision == decision
            ):
                # Prefer a true live completion when two receipts share a timestamp.
                return timestamp, 1 if is_live else 0, path.parent
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return None
        return None

    # This is the normal live path: one explicitly configured, hash-bound
    # predecessor.  It avoids reading every historical recovery manifest
    # while keeping exactly the same completion, temporal, state and bundle
    # checks as the fallback selector.
    direct = _validated_candidate(bootstrap / "run_manifest.json")
    if direct is not None:
        return direct[2]

    # A configured predecessor is a lock-step contract, not a hint.  If it
    # exists but cannot advance Geometry/K9 into this exact decision, scanning
    # the archival estate is both slow and unsafe: it may mask an interrupted
    # hour and only fail after expensive source/feature work.  Fail closed so
    # recovery must materialise the missing hour explicitly.  A missing
    # bootstrap path is retained for offline recovery tooling, which may still
    # use the bounded archival selector below.
    direct_manifest = bootstrap / "run_manifest.json"
    if direct_manifest.is_file():
        raise RuntimeError(
            "configured direct predecessor is not a verified lock-step "
            f"state for {decision.isoformat()}: {bootstrap}; recover the "
            "missing hour before live processing"
        )

    candidates: list[tuple[pd.Timestamp, int, Path]] = []
    manifests = list(ARTIFACTS.glob("strict_r3_successor_*_*/run_manifest.json"))
    manifests.extend(
        ARTIFACTS.glob("strict_r3_stateful_recovery_*/hour_*/run/run_manifest.json")
    )
    for path in manifests:
        candidate = _validated_candidate(path)
        if candidate is not None:
            candidates.append(candidate)
    if candidates:
        return max(candidates, key=lambda item: (item[0], item[1]))[2]
    return bootstrap


def _advance_direct_bootstrap(
    args: argparse.Namespace,
    result: dict[str, object],
) -> str | None:
    """Carry a completed hour forward as the next hour's direct predecessor.

    The live loop is intentionally long-lived.  Leaving its startup bootstrap
    fixed after a successful cycle forces the following hour to scan the whole
    historical artifact estate just to rediscover the immediately preceding
    state.  That is slow and can prevent a fresh decision from starting.

    This helper only updates the in-memory hint.  The normal predecessor
    validator still rechecks the bundle hash, temporal adjacency, Geometry/K9
    continuity, feature-state bundle, and portfolio state before it can be
    consumed.
    """
    raw = result.get("hourly_run")
    if not isinstance(raw, str) or not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    if ROOT not in candidate.parents or not (candidate / "run_manifest.json").is_file():
        return None
    args.bootstrap_previous_run = str(candidate.relative_to(ROOT))
    return args.bootstrap_previous_run


def _verified_runtime_reseal_predecessors(
    *,
    execution_bundle: Path,
    current_bundle: Path,
    current_bundle_hash: str,
    current_payload: dict[str, object],
) -> set[str]:
    """Permit an exact state predecessor only for a declared runtime-only reseal.

    This is intentionally stricter than a generic compatibility escape hatch:
    all non-runtime-code bundle content and every unchanged code hash must be
    identical.  The execution contract names the one permissible parent and
    the sole code paths allowed to differ.
    """
    execution = json.loads(execution_bundle.read_text())
    bridges = execution.get("runtime_reseal_predecessors") or []
    if not isinstance(bridges, list):
        raise ValueError("runtime_reseal_predecessors must be a list")
    compatible = {str(current_bundle_hash)}
    for bridge in bridges:
        if not isinstance(bridge, dict):
            raise ValueError("runtime reseal bridge must be an object")
        # Execution contracts retain the append-only history of runtime-only
        # reseals.  A bridge for an earlier successor is provenance, not an
        # assertion about this active bundle.  Validate only the bridge that
        # explicitly targets the current bundle; an absent matching bridge
        # simply means no predecessor state is compatible and later selection
        # remains fail-closed.
        if str(bridge.get("current_inference_bundle_sha256") or "") != str(current_bundle_hash):
            continue
        relative = str(bridge.get("predecessor_inference_bundle") or "")
        expected = str(bridge.get("predecessor_inference_bundle_sha256") or "")
        allowed = set(map(str, bridge.get("allowed_runtime_code_paths") or []))
        added = set(map(str, bridge.get("added_runtime_code_paths") or []))
        predecessor = (ROOT / relative).resolve()
        if ROOT not in predecessor.parents or not predecessor.is_file() or not expected:
            raise ValueError("runtime reseal predecessor is invalid")
        if _sha(predecessor) != expected:
            raise ValueError("runtime reseal predecessor hash mismatch")
        # Do not import historical sources here: this path exists specifically
        # to bridge a hash-bound predecessor after a reviewed source-only
        # update. The current bundle has already passed normal full source-hash
        # validation; the static predecessor contract is proved identical
        # below and cannot supply executable authority.
        prior_payload = _resolve_historical_static_contract(predecessor)
        if not allowed and not added:
            raise ValueError("runtime reseal bridge has no declared source delta")
        current_code = dict(current_payload.get("runtime_code_sha256") or {})
        prior_code = dict(prior_payload.get("runtime_code_sha256") or {})
        prior_paths = set(prior_code)
        current_paths = set(current_code)
        unexpected_added = current_paths - prior_paths - added
        unexpected_removed = prior_paths - current_paths
        if unexpected_added or unexpected_removed:
            raise ValueError(
                "runtime reseal changes an undeclared sealed source path: "
                f"added={sorted(unexpected_added)} removed={sorted(unexpected_removed)}"
            )
        if added & prior_paths:
            raise ValueError("runtime reseal added source path already exists in parent")
        if added - current_paths:
            raise ValueError("runtime reseal omits a declared added source path")
        changed = {
            key for key in (current_paths & prior_paths)
            if str(current_code[key]) != str(prior_code[key])
        }
        if changed != allowed:
            raise ValueError(
                "runtime reseal code delta differs from its declared scope: "
                f"{sorted(changed)} != {sorted(allowed)}"
            )
        current_static = _static_contract_for_runtime_bridge(current_payload)
        prior_static = _static_contract_for_runtime_bridge(prior_payload)
        calibration_exception = _approved_calibration_static_match(
            current_payload=current_payload,
            prior_payload=prior_payload,
            bridge=bridge,
        )
        if current_static != prior_static and not calibration_exception:
            raise ValueError("runtime reseal changes model, feature, or policy contract")
        compatible.add(expected)
    return compatible


def _universe_symbols(universe: Path) -> list[str]:
    payload = json.loads(universe.read_text())
    source_map = payload.get("source_map", {}) if isinstance(payload, dict) else {}
    if not isinstance(source_map, dict) or not source_map:
        raise RuntimeError(f"frozen universe has no source_map: {universe}")
    return sorted(str(symbol) for symbol in source_map)


def _read_15m_cache_window(symbol: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Read the canonical raw-first 15-minute cache over a tiny PIT window."""
    name = f"{symbol.lower().replace('/', '')}_15m.parquet"
    columns = ["open", "high", "low", "close", "volume", "exchange_observed"]
    frames: list[pd.DataFrame] = []
    for root in (
        ROOT / "data_perp/exchanges/krakenfutures/raw/ohlcv_15m",
        ROOT / "15m_ohlcv_perp",
    ):
        path = root / name
        if not path.exists():
            continue
        filters = [
            ("__index_level_0__", ">=", pd.Timestamp(index.min()).to_pydatetime()),
            ("__index_level_0__", "<=", pd.Timestamp(index.max()).to_pydatetime()),
        ]
        try:
            frame = pd.read_parquet(path, columns=columns, filters=filters)
        except Exception:
            try:
                frame = pd.read_parquet(path, columns=columns[:-1], filters=filters)
            except Exception:
                continue
        if not isinstance(frame.index, pd.DatetimeIndex):
            continue
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        frame = frame.loc[~frame.index.isna()]
        if "exchange_observed" not in frame:
            frame["exchange_observed"] = pd.Series(
                pd.NA, index=frame.index, dtype="boolean",
            )
        else:
            frame["exchange_observed"] = frame["exchange_observed"].astype("boolean")
        frames.append(frame.reindex(index))
    if not frames:
        return pd.DataFrame(index=index, columns=columns)
    # The raw exchange cache wins at exact timestamps; the shared cache is a
    # causal fill source only.  This mirrors the frozen feature source.
    values = frames[0]
    for frame in frames[1:]:
        values = values.combine_first(frame)
    return values.reindex(index)


def _assess_15m_coverage(*, decision: pd.Timestamp, symbols: list[str]) -> pd.DataFrame:
    """Classify completed signal-hour readiness without touching future bars."""
    signal = decision - pd.Timedelta(hours=1)
    signal_index = pd.date_range(signal, decision - pd.Timedelta(minutes=15), freq="15min")
    decision_index = pd.DatetimeIndex([decision])
    # Reuse the exact decision-open source adapter so this receipt reports the
    # same executable-open condition as the candidate materialiser.
    from scripts.run_tp6_sl4_exact170_canonical_consensus import _read_downloaded_15m_decision_open

    def _assess_symbol(symbol: str) -> dict[str, object]:
        """Read one symbol's fixed causal window; independent across symbols."""
        frame = _read_15m_cache_window(symbol, signal_index)
        finite = frame[["open", "high", "low", "close"]].notna().all(axis=1)
        flat_zero = (
            finite
            & frame["open"].eq(frame["high"])
            & frame["high"].eq(frame["low"])
            & frame["low"].eq(frame["close"])
            & pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0).le(0.0)
        )
        observed = frame["exchange_observed"].astype("boolean")
        # Unknown legacy rows retain the former conservative rule.  A true
        # observed bit makes a flat no-trade candle source-valid; false marks
        # a local fill explicitly.
        synthetic = flat_zero & ~observed.fillna(False)
        try:
            decision_open = _read_downloaded_15m_decision_open(
                symbol, decision_index,
            )
            has_decision_open = bool(pd.to_numeric(decision_open, errors="coerce").notna().iloc[0])
        except Exception:
            has_decision_open = False
        return {
            "symbol": symbol,
            "signal_ts": signal.isoformat(),
            "decision_ts": decision.isoformat(),
            "expected_15m_bars": int(len(signal_index)),
            "finite_15m_bars": int(finite.sum()),
            "exchange_observed_15m_bars": int((finite & observed.fillna(False)).sum()),
            "locally_filled_flat_15m_bars": int((finite & observed.eq(False).fillna(False)).sum()),
            "synthetic_flat_15m_bars": int(synthetic.sum()),
            "missing_15m_bar": bool((~finite).any()),
            "synthetic_flat_bar": bool(synthetic.any()),
            "missing_decision_open": not has_decision_open,
            "feature_source_ready": bool(finite.all() and not synthetic.any()),
        }

    # Each symbol reads an independent, fixed four-bar causal window.  Bound
    # parallelism avoids a 170-file serial Parquet metadata scan on the live
    # critical path while preserving both the source contract and universe
    # order.  executor.map returns results in input order deterministically.
    workers = min(16, max(1, len(symbols)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(_assess_symbol, symbols))
    return pd.DataFrame.from_records(records)


def _run_15m_refresh_partitions(
    *,
    start: pd.Timestamp,
    decision: pd.Timestamp,
    universe: Path,
    out_dir: Path,
    prefix: str,
    symbols: list[str] | None = None,
) -> list[int]:
    commands: list[tuple[int, Any, subprocess.Popen[str]]] = []
    for partition in range(16):
        command = [
            sys.executable, str(ROOT / "scripts/download_kraken_15m_hf.py"),
            "--target-free-manifest", str(universe),
            "--force-start", start.isoformat(),
            "--force-end", decision.isoformat(),
            "--hf-data-dir", "15m_ohlcv_perp",
            "--partition-count", "16", "--partition-id", str(partition),
            "--sleep-seconds", "0", "--rate-limit-ms", "1000",
        ]
        if symbols:
            for symbol in symbols:
                command.extend(["--symbol", symbol])
        log = out_dir / f"{prefix}_partition_{partition:02d}.log"
        handle = log.open("w", encoding="utf-8")
        commands.append((partition, handle, subprocess.Popen(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True,
        )))
    failed: list[int] = []
    for partition, handle, process in commands:
        result = process.wait()
        handle.close()
        if result:
            failed.append(partition)
    return failed


def _refresh_15m(
    *,
    decision: pd.Timestamp,
    universe: Path,
    out_dir: Path,
    settled_retry_schedule_seconds: tuple[float, ...] = (30.0, 60.0, 120.0, 180.0),
) -> dict[str, object]:
    """Refresh and audit the exact completed signal hour.

    A refresh with process-level success but incomplete signal bars is
    deliberately labelled ``partial``.  It remains row-local fail-closed: the
    scorer may use ready symbols while incomplete symbols cannot be admitted.
    """
    out_dir.mkdir(parents=True, exist_ok=False)
    start = decision - pd.Timedelta(hours=2)
    symbols = _universe_symbols(universe)
    refresh_started = time.monotonic()
    failed_initial = _run_15m_refresh_partitions(
        start=start, decision=decision, universe=universe, out_dir=out_dir,
        prefix="initial",
    )
    before = _assess_15m_coverage(decision=decision, symbols=symbols)
    before.to_parquet(out_dir / "coverage_before_retry.parquet", index=False)
    print(json.dumps({
        "event": "15m_coverage_before_retry",
        "decision_ts": decision.isoformat(),
        "feature_source_ready": int(before["feature_source_ready"].sum()),
        "missing_15m_bar": int(before["missing_15m_bar"].sum()),
        "synthetic_flat_bar": int(before["synthetic_flat_bar"].sum()),
        "missing_decision_open": int(before["missing_decision_open"].sum()),
    }), flush=True)
    after = before
    retry_attempts: list[dict[str, object]] = []
    failed_retry: list[int] = []
    for target_elapsed in sorted({max(0.0, float(value)) for value in settled_retry_schedule_seconds}):
        retry_symbols = after.loc[
            ~after["feature_source_ready"], "symbol",
        ].astype(str).tolist()
        if not retry_symbols:
            break
        wait_seconds = max(0.0, target_elapsed - (time.monotonic() - refresh_started))
        if wait_seconds:
            time.sleep(wait_seconds)
        failed = _run_15m_refresh_partitions(
            start=start, decision=decision, universe=universe, out_dir=out_dir,
            prefix=f"settled_retry_{int(target_elapsed):03d}s",
            symbols=retry_symbols,
        )
        failed_retry.extend(failed)
        after = _assess_15m_coverage(decision=decision, symbols=symbols)
        artifact = f"coverage_after_retry_{int(target_elapsed):03d}s.parquet"
        after.to_parquet(out_dir / artifact, index=False)
        retry_attempts.append({
            "target_elapsed_seconds": target_elapsed,
            "actual_elapsed_seconds": round(time.monotonic() - refresh_started, 3),
            "symbols_retried": retry_symbols,
            "symbol_count": int(len(retry_symbols)),
            "failed_partitions": failed,
            "coverage_artifact": artifact,
            "feature_source_ready": int(after["feature_source_ready"].sum()),
        })
        print(json.dumps({
            "event": "15m_coverage_after_retry",
            "decision_ts": decision.isoformat(),
            "target_elapsed_seconds": target_elapsed,
            "symbols_retried": int(len(retry_symbols)),
            "feature_source_ready": int(after["feature_source_ready"].sum()),
            "missing_15m_bar": int(after["missing_15m_bar"].sum()),
            "synthetic_flat_bar": int(after["synthetic_flat_bar"].sum()),
        }), flush=True)
    after.to_parquet(out_dir / "coverage_after_retry.parquet", index=False)
    summary = {
        "symbols": int(len(symbols)),
        "feature_source_ready": int(after["feature_source_ready"].sum()),
        "missing_15m_bar": int(after["missing_15m_bar"].sum()),
        "synthetic_flat_bar": int(after["synthetic_flat_bar"].sum()),
        "missing_decision_open": int(after["missing_decision_open"].sum()),
    }
    failed = sorted(set(failed_initial + failed_retry))
    receipt: dict[str, object] = {
        "schema": "strict_r3_live_hourly_15m_refresh_v2",
        "decision_ts": decision.isoformat(), "start": start.isoformat(),
        "end": decision.isoformat(), "partitions": 16,
        "initial_failed_partitions": failed_initial,
        "settled_retry_schedule_seconds": list(settled_retry_schedule_seconds),
        "retry_attempts": retry_attempts,
        "retry_failed_partitions": sorted(set(failed_retry)),
        "failed_partitions": failed,
        "coverage_before_retry": {
            "feature_source_ready": int(before["feature_source_ready"].sum()),
            "missing_15m_bar": int(before["missing_15m_bar"].sum()),
            "synthetic_flat_bar": int(before["synthetic_flat_bar"].sum()),
            "missing_decision_open": int(before["missing_decision_open"].sum()),
        },
        "coverage_after_retry": summary,
        "coverage_artifacts": {
            "before_retry": "coverage_before_retry.parquet",
            "after_retry": "coverage_after_retry.parquet",
        },
        "source_contract": (
            "raw-first 15m cache; exchange_observed=true is accepted even for "
            "flat zero-volume candles; locally-filled or legacy-unknown flat "
            "candles remain unavailable; no post-decision bar is read"
        ),
        "status": "fail_closed" if failed else (
            "pass" if summary["feature_source_ready"] == len(symbols) else "partial"
        ),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if failed:
        raise RuntimeError(f"15m source refresh failed in partitions {failed}")
    return receipt


def _refresh_official_hourly_analytics(
    *,
    decision: pd.Timestamp,
    universe: Path,
    log: Path,
) -> None:
    """Append official mark/OI/L2 analytics for the just-closed signal hour.

    The frozen target-free grid reads these fields from
    ``frozen_contract_backfill_hourly``.  Refreshing only 15-minute OHLCV
    leaves the grid with a missing signal-hour spread and incorrectly rejects
    otherwise valid candidates.  The existing refresh utility preserves prior
    values with ``combine_first`` and only fills the declared one-hour window.
    """
    start = decision - pd.Timedelta(hours=1)
    _run([
        sys.executable,
        str(ROOT / "scripts/backfill_kraken_frozen_contract_inputs.py"),
        "--symbols-json", str(universe),
        "--out-dir", "data_perp/exchanges/krakenfutures/frozen_contract_backfill_hourly",
        "--start", start.isoformat(),
        "--end", decision.isoformat(),
        "--workers", "16",
        "--include-orderbook-analytics",
    ], log=log)


def _refresh_oi_funding_sidecars(
    *,
    decision: pd.Timestamp,
    universe: Path,
    out_dir: Path,
) -> None:
    """Append causal OI/funding observations before live feature materialisation.

    ``post_liquidation_rebound_score`` depends on a funding-change primitive.
    The frozen-contract order-book refresh does not supply that primitive, so
    the live producer must refresh the established OI/funding sidecars as a
    separate source family.  The sidecar utility shifts observations by its
    declared one-hour availability rule; this producer never reads a value
    after ``decision``.  Partitions are independent and bounded so a slow
    product cannot serialize all 170 symbols.

    A product-level API failure is recorded in its partition receipt.  It is
    deliberately not imputed here: the canonical per-row feature gate later
    rejects that product if the frozen field remains unavailable.
    """
    out_dir.mkdir(parents=True, exist_ok=False)
    # Two completed observation hours are sufficient for the one-hour funding
    # change used by the canonical composite; request a small overlap so the
    # sidecar merge remains gap-filling and recovery after a transient outage
    # is deterministic.
    start = decision - pd.Timedelta(hours=3)
    commands: list[tuple[int, Path, Any, subprocess.Popen[str]]] = []
    for partition in range(16):
        log = out_dir / f"partition_{partition:02d}.log"
        handle = log.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(ROOT / "scripts/backfill_kraken_oi_funding_sidecars.py"),
            "--feature-dir", "data_perp/features",
            "--symbols-file", str(universe),
            "--perp-root", "data_perp/exchanges/krakenfutures",
            "--out-dir", str(out_dir / f"partition_{partition:02d}"),
            "--quarantine-corrupt-sidecars-dir",
            "data_perp/exchanges/krakenfutures/corrupt_sidecars",
            "--start-ts", start.isoformat(),
            "--end-ts", decision.isoformat(),
            # Each stable symbol partition fetches its OI/funding endpoints as
            # one bounded batch, then commits a deterministic local merge.
            # Two in-flight public requests per partition keep the critical
            # path below the former serial pair without changing the sidecar
            # availability or gap-filling contract.
            "--workers", "2",
            "--batch-append",
            "--partition-count", "16",
            "--partition-id", str(partition),
        ]
        commands.append((
            partition,
            log,
            handle,
            subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            ),
        ))
    # A public-endpoint worker can occasionally leave a Python process alive
    # after it has atomically committed a COMPLETE partition manifest (for
    # example while a requests worker is unwinding).  Waiting serially on that
    # orphan blocks every later live decision.  The manifest is the durable
    # source receipt: accept it only when it proves all jobs completed without
    # errors, then terminate the now-superfluous child.  All other workers are
    # bounded as one refresh group and fail closed at the deadline.
    refresh_timeout_seconds = 150.0
    manifest_grace_seconds = 2.0
    started = time.monotonic()
    pending = {partition: (handle, process) for partition, _, handle, process in commands}
    completed: dict[int, dict[str, Any]] = {}

    def _read_manifest(partition: int) -> dict[str, Any]:
        manifest_path = out_dir / f"partition_{partition:02d}" / "backfill_manifest.json"
        if not manifest_path.is_file():
            return {}
        try:
            return dict(json.loads(manifest_path.read_text()))
        except (OSError, ValueError, json.JSONDecodeError):
            return {"manifest_parse_error": True}

    def _manifest_complete(payload: dict[str, Any]) -> bool:
        counts = payload.get("result_counts") or {}
        return (
            payload.get("status") == "COMPLETE"
            and not bool(counts.get("error", 0))
            and int(counts.get("jobs", 0)) == int(counts.get("ok", 0)) + int(counts.get("empty", 0)) + int(counts.get("skipped", 0))
        )

    while pending and time.monotonic() - started < refresh_timeout_seconds:
        for partition, (handle, process) in list(pending.items()):
            returncode = process.poll()
            payload = _read_manifest(partition)
            complete_manifest = _manifest_complete(payload)
            if returncode is None and not complete_manifest:
                continue
            accepted_complete_manifest = False
            terminated_after_complete_manifest = False
            if returncode is None and complete_manifest:
                # Give the worker a brief opportunity to exit normally.  If it
                # does not, all durable work is already committed and keeping
                # it alive can only delay the next decision.
                try:
                    returncode = process.wait(timeout=manifest_grace_seconds)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5.0)
                    accepted_complete_manifest = True
                    terminated_after_complete_manifest = True
            if returncode is None:
                continue
            handle.close()
            pending.pop(partition)
            completed[partition] = {
                "partition": partition,
                "returncode": int(returncode),
                "manifest_status": payload.get("status"),
                "result_counts": payload.get("result_counts"),
                "accepted_complete_manifest": accepted_complete_manifest,
                "terminated_after_complete_manifest": terminated_after_complete_manifest,
            }
        if pending:
            time.sleep(0.10)

    for partition, (handle, process) in list(pending.items()):
        payload = _read_manifest(partition)
        complete_manifest = _manifest_complete(payload)
        process.terminate()
        try:
            returncode = process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = process.wait(timeout=5.0)
        handle.close()
        completed[partition] = {
            "partition": partition,
            "returncode": int(returncode),
            "manifest_status": payload.get("status"),
            "result_counts": payload.get("result_counts"),
            "accepted_complete_manifest": bool(complete_manifest),
            "terminated_after_complete_manifest": bool(complete_manifest),
            "timeout_seconds": refresh_timeout_seconds,
            "timed_out": True,
        }

    partition_audits = [completed[partition] for partition in sorted(completed)]
    failed = [
        int(audit["partition"])
        for audit in partition_audits
        if not bool(audit.get("accepted_complete_manifest"))
        and int(audit.get("returncode", 1)) != 0
    ]
    receipt = {
        "schema": "strict_r3_live_hourly_oi_funding_refresh_v1",
        "decision_ts": decision.isoformat(),
        "start": start.isoformat(),
        "end": decision.isoformat(),
        "partitions": 16,
        "refresh_timeout_seconds": refresh_timeout_seconds,
        "failed_partitions": failed,
        "partition_audits": partition_audits,
        "source_contract": "Kraken observed OI/funding shifted +1h; no imputation",
        "status": "pass" if not failed else "fail_closed",
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    if failed:
        raise RuntimeError(f"OI/funding source refresh failed in partitions {failed}")


def run_once(args: argparse.Namespace, *, decision: pd.Timestamp) -> dict[str, Any]:
    decision = _utc(decision)
    invoked_at = pd.Timestamp.now(tz="UTC")
    invocation_monotonic = time.monotonic()
    prior_completed = _completed_live_decision_receipt(decision=decision)
    if prior_completed is not None:
        raise SuccessfulProducerReceiptExists(prior_completed)
    bundle_hash, warmed_bundle, warm_cache_hit = _load_warm_bundle(
        ROOT / args.inference_bundle
    )
    freshness_seconds = min(
        float(dict(warmed_bundle.payload).get("live_decision_freshness_seconds", 900.0)),
        LIVE_EXECUTION_DEADLINE_SECONDS,
    )
    decision_age_seconds = float((invoked_at - decision).total_seconds())
    if decision_age_seconds > freshness_seconds:
        tag = decision.strftime("%Y%m%dT%H%M%SZ")
        runtime_tag = _runtime_tag(args.inference_bundle)
        receipt, _ = _next_receipt(
            f"strict_r3_live_hourly_producer_{runtime_tag}_{tag}"
        )
        receipt.mkdir(parents=True)
        result = {
            "schema": "strict_r3_live_hourly_entry_producer_v1",
            "decision_ts": decision.isoformat(),
            "started_at": invoked_at.isoformat(),
            "completed_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "status": "expired",
            "mode": "live",
            "exchange_order_submission": False,
            "execution_attempt_started": False,
            "inference_bundle_warm_cache_hit": bool(warm_cache_hit),
            "decision_age_seconds": decision_age_seconds,
            "live_decision_freshness_seconds": freshness_seconds,
            "reason": "decision_expired_before_source_refresh",
        }
        _write_json_atomic(receipt / "run_manifest.json", result)
        return result
    direct_terminal = _terminal_direct_execution_receipt(
        decision=decision, inference_bundle_hash=bundle_hash
    )
    if direct_terminal is not None:
        raise TerminalExecutionReceiptExists(direct_terminal)
    tag = decision.strftime("%Y%m%dT%H%M%SZ")
    runtime_tag = _runtime_tag(args.inference_bundle)
    receipt_prefix = f"strict_r3_live_hourly_producer_{runtime_tag}_{tag}"
    receipt, attempt = _next_receipt(receipt_prefix)
    receipt.mkdir(parents=True)
    started = pd.Timestamp.now(tz="UTC")
    _write_json_atomic(receipt / "producer_lease.json", {
        "schema": "strict_r3_live_producer_lease_v1",
        "pid": os.getpid(),
        "started_at": started.isoformat(),
        "decision_ts": decision.isoformat(),
        "execution_attempt_started": False,
    })
    result: dict[str, Any] = {
        "schema": "strict_r3_live_hourly_entry_producer_v1",
        "decision_ts": decision.isoformat(), "started_at": started.isoformat(),
        "scheduler_invoked_at": invoked_at.isoformat(),
        "receipt_reserved_at": started.isoformat(),
        "scheduler_to_receipt_seconds": float(time.monotonic() - invocation_monotonic),
        "inference_bundle_warm_cache_hit": bool(warm_cache_hit),
        "mode": "live", "status": "failed_closed", "exchange_order_submission": False,
    }
    try:
        bundle = ROOT / args.inference_bundle
        # A versioned successor may be a narrow, hash-bound overlay on an
        # immutable base bundle. Resolve it through the same validator used by
        # scoring, rather than assuming every operational bundle stores a full
        # `paths` mapping in its own JSON file.
        bundle_payload = dict(warmed_bundle.payload)
        compatible_bundle_hashes = _verified_runtime_reseal_predecessors(
            execution_bundle=ROOT / args.execution_bundle,
            current_bundle=bundle,
            current_bundle_hash=bundle_hash,
            current_payload=bundle_payload,
        )
        predecessor = _successful_predecessor(
            decision=decision,
            bundle_hash=bundle_hash,
            compatible_bundle_hashes=compatible_bundle_hashes,
            bootstrap=ROOT / args.bootstrap_previous_run,
        )
        result["predecessor"] = str(predecessor.relative_to(ROOT))
        result["compatible_predecessor_bundle_hashes"] = sorted(
            compatible_bundle_hashes
        )
        source_started_at = pd.Timestamp.now(tz="UTC")
        source_started_monotonic = time.monotonic()
        refresh_dir = ARTIFACTS / f"strict_r3_live_15m_refresh_{runtime_tag}_{tag}_v{attempt}"
        oi_funding_refresh_dir = (
            ARTIFACTS / f"strict_r3_live_oi_funding_refresh_{runtime_tag}_{tag}_v{attempt}"
        )
        universe = ROOT / str(bundle_payload["paths"]["frozen_universe_manifest"])
        # These branches write disjoint append-only stores.  They must all
        # pass before feature materialisation, but there is no causal reason
        # to serialise three independent Kraken source waits.
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="strict-r3-source") as pool:
            futures = {
                "refresh_15m": pool.submit(
                    _refresh_15m,
                    decision=decision, universe=universe, out_dir=refresh_dir,
                    settled_retry_schedule_seconds=_retry_schedule(
                        getattr(args, "settled_retry_schedule_seconds", None),
                        first_retry_seconds=getattr(args, "settled_retry_seconds", None),
                    ),
                ),
                "oi_funding": pool.submit(
                    _refresh_oi_funding_sidecars,
                    decision=decision, universe=universe, out_dir=oi_funding_refresh_dir,
                ),
                "official_hourly_analytics": pool.submit(
                    _refresh_official_hourly_analytics,
                    decision=decision, universe=universe,
                    log=receipt / "official_hourly_analytics_refresh.log",
                ),
            }
            refresh_15m = futures["refresh_15m"].result()
            futures["oi_funding"].result()
            futures["official_hourly_analytics"].result()
        result["refresh_15m"] = refresh_15m
        result["source_parallel"] = {
            "started_at": source_started_at.isoformat(),
            "completed_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "duration_seconds": float(time.monotonic() - source_started_monotonic),
            "branches": ["15m", "oi_funding", "official_hourly_analytics"],
            "all_pass_required_before_feature_materialization": True,
        }
        elapsed_after_sources = float(
            (pd.Timestamp.now(tz="UTC") - decision).total_seconds()
        )
        if elapsed_after_sources > freshness_seconds:
            raise ExpiredLiveDecision(
                "decision_expired_after_source_refresh: "
                f"age={elapsed_after_sources:.3f}s limit={freshness_seconds:.3f}s"
            )
        run_dir = ARTIFACTS / f"strict_r3_successor_{runtime_tag}_live_{tag}_v{attempt}"
        feature_contract = str(bundle_payload["runtime"]["feature_state"]["contract_sha256"])
        feature_tail = str(bundle_payload["runtime"]["feature_state"]["panel_tail_hours"])
        # A state re-receipt is allowed only when its sealed predecessor is the
        # exact successful run selected above.  The shadow runner independently
        # verifies both manifests and the byte-identical operator payload before
        # accepting this source, so this cannot become a rolling bridge.
        predecessor_state_bundle = (predecessor / "feature_state/bundle").resolve()
        feature_state_bundle = predecessor_state_bundle
        state_reseal = dict(
            bundle_payload.get("runtime", {}).get("feature_state", {}).get(
                "one_time_state_reseal", {}
            ) or {}
        )
        if state_reseal:
            superseded = (ROOT / str(state_reseal.get("superseded_bundle") or "")).resolve()
            resealed = (ROOT / str(state_reseal.get("resealed_bundle") or "")).resolve()
            if predecessor_state_bundle == superseded:
                if ROOT.resolve() not in resealed.parents or not resealed.is_dir():
                    raise ValueError("sealed feature-state re-receipt bundle is unavailable")
                feature_state_bundle = resealed
                result["feature_state_reseal_source"] = str(
                    resealed.relative_to(ROOT)
                )
        _run([
            sys.executable, str(ROOT / "scripts/run_strict_r3_hourly_shadow_resume_v15.py"),
            "--inference-bundle", str(bundle.relative_to(ROOT)),
            "--portfolio-state-json", str((ROOT / args.live_state).relative_to(ROOT)),
            "--decision-ts", decision.isoformat(), "--out-dir", str(run_dir.relative_to(ROOT)),
            "--previous-shadow-run", str(predecessor.relative_to(ROOT)),
            "--feature-state-bundle", str(feature_state_bundle.relative_to(ROOT)),
            "--feature-state-contract-hash", feature_contract,
            "--feature-state-tail-hours", feature_tail,
            "--portfolio-state-reconciliation", "--enforce-live-wall-clock",
        ], log=receipt / "hourly_shadow.log")
        audit_dir = ARTIFACTS / f"strict_r3_live_hour_audit_{runtime_tag}_{tag}_v{attempt}"
        _run([
            sys.executable, str(ROOT / "scripts/audit_strict_r3_schema_v6_live_hour.py"),
            "--run", str(run_dir.relative_to(ROOT)),
            "--previous-run", str(predecessor.relative_to(ROOT)),
            "--out", str(audit_dir.relative_to(ROOT)), "--enforce-live-wall-clock",
        ], log=receipt / "live_hour_audit.log")
        checkpoint_dir = ARTIFACTS / f"strict_r3_live_runtime_checkpoint_{runtime_tag}_{tag}_v{attempt}"
        _run([
            sys.executable, str(ROOT / "scripts/checkpoint_strict_r3_runtime.py"), "create",
            "--run-dir", str(run_dir.relative_to(ROOT)),
            "--inference-bundle", str(bundle.relative_to(ROOT)),
            "--feature-state-bundle", str((run_dir / "feature_state/bundle").relative_to(ROOT)),
            # The checkpoint binds inputs at the decision boundary.  The next
            # portfolio state is post-auction and therefore newer than that
            # boundary; using it both fails the checkpoint and would be the
            # wrong lineage for an entry decision.
            "--portfolio-state", str((run_dir / "portfolio_reconciliation_state.json").relative_to(ROOT)),
            "--out-dir", str(checkpoint_dir.relative_to(ROOT)),
        ], log=receipt / "runtime_checkpoint.log")
        execution_dir = ARTIFACTS / f"strict_r3_live_execution_{runtime_tag}_{tag}_v{attempt}"
        # Initialise the exchange client/market definitions *before* crossing
        # the execution-intent boundary.  A prewarm failure is a normal
        # fail-closed pre-execution failure and can be retried safely; it must
        # never leave an ambiguous intent receipt behind.
        _load_warm_live_executor(ROOT / args.execution_bundle)
        # A failed process after this point could have reached the exchange.
        # The scheduler must leave it for explicit investigation rather than
        # automatically creating a same-hour retry that might duplicate an
        # entry.  Pre-execution failures remain safely retryable.
        _write_json_atomic(receipt / "execution_attempt_started.json", {
            "schema": "strict_r3_live_execution_intent_v1",
            "decision_ts": decision.isoformat(),
            "started_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "execution_dir": str(execution_dir.relative_to(ROOT)),
        })
        result["execution_attempt_started"] = True
        execution_started_at = pd.Timestamp.now(tz="UTC")
        try:
            execution_result = _execute_verified_hour_warm(
                execution_bundle=ROOT / args.execution_bundle,
                hourly_run=run_dir,
                state_path=ROOT / args.live_state,
                out=execution_dir,
                live_hour_audit=audit_dir,
                runtime_checkpoint=checkpoint_dir,
            )
        except Exception as exc:
            (receipt / "execution.log").write_text(json.dumps({
                "schema": "strict_r3_warm_live_execution_v1",
                "status": "failed",
                "started_at": execution_started_at.isoformat(),
                "failed_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }, indent=2) + "\n")
            raise
        (receipt / "execution.log").write_text(json.dumps({
            "schema": "strict_r3_warm_live_execution_v1",
            "status": "pass",
            "started_at": execution_started_at.isoformat(),
            "completed_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "execution_result": execution_result,
        }, indent=2, default=str) + "\n")
        result.update({
            "status": "pass", "exchange_order_submission": True,
            "hourly_run": str(run_dir.relative_to(ROOT)),
            "live_hour_audit": str(audit_dir.relative_to(ROOT)),
            "runtime_checkpoint": str(checkpoint_dir.relative_to(ROOT)),
            "execution_receipt": str(execution_dir.relative_to(ROOT)),
        })
    except Exception as exc:
        if isinstance(exc, ExpiredLiveDecision):
            result["status"] = "expired"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    result["completed_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    result["decision_age_at_completion_seconds"] = float(
        (pd.Timestamp.now(tz="UTC") - decision).total_seconds()
    )
    (receipt / "run_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-bundle", required=True)
    parser.add_argument("--execution-bundle", required=True)
    parser.add_argument("--live-state", required=True)
    parser.add_argument("--bootstrap-previous-run", required=True)
    parser.add_argument("--decision-ts")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--start-next-fresh-hour",
        action="store_true",
        help=(
            "Prewarm the persistent service now but leave the already-open "
            "hour observation-only; the first execution attempt is the next "
            "fresh UTC decision."
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--failed-retry-seconds", type=float, default=30.0,
        help=(
            "Delay before retrying a current-hour failure that occurred before "
            "the exchange-execution stage.  Execution-stage failures require "
            "explicit investigation and never auto-retry."
        ),
    )
    parser.add_argument(
        "--settled-retry-schedule-seconds", default="30,60,120,180",
        help=(
            "Comma-separated elapsed seconds after the initial source refresh "
            "at which only incomplete completed signal-hour symbols are retried."
        ),
    )
    parser.add_argument(
        "--settled-retry-seconds", type=float, default=None,
        help="Deprecated compatibility override for the first retry checkpoint.",
    )
    args = parser.parse_args()
    if args.loop and args.decision_ts:
        raise ValueError("--loop and --decision-ts are mutually exclusive")
    if args.start_next_fresh_hour and (not args.loop or args.decision_ts):
        raise ValueError("--start-next-fresh-hour requires --loop only")
    if args.decision_ts:
        print(json.dumps(run_once(args, decision=_utc(args.decision_ts)), sort_keys=True))
        return
    if not args.loop:
        raise ValueError("provide --decision-ts for one run or --loop for the service")
    # Parse and validate the sealed bundle before the next boundary.  A later
    # reseal invalidates this cache by content hash and forces another full
    # validation; this is latency optimisation, never a validation bypass.
    _load_warm_bundle(ROOT / args.inference_bundle)
    # Do the heavyweight live-execution import, sealed contract parse and
    # Kraken market-definition load while the service is idle.  The next
    # candle then enters only the small, already-warm verified execution path.
    # This creates no order, reads no candidate, and has no admission authority.
    _load_warm_live_executor(ROOT / args.execution_bundle)
    # A recovered service may be restarted after xx:00.  The sealed bundle,
    # not the scheduler's start second, owns the bounded freshness decision.
    # Running the current decision once here lets a tested recovery complete a
    # still-valid current-hour candidate; it never revisits a prior hour.
    last: pd.Timestamp | None = (
        pd.Timestamp.now(tz="UTC").floor("h")
        if args.start_next_fresh_hour
        else None
    )
    retry_after: pd.Timestamp | None = None
    while True:
        now = pd.Timestamp.now(tz="UTC")
        decision = now.floor("h")
        should_attempt = decision != last or (
            decision == last and retry_after is not None and now >= retry_after
        )
        if should_attempt:
            try:
                result = run_once(args, decision=decision)
            except (SuccessfulProducerReceiptExists, TerminalExecutionReceiptExists) as exc:
                result = {
                    "schema": "strict_r3_live_hourly_entry_producer_v1",
                    "decision_ts": decision.isoformat(),
                    "status": "already_completed",
                    "receipt": str(exc.receipt.relative_to(ROOT)),
                    "exchange_order_submission": False,
                }
            except Exception as exc:
                # A concurrent predecessor can temporarily own the immutable
                # receipt before it writes its terminal manifest.  Preserve a
                # fail-closed event and retry only after a bounded delay.
                result = {
                    "schema": "strict_r3_live_hourly_entry_producer_v1",
                    "decision_ts": decision.isoformat(),
                    "status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "exchange_order_submission": False,
                }
            print(json.dumps(result, sort_keys=True), flush=True)
            last = decision
            status = str(result.get("status") or "")
            if status == "pass":
                updated = _advance_direct_bootstrap(args, result)
                if updated is not None:
                    print(json.dumps({
                        "event": "direct_bootstrap_advanced",
                        "decision_ts": decision.isoformat(),
                        "bootstrap_previous_run": updated,
                    }, sort_keys=True), flush=True)
            execution_started = bool(result.get("execution_attempt_started"))
            if status in {"pass", "already_completed", "expired"} or execution_started:
                retry_after = None
            else:
                retry_after = now + pd.Timedelta(
                    seconds=max(1.0, float(args.failed_retry_seconds))
                )
        # Preserve retry polling, but do not let a five-second sleep phase
        # drift the first attempt tens of seconds past an hourly boundary.
        now_after = pd.Timestamp.now(tz="UTC")
        if retry_after is not None and now_after.floor("h") == last:
            wait = max(0.05, min(
                float(args.poll_seconds),
                float((retry_after - now_after).total_seconds()),
            ))
        else:
            next_boundary = now_after.floor("h") + pd.Timedelta(hours=1)
            until_boundary = float((next_boundary - now_after).total_seconds())
            # Wake just after xx:00 rather than on an arbitrary polling phase.
            wait = max(0.05, min(float(args.poll_seconds), until_boundary + 0.20))
        time.sleep(wait)


if __name__ == "__main__":
    main()
