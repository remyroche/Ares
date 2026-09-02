#!/usr/bin/env python3
"""Rebuild a strict-R3 stateful hourly chain without exchange authority.

This is an operational recovery tool, not an entry producer.  It refreshes
only the declared point-in-time public sources for every requested completed
hour, advances the persisted feature and Geometry/K9 state exactly one hour at
a time, and runs the canonical shadow-only scorer.  It deliberately never
imports the live executor and never submits an order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data_perp" / "artifacts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_inference_bundle import StrictR3InferenceBundle  # noqa: E402
from scripts.run_strict_r3_live_hourly_entry_producer import (  # noqa: E402
    _refresh_15m,
    _refresh_official_hourly_analytics,
    _refresh_oi_funding_sidecars,
)


SCHEMA = "strict_r3_stateful_recovery_v1"


def _utc_hour(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    if stamp != stamp.floor("h"):
        raise ValueError("timestamps must be exact UTC hours")
    return stamp


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads((path / "run_manifest.json").read_text())


def _run(command: list[str], *, log: Path) -> None:
    with log.open("w", encoding="utf-8") as handle:
        complete = subprocess.run(command, cwd=ROOT, text=True, stdout=handle,
                                  stderr=subprocess.STDOUT, check=False)
    if complete.returncode:
        raise RuntimeError(f"stage failed ({complete.returncode}); inspect {log}")


def _hour_tag(decision: pd.Timestamp) -> str:
    return decision.strftime("%Y%m%dT%H%M%SZ")


def _portfolio_state_args(*, is_first_recovered_hour: bool) -> list[str]:
    """Return the only permitted reconciliation scope for recovery.

    The initial seed has a shadow state at the preceding decision boundary and
    needs the canonical live-to-shadow bridge to become exact for the first
    recovered decision.  From the second recovered hour onward, the previous
    recovery's ``next_portfolio_state`` is already exact and must remain the
    counterfactual portfolio predecessor.  Re-bridging against a flat live
    ledger on every hour would intentionally remove simulated entries and make
    the recovery auction non-sequential.
    """
    return ["--portfolio-state-reconciliation"] if is_first_recovered_hour else []


def _refresh_sources(*, decision: pd.Timestamp, universe: Path, receipt: Path) -> dict[str, Any]:
    """Refresh only the one signal hour, concurrently and append-only."""
    source_dir = receipt / "source_refresh"
    source_dir.mkdir(parents=True, exist_ok=False)
    started = pd.Timestamp(datetime.now(timezone.utc))

    def refresh_official_with_retry() -> dict[str, Any]:
        """Retry only a transient public analytics transport failure.

        This recovery-only wrapper never changes a source window, substitutes
        values, or permits scoring without a completed refresh.  Each attempt
        gets its own immutable log so a failed Kraken transport is auditable.
        """
        delays = (0.0, 5.0, 15.0)
        attempts: list[dict[str, Any]] = []
        last_error: Exception | None = None
        for number, delay in enumerate(delays, start=1):
            if delay:
                time.sleep(delay)
            log = source_dir / f"official_analytics_attempt_{number:02d}.log"
            try:
                _refresh_official_hourly_analytics(
                    decision=decision, universe=universe, log=log
                )
            except Exception as exc:  # fail closed after the bounded attempts
                last_error = exc
                attempts.append({
                    "attempt": number,
                    "delay_seconds": delay,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "log": log.name,
                })
                continue
            attempts.append({
                "attempt": number,
                "delay_seconds": delay,
                "status": "complete",
                "log": log.name,
            })
            return {"attempts": attempts}
        raise RuntimeError(
            "official analytics refresh failed after bounded recovery retries: "
            f"{last_error}"
        )

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="strict-r3-recovery") as pool:
        futures = {
            "fifteen_minute": pool.submit(
                _refresh_15m,
                decision=decision,
                universe=universe,
                out_dir=source_dir / "fifteen_minute",
                settled_retry_schedule_seconds=(30.0, 60.0, 120.0, 180.0),
            ),
            "oi_funding": pool.submit(
                _refresh_oi_funding_sidecars,
                decision=decision,
                universe=universe,
                out_dir=source_dir / "oi_funding",
            ),
            "official_analytics": pool.submit(
                refresh_official_with_retry,
            ),
        }
        fifteen = futures["fifteen_minute"].result()
        futures["oi_funding"].result()
        official = futures["official_analytics"].result()
    completed = pd.Timestamp(datetime.now(timezone.utc))
    result = {
        "decision_ts": decision.isoformat(),
        "signal_ts": (decision - pd.Timedelta(hours=1)).isoformat(),
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "duration_seconds": float((completed - started).total_seconds()),
        "fifteen_minute": fifteen,
        "official_analytics": official,
        "append_only": True,
        "future_bars_requested": False,
    }
    (source_dir / "run_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _run_hour(
    *,
    decision: pd.Timestamp,
    predecessor: Path,
    bundle_path: Path,
    live_state: Path,
    output_root: Path,
    feature_contract_hash: str,
    feature_tail_hours: int,
    universe: Path,
    bootstrap_feature_state: Path | None,
    is_first_recovered_hour: bool,
) -> Path:
    tag = _hour_tag(decision)
    receipt = output_root / f"hour_{tag}"
    run_dir = receipt / "run"
    if receipt.exists():
        manifest = receipt / "recovery_hour_manifest.json"
        if manifest.exists():
            payload = json.loads(manifest.read_text())
            if payload.get("status") == "complete":
                return run_dir
        raise FileExistsError(f"incomplete immutable recovery receipt exists: {receipt}")
    receipt.mkdir(parents=True)
    started = pd.Timestamp(datetime.now(timezone.utc))
    source = _refresh_sources(decision=decision, universe=universe, receipt=receipt)
    # Portfolio state chains only from the preceding counterfactual recovery
    # receipt.  The canonical live state is supplied solely to retain a stable
    # state-file contract; it is not used once a predecessor exists.
    # A one-time, explicitly audited implementation reseal may be used only
    # for the first recovered hour.  All later hours must consume the feature
    # state emitted by their immediate recovered predecessor.  The former
    # name-specific v60 exception made this recovery utility unusable for a
    # later verified bootstrap despite an otherwise identical safety contract.
    source_feature_state = (
        bootstrap_feature_state
        if is_first_recovered_hour and bootstrap_feature_state is not None
        else predecessor / "feature_state" / "bundle"
    )
    _run([
        sys.executable, str(ROOT / "scripts" / "run_strict_r3_hourly_shadow_resume_v15.py"),
        "--inference-bundle", str(bundle_path.relative_to(ROOT)),
        "--portfolio-state-json", str(live_state.relative_to(ROOT)),
        "--decision-ts", decision.isoformat(),
        "--out-dir", str(run_dir.relative_to(ROOT)),
        "--previous-shadow-run", str(predecessor.relative_to(ROOT)),
        *_portfolio_state_args(is_first_recovered_hour=is_first_recovered_hour),
        "--feature-state-bundle", str(source_feature_state.relative_to(ROOT)),
        "--feature-state-contract-hash", feature_contract_hash,
        "--feature-state-tail-hours", str(feature_tail_hours),
        "--mode", "shadow-only",
    ], log=receipt / "shadow_recovery.log")
    manifest = _load_manifest(run_dir)
    required = {
        "exchange_calls": 0,
        "order_submission_enabled": False,
        "complete_universe_features_before_actionability_filter": True,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise AssertionError(f"recovery invariant {key}={manifest.get(key)!r}, expected {expected!r}")
    manifest_source = Path(str(manifest.get("stateful_feature_bundle_input") or ""))
    if not manifest_source.is_absolute():
        manifest_source = ROOT / manifest_source
    if manifest_source.resolve() != source_feature_state.resolve():
        raise AssertionError("feature state does not descend from the immediate predecessor")
    if not (run_dir / "feature_state" / "bundle" / "state_bundle_manifest.json").is_file():
        raise FileNotFoundError("recovery hour omitted its output feature-state bundle")
    if not (run_dir / "cycle" / "score" / "geometry_k9_state" / "run_manifest.json").is_file():
        raise FileNotFoundError("recovery hour omitted its Geometry/K9 state")
    completed = pd.Timestamp(datetime.now(timezone.utc))
    receipt_payload = {
        "schema": SCHEMA,
        "status": "complete",
        "decision_ts": decision.isoformat(),
        "predecessor": str(predecessor.relative_to(ROOT)),
        "predecessor_manifest_sha256": _sha(predecessor / "run_manifest.json"),
        "run": str(run_dir.relative_to(ROOT)),
        "run_manifest_sha256": _sha(run_dir / "run_manifest.json"),
        "source": source,
        "geometry_bundle_sha256": manifest["inference_bundle_audit"]["geometry_bundle_sha256"],
        "geometry_state_mode": manifest.get("conversion_state_mode"),
        "feature_contract_sha256": feature_contract_hash,
        "feature_complete_rows": int(manifest["current_feature_parity_rows"]),
        "eligible_rows": int(manifest["eligible_rows"]),
        "admitted_rows": int(manifest["admitted_rows"]),
        "portfolio_accepted_rows": int(manifest["portfolio_accepted_rows"]),
        "exchange_calls": 0,
        "order_submission_enabled": False,
        "portfolio_mode": "counterfactual_predecessor_chain",
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "duration_seconds": float((completed - started).total_seconds()),
    }
    (receipt / "recovery_hour_manifest.json").write_text(json.dumps(receipt_payload, indent=2) + "\n")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-bundle", type=Path, required=True)
    parser.add_argument("--live-state", type=Path, required=True)
    parser.add_argument("--bootstrap-run", type=Path, required=True)
    parser.add_argument("--start-decision", required=True)
    parser.add_argument("--end-decision", required=True,
                        help="Inclusive last completed decision hour to recover")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Finalize an interrupted recovery root using only already-complete "
            "immutable hourly receipts. Existing incomplete receipts remain "
            "fail-closed and are never overwritten."
        ),
    )
    parser.add_argument(
        "--bootstrap-feature-state-bundle", type=Path,
        help=(
            "Optional one-time byte-identical current-code receipt for the "
            "bootstrap run's feature state. It is accepted only for the first "
            "recovered hour; every later hour must use its immediate predecessor."
        ),
    )
    parser.add_argument(
        "--bootstrap-shadow-state-exact",
        action="store_true",
        help=(
            "Declare that --bootstrap-run already carries the exact prior "
            "counterfactual shadow portfolio state.  The first recovered hour "
            "then consumes its next_portfolio_state directly instead of "
            "reconciling against the live ledger a second time."
        ),
    )
    args = parser.parse_args()

    bundle_path = (ROOT / args.inference_bundle).resolve() if not args.inference_bundle.is_absolute() else args.inference_bundle.resolve()
    live_state = (ROOT / args.live_state).resolve() if not args.live_state.is_absolute() else args.live_state.resolve()
    bootstrap = (ROOT / args.bootstrap_run).resolve() if not args.bootstrap_run.is_absolute() else args.bootstrap_run.resolve()
    out_root = (ROOT / args.out_root).resolve() if not args.out_root.is_absolute() else args.out_root.resolve()
    bootstrap_feature_state = (
        ((ROOT / args.bootstrap_feature_state_bundle).resolve()
         if not args.bootstrap_feature_state_bundle.is_absolute()
         else args.bootstrap_feature_state_bundle.resolve())
        if args.bootstrap_feature_state_bundle is not None else None
    )
    start = _utc_hour(args.start_decision)
    end = _utc_hour(args.end_decision)
    if end < start:
        raise ValueError("end-decision precedes start-decision")
    if out_root.exists():
        if not args.resume_existing:
            raise FileExistsError(f"immutable recovery root exists: {out_root}")
        if (out_root / "run_manifest.json").exists():
            raise FileExistsError(f"recovery root is already finalized: {out_root}")
    if not (bootstrap / "run_manifest.json").is_file():
        raise FileNotFoundError("bootstrap run manifest is missing")
    if not (bootstrap / "feature_state" / "bundle" / "state_bundle_manifest.json").is_file():
        raise FileNotFoundError("bootstrap feature-state bundle is missing")
    if bootstrap_feature_state is not None and not (
        bootstrap_feature_state / "state_bundle_manifest.json"
    ).is_file():
        raise FileNotFoundError("bootstrap feature-state receipt is missing its manifest")
    if args.bootstrap_shadow_state_exact and bootstrap_feature_state is not None:
        raise ValueError(
            "an exact shadow bootstrap must use its own immediate feature-state "
            "bundle; do not supply a one-time implementation-rebind receipt"
        )

    bundle = StrictR3InferenceBundle.load(bundle_path, root=ROOT)
    bundle.validate(decision_ts=start)
    runtime = dict(bundle.payload.get("runtime") or {})
    state = dict(runtime.get("feature_state") or {})
    if state.get("mode") != "persisted_state_only":
        raise ValueError("recovery requires persisted_state_only feature state")
    contract_hash = str(state["contract_sha256"])
    tail_hours = int(state["panel_tail_hours"])
    universe = ROOT / str(bundle.payload["paths"]["frozen_universe_manifest"])

    out_root.mkdir(parents=True, exist_ok=args.resume_existing)
    predecessor = bootstrap
    recovered: list[dict[str, Any]] = []
    for index, decision in enumerate(pd.date_range(start, end, freq="h", tz="UTC")):
        predecessor = _run_hour(
            decision=decision,
            predecessor=predecessor,
            bundle_path=bundle_path,
            live_state=live_state,
            output_root=out_root,
            feature_contract_hash=contract_hash,
            feature_tail_hours=tail_hours,
            universe=universe,
            bootstrap_feature_state=bootstrap_feature_state,
            # A verified recovered predecessor already has the exact
            # counterfactual portfolio state stamped for the next decision.
            # Reconciliation is therefore a one-time live-to-shadow bridge,
            # never a property of the first loop iteration by itself.
            is_first_recovered_hour=(
                index == 0 and not args.bootstrap_shadow_state_exact
            ),
        )
        recovered.append(json.loads((predecessor.parent / "recovery_hour_manifest.json").read_text()))
    final = {
        "schema": SCHEMA,
        "status": "complete",
        "bootstrap_run": str(bootstrap.relative_to(ROOT)),
        "bootstrap_run_manifest_sha256": _sha(bootstrap / "run_manifest.json"),
        "bootstrap_feature_state_receipt": (
            str(bootstrap_feature_state.relative_to(ROOT))
            if bootstrap_feature_state is not None else None
        ),
        "bootstrap_shadow_state_exact": bool(args.bootstrap_shadow_state_exact),
        "inference_bundle": str(bundle_path.relative_to(ROOT)),
        "inference_bundle_sha256": _sha(bundle_path),
        "feature_contract_sha256": contract_hash,
        "geometry_bundle_sha256": recovered[-1]["geometry_bundle_sha256"],
        "start_decision": start.isoformat(),
        "end_decision": end.isoformat(),
        "hours": recovered,
        "final_run": str(predecessor.relative_to(ROOT)),
        "final_feature_state_bundle": str((predecessor / "feature_state" / "bundle").relative_to(ROOT)),
        "exchange_calls": 0,
        "order_submission_enabled": False,
        "portfolio_mode": "counterfactual_predecessor_chain",
    }
    (out_root / "run_manifest.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps({"event": "complete", **final}))


if __name__ == "__main__":
    main()
