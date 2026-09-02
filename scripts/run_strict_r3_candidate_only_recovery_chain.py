#!/usr/bin/env python3
"""Advance a bounded archived Strict-R3 recovery chain without live I/O.

Each hour is recovered only after the predecessor's immutable receipt passes.
The runner deliberately has no exchange or order-writing invocation; it stops
at the first missing source, failed parity check, or broken state lineage.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_SHA = "dbf7de6da6bad6927bcbe577d7ad2d2118ecc24a6bdfd35fb7fa190be13d7638"
BUNDLE = "config/strict_r3_inference_overlay_long_v170_v199_feature_state_rereceipt_candidate.json"
SOURCE_ROOT = "data_perp/artifacts/strict_r3_bcf_live_promotion_recovery_20260824_v4_warmcache"


def _score_run(hour: int) -> Path:
    # H03 is v25; subsequent exact score runs advance by two artifact versions.
    return ROOT / f"data_perp/artifacts/strict_r3_bcf_live_promotion_recovery_20260824_v{25 + 2 * (hour - 3)}_h{hour:02d}_exact_score/run"


def _feature_state(hour: int) -> Path:
    return ROOT / f"data_perp/artifacts/strict_r3_feature_state_h{hour:02d}_exact_20260824_v1/bundle"


def _receipt(hour: int) -> Path:
    return _score_run(hour).parent / "recovery_receipt.json"


def _require_pass(hour: int) -> None:
    _require_run_pass(_score_run(hour))


def _require_run_pass(score_run: Path) -> None:
    receipt = score_run.parent / "recovery_receipt.json"
    if not receipt.is_file():
        raise RuntimeError(f"missing predecessor receipt: {receipt}")
    data = json.loads(receipt.read_text())
    parity = data.get("feature_parity", {})
    if not (
        data.get("status") == "complete"
        and data.get("exchange_calls") == 0
        and data.get("order_submission_enabled") is False
        and data.get("geometry_bundle_sha256") == GEOMETRY_SHA
        and parity.get("status") == "pass"
        and parity.get("candidate_ids_exact") is True
        and parity.get("max_numeric_delta") == 0.0
    ):
        raise RuntimeError(f"predecessor did not satisfy exact no-order invariants: {score_run}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-hour", type=int, default=5)
    parser.add_argument("--end-hour", type=int, default=12)
    parser.add_argument("--wait-predecessor-seconds", type=int, default=14_400)
    parser.add_argument("--initial-previous-score-run", type=Path)
    args = parser.parse_args()
    if not (4 <= args.start_hour <= args.end_hour <= 12):
        raise ValueError("this bounded archive runner supports H04 through H12")

    initial_previous = (
        args.initial_previous_score_run.resolve()
        if args.initial_previous_score_run is not None else _score_run(args.start_hour - 1)
    )
    deadline = time.monotonic() + args.wait_predecessor_seconds
    while True:
        try:
            _require_run_pass(initial_previous)
            break
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(30)
    for hour in range(args.start_hour, args.end_hour + 1):
        prior_score = initial_previous if hour == args.start_hour else _score_run(hour - 1)
        _require_run_pass(prior_score)
        source = ROOT / SOURCE_ROOT / f"hour_20260824T{hour:02d}0000Z/run"
        required_source = source / "features/canonical120_features.parquet"
        if not required_source.is_file():
            raise FileNotFoundError(f"no archived canonical source for H{hour:02d}: {required_source}")
        out = _score_run(hour)
        if out.exists():
            _require_pass(hour)
            print(f"H{hour:02d}: existing receipt already passed", flush=True)
            continue
        handoff_version = 24 + 2 * (hour - 3)
        handoff = ROOT / f"data_perp/artifacts/strict_r3_bcf_live_promotion_recovery_20260824_v{handoff_version}_h{hour:02d}_exact_score_handoff"
        work = Path(f"/private/tmp/strict_r3_h{hour:02d}_exact_recovery_work_v4")
        command = [
            sys.executable, "-u", "scripts/recover_strict_r3_archived_hour_exact.py",
            "--inference-bundle", BUNDLE,
            "--source-run", str(source.relative_to(ROOT)),
            "--previous-score-run", str(prior_score.relative_to(ROOT)),
            "--previous-feature-state", str((prior_score / "feature_state/bundle").relative_to(ROOT)),
            "--decision-ts", f"2026-08-24T{hour:02d}:00:00Z",
            "--feature-state-out", str(_feature_state(hour).relative_to(ROOT)),
            "--handoff-out", str(handoff.relative_to(ROOT)),
            "--score-out", str(out.relative_to(ROOT)),
            "--work-dir", str(work),
        ]
        print(f"H{hour:02d}: starting exact candidate-only recovery", flush=True)
        # Validation remains content-addressed: this only enables the existing
        # stat-bound receipt cache for already sealed immutable artifacts.
        # A missing or changed artifact still forces a full SHA-256 check.
        environment = dict(os.environ)
        environment.setdefault(
            "STRICT_R3_VALIDATION_CACHE_DIR",
            "/private/tmp/strict_r3_validation_cache_v199",
        )
        # Recovery invokes the same heavyweight deterministic feature graph as
        # live inference.  Keep compiled modules and plotting configuration on
        # local scratch so every isolated child does not re-read/compile the
        # complete dependency tree before advancing a single hour.
        environment.setdefault(
            "PYTHONPYCACHEPREFIX", "/private/tmp/strict_r3_pycache_v3"
        )
        environment.setdefault("MPLCONFIGDIR", "/private/tmp/strict_r3_mpl_v3")
        # Numba's module-local cache can be slow or wedged after a filesystem
        # metadata refresh.  A local cache holds the same deterministic
        # compiled kernels; it changes neither inputs nor numerical semantics.
        environment.setdefault("NUMBA_CACHE_DIR", "/private/tmp/strict_r3_numba_cache_v2")
        result = subprocess.run(command, cwd=ROOT, env=environment)
        if result.returncode:
            raise RuntimeError(f"H{hour:02d} recovery failed with exit code {result.returncode}")
        _require_pass(hour)
        print(f"H{hour:02d}: exact no-order receipt passed", flush=True)


if __name__ == "__main__":
    main()
