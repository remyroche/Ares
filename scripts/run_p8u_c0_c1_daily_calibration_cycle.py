#!/usr/bin/env python3
"""Advance the no-order P8U C0/C1 daily calibration state once.

This is the operational glue for the append-only C0/C1 calibration contract:

``target-free Router50 score delta -> exact resolved policy labels ->
effective ledger revision -> next UTC-day C0/C1 state``.

It intentionally has no feature construction, model refit, portfolio, account,
network, or exchange-order authority.  The caller supplies a local, immutable
score panel that was produced earlier by the no-order scorer.  The cycle
materialises only labels whose H12 path had resolved before ``--decision-day``;
later score rows are retained in the effective ledger with an unresolved label.

The mutable ledger and daily-state pointers are advanced only after immutable
cycle artifacts have been written.  A preflight failure, a stale day, missing
label support, or a duplicate cycle lock fails closed without publishing a
calibration state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.append_p8u_c0_c1_calibration_policy_ledger import _router50_scores


SCHEMA = "p8u-c0-c1-daily-calibration-cycle-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _utc_day(value: object) -> pd.Timestamp:
    day = _utc(value)
    if day != day.normalize():
        raise ValueError("decision day must be a UTC midnight timestamp")
    return day


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _require_next_state_day(state_root: Path, decision_day: pd.Timestamp) -> None:
    latest_path = state_root / "latest.json"
    if not latest_path.is_file():
        raise FileNotFoundError("daily calibration state lacks latest.json")
    latest = _read_json(latest_path)
    prior = _utc_day(latest.get("decision_day"))
    if decision_day != prior + pd.Timedelta(days=1):
        raise ValueError(
            "daily calibration cycle must advance exactly one day "
            f"(latest={prior.isoformat()}, requested={decision_day.isoformat()})"
        )


def _require_ledger_revision_advance(ledger_root: Path, revision: str) -> None:
    latest = _read_json(ledger_root / "latest.json")
    prior = str(latest.get("revision") or "")
    if not prior or revision <= prior:
        raise ValueError(
            "effective-ledger revision must advance lexically beyond its latest "
            f"revision (latest={prior!r}, requested={revision!r})"
        )


def _score_delta(path: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Validate and trim a source score panel to this cycle's Router50 delta."""
    routed = _router50_scores(path.resolve())
    score = routed.loc[
        routed["__decision_ts__"].ge(start) & routed["__decision_ts__"].lt(end)
    ].copy()
    if score.empty:
        raise RuntimeError("score panel has no target-free Router50 rows in the cycle window")
    if score["candidate_id"].duplicated().any():
        raise AssertionError("cycle score delta has duplicate target-free identities")
    if not np.isfinite(score["base_rank_ts"].to_numpy(float)).all():
        raise AssertionError("cycle score delta has non-finite Base rank coordinates")
    # ``_router50_scores`` normalises the eventual policy-label fields so it
    # can merge them into the effective ledger.  They must not travel in this
    # target-free source delta: the downstream append utility correctly
    # rejects outcome-like columns on a score input.  The original full-panel
    # Router50 cardinality was validated above, before reducing this immutable
    # delta to its score contract.
    return score.loc[:, [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name", "base_rank_ts",
    ]].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            "local calibration component failed:\n"
            + " ".join(command)
            + "\nstdout:\n" + completed.stdout[-4000:]
            + "\nstderr:\n" + completed.stderr[-4000:]
        )


def _create_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"schema": SCHEMA, "pid": os.getpid(), "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-root", type=Path, required=True, help="new immutable output directory")
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--decision-day", required=True, help="UTC midnight for the state to publish")
    parser.add_argument("--revision", required=True, help="new lexically ordered effective-ledger revision")
    parser.add_argument("--upstream-scores", type=Path, required=True, help="local target-free full or Router50 score panel")
    parser.add_argument("--score-start", required=True, help="inclusive UTC score timestamp")
    parser.add_argument("--score-end", required=True, help="exclusive UTC score timestamp")
    parser.add_argument("--frozen-source-manifest", type=Path, required=True)
    parser.add_argument("--frozen-kraken-product-ledger", type=Path, required=True)
    parser.add_argument("--c0-package", type=Path, required=True)
    parser.add_argument("--c1-package", type=Path, required=True)
    parser.add_argument("--frozen-policy", type=Path, required=True)
    parser.add_argument("--minute-root", type=Path, default=ROOT / "data_perp/exchanges/krakenfutures/execution_1m")
    parser.add_argument("--minimum-resolved-rows", type=int, default=500)
    parser.add_argument("--oracle-sample-size", type=int, default=64)
    parser.add_argument(
        "--symbol-load-workers", type=int, default=4,
        help="Bounded read-only exact-label source workers (1-8; default 4).",
    )
    parser.add_argument("--lock-path", type=Path, help="exclusive local cycle lock; defaults under ledger root")
    args = parser.parse_args()

    cycle_root = args.cycle_root.resolve()
    ledger_root = args.ledger_root.resolve()
    state_root = args.state_root.resolve()
    day = _utc_day(args.decision_day)
    start, end = _utc(args.score_start), _utc(args.score_end)
    if end <= start or end > day:
        raise ValueError("score window must be non-empty and may not extend beyond the decision day")
    if cycle_root.exists():
        raise FileExistsError("daily calibration cycle output must be immutable")
    if day > pd.Timestamp.now(tz="UTC").normalize():
        raise ValueError("cannot publish a future calibration day")
    if int(args.minimum_resolved_rows) < 1 or int(args.oracle_sample_size) < 1:
        raise ValueError("minimum support and oracle sample size must be positive")
    if not 1 <= int(args.symbol_load_workers) <= 8:
        raise ValueError("symbol load workers must be between 1 and 8")
    _require_next_state_day(state_root, day)
    _require_ledger_revision_advance(ledger_root, str(args.revision))
    lock = (args.lock_path or (ledger_root / ".daily_cycle.lock")).resolve()
    _create_lock(lock)
    try:
        # Materialise immutable cycle inputs before mutating either append-only
        # pointer.  The output score delta is a target-free audit record in its
        # own right and makes later recovery deterministic.
        score_delta = _score_delta(args.upstream_scores, start=start, end=end)
        stage = cycle_root.parent / f".{cycle_root.name}.{uuid.uuid4().hex}.staging"
        stage.mkdir(parents=True, exist_ok=False)
        try:
            score_path = stage / "router50_score_delta.parquet"
            score_delta.to_parquet(score_path, index=False, compression="zstd")
            # A cycle can legitimately contain only new, unresolved score
            # rows (for example after a late daily score backfill).  Append
            # those target-free identities and publish the normal prior-21d
            # state without attempting to open a future H12 path.  At least
            # one score becomes label-eligible only after 12h05m.
            resolved_mask = (score_delta["__decision_ts__"] + pd.Timedelta(hours=12, minutes=5)).lt(day)
            candidate_dir: Path | None = None
            labels_dir: Path | None = None
            exact_path: Path | None = None
            if bool(resolved_mask.any()):
                resolved_start = score_delta.loc[resolved_mask, "__decision_ts__"].min()
                resolved_end = score_delta.loc[resolved_mask, "__decision_ts__"].max() + pd.Timedelta(hours=1)
                candidate_dir = stage / "exact1m_candidates"
                _run([
                    sys.executable, "scripts/prepare_p8u_c0_c1_calibration_exact1m_candidates.py",
                    "--upstream-scores", str(score_path),
                    "--frozen-source-manifest", str(args.frozen_source_manifest.resolve()),
                    "--frozen-kraken-product-ledger", str(args.frozen_kraken_product_ledger.resolve()),
                    "--start", resolved_start.isoformat(), "--end", resolved_end.isoformat(), "--out", str(candidate_dir),
                ])
                labels_dir = stage / "exact1m_resolved_labels"
                _run([
                    sys.executable, "scripts/materialize_strict_r3_p8u_exact_1m_rich_policy.py",
                    "--candidate-dir", str(candidate_dir), "--skip-parent-policy-comparison",
                    "--label-available-before", day.isoformat(),
                    "--frozen-policy", str(args.frozen_policy.resolve()),
                    "--minute-root", str(args.minute_root.resolve()),
                    "--symbol-load-workers", str(int(args.symbol_load_workers)),
                    "--oracle-sample-size", str(int(args.oracle_sample_size)),
                    "--out-dir", str(labels_dir),
                ])
                exact_path = labels_dir / "exact_1m_policy_outcomes.parquet"
                if not exact_path.is_file():
                    raise FileNotFoundError("exact one-minute materialiser emitted no outcomes")
            # The working artifacts are immutable before their one permitted
            # append-only ledger publication.  A failed subsequent publisher
            # leaves an auditable cycle; it never silently rewrites a state.
            cycle_manifest = {
                "schema": SCHEMA,
                "status": "PASS_PREPARED_NO_ORDER_DAILY_CALIBRATION_CYCLE",
                "decision_day": day.isoformat(), "revision": str(args.revision),
                "score_window": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
                "rows": {"router50_score_delta": int(len(score_delta))},
                "exact_label_source_loading": {
                    "symbol_load_workers": int(args.symbol_load_workers),
                    "contract": "parallel immutable local source reads only; deterministic policy label materialisation",
                },
                "inputs": {
                    "upstream_scores": {"path": str(args.upstream_scores.resolve()), "sha256": _sha256(args.upstream_scores.resolve())},
                    "frozen_source_manifest": {"path": str(args.frozen_source_manifest.resolve()), "sha256": _sha256(args.frozen_source_manifest.resolve())},
                    "frozen_kraken_product_ledger": {"path": str(args.frozen_kraken_product_ledger.resolve()), "sha256": _sha256(args.frozen_kraken_product_ledger.resolve())},
                    "frozen_policy": {"path": str(args.frozen_policy.resolve()), "sha256": _sha256(args.frozen_policy.resolve())},
                },
                "outputs": {
                    "router50_score_delta.parquet": _sha256(score_path),
                    "candidate_manifest.json": None if candidate_dir is None else _sha256(candidate_dir / "candidate_manifest.json"),
                    "exact_1m_policy_outcomes.parquet": None if exact_path is None else _sha256(exact_path),
                    "exact_label_manifest.json": None if labels_dir is None else _sha256(labels_dir / "run_manifest.json"),
                },
                "causality": {
                    "scores": "target-free Router50 only", "labels": "exact rich policy only after H12 resolution",
                    "label_cutoff": f"policy_label_available_ts < {day.isoformat()}",
                    "authority": "no model refit, portfolio, network, exchange, account, or order authority",
                },
            }
            _write_json_once(stage / "cycle_manifest.json", cycle_manifest)
            os.replace(stage, cycle_root)
        except Exception:
            # The immutable cycle directory is deliberately absent on a
            # preparation failure; the caller can safely retry under the same
            # logical date only after fixing the source issue.
            import shutil
            shutil.rmtree(stage, ignore_errors=True)
            raise

        score_path = cycle_root / "router50_score_delta.parquet"
        exact_path = cycle_root / "exact1m_resolved_labels/exact_1m_policy_outcomes.parquet"
        append_command = [
            sys.executable, "scripts/append_p8u_c0_c1_calibration_policy_ledger.py",
            "--ledger-root", str(ledger_root), "--revision", str(args.revision),
            "--upstream-scores", str(score_path),
        ]
        if exact_path.is_file():
            append_command.extend([
                "--exact-outcomes", str(exact_path), "--label-available-before", day.isoformat(),
            ])
        _run(append_command)
        ledger_latest = _read_json(ledger_root / "latest.json")
        effective = ledger_root / str(ledger_latest["ledger_path"])
        _run([
            sys.executable, "scripts/publish_p8u_c0_c1_daily_calibration_state.py",
            "--state-root", str(state_root), "--decision-day", day.isoformat(),
            "--policy-ledger", str(effective), "--c0-package", str(args.c0_package.resolve()),
            "--c1-package", str(args.c1_package.resolve()),
            "--source-manifest", str(cycle_root / "cycle_manifest.json"),
            "--minimum-resolved-rows", str(int(args.minimum_resolved_rows)),
        ])
        state_latest = _read_json(state_root / "latest.json")
        finalized = _read_json(cycle_root / "cycle_manifest.json")
        finalized.update({
            "status": "PASS_PUBLISHED_NO_ORDER_DAILY_CALIBRATION_CYCLE",
            "effective_ledger": {"latest": ledger_latest, "sha256": _sha256(effective)},
            "daily_state": {"latest": state_latest, "receipt_sha256": str(state_latest["receipt_manifest_sha256"])},
        })
        # cycle_manifest was intentionally created immutable before pointer
        # changes; write a separate immutable final receipt instead.
        _write_json_once(cycle_root / "final_receipt.json", finalized)
        print(cycle_root)
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
