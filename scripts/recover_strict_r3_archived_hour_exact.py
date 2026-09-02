#!/usr/bin/env python3
"""Recover one archived Strict-R3 hour with an independently rebuilt state.

The tool is deliberately candidate-only.  It rebuilds the feature state from
the immediate predecessor, compares the complete 170-row matrix to the
archived point-in-time checkpoint, and only then advances the score/K9 chain.
It has no network, exchange or order-writing code path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURE_CONTRACT = "12672f92789107fab4c9ab76a20c0c6504e8adce215b4a7f3fc83171dc5705c4"
FINAL14_CONTRACT = "46103989c66b5f2b286386bca747efe5132151787e22365387922154c783d978"
ORDERBOOK_CONTRACT = "b51a5a9ff5baf838684196e13d0095b1e9b8b5a0eb71da126e1739a08ad4ea27"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], log: Path) -> None:
    # Keep recovery logs observable while a full persisted-state graph is
    # being advanced; buffered stdout made healthy long-running stages look
    # indistinguishable from a stall.
    if command and Path(command[0]).name.startswith("python"):
        command = [command[0], "-u", *command[1:]]
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle,
                                stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"stage failed ({result.returncode}): {log}")


def _feature_parity(reference: Path, rebuilt: Path) -> dict[str, object]:
    left = pd.read_parquet(rebuilt).sort_values("candidate_id").reset_index(drop=True)
    right = pd.read_parquet(reference).sort_values("candidate_id").reset_index(drop=True)
    if left["candidate_id"].tolist() != right["candidate_id"].tolist():
        raise AssertionError("candidate identities differ")
    changed: list[str] = []
    max_delta = 0.0
    for field in sorted(set(left.columns).intersection(right.columns)):
        if field == "candidate_id":
            continue
        lhs, rhs = left[field], right[field]
        if pd.api.types.is_numeric_dtype(lhs) and pd.api.types.is_numeric_dtype(rhs):
            x = pd.to_numeric(lhs, errors="coerce").to_numpy(float)
            y = pd.to_numeric(rhs, errors="coerce").to_numpy(float)
            if not np.array_equal(np.isnan(x), np.isnan(y)):
                changed.append(field)
                continue
            finite = np.isfinite(x) & np.isfinite(y)
            delta = float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else 0.0
            max_delta = max(max_delta, delta)
            if delta != 0.0:
                changed.append(field)
        elif not lhs.equals(rhs):
            changed.append(field)
    return {
        "schema": "strict_r3_archived_hour_feature_parity_v1",
        "status": "pass" if not changed and max_delta == 0.0 else "fail",
        "candidate_ids_exact": True,
        "field_count": int(len(set(left.columns).intersection(right.columns))),
        "changed_fields": changed,
        "max_numeric_delta": max_delta,
        "all_missing_numeric_fields_compared_exactly": True,
        "no_exchange_calls": True,
        "order_submission_enabled": False,
    }


def _write_handoff(*, out: Path, previous_score_run: Path,
                   state_bundle: Path, decision: pd.Timestamp,
                   parity_receipt: Path) -> None:
    if out.exists():
        raise FileExistsError(f"immutable handoff already exists: {out}")
    (out / "feature_state").mkdir(parents=True)
    # Absolute links avoid the copy-depth semantic bug that previously broke
    # a restored state bundle.  They reference only immutable artifacts.
    (out / "cycle").symlink_to((previous_score_run / "cycle").resolve())
    (out / "candidate_grid").symlink_to((previous_score_run / "candidate_grid").resolve())
    (out / "features").symlink_to((previous_score_run / "features").resolve())
    (out / "feature_state" / "bundle").symlink_to(state_bundle.resolve())
    payload = {
        "schema": "strict_r3_exact_feature_score_handoff_v1",
        "decision_ts": decision.isoformat(),
        "previous_score_decision_ts": (decision - pd.Timedelta(hours=1)).isoformat(),
        "score_cycle_source": str(previous_score_run / "cycle"),
        "feature_state_source": str(state_bundle),
        "feature_state_parity_receipt": str(parity_receipt),
        "candidate_only": True,
        "exchange_calls": 0,
        "order_submission_enabled": False,
    }
    (out / "run_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-bundle", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--previous-score-run", type=Path, required=True)
    parser.add_argument("--previous-feature-state", type=Path, required=True)
    parser.add_argument("--decision-ts", required=True)
    parser.add_argument("--feature-state-out", type=Path, required=True)
    parser.add_argument("--handoff-out", type=Path, required=True)
    parser.add_argument("--score-out", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse-validated-feature-state", type=Path,
        help=(
            "Reuse an already immutable, exact-parity feature-state bundle "
            "when only a downstream score/admission retry is required."
        ),
    )
    parser.add_argument(
        "--feature-parity-receipt", type=Path,
        help="Required exact-parity receipt when reusing a feature-state bundle.",
    )
    args = parser.parse_args()

    decision = pd.Timestamp(args.decision_ts)
    decision = decision.tz_localize("UTC") if decision.tzinfo is None else decision.tz_convert("UTC")
    if decision != decision.floor("h"):
        raise ValueError("decision timestamp must be an exact UTC hour")
    source = args.source_run.resolve()
    previous = args.previous_score_run.resolve()
    prior_state = args.previous_feature_state.resolve()
    for required in (
        source / "candidate_grid" / "target_free_candidate_population.parquet",
        source / "features" / "canonical120_features.parquet",
        previous / "cycle" / "score" / "geometry_k9_state" / "run_manifest.json",
        previous / "cycle" / "next_portfolio_state.json",
        prior_state / "state_bundle_manifest.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    source_state = json.loads((prior_state / "state_bundle_manifest.json").read_text())
    # The state emitted by a completed decision contains that decision's
    # signal-hour feature update.  Recovering the next decision therefore
    # starts from the immediately preceding decision's signal hour, then
    # appends this decision's own signal hour.
    expected_input_state = decision - pd.Timedelta(hours=2)
    if pd.Timestamp(source_state["latest_state_timestamp"]).tz_convert("UTC") != expected_input_state:
        raise ValueError("previous feature state does not end at the prior decision signal hour")
    reuse_state = (
        args.reuse_validated_feature_state.resolve()
        if args.reuse_validated_feature_state is not None else None
    )
    if reuse_state is not None and reuse_state != args.feature_state_out.resolve():
        raise ValueError("reused feature state must equal --feature-state-out")
    if args.score_out.exists() or args.handoff_out.exists() or (
        args.feature_state_out.exists() and reuse_state is None
    ):
        raise FileExistsError("recovery outputs must be new immutable paths")
    if reuse_state is None:
        # The parity receipt belongs beside the newly snapshotted state.  Make
        # that immutable parent explicitly before any state operation so a
        # successful snapshot cannot be followed by a late receipt-path
        # failure.  A pre-existing directory is refused to preserve retry
        # isolation and output immutability.
        args.feature_state_out.parent.mkdir(parents=True, exist_ok=False)

    work = args.work_dir.resolve()
    if work.exists():
        raise FileExistsError(f"work directory already exists: {work}")
    work.mkdir(parents=True)
    panel_dir, cache_dir, features_dir = work / "panel", work / "cache", work / "features"
    candidate_path = source / "candidate_grid" / "target_free_candidate_population.parquet"
    if reuse_state is None:
        history_start = str(source_state["panel_start"])
        _run([
            sys.executable, "scripts/update_strict_r3_feature_panel_state.py",
            "--candidates", str(candidate_path), "--history-start", history_start,
            "--end-exclusive", decision.isoformat(), "--state-in",
            str(prior_state / "source_panel" / "feature_panel_state.joblib"),
            "--preserve-sealed-overlap", "--out-dir", str(panel_dir),
        ], work / "panel_update.log")
        _run([
            sys.executable, "scripts/materialize_strict_r3_forward_features_incremental_v13.py",
            "--candidates", str(candidate_path), "--panel-state",
            str(panel_dir / "feature_panel_state.joblib"), "--cache-dir", str(cache_dir),
            "--restore-state-bundle", str(prior_state), "--expected-state-contract-hash",
            FEATURE_CONTRACT, "--stateful-tail-hours", "1536", "--stateful-exact-family",
            "final14", "--stateful-exact-family", "orderbook_precomposite",
            "--expected-final14-contract-hash", FINAL14_CONTRACT,
            "--expected-orderbook-precomposite-contract-hash", ORDERBOOK_CONTRACT,
            "--side", "long", "--out-dir", str(features_dir),
        ], work / "feature_materialize.log")
        _run([
            sys.executable, "scripts/snapshot_strict_r3_feature_state_bundle.py",
            "--cache-dir", str(cache_dir), "--panel-state", str(panel_dir / "feature_panel_state.joblib"),
            "--out-dir", str(args.feature_state_out), "--contract-hash", FEATURE_CONTRACT,
            "--scope", "strict_r3_hourly_canonical120_stateful", "--panel-tail-hours", "1536",
            "--required-state-kind", "raw_rolling", "--required-state-kind", "causal_transform",
            "--required-state-kind", "derived_history", "--required-state-kind", "nested_derived",
            "--required-state-kind", "oi_long_iqr", "--required-state-kind", "fixed_ffd",
            "--required-state-kind", "strict_r3_final14", "--required-state-kind", "orderbook_precomposite",
            "--expected-state-timestamp", (decision - pd.Timedelta(hours=1)).isoformat(),
        ], work / "feature_snapshot.log")
        parity = _feature_parity(
            source / "features" / "canonical120_features.parquet",
            features_dir / "canonical120_features.parquet",
        )
        parity_path = args.feature_state_out.parent / "feature_matrix_parity.json"
        parity_path.write_text(json.dumps(parity, indent=2) + "\n")
    else:
        if args.feature_parity_receipt is None or not args.feature_parity_receipt.is_file():
            raise FileNotFoundError("reused feature state requires its parity receipt")
        parity_path = args.feature_parity_receipt.resolve()
        parity = json.loads(parity_path.read_text())
        state_manifest = json.loads((reuse_state / "state_bundle_manifest.json").read_text())
        state_ts = pd.Timestamp(state_manifest["latest_state_timestamp"])
        state_ts = state_ts.tz_localize("UTC") if state_ts.tzinfo is None else state_ts.tz_convert("UTC")
        if state_ts != decision - pd.Timedelta(hours=1):
            raise ValueError("reused feature state timestamp does not match recovered signal hour")
    if parity.get("status") != "pass" or parity.get("candidate_ids_exact") is not True or float(parity.get("max_numeric_delta", float("nan"))) != 0.0:
        raise AssertionError(json.dumps(parity, sort_keys=True))
    _write_handoff(out=args.handoff_out, previous_score_run=previous,
                   state_bundle=args.feature_state_out, decision=decision,
                   parity_receipt=parity_path)
    previous_decision = pd.Timestamp(
        json.loads((previous / "run_manifest.json").read_text())["decision_ts"]
    )
    previous_decision = (
        previous_decision.tz_localize("UTC")
        if previous_decision.tzinfo is None
        else previous_decision.tz_convert("UTC")
    )
    _run([
        sys.executable, "scripts/run_strict_r3_hourly_shadow_resume_v15.py",
        "--inference-bundle", str(args.inference_bundle), "--portfolio-state-json",
        str(previous / "cycle" / "next_portfolio_state.json"), "--decision-ts", decision.isoformat(),
        # The score successor must inherit the *previous scored decision*, not
        # the feature-state handoff timestamp.  At a UTC midnight boundary the
        # latter is the current decision and would incorrectly make the
        # runtime-ledger builder treat yesterday's intraday carry as same-day.
        # The freshly materialised state is supplied explicitly below, so the
        # scored predecessor remains the authoritative calibration/portfolio
        # lineage.
        "--out-dir", str(args.score_out), "--previous-shadow-run", str(previous),
        "--reuse-current-inputs-from", str(source), "--feature-state-bundle",
        str(args.feature_state_out), "--feature-state-contract-hash", FEATURE_CONTRACT,
        "--feature-state-tail-hours", "1536", "--mode", "shadow-only",
        *(
            ["--candidate-only-reset-calibration-to-sealed-base"]
            if previous_decision.normalize() < decision.normalize() else []
        ),
    ], work / "score_recovery.log")
    manifest = json.loads((args.score_out / "run_manifest.json").read_text())
    required = {
        "exchange_calls": 0,
        "order_submission_enabled": False,
        "complete_universe_features_before_actionability_filter": True,
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise AssertionError("score recovery violated no-order invariants")
    receipt = {
        "schema": "strict_r3_archived_hour_recovery_v1",
        "status": "complete",
        "decision_ts": decision.isoformat(),
        "source_run": str(source),
        "previous_score_run": str(previous),
        "previous_state": str(prior_state),
        "feature_state_out": str(args.feature_state_out),
        "handoff_out": str(args.handoff_out),
        "score_out": str(args.score_out),
        "feature_parity": parity,
        "geometry_bundle_sha256": manifest["inference_bundle_audit"]["geometry_bundle_sha256"],
        "score_manifest_sha256": _sha(args.score_out / "run_manifest.json"),
        "exchange_calls": 0,
        "order_submission_enabled": False,
    }
    (args.score_out.parent / "recovery_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
