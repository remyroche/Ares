#!/usr/bin/env python3
"""Audit a recovered terminal hour by independently re-advancing its state.

The live feature contract is deliberately ``persisted_state_only``.  A raw
tail reconstruction is neither the live implementation nor a valid oracle for
the stateful long-memory primitives.  This audit therefore replays the exact
archived decision inputs from the same recovered predecessor bundle and
compares all frozen feature, score, admission and portfolio outputs.

It is strictly read-only: it never imports an executor or accesses Kraken.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOLERANCE_PCT = 0.01


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _canonical(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "<NULL>"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _frame_audit(
    *, reference: Path, replay: Path, tolerance_pct: float
) -> dict[str, Any]:
    left = pd.read_parquet(reference)
    right = pd.read_parquet(replay)
    if "candidate_id" not in left or "candidate_id" not in right:
        raise AssertionError(f"candidate_id required: {reference}")
    if left["candidate_id"].duplicated().any() or right["candidate_id"].duplicated().any():
        raise AssertionError(f"candidate_id must be unique: {reference}")
    left = left.set_index("candidate_id", drop=False).sort_index()
    right = right.set_index("candidate_id", drop=False).sort_index()
    ids_equal = left.index.equals(right.index)
    columns_equal = list(left.columns) == list(right.columns)
    if not ids_equal or not columns_equal:
        return {
            "reference_rows": int(len(left)),
            "replay_rows": int(len(right)),
            "identity_equal": bool(ids_equal),
            "column_order_equal": bool(columns_equal),
            "pass": False,
            "reason": "candidate identities or ordered columns differ",
        }

    missing_mismatch = 0
    non_numeric_mismatch = 0
    numeric_over_tolerance = 0
    max_abs_delta = 0.0
    max_relative_pct = 0.0
    for column in left.columns:
        lhs = left[column]
        rhs = right[column]
        if pd.api.types.is_numeric_dtype(lhs) and pd.api.types.is_numeric_dtype(rhs):
            lvals = lhs.astype(float).to_numpy()
            rvals = rhs.astype(float).to_numpy()
            lmissing = ~np.isfinite(lvals)
            rmissing = ~np.isfinite(rvals)
            missing_mismatch += int(np.count_nonzero(lmissing != rmissing))
            valid = ~(lmissing | rmissing)
            if np.any(valid):
                delta = np.abs(lvals[valid] - rvals[valid])
                scale = np.maximum(np.maximum(np.abs(lvals[valid]), np.abs(rvals[valid])), 1e-12)
                relative_pct = 100.0 * delta / scale
                max_abs_delta = max(max_abs_delta, float(np.max(delta)))
                max_relative_pct = max(max_relative_pct, float(np.max(relative_pct)))
                numeric_over_tolerance += int(np.count_nonzero(relative_pct > tolerance_pct))
        else:
            lvals = lhs.map(_canonical).to_numpy()
            rvals = rhs.map(_canonical).to_numpy()
            non_numeric_mismatch += int(np.count_nonzero(lvals != rvals))
    passed = not (missing_mismatch or non_numeric_mismatch or numeric_over_tolerance)
    return {
        "reference_rows": int(len(left)),
        "replay_rows": int(len(right)),
        "identity_equal": True,
        "column_order_equal": True,
        "missing_mismatch": missing_mismatch,
        "non_numeric_mismatch": non_numeric_mismatch,
        "numeric_over_tolerance": numeric_over_tolerance,
        "max_abs_delta": max_abs_delta,
        "max_relative_pct": max_relative_pct,
        "pass": bool(passed),
    }


def _state_audit(reference: Path, replay: Path) -> dict[str, Any]:
    left = _load_json(reference)
    right = _load_json(replay)
    keys = [
        "feature_contract_sha256",
        "expected_state_timestamp",
        "earliest_state_timestamp",
        "latest_state_timestamp",
        "panel_state_sha256",
        "state_files",
        "state_bytes",
        "raw_rolling_states",
        "causal_transform_states",
        "derived_history_states",
        "nested_derived_states",
        "orderbook_precomposite_states",
        "strict_r3_final14_states",
    ]
    values = {key: {"reference": left.get(key), "replay": right.get(key)} for key in keys}
    return {"values": values, "pass": all(v["reference"] == v["replay"] for v in values.values())}


def _read_cycle_manifest(run: Path, relative: str) -> dict[str, Any]:
    return _load_json(run / relative)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--replay-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-current-geometry-sha256", required=True)
    parser.add_argument("--tolerance-pct", type=float, default=DEFAULT_TOLERANCE_PCT)
    args = parser.parse_args()
    reference = args.reference_run.resolve()
    replay = args.replay_run.resolve()
    if ROOT not in reference.parents or ROOT not in replay.parents:
        raise ValueError("runs must be inside the Ares repository")

    reference_run = _read_cycle_manifest(reference, "run_manifest.json")
    replay_run = _read_cycle_manifest(replay, "run_manifest.json")
    invariant_keys = [
        "schema", "decision_ts", "mode", "exchange_calls", "order_submission_enabled",
        "stateful_feature_contract_hash", "stateful_feature_bundle_input",
        "portfolio_state_input", "persisted_feature_state_only",
    ]
    invariant_values = {
        key: {"reference": reference_run.get(key), "replay": replay_run.get(key)}
        for key in invariant_keys
    }
    invariant_pass = all(item["reference"] == item["replay"] for item in invariant_values.values())
    no_order_pass = all(
        str(item.get("mode")) == "shadow-only"
        and int(item.get("exchange_calls", -1)) == 0
        and bool(item.get("order_submission_enabled")) is False
        for item in (reference_run, replay_run)
    )
    same_bundle_pass = (
        reference_run.get("hashes", {}).get("inference_bundle")
        == replay_run.get("hashes", {}).get("inference_bundle")
    )

    datasets = [
        "candidate_grid/target_free_candidate_population.parquet",
        "current_hour_inputs/canonical120_features.parquet",
        "cycle/bcf_score/predictions.parquet",
        "cycle/score/predictions.parquet",
        "cycle/dual_admission/admitted_predictions.parquet",
        "cycle/shadow_decisions.parquet",
    ]
    frame_results = {
        relative: _frame_audit(
            reference=reference / relative,
            replay=replay / relative,
            tolerance_pct=args.tolerance_pct,
        )
        for relative in datasets
    }
    state = _state_audit(
        reference / "feature_state/bundle/state_bundle_manifest.json",
        replay / "feature_state/bundle/state_bundle_manifest.json",
    )
    current_geometry = {
        "reference": _read_cycle_manifest(reference, "cycle/score/run_manifest.json").get("geometry_bundle_sha256"),
        "replay": _read_cycle_manifest(replay, "cycle/score/run_manifest.json").get("geometry_bundle_sha256"),
        "expected": args.expected_current_geometry_sha256,
    }
    bcf_geometry = {
        "reference": _read_cycle_manifest(reference, "cycle/bcf_score/run_manifest.json").get("geometry_bundle_sha256"),
        "replay": _read_cycle_manifest(replay, "cycle/bcf_score/run_manifest.json").get("geometry_bundle_sha256"),
    }
    geometry_pass = (
        current_geometry["reference"] == current_geometry["replay"] == current_geometry["expected"]
        and bcf_geometry["reference"] == bcf_geometry["replay"]
    )
    result = {
        "schema": "strict_r3_stateful_recovery_terminal_replay_parity_v1",
        "parity_kind": "same_predecessor_persisted_state_readvance",
        "reference_run": str(reference.relative_to(ROOT)),
        "replay_run": str(replay.relative_to(ROOT)),
        "reference_run_sha256": _sha256(reference / "run_manifest.json"),
        "replay_run_sha256": _sha256(replay / "run_manifest.json"),
        "tolerance_pct": args.tolerance_pct,
        "full_raw_reconstruction_used": False,
        "reason_full_raw_reconstruction_not_used": "sealed persisted_state_only contract prohibits raw-tail reconstruction as a parity oracle",
        "run_invariants": invariant_values,
        "same_inference_bundle": same_bundle_pass,
        "shadow_no_order_pass": no_order_pass,
        "current_geometry": current_geometry,
        "bcf_geometry": bcf_geometry,
        "geometry_pass": geometry_pass,
        "state_bundle": state,
        "datasets": frame_results,
    }
    result["status"] = "pass" if (
        invariant_pass and same_bundle_pass and no_order_pass and geometry_pass and state["pass"]
        and all(item["pass"] for item in frame_results.values())
    ) else "fail"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "out": str(args.out), "datasets": {k: v["pass"] for k, v in frame_results.items()}}, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
