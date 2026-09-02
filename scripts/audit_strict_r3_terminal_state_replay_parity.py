#!/usr/bin/env python3
"""Prove an independent strict-R3 terminal state re-advance is exact.

The primary and replay runs share an already-sealed predecessor.  This audit
intentionally ignores non-deterministic receipt timestamps/paths and compares
the decision-relevant identities, numerical panels and persisted operator
payloads exactly.  It never calls an exchange and writes one immutable receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


ROOT = Path(__file__).resolve().parents[1]
PANELS = (
    "candidate_grid/target_free_candidate_population.parquet",
    "features/canonical120_features.parquet",
    "cycle/bcf_score/predictions.parquet",
    "cycle/score/predictions.parquet",
    "cycle/score/score_decomposition.parquet",
    "cycle/dual_admission/admitted_predictions.parquet",
    "cycle/shadow_decisions.parquet",
    "cycle/shadow_exits.parquet",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _canonical_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    order = sorted(frame.columns)
    frame = frame.loc[:, order]
    sort_keys = [column for column in ("candidate_id", "symbol", "decision_timestamp") if column in frame]
    if sort_keys:
        frame = frame.sort_values(sort_keys, kind="mergesort", na_position="first")
    return frame.reset_index(drop=True)


def _operator_inventory(bundle: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or path.name == "state_bundle_manifest.json":
            continue
        output[str(path.relative_to(bundle))] = _sha(path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-run", type=Path, required=True)
    parser.add_argument("--replay-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    primary = args.primary_run.resolve()
    replay = args.replay_run.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")

    primary_manifest = _read_json(primary / "run_manifest.json")
    replay_manifest = _read_json(replay / "run_manifest.json")
    decision = primary_manifest.get("decision_ts")
    if decision != replay_manifest.get("decision_ts"):
        raise AssertionError("primary/replay decision timestamps differ")
    if primary_manifest.get("mode") != "shadow-only" or replay_manifest.get("mode") != "shadow-only":
        raise AssertionError("terminal parity must compare shadow-only runs")

    panel_rows: dict[str, int] = {}
    for rel in PANELS:
        left = _canonical_frame(primary / rel)
        right = _canonical_frame(replay / rel)
        assert_frame_equal(left, right, check_exact=True, check_dtype=True, check_like=False)
        panel_rows[rel] = len(left)

    primary_state = primary / "feature_state/bundle"
    replay_state = replay / "feature_state/bundle"
    left_inventory = _operator_inventory(primary_state)
    right_inventory = _operator_inventory(replay_state)
    if left_inventory != right_inventory:
        raise AssertionError("persisted operator-state payload inventory differs")
    left_state_manifest = _read_json(primary_state / "state_bundle_manifest.json")
    right_state_manifest = _read_json(replay_state / "state_bundle_manifest.json")
    state_fields = ("expected_state_timestamp", "feature_contract_sha256", "geometry_bundle_sha256")
    state_contract = {field: left_state_manifest.get(field) for field in state_fields}
    if state_contract != {field: right_state_manifest.get(field) for field in state_fields}:
        raise AssertionError("persisted state manifest contract differs")

    payload: dict[str, object] = {
        "schema": "strict_r3_stateful_recovery_terminal_replay_parity_v1",
        "status": "pass",
        "tolerance": "exact",
        "decision_ts": decision,
        "primary_run": str(primary.relative_to(ROOT)),
        "replay_run": str(replay.relative_to(ROOT)),
        "primary_run_manifest_sha256": _sha(primary / "run_manifest.json"),
        "replay_run_manifest_sha256": _sha(replay / "run_manifest.json"),
        "panels_exact": panel_rows,
        "operator_state_files_exact": len(left_inventory),
        "state_contract_exact": state_contract,
        "checks": {
            "candidate_identities_exact": True,
            "feature_values_exact": True,
            "score_values_exact": True,
            "admission_values_exact": True,
            "portfolio_decisions_exact": True,
            "persisted_operator_state_exact": True,
            "geometry_k9_contract_exact": True,
            "primary_order_submission_disabled": True,
            "replay_order_submission_disabled": True,
        },
    }
    _atomic_json(out, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
