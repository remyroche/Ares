#!/usr/bin/env python3
"""Fail closed on whether the raw-score July bridge can form transition inputs.

This audit intentionally does *not* use a neighbouring scorer or candidate
surface as a substitute.  It inspects only the hash-bound 5,760-row bridge and
emits a sealed request identifying the additional frozen inputs that a future
bridge revision must bind before mapped global-book labels or common geometry
can be derived.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRIDGE = ROOT / "data_perp/artifacts/july20_23_retrospective_allscore_bridge_20260730_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/july20_23_retrospective_allscore_transition_readiness_20260730_v1"
SCHEMA = "july20_23_retrospective_allscore_transition_readiness_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(dict(value)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def audit(bridge_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path, seal_path = bridge_root / "manifest.json", bridge_root / "manifest.sha256"
    if not manifest_path.is_file() or not seal_path.is_file() or seal_path.read_text(encoding="utf-8").split()[0] != sha256(manifest_path):
        raise ValueError("bridge manifest is absent or unsealed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bridge_item = manifest.get("outputs", {}).get("retrospective_allscore_bridge", {})
    bridge_path = bridge_root / "retrospective_allscore_bridge.parquet"
    if bridge_item.get("sha256") != sha256(bridge_path):
        raise ValueError("bridge population is not hash-bound")
    bridge = pd.read_parquet(bridge_path)
    required_identity = {"candidate_id", "__ts__", "__symbol__", "side_name", "execution_decision_utc"}
    if not required_identity.issubset(bridge.columns) or bridge["candidate_id"].duplicated().any():
        raise ValueError("bridge identity is not exact")
    mapping_contract = str(manifest.get("contracts", {}).get("mapping", ""))
    score_registry_path = bridge_root / "score_registry.parquet"
    registry = pd.read_parquet(score_registry_path)
    has_mapped = {"mapped_execution_ev", "mapping_available_at"}.issubset(bridge.columns)
    has_map_state = any("calibrator" in str(value).lower() or "isotonic" in str(value).lower() for value in manifest.get("inputs", {}).values())
    geometry_fields = [column for column in bridge.columns if column.startswith("context__")]
    requirements = [
        {
            "requirement": "causal recent-EV mapping state and candidate coordinates",
            "available": bool(has_mapped and has_map_state and "excluded" not in mapping_contract.lower()),
            "missing_columns_or_state": "mapped_execution_ev,mapping_available_at,calibrator_state,seed_history,calibration_support",
            "nearest_frozen_path": str(ROOT / "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/scored"),
            "minimal_request": "Create a successor bridge that binds the scorer population plus causal_recent_ev_state, seed_history and calibration_support, preserving the same candidate_id/decision keys.",
        },
        {
            "requirement": "one pooled-global causal mapped-EV H12 before/after labels",
            "available": False,
            "missing_columns_or_state": "mapped_execution_ev with decision-time availability (selection prerequisite)",
            "nearest_frozen_path": str(ROOT / "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/labels_12h/execution_ev_policy_labels.parquet"),
            "minimal_request": "After the mapping input is bound, join exact H12 labels by candidate_id/__ts__/symbol/side/decision and materialize a single pooled global top10 with candidate-id ties.",
        },
        {
            "requirement": "strict common 90-field signal+1h geometry",
            "available": bool(len(geometry_fields) == 90),
            "missing_columns_or_state": "the bridge exposes no strict context__ 90-field surface or raw nine-field source ledger",
            "nearest_frozen_path": str(ROOT / "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2/candidates_with_frozen_representation/candidate_features_with_representation.parquet"),
            "minimal_request": "Bind an exact candidate surface containing the nine raw common fields, feature_available_at and candidate identity; reconstruct 1/3/12h lags only by exact UTC reindex, without as-of fill.",
        },
    ]
    bounds = {
        "bridge_manifest": str(manifest_path), "bridge_manifest_sha256": sha256(manifest_path),
        "bridge_population": str(bridge_path), "bridge_population_sha256": sha256(bridge_path),
        "score_registry": str(score_registry_path), "score_registry_sha256": sha256(score_registry_path),
        "rows": int(len(bridge)), "candidate_ids": int(bridge.candidate_id.nunique()),
        "sides": bridge.side_name.astype(str).value_counts().to_dict(),
        "first_decision_utc": pd.to_datetime(bridge.execution_decision_utc, utc=True).min(),
        "last_decision_utc": pd.to_datetime(bridge.execution_decision_utc, utc=True).max(),
        "mapping_contract": mapping_contract, "registered_score_names": registry.get("score_name", pd.Series(dtype=str)).astype(str).tolist(),
    }
    return pd.DataFrame(requirements), bounds


def run(*, bridge: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    readiness, bounds = audit(bridge)
    stage = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}."))
    try:
        table_path = stage / "readiness.parquet"
        readiness.to_parquet(table_path, index=False, compression="zstd")
        manifest = {
            "schema": SCHEMA, "status": "FAIL_CLOSED_MISSING_UPSTREAM_PREREQUISITES_NO_MATERIALIZATION",
            "promotion_eligible": False, "materialization_legal": bool(readiness.available.all()),
            "reason": "The raw-score bridge explicitly excludes mapped execution EV and has no hash-bound causal mapping state or strict common geometry; no substitute population was used.",
            "bridge_bounds": bounds,
            "outputs": {"readiness": {"path": str(output_dir / table_path.name), "rows": int(len(readiness)), "sha256": sha256(table_path)}},
            "outputs_sha256": {table_path.name: sha256(table_path)},
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        }
        _write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(sha256(stage / "manifest.json") + "\n", encoding="utf-8")
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(_safe(run(bridge=args.bridge, output_dir=args.output_dir)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
