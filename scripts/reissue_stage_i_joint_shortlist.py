#!/usr/bin/env python3
"""Reissue a completed Round-3 funnel as a joint-meta-only shortlist.

This is a metadata-only migration. It never refits a model or changes the
family winner bundles; it removes the obsolete base-only winner semantics and
rehashes the three mandatory R3/S/O joint finalists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_target_promotion import decide_round3_promotion
from extreme_price_movements.stage_i_shared_population import (
    SharedPopulationError,
    shared_population_contract_reference,
)
from scripts.run_stage_i_base_target_ablation import (
    _export_joint_target_finalists,
    file_sha256,
)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--funnel-dir", type=Path, required=True)
    parser.add_argument("--label-grid-dir", type=Path, required=True)
    parser.add_argument("--shared-population-dir", type=Path)
    args = parser.parse_args()

    root = args.funnel_dir.resolve()
    manifest_path = root / "run_manifest.json"
    scorecard_path = root / "target_selection_scorecard.parquet"
    old_decision_path = root / "target_promotion_decision.json"
    old_finalists_path = root / "winner_bundles" / "joint_finalists" / "target_finalist_contracts.json"
    required = (manifest_path, scorecard_path, old_decision_path, old_finalists_path)
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(f"completed Round-3 funnel is incomplete: {missing}")

    manifest = _read(manifest_path)
    if manifest.get("status") != "complete" or int(manifest.get("completed_round", -1)) != 3:
        raise ValueError("only a completed Round-3 funnel may be reissued")
    old_decision = _read(old_decision_path)
    old_finalists = _read(old_finalists_path)
    source_contract = dict(old_decision.get("source_contract") or {})
    source_contract.update({
        "semantic_migration": "base-only selected winner removed; R3/S/O require matching direct-FQ3 meta",
        "superseded_decision_sha256": str(old_decision.get("decision_sha256", "")),
        "scorecard_sha256": file_sha256(scorecard_path),
        "promotion_gate_module_sha256": file_sha256(
            Path(__file__).resolve().parents[1]
            / "extreme_price_movements" / "stage_i_target_promotion.py"
        ),
    })
    decision = decide_round3_promotion(pd.read_parquet(scorecard_path), source_contract=source_contract)
    decision_path = root / "target_joint_shortlist_decision.json"
    _atomic_json(decision_path, decision)

    r3_source = next(
        item["source"] for item in old_finalists["finalists"]
        if str(item.get("family")) == "R3_control"
    )
    base_selection_dir = Path(r3_source["base_selection_dir"])
    label_manifest = _read(args.label_grid_dir / "manifest.json")
    shared_population_contract = None
    if args.shared_population_dir is not None:
        try:
            shared_population_contract = shared_population_contract_reference(args.shared_population_dir)
        except SharedPopulationError as exc:
            raise ValueError(str(exc)) from exc
    selected_contract = dict(manifest["request"]["selected_feature_contract"])
    joint = _export_joint_target_finalists(
        output_dir=root,
        decision=decision,
        winner_bundles=list(manifest["winner_bundles"]),
        base_selection_dir=base_selection_dir,
        selected_contract=selected_contract,
        label_manifest=label_manifest,
        scorecard_path=scorecard_path,
        shared_population_contract=shared_population_contract,
    )

    legacy = manifest.get("target_promotion_decision")
    manifest["target_promotion_decision"] = None
    manifest["legacy_base_only_promotion_decision"] = {
        "status": "invalidated_base_only_selection_semantics",
        "path": str(old_decision_path),
        "sha256": file_sha256(old_decision_path),
        "prior_manifest_entry": legacy,
    }
    manifest["target_joint_shortlist_decision"] = {
        "path": str(decision_path), "sha256": file_sha256(decision_path), **decision,
    }
    manifest["joint_target_finalists"] = joint
    artifacts = dict(manifest.get("artifact_sha256") or {})
    artifacts[str(decision_path.relative_to(root))] = file_sha256(decision_path)
    artifacts[str(old_finalists_path.relative_to(root))] = file_sha256(old_finalists_path)
    manifest["artifact_sha256"] = artifacts
    manifest["next_step"] = (
        "run matching direct-FQ3 meta for R3, scalar S and ordinal O; select only "
        "on reconstructed joint-stack causal common-bps economics"
    )
    _atomic_json(manifest_path, manifest)
    print(json.dumps({
        "status": "complete",
        "decision": str(decision_path),
        "decision_sha256": decision["decision_sha256"],
        "joint_finalists": joint["path"],
        "joint_finalists_sha256": joint["sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
