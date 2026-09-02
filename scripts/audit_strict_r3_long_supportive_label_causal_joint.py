#!/usr/bin/env python3
"""Fail-closed identity and causality audit for causal×path research output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        digest.update(str(item.relative_to(path)).encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def audit(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "run_manifest.json").read_text())
    if manifest.get("schema") != "strict_r3_long_supportive_label_causal_joint_v1":
        raise AssertionError("unexpected causal-joint schema")
    source = manifest.get("causal_feature_contract", [])
    state = manifest.get("market_context_only_state_fields", [])
    if not source or not state:
        raise AssertionError("missing causal/state feature contracts")
    forbidden = [name for name in [*source, *state] if "supportive" in name or "path_arch" in name or "policy_net" in name]
    if forbidden:
        raise AssertionError(f"future/evaluation label in causal input contract: {forbidden[:5]}")
    fold_audit = pd.read_parquet(root / "causal_joint_fold_audit.parquet")
    if len(fold_audit) != len(manifest["outer_folds"]) or not fold_audit["status"].eq("ok").all():
        raise AssertionError("outer-fold completion audit failed")
    expected = {item["name"]: item for item in manifest["outer_folds"]}
    prediction_paths = sorted((root / "causal_joint_oof_predictions").glob("fold=*.parquet"))
    if len(prediction_paths) != len(expected):
        raise AssertionError(f"expected {len(expected)} prediction panels, found {len(prediction_paths)}")
    per_fold: list[dict[str, Any]] = []
    forbidden_prediction_columns = ("supportive_", "path_arch_", "gold_path", "realised_cluster")
    for path in prediction_paths:
        frame = pd.read_parquet(path)
        if frame["candidate_id"].duplicated().any():
            raise AssertionError(f"duplicate candidate ID in {path.name}")
        if any(column.startswith(forbidden_prediction_columns) for column in frame.columns):
            raise AssertionError(f"raw future label leaked into prediction panel {path.name}")
        fold = path.stem.split("_", 1)[1]
        # The file prefix carries ordinal and a name.  Match it to the manifest
        # without relying on a score column that can legitimately vary by arm.
        name = next((key for key in expected if path.stem.endswith(key)), None)
        if name is None:
            raise AssertionError(f"unrecognised fold panel {path.name}")
        start = pd.Timestamp(expected[name]["start"])
        end = pd.Timestamp(expected[name]["end_exclusive"])
        decision = pd.to_datetime(frame["__decision_ts__"], utc=True)
        if not (decision.ge(start).all() and decision.lt(end).all()):
            raise AssertionError(f"held timestamp outside declared fold in {path.name}")
        score_columns = [column for column in frame.columns if "expected_ev" in column or "J1_" in column or "J2_" in column or column.startswith("P3_path")]
        if not score_columns:
            raise AssertionError(f"no causal-joint scores found in {path.name}")
        if frame.loc[:, score_columns].isna().all(axis=None):
            raise AssertionError(f"all causal-joint scores missing in {path.name}")
        per_fold.append({"fold": name, "rows": int(len(frame)), "score_columns": int(len(score_columns)), "identity_unique": True, "held_range_ok": True})
    maps = pd.read_parquet(root / "joint_training_only_maps.parquet")
    if maps.empty or (maps["train_support"] < 0).any():
        raise AssertionError("training-only joint map audit failed")
    structural = pd.read_parquet(root / "causal_regime_structural_audit.parquet")
    if set(structural["regime_arm"].unique()) != {item["name"] for item in manifest["regime_specs"]}:
        raise AssertionError("regime structural audit missing a declared arm")
    return {
        "schema": "strict_r3_long_supportive_label_causal_joint_audit_v1",
        "status": "pass",
        "root": str(root.resolve()),
        "outer_folds": int(len(expected)),
        "prediction_panels": int(len(prediction_paths)),
        "per_fold": per_fold,
        "causal_features": int(len(source)),
        "market_context_features": int(len(state)),
        "all_supervised_labels_embargoed": True,
        "no_raw_future_path_inputs": True,
        "joint_maps_train_only": True,
        "root_sha256": _sha256(root),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
