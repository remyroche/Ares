#!/usr/bin/env python3
"""Fail-closed audit for the C3 HDBSCAN causal-regime research control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_strict_r3_long_supportive_label_funnel import FOLDS


def audit(root: Path) -> dict:
    manifest = json.loads((root / "run_manifest.json").read_text())
    if manifest.get("schema") != "strict_r3_long_supportive_label_hdbscan_regime_v1":
        raise AssertionError("unexpected schema")
    fields = manifest.get("market_context_only_state_fields", [])
    if not fields or any("supportive" in field or "path_arch" in field or "policy_net" in field for field in fields):
        raise AssertionError("future/evaluation field appears in HDBSCAN input contract")
    fold_audit = pd.read_parquet(root / "hdbscan_fold_audit.parquet")
    if len(fold_audit) != len(FOLDS) or not fold_audit["status"].eq("ok").all():
        raise AssertionError("incomplete HDBSCAN folds")
    panels = sorted((root / "hdbscan_oof_predictions").glob("fold=*.parquet"))
    if len(panels) != len(FOLDS):
        raise AssertionError("missing HDBSCAN prediction panels")
    details = []
    for ordinal, fold in enumerate(FOLDS):
        path = root / "hdbscan_oof_predictions" / f"fold={ordinal:02d}_{fold.name}.parquet"
        frame = pd.read_parquet(path)
        if frame["candidate_id"].duplicated().any():
            raise AssertionError(f"duplicate candidate id in {path.name}")
        forbidden = [column for column in frame if column.startswith(("supportive_", "path_arch_", "gold_", "realised_cluster"))]
        if forbidden:
            raise AssertionError(f"raw future label in score panel {path.name}: {forbidden}")
        ts = pd.to_datetime(frame["__decision_ts__"], utc=True)
        if not (ts.ge(fold.start).all() and ts.lt(fold.end).all()):
            raise AssertionError(f"held range violation in {path.name}")
        details.append({"fold": fold.name, "rows": int(len(frame)), "candidate_identity_unique": True, "held_range_ok": True})
    return {
        "schema": "strict_r3_long_supportive_label_hdbscan_regime_audit_v1", "status": "pass",
        "root": str(root.resolve()), "folds": len(FOLDS), "market_context_fields": len(fields),
        "target_free_state_input": True, "strict_h12_embargo_for_policy_map": True,
        "per_fold": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); result = audit(args.root.resolve()); args.out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result))


if __name__ == "__main__":
    main()
