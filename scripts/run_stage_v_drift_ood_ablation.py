#!/usr/bin/env python3
"""Run the real Stage-V grouped-MDA drift/OOD ablation.

The selector panel supplies frozen Stage-I selected features and grouped-MDA
audits.  ``--oos-panel`` is a row-level, later, source-separated OOS panel
with the same causal raw features and resolved labels; a surface manifest root
is recorded for lineage but never used as a substitute for absent raw fields.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_v_drift_ood_experiment import (
    StageVExperimentConfig,
    StageVLayerSource,
    run_stage_v_drift_ood_ablation,
)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _panel(selector_dir: Path) -> pd.DataFrame:
    features = pd.read_parquet(selector_dir / "selector_features.parquet")
    ledger = pd.read_parquet(selector_dir / "selector_ledger.parquet")
    if features.candidate_id.duplicated().any() or ledger.candidate_id.duplicated().any():
        raise ValueError("selector feature/ledger identities must be unique")
    return ledger.merge(features, on="candidate_id", how="inner", validate="one_to_one")


def _surface_lineage(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {"declared": False}
    if not root.is_dir():
        raise FileNotFoundError(f"OOS surface root is missing: {root}")
    manifests = sorted(root.rglob("manifest.json"))
    return {
        "declared": True,
        "root": str(root.resolve()),
        "manifest_count": len(manifests),
        "manifest_sha256": {str(path.relative_to(root)): _sha(path) for path in manifests},
    }


def _source(
    *, layer: str, side: str, selector_panel: pd.DataFrame, oos: pd.DataFrame,
    selection_dir: Path, target_column: str, surface_lineage: dict[str, Any],
) -> StageVLayerSource:
    manifest_path = selection_dir / side / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError(f"{layer}/{side} selection manifest is not complete")
    report = Path(str(manifest.get("mda_feature_selection_report", "")))
    if not report.is_file():
        raise FileNotFoundError(f"{layer}/{side} frozen MDA report is missing: {report}")
    report_json = _read_json(report)
    group_path = Path(str(report_json.get("group_audit_path", "")))
    if not group_path.is_file():
        raise FileNotFoundError(f"{layer}/{side} frozen group-MDA audit is missing: {group_path}")
    selected = tuple(str(item) for item in manifest.get("selected_feature_contract", manifest.get("selected_features", [])))
    if not selected:
        raise ValueError(f"{layer}/{side} selection manifest has no selected feature contract")
    selector_side = selector_panel.loc[selector_panel.side_name.astype(str).str.lower().eq(side)].copy()
    oos_side = oos.loc[oos.side_name.astype(str).str.lower().eq(side)].copy()
    return StageVLayerSource(
        layer=layer, side=side, selector=selector_side, oos=oos_side,
        raw_feature_names=selected, mda_group_audit=pd.read_csv(group_path), target_column=target_column,
        selector_manifest_sha256=_sha(manifest_path), oos_surface_lineage=surface_lineage,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--base-selection-dir", type=Path, required=True)
    parser.add_argument("--meta-selection-dir", type=Path, required=True)
    parser.add_argument("--oos-panel", type=Path, required=True)
    parser.add_argument("--oos-surface-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-target-column", default="r3_class",
        help="Native R3 class target (0=adverse, 1=weak, 2=robust clear).",
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--max-groups", type=int, default=24)
    args = parser.parse_args()
    selector = _panel(args.selector_dir)
    oos = pd.read_parquet(args.oos_panel)
    lineage = _surface_lineage(args.oos_surface_dir)
    sources = [
        _source(layer=layer, side=side, selector_panel=selector, oos=oos, selection_dir=directory,
                target_column=target, surface_lineage=lineage)
        for layer, directory, target in (
            ("base", args.base_selection_dir, args.base_target_column),
            # Direct FQ3 labels are built fold-locally from exact_net_bps and
            # the arm's strict same-side native base output.  This provenance
            # field is deliberately not a fitted independent meta target.
            ("meta", args.meta_selection_dir, "exact_net_bps"),
        )
        for side in ("long", "short")
    ]
    manifest = run_stage_v_drift_ood_ablation(
        sources=sources, output_dir=args.output_dir,
        config=StageVExperimentConfig(folds=args.folds, min_train_rows=args.min_train_rows, max_groups=args.max_groups),
    )
    print(json.dumps({"status": manifest["status"], "schema": manifest["schema"], "winner": manifest["winner"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
