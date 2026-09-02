#!/usr/bin/env python3
"""Sequential gain-shape screen for the selected short LambdaRank geometry."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_short_policy_conversion_funnel import GAIN_FAMILIES, PolicySpec, run


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


SPECS = tuple(
    PolicySpec(
        name=f"Dg_control_1h_k40_{family}",
        description=f"Selected short control; K=40; {family} label gains.",
        target_kind="policy_bps",
        query_hours=1,
        truncation=40,
        gain_family=family,
        objective="lambdarank",
        lambdarank_norm=True,
    )
    for family in GAIN_FAMILIES
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve(); root.mkdir(parents=True)
    fields = json.loads(args.selection.read_text())["feature_sets"]["90"]
    for fold, oos_start, oos_end in (
        ("mayjun", "2024-05-01", "2024-07-01"),
        ("julaug", "2024-07-01", "2024-09-01"),
        ("sep", "2024-09-01", "2024-10-01"),
    ):
        run(out=root / fold, policies=args.policies.resolve(), features_path=args.features.resolve(), candidates_path=args.candidates.resolve(), fields=fields, train_start=_utc("2023-10-01"), oos_start=_utc(oos_start), oos_end=_utc(oos_end), specs=SPECS)
    (root / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_short_policy_gain_screen_v1", "status": "complete",
        "selection_window": "pre-2024-10 only", "controlled_factor": "label_gain",
        "fixed_target": "policy_bps", "fixed_training_query_hours": 1,
        "fixed_truncation": 40, "fixed_lambdarank_norm": True,
        "fixed_objective": "lambdarank", "gain_families": list(GAIN_FAMILIES),
        "f90_selection": str(args.selection.resolve()),
    }, indent=2) + "\n")
    print(root)


if __name__ == "__main__":
    main()
