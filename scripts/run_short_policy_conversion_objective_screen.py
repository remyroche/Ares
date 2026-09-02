#!/usr/bin/env python3
"""Final objective/normalisation check for the selected short base ranker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_short_policy_conversion_funnel import PolicySpec, run


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


SPECS = (
    PolicySpec("E0_lambdarank_norm", "Selected LambdaRank control.", "policy_bps", truncation=40, gain_family="linear", query_hours=1, objective="lambdarank", lambdarank_norm=True),
    PolicySpec("E1_lambdarank_no_norm", "Matched LambdaRank without normalisation.", "policy_bps", truncation=40, gain_family="linear", query_hours=1, objective="lambdarank", lambdarank_norm=False),
    PolicySpec("E2_rank_xendcg", "Matched RankXENDCG control.", "policy_bps", truncation=40, gain_family="linear", query_hours=1, objective="rank_xendcg", lambdarank_norm=True),
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
        "schema": "strict_r3_short_policy_objective_screen_v1", "status": "complete",
        "selection_window": "pre-2024-10 only", "controlled_factor": "ranker objective / LambdaRank normalisation",
        "fixed_target": "policy_bps", "fixed_training_query_hours": 1,
        "fixed_truncation": 40, "fixed_gain_family": "linear",
        "specs": [spec.name for spec in SPECS], "f90_selection": str(args.selection.resolve()),
    }, indent=2) + "\n")
    print(root)


if __name__ == "__main__":
    main()
