#!/usr/bin/env python3
"""Screen only policy-family supportive labels for the frozen short ranker.

Path auxiliary labels require a longer materialisation before they can be
compared on all three pre-October folds.  This screen first tests the four
already-complete, policy-family alternatives with identical strict-OOF rows.
"""
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
    PolicySpec("S0_policy_bps", "Frozen policy-bps control.", "policy_bps", truncation=40, gain_family="linear", query_hours=1),
    PolicySpec("S1_median_policy", "Median raw policy-family outcome, timestamp ranked.", "median_rank", truncation=40, gain_family="linear", query_hours=1),
    PolicySpec("S2_trimmed_policy", "Trimmed-mean raw policy-family outcome, timestamp ranked.", "trimmed_mean_rank", truncation=40, gain_family="linear", query_hours=1),
    PolicySpec("S3_mean_family_rank", "Mean of seven policy-family timestamp ranks.", "mean_family_rank", truncation=40, gain_family="linear", query_hours=1),
    PolicySpec("S4_p25_family_rank", "P25 of seven policy-family timestamp ranks.", "p25_family_rank", truncation=40, gain_family="linear", query_hours=1),
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
        "schema": "strict_r3_short_policy_supportive_screen_v1", "status": "complete",
        "selection_window": "pre-2024-10 only", "controlled_factor": "policy-family supportive label",
        "frozen_ranker": {"target_control": "policy_bps", "query_hours": 1, "truncation": 40, "gain_family": "linear", "objective": "lambdarank", "lambdarank_norm": True},
        "specs": [spec.name for spec in SPECS], "f90_selection": str(args.selection.resolve()),
    }, indent=2) + "\n")
    print(root)


if __name__ == "__main__":
    main()
