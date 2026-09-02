#!/usr/bin/env python3
"""Sequential LambdaRank truncation screen for the selected short base target.

Formulation and query construction have already been selected on the same
pre-October development folds.  This step changes only ``K``.  Gain shape,
normalisation, objective, feature contract, rows and folds stay frozen.
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


TRUNCATIONS = (12, 16, 24, 32, 40, 48, 64)
SPECS = tuple(
    PolicySpec(
        name=f"Ck_control_1h_k{k}",
        description=f"Selected control target, 1h query, truncation K={k}.",
        target_kind="policy_bps",
        query_hours=1,
        truncation=k,
        gain_family="linear",
        objective="lambdarank",
        lambdarank_norm=True,
    )
    for k in TRUNCATIONS
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True)
    fields = json.loads(args.selection.read_text())["feature_sets"]["90"]
    for fold, oos_start, oos_end in (
        ("mayjun", "2024-05-01", "2024-07-01"),
        ("julaug", "2024-07-01", "2024-09-01"),
        ("sep", "2024-09-01", "2024-10-01"),
    ):
        run(
            out=root / fold,
            policies=args.policies.resolve(),
            features_path=args.features.resolve(),
            candidates_path=args.candidates.resolve(),
            fields=fields,
            train_start=_utc("2023-10-01"),
            oos_start=_utc(oos_start),
            oos_end=_utc(oos_end),
            specs=SPECS,
        )
    (root / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_short_policy_k_screen_v1",
        "status": "complete",
        "selection_window": "pre-2024-10 only",
        "controlled_factor": "lambdarank_truncation_level",
        "fixed_target": "policy_bps",
        "fixed_training_query_hours": 1,
        "fixed_gain_family": "linear",
        "fixed_lambdarank_norm": True,
        "fixed_objective": "lambdarank",
        "truncations": list(TRUNCATIONS),
        "f90_selection": str(args.selection.resolve()),
    }, indent=2) + "\n")
    print(root)


if __name__ == "__main__":
    main()
