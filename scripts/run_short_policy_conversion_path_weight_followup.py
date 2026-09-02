#!/usr/bin/env python3
"""One bounded target × weight confirmation after short path-label screening.

This is deliberately not a factorial search.  It compares the selected
activation target, the exact-policy control, and only the strongest newly
materialised path target under the one weighting recipe that won the prior
screen: equal month authority × equal timestamp-query authority.
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
    PolicySpec(
        "P0_policy_uniform",
        "Exact policy-bps control with uniform authority.",
        "policy_bps", truncation=40, gain_family="linear", query_hours=1,
    ),
    PolicySpec(
        "P9_activation_month_query",
        "Activation-before-adverse with equal month × query authority.",
        "activation_grade", truncation=40, gain_family="linear", query_hours=1,
        weight_kind="month_query",
    ),
    PolicySpec(
        "P7_squeeze_l100_month_query",
        "Three-hour MFE minus 1× pre-activation adverse, same authority.",
        "squeeze_l100_rank", truncation=40, gain_family="linear", query_hours=1,
        weight_kind="month_query",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "selection", "policies", "features", "candidates", "supportive-path"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True)
    fields = json.loads(args.selection.read_text())["feature_sets"]["90"]
    for fold, start, end in (
        ("mayjun", "2024-05-01", "2024-07-01"),
        ("julaug", "2024-07-01", "2024-09-01"),
        ("sep", "2024-09-01", "2024-10-01"),
    ):
        run(
            out=out / fold,
            policies=args.policies.resolve(),
            features_path=args.features.resolve(),
            candidates_path=args.candidates.resolve(),
            supportive_path=getattr(args, "supportive_path").resolve(),
            fields=fields,
            train_start=_utc("2023-10-01"),
            oos_start=_utc(start),
            oos_end=_utc(end),
            specs=SPECS,
        )
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_short_policy_path_weight_followup_v1",
        "status": "complete",
        "selection_window": "pre-2024-10 only",
        "reason": "one predeclared strongest-new-label interaction check",
        "specs": [spec.name for spec in SPECS],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
