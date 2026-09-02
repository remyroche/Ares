#!/usr/bin/env python3
"""Untouched October–December 2024 confirmation for the short-base funnel."""
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
        "Exact policy-bps control retained from development.",
        "policy_bps", truncation=40, gain_family="linear", query_hours=1,
    ),
    PolicySpec(
        "P9_activation_month_query",
        "Selected activation-before-adverse target with equal month × query authority.",
        "activation_grade", truncation=40, gain_family="linear", query_hours=1,
        weight_kind="month_query",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("out", "selection", "policies", "features", "candidates", "supportive-path"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    fields = json.loads(args.selection.read_text())["feature_sets"]["90"]
    run(
        out=out,
        policies=args.policies.resolve(), features_path=args.features.resolve(),
        candidates_path=args.candidates.resolve(), supportive_path=getattr(args, "supportive_path").resolve(),
        fields=fields, train_start=_utc("2024-01-01"),
        oos_start=_utc("2024-10-01"), oos_end=_utc("2025-01-01"), specs=SPECS,
    )
    manifest_path = out / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["confirmation_status"] = "untouched_target_selection_confirmation"
    manifest["selection_history"] = "May–September 2024 only; October–December was not read before this run."
    manifest["hpo_note"] = "Early-stopped broad HPO is not promoted here; this validates the selected frozen-geometry target contract."
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
