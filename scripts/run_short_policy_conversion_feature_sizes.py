#!/usr/bin/env python3
"""Final untouched OOS feature-size report for a frozen policy-MDA ranking."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_short_policy_conversion_funnel import PolicySpec, run  # noqa: E402


SPEC = PolicySpec("P1_policy_bps_k32_linear", "Frozen Round-B winner", "policy_bps", truncation=32, gain_family="linear")


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    if root.exists():
        raise FileExistsError(root)
    payload = json.loads(args.selection.read_text())
    sets = payload.get("feature_sets")
    if not isinstance(sets, dict):
        raise ValueError("MDA selection has no feature_sets")
    root.mkdir(parents=True)
    for size in (15, 30, 60, 90, 115):
        fields = sets.get(str(size))
        if not isinstance(fields, list) or len(fields) != size:
            raise ValueError(f"MDA selection has invalid F{size}")
        run(out=root / f"F{size}", policies=args.policies.resolve(), features_path=args.features.resolve(), candidates_path=args.candidates.resolve(), fields=fields, train_start=_utc("2023-10-01T00:00:00Z"), oos_start=_utc("2024-10-01T00:00:00Z"), oos_end=_utc("2025-01-01T00:00:00Z"), specs=(SPEC,))
    (root / "run_manifest.json").write_text(json.dumps({"schema": "strict_r3_short_policy_conversion_feature_sizes_v1", "status": "complete", "selection": str(args.selection.resolve()), "selection_rule": payload.get("selection_rule"), "development_recommended_size": payload.get("recommended_feature_size_development_only"), "final_oos": "[2024-10-01, 2025-01-01) used for report only, not feature-size selection"}, indent=2) + "\n")
    print(root)


if __name__ == "__main__":
    main()
