#!/usr/bin/env python3
"""Round-D capacity sweep for the development-selected short policy base."""
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
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


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
    size = int(payload["recommended_feature_size_development_only"])
    fields = payload["feature_sets"][str(size)]
    root.mkdir(parents=True)
    for leaves in (15, 31, 63):
        for min_child in (350, 700):
            for trees in (140, 250):
                name = f"L{leaves}_C{min_child}_T{trees}"
                run(out=root / name, policies=args.policies.resolve(), features_path=args.features.resolve(), candidates_path=args.candidates.resolve(), fields=fields, train_start=_utc("2023-10-01T00:00:00Z"), oos_start=_utc("2024-10-01T00:00:00Z"), oos_end=_utc("2025-01-01T00:00:00Z"), specs=(SPEC,), model_overrides={"num_leaves": leaves, "min_child_samples": min_child, "n_estimators": trees})
    (root / "run_manifest.json").write_text(json.dumps({"schema": "strict_r3_short_policy_conversion_capacity_sweep_v1", "status": "complete", "selection": str(args.selection.resolve()), "feature_size": size, "grid": {"num_leaves": [15,31,63], "min_child_samples": [350,700], "n_estimators": [140,250]}, "final_oos": "[2024-10-01,2025-01-01)"}, indent=2) + "\n")
    print(root)


if __name__ == "__main__":
    main()
