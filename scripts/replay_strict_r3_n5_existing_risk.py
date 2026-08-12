#!/usr/bin/env python3
"""Matched risk replay for the already-selected single-forest N5 overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_strict_r3_n5_canonical_selection import _portfolio_full  # noqa: E402
from scripts.run_strict_r3_trust_sizing_ablation import INPUTS, PERIODS, _load  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-2025", type=Path, required=True)
    parser.add_argument("--predictions-2026", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    risks: list[dict[str, object]] = []
    for year, prediction_path in ((2025, args.predictions_2025), (2026, args.predictions_2026)):
        source, _fields, _audit = _load(INPUTS[year])
        prediction = pd.read_parquet(prediction_path)
        prediction = prediction.loc[
            prediction["arm"].astype(str).eq("N5_drf_support_l110_meanrisk")
        ].drop_duplicates("candidate_id", keep="first")
        start, end = PERIODS[year]
        for arm, multiplier in (
            ("equal_control", pd.Series(1.0, index=prediction.index)),
            ("N5_drf_support_l110_meanrisk", prediction["trust_size_multiplier"]),
        ):
            output = prediction.loc[:, ["candidate_id"]].copy()
            output["trust_size_multiplier"] = multiplier.to_numpy(float)
            decisions, equity, monthly, risk = _portfolio_full(
                source,
                output,
                arm=arm,
                start=start,
                end=end,
            )
            risk["year"] = year
            risks.append(risk)
            decisions.to_parquet(args.out_dir / f"decisions_{year}_{arm}.parquet", index=False)
            equity.to_parquet(args.out_dir / f"equity_{year}_{arm}.parquet", index=False)
            monthly.to_parquet(args.out_dir / f"monthly_{year}_{arm}.parquet", index=False)
    pd.DataFrame(risks).to_parquet(args.out_dir / "portfolio_risk_metrics.parquet", index=False)
    manifest = {
        "schema": "strict_r3_existing_n5_matched_risk_v1",
        "n5": "N5_drf_support_l110_meanrisk",
        "control": "true 1.0 relative size, not the historical equal-arm 1.75 mapping",
        "portfolio": "8 concurrent, 2 entries/bar, 1/asset, 80% margin, 7x leverage",
        "initial_wallet": 1000.0,
        "years": [2025, 2026],
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir)}))


if __name__ == "__main__":
    main()
