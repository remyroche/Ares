#!/usr/bin/env python3
"""Audit value coverage of the frozen T2 361-field training contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / (
    "data_perp/artifacts/controlled_target_supportive_prepared_ledger_20260801_v5/"
    "prepared_target_supportive_ledger.parquet"
)
DEFAULT_FEATURES = DEFAULT_LEDGER.parent / "frozen_raw_causal_features.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.features.read_text(encoding="utf-8"))
    fields = [str(x) for x in contract["raw_feature_columns"]]
    frame = pd.read_parquet(args.ledger, columns=fields)
    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    finite = np.isfinite(values)
    rows = int(len(frame))
    coverage = pd.DataFrame(
        {
            "feature": fields,
            "finite_rows": finite.sum(axis=0).astype(int),
            "finite_rate": finite.mean(axis=0),
            "missing_rows": (~finite).sum(axis=0).astype(int),
        }
    ).sort_values(["finite_rate", "feature"], kind="stable")
    complete_rows = finite.all(axis=1)
    summary = {
        "schema": "t2_frozen_feature_coverage_v1",
        "ledger": str(args.ledger.resolve()),
        "feature_contract": str(args.features.resolve()),
        "rows": rows,
        "feature_count": len(fields),
        "features_at_least_90pct_finite": int((coverage["finite_rate"] >= 0.90).sum()),
        "features_below_90pct_finite": int((coverage["finite_rate"] < 0.90).sum()),
        "all_361_finite_rows": int(complete_rows.sum()),
        "all_361_finite_rate": float(complete_rows.mean()),
        "median_feature_finite_rate": float(coverage["finite_rate"].median()),
        "mean_feature_finite_rate": float(coverage["finite_rate"].mean()),
        "note": (
            "This is a value-completeness audit. LightGBM can route missing values, "
            "but this does not make a sparse feature a complete input."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.out_dir / "per_feature_coverage.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
