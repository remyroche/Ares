#!/usr/bin/env python3
"""Compare a schema-v13 incremental score checkpoint with canonical output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _compare(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, object]:
    left = left.copy().set_index("candidate_id").sort_index()
    right = right.copy().set_index("candidate_id").sort_index()
    different: list[str] = []
    maximum_delta = 0.0
    for field in sorted(set(left.columns).intersection(right.columns)):
        a, b = left[field], right[field]
        if pd.api.types.is_numeric_dtype(a) or pd.api.types.is_numeric_dtype(b):
            av = pd.to_numeric(a, errors="coerce").to_numpy(float)
            bv = pd.to_numeric(b, errors="coerce").to_numpy(float)
            delta = np.abs(av - bv)
            if np.isfinite(delta).any():
                maximum_delta = max(maximum_delta, float(np.nanmax(delta)))
            exact = np.allclose(av, bv, atol=0.0, rtol=0.0, equal_nan=True)
        else:
            exact = a.astype(str).equals(b.astype(str))
        if not exact:
            different.append(field)
    return {
        "rows_left": int(len(left)),
        "rows_right": int(len(right)),
        "identities_exact": left.index.equals(right.index),
        "common_fields": int(len(set(left.columns).intersection(right.columns))),
        "different_fields": different,
        "maximum_numeric_delta": maximum_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--incremental", type=Path, required=True)
    parser.add_argument("--decision-ts", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    decision = pd.Timestamp(args.decision_ts)
    decision = decision.tz_localize("UTC") if decision.tzinfo is None else decision.tz_convert("UTC")
    left = pd.read_parquet(args.canonical)
    right = pd.read_parquet(args.incremental)
    left = left.loc[pd.to_datetime(left["__decision_ts__"], utc=True).eq(decision)]
    right = right.loc[pd.to_datetime(right["__decision_ts__"], utc=True).eq(decision)]
    comparison = _compare(left, right)
    checks = {
        "identities_exact": bool(comparison["identities_exact"]),
        "rows_exact": comparison["rows_left"] == comparison["rows_right"],
        "all_common_fields_bit_exact": comparison["different_fields"] == [],
        "zero_numeric_delta": comparison["maximum_numeric_delta"] == 0.0,
    }
    result = {
        "schema": "strict_r3_incremental_score_parity_v1",
        "status": "pass" if all(checks.values()) else "fail",
        "decision_ts": decision.isoformat(),
        "checks": checks,
        "comparison": comparison,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if result["status"] != "pass":
        raise AssertionError(f"incremental score parity failed: {comparison}")


if __name__ == "__main__":
    main()
