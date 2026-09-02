#!/usr/bin/env python3
"""Fail-closed audit for observed-entry supportive H12 path labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = (
    "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
    "supportive_label_available_ts", "supportive_path_valid", "supportive_target_invalid",
    "supportive_peak_mfe_atr_h12", "supportive_path_efficiency_h12", "policy_path_valid",
)


def run(*, root: Path, min_coverage: float) -> dict[str, object]:
    manifest = json.loads((root / "run_manifest.json").read_text())
    if manifest.get("schema") != "strict_r3_long_supportive_path_labels_v2_h12_15m_observed_entry_causal_atr":
        raise AssertionError("unexpected sidecar schema")
    parts = sorted((root / "parts").glob("month=*/side=long.parquet"))
    if not parts:
        raise AssertionError("no monthly label parts")
    rows = valid_rows = policy_invalid_but_valid = 0
    per_month: list[dict[str, object]] = []
    for path in parts:
        frame = pd.read_parquet(path, columns=list(REQUIRED))
        if frame["candidate_id"].duplicated().any():
            raise AssertionError(f"duplicate identity in {path}")
        ts = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        available = pd.to_datetime(frame["supportive_label_available_ts"], utc=True, errors="raise")
        valid = frame["supportive_path_valid"].fillna(False).astype(bool)
        invalid = frame["supportive_target_invalid"].fillna(True).astype(bool)
        if not valid.eq(~invalid).all():
            raise AssertionError(f"validity complement failure in {path}")
        if not available.eq(ts + pd.Timedelta(hours=12)).all():
            raise AssertionError(f"H12 availability failure in {path}")
        forbidden = ["supportive_peak_mfe_atr_h12", "supportive_path_efficiency_h12"]
        if frame.loc[invalid, forbidden].notna().any(axis=None):
            raise AssertionError(f"invalid target encoded as outcome in {path}")
        rows += len(frame); valid_rows += int(valid.sum())
        policy_invalid_but_valid += int((~frame["policy_path_valid"].fillna(False).astype(bool) & valid).sum())
        per_month.append({"month": path.parent.name.removeprefix("month="), "rows": int(len(frame)), "valid_rows": int(valid.sum())})
    coverage = valid_rows / max(rows, 1)
    if coverage < min_coverage:
        raise AssertionError(f"path coverage {coverage:.4%} below required {min_coverage:.2%}")
    return {
        "schema": "strict_r3_long_supportive_path_labels_observed_entry_audit_v1",
        "status": "pass", "rows": rows, "valid_rows": valid_rows, "coverage": coverage,
        "policy_invalid_but_path_valid_rows": policy_invalid_but_valid,
        "path_targets_are_independent_of_policy_label_availability": policy_invalid_but_valid > 0,
        "per_month": per_month,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-coverage", type=float, default=.90)
    args = parser.parse_args()
    result = run(root=args.root.resolve(), min_coverage=float(args.min_coverage))
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
