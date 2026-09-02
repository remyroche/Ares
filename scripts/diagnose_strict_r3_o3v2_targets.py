#!/usr/bin/env python3
"""Recompute target-funnel diagnostics from sealed target-free receipts.

This does not refit or replace any score.  It joins the canonical policy
ledger only after reading each immutable receipt, adding the base-conditional
IC and base-correlation evidence required for the O3-v2 report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import run_strict_r3_o3v2_target_funnel as target


def run(*, target_root: Path, policy_path: Path, control_root: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((target_root / "run_manifest.json").read_text())
    months = tuple(pd.Timestamp(f"{value}-01", tz="UTC") for value in manifest["months"])
    arms = tuple(manifest["arms"])
    policy = pd.read_parquet(policy_path, columns=("candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"))
    policy["policy_label_available_ts"] = pd.to_datetime(policy["policy_label_available_ts"], utc=True, errors="coerce")
    rows = target._control_metrics(control_root, policy, months)
    for arm in arms:
        for month in months:
            source = target_root / "target_free_scores" / arm / f"month={month:%Y-%m}.parquet"
            score = pd.read_parquet(source)
            prohibited = target.PROHIBITED_SCORE_COLUMNS.intersection(score.columns)
            if prohibited:
                raise AssertionError(f"{source}: outcome columns in target-free receipt {sorted(prohibited)}")
            rows.extend(target._metric_rows(score, policy, arm=arm, month=month))
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(out / "target_funnel_diagnostics.parquet", index=False, compression="zstd")
    (out / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_o3v2_target_diagnostics_v1", "target_root": str(target_root),
        "policy_path": str(policy_path), "control_root": str(control_root),
        "causality": "sealed target-free receipts read first; policy joined only for diagnostics; no refit or score mutation",
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(target_root=args.target_root, policy_path=args.policy_path, control_root=args.control_root, out=args.out)


if __name__ == "__main__":
    main()
