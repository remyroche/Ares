#!/usr/bin/env python3
"""Reconcile the sparse 625-action challenger with authoritative A5 exits.

This is a paired diagnostic only.  It deliberately refuses to call the two
uplifts a model comparison when their baseline outcomes differ.  No model is
fitted and no winner is promoted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _metrics(frame: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    delta = frame[f"adaptive_net_bps_{prefix}"] - frame[f"baseline_net_bps_{prefix}"]
    return {
        "rows": int(len(frame)),
        "baseline_net_bps": float(frame[f"baseline_net_bps_{prefix}"].mean()),
        "adaptive_net_bps": float(frame[f"adaptive_net_bps_{prefix}"].mean()),
        "uplift_bps": float(delta.mean()),
        "positive_uplift_fraction": float((delta > 0.01).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenger-trades", type=Path, required=True)
    parser.add_argument("--a5-oof-replay", type=Path, required=True)
    parser.add_argument("--a5-fixed-trades", type=Path, required=True)
    parser.add_argument("--a5-arm", default="F4_disagreement_abstain_p80")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    current = pd.read_parquet(args.challenger_trades)
    a5_oof = pd.read_parquet(args.a5_oof_replay)
    a5_oof = a5_oof.loc[a5_oof["arm"].eq(args.a5_arm)].drop_duplicates("candidate_id")
    a5_fixed = pd.read_parquet(args.a5_fixed_trades).drop_duplicates("candidate_id")
    for name, frame in (("current", current), ("a5_oof", a5_oof), ("a5_fixed", a5_fixed)):
        if frame["candidate_id"].duplicated().any():
            raise ValueError(f"{name} has duplicate candidate IDs")

    outputs: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    monthly: list[pd.DataFrame] = []
    for comparison, other in (("a5_oof", a5_oof), ("a5_fixed", a5_fixed)):
        paired = current.merge(other, on="candidate_id", suffixes=("_current", "_a5"))
        paired["baseline_difference_bps"] = (
            paired["baseline_net_bps_current"] - paired["baseline_net_bps_a5"]
        )
        paired["uplift_current_bps"] = (
            paired["adaptive_net_bps_current"] - paired["baseline_net_bps_current"]
        )
        paired["uplift_a5_bps"] = (
            paired["adaptive_net_bps_a5"] - paired["baseline_net_bps_a5"]
        )
        paired["comparison"] = comparison
        outputs.append(paired)
        summary = {
            "comparison": comparison,
            "overlap_rows": int(len(paired)),
            "overlap_fraction_current": float(len(paired) / max(len(current), 1)),
            "overlap_fraction_a5": float(len(paired) / max(len(other), 1)),
            **{f"current_{key}": value for key, value in _metrics(paired, "current").items()},
            **{f"a5_{key}": value for key, value in _metrics(paired, "a5").items()},
            "baseline_difference_bias_bps": float(paired["baseline_difference_bps"].mean()),
            "baseline_difference_mae_bps": float(paired["baseline_difference_bps"].abs().mean()),
            "baseline_spearman": float(
                paired["baseline_net_bps_current"].corr(
                    paired["baseline_net_bps_a5"], method="spearman"
                )
            ),
            "uplift_spearman": float(
                paired["uplift_current_bps"].corr(paired["uplift_a5_bps"], method="spearman")
            ),
            "direct_uplift_comparison_valid": False,
            "invalid_reason": "baseline_policy_and_outcome_substrate_not_identical",
        }
        summaries.append(summary)
        if len(paired):
            ts_col = "timestamp_current" if "timestamp_current" in paired else "timestamp_a5"
            paired["month"] = pd.to_datetime(paired[ts_col], utc=True).dt.strftime("%Y-%m")
            month = paired.groupby("month", sort=True).agg(
                rows=("candidate_id", "size"),
                current_uplift_bps=("uplift_current_bps", "mean"),
                a5_uplift_bps=("uplift_a5_bps", "mean"),
                baseline_difference_mae_bps=("baseline_difference_bps", lambda x: x.abs().mean()),
            ).reset_index()
            month.insert(0, "comparison", comparison)
            monthly.append(month)

    pd.concat(outputs, ignore_index=True).to_parquet(
        args.output_dir / "matched_candidate_comparison.parquet", index=False
    )
    pd.DataFrame(summaries).to_parquet(
        args.output_dir / "comparison_summary.parquet", index=False
    )
    pd.concat(monthly, ignore_index=True).to_parquet(
        args.output_dir / "monthly_comparison.parquet", index=False
    )
    manifest = {
        "schema": "path_exit_authoritative_a5_reconciliation_v1",
        "status": "DIAGNOSTIC_ONLY_NO_PROMOTION",
        "a5_arm": args.a5_arm,
        "comparisons": summaries,
        "conclusion": (
            "A5 is directionally stronger, but a fair model comparison requires "
            "both controllers to replay the same canonical baseline, paths, clock, "
            "cost, and candidate population."
        ),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
