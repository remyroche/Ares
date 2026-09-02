#!/usr/bin/env python3
"""Apply the frozen Stage-1 base advancement screen to :00-only artifacts.

This is a screen, not a model-selection shortcut.  It combines B0/B1 results
with the corrected canonical-policy B3/B4/B5 artifact and advances no arm
unless it clears every quantitative gate that can be evaluated before the
mandatory downstream residual/MC1 reconstruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data_perp/artifacts"
DEFAULT_B0 = ARTIFACTS / "strict_r3_long_base_recall_funnel_2025dev_holdout_2026oos_20260822_v1"
DEFAULT_UTILITY = ARTIFACTS / "strict_r3_long_base_utility_funnel_2025dev_holdout_2026oos_20260822_v2_canonical_policy"
PERIODS = ("frozen_holdout_2025q4", "frozen_oos_2026jan_jul")


def gate_row(
    base: pd.DataFrame,
    challenger: pd.DataFrame,
    *,
    arm: str,
    family: str,
) -> dict[str, Any]:
    """Evaluate the pre-downstream Stage-1 gates on matching fixed-30% rows."""

    reference = base.set_index("period")
    current = challenger.set_index("period")
    if not set(PERIODS).issubset(reference.index) or not set(PERIODS).issubset(current.index):
        raise AssertionError(f"missing required holdout/portability periods for {arm}")
    result: dict[str, Any] = {"arm": arm, "family": family}
    all_recall, all_mean, all_ic = [], [], []
    for period in PERIODS:
        recall_relative = float(current.loc[period, "recall_composite"] / reference.loc[period, "recall_composite"] - 1.0)
        mean_delta = float(current.loc[period, "routed_policy_net_mean_bps"] - reference.loc[period, "routed_policy_net_mean_bps"])
        ic_delta = float(current.loc[period, "rank_ic"] - reference.loc[period, "rank_ic"])
        recall100_delta = float(current.loc[period, "recall100"] - reference.loc[period, "recall100"])
        prefix = "holdout" if period == PERIODS[0] else "portability"
        result.update({
            f"{prefix}_recall_relative": recall_relative,
            f"{prefix}_mean_net_delta_bps": mean_delta,
            f"{prefix}_rank_ic_delta": ic_delta,
            f"{prefix}_recall100_delta": recall100_delta,
        })
        all_recall.append(recall_relative >= .02)
        all_mean.append(mean_delta >= -5.0)
        all_ic.append(ic_delta >= -.005)
    result["passes_recall_gate"] = bool(all(all_recall))
    result["passes_mean_net_gate"] = bool(all(all_mean))
    result["passes_rank_ic_gate"] = bool(all(all_ic))
    # Per-quarter +100 recall and full downstream economics are deliberately
    # not inferred from a failed aggregate screen.  They are required before
    # acceptance for any arm that reaches this pre-gate.
    result["eligible_for_downstream_rebuild"] = bool(all(all_recall + all_mean + all_ic))
    return result


def _standard_b0(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.loc[metrics["arm"].eq("B0_D2_route30")].copy()
    return result.loc[:, [
        "period", "recall_composite", "routed_policy_net_mean_bps",
        "equal_timestamp_rank_ic", "row_recall__policy_ge_100",
    ]].rename(columns={
        "equal_timestamp_rank_ic": "rank_ic",
        "row_recall__policy_ge_100": "recall100",
    })


def _standard_b1(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.loc[
        metrics["family"].eq("B1") & metrics["route_fraction"].between(.299, .301)
    ].copy()
    return result.loc[:, [
        "arm", "family", "period", "recall_composite", "routed_policy_net_mean_bps",
        "equal_timestamp_rank_ic", "row_recall__policy_ge_100",
    ]].rename(columns={
        "equal_timestamp_rank_ic": "rank_ic",
        "row_recall__policy_ge_100": "recall100",
    })


def _standard_utility(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.loc[metrics["route_fraction"].between(.299, .301)].copy()
    return result.loc[:, [
        "arm", "family", "period", "recall_composite", "routed_policy_net_mean_bps",
        "equal_timestamp_rank_ic", "row_recall__policy_ge_100",
    ]].rename(columns={
        "equal_timestamp_rank_ic": "rank_ic",
        "row_recall__policy_ge_100": "recall100",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-root", type=Path, default=DEFAULT_B0)
    parser.add_argument("--utility-root", type=Path, default=DEFAULT_UTILITY)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    b0 = pd.read_parquet(args.b0_root / "base_recall_metrics.parquet")
    utility = pd.read_parquet(args.utility_root / "utility_base_recall_metrics.parquet")
    reference = _standard_b0(b0)
    challengers = pd.concat([_standard_b1(b0), _standard_utility(utility)], ignore_index=True)
    results = []
    for arm, frame in challengers.groupby("arm", sort=True):
        results.append(gate_row(reference, frame, arm=str(arm), family=str(frame["family"].iloc[0])))
    result = pd.DataFrame(results).sort_values(["eligible_for_downstream_rebuild", "arm"], ascending=[False, True])
    args.out_dir.mkdir(parents=True)
    result.to_parquet(args.out_dir / "stage1_pre_downstream_gate.parquet", index=False)
    manifest = {
        "schema": "strict_r3_stage1_pre_downstream_gate_v1",
        "scope": "offline long-only :00-only research; no live or canonical artifact changed",
        "input_b0": str(args.b0_root),
        "input_utility": str(args.utility_root),
        "gates": {
            "recall_relative": ">= 2% on both 2025Q4 and 2026 portability",
            "routed_policy_net_delta_bps": ">= -5 on both periods",
            "rank_ic_delta": ">= -0.005 on both periods",
            "remaining_required_before_acceptance": [
                "quarter-by-quarter policy>=100 recall never declines",
                "full canonical one-residual downstream reconstruction",
                "MC1 admission, common constrained portfolio, frozen rich parent, Adaptive Exit V1 where exact OOF state exists",
            ],
        },
        "downstream_eligible_arms": result.loc[result["eligible_for_downstream_rebuild"], "arm"].tolist(),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "stage1_pre_downstream_gate_complete", "eligible": manifest["downstream_eligible_arms"]}, sort_keys=True))


if __name__ == "__main__":
    main()
