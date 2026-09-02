#!/usr/bin/env python3
"""Summarise the sequential short absolute-alpha base screen.

The runner deliberately stores each arm in an immutable worker directory.  This
script turns completed workers into a reproducible selection receipt without
re-running a model or looking at any later OOS fold.  It is safe to run more
than once: output files are only created for a new output directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _one_worker(path: Path) -> dict[str, object]:
    tail = pd.read_parquet(path / "global_policy_tail_metrics.parquet")
    within = pd.read_parquet(path / "within_timestamp_scorecard.parquet")
    manifest = json.loads((path / "run_manifest.json").read_text())
    oos = tail.loc[tail.scope.eq("oos")]
    month = tail.loc[~tail.scope.eq("oos")]

    def _tail(fraction: float) -> float:
        value = oos.loc[oos.tail_fraction.eq(fraction), "policy_net_bps"]
        if len(value) != 1:
            raise ValueError(f"{path}: expected exactly one OOS tail {fraction}")
        return float(value.iloc[0])

    policy_ic = within.loc[
        within.scope.eq("oos")
        & within.score_form.eq("score")
        & within.metric.isna(),
        "policy_ic_weighted",
    ]
    if len(policy_ic) != 1:
        raise ValueError(f"{path}: expected exactly one OOS policy IC row")
    row = {
        "worker": str(path),
        "arm": str(manifest["specs"][0]["name"]),
        "description": str(manifest["specs"][0]["description"]),
        "query_hours": int(manifest["specs"][0]["query_hours"]),
        "top_025_net_bps": _tail(0.0025),
        "top_05_net_bps": _tail(0.005),
        "top_1_net_bps": _tail(0.01),
        "top_2_net_bps": _tail(0.02),
        "worst_month_top_05_net_bps": float(
            month.loc[month.tail_fraction.eq(0.005), "policy_net_bps"].min()
        ),
        "policy_ic": float(policy_ic.iloc[0]),
    }
    # Predeclared: prioritize extremely selective absolute economics, then
    # penalise a negative monthly tail.  It has no final-fold authority.
    row["development_selection_score"] = (
        0.4 * row["top_025_net_bps"]
        + 0.3 * row["top_05_net_bps"]
        + 0.2 * row["top_1_net_bps"]
        + 0.1 * row["top_2_net_bps"]
        - max(0.0, -row["worst_month_top_05_net_bps"])
    )
    return row


def run(*, workers: Path, out: Path) -> Path:
    if out.exists():
        raise FileExistsError(out)
    selected = sorted(
        list(workers.glob("A_query_breadth__D1__*/"))
        + list(workers.glob("B_absolute_label__D1__*/"))
    )
    if not selected:
        raise FileNotFoundError("no completed query/label worker directories")
    metrics = pd.DataFrame(_one_worker(worker) for worker in selected)
    metrics = metrics.sort_values(
        ["development_selection_score", "arm"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    winner = metrics.iloc[0]
    gate = {
        "top_05_gt_50_bps": bool(winner.top_05_net_bps > 50.0),
        "top_1_gt_25_bps": bool(winner.top_1_net_bps > 25.0),
        "no_development_month_below_zero_at_top_05": bool(
            winner.worst_month_top_05_net_bps >= 0.0
        ),
    }
    decision = {
        "schema": "strict_r3_short_absolute_alpha_decision_v1",
        "side": "short",
        "stage": "D1 development-only query and absolute-label screen",
        "winner": winner.to_dict(),
        "standalone_base_promotion": bool(all(gate.values())),
        "promotion_gates": gate,
        "decision": (
            "NO_STANDALONE_BASE_PROMOTION: no alternate query or absolute label improves "
            "the frozen Q1 policy control, and the control itself fails every absolute-tail gate."
        ),
        "next_step": (
            "Do not run confirmation, final-fold, hybrid, weighting, or HPO arms for this base branch. "
            "The separately materialised top-1-per-hour oracle diagnostic provides strong admission headroom, "
            "so downstream reliability/admission research remains justified."
        ),
        "strictness": {
            "features": "frozen short F90 causal contract",
            "labels": "canonical policy net after the fixed 100-bps cost, resolved before OOS",
            "scoring": "target-free OOS candidates; labels joined only for metrics",
            "selection": "D1 only; no later fold informed this decision",
        },
    }
    out.mkdir(parents=True)
    metrics.to_parquet(out / "absolute_alpha_screen_metrics.parquet", index=False, compression="zstd")
    (out / "absolute_alpha_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(workers=args.workers.resolve(), out=args.out.resolve()))


if __name__ == "__main__":
    main()
