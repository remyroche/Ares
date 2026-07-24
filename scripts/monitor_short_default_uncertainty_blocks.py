#!/usr/bin/env python3
"""Monitor frozen short-default uncertainty evidence by independent blocks.

The monitor is deliberately observational.  It reads the already-frozen
challenger score and never changes ranks, thresholds, components, or policy.
An evidence block starts only after a cooling period, so consecutive high-score
days are one observation rather than several correlated confirmations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_short_default_uncertainty_ablation import _percentile


GROUP = ("short", "short_default_clean_path")
KEYS = ["__ts__", "side_name", "archetype_policy_key"]
COMPONENTS = (
    "ensemble_risk_std",
    "neighbor_shrunken_adverse_rate",
    "neighbor_weighted_ev_std",
)


def _daily_conjunction(
    challenger: pd.DataFrame,
    train_diagnostics: pd.DataFrame,
    oos_diagnostics: pd.DataFrame,
    *,
    threshold: float,
    reliability_floor: float,
) -> pd.DataFrame:
    train = train_diagnostics.loc[
        train_diagnostics["stage"].eq("train_oof")
        & train_diagnostics["side_name"].eq(GROUP[0])
        & train_diagnostics["archetype_policy_key"].eq(GROUP[1]),
        [*KEYS, *COMPONENTS],
    ].drop_duplicates(KEYS, keep="last")
    score = oos_diagnostics.loc[
        oos_diagnostics["stage"].eq("eval_oos")
        & oos_diagnostics["side_name"].eq(GROUP[0])
        & oos_diagnostics["archetype_policy_key"].eq(GROUP[1]),
        [*KEYS, *COMPONENTS, "neighbor_effective_count"],
    ].drop_duplicates(KEYS, keep="last")
    local = challenger.loc[
        challenger["side_name"].eq(GROUP[0])
        & challenger["archetype_policy_key"].eq(GROUP[1]),
        [
            "__ts__",
            "side_name",
            "archetype_policy_key",
            "short_default_uncertainty_score",
            "parent_rank_v9_residual_error_overlay",
            "frozen_short_default_uncertainty_rank",
            "ev_after_1pct",
            "clean_exec",
        ],
    ].copy()
    for frame in (train, score, local):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    local = local.merge(score, on=KEYS, how="inner", validate="many_to_one")
    if local.empty:
        raise ValueError("No challenger rows match the frozen diagnostic source")
    for name in COMPONENTS:
        reference = pd.to_numeric(train[name], errors="coerce").to_numpy(np.float32)
        local[f"{name}_percentile"] = _percentile(
            pd.to_numeric(local[name], errors="coerce").to_numpy(np.float32),
            reference,
        )
    local["neighbor_reliability"] = (
        pd.to_numeric(local["neighbor_effective_count"], errors="coerce")
        / (pd.to_numeric(local["neighbor_effective_count"], errors="coerce") + 20.0)
    ).astype(np.float32)
    component_high = np.logical_and.reduce(
        [local[f"{name}_percentile"].ge(threshold).to_numpy() for name in COMPONENTS]
    )
    local["conjunction_active"] = (
        local["short_default_uncertainty_score"].ge(threshold).to_numpy()
        & component_high
        & local["neighbor_reliability"].ge(reliability_floor).to_numpy()
    )
    local["day"] = local["__ts__"].dt.floor("D")
    return local.groupby("day", observed=True).agg(
        rows=("day", "size"),
        conjunction_active=("conjunction_active", "max"),
        active_rows=("conjunction_active", "sum"),
        max_uncertainty=("short_default_uncertainty_score", "max"),
        mean_reliability=("neighbor_reliability", "mean"),
        parent_ev=("ev_after_1pct", "mean"),
        parent_selected=("parent_rank_v9_residual_error_overlay", lambda x: int(np.sum(x >= 0.90))),
        challenger_selected=("frozen_short_default_uncertainty_rank", lambda x: int(np.sum(x >= 0.90))),
    ).reset_index()


def _label_blocks(daily: pd.DataFrame, cooling_days: int) -> pd.DataFrame:
    """Assign blocks without allowing a one-day dip to create false evidence."""

    full_days = pd.date_range(daily["day"].min(), daily["day"].max(), freq="D", tz="UTC")
    frame = daily.set_index("day").reindex(full_days).rename_axis("day").reset_index()
    frame["conjunction_active"] = frame["conjunction_active"].fillna(False).astype(bool)
    frame["active_rows"] = frame["active_rows"].fillna(0).astype(np.int32)
    block_id = np.full(len(frame), "normal", dtype=object)
    current: str | None = None
    cooling = 0
    sequence = 0
    for idx, active in enumerate(frame["conjunction_active"].to_numpy(bool)):
        if active:
            if current is None:
                sequence += 1
                current = f"forward_block_{sequence:03d}"
            cooling = 0
            block_id[idx] = current
            continue
        if current is not None:
            cooling += 1
            if cooling < cooling_days:
                block_id[idx] = current
            else:
                current = None
                cooling = 0
    frame["block_id"] = block_id
    return frame


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    reliability_floor = (
        float(args.reliability_floor)
        if args.reliability_floor is not None
        else float(args.minimum_effective_neighbor_support)
        / (float(args.minimum_effective_neighbor_support) + 20.0)
    )
    challenger = pd.read_parquet(args.challenger_predictions)
    train_diagnostics = pd.read_parquet(args.diagnostics / "state_distinguishability_predictions.parquet")
    oos_diagnostics = train_diagnostics
    daily = _daily_conjunction(
        challenger,
        train_diagnostics,
        oos_diagnostics,
        threshold=args.threshold,
        reliability_floor=reliability_floor,
    )
    labeled = _label_blocks(daily, args.cooling_days)
    active = labeled.loc[labeled["block_id"].ne("normal")]
    blocks = active.groupby("block_id", observed=True).agg(
        start=("day", "min"),
        end=("day", "max"),
        calendar_days=("day", "size"),
        active_days=("conjunction_active", "sum"),
        active_rows=("active_rows", "sum"),
        max_uncertainty=("max_uncertainty", "max"),
        mean_reliability=("mean_reliability", "mean"),
    ).reset_index()
    blocks.to_csv(args.output / "prospective_uncertainty_blocks.csv", index=False)
    labeled.to_csv(args.output / "prospective_uncertainty_daily_monitor.csv", index=False)
    manifest = {
        "schema": "short_default_uncertainty_forward_block_monitor_v1",
        "candidate_status": "frozen_research_challenger_not_live",
        "scope": {"side": GROUP[0], "archetype": GROUP[1]},
        "threshold": float(args.threshold),
        "minimum_effective_neighbor_support": float(args.minimum_effective_neighbor_support),
        "reliability_floor": reliability_floor,
        "cooling_days": int(args.cooling_days),
        "block_count": int(len(blocks)),
        "confirmation_contract": {
            "minimum_new_independent_blocks": 3,
            "minimum_improving_blocks": 2,
            "cumulative_ev_delta": "> 0 after outcomes resolve",
            "max_single_block_uplift_share": "<= 0.60",
            "activity_retained": ">= 0.90",
        },
        "leakage_contract": (
            "Block membership uses only frozen challenger scores and train-derived diagnostic percentiles. "
            "Realized outcomes are not used to start, end, or merge blocks."
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenger-predictions", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument(
        "--minimum-effective-neighbor-support",
        type=float,
        default=1.0,
        help="Frozen challenger support floor used to qualify a forward conjunction.",
    )
    parser.add_argument(
        "--reliability-floor",
        type=float,
        help="Explicit override for research only; otherwise derived from minimum effective support.",
    )
    parser.add_argument("--cooling-days", type=int, default=2)
    args = parser.parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be in (0, 1)")
    if args.minimum_effective_neighbor_support < 0.0:
        raise ValueError("--minimum-effective-neighbor-support must be >= 0")
    if args.reliability_floor is not None and not 0.0 <= args.reliability_floor <= 1.0:
        raise ValueError("--reliability-floor must be in [0, 1]")
    if args.cooling_days < 1:
        raise ValueError("--cooling-days must be >= 1")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
