#!/usr/bin/env python3
"""Compare the legacy rank-multiplier admission with EV-unit admission OOS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.inference.threshold_basis_policy import (
    apply_threshold_basis_policy_to_decisions,
    load_threshold_basis_policy,
)


def _load_rows(path: Path, start: pd.Timestamp) -> pd.DataFrame:
    columns = [
        "__ts__",
        "__symbol__",
        "side_name",
        "archetype_policy_key",
        "policy_parent_rank",
        "rank_mlp_direct",
        "expected_ev_rank_score",
        "expected_net_ev_after_1pct_mlp_direct",
        "ev_after_1pct",
        "clean_exec",
        "dirty_positive",
        "full_path_bad_mae_1r",
        "timeout",
    ]
    rows = pd.read_parquet(path, columns=columns)
    rows["__ts__"] = pd.to_datetime(rows["__ts__"], utc=True, errors="coerce")
    rows = rows.loc[rows["__ts__"].ge(start)].copy()
    rows["side_name"] = rows["side_name"].astype(str).str.lower()
    rows["policy_archetype"] = rows["archetype_policy_key"].fillna("missing").astype(str)
    for side in ("long", "short"):
        prefix = f"{side}__"
        mask = rows["side_name"].eq(side) & rows["policy_archetype"].str.startswith(
            prefix, na=False
        )
        rows.loc[mask, "policy_archetype"] = rows.loc[
            mask, "policy_archetype"
        ].str[len(prefix) :]
    return rows.sort_values(["__ts__", "__symbol__", "side_name"], kind="stable")


def _apply(rows: pd.DataFrame, policy_path: Path, arm: str) -> pd.DataFrame:
    policy = load_threshold_basis_policy(policy_path)
    parts: list[pd.DataFrame] = []
    month_key = rows["__ts__"].dt.to_period("M")
    for _, group in rows.groupby(month_key, sort=True, observed=True):
        symbols = group["__symbol__"].astype(str).to_numpy(copy=False)
        sides = group["side_name"].astype(str).to_numpy(copy=False)
        archetypes = group["policy_archetype"].astype(str).to_numpy(copy=False)
        timestamps = group["__ts__"].to_numpy(copy=False)
        parent_ranks = pd.to_numeric(
            group["policy_parent_rank"], errors="coerce"
        ).to_numpy(dtype=np.float64, copy=False)
        ev_ranks = pd.to_numeric(
            group["expected_ev_rank_score"], errors="coerce"
        ).to_numpy(dtype=np.float64, copy=False)
        mapped_evs = pd.to_numeric(
            group["expected_net_ev_after_1pct_mlp_direct"], errors="coerce"
        ).to_numpy(dtype=np.float64, copy=False)
        decisions = [
            {
                "timestamp": timestamps[idx],
                "symbol": symbols[idx],
                "side_name": sides[idx],
                "policy_archetype": archetypes[idx],
                "policy_rank_pct": parent_ranks[idx],
                "v9_tail95_predecessor_rank": parent_ranks[idx],
                "expected_ev_rank_score": ev_ranks[idx],
                "expected_net_ev_after_1pct_side_archetype": mapped_evs[idx],
            }
            for idx in range(len(group))
        ]
        apply_threshold_basis_policy_to_decisions(decisions, policy=policy)
        scored = group.copy()
        scored["selected"] = [
            bool(item.get("threshold_basis_selected", False)) for item in decisions
        ]
        scored["admission_rank"] = [
            float(item.get("threshold_basis_rank_score", 0.0)) for item in decisions
        ]
        scored["mapped_expected_ev"] = [
            item.get("threshold_basis_mapped_expected_ev_side_archetype", np.nan)
            for item in decisions
        ]
        scored["recent_ev_correction"] = [
            item.get("threshold_basis_side_archetype_recent_ev_correction", 0.0)
            for item in decisions
        ]
        scored["corrected_expected_ev"] = [
            item.get("threshold_basis_corrected_expected_ev", np.nan)
            for item in decisions
        ]
        scored["correction_scope"] = [
            item.get("threshold_basis_expected_ev_correction_scope", "")
            for item in decisions
        ]
        scored["arm"] = arm
        parts.append(scored)
    return pd.concat(parts, ignore_index=True, copy=False)


def _metrics(rows: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    selected = rows.loc[rows["selected"]].copy()
    if selected.empty:
        return pd.DataFrame()
    selected["day"] = selected["__ts__"].dt.floor("D")
    selected["month"] = selected["__ts__"].dt.strftime("%Y-%m")
    selected["week"] = selected["__ts__"].dt.to_period("W-SUN").astype(str)
    group_arg: str | list[str] = groups[0] if len(groups) == 1 else groups
    report = selected.groupby(group_arg, dropna=False, observed=True).agg(
        selected_rows=("ev_after_1pct", "size"),
        days=("day", "nunique"),
        mean_net_ev=("ev_after_1pct", "mean"),
        sum_net_ev=("ev_after_1pct", "sum"),
        positive_ev_rate=("ev_after_1pct", lambda value: float((value > 0).mean())),
        clean_exec_rate=("clean_exec", "mean"),
        dirty_positive_rate=("dirty_positive", "mean"),
        bad_mae_rate=("full_path_bad_mae_1r", "mean"),
        timeout_rate=("timeout", "mean"),
        mapped_expected_ev_mean=("mapped_expected_ev", "mean"),
        recent_ev_correction_mean=("recent_ev_correction", "mean"),
        corrected_expected_ev_mean=("corrected_expected_ev", "mean"),
    ).reset_index()
    report["trades_per_day"] = report["selected_rows"] / report["days"].clip(lower=1)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oos-predictions", type=Path, required=True)
    parser.add_argument("--baseline-policy", type=Path)
    parser.add_argument("--candidate-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", default="2026-04-01T00:00:00Z")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    rows = _load_rows(args.oos_predictions, start)
    arms: list[pd.DataFrame] = []
    if args.baseline_policy is not None:
        arms.append(_apply(rows, args.baseline_policy, "legacy_rank_multiplier"))
    arms.append(_apply(rows, args.candidate_policy, "side_archetype_ev_unit"))
    scored = pd.concat(arms, ignore_index=True, copy=False)
    scored.to_parquet(args.output_dir / "oos_policy_rows.parquet", index=False)
    reports: dict[str, str] = {}
    for name, groups in {
        "overall": ["arm"],
        "month": ["arm", "month"],
        "week": ["arm", "week"],
        "side": ["arm", "side_name"],
        "archetype": ["arm", "policy_archetype"],
        "month_side_archetype": ["arm", "month", "side_name", "policy_archetype"],
    }.items():
        report = _metrics(scored, groups)
        path = args.output_dir / f"metrics_{name}.csv"
        report.to_csv(path, index=False)
        reports[name] = str(path)
    manifest = {
        "schema": "side_archetype_expected_ev_policy_oos_comparison_v1",
        "oos_source": str(args.oos_predictions),
        "baseline_policy": (
            str(args.baseline_policy) if args.baseline_policy is not None else None
        ),
        "candidate_policy": str(args.candidate_policy),
        "rows": int(len(rows)),
        "start": start.isoformat(),
        "cost_contract": "ev_after_1pct contains the sole 1% round-trip cost",
        "reports": reports,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
