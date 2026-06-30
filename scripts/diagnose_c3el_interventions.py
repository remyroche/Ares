#!/usr/bin/env python3
"""Diagnose head-native C3el intervention quality.

The promotion report says whether a candidate passes aggregate replay gates.
This script explains *why* by resolving each selected head/timestamp/strategy
cut back to accepted-trade outcomes and, when supplied, exact-state action
labels.  The output is intentionally intervention-level because C3el should be
a sparse default-no-op policy, not a continuous overlay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYS = ["timestamp", "strategy_id"]


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _normalise_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["strategy_id"] = out["strategy_id"].astype(str)
    return out.loc[out["timestamp"].notna()].copy()


def _safe_num(series: pd.Series | Any, default: float = 0.0) -> pd.Series:
    if not isinstance(series, pd.Series):
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _load_exact_support(paths: list[Path]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        frame = _normalise_keys(_read_frame(path))
        if "multiplier" not in frame.columns:
            if "action_value" not in frame.columns:
                continue
            frame["multiplier"] = frame["action_value"]
        frame["multiplier"] = _safe_num(frame["multiplier"]).round(6)
        for col in ["delta_full_J", "delta_immediate_J", "action_binds"]:
            frame[col] = _safe_num(frame[col]) if col in frame.columns else 0.0
        parts.append(frame)
    if not parts:
        return pd.DataFrame()
    exact = pd.concat(parts, ignore_index=True, sort=False)
    exact = exact.loc[exact["multiplier"].lt(1.0) & exact["action_binds"].gt(0.0)].copy()
    if exact.empty:
        return pd.DataFrame()
    agg = exact.groupby(KEYS, as_index=False).agg(
        exact_nonbase_rows=("multiplier", "size"),
        exact_positive_e50_rows=("delta_full_J", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0.0).gt(50.0).sum())),
        exact_best_delta_full_J=("delta_full_J", "max"),
        exact_worst_delta_full_J=("delta_full_J", "min"),
        exact_best_delta_immediate_J=("delta_immediate_J", "max"),
    )
    return agg


def _trade_outcomes(accepted: pd.DataFrame) -> pd.DataFrame:
    accepted = _normalise_keys(accepted)
    if "arm" not in accepted.columns:
        raise ValueError("accepted_trades.csv must contain an arm column")
    for col in ["net_pnl", "gross_pnl", "cost_pnl", "position_size", "net_win", "full_sl", "timeout"]:
        if col in accepted.columns:
            accepted[col] = _safe_num(accepted[col])
        else:
            accepted[col] = 0.0
    grouped = accepted.groupby(["arm", *KEYS], as_index=False).agg(
        trade_count=("symbol", "size"),
        net_pnl=("net_pnl", "sum"),
        gross_pnl=("gross_pnl", "sum"),
        cost_pnl=("cost_pnl", "sum"),
        position_size=("position_size", "sum"),
        net_wins=("net_win", "sum"),
        full_sl=("full_sl", "sum"),
        timeout=("timeout", "sum"),
    )
    wide_parts: list[pd.DataFrame] = []
    for arm, prefix in [("C0_baseline", "baseline"), ("C3el_head_native", "candidate")]:
        part = grouped.loc[grouped["arm"].eq(arm)].drop(columns=["arm"]).copy()
        rename = {col: f"{prefix}_{col}" for col in part.columns if col not in KEYS}
        wide_parts.append(part.rename(columns=rename))
    if not wide_parts:
        return pd.DataFrame(columns=KEYS)
    out = wide_parts[0]
    for part in wide_parts[1:]:
        out = out.merge(part, on=KEYS, how="outer")
    for col in out.columns:
        if col not in KEYS:
            out[col] = _safe_num(out[col])
    return out


def _summarise_interventions(interventions: pd.DataFrame) -> pd.DataFrame:
    if interventions.empty:
        return pd.DataFrame()
    work = interventions.copy()
    work["intervention_count"] = 1
    work["positive_direct_delta"] = work["direct_delta_net_pnl"].gt(0.0).astype(int)
    work["negative_direct_delta"] = work["direct_delta_net_pnl"].lt(0.0).astype(int)
    grouped = work.groupby(["head"], as_index=False).agg(
        intervention_count=("intervention_count", "sum"),
        positive_direct_delta_count=("positive_direct_delta", "sum"),
        negative_direct_delta_count=("negative_direct_delta", "sum"),
        direct_delta_net_pnl=("direct_delta_net_pnl", "sum"),
        direct_delta_gross_pnl=("direct_delta_gross_pnl", "sum"),
        direct_delta_cost_pnl=("direct_delta_cost_pnl", "sum"),
        loss_avoided=("loss_avoided", "sum"),
        winner_pnl_sacrificed=("winner_pnl_sacrificed", "sum"),
        defensive_success=("defensive_success", "sum"),
        baseline_net_pnl=("baseline_net_pnl", "sum"),
        candidate_net_pnl=("candidate_net_pnl", "sum"),
        baseline_trade_count=("baseline_trade_count", "sum"),
        candidate_trade_count=("candidate_trade_count", "sum"),
        baseline_full_sl=("baseline_full_sl", "sum"),
        candidate_full_sl=("candidate_full_sl", "sum"),
        exact_supported_interventions=("exact_supported", "sum"),
        exact_positive_e50_rows=("exact_positive_e50_rows", "sum"),
        exact_delta_full_J_sum=("exact_best_delta_full_J", "sum"),
        exact_delta_immediate_J_sum=("exact_best_delta_immediate_J", "sum"),
        exact_worst_delta_full_J=("exact_worst_delta_full_J", "min"),
    )
    grouped["positive_direct_delta_rate"] = grouped["positive_direct_delta_count"] / grouped["intervention_count"].replace(0, np.nan)
    grouped["exact_positive_e50_rate"] = grouped["exact_positive_e50_rows"] / grouped["intervention_count"].replace(0, np.nan)
    grouped["delta_full_sl"] = grouped["candidate_full_sl"] - grouped["baseline_full_sl"]
    return grouped.fillna(0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--exact-action-panels", type=Path, nargs="*", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir
    schedule = _normalise_keys(pd.read_csv(run_dir / "head_native_size_schedule.csv"))
    scores = _normalise_keys(pd.read_csv(run_dir / "head_native_group_scores.csv"))
    accepted = pd.read_csv(run_dir / "accepted_trades.csv")
    exact_support = _load_exact_support(list(args.exact_action_panels))

    schedule["multiplier"] = _safe_num(schedule["multiplier"], default=1.0).clip(0.0, 1.0)
    interventions = schedule.loc[schedule["multiplier"].lt(1.0)].copy()
    if "head" not in interventions.columns:
        interventions["head"] = interventions["strategy_id"].str.extract(r"^(long_bars|long_dist|short_asset|short_boll)", expand=False)

    score_cols = [
        col
        for col in [
            "head",
            "p_intervene",
            "pred_action_delta_J",
            "selected_multiplier",
            "gate_keep",
            "guard_low_breadth_share",
            "guard_action_feature_min",
            "week_start",
        ]
        if col in scores.columns
    ]
    if score_cols:
        interventions = interventions.merge(scores[KEYS + score_cols], on=KEYS, how="left", suffixes=("", "_score"))
        if "head_score" in interventions.columns:
            interventions["head"] = interventions["head"].fillna(interventions["head_score"])
            interventions = interventions.drop(columns=["head_score"])

    outcomes = _trade_outcomes(accepted)
    interventions = interventions.merge(outcomes, on=KEYS, how="left")
    if not exact_support.empty:
        interventions = interventions.merge(exact_support, on=KEYS, how="left")

    for col in interventions.columns:
        if col not in {"timestamp", "strategy_id", "head", "week_start"}:
            if interventions[col].dtype == object:
                continue
            interventions[col] = _safe_num(interventions[col])

    for prefix in ["baseline", "candidate"]:
        for col in ["trade_count", "net_pnl", "gross_pnl", "cost_pnl", "position_size", "net_wins", "full_sl", "timeout"]:
            name = f"{prefix}_{col}"
            if name not in interventions.columns:
                interventions[name] = 0.0
            interventions[name] = _safe_num(interventions[name])
    for col in ["exact_nonbase_rows", "exact_positive_e50_rows", "exact_best_delta_full_J", "exact_worst_delta_full_J"]:
        if col not in interventions.columns:
            interventions[col] = 0.0
        interventions[col] = _safe_num(interventions[col])
    if "exact_best_delta_immediate_J" not in interventions.columns:
        interventions["exact_best_delta_immediate_J"] = 0.0
    interventions["exact_best_delta_immediate_J"] = _safe_num(interventions["exact_best_delta_immediate_J"])
    interventions["exact_supported"] = interventions["exact_nonbase_rows"].gt(0.0).astype(int)

    interventions["direct_delta_net_pnl"] = interventions["candidate_net_pnl"] - interventions["baseline_net_pnl"]
    interventions["direct_delta_gross_pnl"] = interventions["candidate_gross_pnl"] - interventions["baseline_gross_pnl"]
    interventions["direct_delta_cost_pnl"] = interventions["candidate_cost_pnl"] - interventions["baseline_cost_pnl"]
    interventions["loss_avoided"] = (-interventions["baseline_net_pnl"]).clip(lower=0.0) - (
        -interventions["candidate_net_pnl"]
    ).clip(lower=0.0)
    interventions["loss_avoided"] = interventions["loss_avoided"].clip(lower=0.0)
    interventions["winner_pnl_sacrificed"] = interventions["baseline_net_pnl"].clip(lower=0.0) - interventions[
        "candidate_net_pnl"
    ].clip(lower=0.0)
    interventions["winner_pnl_sacrificed"] = interventions["winner_pnl_sacrificed"].clip(lower=0.0)
    interventions["defensive_success"] = interventions["loss_avoided"] - interventions["winner_pnl_sacrificed"]
    interventions["delta_full_sl"] = interventions["candidate_full_sl"] - interventions["baseline_full_sl"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    interventions.to_csv(args.out_dir / "intervention_diagnostics.csv", index=False)
    by_head = _summarise_interventions(interventions)
    by_head.to_csv(args.out_dir / "intervention_summary_by_head.csv", index=False)
    by_week_head = (
        interventions.assign(week_start=pd.to_datetime(interventions.get("week_start"), utc=True, errors="coerce"))
        .groupby(["week_start", "head"], as_index=False)
        .agg(
            intervention_count=("strategy_id", "size"),
            direct_delta_net_pnl=("direct_delta_net_pnl", "sum"),
            loss_avoided=("loss_avoided", "sum"),
            winner_pnl_sacrificed=("winner_pnl_sacrificed", "sum"),
            defensive_success=("defensive_success", "sum"),
            baseline_full_sl=("baseline_full_sl", "sum"),
            candidate_full_sl=("candidate_full_sl", "sum"),
            exact_supported_interventions=("exact_supported", "sum"),
            exact_positive_e50_rows=("exact_positive_e50_rows", "sum"),
            exact_delta_full_J_sum=("exact_best_delta_full_J", "sum"),
        )
    )
    by_week_head["delta_full_sl"] = by_week_head["candidate_full_sl"] - by_week_head["baseline_full_sl"]
    by_week_head.to_csv(args.out_dir / "intervention_summary_by_week_head.csv", index=False)

    manifest = {
        "generated_by": "diagnose_c3el_interventions",
        "run_dir": str(run_dir),
        "exact_action_panels": [str(p) for p in args.exact_action_panels],
        "interventions": int(len(interventions)),
        "heads": sorted(str(x) for x in interventions["head"].dropna().unique()),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    lines = ["# C3el intervention diagnostics", "", f"Run: `{run_dir}`", "", "## By Head", ""]
    lines.append(by_head.to_markdown(index=False) if not by_head.empty else "No interventions.")
    lines.extend(["", "## By Week And Head", ""])
    lines.append(by_week_head.to_markdown(index=False) if not by_week_head.empty else "No interventions.")
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print((args.out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
