#!/usr/bin/env python3
"""Attribute a frozen short-default challenger by component and independent block.

The analysis is retrospective only.  It uses realized adverse-calendar flags to
define evaluation blocks, never to alter scores.  Counterfactual component rows
answer attribution questions and are not candidate policy variants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from extreme_price_movements.challenger_credibility import (
    PosteriorConfig,
    consecutive_event_blocks,
    daily_decision_deltas,
    hierarchical_student_t_posterior,
)
from scripts.run_short_default_uncertainty_ablation import _adjust_rank, _percentile


GROUP = ("short", "short_default_clean_path")
KEYS = ["__ts__", "side_name", "archetype_policy_key"]
COMPONENTS = (
    "ensemble_risk_std",
    "neighbor_shrunken_adverse_rate",
    "neighbor_weighted_ev_std",
)


def _metrics(frame: pd.DataFrame, rank: np.ndarray) -> dict[str, float]:
    selected = rank >= 0.90
    ev = pd.to_numeric(frame["ev_after_1pct"], errors="coerce").to_numpy(np.float64)
    clean = pd.to_numeric(frame["clean_exec"], errors="coerce").to_numpy(np.float64)
    return {
        "selected_rows": int(selected.sum()),
        "sum_ev": float(np.nansum(ev[selected])),
        "mean_ev": float(np.nanmean(ev[selected])) if selected.any() else np.nan,
        "clean_precision": float(np.nanmean(clean[selected])) if selected.any() else np.nan,
    }


def _load(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics = pd.read_parquet(args.diagnostics / "state_distinguishability_predictions.parquet")
    parent = pd.read_parquet(args.v11_dir / "oos_predictions.parquet")
    challenger = pd.read_parquet(args.challenger_dir / "oos_replication_predictions.parquet")
    for frame in (diagnostics, parent, challenger):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    mask = diagnostics["stage"].eq("eval_oos") & diagnostics["side_name"].eq(GROUP[0]) & diagnostics["archetype_policy_key"].eq(GROUP[1])
    diagnostics = diagnostics.loc[mask, KEYS + list(COMPONENTS) + ["neighbor_effective_count", "neighbor_reliability", "nearest_neighbor_distance"]].drop_duplicates(KEYS, keep="last")
    base = parent.loc[
        parent["side_name"].eq(GROUP[0]) & parent["archetype_policy_key"].eq(GROUP[1]),
        KEYS + ["__symbol__", "parent_rank_v9_residual_error_overlay", "ev_after_1pct", "clean_exec", "adverse_calendar_cell"],
    ]
    frozen = challenger.loc[
        challenger["side_name"].eq(GROUP[0]) & challenger["archetype_policy_key"].eq(GROUP[1]),
        KEYS + ["__symbol__", "short_default_uncertainty_score", "frozen_short_default_uncertainty_rank"],
    ]
    rows = base.merge(frozen, on=[*KEYS, "__symbol__"], how="inner", validate="one_to_one")
    return rows.merge(diagnostics, on=KEYS, how="inner", validate="many_to_one"), diagnostics


def _daily_uncertainty(rows: pd.DataFrame) -> pd.DataFrame:
    local = rows.copy()
    local["day"] = local["__ts__"].dt.floor("D")
    return local.groupby("day", observed=True).agg(
        high_uncertainty=("short_default_uncertainty_score", lambda x: bool(np.nanmax(x) >= 0.85)),
        adverse=("adverse_calendar_cell", "max"),
        parent_ev=("ev_after_1pct", "mean"),
    ).reset_index()


def _block_table(rows: pd.DataFrame) -> pd.DataFrame:
    daily = _daily_uncertainty(rows)
    # A block begins with an adverse or high-uncertainty day and ends after one
    # cooling day. This makes contemporaneous rows a single evidence unit.
    active = daily["high_uncertainty"].astype(bool) | daily["adverse"].astype(bool)
    daily["episode_block"] = consecutive_event_blocks(daily["day"], active)
    daily.loc[daily["episode_block"].eq("normal"), "episode_block"] = "normal_days"
    decision = daily_decision_deltas(
        rows,
        parent_rank="parent_rank_v9_residual_error_overlay",
        challenger_rank="frozen_short_default_uncertainty_rank",
    )
    decision = decision.merge(daily[["day", "episode_block", "high_uncertainty", "adverse"]], on="day", how="left", validate="one_to_one")
    blocks = decision.loc[decision["episode_block"].ne("normal_days")].groupby("episode_block", observed=True).agg(
        start=("day", "min"), end=("day", "max"), days=("day", "size"),
        adverse_days=("adverse", "sum"), high_uncertainty_days=("high_uncertainty", "sum"),
        delta_total_ev=("delta_total_ev", "sum"),
        parent_total_ev=("parent_total_ev", "sum"), challenger_total_ev=("challenger_total_ev", "sum"),
        parent_selected=("parent_selected", "sum"), challenger_selected=("challenger_selected", "sum"),
        parent_clean_sum=("parent_clean_sum", "sum"), challenger_clean_sum=("challenger_clean_sum", "sum"),
    ).reset_index()
    blocks["delta_ev_per_trade"] = blocks["challenger_total_ev"] / blocks["challenger_selected"].clip(lower=1) - blocks["parent_total_ev"] / blocks["parent_selected"].clip(lower=1)
    blocks["delta_clean_precision"] = blocks["challenger_clean_sum"] / blocks["challenger_selected"].clip(lower=1) - blocks["parent_clean_sum"] / blocks["parent_selected"].clip(lower=1)
    blocks["activity_ratio"] = blocks["challenger_selected"] / blocks["parent_selected"].clip(lower=1)
    blocks["month"] = blocks["start"].dt.strftime("%Y-%m")
    blocks["event_family"] = np.where(blocks["adverse_days"].gt(0), "adverse", "benign_high_uncertainty")
    return blocks, decision


def _component_attribution(
    rows: pd.DataFrame, *, v11_dir: Path, diagnostics_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(v11_dir / "train_oof_predictions.parquet")
    train["__ts__"] = pd.to_datetime(train["__ts__"], utc=True)
    # Train diagnostics are the only valid empirical reference distribution.
    train_diag = pd.read_parquet(diagnostics_dir / "state_distinguishability_predictions.parquet")
    train_diag["__ts__"] = pd.to_datetime(train_diag["__ts__"], utc=True)
    mask = train_diag["stage"].eq("train_oof") & train_diag["side_name"].eq(GROUP[0]) & train_diag["archetype_policy_key"].eq(GROUP[1])
    train_diag = train_diag.loc[mask, KEYS + list(COMPONENTS)].drop_duplicates(KEYS, keep="last")
    train = train.loc[train["side_name"].eq(GROUP[0]) & train["archetype_policy_key"].eq(GROUP[1])].merge(train_diag, on=KEYS, how="inner", validate="many_to_one")
    score = rows.copy()
    for component in COMPONENTS:
        reference = pd.to_numeric(train[component], errors="coerce").to_numpy(np.float32)
        train[f"uncertainty__{component}"] = _percentile(reference, reference)
        values = pd.to_numeric(score[component], errors="coerce").to_numpy(np.float32)
        score[f"uncertainty__{component}"] = _percentile(values, reference)
    output: list[dict[str, object]] = []
    percentile_cols = [f"uncertainty__{name}" for name in COMPONENTS]
    source = {name: score[f"uncertainty__{name}"].to_numpy(np.float32) for name in COMPONENTS}
    parent = score["parent_rank_v9_residual_error_overlay"].to_numpy(np.float32)
    for name, values in source.items():
        rank = _adjust_rank(parent, values, 0.85, 0.04)
        metrics = _metrics(score, rank)
        output.append({"counterfactual": f"component_only__{name}", **metrics, "penalized_rows": int((values >= 0.85).sum())})
    full = score["frozen_short_default_uncertainty_rank"].to_numpy(np.float32)
    output.append({"counterfactual": "frozen_full_challenger", **_metrics(score, full), "penalized_rows": int((full < parent).sum())})
    overlap = score.loc[:, percentile_cols].ge(0.85).astype(np.int8)
    overlap["penalty_component_count"] = overlap.sum(axis=1)
    overlap["__ts__"] = score["__ts__"].to_numpy()
    overlap["adverse_calendar_cell"] = score["adverse_calendar_cell"].to_numpy()
    return pd.DataFrame(output), overlap, score


def _group_comparison(
    score: pd.DataFrame, daily: pd.DataFrame
) -> pd.DataFrame:
    labels = daily.loc[:, ["day", "adverse", "high_uncertainty", "parent_total_ev"]].copy()
    labels["group"] = np.select(
        [
            labels["day"].between(pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-04-08", tz="UTC")),
            labels["adverse"].astype(bool),
            labels["high_uncertainty"].astype(bool) & labels["parent_total_ev"].gt(0.0),
        ],
        ["A_april_01_08", "B_other_adverse", "C_benign_high_uncertainty"],
        default="other",
    )
    frame = score.copy()
    frame["day"] = frame["__ts__"].dt.floor("D")
    frame = frame.merge(labels[["day", "group"]], on="day", how="left", validate="many_to_one")
    rows: list[dict[str, object]] = []
    columns = [f"uncertainty__{name}" for name in COMPONENTS]
    for group, local in frame.groupby("group", observed=True):
        for column in columns:
            value = pd.to_numeric(local[column], errors="coerce")
            rows.append(
                {
                    "group": group,
                    "component": column.removeprefix("uncertainty__"),
                    "rows": int(len(local)),
                    "mean_percentile": float(value.mean()),
                    "median_percentile": float(value.median()),
                    "p90_percentile": float(value.quantile(0.90)),
                    "penalized_row_rate": float(value.ge(0.85).mean()),
                    "mean_uncertainty": float(local["short_default_uncertainty_score"].mean()),
                    "effective_neighbor_count": float(local["neighbor_effective_count"].mean()),
                    "neighbor_reliability": float(local["neighbor_reliability"].mean()),
                    "nearest_neighbor_distance": float(local["nearest_neighbor_distance"].mean()),
                }
            )
    return pd.DataFrame(rows), frame


def _neighbor_dates(
    diagnostics_dir: Path, grouped_rows: pd.DataFrame
) -> pd.DataFrame:
    neighbors = pd.read_parquet(diagnostics_dir / "matched_adverse_event_neighbors.parquet")
    neighbors["query_timestamp"] = pd.to_datetime(neighbors["query_timestamp"], utc=True)
    query = grouped_rows.loc[:, ["__ts__", "group"]].drop_duplicates().rename(columns={"__ts__": "query_timestamp"})
    local = neighbors.loc[
        neighbors["side_name"].eq(GROUP[0]) & neighbors["archetype_policy_key"].eq(GROUP[1]) & neighbors["neighbor_rank"].le(5)
    ].merge(query, on="query_timestamp", how="inner", validate="many_to_one")
    return local.groupby(["group", "neighbor_day"], observed=True).agg(
        matched_queries=("query_timestamp", "nunique"),
        mean_neighbor_distance=("neighbor_distance", "mean"),
        historical_adverse_rate=("neighbor_adverse", "mean"),
        historical_neighbor_ev=("neighbor_ev", "mean"),
        historical_neighbor_clean=("neighbor_clean", "mean"),
    ).reset_index().sort_values(["group", "matched_queries", "mean_neighbor_distance"], ascending=[True, False, True], kind="stable")


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    rows, diagnostics = _load(args)
    blocks, daily = _block_table(rows)
    blocks.to_csv(args.output / "equal_weight_episode_blocks.csv", index=False)
    daily.to_csv(args.output / "episode_daily_deltas.csv", index=False)
    attribution, overlap, scored = _component_attribution(
        rows, v11_dir=args.v11_dir, diagnostics_dir=args.diagnostics
    )
    attribution.to_csv(args.output / "component_counterfactuals.csv", index=False)
    overlap.to_csv(args.output / "component_overlap.csv", index=False)
    comparison, grouped_rows = _group_comparison(scored, daily)
    comparison.to_csv(args.output / "april_other_adverse_benign_component_comparison.csv", index=False)
    _neighbor_dates(args.diagnostics, grouped_rows).to_csv(args.output / "component_group_nearest_neighbor_dates.csv", index=False)
    # Equal-block posterior: each block appears once, independent of its duration.
    posterior = hierarchical_student_t_posterior(
        blocks.assign(month="all_blocks"),
        value_column="delta_ev_per_trade",
        config=PosteriorConfig(draws=args.draws, burn_in=args.burn_in, seed=args.seed),
    )
    scale = float(posterior.attrs["scale"])
    sigma = np.exp(posterior["log_sigma"].to_numpy(np.float64)) * scale
    mu = posterior["mu"].to_numpy(np.float64)
    future_positive = student_t.sf((0.0 - mu) / np.maximum(sigma, 1e-12), df=4.0)
    posterior["future_block_positive_probability"] = future_positive
    posterior.to_parquet(args.output / "equal_block_student_t_posterior.parquet", index=False, compression="zstd")
    april = blocks.loc[blocks["start"].eq(pd.Timestamp("2026-04-01", tz="UTC"))]
    manifest = {
        "schema": "short_default_challenger_episode_attribution_v1",
        "episode_count": int(len(blocks)),
        "april_block_found": bool(len(april)),
        "equal_block_posterior": {
            "p_mu_gt_zero": float((mu > 0.0).mean()),
            "mean_future_block_positive_probability": float(future_positive.mean()),
            "p_future_block_positive_gt_50pct": float((future_positive > 0.5).mean()),
        },
        "component_contract": "D=ensemble disagreement, N=shrunken neighbor adverse rate, V=neighbor EV dispersion; all percentiles use train-OOF references.",
        "leakage_contract": "Retrospective adverse blocks are evaluation units only. No component counterfactual is promoted or applied to inference.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v11-dir", type=Path, required=True)
    parser.add_argument("--challenger-dir", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=8_000)
    parser.add_argument("--burn-in", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
