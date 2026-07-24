#!/usr/bin/env python3
"""Summarize residual distinguishability, matched neighbors, and episode states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


TARGET = "bad_residual_event_target"


def _predictive_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = ["stage", "side_name", "archetype_policy_key"]
    for key, local in frame.groupby(groups, observed=True, sort=True):
        target = local[TARGET].to_numpy(np.int8)
        prevalence = float(target.mean())
        for score_name in ("ensemble_risk_mean", "neighbor_adverse_rate"):
            score = local[score_name].to_numpy(np.float32)
            cutoff = float(np.quantile(score, 0.90))
            selected = score >= cutoff
            precision = float(target[selected].mean()) if selected.any() else np.nan
            rows.append(
                {
                    "stage": key[0],
                    "side_name": key[1],
                    "archetype_policy_key": key[2],
                    "score": score_name,
                    "rows": len(local),
                    "adverse_prevalence": prevalence,
                    "roc_auc": float(roc_auc_score(target, score))
                    if np.unique(target).size > 1
                    else np.nan,
                    "average_precision": float(average_precision_score(target, score))
                    if target.sum()
                    else np.nan,
                    "brier": float(brier_score_loss(target, score)),
                    "top10_precision": precision,
                    "top10_lift": precision / prevalence if prevalence > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _neighbor_reports(neighbors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["query_day", "side_name", "archetype_policy_key"]
    summary = (
        neighbors.groupby(keys, observed=True, sort=True)
        .agg(
            query_timestamps=("query_timestamp", "nunique"),
            neighbor_rows=("neighbor_rank", "size"),
            unique_neighbor_days=("neighbor_day", "nunique"),
            neighbor_adverse_rate=("neighbor_adverse", "mean"),
            neighbor_mean_ev=("neighbor_ev", "mean"),
            neighbor_positive_ev_rate=("neighbor_ev", lambda values: float((values > 0).mean())),
            neighbor_clean_rate=("neighbor_clean", "mean"),
            mean_distance=("neighbor_distance", "mean"),
        )
        .reset_index()
    )
    by_day = (
        neighbors.groupby([*keys, "neighbor_day"], observed=True, sort=False)
        .agg(
            minimum_distance=("neighbor_distance", "min"),
            mean_distance=("neighbor_distance", "mean"),
            matches=("neighbor_rank", "size"),
            adverse_rate=("neighbor_adverse", "mean"),
            mean_ev=("neighbor_ev", "mean"),
            clean_rate=("neighbor_clean", "mean"),
        )
        .reset_index()
        .sort_values([*keys, "minimum_distance"], kind="stable")
    )
    by_day["neighbor_day_rank"] = (
        by_day.groupby(keys, observed=True).cumcount() + 1
    )
    return summary, by_day.loc[by_day["neighbor_day_rank"].le(20)]


def _event_classification(
    diagnostics: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    event = diagnostics.loc[diagnostics["adverse_calendar_cell"].gt(0)].copy()
    grouped = (
        event.groupby(
            ["stage", "day", "side_name", "archetype_policy_key"],
            observed=True,
            sort=True,
        )
        .agg(
            timestamps=(TARGET, "size"),
            mean_ev=("ev_after_1pct", "mean"),
            ensemble_risk_mean=("ensemble_risk_mean", "mean"),
            ensemble_std_pct=("ensemble_risk_std_percentile", "mean"),
            neighbor_adverse_rate=("neighbor_adverse_rate", "mean"),
            neighbor_entropy=("neighbor_outcome_entropy", "mean"),
            neighbor_distance_pct=("neighbor_distance_percentile", "mean"),
        )
        .reset_index()
    )
    episode = episodes.loc[
        :,
        [
            "stage",
            "day",
            "side_name",
            "archetype_policy_key",
            "episode_cluster_adverse_prior",
            "episode_cluster_support",
            "episode_posterior_max",
            "episode_posterior_entropy",
        ],
    ].drop_duplicates(["stage", "day", "side_name", "archetype_policy_key"])
    grouped = grouped.merge(
        episode,
        on=["stage", "day", "side_name", "archetype_policy_key"],
        how="left",
        validate="one_to_one",
    )
    rare = grouped["neighbor_distance_pct"].ge(0.75)
    ambiguous = grouped["neighbor_entropy"].ge(0.75) & grouped[
        "ensemble_std_pct"
    ].ge(0.75)
    benign_lookalike = (
        grouped["neighbor_adverse_rate"].lt(0.10)
        & grouped["ensemble_risk_mean"].lt(0.25)
    )
    partially_learnable = grouped["neighbor_adverse_rate"].ge(0.10) | grouped[
        "ensemble_risk_mean"
    ].ge(0.25)
    grouped["diagnosis"] = np.select(
        [
            rare & benign_lookalike,
            ambiguous,
            benign_lookalike,
            partially_learnable,
        ],
        [
            "rare_support_confident_benign_lookalike",
            "high_uncertainty_ambiguous",
            "historical_benign_lookalike",
            "partially_learnable_adverse_state",
        ],
        default="weakly_separated",
    )
    grouped["interpretation"] = np.select(
        [
            grouped["diagnosis"].eq("historical_benign_lookalike"),
            grouped["diagnosis"].eq("rare_support_confident_benign_lookalike"),
            grouped["diagnosis"].eq("high_uncertainty_ambiguous"),
        ],
        [
            "Observable neighbors usually succeeded; likely hidden outcome driver or label ambiguity.",
            "State is poorly supported and historical neighbors look benign; add OOD handling before new rules.",
            "Observable lookalikes disagree and the ensemble is unstable; uncertainty-aware sizing is plausible.",
        ],
        default="Some adverse separation exists; test a calibrated continuous risk feature.",
    )
    return grouped


def run(args: argparse.Namespace) -> dict[str, object]:
    diagnostics = pd.read_parquet(args.input / "state_distinguishability_predictions.parquet")
    neighbors = pd.read_parquet(args.input / "matched_adverse_event_neighbors.parquet")
    episodes = pd.read_parquet(args.input / "episode_cluster_assignments.parquet")
    metrics = _predictive_metrics(diagnostics)
    neighbor_summary, neighbor_days = _neighbor_reports(neighbors)
    events = _event_classification(diagnostics, episodes)
    metrics.to_csv(args.input / "predictive_metrics.csv", index=False)
    neighbor_summary.to_csv(args.input / "matched_neighbor_summary.csv", index=False)
    neighbor_days.to_csv(args.input / "matched_neighbor_top20_days.csv", index=False)
    events.to_csv(args.input / "event_diagnosis.csv", index=False)
    manifest = {
        "schema": "residual_distinguishability_report_v1",
        "input": str(args.input),
        "predictive_metric_rows": len(metrics),
        "event_diagnosis_rows": len(events),
        "neighbor_summary_rows": len(neighbor_summary),
        "interpretation_contract": (
            "Diagnoses describe OOF/OOS distinguishability, not causal certainty. "
            "No difficulty, neighbor, or episode output is activated in inference."
        ),
    }
    (args.input / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
