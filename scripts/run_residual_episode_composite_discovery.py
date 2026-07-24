#!/usr/bin/env python3
"""Discover reusable observable composites for high-surprise calendar episodes."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from extreme_price_movements.residual_leaf_state_discovery import _leaf_paths
from extreme_price_movements.unsupervised_regime_learning.economic_relevance import (
    materialize_composite_features,
)


def _safe_matrix(frame: pd.DataFrame, features: list[str], medians: np.ndarray) -> np.ndarray:
    values = frame[features].to_numpy(dtype=np.float32, copy=True)
    invalid = ~np.isfinite(values)
    if invalid.any():
        values[invalid] = np.take(np.asarray(medians, dtype=np.float32), np.nonzero(invalid)[1])
    return values


def _daily_metrics(score: pd.Series, event: pd.Series) -> dict[str, float]:
    valid = score.notna() & event.notna()
    x = score.loc[valid].to_numpy(dtype=np.float64)
    y = event.loc[valid].astype(bool).to_numpy()
    if not len(x) or y.sum() == 0 or (~y).sum() == 0 or np.nanstd(x) <= 1e-12:
        return {
            "correlation": np.nan,
            "average_precision": np.nan,
            "roc_auc": np.nan,
            "top10_precision": np.nan,
            "top10_recall": np.nan,
            "top10_false_positive_rate": np.nan,
            "top10_lift": np.nan,
        }
    cutoff = float(np.nanquantile(x, 0.90))
    selected = x >= cutoff
    precision = float(y[selected].mean()) if selected.any() else 0.0
    prevalence = float(y.mean())
    return {
        "correlation": float(np.corrcoef(x, y.astype(float))[0, 1]),
        "average_precision": float(average_precision_score(y, x)),
        "roc_auc": float(roc_auc_score(y, x)),
        "top10_precision": precision,
        "top10_recall": float((selected & y).sum() / max(y.sum(), 1)),
        "top10_false_positive_rate": float((selected & ~y).sum() / max((~y).sum(), 1)),
        "top10_lift": float(precision / max(prevalence, 1e-9)),
        "threshold": cutoff,
    }


def _leaf_activation_frame(leaves: np.ndarray) -> tuple[Any, np.ndarray]:
    if leaves.ndim == 1:
        leaves = leaves.reshape(-1, 1)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    encoded = encoder.fit_transform(leaves)
    coordinates: list[tuple[int, int]] = []
    for tree_index, categories in enumerate(encoder.categories_):
        coordinates.extend((tree_index, int(leaf)) for leaf in categories)
    return encoded, np.asarray(coordinates, dtype=np.int32)


def _daily_sparse_mean(encoded: Any, day_codes: np.ndarray, day_count: int) -> np.ndarray:
    output = np.zeros((day_count, encoded.shape[1]), dtype=np.float32)
    counts = np.bincount(day_codes, minlength=day_count).astype(np.float32)
    for day in range(day_count):
        rows = np.flatnonzero(day_codes == day)
        if len(rows):
            output[day] = np.asarray(encoded[rows].mean(axis=0)).ravel()
    return output / np.maximum(counts[:, None] / np.maximum(counts[:, None], 1.0), 1.0)


def _candidate_leaf_table(
    paths: pd.DataFrame,
    coordinates: np.ndarray,
    daily_activation: np.ndarray,
    event: np.ndarray,
) -> pd.DataFrame:
    event_mean = daily_activation[event].mean(axis=0)
    non_mean = daily_activation[~event].mean(axis=0)
    lift = (event_mean + 1e-4) / (non_mean + 1e-4)
    support = (daily_activation[event] > 0).sum(axis=0)
    rows = pd.DataFrame(
        {
            "tree_index": coordinates[:, 0],
            "leaf_index": coordinates[:, 1],
            "event_activation": event_mean,
            "non_event_activation": non_mean,
            "event_lift": lift,
            "episode_days_activated": support,
        }
    )
    return rows.merge(paths, on=["tree_index", "leaf_index"], how="left", validate="one_to_one")


def _cluster_candidate_leaves(
    candidates: pd.DataFrame,
    max_clusters: int,
    seed: int,
) -> pd.DataFrame:
    features = sorted(
        {
            feature
            for value in candidates["path_features"].fillna("")
            for feature in str(value).split("|")
            if feature
        }
    )
    if not features:
        candidates["leaf_pattern_cluster"] = 0
        return candidates
    feature_index = {feature: i for i, feature in enumerate(features)}
    matrix = np.zeros((len(candidates), len(features)), dtype=np.float32)
    for row_index, value in enumerate(candidates["path_features"].fillna("")):
        for feature in str(value).split("|"):
            if feature in feature_index:
                matrix[row_index, feature_index[feature]] = 1.0
    cluster_count = min(int(max_clusters), max(2, len(candidates) // 12), len(candidates))
    if cluster_count <= 1:
        labels = np.zeros(len(candidates), dtype=np.int16)
    else:
        labels = MiniBatchKMeans(
            n_clusters=cluster_count,
            random_state=int(seed),
            n_init=20,
            batch_size=256,
        ).fit_predict(matrix)
    candidates = candidates.copy()
    candidates["leaf_pattern_cluster"] = labels.astype(np.int16)
    return candidates


def _leaf_cluster_scores(
    encoded: Any,
    coordinates: np.ndarray,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, list[tuple[int, int]]]]:
    coordinate_index = {tuple(value): i for i, value in enumerate(coordinates.tolist())}
    columns: dict[str, np.ndarray] = {}
    definitions: dict[int, list[tuple[int, int]]] = {}
    for cluster, group in candidates.groupby("leaf_pattern_cluster", observed=True):
        selected = [
            coordinate_index[(int(row.tree_index), int(row.leaf_index))]
            for row in group.itertuples()
        ]
        weights = np.log1p(group["event_lift"].clip(lower=1.0).to_numpy(dtype=np.float32))
        weights /= max(float(weights.sum()), 1e-8)
        columns[f"leaf_pattern_{int(cluster)}"] = np.asarray(
            encoded[:, selected] @ weights, dtype=np.float32
        ).ravel()
        definitions[int(cluster)] = [
            (int(row.tree_index), int(row.leaf_index)) for row in group.itertuples()
        ]
    return pd.DataFrame(columns), definitions


def _best_unsupervised_pairs(
    frame: pd.DataFrame,
    daily: pd.DataFrame,
    event: pd.Series,
    feature_candidates: list[str],
    max_features: int,
    max_selected: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    features = [name for name in feature_candidates if name in frame.columns][
        : int(max_features)
    ]
    train = frame.loc[frame["__ts__"].lt(pd.Timestamp("2026-04-01", tz="UTC"))]
    thresholds: dict[str, tuple[float, float]] = {}
    for feature in features:
        values = pd.to_numeric(train[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        low, high = values.quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
        if np.isfinite(low) and np.isfinite(high) and low < high:
            thresholds[feature] = (float(low), float(high))
    definitions: list[dict[str, Any]] = []
    for left, right in combinations(thresholds, 2):
        for left_bin in ("low", "mid", "high"):
            for right_bin in ("low", "mid", "high"):
                low_a, high_a = thresholds[left]
                low_b, high_b = thresholds[right]
                definitions.append(
                    {
                        "name": f"episode_pair__{len(definitions)}",
                        "feature": left,
                        "feature_bin": left_bin,
                        "q_low": low_a,
                        "q_high": high_a,
                        "feature_b": right,
                        "feature_b_bin": right_bin,
                        "q_low_b": low_b,
                        "q_high_b": high_b,
                    }
                )
    materialized = materialize_composite_features(
        frame, definitions, include_intensity=True
    )
    materialized["day"] = frame["__ts__"].dt.floor("D").to_numpy()
    metrics: list[dict[str, Any]] = []
    for definition in definitions:
        name = str(definition["name"])
        intensity = f"{name}__intensity"
        score = materialized.groupby("day", observed=True)[intensity].mean().reindex(daily.index)
        row = _daily_metrics(score, event)
        metrics.append({"name": name, **row, **definition})
    report = pd.DataFrame(metrics).sort_values(
        ["correlation", "top10_lift", "top10_recall"],
        ascending=[False, False, False],
        kind="stable",
    )
    selected_defs = report.loc[
        report["correlation"].gt(0) & report["top10_lift"].gt(1.25)
    ].head(int(max_selected))
    names = selected_defs["name"].tolist()
    output_columns = [column for name in names for column in (name, f"{name}__intensity")]
    return materialized[output_columns], selected_defs.to_dict("records"), report


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    states = pd.read_parquet(args.states)
    states["__ts__"] = pd.to_datetime(states["__ts__"], utc=True)
    states = states.loc[
        states["__ts__"].ge(pd.Timestamp("2025-01-01", tz="UTC"))
        & states["__ts__"].lt(pd.Timestamp(args.end, tz="UTC"))
    ]
    calendar = pd.read_csv(args.calendar)
    calendar["day"] = pd.to_datetime(calendar["day"], utc=True).dt.floor("D")
    calendar = calendar.loc[
        calendar["day"].lt(pd.Timestamp(args.end, tz="UTC"))
        & calendar["adverse_event_rows"].gt(0)
    ]
    composite_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    definition_manifest: list[dict[str, Any]] = []
    pair_reports: list[pd.DataFrame] = []
    for model_path in sorted(args.models.glob("*.joblib")):
        bundle = joblib.load(model_path)
        side = str(bundle["side_name"])
        archetype = str(bundle["archetype_policy_key"])
        local = states.loc[states["side_name"].astype(str).str.lower().eq(side)].copy()
        local["archetype_policy_key"] = archetype
        features = list(map(str, bundle["features"]))
        matrix = _safe_matrix(local, features, np.asarray(bundle["medians"]))
        leaves = bundle["model"].predict(matrix, pred_leaf=True).astype(np.int16)
        encoded, coordinates = _leaf_activation_frame(leaves)
        local["day"] = local["__ts__"].dt.floor("D")
        days = pd.Index(sorted(local["day"].unique()), name="day")
        day_codes = pd.Categorical(local["day"], categories=days).codes
        daily_activation = _daily_sparse_mean(encoded, day_codes, len(days))
        is_side_global = archetype == "__side_global__"
        calendar_mask = calendar["side_name"].eq(side)
        if not is_side_global:
            calendar_mask &= calendar["archetype_policy_key"].eq(archetype)
        event_days = set(calendar.loc[calendar_mask, "day"])
        event = np.asarray([day in event_days for day in days], dtype=bool)
        discovery_mask = np.asarray(
            [day < pd.Timestamp("2026-04-01", tz="UTC") for day in days],
            dtype=bool,
        )
        if event.sum() < 2:
            continue
        paths = _leaf_paths(bundle["model"], features)
        leaves_report = _candidate_leaf_table(
            paths,
            coordinates,
            daily_activation[discovery_mask],
            event[discovery_mask],
        )
        candidates = leaves_report.loc[
            leaves_report["event_lift"].ge(float(args.min_leaf_lift))
            & leaves_report["episode_days_activated"].ge(2)
        ].copy()
        if candidates.empty:
            candidates = leaves_report.nlargest(40, "event_lift").copy()
        candidates = _cluster_candidate_leaves(
            candidates, int(args.max_leaf_clusters), int(args.seed)
        )
        leaf_scores, leaf_definitions = _leaf_cluster_scores(
            encoded, coordinates, candidates
        )
        leaf_scores.index = local.index
        leaf_scores["day"] = local["day"]
        daily = pd.DataFrame(index=days)
        event_series = pd.Series(event, index=days)
        selected_leaf_names: list[str] = []
        for name in [column for column in leaf_scores if column.startswith("leaf_pattern_")]:
            daily_score = leaf_scores.groupby("day", observed=True)[name].mean().reindex(days)
            metrics = _daily_metrics(
                daily_score.loc[discovery_mask], event_series.loc[discovery_mask]
            )
            oos_metrics = _daily_metrics(
                daily_score.loc[~discovery_mask], event_series.loc[~discovery_mask]
            )
            metric_rows.append(
                {
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "composite": name,
                    "source": "lgbm_leaf_pattern",
                    **metrics,
                    **{f"oos_{key}": value for key, value in oos_metrics.items()},
                }
            )
            if (
                metrics.get("correlation", -1) >= float(args.min_composite_correlation)
                and metrics.get("top10_lift", 0) >= float(args.min_composite_lift)
            ):
                selected_leaf_names.append(name)
        selected_leaf_names = sorted(
            selected_leaf_names,
            key=lambda name: next(
                row["correlation"]
                for row in metric_rows[::-1]
                if row["side_name"] == side
                and row["archetype_policy_key"] == archetype
                and row["composite"] == name
            ),
            reverse=True,
        )[: int(args.max_leaf_composites)]
        selected_hourly = leaf_scores[selected_leaf_names].copy()
        selected_daily = (
            selected_hourly.assign(day=local["day"].to_numpy())
            .groupby("day", observed=True)
            .mean()
            .reindex(days)
        )
        covered = np.zeros(len(days), dtype=bool)
        for name in selected_leaf_names:
            cutoff = float(selected_daily.loc[discovery_mask, name].quantile(0.90))
            covered |= selected_daily[name].to_numpy() >= cutoff

        top_features = (
            candidates.assign(
                feature_list=candidates["path_features"].fillna("").str.split("|")
            )
            .explode("feature_list")
            .groupby("feature_list", observed=True)["event_lift"]
            .mean()
            .sort_values(ascending=False)
            .index.astype(str)
            .tolist()
        )
        pair_hourly, pair_defs, pair_report = _best_unsupervised_pairs(
            local,
            daily.loc[discovery_mask],
            event_series.loc[discovery_mask]
            & ~pd.Series(covered[discovery_mask], index=days[discovery_mask]),
            top_features,
            int(args.unsupervised_features),
            int(args.max_unsupervised_composites),
        )
        pair_hourly.index = local.index
        pair_report.insert(0, "archetype_policy_key", archetype)
        pair_report.insert(0, "side_name", side)
        pair_reports.append(pair_report)
        pair_names = [str(definition["name"]) + "__intensity" for definition in pair_defs]
        for definition in pair_defs:
            name = str(definition["name"])
            daily_score = pair_hourly.assign(day=local["day"].to_numpy()).groupby(
                "day", observed=True
            )[f"{name}__intensity"].mean().reindex(days)
            metrics = _daily_metrics(
                daily_score.loc[discovery_mask], event_series.loc[discovery_mask]
            )
            oos_metrics = _daily_metrics(
                daily_score.loc[~discovery_mask], event_series.loc[~discovery_mask]
            )
            metric_rows.append(
                {
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "composite": name,
                    "source": "unsupervised_pair_intensity",
                    **metrics,
                    **{f"oos_{key}": value for key, value in oos_metrics.items()},
                }
            )
        output = local[["__ts__", "side_name", "archetype_policy_key"]].copy()
        local_feature_names: list[str] = []
        for index, name in enumerate(selected_leaf_names):
            output_name = f"residual_episode_leaf_composite_{index}"
            output[output_name] = selected_hourly[name].to_numpy(dtype=np.float32)
            local_feature_names.append(output_name)
        for index, name in enumerate(pair_names):
            output_name = f"residual_episode_unsup_composite_{index}"
            output[output_name] = pair_hourly[name].to_numpy(dtype=np.float32)
            local_feature_names.append(output_name)
        if local_feature_names:
            output["residual_episode_composite_max"] = output[local_feature_names].max(axis=1)
            output["residual_episode_composite_mean"] = output[local_feature_names].mean(axis=1)
        composite_parts.append(output)

        daily_outputs = output.assign(day=local["day"].to_numpy()).groupby(
            "day", observed=True
        )[local_feature_names].mean().reindex(days)
        thresholds = {
            name: float(daily_outputs.loc[discovery_mask, name].quantile(0.90))
            for name in local_feature_names
        }
        for day in sorted(event_days):
            if day not in daily_outputs.index:
                continue
            matches = [
                name
                for name in local_feature_names
                if float(daily_outputs.loc[day, name]) >= thresholds[name]
            ]
            coverage_rows.append(
                {
                    "day": day,
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "recognized": bool(matches),
                    "status": "recognized" if matches else "ignored",
                    "matching_composites": "|".join(matches),
                    "best_composite_score": float(
                        max(
                            (
                                daily_outputs.loc[day, name]
                                / max(thresholds[name], 1e-8)
                                for name in local_feature_names
                            ),
                            default=0.0,
                        )
                    ),
                    "evidence_scope": (
                        "final_oos"
                        if day >= pd.Timestamp("2026-04-01", tz="UTC")
                        else "full_period_discovery"
                    ),
                    "recognition_scope": (
                        "side_global" if is_side_global else "side_archetype_local"
                    ),
                }
            )
        definition_manifest.append(
            {
                "side_name": side,
                "archetype_policy_key": archetype,
                "recognition_scope": (
                    "side_global" if is_side_global else "side_archetype_local"
                ),
                "lgbm_leaf_composites": [
                    {
                        "source_name": name,
                        "output_name": f"residual_episode_leaf_composite_{index}",
                        "leaf_coordinates": leaf_definitions[int(name.rsplit("_", 1)[1])],
                    }
                    for index, name in enumerate(selected_leaf_names)
                ],
                "unsupervised_pair_composites": pair_defs,
            }
        )

    composites = pd.concat(composite_parts, ignore_index=True, sort=False)
    composites.to_parquet(
        args.output / "residual_episode_composite_features.parquet",
        index=False,
        compression="zstd",
    )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output / "composite_metrics.csv", index=False)
    coverage = pd.DataFrame(coverage_rows).sort_values(
        ["day", "side_name", "archetype_policy_key"], kind="stable"
    )
    coverage.to_csv(args.output / "episode_coverage.csv", index=False)
    # Expand shared side-level recognition back onto the exact local calendar cells.
    # This keeps the target and final report side x archetype even when one market
    # state intentionally explains several archetypes on the same day.
    if not coverage.empty and "recognition_scope" in coverage:
        shared = coverage.loc[coverage["recognition_scope"].eq("side_global")].drop(
            columns=["archetype_policy_key"]
        )
        local_calendar = calendar[["day", "side_name", "archetype_policy_key"]].drop_duplicates()
        expanded = local_calendar.merge(shared, on=["day", "side_name"], how="left")
        expanded["recognized"] = expanded["recognized"].fillna(False).astype(bool)
        expanded["status"] = np.where(expanded["recognized"], "recognized", "ignored")
        expanded["recognition_scope"] = "side_global_expanded_to_local_cell"
        expanded.to_csv(args.output / "episode_coverage_local_cells.csv", index=False)
    if pair_reports:
        pd.concat(pair_reports, ignore_index=True).to_csv(
            args.output / "unsupervised_pair_search.csv", index=False
        )
    (args.output / "composite_definitions.json").write_text(
        json.dumps(definition_manifest, indent=2, default=str) + "\n"
    )
    summary = (
        coverage.groupby(["evidence_scope", "side_name", "archetype_policy_key"], observed=True)
        .agg(
            episodes=("day", "nunique"),
            recognized=("recognized", "sum"),
            recognition_rate=("recognized", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(args.output / "episode_coverage_summary.csv", index=False)
    manifest = {
        "schema": "residual_episode_composite_discovery_v1",
        "period": ["2025-01-01", args.end],
        "target": "adverse high-surprise calendar episodes only",
        "feature_rows": int(len(composites)),
        "composites": int(len(metrics)),
        "episodes": int(len(coverage)),
        "recognized_episodes": int(coverage["recognized"].sum()),
        "method": (
            "Shallow local LGBM leaf paths are clustered by shared split features. "
            "Uncovered episodes receive train-thresholded pair-bin intensity composites "
            "materialized with unsupervised_regime_learning."
        ),
        "evidence_warning": (
            "Pre-April 2026 episodes select leaf clusters, unsupervised pair definitions, "
            "and thresholds. April-July uses March-frozen LGBMs and train-selected "
            "composites only, and is final OOS evidence."
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, default=Path("data_perp/reports/global_residual_state_discovery_20260712_localmi_v4/side_timestamp_market_states.parquet"))
    parser.add_argument("--calendar", type=Path, default=Path("data_perp/reports/meta_residual_extreme_local_uncaptured_events_202501_20260708_v3/all_extreme_event_cells.csv"))
    parser.add_argument("--models", type=Path, default=Path("data_perp/reports/residual_calendar_leaf_state_discovery_20260712_v1/models"))
    parser.add_argument("--output", type=Path, default=Path("data_perp/reports/residual_episode_composite_discovery_20260712_v1"))
    parser.add_argument("--end", default="2026-07-10")
    parser.add_argument("--min-leaf-lift", type=float, default=1.35)
    parser.add_argument("--max-leaf-clusters", type=int, default=10)
    parser.add_argument("--min-composite-correlation", type=float, default=0.08)
    parser.add_argument("--min-composite-lift", type=float, default=1.25)
    parser.add_argument("--max-leaf-composites", type=int, default=6)
    parser.add_argument("--unsupervised-features", type=int, default=12)
    parser.add_argument("--max-unsupervised-composites", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    manifest = run(args)
    print(json.dumps({"status": "complete", "output": str(args.output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
