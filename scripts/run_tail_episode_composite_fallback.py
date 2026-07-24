#!/usr/bin/env python3
"""Discover diverse low-FPR tail composites for residual calendar episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.residual_leaf_state_discovery import causal_rolling_summary_features
from extreme_price_movements.unsupervised_regime_learning.economic_relevance import (
    TailEpisodeCompositeConfig,
    discover_tail_episode_composites,
    materialize_tail_episode_composites,
)


def _binary_metrics(active: pd.Series, event: pd.Series) -> dict[str, float]:
    valid = active.notna() & event.notna()
    x = active.loc[valid].astype(bool).to_numpy()
    y = event.loc[valid].astype(bool).to_numpy()
    tp = int((x & y).sum())
    precision = float(tp / max(x.sum(), 1))
    recall = float(tp / max(y.sum(), 1))
    fpr = float((x & ~y).sum() / max((~y).sum(), 1))
    corr = float(np.corrcoef(x.astype(float), y.astype(float))[0, 1]) if x.std() and y.std() else np.nan
    return {"precision": precision, "recall": recall, "false_positive_rate": fpr, "correlation": corr}


def _validated_definitions(
    local: pd.DataFrame,
    definitions: list[dict[str, object]],
    event_days: set[pd.Timestamp],
    validation_start: pd.Timestamp,
    fit_end: pd.Timestamp,
    max_rule_fpr: float,
    max_union_fpr: float,
) -> list[dict[str, object]]:
    if not definitions:
        return []
    materialized = materialize_tail_episode_composites(local, definitions)
    materialized["day"] = local["day"].to_numpy()
    daily = materialized.groupby("day", observed=True).max()
    validation = daily.loc[(daily.index >= validation_start) & (daily.index < fit_end)]
    event = pd.Series(validation.index.isin(event_days), index=validation.index)
    prevalence = float(event.mean())
    candidates: list[tuple[float, dict[str, object], pd.Series]] = []
    for definition in definitions:
        active = validation[str(definition["name"])].gt(0)
        metrics = _binary_metrics(active, event)
        if not active.any():
            # A genuinely rare train-supported state may be absent from the
            # short validation slice. Preserve it as dormant evidence rather
            # than misclassifying absence as a failed rule; it adds zero union
            # false positives during validation.
            candidates.append((-0.01, definition, active))
            continue
        if (
            metrics["false_positive_rate"] <= max_rule_fpr
            and metrics["precision"] > prevalence
            and np.isfinite(metrics["correlation"])
            and metrics["correlation"] > 0
        ):
            objective = metrics["correlation"] + 0.20 * metrics["recall"] - 0.30 * metrics["false_positive_rate"]
            candidates.append((objective, definition, active))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[dict[str, object]] = []
    union = pd.Series(False, index=validation.index)
    for _, definition, active in candidates:
        proposed = union | active
        if _binary_metrics(proposed, event)["false_positive_rate"] > max_union_fpr:
            continue
        renamed = dict(definition)
        renamed["name"] = f"tail_episode_composite_{len(selected)}"
        selected.append(renamed)
        union = proposed
    return selected


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    states = pd.read_parquet(args.states)
    states["__ts__"] = pd.to_datetime(states["__ts__"], utc=True)
    states["side_name"] = states["side_name"].astype(str).str.lower()
    states = states.loc[states["__ts__"].lt(pd.Timestamp(args.end, tz="UTC"))]
    calendar = pd.read_csv(args.calendar)
    calendar["day"] = pd.to_datetime(calendar["day"], utc=True).dt.floor("D")
    calendar = calendar.loc[calendar["adverse_event_rows"].gt(0)]
    fit_end = pd.Timestamp(args.fit_end, tz="UTC")
    definitions_manifest: list[dict[str, object]] = []
    feature_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for relevance_path in sorted(args.relevance.glob("feature_relevance__*.csv")):
        relevance = pd.read_csv(relevance_path)
        side = str(relevance["side_name"].iloc[0]).lower()
        archetype = str(relevance["archetype_policy_key"].iloc[0])
        if args.side and side != args.side:
            continue
        if args.archetype and archetype != args.archetype:
            continue
        raw_features = relevance.nlargest(int(args.raw_features), "stable_score")["feature"].tolist()
        raw_features = [feature for feature in raw_features if feature in states.columns]
        local = states.loc[states["side_name"].eq(side), ["__ts__", "side_name"] + raw_features].copy()
        local = local.sort_values("__ts__", kind="stable").reset_index(drop=True)
        rolling = causal_rolling_summary_features(local, raw_features, window=int(args.summary_hours))
        local = pd.concat([local, rolling], axis=1, copy=False)
        local["day"] = local["__ts__"].dt.floor("D")
        event_days = set(calendar.loc[calendar["side_name"].eq(side) & calendar["archetype_policy_key"].eq(archetype), "day"])
        daily = local.groupby("day", observed=True).last()
        daily["event"] = daily.index.isin(event_days).astype(np.int8)
        validation_start = pd.Timestamp(args.validation_start, tz="UTC")
        train_daily = daily.loc[daily.index < validation_start]
        candidate_features = raw_features + rolling.columns.tolist()
        definitions, search = discover_tail_episode_composites(
            train_daily,
            event_col="event",
            feature_columns=candidate_features,
            config=TailEpisodeCompositeConfig(
                min_event_days=int(args.min_event_days),
                max_false_positive_rate=float(args.max_fpr),
                min_lift=float(args.min_lift),
                max_single_candidates=int(args.single_candidates),
                max_pair_candidates=int(args.pair_candidates),
                max_selected=int(args.max_composites),
            ),
        )
        definitions = _validated_definitions(
            local,
            definitions,
            event_days,
            validation_start,
            fit_end,
            float(args.validation_rule_fpr),
            float(args.validation_union_fpr),
        )
        search.insert(0, "archetype_policy_key", archetype)
        search.insert(0, "side_name", side)
        search.to_csv(args.output / f"search__{side}__{archetype}.csv", index=False)
        materialized = materialize_tail_episode_composites(local, definitions)
        output = local[["__ts__", "side_name"]].copy()
        output["archetype_policy_key"] = archetype
        renamed: dict[str, str] = {}
        for column in materialized:
            renamed[column] = f"residual_tail_{archetype}__{column}"
        output = pd.concat([output, materialized.rename(columns=renamed)], axis=1, copy=False)
        feature_parts.append(output)
        binary_columns = [renamed[definition["name"]] for definition in definitions]
        daily_active = output.assign(day=local["day"].to_numpy()).groupby("day", observed=True)[binary_columns].max() if binary_columns else pd.DataFrame(index=daily.index)
        any_active = daily_active.max(axis=1).reindex(daily.index).fillna(0).astype(bool) if binary_columns else pd.Series(False, index=daily.index)
        event = daily["event"].astype(bool)
        for scope, mask in (("train", daily.index < fit_end), ("final_oos", daily.index >= fit_end)):
            metric_rows.append({"side_name": side, "archetype_policy_key": archetype, "scope": scope, "composites": len(definitions), **_binary_metrics(any_active.loc[mask], event.loc[mask])})
        for day in sorted(day for day in event_days if day >= fit_end):
            matches = [column for column in binary_columns if day in daily_active.index and float(daily_active.loc[day, column]) > 0]
            coverage_rows.append({"day": day, "side_name": side, "archetype_policy_key": archetype, "recognized": bool(matches), "status": "recognized" if matches else "ignored", "matching_composites": "|".join(matches)})
        definitions_manifest.append({"side_name": side, "archetype_policy_key": archetype, "fit_end": str(fit_end), "summary_hours": int(args.summary_hours), "definitions": definitions})
    features = pd.concat(feature_parts, ignore_index=True, sort=False)
    features.to_parquet(args.output / "tail_episode_composite_features.parquet", index=False, compression="zstd")
    pd.DataFrame(metric_rows).to_csv(args.output / "metrics.csv", index=False)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(args.output / "oos_episode_coverage.csv", index=False)
    (args.output / "composite_definitions.json").write_text(json.dumps(definitions_manifest, indent=2) + "\n")
    manifest = {"schema": "tail_episode_composite_fallback_v1", "target": "adverse high-residual-autocorrelation calendar episodes only", "fit_end": str(fit_end), "feature_rows": int(len(features)), "definitions": int(sum(len(item["definitions"]) for item in definitions_manifest)), "oos_episodes": int(len(coverage)), "oos_recognized": int(coverage["recognized"].sum())}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, default=Path("data_perp/reports/global_residual_state_discovery_20260712_localmi_v4/side_timestamp_market_states.parquet"))
    parser.add_argument("--calendar", type=Path, default=Path("data_perp/reports/meta_residual_extreme_local_uncaptured_events_202501_20260708_v3/all_extreme_event_cells.csv"))
    parser.add_argument("--relevance", type=Path, default=Path("data_perp/reports/causal_summary_residual_calendar_challenger_20260712_v1"))
    parser.add_argument("--output", type=Path, default=Path("data_perp/reports/tail_episode_composite_fallback_20260712_v1"))
    parser.add_argument("--fit-end", default="2026-04-01")
    parser.add_argument("--validation-start", default="2026-01-01")
    parser.add_argument("--end", default="2026-07-10")
    parser.add_argument("--summary-hours", type=int, default=24)
    parser.add_argument("--side", choices=("long", "short"))
    parser.add_argument("--archetype")
    parser.add_argument("--raw-features", type=int, default=120)
    parser.add_argument("--min-event-days", type=int, default=4)
    parser.add_argument("--max-fpr", type=float, default=0.08)
    parser.add_argument("--min-lift", type=float, default=1.5)
    parser.add_argument("--single-candidates", type=int, default=100)
    parser.add_argument("--pair-candidates", type=int, default=4000)
    parser.add_argument("--max-composites", type=int, default=24)
    parser.add_argument("--validation-rule-fpr", type=float, default=0.10)
    parser.add_argument("--validation-union-fpr", type=float, default=0.15)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
