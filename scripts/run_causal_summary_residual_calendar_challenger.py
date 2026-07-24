#!/usr/bin/env python3
"""Test causal rolling state summaries for local residual-calendar episodes."""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from extreme_price_movements.residual_leaf_state_discovery import (
    ResidualLeafConfig,
    candidate_parameter_grid,
    causal_rolling_summary_features,
    fit_matrix,
    fit_shallow_classifier,
    observable_feature_names,
    stable_binned_mi_screen,
    tail_recognition_metrics,
)
from scripts.run_residual_calendar_leaf_state_discovery import (
    _attach_calendar_target,
    _event_table,
    _weights,
)


def _daily_metrics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    daily = pd.DataFrame(
        {
            "day": frame["__ts__"].dt.floor("D").to_numpy(),
            "score": score,
            "event": frame["__calendar_event"].to_numpy(),
            "severity": frame["__calendar_event_severity"].to_numpy(),
        }
    ).groupby("day", observed=True).agg(
        score=("score", "max"), event=("event", "max"), severity=("severity", "max")
    )
    return tail_recognition_metrics(
        daily["event"].to_numpy(dtype=np.int8),
        daily["severity"].to_numpy(dtype=np.float32),
        daily["score"].to_numpy(dtype=np.float32),
    )


def _augment(local: pd.DataFrame, raw_features: list[str], window: int) -> tuple[pd.DataFrame, list[str]]:
    rolling = causal_rolling_summary_features(local, raw_features, window=window)
    result = pd.concat([local, rolling], axis=1, copy=False)
    return result, raw_features + rolling.columns.tolist()


def _apply_matrix(frame: pd.DataFrame, features: list[str], medians: np.ndarray) -> np.ndarray:
    values = frame[features].to_numpy(dtype=np.float32, copy=True)
    invalid = ~np.isfinite(values)
    if invalid.any():
        values[invalid] = np.take(medians, np.nonzero(invalid)[1])
    return values


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    states = pd.read_parquet(args.states)
    states["__ts__"] = pd.to_datetime(states["__ts__"], utc=True)
    states["side_name"] = states["side_name"].astype(str).str.lower()
    states = states.loc[states["__ts__"].lt(pd.Timestamp(args.end, tz="UTC"))]
    calendar = pd.read_csv(args.calendar)
    calendar["day"] = pd.to_datetime(calendar["day"], utc=True).dt.floor("D")
    calendar = calendar.loc[calendar["adverse_event_rows"].gt(0)]
    config = ResidualLeafConfig(max_features=int(args.max_features), n_estimators=int(args.n_estimators))
    tune_start = pd.Timestamp("2026-01-01", tz="UTC")
    fit_end = pd.Timestamp("2026-04-01", tz="UTC")
    purge = pd.Timedelta(hours=float(args.purge_hours))
    metric_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    pairs = calendar[["side_name", "archetype_policy_key"]].drop_duplicates()
    for pair_index, pair in enumerate(pairs.itertuples(index=False)):
        side, archetype = str(pair.side_name).lower(), str(pair.archetype_policy_key)
        events = _event_table(calendar, side, archetype)
        local = states.loc[states["side_name"].eq(side)].copy().sort_values("__ts__", kind="stable")
        local["archetype_policy_key"] = archetype
        local = _attach_calendar_target(local, events)
        early_raw = local.loc[local["__ts__"].lt(tune_start - purge)]
        if early_raw["__calendar_event"].sum() < 5:
            continue
        candidates = observable_feature_names(early_raw, config)
        raw_features, relevance = stable_binned_mi_screen(
            early_raw,
            early_raw["__calendar_event"].to_numpy(dtype=np.int8),
            candidates,
            max_features=int(args.screened_raw_features),
        )
        local, model_features = _augment(local, raw_features, int(args.summary_hours))
        early = local.loc[local["__ts__"].lt(tune_start - purge)]
        tune = local.loc[local["__ts__"].ge(tune_start) & local["__ts__"].lt(fit_end)]
        train = local.loc[local["__ts__"].lt(fit_end - purge)]
        x_early, [x_tune], _ = fit_matrix(early, [tune], model_features)
        y_early = early["__calendar_event"].to_numpy(dtype=np.int8)
        severity_early = early["__calendar_event_severity"].to_numpy(dtype=np.float32)
        seed = int(args.seed + pair_index * 104729 + zlib.crc32(archetype.encode()))
        search = []
        for params in candidate_parameter_grid(config.n_estimators):
            model = fit_shallow_classifier(x_early, y_early, _weights(y_early, severity_early), params, seed)
            metrics = _daily_metrics(tune, model.predict_proba(x_tune)[:, 1])
            finite = {key: float(value) if np.isfinite(value) else 0.0 for key, value in metrics.items()}
            objective = (
                0.50 * finite.get("top10_precision", 0)
                + 0.25 * finite.get("top10_recall", 0)
                + 0.20 * finite.get("top10_severity_recall", 0)
                - 0.15 * finite.get("top10_false_positive_rate", 0)
            )
            search.append({"params": params, "objective": objective, **metrics})
        best = max(search, key=lambda row: float(row["objective"]))
        x_train, _, medians = fit_matrix(train, [], model_features)
        y_train = train["__calendar_event"].to_numpy(dtype=np.int8)
        severity_train = train["__calendar_event_severity"].to_numpy(dtype=np.float32)
        model = fit_shallow_classifier(x_train, y_train, _weights(y_train, severity_train), best["params"], seed)
        evaluation = local.loc[local["__ts__"].ge(fit_end)].copy()
        x_eval = _apply_matrix(evaluation, model_features, medians)
        score = model.predict_proba(x_eval)[:, 1]
        oos_metrics = _daily_metrics(evaluation, score)
        metric_rows.append({"side_name": side, "archetype_policy_key": archetype, "objective": best["objective"], **oos_metrics})
        daily = pd.DataFrame({"day": evaluation["__ts__"].dt.floor("D"), "score": score}).groupby("day", observed=True)["score"].max()
        train_score = model.predict_proba(x_train)[:, 1]
        train_daily = pd.DataFrame({"day": train["__ts__"].dt.floor("D"), "score": train_score}).groupby("day", observed=True)["score"].max()
        threshold = float(train_daily.quantile(0.90))
        for day in events.loc[events["day"].ge(fit_end), "day"]:
            value = float(daily.get(day, np.nan))
            coverage_rows.append({"day": day, "side_name": side, "archetype_policy_key": archetype, "score": value, "threshold": threshold, "recognized": bool(np.isfinite(value) and value >= threshold)})
        model_path = args.output / "models" / f"{side}__{archetype}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "raw_features": raw_features, "model_features": model_features, "medians": medians, "summary_hours": int(args.summary_hours), "best_params": best["params"], "fit_end": str(fit_end - purge), "feature_transform_contract": "causal rolling min/max/mean including current row only"}, model_path, compress=3)
        relevance.insert(0, "archetype_policy_key", archetype)
        relevance.insert(0, "side_name", side)
        relevance.to_csv(args.output / f"feature_relevance__{side}__{archetype}.csv", index=False)
        manifests.append({"side_name": side, "archetype_policy_key": archetype, "model": str(model_path), "raw_features": len(raw_features), "model_features": len(model_features)})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output / "oos_metrics.csv", index=False)
    coverage = pd.DataFrame(coverage_rows)
    coverage["status"] = np.where(coverage["recognized"], "recognized", "ignored")
    coverage.to_csv(args.output / "oos_episode_coverage.csv", index=False)
    manifest = {"schema": "causal_summary_residual_calendar_challenger_v1", "target": "adverse high-residual-autocorrelation calendar membership only", "fit_end": str(fit_end - purge), "summary_hours": int(args.summary_hours), "models": manifests, "oos_episodes": int(len(coverage)), "oos_recognized": int(coverage["recognized"].sum())}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, default=Path("data_perp/reports/global_residual_state_discovery_20260712_localmi_v4/side_timestamp_market_states.parquet"))
    parser.add_argument("--calendar", type=Path, default=Path("data_perp/reports/meta_residual_extreme_local_uncaptured_events_202501_20260708_v3/all_extreme_event_cells.csv"))
    parser.add_argument("--output", type=Path, default=Path("data_perp/reports/causal_summary_residual_calendar_challenger_20260712_v1"))
    parser.add_argument("--end", default="2026-07-10")
    parser.add_argument("--purge-hours", type=float, default=12.0)
    parser.add_argument("--summary-hours", type=int, default=24)
    parser.add_argument("--max-features", type=int, default=1200)
    parser.add_argument("--screened-raw-features", type=int, default=120)
    parser.add_argument("--n-estimators", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
