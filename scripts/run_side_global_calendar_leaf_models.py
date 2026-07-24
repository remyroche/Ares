#!/usr/bin/env python3
"""Fit shallow long/short models for calendar episodes shared across archetypes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from extreme_price_movements.residual_leaf_state_discovery import (
    ResidualLeafConfig,
    candidate_parameter_grid,
    fit_matrix,
    fit_shallow_classifier,
    observable_feature_names,
    stable_binned_mi_screen,
    tail_recognition_metrics,
)
from scripts.run_residual_calendar_leaf_state_discovery import _weights


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    states = pd.read_parquet(args.states)
    states["__ts__"] = pd.to_datetime(states["__ts__"], utc=True)
    states["side_name"] = states["side_name"].astype(str).str.lower()
    states = states.loc[states["__ts__"].lt(pd.Timestamp(args.end, tz="UTC"))]
    calendar = pd.read_csv(args.calendar)
    calendar["day"] = pd.to_datetime(calendar["day"], utc=True).dt.floor("D")
    calendar = calendar.loc[calendar["adverse_event_rows"].gt(0)]
    config = ResidualLeafConfig(
        max_features=int(args.max_features),
        n_estimators=int(args.n_estimators),
        random_state=int(args.seed),
    )
    tune_start = pd.Timestamp("2026-01-01", tz="UTC")
    fit_end = pd.Timestamp("2026-04-01", tz="UTC")
    purge = pd.Timedelta(hours=float(args.purge_hours))
    manifests = []
    hpo_rows = []
    recognition_rows = []
    for side_index, side in enumerate(("long", "short")):
        local = states.loc[states["side_name"].eq(side)].copy()
        local["day"] = local["__ts__"].dt.floor("D")
        event_days = set(calendar.loc[calendar["side_name"].eq(side), "day"])
        local["__calendar_event"] = local["day"].isin(event_days).astype(np.int8)
        severity = (
            calendar.loc[calendar["side_name"].eq(side)]
            .groupby("day", observed=True)
            .agg(
                persistence_strength=("persistence_strength", "max"),
                large_event_strength=("large_event_strength", "max"),
            )
        )
        local["__calendar_event_severity"] = (
            1.0
            + local["day"].map(severity["persistence_strength"]).fillna(0).clip(0, 5)
            + local["day"].map(severity["large_event_strength"]).fillna(0).clip(0, 5)
        ).where(local["__calendar_event"].gt(0), 0.0).astype(np.float32)
        final_train = local.loc[local["__ts__"].lt(fit_end - purge)]
        early = local.loc[local["__ts__"].lt(tune_start - purge)]
        tune = local.loc[local["__ts__"].ge(tune_start) & local["__ts__"].lt(fit_end)]
        candidates = observable_feature_names(final_train, config)
        features, relevance = stable_binned_mi_screen(
            early,
            early["__calendar_event"].to_numpy(dtype=np.int8),
            candidates,
            max_features=int(args.screened_features),
        )
        relevance.insert(0, "side_name", side)
        relevance["selected"] = relevance["feature"].isin(features)
        relevance.to_csv(args.output / f"feature_relevance__{side}.csv", index=False)
        x_early, [x_tune], _ = fit_matrix(early, [tune], features)
        y_early = early["__calendar_event"].to_numpy(dtype=np.int8)
        severity_early = early["__calendar_event_severity"].to_numpy(dtype=np.float32)
        y_tune = tune["__calendar_event"].to_numpy(dtype=np.int8)
        severity_tune = tune["__calendar_event_severity"].to_numpy(dtype=np.float32)
        search = []
        for params in candidate_parameter_grid(config.n_estimators):
            model = fit_shallow_classifier(
                x_early,
                y_early,
                _weights(y_early, severity_early),
                params,
                int(args.seed + side_index),
            )
            metrics = tail_recognition_metrics(
                y_tune, severity_tune, model.predict_proba(x_tune)[:, 1]
            )
            finite = {k: float(v) if np.isfinite(v) else 0.0 for k, v in metrics.items()}
            objective = (
                0.45 * finite.get("top10_precision", 0)
                + 0.30 * finite.get("top10_recall", 0)
                + 0.20 * finite.get("top10_severity_recall", 0)
                - 0.05 * finite.get("top10_false_positive_rate", 0)
            )
            search.append({"params": params, "objective": objective, **metrics})
        best = max(search, key=lambda row: float(row["objective"]))
        hpo_rows.append({"side_name": side, "objective": best["objective"], **best["params"]})
        x_train, _, medians = fit_matrix(final_train, [], features)
        y_train = final_train["__calendar_event"].to_numpy(dtype=np.int8)
        severity_train = final_train["__calendar_event_severity"].to_numpy(dtype=np.float32)
        model = fit_shallow_classifier(
            x_train,
            y_train,
            _weights(y_train, severity_train),
            best["params"],
            int(args.seed + side_index),
        )
        model_path = args.output / "models" / f"{side}____side_global__.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "features": features,
                "medians": medians,
                "side_name": side,
                "archetype_policy_key": "__side_global__",
                "fit_end": str(fit_end - purge),
                "best_params": best["params"],
            },
            model_path,
            compress=3,
        )
        evaluation = local.loc[local["__ts__"].ge(fit_end)]
        x_eval = _safe_eval_matrix(evaluation, features, medians)
        eval_score = model.predict_proba(x_eval)[:, 1]
        recognition_rows.append(
            {
                "side_name": side,
                **tail_recognition_metrics(
                    evaluation["__calendar_event"].to_numpy(dtype=np.int8),
                    evaluation["__calendar_event_severity"].to_numpy(dtype=np.float32),
                    eval_score,
                ),
            }
        )
        manifests.append(
            {
                "side_name": side,
                "model": str(model_path),
                "features": features,
                "best_params": best["params"],
            }
        )
    pd.DataFrame(hpo_rows).to_csv(args.output / "hpo_summary.csv", index=False)
    pd.DataFrame(recognition_rows).to_csv(args.output / "oos_recognition.csv", index=False)
    manifest = {
        "schema": "side_global_calendar_leaf_models_v1",
        "fit_end": str(fit_end - purge),
        "evaluation_end": args.end,
        "models": manifests,
        "target": "any adverse high-surprise calendar episode on the same side",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _safe_eval_matrix(frame: pd.DataFrame, features: list[str], medians: np.ndarray) -> np.ndarray:
    values = frame[features].to_numpy(dtype=np.float32, copy=True)
    invalid = ~np.isfinite(values)
    if invalid.any():
        values[invalid] = np.take(medians, np.nonzero(invalid)[1])
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, default=Path("data_perp/reports/global_residual_state_discovery_20260712_localmi_v4/side_timestamp_market_states.parquet"))
    parser.add_argument("--calendar", type=Path, default=Path("data_perp/reports/meta_residual_extreme_local_uncaptured_events_202501_20260708_v3/all_extreme_event_cells.csv"))
    parser.add_argument("--output", type=Path, default=Path("data_perp/reports/side_global_calendar_leaf_models_20260712_v1"))
    parser.add_argument("--end", default="2026-07-10")
    parser.add_argument("--purge-hours", type=float, default=12.0)
    parser.add_argument("--max-features", type=int, default=1200)
    parser.add_argument("--screened-features", type=int, default=250)
    parser.add_argument("--n-estimators", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    manifest = run(args)
    print(json.dumps({"status": "complete", "output": str(args.output), "models": len(manifest["models"])}, indent=2))


if __name__ == "__main__":
    main()
