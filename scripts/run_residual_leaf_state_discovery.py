#!/usr/bin/env python3
"""Discover shallow-LGBM residual failure leaves and calendar composites."""

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
    failure_target,
    feature_cluster_composites,
    fit_matrix,
    fit_shallow_classifier,
    fit_time_leaf_clusters,
    leaf_feature_clusters,
    observable_feature_names,
    tail_recognition_metrics,
)


def _token(side: str, archetype: str) -> str:
    return f"{side}_{archetype}".replace("/", "_").replace(" ", "_")


def _score(model, matrix: np.ndarray) -> np.ndarray:
    return model.predict_proba(matrix)[:, 1].astype(np.float32)


def _train_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    finite = np.sort(np.asarray(reference, dtype=np.float32)[np.isfinite(reference)])
    if len(finite) == 0:
        return np.full(len(values), 0.5, dtype=np.float32)
    return (
        np.searchsorted(finite, np.asarray(values, dtype=np.float32), side="right")
        / float(len(finite))
    ).astype(np.float32)


def _target_name(side: str, archetype: str) -> str:
    return f"target_signature_arch__{_token(side, archetype)}_negative_persistence_prev7d"


def _failure_signal(frame: pd.DataFrame, side: str, archetype: str) -> pd.Series:
    """Combine current adverse surprise with persistence in comparable units."""
    prefix = f"target_signature_arch__{_token(side, archetype)}_"
    signed = pd.to_numeric(frame[f"{prefix}signed_surprise"], errors="coerce")
    persistence = pd.to_numeric(
        frame[f"{prefix}negative_persistence_prev7d"], errors="coerce"
    )
    adverse = (-signed).clip(lower=0.0)
    return (adverse + np.sqrt(persistence.clip(lower=0.0))).astype(np.float32)


def _calendar_metrics(
    scored: pd.DataFrame, calendar: pd.DataFrame, score_column: str
) -> dict[str, float]:
    daily = (
        scored.assign(day=scored["__ts__"].dt.floor("D"))
        .groupby("day", observed=True)[score_column]
        .max()
        .rename("score")
        .reset_index()
    )
    events = calendar.loc[
        calendar["side_name"].eq(scored["side_name"].iloc[0])
        & calendar["archetype_policy_key"].eq(scored["archetype_policy_key"].iloc[0])
        & calendar["adverse_event_rows"].gt(0),
        ["day"],
    ].drop_duplicates()
    daily["event"] = daily["day"].isin(events["day"]).astype(np.int8)
    if daily["event"].sum() == 0:
        return {"calendar_adverse_dates": 0}
    metrics = tail_recognition_metrics(
        daily["event"].to_numpy(),
        daily["event"].to_numpy(dtype=np.float32),
        daily["score"].to_numpy(dtype=np.float32),
    )
    return {"calendar_adverse_dates": int(daily["event"].sum()), **metrics}


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    config = ResidualLeafConfig(
        max_features=int(args.max_features),
        target_quantile=float(args.target_quantile),
        n_estimators=int(args.n_estimators),
        feature_cluster_count=int(args.feature_clusters),
        time_cluster_count=int(args.time_clusters),
        random_state=int(args.seed),
    )
    states = pd.read_parquet(args.states)
    states["__ts__"] = pd.to_datetime(states["__ts__"], utc=True, errors="coerce")
    states["side_name"] = states["side_name"].astype(str).str.lower()
    calendar = pd.read_csv(args.calendar)
    calendar["day"] = pd.to_datetime(calendar["day"], utc=True, errors="coerce").dt.floor("D")
    fit_end = pd.Timestamp(args.fit_end, tz="UTC")
    tune_start = pd.Timestamp(args.tune_start, tz="UTC")
    tune_end = pd.Timestamp(args.tune_end, tz="UTC")
    eval_end = pd.Timestamp(args.evaluation_end, tz="UTC")
    purged_fit_end = fit_end - pd.Timedelta(hours=float(args.purge_hours))

    pairs = []
    for column in states.columns:
        prefix = "target_signature_arch__"
        suffix = "_negative_persistence_prev7d"
        if column.startswith(prefix) and column.endswith(suffix):
            body = column[len(prefix) : -len(suffix)]
            side = body.split("_", 1)[0]
            archetype = body[len(side) + 1 :]
            pairs.append((side, archetype, column))

    summaries: list[dict[str, object]] = []
    calendar_rows: list[dict[str, object]] = []
    composite_recognition_rows: list[dict[str, object]] = []
    leaf_parts: list[pd.DataFrame] = []
    composite_parts: list[pd.DataFrame] = []
    manifests: list[dict[str, object]] = []
    for pair_index, (side, archetype, target_column) in enumerate(pairs):
        print(
            json.dumps(
                {
                    "event": "partition_start",
                    "partition": f"{side}__{archetype}",
                    "index": pair_index + 1,
                    "total": len(pairs),
                }
            ),
            flush=True,
        )
        local = states.loc[
            states["side_name"].eq(side)
            & states["__ts__"].lt(eval_end)
        ].sort_values("__ts__", kind="stable").reset_index(drop=True)
        local["__failure_signal"] = _failure_signal(local, side, archetype)
        local = local.loc[local["__failure_signal"].notna()].reset_index(drop=True)
        early = local.loc[local["__ts__"].lt(tune_start)]
        tune = local.loc[local["__ts__"].ge(tune_start) & local["__ts__"].lt(tune_end)]
        final_train = local.loc[local["__ts__"].lt(purged_fit_end)]
        evaluation = local.loc[local["__ts__"].ge(fit_end) & local["__ts__"].lt(eval_end)]
        if min(len(early), len(tune), len(evaluation)) < 100:
            continue
        features = observable_feature_names(final_train, config)
        x_early, [x_tune], _ = fit_matrix(early, [tune], features)
        y_early, w_early, early_threshold = failure_target(
            early["__failure_signal"], config.target_quantile
        )
        y_tune = pd.to_numeric(tune["__failure_signal"], errors="coerce").fillna(0).ge(
            early_threshold
        ).to_numpy(dtype=np.int8)
        severity_tune = pd.to_numeric(tune["__failure_signal"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        search: list[dict[str, object]] = []
        seed = int(args.seed + pair_index * 104729 + zlib.crc32(archetype.encode()))
        for params in candidate_parameter_grid(config.n_estimators):
            model = fit_shallow_classifier(x_early, y_early, w_early, params, seed)
            metrics = tail_recognition_metrics(y_tune, severity_tune, _score(model, x_tune))
            finite_metric = {
                key: float(value) if np.isfinite(value) else 0.0
                for key, value in metrics.items()
            }
            objective = (
                0.45 * finite_metric.get("top10_precision", 0.0)
                + 0.30 * finite_metric.get("top10_severity_recall", 0.0)
                + 0.20 * finite_metric.get("average_precision", 0.0)
                - 0.05 * finite_metric.get("top10_false_positive_rate", 0.0)
            )
            search.append({"params": params, "objective": objective, **metrics})
        best = max(search, key=lambda row: float(row["objective"]))

        # Materialize January-March features from a model fitted only on earlier
        # rows. These are the leakage-safe downstream-meta training features.
        early_model = fit_shallow_classifier(
            x_early, y_early, w_early, best["params"], seed
        )
        early_score = _score(early_model, x_early)
        tune_score = _score(early_model, x_tune)
        early_leaves = early_model.predict(x_early, pred_leaf=True).astype(np.int16)
        tune_leaves = early_model.predict(x_tune, pred_leaf=True).astype(np.int16)
        early_severity = pd.to_numeric(
            early["__failure_signal"], errors="coerce"
        ).fillna(0).to_numpy(dtype=np.float32)
        early_leaf_table, early_mapping = leaf_feature_clusters(
            early_model, features, early_leaves, early_severity, config
        )
        early_feature_count = int(early_leaf_table["feature_cluster"].max()) + 1
        early_feature = feature_cluster_composites(
            early_leaves, early_mapping, early_feature_count
        )
        tune_feature = feature_cluster_composites(
            tune_leaves, early_mapping, early_feature_count
        )
        early_time_model = fit_time_leaf_clusters(
            early_leaves, early_severity, config
        )
        _, _, early_time_risk = early_time_model.transform(early_leaves)
        tune_time_probability, tune_time_state, tune_time_risk = (
            early_time_model.transform(tune_leaves)
        )
        tune_score_rank = _train_percentile(early_score, tune_score)
        tune_time_rank = _train_percentile(early_time_risk, tune_time_risk)
        tune_feature_rank = _train_percentile(
            early_feature.max(axis=1), tune_feature.max(axis=1)
        )
        tune_composite = tune[["__ts__", "side_name"]].copy()
        tune_composite["archetype_policy_key"] = archetype
        tune_composite["residual_leaf_failure_probability"] = tune_score
        tune_composite["residual_leaf_time_state_id"] = tune_time_state
        tune_composite["residual_leaf_time_expected_risk"] = tune_time_risk
        tune_composite["residual_leaf_failure_probability_train_pct"] = tune_score_rank
        tune_composite["residual_leaf_time_risk_train_pct"] = tune_time_rank
        tune_composite["residual_leaf_feature_risk_train_pct"] = tune_feature_rank
        tune_composite["residual_leaf_risk_composite_max"] = np.maximum.reduce(
            [tune_score_rank, tune_time_rank, tune_feature_rank]
        )
        tune_composite["residual_leaf_risk_composite_mean"] = (
            tune_score_rank + tune_time_rank + tune_feature_rank
        ) / np.float32(3.0)
        for index in range(early_feature_count):
            tune_composite[f"residual_leaf_feature_cluster_risk_{index}"] = (
                tune_feature[:, index]
            )
        for index in range(tune_time_probability.shape[1]):
            tune_composite[f"residual_leaf_time_state_probability_{index}"] = (
                tune_time_probability[:, index]
            )
        composite_parts.append(tune_composite)

        x_train, [x_eval], medians = fit_matrix(final_train, [evaluation], features)
        y_train, w_train, final_threshold = failure_target(
            final_train["__failure_signal"], config.target_quantile
        )
        severity_train = pd.to_numeric(final_train["__failure_signal"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        y_eval = pd.to_numeric(evaluation["__failure_signal"], errors="coerce").fillna(0).ge(
            final_threshold
        ).to_numpy(dtype=np.int8)
        severity_eval = pd.to_numeric(evaluation["__failure_signal"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        model = fit_shallow_classifier(x_train, y_train, w_train, best["params"], seed)
        score_train = _score(model, x_train)
        score_eval = _score(model, x_eval)
        train_leaves = model.predict(x_train, pred_leaf=True).astype(np.int16)
        eval_leaves = model.predict(x_eval, pred_leaf=True).astype(np.int16)
        leaf_table, mapping = leaf_feature_clusters(
            model, features, train_leaves, severity_train, config
        )
        leaf_table.insert(0, "archetype_policy_key", archetype)
        leaf_table.insert(0, "side_name", side)
        leaf_parts.append(leaf_table)
        feature_count = int(leaf_table["feature_cluster"].max()) + 1
        train_feature_composite = feature_cluster_composites(
            train_leaves, mapping, feature_count
        )
        feature_composite = feature_cluster_composites(eval_leaves, mapping, feature_count)
        time_model = fit_time_leaf_clusters(train_leaves, severity_train, config)
        _, _, train_time_risk = time_model.transform(train_leaves)
        time_probability, time_state, time_risk = time_model.transform(eval_leaves)

        train_feature_max = train_feature_composite.max(axis=1)
        eval_feature_max = feature_composite.max(axis=1)
        score_rank = _train_percentile(score_train, score_eval)
        time_rank = _train_percentile(train_time_risk, time_risk)
        feature_rank = _train_percentile(train_feature_max, eval_feature_max)
        composite_max = np.maximum.reduce([score_rank, time_rank, feature_rank])
        composite_mean = (score_rank + time_rank + feature_rank) / np.float32(3.0)

        composite = evaluation[["__ts__", "side_name"]].copy()
        composite["archetype_policy_key"] = archetype
        composite["residual_leaf_failure_probability"] = score_eval
        composite["residual_leaf_time_state_id"] = time_state
        composite["residual_leaf_time_expected_risk"] = time_risk
        composite["residual_leaf_failure_probability_train_pct"] = score_rank
        composite["residual_leaf_time_risk_train_pct"] = time_rank
        composite["residual_leaf_feature_risk_train_pct"] = feature_rank
        composite["residual_leaf_risk_composite_max"] = composite_max
        composite["residual_leaf_risk_composite_mean"] = composite_mean
        for index in range(feature_count):
            composite[f"residual_leaf_feature_cluster_risk_{index}"] = feature_composite[:, index]
        for index in range(time_probability.shape[1]):
            composite[f"residual_leaf_time_state_probability_{index}"] = time_probability[:, index]
        composite_parts.append(composite)

        recognition = tail_recognition_metrics(y_eval, severity_eval, score_eval)
        calendar_metric = _calendar_metrics(composite, calendar, "residual_leaf_failure_probability")
        for score_name in (
            "residual_leaf_failure_probability",
            "residual_leaf_time_expected_risk",
            "residual_leaf_feature_risk_train_pct",
            "residual_leaf_risk_composite_max",
            "residual_leaf_risk_composite_mean",
        ):
            composite_recognition_rows.append(
                {
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "composite": score_name,
                    **_calendar_metrics(composite, calendar, score_name),
                }
            )
        summaries.append(
            {
                "side_name": side,
                "archetype_policy_key": archetype,
                "train_rows": len(final_train),
                "evaluation_rows": len(evaluation),
                "features": len(features),
                "target_threshold": final_threshold,
                "best_depth": best["params"]["max_depth"],
                "best_num_leaves": best["params"]["num_leaves"],
                **recognition,
                **{f"calendar_{key}": value for key, value in calendar_metric.items()},
            }
        )
        daily = composite.assign(day=composite["__ts__"].dt.floor("D")).groupby("day", observed=True).agg(
            max_failure_probability=("residual_leaf_failure_probability", "max"),
            mean_failure_probability=("residual_leaf_failure_probability", "mean"),
            mean_time_expected_risk=("residual_leaf_time_expected_risk", "mean"),
        ).reset_index()
        adverse_dates = set(
            calendar.loc[
                calendar["side_name"].eq(side)
                & calendar["archetype_policy_key"].eq(archetype)
                & calendar["adverse_event_rows"].gt(0),
                "day",
            ]
        )
        daily["adverse_calendar_event"] = daily["day"].isin(adverse_dates)
        daily.insert(0, "archetype_policy_key", archetype)
        daily.insert(0, "side_name", side)
        calendar_rows.extend(daily.to_dict("records"))
        bundle = {
            "model": model,
            "features": features,
            "medians": medians,
            "failure_threshold": final_threshold,
            "feature_leaf_mapping": mapping,
            "time_leaf_clusters": time_model,
            "side_name": side,
            "archetype_policy_key": archetype,
            "fit_end": str(purged_fit_end),
        }
        bundle_path = args.output / "models" / f"{_token(side, archetype)}.joblib"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, bundle_path, compress=3)
        manifests.append(
            {
                "side_name": side,
                "archetype_policy_key": archetype,
                "model": str(bundle_path),
                "best_params": best["params"],
                "tuning_objective": best["objective"],
                "features": features,
            }
        )
        print(
            json.dumps(
                {
                    "event": "partition_complete",
                    "partition": f"{side}__{archetype}",
                    "oos_top10_lift": recognition.get("top10_lift"),
                    "calendar_top10_lift": calendar_metric.get("top10_lift"),
                }
            ),
            flush=True,
        )

    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output / "recognition_summary.csv", index=False)
    pd.DataFrame(composite_recognition_rows).to_csv(
        args.output / "calendar_composite_recognition.csv", index=False
    )
    pd.concat(leaf_parts, ignore_index=True).to_csv(args.output / "leaf_feature_clusters.csv", index=False)
    composites = pd.concat(composite_parts, ignore_index=True, sort=False)
    composites.to_parquet(args.output / "oos_leaf_state_composites.parquet", index=False, compression="zstd")
    pd.DataFrame(calendar_rows).to_csv(args.output / "calendar_side_archetype_scores.csv", index=False)
    manifest = {
        "schema": "residual_leaf_state_discovery_v1",
        "states": str(args.states),
        "train_period": [str(states["__ts__"].min()), str(purged_fit_end)],
        "tuning_period": [str(tune_start), str(tune_end)],
        "evaluation_period": [str(fit_end), str(eval_end)],
        "purge_hours": float(args.purge_hours),
        "models": manifests,
        "leakage_contract": (
            "Residual outcomes define train targets only. Feature medians, shallow LGBMs, "
            "leaf risk priors, feature-leaf clusters, SVD, and time-leaf clusters are fitted "
            "before April 2026. April-July rows receive frozen transforms only."
        ),
        "target_contract": (
            "Per side x archetype adverse residual intensity equals max(-signed surprise, 0) "
            "+ sqrt(max(negative persistence versus the shifted prior-week mean, 0)). "
            "The extreme class threshold is the train-only positive-intensity quantile."
        ),
        "downstream_feature_contract": (
            "January-March composite rows are generated by models fitted on 2025 rows. "
            "April-July composites are generated by models fitted through the purged "
            "March boundary. Generic risk-percentile semantics are stable across refits."
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, default=Path("data_perp/reports/global_residual_state_discovery_20260712_localmi_v4/side_timestamp_market_states.parquet"))
    parser.add_argument("--calendar", type=Path, default=Path("data_perp/reports/meta_residual_extreme_local_uncaptured_events_202501_20260708_v3/all_extreme_event_cells.csv"))
    parser.add_argument("--output", type=Path, default=Path("data_perp/reports/residual_leaf_state_discovery_20260712_v1"))
    parser.add_argument("--tune-start", default="2026-01-01")
    parser.add_argument("--tune-end", default="2026-04-01")
    parser.add_argument("--fit-end", default="2026-04-01")
    parser.add_argument("--evaluation-end", default="2026-07-11")
    parser.add_argument("--purge-hours", type=float, default=12.0)
    parser.add_argument("--max-features", type=int, default=1200)
    parser.add_argument("--target-quantile", type=float, default=0.90)
    parser.add_argument("--n-estimators", type=int, default=240)
    parser.add_argument("--feature-clusters", type=int, default=6)
    parser.add_argument("--time-clusters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    manifest = run(args)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "models": len(manifest["models"]),
                "evaluation_period": manifest["evaluation_period"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
