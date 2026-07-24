#!/usr/bin/env python3
"""Learn side x archetype leaf states for high-AC adverse calendar periods only."""

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
    feature_cluster_composites,
    fit_matrix,
    fit_shallow_classifier,
    fit_time_leaf_clusters,
    leaf_feature_clusters,
    observable_feature_names,
    stable_binned_mi_screen,
    tail_recognition_metrics,
)


KEYS = ["__ts__", "side_name", "archetype_policy_key"]
FOCUSED_FEATURES = [
    "residual_leaf_time_risk_train_pct",
    "residual_leaf_feature_risk_train_pct",
]


def _train_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    finite = np.sort(np.asarray(reference, dtype=np.float32)[np.isfinite(reference)])
    if not len(finite):
        return np.full(len(values), 0.5, dtype=np.float32)
    return (
        np.searchsorted(finite, np.asarray(values, dtype=np.float32), side="right")
        / float(len(finite))
    ).astype(np.float32)


def _event_table(calendar: pd.DataFrame, side: str, archetype: str) -> pd.DataFrame:
    local = calendar.loc[
        calendar["side_name"].eq(side)
        & calendar["archetype_policy_key"].eq(archetype)
        & calendar["adverse_event_rows"].gt(0)
    ].copy()
    local["event_severity"] = (
        1.0
        + pd.to_numeric(local["persistence_strength"], errors="coerce").fillna(0).clip(0, 5)
        + pd.to_numeric(local["large_event_strength"], errors="coerce").fillna(0).clip(0, 5)
    ).astype(np.float32)
    return local[["day", "event_severity"]].drop_duplicates("day", keep="last")


def _attach_calendar_target(
    frame: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    result = frame.copy()
    result["day"] = result["__ts__"].dt.floor("D")
    severity = events.set_index("day")["event_severity"]
    result["__calendar_event_severity"] = (
        result["day"].map(severity).fillna(0).astype(np.float32)
    )
    result["__calendar_event"] = result["__calendar_event_severity"].gt(0).astype(np.int8)
    return result


def _weights(label: np.ndarray, severity: np.ndarray) -> np.ndarray:
    positive_rate = max(float(np.mean(label)), 1e-4)
    positive_scale = min(12.0, max(2.0, (1.0 - positive_rate) / positive_rate))
    weight = np.ones(len(label), dtype=np.float32)
    weight[label > 0] = positive_scale * np.clip(severity[label > 0], 1.0, 5.0)
    weight /= max(float(weight.mean()), 1e-6)
    return weight


def _materialize(
    fit: pd.DataFrame,
    apply: pd.DataFrame,
    features: list[str],
    params: dict[str, object],
    config: ResidualLeafConfig,
    seed: int,
) -> tuple[pd.DataFrame, object, pd.DataFrame]:
    x_fit, [x_apply], medians = fit_matrix(fit, [apply], features)
    y_fit = fit["__calendar_event"].to_numpy(dtype=np.int8)
    severity_fit = fit["__calendar_event_severity"].to_numpy(dtype=np.float32)
    model = fit_shallow_classifier(
        x_fit, y_fit, _weights(y_fit, severity_fit), params, seed
    )
    fit_leaves = model.predict(x_fit, pred_leaf=True).astype(np.int16)
    apply_leaves = model.predict(x_apply, pred_leaf=True).astype(np.int16)
    leaf_table, mapping = leaf_feature_clusters(
        model, features, fit_leaves, severity_fit, config
    )
    cluster_count = int(leaf_table["feature_cluster"].max()) + 1
    fit_feature = feature_cluster_composites(
        fit_leaves, mapping, cluster_count
    ).max(axis=1)
    apply_feature = feature_cluster_composites(
        apply_leaves, mapping, cluster_count
    ).max(axis=1)
    time_model = fit_time_leaf_clusters(fit_leaves, severity_fit, config)
    _, _, fit_time = time_model.transform(fit_leaves)
    _, _, apply_time = time_model.transform(apply_leaves)
    output = apply[KEYS].copy()
    output[FOCUSED_FEATURES[0]] = _train_percentile(fit_time, apply_time)
    output[FOCUSED_FEATURES[1]] = _train_percentile(fit_feature, apply_feature)
    bundle = {
        "model": model,
        "features": features,
        "medians": medians,
        "feature_leaf_mapping": mapping,
        "time_leaf_clusters": time_model,
        "side_name": str(apply["side_name"].iloc[0]),
        "archetype_policy_key": str(apply["archetype_policy_key"].iloc[0]),
    }
    return output, bundle, leaf_table


def _recognition_rows(
    composites: pd.DataFrame, calendar: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (side, archetype), local in composites.groupby(
        ["side_name", "archetype_policy_key"], observed=True
    ):
        events = _event_table(calendar, str(side), str(archetype))
        daily = (
            local.assign(day=local["__ts__"].dt.floor("D"))
            .groupby("day", observed=True)
            .agg(
                time_risk=(FOCUSED_FEATURES[0], "max"),
                feature_risk=(FOCUSED_FEATURES[1], "max"),
            )
            .reset_index()
        )
        daily["combined_risk"] = daily[["time_risk", "feature_risk"]].max(axis=1)
        daily["adverse_calendar_event"] = daily["day"].isin(events["day"])
        daily["recognized"] = daily["combined_risk"].ge(0.90)
        for row in daily.loc[daily["adverse_calendar_event"]].itertuples(index=False):
            rows.append(
                {
                    "day": row.day,
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "time_risk": row.time_risk,
                    "feature_risk": row.feature_risk,
                    "combined_risk": row.combined_risk,
                    "recognized": bool(row.recognized),
                    "status": "recognized" if row.recognized else "ignored",
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["day", "side_name", "archetype_policy_key"], kind="stable"
    )


def complete_recognition_calendar(
    composites: pd.DataFrame, calendar: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return every adverse calendar cell and daily confusion metrics."""
    daily = (
        composites.assign(day=composites["__ts__"].dt.floor("D"))
        .groupby(["day", "side_name", "archetype_policy_key"], observed=True)
        .agg(
            time_risk=(FOCUSED_FEATURES[0], "max"),
            feature_risk=(FOCUSED_FEATURES[1], "max"),
        )
        .reset_index()
    )
    daily["combined_risk"] = daily[["time_risk", "feature_risk"]].max(axis=1)
    events = calendar.loc[
        calendar["adverse_event_rows"].gt(0),
        [
            "day",
            "side_name",
            "archetype_policy_key",
            "adverse_event_rows",
            "persistence_strength",
            "large_event_strength",
        ],
    ].drop_duplicates(["day", "side_name", "archetype_policy_key"], keep="last")
    result = events.merge(
        daily,
        on=["day", "side_name", "archetype_policy_key"],
        how="left",
        validate="one_to_one",
    )
    result["recognized"] = result["combined_risk"].ge(0.90)
    result["status"] = np.select(
        [result["combined_risk"].isna(), result["recognized"]],
        ["not_scored_no_prior_training", "recognized"],
        default="ignored",
    )
    event_keys = pd.MultiIndex.from_frame(
        events[["day", "side_name", "archetype_policy_key"]]
    )
    daily["adverse_calendar_event"] = pd.MultiIndex.from_frame(
        daily[["day", "side_name", "archetype_policy_key"]]
    ).isin(event_keys)
    daily["predicted_event"] = daily["combined_risk"].ge(0.90)
    confusion = (
        daily.groupby(["side_name", "archetype_policy_key"], observed=True)
        .apply(
            lambda group: pd.Series(
                {
                    "days": len(group),
                    "event_days": int(group["adverse_calendar_event"].sum()),
                    "predicted_days": int(group["predicted_event"].sum()),
                    "true_positive_days": int(
                        (group["adverse_calendar_event"] & group["predicted_event"]).sum()
                    ),
                    "precision": float(
                        (group["adverse_calendar_event"] & group["predicted_event"]).sum()
                        / max(group["predicted_event"].sum(), 1)
                    ),
                    "recall": float(
                        (group["adverse_calendar_event"] & group["predicted_event"]).sum()
                        / max(group["adverse_calendar_event"].sum(), 1)
                    ),
                    "false_positive_rate": float(
                        ((~group["adverse_calendar_event"]) & group["predicted_event"]).sum()
                        / max((~group["adverse_calendar_event"]).sum(), 1)
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    return result.sort_values(
        ["day", "side_name", "archetype_policy_key"], kind="stable"
    ), confusion


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    config = ResidualLeafConfig(
        max_features=int(args.max_features),
        n_estimators=int(args.n_estimators),
        random_state=int(args.seed),
    )
    states = pd.read_parquet(args.states)
    states["__ts__"] = pd.to_datetime(states["__ts__"], utc=True)
    states["side_name"] = states["side_name"].astype(str).str.lower()
    calendar = pd.read_csv(args.calendar)
    calendar["day"] = pd.to_datetime(calendar["day"], utc=True).dt.floor("D")
    evaluation_end = pd.Timestamp(args.evaluation_end, tz="UTC")
    tune_start = pd.Timestamp("2026-01-01", tz="UTC")
    fit_end = pd.Timestamp("2026-04-01", tz="UTC")
    purge = pd.Timedelta(hours=float(args.purge_hours))
    pairs = (
        calendar.loc[calendar["adverse_event_rows"].gt(0), ["side_name", "archetype_policy_key"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    oof_parts: list[pd.DataFrame] = []
    leaf_parts: list[pd.DataFrame] = []
    search_rows: list[dict[str, object]] = []
    feature_relevance_parts: list[pd.DataFrame] = []
    model_manifests: list[dict[str, object]] = []
    fold_manifest: list[dict[str, object]] = []
    folds = [
        (pd.Timestamp("2025-04-01", tz="UTC"), pd.Timestamp("2025-07-01", tz="UTC")),
        (pd.Timestamp("2025-07-01", tz="UTC"), pd.Timestamp("2025-10-01", tz="UTC")),
        (pd.Timestamp("2025-10-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")),
        (pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-04-01", tz="UTC")),
        (pd.Timestamp("2026-04-01", tz="UTC"), evaluation_end),
    ]
    for pair_index, (side, archetype) in enumerate(pairs):
        side = str(side).lower()
        archetype = str(archetype)
        events = _event_table(calendar, side, archetype)
        local = states.loc[
            states["side_name"].eq(side) & states["__ts__"].lt(evaluation_end)
        ].copy()
        local["archetype_policy_key"] = archetype
        local = _attach_calendar_target(local, events).sort_values("__ts__", kind="stable")
        final_train = local.loc[local["__ts__"].lt(fit_end - purge)]
        candidates = observable_feature_names(final_train, config)
        early = local.loc[local["__ts__"].lt(tune_start - purge)]
        tune = local.loc[local["__ts__"].ge(tune_start) & local["__ts__"].lt(fit_end)]
        if early["__calendar_event"].sum() < 5:
            continue
        features, relevance = stable_binned_mi_screen(
            early,
            early["__calendar_event"].to_numpy(dtype=np.int8),
            candidates,
            max_features=int(args.screened_features),
        )
        relevance.insert(0, "archetype_policy_key", archetype)
        relevance.insert(0, "side_name", side)
        relevance["selected"] = relevance["feature"].isin(features)
        feature_relevance_parts.append(relevance)
        x_early, [x_tune], _ = fit_matrix(early, [tune], features)
        y_early = early["__calendar_event"].to_numpy(dtype=np.int8)
        severity_early = early["__calendar_event_severity"].to_numpy(dtype=np.float32)
        y_tune = tune["__calendar_event"].to_numpy(dtype=np.int8)
        severity_tune = tune["__calendar_event_severity"].to_numpy(dtype=np.float32)
        seed = int(args.seed + pair_index * 104729 + zlib.crc32(archetype.encode()))
        candidates = []
        for params in candidate_parameter_grid(config.n_estimators):
            model = fit_shallow_classifier(
                x_early, y_early, _weights(y_early, severity_early), params, seed
            )
            score = model.predict_proba(x_tune)[:, 1]
            metrics = tail_recognition_metrics(y_tune, severity_tune, score)
            finite = {key: float(value) if np.isfinite(value) else 0.0 for key, value in metrics.items()}
            objective = (
                0.45 * finite.get("top10_precision", 0)
                + 0.30 * finite.get("top10_recall", 0)
                + 0.20 * finite.get("top10_severity_recall", 0)
                - 0.05 * finite.get("top10_false_positive_rate", 0)
            )
            candidates.append({"params": params, "objective": objective, **metrics})
        best = max(candidates, key=lambda row: float(row["objective"]))
        search_rows.append(
            {
                "side_name": side,
                "archetype_policy_key": archetype,
                "objective": best["objective"],
                **best["params"],
            }
        )
        for fold_start, fold_end in folds:
            fit = local.loc[local["__ts__"].lt(fold_start - purge)]
            apply = local.loc[local["__ts__"].ge(fold_start) & local["__ts__"].lt(fold_end)]
            if apply.empty or fit["__calendar_event"].sum() < 5:
                continue
            # Earlier quarterly OOF folds cannot reuse feature selection or HPO
            # performed with later observations. Screen on that fold's train rows
            # and use an ex-ante shallow parameter contract. The April-2026 frozen
            # model may use the Jan-Mar validation-selected configuration above.
            fold_features = features
            fold_params = best["params"]
            parameter_source = "prior_validation_hpo"
            if fold_start < fit_end:
                fold_candidates = observable_feature_names(fit, config)
                fold_features, _ = stable_binned_mi_screen(
                    fit,
                    fit["__calendar_event"].to_numpy(dtype=np.int8),
                    fold_candidates,
                    max_features=int(args.screened_features),
                )
                fold_params = {
                    "max_depth": 3,
                    "num_leaves": 6,
                    "min_child_samples": 72,
                    "reg_alpha": 0.25,
                    "reg_lambda": 5.0,
                    "min_split_gain": 0.01,
                    "n_estimators": int(config.n_estimators),
                }
                parameter_source = "fixed_ex_ante_shallow_contract"
            composite, bundle, leaf_table = _materialize(
                fit, apply, fold_features, fold_params, config, seed
            )
            composite["oof_fold_start"] = fold_start
            composite["oof_fold_end"] = fold_end
            oof_parts.append(composite)
            leaf_table.insert(0, "fold_start", fold_start)
            leaf_table.insert(0, "archetype_policy_key", archetype)
            leaf_table.insert(0, "side_name", side)
            leaf_parts.append(leaf_table)
            fold_manifest.append(
                {
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "fit_end_exclusive": str(fold_start - purge),
                    "apply_start": str(fold_start),
                    "apply_end_exclusive": str(fold_end),
                    "fit_rows": int(len(fit)),
                    "event_hours": int(fit["__calendar_event"].sum()),
                    "selected_features": int(len(fold_features)),
                    "feature_selection_scope": "fold_train_only",
                    "parameter_source": parameter_source,
                }
            )
            if fold_start == fit_end:
                model_path = args.output / "models" / f"{side}__{archetype}.joblib"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(bundle, model_path, compress=3)
                model_manifests.append(
                    {
                        "side_name": side,
                        "archetype_policy_key": archetype,
                        "model": str(model_path),
                        "features": features,
                        "best_params": best["params"],
                    }
                )
    composites = pd.concat(oof_parts, ignore_index=True, sort=False)
    composites.to_parquet(
        args.output / "quarterly_oof_leaf_composites.parquet",
        index=False,
        compression="zstd",
    )
    pd.concat(leaf_parts, ignore_index=True).to_csv(
        args.output / "leaf_feature_clusters_by_fold.csv", index=False
    )
    pd.DataFrame(search_rows).to_csv(args.output / "hpo_summary.csv", index=False)
    pd.concat(feature_relevance_parts, ignore_index=True).to_csv(
        args.output / "archetype_feature_relevance.csv", index=False
    )
    full_calendar, full_confusion = complete_recognition_calendar(composites, calendar)
    full_calendar.to_csv(
        args.output / "calendar_recognized_vs_ignored_full_oof.csv", index=False
    )
    full_confusion.to_csv(
        args.output / "calendar_recognition_confusion_full_oof.csv", index=False
    )
    recognized = _recognition_rows(
        composites.loc[composites["__ts__"].ge(fit_end)], calendar
    )
    recognized.to_csv(args.output / "calendar_recognized_vs_ignored.csv", index=False)
    recognition_summary = (
        recognized.groupby(["side_name", "archetype_policy_key"], observed=True)
        .agg(
            calendar_days=("day", "nunique"),
            recognized_days=("recognized", "sum"),
            recognition_rate=("recognized", "mean"),
            mean_combined_risk=("combined_risk", "mean"),
        )
        .reset_index()
    )
    recognition_summary.to_csv(
        args.output / "calendar_recognition_summary.csv", index=False
    )
    manifest = {
        "schema": "residual_calendar_leaf_state_discovery_v1",
        "target": "high_residual_autocorrelation_adverse_calendar_membership_only",
        "oof_period": ["2025-04-01", str(evaluation_end)],
        "folds": fold_manifest,
        "models": model_manifests,
        "meta_features": FOCUSED_FEATURES,
        "leakage_contract": (
            "Each quarterly apply period receives models, medians, leaf priors, feature-leaf "
            "clusters, and time-leaf clusters fitted only before that period with a 12-hour "
            "purge. Pre-2026 quarterly folds also perform feature screening on fold-train "
            "rows only and use a fixed ex-ante shallow parameter contract. Calendar labels "
            "and severity are training targets only."
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, default=Path("data_perp/reports/global_residual_state_discovery_20260712_localmi_v4/side_timestamp_market_states.parquet"))
    parser.add_argument("--calendar", type=Path, default=Path("data_perp/reports/meta_residual_extreme_local_uncaptured_events_202501_20260708_v3/all_extreme_event_cells.csv"))
    parser.add_argument("--output", type=Path, default=Path("data_perp/reports/residual_calendar_leaf_state_discovery_20260712_v1"))
    parser.add_argument("--evaluation-end", default="2026-07-11")
    parser.add_argument("--purge-hours", type=float, default=12.0)
    parser.add_argument("--max-features", type=int, default=1200)
    parser.add_argument("--n-estimators", type=int, default=240)
    parser.add_argument("--screened-features", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    manifest = run(args)
    print(json.dumps({"status": "complete", "output": str(args.output), "folds": len(manifest["folds"]), "models": len(manifest["models"])}, indent=2))


if __name__ == "__main__":
    main()
